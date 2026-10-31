using System.Text;
using System.Text.Json;
using Trackdub.Domain;
using Trackdub.Infrastructure;
using Trackdub.Infrastructure.Settings;

namespace Trackdub.Infrastructure.Persistence.Repositories;

public sealed class LocalModelCacheRecordStore(TrackdubStoragePaths storagePaths)
{
    private readonly SemaphoreSlim mutationLock = new(1, 1);
    private int revision;
    private string? secondaryIndexPath;
    private IReadOnlyList<LocalModelCacheRecord> mirroredBaseline = [];

    internal int Revision => Volatile.Read(ref revision);

    /// <summary>
    /// Cross-process identity of the persisted index file (mtime + length + content hash),
    /// or null when the file is absent or unreadable. The lookup keys its in-memory cache on
    /// this in addition to <see cref="Revision"/> so writes from another process (CLI installs,
    /// a second instance) invalidate this process's snapshot; <see cref="Revision"/> alone is
    /// process-local. The content hash also catches same-length replacements that preserve
    /// the modification time, which mtime + length cannot distinguish.
    /// </summary>
    internal (long LastWriteTicks, long Length, string ContentHash)? GetIndexIdentity()
    {
        try
        {
            var info = new FileInfo(storagePaths.ModelCacheIndexPath);
            if (!info.Exists)
            {
                return null;
            }

            // Share read/write/delete so this probe never blocks a concurrent
            // atomic replace (File.Move overwrite) of the index on Windows.
            using var stream = new FileStream(
                info.FullName,
                FileMode.Open,
                FileAccess.Read,
                FileShare.ReadWrite | FileShare.Delete);
            string contentHash = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream));
            return (info.LastWriteTimeUtc.Ticks, info.Length, contentHash);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>
    /// Atomically reads the current cache index, applies <paramref name="mutator"/>, and persists the result.
    /// This is the only thread-safe way to update the index: concurrent calls are serialized through an
    /// per-instance and cross-process locks so read-modify-write sequences cannot lose updates. All production mutations must go
    /// through this method.
    /// </summary>
    public async Task MutateAsync(
        Func<IReadOnlyList<LocalModelCacheRecord>, IReadOnlyList<LocalModelCacheRecord>> mutator,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(mutator);

        await mutationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            using var indexLocks = await AcquireIndexLocksAsync(cancellationToken).ConfigureAwait(false);
            IReadOnlyList<LocalModelCacheRecord> current = Load();
            if (secondaryIndexPath is not null)
            {
                IReadOnlyList<LocalModelCacheRecord> secondary = LoadIndex(secondaryIndexPath);
                // A record removed from either index since the last mirrored write must stay removed.
                current = NormalizeRecords([.. current, .. secondary])
                    .Where(record => !ContainsRecord(mirroredBaseline, record) ||
                        (ContainsRecord(current, record) && ContainsRecord(secondary, record)))
                    .ToArray();
            }

            await SaveLockedAsync(mutator(current), cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            mutationLock.Release();
        }
    }

    /// <summary>
    /// Copies model-cache index records to a new cache directory without moving model files.
    /// Existing model roots remain available, while future downloads can use the selected directory.
    /// </summary>
    public async Task CopyRecordsToDirectoryAsync(
        DirectoryInfo destinationModelCacheDirectory,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(destinationModelCacheDirectory);
        string destinationDirectory = destinationModelCacheDirectory.FullName;
        if (RootsEqual(storagePaths.ModelCacheDirectory, destinationDirectory))
        {
            return;
        }

        await mutationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            string destinationIndexPath = Path.Join(destinationDirectory, "model-cache-records.json");
            using var indexLocks = await AcquireIndexLocksAsync(
                [storagePaths.ModelCacheIndexPath, destinationIndexPath], cancellationToken).ConfigureAwait(false);
            IReadOnlyList<LocalModelCacheRecord> sourceRecords = Load();
            IReadOnlyList<LocalModelCacheRecord> destinationRecords = LoadIndex(destinationIndexPath);
            if (sourceRecords.Count == 0 && destinationRecords.Count == 0)
            {
                // Nothing to copy, but still mirror future writes to the new directory.
                secondaryIndexPath = destinationIndexPath;
                mirroredBaseline = [];
                return;
            }

            IReadOnlyList<LocalModelCacheRecord> combined = NormalizeRecords([.. sourceRecords, .. destinationRecords]);
            if (combined.Count > 0)
            {
                await WriteIndexAsync(storagePaths.ModelCacheIndexPath, combined, cancellationToken).ConfigureAwait(false);
                Interlocked.Increment(ref revision);
                await WriteIndexAsync(destinationIndexPath, combined, cancellationToken).ConfigureAwait(false);
                // Arm the mirror only after the destination copy succeeds so a failed write
                // does not leave later SaveAsync calls targeting an unwritable directory.
                secondaryIndexPath = destinationIndexPath;
                mirroredBaseline = combined;
            }
        }
        finally
        {
            mutationLock.Release();
        }
    }

    /// <summary>
    /// Reads the cache index. Safe for concurrent readers; does not acquire the mutation lock.
    /// Callers that modify and persist must use <see cref="MutateAsync"/> instead.
    /// </summary>
    /// <remarks>
    /// The index is small and always read whole, so it is read into a buffer synchronously and
    /// deserialized from memory, matching the synchronous reads in <see cref="FileSmokeVerdictStore"/>
    /// and <see cref="TrackdubStoragePathResolver"/>. Measured on a two-record index, this removes
    /// ~9 ms of the one-time first read and roughly halves the per-read cost after it (0.30 ms →
    /// 0.16 ms) that the async reader spent on its own machinery; the dominant cost is still
    /// building the serializer metadata for the record graph, which this does not change.
    /// <para>
    /// A leading UTF-8 byte order mark is skipped, so an index written by an editor or script that
    /// emits one (Notepad, PowerShell <c>-Encoding utf8</c>) loads exactly as it did through the
    /// stream-based read this replaced. The store never writes a mark itself.
    /// </para>
    /// </remarks>
    public Task<IReadOnlyList<LocalModelCacheRecord>> LoadAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(Load(cancellationToken));
    }

    internal IReadOnlyList<LocalModelCacheRecord> Load(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return LoadIndex(storagePaths.ModelCacheIndexPath);
    }

    private static IReadOnlyList<LocalModelCacheRecord> LoadIndex(string indexPath)
    {
        if (!File.Exists(indexPath))
        {
            return [];
        }

        byte[] payload = File.ReadAllBytes(indexPath);
        ReadOnlySpan<byte> json = payload;
        ReadOnlySpan<byte> preamble = Encoding.UTF8.GetPreamble();
        if (json.StartsWith(preamble))
        {
            json = json[preamble.Length..];
        }

        LocalModelCacheRecord[]? records = JsonSerializer.Deserialize(
            json,
            LocalModelCacheSerializationContext.Default.LocalModelCacheRecordArray);

        return NormalizeRecords(records ?? []);
    }

    /// <summary>
    /// Persists the cache index via temp-file write and replace. Production read-modify-write operations
    /// must route through <see cref="MutateAsync"/>. Internal so tests can seed initial state.
    /// </summary>
    internal async Task SaveAsync(
        IReadOnlyList<LocalModelCacheRecord> records,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(records);

        await mutationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            using var indexLocks = await AcquireIndexLocksAsync(cancellationToken).ConfigureAwait(false);
            await SaveLockedAsync(records, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            mutationLock.Release();
        }
    }

    private async Task SaveLockedAsync(IReadOnlyList<LocalModelCacheRecord> records, CancellationToken cancellationToken)
    {
        await WriteIndexAsync(storagePaths.ModelCacheIndexPath, records, cancellationToken).ConfigureAwait(false);
        Interlocked.Increment(ref revision);
        if (!string.IsNullOrWhiteSpace(secondaryIndexPath) &&
            !RootsEqual(
                Path.GetDirectoryName(storagePaths.ModelCacheIndexPath)!,
                Path.GetDirectoryName(secondaryIndexPath)!))
        {
            await WriteIndexAsync(secondaryIndexPath, records, cancellationToken).ConfigureAwait(false);
            mirroredBaseline = NormalizeRecords(records);
        }
    }

    private Task<IDisposable> AcquireIndexLocksAsync(CancellationToken cancellationToken) =>
        AcquireIndexLocksAsync(secondaryIndexPath is null
            ? [storagePaths.ModelCacheIndexPath]
            : [storagePaths.ModelCacheIndexPath, secondaryIndexPath], cancellationToken);

    private static async Task<IDisposable> AcquireIndexLocksAsync(
        IEnumerable<string> indexPaths, CancellationToken cancellationToken)
    {
        var locks = new List<FileStream>();
        try
        {
            // All writers acquire multiple index locks in the same order to avoid deadlocks.
            foreach (string path in indexPaths.Distinct(RootPathComparer).OrderBy(path => path, RootPathComparer))
            {
                locks.Add(await AcquireFileLockAsync(path + ".lock", cancellationToken).ConfigureAwait(false));
            }

            return new IndexLocks(locks);
        }
        catch
        {
            foreach (FileStream fileLock in locks)
            {
                fileLock.Dispose();
            }

            throw;
        }
    }

    internal static async Task<FileStream> AcquireFileLockAsync(string lockPath, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(lockPath)!);
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                // Keep the lock file in place: deleting it would allow another process to lock a different inode.
                return new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
            }
            catch (IOException ex) when (IsLockContention(ex))
            {
                await Task.Delay(50, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    // Each code signals contention only on the platform where it means that: Windows sharing or
    // lock violations, EWOULDBLOCK on Linux (11) and macOS (35). Elsewhere the same numbers mean
    // other errors (for example EDEADLK or EPIPE), which must not be retried.
    private static bool IsLockContention(IOException ex)
    {
        int code = ex.HResult & 0xffff;
        return (OperatingSystem.IsWindows() && (code is 32 or 33))
            || (OperatingSystem.IsLinux() && code == 11)
            || (OperatingSystem.IsMacOS() && code == 35)
            || ex.Message.Contains("used by another process", StringComparison.OrdinalIgnoreCase);
    }

    private sealed class IndexLocks(List<FileStream> locks) : IDisposable
    {
        public void Dispose()
        {
            foreach (FileStream fileLock in locks)
            {
                fileLock.Dispose();
            }
        }
    }

    private static async Task WriteIndexAsync(
        string indexPath,
        IReadOnlyList<LocalModelCacheRecord> records,
        CancellationToken cancellationToken)
    {
        string directory = Path.GetDirectoryName(indexPath)
            ?? throw new ArgumentException("The cache index path must include a directory.", nameof(indexPath));
        Directory.CreateDirectory(directory);
        string tempPath = indexPath + "." + Guid.NewGuid().ToString("N") + ".tmp";

        try
        {
            await using (var stream = new FileStream(
                             tempPath,
                             FileMode.CreateNew,
                             FileAccess.Write,
                             FileShare.None,
                             bufferSize: 4096,
                             options: FileOptions.Asynchronous))
            {
                // The generated metadata is bound to LocalModelCacheRecord[], so materialize
                // IReadOnlyList<T> implementations (e.g. single-element wrappers) before serializing.
                LocalModelCacheRecord[] materialized = [.. NormalizeRecords(records)];
                await JsonSerializer.SerializeAsync(stream, materialized, LocalModelCacheSerializationContext.Default.LocalModelCacheRecordArray, cancellationToken)
                    .ConfigureAwait(false);
                await stream.FlushAsync(cancellationToken).ConfigureAwait(false);
            }

            File.Move(tempPath, indexPath, overwrite: true);
        }
        catch
        {
            TryDeleteFile(tempPath);
            throw;
        }
    }

    public static bool RootsEqual(string left, string right)
    {
        if (TryNormalizeRoot(left, out string? normalizedLeft) &&
            TryNormalizeRoot(right, out string? normalizedRight))
        {
            return string.Equals(normalizedLeft, normalizedRight, RootPathComparison);
        }

        // Never throw from equality: fall back to a raw trimmed comparison when
        // either path is missing or cannot be normalized.
        return string.Equals(left?.Trim(), right?.Trim(), RootPathComparison);
    }

    public static IReadOnlyList<LocalModelVariantRecord> CompatibleVariants(
        LocalModelCacheRecord record, IEnumerable<LocalModelVariantRecord> variants) =>
        variants.Where(variant =>
                string.Equals(variant.SourceModelRevision, record.Revision, StringComparison.OrdinalIgnoreCase) &&
                (string.IsNullOrWhiteSpace(variant.SourceModelSha256) ||
                 string.Equals(variant.SourceModelSha256, record.Sha256, StringComparison.OrdinalIgnoreCase)))
            .GroupBy(variant => variant.Alias, StringComparer.OrdinalIgnoreCase)
            .Select(group => group.OrderByDescending(variant => variant.CreatedAtUtc)
                .ThenByDescending(variant => variant.IntegrityFailed).First())
            .ToArray();

    private static bool ContainsRecord(IReadOnlyList<LocalModelCacheRecord> records, LocalModelCacheRecord candidate) =>
        records.Any(record => string.Equals(record.ModelId, candidate.ModelId, StringComparison.OrdinalIgnoreCase) &&
            RootsEqual(record.RootPath, candidate.RootPath) &&
            string.Equals(record.Revision, candidate.Revision, StringComparison.Ordinal) &&
            string.Equals(record.Sha256, candidate.Sha256, StringComparison.OrdinalIgnoreCase));

    private static IReadOnlyList<LocalModelCacheRecord> NormalizeRecords(IReadOnlyList<LocalModelCacheRecord> records) =>
        records.GroupBy(record => record.ModelId, StringComparer.OrdinalIgnoreCase)
            .SelectMany(model => model.GroupBy(record => NormalizeRootKey(record.RootPath), RootPathComparer))
            .Select(group =>
            {
                LocalModelCacheRecord newest = group.OrderByDescending(record => record.CachedAtUtc)
                    .ThenByDescending(record => record.IntegrityFailed).First();
                return group.Count() == 1 ? newest : newest with
                {
                    Variants = CompatibleVariants(newest, group.SelectMany(record => record.Variants))
                };
            }).ToArray();

    private static StringComparer RootPathComparer =>
        OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;

    private static StringComparison RootPathComparison =>
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;

    private static string NormalizeRootKey(string rootPath) =>
        TryNormalizeRoot(rootPath, out string? normalized)
            ? normalized
            : (rootPath ?? string.Empty).Trim();

    private static bool TryNormalizeRoot(string path, out string normalized)
    {
        normalized = string.Empty;
        if (string.IsNullOrWhiteSpace(path))
        {
            return false;
        }

        try
        {
            normalized = Path.TrimEndingDirectorySeparator(Path.GetFullPath(path));
            return true;
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
        {
            return false;
        }
    }

    private static void TryDeleteFile(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }
}
