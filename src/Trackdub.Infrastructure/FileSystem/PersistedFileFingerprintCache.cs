using System.Collections.Concurrent;
using System.Text.Json;
using Trackdub.Contracts;

namespace Trackdub.Infrastructure.FileSystem;

/// <summary>
/// Best-effort on-disk cache of file fingerprints keyed by path and validated by file stat
/// (byte length and last write time). Persistence problems are never fatal: a corrupt or
/// unwritable cache file degrades to "no persisted fingerprints" instead of blocking hashing.
/// </summary>
/// <remarks>
/// <para>
/// A persisted fingerprint is an optimization, not an integrity proof. A hit is validated only by
/// full path, byte length, and UTC write time, so byte corruption that leaves the length and the
/// write time unchanged is not detected by this cache. The escape hatch is explicit invalidation:
/// <see cref="Invalidate"/> forces the next computation to rehash.
/// </para>
/// <para>
/// The cache file has a single writer per process. <see cref="Save"/> writes the owning service's
/// whole in-memory map back to the file and invalidation markers are held per instance, so a
/// second service pointed at the same cache file could keep serving an invalidated fingerprint and
/// could write the invalidated entry back into the file. Use exactly one instance per cache file.
/// </para>
/// </remarks>
internal sealed class PersistedFileFingerprintCache(string cacheFilePath, IApplicationLogger logger)
{
    private const int SchemaVersion = 1;
    private const int MaxEntries = 512;
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = false };
    private readonly object sync = new();
    private readonly HashSet<string> invalidatedPaths = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, long> invalidationGenerations = new(StringComparer.Ordinal);

    public long CurrentInvalidationGeneration(string path)
    {
        lock (sync)
        {
            return invalidationGenerations.TryGetValue(path, out long generation) ? generation : 0;
        }
    }

    /// <summary>
    /// Reads the persisted entries, dropping any that are unusable and normalizing the
    /// remaining paths to their full form so they can be compared against file stats.
    /// </summary>
    public IReadOnlyList<Entry> Load()
    {
        lock (sync)
        {
            return ReadEntriesUnlocked()
                .Take(MaxEntries)
                .Select(TryNormalize)
                .OfType<Entry>()
                .ToArray();
        }
    }

    public void Save(IReadOnlyList<Entry> entries)
    {
        lock (sync)
        {
            WriteSnapshotUnlocked(
                new Snapshot(SchemaVersion, entries.Take(MaxEntries).ToArray()),
                "fingerprint cache");
        }
    }

    public bool IsInvalidated(string path)
    {
        lock (sync)
        {
            return invalidationGenerations.ContainsKey(path);
        }
    }

    public void Invalidate(string path)
    {
        lock (sync)
        {
            invalidationGenerations.AddOrUpdate(path, 1, (_, current) => current + 1);
            invalidatedPaths.Add(path);
            Snapshot current = new(SchemaVersion, ReadEntriesUnlocked().Where(entry =>
                !string.Equals(entry.Path, path, StringComparison.Ordinal)).ToArray());
            WriteSnapshotUnlocked(current, "fingerprint invalidation");
        }
    }

    public bool TryClearInvalidation(string path, long capturedGeneration)
    {
        lock (sync)
        {
            long current = invalidationGenerations.TryGetValue(path, out long generation) ? generation : 0;
            if (current != capturedGeneration)
            {
                // A newer invalidation happened since the caller captured the generation; the
                // pending marker belongs to a newer hash operation, so do not publish or clear it.
                return false;
            }

            invalidationGenerations.Remove(path);
            invalidatedPaths.Remove(path);
            return true;
        }
    }

    private IReadOnlyList<Entry> ReadEntriesUnlocked()
    {
        try
        {
            if (!File.Exists(cacheFilePath))
            {
                return [];
            }

            using FileStream stream = File.OpenRead(cacheFilePath);
            Snapshot? snapshot = JsonSerializer.Deserialize<Snapshot>(stream, JsonOptions);
            return snapshot is { Version: SchemaVersion, Entries: not null } ? snapshot.Entries : [];
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException or NotSupportedException)
        {
            logger.LogDebug($"Ignoring unreadable fingerprint cache '{cacheFilePath}': {ex.Message}");
            return [];
        }
    }

    private void WriteSnapshotUnlocked(Snapshot snapshot, string description)
    {
        string temporaryPath = cacheFilePath + ".tmp";
        try
        {
            string? cacheDirectory = Path.GetDirectoryName(cacheFilePath);
            if (!string.IsNullOrEmpty(cacheDirectory))
            {
                Directory.CreateDirectory(cacheDirectory);
            }

            using (FileStream stream = new(temporaryPath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                JsonSerializer.Serialize(stream, snapshot, JsonOptions);
                stream.Flush();
            }
            File.Move(temporaryPath, cacheFilePath, overwrite: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException)
        {
            logger.LogDebug($"Could not persist {description} '{cacheFilePath}': {ex.Message}");
            TryDeleteTemporaryFile(temporaryPath);
        }
    }

    private void TryDeleteTemporaryFile(string path)
    {
        try { File.Delete(path); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { logger.LogDebug(ex.Message); }
    }

    private static Entry? TryNormalize(Entry entry)
    {
        if (entry is not { Path: not (null or ""), Sha256: not (null or "") })
        {
            return null;
        }

        if (!HasValidStat(entry))
        {
            return null;
        }

        try
        {
            return entry with { Path = Path.GetFullPath(entry.Path) };
        }
        catch (Exception ex) when (ex is ArgumentException or IOException or NotSupportedException)
        {
            return null;
        }
    }

    private static bool HasValidStat(Entry entry)
    {
        if (entry.SizeBytes < 0)
        {
            return false;
        }

        if (entry.LastWriteTimeUtcTicks < 0)
        {
            return false;
        }

        return entry.LastWriteTimeUtcTicks <= DateTime.MaxValue.Ticks;
    }

    internal sealed record Entry(string Path, string Sha256, long SizeBytes, long LastWriteTimeUtcTicks);
    private sealed record Snapshot(int Version, IReadOnlyList<Entry> Entries);
}
