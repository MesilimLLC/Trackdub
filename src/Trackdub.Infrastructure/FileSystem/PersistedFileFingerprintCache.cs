using System.Text.Json;
using Trackdub.Contracts;

namespace Trackdub.Infrastructure.FileSystem;

/// <summary>
/// Best-effort on-disk cache of file fingerprints keyed by path and validated by file stat
/// (byte length and last write time). Persistence problems are never fatal: a corrupt or
/// unwritable cache file degrades to "no persisted fingerprints" instead of blocking hashing.
/// </summary>
internal sealed class PersistedFileFingerprintCache(string cacheFilePath, IApplicationLogger logger)
{
    private const int SchemaVersion = 1;
    private const int MaxEntries = 512;
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = false };
    private readonly object sync = new();
    private readonly HashSet<string> invalidatedPaths = new(StringComparer.OrdinalIgnoreCase);

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
            return invalidatedPaths.Contains(path);
        }
    }

    public void Invalidate(string path)
    {
        lock (sync)
        {
            invalidatedPaths.Add(path);
            Snapshot current = new(SchemaVersion, ReadEntriesUnlocked().Where(entry =>
                !string.Equals(entry.Path, path, StringComparison.OrdinalIgnoreCase)).ToArray());
            WriteSnapshotUnlocked(current, "fingerprint invalidation");
        }
    }

    public void ClearInvalidation(string path)
    {
        lock (sync)
        {
            invalidatedPaths.Remove(path);
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
            Directory.CreateDirectory(Path.GetDirectoryName(cacheFilePath)!);
            using (FileStream stream = new(temporaryPath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                JsonSerializer.Serialize(stream, snapshot, JsonOptions);
                stream.Flush(flushToDisk: true);
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

        if (entry is not { SizeBytes: >= 0, LastWriteTimeUtcTicks: >= 0 })
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

    internal sealed record Entry(string Path, string Sha256, long SizeBytes, long LastWriteTimeUtcTicks);
    private sealed record Snapshot(int Version, IReadOnlyList<Entry> Entries);
}
