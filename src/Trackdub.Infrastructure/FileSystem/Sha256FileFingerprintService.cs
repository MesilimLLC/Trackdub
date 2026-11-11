using System.Collections.Concurrent;
using System.Security.Cryptography;
using Trackdub.Contracts;
using Trackdub.Infrastructure.Logging;

namespace Trackdub.Infrastructure.FileSystem;

public sealed class Sha256FileFingerprintService : IFileFingerprintService
{
    internal const int MaxCachedFingerprints = 512;

    private readonly object cacheTrimLock = new();
    private readonly IApplicationLogger logger;
    private readonly PersistedFileFingerprintCache? persistedCache;
    private readonly ConcurrentDictionary<string, FingerprintCacheEntry> fingerprintCache = new(StringComparer.Ordinal);
    private long cacheAccessSequence;

    public Sha256FileFingerprintService(IApplicationLogger? logger = null, string? cacheFilePath = null)
    {
        this.logger = logger ?? new DebugApplicationLogger();
        persistedCache = string.IsNullOrWhiteSpace(cacheFilePath)
            ? null
            : new PersistedFileFingerprintCache(Path.GetFullPath(cacheFilePath), this.logger);
        LoadPersistedFingerprints(persistedCache);
    }

    public async Task<FileFingerprint> ComputeAsync(string path, CancellationToken cancellationToken)
    {
        string fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("File was not found for fingerprinting.", fullPath);
        }

        var fileInfo = new FileInfo(fullPath);

        if (TryGetCachedFingerprint(fileInfo, out FingerprintCacheEntry? cached))
        {
            TouchCachedFingerprint(cached);
            logger.LogDebug($"Fingerprint cache hit for file: '{fullPath}'");
            return new FileFingerprint(cached.Hash, cached.Size, cached.LastWriteTimeUtc);
        }

        long captureGeneration = persistedCache?.CurrentInvalidationGeneration(fullPath) ?? 0;
        logger.LogDebug($"Computing fingerprint for file: '{fullPath}'");

        await using var stream = new FileStream(
            fullPath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 4096,
            options: FileOptions.Asynchronous | FileOptions.SequentialScan);

        byte[] hash = await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false);

        string hashHex = Convert.ToHexString(hash).ToLowerInvariant();

        // Align returned fingerprint and cache entry with the bytes we hashed (post-read metadata).
        fileInfo.Refresh();
        logger.LogDebug($"Fingerprint computed: SHA256={hashHex} Size={fileInfo.Length} bytes");

        bool shouldCache = persistedCache is null || persistedCache.TryClearInvalidation(fullPath, captureGeneration);
        if (shouldCache)
        {
            CacheFingerprint(new FingerprintCacheEntry(
                fullPath,
                fileInfo.Length,
                fileInfo.LastWriteTimeUtc,
                hashHex,
                NextCacheAccessSequence()));
        }

        return new FileFingerprint(hashHex, fileInfo.Length, fileInfo.LastWriteTimeUtc);
    }

    /// <summary>
    /// Drops any cached fingerprint for <paramref name="path"/> so the next computation rehashes the
    /// file even when its length and write time are unchanged.
    /// </summary>
    public void Invalidate(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string fullPath = Path.GetFullPath(path);
        lock (cacheTrimLock)
        {
            fingerprintCache.TryRemove(fullPath, out _);
            persistedCache?.Invalidate(fullPath);
        }
    }

    internal int CachedFingerprintCountForTesting => fingerprintCache.Count;

    internal bool ContainsCachedFingerprintForTesting(string path) =>
        fingerprintCache.ContainsKey(Path.GetFullPath(path));

    private void LoadPersistedFingerprints(PersistedFileFingerprintCache? cache)
    {
        if (cache is null)
        {
            return;
        }

        foreach (PersistedFileFingerprintCache.Entry entry in cache.Load().Reverse())
        {
            fingerprintCache[entry.Path] = new FingerprintCacheEntry(
                entry.Path,
                entry.SizeBytes,
                new DateTime(entry.LastWriteTimeUtcTicks, DateTimeKind.Utc),
                entry.Sha256,
                NextCacheAccessSequence());
        }
    }

    private bool TryGetCachedFingerprint(
        FileInfo fileInfo,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out FingerprintCacheEntry? cached)
    {
        cached = null;
        if (persistedCache is not null)
        {
            if (persistedCache.IsInvalidated(fileInfo.FullName))
            {
                return false;
            }
        }

        if (!fingerprintCache.TryGetValue(fileInfo.FullName, out FingerprintCacheEntry? found))
        {
            return false;
        }

        if ((found.Size, found.LastWriteTimeUtc) != (fileInfo.Length, fileInfo.LastWriteTimeUtc))
        {
            return false;
        }

        cached = found;
        return true;
    }

    private long NextCacheAccessSequence() =>
        Interlocked.Increment(ref cacheAccessSequence);

    private void TouchCachedFingerprint(FingerprintCacheEntry cached)
    {
        var updated = cached with { LastAccessSequence = NextCacheAccessSequence() };
        fingerprintCache.TryUpdate(cached.Path, updated, cached);
    }

    private void CacheFingerprint(FingerprintCacheEntry entry)
    {
        lock (cacheTrimLock)
        {
            if (!fingerprintCache.ContainsKey(entry.Path))
            {
                EvictLeastRecentlyUsedEntries();
            }

            fingerprintCache[entry.Path] = entry;
            persistedCache?.Save(fingerprintCache
                .OrderByDescending(static pair => pair.Value.LastAccessSequence)
                .Select(static pair => new PersistedFileFingerprintCache.Entry(
                    pair.Key, pair.Value.Hash, pair.Value.Size, pair.Value.LastWriteTimeUtc.Ticks))
                .ToArray());
        }
    }

    private void EvictLeastRecentlyUsedEntries()
    {
        while (fingerprintCache.Count >= MaxCachedFingerprints && TryFindLeastRecentlyUsedEntry() is string path)
        {
            fingerprintCache.TryRemove(path, out _);
        }
    }

    private string? TryFindLeastRecentlyUsedEntry()
    {
        string? oldestPath = null;
        long oldestAccess = long.MaxValue;
        foreach ((string candidatePath, FingerprintCacheEntry candidate) in fingerprintCache)
        {
            if (candidate.LastAccessSequence >= oldestAccess)
            {
                continue;
            }

            oldestAccess = candidate.LastAccessSequence;
            oldestPath = candidatePath;
        }

        return oldestPath;
    }

    private sealed record FingerprintCacheEntry(
        string Path,
        long Size,
        DateTime LastWriteTimeUtc,
        string Hash,
        long LastAccessSequence);
}
