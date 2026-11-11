using System.Security.Cryptography;
using System.Text;
using Trackdub.Contracts;
using Trackdub.Infrastructure.FileSystem;

namespace Trackdub.Infrastructure.Tests;

public sealed class Sha256FileFingerprintServiceTests : IDisposable
{
    private readonly Sha256FileFingerprintService service = new();
    private readonly List<string> tempFiles = [];

    public void Dispose()
    {
        foreach (string file in tempFiles)
        {
            try { File.Delete(file); } catch { /* best-effort cleanup */ }
        }
    }

    private string CreateTempFile(string content)
    {
        string path = Path.Join(Path.GetTempPath(), "Trackdub.Infrastructure.Tests", Guid.NewGuid().ToString("N") + ".tmp");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, content);
        tempFiles.Add(path);
        return path;
    }

    private string CreateTempFile(byte[] content)
    {
        string path = Path.Join(Path.GetTempPath(), "Trackdub.Infrastructure.Tests", Guid.NewGuid().ToString("N") + ".tmp");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, content);
        tempFiles.Add(path);
        return path;
    }

    // System.IO.File has no SetLength member; truncating through a write handle is the
    // supported way to restore a file's byte length while leaving the content different.
    private static void SetFileLength(string path, long length)
    {
        using FileStream stream = new(path, FileMode.Open, FileAccess.Write, FileShare.None);
        stream.SetLength(length);
    }

    // ---------------------------------------------------------------
    // RED phase: tests that define expected behavior before optimization
    // ---------------------------------------------------------------

    [Fact]
    public async Task ComputeAsync_returns_expected_sha256_for_known_content()
    {
        string content = "hello world";
        string path = CreateTempFile(content);
        byte[] contentBytes = Encoding.UTF8.GetBytes(content);
        string expectedHash = Convert.ToHexString(SHA256.HashData(contentBytes)).ToLowerInvariant();

        FileFingerprint result = await service.ComputeAsync(path, TestContext.Current.CancellationToken);

        Assert.Equal(expectedHash, result.Sha256);
        Assert.Equal(contentBytes.Length, result.SizeBytes);
    }

    [Fact]
    public async Task ComputeAsync_returns_expected_sha256_for_binary_content()
    {
        byte[] content = [0x00, 0xFF, 0xAB, 0xCD, 0x12, 0x34, 0x56, 0x78];
        string path = CreateTempFile(content);
        string expectedHash = Convert.ToHexString(SHA256.HashData(content)).ToLowerInvariant();

        FileFingerprint result = await service.ComputeAsync(path, TestContext.Current.CancellationToken);

        Assert.Equal(expectedHash, result.Sha256);
    }

    [Fact]
    public async Task ComputeAsync_throws_FileNotFoundException_for_missing_path()
    {
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            service.ComputeAsync("Z:\\__nonexistent__\\file.bin", TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task ComputeAsync_returns_same_fingerprint_for_unchanged_file()
    {
        string path = CreateTempFile("stable content");

        FileFingerprint first = await service.ComputeAsync(path, TestContext.Current.CancellationToken);
        FileFingerprint second = await service.ComputeAsync(path, TestContext.Current.CancellationToken);

        Assert.Equal(first.Sha256, second.Sha256);
        Assert.Equal(first.SizeBytes, second.SizeBytes);
        Assert.Equal(first.LastWriteTimeUtc, second.LastWriteTimeUtc);
    }

    [Fact]
    public async Task ComputeAsync_returns_new_fingerprint_when_content_changes()
    {
        string path = CreateTempFile("original content");

        FileFingerprint first = await service.ComputeAsync(path, TestContext.Current.CancellationToken);

        // Modify — ensure distinct timestamp
        await File.WriteAllTextAsync(path, "modified content which is longer", TestContext.Current.CancellationToken);
        File.SetLastWriteTimeUtc(path, File.GetLastWriteTimeUtc(path).AddSeconds(2));

        FileFingerprint second = await service.ComputeAsync(path, TestContext.Current.CancellationToken);

        Assert.NotEqual(first.Sha256, second.Sha256);
        Assert.NotEqual(first.SizeBytes, second.SizeBytes);
    }

    [Fact]
    public async Task ComputeAsync_returns_new_fingerprint_when_timestamp_changes_but_size_same()
    {
        string path = CreateTempFile("same length!");

        FileFingerprint first = await service.ComputeAsync(path, TestContext.Current.CancellationToken);

        // Touch file to update timestamp (same content length)
        await File.WriteAllTextAsync(path, "SAME LENGTH!", TestContext.Current.CancellationToken);  // same byte count as "same length!"
        File.SetLastWriteTimeUtc(path, File.GetLastWriteTimeUtc(path).AddSeconds(2));

        FileFingerprint second = await service.ComputeAsync(path, TestContext.Current.CancellationToken);

        // Size is same but SHA-256 differs because content differs
        Assert.Equal(first.SizeBytes, second.SizeBytes);
        Assert.NotEqual(first.Sha256, second.Sha256);
    }

    [Fact]
    public async Task ComputeAsync_handles_concurrent_duplicate_requests()
    {
        string path = CreateTempFile("concurrent access test");
        var service = new Sha256FileFingerprintService();

        FileFingerprint[] results = await Task.WhenAll(
            Enumerable.Range(0, 16)
                .Select(_ => service.ComputeAsync(path, TestContext.Current.CancellationToken)));

        Assert.All(results, r => Assert.Equal(results[0].Sha256, r.Sha256));
    }

    [Fact]
    public async Task ComputeAsync_trims_cache_to_max_entries_and_evicts_oldest_entry()
    {
        static int GetRequiredIntField(Type type, string fieldName)
        {
            const System.Reflection.BindingFlags flags =
                System.Reflection.BindingFlags.Public |
                System.Reflection.BindingFlags.NonPublic |
                System.Reflection.BindingFlags.Static;

            var field = type.GetField(fieldName, flags);
            Assert.NotNull(field);

            object? value = field!.GetValue(null);
            Assert.IsType<int>(value);
            return (int)value!;
        }

        static int GetRequiredIntProperty(object target, string propertyName)
        {
            var type = target is Type staticType ? staticType : target.GetType();
            const System.Reflection.BindingFlags flags =
                System.Reflection.BindingFlags.Public |
                System.Reflection.BindingFlags.NonPublic |
                System.Reflection.BindingFlags.Instance |
                System.Reflection.BindingFlags.Static;

            var property = type.GetProperty(propertyName, flags);
            Assert.NotNull(property);

            object? value = property!.GetMethod!.IsStatic
                ? property.GetValue(null)
                : property.GetValue(target is Type ? null : target);

            Assert.IsType<int>(value);
            return (int)value!;
        }

        var service = new Sha256FileFingerprintService();
        int maxCacheEntries = GetRequiredIntField(typeof(Sha256FileFingerprintService), "MaxCachedFingerprints");

        Assert.True(maxCacheEntries > 0);

        var paths = Enumerable.Range(0, maxCacheEntries + 1)
            .Select(i => CreateTempFile($"cache entry {i}"))
            .ToList();

        foreach (string path in paths)
        {
            await service.ComputeAsync(path, TestContext.Current.CancellationToken);
        }

        int cacheCount = GetRequiredIntProperty(service, "CachedFingerprintCountForTesting");
        Assert.Equal(maxCacheEntries, cacheCount);

        string evictedOldestPath = paths[0];
        string retainedNewestPath = paths[^1];

        await using var oldestLock = new FileStream(
            evictedOldestPath, FileMode.Open, FileAccess.Read, FileShare.None);

        await Assert.ThrowsAnyAsync<IOException>(() =>
            service.ComputeAsync(evictedOldestPath, TestContext.Current.CancellationToken));

        // Newest path should still be fingerprintable (no exclusive lock — hashing always opens the file).
        FileFingerprint retained = await service.ComputeAsync(retainedNewestPath, TestContext.Current.CancellationToken);
        Assert.NotNull(retained);
    }

    [Fact]
    public async Task ComputeAsync_second_call_returns_cached_result_without_reopening_locked_file()
    {
        string path = CreateTempFile("cache hit test");
        FileFingerprint first = await service.ComputeAsync(path, TestContext.Current.CancellationToken);

        // File is unchanged (size + mtime match) — a cache hit must short-circuit
        // and never reopen the file, so an exclusive lock elsewhere must not matter.
        await using var lockStream = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.None);

        FileFingerprint second = await service.ComputeAsync(path, TestContext.Current.CancellationToken);

        Assert.Equal(first.Sha256, second.Sha256);
        Assert.Equal(first.SizeBytes, second.SizeBytes);
        Assert.Equal(first.LastWriteTimeUtc, second.LastWriteTimeUtc);
    }

    [Fact]
    public async Task ComputeAsync_cache_hit_does_not_reread_file_contents()
    {
        string path = CreateTempFile("same length!");
        FileFingerprint first = await service.ComputeAsync(path, TestContext.Current.CancellationToken);

        // Overwrite the file's bytes in place without changing size or mtime. Trusting
        // (size, mtime) as the cache key means a cache hit must return the ORIGINAL hash
        // here — this documents the accepted tradeoff (same scheme as Git/MSBuild/NuGet),
        // not a detection guarantee against in-place same-size overwrites.
        DateTime originalWriteTimeUtc = File.GetLastWriteTimeUtc(path);
        byte[] sameLengthDifferentContent = Encoding.UTF8.GetBytes("SAME LENGTH!");
        await File.WriteAllBytesAsync(path, sameLengthDifferentContent, TestContext.Current.CancellationToken);
        File.SetLastWriteTimeUtc(path, originalWriteTimeUtc);

        FileFingerprint second = await service.ComputeAsync(path, TestContext.Current.CancellationToken);

        Assert.Equal(first.Sha256, second.Sha256);
    }

    [Fact]
    public async Task ComputeAsync_cache_miss_when_file_grows()
    {
        string path = CreateTempFile("small");
        FileFingerprint first = await service.ComputeAsync(path, TestContext.Current.CancellationToken);

        await File.AppendAllTextAsync(path, " now with extra content to make it larger");
        File.SetLastWriteTimeUtc(path, File.GetLastWriteTimeUtc(path).AddSeconds(2));

        FileFingerprint second = await service.ComputeAsync(path, TestContext.Current.CancellationToken);

        Assert.NotEqual(first.Sha256, second.Sha256);
        Assert.NotEqual(first.SizeBytes, second.SizeBytes);
    }

    [Fact]
    public async Task ComputeAsync_reuses_persisted_fingerprint_in_a_new_service_instance()
    {
        string path = CreateTempFile("persisted fingerprint");
        string expected = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes("persisted fingerprint")))
            .ToLowerInvariant();
        string cachePath = Path.Join(Path.GetTempPath(), "Trackdub.Infrastructure.Tests", Guid.NewGuid().ToString("N") + ".json");
        tempFiles.Add(cachePath);
        tempFiles.Add(cachePath + ".tmp");
        var first = new Sha256FileFingerprintService(cacheFilePath: cachePath);
        await first.ComputeAsync(path, TestContext.Current.CancellationToken);

        // Capture the stat BEFORE rewriting the bytes, then restore it, so the persisted entry
        // still matches (length, write time) while the content on disk differs.
        var original = new FileInfo(path);
        DateTime originalWriteTime = original.LastWriteTimeUtc;
        long originalLength = original.Length;
        File.WriteAllBytes(path, Encoding.UTF8.GetBytes("different bytes with the same saved stat"));
        SetFileLength(path, originalLength);
        File.SetLastWriteTimeUtc(path, originalWriteTime);
        var second = new Sha256FileFingerprintService(cacheFilePath: cachePath);

        FileFingerprint result = await second.ComputeAsync(path, TestContext.Current.CancellationToken);

        Assert.Equal(expected, result.Sha256);
    }

    [Fact]
    public async Task ComputeAsync_rehashes_when_persisted_file_stat_changes()
    {
        string path = CreateTempFile("before");
        string cachePath = Path.Join(Path.GetTempPath(), "Trackdub.Infrastructure.Tests", Guid.NewGuid().ToString("N") + ".json");
        tempFiles.Add(cachePath);
        var first = new Sha256FileFingerprintService(cacheFilePath: cachePath);
        FileFingerprint before = await first.ComputeAsync(path, TestContext.Current.CancellationToken);
        File.AppendAllText(path, "after");
        var second = new Sha256FileFingerprintService(cacheFilePath: cachePath);

        FileFingerprint result = await second.ComputeAsync(path, TestContext.Current.CancellationToken);

        Assert.NotEqual(before.Sha256, result.Sha256);
        Assert.Equal(new FileInfo(path).Length, result.SizeBytes);
    }

    [Fact]
    public async Task ComputeAsync_ignores_corrupt_persisted_cache()
    {
        string path = CreateTempFile("valid content");
        string cachePath = Path.Join(Path.GetTempPath(), "Trackdub.Infrastructure.Tests", Guid.NewGuid().ToString("N") + ".json");
        tempFiles.Add(cachePath);
        File.WriteAllText(cachePath, "{not-json");
        var service = new Sha256FileFingerprintService(cacheFilePath: cachePath);

        FileFingerprint result = await service.ComputeAsync(path, TestContext.Current.CancellationToken);

        Assert.Equal(Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes("valid content"))).ToLowerInvariant(), result.Sha256);
    }

    [Fact]
    public async Task ComputeAsync_ignores_persisted_entry_with_out_of_range_write_time()
    {
        string path = CreateTempFile("valid content");
        string cachePath = Path.Join(Path.GetTempPath(), "Trackdub.Infrastructure.Tests", Guid.NewGuid().ToString("N") + ".json");
        tempFiles.Add(cachePath);
        File.WriteAllText(cachePath, "{\"Version\":1,\"Entries\":[{\"Path\":\"" + path + "\",\"Sha256\":\"abc\",\"SizeBytes\":13,\"LastWriteTimeUtcTicks\":9223372036854775807}]}");
        var service = new Sha256FileFingerprintService(cacheFilePath: cachePath);

        FileFingerprint result = await service.ComputeAsync(path, TestContext.Current.CancellationToken);

        Assert.Equal(Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes("valid content"))).ToLowerInvariant(), result.Sha256);
    }

    [Fact]
    public async Task Invalidate_forces_rehash_even_when_file_stat_is_unchanged()
    {
        string path = CreateTempFile("original");
        string cachePath = Path.Join(Path.GetTempPath(), "Trackdub.Infrastructure.Tests", Guid.NewGuid().ToString("N") + ".json");
        tempFiles.Add(cachePath);
        var service = new Sha256FileFingerprintService(cacheFilePath: cachePath);
        FileFingerprint before = await service.ComputeAsync(path, TestContext.Current.CancellationToken);
        var original = new FileInfo(path);
        DateTime writeTime = original.LastWriteTimeUtc;
        long length = original.Length;
        File.WriteAllBytes(path, Encoding.UTF8.GetBytes("replaced"));
        SetFileLength(path, length);
        File.SetLastWriteTimeUtc(path, writeTime);

        service.Invalidate(path);

        // A fresh instance must not reload the invalidated entry from disk either.
        var reloaded = new Sha256FileFingerprintService(cacheFilePath: cachePath);
        FileFingerprint result = await reloaded.ComputeAsync(path, TestContext.Current.CancellationToken);

        Assert.NotEqual(before.Sha256, result.Sha256);
    }
}
