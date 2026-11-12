using System.Text.Json;
using Trackdub.Infrastructure.FileSystem;
using Trackdub.Infrastructure.Logging;

namespace Trackdub.Infrastructure.Tests;

public sealed class PersistedFileFingerprintCacheTests : IDisposable
{
    private const string ModelPath = "model.onnx";
    private readonly List<string> tempFiles = [];

    public void Dispose()
    {
        foreach (string file in tempFiles)
        {
            try { File.Delete(file); } catch { /* best-effort cleanup */ }
        }
    }

    [Fact]
    public void Load_keeps_a_valid_entry_and_normalizes_its_path()
    {
        string cachePath = CreateCachePath();
        string denormalized = Path.Join(Path.GetDirectoryName(cachePath)!, ".", Path.GetFileName(cachePath));
        PersistedFileFingerprintCache cache = CreateCache(cachePath, MakeEntry(denormalized));

        PersistedFileFingerprintCache.Entry entry = Assert.Single(cache.Load());

        Assert.Equal(cachePath, entry.Path);
        Assert.Equal(new string('a', 64), entry.Sha256);
    }

    [Fact]
    public void Load_drops_an_entry_with_negative_size()
    {
        AssertDropped(MakeEntry(ModelPath, sizeBytes: -1));
    }

    [Fact]
    public void Load_drops_an_entry_with_negative_write_ticks()
    {
        AssertDropped(MakeEntry(ModelPath, ticks: -1));
    }

    [Fact]
    public void Load_drops_an_entry_with_write_ticks_beyond_datetime_max()
    {
        AssertDropped(MakeEntry(ModelPath, ticks: DateTime.MaxValue.Ticks + 1));
    }

    [Fact]
    public void Invalidate_matches_persisted_paths_case_insensitively()
    {
        PersistedFileFingerprintCache cache = CreateCache(CreateCachePath(), MakeEntry("Model.Onnx"));

        cache.Invalidate(ModelPath);

        Assert.Empty(cache.Load());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TryClearInvalidation_clears_the_marker_only_for_the_captured_generation(bool newerInvalidation)
    {
        PersistedFileFingerprintCache cache = CreateCache(CreateCachePath());
        cache.Invalidate(ModelPath);
        long captured = cache.CurrentInvalidationGeneration(ModelPath);
        if (newerInvalidation)
        {
            cache.Invalidate(ModelPath);
        }

        Assert.Equal(!newerInvalidation, cache.TryClearInvalidation(ModelPath, captured));
        Assert.Equal(newerInvalidation, cache.IsInvalidated(ModelPath));
    }

    [Fact]
    public void Save_truncates_entries_at_the_persisted_cap()
    {
        PersistedFileFingerprintCache cache = CreateCache(CreateCachePath());
        PersistedFileFingerprintCache.Entry[] entries = Enumerable
            .Range(0, PersistedFileFingerprintCache.MaxEntries + 1)
            .Select(i => MakeEntry($"model-{i}.onnx"))
            .ToArray();

        cache.Save(entries);

        Assert.Equal(PersistedFileFingerprintCache.MaxEntries, cache.Load().Count);
    }

    [Fact]
    public void The_in_memory_and_persisted_caps_are_the_same_constant()
    {
        Assert.Equal(PersistedFileFingerprintCache.MaxEntries, Sha256FileFingerprintService.MaxCachedFingerprints);
    }

    private void AssertDropped(PersistedFileFingerprintCache.Entry entry)
    {
        Assert.Empty(CreateCache(CreateCachePath(), entry).Load());
    }

    private string CreateCachePath()
    {
        string path = Path.Join(Path.GetTempPath(), "Trackdub.Infrastructure.Tests", Guid.NewGuid().ToString("N") + ".json");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        tempFiles.Add(path);
        tempFiles.Add(path + ".tmp");
        return path;
    }

    // Serializing a real object keeps Windows path separators escaped; concatenating a raw
    // path into JSON produces an unreadable cache, which would make a guard test pass vacuously.
    private PersistedFileFingerprintCache CreateCache(string cachePath, params PersistedFileFingerprintCache.Entry[] entries)
    {
        File.WriteAllText(cachePath, JsonSerializer.Serialize(new { Version = 1, Entries = entries }));
        return new PersistedFileFingerprintCache(cachePath, new DebugApplicationLogger());
    }

    private static PersistedFileFingerprintCache.Entry MakeEntry(string path, long sizeBytes = 1, long ticks = 1) =>
        new(path, new string('a', 64), sizeBytes, ticks);
}
