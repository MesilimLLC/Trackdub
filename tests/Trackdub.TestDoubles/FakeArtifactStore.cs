using System.Collections.Concurrent;
using System.Text.Json;
using Trackdub.Contracts;

namespace Trackdub.TestDoubles;

/// <summary>
/// In-memory artifact store. Write handles use <see cref="Path.GetTempFileName"/> for the
/// temporary path so callers can write real bytes; <see cref="CommitAsync"/> reads those bytes
/// into the in-memory blob dictionary and deletes the temp file.
/// <see cref="GetPath"/> returns a path registered through <see cref="SeedPath"/> when present;
/// otherwise it returns the relative path unchanged for in-memory-only tests.
/// </summary>
public sealed class FakeArtifactStore(string? rootDirectory = null) : IArtifactStore
{
    private readonly ConcurrentDictionary<string, byte[]> blobs = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentDictionary<string, string> paths = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentDictionary<string, string> pendingCommits = new(StringComparer.OrdinalIgnoreCase);
    private readonly string? rootDirectory = string.IsNullOrWhiteSpace(rootDirectory)
        ? null
        : Path.GetFullPath(rootDirectory);

    public int EnsureLayoutCallCount { get; private set; }

    public string? FailingJsonWriteFileName { get; set; }

    /// <summary>Registers a blob so <see cref="Exists"/> returns true and <see cref="GetPath"/> is meaningful.</summary>
    public void Seed(string relativePath, byte[]? content = null)
    {
        byte[] bytes = content ?? [];
        blobs[relativePath] = bytes;
        if (rootDirectory is null)
        {
            return;
        }

        string fullPath = GetPath(relativePath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        File.WriteAllBytes(fullPath, bytes);
    }

    /// <summary>Maps a project-relative artifact to a real file path for components that open files.</summary>
    public void SeedPath(string relativePath, string fullPath, byte[]? content = null)
    {
        paths[relativePath] = fullPath;
        blobs[relativePath] = content ?? [];
    }

    /// <summary>All committed blobs, keyed by relative path.</summary>
    public IReadOnlyDictionary<string, byte[]> Blobs => blobs;

    public Task EnsureLayoutAsync(CancellationToken cancellationToken)
    {
        EnsureLayoutCallCount++;
        return Task.CompletedTask;
    }

    public ArtifactWriteHandle CreateWriteHandle(string relativePath)
    {
        string tempPath = Path.GetTempFileName();
        string finalPath = GetPath(relativePath);
        pendingCommits[tempPath] = relativePath;
        return new ArtifactWriteHandle(relativePath, finalPath, tempPath);
    }

    public async Task CommitAsync(ArtifactWriteHandle handle, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(handle);
        if (FailingJsonWriteFileName is not null && Path.GetFileName(handle.RelativePath).StartsWith(
            Path.GetFileNameWithoutExtension(FailingJsonWriteFileName), StringComparison.OrdinalIgnoreCase))
            throw new IOException("Simulated artifact promotion failure.");

        if (File.Exists(handle.TemporaryPath))
        {
            byte[] bytes = await File.ReadAllBytesAsync(
                handle.TemporaryPath, cancellationToken).ConfigureAwait(false);
            blobs[handle.RelativePath] = bytes;
            if (rootDirectory is not null)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(handle.FinalPath)!);
                await File.WriteAllBytesAsync(handle.FinalPath, bytes, cancellationToken).ConfigureAwait(false);
            }

            File.Delete(handle.TemporaryPath);
        }
        else
        {
            // Temp file already cleaned up; record an empty blob so Exists returns true.
            blobs.TryAdd(handle.RelativePath, []);
        }

        pendingCommits.Remove(handle.TemporaryPath, out _);
    }

    public async Task CommitNewAsync(ArtifactWriteHandle handle, Trackdub.Contracts.Transcripts.ArtifactPromotionReceipt receipt, CancellationToken cancellationToken)
    {
        if (Exists(handle.RelativePath)) throw new IOException("Artifact destination already exists.");
        await CommitAsync(handle, cancellationToken);
        receipt.MarkCreated();
    }

    public async Task WriteJsonAsync<T>(string relativePath, T value, CancellationToken cancellationToken)
    {
        if (string.Equals(
            Path.GetFileName(relativePath.Replace('/', Path.DirectorySeparatorChar)),
            FailingJsonWriteFileName,
            StringComparison.OrdinalIgnoreCase))
        {
            throw new IOException($"Simulated JSON write failure for '{FailingJsonWriteFileName}'.");
        }

        byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(value);
        blobs[relativePath] = bytes;
        if (rootDirectory is not null)
        {
            string fullPath = GetPath(relativePath);
            Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
            await File.WriteAllBytesAsync(fullPath, bytes, cancellationToken).ConfigureAwait(false);
        }
    }

    public Task<T?> ReadJsonAsync<T>(string relativePath, CancellationToken cancellationToken)
    {
        if (!blobs.TryGetValue(relativePath, out byte[]? bytes) || bytes.Length == 0)
        {
            return Task.FromResult<T?>(default);
        }

        try
        {
            T? result = JsonSerializer.Deserialize<T>(bytes);
            return Task.FromResult(result);
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("The seeded JSON artifact is corrupt.", exception);
        }
    }

    /// <summary>Returns a seeded full path, a rooted test path, or the relative path for in-memory tests.</summary>
    public string GetPath(string relativePath) =>
        paths.TryGetValue(relativePath, out string? path)
            ? path
            : rootDirectory is null
                ? relativePath
                : Path.GetFullPath(Path.Join(rootDirectory, relativePath.Replace('/', Path.DirectorySeparatorChar)));

    public bool Exists(string relativePath) => blobs.ContainsKey(relativePath);
}
