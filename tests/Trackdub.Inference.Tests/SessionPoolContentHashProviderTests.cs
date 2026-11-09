using Trackdub.Contracts;
using Trackdub.Inference.Onnx.Pool;

namespace Trackdub.Inference.Tests;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class SessionPoolContentHashProviderCollection
{
    public const string Name = "Session pool content-hash provider";
}

[Collection(SessionPoolContentHashProviderCollection.Name)]
public sealed class SessionPoolContentHashProviderTests : IDisposable
{
    public void Dispose() => SessionPoolContentHashProvider.Reset();

    [Fact]
    public async Task HashAsync_uses_configured_fingerprint_service()
    {
        string path = Path.Join(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".onnx");
        await File.WriteAllBytesAsync(path, [1, 2, 3], CancellationToken.None);
        string expected = new string('c', 64);
        var service = new FixedFingerprintService(new FileFingerprint(expected, 3, DateTimeOffset.UnixEpoch));
        SessionPoolContentHashProvider.Configure(service);
        try
        {
            string? actual = await SessionPoolKey.HashModelContentAsync(path, CancellationToken.None);
            Assert.Equal(expected, actual);
            Assert.Equal(path, service.RequestedPath);
        }
        finally
        {
            SessionPoolKey.InvalidateModelContentHash(path);
            File.Delete(path);
        }
    }

    [Fact]
    public async Task HashAsync_falls_back_to_content_hash_when_provider_fails()
    {
        string path = Path.Join(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".onnx");
        byte[] bytes = [4, 5, 6];
        await File.WriteAllBytesAsync(path, bytes, CancellationToken.None);
        SessionPoolContentHashProvider.Configure(new ThrowingFingerprintService());
        try
        {
            string? actual = await SessionPoolKey.HashModelContentAsync(path, CancellationToken.None);
            Assert.Equal(Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bytes)).ToLowerInvariant(), actual);
        }
        finally
        {
            SessionPoolKey.InvalidateModelContentHash(path);
            File.Delete(path);
        }
    }

    [Fact]
    public void Reset_removes_the_process_wide_provider()
    {
        SessionPoolContentHashProvider.Configure(new FixedFingerprintService(new FileFingerprint(new string('d', 64), 0, DateTimeOffset.UnixEpoch)));
        SessionPoolContentHashProvider.Reset();
        Assert.Null(SessionPoolContentHashProvider.Current);
    }

    private sealed class FixedFingerprintService(FileFingerprint fingerprint) : IFileFingerprintService
    {
        public string? RequestedPath { get; private set; }
        public Task<FileFingerprint> ComputeAsync(string path, CancellationToken cancellationToken)
        {
            RequestedPath = path;
            return Task.FromResult(fingerprint);
        }
        public void Invalidate(string path)
        {
        }
    }

    private sealed class ThrowingFingerprintService : IFileFingerprintService
    {
        public Task<FileFingerprint> ComputeAsync(string path, CancellationToken cancellationToken) =>
            Task.FromException<FileFingerprint>(new IOException("fingerprint unavailable"));
        public void Invalidate(string path)
        {
        }
    }
}
