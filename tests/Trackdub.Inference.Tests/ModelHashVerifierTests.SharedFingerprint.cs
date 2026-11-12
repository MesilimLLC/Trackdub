using Trackdub.Contracts;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Tests;

/// <summary>
/// Covers <see cref="ModelHashVerifier"/> behaviour that depends on the shared
/// <see cref="IFileFingerprintService"/> rather than on direct streaming hashes.
/// </summary>
public sealed partial class ModelHashVerifierTests
{
    [Fact]
    public async Task Verify_uses_shared_fingerprint_for_actual_hash()
    {
        string filePath = WriteTempFile([1, 2, 3, 4]);
        string expected = new string('a', 64);
        var fingerprints = new FixedFileFingerprintService(new FileFingerprint(expected, 4, DateTimeOffset.UnixEpoch));
        var verifier = new ModelHashVerifier(fingerprints);
        try
        {
            HashVerificationResult result = await verifier.VerifyAsync(expected, filePath);

            Assert.True(result.IsValid);
            Assert.Equal(expected, result.ActualSha256);
            Assert.Equal(filePath, fingerprints.RequestedPath);
        }
        finally
        {
            File.Delete(filePath);
        }
    }

    private sealed class FixedFileFingerprintService(FileFingerprint fingerprint) : IFileFingerprintService
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
}
