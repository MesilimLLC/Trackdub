using System.Security.Cryptography;
using Trackdub.Contracts;

namespace Trackdub.Inference.Runtime.ModelManifest;

public interface IModelHashVerifier
{
    Task<HashVerificationResult> VerifyAsync(
        ModelManifest manifest,
        string filePath,
        CancellationToken cancellationToken = default);

    Task<HashVerificationResult> VerifyAsync(
        string? expectedSha256,
        string filePath,
        CancellationToken cancellationToken = default);
}

public sealed class ModelHashVerifier(IFileFingerprintService? fingerprintService = null) : IModelHashVerifier
{
    private readonly IFileFingerprintService? fingerprintService = fingerprintService;

    public async Task<HashVerificationResult> VerifyAsync(
        ModelManifest manifest,
        string filePath,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

        string? expectedHash = manifest.Sha256?.Trim();
        if (manifest.HashVerificationPolicy.Mode is HashVerificationMode.None)
        {
            return new HashVerificationResult(true, false, null, null, "Hash verification disabled by policy.");
        }

        if (string.IsNullOrWhiteSpace(expectedHash))
        {
            if (manifest.HashVerificationPolicy.Mode is HashVerificationMode.Required)
            {
                return new HashVerificationResult(false, false, null, null, "Hash verification is required but the manifest does not define sha256.");
            }

            return new HashVerificationResult(true, false, null, null, "Manifest does not define sha256; verification skipped.");
        }

        string actualHash = fingerprintService is null
            ? await ComputeSha256Async(filePath, cancellationToken).ConfigureAwait(false)
            : (await fingerprintService.ComputeAsync(filePath, cancellationToken).ConfigureAwait(false)).Sha256;
        bool isMatch = actualHash.Equals(expectedHash, StringComparison.OrdinalIgnoreCase);
        return new HashVerificationResult(
            IsValid: isMatch,
            WasVerified: true,
            ExpectedSha256: expectedHash,
            ActualSha256: actualHash,
            FailureReason: isMatch ? null : "Computed file hash did not match manifest sha256.");
    }

    public async Task<HashVerificationResult> VerifyAsync(
        string? expectedSha256,
        string filePath,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

        string? trimmed = expectedSha256?.Trim();
        if (string.IsNullOrWhiteSpace(trimmed))
        {
            return new HashVerificationResult(true, false, null, null, "No SHA-256 in manifest; verification skipped.");
        }

        string actualHash = fingerprintService is null
            ? await ComputeSha256Async(filePath, cancellationToken).ConfigureAwait(false)
            : (await fingerprintService.ComputeAsync(filePath, cancellationToken).ConfigureAwait(false)).Sha256;
        bool isMatch = actualHash.Equals(trimmed, StringComparison.OrdinalIgnoreCase);
        return new HashVerificationResult(
            IsValid: isMatch,
            WasVerified: true,
            ExpectedSha256: trimmed,
            ActualSha256: actualHash,
            FailureReason: isMatch ? null : "Computed file hash did not match manifest sha256.");
    }

    private static async Task<string> ComputeSha256Async(string filePath, CancellationToken cancellationToken)
    {
        await using FileStream stream = new(
            Path.GetFullPath(filePath),
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 64 * 1024,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        byte[] hash = await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }
}

public sealed record HashVerificationResult(
    bool IsValid,
    bool WasVerified,
    string? ExpectedSha256,
    string? ActualSha256,
    string? FailureReason);
