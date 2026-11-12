using Trackdub.Contracts;

namespace Trackdub.Inference.Onnx.Pool;

/// <summary>
/// Process-wide seam that lets pooled-session identity reuse a shared, persisted file
/// fingerprint instead of re-hashing the model from disk on every pool-key construction.
/// The provider is unconfigured by default (<see cref="Current"/> is <see langword="null"/>),
/// so production wiring is opt-in and the default path keeps the streaming file hash.
/// </summary>
public static class SessionPoolContentHashProvider
{
    /// <summary>
    /// The active content-hash provider, or <see langword="null"/> when no shared fingerprint
    /// service is configured. The delegate returns the model's SHA-256 hex digest, or
    /// <see langword="null"/> when the shared fingerprint is unavailable.
    /// </summary>
    public static Func<string, CancellationToken, Task<string?>>? Current => Volatile.Read(ref _current);
    private static Func<string, CancellationToken, Task<string?>>? _current;

    /// <summary>
    /// Installs <paramref name="fingerprintService"/> as the shared content-hash provider, or
    /// clears the provider when <paramref name="fingerprintService"/> is <see langword="null"/>.
    /// </summary>
    public static void Configure(IFileFingerprintService? fingerprintService) =>
        Volatile.Write(ref _current, fingerprintService is null ? null : (path, cancellationToken) => HashCoreAsync(fingerprintService, path, cancellationToken));

    /// <summary>Removes the process-wide provider.</summary>
    public static void Reset() => Volatile.Write(ref _current, null);

    private static async Task<string?> HashCoreAsync(
        IFileFingerprintService fingerprintService,
        string path,
        CancellationToken cancellationToken)
    {
        try
        {
            return (await fingerprintService.ComputeAsync(path, cancellationToken).ConfigureAwait(false)).Sha256;
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FileNotFoundException or DirectoryNotFoundException)
        {
            System.Diagnostics.Trace.TraceWarning($"Shared model fingerprint failed for '{path}': {ex.Message}");
            return null;
        }
    }
}
