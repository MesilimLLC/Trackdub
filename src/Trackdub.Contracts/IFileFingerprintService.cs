namespace Trackdub.Contracts;

public interface IFileFingerprintService
{
    Task<FileFingerprint> ComputeAsync(string path, CancellationToken cancellationToken);

    /// <summary>
    /// Drops any cached fingerprint for <paramref name="path"/> so the next computation rehashes
    /// the file even when its length and write time are unchanged. The rehash guarantee applies to
    /// the instance that performed the invalidation: it drops its own cached fingerprint and keeps
    /// its own invalidation marker, and other instances that already hold the same fingerprint are
    /// not affected. Implementations that cache fingerprints must override this; the default is a
    /// no-op for implementations that do not cache and therefore have nothing to invalidate.
    /// </summary>
    void Invalidate(string path)
    {
    }
}

public sealed record FileFingerprint(
    string Sha256,
    long SizeBytes,
    DateTimeOffset LastWriteTimeUtc);
