using Trackdub.Contracts;

namespace Trackdub.TestDoubles;

public sealed class FakeFileFingerprintService : IFileFingerprintService
{
    private readonly Dictionary<string, FileFingerprint> fingerprints = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<string> requestedPaths = [];
    private readonly List<string> invalidatedPaths = [];

    public FakeFileFingerprintService()
    {
    }

    public FakeFileFingerprintService(FileFingerprint defaultFingerprint)
    {
        DefaultFingerprint = defaultFingerprint;
    }

    public FileFingerprint? DefaultFingerprint { get; set; }

    public IReadOnlyList<string> RequestedPaths => requestedPaths;

    public IReadOnlyList<string> InvalidatedPaths => invalidatedPaths;

    public void SetFingerprint(string path, FileFingerprint fingerprint)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        fingerprints[path] = fingerprint;
    }

    public Task<FileFingerprint> ComputeAsync(string path, CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        cancellationToken.ThrowIfCancellationRequested();

        requestedPaths.Add(path);
        if (fingerprints.TryGetValue(path, out FileFingerprint? configuredFingerprint))
        {
            return Task.FromResult(configuredFingerprint);
        }

        FileFingerprint fingerprint = DefaultFingerprint ?? CreateDefaultFingerprint(path);
        return Task.FromResult(fingerprint);
    }

    private static FileFingerprint CreateDefaultFingerprint(string path) =>
        new($"hash-{Path.GetFileName(path)}", 0, DateTimeOffset.UnixEpoch);

    public void Invalidate(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        fingerprints.Remove(path);
        invalidatedPaths.Add(path);
    }
}
