namespace Trackdub.Contracts.Pipeline;

public interface IStemSeparationEngine
{
    Task<StemSeparationResult> SeparateAsync(
        StemSeparationRequest request,
        IProgress<StemSeparationProgress>? progress,
        CancellationToken cancellationToken);
}

public sealed record StemSeparationRequest(
    string SourceAudioPath,
    string VocalsOutputPath,
    string AmbianceOutputPath,
    string? PreferredModelAlias = null,
    IReadOnlyDictionary<string, string>? Metadata = null,
    string? MusicOutputPath = null,
    string? SoundEffectsOutputPath = null,
    IReadOnlyDictionary<string, string>? RawStemOutputPaths = null,
    string? PreferredExecutionProvider = null,
    bool RequirePreferredExecutionProvider = false,
    string? PreferredModelVariantAlias = null);

public sealed partial record StemSeparationResult(
    double DurationSeconds,
    int SampleRate,
    int ChannelCount,
    IReadOnlyDictionary<string, string>? Metadata = null);

public sealed record StemSeparationProgress(
    int CompletedChunks,
    int TotalChunks,
    double ChunkStartSeconds,
    double ChunkEndSeconds,
    bool IsFinalizing = false,
    bool IsPersistingArtifacts = false)
{
    public double PercentComplete => TotalChunks <= 0
        ? 0d
        : Math.Clamp(CompletedChunks / (double)TotalChunks, 0d, 1d);
}
