using Trackdub.Domain.Artifacts;

namespace Trackdub.Domain.Mixing;

public sealed record MixPlan
{
    private string sourceAudioRelativePath = string.Empty;
    private string? originalMixAudioRelativePath;
    private int outputChannelCount = 1;

    public MixPlan(
        Guid ProjectId,
        Guid? MediaAssetId,
        ArtifactKind SourceAudioKind,
        string SourceAudioRelativePath,
        double SourceGainDb,
        double DubbedSpeechGainDb,
        double DuckingGainDb,
        double DuckingLeadSeconds,
        double DuckingTailSeconds,
        DateTimeOffset CreatedAtUtc,
        IReadOnlyList<MixSpeechClip> SpeechClips,
        IReadOnlyList<MixDuckRegion> DuckingRegions,
        IReadOnlyList<MixPlanWarning> Warnings,
        string? OriginalMixAudioRelativePath = null,
        int OutputChannelCount = 1,
        bool RestoreOriginalPan = false,
        bool ApplyTimbrePolish = true,
        string? VocalStemRelativePath = null,
        IReadOnlyList<MixMediaClip>? MediaClips = null)
    {
        this.ProjectId = ProjectId;
        this.MediaAssetId = MediaAssetId;
        this.SourceAudioKind = SourceAudioKind;
        this.SourceAudioRelativePath = SourceAudioRelativePath;
        this.SourceGainDb = SourceGainDb;
        this.DubbedSpeechGainDb = DubbedSpeechGainDb;
        this.DuckingGainDb = DuckingGainDb;
        this.DuckingLeadSeconds = DuckingLeadSeconds;
        this.DuckingTailSeconds = DuckingTailSeconds;
        this.CreatedAtUtc = CreatedAtUtc;
        this.SpeechClips = SpeechClips;
        this.DuckingRegions = DuckingRegions;
        this.Warnings = Warnings;
        this.OriginalMixAudioRelativePath = OriginalMixAudioRelativePath ?? string.Empty;
        this.OutputChannelCount = OutputChannelCount;
        this.RestoreOriginalPan = RestoreOriginalPan;
        this.ApplyTimbrePolish = ApplyTimbrePolish;
        this.VocalStemRelativePath = VocalStemRelativePath;
        this.MediaClips = MediaClips ?? [];
    }

    public Guid ProjectId { get; init; }

    public Guid? MediaAssetId { get; init; }

    public ArtifactKind SourceAudioKind { get; init; }

    public string SourceAudioRelativePath
    {
        get => sourceAudioRelativePath;
        init
        {
            sourceAudioRelativePath = value;
            originalMixAudioRelativePath = NormalizeOriginalMixAudioRelativePath(originalMixAudioRelativePath, sourceAudioRelativePath);
        }
    }

    public double SourceGainDb { get; init; }

    public double DubbedSpeechGainDb { get; init; }

    public double DuckingGainDb { get; init; }

    public double DuckingLeadSeconds { get; init; }

    public double DuckingTailSeconds { get; init; }

    public DateTimeOffset CreatedAtUtc { get; init; }

    public IReadOnlyList<MixSpeechClip> SpeechClips { get; init; }

    public IReadOnlyList<MixDuckRegion> DuckingRegions { get; init; }

    public IReadOnlyList<MixPlanWarning> Warnings { get; init; }

    public string OriginalMixAudioRelativePath
    {
        get => originalMixAudioRelativePath ?? SourceAudioRelativePath;
        init => originalMixAudioRelativePath = NormalizeOriginalMixAudioRelativePath(value, SourceAudioRelativePath);
    }

    public int OutputChannelCount
    {
        get => outputChannelCount;
        init => outputChannelCount = NormalizeOutputChannelCount(value);
    }

    public bool RestoreOriginalPan { get; init; }

    public bool ApplyTimbrePolish { get; init; }

    public string? VocalStemRelativePath { get; init; }

    public IReadOnlyList<MixMediaClip> MediaClips { get; init; }

    private static string? NormalizeOriginalMixAudioRelativePath(string? value, string sourceAudioRelativePath)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        string normalizedValue = NormalizeRelativePath(value);
        string normalizedSource = NormalizeRelativePath(sourceAudioRelativePath);
        return string.Equals(normalizedValue, normalizedSource, StringComparison.OrdinalIgnoreCase) ? null : normalizedValue;
    }

    private static string NormalizeRelativePath(string value) =>
        value.Trim().Replace('\\', '/');

    private static int NormalizeOutputChannelCount(int value) =>
        value >= 2 ? 2 : 1;
}

/// <summary>Project-owned audio placed independently of generated dialogue.</summary>
public sealed record MixMediaClip(string AudioRelativePath, double StartSeconds, double DurationSeconds, double GainDb = 0d);

public sealed record MixSpeechClip(
    int SegmentIndex,
    Guid SegmentId,
    double StartSeconds,
    double EndSeconds,
    Guid? TakeId,
    Guid? ArtifactId,
    string? TakeRelativePath,
    double? TakeDurationSeconds,
    bool IsSilentGap,
    string? WarningMessage);

public sealed record MixDuckRegion(
    int SegmentIndex,
    Guid SegmentId,
    double StartSeconds,
    double EndSeconds,
    double GainDb);

public sealed record MixPlanWarning(
    int SegmentIndex,
    Guid SegmentId,
    string Message,
    MixPlanWarningCode Code = MixPlanWarningCode.InvalidTake)
{
    public string SegmentReference => $"Segment {SegmentIndex + 1}";
}

public enum MixPlanWarningCode
{
    InvalidTake = 0,
    MissingTake,
    StaleTake,
    MissingTakeArtifact,
    LipSyncArtifactMissing,
    TakeExceedsSourceAudio
}
