using Trackdub.Domain;
using Trackdub.Domain.Mixing;
using Trackdub.Contracts;

namespace Trackdub.Application.Transcripts;

public sealed record PreviewMixStageRequest(
    Guid ProjectId,
    double StartSeconds,
    double EndSeconds,
    double SourceGainDb = 0d,
    double DubbedSpeechGainDb = 0d,
    double? DuckingGainDb = null,
    bool RestoreOriginalPan = false,
    bool ApplyTimbrePolish = true,
    double BackgroundMediaGainDb = 0d);

public sealed record PreviewMixStageResult(
    StageRunRecord StageRun,
    string PreviewAudioRelativePath,
    MixPlan MixPlan,
    double DurationSeconds,
    IReadOnlyList<MixPlanWarning> Warnings);

public enum ExportSubtitleFormat
{
    Srt = 0,
    Vtt = 1,
    Ass = 2
}

public enum ExportSubtitleSource
{
    Translated = 0,
    Transcript = 1,
    Bilingual = 2
}

/// <summary>
/// Request for an export stage run.
/// <see cref="RawSubtitleFormats"/> carries the caller's original subtitle-format request
/// (null when unspecified, an empty list when explicitly suppressed) before it is resolved into
/// <see cref="SubtitleFormats"/> via the transcript-state-dependent resolution. The export-resume
/// gate compares this raw value against the current run's snapshot so the null-default,
/// empty-list, and explicit-format cases stay distinct without depending on transcript state.
/// </summary>
public sealed record ExportStageRequest(
    Guid ProjectId,
    string OutputPath,
    IReadOnlyList<ExportSubtitleFormat> SubtitleFormats,
    ExportSubtitleSource SubtitleSource = ExportSubtitleSource.Translated,
    bool BurnInSubtitles = false,
    double TargetLufs = ExportLoudnessTargets.OnlineLufs,
    ExportOutputContainer Container = ExportOutputContainer.Mp4,
    double SourceGainDb = 0d,
    double DubbedSpeechGainDb = 0d,
    double? DuckingGainDb = null,
    bool MatchOriginalLoudness = false,
    bool RestoreOriginalPan = false,
    bool ApplyTimbrePolish = true,
    VideoEncoderPreference VideoEncoder = VideoEncoderPreference.Auto,
    IReadOnlyList<string>? RawSubtitleFormats = null,
    double BackgroundMediaGainDb = 0d);

public sealed record ExportStageResult(
    StageRunRecord StageRun,
    string OutputPath,
    string ManifestPath,
    string ExportAudioRelativePath,
    string ExportVideoRelativePath,
    IReadOnlyList<string> SubtitlePaths,
    IReadOnlyList<string> Warnings)
{
    /// <summary>
    /// Indicates that the export was blocked by a tier gate (e.g. duration limit on Free tier).
    /// When true, no encoding was attempted.
    /// </summary>
    public bool IsBlocked { get; private init; }

    /// <summary>
    /// The user-facing reason the export was blocked. Null when not blocked.
    /// </summary>
    public string? BlockedReason { get; private init; }

    /// <summary>
    /// Creates a blocked result indicating the export was rejected by a tier gate.
    /// </summary>
    public static ExportStageResult Blocked(string reason) =>
        new(
            StageRun: null!,
            OutputPath: string.Empty,
            ManifestPath: string.Empty,
            ExportAudioRelativePath: string.Empty,
            ExportVideoRelativePath: string.Empty,
            SubtitlePaths: [],
            Warnings: [])
        {
            IsBlocked = true,
            BlockedReason = reason
        };
}

public sealed record ExportFailureReport(
    Guid ProjectId,
    Guid StageRunId,
    DateTimeOffset CreatedAtUtc,
    IReadOnlyList<ExportFailureCause> Causes);

public sealed record ExportFailureCause(
    string Code,
    string Message,
    Guid? SegmentId = null,
    int? SegmentIndex = null);

public sealed class ExportStageException(
    string message,
    ExportFailureReport report,
    string? reportPath = null,
    Exception? innerException = null)
    : InvalidOperationException(message, innerException)
{
    public ExportFailureReport Report { get; } = report;

    public string? ReportPath { get; } = reportPath;
}
