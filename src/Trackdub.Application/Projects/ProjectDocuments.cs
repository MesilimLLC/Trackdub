using Trackdub.Contracts;
using Trackdub.Contracts.Transcripts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Projects;

namespace Trackdub.Application.Projects;

public static class ProjectDocumentVersions
{
    public const int CurrentProjectSchemaVersion = 2;
}

file static class ProjectDocumentLanguageCodes
{
    internal static string? NormalizeLanguageCode(string? languageCode) =>
        string.IsNullOrWhiteSpace(languageCode)
            ? null
            : languageCode.Trim().ToLowerInvariant();
}

public sealed record ProjectManifest(
    Guid ProjectId,
    string Name,
    int SchemaVersion,
    DateTimeOffset CreatedAtUtc,
    DateTimeOffset UpdatedAtUtc,
    string? TranscriptLanguage = null,
    ProjectUiSettings? UiSettings = null)
{
    public static ProjectManifest FromProject(
        TrackdubProject project,
        string? transcriptLanguage = null,
        ProjectUiSettings? uiSettings = null) =>
        new(
            project.Id,
            project.Name,
            SchemaVersion: ProjectDocumentVersions.CurrentProjectSchemaVersion,
            project.CreatedAtUtc,
            project.UpdatedAtUtc,
            ProjectDocumentLanguageCodes.NormalizeLanguageCode(transcriptLanguage),
            uiSettings?.Normalize());

    public ProjectManifest WithTranscriptLanguage(string? transcriptLanguage) =>
        this with
        {
            SchemaVersion = ProjectDocumentVersions.CurrentProjectSchemaVersion,
            TranscriptLanguage = ProjectDocumentLanguageCodes.NormalizeLanguageCode(transcriptLanguage)
        };

    public ProjectManifest WithUiSettings(ProjectUiSettings? uiSettings) =>
        this with
        {
            SchemaVersion = ProjectDocumentVersions.CurrentProjectSchemaVersion,
            UpdatedAtUtc = DateTimeOffset.UtcNow,
            UiSettings = uiSettings?.Normalize()
        };
}

public sealed record ProjectPipelineSettings
{
    public bool EnableAsrTextRefinement { get; init; }

    public bool EnableSpeechAudioPreparation { get; init; }

    public bool EnableSpeakerDiarization { get; init; } = true;

    public ProjectPipelineSettings Normalize() => this;
}

/// <summary>
/// Per-segment stage-run attribution for mixed translation/ASR provider runs.
/// </summary>
public sealed record SegmentStageRunMap(
    IReadOnlyDictionary<int, Guid>? Translation = null,
    IReadOnlyDictionary<int, Guid>? Asr = null)
{
    public SegmentStageRunMap Normalize() =>
        this with
        {
            Translation = NormalizeMap(Translation),
            Asr = NormalizeMap(Asr),
        };

    private static IReadOnlyDictionary<int, Guid>? NormalizeMap(IReadOnlyDictionary<int, Guid>? map)
    {
        if (map is null || map.Count == 0)
        {
            return null;
        }

        Dictionary<int, Guid> normalized = map
            .Where(static pair => pair.Key >= 0 && pair.Value != Guid.Empty)
            .ToDictionary(static pair => pair.Key, static pair => pair.Value);
        return normalized.Count == 0 ? null : normalized;
    }
}

internal static class VoiceCloneByIdNormalizer
{
    internal static IReadOnlyDictionary<Guid, bool>? NormalizeVoiceCloneBySpeakerId(IReadOnlyDictionary<Guid, bool>? map)
    {
        if (map is null || map.Count == 0)
        {
            return null;
        }

        Dictionary<Guid, bool> normalized = map
            .Where(static pair => pair.Key != Guid.Empty)
            .ToDictionary(static pair => pair.Key, static pair => pair.Value);
        return normalized.Count == 0 ? null : normalized;
    }
}

public sealed record ProjectUiSettings(
    ProjectMixSettings? Mix = null,
    ProjectExportSettings? Export = null,
    IReadOnlyList<ProjectTimelineMediaPlacement>? TimelinePlacements = null,
    string? SelectedTranslationTargetLanguage = null,
    ProjectPipelineSettings? Pipeline = null,
    SegmentStageRunMap? SegmentStageRuns = null,
    IReadOnlyDictionary<Guid, bool>? VoiceCloneBySpeakerId = null)
{
    public ProjectUiSettings Normalize() =>
        this with
        {
            Mix = Mix?.Normalize(),
            Export = Export?.Normalize(),
            TimelinePlacements = TimelinePlacements?
                .Select(static placement => placement.Normalize())
                .ToArray(),
            SelectedTranslationTargetLanguage = ProjectDocumentLanguageCodes.NormalizeLanguageCode(SelectedTranslationTargetLanguage),
            Pipeline = Pipeline?.Normalize(),
            SegmentStageRuns = SegmentStageRuns?.Normalize(),
            VoiceCloneBySpeakerId = VoiceCloneByIdNormalizer.NormalizeVoiceCloneBySpeakerId(VoiceCloneBySpeakerId),
        };
}

public sealed record ProjectMixSettings(
    double SourceGainDb = 0d,
    double DubbedSpeechGainDb = 0d,
    double? DuckingGainDb = null,
    bool DuckingGainExplicit = false,
    bool RestoreOriginalPan = false,
    bool ApplyTimbrePolish = true)
{
    public ProjectMixSettings Normalize() =>
        this with
        {
            SourceGainDb = NormalizeGainDb(SourceGainDb, 0d),
            DubbedSpeechGainDb = NormalizeGainDb(DubbedSpeechGainDb, 0d),
            DuckingGainDb = DuckingGainExplicit && DuckingGainDb is double duckingGainDb
                ? NormalizeGainDb(duckingGainDb, 0d)
                : null,
            DuckingGainExplicit = DuckingGainExplicit && DuckingGainDb is not null
        };

    private static double NormalizeGainDb(double gainDb, double fallback) =>
        double.IsFinite(gainDb)
            ? Math.Clamp(gainDb, -96d, 24d)
            : fallback;
}

public sealed record ProjectExportSettings(
    IReadOnlyList<ExportSubtitleFormat>? SubtitleFormats = null,
    ExportSubtitleSource SubtitleSource = ExportSubtitleSource.Translated,
    bool BurnInSubtitles = false,
    double TargetLufs = ExportLoudnessTargets.OnlineLufs,
    ExportOutputContainer Container = ExportOutputContainer.Mp4,
    bool MatchOriginalLoudness = false,
    VideoEncoderPreference VideoEncoder = VideoEncoderPreference.Auto)
{
    public ProjectExportSettings Normalize()
    {
        ExportSubtitleSource subtitleSource = SubtitleSource is ExportSubtitleSource.Translated
            or ExportSubtitleSource.Transcript
            or ExportSubtitleSource.Bilingual
                ? SubtitleSource
                : ExportSubtitleSource.Translated;
        ExportOutputContainer container = Container is ExportOutputContainer.Mp4 or ExportOutputContainer.Mkv
            ? Container
            : ExportOutputContainer.Mp4;
        ExportSubtitleFormat[] subtitleFormats = (SubtitleFormats ?? [ExportSubtitleFormat.Srt])
            .Where(static format => format is ExportSubtitleFormat.Srt or ExportSubtitleFormat.Vtt or ExportSubtitleFormat.Ass)
            .Distinct()
            .ToArray();

        VideoEncoderPreference videoEncoder = VideoEncoder is VideoEncoderPreference.Auto
            or VideoEncoderPreference.Software
            or VideoEncoderPreference.Nvenc
            or VideoEncoderPreference.Qsv
            or VideoEncoderPreference.Amf
            or VideoEncoderPreference.VideoToolbox
            or VideoEncoderPreference.Vaapi
                ? VideoEncoder
                : VideoEncoderPreference.Auto;

        return this with
        {
            SubtitleFormats = subtitleFormats,
            SubtitleSource = subtitleSource,
            TargetLufs = ExportLoudnessTargets.NormalizeTargetLufs(TargetLufs),
            Container = container,
            VideoEncoder = videoEncoder
        };
    }
}

public sealed record SourceMediaReference(
    string OriginalPath,
    string OriginalFileName,
    FileFingerprint Fingerprint,
    MediaProbeSnapshot Probe,
    DateTimeOffset CapturedAtUtc);

public enum SourceMediaStatus
{
    Unknown = 0,
    Available = 1,
    Missing = 2,
    Changed = 3
}
