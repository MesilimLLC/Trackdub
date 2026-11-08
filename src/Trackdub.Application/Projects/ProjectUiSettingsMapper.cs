using Trackdub.Contracts;
using Trackdub.Contracts.Transcripts;

namespace Trackdub.Application.Projects;

public static class ProjectUiSettingsMapper
{
    public static ProjectUiSettings CreateProjectUiSettings(
        MixGainSettings mix,
        ProjectExportSettings? export,
        IReadOnlyList<ProjectTimelineMediaPlacement>? timelinePlacements = null,
        string? selectedTranslationTargetLanguage = null,
        ProjectPipelineSettings? pipeline = null,
        SegmentStageRunMap? segmentStageRuns = null,
        IReadOnlyDictionary<Guid, bool>? voiceCloneBySpeakerId = null) =>
        new(
            Mix: mix.ToProjectMixSettings(),
            Export: export?.Normalize(),
            TimelinePlacements: timelinePlacements?
                .Select(static placement => placement.Normalize())
                .ToArray(),
            SelectedTranslationTargetLanguage: selectedTranslationTargetLanguage,
            Pipeline: pipeline?.Normalize(),
            SegmentStageRuns: segmentStageRuns?.Normalize(),
            VoiceCloneBySpeakerId: VoiceCloneByIdNormalizer.NormalizeVoiceCloneBySpeakerId(voiceCloneBySpeakerId));

    public static ProjectPipelineSettings ReadPipelineSettings(ProjectUiSettings? settings) =>
        settings?.Pipeline?.Normalize() ?? new ProjectPipelineSettings();

    public static MixGainSettings ReadMixSettings(ProjectUiSettings? settings) =>
        MixGainSettings.FromProjectMixSettings(settings?.Mix);

    public static ProjectExportSettings? ReadExportSettings(ProjectUiSettings? settings) =>
        settings?.Export?.Normalize();

    public static IReadOnlyList<ProjectTimelineMediaPlacement> ReadTimelinePlacements(
        ProjectUiSettings? settings) =>
        settings?.TimelinePlacements?
            .Select(static placement => placement.Normalize())
            .ToArray()
        ?? [];

    public static ProjectExportSettings FromStudioExportSettings(StudioExportSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        var formats = new List<ExportSubtitleFormat>();
        if (settings.Srt)
        {
            formats.Add(ExportSubtitleFormat.Srt);
        }

        if (settings.Vtt)
        {
            formats.Add(ExportSubtitleFormat.Vtt);
        }

        if (settings.Ass)
        {
            formats.Add(ExportSubtitleFormat.Ass);
        }

        ExportSubtitleSource subtitleSource = settings.SubtitleSource switch
        {
            var value when string.Equals(value, StudioExportSettings.TranscriptSubtitleSource, StringComparison.OrdinalIgnoreCase) =>
                ExportSubtitleSource.Transcript,
            var value when string.Equals(value, StudioExportSettings.BilingualSubtitleSource, StringComparison.OrdinalIgnoreCase) =>
                ExportSubtitleSource.Bilingual,
            _ => ExportSubtitleSource.Translated
        };

        ExportOutputContainer container = string.Equals(
            settings.Container,
            StudioExportSettings.MkvContainer,
            StringComparison.OrdinalIgnoreCase)
            ? ExportOutputContainer.Mkv
            : ExportOutputContainer.Mp4;

        return new ProjectExportSettings(
            formats,
            subtitleSource,
            settings.BurnInSubtitles,
            settings.TargetLufs,
            container,
            settings.MatchOriginalLoudness,
            VideoEncoderPreferenceSettings.FromKey(settings.VideoEncoder));
    }

    public static StudioExportSettings ToStudioExportSettings(ProjectExportSettings export)
    {
        ArgumentNullException.ThrowIfNull(export);
        ProjectExportSettings normalized = export.Normalize();
        IReadOnlyList<ExportSubtitleFormat> formats = normalized.SubtitleFormats ?? [];
        string container = normalized.Container == ExportOutputContainer.Mkv
            ? StudioExportSettings.MkvContainer
            : StudioExportSettings.Mp4Container;
        string subtitleSource = normalized.SubtitleSource switch
        {
            ExportSubtitleSource.Transcript => StudioExportSettings.TranscriptSubtitleSource,
            ExportSubtitleSource.Bilingual => StudioExportSettings.BilingualSubtitleSource,
            _ => StudioExportSettings.TranslatedSubtitleSource
        };

        return new StudioExportSettings(
            formats.Contains(ExportSubtitleFormat.Srt),
            formats.Contains(ExportSubtitleFormat.Vtt),
            formats.Contains(ExportSubtitleFormat.Ass),
            normalized.BurnInSubtitles,
            normalized.TargetLufs,
            container,
            subtitleSource,
            normalized.MatchOriginalLoudness,
            VideoEncoderPreferenceSettings.ToKey(normalized.VideoEncoder));
    }

    public static PreviewMixStageRequest? TryBuildPreviewMixRequest(PreviewMixBuildInputs inputs)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        if (inputs.ProjectId == Guid.Empty ||
            inputs.EndSeconds <= inputs.StartSeconds)
        {
            return null;
        }

        MixGainSettings mix = inputs.Mix;
        return new PreviewMixStageRequest(
            inputs.ProjectId,
            inputs.StartSeconds,
            inputs.EndSeconds,
            mix.SourceGainDb,
            mix.DubbedSpeechGainDb,
            mix.DuckingGainExplicit ? mix.DuckingGainDb : null,
            mix.RestoreOriginalPan,
            mix.ApplyTimbrePolish,
            mix.BackgroundMediaGainDb);
    }
}

public sealed record PreviewMixBuildInputs(
    Guid ProjectId,
    double StartSeconds,
    double EndSeconds,
    MixGainSettings Mix);
