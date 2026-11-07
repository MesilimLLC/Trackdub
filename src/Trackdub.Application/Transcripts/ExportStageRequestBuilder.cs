using Trackdub.Contracts;
using Trackdub.Contracts.Projects;

namespace Trackdub.Application.Transcripts;

public static class ExportStageRequestBuilder
{
    public static ExportStageRequest? TryBuild(ExportBuildInputs inputs)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        if (inputs.ProjectId == Guid.Empty ||
            string.IsNullOrWhiteSpace(inputs.OutputPath))
        {
            return null;
        }

        ExportOutputContainer container = inputs.Container is ExportOutputContainer.Mkv
            ? ExportOutputContainer.Mkv
            : ExportOutputContainer.Mp4;

        if (inputs.ValidateOutputExtension &&
            !OutputPathMatchesContainer(inputs.OutputPath, container))
        {
            return null;
        }

        IReadOnlyList<ExportSubtitleFormat> subtitleFormats = inputs.SubtitleFormats ?? [];
        MixGainSettings mix = inputs.Mix;

        return new ExportStageRequest(
            inputs.ProjectId,
            inputs.OutputPath,
            subtitleFormats,
            inputs.SubtitleSource,
            inputs.BurnInSubtitles,
            ExportLoudnessTargets.NormalizeTargetLufs(inputs.TargetLufs),
            container,
            mix.SourceGainDb,
            mix.DubbedSpeechGainDb,
            mix.DuckingGainExplicit ? mix.DuckingGainDb : null,
            inputs.MatchOriginalLoudness,
            mix.RestoreOriginalPan,
            mix.ApplyTimbrePolish,
            inputs.VideoEncoder,
            BackgroundMediaGainDb: mix.BackgroundMediaGainDb);
    }

    public static MixGainSettings ToMixGainSettings(ExportStageRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        return new MixGainSettings(
            request.SourceGainDb,
            request.DubbedSpeechGainDb,
            request.DuckingGainDb,
            DuckingGainExplicit: request.DuckingGainDb is not null)
        {
            BackgroundMediaGainDb = request.BackgroundMediaGainDb
        };
    }

    public static IReadOnlyList<ExportSubtitleFormat> BuildSubtitleFormats(
        bool exportSrt,
        bool exportVtt,
        bool exportAss)
    {
        var formats = new List<ExportSubtitleFormat>();
        if (exportSrt)
        {
            formats.Add(ExportSubtitleFormat.Srt);
        }

        if (exportVtt)
        {
            formats.Add(ExportSubtitleFormat.Vtt);
        }

        if (exportAss)
        {
            formats.Add(ExportSubtitleFormat.Ass);
        }

        return formats;
    }

    public static ExportOutputContainer ResolveContainer(string? containerKey) =>
        string.Equals(containerKey, StudioExportSettings.MkvContainer, StringComparison.OrdinalIgnoreCase)
            ? ExportOutputContainer.Mkv
            : ExportOutputContainer.Mp4;

    public static ExportSubtitleSource ResolveSubtitleSource(string? sourceKey) =>
        sourceKey switch
        {
            var value when string.Equals(value, StudioExportSettings.TranscriptSubtitleSource, StringComparison.OrdinalIgnoreCase) =>
                ExportSubtitleSource.Transcript,
            var value when string.Equals(value, StudioExportSettings.BilingualSubtitleSource, StringComparison.OrdinalIgnoreCase) =>
                ExportSubtitleSource.Bilingual,
            _ => ExportSubtitleSource.Translated
        };

    public static string NormalizeContainerKey(string? value) =>
        string.Equals(value, StudioExportSettings.MkvContainer, StringComparison.OrdinalIgnoreCase)
            ? StudioExportSettings.MkvContainer
            : StudioExportSettings.Mp4Container;

    public static string NormalizeSubtitleSourceKey(string? value) =>
        string.Equals(value, StudioExportSettings.TranscriptSubtitleSource, StringComparison.OrdinalIgnoreCase)
            ? StudioExportSettings.TranscriptSubtitleSource
            : string.Equals(value, StudioExportSettings.BilingualSubtitleSource, StringComparison.OrdinalIgnoreCase)
                ? StudioExportSettings.BilingualSubtitleSource
                : StudioExportSettings.TranslatedSubtitleSource;

    private static bool OutputPathMatchesContainer(string outputPath, ExportOutputContainer container)
    {
        string expectedExtension = container == ExportOutputContainer.Mkv ? ".mkv" : ".mp4";
        return string.Equals(Path.GetExtension(outputPath), expectedExtension, StringComparison.OrdinalIgnoreCase);
    }
}

public sealed record ExportBuildInputs(
    Guid ProjectId,
    string OutputPath,
    IReadOnlyList<ExportSubtitleFormat> SubtitleFormats,
    ExportSubtitleSource SubtitleSource,
    bool BurnInSubtitles,
    double TargetLufs,
    ExportOutputContainer Container,
    bool MatchOriginalLoudness,
    MixGainSettings Mix,
    VideoEncoderPreference VideoEncoder = VideoEncoderPreference.Auto,
    bool ValidateOutputExtension = true);
