using Trackdub.Contracts;
using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts;

namespace Trackdub.Application.Tests;

public sealed class ExportStageRequestBuilderTests
{
    [Fact]
    public void TryBuild_maps_mix_gains_pan_and_loudness_flags()
    {
        Guid projectId = Guid.NewGuid();
        var mix = new MixGainSettings(
            SourceGainDb: -3d,
            DubbedSpeechGainDb: 2d,
            DuckingGainDb: -12d,
            DuckingGainExplicit: true,
            RestoreOriginalPan: true)
        {
            BackgroundMediaGainDb = -7d
        };

        ExportStageRequest? request = ExportStageRequestBuilder.TryBuild(
            new ExportBuildInputs(
                projectId,
                @"C:\out\render.mp4",
                [ExportSubtitleFormat.Srt],
                ExportSubtitleSource.Translated,
                BurnInSubtitles: true,
                TargetLufs: -14d,
                ExportOutputContainer.Mp4,
                MatchOriginalLoudness: true,
                mix));

        Assert.NotNull(request);
        Assert.Equal(projectId, request.ProjectId);
        Assert.Equal(-3d, request.SourceGainDb);
        Assert.Equal(2d, request.DubbedSpeechGainDb);
        Assert.Equal(-12d, request.DuckingGainDb);
        Assert.Equal(-7d, request.BackgroundMediaGainDb);
        Assert.True(request.MatchOriginalLoudness);
        Assert.True(request.RestoreOriginalPan);
        Assert.True(request.BurnInSubtitles);
    }

    [Fact]
    public void TryBuild_returns_null_when_extension_does_not_match_container()
    {
        ExportStageRequest? request = ExportStageRequestBuilder.TryBuild(
            new ExportBuildInputs(
                Guid.NewGuid(),
                @"C:\out\render.mkv",
                [],
                ExportSubtitleSource.Translated,
                false,
                ExportLoudnessTargets.OnlineLufs,
                ExportOutputContainer.Mp4,
                false,
                new MixGainSettings()));

        Assert.Null(request);
    }

    [Fact]
    public void TryBuild_omits_ducking_when_not_explicit()
    {
        ExportStageRequest? request = ExportStageRequestBuilder.TryBuild(
            new ExportBuildInputs(
                Guid.NewGuid(),
                @"C:\out\render.mp4",
                [],
                ExportSubtitleSource.Translated,
                false,
                ExportLoudnessTargets.OnlineLufs,
                ExportOutputContainer.Mp4,
                false,
                new MixGainSettings(DuckingGainDb: -9d, DuckingGainExplicit: false)));

        Assert.NotNull(request);
        Assert.Null(request.DuckingGainDb);
    }

    [Fact]
    public void BuildSubtitleFormats_respects_selected_flags()
    {
        IReadOnlyList<ExportSubtitleFormat> formats = ExportStageRequestBuilder.BuildSubtitleFormats(
            exportSrt: true,
            exportVtt: false,
            exportAss: true);

        Assert.Equal(2, formats.Count);
        Assert.Contains(ExportSubtitleFormat.Srt, formats);
        Assert.Contains(ExportSubtitleFormat.Ass, formats);
        Assert.DoesNotContain(ExportSubtitleFormat.Vtt, formats);
    }
}
