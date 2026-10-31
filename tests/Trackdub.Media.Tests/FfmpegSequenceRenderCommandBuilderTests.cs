using Trackdub.Contracts;
using Trackdub.Contracts.Editing;
using Trackdub.Domain.Editing;
using Trackdub.Media.Muxing;
using Trackdub.Media.Process;

namespace Trackdub.Media.Tests;

public sealed class FfmpegSequenceRenderCommandBuilderTests
{
    [Fact]
    public void BuildArguments_trims_and_concatenates_linked_video_and_audio_in_sequence_order()
    {
        SequenceExportPlan plan = CreatePlan(withAudio: true);
        VideoEncodeProfile softwareProfile = FfmpegVideoEncoderSelector.Resolve(
            VideoEncoderPreference.Software,
            FfmpegVideoEncoderSnapshot.Empty);

        IReadOnlyList<string> arguments = FfmpegSequenceRenderCommandBuilder.BuildArguments(
            plan,
            softwareProfile,
            "sequence.mp4");

        string[] argumentArray = [.. arguments];
        string filterGraph = arguments[Array.IndexOf(argumentArray, "-filter_complex") + 1];
        Assert.Equal(4, arguments.Count(value => value == "-i"));
        Assert.Contains("[0:v:0]setpts=PTS-STARTPTS,trim=start=0:end=6", filterGraph);
        Assert.Contains("[1:v:0]setpts=PTS-STARTPTS,trim=start=3:end=8", filterGraph);
        Assert.Contains("[2:a:0]asetpts=PTS-STARTPTS,atrim=start=0:end=6", filterGraph);
        Assert.Contains("[3:a:0]asetpts=PTS-STARTPTS,atrim=start=3:end=8", filterGraph);
        Assert.Contains("scale=ceil(iw*1.5/2)*2:ceil(ih*1.5/2)*2", filterGraph);
        Assert.Contains("crop=1920:1080:(iw-1920)*0.75:(ih-1080)*0.25", filterGraph);
        Assert.Contains("concat=n=2:v=1:a=1[vconcat][aout]", filterGraph);
        Assert.Contains("[vconcat]null[vout]", filterGraph);
        Assert.Contains("-map", arguments);
        Assert.Contains("[aout]", arguments);
        Assert.Contains("aac", arguments);
        Assert.Contains("DUBBED_BY=Trackdub", arguments);
        Assert.Contains("+faststart", arguments);
    }

    [Fact]
    public void BuildArguments_emits_video_only_output_when_all_audio_paths_are_omitted()
    {
        SequenceExportPlan plan = CreatePlan(withAudio: false) with
        {
            Container = ExportOutputContainer.Mkv,
            BurnInSubtitlePath = "sequence.srt",
            RequiresWatermark = true,
            WatermarkFontPath = "font.ttf"
        };
        VideoEncodeProfile softwareProfile = FfmpegVideoEncoderSelector.Resolve(
            VideoEncoderPreference.Software,
            FfmpegVideoEncoderSnapshot.Empty);

        IReadOnlyList<string> arguments = FfmpegSequenceRenderCommandBuilder.BuildArguments(
            plan,
            softwareProfile,
            "sequence.mkv");

        string[] argumentArray = [.. arguments];
        string filterGraph = arguments[Array.IndexOf(argumentArray, "-filter_complex") + 1];
        Assert.Contains("concat=n=2:v=1:a=0[vconcat]", filterGraph);
        Assert.Contains("subtitles='", filterGraph);
        Assert.Contains("drawtext=fontfile=", filterGraph);
        Assert.DoesNotContain("[aout]", arguments);
        Assert.DoesNotContain("-c:a", arguments);
        Assert.Contains("libx264", arguments);
        Assert.DoesNotContain("libopus", arguments);
    }

    private static SequenceExportPlan CreatePlan(bool withAudio)
    {
        Guid projectId = Guid.NewGuid();
        Guid firstSourceId = Guid.NewGuid();
        Guid secondSourceId = Guid.NewGuid();
        Guid firstPlacementId = Guid.NewGuid();
        Guid secondPlacementId = Guid.NewGuid();
        var sequence = new SequenceDocument(
            SequenceDocument.CurrentVersion,
            projectId,
            new SequenceFrame(1920, 1080),
            [
                new SequenceSource(firstSourceId, projectId, Guid.NewGuid(), null, 20, 1920, 1080),
                new SequenceSource(secondSourceId, Guid.NewGuid(), Guid.NewGuid(), "sequence-sources/second", 20, 1280, 720)
            ],
            [
                new SequencePlacement(firstPlacementId, firstSourceId, 0, 6, new SequenceCrop()),
                new SequencePlacement(secondPlacementId, secondSourceId, 3, 8, new SequenceCrop(1.5, 0.75, 0.25))
            ]);

        return new SequenceExportPlan(
            sequence,
            new Dictionary<Guid, SequenceRenderPlacementAssets>
            {
                [firstPlacementId] = new("first source.mp4", withAudio ? "first mix.wav" : null),
                [secondPlacementId] = new("second source.mp4", withAudio ? "second mix.wav" : null)
            },
            "sequence.mp4",
            ExportOutputContainer.Mp4,
            AudioMode: withAudio ? SequenceExportAudioMode.Dubbed : SequenceExportAudioMode.None,
            VideoEncoder: VideoEncoderPreference.Software);
    }

    [Fact]
    public void BuildArguments_uses_source_audio_without_adding_dub_inputs()
    {
        SequenceExportPlan plan = CreatePlan(withAudio: false) with
        {
            AudioMode = SequenceExportAudioMode.OriginalSource,
        };
        VideoEncodeProfile softwareProfile = FfmpegVideoEncoderSelector.Resolve(
            VideoEncoderPreference.Software,
            FfmpegVideoEncoderSnapshot.Empty);

        IReadOnlyList<string> arguments = FfmpegSequenceRenderCommandBuilder.BuildArguments(
            plan,
            softwareProfile,
            "sequence.mp4");

        string[] argumentArray = [.. arguments];
        string filterGraph = arguments[Array.IndexOf(argumentArray, "-filter_complex") + 1];
        Assert.Equal(2, arguments.Count(value => value == "-i"));
        Assert.Contains("[0:a:0]asetpts=PTS-STARTPTS,atrim=start=0:end=6", filterGraph);
        Assert.Contains("[1:a:0]asetpts=PTS-STARTPTS,atrim=start=3:end=8", filterGraph);
        Assert.Contains("concat=n=2:v=1:a=1[vconcat][aout]", filterGraph);
        Assert.Contains("-map", arguments);
        Assert.Contains("[aout]", arguments);
        Assert.Contains("-c:a", arguments);
    }
}
