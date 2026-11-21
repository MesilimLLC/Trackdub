using Trackdub.Contracts;
using Trackdub.Application.Mixing;
using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Mixing;
using Trackdub.Domain.Projects;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Trackdub.Domain.Tts;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed partial class MixPlanBuilderTests
{
    // ---------------------------------------------------------------------------
    // Tail reflow: takes pulled back from the end of the source audio
    // ---------------------------------------------------------------------------

    [Fact]
    public void Build_pulls_the_last_take_into_slack_left_by_the_previous_one()
    {
        // The POC Chinese clip: segment 1 runs 26.75–40.15s, its take (after the 1.2x partial
        // stretch) is 16.23s, and the previous take ends at 20.86s, well before segment 1 starts.
        MixPlan plan = BuildTwoSegmentPlan(
            (0.14d, 26.75d, 20.72d),
            (26.75d, 40.15d, 16.23d),
            sourceSeconds: 40.171d);

        Assert.Empty(plan.Warnings);
        Assert.Equal(0.14d, plan.SpeechClips[0].StartSeconds, precision: 6);
        Assert.Equal(40.171d - 16.23d, plan.SpeechClips[1].StartSeconds, precision: 6);
        MixDuckRegion duck = plan.DuckingRegions[1];
        Assert.True(duck.StartSeconds < plan.SpeechClips[1].StartSeconds);
    }

    [Fact]
    public void Build_moves_a_take_more_than_the_silence_limit_within_continuous_speech()
    {
        // The second POC Chinese run: segment 1 starts where segment 0 ends, its 17.53s take has to
        // start 4.11s early, and the previous take ends at 18.25s. No silence is crossed, so the
        // 4.0s limit does not apply.
        MixPlan plan = BuildTwoSegmentPlan(
            (0.14d, 26.75d, 18.11d),
            (26.75d, 40.15d, 17.53d),
            sourceSeconds: 40.171d);

        Assert.Empty(plan.Warnings);
        Assert.Equal(40.171d - 17.53d, plan.SpeechClips[1].StartSeconds, precision: 6);
        Assert.Equal(0.14d, plan.SpeechClips[0].StartSeconds, precision: 6);
    }

    [Fact]
    public void Build_moves_the_previous_take_too_when_the_last_one_would_overlap_it()
    {
        // Last take needs to start at 7.0s; the previous take ends at 7.5s, so it moves back 0.5s.
        MixPlan plan = BuildTwoSegmentPlan(
            (2d, 8d, 5.5d),
            (8d, 10d, 3d),
            sourceSeconds: 10d);

        Assert.Empty(plan.Warnings);
        Assert.Equal(1.5d, plan.SpeechClips[0].StartSeconds, precision: 6);
        Assert.Equal(7d, plan.SpeechClips[1].StartSeconds, precision: 6);
    }

    [Fact]
    public void Build_does_not_overlap_a_previous_take_that_cannot_move_far_enough()
    {
        // The last take would need to start at 4.0s, but the previous take fills 0–5s and starts at 0,
        // so it cannot move. The last take stays at 5.0s rather than overlapping it, and the 1.0s
        // overhang is still reported.
        MixPlan plan = BuildTwoSegmentPlan(
            (0d, 5d, 5d),
            (5d, 10d, 6d),
            sourceSeconds: 10d);

        Assert.Equal(0d, plan.SpeechClips[0].StartSeconds, precision: 6);
        Assert.Equal(5d, plan.SpeechClips[1].StartSeconds, precision: 6);
        MixPlanWarning warning = Assert.Single(plan.Warnings);
        Assert.Equal(MixPlanWarningCode.TakeExceedsSourceAudio, warning.Code);
        Assert.Equal(1, warning.SegmentIndex);
    }

    [Fact]
    public void Build_moves_a_take_at_most_the_reflow_lead_into_silence()
    {
        // A long silence before the last segment: the take moves back only 4.0s, not into the gap.
        MixPlan plan = BuildTwoSegmentPlan(
            (0d, 2d, 2d),
            (20d, 25d, 10d),
            sourceSeconds: 25d);

        Assert.Equal(16d, plan.SpeechClips[1].StartSeconds, precision: 6);
        MixPlanWarning warning = Assert.Single(plan.Warnings);
        Assert.Contains("1 s past", warning.Message);
    }

    [Fact]
    public void Build_leaves_takes_that_fit_where_their_segments_start()
    {
        MixPlan plan = BuildTwoSegmentPlan(
            (1d, 5d, 3d),
            (6d, 9d, 3.5d),
            sourceSeconds: 10d);

        Assert.Empty(plan.Warnings);
        Assert.Equal(1d, plan.SpeechClips[0].StartSeconds, precision: 6);
        Assert.Equal(6d, plan.SpeechClips[1].StartSeconds, precision: 6);
    }

    [Fact]
    public void Build_moves_an_earlier_take_that_runs_past_the_source_end_even_when_the_last_one_fits()
    {
        // The first take (2s, 9s long) ends at 11s, past the 10s source; the last take fits. The first
        // take moves to 0 and ends where the last one starts.
        MixPlan plan = BuildPlan(10d, (2d, 9d, 9d), (9d, 10d, 1d));

        Assert.Empty(plan.Warnings);
        Assert.Equal(0d, plan.SpeechClips[0].StartSeconds, precision: 6);
        Assert.Equal(9d, plan.SpeechClips[1].StartSeconds, precision: 6);
    }

    [Fact]
    public void Build_keeps_a_moved_take_after_every_earlier_take_not_only_its_neighbour()
    {
        // A (0–15s) reaches past B (13–14s). C is pulled from 16s towards 14s but must not start
        // before A ends, so it stops at 15s and the remaining 1s overhang is reported.
        MixPlan plan = BuildPlan(17d, (0d, 13d, 15d), (13d, 14d, 1d), (16d, 17d, 3d));

        Assert.Equal(15d, plan.SpeechClips[2].StartSeconds, precision: 6);
        MixPlanWarning warning = Assert.Single(plan.Warnings);
        Assert.Equal(2, warning.SegmentIndex);
    }

    [Fact]
    public void Build_moves_an_earlier_take_that_reaches_past_a_moved_take_beyond_a_fitting_neighbour()
    {
        // C is pulled from 16s to 14s. B fits before it, but A (2–15s) would still push C back to 15s,
        // so A moves 1s earlier and C stays at 14s.
        MixPlan plan = BuildPlan(17d, (2d, 13d, 13d), (13d, 14d, 1d), (16d, 17d, 3d));

        Assert.Empty(plan.Warnings);
        Assert.Equal(1d, plan.SpeechClips[0].StartSeconds, precision: 6);
        Assert.Equal(13d, plan.SpeechClips[1].StartSeconds, precision: 6);
        Assert.Equal(14d, plan.SpeechClips[2].StartSeconds, precision: 6);
    }

    [Fact]
    public void Build_leaves_an_existing_overlap_alone_when_nothing_runs_past_the_source_end()
    {
        MixPlan plan = BuildPlan(20d, (0d, 5d, 7d), (5d, 8d, 2d));

        Assert.Empty(plan.Warnings);
        Assert.Equal(0d, plan.SpeechClips[0].StartSeconds, precision: 6);
        Assert.Equal(5d, plan.SpeechClips[1].StartSeconds, precision: 6);
    }

    private static MixPlan BuildTwoSegmentPlan(
        (double Start, double End, double Take) first,
        (double Start, double End, double Take) second,
        double sourceSeconds) =>
        BuildPlan(sourceSeconds, first, second);

    private static MixPlan BuildPlan(double sourceSeconds, params (double Start, double End, double Take)[] segmentTakes)
    {
        TestProjectContext context = CreateContext();
        ProjectArtifact source = CreateArtifact(context, ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0, durationSeconds: sourceSeconds);
        var segments = new List<TranscriptSegment>();
        var artifacts = new List<ProjectArtifact> { source };
        var takes = new List<TtsTake>();
        foreach ((int index, (double start, double end, double takeSeconds)) in segmentTakes.Select((value, index) => (index, value)))
        {
            segments.Add(TranscriptSegment.Create(context.TranscriptRevisionId, index, start, end, $"Segment {index}", context.SpeakerId));
            ProjectArtifact artifact = CreateArtifact(context, ArtifactKind.TtsTake,
                $"artifacts/tts/take-{index}.wav", createdOffsetSeconds: 2 + index, durationSeconds: takeSeconds);
            artifacts.Add(artifact);
            takes.Add(TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: index)
                .Complete(artifact.Id, durationSamples: (int)Math.Round(takeSeconds * 24000), sampleRate: 24000, provider: "fake"));
        }

        return new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId, context.MediaAssetId, artifacts, segments, [], takes, StageRuns: null));
    }
}
