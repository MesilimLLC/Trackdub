using Trackdub.Application.LipSynthesis;
using Trackdub.Application.Mixing;
using Trackdub.Application.Transcripts;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Mixing;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Tts;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public partial class TranscriptProjectServiceTests
{
    private static async Task<TranscriptProjectState> AssertLipSynthesisUsesCurrentMixPlanDriverAsync(
        FakeServiceScope scope,
        CancellationToken token)
    {
        TranscriptProjectState beforeLipRepair = await scope.CreateReopenedStateService().OpenAsync(null, token);
        Assert.DoesNotContain(beforeLipRepair.ProjectState.Artifacts,
            artifact => artifact.Kind == ArtifactKind.ExportAudio);
        var mixPlanBuilder = new MixPlanBuilder(scope.ArtifactStore);
        var mixPlanStore = new MixPlanStore(scope.ArtifactStore);
        MediaAsset mediaAsset = Assert.IsType<MediaAsset>(beforeLipRepair.ProjectState.MediaAsset);
        MixPlan savedUserPlan = mixPlanBuilder.Build(new MixPlanBuildRequest(
            beforeLipRepair.ProjectState.Project.Id,
            mediaAsset.Id,
            beforeLipRepair.ProjectState.Artifacts,
            beforeLipRepair.TranscriptSegments,
            beforeLipRepair.TranslatedSegments,
            beforeLipRepair.TtsTakes,
            beforeLipRepair.StageRuns,
            SourceGainDb: -8d,
            DubbedSpeechGainDb: -4d,
            DuckingLeadSeconds: 0.3d,
            RestoreOriginalPan: true,
            ApplyTimbrePolish: true,
            CandidateGroups: beforeLipRepair.TtsCandidateGroups,
            MediaDurationSeconds: mediaAsset.DurationSeconds));
        await mixPlanStore.SaveAsync(savedUserPlan, token);

        // Change the current selection after saving the plan: a newer completed take must win
        // when the driver is rebuilt from current project state rather than the saved plan.
        TtsTake savedTake = beforeLipRepair.TtsTakes.First();
        TtsTake newerTake = savedTake with
        {
            Id = Guid.NewGuid(),
            CreatedAtUtc = savedTake.CreatedAtUtc.AddMinutes(1),
        };
        await scope.TtsTakeRepository.SaveAsync(newerTake, token);

        var lipEngine = new FakeLipSynthesisEngine
        {
            OutputDirectory = Path.Join(scope.ArtifactStore.GetPath(string.Empty), "lip-output"),
        };
        var lipRenderer = new FakeMixRenderer();
        using var lipWorkflow = new LipSynthesisWorkflow(
            scope.CreateReopenedStateService(),
            new LipSynthesisStageHandler(
                lipEngine, new FakeFaceDetector(), new FakeFaceLandmarkProvider(), new FakeFacePoseEstimator(),
                scope.ArtifactStore, scope.StageRunStore, new FakeLipSynthesisSegmentRepository(), mediaAssetRepository: scope.MediaAssetRepository),
            scope.ArtifactStore,
            mixPlanStore,
            lipRenderer,
            mixPlanBuilder);
        TranscriptProjectState synthesized = await lipWorkflow.SynthesizeAllAsync(new LipSynthesisRunRequest(), token);
        Assert.NotEmpty(lipEngine.Requests);
        string driverPath = Path.Join(
            scope.ArtifactStore.GetPath(string.Empty), "artifacts", "lip-synthesis", "driver", "dubbed-speech.wav");
        Assert.All(lipEngine.Requests, request => Assert.Equal(driverPath, request.DubbedAudioPath));
        var driverRender = Assert.Single(lipRenderer.Calls);
        Assert.Equal(-200d, driverRender.MixPlan.SourceGainDb);
        Assert.False(driverRender.MixPlan.ApplyTimbrePolish);
        Assert.Empty(driverRender.MixPlan.MediaClips);
        Assert.Equal(-4d, driverRender.MixPlan.DubbedSpeechGainDb);
        Assert.Equal(0.3d, driverRender.MixPlan.DuckingLeadSeconds);
        Assert.True(driverRender.MixPlan.RestoreOriginalPan);
        MixSpeechClip rebuiltClip = Assert.Single(
            driverRender.MixPlan.SpeechClips,
            clip => clip.SegmentIndex == newerTake.SegmentIndex);
        Assert.Equal(newerTake.Id, rebuiltClip.TakeId);
        MixPlan persistedUserPlan = Assert.IsType<MixPlan>(await mixPlanStore.LoadAsync(token));
        Assert.Equal(-8d, persistedUserPlan.SourceGainDb);
        Assert.True(persistedUserPlan.ApplyTimbrePolish);
        Assert.Equal(StageRunStatus.Completed, synthesized.StageRuns.Last(run => run.StageName == StageNames.LipSynthesis).Status);
        return synthesized;
    }
}
