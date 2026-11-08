using Trackdub.Application.Mixing;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Mixing;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed class BackgroundMediaGainTests
{
    [Fact]
    public void MixPlanBuilder_applies_background_gain_to_placed_media()
    {
        Guid projectId = Guid.NewGuid();
        Guid mediaAssetId = Guid.NewGuid();
        var source = new ProjectArtifact(
            Guid.NewGuid(), projectId, mediaAssetId, ArtifactKind.NormalizedAudio,
            "source.wav", "source-hash", 100, 8d, 48000, 1, DateTimeOffset.UtcNow);

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            projectId,
            mediaAssetId,
            [source],
            [],
            [],
            [],
            StageRuns: null,
            MediaClips: [new MixMediaClip("music.wav", 0d, 4d)],
            BackgroundMediaGainDb: -6d));

        Assert.Equal(-6d, Assert.Single(plan.MediaClips).GainDb);
    }

    [Fact]
    public async Task MixPlanStore_round_trips_background_gain()
    {
        var artifactStore = new FakeArtifactStore();
        var store = new MixPlanStore(artifactStore);
        var plan = new MixPlan(
            Guid.NewGuid(), null, ArtifactKind.NormalizedAudio, "source.wav",
            0d, 0d, -13d, 0.05d, 0.18d, DateTimeOffset.UtcNow,
            [], [], [], MediaClips: [new MixMediaClip("music.wav", 0d, 4d, GainDb: -6d)]);

        await store.SaveAsync(plan, TestContext.Current.CancellationToken);
        MixPlan? reloaded = await store.LoadAsync(TestContext.Current.CancellationToken);

        Assert.Equal(-6d, Assert.Single(Assert.IsType<MixPlan>(reloaded).MediaClips).GainDb);
    }
}
