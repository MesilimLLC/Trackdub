using Trackdub.Contracts;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Mixing;
using Trackdub.Media.Mixing;
using Trackdub.Media.Waveforms;
using Trackdub.TestDoubles;

namespace Trackdub.Media.Tests;

public sealed class BackgroundMediaGainRendererTests
{
    [Fact]
    public async Task RenderAsync_applies_background_media_gain_without_changing_source_gain()
    {
        string root = Path.Join(Path.GetTempPath(), "Trackdub.Media.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string source = Path.Join(root, "source.wav");
            string media = Path.Join(root, "media.wav");
            string output = Path.Join(root, "output.wav");
            await WavePcm16.WriteSamplesAsync(source, Enumerable.Repeat(0.1f, 1000).ToArray(), 1000, 1,
                TestContext.Current.CancellationToken);
            await WavePcm16.WriteSamplesAsync(media, Enumerable.Repeat(0.25f, 1000).ToArray(), 1000, 1,
                TestContext.Current.CancellationToken);
            var store = new FakeArtifactStore();
            store.SeedPath("source.wav", source);
            store.SeedPath("media.wav", media);
            var plan = new MixPlan(Guid.NewGuid(), null, ArtifactKind.NormalizedAudio, "source.wav",
                0d, -96d, 0d, 0d, 0d, DateTimeOffset.UtcNow, [], [], [],
                MediaClips: [new MixMediaClip("media.wav", 0d, 1d, GainDb: -6d)]);

            await new PreviewRangeRenderer(store).RenderAsync(
                new PreviewRangeRenderRequest(plan, 0d, 1d, output), TestContext.Current.CancellationToken);

            WavePcm16Samples samples = await WavePcm16.ReadAllSamplesAsync(output, TestContext.Current.CancellationToken);
            Assert.InRange(samples.Samples[100], 0.22f, 0.23f);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
