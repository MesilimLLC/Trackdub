using Trackdub.Inference.Onnx.SamAudio;

namespace Trackdub.Inference.Onnx.Tests.SamAudio;

public sealed class SamAudioChunkedDecoderTests
{
    /// <summary>A pointwise stand-in decoder: sample i is the first latent channel of its frame (target) and its negation (residual).</summary>
    private sealed class PointwiseGraphs : ISamAudioGraphs
    {
        public List<(int Frames, float[] Message)> Calls { get; } = [];

        public float[] Encode(float[] paddedWave, CancellationToken cancellationToken) => throw new NotSupportedException();

        public float[] Velocity(float time, float[] noisy, float[] features, CancellationToken cancellationToken) => throw new NotSupportedException();

        public (float[] Target, float[] Residual) Decode(float[] latent, int frames, float[] message, CancellationToken cancellationToken)
        {
            Calls.Add((frames, message));
            float[] target = new float[frames * SamAudioModelConstants.HopLength];
            for (int i = 0; i < target.Length; i++)
            {
                target[i] = latent[i / SamAudioModelConstants.HopLength * SamAudioModelConstants.LatentChannels];
            }

            return (target, [.. target.Select(static v => -v)]);
        }
    }

    private static float[] Latent(int frames) =>
        [.. Enumerable.Range(0, frames * SamAudioModelConstants.LatentChannels)
            .Select(i => i % SamAudioModelConstants.LatentChannels == 0 ? i / SamAudioModelConstants.LatentChannels + 1f : 0f)];

    [Theory]
    [InlineData(40)]
    [InlineData(150)]
    public void A_latent_that_fits_in_one_chunk_is_decoded_in_a_single_call(int frames)
    {
        PointwiseGraphs graphs = new();

        (float[] target, _) = SamAudioChunkedDecoder.Decode(Latent(frames), new float[32], graphs, CancellationToken.None);

        Assert.Equal((frames, 1), (graphs.Calls.Single().Frames, graphs.Calls.Count));
        Assert.Equal(frames * SamAudioModelConstants.HopLength, target.Length);
    }

    [Theory]
    [InlineData(151)]
    [InlineData(400)]
    [InlineData(750)]
    public void Chunked_decoding_equals_decoding_the_whole_latent(int frames)
    {
        float[] latent = Latent(frames);
        PointwiseGraphs graphs = new();

        (float[] target, float[] residual) = SamAudioChunkedDecoder.Decode(latent, new float[32], graphs, CancellationToken.None);
        (float[] expectedTarget, float[] expectedResidual) = new PointwiseGraphs().Decode(latent, frames, new float[32], CancellationToken.None);

        Assert.Equal(expectedTarget.Length, target.Length);
        Assert.All(Enumerable.Range(0, target.Length).Where(i => i % 97 == 0), i =>
        {
            Assert.Equal(expectedTarget[i], target[i], 1e-3f);
            Assert.Equal(expectedResidual[i], residual[i], 1e-3f);
        });
        Assert.All(graphs.Calls, call => Assert.Equal(SamAudioModelConstants.DecodeChunkFrames, call.Frames));
    }

    [Fact]
    public void Every_chunk_carries_the_same_watermark_message()
    {
        PointwiseGraphs graphs = new();
        float[] message = [.. Enumerable.Range(0, 32).Select(i => (float)(i % 2))];

        SamAudioChunkedDecoder.Decode(Latent(400), message, graphs, CancellationToken.None);

        Assert.True(graphs.Calls.Count > 1);
        Assert.All(graphs.Calls, call => Assert.Same(message, call.Message));
    }

    [Fact]
    public void Chunked_decoding_honours_cancellation()
    {
        using CancellationTokenSource cts = new();
        cts.Cancel();

        Assert.Throws<OperationCanceledException>(() =>
            SamAudioChunkedDecoder.Decode(Latent(400), new float[32], new PointwiseGraphs(), cts.Token));
    }
}
