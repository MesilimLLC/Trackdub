using Trackdub.Inference.Onnx.SamAudio;

namespace Trackdub.Inference.Onnx.Tests.SamAudio;

public sealed class SamAudioChunkedEncoderTests
{
    /// <summary>A pointwise stand-in encoder: channel 0 of a frame is the first sample of that frame, the rest are zero.</summary>
    private sealed class PointwiseGraphs : ISamAudioGraphs
    {
        public List<int> EncodedFrames { get; } = [];

        public float[] Encode(float[] paddedWave, CancellationToken cancellationToken)
        {
            int frames = paddedWave.Length / SamAudioModelConstants.HopLength;
            EncodedFrames.Add(frames);
            float[] features = new float[frames * SamAudioModelConstants.FeatureChannels];
            for (int f = 0; f < frames; f++)
            {
                features[f * SamAudioModelConstants.FeatureChannels] = paddedWave[f * SamAudioModelConstants.HopLength];
            }

            return features;
        }

        public float[] Velocity(float time, float[] noisy, float[] features, CancellationToken cancellationToken) => throw new NotSupportedException();

        public (float[] Target, float[] Residual) Decode(float[] latent, int frames, float[] message, CancellationToken cancellationToken) => throw new NotSupportedException();
    }

    private static float[] Wave(int frames) =>
        [.. Enumerable.Range(0, frames * SamAudioModelConstants.HopLength).Select(i => (float)(i / SamAudioModelConstants.HopLength) + 1f)];

    [Theory]
    [InlineData(40)]
    [InlineData(150)]
    public void A_wave_that_fits_in_one_chunk_is_encoded_in_a_single_call(int frames)
    {
        PointwiseGraphs graphs = new();

        float[] features = SamAudioChunkedEncoder.Encode(Wave(frames), graphs, CancellationToken.None);

        Assert.Equal([frames], graphs.EncodedFrames);
        Assert.Equal(frames * SamAudioModelConstants.FeatureChannels, features.Length);
    }

    [Theory]
    [InlineData(151)]
    [InlineData(400)]
    [InlineData(750)]
    public void Chunked_encoding_equals_encoding_the_whole_wave(int frames)
    {
        float[] wave = Wave(frames);
        PointwiseGraphs graphs = new();

        float[] features = SamAudioChunkedEncoder.Encode(wave, graphs, CancellationToken.None);
        float[] expected = new PointwiseGraphs().Encode(wave, CancellationToken.None);

        Assert.Equal(expected, features);
        Assert.True(graphs.EncodedFrames.Count > 1);
        Assert.All(graphs.EncodedFrames, encoded => Assert.Equal(SamAudioModelConstants.EncodeChunkFrames, encoded));
    }

    [Fact]
    public void Chunked_encoding_honours_cancellation()
    {
        using CancellationTokenSource cts = new();
        cts.Cancel();

        Assert.Throws<OperationCanceledException>(() =>
            SamAudioChunkedEncoder.Encode(Wave(400), new PointwiseGraphs(), cts.Token));
    }
}
