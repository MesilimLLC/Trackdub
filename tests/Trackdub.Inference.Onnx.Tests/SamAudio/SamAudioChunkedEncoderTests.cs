using Trackdub.Inference.Onnx.SamAudio;

namespace Trackdub.Inference.Onnx.Tests.SamAudio;

public sealed class SamAudioChunkedEncoderTests
{
    private const int EdgeFrames = 3;

    /// <summary>
    /// A pointwise stand-in encoder: channel 0 of a frame is the first sample of that frame, the rest are zero.
    /// <see cref="EdgeBias"/> models a convolutional encoder's edge artefacts: the first and last few frames of every call
    /// are off by that much, so stitching at a chunk edge instead of mid-overlap shows up in the features.
    /// </summary>
    private sealed class PointwiseGraphs : ISamAudioGraphs
    {
        public float EdgeBias { get; init; }

        public List<int> EncodedFrames { get; } = [];

        /// <summary>First wave sample of each call (the test wave holds frame index + 1), so chunk spacing is visible.</summary>
        public List<int> ChunkStarts { get; } = [];

        public float[] Encode(float[] paddedWave, CancellationToken cancellationToken)
        {
            int frames = paddedWave.Length / SamAudioModelConstants.HopLength;
            EncodedFrames.Add(frames);
            ChunkStarts.Add((int)paddedWave[0] - 1);
            float[] features = new float[frames * SamAudioModelConstants.FeatureChannels];
            for (int f = 0; f < frames; f++)
            {
                bool atEdge = f < EdgeFrames || f >= frames - EdgeFrames;
                features[f * SamAudioModelConstants.FeatureChannels] = paddedWave[f * SamAudioModelConstants.HopLength] + (atEdge ? EdgeBias : 0f);
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

    /// <summary>Chunk edges are wrong by a large bias here, so only a stitch point well inside the overlap reproduces a whole encode.</summary>
    [Fact]
    public void Chunk_edge_artefacts_never_reach_the_stitched_features()
    {
        foreach (int frames in new[] { 400, 750 })
        {
            float[] wave = Wave(frames);

            float[] features = SamAudioChunkedEncoder.Encode(wave, new PointwiseGraphs { EdgeBias = 1000f }, CancellationToken.None);
            float[] expected = new PointwiseGraphs { EdgeBias = 1000f }.Encode(wave, CancellationToken.None);

            Assert.Equal(expected, features);
        }
    }

    [Fact]
    public void Chunks_start_one_chunk_minus_the_overlap_apart()
    {
        PointwiseGraphs graphs = new();

        SamAudioChunkedEncoder.Encode(Wave(400), graphs, CancellationToken.None);

        Assert.Equal([0, 125, 250], graphs.ChunkStarts);
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
