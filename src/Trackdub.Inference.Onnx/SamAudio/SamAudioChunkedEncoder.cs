namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>
/// Encodes a hop-aligned wave in overlapping chunks and stitches the features at the middle of each overlap. The codec
/// encoder is convolutional with a short receptive field, so a chunk differs from a full encode only within a few frames of
/// its own edges, which the overlap keeps away from the stitch points (identical on a 10 s clip). Encoding a whole 30 s
/// window needs about 4.9 GB of VRAM, the largest of any graph; a 6 s chunk needs a fraction of that.
/// </summary>
internal static class SamAudioChunkedEncoder
{
    public static float[] Encode(float[] paddedWave, ISamAudioGraphs graphs, CancellationToken cancellationToken)
    {
        int frames = SamAudioPadding.FrameCount(paddedWave.Length);
        if (frames <= SamAudioModelConstants.EncodeChunkFrames)
        {
            return graphs.Encode(paddedWave, cancellationToken);
        }

        int chunk = SamAudioModelConstants.EncodeChunkFrames;
        int[] starts = new WindowLayout(frames, chunk, SamAudioModelConstants.EncodeOverlapFrames).Starts();
        float[] features = new float[frames * SamAudioModelConstants.FeatureChannels];
        int keepFrom = 0;
        for (int k = 0; k < starts.Length; k++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            float[] chunkFeatures = graphs.Encode(
                paddedWave.AsSpan(starts[k] * SamAudioModelConstants.HopLength, chunk * SamAudioModelConstants.HopLength).ToArray(),
                cancellationToken);
            int keepTo = k == starts.Length - 1 ? frames : (starts[k + 1] + starts[k] + chunk) / 2;
            Array.Copy(
                chunkFeatures,
                (keepFrom - starts[k]) * SamAudioModelConstants.FeatureChannels,
                features,
                keepFrom * SamAudioModelConstants.FeatureChannels,
                (keepTo - keepFrom) * SamAudioModelConstants.FeatureChannels);
            keepFrom = keepTo;
        }

        return features;
    }
}
