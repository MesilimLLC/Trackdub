namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>
/// Decodes a long latent in overlapping chunks and crossfades the waves. The codec decoder is convolutional with a short
/// receptive field, so a chunk differs from the full decode only near its edges, which the crossfade covers. Decoding a whole
/// 30 s window at once exhausts a 12 GB GPU, while chunks of 6 s take under two seconds per window in total. Every chunk
/// carries the same watermark message so the output watermark is consistent across the window.
/// </summary>
internal static class SamAudioChunkedDecoder
{
    public static (float[] Target, float[] Residual) Decode(
        float[] latent,
        float[] message,
        ISamAudioGraphs graphs,
        CancellationToken cancellationToken)
    {
        int frames = latent.Length / SamAudioModelConstants.LatentChannels;
        if (frames <= SamAudioModelConstants.DecodeChunkFrames)
        {
            return graphs.Decode(latent, frames, message, cancellationToken);
        }

        WindowLayout frameLayout = new(frames, SamAudioModelConstants.DecodeChunkFrames, SamAudioModelConstants.DecodeOverlapFrames);
        List<float[]> targets = [];
        List<float[]> residuals = [];
        foreach (int start in frameLayout.Starts())
        {
            cancellationToken.ThrowIfCancellationRequested();
            float[] chunk = latent.AsSpan(
                start * SamAudioModelConstants.LatentChannels,
                SamAudioModelConstants.DecodeChunkFrames * SamAudioModelConstants.LatentChannels).ToArray();
            (float[] target, float[] residual) = graphs.Decode(chunk, SamAudioModelConstants.DecodeChunkFrames, message, cancellationToken);
            targets.Add(target);
            residuals.Add(residual);
        }

        WindowLayout sampleLayout = new(
            frames * SamAudioModelConstants.HopLength,
            SamAudioModelConstants.DecodeChunkFrames * SamAudioModelConstants.HopLength,
            SamAudioModelConstants.DecodeOverlapFrames * SamAudioModelConstants.HopLength);
        return (SamAudioCrossfade.Join(targets, sampleLayout), SamAudioCrossfade.Join(residuals, sampleLayout));
    }
}
