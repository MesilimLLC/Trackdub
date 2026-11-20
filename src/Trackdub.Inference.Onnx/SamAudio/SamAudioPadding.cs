namespace Trackdub.Inference.Onnx.SamAudio;

internal static class SamAudioPadding
{
    /// <summary>
    /// Pads <paramref name="samples"/> on the right to a multiple of the codec hop, reflecting the signal as the codec's
    /// own padding does. Input too short to reflect is zero-padded instead.
    /// </summary>
    public static float[] PadToHop(float[] samples)
    {
        ArgumentNullException.ThrowIfNull(samples);
        int remainder = samples.Length % SamAudioModelConstants.HopLength;
        if (remainder == 0)
        {
            return samples;
        }

        int pad = SamAudioModelConstants.HopLength - remainder;
        float[] padded = new float[samples.Length + pad];
        samples.CopyTo(padded, 0);
        if (pad < samples.Length)
        {
            for (int i = 0; i < pad; i++)
            {
                padded[samples.Length + i] = samples[samples.Length - 2 - i];
            }
        }

        return padded;
    }

    public static int FrameCount(int paddedLength) => paddedLength / SamAudioModelConstants.HopLength;
}
