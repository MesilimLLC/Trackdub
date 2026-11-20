namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>Separates one mono window (up to the 30 s training length) into speech and the residual bed.</summary>
internal static class SamAudioWindowSeparator
{
    public static (float[] Speech, float[] Bed) Separate(
        float[] window,
        ISamAudioGraphs graphs,
        SamAudioRandom random,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(window);
        ArgumentNullException.ThrowIfNull(random);

        int frames = SamAudioPadding.FrameCount(SamAudioPadding.PadToHop(window).Length);
        SamAudioFlowInputs inputs = new(random.Gaussian(frames * SamAudioModelConstants.LatentChannels), random.WatermarkMessage());
        return Separate(window, graphs, inputs, cancellationToken);
    }

    /// <summary>Separation with explicit flow inputs (the seeded overload draws them).</summary>
    internal static (float[] Speech, float[] Bed) Separate(
        float[] window,
        ISamAudioGraphs graphs,
        SamAudioFlowInputs inputs,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(window);
        ArgumentNullException.ThrowIfNull(graphs);
        ArgumentNullException.ThrowIfNull(inputs);

        float[] padded = SamAudioPadding.PadToHop(window);
        int frames = SamAudioPadding.FrameCount(padded.Length);
        float[] features = graphs.Encode(padded, cancellationToken);

        float[] latent = SamAudioOdeSolver.Midpoint(
            inputs.Noise,
            SamAudioModelConstants.OdeSteps,
            (time, noisy) => graphs.Velocity(time, noisy, features, cancellationToken));

        (float[] target, float[] residual) = graphs.Decode(latent, frames, inputs.WatermarkMessage, cancellationToken);
        return (Crop(target, window.Length), Crop(residual, window.Length));
    }

    private static float[] Crop(float[] decoded, int length)
    {
        if (decoded.Length < length)
        {
            throw new InvalidOperationException(
                $"SAM-Audio decoder returned {decoded.Length} samples for a {length}-sample window.");
        }

        return decoded.Length == length ? decoded : decoded.AsSpan(0, length).ToArray();
    }
}
