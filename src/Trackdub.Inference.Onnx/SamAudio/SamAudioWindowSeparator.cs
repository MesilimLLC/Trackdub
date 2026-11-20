namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>
/// One window (up to the 30 s training length) separated into speech and the residual bed, in the three steps the
/// runner schedules across all windows so that only one graph needs to be loaded at a time.
/// </summary>
internal static class SamAudioWindowSeparator
{
    /// <summary>Codec features of the window, padded to the codec hop, as <c>frames x 128</c>.</summary>
    public static float[] Encode(float[] window, ISamAudioGraphs graphs, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(window);
        ArgumentNullException.ThrowIfNull(graphs);
        return SamAudioChunkedEncoder.Encode(SamAudioPadding.PadToHop(window), graphs, cancellationToken);
    }

    /// <summary>Integrates the flow from the starting noise to the latent, <c>frames x 256</c>.</summary>
    public static float[] Flow(float[] features, float[] noise, ISamAudioGraphs graphs, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(noise);
        ArgumentNullException.ThrowIfNull(graphs);
        return SamAudioOdeSolver.Midpoint(
            noise,
            SamAudioModelConstants.OdeSteps,
            (time, noisy) => graphs.Velocity(time, noisy, features, cancellationToken));
    }

    /// <summary>Decodes the latent to speech and bed waves cropped to the request's window length.</summary>
    public static (float[] Speech, float[] Bed) Decode(
        float[] latent,
        SamAudioDecodeRequest request,
        ISamAudioGraphs graphs,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(latent);
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(graphs);
        (float[] target, float[] residual) = SamAudioChunkedDecoder.Decode(latent, request.WatermarkMessage, graphs, cancellationToken);
        return (Crop(target, request.Length), Crop(residual, request.Length));
    }

    /// <summary>All three steps for one window with explicit flow inputs; the runner uses the steps separately.</summary>
    internal static (float[] Speech, float[] Bed) Separate(
        float[] window,
        ISamAudioGraphs graphs,
        SamAudioFlowInputs inputs,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        float[] features = Encode(window, graphs, cancellationToken);
        float[] latent = Flow(features, inputs.Noise, graphs, cancellationToken);
        return Decode(latent, new SamAudioDecodeRequest(window.Length, inputs.WatermarkMessage), graphs, cancellationToken);
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

/// <summary>What the decode step needs besides the latent: the window's true length and the watermark message.</summary>
internal sealed record SamAudioDecodeRequest(int Length, float[] WatermarkMessage);
