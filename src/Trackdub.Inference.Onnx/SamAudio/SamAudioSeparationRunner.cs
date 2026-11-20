using Trackdub.Contracts.Pipeline;

namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>
/// Separates every channel in overlapping 30 s windows and crossfades the windows back together. Work runs in three passes
/// (encode every window, run the flow on every window, decode every window) so a caller that leases one graph at a time
/// holds a single lease per pass instead of all three at once.
/// </summary>
internal static class SamAudioSeparationRunner
{
    /// <summary>Fixed seed so a rerun of the same job gives the same stems.</summary>
    public const int Seed = 1234;

    public static SamAudioSeparation Run(
        float[][] channels,
        ISamAudioGraphs graphs,
        IProgress<StemSeparationProgress>? progress,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(channels);
        ArgumentNullException.ThrowIfNull(graphs);
        int length = channels.Length > 0 ? channels[0].Length : 0;
        if (length == 0 || channels.Any(channel => channel.Length != length))
        {
            throw new ArgumentException("Channels must be non-empty and equal in length.", nameof(channels));
        }

        WindowLayout layout = new(length, SamAudioModelConstants.WindowSamples, SamAudioModelConstants.OverlapSamples);
        float[][] windows = CutWindows(channels, layout);
        Passes passes = new(graphs, progress, cancellationToken, windows.Length * 3);

        float[][] features = passes.Encode(windows);
        float[][] latents = passes.Flow(features, new SamAudioRandom(Seed));
        (float[][] speechPieces, float[][] bedPieces) = passes.Decode(latents, windows, new SamAudioRandom(Seed + 1));

        return Join(speechPieces, bedPieces, layout, channels.Length);
    }

    /// <summary>Every window of every channel, channel-major: index = channel * windowCount + window.</summary>
    private static float[][] CutWindows(float[][] channels, WindowLayout layout)
    {
        int[] starts = layout.Starts();
        return
        [
            .. channels.SelectMany(channel => starts.Select(start =>
                channel.AsSpan(start, Math.Min(layout.Window, layout.Length - start)).ToArray())),
        ];
    }

    private static SamAudioSeparation Join(float[][] speechPieces, float[][] bedPieces, WindowLayout layout, int channelCount)
    {
        int windowCount = layout.Starts().Length;
        float[][] speech = new float[channelCount][];
        float[][] bed = new float[channelCount][];
        for (int c = 0; c < channelCount; c++)
        {
            speech[c] = SamAudioCrossfade.Join(speechPieces.Skip(c * windowCount).Take(windowCount).ToArray(), layout);
            bed[c] = SamAudioCrossfade.Join(bedPieces.Skip(c * windowCount).Take(windowCount).ToArray(), layout);
        }

        return new SamAudioSeparation(speech, bed, windowCount);
    }

    private sealed class Passes(
        ISamAudioGraphs graphs,
        IProgress<StemSeparationProgress>? progress,
        CancellationToken cancellationToken,
        int totalSteps)
    {
        private int done;

        public float[][] Encode(float[][] windows)
        {
            ReportStart();
            return Each(windows, window => SamAudioWindowSeparator.Encode(window, graphs, cancellationToken));
        }

        public float[][] Flow(float[][] features, SamAudioRandom random) =>
            Each(features, feature => SamAudioWindowSeparator.Flow(
                feature,
                random.Gaussian(feature.Length / SamAudioModelConstants.FeatureChannels * SamAudioModelConstants.LatentChannels),
                graphs,
                cancellationToken));

        public (float[][] Speech, float[][] Bed) Decode(float[][] latents, float[][] windows, SamAudioRandom random)
        {
            (float[] Speech, float[] Bed)[] decoded = new (float[], float[])[latents.Length];
            for (int i = 0; i < latents.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                decoded[i] = SamAudioWindowSeparator.Decode(
                    latents[i], new SamAudioDecodeRequest(windows[i].Length, random.WatermarkMessage()), graphs, cancellationToken);
                ReportStep();
            }

            return ([.. decoded.Select(d => d.Speech)], [.. decoded.Select(d => d.Bed)]);
        }

        private float[][] Each(float[][] inputs, Func<float[], float[]> step)
        {
            float[][] outputs = new float[inputs.Length][];
            for (int i = 0; i < inputs.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                outputs[i] = step(inputs[i]);
                ReportStep();
            }

            return outputs;
        }

        private void ReportStart() => progress?.Report(new StemSeparationProgress(0, totalSteps, 0d, 1d));

        private void ReportStep() =>
            progress?.Report(new StemSeparationProgress(++done, totalSteps, (double)done / totalSteps, 1d));
    }
}
