using Trackdub.Contracts.Pipeline;

namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>
/// Separates every channel in overlapping 30 s windows and crossfades the windows back together. Work runs in three passes
/// (encode every window, run the flow on every window, decode every window) so a caller that leases one graph at a time
/// holds a single lease per pass instead of all three at once. Windows are cut from the source on the fly and decoded
/// windows are accumulated straight into the output, so memory stays near the size of the input and output rather than
/// holding every window and piece as well.
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
        int windowCount = layout.Starts().Length;
        Passes passes = new(graphs, progress, cancellationToken, channels.Length * windowCount * 3);

        float[][] features = passes.Encode(channels, layout);
        float[][] latents = passes.Flow(features, new SamAudioRandom(Seed));
        return passes.Decode(latents, layout, new SamAudioRandom(Seed + 1));
    }

    private sealed class Passes(
        ISamAudioGraphs graphs,
        IProgress<StemSeparationProgress>? progress,
        CancellationToken cancellationToken,
        int totalSteps)
    {
        private int done;

        /// <summary>Features of every window, channel-major: index = channel * windowCount + window.</summary>
        public float[][] Encode(float[][] channels, WindowLayout layout)
        {
            ReportStart();
            int[] starts = layout.Starts();
            float[][] features = new float[channels.Length * starts.Length][];
            for (int i = 0; i < features.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                features[i] = SamAudioWindowSeparator.Encode(CutWindow(channels, layout, starts, i), graphs, cancellationToken);
                ReportStep();
            }

            return features;
        }

        public float[][] Flow(float[][] features, SamAudioRandom random)
        {
            float[][] latents = new float[features.Length][];
            for (int i = 0; i < features.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int frames = features[i].Length / SamAudioModelConstants.FeatureChannels;
                latents[i] = SamAudioWindowSeparator.Flow(
                    features[i], random.Gaussian(frames * SamAudioModelConstants.LatentChannels), graphs, cancellationToken);
                features[i] = [];
                ReportStep();
            }

            return latents;
        }

        public SamAudioSeparation Decode(float[][] latents, WindowLayout layout, SamAudioRandom random)
        {
            int[] starts = layout.Starts();
            int channelCount = latents.Length / starts.Length;
            SamAudioCrossfade[] speech = [.. Enumerable.Range(0, channelCount).Select(_ => new SamAudioCrossfade(layout))];
            SamAudioCrossfade[] bed = [.. Enumerable.Range(0, channelCount).Select(_ => new SamAudioCrossfade(layout))];
            for (int i = 0; i < latents.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                (int channel, int window) = (i / starts.Length, i % starts.Length);
                int windowLength = Math.Min(layout.Window, layout.Length - starts[window]);
                (float[] windowSpeech, float[] windowBed) = SamAudioWindowSeparator.Decode(
                    latents[i], new SamAudioDecodeRequest(windowLength, random.WatermarkMessage()), graphs, cancellationToken);
                latents[i] = [];
                speech[channel].Add(windowSpeech, starts[window], window, starts.Length);
                bed[channel].Add(windowBed, starts[window], window, starts.Length);
                ReportStep();
            }

            return new SamAudioSeparation(
                [.. speech.Select(static channel => channel.Normalized())],
                [.. bed.Select(static channel => channel.Normalized())],
                starts.Length);
        }

        private static float[] CutWindow(float[][] channels, WindowLayout layout, int[] starts, int index)
        {
            int channel = index / starts.Length;
            int start = starts[index % starts.Length];
            return channels[channel].AsSpan(start, Math.Min(layout.Window, layout.Length - start)).ToArray();
        }

        private void ReportStart() => progress?.Report(new StemSeparationProgress(0, totalSteps, 0d, 1d));

        private void ReportStep() =>
            progress?.Report(new StemSeparationProgress(++done, totalSteps, (double)done / totalSteps, 1d));
    }
}
