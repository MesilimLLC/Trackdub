using Trackdub.Contracts.Pipeline;

namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>Runs every channel through overlapping 30 s windows and crossfades the separated windows back together.</summary>
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
        int[] starts = layout.Starts();
        SamAudioRandom random = new(Seed);
        float[][] speech = new float[channels.Length][];
        float[][] bed = new float[channels.Length][];
        int total = channels.Length * starts.Length;
        int done = 0;
        progress?.Report(new StemSeparationProgress(0, total, 0d, 1d));

        for (int c = 0; c < channels.Length; c++)
        {
            List<float[]> speechPieces = [];
            List<float[]> bedPieces = [];
            foreach (int start in starts)
            {
                cancellationToken.ThrowIfCancellationRequested();
                float[] window = channels[c].AsSpan(start, Math.Min(SamAudioModelConstants.WindowSamples, length - start)).ToArray();
                (float[] windowSpeech, float[] windowBed) = SamAudioWindowSeparator.Separate(window, graphs, random, cancellationToken);
                speechPieces.Add(windowSpeech);
                bedPieces.Add(windowBed);
                progress?.Report(new StemSeparationProgress(++done, total, (double)done / total, 1d));
            }

            speech[c] = SamAudioCrossfade.Join(speechPieces, layout);
            bed[c] = SamAudioCrossfade.Join(bedPieces, layout);
        }

        return new SamAudioSeparation(speech, bed, starts.Length);
    }
}
