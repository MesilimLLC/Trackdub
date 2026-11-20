using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Onnx.Audio;

namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>Writes and cleans up the stem files a separation request names.</summary>
internal static class SamAudioStemOutputs
{
    public static float[] DownmixToMono(float[][] channels)
    {
        float[] mono = new float[channels[0].Length];
        foreach (float[] channel in channels)
        {
            for (int i = 0; i < mono.Length; i++)
            {
                mono[i] += channel[i] / channels.Length;
            }
        }

        return mono;
    }

    /// <summary>Speech is written mono; the bed keeps one channel per input channel, so a stereo source gives a stereo bed.</summary>
    public static async Task WriteAsync(StemSeparationRequest request, float[] speech, float[][] bed, CancellationToken cancellationToken)
    {
        int rate = SamAudioModelConstants.SampleRate;
        await WaveAudioWriter.WriteMonoPcm16Async(request.VocalsOutputPath, speech, rate, cancellationToken).ConfigureAwait(false);
        await WaveAudioWriter.WritePcm16Async(request.AmbianceOutputPath, bed, rate, cancellationToken).ConfigureAwait(false);

        // The bed is not split into music and effects: the music path carries the whole bed, as Spleeter's does.
        if (!string.IsNullOrWhiteSpace(request.MusicOutputPath))
        {
            await WaveAudioWriter.WritePcm16Async(request.MusicOutputPath, bed, rate, cancellationToken).ConfigureAwait(false);
        }

        DeleteIfExists(request.SoundEffectsOutputPath);
        await WriteRawStemAsync(request, "vocals", [speech], cancellationToken).ConfigureAwait(false);
        await WriteRawStemAsync(request, "other", bed, cancellationToken).ConfigureAwait(false);
    }

    public static void DeleteAll(StemSeparationRequest request)
    {
        DeleteIfExists(request.VocalsOutputPath);
        DeleteIfExists(request.AmbianceOutputPath);
        DeleteIfExists(request.MusicOutputPath);
        DeleteIfExists(request.SoundEffectsOutputPath);
        foreach (string path in request.RawStemOutputPaths?.Values ?? [])
        {
            DeleteIfExists(path);
        }
    }

    private static Task WriteRawStemAsync(StemSeparationRequest request, string stem, float[][] channels, CancellationToken cancellationToken) =>
        request.RawStemOutputPaths is not null && request.RawStemOutputPaths.TryGetValue(stem, out string? path)
            ? WaveAudioWriter.WritePcm16Async(path, channels, SamAudioModelConstants.SampleRate, cancellationToken)
            : Task.CompletedTask;

    private static void DeleteIfExists(string? path)
    {
        try
        {
            if (!string.IsNullOrWhiteSpace(path) && File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (IOException)
        {
            // Best-effort cleanup only.
        }
        catch (UnauthorizedAccessException)
        {
            // Best-effort cleanup only.
        }
    }
}
