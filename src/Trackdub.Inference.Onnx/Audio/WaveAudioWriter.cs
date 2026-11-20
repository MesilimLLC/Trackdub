using System.Buffers.Binary;

namespace Trackdub.Inference.Onnx.Audio;

internal static class WaveAudioWriter
{
    private const int HeaderBytes = 44;

    public static Task WriteMonoPcm16Async(string path, float[] samples, int sampleRate, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(samples);
        return WritePcm16Async(path, [samples], sampleRate, cancellationToken);
    }

    /// <summary>Writes one array per channel as interleaved 16-bit PCM.</summary>
    public static async Task WritePcm16Async(
        string path,
        float[][] channels,
        int sampleRate,
        CancellationToken cancellationToken)
    {
        if (sampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "Sample rate must be positive.");
        }

        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        byte[] bytes = EncodePcm16(channels, sampleRate);
        await File.WriteAllBytesAsync(fullPath, bytes, cancellationToken).ConfigureAwait(false);
    }

    internal static byte[] EncodeMonoPcm16(float[] samples, int sampleRate) => EncodePcm16([samples], sampleRate);

    internal static byte[] EncodePcm16(float[][] channelSamples, int sampleRate)
    {
        ArgumentNullException.ThrowIfNull(channelSamples);
        if (channelSamples.Length == 0 || channelSamples.Any(channel => channel.Length != channelSamples[0].Length))
        {
            throw new ArgumentException("Channels must be non-empty and equal in length.", nameof(channelSamples));
        }

        ushort channels = checked((ushort)channelSamples.Length);
        const ushort bitsPerSample = 16;
        int frames = channelSamples[0].Length;
        int dataBytes = checked(frames * channels * sizeof(short));
        byte[] wav = new byte[HeaderBytes + dataBytes];
        Span<byte> span = wav;

        "RIFF"u8.CopyTo(span);
        BinaryPrimitives.WriteInt32LittleEndian(span[4..], 36 + dataBytes);
        "WAVE"u8.CopyTo(span[8..]);
        "fmt "u8.CopyTo(span[12..]);
        BinaryPrimitives.WriteInt32LittleEndian(span[16..], 16);
        BinaryPrimitives.WriteUInt16LittleEndian(span[20..], 1);
        BinaryPrimitives.WriteUInt16LittleEndian(span[22..], channels);
        BinaryPrimitives.WriteInt32LittleEndian(span[24..], sampleRate);
        BinaryPrimitives.WriteInt32LittleEndian(span[28..], sampleRate * channels * bitsPerSample / 8);
        BinaryPrimitives.WriteUInt16LittleEndian(span[32..], (ushort)(channels * bitsPerSample / 8));
        BinaryPrimitives.WriteUInt16LittleEndian(span[34..], bitsPerSample);
        "data"u8.CopyTo(span[36..]);
        BinaryPrimitives.WriteInt32LittleEndian(span[40..], dataBytes);

        Span<byte> data = span[HeaderBytes..];
        for (int i = 0; i < frames; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                float clamped = Math.Clamp(channelSamples[c][i], -1f, 1f);
                short pcm = (short)Math.Clamp(clamped * 32767f, short.MinValue, short.MaxValue);
                BinaryPrimitives.WriteInt16LittleEndian(data[(((i * channels) + c) * sizeof(short))..], pcm);
            }
        }

        return wav;
    }
}
