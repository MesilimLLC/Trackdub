using System.Buffers.Binary;
using Trackdub.Inference.Onnx.Qwen3Asr;

namespace Trackdub.Inference.Onnx.Tests;

public sealed class Qwen3AsrLowCostTests
{
    [Fact]
    public void Lookup_span_writes_the_same_row_as_the_allocating_lookup()
    {
        string path = Path.Join(Path.GetTempPath(), $"qwen3-embed-{Guid.NewGuid():N}.bin");
        try
        {
            WriteEmbedTokens(path, vocabSize: 2, hiddenSize: 3, rows: [[1f, 2f, 3f], [4f, 5f, 6f]]);
            Qwen3AsrEmbedTokens embedTokens = Qwen3AsrEmbedTokens.Load(path);

            var destination = new float[3];
            embedTokens.Lookup(1, destination.AsSpan());

            Assert.Equal([4f, 5f, 6f], destination);
            Assert.Equal(embedTokens.Lookup(1), destination);
            Assert.Throws<ArgumentException>(() => embedTokens.Lookup(1, new float[2].AsSpan()));
            Assert.Throws<ArgumentOutOfRangeException>(() => embedTokens.Lookup(2, destination.AsSpan()));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_accepts_a_headerless_known_shape_file()
    {
        const int vocabSize = 151_936;
        const int hiddenSize = 1024;
        string path = Path.Join(Path.GetTempPath(), $"qwen3-embed-{Guid.NewGuid():N}.bin");
        try
        {
            using (var stream = new FileStream(path, FileMode.CreateNew))
            {
                stream.SetLength((long)vocabSize * hiddenSize * sizeof(ushort));
            }

            Qwen3AsrEmbedTokens embedTokens = Qwen3AsrEmbedTokens.Load(path);

            Assert.Equal(vocabSize, embedTokens.VocabularySize);
            Assert.Equal(hiddenSize, embedTokens.HiddenSize);
            var destination = new float[hiddenSize];
            embedTokens.Lookup(0, destination.AsSpan());
            Assert.All(destination, value => Assert.Equal(0f, value));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_rejects_a_header_whose_payload_length_does_not_match()
    {
        string path = Path.Join(Path.GetTempPath(), $"qwen3-embed-{Guid.NewGuid():N}.bin");
        try
        {
            WriteEmbedTokens(path, vocabSize: 2, hiddenSize: 3, rows: [[1f, 2f, 3f], [4f, 5f, 6f]]);
            using (var stream = new FileStream(path, FileMode.Append))
            {
                stream.WriteByte(0);
            }

            Assert.Throws<InvalidDataException>(() => Qwen3AsrEmbedTokens.Load(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Load_rejects_an_unrecognized_file_size()
    {
        string path = Path.Join(Path.GetTempPath(), $"qwen3-embed-{Guid.NewGuid():N}.bin");
        try
        {
            File.WriteAllBytes(path, new byte[10]);

            Assert.Throws<InvalidDataException>(() => Qwen3AsrEmbedTokens.Load(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Too_short_chunk_skip_reason_is_explicit_and_recorded_in_the_summary_detail()
    {
        string reason = Qwen3AsrOnnxAudioTranscriptionEngine.FormatTooShortChunkSkipReason(
            melFrames: 0, audioTokenCount: 0);

        Assert.Contains("too short", reason, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("0 mel frame", reason, StringComparison.Ordinal);

        string detail = Qwen3AsrOnnxAudioTranscriptionEngine.AppendChunkSkipReasons(
            "ONNX Qwen3-ASR greedy decoding.",
            [reason, reason]);

        Assert.Contains("Skipped 2 chunk(s)", detail, StringComparison.Ordinal);
        Assert.Contains(reason, detail, StringComparison.Ordinal);
    }

    private static void WriteEmbedTokens(string path, int vocabSize, int hiddenSize, float[][] rows)
    {
        using var stream = new FileStream(path, FileMode.CreateNew);
        Span<byte> header = stackalloc byte[8];
        BinaryPrimitives.WriteInt32LittleEndian(header, vocabSize);
        BinaryPrimitives.WriteInt32LittleEndian(header[4..], hiddenSize);
        stream.Write(header);
        Span<byte> half = stackalloc byte[2];
        foreach (float[] row in rows)
        {
            foreach (float value in row)
            {
                BinaryPrimitives.WriteUInt16LittleEndian(half, BitConverter.HalfToUInt16Bits((Half)value));
                stream.Write(half);
            }
        }
    }
}
