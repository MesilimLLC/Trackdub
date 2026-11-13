using System.Buffers.Binary;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace Trackdub.Inference.Onnx.Qwen3Asr;

internal sealed class Qwen3AsrEmbedTokens
{
    private readonly Half[] embeddings;

    private Qwen3AsrEmbedTokens(Half[] embeddings, int vocabSize, int hiddenSize)
    {
        this.embeddings = embeddings;
        VocabularySize = vocabSize;
        HiddenSize = hiddenSize;
    }

    public int VocabularySize { get; }

    public int HiddenSize { get; }

    public static Qwen3AsrEmbedTokens Load(string embedTokensPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(embedTokensPath);
        if (!File.Exists(embedTokensPath))
        {
            throw new FileNotFoundException("Qwen3-ASR embed_tokens.bin was not found.", embedTokensPath);
        }

        using FileStream stream = new(embedTokensPath, FileMode.Open, FileAccess.Read, FileShare.Read);
        long fileLength = stream.Length;
        EmbedLayout? layout = ResolveLayout(stream, fileLength);
        if (layout is not { } resolved)
        {
            throw new InvalidDataException(
                $"embed_tokens.bin size {fileLength} is not a recognized float16 embedding matrix.");
        }

        long expectedPayloadBytes = (long)resolved.VocabSize * resolved.HiddenSize * sizeof(ushort);
        if (resolved.PayloadOffset + expectedPayloadBytes != fileLength)
        {
            throw new InvalidDataException(
                $"embed_tokens.bin size {fileLength} does not match expected float16 matrix {resolved.VocabSize}x{resolved.HiddenSize}.");
        }

        int elementCount = checked((int)(expectedPayloadBytes / sizeof(ushort)));
        var matrix = new Half[elementCount];
        stream.Position = resolved.PayloadOffset;
        stream.ReadExactly(MemoryMarshal.AsBytes(matrix.AsSpan()));
        if (!BitConverter.IsLittleEndian)
        {
            Span<ushort> halves = MemoryMarshal.Cast<Half, ushort>(matrix);
            for (int i = 0; i < halves.Length; i++)
            {
                halves[i] = BinaryPrimitives.ReverseEndianness(halves[i]);
            }
        }

        return new Qwen3AsrEmbedTokens(matrix, resolved.VocabSize, resolved.HiddenSize);
    }

    public float[] Lookup(int tokenId)
    {
        var vector = new float[HiddenSize];
        Lookup(tokenId, vector.AsSpan());
        return vector;
    }

    /// <summary>Allocation-free lookup: writes the token row into <paramref name="destination"/>.</summary>
    public void Lookup(int tokenId, Span<float> destination)
    {
        if (tokenId < 0 || tokenId >= VocabularySize)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenId));
        }

        if (destination.Length != HiddenSize)
        {
            throw new ArgumentException(
                $"Destination length {destination.Length} does not match hidden size {HiddenSize}.",
                nameof(destination));
        }

        ReadOnlySpan<Half> row = embeddings.AsSpan(tokenId * HiddenSize, HiddenSize);
        TensorPrimitives.ConvertToSingle(row, destination);
    }

    private static EmbedLayout? ResolveLayout(FileStream stream, long fileLength)
    {
        if (fileLength >= 8 && TryReadHeaderLayout(stream, fileLength) is { } headerLayout)
        {
            return headerLayout;
        }

        return TryInferLayout(fileLength);
    }

    private static EmbedLayout? TryReadHeaderLayout(FileStream stream, long fileLength)
    {
        Span<byte> header = stackalloc byte[8];
        stream.ReadExactly(header);
        int candidateVocab = BinaryPrimitives.ReadInt32LittleEndian(header[..4]);
        int candidateHidden = BinaryPrimitives.ReadInt32LittleEndian(header[4..8]);
        if (candidateVocab <= 0)
        {
            return null;
        }

        if (candidateHidden <= 0)
        {
            return null;
        }

        long expectedPayloadBytes = (long)candidateVocab * candidateHidden * sizeof(ushort);
        if (expectedPayloadBytes != fileLength - header.Length)
        {
            return null;
        }

        return new EmbedLayout(candidateVocab, candidateHidden, header.Length);
    }

    private static EmbedLayout? TryInferLayout(long fileLength)
    {
        if (fileLength % sizeof(ushort) != 0)
        {
            return null;
        }

        long elementCount = fileLength / sizeof(ushort);
        (int vocab, int hidden)[] knownShapes =
        [
            (151_936, 1024),
            (151_936, 2048),
        ];

        foreach ((int vocab, int hidden) in knownShapes)
        {
            if ((long)vocab * hidden == elementCount)
            {
                return new EmbedLayout(vocab, hidden, 0);
            }
        }

        return null;
    }

    private readonly record struct EmbedLayout(int VocabSize, int HiddenSize, long PayloadOffset);
}
