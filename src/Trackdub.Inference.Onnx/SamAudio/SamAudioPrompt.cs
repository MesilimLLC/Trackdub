using System.Buffers.Binary;
using System.Text.Json;

namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>The precomputed text-encoder output for the export's fixed prompt ("speech"), shaped [1, tokens, channels].</summary>
internal sealed record SamAudioPrompt(float[] Embedding, int Tokens, int Channels)
{
    public static SamAudioPrompt Load(string modelRootPath)
    {
        string metadataPath = SamAudioModelConstants.ResolveModelPath(modelRootPath, SamAudioModelConstants.MetadataFileName);
        using JsonDocument metadata = JsonDocument.Parse(File.ReadAllText(metadataPath));
        int[] shape = [.. metadata.RootElement.GetProperty("prompt_shape").EnumerateArray().Select(static d => d.GetInt32())];
        if (shape.Length != 3 || shape[0] != 1)
        {
            throw new InvalidOperationException($"SAM-Audio prompt shape must be [1, tokens, channels], but is [{string.Join(", ", shape)}].");
        }

        byte[] bytes = File.ReadAllBytes(SamAudioModelConstants.ResolveModelPath(modelRootPath, SamAudioModelConstants.PromptEmbeddingFileName));
        int expected = shape[1] * shape[2];
        if (bytes.Length != expected * sizeof(float))
        {
            throw new InvalidOperationException(
                $"SAM-Audio prompt embedding has {bytes.Length} bytes but shape [{string.Join(", ", shape)}] needs {expected * sizeof(float)}.");
        }

        float[] embedding = new float[expected];
        for (int i = 0; i < expected; i++)
        {
            embedding[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * sizeof(float)));
        }

        return new SamAudioPrompt(embedding, shape[1], shape[2]);
    }
}
