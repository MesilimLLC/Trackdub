namespace Trackdub.Inference.Onnx.Madlad;

/// <summary>Shared runtime and AOT profiles for the validated MADLAD-400 3B RTX export.</summary>
internal static class MadladTrtProfiles
{
    internal const int MaximumSequenceLength = 512;
    private const string ExportDirectory = "trt_rtx_mixed_fp16_fp32";

    internal static readonly IReadOnlyDictionary<string, string> Encoder = Build(decoder: false);
    internal static readonly IReadOnlyDictionary<string, string> Decoder = Build(decoder: true);

    // The classic TensorRT EP's CUDA graph replay needs IoBinding; the plain Run the greedy decoder uses
    // returns null outputs under it (ONNX_TYPE_UNKNOWN). TensorRT-RTX captures graphs natively and is unaffected.
    internal static readonly IReadOnlyDictionary<string, string> ClassicEncoder = WithoutCudaGraph(Encoder);
    internal static readonly IReadOnlyDictionary<string, string> ClassicDecoder = WithoutCudaGraph(Decoder);

    internal static IReadOnlyDictionary<string, string>? Resolve(string modelPath) =>
        string.Equals(Path.GetFileName(Path.GetDirectoryName(modelPath)), ExportDirectory, StringComparison.OrdinalIgnoreCase)
            ? Path.GetFileName(modelPath).ToLowerInvariant() switch
            {
                "encoder_model.onnx" => Encoder,
                "decoder_model.onnx" => Decoder,
                _ => null
            }
            : null;

    internal static string? GetAotExclusionReason(string modelPath) =>
        modelPath.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
            .Any(static part => part.StartsWith("madlad400", StringComparison.OrdinalIgnoreCase)) && Resolve(modelPath) is null
            ? "MADLAD TensorRT RTX compilation requires the validated trt-fp16 plain encoder/decoder export. "
                + "Quantized, merged, and cached decoders are excluded before native compilation."
            : null;

    internal static void ValidateLengths(int sourceLength, int generationLength)
    {
        if (sourceLength > MaximumSequenceLength || generationLength > MaximumSequenceLength)
        {
            throw new InvalidOperationException(
                $"MADLAD TensorRT RTX supports at most {MaximumSequenceLength} source and generated tokens per segment. "
                + "Split the segment or select a CPU-compatible MADLAD variant.");
        }
    }

    private static IReadOnlyDictionary<string, string> Build(bool decoder) => new Dictionary<string, string>(StringComparer.Ordinal)
    {
        // Unbounded profiles allow batch/sequence dimensions of 32767 and zero-length inputs.
        // The decoder exceeds TensorRT's tensor-volume limit and can abort in LLVM.
        // Batch one and bounded positive lengths cover the production padded decode buckets.
        ["enable_cuda_graph"] = "1",
        ["trt_profile_min_shapes"] = Shapes(decoder, 1),
        ["trt_profile_opt_shapes"] = Shapes(decoder, 16),
        ["trt_profile_max_shapes"] = Shapes(decoder, MaximumSequenceLength)
    };

    private static IReadOnlyDictionary<string, string> WithoutCudaGraph(IReadOnlyDictionary<string, string> profile) =>
        new Dictionary<string, string>(profile, StringComparer.Ordinal) { ["enable_cuda_graph"] = "0" };

    private static string Shapes(bool decoder, int length) => decoder
        ? $"input_ids:1x{length},encoder_attention_mask:1x{length},encoder_hidden_states:1x{length}x1024"
        : $"input_ids:1x{length},attention_mask:1x{length}";
}
