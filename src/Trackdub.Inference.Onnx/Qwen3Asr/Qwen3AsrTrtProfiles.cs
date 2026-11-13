namespace Trackdub.Inference.Onnx.Qwen3Asr;

/// <summary>
/// TensorRT options for the three Qwen3-ASR graphs. The encoder keeps a bounded mel profile; the
/// decoders stay on the implicit dynamic profile because explicit shape profiles were measured to
/// make the cold JIT build ~3.7x slower (118s vs 32s ASR stage) without improving warm latency.
/// All three graphs opt out of CUDA graph capture: encoder mel length and decoder KV sequence both
/// change per chunk/token, so capture would recapture repeatedly.
/// </summary>
internal static class Qwen3AsrTrtProfiles
{
    // Encoder mel frames: chunks cap at 28s of 16kHz audio (2800 frames at hop 160); 3000 leaves
    // headroom for padding.
    internal static readonly IReadOnlyDictionary<string, string> Encoder = new Dictionary<string, string>
    {
        ["enable_cuda_graph"] = "0",
        ["trt_profile_min_shapes"] = "mel:1x128x1",
        ["trt_profile_max_shapes"] = "mel:1x128x3000",
        ["trt_profile_opt_shapes"] = "mel:1x128x3000",
    };

    // Shared by decoder_init and decoder_step: both see per-chunk/per-token sequence growth, so
    // CUDA graph capture stays off. No trt_profile_*_shapes: measured slower cold builds.
    internal static readonly IReadOnlyDictionary<string, string> Decoder = new Dictionary<string, string>
    {
        ["enable_cuda_graph"] = "0",
    };

    /// <summary>
    /// Resolves the profile for EP-context AOT builds so they bake the same engine the runtime
    /// session would. Gated on the model directory name so unrelated encoder.onnx files elsewhere
    /// keep the default dynamic profile.
    /// </summary>
    internal static IReadOnlyDictionary<string, string>? Resolve(string modelPath)
    {
        string? directory = Path.GetDirectoryName(modelPath);
        if (directory is null
            || directory.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
                .All(static part => !part.Contains("qwen3-asr", StringComparison.OrdinalIgnoreCase)))
        {
            return null;
        }

        return Path.GetFileName(modelPath).ToLowerInvariant() switch
        {
            "encoder.onnx" => Encoder,
            "decoder_init.onnx" or "decoder_step.onnx" => Decoder,
            _ => null,
        };
    }
}
