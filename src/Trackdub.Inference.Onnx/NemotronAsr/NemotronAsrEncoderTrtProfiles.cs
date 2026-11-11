using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime;

namespace Trackdub.Inference.Onnx.NemotronAsr;

internal static class NemotronAsrEncoderTrtProfiles
{
    // Shapes must match the bundled encoder.onnx inputs: processed_signal is mel-major
    // [B,128,T=65] and caches are layers-first ([24,1,56,1024] / [24,1,1024,8]).
    private const string ProfileWithoutPromptIndex =
        "processed_signal:1x128x65,processed_signal_length:1,cache_last_channel:24x1x56x1024,cache_last_time:24x1x1024x8,cache_last_channel_len:1";

    private const string PromptIndexSuffix = ",prompt_index:1";

    private static readonly ConcurrentDictionary<string, bool> PromptInputCache =
        new(StringComparer.OrdinalIgnoreCase);

    internal static IReadOnlyDictionary<string, string> BuildOptions(string encoderModelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(encoderModelPath);
        return BuildOptions(EncoderHasPromptInput(encoderModelPath));
    }

    internal static IReadOnlyDictionary<string, string> BuildOptions(bool hasPromptInput)
    {
        string shapes = hasPromptInput
            ? ProfileWithoutPromptIndex + PromptIndexSuffix
            : ProfileWithoutPromptIndex;

        return new Dictionary<string, string>
        {
            // Bundled Nemotron export expects mel-major [B,mel,T]; TRT profiles must match encoder inputs.
            ["trt_profile_min_shapes"] = shapes,
            ["trt_profile_max_shapes"] = shapes,
            ["trt_profile_opt_shapes"] = shapes,
        };
    }

    // Profile options stay free of enable_cuda_graph so each caller chooses: production and smoke follow
    // the session default (capture on), and EpContextTrtProfiles.Resolve hands this dict to the
    // benchmark path where the session flag must follow the benchmark's own cuda-graph mode.
    // Earlier notes here kept capture off because the decoder can fall back to managed feeds when
    // shared device allocators are unavailable and its bound buffers are scoped to a decode. NVIDIA's
    // native capture recaptures when buffer locations change, so that is a performance question, not a
    // correctness one; add enable_cuda_graph = 0 to opt this model out if a measurement shows a loss.

    internal static SessionOptions BuildRuntimeOptions(string encoderModelPath) =>
        BuildRuntimeOptions(EncoderHasPromptInput(encoderModelPath));

    internal static SessionOptions BuildRuntimeOptions(bool hasPromptInput) =>
        new(
            new Dictionary<string, string>(BuildOptions(hasPromptInput), StringComparer.Ordinal),
            new Dictionary<string, string>(StringComparer.Ordinal));

    internal sealed record SessionOptions(
        IReadOnlyDictionary<string, string> Encoder,
        IReadOnlyDictionary<string, string> Decoder);

    // Opt-in validation harnesses select their capture mode explicitly.
    internal static readonly IReadOnlyDictionary<string, string> DecoderJointCudaGraphOptions =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["enable_cuda_graph"] = "1",
        };

    internal static bool EncoderHasPromptInput(string encoderModelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(encoderModelPath);
        return PromptInputCache.GetOrAdd(encoderModelPath, static path =>
        {
            using InferenceSession session = new(path);
            return session.InputMetadata.ContainsKey("prompt_index");
        });
    }
}
