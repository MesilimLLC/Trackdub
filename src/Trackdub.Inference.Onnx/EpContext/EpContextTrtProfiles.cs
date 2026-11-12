using Microsoft.ML.OnnxRuntime;
using Trackdub.Inference.Onnx.Whisper;
using Trackdub.Inference.Onnx.Madlad;
using Trackdub.Inference.Onnx.NemotronAsr;
using Trackdub.Inference.Onnx.Qwen3Asr;
using Trackdub.Inference.Onnx.SortFormer;

namespace Trackdub.Inference.Onnx.EpContext;

/// <summary>
/// Resolves the TensorRT optimization profile a production session passes for a given graph, so
/// EP-context AOT builds the same engine. Without it TRT-RTX assumes a fully dynamic range, which
/// some graphs (SortFormer streaming) cannot satisfy, and CompileModel() emits no engine.
/// </summary>
internal static class EpContextTrtProfiles
{
    public static IReadOnlyDictionary<string, string>? Resolve(string sourceModelPath)
    {
        if (MadladTrtProfiles.Resolve(sourceModelPath) is { } madladProfile)
        {
            return madladProfile;
        }

        if (Qwen3AsrTrtProfiles.Resolve(sourceModelPath) is { } qwen3Profile)
        {
            return qwen3Profile;
        }

        IReadOnlyCollection<string> inputNames;
        try
        {
            // Metadata only: a CPU session never touches the TRT plugin.
            using InferenceSession session = new(sourceModelPath);
            inputNames = session.InputMetadata.Keys.ToArray();
            // No-past whisper decoder: the hidden width comes from the declared dims, so the
            // profile needs the opened session, not just names.
            if (session.InputMetadata.ContainsKey("input_ids")
                && session.InputMetadata.ContainsKey("encoder_hidden_states"))
            {
                return WhisperOnnxAudioTranscriptionEngine.BuildTrtDecoderOptions(sourceModelPath);
            }
        }
        catch (OnnxRuntimeException)
        {
            return null;
        }

        return Resolve(inputNames);
    }

    internal static IReadOnlyDictionary<string, string>? Resolve(IReadOnlyCollection<string> inputNames)
    {
        var names = new HashSet<string>(inputNames, StringComparer.Ordinal);
        if (SortFormerDiarizationEngine.IsStreamingExportInputSet(names))
        {
            return SortFormerDiarizationEngine.TrtOptions;
        }

        if (names.Contains("processed_signal") && names.Contains("cache_last_channel"))
        {
            return NemotronAsrEncoderTrtProfiles.BuildOptions(hasPromptInput: names.Contains("prompt_index"));
        }

        return null;
    }

    /// <summary>
    /// Parses the registered opt profile ("name:1x2x3,...") into per-input shapes, so benchmark
    /// input generation can match the engine's baked optimization point.
    /// </summary>
    internal static IReadOnlyDictionary<string, int[]>? ResolveOptShapes(string sourceModelPath)
    {
        if (Resolve(sourceModelPath) is not { } options
            || !options.TryGetValue("trt_profile_opt_shapes", out string? spec))
        {
            return null;
        }

        var shapes = new Dictionary<string, int[]>(StringComparer.Ordinal);
        foreach (string entry in spec.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var colon = entry.IndexOf(':');
            if (colon <= 0)
            {
                continue;
            }

            var dims = entry[(colon + 1)..]
                .Split('x', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(part => int.TryParse(part, out int value) ? value : -1)
                .ToArray();
            if (dims.Length == 0 || dims.Any(d => d < 0))
            {
                continue;
            }

            shapes[entry[..colon]] = dims;
        }

        return shapes.Count == 0 ? null : shapes;
    }
}
