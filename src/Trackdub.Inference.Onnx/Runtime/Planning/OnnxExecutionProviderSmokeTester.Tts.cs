using Microsoft.ML.OnnxRuntimeGenAI;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.Planning;
using Trackdub.Inference.Onnx.Runtime;
using Trackdub.Inference.Onnx.ExecutionProviders;
using Trackdub.Inference.Onnx.Qwen3Asr;
using Trackdub.Inference.Onnx.NemotronAsr;
using Trackdub.Inference.Onnx.OpusMt;
using Trackdub.Inference.Onnx.ParakeetTdt;
using Trackdub.Inference.Onnx.CosyVoice;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.SortFormer;
using Trackdub.Inference.Onnx.Spleeter;
using Trackdub.Inference.Onnx.Whisper;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Inference.Onnx.Worker;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Contracts.Pipeline;

namespace Trackdub.Inference.Onnx.Runtime.Planning;

public sealed partial class OnnxExecutionProviderSmokeTester : IExecutionProviderSmokeTester
{
    private static async Task SmokeTestTtsAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        string modelPath = ResolveTtsProbeModelPath(request);
        string poolFamily = ResolveTtsPoolFamily(request);
        using OnnxExecutionSessionFactory.SingleSessionLease sessionLease = await OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = poolFamily, Provider = request.ExecutionProvider, HasExplicitProviderPreference = request.HasExplicitProviderPreference, CancellationToken = cancellationToken, AllowTrtInitFallback = true }, modelPath)
            .ConfigureAwait(false);
        EnsureSelectedProviderMatchesRequested(request.ExecutionProvider, sessionLease.SelectedProvider, request.AllowPolicyProviderSelection);

        using var inputs = CreateTtsInputs(sessionLease.Session.InputMetadata);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _ = sessionLease.Session.RunWithRetry(inputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: request.ExecutionProvider);

        if (IsChatterboxTtsModel(request))
        {
            await WarmChatterboxSidecarsAsync(request, cancellationToken).ConfigureAwait(false);
        }
        else if (IsCosyVoiceTtsModel(request))
        {
            await ProbeCosyVoiceGraphsAsync(request, cancellationToken)
                .ConfigureAwait(false);
        }
    }

    private static string ResolveTtsPoolFamily(ExecutionProviderSmokeTestRequest request)
    {
        if (IsChatterboxTtsModel(request))
        {
            return "chatterbox";
        }

        if (IsCosyVoiceTtsModel(request))
        {
            return "cosyvoice";
        }

        return "kokoro";
    }

    /// <summary>
    /// Creates pooled TRT-pinned sessions for every remaining CosyVoice graph so a hard
    /// pin cannot hit an unproven graph at first synthesis: the verified TTS smoke verdict
    /// persists, so each graph must pass session init here to be safe later without
    /// fallback. Inference stays on the primary probe; the side graphs are init-only,
    /// mirroring the Chatterbox sidecar warmup.
    /// </summary>
    private static async Task ProbeCosyVoiceGraphsAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        string rootPath = ResolveTtsModelRootPath(request, "Cannot resolve CosyVoice TTS smoke-test root path.");
        CosyVoiceModelFiles modelFiles = CosyVoiceModelFiles.Resolve(rootPath, request.Variant);
        string entryFullPath = Path.GetFullPath(request.EntryPath);
        string[] graphPaths =
        [
            modelFiles.CampPlusPath,
            modelFiles.SpeechTokenizerPath,
            modelFiles.TextEncoderPath,
            modelFiles.TokenGeneratorPath,
            modelFiles.FlowEncoderPath,
            modelFiles.FlowDecoderEstimatorPath,
            modelFiles.HiftF0PredictorPath,
            modelFiles.HiftSourcePath,
            modelFiles.HiftVocoderPath,
        ];

        foreach (string graphPath in graphPaths)
        {
            if (string.Equals(Path.GetFullPath(graphPath), entryFullPath, StringComparison.OrdinalIgnoreCase))
            {
                // The primary probe already created and ran this graph's session.
                continue;
            }

            using OnnxExecutionSessionFactory.SingleSessionLease sessionLease = await OnnxExecutionSessionFactory
                .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "cosyvoice", Provider = request.ExecutionProvider, HasExplicitProviderPreference = request.HasExplicitProviderPreference, CancellationToken = cancellationToken, AllowTrtInitFallback = false }, graphPath)
                .ConfigureAwait(false);
            EnsureSelectedProviderMatchesRequested(request.ExecutionProvider, sessionLease.SelectedProvider, request.AllowPolicyProviderSelection);
        }
    }

    /// <summary>
    /// Pre-warms the three Chatterbox CPU sidecar graphs so the TTS stage does not pay
    /// their cold load after the smoke test already warmed the conditional decoder.
    /// Provider selection mirrors ChatterboxVoiceCloneTtsEngine (sidecars on CPU).
    /// </summary>
    private static async Task WarmChatterboxSidecarsAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        string rootPath = ResolveTtsModelRootPath(request, "Cannot resolve Chatterbox TTS warmup root path.");
        string onnxDirectory = ResolveChatterboxOnnxDirectory(rootPath);

        foreach (string graphName in new[] { "speech_encoder", "embed_tokens", "language_model" })
        {
            string graphPath = ResolveChatterboxGraphPath(onnxDirectory, graphName, request.Variant);
            if (!File.Exists(graphPath))
            {
                continue;
            }

            await OnnxExecutionSessionFactory.WarmPooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "chatterbox", Provider = ExecutionProviderKind.Cpu, CancellationToken = cancellationToken }, graphPath).ConfigureAwait(false);
        }
    }

    private static string ResolveChatterboxGraphPath(string onnxDirectory, string graphName, string? variant)
    {
        string? sanitizedVariant = string.IsNullOrWhiteSpace(variant) ? null : Path.GetFileName(variant);
        if (!string.IsNullOrWhiteSpace(sanitizedVariant) &&
            !sanitizedVariant.Equals("default", StringComparison.OrdinalIgnoreCase))
        {
            string variantPath = Path.Join(onnxDirectory, $"{graphName}_{sanitizedVariant}.onnx");
            if (File.Exists(variantPath))
            {
                return variantPath;
            }
        }

        return Path.Join(onnxDirectory, $"{graphName}.onnx");
    }

    private static InputSet CreateTtsInputs(IReadOnlyDictionary<string, NodeMetadata> inputMetadata)
    {
        if (LooksLikeChatterboxConditionalDecoder(inputMetadata))
        {
            return CreateChatterboxDecoderInputs(inputMetadata);
        }

        var values = new List<NamedOnnxValue>(inputMetadata.Count);
        foreach ((string inputName, NodeMetadata metadata) in inputMetadata)
        {
            int[] dims = metadata.Dimensions.Select(d => d > 0 ? d : 1).ToArray();
            NamedOnnxValue? value = CreateTtsInputValue(inputName, metadata.ElementType, dims);
            if (value is not null)
            {
                values.Add(value);
            }
        }

        return new InputSet(values);
    }

    private static bool LooksLikeChatterboxConditionalDecoder(IReadOnlyDictionary<string, NodeMetadata> inputMetadata) =>
        inputMetadata.ContainsKey("speech_tokens")
        && inputMetadata.ContainsKey("speaker_embeddings")
        && inputMetadata.ContainsKey("speaker_features");

    // ISTFT in conditional_decoder uses n_fft=960 / hop=480. One speech token yields 480
    // samples and fails overlap-add (480 by 960). Keep a short but valid token sequence.
    private const int ChatterboxDecoderSmokeSpeechTokenCount = 8;

    private const int ChatterboxDecoderSmokeFeatureFrames = 8;

    private static InputSet CreateChatterboxDecoderInputs(IReadOnlyDictionary<string, NodeMetadata> inputMetadata)
    {
        var values = new List<NamedOnnxValue>(inputMetadata.Count);
        foreach ((string inputName, NodeMetadata metadata) in inputMetadata)
        {
            var spec = new ChatterboxSmokeInputSpec
            {
                InputName = inputName,
                ModelDimensions = metadata.Dimensions,
            };
            int[] dims = ResolveChatterboxDecoderSmokeDimensions(spec);
            NamedOnnxValue? value = CreateTtsInputValue(inputName, metadata.ElementType, dims);
            if (value is not null)
            {
                values.Add(value);
            }
        }

        return new InputSet(values);
    }

    /// <summary>A Chatterbox decoder input's name and declared shape for smoke probing.</summary>
    internal sealed record ChatterboxSmokeInputSpec
    {
        public required string InputName { get; init; }
        public required IReadOnlyList<int> ModelDimensions { get; init; }
    }

    internal static int[] ResolveChatterboxDecoderSmokeDimensionsForTesting(ChatterboxSmokeInputSpec spec) =>
        ResolveChatterboxDecoderSmokeDimensions(spec);

    private static int[] ResolveChatterboxDecoderSmokeDimensions(ChatterboxSmokeInputSpec spec)
    {
        IReadOnlyList<int> modelDimensions = spec.ModelDimensions;
        if (IsChatterboxSpeechTokenInput(spec))
        {
            return
            [
                modelDimensions[0] > 0 ? modelDimensions[0] : 1,
                ChatterboxDecoderSmokeSpeechTokenCount
            ];
        }

        if (IsChatterboxSpeakerFeatureInput(spec))
        {
            return
            [
                modelDimensions[0] > 0 ? modelDimensions[0] : 1,
                modelDimensions[1] > 0 ? modelDimensions[1] : ChatterboxDecoderSmokeFeatureFrames,
                modelDimensions[2] > 0 ? modelDimensions[2] : 80
            ];
        }

        return modelDimensions.Select(static dimension => dimension > 0 ? dimension : 1).ToArray();
    }

    private static bool IsChatterboxSpeechTokenInput(ChatterboxSmokeInputSpec spec) =>
        spec.InputName.Equals("speech_tokens", StringComparison.Ordinal)
        && spec.ModelDimensions.Count == 2;

    private static bool IsChatterboxSpeakerFeatureInput(ChatterboxSmokeInputSpec spec) =>
        spec.InputName.Equals("speaker_features", StringComparison.Ordinal)
        && spec.ModelDimensions.Count == 3;

    private static string ResolveTtsProbeModelPath(ExecutionProviderSmokeTestRequest request)
    {
        if (!IsChatterboxTtsModel(request))
        {
            return request.EntryPath;
        }

        string rootPath = ResolveTtsModelRootPath(request, "Cannot resolve Chatterbox TTS smoke-test root path.");
        string onnxDirectory = ResolveChatterboxOnnxDirectory(rootPath);
        return ResolveChatterboxGraphPath(onnxDirectory, "conditional_decoder", request.Variant);
    }

    private static string ResolveTtsModelRootPath(
        ExecutionProviderSmokeTestRequest request,
        string missingRootMessage) =>
        !string.IsNullOrWhiteSpace(request.ModelRootPath)
            ? request.ModelRootPath
            : Path.GetDirectoryName(request.EntryPath)
                ?? throw new InvalidOperationException(missingRootMessage);

    private static string ResolveChatterboxOnnxDirectory(string rootPath) =>
        string.Equals(Path.GetFileName(rootPath), "onnx", StringComparison.OrdinalIgnoreCase)
            ? rootPath
            : Path.Join(rootPath, "onnx");

    private static bool IsChatterboxTtsModel(ExecutionProviderSmokeTestRequest request)
    {
        if (request.ModelId.Contains("chatterbox", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        return request.ModelAlias.Contains("chatterbox", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsCosyVoiceTtsModel(ExecutionProviderSmokeTestRequest request)
    {
        if (request.ModelId.Contains("cosyvoice", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        return request.ModelAlias.Contains("cosyvoice", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Creates a TTS probe tensor with the declared shape and element type, using unit speed for
    /// the float <c>speed</c> input to avoid Kokoro's zero-speed duration failure and zeros otherwise.
    /// </summary>
    /// <returns>The probe input, or <see langword="null"/> when the element type is unsupported.</returns>
    private static NamedOnnxValue? CreateTtsInputValue(string inputName, Type elementType, int[] dims)
    {
        int elementCount = dims.Aggregate(1, static (a, b) => checked(a * b));

        if (elementType == typeof(long))
        {
            return NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<long>(new long[elementCount], dims));
        }

        if (elementType == typeof(float))
        {
            float[] data = new float[elementCount];
            // Kokoro divides predicted durations by speed; zero yields an empty frame axis and
            // the graph fails in /encoder/shared/Reshape.
            if (string.Equals(inputName, "speed", StringComparison.Ordinal))
            {
                Array.Fill(data, 1f);
            }

            return NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<float>(data, dims));
        }

        if (elementType == typeof(Half))
        {
            return NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<Half>(new Half[elementCount], dims));
        }

        if (elementType == typeof(Float16))
        {
            return NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<Float16>(new Float16[elementCount], dims));
        }

        // Resilient branch (zero-fill/skip) for unknown element types in TTS probe.
        return null;
    }
}
