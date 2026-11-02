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
    private static async Task SmokeTestWhisperAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        string encoderModelPath = request.EntryPath;
        ExecutionProviderKind provider = request.ExecutionProvider;
        string decoderModelPath = ResolveWhisperDecoderPath(encoderModelPath);
        string? onnxDirectory = Path.GetDirectoryName(encoderModelPath);
        string modelRootPath = Path.GetDirectoryName(onnxDirectory ?? string.Empty)
            ?? throw new InvalidOperationException("Whisper smoke test could not resolve model root.");
        int melBins = WhisperOnnxAudioTranscriptionEngine.ReadMelBins(modelRootPath);
        IReadOnlyDictionary<string, string>? encoderOptions = provider is ExecutionProviderKind.TensorRTRtx
            ? WhisperOnnxAudioTranscriptionEngine.BuildTrtEncoderOptions(melBins)
            : null;
        IReadOnlyDictionary<string, string>? decoderOptions = provider is ExecutionProviderKind.TensorRTRtx
            ? WhisperOnnxAudioTranscriptionEngine.BuildTrtDecoderOptions(decoderModelPath)
            : null;
        // Pooled with the same family/TRT profiles WhisperOnnxAudioTranscriptionEngine uses.
        // Production planning excludes whisper-onnx from TensorRT families, while staged Olive
        // validation and explicit smoke probes remain evaluable with automatic downgrade and
        // guarded eviction on failure so a partial TRT-RTX OOM never leaves a poisoned entry behind.
        try
        {
            using OnnxExecutionSessionFactory.WhisperSessionLease sessionLease = await OnnxExecutionSessionFactory
                .CreatePooledWhisperAsync(
                    WhisperOnnxAudioTranscriptionEngine.EngineFamilyName,
                    encoderModelPath,
                    decoderModelPath,
                    provider,
                    cancellationToken,
                    additionalTrtEncoderOptions: encoderOptions,
                    additionalTrtDecoderOptions: decoderOptions)
                .ConfigureAwait(false);
            try
            {
                EnsureSelectedProviderMatchesRequested(provider, sessionLease.SelectedProvider);
            }
            catch (InvalidOperationException ex)
            {
                throw new InvalidOperationException(
                    $"{ex.Message} Session bootstrap: {sessionLease.BootstrapDetail}",
                    ex);
            }

            using var encoderInputs = CreateWhisperEncoderInputs(sessionLease.EncoderSession.InputMetadata, melBins);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = sessionLease.EncoderSession.RunWithRetry(encoderInputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: provider);
            Tensor<float> hiddenStates = ResolveWhisperEncoderHiddenStates(encoderResults);

            using var decoderInputs = CreateWhisperDecoderInputs(hiddenStates);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _ = sessionLease.DecoderSession.RunWithRetry(decoderInputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: provider);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            try
            {
                await InferenceSessionPool.Shared.EvictModelAsync(WhisperOnnxAudioTranscriptionEngine.EngineFamilyName).ConfigureAwait(false);
            }
            catch (Exception)
            {
                // Cleanup only: an eviction failure must not replace the smoke-test failure rethrown below.
            }

            throw;
        }
    }

    private static Tensor<float> ResolveWhisperEncoderHiddenStates(
        IEnumerable<DisposableNamedOnnxValue> encoderResults)
    {
        DisposableNamedOnnxValue[] materialised = encoderResults as DisposableNamedOnnxValue[]
            ?? encoderResults.ToArray();
        foreach (DisposableNamedOnnxValue result in materialised)
        {
            if (result.Name.Contains("hidden", StringComparison.OrdinalIgnoreCase)
                || result.Name.Equals("last_hidden_state", StringComparison.OrdinalIgnoreCase))
            {
                return result.AsTensor<float>();
            }
        }

        string available = string.Join(", ", materialised.Select(static result => result.Name));
        throw new InvalidOperationException(
            "Whisper encoder smoke test did not produce a hidden-state output. "
            + $"Available outputs: {(string.IsNullOrWhiteSpace(available) ? "(none)" : available)}.");
    }

    // One 30-second mel window, matching the engine's real input and TRT profile shapes.
    private const int WhisperSmokeFrameCount = 3000;

    /// <summary>A Whisper encoder input's name, declared shape, and mel-bin count for smoke probing.</summary>
    internal sealed record WhisperSmokeInputSpec
    {
        public required string InputName { get; init; }
        public required IReadOnlyList<int> ModelDimensions { get; init; }
        public required int MelBins { get; init; }
    }

    private static InputSet CreateWhisperEncoderInputs(
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata,
        int melBins)
    {
        var values = new List<NamedOnnxValue>(inputMetadata.Count);
        foreach ((string inputName, NodeMetadata metadata) in inputMetadata)
        {
            var spec = new WhisperSmokeInputSpec
            {
                InputName = inputName,
                ModelDimensions = metadata.Dimensions,
                MelBins = melBins,
            };
            values.Add(CreateWhisperEncoderInput(spec, metadata));
        }

        return new InputSet(values);
    }

    internal static int[] ResolveWhisperEncoderSmokeDimensionsForTesting(
        WhisperSmokeInputSpec spec) =>
        ResolveWhisperEncoderSmokeDimensions(spec);

    private static int[] ResolveWhisperEncoderSmokeDimensions(WhisperSmokeInputSpec spec)
    {
        // The encoder declares input_features as (batch_size, feature_size,
        // encoder_sequence_length) — all symbolic — so the generic all-ones smoke
        // tensor would be [1,1,1] and fail the first Conv (expects feature_size mel
        // channels). Feed a valid zero mel window instead.
        if (HasWhisperFeatureShape(spec))
        {
            return [1, spec.MelBins, WhisperSmokeFrameCount];
        }

        return spec.ModelDimensions.Select(static dimension => dimension > 0 ? dimension : 1).ToArray();
    }

    private static bool HasWhisperFeatureShape(WhisperSmokeInputSpec spec)
    {
        if (spec.ModelDimensions.Count != 3 || spec.MelBins <= 0)
        {
            return false;
        }

        return spec.InputName.Equals("input_features", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsWhisperFeatureInput(WhisperSmokeInputSpec spec, NodeMetadata metadata)
    {
        if (!metadata.IsTensor || metadata.ElementDataType != TensorElementType.Float)
        {
            return false;
        }

        return HasWhisperFeatureShape(spec);
    }

    private static NamedOnnxValue CreateWhisperEncoderInput(
        WhisperSmokeInputSpec spec,
        NodeMetadata metadata)
    {
        if (IsWhisperFeatureInput(spec, metadata))
        {
            int[] dimensions = ResolveWhisperEncoderSmokeDimensions(spec);
            int elementCount = dimensions.Aggregate(1, static (product, dimension) => checked(product * dimension));
            return NamedOnnxValue.CreateFromTensor(
                spec.InputName,
                new DenseTensor<float>(new float[elementCount], dimensions));
        }

        return CreateSmokeTensorInput(spec.InputName, metadata);
    }

    private static InputSet CreateWhisperDecoderInputs(Tensor<float> encoderHiddenStates)
    {
        IReadOnlyList<NamedOnnxValue> values =
        [
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(new long[] { 50258 }, [1, 1])),
            NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates)
        ];

        return new InputSet(values);
    }

    private static string ResolveWhisperDecoderPath(string encoderModelPath)
    {
        string fileName = Path.GetFileName(encoderModelPath);
        string decoderFileName = fileName.Replace("encoder_model", "decoder_model", StringComparison.OrdinalIgnoreCase);
        string candidatePath = Path.Join(Path.GetDirectoryName(encoderModelPath)!, decoderFileName);
        if (File.Exists(candidatePath))
        {
            return Path.GetFullPath(candidatePath);
        }

        candidatePath = Path.Join(Path.GetDirectoryName(encoderModelPath)!, "decoder_model.onnx");
        if (File.Exists(candidatePath))
        {
            return Path.GetFullPath(candidatePath);
        }

        throw new FileNotFoundException("Whisper decoder model was not found next to the encoder model.", candidatePath);
    }
}
