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
    private async Task SmokeTestAsrAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        string engineFamily = request.EngineFamily?.Trim() ?? string.Empty;
        if (engineFamily.Equals("qwen3-asr", StringComparison.OrdinalIgnoreCase))
        {
            // Pool key: production Qwen3 omits modelId/variant; keep smoke identical so the
            // session proven here is the one the ASR stage reuses.
            await SmokeTestQwen3AsrAsync(request.EntryPath, request.ExecutionProvider, cancellationToken).ConfigureAwait(false);
            return;
        }

        if (engineFamily.Equals("nemotron-asr", StringComparison.OrdinalIgnoreCase))
        {
            await SmokeTestNemotronAsrAsync(request, cancellationToken).ConfigureAwait(false);
            return;
        }

        if (engineFamily.Equals(ParakeetTdtOnnxAudioTranscriptionEngine.EngineFamilyName, StringComparison.OrdinalIgnoreCase))
        {
            await SmokeTestParakeetTdtAsync(request, cancellationToken).ConfigureAwait(false);
            return;
        }

        if (engineFamily.Equals("whisper-onnx", StringComparison.OrdinalIgnoreCase))
        {
            await SmokeTestWhisperAsync(request, cancellationToken).ConfigureAwait(false);
            return;
        }

        if (UsesOrtGenAiModelLoad(engineFamily))
        {
            await SmokeTestGenAiLoadAsync(request, cancellationToken)
                .ConfigureAwait(false);
            return;
        }

        await SmokeTestGenericSessionAsync(request, cancellationToken)
            .ConfigureAwait(false);
    }

    private static async Task SmokeTestQwen3AsrAsync(
        string encoderModelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken)
    {
        string root = Path.GetDirectoryName(encoderModelPath)
            ?? throw new InvalidOperationException("Qwen3-ASR smoke test could not resolve model root.");
        string decoderInitPath = Path.Join(root, "decoder_init.onnx");
        string decoderStepPath = Path.Join(root, "decoder_step.onnx");
        using OnnxExecutionSessionFactory.Qwen3AsrSessionLease sessionLease = await OnnxExecutionSessionFactory
            .CreatePooledQwen3AsrAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "qwen3-asr", Provider = provider, CancellationToken = cancellationToken, AdditionalTrtEncoderOptions = Qwen3AsrOnnxAudioTranscriptionEngine.TrtEncoderOptions }, new OnnxExecutionSessionFactory.QwenAsrModelPaths(encoderModelPath, decoderInitPath, decoderStepPath))
            .ConfigureAwait(false);
        EnsureSelectedProviderMatchesRequested(provider, sessionLease.SelectedProvider);

        using var encoderInputs = new InputSet([
            NamedOnnxValue.CreateFromTensor("mel", new DenseTensor<float>(new float[128], [1, 128, 1]))
        ]);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults =
            sessionLease.EncoderSession.RunWithRetry(encoderInputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: provider);
        Tensor<float> audioFeatures = encoderResults.First().AsTensor<float>();
        Qwen3AsrGreedyDecoder.RunSmokeInitAndStep(sessionLease, audioFeatures, provider, cancellationToken);
    }

    private static async Task SmokeTestNemotronAsrAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        string encoderModelPath = request.EntryPath;
        string decoderJointPath = ResolveNemotronDecoderJointPath(encoderModelPath);
        NemotronAsrEncoderTrtProfiles.SessionOptions sessionOptions =
            NemotronAsrEncoderTrtProfiles.BuildRuntimeOptions(encoderModelPath);
        // modelId/variant must match NemotronAsrOnnxAudioTranscriptionEngine or the stage
        // misses this pooled session and pays cold load again.
        using OnnxExecutionSessionFactory.NemotronAsrSessionLease sessionLease = await OnnxExecutionSessionFactory
            .CreatePooledNemotronAsrAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "nemotron-asr", Provider = request.ExecutionProvider, CancellationToken = cancellationToken, ModelId = request.ModelId, Variant = request.Variant, AdditionalTrtEncoderOptions = sessionOptions.Encoder, AdditionalTrtDecoderOptions = sessionOptions.Decoder }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = encoderModelPath, Decoder = decoderJointPath })
            .ConfigureAwait(false);
        EnsureSelectedProviderMatchesRequested(request.ExecutionProvider, sessionLease.SelectedProvider);

        using var encoderInputs = CreateNemotronEncoderInputs(sessionLease.EncoderSession.InputMetadata);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults =
            sessionLease.EncoderSession.RunWithRetry(encoderInputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: request.ExecutionProvider);
        Tensor<float> encoded = encoderResults.Single(static result => result.Name == "encoded").AsTensor<float>();
        using var decoderInputs = CreateNemotronDecoderInputs(sessionLease.DecoderJointSession.InputMetadata, encoded);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _ = sessionLease.DecoderJointSession.RunWithRetry(decoderInputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: request.ExecutionProvider);
    }

    private static async Task SmokeTestParakeetTdtAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        string encoderModelPath = request.EntryPath;
        string root = Path.GetDirectoryName(encoderModelPath)
            ?? throw new InvalidOperationException("Parakeet-TDT smoke test could not resolve model root.");
        // Preprocessor is a separate pooled CPU graph in production; warm the same key so the
        // stage does not rebuild nemo128.onnx after a successful smoke.
        using OnnxExecutionSessionFactory.SingleSessionLease preprocessor = await OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "parakeet-tdt-preprocessor", Provider = ExecutionProviderKind.Cpu, CancellationToken = cancellationToken }, Path.Join(root, "nemo128.onnx"))
            .ConfigureAwait(false);
        EnsureSelectedProviderMatchesRequested(ExecutionProviderKind.Cpu, preprocessor.SelectedProvider);

        using OnnxExecutionSessionFactory.NemotronAsrSessionLease sessionLease = await OnnxExecutionSessionFactory
            .CreatePooledNemotronAsrAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = ParakeetTdtOnnxAudioTranscriptionEngine.EngineFamilyName, Provider = request.ExecutionProvider, CancellationToken = cancellationToken, ModelId = request.ModelId, Variant = request.Variant, AdditionalTrtEncoderOptions = ParakeetTdtOnnxAudioTranscriptionEngine.TrtEncoderOptions }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = encoderModelPath, Decoder = Path.Join(root, "decoder_joint-model.onnx") })
            .ConfigureAwait(false);
        EnsureSelectedProviderMatchesRequested(request.ExecutionProvider, sessionLease.SelectedProvider);

        const int melFrames = 100;
        using var encoderInputs = new InputSet([
            NamedOnnxValue.CreateFromTensor("audio_signal", new DenseTensor<float>(new float[128 * melFrames], [1, 128, melFrames])),
            NamedOnnxValue.CreateFromTensor("length", new DenseTensor<long>(new long[] { melFrames }, [1])),
        ]);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults =
            sessionLease.EncoderSession.RunWithRetry(encoderInputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: request.ExecutionProvider);
        Tensor<float> encoded = encoderResults.Single(static result => result.Name == "outputs").AsTensor<float>();
        using var decoderInputs = CreateNemotronDecoderInputs(sessionLease.DecoderJointSession.InputMetadata, encoded);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _ = sessionLease.DecoderJointSession.RunWithRetry(decoderInputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: request.ExecutionProvider);
    }

    private static InputSet CreateNemotronEncoderInputs(IReadOnlyDictionary<string, NodeMetadata> inputMetadata)
    {
        int[] channelDims = ResolveMetadataDims(inputMetadata, "cache_last_channel", [24, 1, 56, 1024]);
        int[] timeDims = ResolveMetadataDims(inputMetadata, "cache_last_time", [24, 1, 1024, 8]);
        int channelCount = channelDims.Aggregate(1, static (product, dimension) => checked(product * dimension));
        int timeCount = timeDims.Aggregate(1, static (product, dimension) => checked(product * dimension));
        // Bundled Nemotron export expects mel-major [B,mel,T], same layout the engine feeds.
        var values = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("processed_signal", new DenseTensor<float>(new float[128 * 65], [1, 128, 65])),
            NamedOnnxValue.CreateFromTensor("processed_signal_length", new DenseTensor<long>(new long[] { 65 }, [1])),
            NamedOnnxValue.CreateFromTensor("cache_last_channel", new DenseTensor<float>(new float[channelCount], channelDims)),
            NamedOnnxValue.CreateFromTensor("cache_last_time", new DenseTensor<float>(new float[timeCount], timeDims)),
            NamedOnnxValue.CreateFromTensor("cache_last_channel_len", new DenseTensor<long>(new long[] { 0 }, [1]))
        };

        if (inputMetadata.ContainsKey("prompt_index"))
        {
            values.Add(NamedOnnxValue.CreateFromTensor("prompt_index", new DenseTensor<long>(new long[] { 101 }, [1])));
        }

        return new InputSet(values);
    }

    private static InputSet CreateNemotronDecoderInputs(
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata,
        Tensor<float> encoded)
    {
        // The encoder output may be time-major [B,T,H] or hidden-major [B,H,T] depending on
        // the provider's graph; resolve the layout the same way the engine does. The decoder's
        // encoder_outputs metadata declares hidden at index 1 ([batch, hidden, time]).
        int hiddenDim = inputMetadata.TryGetValue("encoder_outputs", out NodeMetadata? encoderOutputsMetadata)
            && encoderOutputsMetadata.Dimensions.Length == 3
            && encoderOutputsMetadata.Dimensions[1] > 0
                ? encoderOutputsMetadata.Dimensions[1]
                : 1024;
        NemotronAsrEncodedTensorLayout.EncodedLayout layout = NemotronAsrEncodedTensorLayout.Resolve(
            encoded.Dimensions,
            encodedLength: int.MaxValue,
            hiddenDim);
        DenseTensor<float> frame = NemotronAsrEncodedTensorLayout.SliceFrame(encoded, layout, frameIndex: 0);
        int[] stateDims = ResolveMetadataDims(inputMetadata, "input_states_1", [2, 1, 640]);
        int stateCount = stateDims.Aggregate(1, static (product, dimension) => checked(product * dimension));

        return new InputSet(
        [
            NamedOnnxValue.CreateFromTensor("encoder_outputs", frame),
            CreateNemotronDecoderTokenInput(inputMetadata, "targets", 0, [1, 1]),
            CreateNemotronDecoderTokenInput(inputMetadata, "target_length", 1, [1]),
            NamedOnnxValue.CreateFromTensor("input_states_1", new DenseTensor<float>(new float[stateCount], stateDims)),
            NamedOnnxValue.CreateFromTensor("input_states_2", new DenseTensor<float>(new float[stateCount], stateDims))
        ]);
    }

    private static NamedOnnxValue CreateNemotronDecoderTokenInput(
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata,
        string inputName,
        long value,
        int[] dimensions)
    {
        TensorElementType elementType = inputMetadata.TryGetValue(inputName, out NodeMetadata? metadata)
            ? metadata.ElementDataType
            : TensorElementType.Int64;   // Nemotron decoder graph exports targets/target_length as int64 (per export + review P2); smoke must not default to int32 or provider verification fails with type mismatch.

        return elementType switch
        {
            TensorElementType.Int32 => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<int>(new[] { checked((int)value) }, dimensions)),
            TensorElementType.Int64 => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<long>(new[] { value }, dimensions)),
            _ => throw new NotSupportedException(
                $"Nemotron decoder input '{inputName}' uses unsupported token tensor element type '{elementType}'.")
        };
    }

    private static string ResolveNemotronDecoderJointPath(string encoderModelPath)
    {
        string candidatePath = Path.Join(Path.GetDirectoryName(encoderModelPath)!, "decoder_joint.onnx");
        if (File.Exists(candidatePath))
        {
            return Path.GetFullPath(candidatePath);
        }

        throw new FileNotFoundException("Nemotron ASR decoder_joint.onnx was not found next to encoder.onnx.", candidatePath);
    }
}
