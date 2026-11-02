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
    private async Task SmokeTestTranslationAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        ThrowIfFatalTensorRtFamily(request);

        if (UsesOrtGenAiTranslationSmoke(request.EngineFamily))
        {
            await SmokeTestTextRefinementGenAiAsync(request, cancellationToken).ConfigureAwait(false);
            return;
        }

        string encoderModelPath = ResolveTranslationEncoderPath(request.EntryPath);
        string decoderModelPath = ResolveOpusDecoderPath(encoderModelPath, request.ModelAlias);
        if (string.Equals(request.EngineFamily, "madlad", StringComparison.OrdinalIgnoreCase))
        {
            await Madlad.MadladTranslationEngine.SmokeTestAsync(
                request, encoderModelPath, decoderModelPath, cancellationToken).ConfigureAwait(false);
            return;
        }

        // Pooled with the same family/options OpusMtTranslationEngine uses, so the
        // sessions this smoke proves are the ones the translation stage leases next.
        // The non-pooled CreateOpusAsync pair would be disposed here and rebuilt by
        // the stage (a second, serial session create inside the stage itself).
        string engineFamily = string.IsNullOrWhiteSpace(request.EngineFamily)
            ? OpusMtTranslationEngine.EngineFamilyName
            : request.EngineFamily;
        try
        {
            using OnnxExecutionSessionFactory.OpusSessionLease sessionLease = await OnnxExecutionSessionFactory
                .CreatePooledOpusAsync(
                    engineFamily,
                    encoderModelPath, decoderModelPath, request.ExecutionProvider, cancellationToken)
                .ConfigureAwait(false);
            EnsureSelectedProviderMatchesRequested(request.ExecutionProvider, sessionLease.SelectedProvider);

            using var encoderInputs = new InputSet([
                NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(new long[] { 0L }, [1, 1])),
                NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(new long[] { 1L }, [1, 1]))
            ]);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = sessionLease.EncoderSession.RunWithRetry(encoderInputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: request.ExecutionProvider);
            Tensor<float> encoderHiddenStates = encoderResults
                .Single(static r => r.Name == "last_hidden_state")
                .AsTensor<float>();

            using var decoderInputs = CreateTranslationDecoderInputs(
                sessionLease.DecoderSession.InputMetadata,
                encoderHiddenStates);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _ = sessionLease.DecoderSession.RunWithRetry(decoderInputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: request.ExecutionProvider);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            try
            {
                // Do not leave an unverified encoder/decoder pair in the shared pool
                // for a later stage to lease (mirrors the VAD/Whisper smoke paths).
                await InferenceSessionPool.Shared.EvictModelAsync(engineFamily).ConfigureAwait(false);
            }
            catch (Exception)
            {
                // Cleanup only: an eviction failure must not replace the smoke-test failure rethrown below.
            }

            throw;
        }
    }

    private static InputSet CreateTranslationDecoderInputs(
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata,
        Tensor<float> encoderHiddenStates)
    {
        var attentionMask = new DenseTensor<long>(new long[] { 1L }, [1, 1]);
        var inputIds = new DenseTensor<long>(new long[] { 0L }, [1, 1]);

        var values = new List<NamedOnnxValue>(inputMetadata.Count);
        foreach ((string inputName, NodeMetadata metadata) in inputMetadata)
        {
            values.Add(inputName switch
            {
                "input_ids" => NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
                "encoder_hidden_states" => NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                "attention_mask" => NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask),
                "encoder_attention_mask" => NamedOnnxValue.CreateFromTensor("encoder_attention_mask", attentionMask),
                "use_cache_branch" => NamedOnnxValue.CreateFromTensor("use_cache_branch", new DenseTensor<bool>(new[] { false }, new[] { 1 })),
                _ when inputName.StartsWith("past_key_values.", StringComparison.Ordinal) =>
                    NamedOnnxValue.CreateFromTensor(inputName, CreateEmptyPastTensor(metadata)),
                _ => throw new NotSupportedException($"Translation decoder input '{inputName}' is not supported in smoke test.")
            });
        }

        return new InputSet(values);
    }

    private static DenseTensor<float> CreateEmptyPastTensor(NodeMetadata metadata)
    {
        int[] sourceDims = metadata.Dimensions;
        int[] dims = new int[sourceDims.Length];
        for (int i = 0; i < sourceDims.Length; i++)
        {
            dims[i] = sourceDims[i] > 0 ? sourceDims[i] : 1;
        }

        dims[0] = 1;                       // batch = 1
        if (dims.Length > 2) dims[2] = 0; // sequence = 0 (empty KV cache)
        return new DenseTensor<float>(Array.Empty<float>(), dims);
    }

    private static string ResolveTranslationEncoderPath(string entryPath)
    {
        string fileName = Path.GetFileName(entryPath);
        if (fileName.StartsWith("decoder_", StringComparison.OrdinalIgnoreCase))
        {
            string directory = Path.GetDirectoryName(entryPath)!;
            string[] candidates = ["encoder_model.onnx", "encoder_model_quantized.onnx", "encoder_model_fp16.onnx", "encoder_model_int8.onnx"];
            foreach (string candidate in candidates)
            {
                string candidatePath = Path.Join(directory, candidate);
                if (File.Exists(candidatePath))
                {
                    return Path.GetFullPath(candidatePath);
                }
            }

            // A decoder file was given but no encoder exists alongside it. Returning the
            // decoder path as the 'encoder' would produce a confusing ONNX load error later.
            // Fail fast with a clear message naming the expected files and directory.
            throw new FileNotFoundException(
                $"Translation encoder model not found in '{directory}'. " +
                $"Expected one of: {string.Join(", ", candidates)}. " +
                "Ensure the encoder model file is present alongside the decoder model.",
                Path.Join(directory, candidates[0]));
        }

        return entryPath;
    }

    private static string ResolveOpusDecoderPath(string encoderModelPath, string modelAlias)
    {
        // Match the decoder preference order used by the active engine family so that the
        // smoke test exercises the same files that translation will actually load at runtime.
        // MADLAD prefers int8 → default → merged; OpusMT and unknowns prefer merged → default → int8.
        bool isMadlad = modelAlias.Contains("madlad", StringComparison.OrdinalIgnoreCase);
        string[] candidates = isMadlad
            ? ["decoder_model_quantized.onnx", "decoder_model_int8.onnx", "decoder_model.onnx", "decoder_model_merged.onnx"]
            : ["decoder_model_merged.onnx", "decoder_model.onnx", "decoder_model_int8.onnx"];

        string directory = Path.GetDirectoryName(encoderModelPath)!;
        foreach (string candidate in candidates)
        {
            string candidatePath = Path.Join(directory, candidate);
            if (File.Exists(candidatePath))
            {
                return Path.GetFullPath(candidatePath);
            }
        }

        string fallback = Path.Join(directory, "decoder_model.onnx");
        throw new FileNotFoundException("Opus decoder model was not found next to the encoder model.", fallback);
    }
}
