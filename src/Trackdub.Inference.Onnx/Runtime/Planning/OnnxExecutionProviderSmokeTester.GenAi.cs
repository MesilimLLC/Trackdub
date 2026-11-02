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
    /// <summary>Cold-start evidence source label for the GenAI smoke's generated token.</summary>
    internal const string GenAiSmokeSource = "smoke:genai";

    /// <summary>Signals a non-evicting GenAI warmup that declined to load its model.</summary>
    private sealed class GenAiWarmupSkippedException(string message) : Exception(message);

    /// <summary>
    /// Leases a GenAI model for a smoke. Authoritative smokes use the regular
    /// (evicting) lease; non-evicting warmup smokes lease only an already-resident
    /// model or a free slot and throw <see cref="GenAiWarmupSkippedException"/> when
    /// loading would evict a resident model.
    /// </summary>
    private async Task<GenAiModelLease> GetGenAiLeaseForSmokeAsync(
        GenAiModelKey key,
        bool nonEvictingWarmup,
        CancellationToken cancellationToken)
    {
        if (!nonEvictingWarmup)
        {
            return await genAiModelPool.GetLeaseAsync(key, cancellationToken)
                .ConfigureAwait(false);
        }

        GenAiModelLease? lease = await genAiModelPool
            .TryGetWarmupLeaseAsync(key, cancellationToken).ConfigureAwait(false);
        return lease ?? throw new GenAiWarmupSkippedException(
            $"GenAI warmup skipped for '{key}': loading it would evict a resident model " +
            "(GenAI pool at capacity or admission budget full). The stage will load it inline.");
    }

    private async Task SmokeTestTextRefinementGenAiAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        ThrowIfGenAiTensorRtProvider(request.ExecutionProvider);

        string genAiRoot = RequireGenAiConfigRoot(
            request,
            "Text refinement smoke test requires genai_config.json in the model root.");

        // Same key the stage adapters use: the model the smoke creates is the one the
        // stage leases next. The lease keeps the shared Model exclusive for the probe.
        GenAiModelKey key = await GenAiModelKey.CreateAsync(
            genAiRoot, request.ExecutionProvider, request.ModelId, request.Variant, deviceId: null,
            modelRevisionHash: request.ModelRevisionHash, cancellationToken: cancellationToken).ConfigureAwait(false);
        using GenAiModelLease lease = await GetGenAiLeaseForSmokeAsync(
            key, request.NonEvictingWarmup, cancellationToken).ConfigureAwait(false);
        using IDisposable? executionAdmission = await CpuExecutionAdmission.Shared
            .AcquireAsync(request.ExecutionProvider, cancellationToken).ConfigureAwait(false);
        Model model = lease.Model;

        await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            using Tokenizer tokenizer = new(model);
            using GeneratorParams generatorParams = new(model);
            using Sequences input = tokenizer.Encode("Hello");
            using Generator generator = new(model, generatorParams);
            generator.AppendTokenSequences(input);
            generator.GenerateNextToken();
            ColdStartCapture.MarkInferenceCompleted(GenAiSmokeSource);
        }, cancellationToken).ConfigureAwait(false);
    }

    internal static bool UsesOrtGenAiModelLoad(string? engineFamily) =>
        engineFamily is not null
        && engineFamily.Equals("whisper-genai", StringComparison.OrdinalIgnoreCase);

    internal static bool UsesOrtGenAiTranslationSmoke(string? engineFamily) =>
        engineFamily is not null
        && (engineFamily.Equals("phi-genai", StringComparison.OrdinalIgnoreCase)
            || engineFamily.Equals("qwen-instruct", StringComparison.OrdinalIgnoreCase));

    private async Task SmokeTestGenAiLoadAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        ThrowIfGenAiTensorRtProvider(request.ExecutionProvider);

        string genAiRoot = RequireGenAiConfigRoot(
            request,
            "GenAI smoke test requires genai_config.json in the model root.");

        // Same key WhisperGenAiAudioTranscriptionEngine uses: the smoke warms the shared
        // resident model and exercises the real audio+generation profile on silence.
        GenAiModelKey key = await GenAiModelKey.CreateAsync(
            genAiRoot, request.ExecutionProvider, request.ModelId, request.Variant, deviceId: null,
            modelRevisionHash: request.ModelRevisionHash, cancellationToken: cancellationToken).ConfigureAwait(false);
        using GenAiModelLease lease = await GetGenAiLeaseForSmokeAsync(
            key, request.NonEvictingWarmup, cancellationToken).ConfigureAwait(false);
        using IDisposable? executionAdmission = await CpuExecutionAdmission.Shared
            .AcquireAsync(request.ExecutionProvider, cancellationToken).ConfigureAwait(false);
        await WhisperGenAiAudioTranscriptionEngine
            .WarmModelAsync(lease.Model, cancellationToken)
            .ConfigureAwait(false);
    }

    // Approved-export guard for the bundled GenAI graphs. Those exports are CPU/CUDA-targeted.
    // A fatal crash cannot be reported as a smoke failure, so the attempt is refused until an
    // export built for the provider is retested in an isolated process. Retest history:
    // GenAI 0.17.1 / Windows ML 2.4 terminated the host on NvTensorRtRtx, and the 2026-10-18
    // isolated-process retest (GenAI 0.17.1, ORT 1.31, Linux container, RTX 5070) confirmed it
    // on the current stack: the NvTensorRtRtx device aborted the host process (exit 134).
    // Classic TensorRT is not an ORT GenAI device at all ("NvTensorRtRtx execution provider
    // is not supported in this build"), so both TensorRT kinds stay refused here.
    private static void ThrowIfGenAiTensorRtProvider(ExecutionProviderKind provider)
    {
        if (provider is ExecutionProviderKind.TensorRTRtx or ExecutionProviderKind.TensorRt)
        {
            throw new NotSupportedException(
                "Bundled GenAI exports are excluded from NvTensorRtRtx: the NvTensorRtRtx device "
                + "terminated the host process in isolated retests (GenAI 0.17.1, ORT 1.31). This is an "
                + "approved-export guard, not a provider-wide ban. Use cpu or cuda until a "
                + "TensorRT-targeted GenAI export is retested.");
        }

        if (provider is ExecutionProviderKind.DirectMl)
        {
            throw new NotSupportedException(
                "Bundled GenAI exports are excluded from DirectML: on GenAI 0.17.1 / Windows ML 2.4 "
                + "those CPU/CUDA exports terminated the host or failed their first DirectML kernel. "
                + "This is an approved-export guard, not a provider-wide ban. Use cpu until a DirectML-targeted export is retested.");
        }
    }

    // TensorRT support is a property of the export, not the model family: an export qualifies
    // when it was built for TensorRT (plain graphs, bucketed static shapes) and has passed an
    // isolated-process smoke on the provider. Session construction for an unqualified
    // encoder-decoder export can terminate the host process (native stack overflow), and a
    // fatal crash cannot be caught and reported, so qualification is checked before session
    // creation; the smoke sweep needs this guard because it bypasses stage allow-lists.
    // Extending support means producing a TensorRT-built export of another variant and
    // qualifying it here after an isolated retest, not lifting the guard wholesale.
    // OPUS-MT is split by provider: its merged-decoder export qualified on classic TensorRT
    // in the 2026-10-18 retest (smoke passed), but session construction on TensorRT-RTX
    // aborted the host process in the same retest (exit 134), so the RTX refusal stays.
    internal static void ThrowIfFatalTensorRtFamily(ExecutionProviderSmokeTestRequest request)
    {
        string? engineFamily = request.EngineFamily;
        ExecutionProviderKind provider = request.ExecutionProvider;
        // Classic TensorRT import of the bundled Qwen3-ASR export terminated the host process
        // (2026-10-18 isolated retest). Refuse before any native call so a sweep keeps going.
        if (string.Equals(engineFamily, "qwen3-asr", StringComparison.OrdinalIgnoreCase) &&
            provider == ExecutionProviderKind.TensorRt)
        {
            throw new NotSupportedException(
                "Engine family 'qwen3-asr' is not TensorRT-qualified on classic TensorRT: "
                + "session import terminates the host process. Use TensorRT-RTX or another provider.");
        }

        if (!IsTensorRtProvider(provider))
        {
            return;
        }

        if (IsUnqualifiedTensorRtExport(request))
        {
            throw new NotSupportedException(
                $"Export for engine family '{engineFamily}' is not TensorRT-qualified: "
                + "InferenceSession construction for an unqualified encoder-decoder export "
                + "terminates the host process (native stack overflow). Qualify a "
                + "TensorRT-built export via an isolated-process retest, or use dml or cpu.");
        }
    }

    private static bool IsTensorRtProvider(ExecutionProviderKind provider) =>
        provider is ExecutionProviderKind.TensorRTRtx or ExecutionProviderKind.TensorRt;

    private static bool IsUnqualifiedTensorRtExport(ExecutionProviderSmokeTestRequest request)
    {
        if (string.Equals(request.EngineFamily, "madlad", StringComparison.OrdinalIgnoreCase))
        {
            return !IsTensorRtQualifiedMadladExport(request);
        }

        return string.Equals(request.EngineFamily, "opus-mt", StringComparison.OrdinalIgnoreCase)
            && request.ExecutionProvider == ExecutionProviderKind.TensorRTRtx;
    }

    // The only currently qualified MADLAD export: the pinned trt-fp16 build (plain
    // encoder/decoder under trt_rtx_mixed_fp16_fp32/). It passed under TensorRT RTX and,
    // in an isolated-process retest on 2026-10-18 (ORT 1.31, TensorRT 10.13, RTX 5070),
    // under classic TensorRT. The quantized, merged-decoder, and int4-kv exports were
    // never built for TensorRT and remain unqualified on both providers.
    private static bool IsTensorRtQualifiedMadladExport(ExecutionProviderSmokeTestRequest request)
    {
        if (!request.Variant.Equals("trt-fp16", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        return string.Equals(
            Path.GetFileName(Path.GetDirectoryName(request.EntryPath)),
            "trt_rtx_mixed_fp16_fp32",
            StringComparison.OrdinalIgnoreCase);
    }

    private static string RequireGenAiConfigRoot(
        ExecutionProviderSmokeTestRequest request,
        string missingConfigMessage)
    {
        string genAiRoot = PlannedRuntimeModelResolver.ResolveGenAiModelRoot(request.ModelRootPath, request.EntryPath);
        string configPath = Path.Join(genAiRoot, "genai_config.json");
        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException($"{missingConfigMessage} Missing: {configPath}", configPath);
        }

        return genAiRoot;
    }

    // Fatal-CTP guards first: these families hard-crash the host under TRT, and the
    // refusal must happen before any bootstrap/native touch (a crash cannot be caught).
    private static void ThrowIfFatalSmokeFamily(ExecutionProviderSmokeTestRequest request)
    {
        if (RequiresGenAiFatalGuard(request))
        {
            ThrowIfGenAiTensorRtProvider(request.ExecutionProvider);
        }

        ThrowIfFatalTensorRtFamily(request);
    }

    private static bool RequiresGenAiFatalGuard(ExecutionProviderSmokeTestRequest request)
    {
        if (request.Stage is RuntimeStage.TextRefinement)
        {
            return true;
        }

        return UsesOrtGenAiModelLoad(request.EngineFamily)
            || UsesOrtGenAiTranslationSmoke(request.EngineFamily);
    }
}
