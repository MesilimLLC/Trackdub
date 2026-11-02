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

public sealed class OnnxExecutionProviderSmokeTester : IExecutionProviderSmokeTester
{
    /// <summary>Cold-start evidence source label for the GenAI smoke's generated token.</summary>
    internal const string GenAiSmokeSource = "smoke:genai";

    private readonly GenAiModelPool genAiModelPool;

    public OnnxExecutionProviderSmokeTester()
        : this(GenAiModelPool.Shared)
    {
    }

    internal OnnxExecutionProviderSmokeTester(GenAiModelPool genAiModelPool)
    {
        this.genAiModelPool = genAiModelPool;
    }

    /// <summary>
    /// Smoke-tests engine families hosted by the ORT 1.30 inference worker on CUDA through that
    /// worker, since this process's ONNX Runtime cannot create CUDA sessions (ADR-0017).
    /// </summary>
    public OnnxExecutionProviderSmokeTester(IInferenceWorkerClient workerClient)
        : this(GenAiModelPool.Shared)
    {
        this.workerClient = workerClient ?? throw new ArgumentNullException(nameof(workerClient));
    }

    private readonly IInferenceWorkerClient? workerClient;

    private async Task<ExecutionProviderSmokeTestResult> SmokeTestThroughWorkerAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        var plan = new StageRuntimePlan
        {
            Stage = request.Stage,
            Status = StageRuntimePlanStatus.Ready,
            ModelId = request.ModelId,
            ModelAlias = request.ModelAlias,
            EngineFamily = request.EngineFamily,
            Variant = request.Variant,
            ExecutionProvider = request.ExecutionProvider,
            RequirePreferredExecutionProvider = true,
            ModelEntryPath = request.EntryPath,
            ModelRootPath = request.ModelRootPath,
            ModelRevisionHash = request.ModelRevisionHash,
        };
        try
        {
            TtsSynthesisResult result = await workerClient!.SynthesizeTtsAsync(
                new TtsSynthesisRequest("Hello.", "en", new VoiceCatalogEntry("af_heart", "en-us", "female", "Heart")),
                plan,
                cancellationToken).ConfigureAwait(false);
            bool onRequestedProvider = string.Equals(
                result.Provider,
                FormatProviderLabel(request.ExecutionProvider),
                StringComparison.OrdinalIgnoreCase);
            return new ExecutionProviderSmokeTestResult(
                onRequestedProvider && result.DurationSamples > 0,
                onRequestedProvider
                    ? $"Inference worker synthesized {result.DurationSamples} samples on {result.Provider}."
                    : $"Inference worker ran on '{result.Provider}' instead of the requested provider.");
        }
        catch (Exception ex) when (ex is InferenceWorkerException or InvalidOperationException or IOException)
        {
            return new ExecutionProviderSmokeTestResult(false, $"Inference worker smoke failed: {ex.Message}");
        }
    }

    public async Task<ExecutionProviderSmokeTestResult> SmokeTestAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();

        // Smoke and warmup inference is real inference that warms pooled sessions; attribute
        // its completions to the smoke phase so cold-start evidence separates it from the
        // first real stage inference.
        using IDisposable smokePhase = ColdStartCapture.EnterPhase(ColdStartPhase.Smoke);
        using IDisposable smokeSpan = BenchmarkPhaseCapture.Start("ep-smoke");

        if (request.ExecutionProvider is ExecutionProviderKind.Cuda &&
            workerClient is not null &&
            OnnxExecutionProviderDiscovery.WorkerCudaEngineFamilies.Contains(request.EngineFamily, StringComparer.OrdinalIgnoreCase) &&
            !CudaOrtProbe.IsCudaProviderListed())
        {
            return await SmokeTestThroughWorkerAsync(request, cancellationToken).ConfigureAwait(false);
        }

        try
        {
            // Fatal-CTP guards first: these families hard-crash the host under TRT, and the
            // refusal must happen before any bootstrap/native touch (a crash cannot be caught).
            if (request.Stage is RuntimeStage.TextRefinement
                || UsesOrtGenAiModelLoad(request.EngineFamily)
                || UsesOrtGenAiTranslationSmoke(request.EngineFamily))
            {
                ThrowIfGenAiTensorRtProvider(request.ExecutionProvider);
            }

            ThrowIfFatalTensorRtFamily(request.EngineFamily, request.ExecutionProvider, request.Variant, request.EntryPath);

            // Register/validate the requested EP before any probe session. When the bootstrapper
            // cannot keep the requested provider selected (e.g. TRT RTX plugin missing and
            // falling back to DirectML), the pair is unproven — fail fast with that detail
            // instead of creating a session that silently lands on a different provider.
            ExecutionProviderBootstrapResult bootstrap = await OnnxExecutionSessionFactory
                .BootstrapForSmokeAsync(request.ExecutionProvider, cancellationToken)
                .ConfigureAwait(false);
            if (!bootstrap.IsRequestFulfilled)
            {
                return new ExecutionProviderSmokeTestResult(
                    false,
                    $"Smoke bootstrap could not select {request.ExecutionProvider} "
                    + $"(selected {bootstrap.SelectedProvider}). {bootstrap.Detail}");
            }

            switch (request.Stage)
            {
                case RuntimeStage.Vad:
                    await SmokeTestVadAsync(request.EntryPath, request.ExecutionProvider, cancellationToken).ConfigureAwait(false);
                    break;
                case RuntimeStage.Asr:
                    await SmokeTestAsrAsync(request, cancellationToken).ConfigureAwait(false);
                    break;
                case RuntimeStage.Separation:
                    await SmokeTestSeparationAsync(request, cancellationToken).ConfigureAwait(false);
                    break;
                case RuntimeStage.Translation:
                    await SmokeTestTranslationAsync(request, cancellationToken).ConfigureAwait(false);
                    break;
                case RuntimeStage.Diarization:
                    await SmokeTestDiarizationAsync(request.EntryPath, request.ExecutionProvider, cancellationToken).ConfigureAwait(false);
                    break;
                case RuntimeStage.Tts:
                    await SmokeTestTtsAsync(
                        request.ModelId,
                        request.ModelAlias,
                        request.ModelRootPath,
                        request.EntryPath,
                        request.Variant,
                        request.ExecutionProvider,
                        cancellationToken).ConfigureAwait(false);
                    break;
                case RuntimeStage.TextRefinement:
                    await SmokeTestTextRefinementGenAiAsync(
                        request.ModelRootPath,
                        request.EntryPath,
                        request.ModelId,
                        request.Variant,
                        request.ModelRevisionHash,
                        request.ExecutionProvider,
                        request.NonEvictingWarmup,
                        cancellationToken).ConfigureAwait(false);
                    break;
                case RuntimeStage.SpeechEnhancement:
                case RuntimeStage.OverlapRescue:
                case RuntimeStage.LipSync:
                case RuntimeStage.LipSynthesis:
                    await SmokeTestGenericSessionAsync(request, cancellationToken)
                        .ConfigureAwait(false);
                    break;
                default:
                    return new ExecutionProviderSmokeTestResult(
                        false,
                        $"Smoke testing is not implemented for runtime stage '{request.Stage}'.");
            }

            return new ExecutionProviderSmokeTestResult(true);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (GenAiWarmupSkippedException ex)
        {
            // Honest non-attempt: warmup declined to evict a resident model. No provider
            // verdict was produced, so this must not read as a smoke failure.
            return new ExecutionProviderSmokeTestResult(false, ex.Message) { NotAttempted = true };
        }
        catch (Exception ex)
        {
            return new ExecutionProviderSmokeTestResult(false, ex.Message);
        }
    }

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
        string modelRootPath,
        string entryPath,
        string? modelId,
        string? variant,
        string? modelRevisionHash,
        ExecutionProviderKind provider,
        bool nonEvictingWarmup,
        CancellationToken cancellationToken)
    {
        ThrowIfGenAiTensorRtProvider(provider);

        string genAiRoot = RequireGenAiConfigRoot(
            modelRootPath,
            entryPath,
            "Text refinement smoke test requires genai_config.json in the model root.");

        // Same key the stage adapters use: the model the smoke creates is the one the
        // stage leases next. The lease keeps the shared Model exclusive for the probe.
        GenAiModelKey key = await GenAiModelKey.CreateAsync(
            genAiRoot, provider, modelId, variant, deviceId: null,
            modelRevisionHash: modelRevisionHash, cancellationToken: cancellationToken).ConfigureAwait(false);
        using GenAiModelLease lease = await GetGenAiLeaseForSmokeAsync(
            key, nonEvictingWarmup, cancellationToken).ConfigureAwait(false);
        using IDisposable? executionAdmission = await CpuExecutionAdmission.Shared
            .AcquireAsync(provider, cancellationToken).ConfigureAwait(false);
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

    private static async Task SmokeTestVadAsync(
        string modelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken)
    {
        // Keyed exactly like SileroVadSpeechRegionDetector, so the session this smoke verifies
        // is the one the VAD stage leases next instead of being built a second time.
        try
        {
            using OnnxExecutionSessionFactory.SingleSessionLease sessionLease = await OnnxExecutionSessionFactory
                .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = SileroVadPoolFamily, Provider = provider, CancellationToken = cancellationToken, AllowTrtInitFallback = true }, modelPath)
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

            using var input = CreateVadInputs();
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _ = sessionLease.Session.RunWithRetry(input.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: provider);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Do not leave a session that failed verification for the stage to reuse.
            await InferenceSessionPool.Shared.EvictModelAsync(SileroVadPoolFamily).ConfigureAwait(false);
            throw;
        }
    }

    private const string SileroVadPoolFamily = "silero-vad";

    private static async Task SmokeTestWhisperAsync(
        string encoderModelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken)
    {
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
                .CreatePooledWhisperAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = WhisperOnnxAudioTranscriptionEngine.EngineFamilyName, Provider = provider, CancellationToken = cancellationToken, AdditionalTrtEncoderOptions = encoderOptions, AdditionalTrtDecoderOptions = decoderOptions }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = encoderModelPath, Decoder = decoderModelPath })
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
            await SmokeTestWhisperAsync(request.EntryPath, request.ExecutionProvider, cancellationToken).ConfigureAwait(false);
            return;
        }

        if (UsesOrtGenAiModelLoad(engineFamily))
        {
            await SmokeTestGenAiLoadAsync(
                    request.ModelRootPath,
                    request.EntryPath,
                    request.ModelId,
                    request.Variant,
                    request.ModelRevisionHash,
                    request.ExecutionProvider,
                    request.NonEvictingWarmup,
                    cancellationToken)
                .ConfigureAwait(false);
            return;
        }

        await SmokeTestGenericSessionAsync(request, cancellationToken)
            .ConfigureAwait(false);
    }

    internal static bool UsesOrtGenAiModelLoad(string? engineFamily) =>
        engineFamily is not null
        && engineFamily.Equals("whisper-genai", StringComparison.OrdinalIgnoreCase);

    internal static bool UsesOrtGenAiTranslationSmoke(string? engineFamily) =>
        engineFamily is not null
        && (engineFamily.Equals("phi-genai", StringComparison.OrdinalIgnoreCase)
            || engineFamily.Equals("qwen-instruct", StringComparison.OrdinalIgnoreCase));

    private async Task SmokeTestGenAiLoadAsync(
        string modelRootPath,
        string entryPath,
        string? modelId,
        string? variant,
        string? modelRevisionHash,
        ExecutionProviderKind provider,
        bool nonEvictingWarmup,
        CancellationToken cancellationToken)
    {
        ThrowIfGenAiTensorRtProvider(provider);

        string genAiRoot = RequireGenAiConfigRoot(
            modelRootPath,
            entryPath,
            "GenAI smoke test requires genai_config.json in the model root.");

        // Same key WhisperGenAiAudioTranscriptionEngine uses: the smoke warms the shared
        // resident model and exercises the real audio+generation profile on silence.
        GenAiModelKey key = await GenAiModelKey.CreateAsync(
            genAiRoot, provider, modelId, variant, deviceId: null,
            modelRevisionHash: modelRevisionHash, cancellationToken: cancellationToken).ConfigureAwait(false);
        using GenAiModelLease lease = await GetGenAiLeaseForSmokeAsync(
            key, nonEvictingWarmup, cancellationToken).ConfigureAwait(false);
        using IDisposable? executionAdmission = await CpuExecutionAdmission.Shared
            .AcquireAsync(provider, cancellationToken).ConfigureAwait(false);
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
    internal static void ThrowIfFatalTensorRtFamily(
        string? engineFamily, ExecutionProviderKind provider, string variant, string entryPath)
    {
        // The only currently qualified MADLAD export: the pinned trt-fp16 build (plain
        // encoder/decoder under trt_rtx_mixed_fp16_fp32/). It passed under TensorRT RTX and,
        // in an isolated-process retest on 2026-10-18 (ORT 1.31, TensorRT 10.13, RTX 5070),
        // under classic TensorRT. The quantized, merged-decoder, and int4-kv exports were
        // never built for TensorRT and remain unqualified on both providers.
        bool isTensorRtQualifiedMadladExport =
            string.Equals(engineFamily, "madlad", StringComparison.OrdinalIgnoreCase) &&
            variant.Equals("trt-fp16", StringComparison.OrdinalIgnoreCase) &&
            string.Equals(Path.GetFileName(Path.GetDirectoryName(entryPath)),
                "trt_rtx_mixed_fp16_fp32", StringComparison.OrdinalIgnoreCase);
        bool unsafeMadlad = string.Equals(engineFamily, "madlad", StringComparison.OrdinalIgnoreCase) &&
            !isTensorRtQualifiedMadladExport;
        bool unsafeOpusMt = string.Equals(engineFamily, "opus-mt", StringComparison.OrdinalIgnoreCase) &&
            provider == ExecutionProviderKind.TensorRTRtx;
        // Classic TensorRT import of the bundled Qwen3-ASR export terminated the host process
        // (2026-10-18 isolated retest). Refuse before any native call so a sweep keeps going.
        if (string.Equals(engineFamily, "qwen3-asr", StringComparison.OrdinalIgnoreCase) &&
            provider == ExecutionProviderKind.TensorRt)
        {
            throw new NotSupportedException(
                "Engine family 'qwen3-asr' is not TensorRT-qualified on classic TensorRT: "
                + "session import terminates the host process. Use TensorRT-RTX or another provider.");
        }

        if (provider is ExecutionProviderKind.TensorRTRtx or ExecutionProviderKind.TensorRt
            && (unsafeMadlad || unsafeOpusMt))
        {
            throw new NotSupportedException(
                $"Export for engine family '{engineFamily}' is not TensorRT-qualified: "
                + "InferenceSession construction for an unqualified encoder-decoder export "
                + "terminates the host process (native stack overflow). Qualify a "
                + "TensorRT-built export via an isolated-process retest, or use dml or cpu.");
        }
    }

    private static string RequireGenAiConfigRoot(
        string modelRootPath,
        string entryPath,
        string missingConfigMessage)
    {
        string genAiRoot = PlannedRuntimeModelResolver.ResolveGenAiModelRoot(modelRootPath, entryPath);
        string configPath = Path.Join(genAiRoot, "genai_config.json");
        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException($"{missingConfigMessage} Missing: {configPath}", configPath);
        }

        return genAiRoot;
    }

    private static async Task SmokeTestGenericSessionAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        string poolFamily = ResolveGenericPoolFamily(request);
        using OnnxExecutionSessionFactory.SingleSessionLease sessionLease = await OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = poolFamily, Provider = request.ExecutionProvider, CancellationToken = cancellationToken, AllowTrtInitFallback = true }, request.EntryPath)
            .ConfigureAwait(false);
        EnsureSelectedProviderMatchesRequested(request.ExecutionProvider, sessionLease.SelectedProvider);

        using var inputs = CreateMetadataDrivenInputs(sessionLease.Session.InputMetadata);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _ = sessionLease.Session.RunWithRetry(inputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: request.ExecutionProvider);
    }

    /// <summary>
    /// Resolves the pool family for generic single-graph smoke targets so the key matches
    /// the production engine's CreatePooledSingleAsync family (DeepFilterNet, SepFormer, LatentSync).
    /// </summary>
    private static string ResolveGenericPoolFamily(ExecutionProviderSmokeTestRequest request)
    {
        string name = $"{request.EngineFamily} {request.ModelId} {Path.GetFileNameWithoutExtension(request.EntryPath)}";
        if (name.Contains("deepfilternet", StringComparison.OrdinalIgnoreCase) ||
            request.Stage is RuntimeStage.SpeechEnhancement)
        {
            if (name.Contains("erb", StringComparison.OrdinalIgnoreCase))
            {
                return "deepfilternet3-erb-dec";
            }

            if (name.Contains("df_dec", StringComparison.OrdinalIgnoreCase) ||
                name.Contains("df-dec", StringComparison.OrdinalIgnoreCase))
            {
                return "deepfilternet3-df-dec";
            }

            if (name.Contains("enc", StringComparison.OrdinalIgnoreCase))
            {
                return "deepfilternet3-enc";
            }
        }

        if (request.Stage is RuntimeStage.OverlapRescue)
        {
            return name.Contains("osd", StringComparison.OrdinalIgnoreCase)
                ? "sepformer-osd"
                : "sepformer";
        }

        if (request.Stage is RuntimeStage.LipSync or RuntimeStage.LipSynthesis)
        {
            return string.IsNullOrWhiteSpace(request.EngineFamily) ? "latentsync" : request.EngineFamily;
        }

        return string.IsNullOrWhiteSpace(request.EngineFamily) ? request.Stage.ToString().ToLowerInvariant() : request.EngineFamily;
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

    private static async Task SmokeTestSeparationAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        string runnableModelPath = request.EntryPath;
        string poolFamily = ResolveSeparationPoolFamily(request.EngineFamily, request.ModelId, runnableModelPath);
        using OnnxExecutionSessionFactory.SingleSessionLease sessionLease = await OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = poolFamily, Provider = request.ExecutionProvider, CancellationToken = cancellationToken, AllowTrtInitFallback = true }, runnableModelPath)
            .ConfigureAwait(false);
        EnsureSelectedProviderMatchesRequested(request.ExecutionProvider, sessionLease.SelectedProvider);

        using var inputs = CreateSeparationInputs(sessionLease.Session);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _ = sessionLease.Session.RunWithRetry(inputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: request.ExecutionProvider);
    }

    /// <summary>
    /// Maps a separation smoke target onto the pool family strings used by the production
    /// separators (Spleeter/SepFormer) so smoke and stage runs share one warm session.
    /// </summary>
    private static string ResolveSeparationPoolFamily(string? engineFamily, string modelId, string modelPath)
    {
        string name = $"{modelId} {Path.GetFileNameWithoutExtension(modelPath)} {engineFamily}";
        if (name.Contains("vocals", StringComparison.OrdinalIgnoreCase))
        {
            return "spleeter-vocals";
        }

        if (name.Contains("accompaniment", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("spleeter-acc", StringComparison.OrdinalIgnoreCase) ||
            (name.Contains("acc", StringComparison.OrdinalIgnoreCase) && name.Contains("spleeter", StringComparison.OrdinalIgnoreCase)))
        {
            return "spleeter-acc";
        }

        if (name.Contains("osd", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("overlap", StringComparison.OrdinalIgnoreCase))
        {
            return "sepformer-osd";
        }

        if (name.Contains("sepformer", StringComparison.OrdinalIgnoreCase))
        {
            return "sepformer";
        }

        return string.IsNullOrWhiteSpace(engineFamily) ? "separation" : engineFamily;
    }

    private static async Task SmokeTestDiarizationAsync(
        string modelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken)
    {
        // Same family + TRT profiles as SortFormerDiarizationEngine so the smoke run warms
        // the session the diarization stage will lease.
        using OnnxExecutionSessionFactory.SingleSessionLease sessionLease = await OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = SortFormerDiarizationEngine.EngineFamilyName, Provider = provider, CancellationToken = cancellationToken, AdditionalTrtOptions = SortFormerDiarizationEngine.TrtOptions, AllowTrtInitFallback = true }, modelPath)
            .ConfigureAwait(false);
        EnsureSelectedProviderMatchesRequested(provider, sessionLease.SelectedProvider);

        using var inputs = CreateDiarizationInputs(sessionLease.Session);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _ = sessionLease.Session.RunWithRetry(inputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: provider);
    }

    private static async Task SmokeTestTtsAsync(
        string modelId,
        string modelAlias,
        string modelRootPath,
        string entryPath,
        string variant,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken)
    {
        string modelPath = ResolveTtsProbeModelPath(modelId, modelAlias, modelRootPath, entryPath, variant);
        string poolFamily = ResolveTtsPoolFamily(modelId, modelAlias);
        using OnnxExecutionSessionFactory.SingleSessionLease sessionLease = await OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = poolFamily, Provider = provider, CancellationToken = cancellationToken, AllowTrtInitFallback = true }, modelPath)
            .ConfigureAwait(false);
        EnsureSelectedProviderMatchesRequested(provider, sessionLease.SelectedProvider);

        using var inputs = CreateTtsInputs(sessionLease.Session.InputMetadata);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _ = sessionLease.Session.RunWithRetry(inputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: provider);

        if (IsChatterboxTtsModel(modelId, modelAlias))
        {
            await WarmChatterboxSidecarsAsync(modelRootPath, entryPath, variant, cancellationToken).ConfigureAwait(false);
        }
        else if (IsCosyVoiceTtsModel(modelId, modelAlias))
        {
            await ProbeCosyVoiceGraphsAsync(modelRootPath, entryPath, variant, provider, cancellationToken)
                .ConfigureAwait(false);
        }
    }

    private static string ResolveTtsPoolFamily(string modelId, string modelAlias)
    {
        if (IsChatterboxTtsModel(modelId, modelAlias))
        {
            return "chatterbox";
        }

        if (IsCosyVoiceTtsModel(modelId, modelAlias))
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
        string modelRootPath,
        string entryPath,
        string variant,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken)
    {
        string rootPath = !string.IsNullOrWhiteSpace(modelRootPath)
            ? modelRootPath
            : Path.GetDirectoryName(entryPath)
                ?? throw new InvalidOperationException("Cannot resolve CosyVoice TTS smoke-test root path.");
        CosyVoiceModelFiles modelFiles = CosyVoiceModelFiles.Resolve(rootPath, variant);
        string entryFullPath = Path.GetFullPath(entryPath);
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
                .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "cosyvoice", Provider = provider, CancellationToken = cancellationToken, AllowTrtInitFallback = false }, graphPath)
                .ConfigureAwait(false);
            EnsureSelectedProviderMatchesRequested(provider, sessionLease.SelectedProvider);
        }
    }

    /// <summary>
    /// Pre-warms the three Chatterbox CPU sidecar graphs so the TTS stage does not pay
    /// their cold load after the smoke test already warmed the conditional decoder.
    /// Provider selection mirrors ChatterboxVoiceCloneTtsEngine (sidecars on CPU).
    /// </summary>
    private static async Task WarmChatterboxSidecarsAsync(
        string modelRootPath,
        string entryPath,
        string variant,
        CancellationToken cancellationToken)
    {
        string rootPath = !string.IsNullOrWhiteSpace(modelRootPath)
            ? modelRootPath
            : Path.GetDirectoryName(entryPath)
                ?? throw new InvalidOperationException("Cannot resolve Chatterbox TTS warmup root path.");
        string onnxDirectory = string.Equals(Path.GetFileName(rootPath), "onnx", StringComparison.OrdinalIgnoreCase)
            ? rootPath
            : Path.Join(rootPath, "onnx");

        foreach (string graphName in new[] { "speech_encoder", "embed_tokens", "language_model" })
        {
            string graphPath = ResolveChatterboxGraphPath(onnxDirectory, graphName, variant);
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

    private static InputSet CreateVadInputs()
    {
        IReadOnlyList<NamedOnnxValue> values =
        [
            NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(new float[512], [1, 512])),
            NamedOnnxValue.CreateFromTensor("state", new DenseTensor<float>(new float[2 * 128], [2, 1, 128])),
            NamedOnnxValue.CreateFromTensor("sr", new DenseTensor<long>(new long[] { 16000 }, [1]))
        ];

        return new InputSet(values);
    }

    // One 30-second mel window, matching the engine's real input and TRT profile shapes.
    private const int WhisperSmokeFrameCount = 3000;

    private static InputSet CreateWhisperEncoderInputs(
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata,
        int melBins)
    {
        var values = new List<NamedOnnxValue>(inputMetadata.Count);
        foreach ((string inputName, NodeMetadata metadata) in inputMetadata)
        {
            values.Add(CreateWhisperEncoderInput(inputName, metadata, melBins));
        }

        return new InputSet(values);
    }

    internal static int[] ResolveWhisperEncoderSmokeDimensionsForTesting(
        string inputName,
        IReadOnlyList<int> modelDimensions,
        int melBins) =>
        ResolveWhisperEncoderSmokeDimensions(inputName, modelDimensions, melBins);

    private static int[] ResolveWhisperEncoderSmokeDimensions(
        string inputName,
        IReadOnlyList<int> modelDimensions,
        int melBins)
    {
        // The encoder declares input_features as (batch_size, feature_size,
        // encoder_sequence_length) — all symbolic — so the generic all-ones smoke
        // tensor would be [1,1,1] and fail the first Conv (expects feature_size mel
        // channels). Feed a valid zero mel window instead.
        if (inputName.Equals("input_features", StringComparison.OrdinalIgnoreCase)
            && modelDimensions.Count == 3
            && melBins > 0)
        {
            return [1, melBins, WhisperSmokeFrameCount];
        }

        return modelDimensions.Select(static dimension => dimension > 0 ? dimension : 1).ToArray();
    }

    private static NamedOnnxValue CreateWhisperEncoderInput(
        string inputName,
        NodeMetadata metadata,
        int melBins)
    {
        if (metadata.IsTensor
            && metadata.ElementDataType == TensorElementType.Float
            && metadata.Dimensions.Length == 3
            && inputName.Equals("input_features", StringComparison.OrdinalIgnoreCase)
            && melBins > 0)
        {
            int[] dimensions = ResolveWhisperEncoderSmokeDimensions(inputName, metadata.Dimensions, melBins);
            int elementCount = dimensions.Aggregate(1, static (product, dimension) => checked(product * dimension));
            return NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<float>(new float[elementCount], dimensions));
        }

        return CreateSmokeTensorInput(inputName, metadata);
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

    private static int[] ResolveMetadataDims(
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata,
        string inputName,
        int[] fallback)
    {
        if (!inputMetadata.TryGetValue(inputName, out NodeMetadata? metadata) ||
            metadata.Dimensions.Length != fallback.Length)
        {
            return fallback;
        }

        var dims = new int[fallback.Length];
        for (int index = 0; index < dims.Length; index++)
        {
            dims[index] = metadata.Dimensions[index] > 0 ? metadata.Dimensions[index] : fallback[index];
        }

        return dims;
    }

    private static InputSet CreateSeparationInputs(InferenceSession session)
    {
        (string inputName, NodeMetadata metadata) = session.InputMetadata.First();
        int[] dimensions = ResolveSeparationSmokeInputDimensions(metadata.Dimensions);
        int elementCount = dimensions.Aggregate(1, static (product, dimension) => checked(product * dimension));
        return new InputSet(
        [
            NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<float>(new float[elementCount], dimensions))
        ]);
    }

    internal static int[] ResolveSeparationSmokeInputDimensionsForTesting(IReadOnlyList<int> modelDimensions) =>
        ResolveSeparationSmokeInputDimensions(modelDimensions);

    private static int[] ResolveSeparationSmokeInputDimensions(IReadOnlyList<int> modelDimensions)
    {
        // 4-D separation graphs are Spleeter STFT blocks. Their channel/split axis
        // order depends on the export (older: [2, num_splits, 512, 1024]; pinned
        // 3e5a4dd: [num_splits, 2, 512, 1024]), so reuse the production separator's
        // layout resolution instead of guessing one order: feeding [2, 1, 512, 1024]
        // to the splits-first export fails with "index: 1 Got: 1 Expected: 2".
        // Unmappable layouts throw there and surface as an honest smoke failure.
        if (modelDimensions.Count == 4)
        {
            return SpleeterOnnxSeparator.ResolveBlockDimensions(modelDimensions);
        }

        int[] dimensions = modelDimensions.Count switch
        {
            2 => [2, 44100],
            3 => [1, 2, 44100],
            _ => modelDimensions
                .Select(static dimension => dimension > 0 ? dimension : 1)
                .ToArray()
        };
        int sampleDimensionIndex = dimensions.Length - 1;
        dimensions[sampleDimensionIndex] = Math.Max(dimensions[sampleDimensionIndex], 44100);
        return dimensions;
    }

    private static int ResolvePositiveDimension(int dimension, int fallback) =>
        dimension > 0 ? dimension : fallback;

    private static InputSet CreateDiarizationInputs(InferenceSession session)
    {
        IReadOnlyDictionary<string, NodeMetadata> inputs = session.InputMetadata;
        if (IsSortFormerStreamingInputSet(inputs.Keys))
        {
            return CreateSortFormerStreamingInputs(inputs);
        }

        if (TryCreateWaveformDiarizationInputs(inputs, out InputSet? waveformInputs))
        {
            return waveformInputs!;
        }

        return CreateMetadataDrivenInputs(inputs);
    }

    private static bool IsSortFormerStreamingInputSet(IEnumerable<string> inputNames)
    {
        HashSet<string> names = inputNames.ToHashSet(StringComparer.Ordinal);
        return names.Contains("chunk") && names.Contains("chunk_lengths") &&
               names.Contains("spkcache") && names.Contains("spkcache_lengths") &&
               names.Contains("fifo") && names.Contains("fifo_lengths");
    }

    // The streaming export expects a fixed 1x3040x128 chunk. Cache and FIFO begin empty.
    private static InputSet CreateSortFormerStreamingInputs(
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata)
    {
        const int chunkFrames = 3040;
        const int melBins = 128;
        const int embeddingDimensions = 512;
        var values = new List<NamedOnnxValue>(6);
        AddFloatInput("chunk", [1, chunkFrames, melBins]);
        AddInt64Input("chunk_lengths", [1], [chunkFrames]);
        AddFloatInput("spkcache", [1, 0, embeddingDimensions]);
        AddInt64Input("spkcache_lengths", [1], [0]);
        AddFloatInput("fifo", [1, 0, embeddingDimensions]);
        AddInt64Input("fifo_lengths", [1], [0]);
        return new InputSet(values);

        void AddFloatInput(string name, int[] dimensions)
        {
            ValidateStreamingInput(name, dimensions, TensorElementType.Float);
            int elementCount = dimensions.Aggregate(1, static (product, dimension) => checked(product * dimension));
            values.Add(NamedOnnxValue.CreateFromTensor(
                name, new DenseTensor<float>(new float[elementCount], dimensions)));
        }

        void AddInt64Input(string name, int[] dimensions, long[] data)
        {
            ValidateStreamingInput(name, dimensions, TensorElementType.Int64);
            values.Add(NamedOnnxValue.CreateFromTensor(
                name, new DenseTensor<long>(data, dimensions)));
        }

        void ValidateStreamingInput(string name, int[] dimensions, TensorElementType elementType)
        {
            if (!inputMetadata.TryGetValue(name, out NodeMetadata? metadata) ||
                !metadata.IsTensor || metadata.ElementDataType != elementType ||
                metadata.Dimensions.Length != dimensions.Length)
            {
                throw new InvalidOperationException(
                    $"SortFormer streaming input '{name}' does not match the expected tensor contract.");
            }

            for (int axis = 0; axis < dimensions.Length; axis++)
            {
                if (metadata.Dimensions[axis] > 0 && metadata.Dimensions[axis] != dimensions[axis])
                {
                    throw new InvalidOperationException(
                        $"SortFormer streaming input '{name}' has an incompatible fixed dimension at axis {axis}.");
                }
            }
        }
    }

    private static bool TryCreateWaveformDiarizationInputs(
        IReadOnlyDictionary<string, NodeMetadata> inputs,
        out InputSet? inputSet)
    {
        inputSet = null;
        IReadOnlyDictionary<string, Type> inputElementTypes = inputs.ToDictionary(
            static kvp => kvp.Key,
            static kvp => kvp.Value.ElementType,
            StringComparer.Ordinal);

        try
        {
            (string waveformName, string? lengthName) = ResolveDiarizationInputNames(inputElementTypes);
            int[] waveformDims = ResolveDiarizationWaveformShape(inputs[waveformName].Dimensions);
            var values = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(waveformName, new DenseTensor<float>(new float[16000], waveformDims))
            };

            if (!string.IsNullOrWhiteSpace(lengthName))
            {
                values.Add(NamedOnnxValue.CreateFromTensor(lengthName, new DenseTensor<long>(new long[] { 16000 }, [1])));
            }

            inputSet = new InputSet(values);
            return true;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
    }

    private static InputSet CreateMetadataDrivenInputs(IReadOnlyDictionary<string, NodeMetadata> inputMetadata)
    {
        var values = new List<NamedOnnxValue>(inputMetadata.Count);
        foreach ((string inputName, NodeMetadata metadata) in inputMetadata)
        {
            values.Add(CreateSmokeTensorInput(inputName, metadata));
        }

        return new InputSet(values);
    }

    private static NamedOnnxValue CreateSmokeTensorInput(string inputName, NodeMetadata metadata)
    {
        if (!metadata.IsTensor)
        {
            throw new NotSupportedException($"Smoke test input '{inputName}' is not a tensor input.");
        }

        int[] dimensions = metadata.Dimensions.Select(static dimension => dimension > 0 ? dimension : 1).ToArray();
        int elementCount = dimensions.Aggregate(1, static (product, dimension) => checked(product * dimension));

        return metadata.ElementDataType switch
        {
            TensorElementType.Float => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<float>(new float[elementCount], dimensions)),
            TensorElementType.Float16 => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<Float16>(new Float16[elementCount], dimensions)),
            TensorElementType.Int32 => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<int>(new int[elementCount], dimensions)),
            TensorElementType.Int64 => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<long>(new long[elementCount], dimensions)),
            TensorElementType.Bool => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<bool>(new bool[elementCount], dimensions)),
            _ => throw new NotSupportedException(
                $"Smoke test input '{inputName}' uses unsupported tensor element type '{metadata.ElementDataType}'.")
        };
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
            int[] dims = ResolveChatterboxDecoderSmokeDimensions(inputName, metadata.Dimensions);
            NamedOnnxValue? value = CreateTtsInputValue(inputName, metadata.ElementType, dims);
            if (value is not null)
            {
                values.Add(value);
            }
        }

        return new InputSet(values);
    }

    internal static int[] ResolveChatterboxDecoderSmokeDimensionsForTesting(string inputName, IReadOnlyList<int> modelDimensions) =>
        ResolveChatterboxDecoderSmokeDimensions(inputName, modelDimensions);

    private static int[] ResolveChatterboxDecoderSmokeDimensions(string inputName, IReadOnlyList<int> modelDimensions)
    {
        if (inputName.Equals("speech_tokens", StringComparison.Ordinal)
            && modelDimensions.Count == 2)
        {
            return
            [
                modelDimensions[0] > 0 ? modelDimensions[0] : 1,
                ChatterboxDecoderSmokeSpeechTokenCount
            ];
        }

        if (inputName.Equals("speaker_features", StringComparison.Ordinal)
            && modelDimensions.Count == 3)
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

    private static string ResolveTtsProbeModelPath(
        string modelId,
        string modelAlias,
        string modelRootPath,
        string entryPath,
        string variant)
    {
        if (!IsChatterboxTtsModel(modelId, modelAlias))
        {
            return entryPath;
        }

        string rootPath = !string.IsNullOrWhiteSpace(modelRootPath)
            ? modelRootPath
            : Path.GetDirectoryName(entryPath)
                ?? throw new InvalidOperationException("Cannot resolve Chatterbox TTS smoke-test root path.");
        string onnxDirectory = string.Equals(Path.GetFileName(rootPath), "onnx", StringComparison.OrdinalIgnoreCase)
            ? rootPath
            : Path.Join(rootPath, "onnx");

        if (!string.IsNullOrWhiteSpace(variant) &&
            !variant.Equals("default", StringComparison.OrdinalIgnoreCase))
        {
            string variantDecoderPath = Path.Join(onnxDirectory, $"conditional_decoder_{variant}.onnx");
            if (File.Exists(variantDecoderPath))
            {
                return variantDecoderPath;
            }
        }

        string defaultDecoderPath = Path.Join(onnxDirectory, "conditional_decoder.onnx");
        if (File.Exists(defaultDecoderPath))
        {
            return defaultDecoderPath;
        }

        return Path.Join(onnxDirectory, "conditional_decoder.onnx");
    }

    private static bool IsChatterboxTtsModel(string modelId, string modelAlias) =>
        modelId.Contains("chatterbox", StringComparison.OrdinalIgnoreCase) ||
        modelAlias.Contains("chatterbox", StringComparison.OrdinalIgnoreCase);

    private static bool IsCosyVoiceTtsModel(string modelId, string modelAlias) =>
        modelId.Contains("cosyvoice", StringComparison.OrdinalIgnoreCase) ||
        modelAlias.Contains("cosyvoice", StringComparison.OrdinalIgnoreCase);

    private static (string WaveformName, string? LengthName) ResolveDiarizationInputNames(
        IReadOnlyDictionary<string, Type> inputElementTypes)
    {
        string waveformName;
        if (inputElementTypes.ContainsKey("waveform"))
        {
            waveformName = "waveform";
        }
        else if (inputElementTypes.ContainsKey("audio_signal"))
        {
            waveformName = "audio_signal";
        }
        else
        {
            string[] floatInputs = inputElementTypes
                .Where(static candidate => candidate.Value == typeof(float))
                .Select(static candidate => candidate.Key)
                .ToArray();
            waveformName = floatInputs.Length switch
            {
                1 => floatInputs[0],
                0 => throw new InvalidOperationException("Smoke test could not locate a float diarization waveform input."),
                _ => throw new InvalidOperationException($"Smoke test found {floatInputs.Length} float diarization inputs; expected exactly one waveform input.")
            };
        }

        string? lengthName;
        if (inputElementTypes.ContainsKey("length"))
        {
            lengthName = "length";
        }
        else if (inputElementTypes.ContainsKey("audio_signal_length"))
        {
            lengthName = "audio_signal_length";
        }
        else
        {
            lengthName = inputElementTypes
                .FirstOrDefault(static candidate =>
                    candidate.Value == typeof(long) &&
                    candidate.Key.Contains("length", StringComparison.OrdinalIgnoreCase))
                .Key;
            if (string.IsNullOrWhiteSpace(lengthName))
            {
                lengthName = null;
            }
        }

        return (waveformName, lengthName);
    }

    /// <summary>Resolves a one-second, 16 kHz probe waveform for rank-one or rank-two diarization inputs.</summary>
    /// <exception cref="InvalidOperationException">The model declares an unsupported waveform rank.</exception>
    private static int[] ResolveDiarizationWaveformShape(IReadOnlyList<int> modelDimensions) =>
        modelDimensions.Count switch
        {
            1 => [16000],
            2 => [1, 16000],
            _ => throw new InvalidOperationException("Smoke test diarization waveform input must be rank 1 or 2.")
        };

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

    private async Task SmokeTestTranslationAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        ThrowIfFatalTensorRtFamily(request.EngineFamily, request.ExecutionProvider, request.Variant, request.EntryPath);

        if (UsesOrtGenAiTranslationSmoke(request.EngineFamily))
        {
            await SmokeTestTextRefinementGenAiAsync(
                request.ModelRootPath,
                request.EntryPath,
                request.ModelId,
                request.Variant,
                request.ModelRevisionHash,
                request.ExecutionProvider,
                request.NonEvictingWarmup,
                cancellationToken).ConfigureAwait(false);
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
                .CreatePooledOpusAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = engineFamily, Provider = request.ExecutionProvider, CancellationToken = cancellationToken }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = encoderModelPath, Decoder = decoderModelPath })
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

    private static void EnsureSelectedProviderMatchesRequested(
        ExecutionProviderKind requestedProvider,
        string selectedProvider)
    {
        string requestedProviderLabel = FormatProviderLabel(requestedProvider);
        if (string.Equals(selectedProvider, requestedProviderLabel, StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        throw new InvalidOperationException(
            $"Smoke test requested provider '{requestedProviderLabel}' but session creation selected effective provider '{selectedProvider}'.");
    }

    private static string FormatProviderLabel(ExecutionProviderKind provider) =>
        provider switch
        {
            ExecutionProviderKind.Cpu => "cpu",
            ExecutionProviderKind.DirectMl => "dml",
            ExecutionProviderKind.TensorRTRtx => "tensorrt-rtx",
            ExecutionProviderKind.Cuda => "cuda",
            ExecutionProviderKind.TensorRt => "tensorrt",
            ExecutionProviderKind.Migraphx => "migraphx",
            ExecutionProviderKind.Dnnl => "dnnl",
            ExecutionProviderKind.Qnn => "qnn",
            ExecutionProviderKind.OpenVinoCatalog => "openvino-catalog",
            ExecutionProviderKind.VitisAi => "vitisai",
            _ => throw new ArgumentOutOfRangeException(nameof(provider), provider, "Unsupported execution provider kind.")
        };

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

    private static string ResolveNemotronDecoderJointPath(string encoderModelPath)
    {
        string candidatePath = Path.Join(Path.GetDirectoryName(encoderModelPath)!, "decoder_joint.onnx");
        if (File.Exists(candidatePath))
        {
            return Path.GetFullPath(candidatePath);
        }

        throw new FileNotFoundException("Nemotron ASR decoder_joint.onnx was not found next to encoder.onnx.", candidatePath);
    }

    private sealed class InputSet(IReadOnlyList<NamedOnnxValue> values) : IDisposable
    {
        public IReadOnlyList<NamedOnnxValue> Values { get; } = values;

        public void Dispose()
        {
            foreach (IDisposable value in Values.OfType<IDisposable>())
            {
                value.Dispose();
            }
        }
    }
}
