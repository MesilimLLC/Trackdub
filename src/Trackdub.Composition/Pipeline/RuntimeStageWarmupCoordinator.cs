using Trackdub.Application.Pipeline;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Licensing;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Onnx.Worker;
using Trackdub.Inference.Runtime.Planning;
using Trackdub.Inference.Runtime.TensorRtRtx;

namespace Trackdub.Composition.Pipeline;

/// <summary>
/// Warms exactly the model the planner would select for an imminent runtime stage by running
/// the real plan + provider smoke profile ahead of execution. The smoke path reuses pooled
/// ONNX sessions and shared GenAI models, so the next stage's first lease is warm. Since
/// planning no longer smokes (C4), this is the single proving create on a first run: its
/// failure is returned with its reason (the engine then skips/blocks the stage before
/// inference), and a pass records the persisted smoke verdict for the next launch. It
/// produces no stage-run records or artifacts.
/// </summary>
public sealed class RuntimeStageWarmupCoordinator(
    IRuntimePlanner runtimePlanner,
    IExecutionProviderSmokeTester executionProviderSmokeTester,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null,
    IInferenceWorkerClient? inferenceWorkerClient = null,
    Func<bool>? hostCudaProviderListed = null,
    bool allowsConcurrentWarmup = false,
    ISmokeVerdictStore? smokeVerdictStore = null,
    IHardwareProfileProvider? hardwareProfileProvider = null)
    : IStageWarmupCoordinator
{
    public bool AllowsConcurrentWarmup => allowsConcurrentWarmup;
    public async Task<StageWarmupResult> WarmAsync(
        StageWarmupRequest request,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        BenchmarkPhaseCapture.Increment("stageWarmupAttempt");

        try
        {
            using (BenchmarkPhaseCapture.Start("stage-warmup"))
            {
                RuntimeModelRequestOptions options =
                    RuntimeModelRequestFactory.CreateOptions(request.Selections);
                RuntimeModelRequest modelRequest = request.Stage switch
                {
                    RuntimeStage.Asr => RuntimeModelRequestFactory.CreateAsrRequest(
                        options, request.SourceLanguageCode),
                    RuntimeStage.Translation => RuntimeModelRequestFactory.CreateTranslationRequest(
                        options,
                        request.SourceLanguageCode ?? string.Empty,
                        request.TargetLanguageCode ?? string.Empty),
                    RuntimeStage.Tts => RuntimeModelRequestFactory.CreateTtsRequest(
                        options, request.RequiresVoiceClone),
                    RuntimeStage.Diarization => RuntimeModelRequestFactory.CreateDiarizationRequest(options),
                    RuntimeStage.OverlapRescue => RuntimeModelRequestFactory.CreateOverlapRescueRequest(options),
                    RuntimeStage.TextRefinement => RuntimeModelRequestFactory.CreateTextRefinementRequest(options),
                    RuntimeStage.LipSync => RuntimeModelRequestFactory.CreateLipSyncRequest(options),
                    RuntimeStage.LipSynthesis => RuntimeModelRequestFactory.CreateLipSynthesisRequest(options),
                    _ => RuntimeModelRequestFactory.CreateStageRequest(options, request.Stage),
                };

                // Warmup planning is file-only: it must never run the planner's
                // authoritative smoke. On a plan-cache miss (e.g. Automatic translation,
                // which readiness never pre-plans), that smoke would take the regular
                // evicting GenAI lease and evict the current stage's resident-idle model
                // before the non-evicting smoke below runs. The coordinator's own smoke
                // still warms the model, and stage execution re-verifies independently
                // (the skip flag is part of the plan-cache key, so this Ready can never
                // satisfy execution's verified request).
                StageRuntimePlanningRequest planningRequest =
                    await StageRuntimePlanningRequestFactory.ApplyPreferredModelTierAsync(
                        new StageRuntimePlanningRequest(
                            modelRequest.Stage,
                            PreferredModelAlias: modelRequest.PreferredModelAlias,
                            SourceLanguage: modelRequest.SourceLanguage,
                            TargetLanguage: modelRequest.TargetLanguage,
                            RequirePreferredModelAlias: modelRequest.RequirePreferredModelAlias,
                            PreferredExecutionProvider: modelRequest.PreferredExecutionProvider,
                            RequirePreferredExecutionProvider: modelRequest.RequirePreferredExecutionProvider,
                            PreferredModelVariantAlias: modelRequest.PreferredModelVariantAlias,
                            SkipProviderSmokeTest: true),
                        runtimePlanningPreferences,
                        cancellationToken).ConfigureAwait(false);

                StageRuntimePlan plan = await runtimePlanner
                    .PlanAsync(planningRequest, cancellationToken)
                    .ConfigureAwait(false);

                if (!plan.IsRunnable()
                    || plan.ExecutionProvider is not { } provider
                    || string.IsNullOrWhiteSpace(plan.ModelId)
                    || string.IsNullOrWhiteSpace(plan.ModelAlias)
                    || string.IsNullOrWhiteSpace(plan.Variant)
                    || string.IsNullOrWhiteSpace(plan.ModelRootPath)
                    || string.IsNullOrWhiteSpace(plan.ModelEntryPath))
                {
                    BenchmarkPhaseCapture.Increment("stageWarmupFailure");
                    return new StageWarmupResult(
                        Attempted: false,
                        Succeeded: false,
                        Detail: $"Stage '{request.Stage}' has no runnable plan to warm "
                            + $"(status {plan.Status}, detail {plan.Fallback?.Detail ?? "none"}).");
                }

                // Worker-backed CUDA TTS pays worker spawn + handshake + Kokoro session
                // construction on its first synthesis. Warm it here — this coordinator is
                // the background path (the pipeline only joins the warmup before the TTS
                // stage), so planning never blocks on it. A warm failure reports its
                // reason and skips the smoke: stage execution still lazy-constructs
                // inline, so no readiness is faked or lost.
                if (IsWorkerBackedCudaTts(plan, provider))
                {
                    try
                    {
                        WorkerWarmResult warm = await inferenceWorkerClient!
                            .WarmTtsAsync(plan, cancellationToken)
                            .ConfigureAwait(false);

                        if (!string.Equals(warm.Provider, "cuda", StringComparison.OrdinalIgnoreCase))
                        {
                            BenchmarkPhaseCapture.Increment("workerTtsWarmFailure");
                            BenchmarkPhaseCapture.Increment("stageWarmupFailure");
                            return new StageWarmupResult(
                                Attempted: true,
                                Succeeded: false,
                                Detail: $"Inference worker warmed on '{warm.Provider}' instead of cuda.");
                        }

                        BenchmarkPhaseCapture.Increment("workerTtsWarmSuccess");
                    }
                    catch (Exception ex) when (ex is not OperationCanceledException)
                    {
                        BenchmarkPhaseCapture.Increment("workerTtsWarmFailure");
                        BenchmarkPhaseCapture.Increment("stageWarmupFailure");
                        return new StageWarmupResult(
                            Attempted: true,
                            Succeeded: false,
                            Detail: $"Inference worker warm failed: {ex.Message} " +
                                "The TTS stage will retry construction inline.");
                    }
                }

                // The smoke profile is the actual per-run warmup: it exercises provider
                // registration plus the pooled session/GenAI load path, so a persisted
                // planner verdict must not skip it. It runs non-evicting for GenAI:
                // this warmup can start before the current stage leases its resident
                // GenAI model, and evicting that idle-but-imminent model to warm the
                // next stage would force a multi-GB reload and lose the overlap.
                ExecutionProviderSmokeTestResult smoke = await SmokeAsync(plan, provider, cancellationToken)
                    .ConfigureAwait(false);

                StageRuntimePlan warmedPlan = plan;
                ExecutionProviderKind warmedProvider = provider;
                if (!smoke.Passed && !smoke.NotAttempted && !IsGenAiRouted(plan))
                {
                    // The file-only plan above never ran the planner's authoritative
                    // smoke, so its provider fallback (e.g. DirectML -> CPU) never had
                    // a chance to run either. For non-GenAI routes, re-plan once with
                    // the authoritative smoke enabled and warm whatever it selects;
                    // GenAI routes keep the non-evicting behaviour this warmup exists
                    // for, and a would-evict skip (NotAttempted) never re-plans.
                    StageRuntimePlan replanned = await runtimePlanner
                        .PlanAsync(planningRequest with { SkipProviderSmokeTest = false }, cancellationToken)
                        .ConfigureAwait(false);
                    if (replanned.IsRunnable()
                        && replanned.ExecutionProvider is { } fallbackProvider
                        && (fallbackProvider != provider
                            || !string.Equals(replanned.ModelEntryPath, plan.ModelEntryPath, StringComparison.Ordinal)))
                    {
                        smoke = await SmokeAsync(replanned, fallbackProvider, cancellationToken)
                            .ConfigureAwait(false);
                        warmedPlan = replanned;
                        warmedProvider = fallbackProvider;
                    }
                }

                if (smoke.NotAttempted)
                {
                    BenchmarkPhaseCapture.Increment("stageWarmupSkipped");
                    return new StageWarmupResult(
                        Attempted: false,
                        Succeeded: false,
                        Detail: smoke.Detail
                            ?? "Warmup skipped: loading the model would evict a resident model.");
                }

                if (smoke.Passed)
                {
                    await RecordVerdictAsync(warmedPlan, warmedProvider, cancellationToken).ConfigureAwait(false);
                }

                BenchmarkPhaseCapture.Increment(
                    smoke.Passed ? "stageWarmupSuccess" : "stageWarmupFailure");
                return new StageWarmupResult(
                    Attempted: true,
                    Succeeded: smoke.Passed,
                    Detail: smoke.Passed ? null : smoke.Detail);
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            BenchmarkPhaseCapture.Increment("stageWarmupFailure");
            throw;
        }
        catch (Exception ex)
        {
            // Optimization must never fail the run.
            BenchmarkPhaseCapture.Increment("stageWarmupFailure");
            return new StageWarmupResult(Attempted: true, Succeeded: false, Detail: ex.Message);
        }
    }

    /// <summary>
    /// Persists the proof this warmup just produced so the next launch's planner can return
    /// Verified without any smoke. Planning no longer smokes or records verdicts (C4), so the
    /// coordinator — the single proving create, leasing the pooled session the stage reuses —
    /// is now the verdict recorder. Best-effort: a recording failure must not turn a passed
    /// warmup into a failure. Local optimized variants are excluded, mirroring the planner:
    /// a source-sha verdict cannot prove a different graph file.
    /// </summary>
    private async Task RecordVerdictAsync(
        StageRuntimePlan plan,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken)
    {
        ISmokeVerdictStore? verdictStore = smokeVerdictStore;
        IHardwareProfileProvider? profileProvider = hardwareProfileProvider;
        string? modelRevisionHash = plan.ModelRevisionHash;
        // Catalog placement can change with the saved policy, which the persisted key lacks.
        if (verdictStore is null || profileProvider is null || string.IsNullOrWhiteSpace(modelRevisionHash) ||
            !ShouldRecordVerifiedVerdict(plan, provider))
        {
            return;
        }

        try
        {
            HardwareProfile hardwareProfile = await profileProvider
                .GetCurrentAsync(cancellationToken).ConfigureAwait(false);
            verdictStore.RecordVerified(new SmokeVerdictKey(
                modelRevisionHash,
                provider,
                hardwareProfile.NvidiaGpuArchitecture,
                hardwareProfile.GpuDriverVersion,
                TensorRtRtxProviderConstants.BundledFingerprintVersion,
                hardwareProfile.OrtRuntimeVersion));
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Best-effort persistence only (see summary).
        }
    }

    private static bool ShouldRecordVerifiedVerdict(
        StageRuntimePlan plan,
        ExecutionProviderKind provider)
    {
        if (plan.IsLocalOptimizedVariant)
        {
            return false;
        }
        if (!OperatingSystem.IsWindows())
        {
            return true;
        }

        return !SmokeVerdictPolicy.IsWindowsMlPolicyDependent(provider);
    }
    private Task<ExecutionProviderSmokeTestResult> SmokeAsync(
        StageRuntimePlan plan,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken) =>
        executionProviderSmokeTester.SmokeTestAsync(
            new ExecutionProviderSmokeTestRequest(
                plan.Stage,
                plan.ModelId!,
                plan.ModelAlias!,
                plan.EngineFamily,
                plan.Variant!,
                provider,
                plan.ModelRootPath!,
                plan.ModelEntryPath!,
                plan.ModelRevisionHash)
            {
                NonEvictingWarmup = true,
                HasExplicitProviderPreference = plan.HasExplicitProviderPreference,
            },
            cancellationToken);

    /// <summary>
    /// Routes whose smoke goes through the GenAI model pool; re-planning them with the
    /// authoritative smoke would take the evicting lease this warmup exists to avoid.
    /// </summary>
    private static bool IsGenAiRouted(StageRuntimePlan plan) =>
        plan.Stage is RuntimeStage.TextRefinement
        || OnnxExecutionProviderSmokeTester.UsesOrtGenAiModelLoad(plan.EngineFamily)
        || OnnxExecutionProviderSmokeTester.UsesOrtGenAiTranslationSmoke(plan.EngineFamily);

    /// <summary>
    /// Mirrors <see cref="Trackdub.Inference.Onnx.Kokoro.KokoroTtsEngine"/>'s worker routing:
    /// a CUDA TTS plan for a worker-hosted family while this process cannot list CUDA itself.
    /// </summary>
    private bool IsWorkerBackedCudaTts(StageRuntimePlan plan, ExecutionProviderKind provider) =>
        inferenceWorkerClient is { IsInstalled: true } &&
        plan.Stage is RuntimeStage.Tts &&
        provider is ExecutionProviderKind.Cuda &&
        plan.EngineFamily is not null &&
        OnnxExecutionProviderDiscovery.WorkerCudaEngineFamilies.Contains(
            plan.EngineFamily, StringComparer.OrdinalIgnoreCase) &&
        !(hostCudaProviderListed?.Invoke() ?? CudaOrtProbe.IsCudaProviderListed());
}
