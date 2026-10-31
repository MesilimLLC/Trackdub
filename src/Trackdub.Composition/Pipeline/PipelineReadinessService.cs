using System.Collections.Concurrent;
using Trackdub.Application.Pipeline;
using Trackdub.Application.Transcripts;
using Trackdub.Application.Cloud;
using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Cloud;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.StageRuns;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Runtime.Planning;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.Pipeline;

/// <summary>
/// Evaluates pipeline readiness for all enabled stages against a set of runtime
/// model selections. Pure read — never opens dialogs or downloads models.
/// Lives in Composition because it references IRuntimePlanner (Inference layer).
/// </summary>
public sealed class PipelineReadinessService(
    IRuntimePlanner runtimePlanner,
    ICloudApiKeyProvider cloudApiKeyProvider,
    IConsentService consentService,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null,
    int? maxConcurrentStageEvaluations = null,
    TimeSpan? stageEvaluationTimeout = null,
    ICloudEgressConsentService? cloudEgressConsentService = null,
    BundledModelManifestRegistry? modelRegistry = null)
    : IPipelineReadinessService
{
    private readonly IRuntimePlanner _runtimePlanner =
        runtimePlanner ?? throw new ArgumentNullException(nameof(runtimePlanner));
    private readonly ICloudApiKeyProvider _cloudApiKeyProvider =
        cloudApiKeyProvider ?? throw new ArgumentNullException(nameof(cloudApiKeyProvider));
    private readonly IConsentService _consentService =
        consentService ?? throw new ArgumentNullException(nameof(consentService));
    private readonly IRuntimePlanningPreferences? _runtimePlanningPreferences = runtimePlanningPreferences;
    private readonly ICloudEgressConsentService? _cloudEgressConsentService = cloudEgressConsentService;

    // Readiness smoke tests cold-load full ONNX graphs. Cap parallel stage checks so a
    // multi-stage sweep overlaps load work without stampeding GPU/RAM admission.
    private const int DefaultMaxConcurrentStageEvaluations = 3;

    // Hard per-stage ceiling (audit: hard timeouts with cleanup). Linked to the caller
    // token so user cancel still wins; a timeout is reported as RuntimeMissing rather
    // than aborting the whole sweep. Cancellation is cooperative: native ONNX session
    // construction already in flight may ignore the token and keep this throttle slot
    // occupied until it returns. Large multi-GB graphs (qwen3-asr-1.7b encoder) can
    // exceed 180s on cold load + smoke, so the default is 10 minutes.
    private static readonly TimeSpan DefaultStageEvaluationTimeout = TimeSpan.FromSeconds(600);

    private readonly int _maxConcurrentStageEvaluations =
        maxConcurrentStageEvaluations is null
            ? DefaultMaxConcurrentStageEvaluations
            : maxConcurrentStageEvaluations.Value > 0
                ? maxConcurrentStageEvaluations.Value
                : throw new ArgumentOutOfRangeException(nameof(maxConcurrentStageEvaluations));
    private readonly TimeSpan _stageEvaluationTimeout =
        stageEvaluationTimeout is null
            ? DefaultStageEvaluationTimeout
            : stageEvaluationTimeout.Value > TimeSpan.Zero || stageEvaluationTimeout.Value == Timeout.InfiniteTimeSpan
                ? stageEvaluationTimeout.Value
                : throw new ArgumentOutOfRangeException(nameof(stageEvaluationTimeout));

    // Cache key covers every input that reaches the runtime planner: stage, model alias,
    // language context, validation mode, per-stage execution-provider override, the
    // require-preferred flag (dev build / require-providers settings), model variant
    // override, and the preferred model tier. Anything omitted would let a stale plan
    // computed under different inputs masquerade as current readiness.
    private readonly record struct ReadinessCacheKey(
        RuntimeStage Stage,
        string? ModelAlias,
        string? SourceLanguage,
        string? TargetLanguage,
        bool ValidateRuntime,
        ExecutionProviderKind? PreferredProvider,
        bool RequirePreferredProvider,
        string? PreferredVariantAlias,
        string? PreferredModelTier,
        RuntimeModelSelectionIntent SelectionIntent);

    private readonly ConcurrentDictionary<ReadinessCacheKey, StageReadiness> _cache = new();

    public async Task<PipelineReadinessReport> EvaluateAsync(
        IReadOnlyList<RuntimeStage> enabledStages,
        RuntimeModelSelections selections,
        TranscriptProjectState? state,
        CancellationToken cancellationToken = default,
        string? sourceLanguageCode = null,
        string? targetLanguageCode = null,
        bool validateRuntime = true,
        IReadOnlyDictionary<string, bool>? explicitEgressConsents = null)
    {
        ArgumentNullException.ThrowIfNull(enabledStages);
        ArgumentNullException.ThrowIfNull(selections);

        string? preferredModelTier = _runtimePlanningPreferences is null
            ? null
            : await _runtimePlanningPreferences
                .GetPreferredModelTierAsync(cancellationToken)
                .ConfigureAwait(false);

        var stageReadinesses = new StageReadiness[enabledStages.Count];
        using var throttle = new SemaphoreSlim(_maxConcurrentStageEvaluations, _maxConcurrentStageEvaluations);
        var stageTasks = new Task[enabledStages.Count];

        for (int index = 0; index < enabledStages.Count; index++)
        {
            int slot = index;
            RuntimeStage stage = enabledStages[index];
            stageTasks[slot] = Task.Run(
                () => EvaluateStageSlotAsync(
                    slot,
                    stage,
                    selections,
                    state,
                    preferredModelTier,
                    sourceLanguageCode,
                    targetLanguageCode,
                    validateRuntime,
                    explicitEgressConsents,
                    stageReadinesses,
                    throttle,
                    cancellationToken),
                cancellationToken);
        }

        await Task.WhenAll(stageTasks).ConfigureAwait(false);
        return new PipelineReadinessReport(stageReadinesses);
    }

    private async Task EvaluateStageSlotAsync(
        int slot,
        RuntimeStage stage,
        RuntimeModelSelections selections,
        TranscriptProjectState? state,
        string? preferredModelTier,
        string? sourceLanguageCode,
        string? targetLanguageCode,
        bool validateRuntime,
        IReadOnlyDictionary<string, bool>? explicitEgressConsents,
        StageReadiness[] stageReadinesses,
        SemaphoreSlim throttle,
        CancellationToken cancellationToken)
    {
        await throttle.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeoutCts.CancelAfter(_stageEvaluationTimeout);
            CancellationToken stageToken = timeoutCts.Token;

            try
            {
                stageReadinesses[slot] = await EvaluateStageAsync(
                        stage,
                        selections,
                        state,
                        preferredModelTier,
                        sourceLanguageCode,
                        targetLanguageCode,
                        validateRuntime,
                        explicitEgressConsents,
                        stageToken)
                    .ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested && stageToken.IsCancellationRequested)
            {
                // Per-stage timeout only: keep the sweep alive and surface a blocking state.
                stageReadinesses[slot] = new StageReadiness(
                    StageName: StageNameFor(stage),
                    Status: ReadinessState.RuntimeMissing,
                    Detail: $"Readiness check timed out after {_stageEvaluationTimeout.TotalSeconds:0}s; verify runtime installation",
                    ModelId: null,
                    ModelAlias: GetModelAlias(stage, selections),
                    ResolveAction: "install-runtime");
            }
        }
        finally
        {
            throttle.Release();
        }
    }

    private async Task<StageReadiness> EvaluateStageAsync(
        RuntimeStage stage,
        RuntimeModelSelections selections,
        TranscriptProjectState? state,
        string? preferredModelTier,
        string? sourceLanguageCode,
        string? targetLanguageCode,
        bool validateRuntime,
        IReadOnlyDictionary<string, bool>? explicitEgressConsents,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        string? modelAlias = GetModelAlias(stage, selections);
        string? planningSourceLanguageCode = stage == RuntimeStage.Translation
            ? TranscriptWorkflowUtilities.NormalizeTranscriptLanguageCode(sourceLanguageCode)
            : sourceLanguageCode;
        string? planningTargetLanguageCode = stage == RuntimeStage.Translation
            ? TranscriptWorkflowUtilities.NormalizeTranslationTargetLanguageCodeOrNull(targetLanguageCode)
            : targetLanguageCode;

        StageReadiness? skipped = GetDisabledOptionalStage(stage, selections);
        if (skipped is not null) return skipped;

        // Automatic translation must select its pair after the transcript establishes
        // the source. Generic preflight planning could otherwise download a large
        // multilingual fallback before the pair-specific preparation gate runs.
        if (stage == RuntimeStage.Translation &&
            selections.GetSelectionIntent(stage) == RuntimeModelSelectionIntent.Automatic)
        {
            return new StageReadiness(StageNameFor(stage), ReadinessState.PendingTranslationResolution,
                "Translation model selection and provisioning will be checked after source language resolution.",
                ModelId: null, ModelAlias: null, ResolveAction: null);
        }

        // The planning request carries every planner input; its fields double as the
        // rest of the cache key so plans computed under different provider, variant,
        // or tier inputs can never collide.
        StageRuntimePlanningRequest planningRequest = BuildPlanningRequest(
            stage,
            modelAlias,
            selections,
            planningSourceLanguageCode,
            planningTargetLanguageCode,
            skipProviderSmokeTest: !validateRuntime);
        var cacheKey = new ReadinessCacheKey(
            stage,
            modelAlias,
            planningSourceLanguageCode,
            planningTargetLanguageCode,
            validateRuntime,
            planningRequest.PreferredExecutionProvider,
            planningRequest.RequirePreferredExecutionProvider,
            planningRequest.PreferredModelVariantAlias,
            preferredModelTier, selections.GetSelectionIntent(stage));

        if (IsCloudAlias(stage, modelAlias))
        {
            StageReadiness cloudReadiness = await EvaluateCloudStageAsync(
                stage, modelAlias!, explicitEgressConsents, cancellationToken).ConfigureAwait(false);
            return ApplyVoiceCloneConsent(stage, state, cloudReadiness);
        }

        if (!_cache.TryGetValue(cacheKey, out StageReadiness? readiness))
        {
            readiness = await EvaluateLocalStageAsync(
                    planningRequest,
                    stage,
                    preferredModelTier,
                    selections.GetSelectionIntent(stage),
                    cancellationToken).ConfigureAwait(false);
            _cache[cacheKey] = readiness;
        }

        if (stage == RuntimeStage.Tts && !string.IsNullOrWhiteSpace(targetLanguageCode) && modelRegistry is not null)
        {
            string target = TranscriptWorkflowUtilities.NormalizeTranslationTargetLanguageCode(targetLanguageCode);
            var model = modelRegistry.Entries.FirstOrDefault(entry =>
                (!string.IsNullOrEmpty(readiness.ModelId) && entry.ModelId.Equals(readiness.ModelId, StringComparison.OrdinalIgnoreCase)) ||
                ((readiness.ModelAlias ?? modelAlias) is { Length: > 0 } alias && entry.Aliases.Contains(alias, StringComparer.OrdinalIgnoreCase)));
            if (model?.LanguageCoverage.TargetLanguages is { Count: > 0 } languages &&
                !languages.Any(language => language.Equals(target, StringComparison.OrdinalIgnoreCase) || language is "*" or "multi" or "any"))
            {
                return new StageReadiness(StageNames.Tts, ReadinessState.UnsupportedLanguage,
                    $"The selected speech model does not support '{target}'. Translation is available; choose a compatible speech model to dub this language.",
                    model?.ModelId ?? readiness.ModelId, readiness.ModelAlias, null);
            }
        }

        return ApplyVoiceCloneConsent(stage, state, readiness);
    }

    private static StageReadiness? GetDisabledOptionalStage(RuntimeStage stage, RuntimeModelSelections selections)
    {
        if (stage == RuntimeStage.Separation && IsSeparationSkipped(selections))
        {
            return new StageReadiness(StageNameFor(stage), ReadinessState.SkippableOptional,
                "Separation is optional and currently disabled", null, null, null);
        }

        if (stage == RuntimeStage.TextRefinement && !selections.EnableAsrTextRefinement)
        {
            return new StageReadiness(StageNames.TextRefinementAsr, ReadinessState.SkippableOptional,
                "ASR text polish is disabled.", null, null, null);
        }

        return null;
    }

    // Applied per call, never cached, so consent changes take effect immediately.
    private StageReadiness ApplyVoiceCloneConsent(RuntimeStage stage, TranscriptProjectState? state, StageReadiness readiness)
    {
        if (stage == RuntimeStage.Tts
            && readiness.Status is ReadinessState.Ready or ReadinessState.Unverified
            && !_consentService.IsVoiceCloningConsentGranted
            && HasVoiceCloneRequest(state))
        {
            return readiness with
            {
                Status = ReadinessState.ConsentRequired,
                Detail = "Voice cloning requires session consent",
                ResolveAction = "grant-consent",
            };
        }

        return readiness;
    }

    public void InvalidateCache(IReadOnlyList<RuntimeStage>? stages = null)
    {
        if (stages is null)
        {
            _cache.Clear();
            return;
        }

        foreach (RuntimeStage stage in stages)
        {
            foreach (var key in _cache.Keys.Where(k => k.Stage == stage).ToList())
            {
                _cache.TryRemove(key, out _);
            }
        }
    }

    // ── Cloud stage evaluation ─────────────────────────────────────────────────

    private async Task<StageReadiness> EvaluateCloudStageAsync(
        RuntimeStage stage,
        string modelAlias,
        IReadOnlyDictionary<string, bool>? explicitEgressConsents,
        CancellationToken cancellationToken)
    {
        string stageName = StageNameFor(stage);
        string providerKey = CloudProviderKey(modelAlias);

        // A missing consent service means consent gating is not configured (e.g. a host
        // that constructs the readiness service without wiring egress consent). Treat it
        // as "no gating required" so cloud stages are not permanently blocked, matching
        // the engines' own null-service handling.
        if (_cloudEgressConsentService is not null)
        {
            IReadOnlyList<string> requiredConsentKeys = _cloudEgressConsentService
                .GetRequiredConsentKeys(stageName, modelAlias);
            string? missingConsentKey = await ResolveMissingConsentKeyAsync(
                requiredConsentKeys, explicitEgressConsents, cancellationToken).ConfigureAwait(false);
            if (missingConsentKey is not null)
            {
                return new StageReadiness(
                    StageName: stageName,
                    Status: ReadinessState.CloudEgressConsentRequired,
                    Detail: $"Cloud egress to {providerKey} ({missingConsentKey}) requires user consent",
                    ModelId: null,
                    ModelAlias: modelAlias,
                    ResolveAction: "grant-egress-consent");
            }
        }

        string? apiKey = await _cloudApiKeyProvider
            .GetApiKeyAsync(providerKey, cancellationToken)
            .ConfigureAwait(false);

        if (string.IsNullOrWhiteSpace(apiKey))
        {
            return new StageReadiness(
                StageName: stageName,
                Status: ReadinessState.CloudKeyMissing,
                Detail: $"{providerKey} API key is not configured",
                ModelId: null,
                ModelAlias: modelAlias,
                ResolveAction: "set-api-key");
        }

        return new StageReadiness(
            StageName: stageName,
            Status: ReadinessState.Ready,
            Detail: null,
            ModelId: null,
            ModelAlias: modelAlias,
            ResolveAction: null);
    }

    private async Task<string?> ResolveMissingConsentKeyAsync(
        IReadOnlyList<string> requiredConsentKeys,
        IReadOnlyDictionary<string, bool>? explicitEgressConsents,
        CancellationToken cancellationToken)
    {
        foreach (string key in requiredConsentKeys)
        {
            if (explicitEgressConsents is not null &&
                explicitEgressConsents.TryGetValue(key.Trim().ToLowerInvariant(), out bool explicitlyConsented) &&
                explicitlyConsented)
            {
                continue;
            }

            if (_cloudEgressConsentService is not null &&
                await _cloudEgressConsentService.HasConsentAsync(key, cancellationToken).ConfigureAwait(false))
            {
                continue;
            }

            return key;
        }

        return null;
    }

    // ── Local stage evaluation ─────────────────────────────────────────────────

    private async Task<StageReadiness> EvaluateLocalStageAsync(
        StageRuntimePlanningRequest request,
        RuntimeStage stage,
        string? preferredModelTier,
        RuntimeModelSelectionIntent selectionIntent,
        CancellationToken cancellationToken)
    {
        if (!string.IsNullOrWhiteSpace(preferredModelTier))
        {
            request = request with { PreferredModelTier = preferredModelTier };
        }

        StageRuntimePlan plan = await _runtimePlanner
            .PlanAsync(request, cancellationToken)
            .ConfigureAwait(false);

        return MapPlanToReadiness(stage, plan, selectionIntent);
    }

    private static StageReadiness MapPlanToReadiness(
        RuntimeStage stage,
        StageRuntimePlan plan,
        RuntimeModelSelectionIntent selectionIntent)
    {
        string stageName = StageNameFor(stage);

        if (plan.Status is StageRuntimePlanStatus.Ready or StageRuntimePlanStatus.Verified)
        {
            // Files/metadata passed but this plan was never smoke-proven: either the caller
            // skipped the provider smoke test, or planning returned the honest file-only
            // Ready for a non-CPU provider with no persisted verdict (C4 — proving is
            // deferred to warmup/stage execution). Never promote that to Ready.
            // CPU plans are never smoke-tested, so Ready is already their full check.
            bool smokeNotProven = plan.Status == StageRuntimePlanStatus.Ready
                && plan.ExecutionProvider is not (null or ExecutionProviderKind.Cpu);
            if (smokeNotProven)
            {
                return new StageReadiness(
                    stageName,
                    ReadinessState.Unverified,
                    Detail: $"Model '{plan.ModelId}' is present; runtime path not verified (provider smoke has not proven this plan)",
                    plan.ModelId,
                    plan.ModelAlias,
                    null);
            }

            return new StageReadiness(stageName, ReadinessState.Ready, null, plan.ModelId, plan.ModelAlias, null);
        }

        if (plan.Status == StageRuntimePlanStatus.DownloadRequired)
        {
            return new StageReadiness(
                stageName,
                ReadinessState.DownloadRequired,
                Detail: $"Model '{plan.ModelId}' needs to be downloaded",
                ModelId: plan.ModelId,
                ModelAlias: plan.ModelAlias,
                ResolveAction: plan.Fallback?.Code == RuntimePlanFallbackCode.ModelIntegrityMismatch
                    || selectionIntent == RuntimeModelSelectionIntent.Explicit ? "download" : "bundle-needed");
        }

        // Blocked — inspect fallback code
        return plan.Fallback?.Code switch
        {
            RuntimePlanFallbackCode.CommercialSafeExcluded =>
                new StageReadiness(stageName, ReadinessState.CommercialBlocked,
                    "Model is non-commercial only; switch to a commercial-safe model",
                    plan.ModelId, plan.ModelAlias, null),

            RuntimePlanFallbackCode.ModelIntegrityMismatch =>
                new StageReadiness(stageName, ReadinessState.IntegrityFailed,
                    "Model checksum verification failed; re-download the model",
                    plan.ModelId, plan.ModelAlias, "download"),

            RuntimePlanFallbackCode.ProviderUnavailable =>
                new StageReadiness(stageName, ReadinessState.ProviderMissing,
                    plan.Fallback?.Detail ?? "No compatible execution provider found",
                    plan.ModelId, plan.ModelAlias, null),

            RuntimePlanFallbackCode.ProviderSmokeTestFailed =>
                new StageReadiness(stageName, ReadinessState.RuntimeMissing,
                    plan.Fallback?.Detail ?? "Execution provider smoke test failed; verify runtime installation",
                    plan.ModelId, plan.ModelAlias, "install-runtime"),

            _ =>
                new StageReadiness(stageName, ReadinessState.ImportRequired,
                    Detail: plan.Fallback is { } f
                        ? $"{f.Code}{(f.Detail is not null ? $": {f.Detail}" : string.Empty)}"
                        : "No compatible model found; import or download a model file",
                    plan.ModelId, plan.ModelAlias, "import"),
        };
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private static string? GetModelAlias(RuntimeStage stage, RuntimeModelSelections selections) =>
        stage switch
        {
            RuntimeStage.Asr => selections.AsrModelAlias,
            RuntimeStage.Translation => selections.TranslationModelAlias,
            RuntimeStage.Tts => selections.TtsModelAlias,
            RuntimeStage.Separation => selections.SeparationModelAlias,
            RuntimeStage.OverlapRescue => selections.OverlapRescueModelAlias,
            RuntimeStage.Diarization => selections.DiarizationModelAlias,
            RuntimeStage.TextRefinement => selections.TextRefinementModelAlias,
            RuntimeStage.LipSync => selections.LipSyncModelAlias,
            RuntimeStage.LipSynthesis => selections.LipSynthesisModelAlias,
            _ => null,
        };

    private static bool IsCloudAlias(RuntimeStage stage, string? alias) =>
        stage switch
        {
            RuntimeStage.Asr => AsrModelOverrideSettings.IsCloudAlias(alias),
            RuntimeStage.Translation => IsCloudTranslationAlias(alias),
            RuntimeStage.Tts => TtsModelOverrideSettings.IsCloudAlias(alias),
            RuntimeStage.TextRefinement => IsGeminiRefinementAlias(alias),
            _ => false,
        };

    private static bool IsGeminiRefinementAlias(string? alias) =>
        !string.IsNullOrWhiteSpace(alias) &&
        alias.StartsWith("gemini", StringComparison.OrdinalIgnoreCase);

    private static bool IsCloudTranslationAlias(string? alias) =>
        TranslationModelOverrideSettings.IsDeepLModelAlias(alias)
        || TranslationModelOverrideSettings.IsOpenAiGptAlias(alias)
        || TranslationModelOverrideSettings.IsGeminiTranslationAlias(alias);

    private static string CloudProviderKey(string? alias)
    {
        if (AsrModelOverrideSettings.IsOpenAiWhisperAlias(alias)) return "openai";
        if (AsrModelOverrideSettings.IsGeminiAsrAlias(alias)) return "gemini";
        if (TranslationModelOverrideSettings.IsDeepLModelAlias(alias)) return "deepl";
        if (TranslationModelOverrideSettings.IsOpenAiGptAlias(alias)) return "openai";
        if (TranslationModelOverrideSettings.IsGeminiTranslationAlias(alias)) return "gemini";
        if (TtsModelOverrideSettings.IsElevenLabsAlias(alias)) return "elevenlabs";
        if (TtsModelOverrideSettings.IsOpenAiTtsAlias(alias)) return "openai";
        if (TtsModelOverrideSettings.IsGoogleTtsAlias(alias)) return "google";
        if (TtsModelOverrideSettings.IsGeminiTtsAlias(alias)) return "gemini";
        if (IsGeminiRefinementAlias(alias)) return "gemini";
        return "unknown";
    }

    private static bool IsSeparationSkipped(RuntimeModelSelections selections) =>
        // Treat null/empty separation alias with Auto override as user-opted-out.
        // The explicit skip is signaled by SeparationModelOverride.Auto with no alias.
        selections.SeparationModelOverride == SeparationModelOverride.Auto
        && string.IsNullOrWhiteSpace(selections.SeparationModelAlias);

    private static StageRuntimePlanningRequest BuildPlanningRequest(
        RuntimeStage stage,
        string? modelAlias,
        RuntimeModelSelections selections,
        string? sourceLanguageCode = null,
        string? targetLanguageCode = null,
        bool skipProviderSmokeTest = false)
    {
        RuntimeModelRequestOptions options = RuntimeModelRequestFactory.CreateOptions(selections);
        ExecutionProviderKind? preferredProvider =
            RuntimeModelRequestFactory.ResolvePreferredExecutionProvider(options, stage);

        return new(
            Stage: stage,
            PreferredModelAlias: modelAlias,
            PreferredExecutionProvider: preferredProvider,
            RequirePreferredExecutionProvider: RuntimeModelRequestFactory.IsPreferredExecutionProviderRequired(options, stage),
            PreferredModelVariantAlias: RuntimeModelRequestFactory.ResolvePreferredModelVariantAlias(
                options,
                stage,
                modelAlias),
            SourceLanguage: stage switch
            {
                RuntimeStage.Asr => sourceLanguageCode,
                RuntimeStage.Translation => sourceLanguageCode,
                _ => null,
            },
            TargetLanguage: stage == RuntimeStage.Translation ? targetLanguageCode : null,
            SkipProviderSmokeTest: skipProviderSmokeTest);
    }

    private static bool HasVoiceCloneRequest(TranscriptProjectState? state) =>
        // Voice-clone consent only required when at least one voice assignment has a reference clip.
        state?.VoiceAssignments.Any(v => v.ReferenceClipArtifactId is not null) == true;

    private static string StageNameFor(RuntimeStage stage) =>
        stage switch
        {
            RuntimeStage.Vad => StageNames.Vad,
            RuntimeStage.Asr => StageNames.Asr,
            RuntimeStage.Diarization => StageNames.Diarization,
            RuntimeStage.Translation => StageNames.Translation,
            RuntimeStage.Tts => StageNames.Tts,
            RuntimeStage.Separation => StageNames.Separation,
            RuntimeStage.SpeechEnhancement => StageNames.SpeechEnhancement,
            RuntimeStage.OverlapRescue => StageNames.OverlapRescue,
            RuntimeStage.TextRefinement => StageNames.TextRefinementAsr,
            RuntimeStage.LipSync => StageNames.LipSync,
            RuntimeStage.LipSynthesis => StageNames.LipSynthesis,
            _ => stage.ToString().ToLowerInvariant(),
        };
}
