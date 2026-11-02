using Trackdub.Application.Pipeline;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Dubbing;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.Transcripts;
using Trackdub.Domain;
using Trackdub.Domain.Tts;

namespace Trackdub.Application.Dubbing;

public sealed partial class DubbingPipelineEngine
{
    /// <summary>What the pre-flight evaluates: the session, its options, and the requested stages.</summary>
    private sealed record PreFlightTarget(
        IDubbingSession Session,
        DubbingSessionOptions Options,
        string[] StageNames,
        TranscriptProjectState? State);

    /// <summary>Run bookkeeping carried into any pre-flight failure result.</summary>
    private sealed record PreFlightRun(
        Guid RunId,
        DateTimeOffset RunStart,
        List<StageOutcome> StageOutcomes,
        Dictionary<string, string> ExecutionSnapshot);

    private sealed record PreFlightProvisioning(
        IPipelineReadinessService ReadinessService,
        RuntimeModelSetupCoordinator Coordinator,
        IProgress<PipelineProgressEvent>? Progress,
        CancellationToken CancellationToken);

    private sealed record ReadinessPreFlightContext(
        PreFlightTarget Target,
        PreFlightRun Run,
        PreFlightProvisioning Provisioning)
    {
        public IDubbingSession Session => Target.Session;
        public DubbingSessionOptions Options => Target.Options;
        public string[] StageNames => Target.StageNames;
        public TranscriptProjectState? State => Target.State;
        public Dictionary<string, string> ExecutionSnapshot => Run.ExecutionSnapshot;
        public IPipelineReadinessService ReadinessService => Provisioning.ReadinessService;
        public RuntimeModelSetupCoordinator Coordinator => Provisioning.Coordinator;
        public IProgress<PipelineProgressEvent>? Progress => Provisioning.Progress;
        public CancellationToken CancellationToken => Provisioning.CancellationToken;

        /// <summary>
        /// Headless sessions auto-download-or-cancel; interactive hosts show dialogs.
        /// </summary>
        public RuntimeModelSetupCallbacks CreateSetupCallbacks() =>
            TryResolveService<IPipelineModelSetupInteraction>(Session)
                ?.CreateCallbacks(Progress, CancellationToken)
            ?? BuildHeadlessCallbacks(CancellationToken);

        /// <summary>
        /// The project state is passed so consent gates (voice cloning) are
        /// evaluated up front instead of failing mid-run.
        /// </summary>
        public async Task<PipelineReadinessReport> EvaluateReadinessAsync(
            List<RuntimeStage> stages,
            RuntimeModelSelections selections) =>
            await ReadinessService
                .EvaluateAsync(
                    stages,
                    selections,
                    State,
                    CancellationToken,
                    Options.SourceLanguageCode,
                    Options.TargetLanguageCode,
                    explicitEgressConsents: BuildExplicitCloudEgressConsents(Options.ExplicitCloudEgressConsents))
                .ConfigureAwait(false);

        public DubbingRunResult PreFlightFailed(IReadOnlyList<string> failures) =>
            BuildErrorResult(
                new RunScope(Run.RunId, Run.RunStart, Run.StageOutcomes, Run.ExecutionSnapshot),
                DubbingRunStatus.PreFlightFailed,
                failures);

        public DubbingRunResult ProvisioningCancelled() =>
            PreFlightFailed(["Model provisioning was cancelled during pre-flight."]);
    }

    private readonly record struct ProvisioningOutcome(
        DubbingRunResult? Error,
        RuntimeModelSelections Selections,
        PipelineReadinessReport Report,
        IReadOnlySet<string> DeclinedStages);

    /// <summary>
    /// Pre-flight using the consolidated readiness gate.
    /// Evaluates all stages upfront, auto-provisions downloadable models before the stage loop,
    /// and fails fast with an aggregated error if anything is still blocking.
    /// </summary>
    private static async Task<(DubbingRunResult? Error, IReadOnlySet<string> DeclinedStages)> RunPreFlightWithReadinessServiceAsync(
        ReadinessPreFlightContext context,
        RuntimeModelSelections selections)
    {
        List<RuntimeStage> enabledStages = await CollectStagesToEvaluateAsync(context).ConfigureAwait(false);
        if (enabledStages.Count == 0)
            return (null, EmptyDeclinedStages);

        PipelineReadinessReport report;
        using (BenchmarkPhaseCapture.Start("readiness-evaluate"))
        {
            report = await context.EvaluateReadinessAsync(enabledStages, selections).ConfigureAwait(false);
        }

        IReadOnlySet<string> declinedStages = EmptyDeclinedStages;
        if (report.Stages.Any(s => s.Status == ReadinessState.DownloadRequired))
        {
            ProvisioningOutcome provisioned = await ProvisionAndReevaluateAsync(
                context, enabledStages, selections, report).ConfigureAwait(false);
            if (provisioned.Error is not null)
                return (provisioned.Error, EmptyDeclinedStages);

            declinedStages = provisioned.DeclinedStages;
            if (enabledStages.Count == 0)
                return (null, declinedStages);

            (selections, report) = (provisioned.Selections, provisioned.Report);
        }

        // Collect all remaining blocking states into an aggregated failure list.
        List<string> failures = report.BlockingStages.Select(FormatPreFlightFailure).ToList();
        if (failures.Count > 0)
            return (context.PreFlightFailed(failures), declinedStages);

        DubbingRunResult? companionError = await EnsureCompanionCustomVoiceModelAsync(
            context, enabledStages, selections).ConfigureAwait(false);
        return (companionError, declinedStages);
    }

    /// <summary>
    /// Maps stage names to <see cref="RuntimeStage"/>, skipping stages with valid existing
    /// artifacts (resumable).
    /// </summary>
    private static async Task<List<RuntimeStage>> CollectStagesToEvaluateAsync(ReadinessPreFlightContext context)
    {
        var enabledStages = new List<RuntimeStage>();
        foreach (string stageName in context.StageNames)
        {
            if (MapStageNameToRuntimeStage(stageName) is { } stage &&
                await IsStageEnabledAsync(context, stage, stageName).ConfigureAwait(false))
            {
                enabledStages.Add(stage);
            }
        }

        return enabledStages;
    }

    /// <summary>
    /// A stage is evaluated unless separation was disabled or resumable artifacts from a
    /// prior run already satisfy it.
    /// </summary>
    private static async Task<bool> IsStageEnabledAsync(
        ReadinessPreFlightContext context,
        RuntimeStage stage,
        string stageName)
    {
        if (stage == RuntimeStage.Separation && !context.Options.EnableStemSeparation)
            return false;

        return context.Options.ForceRerun ||
            !await HasValidExistingArtifactsAsync(
                new ArtifactResumeContext(context.Session, context.Options, context.ExecutionSnapshot),
                stageName,
                context.CancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Provisions downloadable models upfront via the host's interaction surface, drops stages
    /// whose optional model was declined, then re-evaluates with refreshed selections in case
    /// settings changed. Leaves the report unevaluated when no stage remains.
    /// </summary>
    private static async Task<ProvisioningOutcome> ProvisionAndReevaluateAsync(
        ReadinessPreFlightContext context,
        List<RuntimeStage> enabledStages,
        RuntimeModelSelections selections,
        PipelineReadinessReport report)
    {
        RuntimeModelSetupCallbacks setupCallbacks = context.CreateSetupCallbacks();
        RuntimeModelSetupResult provisionResult;
        using (BenchmarkPhaseCapture.Start("model-provision"))
        {
            provisionResult = await context.Coordinator
                .EnsurePipelineModelsAvailableAsync(
                    context.Session.Workspace,
                    selections,
                    report,
                    setupCallbacks,
                    context.Options.SourceLanguageCode,
                    context.Options.TargetLanguageCode,
                    context.CancellationToken)
                .ConfigureAwait(false);
        }

        if (!provisionResult.IsReady)
            return new ProvisioningOutcome(context.ProvisioningCancelled(), selections, report, EmptyDeclinedStages);

        IReadOnlySet<string> declinedStages = RemoveDeclinedOptionalStages(
            context.StageNames, provisionResult, enabledStages);
        if (enabledStages.Count == 0)
            return new ProvisioningOutcome(null, selections, report, declinedStages);

        RuntimeModelSelections refreshedSelections = await CreateRuntimeSelectionsAsync(
                context.Session, context.Options, context.CancellationToken, context.State)
            .ConfigureAwait(false);
        DubbingExecutionSnapshot.MergeRuntimeModelSelectionsIntoSnapshot(
            context.ExecutionSnapshot,
            refreshedSelections,
            TryResolveService<IModelAliasResolver>(context.Session));
        context.ReadinessService.InvalidateCache(enabledStages);
        PipelineReadinessReport refreshedReport = await context
            .EvaluateReadinessAsync(enabledStages, refreshedSelections)
            .ConfigureAwait(false);
        return new ProvisioningOutcome(null, refreshedSelections, refreshedReport, declinedStages);
    }

    /// <summary>
    /// Stages whose optional model the user declined must not re-block on re-evaluation
    /// (the model is still absent). Stages with an in-place fallback still execute and
    /// degrade; stages without one are skipped.
    /// </summary>
    private static HashSet<string> RemoveDeclinedOptionalStages(
        string[] stageNames,
        RuntimeModelSetupResult provisionResult,
        List<RuntimeStage> enabledStages)
    {
        var declinedStages = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string stageName in stageNames)
        {
            if (MapStageNameToRuntimeStage(stageName) is not { } skippedStage ||
                !provisionResult.SkippedStages.Contains(skippedStage))
            {
                continue;
            }

            enabledStages.Remove(skippedStage);
            if (!StagesWithOptionalModelFallback.Contains(skippedStage))
            {
                declinedStages.Add(stageName);
            }
        }

        return declinedStages;
    }

    private static async Task<DubbingRunResult?> EnsureCompanionCustomVoiceModelAsync(
        ReadinessPreFlightContext context,
        List<RuntimeStage> enabledStages,
        RuntimeModelSelections selections)
    {
        if (!enabledStages.Contains(RuntimeStage.Tts) ||
            !UnattendedVoiceSelection.RequiresCompanionCustomVoiceModel(selections, context.Options, context.State))
        {
            return null;
        }

        RuntimeModelSetupCallbacks setupCallbacks = context.CreateSetupCallbacks();
        RuntimeModelSetupResult companion = await context.Coordinator
            .EnsureTtsModelAvailableAsync(
                context.Session.Workspace,
                selections with { TtsModelAlias = Qwen3TtsDefaults.ResolveCustomVoiceAlias(tier: null) },
                requiresVoiceClone: false,
                setupCallbacks,
                context.CancellationToken)
            .ConfigureAwait(false);
        return companion.IsReady ? null : context.ProvisioningCancelled();
    }

    /// <summary>
    /// Optional stages whose runtime degrades in place when the model is absent.
    /// A declined model for these stages is exempt from post-provisioning re-blocking,
    /// and the stage still executes (speech enhancement falls back to FFmpeg/AFX).
    /// </summary>
    private static readonly IReadOnlySet<RuntimeStage> StagesWithOptionalModelFallback =
        new HashSet<RuntimeStage> { RuntimeStage.SpeechEnhancement };

    private static readonly IReadOnlyDictionary<ReadinessState, string> FixedPreFlightFailureMessages =
        new Dictionary<ReadinessState, string>
        {
            [ReadinessState.CommercialBlocked] = "selected model is non-commercial only",
            [ReadinessState.IntegrityFailed] = "model checksum verification failed — re-download the model",
            [ReadinessState.ConsentRequired] = "voice-clone consent required",
        };

    private static string FormatPreFlightFailure(StageReadiness s)
    {
        if (FixedPreFlightFailureMessages.TryGetValue(s.Status, out string? message))
        {
            return $"{s.StageName}: {message}";
        }

        return s.Status switch
        {
            ReadinessState.CloudKeyMissing =>
                $"{s.StageName}: API key not configured for cloud provider{FormatModelAliasSuffix(s.ModelAlias)}",
            ReadinessState.ImportRequired =>
                $"{s.StageName}: model requires manual import{FormatModelIdSuffix(s.ModelId)}",
            ReadinessState.CloudEgressConsentRequired =>
                $"{s.StageName}: {s.Detail ?? "cloud egress consent required. Set TRACKDUB_CLOUD_EGRESS_CONSENT or use the app to consent interactively."}",
            _ => FormatUnclassifiedFailure(s),
        };
    }

    private static string FormatUnclassifiedFailure(StageReadiness s) =>
        s.Detail is not null
            ? $"{s.StageName}: {s.Detail}"
            : $"{s.StageName}: not ready ({s.Status})";

    private static string FormatModelAliasSuffix(string? modelAlias) =>
        modelAlias is null ? string.Empty : $" ({modelAlias})";

    private static string FormatModelIdSuffix(string? modelId) =>
        modelId is null ? string.Empty : $" — {modelId}";

    /// <summary>
    /// Headless provisioning callbacks: auto-download when possible, cancel otherwise.
    /// No UI dialogs, no file pickers. Used when the session does not register an
    /// <see cref="IPipelineModelSetupInteraction"/> (e.g. minimal test hosts).
    /// </summary>
    private static RuntimeModelSetupCallbacks BuildHeadlessCallbacks(CancellationToken cancellationToken) =>
        HeadlessRuntimeModelSetup.CreateCallbacks(cancellationToken);
}
