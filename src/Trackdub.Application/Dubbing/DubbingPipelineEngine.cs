using System.Globalization;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.Extensions.DependencyInjection;
using Trackdub.Application.LipSync;
using Trackdub.Application.LipSynthesis;
using Trackdub.Application.Pipeline;
using Trackdub.Application.Transcripts;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Contracts;
using Trackdub.Contracts.Dubbing;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Contracts.Licensing;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.Transcripts;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Pipeline;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Tts;

namespace Trackdub.Application.Dubbing;

/// <summary>
/// Primary entry point for SDK consumers to execute the dubbing pipeline.
/// Orchestrates validation, pre-flight checks, stage execution, and result aggregation.
/// </summary>
/// <param name="sessionFactory">Factory used to create per-run sessions.</param>
/// <param name="transientFaultBus">
/// Optional shared bus for transient-fault telemetry. When null the engine owns
/// its own bus internally so the <see cref="ITransientFaultReporting"/> surface
/// is always observable. Callers that need shared visibility across engine +
/// diagnostics exporter should DI-register the bus as a singleton.
/// </param>
public sealed partial class DubbingPipelineEngine(
    IDubbingSessionFactory sessionFactory,
    PipelineTransientFaultBus? transientFaultBus = null) : IDubbingPipelineEngine, ITransientFaultReporting
{
    private readonly IDubbingSessionFactory _sessionFactory = sessionFactory ?? throw new ArgumentNullException(nameof(sessionFactory));
    private readonly PipelineTransientFaultBus _transientFaultBus = transientFaultBus ?? new PipelineTransientFaultBus();

    /// <summary>
    /// The bus instance this engine publishes to. Exposed as <c>internal</c> so the
    /// test/Composition surface can pin identity against the Composition-registered
    /// singleton (spec §4.4 C8 follow-up) without resorting to reflection.
    /// </summary>
    internal PipelineTransientFaultBus TransientFaultBus => _transientFaultBus;

    internal static string? NormalizeAsrSourceLanguageCode(string? sourceLanguageCode) =>
        TranscriptWorkflowUtilities.NormalizeTranscriptLanguageCode(sourceLanguageCode);

    /// <summary>
    /// Executes the dubbing pipeline according to the provided options.
    /// </summary>
    /// <param name="options">Immutable configuration for this pipeline run.</param>
    /// <param name="progress">Optional progress reporter for stage-level events.</param>
    /// <param name="cancellationToken">Token to request clean abort between stages.</param>
    /// <returns>An immutable <see cref="DubbingRunResult"/> describing the outcome of every stage.</returns>
    public async Task<DubbingRunResult> ExecuteAsync(
        DubbingSessionOptions options,
        IProgress<PipelineProgressEvent>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);

        // Cold-start instrumentation: one capture per run, active for everything the run
        // spawns, so every inference choke point can report its first completed inference
        // here regardless of which thread it completes on.
        var coldStart = new ColdStartCapture();
        using (ColdStartCapture.Activate(coldStart))
        {
            DubbingRunResult result = await ExecuteCoreAsync(options, progress, cancellationToken, coldStart).ConfigureAwait(false);
            PipelineColdStartTiming timing = coldStart.BuildTiming();
            coldStart.Logger?.LogInformation(ColdStartCapture.BuildSummaryMessage(timing));
            return result with { ColdStartTiming = timing };
        }
    }

    private async Task<DubbingRunResult> ExecuteCoreAsync(
        DubbingSessionOptions options,
        IProgress<PipelineProgressEvent>? progress,
        CancellationToken cancellationToken,
        ColdStartCapture coldStart)
    {
        Guid runId = Guid.NewGuid();
        // All preflight/nested/stage events of this run carry the same RunId and a
        // strictly increasing SequenceNumber; downstream callers see only the wrapper.
        var scopedProgress = new RunScopedProgressReporter(runId, progress);
        DateTimeOffset runStart = DateTimeOffset.UtcNow;
        Guid projectId = Guid.Empty;
        var stageOutcomes = new List<StageOutcome>();
        string[] stagesToRun = ResolveStageOrder(options.StageFilter);

        var scope = new RunScope(runId, runStart, stageOutcomes);
        DubbingRunResult? selectionError = ValidateStageSelection(options, stagesToRun, scope);
        if (selectionError is not null)
        {
            return selectionError;
        }

        bool requiresSourceMedia = stagesToRun
            .Any(stage => DubbingPipelineStages.RequiresSourceMedia(stage));
        (DubbingSessionOptions effectiveOptions, DubbingRunResult? mediaError) =
            await ResolveEffectiveSourceMediaAsync(
                options, requiresSourceMedia, scope, cancellationToken)
                .ConfigureAwait(false);
        if (mediaError is not null)
        {
            return mediaError;
        }

        options = UnattendedVoiceSelection.ApplyVoiceCloningDefaults(effectiveOptions);

        // --- Validation: create output directory if missing ---
        string projectOutputDirectory = EnsureProjectDirectory(options);

        // --- Capture immutable ExecutionSnapshot ---
        Dictionary<string, string> executionSnapshot = DubbingExecutionSnapshot.CaptureExecutionSnapshot(options);

        // --- Create session ---
        IDubbingSession? session;
        DubbingRunResult? sessionError;
        using (BenchmarkPhaseCapture.Start("session-create"))
            (session, sessionError) = CreateSessionOrError(
                projectOutputDirectory, options, scope with { ExecutionSnapshot = executionSnapshot });

        if (sessionError is not null || session is null)
        {
            return sessionError!;
        }

        await using (session.ConfigureAwait(false))
        {
            // Explicit cloud egress consents are applied as a run-scoped overlay via the
            // session settings seeded in CreateSessionOrError. They are never written to the
            // shared consent store, so a later run that omits the option is still gated by
            // what was actually persisted.

            (TranscriptProjectState? initialProjectState, projectId) = await PrepareProjectContextAsync(
                new ProjectContextRequest
                {
                    Session = session,
                    Options = options,
                    RequiresSourceMedia = requiresSourceMedia,
                    ExecutionSnapshot = executionSnapshot,
                    ColdStart = coldStart,
                },
                cancellationToken).ConfigureAwait(false);

            // --- Pre-flight checks and runtime selections ---
            (DubbingRunResult? preFlightResult, IReadOnlySet<string> declinedOptionalStages,
                RuntimeModelSelections? runtimeSelections) = await PrepareStageInputsAsync(
                    new PreFlightTarget(session, options, stagesToRun, initialProjectState),
                    new PreFlightRun(runId, runStart, stageOutcomes, executionSnapshot),
                    scopedProgress,
                    cancellationToken).ConfigureAwait(false);
            if (preFlightResult is not null)
            {
                return preFlightResult;
            }

            // --- Execute stages ---
            var runContext = new StageRunContext(session, options, runtimeSelections!, executionSnapshot);
            var telemetry = new StageRunTelemetry(scopedProgress, projectId, runId);
            var loopPlan = new StageLoopPlan(stagesToRun, declinedOptionalStages, stageOutcomes);
            await RunStageLoopAndPostExportAsync(runContext, telemetry, loopPlan, cancellationToken)
                .ConfigureAwait(false);

            // --- Build final result ---
            return BuildCompletedResult(runId, runStart, stageOutcomes, executionSnapshot);
        }
    }

    /// <summary>Stage execution followed by the post-lip-synthesis export check.</summary>
    private async Task RunStageLoopAndPostExportAsync(
        StageRunContext runContext,
        StageRunTelemetry telemetry,
        StageLoopPlan loopPlan,
        CancellationToken cancellationToken)
    {
        await RunStageLoopAsync(runContext, telemetry, loopPlan, cancellationToken).ConfigureAwait(false);

        await RunPostLipSynthesisExportIfNeededAsync(
            runContext,
            telemetry,
            loopPlan,
            cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// In-session setup that runs before pre-flight: cold-start evidence channel,
    /// voice-cloning consent, media spine creation for fresh SDK runs, and the
    /// telemetry project id hoist.
    /// </summary>
    private static async Task<(TranscriptProjectState? State, Guid ProjectId)> PrepareProjectContextAsync(
        ProjectContextRequest request,
        CancellationToken cancellationToken)
    {
        // Best-effort evidence channel for the end-of-run cold-start summary line.
        request.ColdStart.SetLogger(TryResolveService<IApplicationLogger>(request.Session));

        if (UnattendedVoiceSelection.RequestsVoiceCloning(request.Options))
        {
            TryResolveService<IConsentService>(request.Session)?.GrantVoiceCloningConsent();
        }

        TranscriptProjectState? initialProjectState = null;

        // --- Ensure project/media spine exists for fresh SDK runs ---
        if (request.RequiresSourceMedia)
        {
            using (BenchmarkPhaseCapture.Start("import"))
                initialProjectState = await EnsureMediaSpineCreatedAsync(request.Session, request.Options, cancellationToken).ConfigureAwait(false);
        }

        (initialProjectState, Guid projectId) = await ResolveTelemetryProjectStateAsync(
            request.Session, initialProjectState, cancellationToken).ConfigureAwait(false);

        if (initialProjectState is not null)
        {
            DubbingExecutionSnapshot.UpdateExecutionSnapshotWithSourceAudioKind(
                request.ExecutionSnapshot,
                request.Options,
                ResolveSourceAudioKind(initialProjectState));
        }

        return (initialProjectState, projectId);
    }

    /// <summary>
    /// Pre-flight checks followed by a selections reload (starter-pack or settings
    /// changes made during pre-flight become visible to execution) and the
    /// execution-snapshot merge.
    /// </summary>
    private static async Task<(DubbingRunResult? Error, IReadOnlySet<string> DeclinedStages, RuntimeModelSelections? Selections)>
        PrepareStageInputsAsync(
            PreFlightTarget target,
            PreFlightRun run,
            IProgress<PipelineProgressEvent>? progress,
            CancellationToken cancellationToken)
    {
        DubbingRunResult? preFlightResult;
        IReadOnlySet<string> declinedOptionalStages;
        using (BenchmarkPhaseCapture.Start("preflight"))
        {
            (preFlightResult, declinedOptionalStages) = await RunPreFlightChecksAsync(
                target,
                run,
                progress,
                cancellationToken).ConfigureAwait(false);
        }

        if (preFlightResult is not null)
        {
            return (preFlightResult, declinedOptionalStages, null);
        }

        // Reload selections after pre-flight so starter-pack or settings changes are visible to execution.
        RuntimeModelSelections runtimeSelections = await CreateRuntimeSelectionsAsync(
            target.Session,
            target.Options,
            cancellationToken,
            target.State).ConfigureAwait(false);

        DubbingExecutionSnapshot.MergeRuntimeModelSelectionsIntoSnapshot(
            run.ExecutionSnapshot,
            runtimeSelections,
            TryResolveService<IModelAliasResolver>(target.Session));

        return (null, declinedOptionalStages, runtimeSelections);
    }

    /// <summary>
    /// Validates the stage filter and target-language selection before any I/O.
    /// Returns an error result when validation fails, otherwise null.
    /// </summary>
    private static DubbingRunResult? ValidateStageSelection(
        DubbingSessionOptions options,
        string[] stagesToRun,
        RunScope scope)
    {
        if (stagesToRun.Length == 0 && options.StageFilter is { Count: > 0 })
        {
            return BuildErrorResult(scope,
                DubbingRunStatus.Failed,
                preFlightFailures: [$"StageFilter did not match any known pipeline stage: {string.Join(", ", options.StageFilter)}"]);
        }

        string[] stagesMissingTargetLanguage = stagesToRun
            .Where(DubbingPipelineStages.RequiresTargetLanguage)
            .ToArray();
        if (string.IsNullOrWhiteSpace(options.TargetLanguageCode)
            && stagesMissingTargetLanguage.Length > 0)
        {
            return BuildErrorResult(scope,
                DubbingRunStatus.Failed,
                preFlightFailures:
                [$"Target language is required for stage(s): {string.Join(", ", stagesMissingTargetLanguage)}."]);
        }

        return null;
    }

    /// <summary>
    /// Resolves the effective options, falling back to the stored source-media path
    /// from SQLite for existing projects. The media file must exist ONLY if a stage
    /// in this run consumes it. Returns an error result when the media is missing.
    /// </summary>
    private async Task<(DubbingSessionOptions Options, DubbingRunResult? Error)> ResolveEffectiveSourceMediaAsync(
        DubbingSessionOptions options,
        bool requiresSourceMedia,
        RunScope scope,
        CancellationToken cancellationToken)
    {
        // Stages such as Translation, Tts, and Export operate against cached artifacts in
        // the project directory and do not need the original source media file to be present.
        // For existing projects, resolve the stored source path from SQLite before failing.
        DubbingSessionOptions effectiveOptions = options;

        if (NeedsStoredSourceMediaLookup(options, requiresSourceMedia))
        {
            DubbingProjectContext? projectContext = await DubbingProjectContextResolver
                .TryOpenAsync(_sessionFactory, options.ProjectOutputDirectory!, cancellationToken)
                .ConfigureAwait(false);

            if (!string.IsNullOrWhiteSpace(projectContext?.SourceMediaPath))
            {
                effectiveOptions = options with { SourceMediaPath = projectContext.SourceMediaPath };
            }
        }

        if (requiresSourceMedia && !File.Exists(effectiveOptions.SourceMediaPath))
        {
            return (effectiveOptions, BuildErrorResult(scope,
                DubbingRunStatus.Failed,
                preFlightFailures: [$"Media file not found: {effectiveOptions.SourceMediaPath}"]));
        }

        return (effectiveOptions, null);
    }

    private static bool NeedsStoredSourceMediaLookup(DubbingSessionOptions options, bool requiresSourceMedia) =>
        requiresSourceMedia
        && !File.Exists(options.SourceMediaPath)
        && options.ProjectOutputDirectory is not null;

    private static IReadOnlyDictionary<string, bool> BuildExplicitCloudEgressConsents(
        IReadOnlyList<string>? explicitConsents)
    {
        var map = new Dictionary<string, bool>(StringComparer.OrdinalIgnoreCase);
        if (explicitConsents is not null)
        {
            foreach (string key in explicitConsents.Where(static key => !string.IsNullOrWhiteSpace(key)))
            {
                map[key.Trim().ToLowerInvariant()] = true;
            }
        }

        return map;
    }

    private static string EnsureProjectDirectory(DubbingSessionOptions options)
    {
        string projectOutputDirectory = DubbingProjectPaths.ResolveProjectDirectory(
            options.SourceMediaPath,
            options.ProjectOutputDirectory);

        if (!Directory.Exists(projectOutputDirectory))
        {
            Directory.CreateDirectory(projectOutputDirectory);
        }

        return projectOutputDirectory;
    }

    private (IDubbingSession? Session, DubbingRunResult? Error) CreateSessionOrError(
        string projectOutputDirectory,
        DubbingSessionOptions options,
        RunScope scope)
    {
        try
        {
            StudioSettings sessionSettings = StudioSettings.Default with
            {
                DefaultSourceLanguage = options.SourceLanguageCode,
                DefaultTargetLanguage = options.TargetLanguageCode,
                // Explicit per-run TTS timing (CLI/SDK) wins over StudioSettings.Default.
                TtsTiming = options.TtsTiming ?? StudioSettings.Default.TtsTiming,
                // WinML catalog device policy for this run when the host pinned one.
                WindowsMlExecutionDevicePolicy = options.WindowsMlExecutionDevicePolicy
                    ?? StudioSettings.Default.WindowsMlExecutionDevicePolicy,
                // Seeds SessionCloudEgressConsentOverlay for this run. Engine consent
                // checks read it from the session; it is not written to the shared store.
                CloudEgressConsents = BuildExplicitCloudEgressConsents(options.ExplicitCloudEgressConsents),
            };
            return (_sessionFactory.CreateSession(projectOutputDirectory, sessionSettings), null);
        }
        catch (Exception ex)
        {
            return (null, BuildErrorResult(scope,
                DubbingRunStatus.Failed,
                preFlightFailures: [$"Failed to create session: {ex.Message}"]));
        }
    }

    /// <summary>
    /// Captures the project id for transient-fault telemetry (spec §4.4 follow-up lane V2).
    /// Without this hoist, per-project aggregation silently drops engine-emitted faults whose
    /// ProjectId defaults to Guid.Empty (the bus's CountsByKindForProject rejects Empty).
    /// Tied to the ensure-media-spine path so we don't OpenAsync three times per run.
    /// </summary>
    private static async Task<(TranscriptProjectState? State, Guid ProjectId)> ResolveTelemetryProjectStateAsync(
        IDubbingSession session,
        TranscriptProjectState? initialProjectState,
        CancellationToken cancellationToken)
    {
        try
        {
            if (initialProjectState is not null)
            {
                return (initialProjectState, initialProjectState.ProjectState.Project.Id);
            }

            TranscriptProjectState state = await session.Workspace.Project
                .OpenAsync(cancellationToken).ConfigureAwait(false);
            return (state, state.ProjectState.Project.Id);
        }
        // OperationCanceledException naturally propagates through filter-less try-blocks; no
        // explicit rethrow arm needed. Per AGENTS.md "Unnecessary try/catch blocks. Prefer to remove those."
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Best-effort fallback: Snapshot() consumers (DiagnosticsBundleExporter) still receive
            // engine-emitted rows; per-project filter consumers see them as Guid.Empty.
            return (initialProjectState, Guid.Empty);
        }
    }

    /// <summary>
    /// Exports the repaired video after successful lip synthesis, replacing any
    /// earlier mix-producing Export outcome.
    /// </summary>
    private async Task RunPostLipSynthesisExportIfNeededAsync(
        StageRunContext run,
        StageRunTelemetry telemetry,
        StageLoopPlan plan,
        CancellationToken cancellationToken)
    {
        if (!ShouldRunPostLipSynthesisExport(plan.StagesToRun, plan.StageOutcomes))
        {
            return;
        }

        int priorExportIndex = plan.StageOutcomes.FindLastIndex(static outcome =>
            string.Equals(outcome.StageName, StageNames.Export, StringComparison.OrdinalIgnoreCase));
        if (priorExportIndex >= 0)
        {
            plan.StageOutcomes.RemoveAt(priorExportIndex);
        }

        StageOutcome postLipExportOutcome = await ExecuteStageAsync(
            run,
            telemetry,
            new StageStep(StageNames.Export, cancellationToken)).ConfigureAwait(false);
        plan.StageOutcomes.Add(postLipExportOutcome);
    }

    private static DubbingRunResult BuildCompletedResult(
        Guid runId,
        DateTimeOffset runStart,
        List<StageOutcome> stageOutcomes,
        Dictionary<string, string> executionSnapshot)
    {
        DubbingRunStatus overallStatus = DetermineOverallStatus(stageOutcomes);
        return new DubbingRunResult
        {
            RunId = runId,
            CorrelationId = runId,
            StartTime = runStart,
            EndTime = DateTimeOffset.UtcNow,
            OverallStatus = overallStatus,
            StageOutcomes = stageOutcomes.AsReadOnly(),
            ExecutionSnapshot = executionSnapshot.AsReadOnly(),
        };
    }

    /// <summary>
    /// Runs pre-flight checks for all stages that will execute.
    /// Uses <see cref="IPipelineReadinessService"/> when available to evaluate all stages
    /// upfront, auto-provision downloadable models before the stage loop, and fail fast
    /// with an aggregated error list instead of discovering failures mid-run.
    /// Falls back to <see cref="IPipelinePreFlightChecker"/> when the readiness service
    /// is not registered (backward-compatible).
    /// </summary>
    private static async Task<(DubbingRunResult? Error, IReadOnlySet<string> DeclinedStages)> RunPreFlightChecksAsync(
        PreFlightTarget target,
        PreFlightRun run,
        IProgress<PipelineProgressEvent>? progress,
        CancellationToken cancellationToken)
    {
        if (target.Session.Workspace.Project is null)
        {
            return (null, EmptyDeclinedStages); // No project yet; pre-flight not applicable.
        }

        RuntimeModelSelections preFlightSelections = await CreateRuntimeSelectionsAsync(
            target.Session,
            target.Options,
            cancellationToken,
            target.State).ConfigureAwait(false);
        DubbingExecutionSnapshot.MergeRuntimeModelSelectionsIntoSnapshot(
            run.ExecutionSnapshot,
            preFlightSelections,
            TryResolveService<IModelAliasResolver>(target.Session));

        // ── New path: IPipelineReadinessService ───────────────────────────────
        IPipelineReadinessService? readinessService = TryResolveService<IPipelineReadinessService>(target.Session);
        RuntimeModelSetupCoordinator? coordinator = TryResolveService<RuntimeModelSetupCoordinator>(target.Session);

        if (readinessService is not null && coordinator is not null)
        {
            var readinessContext = new ReadinessPreFlightContext(
                target,
                run,
                new PreFlightProvisioning(readinessService, coordinator, progress, cancellationToken));
            return await RunPreFlightWithReadinessServiceAsync(readinessContext, preFlightSelections)
                .ConfigureAwait(false);
        }

        // ── Legacy fallback: IPipelinePreFlightChecker ────────────────────────
        IPipelinePreFlightChecker? checker = TryResolveService<IPipelinePreFlightChecker>(target.Session);
        if (checker is null)
        {
            return (null, EmptyDeclinedStages);
        }

        return (await RunLegacyModelPreFlightAsync(
            target,
            run,
            checker,
            cancellationToken).ConfigureAwait(false), EmptyDeclinedStages);
    }

    /// <summary>
    /// Legacy pre-flight path: ensures models per stage via
    /// <see cref="IPipelinePreFlightChecker"/>, deferring auto-downloadable
    /// VAD/ASR/Diarization models to stage execution.
    /// </summary>
    private static async Task<DubbingRunResult?> RunLegacyModelPreFlightAsync(
        PreFlightTarget target,
        PreFlightRun run,
        IPipelinePreFlightChecker checker,
        CancellationToken cancellationToken)
    {
        var failures = new List<string>();

        foreach (string stageName in target.StageNames)
        {
            if (ShouldSkipModelPreFlight(stageName, target.Options.ModelPreferences))
                continue;

            if (!await NeedsModelCheckAsync(target, run, stageName, cancellationToken).ConfigureAwait(false))
                continue;

            string? failure = await CheckStageModelsAsync(target, stageName, checker, cancellationToken)
                .ConfigureAwait(false);
            if (failure is not null)
            {
                failures.Add(failure);
            }
        }

        return failures.Count > 0
            ? BuildErrorResult(
                new RunScope(run.RunId, run.RunStart, run.StageOutcomes, run.ExecutionSnapshot),
                DubbingRunStatus.PreFlightFailed,
                failures)
            : null;
    }

    private static async Task<bool> NeedsModelCheckAsync(
        PreFlightTarget target,
        PreFlightRun run,
        string stageName,
        CancellationToken cancellationToken) =>
        target.Options.ForceRerun
        || !await HasValidExistingArtifactsAsync(
            new ArtifactResumeContext(target.Session, target.Options, run.ExecutionSnapshot),
            stageName,
            cancellationToken).ConfigureAwait(false);

    /// <summary>
    /// Ensures one stage's models through the legacy checker. Returns the failure detail
    /// when the stage cannot proceed, or null when its models are satisfied (including
    /// auto-downloadable VAD/ASR/diarization models deferred to stage execution).
    /// </summary>
    private static async Task<string?> CheckStageModelsAsync(
        PreFlightTarget target,
        string stageName,
        IPipelinePreFlightChecker checker,
        CancellationToken cancellationToken)
    {
        try
        {
            await checker.EnsureModelsAvailableAsync(
                stageName,
                cancellationToken,
                string.Equals(stageName, StageNames.Asr, StringComparison.OrdinalIgnoreCase)
                    ? target.Options.SourceLanguageCode
                    : null).ConfigureAwait(false);
            return null;
        }
        catch (RequiredModelNotAvailableException ex)
        {
            // Legacy: auto-downloadable VAD/ASR/Diar still deferred to stage execution.
            return !ex.CanAutoDownload || !IsDeferredModelStage(stageName)
                ? $"{stageName}: {ex.ModelId}"
                : null;
        }
        catch (OperationCanceledException) { throw; }
        catch (Exception ex)
        {
            // Catch any other unexpected exception during pre-flight model check
            // and record it as a failure for this stage
            return $"{stageName}: {ex.Message}";
        }
    }

    private static bool IsDeferredModelStage(string stageName) =>
        string.Equals(stageName, StageNames.Vad, StringComparison.OrdinalIgnoreCase)
        || string.Equals(stageName, StageNames.Asr, StringComparison.OrdinalIgnoreCase)
        || string.Equals(stageName, StageNames.Diarization, StringComparison.OrdinalIgnoreCase);

    private static readonly IReadOnlySet<string> EmptyDeclinedStages =
        new HashSet<string>(StringComparer.OrdinalIgnoreCase);

    internal static bool ShouldSkipModelPreFlight(
        string stageName,
        IReadOnlyDictionary<string, string>? modelPreferences)
    {
        if (!TryGetModelOverrideAlias(modelPreferences, stageName, out string modelAlias))
        {
            return false;
        }

        return ResolveCloudAliasPredicate(stageName)?.Invoke(modelAlias) ?? false;
    }

    private static bool TryGetModelOverrideAlias(
        IReadOnlyDictionary<string, string>? modelPreferences,
        string stageName,
        out string modelAlias)
    {
        modelAlias = string.Empty;
        if (!HasModelOverrideAlias(modelPreferences, stageName, out string? alias))
        {
            return false;
        }

        modelAlias = alias!;
        return true;
    }

    private static bool HasModelOverrideAlias(
        IReadOnlyDictionary<string, string>? modelPreferences,
        string stageName,
        out string? alias)
    {
        alias = null;
        if (modelPreferences is null)
        {
            return false;
        }

        if (!modelPreferences.TryGetValue(stageName, out alias))
        {
            return false;
        }

        return !string.IsNullOrWhiteSpace(alias);
    }

    private static Func<string, bool>? ResolveCloudAliasPredicate(string stageName)
    {
        if (string.Equals(stageName, StageNames.Asr, StringComparison.OrdinalIgnoreCase))
        {
            return AsrModelOverrideSettings.IsCloudAlias;
        }

        if (string.Equals(stageName, StageNames.Translation, StringComparison.OrdinalIgnoreCase))
        {
            return TranslationModelOverrideSettings.IsCloudAlias;
        }

        if (string.Equals(stageName, StageNames.Tts, StringComparison.OrdinalIgnoreCase))
        {
            return TtsModelOverrideSettings.IsCloudAlias;
        }

        return IsTextRefinementStage(stageName) ? IsGeminiModelAlias : null;
    }

    private static bool IsTextRefinementStage(string stageName) =>
        string.Equals(stageName, StageNames.TextRefinementAsr, StringComparison.OrdinalIgnoreCase)
        || string.Equals(stageName, StageNames.TextRefinementTranslation, StringComparison.OrdinalIgnoreCase)
        || string.Equals(stageName, StageNames.TextRefinement, StringComparison.OrdinalIgnoreCase);

    private static bool IsGeminiModelAlias(string modelAlias) =>
        modelAlias.StartsWith("gemini", StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Ensures the project's media spine (project record + normalized audio) exists.
    /// For fresh SDK runs the project has not been created yet; this creates it from the
    /// source media path without running any transcription stages.
    /// </summary>
    private static async Task<TranscriptProjectState?> EnsureMediaSpineCreatedAsync(
        IDubbingSession session,
        DubbingSessionOptions options,
        CancellationToken cancellationToken)
    {
        TranscriptWorkspace workspace = session.Workspace;
        try
        {
            return await workspace.Project.OpenAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (IsProjectMissingException(ex))
        {
            // Project does not exist yet — create the media spine below.
        }

        string projectName = Path.GetFileNameWithoutExtension(options.SourceMediaPath);
        await workspace.CreateMediaSpineAsync(
            new CreateTranscriptProjectRequest(projectName, options.SourceMediaPath),
            cancellationToken).ConfigureAwait(false);

        return null;
    }

    private static bool IsProjectMissingException(Exception ex)
    {
        // ProjectMediaIngestService.OpenAsync throws this exact InvalidOperationException
        // when no project record exists yet.
        if (ex is InvalidOperationException &&
            ex.Message.Contains("does not contain a project record", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        // On a brand-new SDK run the SQLite file can exist (created lazily on connection
        // open) before migrations have created the "projects" table — same "fresh project"
        // signal as above, just surfaced lower in the stack.
        if (string.Equals(ex.GetType().FullName, "Microsoft.Data.Sqlite.SqliteException", StringComparison.Ordinal) &&
            ex.Message.Contains("no such table: projects", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        // Do not widen this further to bare exception types or loose message substrings:
        // those also fire for real failures (DB corruption, permission errors, unrelated
        // schema mismatches) and would silently misroute them into CreateMediaSpineAsync
        // instead of surfacing them.
        return false;
    }

    private static T? TryResolveService<T>(IDubbingSession session) where T : class
    {
        try { return session.Services.GetService<T>(); }
        catch (ObjectDisposedException ex)
        {
            TryLogServiceResolutionFailure<T>(session, ex);
            return null;
        }
        catch (InvalidOperationException ex)
        {
            TryLogServiceResolutionFailure<T>(session, ex);
            return null;
        }
    }

    private static void TryLogServiceResolutionFailure<T>(IDubbingSession session, Exception ex)
    {
        try
        {
            session.Services.GetService<IApplicationLogger>()
                ?.LogWarning(
                    $"Service resolution failed for {typeof(T).Name}; dependent pipeline checks are disabled for this run.",
                    ex);
        }
        catch (Exception)
        {
            // Best-effort logging: swallow any exception from the replaceable logger.
        }
    }

    private static readonly IReadOnlyDictionary<string, RuntimeStage> RuntimeStageByStageName =
        new Dictionary<string, RuntimeStage>(StringComparer.OrdinalIgnoreCase)
        {
            [StageNames.Vad] = RuntimeStage.Vad,
            [StageNames.Asr] = RuntimeStage.Asr,
            [StageNames.Diarization] = RuntimeStage.Diarization,
            [StageNames.Translation] = RuntimeStage.Translation,
            [StageNames.Tts] = RuntimeStage.Tts,
            [StageNames.Separation] = RuntimeStage.Separation,
            [StageNames.AudioPreparation] = RuntimeStage.SpeechEnhancement,
            [StageNames.SpeechEnhancement] = RuntimeStage.SpeechEnhancement,
            [StageNames.OverlapRescue] = RuntimeStage.OverlapRescue,
            [StageNames.TextRefinementAsr] = RuntimeStage.TextRefinement,
            [StageNames.LipSync] = RuntimeStage.LipSync,
            [StageNames.LipSynthesis] = RuntimeStage.LipSynthesis,
        };

    internal static RuntimeStage? MapStageNameToRuntimeStage(string stageName) =>
        RuntimeStageByStageName.TryGetValue(stageName, out RuntimeStage stage) ? stage : null;

    internal readonly record struct StageWorkflowResult(
        IReadOnlyList<string> ArtifactPaths,
        IReadOnlyList<string>? DegradationRecords,
        StageStatus Status = StageStatus.Succeeded,
        string? ReasonCode = null);

    /// <summary>A stage progress update bundled for <see cref="ReportProgress"/>.</summary>
    private readonly record struct StageProgressEvent(
        string StageName,
        PipelineProgressEventKind Kind,
        string? Message,
        TimeSpan Elapsed = default);

    /// <summary>Run-scoped bookkeeping shared by validation, session create, and error results.</summary>
    private sealed record RunScope(
        Guid RunId,
        DateTimeOffset RunStart,
        List<StageOutcome> StageOutcomes,
        Dictionary<string, string>? ExecutionSnapshot = null);

    /// <summary>A transient fault observed during stage execution, bundled for the telemetry bus.</summary>
    private sealed record EngineTransientFault(
        Guid ProjectId,
        string StageName,
        TransientFailureKind Kind,
        Exception Exception,
        DateTimeOffset StageStart,
        int AttemptNumber,
        Guid RunId);

    /// <summary>Inputs for the in-session project-context setup that precedes pre-flight.</summary>
    private sealed record ProjectContextRequest
    {
        public required IDubbingSession Session { get; init; }
        public required DubbingSessionOptions Options { get; init; }
        public required bool RequiresSourceMedia { get; init; }
        public required Dictionary<string, string> ExecutionSnapshot { get; init; }
        public required ColdStartCapture ColdStart { get; init; }
    }

    /// <summary>
    /// Executes a single pipeline stage, wrapping it in timing, transient retry, and error handling.
    /// Transient failures (filesystem locks, SQLite busy, HF mirror 5xx) are retried up to
    /// <see cref="StageRetryBudget.Default"/> attempts with exponential backoff before being
    /// reported as terminal failures.
    /// </summary>
    private async Task<StageOutcome> ExecuteStageAsync(
        StageRunContext run,
        StageRunTelemetry telemetry,
        StageStep step,
        TtsStreamingPrefetch? ttsPrefetch = null)
    {
        var context = new StageExecutionContext
        {
            Session = run.Session,
            Options = run.Options,
            RuntimeSelections = run.RuntimeSelections,
            Progress = telemetry.Progress,
            TtsPrefetch = ttsPrefetch,
        };
        DateTimeOffset stageStart = DateTimeOffset.UtcNow;
        ReportProgress(
            telemetry.Progress,
            new StageProgressEvent(step.StageName, PipelineProgressEventKind.Started, null));

        StageRetryBudget retryBudget = StageRetryBudget.Default;
        int attempt = 1;
        while (true)
        {
            try
            {
                StageWorkflowResult workflowResult = await RunStageWorkflowAsync(
                    context,
                    step.StageName,
                    step.CancellationToken).ConfigureAwait(false);

                return BuildStageOutcome(step, telemetry, stageStart, workflowResult);
            }
            catch (OperationCanceledException) when (step.CancellationToken.IsCancellationRequested)
            {
                return BuildCancelledStageOutcome(step, telemetry, stageStart, "Cancelled");
            }
            catch (Exception ex) when (TransientFailureClassifier.IsTransient(ex))
            {
                TransientFailureKind kind = TransientFailureClassifier.Classify(ex);
                PublishEngineTransient(
                    new EngineTransientFault(
                        telemetry.ProjectId,
                        step.StageName,
                        kind,
                        ex,
                        stageStart,
                        attempt,
                        telemetry.RunId));

                if (attempt >= retryBudget.MaxAttempts)
                {
                    string retryExhaustedMessage = $"{ex.GetType().Name}: {ex.Message} (transient retry exhausted after {attempt} attempts)";
                    return BuildFailedStageOutcome(step, telemetry, stageStart,
                        new StageFailureDetail(retryExhaustedMessage, "STAGE_FAILED_TRANSIENT", [retryExhaustedMessage]));
                }

                try
                {
                    await Task.Delay(retryBudget.BackoffFor(attempt), step.CancellationToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (step.CancellationToken.IsCancellationRequested)
                {
                    return BuildCancelledStageOutcome(step, telemetry, stageStart, "Cancelled during retry backoff");
                }
                attempt++;
            }
            catch (Exception ex)
            {
                // Catch all exceptions from stage execution to ensure we always
                // return a StageOutcome rather than letting exceptions propagate.
                // This allows the pipeline to report failures gracefully.
                return BuildFailedStageOutcome(step, telemetry, stageStart,
                    new StageFailureDetail(ex.Message, "STAGE_FAILED", [ex.Message]));
            }
        }
    }

    private static StageOutcome BuildStageOutcome(
        StageStep step,
        StageRunTelemetry telemetry,
        DateTimeOffset stageStart,
        StageWorkflowResult workflowResult)
    {
        DateTimeOffset stageEnd = DateTimeOffset.UtcNow;
        PipelineProgressEventKind eventKind = workflowResult.Status switch
        {
            StageStatus.Skipped => PipelineProgressEventKind.Skipped,
            StageStatus.Failed => PipelineProgressEventKind.Failed,
            // PartiallySucceeded reports Completed: the stage finished, and the
            // degradation detail is carried in the message and outcome records.
            _ => PipelineProgressEventKind.Completed,
        };
        string? progressMessage = workflowResult.ReasonCode
            ?? (workflowResult.DegradationRecords is { Count: > 0 } records ? records[0] : null);
        ReportProgress(
            telemetry.Progress,
            new StageProgressEvent(step.StageName, eventKind, progressMessage, stageEnd - stageStart));

        return new StageOutcome
        {
            StageName = step.StageName,
            Status = workflowResult.Status,
            StartTime = stageStart,
            EndTime = stageEnd,
            ArtifactPaths = workflowResult.ArtifactPaths,
            DegradationRecords = workflowResult.DegradationRecords,
            ReasonCode = workflowResult.ReasonCode,
        };
    }

    private static StageOutcome BuildCancelledStageOutcome(
        StageStep step,
        StageRunTelemetry telemetry,
        DateTimeOffset stageStart,
        string message) =>
        BuildFailedStageOutcome(step, telemetry, stageStart, new StageFailureDetail(message, "CANCELLED", null));

    private static StageOutcome BuildFailedStageOutcome(
        StageStep step,
        StageRunTelemetry telemetry,
        DateTimeOffset stageStart,
        StageFailureDetail failure)
    {
        DateTimeOffset stageEnd = DateTimeOffset.UtcNow;
        ReportProgress(
            telemetry.Progress,
            new StageProgressEvent(step.StageName, PipelineProgressEventKind.Failed, failure.Message, stageEnd - stageStart));

        return new StageOutcome
        {
            StageName = step.StageName,
            Status = StageStatus.Failed,
            StartTime = stageStart,
            EndTime = stageEnd,
            ArtifactPaths = [],
            ReasonCode = failure.ReasonCode,
            DegradationRecords = failure.DegradationRecords,
        };
    }

    /// <summary>Failure detail carried into a failed or cancelled stage outcome.</summary>
    private sealed record StageFailureDetail(
        string Message,
        string ReasonCode,
        IReadOnlyList<string>? DegradationRecords);

    private static async Task<RuntimeModelSelections> CreateRuntimeSelectionsAsync(
        IDubbingSession session,
        DubbingSessionOptions options,
        CancellationToken cancellationToken,
        TranscriptProjectState? state = null)
    {
        InferenceModelPreferences? preferences = DubbingModelPreferences.BuildModelPreferences(options);
        RuntimeModelSelections selections;

        if (TryResolveService<IPipelineRuntimeSelectionsProvider>(session) is { } selectionsProvider)
        {
            RuntimeModelSelections? provided = await selectionsProvider
                .CreateSelectionsAsync(state, options, cancellationToken)
                .ConfigureAwait(false);
            if (provided is not null)
            {
                // Explicit per-stage aliases in options.ModelPreferences are caller
                // intent (e.g. the Chatterbox pin ApplyVoiceCloningDefaults installs for
                // clone runs) and take precedence over the host's UI-side selections,
                // matching the precedence the settings path applies via
                // CreateSelectionsFromSettings. ApplyPresetVoiceModelSelection may still
                // retarget TTS when the voices that will speak are Qwen presets, because
                // synthesis will not use a non-CustomVoice model for those voices.
                selections = DubbingModelPreferences.ApplyModelPreferenceAliases(provided with
                {
                    SelectionIntents = Enum.GetValues<RuntimeStage>().ToDictionary(stage => stage, provided.GetSelectionIntent)
                }, preferences);
                selections = UnattendedVoiceSelection.ApplyPresetVoiceModelSelection(selections, options, state);
                return DubbingModelPreferences.ApplyExecutionProviderPins(selections, options, preferences);
            }
        }

        StudioSettings settings = StudioSettings.Default;
        IServiceProvider? serviceProvider = session.Services;
        if (serviceProvider?.GetService<IStudioSettingsService>() is IStudioSettingsService settingsService)
        {
            settings = await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
        }

        selections = RuntimeModelRequestFactory.CreateSelectionsFromSettings(
            settings,
            preferences);
        selections = DubbingModelPreferences.ApplyModelPreferenceAliases(selections, preferences);
        selections = UnattendedVoiceSelection.ApplyPresetVoiceModelSelection(selections, options, state);
        return DubbingModelPreferences.ApplyExecutionProviderPins(selections, options, preferences);
    }

    private static ArtifactKind ResolveSourceAudioKind(TranscriptProjectState state) =>
        TranscriptWorkflowUtilities.GetLatestAcceptedAmbianceStem(state.ProjectState.Artifacts, state.StageRuns) is not null
            ? ArtifactKind.Ambiance
            : ArtifactKind.NormalizedAudio;

    /// <summary>
    /// Best-effort transient-fault publisher used by <see cref="ExecuteStageAsync"/>'s
    /// transient-exception arm. Wrapped in try/catch so telemetry can never break
    /// pipeline flow. Mirrors <see cref="StageRunHelper.PublishTransient"/>'s contract
    /// but skips stage-run persistence since the engine layer does not own that
    /// lifecycle. See <c>docs/internal/pipeline-readiness-spec.md</c> section 4.4.
    /// </summary>
    private void PublishEngineTransient(EngineTransientFault fault)
    {
        try
        {
            var context = new Dictionary<string, string>
            {
                ["Engine"] = "DubbingPipelineEngine",
                ["StageStart"] = fault.StageStart.ToString("O"),
            };
            if (fault.RunId != Guid.Empty)
            {
                context["RunId"] = fault.RunId.ToString("N");
            }
            if (fault.Exception.GetType().FullName is { Length: > 0 } typeName)
            {
                context["ExceptionType"] = typeName;
            }
            _transientFaultBus.Publish(new PipelineTransientFault(
                fault.ProjectId,
                fault.StageName,
                fault.Kind,
                fault.Exception.Message,
                DateTimeOffset.UtcNow,
                fault.AttemptNumber,
                context));
        }
        catch
        {
            // Telemetry must never break pipeline flow — bus surface is best-effort.
        }
    }

    /// <summary>
    /// Streams transient-fault records emitted during this engine's lifetime.
    /// Bridges <see cref="PipelineTransientFaultBus"/> (IObservable) into an
    /// <see cref="IAsyncEnumerable{T}"/> via a <see cref="Channel{T}"/> so subscribers
    /// do not have to take an Rx dependency. Drains snapshot (already replayed by bus
    /// Subscribe), then forwards live events until the consumer cancels. See
    /// <c>docs/internal/pipeline-readiness-spec.md</c> section 4.4.
    /// </summary>
    public async IAsyncEnumerable<PipelineTransientFault> TransientFaultsAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // Bounded 256-capacity ring with DropOldest. Reasoning: a stalled consumer (e.g.
        // UI shell paused on UI thread) must not OOM the process. Stale faults are less
        // valuable than recent telemetry for the diagnostics-bundle snapshot cap; the bus
        // already retains its own 50-item ring so late-visible replay is still possible.
        var channel = Channel.CreateBounded<PipelineTransientFault>(
            new BoundedChannelOptions(256)
            {
                FullMode = BoundedChannelFullMode.DropOldest,
                SingleReader = true,
                SingleWriter = false,
            });

        var observer = new ChannelWriterObserver(channel.Writer);
        using IDisposable subscription = _transientFaultBus.Subscribe(observer);
        try
        {
            await foreach (PipelineTransientFault fault in
                channel.Reader.ReadAllAsync(cancellationToken).ConfigureAwait(false))
            {
                yield return fault;
            }
        }
        finally
        {
            channel.Writer.TryComplete();
        }
    }

    private sealed class ChannelWriterObserver(ChannelWriter<PipelineTransientFault> writer) : IObserver<PipelineTransientFault>
    {
        public void OnNext(PipelineTransientFault value) => writer.TryWrite(value);

        public void OnError(Exception error) => writer.TryComplete(error);

        public void OnCompleted() => writer.TryComplete();
    }
}
