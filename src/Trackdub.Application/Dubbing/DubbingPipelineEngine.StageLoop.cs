using Microsoft.Extensions.DependencyInjection;
using Trackdub.Application.Pipeline;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Contracts.Dubbing;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Dubbing;

public sealed partial class DubbingPipelineEngine
{
    /// <summary>Session-scoped inputs every stage execution reads.</summary>
    private sealed record StageRunContext(
        IDubbingSession Session,
        DubbingSessionOptions Options,
        RuntimeModelSelections RuntimeSelections,
        Dictionary<string, string> ExecutionSnapshot);

    private sealed record StageRunTelemetry(
        IProgress<PipelineProgressEvent>? Progress,
        Guid ProjectId,
        Guid RunId);

    /// <summary>The ordered stages for this run and where their outcomes are collected.</summary>
    private sealed record StageLoopPlan(
        string[] StagesToRun,
        IReadOnlySet<string> DeclinedOptionalStages,
        List<StageOutcome> StageOutcomes);

    /// <summary>The ordered stages to run, and the inputs for choosing which later stage to warm up next.</summary>
    internal sealed record StageSchedule(
        string[] StagesToRun,
        RuntimeModelSelections RuntimeSelections,
        DubbingSessionOptions Options,
        IReadOnlySet<string> DeclinedOptionalStages);

    /// <summary>A named stage together with the token governing its execution.</summary>
    private readonly record struct StageStep(string StageName, CancellationToken CancellationToken);

    /// <summary>The index of the next stage to warm up and the token governing that warmup.</summary>
    private readonly record struct WarmupStart(int NextIndex, CancellationToken CancellationToken);

    /// <summary>
    /// Per-stage hooks for <see cref="RunScheduledStagesAsync"/>: a gate that may return a skip
    /// outcome, the execution itself, where outcomes go, and where warmup skips are reported.
    /// </summary>
    internal sealed record StageCallbacks(
        Func<string, CancellationToken, Task<StageOutcome?>> GateStageAsync,
        Func<string, CancellationToken, Task<StageOutcome>> ExecuteStageAsync,
        Action<StageOutcome> RecordOutcome,
        IProgress<PipelineProgressEvent>? Progress);

    /// <summary>
    /// Runs each requested stage in order, honoring cancellation, failed prerequisites,
    /// and artifact resume. Outcomes are appended to <see cref="StageLoopPlan.StageOutcomes"/>.
    /// </summary>
    private async Task RunStageLoopAsync(
        StageRunContext run,
        StageRunTelemetry telemetry,
        StageLoopPlan plan,
        CancellationToken cancellationToken)
    {
        var loop = new StageLoop(this, run, telemetry, plan);
        try
        {
            await RunScheduledStagesAsync(
                new StageSchedule(plan.StagesToRun, run.RuntimeSelections, run.Options, plan.DeclinedOptionalStages),
                run.Session.Services?.GetService<IStageWarmupCoordinator>(),
                new StageCallbacks(
                    (stageName, token) => loop.GateAsync(new StageStep(stageName, token)),
                    (stageName, token) => loop.ExecuteAsync(new StageStep(stageName, token)),
                    loop.Record,
                    telemetry.Progress),
                cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            await loop.DisposeAsync().ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Stage sequence with next-stage warmup lookahead: the warmup for the next warmable
    /// stage is started (not awaited) immediately before the current stage executes, so
    /// its proving session create overlaps the current stage's work instead of sitting
    /// serially in the gap between stages. Each stage's gate runs before its pending
    /// warmup is joined, and a failed warmup verdict is applied only when the gate
    /// would let the stage execute: it skips that stage with
    /// <see cref="StageSkipReasonCodes.WarmupFailed"/> and the failure detail (the
    /// coordinator records no verdict for a failed smoke), while a passed or
    /// not-attempted warmup leaves execution untouched. Gate-skipped
    /// stages (cancelled, prerequisite-failed, declined, resume-valid) keep their
    /// gate outcome and start no new warmup, matching the previous post-stage
    /// scheduling for those paths.
    /// </summary>
    internal static async Task RunScheduledStagesAsync(
        StageSchedule schedule,
        IStageWarmupCoordinator? warmupCoordinator,
        StageCallbacks callbacks,
        CancellationToken cancellationToken)
    {
        var lookahead = new WarmupLookahead(warmupCoordinator, schedule);

        try
        {
            for (int i = 0; i < schedule.StagesToRun.Length; i++)
            {
                var step = new StageStep(schedule.StagesToRun[i], cancellationToken);

                // Gate first: a gate-skipped stage (cancelled, prerequisite-failed,
                // declined, resume-valid) keeps its gate outcome even when its
                // lookahead warmup failed — the warmup verdict applies only to a
                // stage that would actually execute.
                StageOutcome? gateOutcome = await callbacks.GateStageAsync(step.StageName, step.CancellationToken).ConfigureAwait(false);
                if (gateOutcome is not null)
                {
                    await lookahead.DiscardIfTargetingAsync(step).ConfigureAwait(false);
                    callbacks.RecordOutcome(gateOutcome);
                    continue;
                }

                StageOutcome? warmupSkip = await lookahead
                    .TakeFailedVerdictAsync(step, callbacks.Progress)
                    .ConfigureAwait(false);
                if (warmupSkip is not null)
                {
                    callbacks.RecordOutcome(warmupSkip);
                    continue;
                }

                lookahead.StartConcurrentIfAllowed(new WarmupStart(i + 1, cancellationToken));
                StageOutcome outcome = await callbacks.ExecuteStageAsync(step.StageName, step.CancellationToken).ConfigureAwait(false);
                callbacks.RecordOutcome(outcome);
                lookahead.StartIfIdle(new WarmupStart(i + 1, cancellationToken));
            }
        }
        finally
        {
            await lookahead.ObserveRemainingAsync(cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Gate, execution and outcome bookkeeping for one run's stage loop, including the
    /// optional TTS prefetch that overlaps translation.
    /// </summary>
    private sealed class StageLoop(
        DubbingPipelineEngine engine,
        StageRunContext run,
        StageRunTelemetry telemetry,
        StageLoopPlan plan) : IAsyncDisposable
    {
        private string? _failedPrerequisiteStage;
        private TtsStreamingPrefetch? _ttsPrefetch;

        public async Task<StageOutcome?> GateAsync(StageStep step)
        {
            if (step.CancellationToken.IsCancellationRequested)
                return Skip(step.StageName, "CANCELLED", "Cancelled");

            if (_failedPrerequisiteStage is not null)
            {
                return Skip(
                    step.StageName,
                    StageSkipReasonCodes.PrerequisiteFailed,
                    $"Skipped due to failed prerequisite: {_failedPrerequisiteStage}");
            }

            // Optional stages whose model the user declined during pre-flight provisioning
            // would fail mid-run without the model.
            if (plan.DeclinedOptionalStages.Contains(step.StageName))
            {
                return Skip(
                    step.StageName,
                    StageSkipReasonCodes.OptionalModelDeclined,
                    "Skipped — optional model setup declined");
            }

            bool hasValidArtifacts = !run.Options.ForceRerun &&
                await HasValidExistingArtifactsAsync(
                    new ArtifactResumeContext(run.Session, run.Options, run.ExecutionSnapshot),
                    step.StageName,
                    step.CancellationToken).ConfigureAwait(false);
            return hasValidArtifacts
                ? Skip(step.StageName, StageSkipReasonCodes.ExistingArtifactsValid, "Skipped — valid artifacts from prior run")
                : null;
        }

        public async Task<StageOutcome> ExecuteAsync(StageStep step)
        {
            await StartTtsPrefetchIfOverlappingAsync(step).ConfigureAwait(false);

            StageOutcome outcome = await engine.ExecuteStageAsync(
                run,
                telemetry,
                step,
                _ttsPrefetch).ConfigureAwait(false);

            if (IsTtsPrefetchSpent(step, outcome))
            {
                await DisposeAsync().ConfigureAwait(false);
            }

            return outcome;
        }

        public void Record(StageOutcome outcome)
        {
            plan.StageOutcomes.Add(outcome);
            if (IsBlockingPrerequisiteOutcome(outcome))
            {
                _failedPrerequisiteStage = outcome.StageName;
            }
        }

        public async ValueTask DisposeAsync()
        {
            if (_ttsPrefetch is null)
                return;

            await _ttsPrefetch.DisposeAsync().ConfigureAwait(false);
            _ttsPrefetch = null;
        }

        private StageOutcome Skip(string stageName, string reasonCode, string message)
        {
            ReportProgress(
                telemetry.Progress,
                new StageProgressEvent(stageName, PipelineProgressEventKind.Skipped, message));
            return BuildSkippedOutcome(stageName, reasonCode);
        }

        private async Task StartTtsPrefetchIfOverlappingAsync(StageStep step)
        {
            if (_ttsPrefetch is not null ||
                !ShouldOverlapTtsWithTranslation(
                    run.Options,
                    plan.StagesToRun,
                    Array.IndexOf(plan.StagesToRun, step.StageName),
                    plan.DeclinedOptionalStages))
            {
                return;
            }

            _ttsPrefetch = await TryCreateTtsPrefetchAsync(
                run.Session,
                run.Options,
                run.RuntimeSelections,
                step.CancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Unclaimed clips are useless once TTS has run or translation produced no revision.
        /// </summary>
        private static bool IsTtsPrefetchSpent(StageStep step, StageOutcome outcome) =>
            string.Equals(step.StageName, StageNames.Tts, StringComparison.OrdinalIgnoreCase) ||
            (string.Equals(step.StageName, StageNames.Translation, StringComparison.OrdinalIgnoreCase) &&
             outcome.Status == StageStatus.Failed);
    }

    /// <summary>
    /// Tracks the single in-flight lookahead warmup and decides when to start, join or
    /// discard it.
    /// </summary>
    private sealed class WarmupLookahead(IStageWarmupCoordinator? coordinator, StageSchedule schedule)
    {
        private PendingStageWarmup? _pending;

        /// <summary>Joins and discards the verdict so no warmup task is left unobserved.</summary>
        public async Task DiscardIfTargetingAsync(StageStep step)
        {
            if (TakePendingFor(step) is { } pending)
            {
                await ObserveWarmupAsync(pending.Task, step.CancellationToken).ConfigureAwait(false);
            }
        }

        public async Task<StageOutcome?> TakeFailedVerdictAsync(
            StageStep step,
            IProgress<PipelineProgressEvent>? progress)
        {
            if (TakePendingFor(step) is not { } pending)
                return null;

            StageWarmupResult? warmupResult =
                await AwaitWarmupResultAsync(pending.Task, step.CancellationToken).ConfigureAwait(false);
            StageOutcome? warmupSkip = BuildWarmupSkipOutcomeIfFailed(step.StageName, warmupResult);
            if (warmupSkip is not null)
            {
                ReportProgress(
                    progress,
                    new StageProgressEvent(
                        step.StageName,
                        PipelineProgressEventKind.Skipped,
                        warmupResult?.Detail is { } detail
                            ? $"Skipped — stage warmup failed: {detail}"
                            : "Skipped — stage warmup failed"));
            }

            return warmupSkip;
        }

        /// <summary>
        /// Capacity permitting, starts the next stage's proving warmup now so it overlaps this
        /// stage's execution; it is joined at the top of its target iteration.
        /// </summary>
        public void StartConcurrentIfAllowed(WarmupStart next)
        {
            if (ShouldStartConcurrentWarmup(coordinator, _pending))
            {
                _pending = Start(coordinator!, next);
            }
        }

        /// <summary>
        /// At single-entry capacity (or without concurrent warmup), starts the next warmup only
        /// once the current stage has completed and released its model lease.
        /// </summary>
        public void StartIfIdle(WarmupStart next)
        {
            if (coordinator is not null && _pending is null)
            {
                _pending = Start(coordinator, next);
            }
        }

        /// <summary>
        /// Stops joining warmup promptly on cancellation; the coordinator receives the same
        /// token and will stop at its next cancellable checkpoint.
        /// </summary>
        public async Task ObserveRemainingAsync(CancellationToken cancellationToken)
        {
            if (_pending is not null)
            {
                await ObserveWarmupAsync(_pending.Task, cancellationToken).ConfigureAwait(false);
            }
        }

        private PendingStageWarmup? TakePendingFor(StageStep step)
        {
            if (_pending is null ||
                !string.Equals(_pending.StageName, step.StageName, StringComparison.OrdinalIgnoreCase))
            {
                return null;
            }

            PendingStageWarmup pending = _pending;
            _pending = null;
            return pending;
        }

        private PendingStageWarmup? Start(
            IStageWarmupCoordinator warmupCoordinator,
            WarmupStart next) =>
            StartNextStageWarmup(
                warmupCoordinator,
                schedule.StagesToRun,
                next.NextIndex,
                schedule.RuntimeSelections,
                schedule.Options,
                schedule.DeclinedOptionalStages,
                next.CancellationToken);
    }

    /// <summary>
    /// A prerequisite outcome blocks downstream stages (and any deliverable) when it
    /// failed outright or was skipped for a non-benign reason — e.g. a stage skipped
    /// because its proving warmup failed (<see cref="StageSkipReasonCodes.WarmupFailed"/>)
    /// produced nothing just like a failed stage. Benign skips (valid existing
    /// artifacts, declined optional model, …) never block.
    /// </summary>
    internal static bool IsBlockingPrerequisiteOutcome(StageOutcome outcome) =>
        DubbingPipelineStages.PrerequisiteStages.Contains(outcome.StageName)
        && (outcome.Status == StageStatus.Failed
            || (outcome.Status == StageStatus.Skipped
                && !IsBenignSkipReasonCode(outcome.ReasonCode)));

    /// <summary>
    /// TTS overlaps translation when segment streaming is on and TTS runs later in this run:
    /// the stage at <paramref name="index"/> must be translation.
    /// </summary>
    internal static bool ShouldOverlapTtsWithTranslation(
        DubbingSessionOptions options,
        string[] stagesToRun,
        int index,
        IReadOnlySet<string> declinedOptionalStages) =>
        options.EnableTranslationSegmentStreaming &&
        string.Equals(stagesToRun[index], StageNames.Translation, StringComparison.OrdinalIgnoreCase) &&
        !declinedOptionalStages.Contains(StageNames.Tts) &&
        stagesToRun.Skip(index + 1).Contains(StageNames.Tts, StringComparer.OrdinalIgnoreCase);

    private static async Task<TtsStreamingPrefetch?> TryCreateTtsPrefetchAsync(
        IDubbingSession session,
        DubbingSessionOptions options,
        RuntimeModelSelections runtimeSelections,
        CancellationToken cancellationToken)
    {
        try
        {
            TranscriptProjectState state = await session.Workspace.Project.OpenAsync(cancellationToken).ConfigureAwait(false);
            if (state.TranscriptSegments.Count == 0)
            {
                return null;
            }

            return await session.Workspace.CreateTtsStreamingPrefetchAsync(
                BuildPipelineTtsRequest(state, options, runtimeSelections),
                options.TargetLanguageCode,
                cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Overlap is an optimization: without it the TTS stage synthesizes every segment itself.
            TryResolveService<IApplicationLogger>(session)?.LogWarning(
                "TTS will not overlap translation: the streaming prefetch could not be prepared.", ex);
            return null;
        }
    }

    /// <summary>
    /// Concurrent warmup is safe only when the coordinator reports the GenAI pool
    /// capacity can hold the next model alongside the active one, and no warmup is
    /// already in flight.
    /// </summary>
    internal static bool ShouldStartConcurrentWarmup(
        IStageWarmupCoordinator? coordinator,
        PendingStageWarmup? pendingWarmup) =>
        coordinator is { AllowsConcurrentWarmup: true } && pendingWarmup is null;

    /// <summary>An in-flight warmup for a named pipeline stage; optimization-only.</summary>
    internal sealed record PendingStageWarmup(string StageName, Task<StageWarmupResult> Task);

    /// <summary>
    /// Starts warmup for the first stage after <paramref name="startIndex"/> that maps to a
    /// runtime stage and was not declined in pre-flight. Returns without awaiting so the
    /// model loads while the current stage executes.
    /// </summary>
    internal static PendingStageWarmup? StartNextStageWarmup(
        IStageWarmupCoordinator coordinator,
        string[] stagesToRun,
        int startIndex,
        RuntimeModelSelections runtimeSelections,
        DubbingSessionOptions options,
        IReadOnlySet<string> declinedOptionalStages,
        CancellationToken cancellationToken)
    {
        for (int i = startIndex; i < stagesToRun.Length; i++)
        {
            if (declinedOptionalStages.Contains(stagesToRun[i])
                || MapStageNameToRuntimeStage(stagesToRun[i]) is not { } runtimeStage)
            {
                continue;
            }

            Task<StageWarmupResult> task;
            try
            {
                task = coordinator.WarmAsync(
                    new StageWarmupRequest(
                        runtimeStage,
                        runtimeSelections,
                        options.SourceLanguageCode,
                        options.TargetLanguageCode,
                        UnattendedVoiceSelection.RequestsVoiceCloning(options)),
                    cancellationToken);
            }
            catch (Exception ex)
            {
                task = Task.FromResult(new StageWarmupResult(
                    Attempted: true,
                    Succeeded: false,
                    Detail: ex.Message));
            }

            return new PendingStageWarmup(stagesToRun[i], task);
        }

        return null;
    }

    /// <summary>
    /// Builds the skip outcome for a stage whose proving warmup failed, or null when the
    /// warmup passed, was never attempted (no runnable plan — including cloud selections
    /// the local planner cannot resolve; the stage must execute and surface its own
    /// outcome), or produced no result (e.g. a swallowed coordinator exception — those
    /// stay optimization-only and the stage attempts its own execution). The
    /// failure detail is preserved on the outcome; the coordinator has recorded no
    /// verdict for it, so nothing is reported ready on the strength of a failed smoke.
    /// </summary>
    internal static StageOutcome? BuildWarmupSkipOutcomeIfFailed(
        string stageName,
        StageWarmupResult? warmupResult)
    {
        if (warmupResult is not { Attempted: true, Succeeded: false })
        {
            return null;
        }

        StageOutcome outcome = BuildSkippedOutcome(stageName, StageSkipReasonCodes.WarmupFailed);
        return string.IsNullOrWhiteSpace(warmupResult.Detail)
            ? outcome
            : outcome with { DegradationRecords = [warmupResult.Detail] };
    }

    /// <summary>
    /// Joins a warmup task and returns its result: ordinary failures are returned (or
    /// null for a faulted task) rather than thrown, and cancellation is swallowed only
    /// when the run itself is cancelled.
    /// </summary>
    internal static async Task<StageWarmupResult?> AwaitWarmupResultAsync(
        Task<StageWarmupResult> warmupTask,
        CancellationToken cancellationToken)
    {
        try
        {
            return await warmupTask.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            // Run cancelled — the warmup was optimization-only.
            return null;
        }
        catch (OperationCanceledException)
        {
            // A cancelled warmup on a live run token is real cancellation, not a failure.
            throw;
        }
        catch (Exception)
        {
            // A faulted coordinator task carries no result to consume; the target stage
            // attempts its own execution and surfaces any real failure itself.
            return null;
        }
    }

    /// <summary>
    /// Joins a warmup task without letting its outcome affect the run: ordinary failures are
    /// optimization-only, and cancellation is swallowed only when the run itself is cancelled.
    /// </summary>
    internal static async Task ObserveWarmupAsync(
        Task<StageWarmupResult> warmupTask,
        CancellationToken cancellationToken)
    {
        _ = await AwaitWarmupResultAsync(warmupTask, cancellationToken).ConfigureAwait(false);
    }
}
