using Trackdub.Application.Transcripts;
using Trackdub.Contracts;
using Trackdub.Contracts.Transcripts;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Transcripts.Pipeline;

public interface ITranscriptPipelineBuilder
{
    ITranscriptPipelineBuilder AddStage(ITranscriptGenerationStage stage, StageOptions? options = null);
    ITranscriptGenerationPipeline Build();
}

public interface ITranscriptGenerationPipeline
{
    Task<TranscriptGenerationContext> ExecuteAsync(
        TranscriptGenerationContext context,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress = null);
}

public sealed class TranscriptGenerationPipeline : ITranscriptGenerationPipeline
{
    private readonly IReadOnlyList<(ITranscriptGenerationStage Stage, StageOptions Options)> stages;
    private readonly IArtifactStore? artifactStore;
    private readonly TranscriptArtifactWriter? artifactWriter;
    private readonly ITranscriptRepository? transcriptRepository;
    private readonly PipelineDegradationWriter? degradationWriter;
    private readonly IProjectStageRunStore? stageRunStore;

    public TranscriptGenerationPipeline(
        IReadOnlyList<(ITranscriptGenerationStage Stage, StageOptions Options)> stages,
        IArtifactStore? artifactStore = null,
        TranscriptArtifactWriter? artifactWriter = null,
        ITranscriptRepository? transcriptRepository = null,
        PipelineDegradationWriter? degradationWriter = null,
        IProjectStageRunStore? stageRunStore = null)
    {
        this.stages = stages ?? throw new ArgumentNullException(nameof(stages));
        this.artifactStore = artifactStore;
        this.artifactWriter = artifactWriter;
        this.transcriptRepository = transcriptRepository;
        this.degradationWriter = degradationWriter;
        this.stageRunStore = stageRunStore;
    }

    /// <summary>
    /// Convenience constructor: wraps each stage with <see cref="StageOptions.Default"/> (no timeout).
    /// </summary>
    public TranscriptGenerationPipeline(
        IReadOnlyList<ITranscriptGenerationStage> stages,
        IArtifactStore? artifactStore = null,
        TranscriptArtifactWriter? artifactWriter = null,
        ITranscriptRepository? transcriptRepository = null,
        PipelineDegradationWriter? degradationWriter = null,
        IProjectStageRunStore? stageRunStore = null)
        : this(
            (stages ?? throw new ArgumentNullException(nameof(stages)))
                .Select(static s => (s, StageOptions.Default))
                .ToArray(),
            artifactStore,
            artifactWriter,
            transcriptRepository,
            degradationWriter,
            stageRunStore)
    {
    }

    public async Task<TranscriptGenerationContext> ExecuteAsync(
        TranscriptGenerationContext context,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress = null)
    {
        foreach ((ITranscriptGenerationStage stage, StageOptions options) in stages)
        {
            if (await ShouldResumeSkipStageAsync(context, stage.StageName, cancellationToken).ConfigureAwait(false))
            {
                DateTimeOffset resumedStageStart = DateTimeOffset.UtcNow;
                PipelineProgressReporter.Started(progress, stage.StageName);
                const string resumeSkipReason = "Skipped — valid artifacts from prior run";
                if (stageRunStore is not null)
                {
                    StageRunRecord resumedRun = await StageRunHelper
                        .StartAsync(stageRunStore, context.Project.Id, stage.StageName, cancellationToken)
                        .ConfigureAwait(false);
                    await StageRunHelper
                        .SkipAsync(stageRunStore, resumedRun, null, resumeSkipReason, cancellationToken)
                        .ConfigureAwait(false);
                }

                if (artifactStore is not null &&
                    artifactWriter is not null &&
                    transcriptRepository is not null)
                {
                    context = await TranscriptPipelineResumeHydrator.HydrateSkippedStageAsync(
                            context,
                            stage.StageName,
                            artifactWriter,
                            artifactStore,
                            transcriptRepository,
                            cancellationToken)
                        .ConfigureAwait(false);
                }

                PipelineProgressReporter.Skipped(
                    progress,
                    stage.StageName,
                    resumeSkipReason,
                    DateTimeOffset.UtcNow - resumedStageStart);
                continue;
            }

            // Create a per-stage CTS when a timeout is configured. It is linked to the outer
            // token so that user/workspace cancellation still propagates; the stage sees a
            // single token that fires on whichever source triggers first.
            CancellationTokenSource? stageTimeoutCts = null;
            if (options.Timeout is { } budget)
            {
                stageTimeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                stageTimeoutCts.CancelAfter(budget);
            }

            CancellationToken stageToken = stageTimeoutCts?.Token ?? cancellationToken;
            DateTimeOffset stageStart = DateTimeOffset.UtcNow;
            PipelineProgressReporter.Started(progress, stage.StageName);

            try
            {
                PipelineProgressReporter.Phase(progress, stage.StageName, "Running");
                context = await stage.ExecuteAsync(context, stageToken, progress).ConfigureAwait(false);
                ReportTerminalProgress(progress, stage, context, DateTimeOffset.UtcNow - stageStart);
            }
            catch (OperationCanceledException)
                when (stageToken.IsCancellationRequested)
            {
                PipelineProgressReporter.Failed(
                    progress,
                    stage.StageName,
                    "Stage canceled.",
                    DateTimeOffset.UtcNow - stageStart);
                // Both user/workspace cancel and per-stage timeout are intentional terminations.
                // Re-throw without writing an unhandled-exception degradation record.
                // (StageRunHelper.RunStageAsync — called inside stage.ExecuteAsync — already
                // records the stage run as Canceled with an appropriate reason string.)
                // Note: stageToken is either the timeout-linked CTS token (which is itself linked
                // to the outer cancellationToken) or cancellationToken directly, so this guard
                // covers both sources without needing to inspect each CTS individually.
                throw;
            }
            catch (Exception ex)
            {
                PipelineProgressReporter.Failed(
                    progress,
                    stage.StageName,
                    ex.Message,
                    DateTimeOffset.UtcNow - stageStart);
                await WriteUnhandledStageDegradationAsync(stage, context, ex, cancellationToken).ConfigureAwait(false);
                throw;
            }
            finally
            {
                stageTimeoutCts?.Dispose();
            }
        }

        return context;
    }

    private async Task<bool> ShouldResumeSkipStageAsync(
        TranscriptGenerationContext context,
        string stageName,
        CancellationToken cancellationToken)
    {
        if (artifactStore is null)
        {
            return false;
        }

        if (context.ForceRerun)
        {
            return false;
        }

        if (context.ProjectState is null)
        {
            return false;
        }

        if (string.IsNullOrWhiteSpace(context.ProjectRootPath))
        {
            return false;
        }

        if (await VadSnapshotIsNotResumableAsync(context, stageName, cancellationToken).ConfigureAwait(false))
        {
            return false;
        }

        if (!HasCurrentVadInputsForResume(context, stageName) ||
            !StageConfigurationAllowsResume(context, stageName))
        {
            return false;
        }

        return StageArtifactResumeEvaluator.CanResumeStage(
            context.ProjectState!,
            artifactStore!,
            stageName,
            context.ExecutionSnapshot,
            context.ProjectRootPath!);
    }

    private async Task<bool> VadSnapshotIsNotResumableAsync(
        TranscriptGenerationContext context,
        string stageName,
        CancellationToken cancellationToken)
    {
        if (!string.Equals(stageName, StageNames.Vad, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        return !await HasCurrentVadSnapshotForResumeAsync(context, cancellationToken).ConfigureAwait(false);
    }

    private async Task<bool> HasCurrentVadSnapshotForResumeAsync(
        TranscriptGenerationContext context,
        CancellationToken cancellationToken)
    {
        if (artifactWriter is null)
        {
            return false;
        }

        SpeechRegionsArtifactSnapshot? snapshot = await artifactWriter
            .TryReadSpeechRegionsSnapshotAsync(
                context.Project.Id,
                context.AudioRoutingPlan.VadAudioArtifact,
                cancellationToken)
            .ConfigureAwait(false);
        StageRunRecord? latestVadRun = StageArtifactResumeEvaluator.GetLatestSuccessfulRun(
            context.ProjectState!.StageRuns,
            StageNames.Vad);
        return SnapshotMatchesLatestVadRun(snapshot, latestVadRun);
    }

    private static bool SnapshotMatchesLatestVadRun(
        SpeechRegionsArtifactSnapshot? snapshot,
        StageRunRecord? latestVadRun)
    {
        if (snapshot is null || latestVadRun is null)
        {
            return false;
        }

        return snapshot.StageRunId == latestVadRun.Id;
    }

    private bool StageConfigurationAllowsResume(TranscriptGenerationContext context, string stageName) =>
        (!string.Equals(stageName, StageNames.Diarization, StringComparison.OrdinalIgnoreCase) ||
         context.EnableSpeakerDiarization) &&
        (!string.Equals(stageName, StageNames.TextRefinementAsr, StringComparison.OrdinalIgnoreCase) ||
         context.ModelPreferences.EnableAsrTextRefinement) &&
        (!string.Equals(stageName, StageNames.SpeakerAssignment, StringComparison.OrdinalIgnoreCase) ||
         SpeakerAssignmentInputsCanResume(context));

    private bool SpeakerAssignmentInputsCanResume(TranscriptGenerationContext context)
    {
        if (!TranscriptResumeLineageEvaluator.HasCurrentRawAsrArtifact(
                context.ProjectState,
                context.AudioRoutingPlan,
                context.VadStageRunId) ||
            !StageArtifactResumeEvaluator.CanResumeStage(
                context.ProjectState!,
                artifactStore!,
                StageNames.Asr,
                context.ExecutionSnapshot,
                context.ProjectRootPath!))
        {
            return false;
        }

        return !context.ModelPreferences.EnableAsrTextRefinement ||
               StageArtifactResumeEvaluator.CanResumeStage(
                   context.ProjectState!,
                   artifactStore!,
                   StageNames.TextRefinementAsr,
                   context.ExecutionSnapshot,
                   context.ProjectRootPath!);
    }

    private static bool HasCurrentVadInputsForResume(TranscriptGenerationContext context, string stageName) =>
        TranscriptResumeLineageEvaluator.HasCurrentVadInputsForResume(context, stageName);

    internal static bool HasCurrentVadRunForResume(TranscriptGenerationContext context, string stageName)
    {
        ArgumentNullException.ThrowIfNull(context);
        return TranscriptResumeLineageEvaluator.HasCurrentVadRunForStage(
            context.ProjectState,
            context.VadStageRunId,
            context.SpeechRegions,
            stageName);
    }

    private async Task WriteUnhandledStageDegradationAsync(
        ITranscriptGenerationStage stage,
        TranscriptGenerationContext context,
        Exception exception,
        CancellationToken cancellationToken)
    {
        if (degradationWriter is null)
        {
            return;
        }

        try
        {
            await degradationWriter.WriteAsync(
                new PipelineDegradationRecord(
                    stage.StageName,
                    "PIPELINE_STAGE_UNHANDLED_EXCEPTION",
                    $"Unhandled exception escaped the {stage.StageName} pipeline stage.",
                    Detail: exception.Message,
                    SelectedFallback: null,
                    RecommendedAction: "Review the stage failure details and retry after fixing the underlying issue.",
                    DateTimeOffset.UtcNow,
                    StageRunId: null),
                context.Project.Id,
                context.MediaAsset.Id,
                cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Degradation writes are best-effort and must not mask the original stage failure.
        }
    }

    private static void ReportTerminalProgress(
        IProgress<PipelineProgressEvent>? progress,
        ITranscriptGenerationStage stage,
        TranscriptGenerationContext context,
        TimeSpan elapsed)
    {
        if (string.Equals(stage.StageName, StageNames.Asr, StringComparison.OrdinalIgnoreCase))
        {
            StageRunStatus? asrStatus = context.AsrResult?.StageRun.Status;
            string? modelAlias = context.AsrResult?.StageRun.RuntimeInfo?.ModelAlias;
            string? provider = context.AsrResult?.StageRun.RuntimeInfo?.SelectedProvider;
            string runtimeLabel = string.IsNullOrWhiteSpace(modelAlias)
                ? string.IsNullOrWhiteSpace(provider) ? "auto" : provider
                : string.IsNullOrWhiteSpace(provider) ? modelAlias : $"{modelAlias} @ {provider}";

            if (asrStatus == StageRunStatus.Skipped)
            {
                PipelineProgressReporter.Skipped(
                    progress,
                    stage.StageName,
                    $"ASR skipped ({context.AsrResult?.StageRun.FailureReason ?? "no speech"}).",
                    elapsed);
                return;
            }

            int segmentCount = context.AsrResult?.Segments.Count ?? 0;
            if (asrStatus is StageRunStatus.Completed or StageRunStatus.PartiallyCompleted && segmentCount == 0)
            {
                PipelineProgressReporter.Failed(
                    progress,
                    stage.StageName,
                    $"ASR '{runtimeLabel}' finished with 0 transcript segments — translation/TTS cannot start.",
                    elapsed);
                return;
            }

            PipelineProgressReporter.Completed(
                progress,
                stage.StageName,
                elapsed,
                $"ASR '{runtimeLabel}': {segmentCount} segment(s).");
            return;
        }

        if (string.Equals(stage.StageName, StageNames.Diarization, StringComparison.OrdinalIgnoreCase) &&
            context.EnableSpeakerDiarization &&
            context.SpeechRegions.Count == 0 &&
            context.DiarizationResult is null)
        {
            PipelineProgressReporter.Skipped(progress, stage.StageName, "Diarization skipped (no speech regions).", elapsed);
            return;
        }

        PipelineProgressReporter.Completed(progress, stage.StageName, elapsed);
    }
}

public sealed class TranscriptPipelineBuilder(
    PipelineDegradationWriter? degradationWriter = null,
    IArtifactStore? artifactStore = null,
    TranscriptArtifactWriter? artifactWriter = null,
    ITranscriptRepository? transcriptRepository = null,
    IProjectStageRunStore? stageRunStore = null)
    : ITranscriptPipelineBuilder
{
    private readonly List<(ITranscriptGenerationStage Stage, StageOptions Options)> stages = [];

    public ITranscriptPipelineBuilder AddStage(ITranscriptGenerationStage stage, StageOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(stage);
        stages.Add((stage, options ?? StageOptions.Default));
        return this;
    }

    public ITranscriptGenerationPipeline Build()
    {
        return new TranscriptGenerationPipeline(
            stages.ToArray(),
            artifactStore,
            artifactWriter,
            transcriptRepository,
            degradationWriter,
            stageRunStore);
    }
}
