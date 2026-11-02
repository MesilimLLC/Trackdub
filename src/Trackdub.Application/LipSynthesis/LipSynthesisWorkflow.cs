using Trackdub.Application.Artifacts;
using Trackdub.Application.Mixing;
using Trackdub.Application.Transcripts;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Mixing;

namespace Trackdub.Application.LipSynthesis;

/// <summary>Request to run the M23 lip-synthesis stage for the current project.</summary>
public sealed record LipSynthesisRunRequest(
    bool IsLicenseApproved = true,
    bool AllowExperimentalExecution = false,
    string? PreferredModelAlias = null);

/// <summary>
/// Orchestrates the M23 video lip-synthesis stage over the current project state. Builds per-speaker
/// turns from diarization, runs the gated experimental LatentSync engine, and reloads
/// the per-turn results. User-triggered — never auto-runs on media load. Mirrors LipSyncWorkflow.
/// </summary>
public sealed class LipSynthesisWorkflow(
    TranscriptProjectStateService stateService,
    LipSynthesisStageHandler lipSynthesisStageHandler,
    IArtifactStore artifactStore,
    MixPlanStore? mixPlanStore = null,
    IPreviewRangeRenderer? speechRenderer = null,
    MixPlanBuilder? mixPlanBuilder = null)
    : IDisposable
{
    private const string ConcurrentOperationMessage = "A lip-synthesis operation is already running for this project.";

    private readonly TranscriptProjectStateService stateService = stateService ?? throw new ArgumentNullException(nameof(stateService));
    private readonly LipSynthesisStageHandler lipSynthesisStageHandler = lipSynthesisStageHandler ?? throw new ArgumentNullException(nameof(lipSynthesisStageHandler));
    private readonly IArtifactStore artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));
    private readonly MixPlanBuilder? mixPlanBuilder = mixPlanBuilder;
    private readonly SemaphoreSlim operationGate = new(1, 1);
    private readonly DubbedSpeechDriverRenderer? speechDriverRenderer =
        mixPlanStore is not null && speechRenderer is not null
            ? new DubbedSpeechDriverRenderer(mixPlanStore, speechRenderer, artifactStore)
            : null;

    public void Dispose() => operationGate.Dispose();

    public async Task<TranscriptProjectState> SynthesizeAllAsync(
        LipSynthesisRunRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);

        cancellationToken.ThrowIfCancellationRequested();
        if (!operationGate.Wait(0))
            throw new InvalidOperationException(ConcurrentOperationMessage);

        try
        {
            TranscriptProjectState currentState = await stateService
                .OpenAsync(requestedTranslationTargetLanguage: null, cancellationToken)
                .ConfigureAwait(false);

            // Prerequisite: a media asset that actually has video.
            if (currentState.ProjectState.MediaAsset is not { } mediaAsset || !mediaAsset.HasVideo)
                return currentState;

            // Mirror LipSync: record an honest skipped stage run when dubbing prerequisites are missing.
            if (!TranscriptWorkflowUtilities.HasGeneratedTtsTakes(currentState))
            {
                var emptyRequest = new LipSynthesisStageRequest(
                    ProjectId: currentState.ProjectState.Project.Id,
                    MediaAsset: mediaAsset,
                    SourceVideoPath: mediaAsset.SourceFilePath,
                    DubbedAudioPath: string.Empty,
                    SpeakerTurns: [],
                    IsEnabled: true,
                    IsLicenseApproved: request.IsLicenseApproved,
                    AllowExperimentalExecution: request.AllowExperimentalExecution,
                    PreferredModelAlias: request.PreferredModelAlias);

                await lipSynthesisStageHandler
                    .HandleAsync(emptyRequest, cancellationToken).ConfigureAwait(false);

                return await stateService
                    .OpenAsync(currentState.SelectedTranslationTargetLanguage, cancellationToken)
                    .ConfigureAwait(false);
            }

            string sourceVideoPath = mediaAsset.SourceFilePath;
            ProjectArtifact? dubbedDriver = TranscriptWorkflowUtilities
                .ResolveLipSynthesisDriverAudioArtifact(currentState);
            string dubbedAudioPath = dubbedDriver is not null
                ? artifactStore.GetPath(dubbedDriver.RelativePath)
                : string.Empty;
            // Build from the current project so lip repair does not require a video export or
            // overwrite the user's saved preview/export mix settings. Mouth-driving models
            // follow dubbed speech alone; the final mix also carries ambiance and music.
            string? speechOnly = await TryRenderCurrentSpeechDriverAsync(
                currentState,
                mediaAsset.DurationSeconds,
                cancellationToken).ConfigureAwait(false);
            dubbedAudioPath = speechOnly ?? dubbedAudioPath;

            // Process by diarization speaker turn (not the whole file at once).
            var turns = currentState.SpeakerTurns
                .Select(t => new LipSynthesisTurn(
                    SegmentId: t.Id,
                    Start: TimeSpan.FromSeconds(t.StartSeconds),
                    End: TimeSpan.FromSeconds(t.EndSeconds),
                    SpeakerId: t.SpeakerId.ToString("D")))
                .ToList();

            var stageRequest = new LipSynthesisStageRequest(
                ProjectId: currentState.ProjectState.Project.Id,
                MediaAsset: mediaAsset,
                SourceVideoPath: sourceVideoPath,
                DubbedAudioPath: dubbedAudioPath,
                SpeakerTurns: turns,
                IsEnabled: true,
                IsLicenseApproved: request.IsLicenseApproved,
                AllowExperimentalExecution: request.AllowExperimentalExecution,
                PreferredModelAlias: request.PreferredModelAlias);

            await lipSynthesisStageHandler
                .HandleAsync(stageRequest, cancellationToken).ConfigureAwait(false);

            return await stateService
                .OpenAsync(currentState.SelectedTranslationTargetLanguage, cancellationToken)
                .ConfigureAwait(false);
        }
        finally
        {
            operationGate.Release();
        }
    }

    private async Task<string?> TryRenderCurrentSpeechDriverAsync(
        TranscriptProjectState currentState,
        double? mediaDurationSeconds,
        CancellationToken cancellationToken)
    {
        if (speechDriverRenderer is null)
        {
            return null;
        }

        if (mixPlanBuilder is null)
        {
            // Compatibility path for callers that only provide the previously saved mix plan.
            return await speechDriverRenderer
                .TryRenderAsync(mediaDurationSeconds, cancellationToken)
                .ConfigureAwait(false);
        }

        MixPlan? savedPlan = currentState.CurrentMixPlan;
        MixPlan mixPlan;
        try
        {
            mixPlan = mixPlanBuilder.Build(new MixPlanBuildRequest(
                currentState.ProjectState.Project.Id,
                currentState.ProjectState.MediaAsset?.Id,
                currentState.ProjectState.Artifacts,
                currentState.TranscriptSegments,
                currentState.TranslatedSegments,
                currentState.TtsTakes,
                currentState.StageRuns,
                SourceGainDb: savedPlan?.SourceGainDb ?? 0d,
                DubbedSpeechGainDb: savedPlan?.DubbedSpeechGainDb ?? 0d,
                DuckingGainDb: savedPlan?.DuckingGainDb,
                DuckingLeadSeconds: savedPlan?.DuckingLeadSeconds ?? 0.05d,
                DuckingTailSeconds: savedPlan?.DuckingTailSeconds ?? 0.18d,
                RestoreOriginalPan: savedPlan?.RestoreOriginalPan ?? false,
                ApplyTimbrePolish: savedPlan?.ApplyTimbrePolish ?? true,
                CandidateGroups: currentState.TtsCandidateGroups,
                MediaDurationSeconds: mediaDurationSeconds));
        }
        catch (InvalidOperationException)
        {
            // Keep the export-audio fallback when current source audio is unavailable.
            return null;
        }

        return await speechDriverRenderer
            .TryRenderAsync(mixPlan, mediaDurationSeconds, cancellationToken)
            .ConfigureAwait(false);
    }
}
