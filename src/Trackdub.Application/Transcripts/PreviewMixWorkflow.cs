using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Application.Mixing;
using Trackdub.Contracts.Projects;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Mixing;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Tts;

namespace Trackdub.Application.Transcripts;

public sealed class PreviewMixWorkflow(
    MixPlanBuilder mixPlanBuilder,
    MixPlanStore mixPlanStore,
    IPreviewRangeRenderer previewRangeRenderer,
    IArtifactStore artifactStore,
    IFileFingerprintService fileFingerprintService,
    IMediaAssetRepository mediaAssetRepository,
    IProjectStageRunStore stageRunStore,
    ITtsCandidateGroupRepository? candidateGroupRepository = null,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null,
    TimelineMediaMixPreparer? timelineMediaMixPreparer = null)
{
    private readonly MixPlanBuilder mixPlanBuilder = mixPlanBuilder ?? throw new ArgumentNullException(nameof(mixPlanBuilder));
    private readonly MixPlanStore mixPlanStore = mixPlanStore ?? throw new ArgumentNullException(nameof(mixPlanStore));
    private readonly IPreviewRangeRenderer previewRangeRenderer = previewRangeRenderer ?? throw new ArgumentNullException(nameof(previewRangeRenderer));
    private readonly IArtifactStore artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));
    private readonly IFileFingerprintService fileFingerprintService = fileFingerprintService ?? throw new ArgumentNullException(nameof(fileFingerprintService));
    private readonly IMediaAssetRepository mediaAssetRepository = mediaAssetRepository ?? throw new ArgumentNullException(nameof(mediaAssetRepository));
    private readonly IProjectStageRunStore stageRunStore = stageRunStore ?? throw new ArgumentNullException(nameof(stageRunStore));
    private readonly ITtsCandidateGroupRepository? candidateGroupRepository = candidateGroupRepository;

    public async Task<PreviewMixStageResult> GeneratePreviewAsync(
        TranscriptProjectState currentState,
        PreviewMixStageRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(currentState);
        ArgumentNullException.ThrowIfNull(request);
        if (request.ProjectId != currentState.ProjectState.Project.Id)
        {
            throw new InvalidOperationException("Preview mix request does not match the loaded project.");
        }

        MediaAsset mediaAsset = TranscriptWorkflowUtilities.GetRequiredMediaAsset(currentState);
        StageRunRecord stageRun = await StageRunHelper
            .StartAsync(stageRunStore, request.ProjectId, StageNames.PreviewMix, cancellationToken)
            .ConfigureAwait(false);

        try
        {
            IReadOnlyList<TtsCandidateGroup>? candidateGroups = candidateGroupRepository is not null
                ? await candidateGroupRepository.GetByProjectAsync(request.ProjectId, cancellationToken).ConfigureAwait(false)
                : null;

            MixPlan mixPlan = mixPlanBuilder.Build(new MixPlanBuildRequest(
                request.ProjectId,
                mediaAsset.Id,
                currentState.ProjectState.Artifacts,
                currentState.TranscriptSegments,
                currentState.TranslatedSegments,
                currentState.TtsTakes,
                currentState.StageRuns,
                request.SourceGainDb,
                request.DubbedSpeechGainDb,
                request.DuckingGainDb,
                RestoreOriginalPan: request.RestoreOriginalPan,
                ApplyTimbrePolish: request.ApplyTimbrePolish,
                CandidateGroups: candidateGroups, MediaDurationSeconds: mediaAsset.DurationSeconds,
                BackgroundMediaGainDb: request.BackgroundMediaGainDb,
                MediaClips: await TimelineMediaMixPreparer.PrepareAsync(
                    timelineMediaMixPreparer, currentState, cancellationToken).ConfigureAwait(false)));
            await mixPlanStore.SaveAsync(mixPlan, cancellationToken).ConfigureAwait(false);

            string previewRelativePath = ProjectArtifactPaths.GetPreviewMixRelativePath(stageRun.Id);
            await using ArtifactWriteHandle writeHandle = artifactStore.CreateWriteHandle(previewRelativePath);
            PreviewRangeRenderResult renderResult = await previewRangeRenderer
                .RenderAsync(
                    new PreviewRangeRenderRequest(
                        mixPlan,
                        request.StartSeconds,
                        request.EndSeconds,
                        writeHandle.TemporaryPath),
                    cancellationToken)
                .ConfigureAwait(false);
            await artifactStore.CommitAsync(writeHandle, cancellationToken).ConfigureAwait(false);

            FileFingerprint fingerprint = await fileFingerprintService
                .ComputeAsync(writeHandle.FinalPath, cancellationToken)
                .ConfigureAwait(false);
            var previewArtifact = new ProjectArtifact(
                Guid.NewGuid(),
                currentState.ProjectState.Project.Id,
                mediaAsset.Id,
                ArtifactKind.PreviewMix,
                previewRelativePath,
                fingerprint.Sha256,
                fingerprint.SizeBytes,
                renderResult.DurationSeconds,
                renderResult.SampleRate,
                renderResult.ChannelCount,
                DateTimeOffset.UtcNow,
                stageRun.Id,
                "preview-mix");
            await mediaAssetRepository.SaveArtifactAsync(previewArtifact, cancellationToken).ConfigureAwait(false);

            StageRunRecord completed = await StageRunHelper
                .CompleteAsync(stageRunStore, stageRun, runtimeReporter: null, cancellationToken, runtimePlanningPreferences)
                .ConfigureAwait(false);
            // Unlike ExportStageHandler, warnings (missing/stale takes) are intentionally
            // non-blocking here: preview supports iterating on an in-progress project before
            // every take is finished. Callers can inspect mixPlan.Warnings themselves.
            return new PreviewMixStageResult(
                completed,
                previewRelativePath,
                mixPlan,
                renderResult.DurationSeconds,
                mixPlan.Warnings);
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not TaskCanceledException)
        {
            await StageRunHelper
                .FailAsync(stageRunStore, stageRun, runtimeReporter: null, ex.Message, cancellationToken, runtimePlanningPreferences)
                .ConfigureAwait(false);
            throw;
        }
        catch (OperationCanceledException)
        {
            await StageRunHelper
                .CancelAsync(stageRunStore, stageRun, runtimeReporter: null, "Preview mix canceled.", CancellationToken.None, runtimePlanningPreferences)
                .ConfigureAwait(false);
            throw;
        }
    }
}
