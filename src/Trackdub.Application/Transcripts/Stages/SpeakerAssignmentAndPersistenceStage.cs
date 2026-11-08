using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts;
using Trackdub.Contracts.Projects;
using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;

namespace Trackdub.Application.Transcripts.Stages;

public sealed class SpeakerAssignmentAndPersistenceStage(
    SpeakerAssignmentService speakerAssignmentService,
    ITranscriptRepository transcriptRepository,
    TranscriptArtifactWriter artifactWriter,
    IArtifactStore artifactStore,
    IProjectStageRunStore stageRunStore)
    : ITranscriptGenerationStage
{
    private readonly SpeakerAssignmentService speakerAssignmentService = speakerAssignmentService ?? throw new ArgumentNullException(nameof(speakerAssignmentService));
    private readonly ITranscriptRepository transcriptRepository = transcriptRepository ?? throw new ArgumentNullException(nameof(transcriptRepository));
    private readonly TranscriptArtifactWriter artifactWriter = artifactWriter ?? throw new ArgumentNullException(nameof(artifactWriter));
    private readonly IArtifactStore artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));
    private readonly IProjectStageRunStore stageRunStore = stageRunStore ?? throw new ArgumentNullException(nameof(stageRunStore));

    public string StageName => StageNames.SpeakerAssignment;

    public async Task<TranscriptGenerationContext> ExecuteAsync(
        TranscriptGenerationContext context,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress = null)
    {
        if (context.AsrResult is null || context.RegionPlan is null)
        {
            throw new InvalidOperationException("ASR result or Region Plan is missing.");
        }

        Guid projectId = context.Project.Id;
        (_, SpeakerAssignmentResult speakerAssignment) = await StageRunHelper.RunStageAsync(
            stageRunStore,
            projectId,
            StageName,
            runtimeReporter: null,
            async (_, stageCancellationToken) => await PersistTranscriptAsync(
                    context,
                    progress,
                    stageCancellationToken)
                .ConfigureAwait(false),
            canceledReason: "Speaker assignment canceled.",
            cancellationToken).ConfigureAwait(false);

        return context with { SpeakerAssignment = speakerAssignment };
    }

    private async Task<SpeakerAssignmentResult> PersistTranscriptAsync(
        TranscriptGenerationContext context,
        IProgress<PipelineProgressEvent>? progress,
        CancellationToken cancellationToken)
    {
        if (context.AsrResult is null || context.RegionPlan is null)
        {
            throw new InvalidOperationException("ASR result or Region Plan is missing.");
        }

        AsrStageResult asrResult = context.AsrResult;
        TranscriptRegionPlan regionPlan = context.RegionPlan;
        Guid projectId = context.Project.Id;
        PipelineProgressReporter.Phase(progress, StageName, "Assigning speakers", "Resolving speaker labels for transcript segments.");
        SpeakerAssignmentResult speakerAssignment = context.DiarizationResult is not null
            ? new SpeakerAssignmentResult(
                context.DiarizationResult.Speakers,
                context.DiarizationResult.Turns,
                regionPlan.SpeakerIdsBySegmentIndex.Count > 0
                    ? regionPlan.SpeakerIdsBySegmentIndex
                    : SpeakerAssignmentService.AssignTranscriptSegmentsToSpeakers(asrResult.Segments, context.DiarizationResult.Speakers, context.DiarizationResult.Turns))
            : await speakerAssignmentService.CreateDefaultSpeakerAssignmentAsync(
                projectId,
                asrResult.Segments,
                cancellationToken).ConfigureAwait(false);

        int revisionNumber = await transcriptRepository.GetNextRevisionNumberAsync(projectId, cancellationToken).ConfigureAwait(false);
        Guid revisionStageRunId = TextRefinementSegmentResolution.ResolveRevisionStageRunId(
            asrResult,
            context.TextRefinementResult);
        TranscriptRevision revision = TranscriptRevision.Create(projectId, revisionStageRunId, revisionNumber, DateTimeOffset.UtcNow);
        string activeProvenance = TextRefinementSegmentResolution.ResolveActiveTranscriptProvenance(context.TextRefinementResult);

        TranscriptSegment[] segments = asrResult.Segments
            .OrderBy(segment => segment.Index)
            .Select(segment => (
                Segment: segment,
                Text: TextRefinementSegmentResolution.ResolveDisplayedText(segment, context.TextRefinementResult)))
            .Where(item => !string.IsNullOrWhiteSpace(item.Text))
            .Select(item => TranscriptSegment.Create(
                revision.Id,
                item.Segment.Index,
                item.Segment.StartSeconds,
                item.Segment.EndSeconds,
                item.Text,
                speakerAssignment.SegmentSpeakerIdsByIndex.TryGetValue(item.Segment.Index, out Guid speakerId)
                    ? speakerId
                    : null,
                item.Segment.DetectedLanguage,
                TranscriptWorkflowUtilities.CreateTranscriptWords(item.Segment.Words),
                SpeakerAssignmentOrigin.Automatic))
            .ToArray();

        int droppedEmptyCount = asrResult.Segments.Count - segments.Length;
        if (droppedEmptyCount > 0)
        {
            PipelineProgressReporter.Phase(
                progress,
                StageName,
                "Assigning speakers",
                $"Dropped {droppedEmptyCount} empty ASR region(s).");
        }

        PipelineProgressReporter.Phase(progress, StageName, "Persisting transcript", $"Saving {segments.Length} transcript segment(s).");
        await transcriptRepository.SaveRevisionAsync(revision, segments, cancellationToken).ConfigureAwait(false);
        await PersistRawAsrArtifactAsync(context, asrResult, cancellationToken).ConfigureAwait(false);
        await artifactWriter.WriteTranscriptArtifactAsync(
            projectId,
            context.MediaAsset,
            revision,
            segments,
            revisionStageRunId,
            activeProvenance,
            cancellationToken).ConfigureAwait(false);

        if (segments.Length > 0)
        {
            PipelineProgressReporter.Output(
                progress,
                StageName,
                PipelineOutputKind.TranscriptSegmentPersisted,
                segments[0].SegmentIndex,
                revisionId: revision.Id,
                segmentId: segments[0].Id,
                message: $"Persisted transcript revision {revisionNumber} ({segments.Length} segment(s)).");
        }

        IReadOnlyList<TranscriptSegmentTextProvenance> provenance = TextRefinementSegmentResolution.BuildProvenance(
            asrResult,
            context.TextRefinementResult);
        if (provenance.Count > 0)
        {
            await artifactWriter.WriteTextRefinementProvenanceArtifactAsync(
                projectId,
                context.MediaAsset,
                new TextRefinementProvenanceArtifactDocument(
                    revision.Id,
                    context.TextRefinementResult?.StageRun.Id,
                    TextRefinementScope.Asr,
                    provenance,
                    DateTimeOffset.UtcNow),
                cancellationToken).ConfigureAwait(false);
        }

        PipelineProgressReporter.Phase(progress, StageName, "Updating transcript language");
        await PersistDetectedTranscriptLanguageAsync(
            context,
            ResolveTranscriptLanguage(context, asrResult),
            cancellationToken).ConfigureAwait(false);

        await SeedAsrSegmentStageRunsAsync(context, segments, cancellationToken).ConfigureAwait(false);

        return speakerAssignment;
    }

    private Task PersistRawAsrArtifactAsync(
        TranscriptGenerationContext context,
        AsrStageResult asrResult,
        CancellationToken cancellationToken)
    {
        RawAsrTranscriptArtifactWriteRequest request = new()
        {
            ProjectId = context.Project.Id,
            MediaAsset = context.MediaAsset,
            Segments = asrResult.Segments,
            StageRunId = asrResult.StageRun.Id,
            Lineage = RawAsrTranscriptArtifactLineage.Create(
                context.AudioRoutingPlan.AsrAudioArtifact,
                context.AudioRoutingPlan.VadAudioArtifact,
                context.SpeechRegions.Count > 0 ? context.VadStageRunId : null)
        };
        return artifactWriter.WriteRawAsrTranscriptArtifactAsync(request, cancellationToken);
    }

    private static string? ResolveTranscriptLanguage(
        TranscriptGenerationContext context,
        AsrStageResult asrResult)
    {
        string? detectedLanguage = TranscriptWorkflowUtilities.ResolveDetectedTranscriptLanguage(asrResult.Segments);
        if (detectedLanguage is not null)
        {
            return detectedLanguage;
        }

        string? requestedSourceLanguage = TranscriptWorkflowUtilities.NormalizeTranscriptLanguageCode(context.SourceLanguage);
        return string.Equals(requestedSourceLanguage, "auto", StringComparison.OrdinalIgnoreCase)
            ? null
            : requestedSourceLanguage;
    }

    private async Task SeedAsrSegmentStageRunsAsync(
        TranscriptGenerationContext context,
        TranscriptSegment[] segments,
        CancellationToken cancellationToken)
    {
        Guid asrStageRunId = context.AsrResult!.StageRun.Id;
        if (asrStageRunId == Guid.Empty || segments.Length == 0)
        {
            return;
        }

        ProjectManifest? manifest = await artifactStore
            .ReadJsonAsync<ProjectManifest>(ProjectArtifactPaths.ManifestRelativePath, cancellationToken)
            .ConfigureAwait(false);
        manifest ??= ProjectManifest.FromProject(context.Project);

        int[] segmentIndices = segments.Select(static segment => segment.SegmentIndex).ToArray();
        ProjectUiSettings updatedUiSettings = SegmentStageRunProvenanceStore.RecordAsrRuns(
            manifest.UiSettings,
            segmentIndices,
            segmentIndices.ToHashSet(),
            asrStageRunId);

        await SegmentStageRunProvenanceStore.PersistUiSettingsAsync(
            artifactStore,
            context.Project,
            manifest.TranscriptLanguage,
            updatedUiSettings,
            cancellationToken).ConfigureAwait(false);
    }

    private async Task PersistDetectedTranscriptLanguageAsync(
        TranscriptGenerationContext context,
        string? detectedTranscriptLanguage,
        CancellationToken cancellationToken)
    {
        if (detectedTranscriptLanguage is null)
        {
            return;
        }

        ProjectManifest? manifest = await artifactStore.ReadJsonAsync<ProjectManifest>(
            ProjectArtifactPaths.ManifestRelativePath,
            cancellationToken).ConfigureAwait(false);
        manifest ??= ProjectManifest.FromProject(context.Project);
        if (string.Equals(manifest.TranscriptLanguage, detectedTranscriptLanguage, StringComparison.Ordinal))
        {
            return;
        }

        await artifactStore.WriteJsonAsync(
            ProjectArtifactPaths.ManifestRelativePath,
            manifest.WithTranscriptLanguage(detectedTranscriptLanguage),
            cancellationToken).ConfigureAwait(false);
    }
}
