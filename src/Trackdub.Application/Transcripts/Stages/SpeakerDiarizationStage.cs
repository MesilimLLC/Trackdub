using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Application.Transcripts;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Transcripts.Stages;

public sealed class SpeakerDiarizationStage(
    SpeakerAssignmentService speakerAssignmentService,
    TranscriptArtifactWriter artifactWriter,
    IArtifactStore artifactStore,
    IProjectStageRunStore stageRunStore)
    : ITranscriptGenerationStage
{
    private readonly SpeakerAssignmentService speakerAssignmentService = speakerAssignmentService ?? throw new ArgumentNullException(nameof(speakerAssignmentService));
    private readonly TranscriptArtifactWriter artifactWriter = artifactWriter ?? throw new ArgumentNullException(nameof(artifactWriter));
    private readonly IArtifactStore artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));
    private readonly IProjectStageRunStore stageRunStore = stageRunStore ?? throw new ArgumentNullException(nameof(stageRunStore));

    public string StageName => StageNames.Diarization;

    public async Task<TranscriptGenerationContext> ExecuteAsync(
        TranscriptGenerationContext context,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress = null)
    {
        ProjectArtifact diarizationAudioArtifact = context.AudioRoutingPlan.DiarizationAudioArtifact;
        double durationSeconds = diarizationAudioArtifact.DurationSeconds
                                 ?? context.NormalizedAudioArtifact.DurationSeconds
                                 ?? context.MediaAsset.DurationSeconds;

        if (context.EnableSpeakerDiarization && context.SpeechRegions.Count == 0)
        {
            StageRunRecord stageRun = await StageRunHelper
                .StartAsync(stageRunStore, context.Project.Id, StageName, cancellationToken)
                .ConfigureAwait(false);
            await StageRunHelper
                .SkipAsync(stageRunStore, stageRun, null, StageSkipReasonCodes.NoSpeechRegions, cancellationToken)
                .ConfigureAwait(false);

            PipelineProgressReporter.Phase(
                progress,
                StageName,
                "Skipping diarization",
                "VAD detected no speech regions; no speaker turns will be created.");
            return context with
            {
                DiarizationResult = null,
                RegionPlan = TranscriptWorkflowUtilities.BuildTranscriptRegionPlan([], null, durationSeconds)
            };
        }

        string speechAudioPath = artifactStore.GetPath(diarizationAudioArtifact.RelativePath);

        PipelineProgressReporter.Phase(
            progress,
            StageName,
            context.EnableSpeakerDiarization ? "Identifying speakers" : "Building default region plan",
            context.EnableSpeakerDiarization ? "Running speaker diarization." : "Speaker diarization is disabled for this run.");
        DiarizationResult? diarizationResult = context.EnableSpeakerDiarization
            ? await speakerAssignmentService.CreateDiarizationAsync(
                context.Project.Id,
                context.MediaAsset.Id,
                speechAudioPath,
                durationSeconds,
                context.SpeechRegions,
                context.ModelPreferences.DiarizationModelAlias,
                context.ModelPreferences.GetPreferredExecutionProvider(RuntimeStage.Diarization),
                context.ModelPreferences.RequiresPreferredExecutionProvider(RuntimeStage.Diarization),
                context.ModelPreferences.GetPreferredModelVariantAlias(RuntimeStage.Diarization),
                cancellationToken).ConfigureAwait(false)
            : null;

        if (diarizationResult is not null)
        {
            PipelineProgressReporter.Phase(
                progress,
                StageName,
                "Persisting speaker turns",
                $"Saving {diarizationResult.Turns.Count} speaker turn(s).");
            await PersistDiarizationArtifactAsync(context, diarizationResult, cancellationToken)
                .ConfigureAwait(false);
        }

        PipelineProgressReporter.Phase(progress, StageName, "Planning regions", "Building transcript region plan.");
        TranscriptRegionPlan regionPlan = TranscriptWorkflowUtilities.BuildTranscriptRegionPlan(
            context.SpeechRegions,
            diarizationResult,
            durationSeconds);

        return context with { DiarizationResult = diarizationResult, RegionPlan = regionPlan };
    }

    private Task PersistDiarizationArtifactAsync(
        TranscriptGenerationContext context,
        DiarizationResult diarizationResult,
        CancellationToken cancellationToken)
    {
        Guid stageRunId = diarizationResult.Turns
            .FirstOrDefault(static turn => turn.StageRunId.HasValue)
            ?.StageRunId ?? Guid.Empty;
        DiarizationArtifactWriteRequest request = new()
        {
            ProjectId = context.Project.Id,
            MediaAsset = context.MediaAsset,
            Speakers = diarizationResult.Speakers,
            Turns = diarizationResult.Turns,
            StageRunId = stageRunId,
            VadStageRunId = context.VadStageRunId
        };
        return artifactWriter.WriteDiarizationArtifactAsync(request, cancellationToken);
    }
}
