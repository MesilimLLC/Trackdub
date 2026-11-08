using Trackdub.Contracts;
using Trackdub.Contracts.Transcripts;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Transcripts.Stages;

public sealed class VadGenerationStage(
    VadStageHandler vadStageHandler,
    TranscriptArtifactWriter artifactWriter,
    IArtifactStore artifactStore)
    : ITranscriptGenerationStage
{
    private readonly VadStageHandler vadStageHandler = vadStageHandler ?? throw new ArgumentNullException(nameof(vadStageHandler));
    private readonly TranscriptArtifactWriter artifactWriter = artifactWriter ?? throw new ArgumentNullException(nameof(artifactWriter));
    private readonly IArtifactStore artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));

    public string StageName => StageNames.Vad;

    public async Task<TranscriptGenerationContext> ExecuteAsync(
        TranscriptGenerationContext context,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress = null)
    {
        PipelineProgressReporter.Phase(progress, StageName, "Validating audio", "Reading speech audio for VAD.");
        ProjectArtifact vadAudioArtifact = context.AudioRoutingPlan.VadAudioArtifact;
        string speechAudioPath = artifactStore.GetPath(vadAudioArtifact.RelativePath);
        AudioArtifactValidator.AudioFileInfo audioInfo =
            await AudioArtifactValidator.ReadAndValidateAsync(speechAudioPath, cancellationToken).ConfigureAwait(false);
        double durationSeconds = audioInfo.DurationSeconds;

        PipelineProgressReporter.Phase(progress, StageName, "Detecting speech", "Running voice activity detection.");
        VadStageResult vadResult = await vadStageHandler.HandleAsync(
            new VadStageRequest(
                context.Project.Id,
                speechAudioPath,
                durationSeconds,
                context.ModelPreferences.VadModelAlias,
                context.ModelPreferences.GetPreferredExecutionProvider(RuntimeStage.Vad),
                context.ModelPreferences.RequiresPreferredExecutionProvider(RuntimeStage.Vad),
                context.ModelPreferences.GetPreferredModelVariantAlias(RuntimeStage.Vad)),
            cancellationToken).ConfigureAwait(false);

        IReadOnlyList<SpeechRegion> regions = vadResult.Regions;

        PipelineProgressReporter.Phase(progress, StageName, "Persisting regions", $"Saving {regions.Count} speech region(s).");
        await artifactWriter.WriteSpeechRegionsArtifactAsync(
            new SpeechRegionsArtifactWriteRequest(
                context.Project,
                context.MediaAsset,
                regions,
                vadResult.StageRun,
                context.AudioRoutingPlan.VadAudioArtifact),
            cancellationToken).ConfigureAwait(false);

        return context with { SpeechRegions = regions, VadStageRunId = vadResult.StageRun.Id };
    }
}
