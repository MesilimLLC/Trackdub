using Trackdub.Contracts;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Transcripts.Stages;

/// <summary>
/// Pipeline stage that wraps <see cref="SpeechAudioEnhancementStageHandler"/> to enhance speech audio
/// quality (denoising, dereverberation) before VAD and downstream processing.
/// The stage updates the audio routing plan so all subsequent stages use the enhanced audio artifact.
/// </summary>
public sealed class SpeechEnhancementGenerationStage(
    SpeechAudioEnhancementStageHandler enhancementHandler,
    IArtifactStore artifactStore,
    IMediaAssetRepository mediaAssetRepository)
    : ITranscriptGenerationStage
{
    private readonly SpeechAudioEnhancementStageHandler enhancementHandler = enhancementHandler ?? throw new ArgumentNullException(nameof(enhancementHandler));
    private readonly IArtifactStore artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));
    private readonly IMediaAssetRepository mediaAssetRepository = mediaAssetRepository ?? throw new ArgumentNullException(nameof(mediaAssetRepository));

    public string StageName => StageNames.SpeechEnhancement;

    public async Task<TranscriptGenerationContext> ExecuteAsync(
        TranscriptGenerationContext context,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress = null)
    {
        PipelineProgressReporter.Phase(progress, StageName, "Enhancing speech", "Running speech audio enhancement.");

        // Use the VAD audio artifact from the routing plan as the source for enhancement.
        // This is consistent with how ProjectWorkflow feeds normalized/vocal-stem audio into enhancement.
        ProjectArtifact sourceArtifact = context.AudioRoutingPlan.VadAudioArtifact;

        // Fetch existing enhanced audio artifacts for ID reuse (the handler reuses the latest ID).
        IReadOnlyList<ProjectArtifact> existingArtifacts = await mediaAssetRepository
            .GetArtifactsAsync(context.MediaAsset.Id, cancellationToken)
            .ConfigureAwait(false);

        var request = new SpeechAudioEnhancementStageRequest(
            context.Project.Id,
            context.MediaAsset,
            sourceArtifact,
            existingArtifacts);

        SpeechAudioEnhancementStageResult result = await enhancementHandler
            .HandleAsync(request, cancellationToken)
            .ConfigureAwait(false);

        PipelineProgressReporter.Phase(progress, StageName, "Updating routing", "Updating audio routing with enhanced audio.");

        // VAD and diarization use the enhanced audio. ASR uses Studio Voice only when the native
        // AFX effect actually produced it; all fallback paths retain the unprocessed ASR source.
        TranscriptAudioRoutingPlan enhancedRoutingPlan = context.AudioRoutingPlan with
        {
            VadAudioArtifact = result.EnhancedAudioArtifact,
            DiarizationAudioArtifact = result.EnhancedAudioArtifact
        };

        enhancedRoutingPlan = enhancedRoutingPlan.WithSpeechEnhancementAsrSource(
            result.EnhancedAudioArtifact,
            result.Backend,
            result.BackendProfile);

        return context with
        {
            AudioRoutingPlan = enhancedRoutingPlan,
            EnhancedAudioArtifact = result.EnhancedAudioArtifact
        };
    }
}
