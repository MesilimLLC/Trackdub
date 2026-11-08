using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Transcripts;

internal static class TranscriptResumeLineageEvaluator
{
    private static readonly HashSet<string> VadDependentStages = new(StringComparer.OrdinalIgnoreCase)
    {
        StageNames.Diarization,
        StageNames.Asr,
        StageNames.TextRefinementAsr,
        StageNames.SpeakerAssignment
    };

    private static readonly HashSet<string> RawAsrDependentStages = new(StringComparer.OrdinalIgnoreCase)
    {
        StageNames.Asr,
        StageNames.TextRefinementAsr,
        StageNames.SpeakerAssignment
    };

    public static bool IsVadDependentStage(string stageName) => VadDependentStages.Contains(stageName);

    public static bool RequiresRawAsrLineage(string stageName) => RawAsrDependentStages.Contains(stageName);

    public static bool HasCurrentVadInputsForResume(
        TranscriptGenerationContext context,
        string stageName)
    {
        if (string.Equals(stageName, StageNames.Vad, StringComparison.OrdinalIgnoreCase))
        {
            return SpeechRegionsMatchCurrentAudio(
                context.ProjectState,
                context.AudioRoutingPlan.VadAudioArtifact);
        }

        if (!IsVadDependentStage(stageName))
        {
            return true;
        }

        if (!HasCurrentVadRunForStage(
                context.ProjectState,
                context.VadStageRunId,
                context.SpeechRegions,
                stageName))
        {
            return false;
        }

        return HasCurrentRawAsrInputsIfRequired(context, stageName);
    }

    private static bool HasCurrentRawAsrInputsIfRequired(
        TranscriptGenerationContext context,
        string stageName)
    {
        if (!RequiresRawAsrLineage(stageName))
        {
            return true;
        }

        return HasCurrentRawAsrArtifact(
            context.ProjectState,
            context.AudioRoutingPlan,
            context.VadStageRunId);
    }

    public static bool SpeechRegionsMatchCurrentAudio(
        TranscriptProjectState? state,
        ProjectArtifact currentVadAudioArtifact)
    {
        if (state is null)
        {
            return false;
        }

        ProjectArtifact? latestRegions = state.ProjectState.Artifacts
            .Where(static artifact => artifact.Kind == ArtifactKind.SpeechRegions)
            .OrderByDescending(static artifact => artifact.CreatedAtUtc)
            .FirstOrDefault();
        StageRunRecord? latestVadRun = StageArtifactResumeEvaluator.GetLatestSuccessfulRun(
            state.StageRuns,
            StageNames.Vad);
        if (latestVadRun is null || latestRegions?.StageRunId != latestVadRun.Id)
        {
            return false;
        }

        return string.Equals(
            latestRegions.Provenance,
            TranscriptArtifactWriter.GetSpeechRegionsProvenance(currentVadAudioArtifact),
            StringComparison.Ordinal);
    }

    public static bool HasCurrentVadRunForStage(
        TranscriptProjectState? state,
        Guid? currentVadStageRunId,
        IReadOnlyList<SpeechRegion> speechRegions,
        string stageName)
    {
        if (state is null)
        {
            return false;
        }

        if (currentVadStageRunId is not Guid currentVadRunId)
        {
            return false;
        }

        if (speechRegions.Count == 0)
        {
            return false;
        }

        StageRunRecord? latestVadRun = StageArtifactResumeEvaluator.GetLatestSuccessfulRun(
            state.StageRuns,
            StageNames.Vad);
        if (latestVadRun?.Id != currentVadRunId)
        {
            return false;
        }

        return DependentStageStartedAfterVad(state, stageName, latestVadRun);
    }

    public static bool DependentStageStartedAfterVad(
        TranscriptProjectState state,
        string stageName,
        StageRunRecord? latestVadRun)
    {
        if (latestVadRun?.CompletedAtUtc is not DateTimeOffset vadCompletedAt)
        {
            return false;
        }

        StageRunRecord? latestStageRun = StageArtifactResumeEvaluator.GetLatestSuccessfulRun(
            state.StageRuns,
            stageName);
        if (latestStageRun is null)
        {
            return false;
        }

        return latestStageRun.StartedAtUtc >= vadCompletedAt;
    }

    public static bool HasCurrentRawAsrArtifact(
        TranscriptProjectState? state,
        TranscriptAudioRoutingPlan audioRoutingPlan,
        Guid? vadStageRunId)
    {
        if (state is null)
        {
            return false;
        }

        if (vadStageRunId is not Guid currentVadStageRunId)
        {
            return false;
        }

        StageRunRecord? latestAsrRun = StageArtifactResumeEvaluator.GetLatestSuccessfulRun(
            state.StageRuns,
            StageNames.Asr);
        if (latestAsrRun is null)
        {
            return false;
        }

        RawAsrTranscriptArtifactLineage lineage = RawAsrTranscriptArtifactLineage.Create(
            audioRoutingPlan.AsrAudioArtifact,
            audioRoutingPlan.VadAudioArtifact,
            currentVadStageRunId);
        string expectedProvenance = TranscriptArtifactWriter.GetRawAsrProvenance(lineage);
        return state.ProjectState.Artifacts.Any(artifact =>
            IsCurrentRawAsrArtifact(artifact, latestAsrRun.Id, expectedProvenance));
    }

    private static bool IsCurrentRawAsrArtifact(
        ProjectArtifact artifact,
        Guid latestAsrRunId,
        string expectedProvenance) =>
        artifact.Kind == ArtifactKind.TranscriptRevision &&
        artifact.StageRunId == latestAsrRunId &&
        string.Equals(artifact.Provenance, expectedProvenance, StringComparison.Ordinal);
}
