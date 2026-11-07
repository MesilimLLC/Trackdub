using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Application.Projects;
using Trackdub.Contracts.Projects;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;

namespace Trackdub.Application.Transcripts.Pipeline;

internal static class TranscriptPipelineResumeHydrator
{
    public static async Task<TranscriptGenerationContext> HydrateSkippedStageAsync(
        TranscriptGenerationContext context,
        string stageName,
        TranscriptArtifactWriter artifactWriter,
        IArtifactStore artifactStore,
        ITranscriptRepository transcriptRepository,
        CancellationToken cancellationToken)
    {
        if (string.Equals(stageName, StageNames.SpeechEnhancement, StringComparison.OrdinalIgnoreCase))
        {
            return HydrateSpeechEnhancement(context, context.ProjectStateArtifacts());
        }

        if (string.Equals(stageName, StageNames.Vad, StringComparison.OrdinalIgnoreCase))
        {
            return await HydrateVadAsync(context, artifactWriter, cancellationToken).ConfigureAwait(false);
        }

        if (string.Equals(stageName, StageNames.Diarization, StringComparison.OrdinalIgnoreCase))
        {
            return await HydrateDiarizationAsync(context, artifactWriter, cancellationToken).ConfigureAwait(false);
        }

        if (string.Equals(stageName, StageNames.Asr, StringComparison.OrdinalIgnoreCase))
        {
            return await HydrateAsrAsync(context, artifactWriter, cancellationToken).ConfigureAwait(false);
        }

        if (string.Equals(stageName, StageNames.TextRefinementAsr, StringComparison.OrdinalIgnoreCase))
        {
            return await HydrateTextRefinementAsrAsync(
                    context,
                    artifactWriter,
                    artifactStore,
                    transcriptRepository,
                    cancellationToken)
                .ConfigureAwait(false);
        }

        if (string.Equals(stageName, StageNames.SpeakerAssignment, StringComparison.OrdinalIgnoreCase))
        {
            return await HydrateSpeakerAssignmentAsync(
                    context,
                    artifactWriter,
                    artifactStore,
                    transcriptRepository,
                    cancellationToken)
                .ConfigureAwait(false);
        }

        return context;
    }

    private static TranscriptGenerationContext HydrateSpeechEnhancement(
        TranscriptGenerationContext context,
        IReadOnlyList<ProjectArtifact> artifacts)
    {
        ProjectArtifact? enhancedArtifact = TranscriptWorkflowUtilities.GetLatestArtifactByKind(
            artifacts,
            ArtifactKind.SpeechEnhancedAudio);
        if (enhancedArtifact is null)
        {
            return context;
        }

        // Accept the normalized audio and the pre-enhancement VAD source as valid lineage; the
        // enhancement stage consumes one of those, and a stale artifact from a changed source
        // must not reach ASR.
        Guid[] sourceArtifactIds =
        [
            context.NormalizedAudioArtifact.Id,
            context.AudioRoutingPlan.VadAudioArtifact.Id
        ];

        // A Studio Voice artifact whose persisted source no longer matches the current source
        // must not feed any stage; leave the original routing so the run uses the current audio.
        if (TranscriptAudioRoutingPlan.IsStudioVoiceEnhancementArtifact(enhancedArtifact)
            && !TranscriptAudioRoutingPlan.IsStudioVoiceEnhancementArtifact(enhancedArtifact, sourceArtifactIds))
        {
            return context;
        }

        TranscriptAudioRoutingPlan enhancedRoutingPlan = context.AudioRoutingPlan with
        {
            VadAudioArtifact = enhancedArtifact,
            DiarizationAudioArtifact = enhancedArtifact
        };
        enhancedRoutingPlan = enhancedRoutingPlan.WithStudioVoiceAsrSourceIfApplicable(
            enhancedArtifact,
            sourceArtifactIds);

        return context with
        {
            AudioRoutingPlan = enhancedRoutingPlan,
            EnhancedAudioArtifact = enhancedArtifact
        };
    }

    private static async Task<TranscriptGenerationContext> HydrateVadAsync(
        TranscriptGenerationContext context,
        TranscriptArtifactWriter artifactWriter,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<SpeechRegion>? speechRegions = await artifactWriter
            .TryReadSpeechRegionsAsync(
                context.AudioRoutingPlan.VadAudioArtifact,
                cancellationToken)
            .ConfigureAwait(false);
        if (speechRegions is null)
        {
            return context;
        }

        SpeechRegion[] regions = speechRegions.OrderBy(static region => region.Index).ToArray();
        Guid? vadStageRunId = context.ProjectState?.StageRuns
            .Where(run => string.Equals(run.StageName, StageNames.Vad, StringComparison.OrdinalIgnoreCase))
            .Where(run => run.Status is StageRunStatus.Completed or StageRunStatus.PartiallyCompleted)
            .OrderByDescending(static run => run.CompletedAtUtc ?? run.StartedAtUtc)
            .Select(static run => (Guid?)run.Id)
            .FirstOrDefault();

        return context with
        {
            SpeechRegions = regions,
            VadStageRunId = vadStageRunId
        };
    }

    private static async Task<TranscriptGenerationContext> HydrateDiarizationAsync(
        TranscriptGenerationContext context,
        TranscriptArtifactWriter artifactWriter,
        CancellationToken cancellationToken)
    {
        context = await EnsureSpeechRegionsAsync(context, artifactWriter, cancellationToken).ConfigureAwait(false);

        DiarizationResult? diarizationResult = await artifactWriter
            .TryReadDiarizationResultAsync(context.Project.Id, cancellationToken)
            .ConfigureAwait(false);
        double durationSeconds = context.AudioRoutingPlan.DiarizationAudioArtifact.DurationSeconds
                                 ?? context.NormalizedAudioArtifact.DurationSeconds
                                 ?? context.MediaAsset.DurationSeconds;
        TranscriptRegionPlan regionPlan = TranscriptWorkflowUtilities.BuildTranscriptRegionPlan(
            context.SpeechRegions,
            diarizationResult,
            durationSeconds);

        return context with
        {
            DiarizationResult = diarizationResult,
            RegionPlan = regionPlan
        };
    }

    private static async Task<TranscriptGenerationContext> HydrateAsrAsync(
        TranscriptGenerationContext context,
        TranscriptArtifactWriter artifactWriter,
        CancellationToken cancellationToken)
    {
        context = await HydrateDiarizationAsync(context, artifactWriter, cancellationToken).ConfigureAwait(false);

        RawAsrTranscriptArtifact? rawAsr = await artifactWriter
            .TryReadRawAsrTranscriptAsync(context.Project.Id, cancellationToken)
            .ConfigureAwait(false);
        if (rawAsr is null || rawAsr.Segments.Count == 0)
        {
            return context;
        }

        DateTimeOffset now = DateTimeOffset.UtcNow;
        var asrStageRun = new StageRunRecord(
            rawAsr.StageRunId,
            context.Project.Id,
            StageNames.Asr,
            StageRunStatus.Completed,
            now,
            now,
            FailureReason: null);

        return context with { AsrResult = new AsrStageResult(asrStageRun, rawAsr.Segments) };
    }

    private static async Task<TranscriptGenerationContext> HydrateTextRefinementAsrAsync(
        TranscriptGenerationContext context,
        TranscriptArtifactWriter artifactWriter,
        IArtifactStore artifactStore,
        ITranscriptRepository transcriptRepository,
        CancellationToken cancellationToken)
    {
        context = await HydrateAsrAsync(context, artifactWriter, cancellationToken).ConfigureAwait(false);
        if (context.AsrResult is null)
        {
            return context;
        }

        TranscriptRevision? revision = await transcriptRepository
            .GetCurrentRevisionAsync(context.Project.Id, cancellationToken)
            .ConfigureAwait(false);
        if (revision is null)
        {
            return context;
        }

        string provenanceRelativePath = ProjectArtifactPaths.GetTextRefinementProvenanceRelativePath(revision.Id);
        TextRefinementProvenanceArtifactDocument? provenance = await artifactStore
            .ReadJsonAsync<TextRefinementProvenanceArtifactDocument>(provenanceRelativePath, cancellationToken)
            .ConfigureAwait(false);
        if (provenance?.TextRefinementStageRunId is not Guid refinementStageRunId)
        {
            return context;
        }

        StageRunRecord? refinementStageRun = context.ProjectState?.StageRuns
            .FirstOrDefault(run => run.Id == refinementStageRunId);
        if (refinementStageRun is null)
        {
            DateTimeOffset now = DateTimeOffset.UtcNow;
            refinementStageRun = new StageRunRecord(
                refinementStageRunId,
                context.Project.Id,
                StageNames.TextRefinementAsr,
                StageRunStatus.Completed,
                now,
                now,
                FailureReason: null);
        }

        RefinedTextSegment[] refinedSegments = context.AsrResult.Segments
            .OrderBy(static segment => segment.Index)
            .Select(asrSegment =>
            {
                TranscriptSegmentTextProvenance? provenanceSegment = provenance.Segments
                    .FirstOrDefault(segment => segment.SegmentIndex == asrSegment.Index);
                if (provenanceSegment is null)
                {
                    return new RefinedTextSegment(
                        asrSegment.Index,
                        asrSegment.StartSeconds,
                        asrSegment.EndSeconds,
                        asrSegment.Text,
                        asrSegment.Text,
                        asrSegment.Text,
                        Accepted: false,
                        TextRefinementGuardStatus.Unchanged,
                        AppliedCorrections: []);
                }

                return new RefinedTextSegment(
                    asrSegment.Index,
                    asrSegment.StartSeconds,
                    asrSegment.EndSeconds,
                    provenanceSegment.OriginalText,
                    provenanceSegment.RefinedText ?? provenanceSegment.DisplayedText,
                    provenanceSegment.DisplayedText,
                    provenanceSegment.Accepted,
                    TextRefinementGuardStatus.Unchanged,
                    provenanceSegment.AppliedCorrections);
            })
            .ToArray();

        return context with
        {
            TextRefinementResult = new TextRefinementStageResult(
                refinementStageRun,
                TextRefinementScope.Asr,
                refinedSegments)
        };
    }

    private static async Task<TranscriptGenerationContext> HydrateSpeakerAssignmentAsync(
        TranscriptGenerationContext context,
        TranscriptArtifactWriter artifactWriter,
        IArtifactStore artifactStore,
        ITranscriptRepository transcriptRepository,
        CancellationToken cancellationToken)
    {
        context = await HydrateTextRefinementAsrAsync(
                context,
                artifactWriter,
                artifactStore,
                transcriptRepository,
                cancellationToken)
            .ConfigureAwait(false);
        if (context.AsrResult is null)
        {
            context = await HydrateAsrAsync(context, artifactWriter, cancellationToken).ConfigureAwait(false);
        }

        if (context.AsrResult is null || context.RegionPlan is null)
        {
            return context;
        }

        IReadOnlyList<TranscriptSegment> segments = context.ProjectState?.TranscriptSegments ?? [];
        if (segments.Count == 0)
        {
            TranscriptRevision? revision = await transcriptRepository
                .GetCurrentRevisionAsync(context.Project.Id, cancellationToken)
                .ConfigureAwait(false);
            if (revision is null)
            {
                return context;
            }

            segments = await transcriptRepository
                .GetSegmentsAsync(revision.Id, cancellationToken)
                .ConfigureAwait(false);
        }

        if (segments.Count == 0)
        {
            return context;
        }

        IReadOnlyList<ProjectSpeaker> speakers = context.DiarizationResult?.Speakers
                                                  ?? context.ProjectState?.Speakers
                                                  ?? [];
        IReadOnlyList<SpeakerTurn> turns = context.DiarizationResult?.Turns
                                             ?? context.ProjectState?.SpeakerTurns
                                             ?? [];
        Dictionary<int, Guid> segmentSpeakerIdsByIndex = segments
            .Where(segment => segment.SpeakerId is Guid speakerId && speakerId != Guid.Empty)
            .ToDictionary(segment => segment.SegmentIndex, segment => segment.SpeakerId!.Value);

        return context with
        {
            SpeakerAssignment = new SpeakerAssignmentResult(
                speakers,
                turns,
                segmentSpeakerIdsByIndex)
        };
    }

    private static async Task<TranscriptGenerationContext> EnsureSpeechRegionsAsync(
        TranscriptGenerationContext context,
        TranscriptArtifactWriter artifactWriter,
        CancellationToken cancellationToken)
    {
        if (context.SpeechRegions.Count > 0)
        {
            return context;
        }

        return await HydrateVadAsync(context, artifactWriter, cancellationToken).ConfigureAwait(false);
    }

    private static IReadOnlyList<ProjectArtifact> ProjectStateArtifacts(this TranscriptGenerationContext context) =>
        context.ProjectState?.ProjectState.Artifacts ?? [];
}
