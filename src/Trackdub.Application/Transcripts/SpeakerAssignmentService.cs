using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Projects;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;

namespace Trackdub.Application.Transcripts;

public sealed class SpeakerAssignmentService(
    ISpeakerRepository speakerRepository,
    ITranscriptRepository transcriptRepository,
    SegmentEditingService segmentEditingService,
    IArtifactStore artifactStore,
    IProjectStageRunStore stageRunStore,
    ISpeakerDiarizationEngine diarizationEngine,
    SpeakerReferenceClipService referenceClipService,
    TranscriptArtifactWriter transcriptArtifactWriter,
    DiarizationStageHandler diarizationStageHandler,
    IApplicationLogger? logger = null,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null,
    PipelineDegradationWriter? degradationWriter = null)
{
    private readonly ISpeakerRepository speakerRepository = speakerRepository ?? throw new ArgumentNullException(nameof(speakerRepository));
    private readonly SegmentEditingService segmentEditingService = segmentEditingService ?? throw new ArgumentNullException(nameof(segmentEditingService));
    private readonly IArtifactStore artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));
    private readonly IProjectStageRunStore stageRunStore = stageRunStore ?? throw new ArgumentNullException(nameof(stageRunStore));
    private readonly ISpeakerDiarizationEngine diarizationEngine = diarizationEngine ?? throw new ArgumentNullException(nameof(diarizationEngine));
    private readonly SpeakerReferenceClipService referenceClipService = referenceClipService ?? throw new ArgumentNullException(nameof(referenceClipService));
    private readonly TranscriptArtifactWriter transcriptArtifactWriter = transcriptArtifactWriter ?? throw new ArgumentNullException(nameof(transcriptArtifactWriter));
    private readonly DiarizationStageHandler diarizationStageHandler = diarizationStageHandler ?? throw new ArgumentNullException(nameof(diarizationStageHandler));
    private readonly RenameSpeakerHandler renameSpeakerHandler = new(speakerRepository);
    private readonly MergeSpeakersHandler mergeSpeakersHandler = new(transcriptRepository ?? throw new ArgumentNullException(nameof(transcriptRepository)));

    public async Task RenameSpeakerAsync(
        TranscriptProjectState currentState,
        RenameSpeakerRequest request,
        CancellationToken cancellationToken)
    {
        await renameSpeakerHandler.HandleAsync(
            currentState.ProjectState.Project.Id,
            request.SpeakerId,
            request.DisplayName,
            cancellationToken).ConfigureAwait(false);
    }

    public async Task MergeSpeakersAsync(
        TranscriptProjectState currentState,
        MergeSpeakersRequest request,
        CancellationToken cancellationToken)
    {
        await mergeSpeakersHandler.HandleAsync(
            currentState.ProjectState.Project.Id,
            request.SourceSpeakerId,
            request.TargetSpeakerId,
            cancellationToken).ConfigureAwait(false);
    }

    public Task<IReadOnlyList<MergedSpeakerEntry>> ListMergedSpeakersAsync(
        TranscriptProjectState currentState,
        CancellationToken cancellationToken) =>
        speakerRepository.ListMergedSpeakersAsync(
            currentState.ProjectState.Project.Id,
            cancellationToken);

    public Task DeleteEmptySpeakerAsync(
        TranscriptProjectState currentState,
        DeleteEmptySpeakerRequest request,
        CancellationToken cancellationToken) =>
        speakerRepository.DeleteEmptySpeakerAsync(
            currentState.ProjectState.Project.Id,
            request.SpeakerId,
            cancellationToken);

    public Task AssignSpeakerToSegmentAsync(
        TranscriptProjectState currentState,
        AssignSpeakerToSegmentRequest request,
        CancellationToken cancellationToken) =>
        AssignSpeakerToSegmentsAsync(
            currentState,
            new AssignSpeakerToSegmentsRequest(
                request.TranscriptRevisionId,
                [request.SegmentId],
                request.SpeakerId),
            cancellationToken);

    public async Task CreateSpeakerFromSegmentsAsync(
        TranscriptProjectState currentState,
        CreateSpeakerFromSegmentsRequest request,
        CancellationToken cancellationToken)
    {
        TranscriptRevision currentRevision = TranscriptWorkflowUtilities.GetRequiredTranscriptRevision(currentState);
        TranscriptWorkflowUtilities.EnsureRevisionMatches(
            currentRevision,
            request.TranscriptRevisionId,
            "Speaker assignment was based on an out-of-date transcript revision.");

        Guid[] requestedSegmentIds = request.SegmentIds.Distinct().ToArray();
        if (requestedSegmentIds.Length == 0)
        {
            throw new InvalidOperationException("Select at least one segment before creating a speaker.");
        }

        HashSet<Guid> requestedIds = requestedSegmentIds.ToHashSet();
        int matchedCount = currentState.TranscriptSegments.Count(segment => requestedIds.Contains(segment.Id));
        if (matchedCount != requestedSegmentIds.Length)
        {
            throw new InvalidOperationException("One or more selected segments were not found in the current transcript revision.");
        }

        ProjectSpeaker speaker = await speakerRepository
            .CreateSpeakerAsync(currentState.ProjectState.Project.Id, cancellationToken)
            .ConfigureAwait(false);
        TranscriptProjectState stateWithSpeaker = currentState with
        {
            Speakers = currentState.Speakers.Append(speaker).ToArray()
        };

        await AssignSpeakerToSegmentsAsync(
            stateWithSpeaker,
            new AssignSpeakerToSegmentsRequest(
                request.TranscriptRevisionId,
                requestedSegmentIds,
                speaker.Id),
            cancellationToken).ConfigureAwait(false);
    }

    public Task AssignSpeakerToSegmentsAsync(
        TranscriptProjectState currentState,
        AssignSpeakerToSegmentsRequest request,
        CancellationToken cancellationToken)
    {
        TranscriptRevision currentRevision = TranscriptWorkflowUtilities.GetRequiredTranscriptRevision(currentState);
        TranscriptWorkflowUtilities.EnsureRevisionMatches(
            currentRevision,
            request.TranscriptRevisionId,
            "Speaker assignment was based on an out-of-date transcript revision.");
        if (!currentState.Speakers.Any(speaker => speaker.Id == request.SpeakerId))
        {
            throw new InvalidOperationException("The selected speaker was not found.");
        }

        Guid[] requestedSegmentIds = request.SegmentIds.Distinct().ToArray();
        if (requestedSegmentIds.Length == 0)
        {
            throw new InvalidOperationException("Select at least one segment before assigning a speaker.");
        }

        HashSet<Guid> requestedIds = requestedSegmentIds.ToHashSet();
        int matchedCount = currentState.TranscriptSegments.Count(segment => requestedIds.Contains(segment.Id));
        if (matchedCount != requestedSegmentIds.Length)
        {
            throw new InvalidOperationException("One or more selected segments were not found in the current transcript revision.");
        }

        TranscriptSegment[] revisedSegments = currentState.TranscriptSegments
            .OrderBy(segment => segment.SegmentIndex)
            .Select((segment, index) => TranscriptSegment.Create(
                currentRevision.Id,
                index,
                segment.StartSeconds,
                segment.EndSeconds,
                segment.Text,
                requestedIds.Contains(segment.Id)
                    ? request.SpeakerId
                    : segment.SpeakerId,
                segment.DetectedLanguage,
                TranscriptWorkflowUtilities.CloneWords(segment.Words),
                requestedIds.Contains(segment.Id)
                    ? SpeakerAssignmentOrigin.Manual
                    : segment.SpeakerAssignmentOrigin))
            .ToArray();

        return segmentEditingService.SaveTranscriptRevisionAsync(
            currentState,
            revisedSegments,
            "speaker-assignment",
            cancellationToken);
    }

    public async Task SplitSpeakerTurnAsync(
        TranscriptProjectState currentState,
        SplitSpeakerTurnRequest request,
        CancellationToken cancellationToken)
    {
        await speakerRepository.SplitTurnAsync(
            currentState.ProjectState.Project.Id,
            request.SpeakerTurnId,
            request.SplitSeconds,
            cancellationToken).ConfigureAwait(false);
    }

    public Task ExtractReferenceClipAsync(
        TranscriptProjectState currentState,
        ExtractReferenceClipRequest request,
        CancellationToken cancellationToken) =>
        referenceClipService.ExtractReferenceClipAsync(currentState, request, cancellationToken);

    public Task ImportReferenceClipAsync(
        TranscriptProjectState currentState,
        ImportReferenceClipRequest request,
        CancellationToken cancellationToken) =>
        referenceClipService.ImportReferenceClipAsync(currentState, request, cancellationToken);

    public async Task RerunDiarizationAsync(
        TranscriptProjectState currentState,
        RerunDiarizationRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(currentState);
        ArgumentNullException.ThrowIfNull(request);

        TranscriptRevision revision = TranscriptWorkflowUtilities.GetRequiredTranscriptRevision(currentState);
        TranscriptSegment[] segments = currentState.TranscriptSegments
            .OrderBy(static segment => segment.SegmentIndex)
            .ToArray();
        if (segments.Length == 0)
        {
            throw new InvalidOperationException("Transcript segments are required before speaker identification can run.");
        }

        MediaAsset mediaAsset = TranscriptWorkflowUtilities.GetRequiredMediaAsset(currentState);
        ProjectArtifact diarizationAudio = TranscriptWorkflowUtilities.ResolveAsrAudioArtifact(
                currentState.ProjectState.Artifacts,
                currentState.StageRuns)
            ?? throw new InvalidOperationException("The project does not contain audio for speaker diarization.");
        string speechAudioPath = artifactStore.GetPath(diarizationAudio.RelativePath);
        double durationSeconds = diarizationAudio.DurationSeconds ?? mediaAsset.DurationSeconds;
        if (durationSeconds <= 0d)
        {
            durationSeconds = segments.Max(static segment => segment.EndSeconds);
        }
        Guid projectId = currentState.ProjectState.Project.Id;
        IReadOnlyList<SpeechRegion> speechRegions = await ResolveSpeechRegionsForDiarizationAsync(
                projectId,
                segments,
                cancellationToken)
            .ConfigureAwait(false);
        DiarizationResult? diarizationResult = await CreateDiarizationAsync(
            projectId,
            mediaAsset.Id,
            speechAudioPath,
            durationSeconds,
            speechRegions,
            request.PreferredModelAlias,
            request.PreferredExecutionProvider,
            request.RequirePreferredExecutionProvider,
            request.PreferredModelVariantAlias,
            cancellationToken).ConfigureAwait(false);
        if (diarizationResult is null)
        {
            throw new InvalidOperationException("Speaker diarization did not produce any speakers.");
        }

        RecognizedTranscriptSegment[] recognizedSegments = segments
            .Select(static segment => new RecognizedTranscriptSegment(
                segment.SegmentIndex,
                segment.StartSeconds,
                segment.EndSeconds,
                segment.Text,
                segment.DetectedLanguage))
            .ToArray();
        Dictionary<int, Guid> speakerIdsBySegmentIndex = AssignTranscriptSegmentsToSpeakers(
            recognizedSegments,
            diarizationResult.Speakers,
            diarizationResult.Turns);

        TranscriptSegment[] revisedSegments = segments
            .Select((segment, index) => TranscriptSegment.Create(
                revision.Id,
                index,
                segment.StartSeconds,
                segment.EndSeconds,
                segment.Text,
                (segment.SpeakerAssignmentOrigin == SpeakerAssignmentOrigin.Automatic ||
                    (segment.SpeakerAssignmentOrigin == SpeakerAssignmentOrigin.Unknown && segment.SpeakerId is null)) &&
                    speakerIdsBySegmentIndex.TryGetValue(segment.SegmentIndex, out Guid speakerId)
                    ? speakerId
                    : segment.SpeakerId,
                segment.DetectedLanguage,
                TranscriptWorkflowUtilities.CloneWords(segment.Words),
                segment.SpeakerAssignmentOrigin == SpeakerAssignmentOrigin.Unknown &&
                    segment.SpeakerId is null && speakerIdsBySegmentIndex.ContainsKey(segment.SegmentIndex)
                    ? SpeakerAssignmentOrigin.Automatic
                    : segment.SpeakerAssignmentOrigin))
            .ToArray();

        await segmentEditingService.SaveTranscriptRevisionAsync(
            currentState,
            revisedSegments,
            "rerun-diarization",
            cancellationToken).ConfigureAwait(false);

        Guid stageRunId = GetDiarizationStageRunId(diarizationResult);
        await transcriptArtifactWriter.WriteDiarizationArtifactAsync(
            CreateDiarizationArtifactWriteRequest(currentState, mediaAsset, diarizationResult, stageRunId),
            cancellationToken).ConfigureAwait(false);
    }

    private static Guid GetDiarizationStageRunId(DiarizationResult result)
    {
        // Guid.Empty means no turn has a persisted stage run; do not invent an orphan id.
        return result.Turns.FirstOrDefault(static turn => turn.StageRunId.HasValue)?.StageRunId ?? Guid.Empty;
    }

    private static DiarizationArtifactWriteRequest CreateDiarizationArtifactWriteRequest(
        TranscriptProjectState currentState,
        MediaAsset mediaAsset,
        DiarizationResult result,
        Guid stageRunId)
    {
        Guid? vadStageRunId = StageArtifactResumeEvaluator.GetLatestSuccessfulRun(
            currentState.StageRuns,
            StageNames.Vad)?.Id;
        return new DiarizationArtifactWriteRequest
        {
            ProjectId = currentState.ProjectState.Project.Id,
            MediaAsset = mediaAsset,
            Speakers = result.Speakers,
            Turns = result.Turns,
            StageRunId = stageRunId,
            VadStageRunId = vadStageRunId
        };
    }

    public async Task<DiarizationResult?> CreateDiarizationAsync(
        Guid projectId,
        Guid mediaAssetId,
        string normalizedAudioPath,
        double durationSeconds,
        IReadOnlyList<SpeechRegion> regions,
        string? preferredModelAlias,
        ExecutionProviderKind? preferredExecutionProvider = null,
        bool requirePreferredExecutionProvider = false,
        string? preferredModelVariantAlias = null,
        CancellationToken cancellationToken = default)
    {
        StageRunRecord stageRun = await StageRunHelper
            .StartAsync(stageRunStore, projectId, StageNames.Diarization, cancellationToken)
            .ConfigureAwait(false);
        object diarizationRuntimeReporter = diarizationStageHandler;

        try
        {
            IReadOnlyList<DiarizedSpeakerTurn> diarizedTurns = await diarizationStageHandler.DiarizeAsync(
                    normalizedAudioPath,
                    durationSeconds,
                    regions,
                    preferredModelAlias,
                    preferredExecutionProvider,
                    requirePreferredExecutionProvider,
                    preferredModelVariantAlias,
                    cancellationToken: cancellationToken).ConfigureAwait(false);

            if (diarizedTurns.Count == 0)
            {
                await StageRunHelper
                    .CompleteAsync(stageRunStore, stageRun, diarizationRuntimeReporter, cancellationToken, runtimePlanningPreferences)
                    .ConfigureAwait(false);
                return null;
            }

            DateTimeOffset now = DateTimeOffset.UtcNow;
            HashSet<string> reservedSpeakerNames = (await speakerRepository
                .ListSpeakersAsync(projectId, cancellationToken)
                .ConfigureAwait(false))
                .Select(static speaker => speaker.DisplayName)
                .ToHashSet(StringComparer.OrdinalIgnoreCase);
            var speakerData = diarizedTurns
                .OrderBy(turn => turn.StartSeconds)
                .GroupBy(turn => turn.NormalizedSpeakerKey, StringComparer.OrdinalIgnoreCase)
                .Select((group, index) =>
                {
                    string displayName = BuildNextDiarizedSpeakerDisplayName(reservedSpeakerNames);
                    reservedSpeakerNames.Add(displayName);
                    ProjectSpeaker speaker = ProjectSpeaker.Create(projectId, displayName, now.AddMilliseconds(index));
                    return new { SpeakerKey = group.Key, Speaker = speaker, SpeakerId = speaker.Id };
                })
                .ToArray();

            ProjectSpeaker[] speakers = speakerData.Select(entry => entry.Speaker).ToArray();
            Dictionary<string, Guid> speakerIdsByKey = speakerData.ToDictionary(
                entry => entry.SpeakerKey,
                entry => entry.SpeakerId,
                StringComparer.OrdinalIgnoreCase);

            SpeakerTurn[] turns = diarizedTurns
                .OrderBy(turn => turn.StartSeconds)
                .Select(turn => SpeakerTurn.Create(
                    projectId,
                    speakerIdsByKey[turn.NormalizedSpeakerKey],
                    turn.StartSeconds,
                    turn.EndSeconds,
                    turn.Confidence,
                    turn.HasOverlap,
                    stageRun.Id))
                .ToArray();

            await speakerRepository.ReplaceDiarizationAsync(
                projectId,
                speakers,
                turns,
                cancellationToken).ConfigureAwait(false);

            await StageRunHelper
                .CompleteAsync(stageRunStore, stageRun, diarizationRuntimeReporter, cancellationToken, runtimePlanningPreferences)
                .ConfigureAwait(false);

            return new DiarizationResult(speakers, turns);
        }
        catch (Exception ex)
        {
            if (ex is OperationCanceledException or TaskCanceledException)
            {
                await StageRunHelper
                    .CancelAsync(stageRunStore, stageRun, diarizationRuntimeReporter, "Diarization canceled.", CancellationToken.None, runtimePlanningPreferences, logger)
                    .ConfigureAwait(false);
                throw;
            }

            if (ex is RequiredModelNotAvailableException modelEx && degradationWriter is not null)
            {
                try
                {
                    await degradationWriter.WriteAsync(
                        new PipelineDegradationRecord(
                            StageNames.Diarization,
                            "DIARIZATION_MODEL_UNAVAILABLE",
                            $"Required diarization model not available: {modelEx.Message}",
                            Detail: $"Model: {modelEx.ModelId}, Path: {modelEx.ModelPath}",
                            SelectedFallback: null,
                            RecommendedAction: modelEx.CanAutoDownload ? "Download model from Model Manager" : "Install model manually",
                            DateTimeOffset.UtcNow,
                            stageRun.Id),
                        projectId,
                        mediaAssetId,
                        CancellationToken.None).ConfigureAwait(false);
                }
                catch (Exception degradationEx) when (degradationEx is not OperationCanceledException)
                {
                    logger?.LogWarning("Failed to write degradation record for diarization model unavailable.", degradationEx);
                }
            }

            await StageRunHelper
                .FailAsync(stageRunStore, stageRun, diarizationRuntimeReporter, ex.Message, cancellationToken, runtimePlanningPreferences, logger)
                .ConfigureAwait(false);

            return null;
        }
    }

    public async Task<SpeakerAssignmentResult> CreateDefaultSpeakerAssignmentAsync(
        Guid projectId,
        IReadOnlyList<RecognizedTranscriptSegment> recognizedSegments,
        CancellationToken cancellationToken)
    {
        ProjectSpeaker defaultSpeaker = await speakerRepository.EnsureDefaultSpeakerAsync(projectId, cancellationToken).ConfigureAwait(false);
        return CreateDefaultSpeakerAssignment(defaultSpeaker, recognizedSegments);
    }

    public static SpeakerAssignmentResult CreateDefaultSpeakerAssignment(
        ProjectSpeaker speaker,
        IReadOnlyList<RecognizedTranscriptSegment> recognizedSegments)
    {
        Dictionary<int, Guid> segmentSpeakerIdsByIndex = recognizedSegments
            .OrderBy(segment => segment.Index)
            .ToDictionary(segment => segment.Index, _ => speaker.Id);
        return new SpeakerAssignmentResult([speaker], [], segmentSpeakerIdsByIndex);
    }

    public static Dictionary<int, Guid> AssignTranscriptSegmentsToSpeakers(
        IReadOnlyList<RecognizedTranscriptSegment> recognizedSegments,
        IReadOnlyList<ProjectSpeaker> speakers,
        IReadOnlyList<SpeakerTurn> turns)
    {
        Guid? fallbackSpeakerId = speakers.OrderBy(speaker => speaker.CreatedAtUtc).FirstOrDefault()?.Id;
        var speakerIdsBySegmentIndex = new Dictionary<int, Guid>();

        // Sort turns by StartSeconds once so we can advance a cursor as segments
        // advance in time, replacing the per-segment O(turns) full scan with an
        // amortized O(1) sweep. Segments are also iterated in start-time order.
        SpeakerTurn[] turnsByStart = turns
            .OrderBy(turn => turn.StartSeconds)
            .ThenBy(turn => turn.EndSeconds)
            .ToArray();
        RecognizedTranscriptSegment[] orderedSegments = recognizedSegments
            .OrderBy(segment => segment.StartSeconds)
            .ThenBy(segment => segment.Index)
            .ToArray();

        int cursor = 0;
        foreach (RecognizedTranscriptSegment segment in orderedSegments)
        {
            // Advance the cursor past turns whose end is before the segment start —
            // those cannot overlap any segment that comes later either (segments are sorted).
            while (cursor < turnsByStart.Length && turnsByStart[cursor].EndSeconds < segment.StartSeconds)
            {
                cursor++;
            }

            SpeakerTurn? bestTurn = null;
            double bestOverlap = 0d;
            double bestConfidence = double.NegativeInfinity;
            double bestStart = double.PositiveInfinity;

            for (int i = cursor; i < turnsByStart.Length; i++)
            {
                SpeakerTurn turn = turnsByStart[i];
                if (turn.StartSeconds > segment.EndSeconds)
                {
                    // Remaining turns start after this segment ends; none can overlap.
                    break;
                }

                double overlap = GetOverlapSeconds(segment.StartSeconds, segment.EndSeconds, turn.StartSeconds, turn.EndSeconds);
                if (overlap <= 0d)
                {
                    continue;
                }

                double confidence = turn.Confidence ?? -1d;
                if (overlap > bestOverlap ||
                    (overlap == bestOverlap && confidence > bestConfidence) ||
                    (overlap == bestOverlap && confidence == bestConfidence && turn.StartSeconds < bestStart))
                {
                    bestTurn = turn;
                    bestOverlap = overlap;
                    bestConfidence = confidence;
                    bestStart = turn.StartSeconds;
                }
            }

            if (bestTurn is not null)
            {
                speakerIdsBySegmentIndex[segment.Index] = bestTurn.SpeakerId;
            }
            else if (fallbackSpeakerId is Guid speakerId)
            {
                speakerIdsBySegmentIndex[segment.Index] = speakerId;
            }
        }

        return speakerIdsBySegmentIndex;
    }

    private static double GetOverlapSeconds(
        double leftStartSeconds,
        double leftEndSeconds,
        double rightStartSeconds,
        double rightEndSeconds)
    {
        double overlap = Math.Min(leftEndSeconds, rightEndSeconds) - Math.Max(leftStartSeconds, rightStartSeconds);
        return overlap > 0d ? overlap : 0d;
    }

    private async Task<IReadOnlyList<SpeechRegion>> ResolveSpeechRegionsForDiarizationAsync(
        Guid projectId,
        IReadOnlyList<TranscriptSegment> segments,
        CancellationToken cancellationToken)
    {
        SpeechRegionsArtifactSnapshot? artifact = await transcriptArtifactWriter
            .TryReadSpeechRegionsSnapshotAsync(projectId, null, cancellationToken)
            .ConfigureAwait(false);
        if (artifact?.Regions is { Count: > 0 })
        {
            return artifact.Regions;
        }

        return segments
            .OrderBy(static segment => segment.SegmentIndex)
            .Select(static segment => new SpeechRegion(segment.SegmentIndex, segment.StartSeconds, segment.EndSeconds))
            .ToArray();
    }

    private static string BuildNextDiarizedSpeakerDisplayName(HashSet<string> existingNames)
    {
        const string prefix = "Speaker ";
        int nextNumber = existingNames
            .Select(static name => name.Trim())
            .Where(static name => name.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            .Select(static name => int.TryParse(name[prefix.Length..], out int number) ? number : 0)
            .DefaultIfEmpty(0)
            .Max() + 1;

        string candidate;
        do
        {
            candidate = $"{prefix}{nextNumber++}";
        }
        while (existingNames.Contains(candidate));

        return candidate;
    }
}

public sealed record SpeakerAssignmentResult(
    IReadOnlyList<ProjectSpeaker> Speakers,
    IReadOnlyList<SpeakerTurn> Turns,
    IReadOnlyDictionary<int, Guid> SegmentSpeakerIdsByIndex);

public sealed record DiarizationResult(
    IReadOnlyList<ProjectSpeaker> Speakers,
    IReadOnlyList<SpeakerTurn> Turns);

internal sealed record ClipRange(
    double StartSeconds,
    double EndSeconds);
