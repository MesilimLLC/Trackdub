using Trackdub.Application.Logging;
using Trackdub.Domain;
using Trackdub.Domain.StageRuns;
using Trackdub.Contracts;
using Trackdub.Application.Pipeline;
using Trackdub.Contracts.Projects;
using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Application.Transcripts.Stages;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Projects;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.Transcript;
namespace Trackdub.Application.Transcripts;

public sealed class TranscriptGenerationService(
    ITranscriptRepository transcriptRepository,
    IArtifactStore artifactStore,
    AsrStageHandler asrStageHandler,
    TranscriptArtifactWriter artifactWriter,
    VadGenerationStage vadGenerationStage,
    ITranscriptGenerationStage speechEnhancementGenerationStage,
    SpeakerDiarizationStage speakerDiarizationStage,
    AsrGenerationStage asrGenerationStage,
    TextRefinementGenerationStage textRefinementGenerationStage,
    SpeakerAssignmentAndPersistenceStage speakerAssignmentAndPersistenceStage,
    IPipelinePreFlightChecker preFlightChecker,
    IProjectStageRunStore stageRunStore,
    IMediaAssetRepository mediaAssetRepository,
    ISpeakerRepository speakerRepository,
    PipelineDegradationWriter? degradationWriter = null,
    Trackdub.Contracts.Pipeline.IPipelineRunLifecycle? pipelineRunLifecycle = null,
    IApplicationLogger? logger = null,
    IModelAliasResolver? modelAliasResolver = null)
{
    private const double ShortAudioFallbackMaximumSeconds = TranscriptPipelineConstants.ShortAudioFallbackMaximumSeconds;

    private readonly IApplicationLogger? logger = logger;
    private readonly IModelAliasResolver? modelAliasResolver = modelAliasResolver;
    private readonly PipelineDegradationWriter? degradationWriter = degradationWriter;
    private readonly ITranscriptRepository transcriptRepository = transcriptRepository ?? throw new ArgumentNullException(nameof(transcriptRepository));
    private readonly IArtifactStore artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));
    private readonly AsrStageHandler asrStageHandler = asrStageHandler ?? throw new ArgumentNullException(nameof(asrStageHandler));
    private readonly TranscriptArtifactWriter artifactWriter = artifactWriter ?? throw new ArgumentNullException(nameof(artifactWriter));
    private readonly VadGenerationStage vadGenerationStage = vadGenerationStage ?? throw new ArgumentNullException(nameof(vadGenerationStage));
    private readonly ITranscriptGenerationStage speechEnhancementGenerationStage = speechEnhancementGenerationStage ?? throw new ArgumentNullException(nameof(speechEnhancementGenerationStage));
    private readonly SpeakerDiarizationStage speakerDiarizationStage = speakerDiarizationStage ?? throw new ArgumentNullException(nameof(speakerDiarizationStage));
    private readonly AsrGenerationStage asrGenerationStage = asrGenerationStage ?? throw new ArgumentNullException(nameof(asrGenerationStage));
    private readonly TextRefinementGenerationStage textRefinementGenerationStage = textRefinementGenerationStage ?? throw new ArgumentNullException(nameof(textRefinementGenerationStage));
    private readonly SpeakerAssignmentAndPersistenceStage speakerAssignmentAndPersistenceStage = speakerAssignmentAndPersistenceStage ?? throw new ArgumentNullException(nameof(speakerAssignmentAndPersistenceStage));
    private readonly IPipelinePreFlightChecker preFlightChecker = preFlightChecker ?? throw new ArgumentNullException(nameof(preFlightChecker));
    private readonly IProjectStageRunStore stageRunStore = stageRunStore ?? throw new ArgumentNullException(nameof(stageRunStore));
    private readonly IMediaAssetRepository mediaAssetRepository = mediaAssetRepository ?? throw new ArgumentNullException(nameof(mediaAssetRepository));
    private readonly ISpeakerRepository speakerRepository = speakerRepository ?? throw new ArgumentNullException(nameof(speakerRepository));

    // The pipeline is built once and cached: the stage instances are injected at construction time
    // and hold no per-run mutable state that would require a fresh builder on each call.
    // Optional polish stays on the spine and no-ops when disabled (same pattern as diarization).
    private readonly Trackdub.Application.Transcripts.Pipeline.ITranscriptGenerationPipeline _pipeline =
        new Trackdub.Application.Transcripts.Pipeline.TranscriptPipelineBuilder(
                degradationWriter,
                artifactStore,
                artifactWriter,
                transcriptRepository,
                stageRunStore)
            .AddStage(speechEnhancementGenerationStage ?? throw new ArgumentNullException(nameof(speechEnhancementGenerationStage)))
            .AddStage(vadGenerationStage ?? throw new ArgumentNullException(nameof(vadGenerationStage)))
            .AddStage(speakerDiarizationStage ?? throw new ArgumentNullException(nameof(speakerDiarizationStage)))
            .AddStage(asrGenerationStage ?? throw new ArgumentNullException(nameof(asrGenerationStage)))
            .AddStage(textRefinementGenerationStage ?? throw new ArgumentNullException(nameof(textRefinementGenerationStage)))
            .AddStage(speakerAssignmentAndPersistenceStage ?? throw new ArgumentNullException(nameof(speakerAssignmentAndPersistenceStage)))
            .Build();

    public async Task GenerateTranscriptAsync(
        TrackdubProject project,
        MediaAsset mediaAsset,
        ProjectArtifact normalizedAudioArtifact,
        TranscriptAudioRoutingPlan audioRoutingPlan,
        bool enableSpeakerDiarization,
        InferenceModelPreferences? modelPreferences,
        CancellationToken cancellationToken,
        string? sourceLanguage = null,
        IProgress<PipelineProgressEvent>? progress = null,
        bool forceRerun = false)
    {
        IReadOnlyList<StageRunRecord> existingRuns = await ReconcileStaleStageRunsAsync(
            project.Id,
            cancellationToken)
            .ConfigureAwait(false);

        InferenceModelPreferences preferences = modelPreferences ?? InferenceModelPreferences.Empty;
        IReadOnlyDictionary<string, string> executionSnapshot = BuildExecutionSnapshot(modelPreferences, sourceLanguage);
        string projectRootPath = ResolveProjectRootPath();
        TranscriptProjectState? resumeState = await LoadResumeStateAsync(
            project,
            mediaAsset,
            existingRuns,
            cancellationToken).ConfigureAwait(false);

        PipelineProgressReporter.Phase(progress, StageNames.Asr, "Checking models", "Checking transcript model readiness.");

        // Run preflight checks — each may trigger RuntimePlanner hardware enumeration.
        var stagesToCheck = new List<string>(4) { StageNames.Vad };
        if (enableSpeakerDiarization)
        {
            stagesToCheck.Add(StageNames.Diarization);
        }

        stagesToCheck.Add(StageNames.Asr);

        if (preferences.EnableAsrTextRefinement)
        {
            stagesToCheck.Add(StageNames.TextRefinementAsr);
        }

        HashSet<string> skipPreflightStages = BuildResumableStageSet(
            resumeState,
            projectRootPath,
            executionSnapshot,
            enableSpeakerDiarization,
            preferences.EnableAsrTextRefinement,
            forceRerun);

        // Check model readiness for each stage. VAD/ASR/Diarization have handler-level auto-download,
        // so defer DownloadRequired exceptions for those stages to allow handlers to provision models.
        foreach (string stage in stagesToCheck)
        {
            if (skipPreflightStages.Contains(stage))
            {
                continue;
            }

            await EnsureStageModelsReadyAsync(stage, sourceLanguage, cancellationToken)
                .ConfigureAwait(false);
        }

        var context = new Trackdub.Application.Transcripts.Pipeline.TranscriptGenerationContext(
            project,
            mediaAsset,
            normalizedAudioArtifact,
            audioRoutingPlan,
            enableSpeakerDiarization,
            sourceLanguage,
            modelPreferences)
        {
            ExecutionSnapshot = executionSnapshot,
            ProjectState = resumeState,
            ProjectRootPath = projectRootPath,
            ForceRerun = forceRerun
        };

        pipelineRunLifecycle?.BeginRun();
        try
        {
            await _pipeline.ExecuteAsync(context, cancellationToken, progress).ConfigureAwait(false);
        }
        finally
        {
            pipelineRunLifecycle?.EndRun();
        }
    }

    /// <summary>
    /// Runs a single transcript pipeline stage (VAD, diarization, or ASR) against an existing project
    /// with normalized audio and routing already prepared.
    /// </summary>
    public async Task GenerateTranscriptStageAsync(
        TrackdubProject project,
        MediaAsset mediaAsset,
        ProjectArtifact normalizedAudioArtifact,
        TranscriptAudioRoutingPlan audioRoutingPlan,
        string stageName,
        bool enableSpeakerDiarization,
        InferenceModelPreferences? modelPreferences,
        CancellationToken cancellationToken,
        string? sourceLanguage = null,
        IProgress<PipelineProgressEvent>? progress = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(stageName);

        await ReconcileStaleStageRunsAsync(project.Id, cancellationToken)
            .ConfigureAwait(false);

        IReadOnlyList<ITranscriptGenerationStage> stages = ResolveTranscriptStages(stageName);

        PipelineProgressReporter.Phase(progress, stageName, "Checking models", "Checking transcript model readiness.");
        // Skip preflight for isolated TextRefinementAsr when polish is disabled so it remains a true no-op.
        InferenceModelPreferences preferences = modelPreferences ?? InferenceModelPreferences.Empty;
        bool shouldSkipPreflight = string.Equals(stageName, StageNames.TextRefinementAsr, StringComparison.OrdinalIgnoreCase)
                                   && !preferences.EnableAsrTextRefinement;
        if (!shouldSkipPreflight)
        {
            await EnsureStageModelsReadyAsync(stageName, sourceLanguage, cancellationToken)
                .ConfigureAwait(false);
        }

        var context = new TranscriptGenerationContext(
            project,
            mediaAsset,
            normalizedAudioArtifact,
            audioRoutingPlan,
            enableSpeakerDiarization,
            sourceLanguage,
            modelPreferences);

        TranscriptGenerationPipeline singleStagePipeline = CreateSingleStagePipeline(stages);

        pipelineRunLifecycle?.BeginRun();
        try
        {
            context = await PrepareStageContextAsync(context, stageName, cancellationToken, progress)
                .ConfigureAwait(false);
            await singleStagePipeline.ExecuteAsync(context, cancellationToken, progress).ConfigureAwait(false);
        }
        finally
        {
            pipelineRunLifecycle?.EndRun();
        }
    }

    private async Task<TranscriptGenerationContext> PrepareStageContextAsync(
        TranscriptGenerationContext context,
        string stageName,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress)
    {
        bool hasCurrentSpeechRegions = await artifactWriter
            .TryReadSpeechRegionsAsync(context.AudioRoutingPlan.VadAudioArtifact, cancellationToken)
            .ConfigureAwait(false) is not null;
        if (!string.Equals(stageName, StageNames.Vad, StringComparison.OrdinalIgnoreCase) && !hasCurrentSpeechRegions)
        {
            await EnsureStageModelsReadyAsync(StageNames.Vad, context.SourceLanguage, cancellationToken)
                .ConfigureAwait(false);
            context = await vadGenerationStage.ExecuteAsync(context, cancellationToken, progress)
                .ConfigureAwait(false);
        }

        return await HydrateContextForSingleStageAsync(context, stageName, cancellationToken)
            .ConfigureAwait(false);
    }

    private IReadOnlyList<ITranscriptGenerationStage> ResolveTranscriptStages(string stageName) =>
        stageName switch
        {
            StageNames.Vad => [vadGenerationStage],
            StageNames.Diarization => [speakerDiarizationStage],
            // ASR must be followed by persistence so the TranscriptRevision is saved to the database.
            // Polish stays on the spine and no-ops when disabled (same pattern as diarization).
            StageNames.Asr => [asrGenerationStage, textRefinementGenerationStage, speakerAssignmentAndPersistenceStage],
            StageNames.TextRefinementAsr =>
            [
                textRefinementGenerationStage,
                speakerAssignmentAndPersistenceStage
            ],
            _ => throw new ArgumentException(
                $"Unsupported isolated transcript stage '{stageName}'. Supported stages: {StageNames.Vad}, {StageNames.Diarization}, {StageNames.Asr}, {StageNames.TextRefinementAsr}.",
                nameof(stageName)),
        };

    private async Task<TranscriptGenerationContext> HydrateContextForSingleStageAsync(
        TranscriptGenerationContext context,
        string stageName,
        CancellationToken cancellationToken)
    {
        if (string.Equals(stageName, StageNames.Vad, StringComparison.OrdinalIgnoreCase))
        {
            return context;
        }

        if (string.Equals(stageName, StageNames.TextRefinementAsr, StringComparison.OrdinalIgnoreCase))
        {
            return await HydrateContextForTextRefinementAsrAsync(context, cancellationToken).ConfigureAwait(false);
        }

        IReadOnlyList<SpeechRegion> speechRegions = await RequireSpeechRegionsArtifactAsync(
            context.AudioRoutingPlan.VadAudioArtifact,
            cancellationToken)
            .ConfigureAwait(false);

        bool usedShortAudioFallback = false;
        if (string.Equals(stageName, StageNames.Asr, StringComparison.OrdinalIgnoreCase))
        {
            bool hadSpeechRegions = speechRegions.Count > 0;
            speechRegions = await ApplyShortAudioFallbackAsync(context, speechRegions, cancellationToken)
                .ConfigureAwait(false);
            usedShortAudioFallback = !hadSpeechRegions && speechRegions.Count > 0;
        }

        SpeechRegion[] regions = speechRegions.OrderBy(static region => region.Index).ToArray();
        context = context with { SpeechRegions = regions };

        if (string.Equals(stageName, StageNames.Diarization, StringComparison.OrdinalIgnoreCase))
        {
            return context;
        }

        if (usedShortAudioFallback)
        {
            return context with
            {
                DiarizationResult = null,
                RegionPlan = new TranscriptRegionPlan(regions, new Dictionary<int, Guid>())
            };
        }

        var (_, diarizationResult, regionPlan) = await BuildRegionPlanAsync(
            context,
            context.SpeechRegions!,
            cancellationToken)
            .ConfigureAwait(false);

        return context with
        {
            DiarizationResult = diarizationResult,
            RegionPlan = regionPlan
        };
    }

    private async Task<IReadOnlyList<StageRunRecord>> ReconcileStaleStageRunsAsync(
        Guid projectId,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<StageRunRecord> existingRuns = await stageRunStore
            .ListByProjectAsync(projectId, cancellationToken)
            .ConfigureAwait(false);
        await StageRunHygiene.ReconcileStaleRunningAsync(
            stageRunStore,
            existingRuns,
            logger,
            cancellationToken)
            .ConfigureAwait(false);
        return existingRuns;
    }

    private async Task EnsureStageModelsReadyAsync(
        string stageName,
        string? sourceLanguage,
        CancellationToken cancellationToken)
    {
        bool isAsrStage = string.Equals(stageName, StageNames.Asr, StringComparison.OrdinalIgnoreCase);
        bool isDiarizationStage = string.Equals(stageName, StageNames.Diarization, StringComparison.OrdinalIgnoreCase);

        try
        {
            await preFlightChecker.EnsureModelsAvailableAsync(
                stageName,
                cancellationToken,
                isAsrStage ? sourceLanguage : null).ConfigureAwait(false);
        }
        catch (RequiredModelNotAvailableException ex)
        {
            // Only diarization has handler-level model provisioning (EnsureModelAvailableAsync).
            // VAD/ASR routed engines reject non-runnable DownloadRequired plans, so keep preflight failures visible.
            if (!ex.CanAutoDownload || !isDiarizationStage)
            {
                throw;
            }

            logger?.LogInformation(
                "Deferring {StageName} model download to stage execution: {ModelId}",
                stageName,
                ex.ModelId);
        }
    }

    private TranscriptGenerationPipeline CreateSingleStagePipeline(
        IReadOnlyList<ITranscriptGenerationStage> stages)
    {
        return new TranscriptGenerationPipeline(
            stages,
            artifactStore,
            artifactWriter,
            transcriptRepository,
            degradationWriter,
            stageRunStore);
    }

    private async Task<TranscriptGenerationContext> HydrateContextForTextRefinementAsrAsync(
        TranscriptGenerationContext context,
        CancellationToken cancellationToken)
    {
        RawAsrTranscriptArtifact rawAsr = await RequireRawAsrTranscriptAsync(
            context.Project.Id,
            cancellationToken)
            .ConfigureAwait(false);

        IReadOnlyList<SpeechRegion> speechRegions = await RequireSpeechRegionsArtifactAsync(
            context.AudioRoutingPlan.VadAudioArtifact,
            cancellationToken)
            .ConfigureAwait(false);

        var (regions, diarizationResult, regionPlan) = await BuildRegionPlanAsync(
            context,
            speechRegions,
            cancellationToken)
            .ConfigureAwait(false);

        StageRunRecord asrStageRun = await LoadOrCreateAsrStageRunAsync(
            rawAsr.StageRunId,
            context.Project.Id,
            cancellationToken)
            .ConfigureAwait(false);

        return context with
        {
            SpeechRegions = regions,
            DiarizationResult = diarizationResult,
            RegionPlan = regionPlan,
            AsrResult = new AsrStageResult(asrStageRun, rawAsr.Segments)
        };
    }

    private async Task<RawAsrTranscriptArtifact> RequireRawAsrTranscriptAsync(
        Guid projectId,
        CancellationToken cancellationToken)
    {
        RawAsrTranscriptArtifact? rawAsr = await artifactWriter
            .TryReadRawAsrTranscriptAsync(projectId, cancellationToken)
            .ConfigureAwait(false);

        if (rawAsr is null or { Segments.Count: 0 })
        {
            throw new InvalidOperationException(
                "Raw ASR transcript artifact is required before running ASR text refinement. Run the ASR stage first.");
        }

        return rawAsr;
    }

    private async Task<IReadOnlyList<SpeechRegion>> RequireSpeechRegionsArtifactAsync(
        ProjectArtifact expectedInputAudioArtifact,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<SpeechRegion>? speechRegions = await artifactWriter
            .TryReadSpeechRegionsAsync(expectedInputAudioArtifact, cancellationToken)
            .ConfigureAwait(false);

        if (speechRegions is null)
        {
            throw new InvalidOperationException(
                "Speech regions artifact is required before running ASR text refinement. Run the VAD stage first.");
        }

        return speechRegions;
    }

    private async Task<IReadOnlyList<SpeechRegion>> ApplyShortAudioFallbackAsync(
        TranscriptGenerationContext context,
        IReadOnlyList<SpeechRegion> speechRegions,
        CancellationToken cancellationToken)
    {
        if (speechRegions.Count > 0)
        {
            return speechRegions;
        }

        double durationSeconds = context.AudioRoutingPlan.AsrAudioArtifact.DurationSeconds
                                 ?? context.NormalizedAudioArtifact.DurationSeconds
                                 ?? context.MediaAsset.DurationSeconds;
        if (!double.IsFinite(durationSeconds)
            || durationSeconds <= 0d
            || durationSeconds >= ShortAudioFallbackMaximumSeconds)
        {
            return speechRegions;
        }

        var fallbackRegion = new SpeechRegion(0, 0d, durationSeconds);
        logger?.LogInformation(
            "VAD detected no regions in {DurationSeconds:F1}s audio; ASR will use the full audio fallback.",
            durationSeconds);

        if (degradationWriter is not null)
        {
            try
            {
                await degradationWriter.WriteAsync(
                    new PipelineDegradationRecord(
                        StageNames.Asr,
                        "VAD_NO_REGIONS_SHORT_AUDIO_FALLBACK",
                        "VAD detected no speech regions in short audio; ASR will process the full audio.",
                        Detail: $"Duration: {durationSeconds:F3}s",
                        SelectedFallback: "full-audio-asr",
                        RecommendedAction: "Review the transcript before export.",
                        DateTimeOffset.UtcNow,
                        StageRunId: null),
                    context.Project.Id,
                    context.MediaAsset.Id,
                    cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                // Best-effort degradation logging: any persistence failure must not block the ASR fallback.
                logger?.LogWarning(
                    "Failed to persist the short-audio VAD fallback record. {ExceptionType}: {Message}",
                    ex.GetType().Name,
                    ex.Message);
            }
        }

        return [fallbackRegion];
    }

    private async Task<(SpeechRegion[] Regions, DiarizationResult? Diarization, TranscriptRegionPlan RegionPlan)>
        BuildRegionPlanAsync(
            TranscriptGenerationContext context,
            IReadOnlyList<SpeechRegion> speechRegions,
            CancellationToken cancellationToken)
    {
        SpeechRegion[] regions = speechRegions.OrderBy(static region => region.Index).ToArray();

        DiarizationResult? diarizationResult = await artifactWriter
            .TryReadDiarizationResultAsync(context.Project.Id, cancellationToken)
            .ConfigureAwait(false);

        double durationSeconds = context.AudioRoutingPlan.DiarizationAudioArtifact.DurationSeconds
                                 ?? context.NormalizedAudioArtifact.DurationSeconds
                                 ?? context.MediaAsset.DurationSeconds;

        TranscriptRegionPlan regionPlan = TranscriptWorkflowUtilities.BuildTranscriptRegionPlan(
            regions,
            diarizationResult,
            durationSeconds);

        return (regions, diarizationResult, regionPlan);
    }

    private async Task<StageRunRecord> LoadOrCreateAsrStageRunAsync(
        Guid stageRunId,
        Guid projectId,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<StageRunRecord> projectRuns = await stageRunStore
            .ListByProjectAsync(projectId, cancellationToken)
            .ConfigureAwait(false);

        StageRunRecord? existingRun = projectRuns.FirstOrDefault(run => run.Id == stageRunId);
        if (existingRun is not null)
        {
            return existingRun;
        }

        // Fallback: create synthetic record if stage run not found in store.
        // This can occur when hydrating context for text refinement after ASR completed
        // but before the stage run was persisted, or in test scenarios.
        DateTimeOffset now = DateTimeOffset.UtcNow;
        return new StageRunRecord(
            stageRunId,
            projectId,
            StageNames.Asr,
            StageRunStatus.Completed,
            now,
            now,
            FailureReason: null);
    }

    public async Task RetranscribeSegmentsAsync(
        TranscriptProjectState currentState,
        RetranscribeTranscriptSegmentsRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        TranscriptRevision currentRevision = TranscriptWorkflowUtilities.GetRequiredTranscriptRevision(currentState);
        TranscriptWorkflowUtilities.EnsureRevisionMatches(
            currentRevision,
            request.TranscriptRevisionId,
            "Re-transcription was based on an out-of-date transcript revision.");

        Guid[] requestedSegmentIds = request.SegmentIds.Distinct().ToArray();
        TranscriptSegment[] existingSegments = currentState.TranscriptSegments
            .OrderBy(segment => segment.SegmentIndex)
            .ToArray();

        (TranscriptSegment[] selectedSegments, HashSet<Guid> requestedIds) = ValidateAndSelectSegments(
            requestedSegmentIds,
            existingSegments);

        ProjectArtifact asrAudioArtifact = TranscriptWorkflowUtilities.ResolveAsrAudioArtifact(
                currentState.ProjectState.Artifacts,
                currentState.StageRuns)
            ?? throw new InvalidOperationException("The project does not contain audio for transcription.");
        string asrAudioPath = artifactStore.GetPath(asrAudioArtifact.RelativePath);
        SpeechRegion[] regions = selectedSegments
            .Select(segment => new SpeechRegion(segment.SegmentIndex, segment.StartSeconds, segment.EndSeconds))
            .ToArray();

        Guid projectId = currentState.ProjectState.Project.Id;
        AsrStageRequest asrRequest = BuildAsrStageRequest(
            projectId,
            asrAudioPath,
            regions,
            request,
            currentState.TranscriptLanguage);
        AsrStageResult asrResult = await asrStageHandler.HandleAsync(asrRequest, cancellationToken).ConfigureAwait(false);

        Dictionary<int, RecognizedTranscriptSegment> recognizedByIndex = asrResult.Segments
            .GroupBy(segment => segment.Index)
            .ToDictionary(group => group.Key, group => group.Last());
        Queue<RecognizedTranscriptSegment> recognizedByOrder = new(asrResult.Segments.OrderBy(segment => segment.Index));
        int revisionNumber = await transcriptRepository.GetNextRevisionNumberAsync(projectId, cancellationToken).ConfigureAwait(false);
        TranscriptRevision revision = TranscriptRevision.Create(projectId, asrResult.StageRun.Id, revisionNumber, DateTimeOffset.UtcNow);

        TranscriptSegment[] revisedSegmentArray = CreateRevisedSegments(
            existingSegments,
            requestedIds,
            recognizedByIndex,
            recognizedByOrder,
            revision.Id);

        MediaAsset mediaAsset = TranscriptWorkflowUtilities.GetRequiredMediaAsset(currentState);
        await transcriptRepository.SaveRevisionAsync(revision, revisedSegmentArray, cancellationToken).ConfigureAwait(false);
        await artifactWriter.WriteTranscriptArtifactAsync(
            projectId,
            mediaAsset,
            revision,
            revisedSegmentArray,
            asrResult.StageRun.Id,
            "segment-retranscribe",
            cancellationToken).ConfigureAwait(false);
        logger?.LogInformation(
            PipelineRuntimeProvenanceFormatter.FormatStageSegmentLogLine(
                StageNames.Asr,
                selectedSegments[0].SegmentIndex,
                asrResult.StageRun.RuntimeInfo));
        if (selectedSegments.Length > 1)
        {
            logger?.LogInformation(
                $"ASR re-transcribed {selectedSegments.Length} segments in one run (segment indices: {string.Join(", ", selectedSegments.Select(s => s.SegmentIndex))}).");
        }

        HashSet<int> updatedIndices = selectedSegments.Select(s => s.SegmentIndex).ToHashSet();
        ProjectUiSettings updatedUiSettings = SegmentStageRunProvenanceStore.RecordAsrRuns(
            currentState.ProjectUiSettings,
            existingSegments.Select(s => s.SegmentIndex),
            updatedIndices,
            asrResult.StageRun.Id,
            currentRevision.StageRunId);
        await SegmentStageRunProvenanceStore.PersistUiSettingsAsync(
            artifactStore,
            currentState.ProjectState.Project,
            currentState.TranscriptLanguage,
            updatedUiSettings,
            cancellationToken).ConfigureAwait(false);
    }

    private IReadOnlyDictionary<string, string> BuildExecutionSnapshot(
        InferenceModelPreferences? modelPreferences,
        string? sourceLanguage)
    {
        var snapshot = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        if (!string.IsNullOrWhiteSpace(sourceLanguage))
        {
            snapshot["SourceLanguage"] = sourceLanguage;
        }

        if (modelPreferences is null)
        {
            return snapshot;
        }

        // Capture model aliases for each stage
        if (!string.IsNullOrWhiteSpace(modelPreferences.VadModelAlias))
        {
            snapshot[$"Model:{StageNames.Vad}"] = modelPreferences.VadModelAlias;
            AddModelId(snapshot, StageNames.Vad, modelPreferences.VadModelAlias);
        }

        if (!string.IsNullOrWhiteSpace(modelPreferences.AsrModelAlias))
        {
            snapshot[$"Model:{StageNames.Asr}"] = modelPreferences.AsrModelAlias;
            AddModelId(snapshot, StageNames.Asr, modelPreferences.AsrModelAlias);
        }

        if (!string.IsNullOrWhiteSpace(modelPreferences.DiarizationModelAlias))
        {
            snapshot[$"Model:{StageNames.Diarization}"] = modelPreferences.DiarizationModelAlias;
            AddModelId(snapshot, StageNames.Diarization, modelPreferences.DiarizationModelAlias);
        }

        if (!string.IsNullOrWhiteSpace(modelPreferences.TextRefinementModelAlias))
        {
            snapshot[$"Model:{StageNames.TextRefinementAsr}"] = modelPreferences.TextRefinementModelAlias;
            AddModelId(snapshot, StageNames.TextRefinementAsr, modelPreferences.TextRefinementModelAlias);
        }

        AddPreferredModelVariant(snapshot, modelPreferences, StageNames.Vad, RuntimeStage.Vad);
        AddPreferredModelVariant(snapshot, modelPreferences, StageNames.Asr, RuntimeStage.Asr);
        AddPreferredModelVariant(snapshot, modelPreferences, StageNames.Diarization, RuntimeStage.Diarization);
        AddPreferredModelVariant(snapshot, modelPreferences, StageNames.TextRefinementAsr, RuntimeStage.TextRefinement);

        return snapshot;
    }

    private void AddModelId(Dictionary<string, string> snapshot, string stageName, string modelAlias)
    {
        if (modelAliasResolver is not null &&
            modelAliasResolver.TryResolveModelId(modelAlias, out string? modelId) &&
            !string.IsNullOrWhiteSpace(modelId))
        {
            snapshot[$"ModelId:{stageName}"] = modelId;
        }
    }

    private static void AddPreferredModelVariant(
        Dictionary<string, string> snapshot,
        InferenceModelPreferences modelPreferences,
        string stageName,
        RuntimeStage runtimeStage)
    {
        string? variantAlias = modelPreferences.GetPreferredModelVariantAlias(runtimeStage);
        if (!string.IsNullOrWhiteSpace(variantAlias))
        {
            snapshot[$"ModelVariant:{stageName}"] = variantAlias;
        }
    }

    private string ResolveProjectRootPath()
    {
        string manifestPath = artifactStore.GetPath(ProjectArtifactPaths.ManifestRelativePath);
        string? directory = Path.GetDirectoryName(manifestPath);
        return string.IsNullOrWhiteSpace(directory) ? manifestPath : directory;
    }

    private async Task<TranscriptProjectState?> LoadResumeStateAsync(
        TrackdubProject project,
        MediaAsset mediaAsset,
        IReadOnlyList<StageRunRecord> stageRuns,
        CancellationToken cancellationToken)
    {
        try
        {
            // Parallelize independent repository calls for better performance
            Task<TranscriptRevision?> revisionTask = transcriptRepository
                .GetCurrentRevisionAsync(project.Id, cancellationToken);
            Task<IReadOnlyList<ProjectArtifact>> artifactsTask = mediaAssetRepository
                .GetArtifactsAsync(project.Id, cancellationToken);
            Task<IReadOnlyList<SpeakerTurn>> turnsTask = speakerRepository
                .ListTurnsAsync(project.Id, cancellationToken);
            Task<IReadOnlyList<ProjectSpeaker>> speakersTask = speakerRepository
                .ListSpeakersAsync(project.Id, cancellationToken);

            await Task.WhenAll(revisionTask, artifactsTask, turnsTask, speakersTask)
                .ConfigureAwait(false);

            TranscriptRevision? currentRevision = await revisionTask.ConfigureAwait(false);

            // Load segments only if revision exists
            IReadOnlyList<TranscriptSegment> segments = currentRevision is null
                ? []
                : await transcriptRepository.GetSegmentsAsync(currentRevision.Id, cancellationToken)
                    .ConfigureAwait(false);

            return BuildTranscriptProjectState(
                project,
                mediaAsset,
                currentRevision,
                segments,
                await artifactsTask.ConfigureAwait(false),
                await speakersTask.ConfigureAwait(false),
                await turnsTask.ConfigureAwait(false),
                stageRuns);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            logger?.LogWarning(ex, "Transcript resume state unavailable for project {ProjectId}", project.Id);
            return null;
        }
    }

    private static TranscriptProjectState BuildTranscriptProjectState(
        TrackdubProject project,
        MediaAsset mediaAsset,
        TranscriptRevision? currentRevision,
        IReadOnlyList<TranscriptSegment> segments,
        IReadOnlyList<ProjectArtifact> artifacts,
        IReadOnlyList<ProjectSpeaker> speakers,
        IReadOnlyList<SpeakerTurn> speakerTurns,
        IReadOnlyList<StageRunRecord> stageRuns)
    {
        var projectState = new OpenProjectResult(
            project,
            mediaAsset,
            SourceReference: null,
            SourceMediaStatus.Available,
            SourceStatusMessage: null,
            artifacts,
            TranscriptLanguage: null);

        return new TranscriptProjectState(
            projectState,
            currentRevision,
            segments,
            speakers,
            speakerTurns,
            CurrentTranslationRevision: null,
            TranslatedSegments: [],
            IsTranslationStale: false,
            TranscriptLanguage: null,
            stageRuns,
            SupportedTargetLanguages: [],
            SelectedTranslationTargetLanguage: null,
            StaleTranslatedSegmentIndices: new HashSet<int>(),
            WaveformSummary: null,
            AvailableVoices: [],
            VoiceAssignments: [],
            TtsTakes: [],
            TtsSegmentStates: [],
            VoiceAssignmentWarnings: []);
    }

    private static readonly string[] ResumableStageNames =
    [
        StageNames.SpeechEnhancement,
        StageNames.Vad,
        StageNames.Diarization,
        StageNames.Asr,
        StageNames.TextRefinementAsr,
        StageNames.SpeakerAssignment
    ];

    private HashSet<string> BuildResumableStageSet(
        TranscriptProjectState? resumeState,
        string projectRootPath,
        IReadOnlyDictionary<string, string> executionSnapshot,
        bool enableSpeakerDiarization,
        bool enableAsrTextRefinement,
        bool forceRerun)
    {
        var resumableStages = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        if (forceRerun || resumeState is null)
        {
            return resumableStages;
        }

        foreach (string stageName in ResumableStageNames)
        {
            if (!IsStageEligibleForResume(stageName, enableSpeakerDiarization, enableAsrTextRefinement, resumableStages))
            {
                continue;
            }

            if (StageArtifactResumeEvaluator.CanResumeStage(
                    resumeState,
                    artifactStore,
                    stageName,
                    executionSnapshot,
                    projectRootPath))
            {
                resumableStages.Add(stageName);
            }
        }

        return resumableStages;
    }

    private static bool IsStageEligibleForResume(
        string stageName,
        bool enableSpeakerDiarization,
        bool enableAsrTextRefinement,
        HashSet<string> resumableStages)
    {
        return stageName switch
        {
            _ when string.Equals(stageName, StageNames.Diarization, StringComparison.OrdinalIgnoreCase)
                => enableSpeakerDiarization,

            _ when string.Equals(stageName, StageNames.TextRefinementAsr, StringComparison.OrdinalIgnoreCase)
                => enableAsrTextRefinement,

            _ when string.Equals(stageName, StageNames.SpeakerAssignment, StringComparison.OrdinalIgnoreCase)
                => resumableStages.Contains(StageNames.Asr) &&
                   (!enableAsrTextRefinement || resumableStages.Contains(StageNames.TextRefinementAsr)),

            _ => true
        };
    }

    private static RecognizedTranscriptSegment? ResolveRecognition(
        TranscriptSegment segment,
        Dictionary<int, RecognizedTranscriptSegment> byIndex,
        Queue<RecognizedTranscriptSegment> byOrder)
    {
        if (byIndex.TryGetValue(segment.SegmentIndex, out var recognized))
            return recognized;

        return byOrder.Count > 0 ? byOrder.Dequeue() : null;
    }

    private static TranscriptSegment CreateRevisedSegment(
        Guid revisionId,
        int segmentIndex,
        TranscriptSegment source,
        RecognizedTranscriptSegment? recognized)
    {
        if (recognized is null)
        {
            return TranscriptSegment.Create(
                revisionId,
                segmentIndex,
                source.StartSeconds,
                source.EndSeconds,
                source.Text,
                source.SpeakerId,
                source.DetectedLanguage,
                TranscriptWorkflowUtilities.CloneWords(source.Words),
                source.SpeakerAssignmentOrigin);
        }

        (double startSeconds, double endSeconds) = TranscriptWorkflowUtilities.ResolveRecognizedTiming(source, recognized);

        return TranscriptSegment.Create(
            revisionId,
            segmentIndex,
            startSeconds,
            endSeconds,
            string.IsNullOrWhiteSpace(recognized.Text) ? source.Text : recognized.Text,
            source.SpeakerId,
            recognized.DetectedLanguage ?? source.DetectedLanguage,
            TranscriptWorkflowUtilities.CreateTranscriptWords(recognized.Words),
            source.SpeakerAssignmentOrigin);
    }

    private static (TranscriptSegment[] Selected, HashSet<Guid> RequestedIds) ValidateAndSelectSegments(
        Guid[] requestedSegmentIds,
        TranscriptSegment[] existingSegments)
    {
        if (requestedSegmentIds.Length == 0)
        {
            throw new InvalidOperationException("Select at least one segment before re-transcribing.");
        }

        HashSet<Guid> requestedIds = requestedSegmentIds.ToHashSet();
        TranscriptSegment[] selectedSegments = existingSegments
            .Where(segment => requestedIds.Contains(segment.Id))
            .OrderBy(segment => segment.SegmentIndex)
            .ToArray();

        if (selectedSegments.Length != requestedSegmentIds.Length)
        {
            throw new InvalidOperationException("One or more selected segments were not found in the current transcript revision.");
        }

        return (selectedSegments, requestedIds);
    }

    private static AsrStageRequest BuildAsrStageRequest(
        Guid projectId,
        string asrAudioPath,
        SpeechRegion[] regions,
        RetranscribeTranscriptSegmentsRequest request,
        string? transcriptLanguage)
    {
        return new AsrStageRequest(
            projectId,
            asrAudioPath,
            regions,
            request.PreferredModelAlias,
            request.RequirePreferredModelAlias,
            transcriptLanguage,
            request.PreferredExecutionProvider,
            request.RequirePreferredExecutionProvider,
            request.PreferredModelVariantAlias);
    }

    private static TranscriptSegment[] CreateRevisedSegments(
        TranscriptSegment[] existingSegments,
        HashSet<Guid> requestedIds,
        Dictionary<int, RecognizedTranscriptSegment> recognizedByIndex,
        Queue<RecognizedTranscriptSegment> recognizedByOrder,
        Guid revisionId)
    {
        return existingSegments
            .Select(segment =>
            {
                RecognizedTranscriptSegment? recognized = requestedIds.Contains(segment.Id)
                    ? ResolveRecognition(segment, recognizedByIndex, recognizedByOrder)
                    : null;

                // Preserve each segment's existing identity index. Using the enumeration
                // position here renumbered sparse indices (e.g. 1,4,21,26 -> 0,1,2,3),
                // which reordered the transcript panel and broke segment references.
                return CreateRevisedSegment(revisionId, segment.SegmentIndex, segment, recognized);
            })
            .ToArray();
    }
}
