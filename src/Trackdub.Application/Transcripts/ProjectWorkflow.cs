using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Contracts.Projects;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.AudioQuality;
using Trackdub.Domain.Media;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Transcripts;

public sealed class ProjectWorkflow(
    ProjectMediaIngestService projectMediaIngestService,
    TranscriptProjectStateService stateService,
    TranscriptGenerationService transcriptGenerationService,
    StemSeparationStageHandler? stemSeparationStageHandler = null,
    SpeechAudioPreparationStageHandler? speechAudioPreparationStageHandler = null,
    PipelineDegradationWriter? degradationWriter = null,
    SpeechAudioEnhancementStageHandler? speechAudioEnhancementStageHandler = null,
    OverlapRescueWorkflow? overlapRescueWorkflow = null,
    IProjectStageRunStore? stageRunStore = null)
{
    private const string DialogueIsolationUnavailableCode = "DIALOGUE_ISOLATION_UNAVAILABLE";
    private const string DialogueIsolationUnavailableMessage =
        "Dialogue isolation model unavailable; no clean ambiance track was generated.";

    private readonly ProjectMediaIngestService projectMediaIngestService = projectMediaIngestService ?? throw new ArgumentNullException(nameof(projectMediaIngestService));
    private readonly TranscriptProjectStateService stateService = stateService ?? throw new ArgumentNullException(nameof(stateService));
    private readonly TranscriptGenerationService transcriptGenerationService = transcriptGenerationService ?? throw new ArgumentNullException(nameof(transcriptGenerationService));
    private readonly SpeechAudioEnhancementStageHandler? speechAudioEnhancementStageHandler = speechAudioEnhancementStageHandler;
    private readonly OverlapRescueWorkflow? overlapRescueWorkflow = overlapRescueWorkflow;
    private readonly IProjectStageRunStore? stageRunStore = stageRunStore;

    public async Task<TranscriptProjectState> CreateAsync(
        CreateTranscriptProjectRequest request,
        CancellationToken cancellationToken)
    {
        string? sourceLanguage = TranscriptWorkflowUtilities.NormalizeTranscriptLanguageCode(request.SourceLanguage);
        CreateProjectFromMediaResult createResult = await projectMediaIngestService.CreateAsync(
            new CreateProjectFromMediaRequest(request.ProjectName, request.SourceMediaPath),
            cancellationToken).ConfigureAwait(false);

        var currentArtifacts = new List<ProjectArtifact> { createResult.AudioArtifact };

        TranscriptAudioRoutingPlan audioRoutingPlan = await TryPrepareSpeechAudioAsync(
            new TranscriptAudioRoutingRequest
            {
                ProjectId = createResult.Project.Id,
                MediaAsset = createResult.MediaAsset,
                NormalizedAudioArtifact = createResult.AudioArtifact,
                ExistingArtifacts = currentArtifacts,
                CancellationToken = cancellationToken,
                PrepareSpeechAudio = true,
            }).ConfigureAwait(false);

        await transcriptGenerationService.GenerateTranscriptAsync(
            createResult.Project,
            createResult.MediaAsset,
            createResult.AudioArtifact,
            audioRoutingPlan,
            request.EnableSpeakerDiarization,
            request.ModelPreferences,
            cancellationToken,
            sourceLanguage).ConfigureAwait(false);

        return await ReloadAsync(requestedTranslationTargetLanguage: null, cancellationToken).ConfigureAwait(false);
    }

    public async Task<TranscriptProjectState> CreateMediaSpineAsync(
        CreateTranscriptProjectRequest request,
        CancellationToken cancellationToken)
    {
        await projectMediaIngestService.CreateMediaSpineAsync(
            new CreateProjectFromMediaRequest(request.ProjectName, request.SourceMediaPath),
            cancellationToken).ConfigureAwait(false);

        return await ReloadAsync(requestedTranslationTargetLanguage: null, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Extracts normalized audio and waveform after a lightweight media-spine import.
    /// </summary>
    public async Task<TranscriptProjectState> EnsureNormalizedProjectAudioAsync(
        CancellationToken cancellationToken,
        int? ffmpegThreadBudget = null)
    {
        TranscriptProjectState currentState = await OpenAsync(cancellationToken).ConfigureAwait(false);
        MediaAsset mediaAsset = TranscriptWorkflowUtilities.GetRequiredMediaAsset(currentState);
        await projectMediaIngestService.EnsureNormalizedAudioAsync(
            mediaAsset,
            currentState.ProjectState.Artifacts,
            cancellationToken,
            ffmpegThreadBudget).ConfigureAwait(false);

        return await stateService.RefreshArtifactsAndStageRunsAsync(currentState, cancellationToken).ConfigureAwait(false);
    }

    public Task<TranscriptProjectState> RunInitialTranscriptionAsync(
        TranscriptRunOptions options,
        CancellationToken cancellationToken) =>
        RunPreparedGenerationAsync(
            new PreparedGenerationRequest
            {
                IsInitial = true,
                StageName = StageNames.Asr,
                AudioPurpose = "transcription",
                Options = options,
                CancellationToken = cancellationToken,
            });

    public Task<TranscriptProjectState> RunTranscriptStageAsync(
        string stageName,
        TranscriptRunOptions options,
        CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(stageName);
        return RunPreparedGenerationAsync(
            new PreparedGenerationRequest
            {
                IsInitial = false,
                StageName = stageName,
                AudioPurpose = "this stage",
                Options = options,
                CancellationToken = cancellationToken,
            });
    }

    private async Task<TranscriptProjectState> RunPreparedGenerationAsync(PreparedGenerationRequest request)
    {
        ArgumentNullException.ThrowIfNull(request.Options);
        string? normalizedSourceLanguage = TranscriptWorkflowUtilities.NormalizeTranscriptLanguageCode(request.Options.SourceLanguage);
        PreparedTranscriptAudio audio = await PrepareTranscriptAudioAsync(request).ConfigureAwait(false);

        if (request.IsInitial)
        {
            await transcriptGenerationService.GenerateTranscriptAsync(
                audio.ProjectState.ProjectState.Project,
                audio.MediaAsset,
                audio.NormalizedAudioArtifact,
                audio.AudioRoutingPlan,
                request.Options.EnableSpeakerDiarization,
                request.Options.ModelPreferences ?? InferenceModelPreferences.Empty,
                request.CancellationToken,
                normalizedSourceLanguage,
                request.Options.Progress).ConfigureAwait(false);
        }
        else
        {
            await transcriptGenerationService.GenerateTranscriptStageAsync(
                audio.ProjectState.ProjectState.Project,
                audio.MediaAsset,
                audio.NormalizedAudioArtifact,
                audio.AudioRoutingPlan,
                request.StageName,
                request.Options.EnableSpeakerDiarization,
                request.Options.ModelPreferences ?? InferenceModelPreferences.Empty,
                request.CancellationToken,
                normalizedSourceLanguage,
                request.Options.Progress).ConfigureAwait(false);
        }

        return await ReloadAsync(audio.ProjectState.SelectedTranslationTargetLanguage, request.CancellationToken).ConfigureAwait(false);
    }

    public Task<TranscriptProjectState> OpenAsync(CancellationToken cancellationToken) =>
        ReloadAsync(requestedTranslationTargetLanguage: null, cancellationToken);

    public Task<TranscriptProjectState> OpenProjectShellAsync(CancellationToken cancellationToken) =>
        stateService.OpenProjectShellAsync(requestedTranslationTargetLanguage: null, cancellationToken);

    public Task<TranscriptProjectState> ReloadAsync(
        string? requestedTranslationTargetLanguage,
        CancellationToken cancellationToken) =>
        stateService.OpenAsync(requestedTranslationTargetLanguage, cancellationToken);

    public async Task<TranscriptProjectState> SaveUiSettingsAsync(
        ProjectUiSettings settings,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(settings);

        await projectMediaIngestService.SaveUiSettingsAsync(settings, cancellationToken).ConfigureAwait(false);

        return await ReloadAsync(settings.SelectedTranslationTargetLanguage, cancellationToken).ConfigureAwait(false);
    }

    public async Task<TranscriptProjectState> RelocateSourceAsync(
        RelocateTranscriptSourceRequest request,
        CancellationToken cancellationToken)
    {
        await projectMediaIngestService.RelocateSourceAsync(
            new RelocateSourceMediaRequest(request.NewSourceMediaPath),
            cancellationToken).ConfigureAwait(false);

        return await ReloadAsync(request.SelectedTranslationTargetLanguage, cancellationToken).ConfigureAwait(false);
    }

    public async Task<TranscriptProjectState> RenameProjectAsync(
        RenameProjectRequest request,
        CancellationToken cancellationToken)
    {
        await projectMediaIngestService.RenameProjectAsync(request, cancellationToken).ConfigureAwait(false);

        return await ReloadAsync(request.SelectedTranslationTargetLanguage, cancellationToken).ConfigureAwait(false);
    }

    public async Task<TranscriptProjectState> RunStemSeparationAsync(
        CancellationToken cancellationToken,
        IProgress<StemSeparationProgress>? progress = null,
        string? preferredModelAlias = null,
        InferenceModelPreferences? modelPreferences = null)
    {
        TranscriptProjectState currentState = await OpenAsync(cancellationToken).ConfigureAwait(false);
        MediaAsset mediaAsset = TranscriptWorkflowUtilities.GetRequiredMediaAsset(currentState);
        ProjectArtifact sourceAudioArtifact = await projectMediaIngestService
            .EnsureStemSeparationAudioAsync(mediaAsset, currentState.ProjectState.Artifacts, cancellationToken)
            .ConfigureAwait(false);

        if (stemSeparationStageHandler is null)
        {
            throw new InvalidOperationException("Stem separation is not configured.");
        }

        try
        {
            await stemSeparationStageHandler.HandleAsync(
                new StemSeparationStageRequest(
                    currentState.ProjectState.Project.Id,
                    mediaAsset,
                    sourceAudioArtifact,
                    currentState.ProjectState.Artifacts,
                    preferredModelAlias,
                    modelPreferences?.GetPreferredExecutionProvider(RuntimeStage.Separation),
                    modelPreferences?.RequiresPreferredExecutionProvider(RuntimeStage.Separation) == true,
                    modelPreferences?.GetPreferredModelVariantAlias(RuntimeStage.Separation)),
                progress,
                cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not TaskCanceledException)
        {
            await WriteDegradationAsync(
                DialogueIsolationUnavailable,
                new DegradationTarget(currentState.ProjectState.Project.Id, mediaAsset.Id),
                ex.Message,
                CancellationToken.None).ConfigureAwait(false);
            throw;
        }

        // Stems feed voice-clone references and the mix bed only; transcript stages always
        // route from the full mix, so a separation rerun never invalidates the transcript.
        return await ReloadAsync(currentState.SelectedTranslationTargetLanguage, cancellationToken).ConfigureAwait(false);
    }

    public async Task<TranscriptProjectState> RunSpeechAudioPreparationAsync(
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress = null)
    {
        TranscriptProjectState currentState = await OpenAsync(cancellationToken).ConfigureAwait(false);
        MediaAsset mediaAsset = TranscriptWorkflowUtilities.GetRequiredMediaAsset(currentState);
        IReadOnlyList<ProjectArtifact> artifacts = currentState.ProjectState.Artifacts;
        PipelineProgressReporter.Started(progress, StageNames.SpeechEnhancement, "Starting speech audio cleanup.");
        PipelineProgressReporter.Phase(progress, StageNames.SpeechEnhancement, "Preparing audio", "Ensuring normalized audio for cleanup.");
        ProjectArtifact normalizedAudio = await EnsureNormalizedAudioForPipelineAsync(
            mediaAsset,
            artifacts,
            cancellationToken).ConfigureAwait(false);

        PipelineProgressReporter.Phase(progress, StageNames.SpeechEnhancement, "Processing speech audio", "Running speech audio preparation.");

        // Dedicated stage entry point: callers invoke this only when the engine's resume
        // gate declined prior artifacts or the run was explicitly forced, so existing
        // processed output must not be silently reused here.
        DateTimeOffset stageWorkStartedUtc = DateTimeOffset.UtcNow;
        await TryPrepareSpeechAudioAsync(
            new TranscriptAudioRoutingRequest
            {
                ProjectId = currentState.ProjectState.Project.Id,
                MediaAsset = mediaAsset,
                NormalizedAudioArtifact = normalizedAudio,
                ExistingArtifacts = artifacts,
                StageRuns = currentState.StageRuns,
                CancellationToken = cancellationToken,
                PrepareSpeechAudio = true,
                AllowExistingReuse = false,
            }).ConfigureAwait(false);

        TranscriptProjectState refreshed = await stateService
            .RefreshArtifactsAndStageRunsAsync(currentState, cancellationToken)
            .ConfigureAwait(false);

        // The fallback path (preparation handler unavailable) records no AudioPreparation
        // run of its own; persist a partial marker so the stage outcome reflects what
        // happened instead of surfacing a stale earlier run.
        StageRunRecord? latestPreparationRun = refreshed.StageRuns
            .Where(run =>
                string.Equals(run.StageName, StageNames.AudioPreparation, StringComparison.OrdinalIgnoreCase) &&
                run.StartedAtUtc >= stageWorkStartedUtc)
            .OrderByDescending(run => run.StartedAtUtc)
            .FirstOrDefault();
        bool needsFallbackRun = latestPreparationRun is null
            or { Status: StageRunStatus.Failed };
        if (needsFallbackRun && stageRunStore is not null)
        {
            StageRunRecord fallbackRun = await StageRunHelper
                .StartAsync(stageRunStore, currentState.ProjectState.Project.Id, StageNames.AudioPreparation, cancellationToken)
                .ConfigureAwait(false);
            await StageRunHelper
                .PartiallyCompleteAsync(
                    stageRunStore,
                    fallbackRun,
                    runtimeReporter: null,
                    "Speech audio preparation was unavailable; continuing with available source audio.",
                    CancellationToken.None)
                .ConfigureAwait(false);
            refreshed = await stateService
                .RefreshArtifactsAndStageRunsAsync(refreshed, cancellationToken)
                .ConfigureAwait(false);
        }

        PipelineProgressReporter.Completed(progress, StageNames.SpeechEnhancement, TimeSpan.Zero, "Speech audio cleanup finished.");
        return refreshed;
    }

    public async Task<TranscriptProjectState> RunOverlapRescueAsync(
        CancellationToken cancellationToken,
        IProgress<OverlapRescueProgress>? progress = null,
        string? preferredModelAlias = null,
        InferenceModelPreferences? modelPreferences = null,
        bool retranscribeCandidates = false)
    {
        if (overlapRescueWorkflow is null)
        {
            throw new InvalidOperationException("Overlap speech rescue is not configured.");
        }

        TranscriptProjectState currentState = await OpenAsync(cancellationToken).ConfigureAwait(false);
        await overlapRescueWorkflow
            .RunAsync(
                currentState,
                cancellationToken,
                progress,
                preferredModelAlias,
                modelPreferences,
                retranscribeCandidates)
            .ConfigureAwait(false);

        return await ReloadAsync(currentState.SelectedTranslationTargetLanguage, cancellationToken).ConfigureAwait(false);
    }

    private static readonly PreparationDegradation DialogueIsolationUnavailable = new()
    {
        StageName = StageNames.Separation,
        Code = DialogueIsolationUnavailableCode,
        Message = DialogueIsolationUnavailableMessage,
        SelectedFallback = "raw-audio",
        RecommendedAction = "Set up a separation model and regenerate stems.",
    };

    private static readonly PreparationDegradation PreparationFailed = new()
    {
        StageName = StageNames.AudioPreparation,
        Code = "SPEECH_PREPARATION_FAILED",
        Message = "Speech audio preparation was unavailable or failed; the unprocessed audio will be used for transcription.",
        SelectedFallback = "unprocessed-audio",
    };

    private static readonly PreparationDegradation EnhancementFailed = new()
    {
        StageName = StageNames.SpeechEnhancement,
        Code = "SPEECH_ENHANCEMENT_FAILED",
        Message = "Speech enhancement failed; the unenhanced audio will be used for transcription.",
        SelectedFallback = "unenhanced-audio",
    };

    private async Task WriteDegradationAsync(
        PreparationDegradation degradation,
        DegradationTarget target,
        string? detail,
        CancellationToken cancellationToken)
    {
        if (degradationWriter is null)
        {
            return;
        }

        try
        {
            await degradationWriter.WriteAsync(
                new PipelineDegradationRecord(
                    degradation.StageName,
                    degradation.Code,
                    degradation.Message,
                    Detail: detail,
                    SelectedFallback: degradation.SelectedFallback,
                    RecommendedAction: degradation.RecommendedAction,
                    DateTimeOffset.UtcNow,
                    StageRunId: null),
                target.ProjectId,
                target.MediaAssetId,
                cancellationToken).ConfigureAwait(false);
        }
        catch (Exception) when (!cancellationToken.IsCancellationRequested)
        {
            // Degradation write is best-effort; failure must not abort the fallback.
        }
    }

    private Task<ProjectArtifact> EnsureNormalizedAudioForPipelineAsync(
        MediaAsset mediaAsset,
        IReadOnlyList<ProjectArtifact> artifacts,
        CancellationToken cancellationToken) =>
        projectMediaIngestService.EnsureNormalizedAudioAsync(mediaAsset, artifacts, cancellationToken);

    private async Task<PreparedTranscriptAudio> PrepareTranscriptAudioAsync(PreparedGenerationRequest request)
    {
        TranscriptProjectState projectState = await OpenAsync(request.CancellationToken).ConfigureAwait(false);
        MediaAsset mediaAsset = TranscriptWorkflowUtilities.GetRequiredMediaAsset(projectState);
        IReadOnlyList<ProjectArtifact> artifacts = projectState.ProjectState.Artifacts;
        PipelineProgressReporter.Phase(
            request.Options.Progress,
            request.StageName,
            "Preparing audio",
            $"Ensuring normalized audio for {request.AudioPurpose}.");
        ProjectArtifact normalizedAudioArtifact = await EnsureNormalizedAudioForPipelineAsync(
            mediaAsset,
            artifacts,
            request.CancellationToken).ConfigureAwait(false);
        PipelineProgressReporter.Phase(
            request.Options.Progress,
            request.StageName,
            request.Options.PrepareSpeechAudio ? "Routing speech audio" : "Using source audio",
            request.Options.PrepareSpeechAudio
                ? $"Preparing the best speech audio for {request.AudioPurpose}."
                : "Source cleanup is off; using normalized source audio.");
        var routingRequest = new TranscriptAudioRoutingRequest
        {
            ProjectId = projectState.ProjectState.Project.Id,
            MediaAsset = mediaAsset,
            NormalizedAudioArtifact = normalizedAudioArtifact,
            ExistingArtifacts = artifacts,
            StageRuns = projectState.StageRuns,
            CancellationToken = request.CancellationToken,
            PrepareSpeechAudio = request.Options.PrepareSpeechAudio
        };
        TranscriptAudioRoutingPlan routingPlan = await PrepareSpeechAudioRoutingAsync(routingRequest)
            .ConfigureAwait(false);

        return new PreparedTranscriptAudio(projectState, mediaAsset, normalizedAudioArtifact, routingPlan);
    }

    private Task<TranscriptAudioRoutingPlan> PrepareSpeechAudioRoutingAsync(TranscriptAudioRoutingRequest request)
    {
        if (!request.PrepareSpeechAudio)
        {
            return Task.FromResult(
                TranscriptAudioRoutingPlan.Raw(request.NormalizedAudioArtifact, SpeechAudioSourceKind.FullMix)
                    .WithUnprocessedAsrSource(request.NormalizedAudioArtifact));
        }

        return TryPrepareSpeechAudioAsync(request);
    }

    private async Task<TranscriptAudioRoutingPlan> TryPrepareSpeechAudioAsync(TranscriptAudioRoutingRequest request)
    {
        // Transcript stages route from the full mix, not a separated vocal stem. Optional audio
        // preparation may enhance and clean that mix before VAD, ASR, and diarization.

        // A SpeechEnhancedAudio artifact from a completed enhancement run stays eligible
        // as the preparation input either way; when reuse is suppressed it must not also
        // suppress a fresh enhancement pass.
        ProjectArtifact? existingEnhanced =
            TranscriptWorkflowUtilities.GetAcceptedSpeechEnhancedAudio(request.ExistingArtifacts, request.StageRuns);

        // If speech processed audio from a completed preparation run already exists, skip
        // both enhancement and preparation. Artifacts registered by a canceled or failed
        // producer are never reused, even when they are the newest rows.
        // Reuse is suppressed when the caller requires fresh evidence of the stage's work.
        ProjectArtifact? existingProcessed = request.AllowExistingReuse
            ? TranscriptWorkflowUtilities.GetAcceptedSpeechProcessedAudio(request.ExistingArtifacts, request.StageRuns)
            : null;
        if (existingProcessed is not null)
        {
            return BuildRoutingPlan(request, existingProcessed, enhancementResult: null, existingEnhanced);
        }

        SpeechAudioEnhancementStageResult? enhancementResult = await RunEnhancementAsync(
            request,
            existingEnhanced).ConfigureAwait(false);

        if (speechAudioPreparationStageHandler is null)
        {
            await WriteDegradationAsync(
                PreparationFailed,
                new DegradationTarget(request.ProjectId, request.MediaAsset.Id),
                "Speech audio preparation is not configured in this host.",
                request.CancellationToken).ConfigureAwait(false);
            return BuildRoutingPlan(
                request,
                ResolveFallbackSource(request, enhancementResult, existingEnhanced),
                enhancementResult,
                existingEnhanced);
        }

        ProjectArtifact prepNormalizedAudio = ResolveFallbackSource(request, enhancementResult, existingEnhanced);

        try
        {
            TranscriptAudioRoutingPlan preparedPlan = await speechAudioPreparationStageHandler
                .HandleAsync(
                    new SpeechAudioPreparationStageRequest(
                        request.ProjectId,
                        request.MediaAsset,
                        prepNormalizedAudio,
                        request.ExistingArtifacts),
                    request.CancellationToken)
                .ConfigureAwait(false);
            return ApplyStudioVoiceAsrRouting(
                preparedPlan.WithUnprocessedAsrSource(request.NormalizedAudioArtifact),
                enhancementResult,
                existingEnhanced,
                request.NormalizedAudioArtifact.Id);
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not TaskCanceledException)
        {
            await WriteDegradationAsync(
                PreparationFailed,
                new DegradationTarget(request.ProjectId, request.MediaAsset.Id),
                ex.Message,
                request.CancellationToken).ConfigureAwait(false);
            return BuildRoutingPlan(
                request,
                ResolveFallbackSource(request, enhancementResult, existingEnhanced),
                enhancementResult,
                existingEnhanced);
        }
    }

    private async Task<SpeechAudioEnhancementStageResult?> RunEnhancementAsync(
        TranscriptAudioRoutingRequest request,
        ProjectArtifact? existingEnhanced)
    {
        if (speechAudioEnhancementStageHandler is null)
        {
            return null;
        }

        if (!ShouldRunEnhancement(existingEnhanced, request.AllowExistingReuse))
        {
            return null;
        }

        try
        {
            return await speechAudioEnhancementStageHandler
                .HandleAsync(
                    new SpeechAudioEnhancementStageRequest(
                        request.ProjectId,
                        request.MediaAsset,
                        request.NormalizedAudioArtifact,
                        request.ExistingArtifacts),
                    request.CancellationToken)
                .ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not TaskCanceledException)
        {
            await WriteDegradationAsync(
                EnhancementFailed,
                new DegradationTarget(request.ProjectId, request.MediaAsset.Id),
                ex.Message,
                request.CancellationToken).ConfigureAwait(false);
            return null;
        }
    }

    private static bool ShouldRunEnhancement(ProjectArtifact? existingEnhanced, bool allowExistingReuse)
    {
        if (existingEnhanced is null)
        {
            return true;
        }

        return !allowExistingReuse;
    }

    private static ProjectArtifact ResolveFallbackSource(
        TranscriptAudioRoutingRequest request,
        SpeechAudioEnhancementStageResult? enhancementResult,
        ProjectArtifact? existingEnhanced) =>
        enhancementResult?.EnhancedAudioArtifact
            ?? existingEnhanced
            ?? request.NormalizedAudioArtifact;

    private static TranscriptAudioRoutingPlan BuildRoutingPlan(
        TranscriptAudioRoutingRequest request,
        ProjectArtifact source,
        SpeechAudioEnhancementStageResult? enhancementResult,
        ProjectArtifact? existingEnhanced) =>
        ApplyStudioVoiceAsrRouting(
            TranscriptAudioRoutingPlan.Raw(source, SpeechAudioSourceKind.FullMix)
                .WithUnprocessedAsrSource(request.NormalizedAudioArtifact),
            enhancementResult,
            existingEnhanced,
            request.NormalizedAudioArtifact.Id);

    /// <summary>
    /// Routes ASR to the Studio Voice artifact only when the native effect produced it. A fresh
    /// enhancement result is authoritative; a reused artifact must still carry matching source
    /// lineage so a changed source cannot transcribe stale enhanced audio.
    /// </summary>
    private static TranscriptAudioRoutingPlan ApplyStudioVoiceAsrRouting(
        TranscriptAudioRoutingPlan plan,
        SpeechAudioEnhancementStageResult? enhancementResult,
        ProjectArtifact? existingEnhanced,
        Guid sourceArtifactId)
    {
        if (enhancementResult is not null)
        {
            return plan.WithSpeechEnhancementAsrSource(
                enhancementResult.EnhancedAudioArtifact,
                enhancementResult.Backend,
                enhancementResult.BackendProfile);
        }

        return existingEnhanced is not null
            ? plan.WithStudioVoiceAsrSourceIfApplicable(existingEnhanced, [sourceArtifactId])
            : plan;
    }

    private sealed record TranscriptAudioRoutingRequest
    {
        public required Guid ProjectId { get; init; }

        public required MediaAsset MediaAsset { get; init; }

        public required ProjectArtifact NormalizedAudioArtifact { get; init; }

        public required IReadOnlyList<ProjectArtifact> ExistingArtifacts { get; init; }

        public IReadOnlyList<StageRunRecord>? StageRuns { get; init; }

        public required CancellationToken CancellationToken { get; init; }

        public required bool PrepareSpeechAudio { get; init; }

        public bool AllowExistingReuse { get; init; } = true;
    }

    private sealed record PreparedGenerationRequest
    {
        public required bool IsInitial { get; init; }

        public required string StageName { get; init; }

        public required string AudioPurpose { get; init; }

        public required TranscriptRunOptions Options { get; init; }

        public required CancellationToken CancellationToken { get; init; }
    }

    private sealed record PreparationDegradation
    {
        public required string StageName { get; init; }

        public required string Code { get; init; }

        public required string Message { get; init; }

        public required string SelectedFallback { get; init; }

        public string? RecommendedAction { get; init; }
    }

    private sealed record DegradationTarget(Guid ProjectId, Guid MediaAssetId);

    private sealed record PreparedTranscriptAudio(
        TranscriptProjectState ProjectState,
        MediaAsset MediaAsset,
        ProjectArtifact NormalizedAudioArtifact,
        TranscriptAudioRoutingPlan AudioRoutingPlan);
}
