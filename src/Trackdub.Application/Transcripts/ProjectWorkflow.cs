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
            createResult.Project.Id,
            createResult.MediaAsset,
            createResult.AudioArtifact,
            currentArtifacts,
            cancellationToken).ConfigureAwait(false);

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

    public async Task<TranscriptProjectState> RunInitialTranscriptionAsync(
        TranscriptRunOptions options,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(options);
        string? normalizedSourceLanguage = TranscriptWorkflowUtilities.NormalizeTranscriptLanguageCode(options.SourceLanguage);
        InferenceModelPreferences preferences = options.ModelPreferences ?? InferenceModelPreferences.Empty;
        PreparedTranscriptAudio audio = await PrepareTranscriptAudioAsync(
            StageNames.Asr,
            "transcription",
            options,
            cancellationToken).ConfigureAwait(false);

        await transcriptGenerationService.GenerateTranscriptAsync(
            audio.ProjectState.ProjectState.Project,
            audio.MediaAsset,
            audio.NormalizedAudioArtifact,
            audio.AudioRoutingPlan,
            options.EnableSpeakerDiarization,
            preferences,
            cancellationToken,
            normalizedSourceLanguage,
            options.Progress).ConfigureAwait(false);

        return await ReloadAsync(audio.ProjectState.SelectedTranslationTargetLanguage, cancellationToken).ConfigureAwait(false);
    }

    public async Task<TranscriptProjectState> RunTranscriptStageAsync(
        string stageName,
        TranscriptRunOptions options,
        CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(stageName);
        ArgumentNullException.ThrowIfNull(options);
        string? normalizedSourceLanguage = TranscriptWorkflowUtilities.NormalizeTranscriptLanguageCode(options.SourceLanguage);
        PreparedTranscriptAudio audio = await PrepareTranscriptAudioAsync(
            stageName,
            "this stage",
            options,
            cancellationToken).ConfigureAwait(false);

        await transcriptGenerationService.GenerateTranscriptStageAsync(
            audio.ProjectState.ProjectState.Project,
            audio.MediaAsset,
            audio.NormalizedAudioArtifact,
            audio.AudioRoutingPlan,
            stageName,
            options.EnableSpeakerDiarization,
            options.ModelPreferences ?? InferenceModelPreferences.Empty,
            cancellationToken,
            normalizedSourceLanguage,
            options.Progress).ConfigureAwait(false);

        return await ReloadAsync(audio.ProjectState.SelectedTranslationTargetLanguage, cancellationToken).ConfigureAwait(false);
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
            await WriteDialogueIsolationUnavailableAsync(
                currentState.ProjectState.Project.Id,
                mediaAsset.Id,
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
            currentState.ProjectState.Project.Id,
            mediaAsset,
            normalizedAudio,
            artifacts,
            cancellationToken,
            allowExistingReuse: false,
            stageRuns: currentState.StageRuns).ConfigureAwait(false);

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

    private async Task WriteDialogueIsolationUnavailableAsync(
        Guid projectId,
        Guid mediaAssetId,
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
                    StageNames.Separation,
                    DialogueIsolationUnavailableCode,
                    DialogueIsolationUnavailableMessage,
                    Detail: detail,
                    SelectedFallback: "raw-audio",
                    RecommendedAction: "Set up a separation model and regenerate stems.",
                    DateTimeOffset.UtcNow,
                    StageRunId: null),
                projectId,
                mediaAssetId,
                cancellationToken).ConfigureAwait(false);
        }
        catch (Exception) when (!cancellationToken.IsCancellationRequested)
        {
            // Degradation write is best-effort; failure must not abort the raw-audio fallback.
        }
    }

    private async Task WritePreparationFallbackAsync(
        Guid projectId,
        Guid mediaAssetId,
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
                    StageNames.AudioPreparation,
                    "SPEECH_PREPARATION_FAILED",
                    "Speech audio preparation was unavailable or failed; the unprocessed audio will be used for transcription.",
                    Detail: detail,
                    SelectedFallback: "unprocessed-audio",
                    RecommendedAction: null,
                    DateTimeOffset.UtcNow,
                    StageRunId: null),
                projectId,
                mediaAssetId,
                cancellationToken).ConfigureAwait(false);
        }
        catch (Exception) when (!cancellationToken.IsCancellationRequested)
        {
            // Degradation write is best-effort.
        }
    }

    private async Task WriteEnhancementFallbackAsync(
        Guid projectId,
        Guid mediaAssetId,
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
                    StageNames.SpeechEnhancement,
                    "SPEECH_ENHANCEMENT_FAILED",
                    "Speech enhancement failed; the unenhanced audio will be used for transcription.",
                    Detail: detail,
                    SelectedFallback: "unenhanced-audio",
                    RecommendedAction: null,
                    DateTimeOffset.UtcNow,
                    StageRunId: null),
                projectId,
                mediaAssetId,
                cancellationToken).ConfigureAwait(false);
        }
        catch (Exception) when (!cancellationToken.IsCancellationRequested)
        {
            // Degradation write is best-effort.
        }
    }

    private Task<ProjectArtifact> EnsureNormalizedAudioForPipelineAsync(
        MediaAsset mediaAsset,
        IReadOnlyList<ProjectArtifact> artifacts,
        CancellationToken cancellationToken) =>
        projectMediaIngestService.EnsureNormalizedAudioAsync(mediaAsset, artifacts, cancellationToken);

    private async Task<PreparedTranscriptAudio> PrepareTranscriptAudioAsync(
        string stageName,
        string audioPurpose,
        TranscriptRunOptions options,
        CancellationToken cancellationToken)
    {
        TranscriptProjectState projectState = await OpenAsync(cancellationToken).ConfigureAwait(false);
        MediaAsset mediaAsset = TranscriptWorkflowUtilities.GetRequiredMediaAsset(projectState);
        IReadOnlyList<ProjectArtifact> artifacts = projectState.ProjectState.Artifacts;
        PipelineProgressReporter.Phase(
            options.Progress,
            stageName,
            "Preparing audio",
            $"Ensuring normalized audio for {audioPurpose}.");
        ProjectArtifact normalizedAudioArtifact = await EnsureNormalizedAudioForPipelineAsync(
            mediaAsset,
            artifacts,
            cancellationToken).ConfigureAwait(false);
        PipelineProgressReporter.Phase(
            options.Progress,
            stageName,
            options.PrepareSpeechAudio ? "Routing speech audio" : "Using source audio",
            options.PrepareSpeechAudio
                ? $"Preparing the best speech audio for {audioPurpose}."
                : "Source cleanup is off; using normalized source audio.");
        var routingRequest = new TranscriptAudioRoutingRequest
        {
            ProjectId = projectState.ProjectState.Project.Id,
            MediaAsset = mediaAsset,
            NormalizedAudioArtifact = normalizedAudioArtifact,
            ExistingArtifacts = artifacts,
            StageRuns = projectState.StageRuns,
            CancellationToken = cancellationToken,
            PrepareSpeechAudio = options.PrepareSpeechAudio
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

        return TryPrepareSpeechAudioAsync(
            request.ProjectId,
            request.MediaAsset,
            request.NormalizedAudioArtifact,
            request.ExistingArtifacts,
            request.CancellationToken,
            stageRuns: request.StageRuns);
    }

    private async Task<TranscriptAudioRoutingPlan> TryPrepareSpeechAudioAsync(
        Guid projectId,
        MediaAsset mediaAsset,
        ProjectArtifact normalizedAudioArtifact,
        IReadOnlyList<ProjectArtifact> existingArtifacts,
        CancellationToken cancellationToken,
        bool allowExistingReuse = true,
        IReadOnlyList<StageRunRecord>? stageRuns = null)
    {
        // Transcript stages route from the full mix, not a separated vocal stem. Optional audio
        // preparation may enhance and clean that mix before VAD, ASR, and diarization.
        ProjectArtifact selectedSource = normalizedAudioArtifact;
        const SpeechAudioSourceKind sourceKind = SpeechAudioSourceKind.FullMix;

        // If speech processed audio from a completed preparation run already exists, skip
        // both enhancement and preparation. Artifacts registered by a canceled or failed
        // producer are never reused, even when they are the newest rows.
        // Reuse is suppressed when the caller requires fresh evidence of the stage's work.
        ProjectArtifact? existingProcessed = allowExistingReuse
            ? TranscriptWorkflowUtilities.GetAcceptedSpeechProcessedAudio(existingArtifacts, stageRuns)
            : null;
        if (existingProcessed is not null)
        {
            return TranscriptAudioRoutingPlan.Raw(existingProcessed, sourceKind)
                .WithUnprocessedAsrSource(normalizedAudioArtifact);
        }

        SpeechAudioEnhancementStageResult? enhancementResult = null;

        // A SpeechEnhancedAudio artifact from a completed enhancement run stays eligible
        // as the preparation input either way; when reuse is suppressed it must not also
        // suppress a fresh enhancement pass.
        ProjectArtifact? existingEnhanced =
            TranscriptWorkflowUtilities.GetAcceptedSpeechEnhancedAudio(existingArtifacts, stageRuns);
        if (speechAudioEnhancementStageHandler is not null && (existingEnhanced is null || !allowExistingReuse))
        {
            try
            {
                enhancementResult = await speechAudioEnhancementStageHandler
                    .HandleAsync(
                        new SpeechAudioEnhancementStageRequest(
                            projectId,
                            mediaAsset,
                            selectedSource,
                            existingArtifacts),
                        cancellationToken)
                    .ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is not OperationCanceledException and not TaskCanceledException)
            {
                await WriteEnhancementFallbackAsync(projectId, mediaAsset.Id, ex.Message, cancellationToken)
                    .ConfigureAwait(false);
            }
        }

        if (speechAudioPreparationStageHandler is null)
        {
            await WritePreparationFallbackAsync(
                projectId,
                mediaAsset.Id,
                "Speech audio preparation is not configured in this host.",
                cancellationToken).ConfigureAwait(false);
            ProjectArtifact fallbackSource = enhancementResult?.EnhancedAudioArtifact
                ?? existingEnhanced
                ?? selectedSource;
            return TranscriptAudioRoutingPlan.Raw(fallbackSource, sourceKind)
                .WithUnprocessedAsrSource(normalizedAudioArtifact);
        }

        ProjectArtifact prepNormalizedAudio = enhancementResult?.EnhancedAudioArtifact
            ?? existingEnhanced
            ?? normalizedAudioArtifact;

        try
        {
            TranscriptAudioRoutingPlan preparedPlan = await speechAudioPreparationStageHandler
                .HandleAsync(
                    new SpeechAudioPreparationStageRequest(
                        projectId,
                        mediaAsset,
                        prepNormalizedAudio,
                        existingArtifacts),
                    cancellationToken)
                .ConfigureAwait(false);
            return preparedPlan.WithUnprocessedAsrSource(normalizedAudioArtifact);
        }
        catch (Exception ex) when (ex is not OperationCanceledException and not TaskCanceledException)
        {
            await WritePreparationFallbackAsync(projectId, mediaAsset.Id, ex.Message, cancellationToken)
                .ConfigureAwait(false);
            ProjectArtifact fallbackSource = enhancementResult?.EnhancedAudioArtifact
                ?? existingEnhanced
                ?? selectedSource;
            return TranscriptAudioRoutingPlan.Raw(fallbackSource, sourceKind)
                .WithUnprocessedAsrSource(normalizedAudioArtifact);
        }
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
    }

    private sealed record PreparedTranscriptAudio(
        TranscriptProjectState ProjectState,
        MediaAsset MediaAsset,
        ProjectArtifact NormalizedAudioArtifact,
        TranscriptAudioRoutingPlan AudioRoutingPlan);
}
