using Trackdub.Application.Pipeline;
using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Contracts;
using Trackdub.Contracts.Translation;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Projects;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Projects;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace Trackdub.Application.Transcripts;

public sealed partial class TranslationOrchestrationService(
    ITranslationRepository translationRepository,
    GlossaryService glossaryService,
    IGlossaryTermMatcher glossaryTermMatcher,
    ITranslationLanguageRouter translationLanguageRouter,
    ITranslationEngine translationEngine,
    ITtsTakeRepository ttsTakeRepository,
    IProjectStageRunStore stageRunStore,
    IArtifactStore artifactStore,
    TranscriptArtifactWriter artifactWriter,
    PipelineDegradationWriter? degradationWriter = null,
    ITranslatedWordAlignmentService? translatedWordAlignmentService = null,
    ILogger<TranslationOrchestrationService>? logger = null,
    IApplicationLogger? applicationLogger = null,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null,
    IAtomicRevisionCommitBoundary? commitBoundary = null,
    ITranslationModelProvisioner? modelProvisioner = null)
{
    // Sentinel language code meaning "follow the transcript's persisted or detected language".
    private const string AutoLanguageCode = "auto";

    // Serializes manifest read-modify-write operations to prevent concurrent language-change calls
    // from interleaving and clobbering each other's writes.
    private readonly SemaphoreSlim _manifestLock = new(1, 1);

    private readonly ITranslationRepository translationRepository = translationRepository ?? throw new ArgumentNullException(nameof(translationRepository));
    private readonly GlossaryService glossaryService = glossaryService ?? throw new ArgumentNullException(nameof(glossaryService));
    private readonly IGlossaryTermMatcher glossaryTermMatcher = glossaryTermMatcher ?? throw new ArgumentNullException(nameof(glossaryTermMatcher));
    private readonly ITranslationLanguageRouter translationLanguageRouter = translationLanguageRouter ?? throw new ArgumentNullException(nameof(translationLanguageRouter));
    private readonly ITranslationEngine translationEngine = translationEngine ?? throw new ArgumentNullException(nameof(translationEngine));
    private readonly ITranslatedWordAlignmentService translatedWordAlignmentService = translatedWordAlignmentService ?? new UnavailableTranslatedWordAlignmentService();
    private readonly ITtsTakeRepository ttsTakeRepository = ttsTakeRepository ?? throw new ArgumentNullException(nameof(ttsTakeRepository));
    private readonly IProjectStageRunStore stageRunStore = stageRunStore ?? throw new ArgumentNullException(nameof(stageRunStore));
    private readonly IArtifactStore artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));
    private readonly TranscriptArtifactWriter artifactWriter = artifactWriter ?? throw new ArgumentNullException(nameof(artifactWriter));
    private readonly ILogger<TranslationOrchestrationService> logger = logger ?? NullLogger<TranslationOrchestrationService>.Instance;
    private readonly IApplicationLogger? applicationLogger = applicationLogger;

    public async Task SetTranscriptLanguageAsync(
        TranscriptProjectState currentState,
        SetTranscriptLanguageRequest request,
        CancellationToken cancellationToken)
    {
        string? transcriptLanguage = TranscriptWorkflowUtilities.NormalizeTranscriptLanguageCode(request.TranscriptLanguage);
        if (string.Equals(currentState.TranscriptLanguage, transcriptLanguage, StringComparison.Ordinal))
        {
            return;
        }

        // Serialize the manifest read-modify-write so concurrent language-change calls cannot
        // interleave their reads and writes and lose each other's updates.
        await _manifestLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ProjectManifest manifest = await ReadProjectManifestAsync(
                currentState.ProjectState.Project,
                currentState.TranscriptLanguage,
                cancellationToken).ConfigureAwait(false);
            string? selectedTranslationTargetLanguage =
                TranscriptWorkflowUtilities.NormalizeTranslationTargetLanguageCodeOrNull(
                    request.SelectedTranslationTargetLanguage);
            ProjectUiSettings uiSettings = (manifest.UiSettings ?? new ProjectUiSettings()).Normalize() with
            {
                SelectedTranslationTargetLanguage = selectedTranslationTargetLanguage
                    ?? manifest.UiSettings?.SelectedTranslationTargetLanguage,
            };

            await artifactStore.WriteJsonAsync(
                ProjectArtifactPaths.ManifestRelativePath,
                manifest
                    .WithTranscriptLanguage(transcriptLanguage)
                    .WithUiSettings(uiSettings),
                cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _manifestLock.Release();
        }
    }

    /// <summary>
    /// Resolves the effective source language for a bulk translation run.
    /// A null, empty, or "auto" request follows project truth: the transcript language persisted
    /// at ASR time, falling back to the dominant per-segment detected language (persisting it so
    /// later stages and reloads see the same value). An explicit code must match the persisted
    /// transcript language; when none is persisted yet, the explicit code is adopted and persisted.
    /// </summary>
    private async Task<string> ResolveSourceLanguageAsync(
        TranscriptProjectState currentState,
        string? requestedSourceLanguage,
        CancellationToken cancellationToken)
    {
        string? requested = TranscriptWorkflowUtilities.NormalizeTranscriptLanguageCode(requestedSourceLanguage);
        if (IsExplicitLanguage(requested))
        {
            return await ResolveExplicitSourceLanguageAsync(currentState, requested!, cancellationToken).ConfigureAwait(false);
        }

        return await ResolveProjectSourceLanguageAsync(currentState, cancellationToken).ConfigureAwait(false);
    }

    private static bool IsExplicitLanguage(string? requested)
    {
        if (requested is null)
        {
            return false;
        }

        return !string.Equals(requested, AutoLanguageCode, StringComparison.Ordinal);
    }

    private async Task<string> ResolveExplicitSourceLanguageAsync(
        TranscriptProjectState currentState,
        string requested,
        CancellationToken cancellationToken)
    {
        string? transcriptLanguage = TranscriptWorkflowUtilities.NormalizeTranscriptLanguageCode(currentState.TranscriptLanguage);
        if (transcriptLanguage is null)
        {
            await SetTranscriptLanguageAsync(
                currentState,
                new SetTranscriptLanguageRequest(requested),
                cancellationToken).ConfigureAwait(false);
            return requested;
        }

        if (!string.Equals(transcriptLanguage, requested, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"Requested source language '{requested}' does not match the transcript language '{transcriptLanguage}'. " +
                "Change the transcript language first or request translation with the matching source language.");
        }

        return requested;
    }

    private async Task<string> ResolveProjectSourceLanguageAsync(
        TranscriptProjectState currentState,
        CancellationToken cancellationToken)
    {
        string? transcriptLanguage = TranscriptWorkflowUtilities.NormalizeTranscriptLanguageCode(currentState.TranscriptLanguage);
        if (IsUsableLanguage(transcriptLanguage))
        {
            return transcriptLanguage!;
        }

        string? detectedLanguage = TranscriptWorkflowUtilities.ResolveDetectedTranscriptLanguage(currentState.TranscriptSegments);
        if (IsUsableLanguage(detectedLanguage))
        {
            await SetTranscriptLanguageAsync(
                currentState,
                new SetTranscriptLanguageRequest(detectedLanguage!),
                cancellationToken).ConfigureAwait(false);
            return detectedLanguage!;
        }

        throw new InvalidOperationException(
            "Source language is unknown: no transcript language is set and ASR did not detect one. " +
            "Set the transcript language or request translation with an explicit source language.");
    }

    private static bool IsUsableLanguage(string? language)
    {
        if (language is null)
        {
            return false;
        }

        return !string.Equals(language, AutoLanguageCode, StringComparison.Ordinal);
    }

    public async Task SetSelectedTranslationTargetLanguageAsync(
        TranscriptProjectState currentState,
        string? targetLanguage,
        CancellationToken cancellationToken)
    {
        string? normalizedTargetLanguage =
            TranscriptWorkflowUtilities.NormalizeTranslationTargetLanguageCodeOrNull(targetLanguage);
        ProjectUiSettings currentUiSettings = currentState.ProjectState.UiSettings?.Normalize() ?? new ProjectUiSettings();

        await _manifestLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ProjectManifest manifest = await ReadProjectManifestAsync(
                currentState.ProjectState.Project,
                currentState.TranscriptLanguage,
                cancellationToken).ConfigureAwait(false);
            // Compare against the manifest under the lock: the loaded state can be stale if another
            // session changed the target after it was read.
            ProjectUiSettings baseSettings = (manifest.UiSettings ?? currentUiSettings).Normalize();
            if (string.Equals(baseSettings.SelectedTranslationTargetLanguage, normalizedTargetLanguage, StringComparison.Ordinal))
            {
                return;
            }

            ProjectUiSettings uiSettings = baseSettings with
            {
                SelectedTranslationTargetLanguage = normalizedTargetLanguage,
            };

            await artifactStore.WriteJsonAsync(
                ProjectArtifactPaths.ManifestRelativePath,
                manifest.WithUiSettings(uiSettings),
                cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _manifestLock.Release();
        }
    }

    public Task GenerateTranslationAsync(
        TranscriptProjectState currentState,
        GenerateTranslationRequest request,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress = null) =>
        GenerateTranslationAsync(
            currentState,
            request,
            new TranslationRunProgress(progress, SegmentObserver: null),
            cancellationToken);

    internal async Task GenerateTranslationAsync(
        TranscriptProjectState currentState,
        GenerateTranslationRequest request,
        TranslationRunProgress runProgress,
        CancellationToken cancellationToken)
    {
        DateTimeOffset progressStartedAt = DateTimeOffset.UtcNow;
        PipelineProgressReporter.Started(runProgress.Progress, StageNames.Translation, phase: "Preparing");

        TranscriptRevision currentTranscriptRevision = TranscriptWorkflowUtilities.GetRequiredTranscriptRevision(currentState);
        TranslationRunPreparation? preparation =
            await PrepareTranslationRunAsync(currentState, request, cancellationToken).ConfigureAwait(false);
        if (preparation is null)
        {
            await SetSelectedTranslationTargetLanguageAsync(currentState, request.TargetLanguage, cancellationToken).ConfigureAwait(false);
            await RecordSkippedTranslationAsync(currentState, runProgress.Progress, progressStartedAt, cancellationToken).ConfigureAwait(false);
            return;
        }

        (string sourceLanguage, string targetLanguage, TranslationScopePlan scopePlan) = preparation;
        TranslationRouteSelection route = await ResolveTranslationRouteAsync(
            new TranslationRouteRequest(sourceLanguage, targetLanguage, request.PreferredModelAlias),
            runProgress.Progress,
            cancellationToken).ConfigureAwait(false);
        StageRunRecord translationStageRun = await StageRunHelper
            .StartAsync(stageRunStore, currentState.ProjectState.Project.Id, StageNames.Translation, cancellationToken)
            .ConfigureAwait(false);

        var input = new TranslationGenerationInput(
            currentState,
            request,
            currentTranscriptRevision,
            scopePlan,
            route,
            translationStageRun,
            sourceLanguage,
            targetLanguage);

        try
        {
            await ExecuteTranslationAsync(input, runProgress, cancellationToken).ConfigureAwait(false);
        }
        // Only the caller's token means user cancellation. A TaskCanceledException from a
        // cloud-engine timeout must fall through to the failure arm, not read as "canceled".
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            await StageRunHelper
                .CancelAsync(stageRunStore, translationStageRun, translationEngine, "Translation canceled.", CancellationToken.None, runtimePlanningPreferences)
                .ConfigureAwait(false);
            PipelineProgressReporter.Failed(
                runProgress.Progress,
                StageNames.Translation,
                "Translation canceled.",
                DateTimeOffset.UtcNow - progressStartedAt);
            throw;
        }
        catch (Exception ex)
        {
            await StageRunHelper
                .FailAsync(stageRunStore, translationStageRun, translationEngine, ex.Message, cancellationToken, runtimePlanningPreferences)
                .ConfigureAwait(false);
            PipelineProgressReporter.Failed(
                runProgress.Progress,
                StageNames.Translation,
                ex.Message,
                DateTimeOffset.UtcNow - progressStartedAt);
            throw;
        }

        PipelineProgressReporter.Completed(
            runProgress.Progress,
            StageNames.Translation,
            DateTimeOffset.UtcNow - progressStartedAt,
            "Translation finished.");
    }

    private async Task RecordSkippedTranslationAsync(
        TranscriptProjectState currentState,
        IProgress<PipelineProgressEvent>? progress,
        DateTimeOffset progressStartedAt,
        CancellationToken cancellationToken)
    {
        // Record a benign skipped run so the pipeline engine can bind this invocation to
        // a stage row instead of reporting STAGE_RUN_MISSING and blocking downstream stages.
        StageRunRecord skippedTranslationRun = await StageRunHelper
            .StartAsync(stageRunStore, currentState.ProjectState.Project.Id, StageNames.Translation, cancellationToken)
            .ConfigureAwait(false);
        await StageRunHelper
            .SkipAsync(stageRunStore, skippedTranslationRun, null, StageSkipReasonCodes.ExistingArtifactsValid, cancellationToken)
            .ConfigureAwait(false);
        PipelineProgressReporter.Skipped(progress, StageNames.Translation,
            "No translations need updating.", DateTimeOffset.UtcNow - progressStartedAt);
    }

    private async Task<TranslationRouteSelection> ResolveTranslationRouteAsync(
        TranslationRouteRequest routeRequest,
        IProgress<PipelineProgressEvent>? progress,
        CancellationToken cancellationToken)
    {
        PipelineProgressReporter.Phase(progress, StageNames.Translation, "Resolving model");
        string? provisionedAlias = modelProvisioner is null ? routeRequest.PreferredModelAlias :
            await modelProvisioner.EnsureAsync(
                routeRequest.SourceLanguage,
                routeRequest.TargetLanguage,
                routeRequest.PreferredModelAlias,
                cancellationToken,
                new TranslationPreparationProgress(progress)).ConfigureAwait(false);
        PipelineProgressReporter.Phase(progress, StageNames.Translation, "Initializing translation model");
        TranslationRouteSelection route = await translationLanguageRouter.ResolveRouteAsync(
            routeRequest.SourceLanguage,
            routeRequest.TargetLanguage,
            cancellationToken,
            provisionedAlias).ConfigureAwait(false);
        if (!route.IsAvailable)
        {
            throw new InvalidOperationException(
                route.UnavailableReason ??
                $"Translation route {routeRequest.SourceLanguage} -> {routeRequest.TargetLanguage} is not available.");
        }

        return route;
    }

    private async Task ExecuteTranslationAsync(
        TranslationGenerationInput input,
        TranslationRunProgress runProgress,
        CancellationToken cancellationToken)
    {
        TranslationRequest translationRequest = await BuildTranslationRequestAsync(input, runProgress, cancellationToken).ConfigureAwait(false);
        (TranslationStreamCommit? streamCommit, IReadOnlyList<TranslatedTextSegment> translatedTextSegments) =
            await TranslateAsync(input, translationRequest, runProgress, cancellationToken).ConfigureAwait(false);

        TranslationRevision translationRevision = CreateTranslationRevision(input);
        var buildRequest = new TranslatedSegmentBuildRequest(
            translationRevision,
            translatedTextSegments,
            input.CurrentState.TranscriptSegments
                .OrderBy(segment => segment.SegmentIndex)
                .ToDictionary(segment => segment.SegmentIndex));
        List<TranslatedSegment> translatedSegments = await BuildTranslatedSegmentsAsync(
            input,
            buildRequest,
            runProgress,
            cancellationToken).ConfigureAwait(false);
        IReadOnlyList<TranslatedSegment> revisionSegments = input.ScopePlan.BuildRevision(
            translationRevision.Id,
            translatedTextSegments,
            translatedSegments);

        PipelineProgressReporter.Phase(runProgress.Progress, StageNames.Translation, "Saving translation");
        await CommitTranslationRevisionAsync(
            input,
            new TranslationCommitRequest(translationRevision, revisionSegments, streamCommit),
            cancellationToken).ConfigureAwait(false);

        await ReportTranslationDegradationsAsync(input, translatedTextSegments, translatedSegments, cancellationToken).ConfigureAwait(false);
        await PersistTranslationProvenanceAsync(input, cancellationToken).ConfigureAwait(false);
        await CompleteTranslationRunAsync(input, cancellationToken).ConfigureAwait(false);
    }

    private async Task<TranslationRequest> BuildTranslationRequestAsync(
        TranslationGenerationInput input,
        TranslationRunProgress runProgress,
        CancellationToken cancellationToken)
    {
        PipelineProgressReporter.Phase(runProgress.Progress, StageNames.Translation, "Preparing segments");
        IReadOnlyList<GlossaryEntry> glossaryEntries = await glossaryService.GetMergedEntriesAsync(
            input.CurrentState.ProjectState.Project.Id,
            input.SourceLanguage,
            input.TargetLanguage,
            cancellationToken).ConfigureAwait(false);
        TranslationInputSegment[] translationInputSegments = input.ScopePlan.SelectedSegments
            .OrderBy(segment => segment.SegmentIndex)
            .Select(segment => new TranslationInputSegment(
                segment.SegmentIndex,
                segment.StartSeconds,
                segment.EndSeconds,
                segment.Text))
            .ToArray();
        IReadOnlyList<TranslationGlossaryHint> glossaryHints = glossaryTermMatcher.BuildHints(
            input.SourceLanguage,
            translationInputSegments,
            glossaryEntries);

        PipelineProgressReporter.Phase(
            runProgress.Progress,
            StageNames.Translation,
            "Translating",
            $"{translationInputSegments.Length} segment(s) queued.");
        return new TranslationRequest(
            input.SourceLanguage,
            input.TargetLanguage,
            translationInputSegments,
            PreferredModelAlias: input.Request.PreferredModelAlias,
            GlossaryHints: glossaryHints,
            PreferredExecutionProvider: input.Request.PreferredExecutionProvider?.ToString(),
            RequirePreferredExecutionProvider: input.Request.RequirePreferredExecutionProvider,
            PreferredModelVariantAlias: input.Request.PreferredModelVariantAlias);
    }

    private async Task<(TranslationStreamCommit? StreamCommit, IReadOnlyList<TranslatedTextSegment> Segments)> TranslateAsync(
        TranslationGenerationInput input,
        TranslationRequest translationRequest,
        TranslationRunProgress runProgress,
        CancellationToken cancellationToken)
    {
        if (!input.Request.EnableSegmentStreaming || translationEngine is not IStreamingTranslationEngine streamingEngine)
        {
            IReadOnlyList<TranslatedTextSegment> segments = await translationEngine.TranslateAsync(
                translationRequest,
                cancellationToken).ConfigureAwait(false);
            return (null, segments);
        }

        TranslationStreamCommit streamCommit = await CollectTranslationStreamAsync(
            new TranslationStreamRequest(
                streamingEngine,
                translationRequest,
                input.StageRun.Id,
                input.TranscriptRevision.Id,
                input.TargetLanguage,
                translationRequest.Segments.Count,
                runProgress.Progress,
                runProgress.Observer),
            cancellationToken).ConfigureAwait(false);
        return (streamCommit, streamCommit.Items.Select(item => item.Payload).ToArray());
    }

    private TranslationRevision CreateTranslationRevision(TranslationGenerationInput input)
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        TranslationExecutionMetadata? executionMetadata = GetTranslationExecutionMetadata(translationEngine);
        return TranslationRevision.Create(
            input.CurrentState.ProjectState.Project.Id,
            input.StageRun.Id,
            input.TranscriptRevision.Id,
            input.TargetLanguage,
            revisionNumber: 1,
            now,
            translationProvider: executionMetadata?.ProviderName ?? input.Route.ProviderName,
            modelId: executionMetadata?.ModelId ?? input.Route.ModelId,
            executionProvider: executionMetadata?.SelectedExecutionProvider,
            modelProvenance: executionMetadata?.ModelProvenance ?? input.Route.ModelProvenance);
    }

    private async Task<List<TranslatedSegment>> BuildTranslatedSegmentsAsync(
        TranslationGenerationInput input,
        TranslatedSegmentBuildRequest buildRequest,
        TranslationRunProgress runProgress,
        CancellationToken cancellationToken)
    {
        // Engine results for indices outside the transcript do not belong in the revision.
        TranslatedTextSegment[] orderedSegments = buildRequest.TextSegments
            .Where(segment => buildRequest.SourceSegmentsByIndex.ContainsKey(segment.Index))
            .OrderBy(static segment => segment.Index)
            .ToArray();
        PipelineProgressReporter.Phase(
            runProgress.Progress,
            StageNames.Translation,
            "Building revision",
            $"{orderedSegments.Length} segment(s) returned.");
        List<TranslatedSegment> translatedSegments = [];
        int processed = 0;
        foreach (TranslatedTextSegment segment in orderedSegments)
        {
            TranslatedSegment? translatedSegment = await BuildTranslatedSegmentAsync(
                input,
                buildRequest,
                segment,
                cancellationToken).ConfigureAwait(false);
            if (translatedSegment is not null)
            {
                translatedSegments.Add(translatedSegment);
            }

            processed++;
            PipelineProgressReporter.Determinate(
                runProgress.Progress,
                StageNames.Translation,
                processed,
                orderedSegments.Length,
                "Building revision",
                currentItemLabel: $"Segment {segment.Index}");
        }

        return translatedSegments;
    }

    private async Task<TranslatedSegment?> BuildTranslatedSegmentAsync(
        TranslationGenerationInput input,
        TranslatedSegmentBuildRequest buildRequest,
        TranslatedTextSegment segment,
        CancellationToken cancellationToken)
    {
        bool hasText = !string.IsNullOrWhiteSpace(segment.Text);
        bool hasSourceSegment = buildRequest.SourceSegmentsByIndex.TryGetValue(segment.Index, out TranscriptSegment? sourceSegment);
        string? text = hasText
            ? segment.Text
            : (hasSourceSegment
                ? sourceSegment!.Text
                : null);

        // If neither the engine output nor the source fallback produced usable text,
        // skip this segment rather than passing empty text to TranslatedSegment.Create,
        // which rejects null/empty/whitespace text with ArgumentException.
        if (string.IsNullOrWhiteSpace(text))
        {
            return null;
        }

        TranslatedSegment translatedSegment = TranslatedSegment.Create(
            buildRequest.Revision.Id,
            segment.Index,
            segment.StartSeconds,
            segment.EndSeconds,
            text,
            hasText && hasSourceSegment
                ? TranscriptWorkflowUtilities.ComputeSourceSegmentHash(sourceSegment!)
                : null) with
        { Origin = TranslationSegmentOrigin.Generated };

        if (hasText)
        {
            if (hasSourceSegment)
            {
                translatedSegment = await AlignTranslatedWordsAsync(
                    translatedSegment,
                    sourceSegment!,
                    new TranslationAlignmentOptions(
                        input.SourceLanguage,
                        input.TargetLanguage,
                        input.Request.PreferredModelAlias,
                        input.Request.PreferredExecutionProvider,
                        input.Request.RequirePreferredExecutionProvider),
                    cancellationToken).ConfigureAwait(false);
            }
        }

        return translatedSegment;
    }

    private async Task CommitTranslationRevisionAsync(
        TranslationGenerationInput input,
        TranslationCommitRequest commitRequest,
        CancellationToken cancellationToken)
    {
        IAtomicRevisionCommitBoundary boundary = commitBoundary
            ?? throw new InvalidOperationException("Atomic revision commit boundary is not configured.");
        await boundary.CommitRevisionAsync(
            new AtomicRevisionCommitRequest(
                commitRequest.Revision,
                input.ScopePlan.PreviousRevision?.Id,
                commitRequest.Segments,
                commitRequest.StreamCommit),
            (revision, ct) => artifactWriter.PrepareTranslationArtifactAsync(
                input.CurrentState.ProjectState.Project.Id,
                TranscriptWorkflowUtilities.GetRequiredMediaAsset(input.CurrentState),
                revision,
                commitRequest.Segments,
                input.StageRun.Id,
                "generated-translation",
                ct),
            cancellationToken).ConfigureAwait(false);
    }

    private async Task ReportTranslationDegradationsAsync(
        TranslationGenerationInput input,
        IReadOnlyList<TranslatedTextSegment> translatedTextSegments,
        IReadOnlyList<TranslatedSegment> translatedSegments,
        CancellationToken cancellationToken)
    {
        if (degradationWriter is null)
        {
            return;
        }

        await ReportEmptyOutputDegradationAsync(input, translatedTextSegments, cancellationToken).ConfigureAwait(false);
        await ReportDroppedSegmentDegradationAsync(input, translatedSegments, cancellationToken).ConfigureAwait(false);
    }

    private async Task ReportEmptyOutputDegradationAsync(
        TranslationGenerationInput input,
        IReadOnlyList<TranslatedTextSegment> translatedTextSegments,
        CancellationToken cancellationToken)
    {
        int[] emptyIndices = translatedTextSegments
            .Where(static s => string.IsNullOrWhiteSpace(s.Text))
            .Select(static s => s.Index)
            .ToArray();
        if (emptyIndices.Length == 0)
        {
            return;
        }

        // Aggregate all empty-output segments into a single degradation record so
        // long transcripts with widespread engine failures don't flood the artifact
        // store with one record per segment.
        string indexSummary = SummarizeIndices(emptyIndices);
        await TryWriteDegradationAsync(
            new PipelineDegradationRecord(
                StageNames.Translation,
                "TRANSLATION_EMPTY_OUTPUT",
                $"Translation engine returned empty output for {emptyIndices.Length} segment(s) (indices: {indexSummary}); source text used as fallback.",
                Detail: null,
                SelectedFallback: "source-text",
                RecommendedAction: "Review source text for the affected segments or try a different translation model.",
                DateTimeOffset.UtcNow,
                input.StageRun.Id),
            input.CurrentState,
            cancellationToken).ConfigureAwait(false);
    }

    private async Task ReportDroppedSegmentDegradationAsync(
        TranslationGenerationInput input,
        IReadOnlyList<TranslatedSegment> translatedSegments,
        CancellationToken cancellationToken)
    {
        // Identify requested segments that produced no revision entry: the engine omitted them,
        // or returned no text and no source-text fallback was available. These are distinct from
        // TRANSLATION_EMPTY_OUTPUT (engine empty but source fallback succeeded): here no text at
        // all could be produced, so downstream stages (TTS, alignment) will see a gap in indices.
        HashSet<int> builtIndices = translatedSegments
            .Select(static s => s.SegmentIndex)
            .ToHashSet();
        int[] droppedIndices = input.ScopePlan.SelectedSegments
            .Select(static s => s.SegmentIndex)
            .Where(i => !builtIndices.Contains(i))
            .ToArray();
        if (droppedIndices.Length == 0)
        {
            return;
        }

        string indexSummary = SummarizeIndices(droppedIndices);
        await TryWriteDegradationAsync(
            new PipelineDegradationRecord(
                StageNames.Translation,
                "TRANSLATION_SEGMENT_DROPPED",
                $"Translation produced no usable text for {droppedIndices.Length} segment(s) (indices: {indexSummary}); " +
                "the engine omitted them or returned empty output with no source-text fallback available. These segments are absent from the revision.",
                Detail: null,
                SelectedFallback: null,
                RecommendedAction: "Check that all transcript segment indices are present in the translation engine output and that the source transcript is not empty for these indices.",
                DateTimeOffset.UtcNow,
                input.StageRun.Id),
            input.CurrentState,
            cancellationToken).ConfigureAwait(false);
    }

    private Task PersistTranslationProvenanceAsync(
        TranslationGenerationInput input,
        CancellationToken cancellationToken)
    {
        int[] allSegmentIndices = input.CurrentState.TranscriptSegments
            .Select(static segment => segment.SegmentIndex)
            .ToArray();
        ProjectUiSettings updatedUiSettings = SegmentStageRunProvenanceStore.RecordTranslationRuns(
            input.CurrentState.ProjectUiSettings,
            allSegmentIndices,
            input.ScopePlan.SelectedSegments.Select(segment => segment.SegmentIndex).ToHashSet(),
            input.StageRun.Id,
            input.ScopePlan.PreviousRevision?.StageRunId) with
        {
            SelectedTranslationTargetLanguage = input.TargetLanguage,
        };
        return PersistTranslationUiSettingsAsync(input.CurrentState, updatedUiSettings, cancellationToken);
    }

    private async Task CompleteTranslationRunAsync(
        TranslationGenerationInput input,
        CancellationToken cancellationToken)
    {
        // Mark the run completed only after the revision, stale-marking, artifact, and
        // provenance persist — a Completed row must imply durable output.
        await StageRunHelper
            .CompleteAsync(stageRunStore, input.StageRun, translationEngine, cancellationToken, runtimePlanningPreferences)
            .ConfigureAwait(false);
    }

    public async Task RetranslateSegmentAsync(
        TranscriptProjectState currentState,
        RetranslateSegmentRequest request,
        CancellationToken cancellationToken)
    {
        RetranslationContext context = await PrepareRetranslationAsync(currentState, request, cancellationToken).ConfigureAwait(false);
        try
        {
            await ExecuteRetranslationAsync(context, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            await StageRunHelper
                .CancelAsync(stageRunStore, context.StageRun, translationEngine, "Translation canceled.", CancellationToken.None, runtimePlanningPreferences)
                .ConfigureAwait(false);
            throw;
        }
        catch (Exception ex)
        {
            await StageRunHelper
                .FailAsync(stageRunStore, context.StageRun, translationEngine, ex.Message, cancellationToken, runtimePlanningPreferences)
                .ConfigureAwait(false);
            throw;
        }
    }

    private async Task<RetranslationContext> PrepareRetranslationAsync(
        TranscriptProjectState currentState,
        RetranslateSegmentRequest request,
        CancellationToken cancellationToken)
    {
        TranscriptRevision currentTranscriptRevision = TranscriptWorkflowUtilities.GetRequiredTranscriptRevision(currentState);
        TranslationRevision currentTranslationRevision = currentState.CurrentTranslationRevision
            ?? throw new InvalidOperationException("Generate translation before re-translating a segment.");

        if (request.TranslationRevisionId != currentTranslationRevision.Id)
        {
            throw new InvalidOperationException("The selected translation revision is no longer current.");
        }

        string sourceLanguage = TranscriptWorkflowUtilities.NormalizeRequiredTranscriptLanguageCode(request.SourceLanguage);
        string targetLanguage = TranscriptWorkflowUtilities.NormalizeTranslationTargetLanguageCode(request.TargetLanguage);
        if (!string.Equals(currentTranslationRevision.TargetLanguage, targetLanguage, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                "The selected translation target language does not match the current translation revision.");
        }

        if (!string.Equals(currentState.TranscriptLanguage, sourceLanguage, StringComparison.Ordinal))
        {
            throw new InvalidOperationException("Set the transcript language before starting translation.");
        }

        TranscriptSegment sourceSegment = currentState.TranscriptSegments
            .FirstOrDefault(segment => segment.Id == request.SegmentId)
            ?? throw new InvalidOperationException("The selected transcript segment is no longer available.");

        TranslationRouteSelection route = await ResolveTranslationRouteAsync(
            new TranslationRouteRequest(sourceLanguage, targetLanguage, request.PreferredModelAlias),
            progress: null,
            cancellationToken).ConfigureAwait(false);
        StageRunRecord translationStageRun = await StageRunHelper
            .StartAsync(stageRunStore, currentState.ProjectState.Project.Id, StageNames.Translation, cancellationToken)
            .ConfigureAwait(false);

        return new RetranslationContext(
            currentState,
            request,
            currentTranscriptRevision,
            currentTranslationRevision,
            sourceSegment,
            route,
            translationStageRun,
            sourceLanguage,
            targetLanguage);
    }

    private async Task ExecuteRetranslationAsync(
        RetranslationContext context,
        CancellationToken cancellationToken)
    {
        (string replacementText, bool hasUsableEngineText) = await TranslateSegmentAsync(context, cancellationToken).ConfigureAwait(false);
        if (!hasUsableEngineText && degradationWriter is not null)
        {
            await TryWriteDegradationAsync(
                new PipelineDegradationRecord(
                    StageNames.Translation,
                    "TRANSLATION_EMPTY_OUTPUT",
                    $"Translation engine returned empty output for segment {context.SourceSegment.SegmentIndex}; source text used as fallback.",
                    Detail: null,
                    SelectedFallback: "source-text",
                    RecommendedAction: "Review source text for this segment or try a different translation model.",
                    DateTimeOffset.UtcNow,
                    context.StageRun.Id),
                context.CurrentState,
                cancellationToken).ConfigureAwait(false);
        }

        TranslationRevision translationRevision = await CreateRetranslationRevisionAsync(context, cancellationToken).ConfigureAwait(false);
        TranslatedSegment replacementSegment = await CreateReplacementSegmentAsync(
            context,
            translationRevision,
            new RetranslationText(replacementText, hasUsableEngineText),
            cancellationToken).ConfigureAwait(false);
        await CommitRetranslationAsync(context, translationRevision, replacementSegment, cancellationToken).ConfigureAwait(false);
    }

    private async Task<(string ReplacementText, bool HasUsableEngineText)> TranslateSegmentAsync(
        RetranslationContext context,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<TranslatedTextSegment> translatedTextSegments = await translationEngine.TranslateAsync(
            new TranslationRequest(
                context.SourceLanguage,
                context.TargetLanguage,
                [
                    new TranslationInputSegment(
                        context.SourceSegment.SegmentIndex,
                        context.SourceSegment.StartSeconds,
                        context.SourceSegment.EndSeconds,
                        context.SourceSegment.Text)
                ],
                PreferredModelAlias: context.Request.PreferredModelAlias,
                PreferredExecutionProvider: context.Request.PreferredExecutionProvider?.ToString(),
                RequirePreferredExecutionProvider: context.Request.RequirePreferredExecutionProvider,
                PreferredModelVariantAlias: context.Request.PreferredModelVariantAlias),
            cancellationToken).ConfigureAwait(false);

        TranslatedTextSegment? translatedTextSegment = translatedTextSegments
            .FirstOrDefault(segment => segment.Index == context.SourceSegment.SegmentIndex);
        bool hasUsableEngineText = !string.IsNullOrWhiteSpace(translatedTextSegment?.Text);
        string replacementText = hasUsableEngineText
            ? translatedTextSegment!.Text
            : context.SourceSegment.Text;
        return (replacementText, hasUsableEngineText);
    }

    private async Task<TranslationRevision> CreateRetranslationRevisionAsync(
        RetranslationContext context,
        CancellationToken cancellationToken)
    {
        int nextRevisionNumber = await translationRepository.GetNextRevisionNumberAsync(
            context.CurrentState.ProjectState.Project.Id,
            context.TargetLanguage,
            cancellationToken).ConfigureAwait(false);
        DateTimeOffset now = DateTimeOffset.UtcNow;
        TranslationExecutionMetadata? executionMetadata = GetTranslationExecutionMetadata(translationEngine);
        return TranslationRevision.Create(
            context.CurrentState.ProjectState.Project.Id,
            context.StageRun.Id,
            context.TranscriptRevision.Id,
            context.TargetLanguage,
            nextRevisionNumber,
            now,
            translationProvider: executionMetadata?.ProviderName ?? context.Route.ProviderName,
            modelId: executionMetadata?.ModelId ?? context.Route.ModelId,
            executionProvider: executionMetadata?.SelectedExecutionProvider,
            modelProvenance: executionMetadata?.ModelProvenance ?? context.Route.ModelProvenance);
    }

    private async Task<TranslatedSegment> CreateReplacementSegmentAsync(
        RetranslationContext context,
        TranslationRevision translationRevision,
        RetranslationText replacement,
        CancellationToken cancellationToken)
    {
        TranslatedSegment replacementSegment = TranslatedSegment.Create(
            translationRevision.Id,
            context.SourceSegment.SegmentIndex,
            context.SourceSegment.StartSeconds,
            context.SourceSegment.EndSeconds,
            replacement.Text,
            replacement.HasUsableEngineText
                ? TranscriptWorkflowUtilities.ComputeSourceSegmentHash(context.SourceSegment)
                : null) with
        { Origin = TranslationSegmentOrigin.Generated };
        if (!replacement.HasUsableEngineText)
        {
            return replacementSegment;
        }

        return await AlignTranslatedWordsAsync(
            replacementSegment,
            context.SourceSegment,
            new TranslationAlignmentOptions(
                context.SourceLanguage,
                context.TargetLanguage,
                context.Request.PreferredModelAlias,
                context.Request.PreferredExecutionProvider,
                context.Request.RequirePreferredExecutionProvider),
            cancellationToken).ConfigureAwait(false);
    }

    private async Task CommitRetranslationAsync(
        RetranslationContext context,
        TranslationRevision translationRevision,
        TranslatedSegment replacementSegment,
        CancellationToken cancellationToken)
    {
        // Use PatchRevisionAsync: only the single changed/new segment is passed; the repository
        // handles re-binding unchanged segments from the previous revision.  This avoids the O(n)
        // in-memory reconstruction for the common single-segment retranslation case.
        await translationRepository.PatchRevisionAsync(
            context.CurrentTranslationRevision, translationRevision, [replacementSegment], cancellationToken)
            .ConfigureAwait(false);

        // Read back the merged list so the artifact writer has the full set.
        IReadOnlyList<TranslatedSegment> translatedSegments = await translationRepository
            .GetSegmentsAsync(translationRevision.Id, cancellationToken)
            .ConfigureAwait(false);

        if (HasRetranslationChanged(context, replacementSegment))
        {
            await ttsTakeRepository.MarkBySegmentIndicesStaleAsync(
                context.CurrentState.ProjectState.Project.Id,
                new HashSet<int> { context.SourceSegment.SegmentIndex },
                cancellationToken).ConfigureAwait(false);
        }

        await artifactWriter.WriteTranslationArtifactAsync(
            context.CurrentState.ProjectState.Project.Id,
            TranscriptWorkflowUtilities.GetRequiredMediaAsset(context.CurrentState),
            translationRevision,
            translatedSegments,
            context.StageRun.Id,
            provenance: "single-segment-retranslation",
            cancellationToken).ConfigureAwait(false);

        await PersistRetranslationProvenanceAsync(context, cancellationToken).ConfigureAwait(false);

        // Complete the run only after the revision patch, stale-marking, artifact, and
        // provenance persist — a Completed row must imply durable output.
        StageRunRecord completedRun = await StageRunHelper
            .CompleteAsync(stageRunStore, context.StageRun, translationEngine, cancellationToken, runtimePlanningPreferences)
            .ConfigureAwait(false);
        applicationLogger?.LogInformation(
            PipelineRuntimeProvenanceFormatter.FormatStageSegmentLogLine(
                StageNames.Translation,
                context.SourceSegment.SegmentIndex,
                completedRun.RuntimeInfo));
    }

    private static bool HasRetranslationChanged(
        RetranslationContext context,
        TranslatedSegment replacementSegment)
    {
        TranslatedSegment? existingTargetSegment = context.CurrentState.TranslatedSegments
            .FirstOrDefault(segment => segment.SegmentIndex == context.SourceSegment.SegmentIndex);
        return existingTargetSegment is null
            || !string.Equals(existingTargetSegment.Text, replacementSegment.Text, StringComparison.Ordinal);
    }

    private Task PersistRetranslationProvenanceAsync(
        RetranslationContext context,
        CancellationToken cancellationToken)
    {
        int[] allSegmentIndices = context.CurrentState.TranscriptSegments
            .Select(static segment => segment.SegmentIndex)
            .ToArray();
        ProjectUiSettings updatedUiSettings = SegmentStageRunProvenanceStore.RecordTranslationRuns(
            context.CurrentState.ProjectUiSettings,
            allSegmentIndices,
            new HashSet<int> { context.SourceSegment.SegmentIndex },
            context.StageRun.Id,
            context.CurrentTranslationRevision.StageRunId);
        return PersistTranslationUiSettingsAsync(context.CurrentState, updatedUiSettings, cancellationToken);
    }

    private async Task TryWriteDegradationAsync(
        PipelineDegradationRecord record,
        TranscriptProjectState currentState,
        CancellationToken cancellationToken)
    {
        try
        {
            await degradationWriter!.WriteAsync(
                record,
                currentState.ProjectState.Project.Id,
                TranscriptWorkflowUtilities.GetRequiredMediaAsset(currentState).Id,
                cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Degradation write is best-effort; failure must not abort translation persistence.
        }
    }

    private Task PersistTranslationUiSettingsAsync(
        TranscriptProjectState currentState,
        ProjectUiSettings updatedUiSettings,
        CancellationToken cancellationToken) =>
        SegmentStageRunProvenanceStore.PersistUiSettingsAsync(
            artifactStore,
            currentState.ProjectState.Project,
            currentState.TranscriptLanguage,
            updatedUiSettings,
            cancellationToken);

    public async Task SaveTranslationEditsAsync(
        TranscriptProjectState currentState,
        SaveTranslationEditsRequest request,
        CancellationToken cancellationToken)
    {
        TranslationRevision currentTranslationRevision = currentState.CurrentTranslationRevision
            ?? throw new InvalidOperationException("The project does not contain a translation revision.");

        if (currentTranslationRevision.Id != request.TranslationRevisionId)
        {
            throw new InvalidOperationException("Translation edits were based on an out-of-date revision.");
        }

        string targetLanguage = TranscriptWorkflowUtilities.NormalizeTranslationTargetLanguageCode(
            request.TargetLanguage);
        if (!string.Equals(currentTranslationRevision.TargetLanguage, targetLanguage, StringComparison.Ordinal))
        {
            throw new InvalidOperationException("Translation edits were based on a different target language.");
        }

        // Trim before comparing so whitespace-only edits are treated as unchanged and do not
        // drop word alignment or mark TTS takes stale. Empty results are ignored entirely.
        Dictionary<int, string> replacements = request.Segments
            .Select(segment => (segment.SegmentIndex, Text: segment.Text.Trim()))
            .Where(entry => entry.Text.Length > 0)
            .ToDictionary(entry => entry.SegmentIndex, entry => entry.Text);
        int nextRevisionNumber = await translationRepository.GetNextRevisionNumberAsync(
            currentState.ProjectState.Project.Id,
            targetLanguage,
            cancellationToken).ConfigureAwait(false);

        TranslationRevision editedRevision = CreateEditedRevision(
            currentState,
            currentTranslationRevision,
            targetLanguage,
            nextRevisionNumber);
        HashSet<int> changedSegmentIndices = currentState.TranslatedSegments
            .Where(segment => replacements.TryGetValue(segment.SegmentIndex, out string? replacement) &&
                              !string.Equals(segment.Text, replacement, StringComparison.Ordinal))
            .Select(segment => segment.SegmentIndex)
            .ToHashSet();

        // Build only the changed segments; PatchRevisionAsync handles re-binding unchanged segments
        // from the previous revision, avoiding the O(n) full reconstruction when only a subset changed.
        TranslatedSegment[] changedSegmentObjects = currentState.TranslatedSegments
            .Where(segment => replacements.ContainsKey(segment.SegmentIndex))
            .Select(segment => CreateEditedTranslationSegment(
                editedRevision,
                segment,
                replacements[segment.SegmentIndex],
                changedSegmentIndices))
            .ToArray();

        await translationRepository.PatchRevisionAsync(
            currentTranslationRevision, editedRevision, changedSegmentObjects, cancellationToken)
            .ConfigureAwait(false);

        // Read back the merged list so the artifact writer has the full set.
        IReadOnlyList<TranslatedSegment> editedSegments = await translationRepository
            .GetSegmentsAsync(editedRevision.Id, cancellationToken)
            .ConfigureAwait(false);

        if (changedSegmentIndices.Count > 0)
        {
            await ttsTakeRepository.MarkBySegmentIndicesStaleAsync(
                currentState.ProjectState.Project.Id,
                changedSegmentIndices,
                cancellationToken).ConfigureAwait(false);
        }

        await artifactWriter.WriteTranslationArtifactAsync(
            currentState.ProjectState.Project.Id,
            TranscriptWorkflowUtilities.GetRequiredMediaAsset(currentState),
            editedRevision,
            editedSegments,
            stageRunId: null,
            provenance: "manual-edit",
            cancellationToken).ConfigureAwait(false);
    }

    private static TranslationRevision CreateEditedRevision(
        TranscriptProjectState currentState,
        TranslationRevision currentTranslationRevision,
        string targetLanguage,
        int nextRevisionNumber) =>
        TranslationRevision.Create(
            currentState.ProjectState.Project.Id,
            stageRunId: null,
            currentTranslationRevision.SourceTranscriptRevisionId,
            targetLanguage,
            nextRevisionNumber,
            DateTimeOffset.UtcNow,
            currentTranslationRevision.TranslationProvider,
            currentTranslationRevision.ModelId,
            currentTranslationRevision.ExecutionProvider,
            modelProvenance: currentTranslationRevision.ModelProvenance);

    public async Task RestoreTranslationRevisionAsync(
        TranscriptProjectState currentState,
        string targetLanguage,
        IReadOnlyList<TranslatedSegment> translatedSegments,
        CancellationToken cancellationToken)
    {
        TranslationRevision? sourceRevision = currentState.CurrentTranslationRevision;
        int nextRevisionNumber = await translationRepository.GetNextRevisionNumberAsync(
            currentState.ProjectState.Project.Id,
            targetLanguage,
            cancellationToken).ConfigureAwait(false);
        DateTimeOffset now = DateTimeOffset.UtcNow;
        TranslationRevision restoredRevision = TranslationRevision.Create(
            currentState.ProjectState.Project.Id,
            stageRunId: null,
            currentState.CurrentTranscriptRevision!.Id,
            targetLanguage,
            nextRevisionNumber,
            now,
            sourceRevision?.TranslationProvider,
            sourceRevision?.ModelId,
            sourceRevision?.ExecutionProvider,
            modelProvenance: sourceRevision?.ModelProvenance);

        TranslatedSegment[] restoredSegments = translatedSegments
            .OrderBy(segment => segment.SegmentIndex)
            .Select(segment => TranslatedSegment.Create(
                restoredRevision.Id,
                segment.SegmentIndex,
                segment.StartSeconds,
                segment.EndSeconds,
                segment.Text,
                segment.SourceSegmentHash,
                segment.Words) with
            { Origin = segment.Origin })
            .ToArray();

        await translationRepository.SaveRevisionAsync(restoredRevision, restoredSegments, cancellationToken).ConfigureAwait(false);
        await artifactWriter.WriteTranslationArtifactAsync(
            currentState.ProjectState.Project.Id,
            TranscriptWorkflowUtilities.GetRequiredMediaAsset(currentState),
            restoredRevision,
            restoredSegments,
            stageRunId: null,
            provenance: "history-restore",
            cancellationToken).ConfigureAwait(false);
    }

    private async Task<ProjectManifest> ReadProjectManifestAsync(
        TrackdubProject project,
        string? transcriptLanguage,
        CancellationToken cancellationToken)
    {
        ProjectManifest? manifest = await artifactStore.ReadJsonAsync<ProjectManifest>(
            ProjectArtifactPaths.ManifestRelativePath,
            cancellationToken).ConfigureAwait(false);
        return manifest ?? ProjectManifest.FromProject(project, transcriptLanguage);
    }

    /// <summary>
    /// Opt-in (<see cref="GenerateTranslationRequest.EnableSegmentStreaming"/>): drives a
    /// streaming translation engine through the bounded channel, retaining every item's
    /// envelope for final identity validation at the commit boundary. Persistence
    /// happens once after the complete stream; nothing here commits a partial revision.
    /// </summary>
    private static async Task<TranslationStreamCommit> CollectTranslationStreamAsync(
        TranslationStreamRequest request,
        CancellationToken cancellationToken)
    {
        string snapshotId = $"{request.SourceRevisionId:N}:{request.TargetLanguage}";
        List<PipelineStreamItem<TranslatedTextSegment>> collected = [];
        await BoundedPipelineRunner.RunAsync<TranslatedTextSegment>(
            new BoundedPipelineChannelOptions(itemCapacity: 8, byteCapacity: 4 * 1024 * 1024),
            async (channel, ct) =>
            {
                await foreach (var item in request.StreamingEngine.TranslateStreamAsync(
                    request.TranslationRequest, request.RunId, snapshotId, request.SourceRevisionId, ct).ConfigureAwait(false))
                {
                    await channel.WriteAsync(item, ct).ConfigureAwait(false);
                }
            },
            async (items, ct) =>
            {
                await foreach (var item in items.WithCancellation(ct).ConfigureAwait(false))
                {
                    // Retain the complete envelope. Revision acceptance belongs to the real commit boundary.
                    collected.Add(item);
                    request.SegmentObserver?.OnSegmentTranslated(item.Payload.Index, item.Payload.Text);
                    PipelineProgressReporter.Phase(request.Progress, StageNames.Translation, "Translating",
                        $"{collected.Count}/{request.InputSegmentCount} segment(s) translated.");
                }
            }, cancellationToken).ConfigureAwait(false);
        return new TranslationStreamCommit(request.RunId, snapshotId, request.SourceRevisionId,
            request.TranslationRequest.Segments.Select(segment => segment.Index).ToArray(), collected);
    }
}
