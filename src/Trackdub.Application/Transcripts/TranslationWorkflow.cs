namespace Trackdub.Application.Transcripts;

using Trackdub.Contracts.Pipeline;

public sealed class TranslationWorkflow(
    TranscriptProjectStateService stateService,
    TranslationOrchestrationService translationOrchestrationService)
{
    private readonly TranscriptProjectStateService stateService = stateService ?? throw new ArgumentNullException(nameof(stateService));
    private readonly TranslationOrchestrationService translationOrchestrationService = translationOrchestrationService ?? throw new ArgumentNullException(nameof(translationOrchestrationService));

    public async Task<TranscriptProjectState> SelectTranslationTargetAsync(
        SetTranslationTargetRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        TranscriptProjectState currentState = await OpenAsync(cancellationToken).ConfigureAwait(false);
        await translationOrchestrationService.SetSelectedTranslationTargetLanguageAsync(
            currentState,
            request.TargetLanguage,
            cancellationToken).ConfigureAwait(false);
        return await ReloadAsync(request.TargetLanguage, cancellationToken).ConfigureAwait(false);
    }

    public async Task<TranscriptProjectState> SetTranscriptLanguageAsync(
        SetTranscriptLanguageRequest request,
        CancellationToken cancellationToken)
    {
        TranscriptProjectState currentState = await OpenAsync(cancellationToken).ConfigureAwait(false);
        await translationOrchestrationService.SetTranscriptLanguageAsync(currentState, request, cancellationToken).ConfigureAwait(false);
        string? selectedTranslationTargetLanguage =
            request.SelectedTranslationTargetLanguage ?? currentState.SelectedTranslationTargetLanguage;
        return await ReloadAsync(selectedTranslationTargetLanguage, cancellationToken).ConfigureAwait(false);
    }

    public Task<TranscriptProjectState> GenerateTranslationAsync(
        GenerateTranslationRequest request,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress = null) =>
        GenerateTranslationAsync(request, cancellationToken, progress, segmentObserver: null);

    internal async Task<TranscriptProjectState> GenerateTranslationAsync(
        GenerateTranslationRequest request,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress,
        ITranslatedSegmentObserver? segmentObserver)
    {
        TranscriptProjectState currentState = await OpenAsync(cancellationToken).ConfigureAwait(false);
        await translationOrchestrationService
            .GenerateTranslationAsync(
                currentState,
                request,
                new TranslationOrchestrationService.TranslationRunProgress(progress, segmentObserver),
                cancellationToken)
            .ConfigureAwait(false);
        return await ReloadAsync(
            TranscriptWorkflowUtilities.NormalizeTranslationTargetLanguageCode(request.TargetLanguage),
            cancellationToken).ConfigureAwait(false);
    }

    public Task<TranscriptProjectState> RetranslateSegmentAsync(
        RetranslateSegmentRequest request,
        CancellationToken cancellationToken) =>
        ExecuteAsync(
            (state, ct) => translationOrchestrationService.RetranslateSegmentAsync(state, request, ct),
            TranscriptWorkflowUtilities.NormalizeTranslationTargetLanguageCode(request.TargetLanguage),
            cancellationToken);

    public Task<TranscriptProjectState> SaveTranslationEditsAsync(
        SaveTranslationEditsRequest request,
        CancellationToken cancellationToken) =>
        ExecuteAsync(
            (state, ct) => translationOrchestrationService.SaveTranslationEditsAsync(state, request, ct),
            TranscriptWorkflowUtilities.NormalizeTranslationTargetLanguageCode(request.TargetLanguage),
            cancellationToken);

    private async Task<TranscriptProjectState> ExecuteAsync(
        Func<TranscriptProjectState, CancellationToken, Task> action,
        string? targetLanguage,
        CancellationToken cancellationToken)
    {
        TranscriptProjectState currentState = await OpenAsync(cancellationToken).ConfigureAwait(false);
        await action(currentState, cancellationToken).ConfigureAwait(false);
        return await ReloadAsync(targetLanguage, cancellationToken).ConfigureAwait(false);
    }

    private Task<TranscriptProjectState> OpenAsync(CancellationToken cancellationToken) =>
        stateService.OpenAsync(requestedTranslationTargetLanguage: null, cancellationToken);

    private Task<TranscriptProjectState> ReloadAsync(
        string? requestedTranslationTargetLanguage,
        CancellationToken cancellationToken) =>
        stateService.OpenAsync(requestedTranslationTargetLanguage, cancellationToken);
}
