using Trackdub.Domain.Translation;

namespace Trackdub.Application.Transcripts;

public sealed partial class TranslationOrchestrationService
{
    private async Task<TranslationRunPreparation?> PrepareTranslationRunAsync(
        TranscriptProjectState currentState,
        GenerateTranslationRequest request,
        CancellationToken cancellationToken)
    {
        EnsureTranslationHasSourceSegments(currentState);
        string targetLanguage = TranscriptWorkflowUtilities.NormalizeTranslationTargetLanguageCode(request.TargetLanguage);
        TranslationScopePlan scopePlan = await PrepareTranslationScopePlanAsync(
            currentState,
            targetLanguage,
            request.Scope,
            cancellationToken).ConfigureAwait(false);
        if (scopePlan.SelectedSegments.Count == 0)
        {
            return null;
        }

        // Resolve the language only once there is work to do: a legacy project with no
        // detected language should report "nothing to update" rather than throw.
        string sourceLanguage = await ResolveSourceLanguageAsync(
            currentState,
            request.SourceLanguage,
            cancellationToken).ConfigureAwait(false);
        return new TranslationRunPreparation(sourceLanguage, targetLanguage, scopePlan);
    }

    private async Task<TranslationScopePlan> PrepareTranslationScopePlanAsync(
        TranscriptProjectState currentState,
        string targetLanguage,
        TranslationRunScope scope,
        CancellationToken cancellationToken)
    {
        TranslationRevision? previousRevision = await translationRepository.GetCurrentRevisionAsync(
            currentState.ProjectState.Project.Id,
            targetLanguage,
            cancellationToken).ConfigureAwait(false);
        IReadOnlyList<TranslatedSegment> previousSegments = await LoadTranslationSegmentsAsync(
            previousRevision,
            cancellationToken).ConfigureAwait(false);

        return TranslationScopePlanner.CreatePlan(
            scope,
            currentState.TranscriptSegments,
            previousSegments,
            previousRevision);
    }

    private Task<IReadOnlyList<TranslatedSegment>> LoadTranslationSegmentsAsync(
        TranslationRevision? revision,
        CancellationToken cancellationToken) =>
        revision is null
            ? Task.FromResult<IReadOnlyList<TranslatedSegment>>([])
            : translationRepository.GetSegmentsAsync(revision.Id, cancellationToken);

    private static void EnsureTranslationHasSourceSegments(TranscriptProjectState currentState)
    {
        if (currentState.TranscriptSegments.Count == 0)
        {
            throw new InvalidOperationException("The project's transcript revision has no segments to translate.");
        }
    }

    private static TranslatedSegment CreateEditedTranslationSegment(
        TranslationRevision editedRevision,
        TranslatedSegment segment,
        string replacement,
        IReadOnlySet<int> changedSegmentIndices)
    {
        bool textChanged = changedSegmentIndices.Contains(segment.SegmentIndex);
        return TranslatedSegment.Create(
            editedRevision.Id,
            segment.SegmentIndex,
            segment.StartSeconds,
            segment.EndSeconds,
            replacement,
            segment.SourceSegmentHash,
            textChanged ? [] : segment.Words) with
        { Origin = textChanged ? TranslationSegmentOrigin.Manual : segment.Origin };
    }

    private sealed record TranslationRunPreparation(
        string SourceLanguage,
        string TargetLanguage,
        TranslationScopePlan ScopePlan);
}
