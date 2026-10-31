using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;

namespace Trackdub.Application.Transcripts;

/// <summary>Selects translation inputs without replacing manual or legacy text implicitly.</summary>
public static class TranslationScopePlanner
{
    internal static TranslationScopePlan CreatePlan(
        TranslationRunScope scope,
        IReadOnlyList<TranscriptSegment> source,
        IReadOnlyList<TranslatedSegment> existing,
        TranslationRevision? previousRevision)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(existing);

        // Deleted transcript segments are renumbered, so a prior revision can still carry
        // translations for indices that no longer exist in the current source. Preserving
        // those would publish a revision with more segments than the transcript.
        HashSet<int> sourceIndices = source.Select(static segment => segment.SegmentIndex).ToHashSet();
        IReadOnlyList<TranslatedSegment> currentExisting = existing
            .Where(segment => sourceIndices.Contains(segment.SegmentIndex))
            .ToArray();
        return new(scope, Select(scope, source, currentExisting), currentExisting, previousRevision);
    }

    public static IReadOnlyList<TranscriptSegment> Select(
        TranslationRunScope scope,
        IReadOnlyList<TranscriptSegment> source,
        IReadOnlyList<TranslatedSegment> existing)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(existing);
        if (scope == TranslationRunScope.All)
        {
            return source.OrderBy(segment => segment.SegmentIndex).ToArray();
        }

        if (scope != TranslationRunScope.MissingOrOutdated)
        {
            throw new ArgumentOutOfRangeException(nameof(scope));
        }

        Dictionary<int, TranslatedSegment> byIndex = existing.ToDictionary(segment => segment.SegmentIndex);
        return source.Where(segment =>
            !byIndex.TryGetValue(segment.SegmentIndex, out TranslatedSegment? translated) ||
            (translated.Origin == TranslationSegmentOrigin.Generated &&
             !string.Equals(translated.SourceSegmentHash,
                 TranscriptWorkflowUtilities.ComputeSourceSegmentHash(segment), StringComparison.Ordinal)))
            .OrderBy(segment => segment.SegmentIndex)
            .ToArray();
    }
}

internal sealed record TranslationScopePlan(
    TranslationRunScope Scope,
    IReadOnlyList<TranscriptSegment> SelectedSegments,
    IReadOnlyList<TranslatedSegment> ExistingSegments,
    TranslationRevision? PreviousRevision)
{
    public IReadOnlyList<TranslatedSegment> BuildRevision(
        Guid revisionId,
        IReadOnlyList<TranslatedTextSegment> engineResults,
        IReadOnlyList<TranslatedSegment> generatedSegments)
    {
        if (Scope == TranslationRunScope.All)
        {
            return generatedSegments;
        }

        HashSet<int> selectedIndices = SelectedSegments.Select(static segment => segment.SegmentIndex).ToHashSet();
        HashSet<int> generatedIndices = generatedSegments.Select(static segment => segment.SegmentIndex).ToHashSet();

        // An engine result with no text that was only rescued by the source-text fallback means
        // the engine output is unreliable, so keep the existing translations. This applies only
        // where an existing translation would be lost; a fallback-only segment with nothing to
        // preserve is accepted. A segment with neither engine text nor a fallback is dropped; it
        // stays absent and is reported as TRANSLATION_SEGMENT_DROPPED while the rest of the run still commits.
        HashSet<int> existingIndices = ExistingSegments.Select(static segment => segment.SegmentIndex).ToHashSet();
        bool fallbackOnlyOutput = engineResults.Any(segment =>
            string.IsNullOrWhiteSpace(segment.Text) &&
            generatedIndices.Contains(segment.Index) &&
            existingIndices.Contains(segment.Index));
        if (fallbackOnlyOutput)
        {
            throw new InvalidDataException("Translation did not return every requested segment; existing translations were preserved.");
        }

        return generatedSegments
            .Concat(ExistingSegments
                .Where(segment => !selectedIndices.Contains(segment.SegmentIndex))
                .Select(segment => TranslatedSegment.Create(
                    revisionId,
                    segment.SegmentIndex,
                    segment.StartSeconds,
                    segment.EndSeconds,
                    segment.Text,
                    segment.SourceSegmentHash,
                    segment.Words) with
                { Origin = segment.Origin }))
            .OrderBy(static segment => segment.SegmentIndex)
            .ToArray();
    }
}
