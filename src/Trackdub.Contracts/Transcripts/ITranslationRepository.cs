using Trackdub.Domain.Translation;
using DomainTranslatedSegment = Trackdub.Domain.Translation.TranslatedSegment;

namespace Trackdub.Contracts.Transcripts;

public interface ITranslationRepository
{
    Task<TranslationRevision?> GetCurrentRevisionAsync(
        Guid projectId,
        string targetLanguage,
        CancellationToken cancellationToken);

    Task<IReadOnlyList<DomainTranslatedSegment>> GetSegmentsAsync(
        Guid translationRevisionId,
        CancellationToken cancellationToken);

    Task<int> GetNextRevisionNumberAsync(
        Guid projectId,
        string targetLanguage,
        CancellationToken cancellationToken);

    Task SaveRevisionAsync(
        TranslationRevision revision,
        IReadOnlyList<DomainTranslatedSegment> segments,
        CancellationToken cancellationToken);

    /// <summary>
    /// Saves a new translation revision where only <paramref name="changedSegments"/> differ from the
    /// previous revision.  The default implementation fetches the previous revision's segments, merges
    /// <paramref name="changedSegments"/> into them (by <c>SegmentIndex</c>), and delegates to
    /// <see cref="SaveRevisionAsync"/>.  Infrastructure implementations may override this to perform a
    /// more efficient partial-upsert at the SQL layer, avoiding the full O(n) segment read+write for a
    /// single-segment retranslation.
    /// </summary>
    async Task PatchRevisionAsync(
        TranslationRevision previousRevision,
        TranslationRevision newRevision,
        IReadOnlyList<DomainTranslatedSegment> changedSegments,
        CancellationToken cancellationToken)
    {
        // Default: read all current segments, merge the changed ones, then write the full set.
        IReadOnlyList<DomainTranslatedSegment> current = await GetSegmentsAsync(
            previousRevision.Id, cancellationToken).ConfigureAwait(false);

        Dictionary<int, DomainTranslatedSegment> changedByIndex = changedSegments
            .ToDictionary(s => s.SegmentIndex);

        List<DomainTranslatedSegment> merged = current
            .OrderBy(s => s.SegmentIndex)
            .Select(s => changedByIndex.TryGetValue(s.SegmentIndex, out DomainTranslatedSegment? changed)
                ? changed
                : DomainTranslatedSegment.Create(
                    newRevision.Id,
                    s.SegmentIndex,
                    s.StartSeconds,
                    s.EndSeconds,
                    s.Text,
                    s.SourceSegmentHash,
                    s.Words) with
                { Origin = s.Origin })
            .ToList();

        // Append any brand-new segments that were not in the previous revision
        foreach (DomainTranslatedSegment extra in changedSegments.Where(s => !current.Any(c => c.SegmentIndex == s.SegmentIndex)))
        {
            merged.Add(extra);
        }

        merged = merged.OrderBy(s => s.SegmentIndex).ToList();
        await SaveRevisionAsync(newRevision, merged, cancellationToken).ConfigureAwait(false);
    }
}
