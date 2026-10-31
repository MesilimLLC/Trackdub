namespace Trackdub.Domain.Translation;

public sealed record TranslatedSegment
{
    public TranslatedSegment(
        Guid id,
        Guid translationRevisionId,
        int segmentIndex,
        double startSeconds,
        double endSeconds,
        string text,
        string? sourceSegmentHash,
        IReadOnlyList<TranslatedWord>? words = null)
    {
        Id = id;
        TranslationRevisionId = translationRevisionId;
        SegmentIndex = segmentIndex;
        StartSeconds = startSeconds;
        EndSeconds = endSeconds;
        Text = text;
        SourceSegmentHash = NormalizeOptional(sourceSegmentHash);
        Words = NormalizeWords(words);
    }

    public Guid Id { get; init; }

    public Guid TranslationRevisionId { get; init; }

    public int SegmentIndex { get; init; }

    public double StartSeconds { get; init; }

    public double EndSeconds { get; init; }

    public string Text { get; init; }

    public string? SourceSegmentHash { get; init; }

    public IReadOnlyList<TranslatedWord> Words { get; init; }

    public TranslationSegmentOrigin Origin { get; init; } = TranslationSegmentOrigin.Unknown;

    public static TranslatedSegment Create(
        Guid translationRevisionId,
        int segmentIndex,
        double startSeconds,
        double endSeconds,
        string text,
        string? sourceSegmentHash = null,
        IReadOnlyList<TranslatedWord>? words = null)
    {
        if (translationRevisionId == Guid.Empty)
        {
            throw new ArgumentException("Translation revision id is required.", nameof(translationRevisionId));
        }

        if (segmentIndex < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(segmentIndex), "Segment index cannot be negative.");
        }

        if (!double.IsFinite(startSeconds) || startSeconds < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(startSeconds), "Segment start must be finite and non-negative.");
        }

        if (!double.IsFinite(endSeconds) || endSeconds < startSeconds)
        {
            throw new ArgumentOutOfRangeException(nameof(endSeconds), "Segment end must be finite and greater than or equal to the start.");
        }

        if (string.IsNullOrWhiteSpace(text))
        {
            throw new ArgumentException("Segment text is required.", nameof(text));
        }

        return new TranslatedSegment(
            Guid.NewGuid(),
            translationRevisionId,
            segmentIndex,
            startSeconds,
            endSeconds,
            text.Trim(),
            NormalizeOptional(sourceSegmentHash),
            words);
    }

    private static string? NormalizeOptional(string? value) =>
        string.IsNullOrWhiteSpace(value) ? null : value.Trim();

    private static IReadOnlyList<TranslatedWord> NormalizeWords(IReadOnlyList<TranslatedWord>? words) =>
        words is null || words.Count == 0
            ? []
            : words
                .OrderBy(static word => word.WordIndex)
                .Select((word, index) => TranslatedWord.Create(
                    index,
                    word.StartSeconds,
                    word.EndSeconds,
                    word.Text))
                .ToArray();
}
