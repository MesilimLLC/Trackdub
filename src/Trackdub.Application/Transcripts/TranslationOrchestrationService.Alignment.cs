using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.Transcripts;
using Trackdub.Domain;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Microsoft.Extensions.Logging;

namespace Trackdub.Application.Transcripts;

public sealed partial class TranslationOrchestrationService
{
    private static string SummarizeIndices(int[] indices)
    {
        const int maxListed = 10;
        return indices.Length <= maxListed
            ? string.Join(", ", indices)
            : string.Join(", ", indices.Take(maxListed)) + $" … ({indices.Length - maxListed} more)";
    }

    private sealed class TranslationPreparationProgress(IProgress<PipelineProgressEvent>? progress) : IProgress<string>
    {
        public void Report(string value) => PipelineProgressReporter.Phase(progress, StageNames.Translation, value);
    }

    private static TranslationExecutionMetadata? GetTranslationExecutionMetadata(object stageEngine) =>
        stageEngine is ITranslationExecutionMetadataReporter reporter
            ? reporter.LastExecutionMetadata
            : null;

    private async Task<TranslatedSegment> AlignTranslatedWordsAsync(
        TranslatedSegment translatedSegment,
        TranscriptSegment sourceSegment,
        TranslationAlignmentOptions options,
        CancellationToken cancellationToken)
    {
        TranslatedWordAlignmentResult alignment;
        try
        {
            alignment = await translatedWordAlignmentService.AlignAsync(
                new TranslatedWordAlignmentRequest(
                    sourceSegment,
                    translatedSegment,
                    options.SourceLanguage,
                    options.TargetLanguage,
                    options.PreferredModelAlias,
                    options.PreferredExecutionProvider,
                    options.RequirePreferredExecutionProvider),
                cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex,
                "Word alignment failed for segment {SegmentIndex}; karaoke timing will be omitted for this segment.",
                translatedSegment.SegmentIndex);
            return translatedSegment with { Words = [] };
        }

        if (alignment.Outcome is not TranslatedWordAlignmentOutcomeKind.Succeeded)
        {
            return translatedSegment with { Words = [] };
        }

        return HasRenderableTranslatedWordAlignment(translatedSegment, alignment.Words)
            ? new TranslatedSegment(
                translatedSegment.Id,
                translatedSegment.TranslationRevisionId,
                translatedSegment.SegmentIndex,
                translatedSegment.StartSeconds,
                translatedSegment.EndSeconds,
                translatedSegment.Text,
                translatedSegment.SourceSegmentHash,
                alignment.Words) with
            { Origin = translatedSegment.Origin }
            : translatedSegment with { Words = [] };
    }

    private static bool HasRenderableTranslatedWordAlignment(
        TranslatedSegment translatedSegment,
        IReadOnlyList<TranslatedWord> words)
    {
        if (words.Count == 0)
        {
            return false;
        }

        double previousEnd = translatedSegment.StartSeconds;
        foreach (TranslatedWord word in words.OrderBy(static word => word.WordIndex))
        {
            if (!IsWordWithinBounds(translatedSegment, word, previousEnd))
            {
                return false;
            }

            previousEnd = word.EndSeconds;
        }

        return CanMapWordsToTranslatedText(translatedSegment.Text, words);
    }

    private static bool IsWordWithinBounds(
        TranslatedSegment translatedSegment,
        TranslatedWord word,
        double previousEnd)
    {
        if (word.StartSeconds < translatedSegment.StartSeconds)
        {
            return false;
        }

        if (word.EndSeconds > translatedSegment.EndSeconds)
        {
            return false;
        }

        return word.StartSeconds >= previousEnd;
    }

    private static bool CanMapWordsToTranslatedText(
        string translatedText,
        IReadOnlyList<TranslatedWord> words)
    {
        int searchStart = 0;
        foreach (TranslatedWord word in words.OrderBy(static word => word.WordIndex))
        {
            int matchIndex = translatedText.IndexOf(word.Text, searchStart, StringComparison.OrdinalIgnoreCase);
            if (matchIndex < 0)
            {
                return false;
            }

            searchStart = matchIndex + word.Text.Length;
        }

        return true;
    }

    internal sealed record TranslationRunProgress(
        IProgress<PipelineProgressEvent>? Progress,
        ITranslatedSegmentObserver? SegmentObserver)
    {
        public ITranslatedSegmentObserver? Observer => SegmentObserver;
    }

    private sealed record TranslationRouteRequest(
        string SourceLanguage,
        string TargetLanguage,
        string? PreferredModelAlias);

    private sealed record TranslationGenerationInput(
        TranscriptProjectState CurrentState,
        GenerateTranslationRequest Request,
        TranscriptRevision TranscriptRevision,
        TranslationScopePlan ScopePlan,
        TranslationRouteSelection Route,
        StageRunRecord StageRun,
        string SourceLanguage,
        string TargetLanguage);

    private sealed record TranslatedSegmentBuildRequest(
        TranslationRevision Revision,
        IReadOnlyList<TranslatedTextSegment> TextSegments,
        IReadOnlyDictionary<int, TranscriptSegment> SourceSegmentsByIndex);

    private sealed record TranslationCommitRequest(
        TranslationRevision Revision,
        IReadOnlyList<TranslatedSegment> Segments,
        TranslationStreamCommit? StreamCommit);

    private sealed record TranslationStreamRequest(
        IStreamingTranslationEngine StreamingEngine,
        TranslationRequest TranslationRequest,
        Guid RunId,
        Guid SourceRevisionId,
        string TargetLanguage,
        int InputSegmentCount,
        IProgress<PipelineProgressEvent>? Progress,
        ITranslatedSegmentObserver? SegmentObserver);

    private sealed record TranslationAlignmentOptions(
        string SourceLanguage,
        string TargetLanguage,
        string? PreferredModelAlias,
        ExecutionProviderKind? PreferredExecutionProvider,
        bool RequirePreferredExecutionProvider);

    private sealed record RetranslationContext(
        TranscriptProjectState CurrentState,
        RetranslateSegmentRequest Request,
        TranscriptRevision TranscriptRevision,
        TranslationRevision CurrentTranslationRevision,
        TranscriptSegment SourceSegment,
        TranslationRouteSelection Route,
        StageRunRecord StageRun,
        string SourceLanguage,
        string TargetLanguage);

    private sealed record RetranslationText(
        string Text,
        bool HasUsableEngineText);
}
