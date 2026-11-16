using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private static bool CoversLanguagePair(ModelLanguageCoverage coverage, RouteQuery query)
    {
        if (coverage.LanguagePairs.Select(NormalizePair).Any(pair => pair == (query.SourceLanguage, query.TargetLanguage)))
        {
            return true;
        }

        return CoversLanguage(coverage.SourceLanguages, query.SourceLanguage) &&
               CoversLanguage(coverage.TargetLanguages, query.TargetLanguage);
    }

    private static bool CoversLanguage(IReadOnlyList<string> supportedLanguages, string language) =>
        supportedLanguages.Any(candidate =>
            string.Equals(candidate, language, StringComparison.OrdinalIgnoreCase) ||
            string.Equals(candidate, "multi", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(candidate, "any", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(candidate, "*", StringComparison.OrdinalIgnoreCase));

    private static bool HasDirectPair(BundledModelManifestEntry entry, RouteQuery query) =>
        entry.LanguageCoverage.LanguagePairs.Any(pair =>
            string.Equals(pair.SourceLanguage, query.SourceLanguage, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(pair.TargetLanguage, query.TargetLanguage, StringComparison.OrdinalIgnoreCase));

    private static (string SourceLanguage, string TargetLanguage)? NormalizePair(ModelLanguagePair pair)
    {
        string? sourceLanguage = NormalizeLanguageCode(pair.SourceLanguage);
        string? targetLanguage = NormalizeLanguageCode(pair.TargetLanguage);
        return sourceLanguage is null || targetLanguage is null
            ? null
            : (sourceLanguage, targetLanguage);
    }

    private static IEnumerable<(string SourceLanguage, string TargetLanguage)> EnumerateDirectPairs(BundledModelManifestEntry entry)
    {
        HashSet<(string SourceLanguage, string TargetLanguage)> pairs = [];
        foreach ((string SourceLanguage, string TargetLanguage)? explicitPair in entry.LanguageCoverage.LanguagePairs.Select(NormalizePair))
        {
            if (explicitPair is { } value && pairs.Add(value))
            {
                yield return value;
            }
        }
    }

    private static string? NormalizeLanguageCode(string? languageCode) =>
        string.IsNullOrWhiteSpace(languageCode)
            ? null
            : languageCode.Trim().ToLowerInvariant();
}
