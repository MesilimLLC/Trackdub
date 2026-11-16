using Trackdub.Contracts.Pipeline;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private TranslationTargetLanguageOption[] BuildTargetOptions(string sourceLanguage, TranslationRoutingContext context)
    {
        bool sourceUnresolved = IsSourceUnresolved(sourceLanguage);
        return EnumerateCandidateLanguages(sourceLanguage, context, sourceUnresolved)
            .DistinctBy(language => language.Code)
            .Select(language => sourceUnresolved
                ? CreateUnresolvedOption(language)
                : CreateResolvedOption(context, sourceLanguage, language))
            .ToArray();
    }

    private static IEnumerable<TranslationLanguageDefinition> EnumerateCandidateLanguages(
        string sourceLanguage,
        TranslationRoutingContext context,
        bool includeAll)
    {
        IEnumerable<TranslationLanguageDefinition> matrix = includeAll
            ? TranslationLanguageCoverageMatrix.AllLanguages
            : TranslationLanguageCoverageMatrix.GetTargets(sourceLanguage);
        return matrix.Concat(EnumerateDirectLanguages(sourceLanguage, context, includeAll));
    }

    private static IEnumerable<TranslationLanguageDefinition> EnumerateDirectLanguages(
        string sourceLanguage,
        TranslationRoutingContext context,
        bool includeAll) =>
        context.DirectEntriesByPair.Keys
            .Where(pair => includeAll || MatchesSource(pair.SourceLanguage, sourceLanguage))
            .Select(pair => CreateLanguageDefinition(pair.TargetLanguage));

    private static bool MatchesSource(string pairSourceLanguage, string sourceLanguage) =>
        pairSourceLanguage.Equals(sourceLanguage, StringComparison.OrdinalIgnoreCase);

    private static TranslationLanguageDefinition CreateLanguageDefinition(string code) =>
        new(code, code, code);

    private static bool IsSourceUnresolved(string sourceLanguage) =>
        string.IsNullOrWhiteSpace(sourceLanguage) ||
        sourceLanguage.Equals("auto", StringComparison.OrdinalIgnoreCase) ||
        sourceLanguage.Equals("und", StringComparison.OrdinalIgnoreCase);
}
