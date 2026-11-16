using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.Translation;

public sealed partial class CatalogTranslationModelProvisioner
{
    private static bool CoversPair(
        BundledModelManifestEntry selected,
        string sourceLanguage,
        string targetLanguage)
    {
        if (HasExplicitPair(selected, sourceLanguage, targetLanguage))
        {
            return true;
        }

        return Covers(selected.LanguageCoverage.SourceLanguages, sourceLanguage) &&
            Covers(selected.LanguageCoverage.TargetLanguages, targetLanguage);
    }

    private static bool HasExplicitPair(
        BundledModelManifestEntry selected,
        string sourceLanguage,
        string targetLanguage) =>
        selected.LanguageCoverage.LanguagePairs.Any(pair =>
            pair.SourceLanguage.Equals(sourceLanguage, StringComparison.OrdinalIgnoreCase) &&
            pair.TargetLanguage.Equals(targetLanguage, StringComparison.OrdinalIgnoreCase));

    private static bool Covers(IReadOnlyList<string> languages, string language) =>
        languages.Any(candidate => candidate.Equals(language, StringComparison.OrdinalIgnoreCase) || candidate is "multi" or "any" or "*");

    private bool HasRegisteredDirectPair(string sourceLanguage, string targetLanguage) =>
        registry.Entries.Any(entry => entry.Task == "translation" &&
            entry.LanguageCoverage.LanguagePairs.Any(pair =>
                pair.SourceLanguage.Equals(sourceLanguage, StringComparison.OrdinalIgnoreCase) &&
                pair.TargetLanguage.Equals(targetLanguage, StringComparison.OrdinalIgnoreCase)));
}
