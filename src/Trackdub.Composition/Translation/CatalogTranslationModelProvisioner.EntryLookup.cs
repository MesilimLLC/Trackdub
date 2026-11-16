using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.Translation;

public sealed partial class CatalogTranslationModelProvisioner
{
    private static BundledModelManifestEntry? FindAliasedEntry(
        IEnumerable<BundledModelManifestEntry> candidates,
        string effectiveAlias) =>
        candidates.FirstOrDefault(entry =>
            entry.ModelId.Equals(effectiveAlias, StringComparison.OrdinalIgnoreCase) ||
            entry.Aliases.Contains(effectiveAlias, StringComparer.OrdinalIgnoreCase));

    private static BundledModelManifestEntry? FindDefaultOpusEntry(
        IEnumerable<BundledModelManifestEntry> candidates,
        string sourceLanguage,
        string targetLanguage) =>
        candidates.FirstOrDefault(entry =>
            entry.EngineFamily == "opus-mt" &&
            entry.LanguageCoverage.LanguagePairs.Any(pair =>
                pair.SourceLanguage.Equals(sourceLanguage, StringComparison.OrdinalIgnoreCase) &&
                pair.TargetLanguage.Equals(targetLanguage, StringComparison.OrdinalIgnoreCase)));
}
