using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private static IReadOnlyList<BundledModelManifestEntry> GetDirectEntriesForPair(
        TranslationRoutingContext context,
        RouteQuery query) =>
        context.DirectEntriesByPair.TryGetValue((query.SourceLanguage, query.TargetLanguage), out IReadOnlyList<BundledModelManifestEntry>? entries)
            ? entries
            : [];

    private static IEnumerable<BundledModelManifestEntry> GetPivotEntriesForPair(
        TranslationRoutingContext context,
        RouteQuery query) =>
        context.PivotEntries.Where(entry => CoversLanguagePair(entry.LanguageCoverage, query));

    private static bool HasCapability(BundledModelManifestEntry entry, string capability) =>
        entry.Capabilities.Any(candidate => string.Equals(candidate, capability, StringComparison.OrdinalIgnoreCase));

    private static bool IsDirectTranslationEntry(BundledModelManifestEntry entry) =>
        ResolveTranslationEntryKind(entry) == TranslationEntryKind.Direct;

    private static bool IsPivotTranslationEntry(BundledModelManifestEntry entry) =>
        ResolveTranslationEntryKind(entry) == TranslationEntryKind.Pivot;

    private static TranslationEntryKind ResolveTranslationEntryKind(BundledModelManifestEntry entry)
    {
        if (HasCapability(entry, "direct-translation"))
        {
            return TranslationEntryKind.Direct;
        }

        if (HasCapability(entry, "pivot-translation"))
        {
            return TranslationEntryKind.Pivot;
        }

        if (entry.LanguageCoverage.LanguagePairs.Count > 0)
        {
            return TranslationEntryKind.Direct;
        }

        if (entry.LanguageCoverage.SourceLanguages.Count > 0 || entry.LanguageCoverage.TargetLanguages.Count > 0)
        {
            return TranslationEntryKind.Pivot;
        }

        return TranslationEntryKind.None;
    }

    private enum TranslationEntryKind
    {
        None,
        Direct,
        Pivot
    }
}
