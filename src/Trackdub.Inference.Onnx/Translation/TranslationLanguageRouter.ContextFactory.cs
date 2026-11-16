using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private static IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> BuildCacheIndex(
        IReadOnlyList<LocalModelCacheRecord> cacheRecords) =>
        cacheRecords
            .GroupBy(record => record.ModelId, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(
                group => group.Key,
                group => (IReadOnlyList<LocalModelCacheRecord>)group
                    .OrderByDescending(record => record.CachedAtUtc)
                    .ToArray(),
                StringComparer.OrdinalIgnoreCase);

    private static Dictionary<(string SourceLanguage, string TargetLanguage), IReadOnlyList<BundledModelManifestEntry>> BuildDirectEntriesByPair(
        IReadOnlyList<BundledModelManifestEntry> manifestEntries,
        IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> cacheIndex) =>
        manifestEntries
            .Where(static entry => IsDirectTranslationManifestEntry(entry))
            .SelectMany(entry => EnumerateDirectPairs(entry).Select(pair => (Entry: entry, Pair: pair)))
            .GroupBy(candidate => candidate.Pair)
            .ToDictionary(
                group => group.Key,
                group => (IReadOnlyList<BundledModelManifestEntry>)group.Select(candidate => candidate.Entry)
                    .OrderBy(entry => entry.LanguageCoverage.LanguagePairs.Count == 1 ? 0 : 1)
                    .ThenBy(entry => cacheIndex.ContainsKey(entry.ModelId) ? 0 : 1)
                    .ThenBy(entry => entry.SourceUrl.Contains("/onnx-community/", StringComparison.Ordinal) ? 0 : 1)
                    .ThenBy(entry => entry.ModelId, StringComparer.Ordinal).ToArray());

    private static BundledModelManifestEntry[] BuildPivotEntries(
        IReadOnlyList<BundledModelManifestEntry> manifestEntries) =>
        manifestEntries.Where(static entry => IsPivotTranslationManifestEntry(entry)).ToArray();

    private static bool IsDirectTranslationManifestEntry(BundledModelManifestEntry entry) =>
        string.Equals(entry.Task, "translation", StringComparison.OrdinalIgnoreCase) && IsDirectTranslationEntry(entry);

    private static bool IsPivotTranslationManifestEntry(BundledModelManifestEntry entry) =>
        string.Equals(entry.Task, "translation", StringComparison.OrdinalIgnoreCase) && IsPivotTranslationEntry(entry);
}
