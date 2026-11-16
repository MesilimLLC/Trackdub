using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.Translation;

public sealed partial class CatalogTranslationModelProvisioner
{
    private async Task<BundledModelManifestEntry?> SelectEntryAsync(
        string sourceLanguage,
        string targetLanguage,
        string? effectiveAlias,
        CancellationToken cancellationToken)
    {
        HashSet<string> installed = await LoadInstalledModelIdsAsync(cancellationToken).ConfigureAwait(false);
        IEnumerable<BundledModelManifestEntry> candidates = BuildCandidates(installed);
        if (string.IsNullOrWhiteSpace(effectiveAlias))
        {
            return FindDefaultOpusEntry(candidates, sourceLanguage, targetLanguage);
        }

        return FindAliasedEntry(candidates, effectiveAlias);
    }

    private async Task<HashSet<string>> LoadInstalledModelIdsAsync(CancellationToken cancellationToken)
    {
        if (cacheInventory is null)
        {
            return new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        }

        IReadOnlyList<LocalModelCacheRecord> records = await cacheInventory.LoadAsync(cancellationToken).ConfigureAwait(false);
        return records.Select(record => record.ModelId).ToHashSet(StringComparer.OrdinalIgnoreCase);
    }

    private IEnumerable<BundledModelManifestEntry> BuildCandidates(HashSet<string> installed) =>
        registry.Entries
            .Where(entry => entry.Task == "translation" && entry.CommercialAllowed && entry.CommercialUseVerified)
            .OrderBy(entry => installed.Contains(entry.ModelId) ? 0 : 1)
            .ThenBy(entry => entry.LanguageCoverage.LanguagePairs.Count == 1 ? 0 : 1)
            .ThenBy(entry => entry.SourceUrl.Contains("/onnx-community/", StringComparison.Ordinal) ? 0 : 1)
            .ThenBy(entry => entry.ModelId, StringComparer.Ordinal);
}
