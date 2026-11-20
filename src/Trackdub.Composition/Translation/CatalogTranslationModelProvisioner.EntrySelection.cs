using Trackdub.Composition.Runtime;
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
        ILookup<string, LocalModelCacheRecord> cacheByModelId = await LoadCacheByModelIdAsync(cancellationToken).ConfigureAwait(false);
        IEnumerable<BundledModelManifestEntry> candidates = BuildCandidates(cacheByModelId);
        if (string.IsNullOrWhiteSpace(effectiveAlias))
        {
            // Automatic choice only among verified models: a risk-accepted model is opt-in by explicit alias.
            return FindDefaultOpusEntry(candidates.Where(static entry => entry.CommercialUseVerified), sourceLanguage, targetLanguage);
        }

        return FindAliasedEntry(candidates, effectiveAlias);
    }

    private async Task<ILookup<string, LocalModelCacheRecord>> LoadCacheByModelIdAsync(CancellationToken cancellationToken)
    {
        if (cacheInventory is null)
        {
            return Array.Empty<LocalModelCacheRecord>().ToLookup(record => record.ModelId, StringComparer.OrdinalIgnoreCase);
        }

        IReadOnlyList<LocalModelCacheRecord> records = await cacheInventory.LoadAsync(cancellationToken).ConfigureAwait(false);
        return records.ToLookup(record => record.ModelId, StringComparer.OrdinalIgnoreCase);
    }

    private IEnumerable<BundledModelManifestEntry> BuildCandidates(
        ILookup<string, LocalModelCacheRecord> cacheByModelId) =>
        registry.Entries
            .Where(entry => entry.Task == "translation" && entry.CommercialAllowed && entry.CommercialSafeMode)
            .OrderBy(entry => IsUsableInstalled(entry, cacheByModelId) ? 0 : 1)
            .ThenBy(entry => entry.LanguageCoverage.LanguagePairs.Count == 1 ? 0 : 1)
            .ThenBy(entry => entry.SourceUrl.Contains("/onnx-community/", StringComparison.Ordinal) ? 0 : 1)
            .ThenBy(entry => entry.ModelId, StringComparer.Ordinal);

    private static bool IsUsableInstalled(
        BundledModelManifestEntry entry,
        ILookup<string, LocalModelCacheRecord> cacheByModelId) =>
        cacheByModelId[entry.ModelId].Any(record =>
            ModelInventoryService.DetermineState(entry, record) is ModelCacheState.Installed or ModelCacheState.Ready);
}
