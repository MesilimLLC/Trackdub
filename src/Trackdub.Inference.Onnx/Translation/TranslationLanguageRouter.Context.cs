using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private async Task<TranslationRoutingContext> BuildContextAsync(CancellationToken cancellationToken)
    {
        if (modelProvisioner is not null)
        {
            await modelProvisioner.RefreshAsync(cancellationToken).ConfigureAwait(false);
        }

        IReadOnlyList<LocalModelCacheRecord> cacheRecords = await modelCacheInventory.LoadAsync(cancellationToken).ConfigureAwait(false);
        IReadOnlyList<BundledModelManifestEntry> manifestEntries = manifestRegistry.Entries;
        IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> cacheIndex = BuildCacheIndex(cacheRecords);
        Dictionary<(string SourceLanguage, string TargetLanguage), IReadOnlyList<BundledModelManifestEntry>> directEntriesByPair =
            BuildDirectEntriesByPair(manifestEntries, cacheIndex);
        BundledModelManifestEntry[] pivotEntries = BuildPivotEntries(manifestEntries);
        IReadOnlyDictionary<string, PlanAvailability> planAvailability = await BuildPlanAvailabilityAsync(
            directEntriesByPair.Values.SelectMany(static entries => entries).Concat(pivotEntries),
            cancellationToken).ConfigureAwait(false);
        bool canDownload = await ResolveCanDownloadAsync(cancellationToken).ConfigureAwait(false);
        return new TranslationRoutingContext
        {
            CacheIndex = cacheIndex,
            DirectEntriesByPair = directEntriesByPair,
            PivotEntries = pivotEntries,
            PlanAvailabilityByModelId = planAvailability,
            CanDownload = canDownload
        };
    }
}
