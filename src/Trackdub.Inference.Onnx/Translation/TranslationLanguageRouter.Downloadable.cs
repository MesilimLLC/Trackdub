using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private static BundledModelManifestEntry? FindDownloadableEntry(TranslationRoutingContext context, RouteQuery query) =>
        // Skip cache hits here: installed resolution already failed for them above, so an
        // uncached covering candidate must remain eligible for DownloadRequired.
        GetDirectEntriesForPair(context, query)
            .FirstOrDefault(entry => IsUncachedDownloadableEntry(context, entry, query));

    private static bool IsUncachedDownloadableEntry(
        TranslationRoutingContext context,
        BundledModelManifestEntry entry,
        RouteQuery query)
    {
        if (context.CacheIndex.ContainsKey(entry.ModelId))
        {
            return false;
        }

        return IsDownloadableEntry(entry, query);
    }

    private static bool IsDownloadableEntry(BundledModelManifestEntry entry, RouteQuery query)
    {
        if (!entry.CommercialAllowed || !entry.CommercialClearance)
        {
            return false;
        }

        if (entry.DownloadFiles.Count == 0)
        {
            return false;
        }

        if (query.IsOpusMtAuto)
        {
            return string.Equals(entry.EngineFamily, "opus-mt", StringComparison.OrdinalIgnoreCase);
        }

        return !query.HasPreferredModel || HasAlias(entry, query.PreferredModelAlias!);
    }

    private static bool HasAlias(BundledModelManifestEntry entry, string alias) =>
        entry.Aliases.Any(candidate => candidate.Equals(alias.Trim(), StringComparison.OrdinalIgnoreCase));

    private static bool IsUninstalledCommercialDownload(
        TranslationRoutingContext context,
        BundledModelManifestEntry candidate)
    {
        if (context.CacheIndex.ContainsKey(candidate.ModelId))
        {
            return false;
        }

        bool commerciallyUsable = candidate.CommercialAllowed && candidate.CommercialClearance;
        bool hasDownloadFiles = candidate.DownloadFiles.Count > 0 || candidate.Variants.Any(variant => variant.DownloadFiles.Count > 0);
        return commerciallyUsable && hasDownloadFiles;
    }
}
