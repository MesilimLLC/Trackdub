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

    /// <summary>
    /// A verified commercial model may be chosen automatically; a risk-accepted one only when the request names it by alias
    /// (MODEL_LICENSE_POLICY.md: an accepted-risk model is opt-in and never selected automatically).
    /// </summary>
    private static bool IsCommerciallyEligible(BundledModelManifestEntry entry, RouteQuery query) =>
        entry.CommercialAllowed &&
        (entry.CommercialUseVerified || (entry.CommercialSafeMode && query.HasPreferredModel && !query.IsOpusMtAuto));

    private static bool IsDownloadableEntry(BundledModelManifestEntry entry, RouteQuery query)
    {
        if (!IsCommerciallyEligible(entry, query))
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

        bool commerciallyUsable = candidate.CommercialAllowed && candidate.CommercialSafeMode;
        bool hasDownloadFiles = candidate.DownloadFiles.Count > 0 || candidate.Variants.Any(variant => variant.DownloadFiles.Count > 0);
        return commerciallyUsable && hasDownloadFiles;
    }
}
