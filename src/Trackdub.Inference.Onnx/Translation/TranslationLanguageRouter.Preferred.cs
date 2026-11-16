using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private bool TryResolvePreferredRoute(
        TranslationRoutingContext context,
        RouteQuery query,
        out TranslationRouteSelection? route)
    {
        route = null;
        if (!query.HasPreferredModel)
        {
            return false;
        }

        foreach (BundledModelManifestEntry entry in GetPreferredCandidates(context, query))
        {
            if (TryResolvePreferredEntry(context, query, entry, out route))
            {
                return true;
            }
        }

        return TryResolvePreferredGenAiDownload(context, query, out route);
    }

    private static IEnumerable<BundledModelManifestEntry> GetPreferredCandidates(
        TranslationRoutingContext context,
        RouteQuery query) =>
        GetDirectEntriesForPair(context, query)
            .Concat(GetPivotEntriesForPair(context, query))
            .Where(entry => HasAlias(entry, query.PreferredModelAlias!));

    private bool TryResolvePreferredEntry(
        TranslationRoutingContext context,
        RouteQuery query,
        BundledModelManifestEntry entry,
        out TranslationRouteSelection? route)
    {
        if (TryResolvePreferredDirect(context, query, entry, out route))
        {
            return true;
        }

        if (TryResolvePreferredPivot(context, query, entry, out route))
        {
            return true;
        }

        return TryResolvePreferredGenAi(context, query, entry, out route);
    }
}
