using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private bool TryResolveOpusAutoDirectRoute(
        TranslationRoutingContext context,
        RouteQuery query,
        out TranslationRouteSelection? route)
    {
        route = null;
        foreach (BundledModelManifestEntry entry in GetDirectEntriesForPair(context, query)
                     .Where(static entry => string.Equals(entry.EngineFamily, "opus-mt", StringComparison.OrdinalIgnoreCase)))
        {
            if (TryResolveDirectRoute(context, query, entry, out route))
            {
                return true;
            }
        }

        return false;
    }

    private bool TryResolveDirectRoute(
        TranslationRoutingContext context,
        RouteQuery query,
        out TranslationRouteSelection? route)
    {
        route = null;
        foreach (BundledModelManifestEntry entry in GetDirectEntriesForPair(context, query))
        {
            if (TryResolveDirectRoute(context, query, entry, out route))
            {
                return true;
            }
        }

        return false;
    }
}
