using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private bool TryResolveMadladPivotRoute(
        TranslationRoutingContext context,
        RouteQuery query,
        out TranslationRouteSelection? route)
    {
        route = null;
        foreach (BundledModelManifestEntry entry in GetPivotEntriesForPair(context, query)
                     .Where(static entry => string.Equals(entry.EngineFamily, "madlad", StringComparison.OrdinalIgnoreCase)))
        {
            if (TryResolvePivotRoute(context, query, entry, out route))
            {
                return true;
            }
        }

        return false;
    }

    private bool TryResolvePivotRoute(
        TranslationRoutingContext context,
        RouteQuery query,
        out TranslationRouteSelection? route)
    {
        route = null;
        foreach (BundledModelManifestEntry entry in GetPivotEntriesForPair(context, query))
        {
            if (TryResolvePivotRoute(context, query, entry, out route))
            {
                return true;
            }
        }

        return false;
    }
}
