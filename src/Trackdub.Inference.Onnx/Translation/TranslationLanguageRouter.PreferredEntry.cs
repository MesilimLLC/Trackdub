using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private bool TryResolvePreferredDirect(
        TranslationRoutingContext context,
        RouteQuery query,
        BundledModelManifestEntry entry,
        out TranslationRouteSelection? route)
    {
        route = null;
        if (!HasDirectPair(entry, query))
        {
            return false;
        }

        return TryResolveDirectRoute(context, query, entry, out route);
    }

    private bool TryResolvePreferredPivot(
        TranslationRoutingContext context,
        RouteQuery query,
        BundledModelManifestEntry entry,
        out TranslationRouteSelection? route)
    {
        route = null;
        if (!IsPivotTranslationEntry(entry))
        {
            return false;
        }

        return TryResolvePivotRoute(context, query, entry, out route);
    }

    private bool TryResolvePreferredGenAi(
        TranslationRoutingContext context,
        RouteQuery query,
        BundledModelManifestEntry entry,
        out TranslationRouteSelection? route)
    {
        route = null;
        if (!IsGenAiPivotEngineFamily(entry.EngineFamily))
        {
            return false;
        }

        return TryResolveGenAiPivotRoute(context, query, entry, out route);
    }
}
