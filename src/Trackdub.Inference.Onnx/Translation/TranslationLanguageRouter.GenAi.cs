using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private bool TryResolveGenAiPivotRoute(
        TranslationRoutingContext context,
        RouteQuery query,
        out TranslationRouteSelection? route)
    {
        route = null;
        foreach (BundledModelManifestEntry entry in GetPivotEntriesForPair(context, query)
                     .Where(static entry => IsGenAiPivotEngineFamily(entry.EngineFamily)))
        {
            if (TryResolveGenAiPivotRoute(context, query, entry, out route))
            {
                return true;
            }
        }

        return false;
    }
}
