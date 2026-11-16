using Trackdub.Contracts.Pipeline;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private TranslationRouteSelection ResolveRoute(TranslationRoutingContext context, RouteQuery query)
    {
        if (query.IsSourceUnresolved)
        {
            return CreateSourceUnresolvedRoute(query);
        }

        TranslationLanguageCoverageMatrix.TryGetLanguage(query.TargetLanguage, out TranslationLanguageDefinition? targetDefinition);
        if (!HasKnownRoute(context, query, targetDefinition))
        {
            return CreateUnknownPairRoute(context, query);
        }

        string displayName = query.ResolveDisplayName(targetDefinition);
        return ResolveEarlyRoute(context, query)
            ?? ResolveInstalledRoute(context, query)
            ?? ResolveFallbackRoute(context, query, displayName);
    }

    private static bool HasKnownRoute(
        TranslationRoutingContext context,
        RouteQuery query,
        TranslationLanguageDefinition? targetDefinition)
    {
        if (context.DirectEntriesByPair.ContainsKey((query.SourceLanguage, query.TargetLanguage)))
        {
            return true;
        }

        if (targetDefinition is null)
        {
            return false;
        }

        return TranslationLanguageCoverageMatrix.GetTargets(query.SourceLanguage)
            .Any(candidate => string.Equals(candidate.Code, query.TargetLanguage, StringComparison.Ordinal));
    }

    private TranslationRouteSelection? ResolveEarlyRoute(TranslationRoutingContext context, RouteQuery query)
    {
        if (!query.HasPreferredModel)
        {
            return null;
        }

        return TryResolvePreferredRoute(context, query, out TranslationRouteSelection? preferredRoute)
            ? preferredRoute
            : null;
    }
}
