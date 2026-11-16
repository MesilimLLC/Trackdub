using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private TranslationRouteSelection? ResolveInstalledRoute(TranslationRoutingContext context, RouteQuery query)
    {
        if (query.IsOpusMtAuto)
        {
            return TryResolveOpusAutoDirectRoute(context, query, out TranslationRouteSelection? opusRoute)
                ? opusRoute
                : ResolveDownloadRequiredRoute(context, query);
        }

        if (query.HasPreferredModel)
        {
            return ResolveDownloadRequiredRoute(context, query);
        }

        return TryResolveDirectRoute(context, query, out TranslationRouteSelection? directRoute)
            ? directRoute
            : ResolveDownloadRequiredRoute(context, query);
    }

    private TranslationRouteSelection? ResolveDownloadRequiredRoute(TranslationRoutingContext context, RouteQuery query)
    {
        BundledModelManifestEntry? downloadable = FindDownloadableEntry(context, query);
        if (downloadable is null)
        {
            return null;
        }

        if (!context.CanDownload)
        {
            return null;
        }

        if (context.CacheIndex.ContainsKey(downloadable.ModelId))
        {
            return null;
        }

        return new TranslationRouteSelection(
            query.SourceLanguage,
            query.TargetLanguage,
            TranslationRoutingKind.Direct,
            IsAvailable: false,
            ProviderName: ResolveProviderName(downloadable),
            RouteDetail: "Download required",
            ModelId: downloadable.ModelId,
            PreferredModelAlias: downloadable.Aliases.FirstOrDefault(),
            EngineFamily: downloadable.EngineFamily,
            PreparationStatus: TranslationPreparationStatus.DownloadRequired,
            ExpectedDownloadBytes: downloadable.ExpectedDownloadBytes);
    }

    private TranslationRouteSelection ResolveFallbackRoute(
        TranslationRoutingContext context,
        RouteQuery query,
        string displayName)
    {
        if (query.IsOpusMtAuto)
        {
            return CreateUnavailablePreferredRoute(context, query);
        }

        if (query.HasPreferredModel)
        {
            return CreateUnavailablePreferredRoute(context, query);
        }

        if (TryResolveMadladPivotRoute(context, query, out TranslationRouteSelection? madladRoute))
        {
            return madladRoute!;
        }

        if (TryResolvePivotRoute(context, query, out TranslationRouteSelection? pivotRoute))
        {
            return pivotRoute!;
        }

        if (TryResolveGenAiPivotRoute(context, query, out TranslationRouteSelection? genAiRoute))
        {
            return genAiRoute!;
        }

        return CreateMissingRoute(context, query, displayName);
    }
}
