using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private static bool TryResolvePreferredGenAiDownload(
        TranslationRoutingContext context,
        RouteQuery query,
        out TranslationRouteSelection? route)
    {
        route = null;
        if (!context.CanDownload)
        {
            return false;
        }

        BundledModelManifestEntry? entry = GetPivotEntriesForPair(context, query)
            .FirstOrDefault(candidate => IsUninstalledPreferredGenAiDownload(context, candidate, query));
        if (entry is null)
        {
            return false;
        }

        route = new TranslationRouteSelection(
            query.SourceLanguage,
            query.TargetLanguage,
            TranslationRoutingKind.Pivot,
            IsAvailable: false,
            ProviderName: entry.EngineFamily,
            RouteDetail: "Download required",
            ModelId: entry.ModelId,
            PreferredModelAlias: entry.Aliases.FirstOrDefault(),
            EngineFamily: entry.EngineFamily,
            PreparationStatus: TranslationPreparationStatus.DownloadRequired,
            ExpectedDownloadBytes: entry.ExpectedDownloadBytes);
        return true;
    }

    private static bool IsUninstalledPreferredGenAiDownload(
        TranslationRoutingContext context,
        BundledModelManifestEntry candidate,
        RouteQuery query) =>
        IsPreferredGenAiEntry(candidate, query) && IsUninstalledCommercialDownload(context, candidate);

    private static bool IsPreferredGenAiEntry(BundledModelManifestEntry candidate, RouteQuery query) =>
        HasAlias(candidate, query.PreferredModelAlias!) && IsGenAiPivotEngineFamily(candidate.EngineFamily);
}
