using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private bool TryResolvePivotRoute(
        TranslationRoutingContext context,
        RouteQuery query,
        BundledModelManifestEntry entry,
        out TranslationRouteSelection? route)
    {
        route = null;
        if (!IsPlanRunnableOrUngated(context, entry))
        {
            return false;
        }

        foreach (string entryPath in EnumerateResolvedMadladEntryPaths(context.CacheIndex, entry))
        {
            string modelRootPath = Path.GetDirectoryName(entryPath)
                ?? throw new InvalidOperationException("MADLAD model root path could not be resolved.");
            if (!HasMadladSupportingFiles(modelRootPath))
            {
                continue;
            }

            string routeDetail = ResolvePivotRouteDetail(
                entry,
                context.DirectEntriesByPair.ContainsKey((query.SourceLanguage, query.TargetLanguage)));

            route = new TranslationRouteSelection(
                query.SourceLanguage,
                query.TargetLanguage,
                TranslationRoutingKind.Pivot,
                IsAvailable: true,
                ProviderName: ResolveProviderName(entry),
                RouteDetail: routeDetail,
                ModelId: entry.ModelId,
                PreferredModelAlias: entry.Aliases.FirstOrDefault(),
                ResolvedModelEntryPath: entryPath,
                EngineFamily: entry.EngineFamily,
                ModelProvenance: new Trackdub.Domain.Translation.TranslationModelProvenance(
                    ResolveProvenanceRepository(entry), entry.Revision, entry.Sha256,
                    new Dictionary<string, string>(entry.DownloadFileHashes),
                    entry.TargetPrefixes?.GetValueOrDefault(query.TargetLanguage), entry.CatalogPolicyVersion));
            return true;
        }

        return false;
    }
}
