using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private bool TryResolveDirectRoute(
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

        foreach (string entryPath in EnumerateResolvedEntryPaths(context.CacheIndex, entry))
        {
            string modelRootPath = ResolveOpusModelRootPath(entryPath);
            if (!HasOpusSupportingFiles(modelRootPath))
            {
                continue;
            }

            route = new TranslationRouteSelection(
                query.SourceLanguage,
                query.TargetLanguage,
                TranslationRoutingKind.Direct,
                IsAvailable: true,
                ProviderName: ResolveProviderName(entry),
                RouteDetail: ResolveDirectRouteDetail(entry),
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
