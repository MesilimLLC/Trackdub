using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private bool TryResolveGenAiPivotRoute(
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

        string? plannedEntryPath = context.PlanAvailabilityByModelId.TryGetValue(entry.ModelId, out PlanAvailability? planned)
            ? planned.PlannedEntryPath
            : null;
        // The planner's chosen variant goes first so route provenance describes the variant that will run.
        foreach (string entryPath in EnumerateResolvedEntryPaths(context.CacheIndex, entry, includeVariants: true)
                     .OrderBy(path => IsPlannedEntryPath(path, plannedEntryPath) ? 0 : 1))
        {
            string? modelRootPath = Path.GetDirectoryName(entryPath);
            if (modelRootPath is null || !HasGenAiSupportingFiles(modelRootPath))
            {
                continue;
            }

            route = new TranslationRouteSelection(
                query.SourceLanguage,
                query.TargetLanguage,
                TranslationRoutingKind.Pivot,
                IsAvailable: true,
                ProviderName: entry.EngineFamily,
                RouteDetail: ResolvePivotRouteLabel(entry),
                ModelId: entry.ModelId,
                PreferredModelAlias: entry.Aliases.FirstOrDefault(),
                ResolvedModelEntryPath: entryPath,
                EngineFamily: entry.EngineFamily,
                ModelProvenance: GenAiPivotProvenanceResolver.Resolve(new GenAiPivotProvenanceRequest
                {
                    CacheIndex = context.CacheIndex,
                    Entry = entry,
                    ResolvedEntryPath = entryPath,
                    TargetLanguage = query.TargetLanguage
                }));
            return true;
        }

        return false;
    }
}
