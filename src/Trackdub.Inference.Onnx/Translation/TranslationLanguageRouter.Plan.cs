using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    /// <summary>
    /// Asks the shared <see cref="IRuntimePlanner"/> whether each candidate translation model
    /// is currently runnable. The router only marks a route as available when the planner
    /// agrees, so model integrity checks, EP availability, and smoke tests are honored
    /// uniformly with the rest of the pipeline.
    /// </summary>
    private async Task<IReadOnlyDictionary<string, PlanAvailability>> BuildPlanAvailabilityAsync(
        IEnumerable<BundledModelManifestEntry> entries,
        CancellationToken cancellationToken)
    {
        if (runtimePlanner is null)
        {
            return new Dictionary<string, PlanAvailability>(StringComparer.OrdinalIgnoreCase);
        }

        var availability = new Dictionary<string, PlanAvailability>(StringComparer.OrdinalIgnoreCase);
        foreach (BundledModelManifestEntry entry in entries.DistinctBy(e => e.ModelId, StringComparer.OrdinalIgnoreCase))
        {
            string? primaryAlias = entry.Aliases.FirstOrDefault();
            try
            {
                StageRuntimePlan plan = await runtimePlanner.PlanAsync(
                    new StageRuntimePlanningRequest(
                        Stage: RuntimeStage.Translation,
                        PreferredModelAlias: primaryAlias,
                        PreferredEngineFamily: entry.EngineFamily,
                        RequirePreferredModelAlias: !string.IsNullOrWhiteSpace(primaryAlias)),
                    cancellationToken).ConfigureAwait(false);
                availability[entry.ModelId] = new PlanAvailability(plan.IsRunnable(), plan.ModelEntryPath);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch
            {
                // Planner failure is treated as "not runnable" — a strictly more conservative
                // outcome than silently masking an underlying problem with the planner stack.
                availability[entry.ModelId] = new PlanAvailability(false, null);
            }
        }

        return availability;
    }

    private bool IsPlanRunnableOrUngated(TranslationRoutingContext context, BundledModelManifestEntry entry)
    {
        // When no planner is configured, fall back to the inline file checks each TryResolve
        // method already performs. With a planner present, route availability requires the
        // planner to confirm the model is in a runnable status (Ready or Verified).
        if (runtimePlanner is null)
        {
            return true;
        }

        return context.PlanAvailabilityByModelId.TryGetValue(entry.ModelId, out PlanAvailability? planAvailability) &&
            planAvailability.IsRunnable;
    }
}
