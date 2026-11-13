using Trackdub.Domain;

namespace Trackdub.Inference.Runtime.Planning;

internal sealed record PreferredVariantFallbackRequest
{
    public required StageRuntimePlan ResolvedPlan { get; init; }
    public string? PreferredVariantAlias { get; init; }
    public required bool RequirePreferredExecutionProvider { get; init; }
    public required Func<string, CancellationToken, Task<StageRuntimePlan>> ResolveFallbackAsync { get; init; }
}

internal static class RuntimePlannerPreferredVariantFallback
{
    public static async Task<StageRuntimePlan> ResolveAsync(
        PreferredVariantFallbackRequest request,
        CancellationToken cancellationToken)
    {
        if (request.ResolvedPlan.Status != StageRuntimePlanStatus.Blocked)
        {
            return RuntimePlanner.StampRequirePreferred(
                request.ResolvedPlan,
                request.RequirePreferredExecutionProvider);
        }

        if (request.PreferredVariantAlias is not string alias)
        {
            return RuntimePlanner.StampRequirePreferred(
                request.ResolvedPlan,
                request.RequirePreferredExecutionProvider);
        }

        StageRuntimePlan fallbackPlan = await request.ResolveFallbackAsync(alias, cancellationToken).ConfigureAwait(false);
        return fallbackPlan.Status.IsRunnable()
            ? fallbackPlan
            : RuntimePlanner.StampRequirePreferred(request.ResolvedPlan, request.RequirePreferredExecutionProvider);
    }
}
