using Trackdub.Domain;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx;

/// <summary>
/// The provider a pooled session is requested on and how strictly that request binds. Value
/// equality lets a pinned session decide whether it still matches a new request.
/// </summary>
internal readonly record struct PooledProviderRequest(
    ExecutionProviderKind Provider,
    bool AllowTrtInitFallback = true,
    bool HasExplicitProviderPreference = false)
{
    public static PooledProviderRequest ForPlan(StageRuntimePlan plan) => new(
        plan.ExecutionProvider ?? ExecutionProviderKind.Cpu,
        !plan.RequirePreferredExecutionProvider,
        plan.HasExplicitProviderPreference);

    public OnnxExecutionSessionFactory.PooledSessionRequest ToSessionRequest(
        string engineFamily,
        CancellationToken cancellationToken) =>
        new()
        {
            EngineFamily = engineFamily,
            Provider = Provider,
            CancellationToken = cancellationToken,
            AllowTrtInitFallback = AllowTrtInitFallback,
            HasExplicitProviderPreference = HasExplicitProviderPreference,
        };
}
