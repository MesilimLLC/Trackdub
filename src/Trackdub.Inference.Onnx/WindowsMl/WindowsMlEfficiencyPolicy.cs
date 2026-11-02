using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Domain;

namespace Trackdub.Inference.Onnx.WindowsMl;

/// <summary>
/// Single owner of the "efficiency intent" decision: when a resolved
/// <see cref="WindowsMlExecutionDevicePolicy"/> is efficiency-flavored and the session's
/// provider is an NPU-capable Windows ML catalog EP, apply
/// <c>ep.dynamic.workload_type=Efficient</c> to the live session (MS "Efficiency tips for
/// Windows ML inference": portable NPU lever, expressed through ONNX Runtime rather than
/// vendor knobs).
/// </summary>
internal static class WindowsMlEfficiencyPolicy
{
    /// <summary>
    /// Device policies that express efficiency intent in addition to device selection.
    /// Explicit/MaxPerformance/PreferNpu/DefaultRender select devices only.
    /// </summary>
    internal static bool RequestsEfficientWorkload(WindowsMlExecutionDevicePolicy policy) =>
        policy is WindowsMlExecutionDevicePolicy.MaxEfficiency
            or WindowsMlExecutionDevicePolicy.MinOverallPower
            or WindowsMlExecutionDevicePolicy.MinPower;

    /// <summary>
    /// Windows ML catalog EPs whose devices are NPUs. Mirrors the NPU subset of
    /// ProviderNames.IsWindowsMlCatalogProvider (Migraphx is a catalog EP but a
    /// GPU EP; only NPU EPs act on the workload_type signal per the MS doc). Keep in sync with
    /// <c>Trackdub.Domain/Common/RuntimePlanning.cs</c> enum docs.
    /// </summary>
    internal static bool IsNpuCatalogProvider(ExecutionProviderKind provider) =>
        provider is ExecutionProviderKind.Qnn
            or ExecutionProviderKind.VitisAi
            or ExecutionProviderKind.OpenVinoCatalog;

    /// <summary>
    /// Applies <c>ep.dynamic.workload_type=Efficient</c> when policy and provider both call for
    /// it. Returns the honest outcome: <c>(false, null)</c> means "not applicable" (no call made);
    /// <c>(false, error)</c> means the call was attempted and failed — the caller must surface
    /// that reason, never report the efficiency mode as active.
    /// </summary>
    internal static (bool Applied, string? Error) ApplyToSession(
        InferenceSession session,
        ExecutionProviderKind selectedProvider,
        WindowsMlExecutionDevicePolicy devicePolicy)
    {
        if (!RequestsEfficientWorkload(devicePolicy) || !IsNpuCatalogProvider(selectedProvider))
        {
            return (false, null);
        }

        return OrtEpDynamicOptions.TrySetWorkloadType(
            session, OrtEpDynamicOptions.EfficientWorkload, out string? error)
            ? (true, null)
            : (false, error);
    }
}
