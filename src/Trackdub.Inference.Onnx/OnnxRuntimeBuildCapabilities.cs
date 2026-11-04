using Trackdub.Domain;

namespace Trackdub.Inference.Onnx;

/// <summary>
/// Reports which execution-provider routes this build can actually create sessions for.
/// The portable net11.0 target carries plain ONNX Runtime plus the TensorRT RTX plugin
/// route; DirectML and the Windows ML catalog EPs require the net11.0-windows10.0.19041.0
/// build. Hardware, license, and bundle readiness are still decided by discovery probes.
/// </summary>
public static class OnnxRuntimeBuildCapabilities
{
#if WINDOWS
    public static bool SupportsWindowsMlRoutes { get; } = true;
#else
    public static bool SupportsWindowsMlRoutes { get; } = false;
#endif

    private static bool SupportsLinuxNativeRoutes => OperatingSystem.IsLinux();

    private static bool SupportsCoreMlRoute => OperatingSystem.IsMacOS();

    public static bool IsProviderSupportedInThisBuild(ExecutionProviderKind provider) =>
        provider switch
        {
            ExecutionProviderKind.DirectMl or
            ExecutionProviderKind.OpenVinoCatalog or
            ExecutionProviderKind.Qnn or
            ExecutionProviderKind.VitisAi => SupportsWindowsMlRoutes,

            ExecutionProviderKind.Migraphx => SupportsWindowsMlRoutes || SupportsLinuxNativeRoutes,

            ExecutionProviderKind.CoreMl => SupportsCoreMlRoute,

            _ => true,
        };
}
