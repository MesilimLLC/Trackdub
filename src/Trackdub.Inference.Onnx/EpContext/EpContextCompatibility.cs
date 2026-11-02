using Microsoft.ML.OnnxRuntime;
using Trackdub.Inference.Runtime.TensorRtRtx;

namespace Trackdub.Inference.Onnx.EpContext;

/// <summary>
/// Reuse of a compiled EP-context model requires
/// <see cref="OrtCompiledModelCompatibility.EP_SUPPORTED_OPTIMAL"/> for the device the session
/// will select. Supported-but-recompilation-preferred, unsupported, not-applicable, missing
/// metadata, an empty device group, and API failures are cache misses. The original model is
/// then used, and a later compile can replace the artifact.
/// </summary>
internal static class EpContextCompatibility
{
    public static bool ShouldReuse(OrtCompiledModelCompatibility compatibility) =>
        compatibility == OrtCompiledModelCompatibility.EP_SUPPORTED_OPTIMAL;

    public static bool IsSelectedDeviceGroup(IReadOnlyList<OrtEpDevice> devices, string executionProviderName)
    {
        if (devices.Count == 0 || string.IsNullOrWhiteSpace(executionProviderName))
        {
            return false;
        }

        foreach (OrtEpDevice device in devices)
        {
            if (!string.Equals(device.EpName, executionProviderName, StringComparison.Ordinal))
            {
                return false;
            }
        }

        return true;
    }

    public static bool AllowsTensorRtRtxArtifact(string compiledModelPath) =>
        AllowsCachedArtifact(compiledModelPath, TensorRtRtxProviderConstants.PluginOrtExecutionProviderName);

    public static bool AllowsCachedArtifact(string compiledModelPath, string executionProviderName)
    {
        try
        {
            // Session creation appends exactly this device (AppendTensorRtRtxOrFallbackProvider).
            OrtEpDevice? sessionDevice = OrtEnv.Instance()
                .GetEpDevices()
                .FirstOrDefault(device => ProviderNames.IsTensorRtRtxDeviceCandidate(
                    device.EpName,
                    device.HardwareDevice.Type));
            OrtEpDevice[] selected = sessionDevice is null ? [] : [sessionDevice];
            if (!IsSelectedDeviceGroup(selected, executionProviderName))
            {
                return false;
            }

            string? info = OrtEnv.Instance().GetCompatibilityInfoFromModel(compiledModelPath, executionProviderName);
            if (string.IsNullOrWhiteSpace(info))
            {
                return false;
            }

            return ShouldReuse(OrtEnv.Instance().GetModelCompatibilityForEpDevices(selected, info));
        }
        catch (Exception ex) when (ex is OnnxRuntimeException
            or InvalidOperationException
            or DllNotFoundException
            or EntryPointNotFoundException
            or IOException
            or NotSupportedException
            or ArgumentException)
        {
            return false;
        }
    }
}
