using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Dnnl;
using Trackdub.Inference.Runtime.TensorRtRtx;

namespace Trackdub.Inference.Onnx;

/// <summary>
/// Canonical execution-provider names, device-candidate predicates, and display labels.
/// Single source of truth for the EP names ORT reports via <c>OrtEpDevice.EpName</c>.
/// </summary>
internal static class ProviderNames
{
    // Canonical EP names returned by ORT device discovery. TensorRT RTX uses the standalone
    // EP ABI plugin, not the Windows ML catalog spelling. Single source of truth — must match the
    // name the plugin is registered under (see TensorRtRtxPluginService.RegistrationName); ORT reports
    // OrtEpDevice.EpName as that registration name.
    private const string TensorRtRtxExecutionProviderName = TensorRtRtxProviderConstants.PluginOrtExecutionProviderName;
    private const string DnnlExecutionProviderName = DnnlOrtProbe.OrtExecutionProviderName;
    private const string DnnlUpperExecutionProviderName = DnnlOrtProbe.OrtExecutionProviderNameUpper;
    internal const string DirectMlExecutionProviderName = "DmlExecutionProvider";
    private const string ClassicTensorRtExecutionProviderName = "TensorrtExecutionProvider";
    private const string DirectMlLongExecutionProviderName = "DirectMLExecutionProvider";

    private static readonly IReadOnlyDictionary<ExecutionProviderKind, string> ProviderLabels =
        new Dictionary<ExecutionProviderKind, string>
        {
            [ExecutionProviderKind.Cpu] = "cpu",
            [ExecutionProviderKind.DirectMl] = "dml",
            [ExecutionProviderKind.TensorRTRtx] = "tensorrt-rtx",
            [ExecutionProviderKind.OpenVino] = "openvino",
            [ExecutionProviderKind.CoreMl] = "coreml",
            [ExecutionProviderKind.Cuda] = "cuda",
            [ExecutionProviderKind.TensorRt] = "tensorrt",
            [ExecutionProviderKind.Migraphx] = "migraphx",
            [ExecutionProviderKind.Dnnl] = "dnnl",
            [ExecutionProviderKind.Qnn] = "qnn",
            [ExecutionProviderKind.OpenVinoCatalog] = "openvino-catalog",
            [ExecutionProviderKind.VitisAi] = "vitisai",
        };

    internal static string FormatProviderLabel(ExecutionProviderKind provider) =>
        ProviderLabels.TryGetValue(provider, out string? label)
            ? label
            : throw new ArgumentOutOfRangeException(nameof(provider), provider, "Unsupported execution provider kind.");

    internal static bool IsTensorRtDeviceCandidate(string epName, OrtHardwareDeviceType hardwareDeviceType) =>
        hardwareDeviceType is OrtHardwareDeviceType.GPU &&
        string.Equals(epName, ClassicTensorRtExecutionProviderName, StringComparison.Ordinal);

    internal static bool IsTensorRtRtxDeviceCandidate(string epName, OrtHardwareDeviceType hardwareDeviceType) =>
        // Only accept the single canonical standalone EP ABI plugin name.
        // "NvTensorRtExecutionProvider" and "TensorrtExecutionProvider" are the old CUDA-based TensorRT EP —
        // they must NOT be treated as TRT RTX candidates (different EP family, different options).
        hardwareDeviceType is OrtHardwareDeviceType.GPU &&
        string.Equals(epName, TensorRtRtxExecutionProviderName, StringComparison.Ordinal);

    internal static bool IsDirectMlExecutionProviderName(string epName) =>
        string.Equals(epName, DirectMlExecutionProviderName, StringComparison.OrdinalIgnoreCase) ||
        string.Equals(epName, DirectMlLongExecutionProviderName, StringComparison.OrdinalIgnoreCase);

    internal static bool IsDirectMlDeviceCandidate(string epName, OrtHardwareDeviceType hardwareDeviceType) =>
        hardwareDeviceType is OrtHardwareDeviceType.GPU &&
        IsDirectMlExecutionProviderName(epName);

    internal static bool IsDnnlExecutionProviderName(string epName) =>
        string.Equals(epName, DnnlExecutionProviderName, StringComparison.OrdinalIgnoreCase) ||
        string.Equals(epName, DnnlUpperExecutionProviderName, StringComparison.OrdinalIgnoreCase);

    internal static bool ShouldUseCatalogDevicePolicy(
        WindowsMlExecutionDevicePolicy devicePolicy,
        ExecutionProviderKind provider) =>
        OperatingSystem.IsWindows() &&
        devicePolicy != WindowsMlExecutionDevicePolicy.Explicit &&
        IsWindowsMlCatalogProvider(provider);

    /// <summary>
    /// WinML catalog EPs that can take ORT <c>SetEpSelectionPolicy</c> (device auto-select among
    /// registered catalog devices). DirectML stays on the legacy explicit-append path; native
    /// CUDA/TensorRT/TensorRT RTX/DNNL/CoreML are not Windows ML catalog routes.
    /// </summary>
    private static bool IsWindowsMlCatalogProvider(ExecutionProviderKind provider) =>
        provider is ExecutionProviderKind.Migraphx
            or ExecutionProviderKind.Qnn
            or ExecutionProviderKind.VitisAi
            or ExecutionProviderKind.OpenVinoCatalog;
}
