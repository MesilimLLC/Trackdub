using Trackdub.Contracts.ApplicationContracts;

namespace Trackdub.Inference.Runtime.TensorRtRtx;

public static class TensorRtRtxProviderConstants
{
    public const string PluginOrtExecutionProviderName = "NvTensorRTRTXExecutionProvider";

    public const string PluginLibraryFileNameWindows = "onnxruntime_providers_nv_tensorrt_rtx.dll";

    public const string PluginLibraryFileNameLinux = "libonnxruntime_providers_nv_tensorrt_rtx.so";

    public const string TensorRtRuntimeFileNameWindows = "tensorrt_rtx_1_7.dll";

    public const string TensorRtRuntimeFileNameLinux = "libtensorrt_rtx.so";

    public const string TensorRtOnnxParserFileNameWindows = "tensorrt_onnxparser_rtx_1_7.dll";

    public const string TensorRtOnnxParserFileNameLinux = "libtensorrt_onnxparser_rtx.so";

    public const string PluginDirectoryEnvironmentVariable = "TRACKDUB_TRT_RTX_EP_DIR";

    public const string CudaRuntimeBinDirectoryEnvironmentVariable = "TRACKDUB_CUDA_BIN_DIR";

    // The pin is per-platform. Windows installs NVIDIA's EP ABI GitHub bundle from
    // runtime/trt-rtx-ep.manifest.json. Linux installs through LinML.Catalog, which pins its own
    // onnxruntime-ep-nv-tensorrt-rtx wheel (0.5.0, TensorRT-RTX 1.7.1 as of LinML.Catalog 0.6.0), so
    // the Linux values must follow LinML.Catalog, not the manifest.
    public const string BundledVersionWindows = "0.5.0";

    public const string BundledVersionLinux = "0.5.0";

    public const string BundledCudaVariant = "cu13";

    /// <summary>
    /// TensorRT-RTX runtime vendored by the pinned EP ABI bundle. Tracked separately from the EP ABI
    /// version because the runtime lineage can move without the plugin version moving.
    /// </summary>
    public static string BundledTrtRtxRuntimeVersion =>
        OperatingSystem.IsLinux() ? BundledTrtRtxRuntimeVersionLinux : BundledTrtRtxRuntimeVersionWindows;

    public const string BundledTrtRtxRuntimeVersionWindows = "1.7.1";

    public const string BundledTrtRtxRuntimeVersionLinux = "1.7.1";

    public static string BundledVersion =>
        OperatingSystem.IsLinux() ? BundledVersionLinux : BundledVersionWindows;

    /// <summary>
    /// Ceiling, in bytes, on the kernel (tactic) shared memory TensorRT-RTX may use when it builds
    /// engines on Windows. A TensorRT-RTX launch after a DirectML dispatch in the same process fails
    /// with CUDA_ERROR_INVALID_VALUE unless the engine was built under this limit (NVIDIA TensorRT-RTX
    /// 1.7 release notes, known issues). Trackdub mixes DirectML and TensorRT-RTX stages in one
    /// process, so every Windows engine is built under it.
    /// </summary>
    public const long WindowsDirectMlInteropSharedMemoryLimitBytes = 99 * 1024;

    /// <summary>
    /// Version identity for anything that must invalidate with the engine cache (smoke verdicts,
    /// EP-context stamps, the runtime cache directory): changes when the EP ABI plugin or the
    /// TRT-RTX runtime changes, and on Windows when the engine-build shared-memory limit changes,
    /// because engines and kernels built without the limit fail after a DirectML dispatch.
    /// </summary>
    public static string BundledFingerprintVersion =>
        $"{BundledVersion}+{BundledCudaVariant}+trt-rtx-{BundledTrtRtxRuntimeVersion}{EngineBuildPolicyTag}";

    private static string EngineBuildPolicyTag =>
        OperatingSystem.IsWindows() ? $"+smem{WindowsDirectMlInteropSharedMemoryLimitBytes / 1024}k" : string.Empty;

    public const string NativeOrtExecutionProviderName = "TensorrtExecutionProvider";

    public static string ProviderIdPluginEpAbi => TensorRtRtxProviderIds.PluginEpAbi;

    public static string ProviderIdNative => TensorRtRtxProviderIds.Native;

    public static string PluginLibraryFileName =>
        OperatingSystem.IsLinux()
            ? PluginLibraryFileNameLinux
            : PluginLibraryFileNameWindows;

    public static string TensorRtRuntimeFileName =>
        OperatingSystem.IsLinux()
            ? TensorRtRuntimeFileNameLinux
            : TensorRtRuntimeFileNameWindows;

    public static string TensorRtOnnxParserFileName =>
        OperatingSystem.IsLinux()
            ? TensorRtOnnxParserFileNameLinux
            : TensorRtOnnxParserFileNameWindows;

    public static IReadOnlyList<string> RequiredPluginFileNames =>
    [
        PluginLibraryFileName,
        TensorRtRuntimeFileName,
        TensorRtOnnxParserFileName
    ];

    public const string LinuxInstallHint =
        "Use Install in Model Manager to download NVIDIA's TensorRT-RTX EP ABI plugin (v" + BundledVersionLinux + " " + BundledCudaVariant
        + ") through the LinML catalog, then refresh readiness. "
        + "The plugin ships its own CUDA 13 runtime (libcudart.so.13). "
        + "To use your own build instead, set " + PluginDirectoryEnvironmentVariable + ".";

    public const string WindowsInstallHint =
        "Use Install in Model Manager to download TensorRT-RTX-EP-ABI v" + BundledVersionWindows + " " + BundledCudaVariant
        + " for win-x64, or run tools/dev/Fetch-TrtRtxEp.ps1, then refresh readiness. "
        + "No CUDA Toolkit is required: the CUDA runtime is statically linked into the TensorRT-RTX 1.7 bundle.";

    public static string GetDefaultInstallDirectory(string userDataRoot, string runtimeIdentifier) =>
        Path.Join(
            Path.GetFullPath(Environment.ExpandEnvironmentVariables(userDataRoot)),
            "Providers",
            "trt-rtx",
            BundledVersion,
            BundledCudaVariant,
            runtimeIdentifier);

    /// <summary>
    /// True when <paramref name="candidateDirectory"/> is a different version/variant directory under
    /// the same managed <c>Providers/trt-rtx</c> root as <paramref name="defaultInstallDirectory"/>.
    /// </summary>
    public static bool IsSupersededManagedInstallDirectory(string? candidateDirectory, string? defaultInstallDirectory)
    {
        if (string.IsNullOrWhiteSpace(candidateDirectory) || string.IsNullOrWhiteSpace(defaultInstallDirectory))
        {
            return false;
        }

        try
        {
            string candidate = Path.TrimEndingDirectorySeparator(Path.GetFullPath(candidateDirectory));
            string current = Path.TrimEndingDirectorySeparator(
                Path.GetFullPath(Environment.ExpandEnvironmentVariables(defaultInstallDirectory)));

            // <root>/<version>/<cudaVariant>/<rid>
            string? managedRoot = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(current)));
            if (string.IsNullOrEmpty(managedRoot) ||
                string.Equals(candidate, current, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            string? candidateRoot = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(candidate)));
            return string.Equals(candidateRoot, managedRoot, StringComparison.OrdinalIgnoreCase);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
        {
            return false;
        }
    }
}
