using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Inference.Onnx.Dnnl;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Migraphx;
using Trackdub.Inference.Onnx.WinMlCatalog;
using Trackdub.Inference.Onnx.TensorRtRtx;
using Trackdub.Inference.Runtime.Migraphx;
using Trackdub.Inference.Runtime.NativeCudaTensorRt;
using Trackdub.Inference.Runtime.Planning;
using Trackdub.Inference.Onnx.Worker;

namespace Trackdub.Inference.Onnx.Runtime.Planning;

public sealed class OnnxExecutionProviderDiscovery : IExecutionProviderDiscovery
{
    private readonly IOpenVinoAvailabilityProvider _openVino;
    private readonly ExecutionProviders.LinMl.ILinuxOpenVinoCatalog? _linuxOpenVinoCatalog;
    private readonly ILinuxNativeGpuRuntimeProbe _linuxRuntimeProbe;
    private readonly INativeCudaTensorRtWindowsPolicy _nativeCudaTensorRtWindowsPolicy;
    private readonly IMigraphxReadinessProbe _migraphxReadinessProbe;
    private readonly IDnnlReadinessProbe _dnnlReadinessProbe;
    private readonly IOpenVinoCatalogReadinessProbe _openVinoCatalogReadinessProbe;
    private readonly IQnnCatalogReadinessProbe _qnnCatalogReadinessProbe;
    private readonly IVitisAiCatalogReadinessProbe _vitisAiCatalogReadinessProbe;
    private readonly ITensorRtRtxReadinessProbe _tensorRtRtxReadinessProbe;
    private readonly Func<CancellationToken, Task<bool>> _isTensorRtRtxEnabled;
    private readonly Func<string?> _directMlUnavailableReason;
    private readonly IInferenceWorkerClient? _workerClient;

    /// <summary>Engine families the ORT 1.30 inference worker runs on CUDA (ADR-0017).</summary>
    internal static IReadOnlyList<string> WorkerCudaEngineFamilies { get; } = ["kokoro"];

    public OnnxExecutionProviderDiscovery()
        : this(new NullOpenVinoAvailabilityProvider())
    {
    }

    public OnnxExecutionProviderDiscovery(IOpenVinoAvailabilityProvider openVino)
        : this(openVino, new LinuxNativeGpuRuntimeProbe(), NullNativeCudaTensorRtWindowsPolicy.Instance)
    {
    }

    public OnnxExecutionProviderDiscovery(
        IOpenVinoAvailabilityProvider openVino,
        ILinuxNativeGpuRuntimeProbe linuxRuntimeProbe)
        : this(openVino, linuxRuntimeProbe, NullNativeCudaTensorRtWindowsPolicy.Instance)
    {
    }

    public OnnxExecutionProviderDiscovery(
        IOpenVinoAvailabilityProvider openVino,
        ILinuxNativeGpuRuntimeProbe linuxRuntimeProbe,
        INativeCudaTensorRtWindowsPolicy nativeCudaTensorRtWindowsPolicy)
        : this(openVino, linuxRuntimeProbe, nativeCudaTensorRtWindowsPolicy, new MigraphxReadinessProbe())
    {
    }

    public OnnxExecutionProviderDiscovery(
        IOpenVinoAvailabilityProvider openVino,
        ILinuxNativeGpuRuntimeProbe linuxRuntimeProbe,
        INativeCudaTensorRtWindowsPolicy nativeCudaTensorRtWindowsPolicy,
        IMigraphxReadinessProbe migraphxReadinessProbe)
        : this(
            openVino,
            linuxRuntimeProbe,
            nativeCudaTensorRtWindowsPolicy,
            migraphxReadinessProbe,
            new DnnlReadinessProbe(),
            new TensorRtRtxReadinessProbe(),
            new OpenVinoCatalogReadinessProbe(),
            new QnnCatalogReadinessProbe(),
            new VitisAiCatalogReadinessProbe())
    {
    }

    public OnnxExecutionProviderDiscovery(
        IOpenVinoAvailabilityProvider openVino,
        ILinuxNativeGpuRuntimeProbe linuxRuntimeProbe,
        INativeCudaTensorRtWindowsPolicy nativeCudaTensorRtWindowsPolicy,
        IMigraphxReadinessProbe migraphxReadinessProbe,
        IDnnlReadinessProbe dnnlReadinessProbe,
        ITensorRtRtxReadinessProbe tensorRtRtxReadinessProbe,
        IOpenVinoCatalogReadinessProbe openVinoCatalogReadinessProbe,
        IQnnCatalogReadinessProbe qnnCatalogReadinessProbe,
        IVitisAiCatalogReadinessProbe vitisAiCatalogReadinessProbe,
        Func<CancellationToken, Task<bool>>? isTensorRtRtxEnabled = null,
        Func<string?>? directMlUnavailableReason = null,
        IInferenceWorkerClient? workerClient = null,
        ExecutionProviders.LinMl.ILinuxOpenVinoCatalog? linuxOpenVinoCatalog = null)
    {
        _openVino = openVino ?? throw new ArgumentNullException(nameof(openVino));
        _linuxOpenVinoCatalog = linuxOpenVinoCatalog ?? DefaultLinuxOpenVinoCatalog();
        _linuxRuntimeProbe = linuxRuntimeProbe ?? throw new ArgumentNullException(nameof(linuxRuntimeProbe));
        _nativeCudaTensorRtWindowsPolicy = nativeCudaTensorRtWindowsPolicy
            ?? throw new ArgumentNullException(nameof(nativeCudaTensorRtWindowsPolicy));
        _migraphxReadinessProbe = migraphxReadinessProbe ?? throw new ArgumentNullException(nameof(migraphxReadinessProbe));
        _dnnlReadinessProbe = dnnlReadinessProbe ?? throw new ArgumentNullException(nameof(dnnlReadinessProbe));
        _tensorRtRtxReadinessProbe = tensorRtRtxReadinessProbe ?? throw new ArgumentNullException(nameof(tensorRtRtxReadinessProbe));
        _openVinoCatalogReadinessProbe = openVinoCatalogReadinessProbe
            ?? throw new ArgumentNullException(nameof(openVinoCatalogReadinessProbe));
        _qnnCatalogReadinessProbe = qnnCatalogReadinessProbe
            ?? throw new ArgumentNullException(nameof(qnnCatalogReadinessProbe));
        _vitisAiCatalogReadinessProbe = vitisAiCatalogReadinessProbe
            ?? throw new ArgumentNullException(nameof(vitisAiCatalogReadinessProbe));
        _isTensorRtRtxEnabled = isTensorRtRtxEnabled ?? (static _ => Task.FromResult(false));
        _directMlUnavailableReason = directMlUnavailableReason ?? DirectMlRuntimeProbe.GetUnavailableReason;
        _workerClient = workerClient;
    }

    private static ExecutionProviders.LinMl.ILinuxOpenVinoCatalog? DefaultLinuxOpenVinoCatalog() =>
#if LINML
        OperatingSystem.IsLinux() ? ExecutionProviders.LinMl.LinMlOpenVinoCatalog.Default : null;
#else
        null;
#endif

    public async Task<IReadOnlyList<ExecutionProviderAvailability>> DiscoverAsync(
        HardwareProfile hardwareProfile,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(hardwareProfile);
        cancellationToken.ThrowIfCancellationRequested();

#if WINDOWS
        // DirectML is only visible through the WinML catalog route on this ORT build (no native
        // OrtSessionOptionsAppendExecutionProvider_DML export). The catalog EP must be registered
        // before the first OrtEnv touch or GetEpDevices() never lists a DML device and every later
        // session silently appends nothing and falls back to CPU — which then fails smoke.
        WindowsMl.WindowsMlOnnxRuntimeNativeResolver.EnsureInitialized();
        await WindowsMl.WindowsMlProviderRegistrationPolicy.Shared
            .RegisterForReadinessAsync(ExecutionProviderKind.DirectMl, cancellationToken)
            .ConfigureAwait(false);
#endif

        bool allowNativeCudaTensorRtOnWindows = await _nativeCudaTensorRtWindowsPolicy
            .IsNativeProvidersAllowedOnWindowsAsync(cancellationToken)
            .ConfigureAwait(false);

        var availabilities = new List<ExecutionProviderAvailability>
        {
            new(ExecutionProviderKind.Cpu, true, "CPU execution is always available.")
        };

        bool isWindows = hardwareProfile.OperatingSystem.Equals("windows", StringComparison.OrdinalIgnoreCase);
        bool isMacOs = hardwareProfile.OperatingSystem.Equals("macos", StringComparison.OrdinalIgnoreCase);
        bool isLinux = hardwareProfile.OperatingSystem.Equals("linux", StringComparison.OrdinalIgnoreCase);
        // WSL2 has no NVIDIA PCI device in the hardware profile, but a loaded driver (GPU passthrough) is an NVIDIA GPU.
        bool isNvidiaGpu = (hardwareProfile.GpuDescription?.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase) ?? false)
            || (isLinux && _linuxRuntimeProbe.IsNvidiaDriverLoaded());
        bool isAmdGpu = MigraphxProviderOrdering.ShouldPreferMigraphxOnAmdGpu(hardwareProfile);
        bool tensorRtRtxEnabled = await _isTensorRtRtxEnabled(cancellationToken).ConfigureAwait(false);

        // Windows providers
        bool directMlBuildRoute = isWindows && hardwareProfile.HasGpu
            && OnnxRuntimeBuildCapabilities.SupportsWindowsMlRoutes;
        // The build can carry DirectML while the loaded onnxruntime.dll cannot (an app that
        // replaces Windows ML's runtime with a CPU/GPU ORT build). Report that, instead of
        // letting every DirectML session silently fall back to the CPU.
        string? directMlUnavailableReason = directMlBuildRoute ? _directMlUnavailableReason() : null;
        availabilities.Add(directMlBuildRoute && directMlUnavailableReason is null
            ? new(ExecutionProviderKind.DirectMl, true,
                "Windows ML legacy DirectML route can be probed on this machine.")
            : new(ExecutionProviderKind.DirectMl, false,
                directMlUnavailableReason
                ?? (isWindows && hardwareProfile.HasGpu
                    ? "DirectML requires the net11.0-windows10.0.19041.0 build; this build supports TensorRT RTX and CPU."
                    : "DirectML legacy GPU probing requires Windows with a GPU-capable Windows ML path.")));

        TensorRtRtxReadinessReport? tensorRtRtxReport = null;
        bool tensorRtAvailable = false;
        if (tensorRtRtxEnabled && (isWindows || isLinux) && isNvidiaGpu)
        {
            tensorRtRtxReport = await _tensorRtRtxReadinessProbe
                .ProbeAsync(allowProviderDownloads: false, cancellationToken)
                .ConfigureAwait(false);
            tensorRtAvailable = tensorRtRtxReport.IsReady;
        }
        availabilities.Add(tensorRtAvailable
            ? new(ExecutionProviderKind.TensorRTRtx, true,
                "NVIDIA GPU detected and TensorRT RTX EP ABI plugin is registered for installed-provider readiness.")
            : new(ExecutionProviderKind.TensorRTRtx, false,
                tensorRtRtxEnabled
                    ? ResolveTensorRtRtxUnavailableDetail(isWindows, isLinux, isNvidiaGpu, tensorRtRtxReport)
                    : "TensorRT RTX EP ABI plugin is disabled until the NVIDIA TensorRT RTX license is accepted in Model Manager."));

        availabilities.Add(await ResolveMigraphxAvailabilityAsync(
            isWindows,
            isLinux,
            isAmdGpu,
            cancellationToken).ConfigureAwait(false));

        // macOS providers
        availabilities.Add(isMacOs
            ? new(ExecutionProviderKind.CoreMl, true,
                "CoreML is always available on macOS 10.15+.")
            : new(ExecutionProviderKind.CoreMl, false,
                "CoreML is macOS-only."));

        // Native ORT CUDA / TensorRT (Linux always; Windows when advanced setting enabled)
        bool linuxNvidiaDriverLoaded = isLinux && isNvidiaGpu && _linuxRuntimeProbe.IsNvidiaDriverLoaded();
        bool linuxTensorRtAvailable = linuxNvidiaDriverLoaded && _linuxRuntimeProbe.IsNativeTensorRtAvailable();

        bool windowsNativeProbe = isWindows && isNvidiaGpu && allowNativeCudaTensorRtOnWindows;
        bool windowsCudaListed = windowsNativeProbe && CudaOrtProbe.IsCudaProviderListed();
        bool windowsTensorRtLibs = windowsNativeProbe && NativeTensorRtLibraryProbe.IsNativeTensorRtAvailable();
        bool windowsTensorRtListed = windowsNativeProbe && TensorRtRtxOrtProbe.IsNativeTensorRtProviderListed();
        bool windowsCudaAvailable = windowsCudaListed;
        bool windowsTensorRtAvailable = windowsTensorRtLibs && windowsTensorRtListed;

        bool linuxCudaUsable = linuxNvidiaDriverLoaded && _linuxRuntimeProbe.IsCudaOrtProviderAvailable();

        ExecutionProviderAvailability cudaAvailability = ResolveCudaAvailability(
            isLinux,
            linuxNvidiaDriverLoaded,
            linuxCudaUsable,
            isWindows,
            isNvidiaGpu,
            allowNativeCudaTensorRtOnWindows,
            windowsCudaAvailable);
        if (!cudaAvailability.IsAvailable && isWindows && isNvidiaGpu &&
            await ResolveWorkerCudaAvailabilityAsync(cancellationToken).ConfigureAwait(false) is { } workerCuda)
        {
            cudaAvailability = workerCuda;
        }

        availabilities.Add(cudaAvailability);

        availabilities.Add(ResolveTensorRtAvailability(
            isLinux,
            linuxNvidiaDriverLoaded,
            linuxTensorRtAvailable,
            isWindows,
            isNvidiaGpu,
            allowNativeCudaTensorRtOnWindows,
            windowsTensorRtAvailable,
            windowsTensorRtLibs,
            windowsTensorRtListed));

        DnnlReadinessReport dnnlReport = await _dnnlReadinessProbe
            .ProbeAsync(allowProviderDownloads: false, cancellationToken)
            .ConfigureAwait(false);
        availabilities.Add(dnnlReport.IsReady
            ? new(ExecutionProviderKind.Dnnl, true, dnnlReport.Detail)
            : new(ExecutionProviderKind.Dnnl, false, dnnlReport.Detail));

        if (_linuxOpenVinoCatalog is not null)
        {
            // OpenVINO on Linux — sessions append the LinML catalog's plugin EP. Discovery never
            // downloads: it registers an installed plugin and reports ready only with a usable device.
            ExecutionProviders.LinMl.LinuxPluginCatalogResult openVinoCatalog = await _linuxOpenVinoCatalog
                .EnsureReadyAsync(allowDownloads: false, cancellationToken)
                .ConfigureAwait(false);
            availabilities.Add(new(ExecutionProviderKind.OpenVino, openVinoCatalog.IsReady, openVinoCatalog.Detail));
        }
        else
        {
            // OpenVINO — via runtime availability check
            availabilities.Add(_openVino.IsAvailable
                ? new(ExecutionProviderKind.OpenVino, true, "OpenVINO runtime loaded.")
                : new(ExecutionProviderKind.OpenVino, false, "OpenVINO not installed."));
        }

        availabilities.Add(await ResolveWinMlCatalogEpAvailabilityAsync(
            ExecutionProviderKind.OpenVinoCatalog,
            isWindows,
            _openVinoCatalogReadinessProbe.ProbeAsync,
            cancellationToken).ConfigureAwait(false));
        availabilities.Add(await ResolveWinMlCatalogEpAvailabilityAsync(
            ExecutionProviderKind.Qnn,
            isWindows,
            _qnnCatalogReadinessProbe.ProbeAsync,
            cancellationToken).ConfigureAwait(false));
        availabilities.Add(await ResolveWinMlCatalogEpAvailabilityAsync(
            ExecutionProviderKind.VitisAi,
            isWindows,
            _vitisAiCatalogReadinessProbe.ProbeAsync,
            cancellationToken).ConfigureAwait(false));

        return availabilities;
    }

    /// <summary>
    /// CUDA through the inference worker, for the engine families it hosts. Null when the worker is
    /// not installed or its handshake fails; unavailable (with the worker's reason) when it runs but
    /// cannot use CUDA, for example without cuDNN 9.
    /// </summary>
    private async Task<ExecutionProviderAvailability?> ResolveWorkerCudaAvailabilityAsync(CancellationToken cancellationToken)
    {
        if (_workerClient is not { IsInstalled: true } ||
            await _workerClient.TryGetHelloAsync(cancellationToken).ConfigureAwait(false) is not { } hello)
        {
            return null;
        }

        return hello.CudaAvailable
            ? new(ExecutionProviderKind.Cuda, true,
                $"CUDA through the ONNX Runtime {hello.OnnxRuntimeVersion} inference worker: {hello.CudaDetail}",
                WorkerCudaEngineFamilies)
            : new(ExecutionProviderKind.Cuda, false,
                $"The inference worker cannot use CUDA: {hello.CudaDetail}");
    }

    private static ExecutionProviderAvailability ResolveCudaAvailability(
        bool isLinux,
        bool linuxNvidiaDriverLoaded,
        bool linuxCudaUsable,
        bool isWindows,
        bool isNvidiaGpu,
        bool allowNativeCudaTensorRtOnWindows,
        bool windowsCudaAvailable)
    {
        if (isLinux)
        {
            if (!linuxNvidiaDriverLoaded)
            {
                return new(ExecutionProviderKind.Cuda, false, "No loaded NVIDIA driver detected on Linux.");
            }

            return linuxCudaUsable
                ? new(ExecutionProviderKind.Cuda, true,
                    "NVIDIA GPU detected and CUDAExecutionProvider is usable.")
                : new(ExecutionProviderKind.Cuda, false,
                    "NVIDIA GPU detected, but CUDAExecutionProvider is not usable in the loaded ONNX Runtime "
                    + "(it needs a CUDA-enabled build and the CUDA libraries).");
        }

        if (isWindows && allowNativeCudaTensorRtOnWindows)
        {
            return windowsCudaAvailable
                ? new(ExecutionProviderKind.Cuda, true,
                    "Native CUDAExecutionProvider is listed; use for explicit CUDA requests (advanced).")
                : new(ExecutionProviderKind.Cuda, false,
                    isNvidiaGpu
                        ? "CUDAExecutionProvider is not listed. Install NVIDIA drivers, CUDA, and ONNX Runtime GPU package."
                        : "Native CUDA on Windows requires an NVIDIA GPU.");
        }

        return new(ExecutionProviderKind.Cuda, false,
            isWindows
                ? "CUDA native EP on Windows is off by default. Use TensorRT RTX instead."
                : "CUDA native EP is available on Linux and Windows (when enabled).");
    }

    private static ExecutionProviderAvailability ResolveTensorRtAvailability(
        bool isLinux,
        bool linuxNvidiaDriverLoaded,
        bool linuxTensorRtAvailable,
        bool isWindows,
        bool isNvidiaGpu,
        bool allowNativeCudaTensorRtOnWindows,
        bool windowsTensorRtAvailable,
        bool windowsTensorRtLibs,
        bool windowsTensorRtListed)
    {
        if (isLinux)
        {
            return linuxTensorRtAvailable
                ? new(ExecutionProviderKind.TensorRt, true,
                    "NVIDIA driver, libnvinfer and TensorrtExecutionProvider are usable.")
                : new(ExecutionProviderKind.TensorRt, false,
                    ResolveLinuxTensorRtUnavailableDetail(isLinux: true, linuxNvidiaDriverLoaded));
        }

        if (isWindows && allowNativeCudaTensorRtOnWindows)
        {
            return windowsTensorRtAvailable
                ? new(ExecutionProviderKind.TensorRt, true,
                    "Native TensorRT libraries and TensorrtExecutionProvider are available (advanced).")
                : new(ExecutionProviderKind.TensorRt, false,
                    ResolveWindowsNativeTensorRtUnavailableDetail(isNvidiaGpu, windowsTensorRtLibs, windowsTensorRtListed));
        }

        return new(ExecutionProviderKind.TensorRt, false,
            isWindows
                ? "Native TensorRT on Windows is off by default. Use TensorRT RTX instead."
                : "TensorRT native EP is available on Linux and Windows (when enabled).");
    }

    private static string ResolveWindowsNativeTensorRtUnavailableDetail(
        bool isNvidiaGpu,
        bool librariesPresent,
        bool ortProviderListed)
    {
        if (!isNvidiaGpu)
        {
            return "Native TensorRT on Windows requires an NVIDIA GPU.";
        }

        if (!librariesPresent && !ortProviderListed)
        {
            return "TensorRT libraries and TensorrtExecutionProvider are not available.";
        }

        if (!librariesPresent)
        {
            return "nvinfer.dll was not found on PATH or common install locations.";
        }

        return "TensorrtExecutionProvider is not listed by ONNX Runtime.";
    }

    private static string ResolveLinuxTensorRtUnavailableDetail(bool isLinux, bool linuxNvidiaDriverLoaded)
    {
        if (!isLinux)
        {
            return "TensorRT native EP is Linux-only.";
        }

        return linuxNvidiaDriverLoaded
            ? "libnvinfer not found or TensorrtExecutionProvider is not usable; TensorRT native EP is unavailable."
            : "No loaded NVIDIA driver detected on Linux; TensorRT native EP is unavailable.";
    }

    private static string ResolveTensorRtRtxUnavailableDetail(
        bool isWindows,
        bool isLinux,
        bool isNvidiaGpu,
        TensorRtRtxReadinessReport? readinessReport)
    {
        if (!isWindows && !isLinux)
        {
            return "TensorRT RTX EP ABI plugin is not supported on this platform.";
        }

        if (!isNvidiaGpu)
        {
            return "TensorRT RTX path requires an NVIDIA GPU.";
        }

        return readinessReport?.Detail ?? "TensorRT RTX EP ABI plugin is not registered.";
    }

    private async Task<ExecutionProviderAvailability> ResolveMigraphxAvailabilityAsync(
        bool isWindows,
        bool isLinux,
        bool isAmdGpu,
        CancellationToken cancellationToken)
    {
        if (!isWindows && !isLinux)
        {
            return new(
                ExecutionProviderKind.Migraphx,
                false,
                "MIGraphX is available on Windows (WinML catalog) and Linux (LinML catalog plugin EP) only.");
        }

        if (isWindows && !OnnxRuntimeBuildCapabilities.SupportsWindowsMlRoutes)
        {
            return new(
                ExecutionProviderKind.Migraphx,
                false,
                "MIGraphX requires the net11.0-windows10.0.19041.0 build on Windows, or Linux with ROCm and MIGraphX 2.17 or newer.");
        }

        if (isWindows && !isAmdGpu)
        {
            return new(
                ExecutionProviderKind.Migraphx,
                false,
                "MIGraphX WinML route requires an AMD GPU with the documented driver version.");
        }

        if (isLinux && !isAmdGpu)
        {
            return new(
                ExecutionProviderKind.Migraphx,
                false,
                "MIGraphX on Linux requires an AMD GPU and ROCm with MIGraphX 2.17 or newer.");
        }

        MigraphxReadinessReport report = await _migraphxReadinessProbe
            .ProbeAsync(allowProviderDownloads: false, cancellationToken)
            .ConfigureAwait(false);

        return new(
            ExecutionProviderKind.Migraphx,
            report.IsReady,
            report.Detail);
    }

    private static async Task<ExecutionProviderAvailability> ResolveWinMlCatalogEpAvailabilityAsync(
        ExecutionProviderKind provider,
        bool isWindows,
        Func<bool, CancellationToken, Task<WinMlCatalogReadinessReport>> probeAsync,
        CancellationToken cancellationToken)
    {
        if (!isWindows)
        {
            return new(provider, false, $"{provider} WinML catalog route is Windows-only.");
        }

        if (!OnnxRuntimeBuildCapabilities.SupportsWindowsMlRoutes)
        {
            return new(provider, false,
                $"{provider} requires the net11.0-windows10.0.19041.0 build for the Windows ML catalog route.");
        }

        WinMlCatalogReadinessReport report = await probeAsync(false, cancellationToken).ConfigureAwait(false);
        return new(provider, report.IsReady, report.Detail);
    }

    private sealed class NullNativeCudaTensorRtWindowsPolicy : INativeCudaTensorRtWindowsPolicy
    {
        public static NullNativeCudaTensorRtWindowsPolicy Instance { get; } = new();

        public Task<bool> IsNativeProvidersAllowedOnWindowsAsync(CancellationToken cancellationToken = default) =>
            Task.FromResult(false);
    }
}
