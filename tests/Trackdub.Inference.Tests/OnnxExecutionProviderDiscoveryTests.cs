using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Domain;
using Trackdub.Inference.Onnx;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Runtime.Planning;
using Xunit;

namespace Trackdub.Inference.Tests;

public sealed class OnnxExecutionProviderDiscoveryTests
{
    [Fact]
    public async Task DiscoverAsync_LinuxWithNvidiaDriverWithoutTensorRt_ReportsCudaOnly()
    {
        var discovery = new OnnxExecutionProviderDiscovery(
            new StubOpenVinoProvider(false),
            new StubLinuxNativeGpuRuntimeProbe(nvidiaDriverLoaded: true, nativeTensorRtAvailable: false));

        IReadOnlyList<ExecutionProviderAvailability> availabilities = await discovery.DiscoverAsync(
            new HardwareProfile("linux", "x64", HasGpu: true, GpuDescription: "NVIDIA RTX 4090"),
            CancellationToken.None);

        Assert.True(availabilities.Single(a => a.Provider == ExecutionProviderKind.Cuda).IsAvailable);
        ExecutionProviderAvailability tensorRt = availabilities.Single(a => a.Provider == ExecutionProviderKind.TensorRt);
        Assert.False(tensorRt.IsAvailable);
        Assert.Contains("libnvinfer", tensorRt.Detail, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task DiscoverAsync_LinuxWithLoadedNvidiaDriverButNoNvidiaPciDevice_TreatsTheGpuAsNvidia()
    {
        // WSL2 passthrough: the hardware profile lists a Microsoft virtual adapter, not the NVIDIA GPU.
        var discovery = new OnnxExecutionProviderDiscovery(
            new StubOpenVinoProvider(false),
            new StubLinuxNativeGpuRuntimeProbe(nvidiaDriverLoaded: true, nativeTensorRtAvailable: false));

        IReadOnlyList<ExecutionProviderAvailability> availabilities = await discovery.DiscoverAsync(
            new HardwareProfile("linux", "x64", HasGpu: true, GpuDescription: "Microsoft Corporation Device 008e"),
            CancellationToken.None);

        Assert.True(availabilities.Single(a => a.Provider == ExecutionProviderKind.Cuda).IsAvailable);
    }

    [Fact]
    public async Task DiscoverAsync_LinuxWithNvidiaDriverButUnusableCudaProvider_ReportsCudaUnavailable()
    {
        var discovery = new OnnxExecutionProviderDiscovery(
            new StubOpenVinoProvider(false),
            new StubLinuxNativeGpuRuntimeProbe(nvidiaDriverLoaded: true, nativeTensorRtAvailable: false, cudaOrtProviderAvailable: false));

        IReadOnlyList<ExecutionProviderAvailability> availabilities = await discovery.DiscoverAsync(
            new HardwareProfile("linux", "x64", HasGpu: true, GpuDescription: "NVIDIA RTX 4090"),
            CancellationToken.None);

        ExecutionProviderAvailability cuda = availabilities.Single(a => a.Provider == ExecutionProviderKind.Cuda);
        Assert.False(cuda.IsAvailable);
        Assert.Contains("not usable", cuda.Detail, StringComparison.Ordinal);
    }

    [Fact]
    public async Task DiscoverAsync_LinuxWithNvidiaDriverAndTensorRt_ReportsTensorRtAvailable()
    {
        var discovery = new OnnxExecutionProviderDiscovery(
            new StubOpenVinoProvider(false),
            new StubLinuxNativeGpuRuntimeProbe(nvidiaDriverLoaded: true, nativeTensorRtAvailable: true));

        IReadOnlyList<ExecutionProviderAvailability> availabilities = await discovery.DiscoverAsync(
            new HardwareProfile("linux", "x64", HasGpu: true, GpuDescription: "NVIDIA RTX 4090"),
            CancellationToken.None);

        Assert.True(availabilities.Single(a => a.Provider == ExecutionProviderKind.Cuda).IsAvailable);
        Assert.True(availabilities.Single(a => a.Provider == ExecutionProviderKind.TensorRt).IsAvailable);
    }

    [Fact]
    public async Task DiscoverAsync_LinuxWithoutNvidiaDriver_ReportsCudaAndTensorRtUnavailable()
    {
        var discovery = new OnnxExecutionProviderDiscovery(
            new StubOpenVinoProvider(false),
            new StubLinuxNativeGpuRuntimeProbe(nvidiaDriverLoaded: false, nativeTensorRtAvailable: true));

        IReadOnlyList<ExecutionProviderAvailability> availabilities = await discovery.DiscoverAsync(
            new HardwareProfile("linux", "x64", HasGpu: true, GpuDescription: "NVIDIA RTX 4090"),
            CancellationToken.None);

        Assert.False(availabilities.Single(a => a.Provider == ExecutionProviderKind.Cuda).IsAvailable);
        Assert.False(availabilities.Single(a => a.Provider == ExecutionProviderKind.TensorRt).IsAvailable);
    }

    [Fact]
    public async Task DiscoverAsync_Windows_IncludesWinMlCatalogEpProviders()
    {
        var discovery = new OnnxExecutionProviderDiscovery(
            new StubOpenVinoProvider(false),
            new StubLinuxNativeGpuRuntimeProbe(nvidiaDriverLoaded: false, nativeTensorRtAvailable: false),
            new StubNativeCudaTensorRtWindowsPolicy(allowed: false),
            new StubMigraphxReadinessProbe(),
            new StubDnnlReadinessProbe(isReady: false),
            new StubTensorRtRtxReadinessProbe(),
            new StubWinMlCatalogReadinessProbe(),
            new StubWinMlCatalogReadinessProbe(),
            new StubWinMlCatalogReadinessProbe());

        IReadOnlyList<ExecutionProviderAvailability> availabilities = await discovery.DiscoverAsync(
            new HardwareProfile("windows", "x64", HasGpu: true, GpuDescription: "NVIDIA RTX 5080"),
            CancellationToken.None);

        Assert.Contains(availabilities, a => a.Provider == ExecutionProviderKind.Qnn);
        Assert.Contains(availabilities, a => a.Provider == ExecutionProviderKind.OpenVinoCatalog);
        Assert.Contains(availabilities, a => a.Provider == ExecutionProviderKind.VitisAi);
    }

    [Fact]
    public async Task DiscoverAsync_Windows_DirectMlFollowsBuildCapabilities()
    {
        var discovery = new OnnxExecutionProviderDiscovery(
            new StubOpenVinoProvider(false),
            new StubLinuxNativeGpuRuntimeProbe(nvidiaDriverLoaded: false, nativeTensorRtAvailable: false),
            new StubNativeCudaTensorRtWindowsPolicy(allowed: false),
            new StubMigraphxReadinessProbe(),
            new StubDnnlReadinessProbe(isReady: false),
            new StubTensorRtRtxReadinessProbe(),
            new StubWinMlCatalogReadinessProbe(),
            new StubWinMlCatalogReadinessProbe(),
            new StubWinMlCatalogReadinessProbe(),
            directMlUnavailableReason: static () => null);

        IReadOnlyList<ExecutionProviderAvailability> availabilities = await discovery.DiscoverAsync(
            new HardwareProfile("windows", "x64", HasGpu: true, GpuDescription: "NVIDIA RTX 5070"),
            CancellationToken.None);

        ExecutionProviderAvailability directMl = Assert.Single(
            availabilities, a => a.Provider == ExecutionProviderKind.DirectMl);
        if (OnnxRuntimeBuildCapabilities.SupportsWindowsMlRoutes)
        {
            Assert.True(directMl.IsAvailable);
        }
        else
        {
            Assert.False(directMl.IsAvailable);
            Assert.Contains("net11.0-windows10.0.19041.0", directMl.Detail);
        }
    }

    [Fact]
    public async Task DiscoverAsync_Windows_LoadedRuntimeWithoutDirectMl_ReportsDirectMlUnavailable()
    {
        var discovery = new OnnxExecutionProviderDiscovery(
            new StubOpenVinoProvider(false),
            new StubLinuxNativeGpuRuntimeProbe(nvidiaDriverLoaded: false, nativeTensorRtAvailable: false),
            new StubNativeCudaTensorRtWindowsPolicy(allowed: false),
            new StubMigraphxReadinessProbe(),
            new StubDnnlReadinessProbe(isReady: false),
            new StubTensorRtRtxReadinessProbe(),
            new StubWinMlCatalogReadinessProbe(),
            new StubWinMlCatalogReadinessProbe(),
            new StubWinMlCatalogReadinessProbe(),
            directMlUnavailableReason: static () => "The loaded ONNX Runtime (1.30.0) cannot run DirectML.");

        IReadOnlyList<ExecutionProviderAvailability> availabilities = await discovery.DiscoverAsync(
            new HardwareProfile("windows", "x64", HasGpu: true, GpuDescription: "NVIDIA RTX 5070"),
            CancellationToken.None);

        ExecutionProviderAvailability directMl = Assert.Single(
            availabilities, a => a.Provider == ExecutionProviderKind.DirectMl);
        Assert.False(directMl.IsAvailable);
        if (OnnxRuntimeBuildCapabilities.SupportsWindowsMlRoutes)
        {
            Assert.Equal("The loaded ONNX Runtime (1.30.0) cannot run DirectML.", directMl.Detail);
        }
    }

    [Fact]
    public async Task DiscoverAsync_Windows_WinMlCatalogProvidersFollowBuildCapabilities()
    {
        var openVinoCatalogProbe = new StubWinMlCatalogReadinessProbe();
        var qnnProbe = new StubWinMlCatalogReadinessProbe();
        var vitisAiProbe = new StubWinMlCatalogReadinessProbe();
        var discovery = new OnnxExecutionProviderDiscovery(
            new StubOpenVinoProvider(false),
            new StubLinuxNativeGpuRuntimeProbe(nvidiaDriverLoaded: false, nativeTensorRtAvailable: false),
            new StubNativeCudaTensorRtWindowsPolicy(allowed: false),
            new StubMigraphxReadinessProbe(),
            new StubDnnlReadinessProbe(isReady: false),
            new StubTensorRtRtxReadinessProbe(),
            openVinoCatalogProbe,
            qnnProbe,
            vitisAiProbe);

        IReadOnlyList<ExecutionProviderAvailability> availabilities = await discovery.DiscoverAsync(
            new HardwareProfile("windows", "x64", HasGpu: true, GpuDescription: "NVIDIA RTX 5070"),
            CancellationToken.None);

        if (OnnxRuntimeBuildCapabilities.SupportsWindowsMlRoutes)
        {
            Assert.True(openVinoCatalogProbe.CallCount > 0);
        }
        else
        {
            // The portable build cannot load the WinML catalog route; probes must not run.
            Assert.Equal(0, openVinoCatalogProbe.CallCount);
            Assert.Equal(0, qnnProbe.CallCount);
            Assert.Equal(0, vitisAiProbe.CallCount);
            foreach (ExecutionProviderAvailability availability in availabilities.Where(
                         a => a.Provider is ExecutionProviderKind.OpenVinoCatalog
                             or ExecutionProviderKind.Qnn
                             or ExecutionProviderKind.VitisAi))
            {
                Assert.False(availability.IsAvailable);
                Assert.Contains("net11.0-windows10.0.19041.0", availability.Detail);
            }
        }
    }

    [Fact]
    public async Task DiscoverAsync_WhenTensorRtRtxLicenseIsNotAccepted_DoesNotProbeOrRegisterPlugin()
    {
        var tensorRtProbe = new StubTensorRtRtxReadinessProbe();
        var discovery = new OnnxExecutionProviderDiscovery(
            new StubOpenVinoProvider(false),
            new StubLinuxNativeGpuRuntimeProbe(nvidiaDriverLoaded: false, nativeTensorRtAvailable: false),
            new StubNativeCudaTensorRtWindowsPolicy(allowed: false),
            new StubMigraphxReadinessProbe(),
            new StubDnnlReadinessProbe(isReady: false),
            tensorRtProbe,
            new StubWinMlCatalogReadinessProbe(),
            new StubWinMlCatalogReadinessProbe(),
            new StubWinMlCatalogReadinessProbe(),
            isTensorRtRtxEnabled: static _ => Task.FromResult(false));

        IReadOnlyList<ExecutionProviderAvailability> availabilities = await discovery.DiscoverAsync(
            new HardwareProfile("windows", "x64", HasGpu: true, GpuDescription: "NVIDIA RTX 5070"),
            CancellationToken.None);

        ExecutionProviderAvailability tensorRt = Assert.Single(
            availabilities,
            availability => availability.Provider == ExecutionProviderKind.TensorRTRtx);
        Assert.False(tensorRt.IsAvailable);
        Assert.Contains("license", tensorRt.Detail, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(0, tensorRtProbe.CallCount);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task DiscoverAsync_IncludesDnnlReadinessProbeResult(bool dnnlReady)
    {
        var discovery = new OnnxExecutionProviderDiscovery(
            new StubOpenVinoProvider(false),
            new StubLinuxNativeGpuRuntimeProbe(nvidiaDriverLoaded: false, nativeTensorRtAvailable: false),
            new StubNativeCudaTensorRtWindowsPolicy(allowed: false),
            new StubMigraphxReadinessProbe(),
            new StubDnnlReadinessProbe(dnnlReady),
            new StubTensorRtRtxReadinessProbe(),
            new StubWinMlCatalogReadinessProbe(),
            new StubWinMlCatalogReadinessProbe(),
            new StubWinMlCatalogReadinessProbe());

        IReadOnlyList<ExecutionProviderAvailability> availabilities = await discovery.DiscoverAsync(
            new HardwareProfile("linux", "x64", HasGpu: false, GpuDescription: null),
            CancellationToken.None);

        ExecutionProviderAvailability dnnl = Assert.Single(availabilities, a => a.Provider == ExecutionProviderKind.Dnnl);
        Assert.Equal(dnnlReady, dnnl.IsAvailable);
        Assert.Contains("DNNL fake", dnnl.Detail, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task DiscoverAsync_reports_OpenVino_from_the_Linux_catalog_without_downloads(bool ready)
    {
        var catalog = new StubLinuxOpenVinoCatalog(ready);
        var discovery = new OnnxExecutionProviderDiscovery(
            new StubOpenVinoProvider(isAvailable: !ready),
            new StubLinuxNativeGpuRuntimeProbe(nvidiaDriverLoaded: false, nativeTensorRtAvailable: false),
            new StubNativeCudaTensorRtWindowsPolicy(allowed: false),
            new StubMigraphxReadinessProbe(),
            new StubDnnlReadinessProbe(false),
            new StubTensorRtRtxReadinessProbe(),
            new StubWinMlCatalogReadinessProbe(),
            new StubWinMlCatalogReadinessProbe(),
            new StubWinMlCatalogReadinessProbe(),
            linuxOpenVinoCatalog: catalog);

        IReadOnlyList<ExecutionProviderAvailability> availabilities = await discovery.DiscoverAsync(
            new HardwareProfile("linux", "x64", HasGpu: false, GpuDescription: null),
            CancellationToken.None);

        ExecutionProviderAvailability openVino = Assert.Single(availabilities, a => a.Provider == ExecutionProviderKind.OpenVino);
        Assert.Equal(ready, openVino.IsAvailable);
        Assert.Equal("catalog detail", openVino.Detail);
        Assert.Equal([false], catalog.EnsureCalls);
    }

    private sealed class StubLinuxOpenVinoCatalog(bool ready) : Trackdub.Inference.Onnx.ExecutionProviders.LinMl.ILinuxOpenVinoCatalog
    {
        public List<bool> EnsureCalls { get; } = [];

        public Task<Trackdub.Inference.Onnx.ExecutionProviders.LinMl.LinuxPluginCatalogResult> EnsureReadyAsync(
            bool allowDownloads,
            CancellationToken cancellationToken)
        {
            EnsureCalls.Add(allowDownloads);
            return Task.FromResult(new Trackdub.Inference.Onnx.ExecutionProviders.LinMl.LinuxPluginCatalogResult(
                ready
                    ? Trackdub.Inference.Onnx.ExecutionProviders.LinMl.LinuxPluginCatalogStatus.Ready
                    : Trackdub.Inference.Onnx.ExecutionProviders.LinMl.LinuxPluginCatalogStatus.NotInstalled,
                "catalog detail"));
        }

        public bool TryAppend(Microsoft.ML.OnnxRuntime.SessionOptions options, bool cpuOnly, out string? failureReason)
        {
            failureReason = null;
            return false;
        }
    }

    private sealed class StubNativeCudaTensorRtWindowsPolicy(bool allowed) : INativeCudaTensorRtWindowsPolicy
    {
        public Task<bool> IsNativeProvidersAllowedOnWindowsAsync(CancellationToken cancellationToken = default) =>
            Task.FromResult(allowed);
    }

    private sealed class StubMigraphxReadinessProbe : IMigraphxReadinessProbe
    {
        public Task<MigraphxReadinessReport> ProbeAsync(
            bool allowProviderDownloads,
            CancellationToken cancellationToken = default,
            IProgress<double>? progress = null) =>
            Task.FromResult(new MigraphxReadinessReport(
                ProviderId: string.Empty,
                MigraphxPlatformRoute.None,
                MigraphxReadinessBlocker.PlatformUnsupported,
                IsHardwareEligible: false,
                IsOrtProviderListed: false,
                IsRegisteredWithOrt: false,
                Detail: "MIGraphX not part of this test."));
    }

    private sealed class StubTensorRtRtxReadinessProbe : ITensorRtRtxReadinessProbe
    {
        public int CallCount { get; private set; }

        public Task<TensorRtRtxReadinessReport> ProbeAsync(
            bool allowProviderDownloads,
            CancellationToken cancellationToken = default)
        {
            CallCount++;
            return Task.FromResult(new TensorRtRtxReadinessReport(
                TensorRtRtxProviderIds.PluginEpAbi,
                TensorRtRtxPlatformRoute.PluginEpAbi,
                TensorRtRtxReadinessBlocker.EpNotPresent,
                IsHardwareEligible: true,
                IsOrtProviderListed: false,
                IsRegisteredWithOrt: false,
                Detail: "Plugin fake not registered."));
        }
    }

    private sealed class StubDnnlReadinessProbe(bool isReady) : IDnnlReadinessProbe
    {
        public Task<DnnlReadinessReport> ProbeAsync(
            bool allowProviderDownloads,
            CancellationToken cancellationToken = default) =>
            Task.FromResult(new DnnlReadinessReport(
                DnnlProviderIds.NativeOrt,
                isReady ? DnnlReadinessBlocker.None : DnnlReadinessBlocker.OrtProviderUnavailable,
                IsSupportedRid: true,
                IsOrtProviderListed: isReady,
                CanAppendSessionOptions: isReady,
                SmokeTestPassed: isReady,
                Detail: isReady ? "DNNL fake ready." : "DNNL fake unavailable."));
    }

    private sealed class StubWinMlCatalogReadinessProbe :
        IOpenVinoCatalogReadinessProbe,
        IQnnCatalogReadinessProbe,
        IVitisAiCatalogReadinessProbe
    {
        public int CallCount { get; private set; }

        public Task<WinMlCatalogReadinessReport> ProbeAsync(
            bool allowProviderDownloads,
            CancellationToken cancellationToken = default)
        {
            CallCount++;
            return Task.FromResult(new WinMlCatalogReadinessReport(
                ProviderId: string.Empty,
                WinMlCatalogPlatformRoute.WinMlCatalog,
                WinMlCatalogReadinessBlocker.EpNotPresent,
                IsHardwareEligible: true,
                IsOrtProviderListed: false,
                IsRegisteredWithOrt: false,
                Detail: "Catalog fake not installed."));
        }
    }

    private sealed class StubOpenVinoProvider(bool isAvailable) : IOpenVinoAvailabilityProvider
    {
        public bool IsAvailable { get; } = isAvailable;

        public bool UseOpenVinoCpuProxy => false;
    }

    private sealed class StubLinuxNativeGpuRuntimeProbe(
        bool nvidiaDriverLoaded,
        bool nativeTensorRtAvailable,
        bool amdGpuPresent = false,
        bool cudaOrtProviderAvailable = true)
        : ILinuxNativeGpuRuntimeProbe
    {
        public bool IsNvidiaDriverLoaded() => nvidiaDriverLoaded;

        public bool IsAmdGpuPresent() => amdGpuPresent;

        public bool IsNativeTensorRtAvailable() => nativeTensorRtAvailable;

        public bool IsCudaOrtProviderAvailable() => cudaOrtProviderAvailable;

        public bool IsMigraphxOrtProviderAvailable() => false;
    }
}
