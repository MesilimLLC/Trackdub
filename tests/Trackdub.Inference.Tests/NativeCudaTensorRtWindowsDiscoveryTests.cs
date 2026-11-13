using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Runtime.Planning;
using Xunit;

namespace Trackdub.Inference.Tests;

public sealed class NativeCudaTensorRtWindowsDiscoveryTests
{
    [Fact]
    public async Task DiscoverAsync_WindowsWithNativePolicyDisabled_ReportsCudaUnavailable()
    {
        var discovery = new OnnxExecutionProviderDiscovery(
            new StubOpenVinoProvider(false),
            new StubLinuxNativeGpuRuntimeProbe(nvidiaDriverLoaded: false, nativeTensorRtAvailable: false),
            new StubNativeCudaTensorRtWindowsPolicy(allowed: false));

        IReadOnlyList<ExecutionProviderAvailability> availabilities = await discovery.DiscoverAsync(
            new HardwareProfile("windows", "x64", HasGpu: true, GpuDescription: "NVIDIA RTX 4090"),
            CancellationToken.None);

        ExecutionProviderAvailability cuda = availabilities.Single(a => a.Provider == ExecutionProviderKind.Cuda);
        Assert.False(cuda.IsAvailable);
        Assert.Contains("host policy", cuda.Detail, StringComparison.OrdinalIgnoreCase);
    }

    private sealed class StubOpenVinoProvider(bool isAvailable) : IOpenVinoAvailabilityProvider
    {
        public bool IsAvailable { get; } = isAvailable;

        public bool UseOpenVinoCpuProxy => false;
    }

    private sealed class StubLinuxNativeGpuRuntimeProbe(
        bool nvidiaDriverLoaded,
        bool nativeTensorRtAvailable)
        : ILinuxNativeGpuRuntimeProbe
    {
        public bool IsNvidiaDriverLoaded() => nvidiaDriverLoaded;

        public bool IsAmdGpuPresent() => false;

        public bool IsNativeTensorRtAvailable() => nativeTensorRtAvailable;

        public bool IsCudaOrtProviderAvailable() => true;

        public bool IsMigraphxOrtProviderAvailable() => false;
    }

    private sealed class StubNativeCudaTensorRtWindowsPolicy(bool allowed) : INativeCudaTensorRtWindowsPolicy
    {
        public Task<bool> IsNativeProvidersAllowedOnWindowsAsync(CancellationToken cancellationToken = default) =>
            Task.FromResult(allowed);
    }
}
