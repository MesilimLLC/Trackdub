using Trackdub.Contracts.Benchmarking;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Sdk;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.Planning;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace Trackdub.Sdk.Tests;

/// <summary>
/// The SDK builds the headless composition directly in <see cref="TrackdubBuilder.Build"/>
/// (not through <see cref="Trackdub.Composition.Headless.HeadlessDubbingHost"/>), so it must
/// hand the container's process-GPU reader to the shared ONNX session pool itself — otherwise
/// the advertised default-on accelerator admission silently stays reservation-only on the SDK
/// path. Disposing the factory releases that process-wide binding.
/// </summary>
public sealed class TrackdubBuilderProcessGpuAdmissionTests
{
    [Fact]
    public void Build_binds_gpu_device_luids_and_dispose_preserves_a_newer_owner()
    {
        IReadOnlyDictionary<int, long>? previous = SharedPoolOptions.AdapterLuidMap;
        SharedPoolOptions.UseAdapterLuidMap(null);
        try
        {
            using var first = new TrackdubBuilder().ConfigureServices(services =>
                services.Replace(ServiceDescriptor.Singleton<IDeviceEnumerator>(new FixedDeviceEnumerator())))
                .Build();
#if WINDOWS
            IReadOnlyDictionary<int, long>? firstMap = SharedPoolOptions.AdapterLuidMap;
            Assert.NotNull(firstMap);
            Assert.Equal(100, firstMap[0]);
            Assert.Equal(200, firstMap[1]);
            Assert.Equal(2, firstMap.Count);

            using var second = new TrackdubBuilder().ConfigureServices(services =>
                services.Replace(ServiceDescriptor.Singleton<IDeviceEnumerator>(new FixedDeviceEnumerator())))
                .Build();
            IReadOnlyDictionary<int, long>? secondMap = SharedPoolOptions.AdapterLuidMap;
            Assert.NotSame(firstMap, secondMap);
            first.Dispose();
            Assert.Same(secondMap, SharedPoolOptions.AdapterLuidMap);
            second.Dispose();
            Assert.Null(SharedPoolOptions.AdapterLuidMap);
#else
            Assert.Null(SharedPoolOptions.AdapterLuidMap);
#endif
        }
        finally
        {
            SharedPoolOptions.UseAdapterLuidMap(previous);
        }
    }

    [Fact]
    public void Enumeration_failure_keeps_sdk_construction_available()
    {
        IReadOnlyDictionary<int, long>? previous = SharedPoolOptions.AdapterLuidMap;
        SharedPoolOptions.UseAdapterLuidMap(null);
        try
        {
            using var factory = new TrackdubBuilder().ConfigureServices(services =>
                services.Replace(ServiceDescriptor.Singleton<IDeviceEnumerator>(new FixedDeviceEnumerator(fail: true))))
                .Build();
            Assert.Null(SharedPoolOptions.AdapterLuidMap);
        }
        finally
        {
            SharedPoolOptions.UseAdapterLuidMap(previous);
        }
    }

    private sealed class FixedDeviceEnumerator(bool fail = false) : IDeviceEnumerator
    {
        public Task<IReadOnlyList<DeviceEntry>> GetDevicesAsync(CancellationToken cancellationToken = default)
        {
            if (fail)
            {
                throw new InvalidOperationException("Enumeration unavailable");
            }

            IReadOnlyList<DeviceEntry> devices =
            [
                new(DeviceKind.DiscreteGpu, 0, "GPU A", "Test", 4096, 0, [], 100),
                new(DeviceKind.IntegratedGpu, 1, "GPU B", "Test", 0, 4096, [], 200),
                new(DeviceKind.Cpu, 2, "CPU", "Test", 0, 0, [], 300),
                new(DeviceKind.DiscreteGpu, 3, "Unmapped GPU", "Test", 4096, 0, [])
            ];
            return Task.FromResult(devices);
        }

        public Task<IReadOnlyList<DeviceEntry>> ReEnumerateAsync(CancellationToken cancellationToken = default) =>
            GetDevicesAsync(cancellationToken);
    }

    [Fact]
    public void Build_binds_process_gpu_reader_to_shared_pool_admission_and_dispose_releases_it()
    {
        IProcessGpuMemoryReader? previous = SharedPoolOptions.ProcessGpuMemoryReader;
        using var factory = new TrackdubBuilder().Build();
        try
        {
            // Windows reads performance counters (WINDOWS build only) and Linux reads NVML;
            // elsewhere there is no platform reader and the pool keeps its reservation-only model.
#if WINDOWS
            const bool hasProcessGpuSource = true;
#else
            bool hasProcessGpuSource = OperatingSystem.IsLinux();
#endif
            if (hasProcessGpuSource)
            {
                Assert.NotNull(SharedPoolOptions.ProcessGpuMemoryReader);
                factory.Dispose();
                Assert.Null(SharedPoolOptions.ProcessGpuMemoryReader);
            }
            else
            {
                Assert.Null(SharedPoolOptions.ProcessGpuMemoryReader);
                factory.Dispose();
            }
        }
        finally
        {
            SharedPoolOptions.UseProcessGpuMemoryReader(previous);
        }
    }
}
