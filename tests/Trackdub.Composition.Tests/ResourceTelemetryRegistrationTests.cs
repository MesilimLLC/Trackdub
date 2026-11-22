using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Trackdub.Application.Benchmarking;
using Trackdub.Composition.Headless;
using Trackdub.Composition.Runtime;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Infrastructure.Diagnostics;
using Trackdub.Inference.Onnx.Pool;

namespace Trackdub.Composition.Tests;

/// <summary>
/// Hosts bind a process-wide reader to the shared pool, so these tests must not run in parallel
/// with other collections that create hosts.
/// </summary>
[CollectionDefinition(nameof(ResourceTelemetryRegistrationTests), DisableParallelization = true)]
public sealed class ResourceTelemetryRegistrationCollection;

[Collection(nameof(ResourceTelemetryRegistrationTests))]
public sealed class ResourceTelemetryRegistrationTests
{
    // Windows reads the GPU Process Memory counters and Linux reads NVML; other platforms have
    // no process-GPU source, so the pool keeps its reservation-only model there. The Windows
    // reader exists only in the WINDOWS build, so the predicate follows that build define.
#if WINDOWS
    private static bool HasProcessGpuSource => true;
#else
    private static bool HasProcessGpuSource => OperatingSystem.IsLinux();
#endif

    [Fact]
    public void Core_composition_registers_the_reader_for_the_platform()
    {
        var services = new ServiceCollection();
        services.AddTrackdub();
        using var provider = services.BuildServiceProvider();

        IProcessGpuMemoryReader reader = provider.GetRequiredService<IProcessGpuMemoryReader>();

        if (OperatingSystem.IsLinux())
        {
            Assert.IsType<LinuxProcessGpuMemoryReader>(reader);
        }
        else if (!OperatingSystem.IsWindows())
        {
            Assert.IsType<UnavailableProcessGpuMemoryReader>(reader);
        }
    }

    [Fact]
    public void Headless_does_not_bind_an_unavailable_reader_where_the_platform_has_no_source()
    {
        Assert.SkipWhen(OperatingSystem.IsWindows(), "Windows binds whatever reader the host registered.");
        IProcessGpuMemoryReader? previous = SharedPoolOptions.ProcessGpuMemoryReader;
        var services = new ServiceCollection();
        services.AddSingleton<IProcessGpuMemoryReader>(new UnavailableProcessGpuMemoryReader());
        services.AddHeadlessTrackdub();
        using var provider = services.BuildServiceProvider();

        try
        {
            Assert.Null(HeadlessCompositionRoot.BindSharedPoolProcessGpuAdmission(provider));
            Assert.Null(SharedPoolOptions.ProcessGpuMemoryReader);
        }
        finally
        {
            SharedPoolOptions.UseProcessGpuMemoryReader(previous);
        }
    }

    [Fact]
    public void Core_composition_registers_gpu_reader_and_preserves_host_override()
    {
        var defaults = new ServiceCollection();
        defaults.AddTrackdub();
        using var defaultProvider = defaults.BuildServiceProvider();
        Assert.NotNull(defaultProvider.GetRequiredService<IProcessGpuMemoryReader>());

        var custom = new UnavailableProcessGpuMemoryReader();
        var services = new ServiceCollection();
        services.AddSingleton<IProcessGpuMemoryReader>(custom);
        services.AddTrackdub();
        using var provider = services.BuildServiceProvider();
        Assert.Same(custom, provider.GetRequiredService<IProcessGpuMemoryReader>());
    }

    [Fact]
    public void Headless_registers_resolvable_singleton_resource_services()
    {
        var services = new ServiceCollection();
        services.AddHeadlessTrackdub();
        using var provider = services.BuildServiceProvider();
        var collector = provider.GetRequiredService<IResourceTelemetryCollector>();
        var validator = provider.GetRequiredService<IResourceTelemetryValidator>();
        var workingSetSampler = provider.GetRequiredService<IWorkingSetSampler>();
        var gpuMemoryReader = provider.GetRequiredService<IProcessGpuMemoryReader>();
        Assert.IsType<ProcessResourceTelemetryCollector>(collector);
        Assert.IsType<ResourceTelemetryValidator>(validator);
        Assert.IsType<ProcessWorkingSetSampler>(workingSetSampler);
        Assert.Same(collector, provider.GetRequiredService<IResourceTelemetryCollector>());
        Assert.Same(validator, provider.GetRequiredService<IResourceTelemetryValidator>());
        Assert.Same(workingSetSampler, provider.GetRequiredService<IWorkingSetSampler>());
        Assert.Same(gpuMemoryReader, provider.GetRequiredService<IProcessGpuMemoryReader>());
    }

    [Fact]
    public void Headless_process_gpu_memory_reader_degrades_instead_of_throwing()
    {
        var services = new ServiceCollection();
        services.AddHeadlessTrackdub();
        using var provider = services.BuildServiceProvider();

        IProcessGpuMemoryReader reader = provider.GetRequiredService<IProcessGpuMemoryReader>();

        // Exercised through the same port the collector uses, so this covers the real
        // Windows PDH probe where the host provides it and the explicit unavailable fallback
        // everywhere else. Either way the reading never throws and never invents a zero.
        long? reading = reader.ReadDedicatedGpuMemoryBytes();
        if (reading is null)
        {
            Assert.False(string.IsNullOrWhiteSpace(reader.UnavailableReason));
        }
        else
        {
            Assert.True(reading.Value >= 0);
        }
    }

    [Fact]
    public void Headless_binds_the_process_gpu_reader_to_shared_pool_admission()
    {
        // Creating the host is what hands its reader to the shared session pool, so accelerator
        // admission accounts for this process's real dedicated GPU footprint instead of only the
        // pool's own reservations. Resolving the reader alone must not bind anything.
        var services = new ServiceCollection();
        services.AddHeadlessTrackdub();
        using (var provider = services.BuildServiceProvider())
        {
            _ = provider.GetRequiredService<IProcessGpuMemoryReader>();
            Assert.Null(SharedPoolOptions.ProcessGpuMemoryReader);
        }

        // Capture the ambient binding before this test registers its own host, so the
        // finally restores exactly what was there instead of unconditionally clearing a
        // reader a previous test or host bound.
        IProcessGpuMemoryReader? previous = SharedPoolOptions.ProcessGpuMemoryReader;
        using var host = HeadlessDubbingHost.Create();
        try
        {
            IProcessGpuMemoryReader reader = host.Services.GetRequiredService<IProcessGpuMemoryReader>();
            if (HasProcessGpuSource)
            {
                Assert.Same(reader, SharedPoolOptions.ProcessGpuMemoryReader);
            }
            else
            {
                // No platform reader here: the pool keeps its reservation-only model.
                Assert.Null(SharedPoolOptions.ProcessGpuMemoryReader);
            }
        }
        finally
        {
            SharedPoolOptions.UseProcessGpuMemoryReader(previous);
        }
    }

    [Fact]
    public void Headless_host_dispose_releases_only_its_own_pool_binding()
    {
        IProcessGpuMemoryReader? previous = SharedPoolOptions.ProcessGpuMemoryReader;
        using var first = HeadlessDubbingHost.Create();
        using var second = HeadlessDubbingHost.Create();
        try
        {
            if (HasProcessGpuSource)
            {
                // The latest host wins the process-wide registration.
                Assert.Same(
                    second.Services.GetRequiredService<IProcessGpuMemoryReader>(),
                    SharedPoolOptions.ProcessGpuMemoryReader);

                // Disposing the older host must not tear down the newer host's registration.
                first.Dispose();
                Assert.Same(
                    second.Services.GetRequiredService<IProcessGpuMemoryReader>(),
                    SharedPoolOptions.ProcessGpuMemoryReader);

                // Disposing the owning host clears the binding instead of leaking a stale reader
                // into later hosts and tests in the same process.
                second.Dispose();
                Assert.Null(SharedPoolOptions.ProcessGpuMemoryReader);
            }
            else
            {
                Assert.Null(SharedPoolOptions.ProcessGpuMemoryReader);
                first.Dispose();
                second.Dispose();
            }
        }
        finally
        {
            SharedPoolOptions.UseProcessGpuMemoryReader(previous);
        }
    }

    [Fact]
    public void Headless_composition_root_binds_and_clears_shared_pool_admission()
    {
        // The shared helper every headless composition owner calls — HeadlessDubbingHost and
        // the SDK's TrackdubBuilder.Build — must bind the container's reader and release the
        // binding on dispose without touching a previous registration.
        IProcessGpuMemoryReader? previous = SharedPoolOptions.ProcessGpuMemoryReader;
        var services = new ServiceCollection();
        services.AddHeadlessTrackdub();
        using var provider = services.BuildServiceProvider();

        try
        {
            IProcessGpuMemoryReader? bound = HeadlessCompositionRoot.BindSharedPoolProcessGpuAdmission(provider);
            if (HasProcessGpuSource)
            {
                Assert.Same(provider.GetRequiredService<IProcessGpuMemoryReader>(), bound);
                Assert.Same(bound, SharedPoolOptions.ProcessGpuMemoryReader);
                HeadlessCompositionRoot.ClearSharedPoolProcessGpuAdmission(bound);
                Assert.Null(SharedPoolOptions.ProcessGpuMemoryReader);
            }
            else
            {
                // No platform reader here: the pool keeps its reservation-only model.
                Assert.Null(bound);
                Assert.Null(SharedPoolOptions.ProcessGpuMemoryReader);
            }
        }
        finally
        {
            SharedPoolOptions.UseProcessGpuMemoryReader(previous);
        }
    }

    [Fact]
    public void Headless_preserves_pre_registered_resource_services()
    {
        var collector = new ProcessResourceTelemetryCollector();
        var validator = new ResourceTelemetryValidator();
        var services = new ServiceCollection();
        services.AddSingleton<IResourceTelemetryCollector>(collector);
        services.AddSingleton<IResourceTelemetryValidator>(validator);
        var workingSetSampler = new ProcessWorkingSetSampler();
        services.AddSingleton<IWorkingSetSampler>(workingSetSampler);
        var gpuMemoryReader = new UnavailableProcessGpuMemoryReader();
        services.AddSingleton<IProcessGpuMemoryReader>(gpuMemoryReader);
        services.AddHeadlessTrackdub();
        using var provider = services.BuildServiceProvider();
        Assert.Same(collector, provider.GetRequiredService<IResourceTelemetryCollector>());
        Assert.Same(validator, provider.GetRequiredService<IResourceTelemetryValidator>());
        Assert.Same(workingSetSampler, provider.GetRequiredService<IWorkingSetSampler>());
        Assert.Same(gpuMemoryReader, provider.GetRequiredService<IProcessGpuMemoryReader>());
    }

    [Fact]
    public void Headless_configurator_can_replace_resource_services_after_defaults()
    {
        var collector = new ProcessResourceTelemetryCollector();
        var validator = new ResourceTelemetryValidator();
        var services = new ServiceCollection();
        services.AddHeadlessTrackdub(new HeadlessTrackdubOptions
        {
            ServiceConfigurator = configured =>
            {
                Assert.Contains(configured, item => item.ServiceType == typeof(IResourceTelemetryCollector));
                Assert.Contains(configured, item => item.ServiceType == typeof(IResourceTelemetryValidator));
                Assert.Contains(configured, item => item.ServiceType == typeof(IWorkingSetSampler));
                configured.Replace(ServiceDescriptor.Singleton<IResourceTelemetryCollector>(collector));
                configured.Replace(ServiceDescriptor.Singleton<IResourceTelemetryValidator>(validator));
            }
        });
        using var provider = services.BuildServiceProvider();
        Assert.Same(collector, provider.GetRequiredService<IResourceTelemetryCollector>());
        Assert.Same(validator, provider.GetRequiredService<IResourceTelemetryValidator>());
    }
}
