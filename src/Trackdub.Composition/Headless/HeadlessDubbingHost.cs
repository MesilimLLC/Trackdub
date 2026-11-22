using Trackdub.Application.Dubbing;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Contracts;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Infrastructure.Diagnostics;
using Microsoft.Extensions.DependencyInjection;

namespace Trackdub.Composition.Headless;

/// <summary>
/// Owns the headless DI container and session factory for pipeline execution.
/// Used by benchmarks and other non-SDK headless hosts.
/// </summary>
public sealed class HeadlessDubbingHost : IDisposable
{
    private readonly HeadlessDubbingSessionFactory _sessionFactory;
    private readonly IServiceProvider _serviceProvider;
    private readonly IProcessGpuMemoryReader? _processGpuReader;
    private readonly IReadOnlyDictionary<int, long>? _adapterLuidMap;

    private HeadlessDubbingHost(
        HeadlessDubbingSessionFactory sessionFactory,
        IServiceProvider serviceProvider,
        IProcessGpuMemoryReader? processGpuReader,
        IReadOnlyDictionary<int, long>? adapterLuidMap)
    {
        _sessionFactory = sessionFactory;
        _serviceProvider = serviceProvider;
        _processGpuReader = processGpuReader;
        _adapterLuidMap = adapterLuidMap;
    }

    /// <summary>
    /// Session factory for creating per-project dubbing sessions.
    /// </summary>
    public IDubbingSessionFactory SessionFactory => _sessionFactory;

    /// <summary>
    /// The root service provider backing this headless host.
    /// </summary>
    public IServiceProvider Services => _serviceProvider;

    /// <summary>
    /// Builds a headless host with the given options.
    /// </summary>
    public static HeadlessDubbingHost Create(HeadlessTrackdubOptions? options = null)
    {
        options ??= new HeadlessTrackdubOptions();

        if (options.ModelDirectory is not null && !Directory.Exists(options.ModelDirectory))
        {
            throw new DirectoryNotFoundException($"Model directory not found: {options.ModelDirectory}");
        }

        if (options.FfmpegPath is not null && !File.Exists(options.FfmpegPath))
        {
            throw new FileNotFoundException($"FFmpeg executable not found: {options.FfmpegPath}", options.FfmpegPath);
        }

        if (options.FfprobePath is not null && !File.Exists(options.FfprobePath))
        {
            throw new FileNotFoundException($"FFprobe executable not found: {options.FfprobePath}", options.FfprobePath);
        }

        var services = new ServiceCollection();
        services.AddHeadlessTrackdub(options);
        ServiceProvider serviceProvider = services.BuildServiceProvider();

        // This host owns a plain ServiceProvider, so start the warm queue explicitly.
        serviceProvider.GetRequiredService<Trackdub.Composition.Pipeline.EngineAutoWarmInstaller>()
            .StartAsync(CancellationToken.None).GetAwaiter().GetResult();

        // Arm the shared session pool's process-GPU admission explicitly at host construction:
        // a lazy DI factory would only bind when something happens to resolve the telemetry
        // reader, leaving normal host paths on reservation-only accounting despite admission
        // being default-on, and a host-provided reader override would bypass the factory
        // entirely. Resolving here also pays the performance-counter warm-up during setup
        // rather than inside a measured stage. Best-effort: telemetry must never fail host
        // creation.
        IProcessGpuMemoryReader? processGpuReader = null;
        IReadOnlyDictionary<int, long>? adapterLuidMap = null;
#if WINDOWS
        try
        {
            processGpuReader = serviceProvider.GetService<IProcessGpuMemoryReader>();
            if (processGpuReader is not null)
            {
                SharedPoolOptions.UseProcessGpuMemoryReader(processGpuReader);
            }

            adapterLuidMap = HeadlessCompositionRoot.BindSharedPoolAdapterLuidMap(serviceProvider);
        }
        catch
        {
            processGpuReader = null;
            adapterLuidMap = null;
        }
#else
        // No adapter LUIDs off Windows: the reader reports a process total only, which the pool
        // charges to every device. Null when the platform has no process-GPU source.
        processGpuReader = HeadlessCompositionRoot.BindSharedPoolProcessGpuAdmission(serviceProvider);
#endif

        return new HeadlessDubbingHost(new HeadlessDubbingSessionFactory(serviceProvider), serviceProvider, processGpuReader, adapterLuidMap);
    }

    /// <summary>
    /// Creates a new <see cref="DubbingPipelineEngine"/> bound to this host's session factory.
    /// Resolves the Composition-singleton <see cref="PipelineTransientFaultBus"/> so the
    /// engine publishes to the same stream the diagnostics exporter reads from.
    /// </summary>
    public DubbingPipelineEngine CreateEngine() =>
        new(_sessionFactory, _serviceProvider.GetRequiredService<PipelineTransientFaultBus>());

    /// <summary>
    /// Composition-singleton transient-fault bus. Exposed for Composition-level tests so
    /// they can pin the shared-bus identity contract across the engine and any consumer
    /// that takes the bus via DI.
    /// </summary>
    public PipelineTransientFaultBus TransientFaultBus =>
        _serviceProvider.GetRequiredService<PipelineTransientFaultBus>();

    /// <summary>
    /// Composition-singleton diagnostics bundle exporter. Exposed so tests can pin
    /// shared-bus identity on the exporter side of the C8 wire (spec §4.4 follow-up)
    /// without resorting to reflection. The exporter's own
    /// <c>TransientFaultBus</c> property surfaces the same singleton the engine publishes to.
    /// Cast to concrete is safe: the headless composition only registers the concrete
    /// <see cref="DiagnosticsBundleExporter"/> under the
    /// <c>IDiagnosticsBundleExporter</c> contract via
    /// <c>CompositionRoot.TryAddSingleton&lt;IDiagnosticsBundleExporter, DiagnosticsBundleExporter&gt;</c>.
    /// </summary>
    public DiagnosticsBundleExporter Exporter =>
        (DiagnosticsBundleExporter)_serviceProvider.GetRequiredService<IDiagnosticsBundleExporter>();

    /// <inheritdoc />
    public void Dispose()
    {
        // The pool bindings are process-wide but were registered by this host: clear them on
        // dispose only while they still refer to this host's registrations, so a later host's
        // registrations are never torn down and later tests never observe stale state. The
        // clears are atomic: a host disposing while another host registers cannot null the
        // newer registrations.
        if (_processGpuReader is not null)
        {
            SharedPoolOptions.TryClearProcessGpuMemoryReader(_processGpuReader);
        }

        if (_adapterLuidMap is not null)
        {
            SharedPoolOptions.TryClearAdapterLuidMap(_adapterLuidMap);
        }

        _sessionFactory.Dispose();
    }
}
