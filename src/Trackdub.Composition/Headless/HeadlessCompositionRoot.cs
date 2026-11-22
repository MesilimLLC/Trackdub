using Trackdub.Contracts;
using Trackdub.Domain;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Pipeline;
using Trackdub.Application.Benchmarking;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Composition.DeepFilterNet;
using Trackdub.Composition.NvidiaAfx;
using Trackdub.Composition.Runtime;
using Trackdub.Infrastructure.Diagnostics;
using Trackdub.Infrastructure.Settings;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.Runtime;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;
using Trackdub.Media.Enhancement;
using Trackdub.Media.Extraction;
using Trackdub.Media.Loudness;
using Trackdub.Media.Muxing;
using Trackdub.Media.Playback;
using Trackdub.Media.Probe;
using Trackdub.Media.Process;
using Trackdub.Media.Stretch;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace Trackdub.Composition.Headless;

/// <summary>
/// Headless variant of <see cref="CompositionRoot.AddTrackdub"/> that omits UI-coupled
/// services and substitutes no-op implementations suitable for CLI, benchmarks, and server-side scenarios.
/// </summary>
public static class HeadlessCompositionRoot
{
    /// <summary>
    /// Registers all standard Trackdub services and then replaces UI-coupled registrations
    /// with headless alternatives.
    /// </summary>
    public static IServiceCollection AddHeadlessTrackdub(
        this IServiceCollection services,
        HeadlessTrackdubOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(services);
        options ??= new HeadlessTrackdubOptions();

        // Step 1: Register all standard services (same as UI app).
        services.AddTrackdub();

        // Step 1b: Register the cross-process transient-fault bus as a singleton so the
        // DubbingPipelineEngine ctor + DiagnosticsBundleExporter ctor resolve to the same
        // instance. Eliminates reviewer C8's "each engine instance owns a private bus" gap.
        services.AddSingleton<PipelineTransientFaultBus>();

        // Step 2: Override storage paths if custom directories provided.
        if (options.ModelDirectory is not null || options.ModelCacheDirectory is not null || options.LogDirectory is not null)
        {
            TrackdubStoragePaths storagePaths = HeadlessPersistedEpLicenses.ResolveStoragePaths(options);

            // Reapply to the process env too: static consumers that read these vars directly
            // (e.g. OnnxExecutionSessionFactory, FfmpegAutoDownloader) don't go through DI.
            // Registered as a singleton so it's disposed (and the env restored) when this
            // host's ServiceProvider disposes, instead of leaking into other hosts created
            // later in the same process. Callers must resolve HeadlessStorageEnvironmentScope
            // eagerly right after building the provider — unrequested singletons are never
            // instantiated by the container, so the override wouldn't otherwise take effect.
            services.AddSingleton(_ => new HeadlessStorageEnvironmentScope(
                TrackdubStoragePathResolver.ApplyToCurrentProcessScoped(storagePaths)));

            services.Replace(ServiceDescriptor.Singleton(storagePaths));
            services.Replace(ServiceDescriptor.Singleton<IAppStoragePaths>(storagePaths));
        }

        // Step 3: Replace UI-coupled services with headless alternatives.
        services.Replace(ServiceDescriptor.Scoped<IAudioPreviewTransport, NullAudioPreviewTransport>());
        services.RemoveAll<PlaybackService>();
        services.RemoveAll<PlaybackCapabilityProbe>();
        services.RemoveAll<IPlaybackBackendFactory>();

        // Step 4: Replace settings service with in-memory headless variant.
        // Hardware pins stay process-local. Vendor EP license flags seed from disk
        // so CLI `providers trt-rtx install --accept-license` survives to the next dub.
        services.Replace(ServiceDescriptor.Singleton<IStudioSettingsService>(sp =>
        {
            IAppStoragePaths storagePaths = sp.GetRequiredService<IAppStoragePaths>();
            StudioSettings persisted = HeadlessPersistedEpLicenses.Load(storagePaths);
            return new InMemoryStudioSettingsService(options, persisted);
        }));

        // Step 5: Wire explicit FFmpeg/FFprobe paths into media services when configured.
        // AddTrackdub() registers these with null paths (PATH/cache discovery). Headless hosts
        // that pass FfmpegPath/FfprobePath must replace those registrations.
        if (options.FfmpegPath is not null || options.FfprobePath is not null)
        {
            string? ffmpegPath = options.FfmpegPath;
            string? ffprobePath = options.FfprobePath;

            services.Replace(ServiceDescriptor.Singleton<IMediaProbe>(
                _ => new FfmpegMediaProbe(ffmpegPath, ffprobePath)));
            services.Replace(ServiceDescriptor.Singleton<IExportToolAvailabilityService>(
                _ => new FfmpegExportToolAvailabilityService(ffmpegPath, ffprobePath)));
            services.Replace(ServiceDescriptor.Singleton<IAudioExtractionService>(
                _ => new FfmpegAudioExtractionService(ffmpegPath)));
            services.Replace(ServiceDescriptor.Singleton<IFfmpegVideoEncoderCapabilities>(
                _ => new FfmpegVideoEncoderCapabilityService(ffmpegPath)));
            services.Replace(ServiceDescriptor.Singleton<ILoudnessNormalizer>(
                _ => new FfmpegLoudnessNormalizer(ffmpegPath)));
            services.Replace(ServiceDescriptor.Singleton<IExportRenderer>(
                _ => new FfmpegMuxer(ffmpegPath)));
            services.Replace(ServiceDescriptor.Singleton<IVideoRecomposer>(
                _ => new FfmpegVideoRecomposer(ffmpegPath)));
            services.Replace(ServiceDescriptor.Singleton<ISpeechAudioProcessingService>(
                _ => new FfmpegSpeechAudioProcessingService(ffmpegPath)));
            services.Replace(ServiceDescriptor.Singleton<ISpeechAudioEnhancementService>(sp =>
                new NvidiaAfxSpeechAudioEnhancementService(
                    sp.GetRequiredService<INvidiaAfxRuntimeReadinessService>(),
                    new ResolvingSpeechAudioEnhancementService(
                        sp.GetService<BundledModelManifestRegistry>(),
                        sp.GetService<IModelCacheInventory>()),
                    logger: sp.GetService<IApplicationLogger>())));
            services.Replace(ServiceDescriptor.Scoped<IAudioTimeStretchService>(
                _ => new AudioTimeStretchService(ffmpegPath)));
        }

        // Step 6: Override logger if provided.
        if (options.Logger is not null)
        {
            services.Replace(ServiceDescriptor.Singleton(options.Logger));
        }

        // Resource telemetry defaults remain replaceable by headless hosts.
#if WINDOWS
        // DXGI can report free VRAM on Windows; IVramMonitor carries the existing adapter query.
        services.TryAddSingleton<IVramMonitor>(sp => new WindowsVramMonitor(
            sp.GetRequiredService<IDeviceEnumerator>()));
        services.TryAddSingleton<IAvailableVramReader>(sp =>
            new WindowsAvailableVramReader(sp.GetRequiredService<IVramMonitor>()));
        // The GPU Process Memory performance counters give the process-isolated attribution the
        // adapter-wide DXGI reading cannot, so both readings are kept and reported separately.
        // Registering the reader has no process-wide side effect on its own: the host that owns
        // this container hands it to the shared session pool explicitly when it is built (see
        // HeadlessDubbingHost.Create), so resolving telemetry services can never silently arm
        // accelerator admission, and a host-provided override binds exactly like the default.
        // The reader's first read initializes the performance-counter subsystem (~1 s once per
        // process), so a host that measures stage timings should resolve the reader while
        // preparing rather than inside a measured stage.
        services.TryAddSingleton<IProcessGpuMemoryReader>(static _ => new WindowsProcessGpuMemoryReader());
#else
        // No DXGI on this platform: telemetry records an explicit unavailable reading, and the
        // shared pool keeps accounting for its own reservations only.
        services.TryAddSingleton<IAvailableVramReader, UnavailableAvailableVramReader>();
#endif
        // IProcessGpuMemoryReader itself is registered by the core AddTrackdub() path, so the
        // desktop and headless hosts share one registration; the headless session factory binds
        // the final (possibly host-replaced) reader to the shared pool's admission before the
        // first session is created. Its first read initializes the performance-counter subsystem
        // (~1 s once per process), so a host that measures stage timings should construct the
        // reader while preparing rather than inside a measured stage.
        services.TryAddSingleton<IWorkingSetSampler, ProcessWorkingSetSampler>();
        services.TryAddSingleton<IResourceTelemetryCollector, ProcessResourceTelemetryCollector>();
        services.TryAddSingleton<IResourceTelemetryValidator, ResourceTelemetryValidator>();

        // Step 7: Apply user-provided service overrides.
        options.ServiceConfigurator?.Invoke(services);

        return services;
    }

    /// <summary>
    /// Hands the container's process-GPU reader to the shared ONNX session pool so accelerator
    /// admission accounts for this process's real dedicated GPU footprint (opt-out via
    /// <c>TRACKDUB_SESSION_PROCESS_GPU_ADMISSION</c>). Every headless composition owner must call
    /// this once the <see cref="IServiceProvider"/> is built — <see cref="HeadlessDubbingHost"/> and
    /// the SDK's <c>TrackdubBuilder.Build</c> — or the advertised default-on admission silently
    /// stays reservation-only for that path. Best-effort: a failing reader must never fail
    /// composition. On a platform with no process-GPU source (neither the Windows nor the Linux
    /// reader) the pool keeps its reservation-only model and this returns null.
    /// </summary>
    /// <returns>The bound reader (null when there is nothing to bind).</returns>
    public static IProcessGpuMemoryReader? BindSharedPoolProcessGpuAdmission(IServiceProvider services)
    {
        IProcessGpuMemoryReader? reader = null;
        try
        {
            reader = services.GetService<IProcessGpuMemoryReader>();
#if !WINDOWS
            // The unavailable reader is the registration for platforms without a source;
            // binding it would arm nothing.
            if (reader is UnavailableProcessGpuMemoryReader)
            {
                reader = null;
            }
#endif
            if (reader is not null)
            {
                SharedPoolOptions.UseProcessGpuMemoryReader(reader);
            }
        }
        catch
        {
            reader = null;
        }

        return reader;
    }

    /// <summary>
    /// Clears the shared pool's process-GPU reader binding only while it still refers to
    /// <paramref name="reader"/>, so disposing one composition owner never tears down a newer
    /// owner's registration. Safe to call with null.
    /// </summary>
    public static void ClearSharedPoolProcessGpuAdmission(IProcessGpuMemoryReader? reader)
    {
        if (reader is not null)
        {
            SharedPoolOptions.TryClearProcessGpuMemoryReader(reader);
        }
    }

    /// <summary>
    /// Hands the container's device-index to adapter-LUID map to the shared ONNX session pool so
    /// the process-GPU observation can be attributed per adapter instead of charging every
    /// device the process total. Every composition owner that binds the reader must bind this
    /// too, or multi-GPU hosts over-restrict admission on otherwise-free adapters. Best-effort
    /// and synchronous (device enumeration caches): telemetry must never fail composition.
    /// </summary>
    /// <returns>The bound map (null when there is nothing to bind).</returns>
    public static IReadOnlyDictionary<int, long>? BindSharedPoolAdapterLuidMap(IServiceProvider services)
    {
        IReadOnlyDictionary<int, long>? map = null;
#if WINDOWS
        try
        {
            map = QueryAdapterLuidMap(services);
            if (map is not null)
            {
                SharedPoolOptions.UseAdapterLuidMap(map);
            }
        }
        catch
        {
            map = null;
        }
#endif
        return map;
    }

    /// <summary>
    /// Clears the shared pool's adapter-LUID map only while it still refers to
    /// <paramref name="map"/>. Safe to call with null.
    /// </summary>
    public static void ClearSharedPoolAdapterLuidMap(IReadOnlyDictionary<int, long>? map)
    {
        if (map is not null)
        {
            SharedPoolOptions.TryClearAdapterLuidMap(map);
        }
    }

#if WINDOWS
    private static IReadOnlyDictionary<int, long>? QueryAdapterLuidMap(IServiceProvider services)
    {
        try
        {
            IDeviceEnumerator? enumerator = services.GetService<IDeviceEnumerator>();
            if (enumerator is null)
            {
                return null;
            }

            IReadOnlyList<DeviceEntry> devices = enumerator
                .GetDevicesAsync(CancellationToken.None)
                .GetAwaiter()
                .GetResult();
            var map = new Dictionary<int, long>();
            foreach (DeviceEntry device in devices)
            {
                if (device.Kind is DeviceKind.DiscreteGpu or DeviceKind.IntegratedGpu
                    && device.AdapterLuid is long luid)
                {
                    map[device.DeviceIndex] = luid;
                }
            }

            return map.Count == 0 ? null : map;
        }
        catch
        {
            return null;
        }
    }
#endif
}

/// <summary>
/// DI-owned handle for a headless host's process-env storage-path override. The container
/// only instantiates unrequested singletons on demand, so callers must resolve this type
/// eagerly right after building the <see cref="System.IServiceProvider"/> (before any
/// pipeline work starts) to actually apply the override; disposing the provider then
/// restores the prior environment values via <see cref="Dispose"/>.
/// </summary>
public sealed class HeadlessStorageEnvironmentScope(IDisposable inner) : IDisposable
{
    public void Dispose() => inner.Dispose();
}
