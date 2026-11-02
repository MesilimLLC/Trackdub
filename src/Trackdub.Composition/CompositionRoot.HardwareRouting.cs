using Trackdub.Application.Cloud;
using Trackdub.Contracts;
using Trackdub.Contracts.Editing;
using Trackdub.Contracts.Cloud;
using Trackdub.Contracts.Serving;
using Trackdub.Contracts.LipSync;
using Trackdub.Contracts.Licensing;
using Trackdub.Application.Mixing;
using Trackdub.Application.Editing;
using Trackdub.Application.Updates;
using Trackdub.Contracts.Projects;
using Trackdub.Application.Pipeline;
using Trackdub.Application.Runtime;
using Trackdub.Application.Settings;
using Trackdub.Composition.ForcedAlignment;
using Trackdub.Composition.LipSynthesis;
using Trackdub.Composition.NvidiaAfx;
using Trackdub.Composition.Pipeline;
using Trackdub.Inference.Onnx.ForcedAlignment;
using Trackdub.Contracts.Transcripts;
using Trackdub.Application.Transcripts;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Application.Transcripts.Stages;
using Trackdub.Composition.HardwareProfiler;
using Trackdub.Composition.Runtime.Planning;
using Trackdub.Composition.Runtime;
using Trackdub.Composition.StarterPacks;
using Trackdub.Composition.Transcription;
using Trackdub.Composition.Translation;
using Trackdub.Contracts.Translation;
using Trackdub.Composition.Tts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Inference.Onnx.Migraphx;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.StarterPacks;
using Trackdub.Infrastructure.Components;
using Trackdub.Infrastructure.Components.NvidiaAfx;
using Trackdub.Infrastructure.Diagnostics;
using Trackdub.Infrastructure.FileSystem;
using Trackdub.Infrastructure.Licensing;
using Trackdub.Infrastructure.Logging;
using Trackdub.Infrastructure.Persistence.Repositories;
using Trackdub.Infrastructure.Persistence.Sqlite;
using Trackdub.Infrastructure.ModelOptimization;
using Trackdub.Infrastructure.Runtime.TrtRtxEp;
using Trackdub.Infrastructure.Settings;
using Trackdub.Infrastructure.StarterPacks;
using Trackdub.Infrastructure.Updates;
using Trackdub.Contracts.ModelOptimization;
using Trackdub.Infrastructure.Transcripts;
using Trackdub.Infrastructure.Translation;
using Trackdub.Infrastructure.Transcription;
using Trackdub.Infrastructure.Tts;
using Trackdub.Infrastructure.Dubbing;
using Trackdub.Inference.Onnx;
using Trackdub.Inference.Onnx.Worker;
using Trackdub.Inference.Onnx.Dnnl;
using Trackdub.Inference.Onnx.Chatterbox;
using Trackdub.Inference.Onnx.CosyVoice;
using Trackdub.Inference.Onnx.SepFormer;
using Trackdub.Inference.Onnx.Spleeter;
using Trackdub.Inference.Onnx.Kokoro;
using Trackdub.Inference.Onnx.Qwen3Tts;
using Trackdub.Inference.Onnx.Madlad;
using Trackdub.Inference.Onnx.OpusMt;
using Trackdub.Inference.Onnx.Runtime;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Onnx.Runtime.Routing;
using Trackdub.Inference.Onnx.WinMlCatalog;
using Trackdub.Inference.Onnx.SileroVad;
using Trackdub.Inference.Onnx.SortFormer;
using Trackdub.Inference.Onnx.TensorRtRtx;
using Trackdub.Inference.Onnx.Translation;
using Trackdub.Inference.Onnx.ExecutionProviders;
using Trackdub.Inference.Onnx.Whisper;
#if WINDOWS
using Trackdub.Inference.Onnx.WindowsMl;
#endif
using Trackdub.Application.Hardware;
using Trackdub.Composition.Hardware;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.ParakeetTdt;
using Trackdub.Inference.Onnx.Qwen3Asr;
using Trackdub.Inference.Onnx.NemotronAsr;
using Trackdub.Inference.Onnx.QwenTextRefinement;
using Trackdub.Inference.Onnx.QwenAssistant;
using Trackdub.Inference.Onnx.DeepFilterNet;
using Trackdub.Inference.Onnx.Phi;
using Trackdub.Inference.Runtime.TensorRtRtx;
using Trackdub.Inference.Runtime.Planning;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Domain;
using Trackdub.Licensing;
using Trackdub.Media.Enhancement;
using Trackdub.Media.Extraction;
using Trackdub.Media.Loudness;
using Trackdub.Media.Mixing;
using Trackdub.Media.Muxing;
using Trackdub.Media.Playback;
using Trackdub.Media.Probe;
using Trackdub.Media.Process;
using Trackdub.Media.Quality;
using Trackdub.Media.Stretch;
using Trackdub.Media.Tts;
using Trackdub.Media.Waveforms;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;
using System.Net;
using System.Runtime.Versioning;

namespace Trackdub.Composition;

public static partial class CompositionRoot
{
    [SupportedOSPlatform("linux")]
    private static IExecutionProviderBootstrapper CreateLinuxExecutionProviderBootstrapper(IServiceProvider sp) =>
        new Trackdub.Inference.Onnx.ExecutionProviders.Linux.LinuxExecutionProviderBootstrapper(
            sp.GetRequiredService<IOpenVinoAvailabilityProvider>(),
            new LinuxNativeGpuRuntimeProbe(),
            sp.GetRequiredService<ITensorRtRtxProviderBootstrap>());

    private static void AddHardwareRouting(IServiceCollection services)
    {
        AddComponentStoreServices(services);
        AddDeviceEnumerationServices(services);
        AddExecutionProviderBootstrapper(services);
        AddHardwareMatrixAndBenchmarking(services);
        AddDeviceExclusionAndDownloaders(services);
    }

    private static void AddComponentStoreServices(IServiceCollection services)
    {
        // ComponentStore — manages optional downloadable runtime components.
        services.TryAddSingleton<ComponentStore>(sp =>
            new ComponentStore(
                sp.GetRequiredService<TrackdubStoragePaths>().ComponentCacheDirectory,
                sp.GetRequiredService<IApplicationLogger>()));

        // OpenVinoBootstrapper — singleton that loads OpenVINO native libraries at startup.
        // Registered as both its concrete type and the IOpenVinoAvailabilityProvider interface
        // so that WindowsDeviceEnumerator and EP discovery share the same instance.
        services.TryAddSingleton<OpenVinoBootstrapper>(sp =>
        {
            ComponentStore componentStore = sp.GetRequiredService<ComponentStore>();
            DeviceAffinitySettings deviceAffinitySettings = sp.GetRequiredService<DeviceAffinitySettings>();
            return new OpenVinoBootstrapper(
                isComponentInstalled: componentStore.IsInstalled,
                getComponentInstallPath: componentStore.GetInstallPath,
                useOpenVinoCpuProxy: deviceAffinitySettings.UseOpenVinoCpuProxy,
                logger: sp.GetRequiredService<ILogger<OpenVinoBootstrapper>>());
        });
        services.TryAddSingleton<IOpenVinoAvailabilityProvider>(sp =>
            sp.GetRequiredService<OpenVinoBootstrapper>());
    }

    private static void AddDeviceEnumerationServices(IServiceCollection services)
    {
        // IDeviceEnumerator — platform-specific GPU/NPU discovery, singleton. The platform
        // dispatch lives in DeviceEnumeratorFactory so the pre-container callers share it.
        // Resolved only on Linux (below); registering it elsewhere is harmless.
#pragma warning disable CA1416 // PhysicalSysfsReader is Linux-only and is only resolved on Linux
        services.TryAddSingleton<ISysfsReader, PhysicalSysfsReader>();
#pragma warning restore CA1416
        services.TryAddSingleton<IDeviceEnumerator>(sp =>
            OperatingSystem.IsLinux()
                ? DeviceEnumeratorFactory.CreateLinux(
                    sp.GetRequiredService<IOpenVinoAvailabilityProvider>(),
                    sp.GetRequiredService<ILoggerFactory>(),
                    sp.GetRequiredService<ISysfsReader>())
                : DeviceEnumeratorFactory.Create(
                    sp.GetRequiredService<IOpenVinoAvailabilityProvider>(),
                    sp.GetRequiredService<ILoggerFactory>()));

        // IProcessGpuMemoryReader — this process's dedicated GPU memory, used both as benchmark
        // evidence and as the shared session pool's accelerator-admission observation. Registered
        // in the core path (not only headless) so every host that calls AddTrackdub — desktop,
        // headless, benchmark — offers a reader, and so a host that pre-registers its own wins.
        // Binding the final registration to the pool happens at host construction, after all
        // service overrides, rather than inside this factory: admission must not depend on
        // something later resolving the telemetry services (see HeadlessDubbingSessionFactory).
#if WINDOWS
        services.TryAddSingleton<Trackdub.Contracts.Benchmarking.IProcessGpuMemoryReader, WindowsProcessGpuMemoryReader>();
#else
        services.TryAddSingleton<Trackdub.Contracts.Benchmarking.IProcessGpuMemoryReader, UnavailableProcessGpuMemoryReader>();
#endif
    }

    private static void AddExecutionProviderBootstrapper(IServiceCollection services)
    {
        // IExecutionProviderBootstrapper — platform-specific EP bootstrap logic, singleton.
        // Also calls OnnxExecutionSessionFactory.Initialize() so the static factory uses the
        // DI-wired bootstrapper (with real IOpenVinoAvailabilityProvider on Linux).
        services.TryAddSingleton<IExecutionProviderBootstrapper>(CreateExecutionProviderBootstrapper);
    }

    private static IExecutionProviderBootstrapper CreateExecutionProviderBootstrapper(IServiceProvider sp)
    {
#if WINDOWS
        IExecutionProviderBootstrapper bootstrapper =
            new Trackdub.Inference.Onnx.ExecutionProviders.Windows.WindowsExecutionProviderBootstrapper(
                WindowsMlProviderRegistrationPolicy.Shared,
                sp.GetRequiredService<INativeCudaTensorRtWindowsPolicy>(),
                sp.GetRequiredService<ITensorRtRtxProviderBootstrap>());
#else
        // The portable TFM picks the platform bootstrapper at run time, so the same artifact
        // behaves the same whichever OS built it.
        IExecutionProviderBootstrapper bootstrapper =
            OperatingSystem.IsMacOSVersionAtLeast(10, 15)
                ? new Trackdub.Inference.Onnx.ExecutionProviders.Mac.MacExecutionProviderBootstrapper()
                : OperatingSystem.IsLinux()
                    ? CreateLinuxExecutionProviderBootstrapper(sp)
                    : new PortableExecutionProviderBootstrapper(
                        sp.GetRequiredService<ITensorRtRtxProviderBootstrap>());
#endif
        OnnxExecutionProviderBootstrapperRegistry.Initialize(
            bootstrapper,
            sp.GetRequiredService<IWindowsMlEpDevicePolicyProvider>(),
            openVinoAvailabilityProvider: sp.GetRequiredService<IOpenVinoAvailabilityProvider>());
        return bootstrapper;
    }

    private static void AddHardwareMatrixAndBenchmarking(IServiceCollection services)
    {
        // HardwareMatrix — stateless singleton scoring engine.
        services.TryAddSingleton<IHardwareMatrix, HardwareMatrix>();

        // Hardware profiler — persists benchmark results; required by RuntimeSelectionService.
        services.TryAddSingleton(sp =>
            new SqliteUserBenchmarkDatabase(sp.GetRequiredService<IAppStoragePaths>().UserDataRoot));
        services.TryAddSingleton<Trackdub.Contracts.Persistence.IUserBenchmarkRepository, UserBenchmarkRepository>();
        services.TryAddSingleton<Trackdub.Contracts.Persistence.IBenchmarkEvidenceRepository, BenchmarkEvidenceRepository>();
        services.TryAddSingleton<HardwareProfilerHistoryRecorder>();
        services.TryAddSingleton<JsonHardwareProfilerStore>();
        services.TryAddSingleton<IHardwareProfilerService, HardwareProfilerService>();
    }

    private static void AddDeviceExclusionAndDownloaders(IServiceCollection services)
    {
        // PipelineDeviceExclusionProvider — singleton managing per-run device exclusions.
        // Registered as its concrete type first so both interface registrations share the same instance.
        services.TryAddSingleton<PipelineDeviceExclusionProvider>(sp =>
            new PipelineDeviceExclusionProvider(
                sp.GetService<ILogger<PipelineDeviceExclusionProvider>>()));
        services.TryAddSingleton<IPipelineDeviceExclusionProvider>(sp =>
            sp.GetRequiredService<PipelineDeviceExclusionProvider>());
        services.TryAddSingleton<Trackdub.Contracts.Pipeline.IPipelineRunLifecycle>(sp =>
            sp.GetRequiredService<PipelineDeviceExclusionProvider>());

        services.AddHttpClient("OpenVinoComponentDownloader", client =>
        {
            client.Timeout = TimeSpan.FromMinutes(10);
            client.DefaultRequestHeaders.UserAgent.ParseAdd("Trackdub-OpenVinoDownloader/1.0");
        }).ConfigurePrimaryHttpMessageHandler(() => new HttpClientHandler
        {
            AutomaticDecompression = DecompressionMethods.GZip | DecompressionMethods.Deflate | DecompressionMethods.Brotli
        });

        services.AddHttpClient("NvidiaAfxRuntimeDownloader", client =>
        {
            client.Timeout = TimeSpan.FromMinutes(30);
            client.DefaultRequestHeaders.UserAgent.ParseAdd("Trackdub-NvidiaAfxDownloader/1.0");
        }).ConfigurePrimaryHttpMessageHandler(() => new HttpClientHandler
        {
            AutomaticDecompression = DecompressionMethods.GZip | DecompressionMethods.Deflate | DecompressionMethods.Brotli
        });

        // OpenVinoComponentDownloader — transient; each download operation gets a fresh instance.
        services.TryAddTransient<OpenVinoComponentDownloader>(sp =>
        {
            return new OpenVinoComponentDownloader(
                sp.GetRequiredService<ComponentStore>(),
                sp.GetRequiredService<IHttpClientFactory>().CreateClient("OpenVinoComponentDownloader"),
                sp.GetRequiredService<IApplicationLogger>(),
                CreateOpenVinoComponentSettings(sp.GetRequiredService<IApplicationLogger>()),
                allowInsecureComponentDownload: false);
        });

        // DeviceAffinitySettings — loaded from disk once, registered as singleton.
        services.TryAddSingleton<DeviceAffinitySettings>(_ => DeviceAffinitySettings.Load());

        services.TryAddSingleton<IWindowsMlEpDevicePolicyProvider, StudioSettingsWindowsMlEpDevicePolicyProvider>();
        services.TryAddSingleton<IInferenceSessionPoolEvictor, InferenceSessionPoolEvictor>();
        services.TryAddSingleton<IHardwarePolicyCoordinator, HardwarePolicyCoordinator>();
    }

    private static long? TryGetEnvironmentLong(string variableName)
    {
        string? raw = Environment.GetEnvironmentVariable(variableName);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }

        return long.TryParse(
            raw,
            System.Globalization.NumberStyles.Integer,
            System.Globalization.CultureInfo.InvariantCulture,
            out long value)
            ? value
            : null;
    }

    private static OpenVinoComponentSettings CreateOpenVinoComponentSettings(IApplicationLogger logger) =>
        IsOpenVinoOverrideEnabled()
            ? CreateOverriddenOpenVinoComponentSettings(logger)
            : CreateDefaultOpenVinoComponentSettings();

    private static OpenVinoComponentSettings CreateDefaultOpenVinoComponentSettings() =>
        new()
        {
            DownloadUrl = OpenVinoComponentDefaults.DownloadUrl,
            ExpectedSha256Hash = OpenVinoComponentDefaults.ExpectedSha256Hash,
            ExpectedFileSizeBytes = OpenVinoComponentDefaults.ExpectedFileSizeBytes
        };

    private static OpenVinoComponentSettings CreateOverriddenOpenVinoComponentSettings(IApplicationLogger logger)
    {
        OpenVinoOverrides overrides = ReadOpenVinoOverrides(logger);
        return new OpenVinoComponentSettings
        {
            DownloadUrl = overrides.DownloadUrl ?? OpenVinoComponentDefaults.DownloadUrl,
            ExpectedSha256Hash = overrides.Sha256 ?? OpenVinoComponentDefaults.ExpectedSha256Hash,
            ExpectedFileSizeBytes = overrides.FileSizeBytes ?? OpenVinoComponentDefaults.ExpectedFileSizeBytes
        };
    }

    private sealed record OpenVinoOverrides(string? DownloadUrl, string? Sha256, long? FileSizeBytes);

    private static OpenVinoOverrides ReadOpenVinoOverrides(IApplicationLogger logger) =>
        new(
            ReadOpenVinoUrlOverride(logger),
            ReadOpenVinoSha256Override(logger),
            ReadOpenVinoFileSizeOverride(logger));

    private static string? ReadOpenVinoUrlOverride(IApplicationLogger logger)
    {
        string? raw = Environment.GetEnvironmentVariable("TRACKDUB_OPENVINO_DOWNLOAD_URL");
        string? validated = TryGetValidatedOpenVinoDownloadUrl(raw);
        if (!string.IsNullOrWhiteSpace(raw) && validated is null)
        {
            logger.LogWarning("Ignoring TRACKDUB_OPENVINO_DOWNLOAD_URL because it is not a valid absolute HTTPS URL.");
        }

        return validated;
    }

    private static string? ReadOpenVinoSha256Override(IApplicationLogger logger)
    {
        string? raw = Environment.GetEnvironmentVariable("TRACKDUB_OPENVINO_SHA256");
        string? validated = TryGetValidatedOpenVinoSha256(raw);
        if (!string.IsNullOrWhiteSpace(raw) && validated is null)
        {
            logger.LogWarning("Ignoring TRACKDUB_OPENVINO_SHA256 because it is not a valid SHA-256 hex value.");
        }

        return validated;
    }

    private static long? ReadOpenVinoFileSizeOverride(IApplicationLogger logger)
    {
        string? raw = Environment.GetEnvironmentVariable("TRACKDUB_OPENVINO_FILE_SIZE_BYTES");
        long? parsed = TryGetEnvironmentLong("TRACKDUB_OPENVINO_FILE_SIZE_BYTES");
        long? validated = parsed is > 0 ? parsed : null;
        if (!string.IsNullOrWhiteSpace(raw) && validated is null)
        {
            logger.LogWarning("Ignoring TRACKDUB_OPENVINO_FILE_SIZE_BYTES because it is not a positive integer.");
        }

        return validated;
    }

    private static bool IsOpenVinoOverrideEnabled()
    {
        return string.Equals(
            Environment.GetEnvironmentVariable("TRACKDUB_ALLOW_OPENVINO_OVERRIDE"),
            "true",
            StringComparison.OrdinalIgnoreCase);
    }

    private static string? TryGetValidatedOpenVinoDownloadUrl(string? rawUrl)
    {
        if (string.IsNullOrWhiteSpace(rawUrl))
        {
            return null;
        }

        if (!Uri.TryCreate(rawUrl, UriKind.Absolute, out Uri? uri))
        {
            return null;
        }

        if (!string.Equals(uri.Scheme, Uri.UriSchemeHttps, StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        return uri.ToString();
    }

    private static string? TryGetValidatedOpenVinoSha256(string? rawHash)
    {
        if (string.IsNullOrWhiteSpace(rawHash))
        {
            return null;
        }

        string normalized = rawHash.Trim();
        return System.Text.RegularExpressions.Regex.IsMatch(normalized, "^[A-Fa-f0-9]{64}$")
            ? normalized.ToLowerInvariant()
            : null;
    }
}
