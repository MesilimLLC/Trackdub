using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Licensing;
using Trackdub.Application.Projects;
using Trackdub.Application.Runtime;
using Trackdub.Application.Transcripts;
using Trackdub.Composition;
using Trackdub.Composition.NvidiaAfx;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Contracts.Pipeline;
using Trackdub.Infrastructure.FileSystem;
using Trackdub.Infrastructure.Logging;
using Trackdub.Infrastructure.Persistence.Sqlite;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.ExecutionProviders;
using Trackdub.Inference.Onnx.QwenTextRefinement;
using Trackdub.Inference.Runtime.Planning;
using Trackdub.TestDoubles;
using Microsoft.Extensions.DependencyInjection;

namespace Trackdub.Composition.Tests;

public sealed class CompositionRootTests : IDisposable
{
    private readonly List<string> tempDirectories = [];

    [Fact]
    public void AddTrackdub_builds_with_scope_validation()
    {
        using ServiceProvider provider = BuildProvider();
        using IServiceScope scope = provider.CreateScope();

        Assert.NotNull(provider.GetRequiredService<TranscriptWorkspaceFactory>());
        Assert.NotNull(provider.GetRequiredService<ProjectSessionService>());
        Assert.NotNull(scope.ServiceProvider.GetRequiredService<IRuntimeModelBootstrapService>());
    }

    [Fact]
    public void AddTrackdub_resolves_routed_text_refinement_engine_that_prefers_local_qwen()
    {
        using ServiceProvider provider = BuildProvider();
        using IServiceScope scope = provider.CreateScope();

        ITextRefinementEngine engine = scope.ServiceProvider.GetRequiredService<ITextRefinementEngine>();

        Assert.IsType<RoutedTextRefinementEngine>(engine);
        Assert.Equal(QwenTextRefinementEngine.EngineFamilyName, engine.EngineFamily);
    }

    [Fact]
    public void AddTrackdub_registers_subtitle_export_service_for_headless_paths()
    {
        // ExportStageHandler requires SubtitleExportService. The Avalonia shell registers it in
        // AvaloniaPlaybackComposition, but headless consumers (SDK/CLI/API/Worker) only get
        // CompositionRoot registrations; without this one, first-dub session creation fails.
        var services = new ServiceCollection();
        services.AddTrackdub();

        Assert.Contains(services, descriptor => descriptor.ServiceType == typeof(SubtitleExportService));
    }

    [Fact]
    public async Task AddTrackdub_execution_provider_discovery_uses_registered_trt_rtx_readiness_probe()
    {
        var settings = new FakeStudioSettingsService();
        await settings.SaveAsync(
            StudioSettings.Default with { NvidiaTensorRtRtxLicenseAccepted = true },
            CancellationToken.None);
        var readinessProbe = new StubTensorRtRtxReadinessProbe(new TensorRtRtxReadinessReport(
            ProviderId: TensorRtRtxProviderIds.PluginEpAbi,
            Route: TensorRtRtxPlatformRoute.PluginEpAbi,
            Blocker: TensorRtRtxReadinessBlocker.None,
            IsHardwareEligible: true,
            IsOrtProviderListed: true,
            IsRegisteredWithOrt: true,
            Detail: "DI TensorRT RTX readiness probe used."));
        using ServiceProvider provider = BuildProvider(services =>
        {
            services.AddSingleton<IStudioSettingsService>(settings);
            services.AddSingleton<ITensorRtRtxReadinessProbe>(readinessProbe);
        });

        IExecutionProviderDiscovery discovery = provider.GetRequiredService<IExecutionProviderDiscovery>();
        IReadOnlyList<ExecutionProviderAvailability> availabilities = await discovery.DiscoverAsync(
            new HardwareProfile("linux", "x64", HasGpu: true, GpuDescription: "NVIDIA RTX 4090"),
            CancellationToken.None);

        Assert.Equal(1, readinessProbe.CallCount);
        Assert.True(availabilities.Single(a => a.Provider == ExecutionProviderKind.TensorRTRtx).IsAvailable);
    }

    [Fact]
    public async Task AddTrackdub_linux_bootstrapper_uses_registered_trt_rtx_provider_bootstrap()
    {
        if (!OperatingSystem.IsLinux())
        {
            return;
        }

        var providerBootstrap = new StubTensorRtRtxProviderBootstrap(new TensorRtRtxBootstrapResult(
            Succeeded: true,
            ProviderId: TensorRtRtxProviderIds.PluginEpAbi,
            Blocker: null,
            Detail: "DI TensorRT RTX provider bootstrap used."));
        using ServiceProvider provider = BuildProvider(services =>
            services.AddSingleton<ITensorRtRtxProviderBootstrap>(providerBootstrap));

        IExecutionProviderBootstrapper bootstrapper = provider.GetRequiredService<IExecutionProviderBootstrapper>();
        ExecutionProviderBootstrapResult result = await bootstrapper.BootstrapAsync(
            ExecutionProviderKind.TensorRTRtx,
            allowDownloads: false,
            CancellationToken.None);

        Assert.Equal(1, providerBootstrap.CallCount);
        Assert.True(result.Succeeded);
        Assert.Equal(ExecutionProviderKind.TensorRTRtx, result.SelectedProvider);
        Assert.Equal("DI TensorRT RTX provider bootstrap used.", result.Detail);
    }

#if !WINDOWS
    /// <summary>
    /// The portable build selects the platform bootstrapper and device enumerator at run time, not
    /// by the OS that built it.
    /// </summary>
    [Fact]
    public void AddTrackdub_portable_build_selects_linux_services_at_run_time()
    {
        if (!OperatingSystem.IsLinux())
        {
            return;
        }

        using ServiceProvider provider = BuildProvider(_ => { });

        Assert.IsType<Trackdub.Inference.Onnx.ExecutionProviders.Linux.LinuxExecutionProviderBootstrapper>(
            provider.GetRequiredService<IExecutionProviderBootstrapper>());
        Assert.IsType<Trackdub.Inference.Onnx.Runtime.LinuxDeviceEnumerator>(
            provider.GetRequiredService<Trackdub.Inference.Runtime.Planning.IDeviceEnumerator>());
    }

    /// <summary>
    /// macOS counterpart of the Linux run-time selection test.
    /// </summary>
    [Fact]
    public void AddTrackdub_portable_build_selects_macos_services_at_run_time()
    {
        if (!OperatingSystem.IsMacOSVersionAtLeast(10, 15))
        {
            return;
        }

        using ServiceProvider provider = BuildProvider(_ => { });

        Assert.IsType<Trackdub.Inference.Onnx.ExecutionProviders.Mac.MacExecutionProviderBootstrapper>(
            provider.GetRequiredService<IExecutionProviderBootstrapper>());
        Assert.IsType<Trackdub.Inference.Onnx.Runtime.MacDeviceEnumerator>(
            provider.GetRequiredService<Trackdub.Inference.Runtime.Planning.IDeviceEnumerator>());
    }
#endif

#if WINDOWS
    /// <summary>
    /// Regression test: the Windows bootstrapper wired by CompositionRoot must route TRT-RTX
    /// bootstrap through the DI-registered <see cref="ITensorRtRtxProviderBootstrap"/> and NOT
    /// through the null-provider <c>TensorRtRtxPluginService.Shared</c> singleton.
    /// </summary>
    [WindowsOnlyFact]
    public async Task AddTrackdub_windows_bootstrapper_uses_registered_trt_rtx_provider_bootstrap()
    {
        var providerBootstrap = new StubTensorRtRtxProviderBootstrap(new TensorRtRtxBootstrapResult(
            Succeeded: true,
            ProviderId: TensorRtRtxProviderIds.PluginEpAbi,
            Blocker: null,
            Detail: "DI TensorRT RTX provider bootstrap used (Windows)."));
        using ServiceProvider provider = BuildProvider(services =>
            services.AddSingleton<ITensorRtRtxProviderBootstrap>(providerBootstrap));

        IExecutionProviderBootstrapper bootstrapper = provider.GetRequiredService<IExecutionProviderBootstrapper>();
        ExecutionProviderBootstrapResult result = await bootstrapper.BootstrapAsync(
            ExecutionProviderKind.TensorRTRtx,
            allowDownloads: false,
            CancellationToken.None);

        Assert.Equal(1, providerBootstrap.CallCount);
        Assert.True(result.Succeeded);
        Assert.Equal(ExecutionProviderKind.TensorRTRtx, result.SelectedProvider);
        Assert.Equal("DI TensorRT RTX provider bootstrap used (Windows).", result.Detail);
    }
#endif

    /// <summary>
    /// Regression guard: the readiness probes must be registered as their caching decorators so
    /// callers can invalidate the memoized result after a state-changing install (the mechanism
    /// TrtRtxProvidersHandler and GpuRuntimeInstallOrchestrator rely on). A future DI change that
    /// dropped the caching decorator would silently reintroduce the stale-read bug; the soft
    /// <c>is IReadinessProbeCache</c> cast would just skip invalidation. This test fails loudly if
    /// that happens.
    /// </summary>
    [Fact]
    public void AddTrackdub_registers_caching_readiness_probes_that_support_invalidation()
    {
        using ServiceProvider provider = BuildProvider();

        Assert.IsAssignableFrom<IReadinessProbeCache>(provider.GetRequiredService<ITensorRtRtxReadinessProbe>());
        Assert.IsAssignableFrom<IReadinessProbeCache>(provider.GetRequiredService<IOpenVinoCatalogReadinessProbe>());
        Assert.IsAssignableFrom<IReadinessProbeCache>(provider.GetRequiredService<IQnnCatalogReadinessProbe>());
        Assert.IsAssignableFrom<IReadinessProbeCache>(provider.GetRequiredService<IVitisAiCatalogReadinessProbe>());
    }

    [Fact]
    public void AddTrackdub_registers_update_service()
    {
        using ServiceProvider provider = BuildProvider();
        Trackdub.Application.Services.IUpdateService? service = provider.GetService<Trackdub.Application.Services.IUpdateService>();
        Assert.NotNull(service);
    }

    [Fact]
    public void AddTrackdub_does_not_register_legacy_project_service_adapter()
    {
        using ServiceProvider provider = BuildProvider();
        using IServiceScope scope = provider.CreateScope();

#pragma warning disable CS0618 // The assertion documents that the transitional adapter is not a DI entry point.
        Assert.Null(scope.ServiceProvider.GetService<TranscriptProjectService>());
#pragma warning restore CS0618
    }

    [Fact]
    public void AddTrackdub_registers_bounded_file_logger()
    {
        using ServiceProvider provider = BuildProvider();

        var logger = Assert.IsType<RollingFileApplicationLogger>(provider.GetRequiredService<IApplicationLogger>());
        Assert.EndsWith(Path.Join("Trackdub", "trackdub.log"), logger.LogFilePath);
        Assert.Equal(ApplicationLogLevel.Information, logger.MinimumLevel);
        Assert.Equal(1 * 1024 * 1024, logger.MaxFileBytes);
        Assert.Equal(RollingFileApplicationLogger.SessionMaxArchiveFiles, logger.MaxArchiveFiles);
    }

    [Fact]
    public void AddTrackdub_registers_diagnostics_runtime_info_provider()
    {
        using ServiceProvider provider = BuildProvider();

        Assert.NotNull(provider.GetRequiredService<IDiagnosticsRuntimeInfo>());
    }

    /// <summary>
    /// Regression guard for hosts that offer a "warm caches" action outside the CLI. The desktop
    /// shell resolves <see cref="IEpContextWarmupService"/> from this same graph, so a DI change
    /// that dropped it — or left one of its dependencies unregistered — would surface only when a
    /// user clicked warm, not at build time.
    /// </summary>
    [Fact]
    public void AddTrackdub_registers_ep_context_warmup_service()
    {
        using ServiceProvider provider = BuildProvider();

        Assert.NotNull(provider.GetRequiredService<IEpContextWarmupService>());
    }

    /// <summary>
    /// Regression test for P0-5: IHardwareInfoService and IFfmpegVideoEncoderCapabilities were
    /// unregistered, causing SettingsWindowViewModel's real constructor to be unsatisfiable. DI
    /// silently fell back to the parameterless ctor (LocalModels = null!) → blank Model Manager tab.
    /// </summary>
    [Fact]
    public void AddTrackdub_registers_settings_window_required_deps()
    {
        using ServiceProvider provider = BuildProvider();

        Assert.NotNull(provider.GetService<IHardwareInfoService>());
        Assert.NotNull(provider.GetService<IFfmpegVideoEncoderCapabilities>());
    }

    /// <summary>
    /// Regression test: IMediaHardwareCapabilitiesService and its IMediaGpuHintProvider dependency
    /// were unregistered, so AvaloniaMainWindowViewModel.RefreshMediaCapabilitiesAsync threw
    /// "No service for type 'IMediaHardwareCapabilitiesService' has been registered" and the media
    /// capability probe (FFmpeg encoders + GPU hint) was silently lost on startup.
    /// </summary>
    [Fact]
    public void AddTrackdub_registers_media_hardware_capabilities_service()
    {
        using ServiceProvider provider = BuildProvider();
        using IServiceScope scope = provider.CreateScope();

        Assert.NotNull(scope.ServiceProvider.GetService<IMediaHardwareCapabilitiesService>());
        Assert.NotNull(scope.ServiceProvider.GetService<IMediaGpuHintProvider>());
    }

    [Fact]
    public void AddTrackdub_registers_managed_glossary_analyzers()
    {
        using ServiceProvider provider = BuildProvider();
        using IServiceScope scope = provider.CreateScope();

        IGlossaryAnalyzerCatalog catalog = scope.ServiceProvider.GetRequiredService<IGlossaryAnalyzerCatalog>();
        IGlossaryTermMatcher matcher = scope.ServiceProvider.GetRequiredService<IGlossaryTermMatcher>();
        IGlossaryLanguageAnalyzer[] analyzers = scope.ServiceProvider
            .GetServices<IGlossaryLanguageAnalyzer>()
            .ToArray();

        Assert.NotNull(catalog);
        Assert.IsType<GlossaryTermMatcher>(matcher);
        Assert.Contains(analyzers, analyzer => analyzer.Supports("ja"));
        Assert.Contains(analyzers, analyzer => analyzer.Supports("zh-Hans"));
        Assert.Contains(analyzers, analyzer => analyzer.Supports("ar"));
        Assert.DoesNotContain(analyzers, analyzer => analyzer.Supports("ko"));
        Assert.NotEmpty(Assert.Single(analyzers, analyzer => analyzer.Supports("ja")).Analyze("ja", "寿司を食べる"));
        Assert.NotEmpty(Assert.Single(analyzers, analyzer => analyzer.Supports("zh-Hans")).Analyze("zh-Hans", "我是中国人"));
        Assert.NotEmpty(Assert.Single(analyzers, analyzer => analyzer.Supports("ar")).Analyze("ar", "والكِتاب"));
    }

    [Fact]
    public void Workspace_sessions_use_distinct_project_scopes()
    {
        using ServiceProvider provider = BuildProvider();
        TranscriptWorkspaceFactory factory = provider.GetRequiredService<TranscriptWorkspaceFactory>();

        using TranscriptWorkspaceSession first = factory.Create(CreateTempDirectory());
        using TranscriptWorkspaceSession second = factory.Create(CreateTempDirectory());

        Assert.IsType<AtomicRevisionCommitBoundary>(first.Services.GetRequiredService<IAtomicRevisionCommitBoundary>());
        Assert.IsType<SqliteAtomicRevisionPersistence>(first.Services.GetRequiredService<Trackdub.Contracts.Transcripts.IAtomicRevisionPersistence>());

        var firstDatabase = first.Services.GetRequiredService<SqliteProjectDatabase>();
        var secondDatabase = second.Services.GetRequiredService<SqliteProjectDatabase>();
        var firstStore = Assert.IsType<FileSystemArtifactStore>(first.Services.GetRequiredService<IArtifactStore>());
        var secondStore = Assert.IsType<FileSystemArtifactStore>(second.Services.GetRequiredService<IArtifactStore>());
        IVoiceCatalog firstVoiceCatalog = first.Services.GetRequiredService<IVoiceCatalog>();
        IVoiceCatalog secondVoiceCatalog = second.Services.GetRequiredService<IVoiceCatalog>();

        Assert.NotEqual(firstDatabase.DatabasePath, secondDatabase.DatabasePath);
        Assert.NotEqual(firstStore.GetPath("manifest.json"), secondStore.GetPath("manifest.json"));
        Assert.NotSame(firstVoiceCatalog, secondVoiceCatalog);
    }

    [Fact]
    public void Workspace_session_applies_tts_timing_settings()
    {
        using ServiceProvider provider = BuildProvider();
        TranscriptWorkspaceFactory factory = provider.GetRequiredService<TranscriptWorkspaceFactory>();
        StudioSettings settings = StudioSettings.Default with
        {
            TtsTiming = new TtsTimingSettings(
                EnableRubberbandStretch: true,
                RubberbandStretchThreshold: 0.42d)
        };

        using TranscriptWorkspaceSession session = factory.Create(CreateTempDirectory(), settings);
        TtsTimingOptions options = session.Services.GetRequiredService<TtsTimingOptions>();

        Assert.True(options.EnableRubberbandStretch);
        Assert.Equal(0.42d, options.RubberbandStretchThreshold);
    }

    [Fact]
    public void Workspace_session_applies_auto_stretch_max_overrun_setting()
    {
        using ServiceProvider provider = BuildProvider();
        TranscriptWorkspaceFactory factory = provider.GetRequiredService<TranscriptWorkspaceFactory>();
        StudioSettings settings = StudioSettings.Default with
        {
            TtsTiming = new TtsTimingSettings(
                EnableRubberbandStretch: false,
                RubberbandStretchThreshold: 0.15d,
                AutoStretchMaxOverrun: 0.45d)
        };

        using TranscriptWorkspaceSession session = factory.Create(CreateTempDirectory(), settings);
        TtsTimingOptions options = session.Services.GetRequiredService<TtsTimingOptions>();

        Assert.Equal(0.45d, options.AutoStretchMaxOverrun);
    }

    [Fact]
    public void Workspace_session_keeps_default_auto_stretch_max_overrun_when_unset()
    {
        using ServiceProvider provider = BuildProvider();
        TranscriptWorkspaceFactory factory = provider.GetRequiredService<TranscriptWorkspaceFactory>();

        using TranscriptWorkspaceSession session = factory.Create(CreateTempDirectory());
        TtsTimingOptions options = session.Services.GetRequiredService<TtsTimingOptions>();

        Assert.Equal(TtsTimingOptions.Default.AutoStretchMaxOverrun, options.AutoStretchMaxOverrun);
    }

    [Fact]
    public void Workspace_session_owns_scoped_engine_lifetime()
    {
        using ServiceProvider provider = BuildProvider(services =>
        {
            services.AddScoped<ITtsEngine, DisposableTtsEngine>();
        });
        TranscriptWorkspaceFactory factory = provider.GetRequiredService<TranscriptWorkspaceFactory>();

        TranscriptWorkspaceSession first = factory.Create(CreateTempDirectory());
        using TranscriptWorkspaceSession second = factory.Create(CreateTempDirectory());
        var firstEngine = Assert.IsType<DisposableTtsEngine>(first.Services.GetRequiredService<ITtsEngine>());
        var secondEngine = Assert.IsType<DisposableTtsEngine>(second.Services.GetRequiredService<ITtsEngine>());

        Assert.Same(firstEngine, first.Services.GetRequiredService<ITtsEngine>());
        Assert.NotSame(firstEngine, secondEngine);

        first.Dispose();
        first.Dispose();

        Assert.Equal(1, firstEngine.DisposeCount);
        Assert.Equal(0, secondEngine.DisposeCount);
    }

    [WindowsOnlyFact]
    public async Task AddTrackdub_resolves_the_probe_based_afx_readiness_service_with_saved_settings()
    {
        string runtimeDirectory = CreateTempDirectory();
        var settings = new FakeStudioSettingsService();
        await settings.SaveAsync(
            StudioSettings.Default with
            {
                NvidiaAfxRuntimeDirectory = runtimeDirectory,
                NvidiaAfxLicenseAccepted = true,
            },
            CancellationToken.None);
        using ServiceProvider provider = BuildProvider(services =>
            services.AddSingleton<IStudioSettingsService>(settings));

        INvidiaAfxRuntimeReadinessService readiness = provider.GetRequiredService<INvidiaAfxRuntimeReadinessService>();
        NvidiaAfxRuntimeReadiness result = readiness.GetReadiness(NvidiaAfxProfile.NoiseAndReverb);

        Assert.IsType<NvidiaAfxRuntimeReadinessService>(readiness);
        Assert.False(result.IsReady);
        Assert.Equal("Missing native library", result.StatusLabel);
        Assert.Equal(runtimeDirectory, result.RuntimeRoot);
    }

    public void Dispose()
    {
        foreach (string tempDirectory in tempDirectories)
        {
            try
            {
                if (Directory.Exists(tempDirectory))
                {
                    Directory.Delete(tempDirectory, recursive: true);
                }
            }
            catch
            {
                // Best-effort cleanup for temp project folders created by tests.
            }
        }
    }

    private static ServiceProvider BuildProvider(Action<IServiceCollection>? configure = null)
    {
        var services = new ServiceCollection();
        configure?.Invoke(services);
        services.AddTrackdub();
        return services.BuildServiceProvider(new ServiceProviderOptions
        {
            ValidateOnBuild = true,
            ValidateScopes = true
        });
    }

    private string CreateTempDirectory()
    {
        string path = Path.Join(Path.GetTempPath(), "Trackdub.Composition.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        tempDirectories.Add(path);
        return path;
    }

    private sealed class StubTensorRtRtxReadinessProbe(TensorRtRtxReadinessReport report)
        : ITensorRtRtxReadinessProbe
    {
        public int CallCount { get; private set; }

        public Task<TensorRtRtxReadinessReport> ProbeAsync(
            bool allowProviderDownloads,
            CancellationToken cancellationToken = default)
        {
            CallCount++;
            return Task.FromResult(report);
        }
    }

    private sealed class StubTensorRtRtxProviderBootstrap(TensorRtRtxBootstrapResult result)
        : ITensorRtRtxProviderBootstrap
    {
        public int CallCount { get; private set; }

        public Task<TensorRtRtxBootstrapResult> EnsureRegisteredAsync(
            bool allowProviderDownloads,
            CancellationToken cancellationToken = default)
        {
            CallCount++;
            return Task.FromResult(result);
        }
    }

    private sealed class DisposableTtsEngine : ITtsEngine, IDisposable
    {
        public int DisposeCount { get; private set; }

        public Task<TtsSynthesisResult> SynthesizeAsync(
            TtsSynthesisRequest request,
            CancellationToken cancellationToken)
        {
            var result = new TtsSynthesisResult(
                [],
                DurationSamples: 0,
                SampleRate: 24_000,
                ModelId: "fake-tts",
                VoiceId: request.Voice.VoiceId,
                Provider: "fake");
            return Task.FromResult(result);
        }

        public void Dispose()
        {
            DisposeCount++;
        }
    }
}
