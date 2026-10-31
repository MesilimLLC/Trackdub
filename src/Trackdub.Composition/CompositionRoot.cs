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

internal static class StemTempCleanup
{
    private static int processExitHandlerRegistered;

    public static void RegisterProcessExitHandler()
    {
        if (Interlocked.Exchange(ref processExitHandlerRegistered, 1) == 1)
        {
            return;
        }

        AppDomain.CurrentDomain.ProcessExit += OnProcessExit;
    }

    public static void OnProcessExit(object? sender, EventArgs e)
    {
        StemSeparationTempDirectories.CleanupStale(DateTimeOffset.UtcNow);
    }
}

internal sealed class CpuOnlyDeviceEnumerator : IDeviceEnumerator
{
    private static readonly IReadOnlyList<DeviceEntry> CpuOnlyDevices =
    [
        new DeviceEntry(
            DeviceKind.Cpu,
            0,
            "CPU",
            "Fallback",
            0,
            0,
            [ExecutionProviderKind.Cpu])
    ];

    public Task<IReadOnlyList<DeviceEntry>> GetDevicesAsync(CancellationToken cancellationToken = default) =>
        Task.FromResult(CpuOnlyDevices);

    public Task<IReadOnlyList<DeviceEntry>> ReEnumerateAsync(CancellationToken cancellationToken = default) =>
        Task.FromResult(CpuOnlyDevices);
}

public static class CompositionRoot
{
    public static IServiceCollection AddTrackdub(this IServiceCollection services)
    {
        StemTempCleanup.RegisterProcessExitHandler();
        AddInfrastructure(services);
        AddApplication(services);
        AddSequenceExport(services);
        AddInference(services);
        AddHardwareRouting(services);
        return services;
    }

    private static void AddInfrastructure(IServiceCollection services)
    {
        var storagePaths = new TrackdubStoragePaths();
        TrackdubStoragePathResolver.ApplyToCurrentProcess(storagePaths);

        services.TryAddSingleton(storagePaths);
        services.TryAddSingleton<IAppStoragePaths>(storagePaths);
        services.TryAddSingleton<IEngineCacheMaintenanceService, EngineCacheMaintenanceService>();
        services.TryAddSingleton<IEpContextWarmupService>(sp => new Trackdub.Inference.Onnx.EpContext.EpContextWarmupService(
            sp.GetRequiredService<IAppStoragePaths>(),
            sp.GetRequiredService<IHardwareProfileProvider>(),
            new Trackdub.Inference.Onnx.EpContext.EpContextCompiler(
                sp.GetRequiredService<ITensorRtRtxProviderBootstrap>())));
        services.TryAddSingleton<IResidentInferenceHost>(sp =>
        {
            IExecutionProviderDiscovery discovery = sp.GetRequiredService<IExecutionProviderDiscovery>();
            IHardwareProfileProvider hardware = sp.GetRequiredService<IHardwareProfileProvider>();
            return new Trackdub.Inference.Onnx.Serving.ResidentInferenceHost(async cancellationToken =>
            {
                HardwareProfile profile = await hardware.GetCurrentAsync(cancellationToken).ConfigureAwait(false);
                IReadOnlyList<ExecutionProviderAvailability> availability =
                    await discovery.DiscoverAsync(profile, cancellationToken).ConfigureAwait(false);
                return ExecutionProviderProbeOrder.InProcessAvailable(availability);
            });
        });
        // Process-level efficiency control for host processes (serve --ecoqos). Resolved through
        // DI because the CLI must not reference the inference runtime directly (AGENTS.md graph).
        services.TryAddSingleton<IProcessEfficiencyControl>(_ => new Trackdub.Inference.Onnx.Serving.ProcessEfficiencyControl());
        // Readiness probes are wrapped in thread-safe caching decorators so a single
        // `providers list` invocation runs each underlying native/ORT readiness check exactly
        // once. Both OnnxExecutionProviderDiscovery and the *RuntimeReadinessService wrappers
        // resolve these same cached singletons, so the discovery pass and the remediation pass
        // reuse one probe result instead of double-probing. Because these singletons live for the
        // process lifetime, callers that perform a state-changing install/register (e.g.
        // TrtRtxProvidersHandler.InstallAsync) invalidate the cache via IReadinessProbeCache before
        // re-probing so verification observes the new state, not the stale pre-change snapshot.
        services.TryAddSingleton<IMigraphxReadinessProbe>(_ =>
            new CachingMigraphxReadinessProbe(new MigraphxReadinessProbe()));
        services.TryAddSingleton<IDnnlReadinessProbe>(_ =>
            new CachingDnnlReadinessProbe(new DnnlReadinessProbe()));
        services.TryAddSingleton<IOpenVinoCatalogReadinessProbe>(_ =>
            new CachingOpenVinoCatalogReadinessProbe(new OpenVinoCatalogReadinessProbe()));
        services.TryAddSingleton<IQnnCatalogReadinessProbe>(_ =>
            new CachingQnnCatalogReadinessProbe(new QnnCatalogReadinessProbe()));
        services.TryAddSingleton<IVitisAiCatalogReadinessProbe>(_ =>
            new CachingVitisAiCatalogReadinessProbe(new VitisAiCatalogReadinessProbe()));
        services.TryAddSingleton<IDiagnosticsRuntimeInfo, DiagnosticsRuntimeInfoProvider>();
        services.TryAddSingleton<IApplicationLogger>(sp =>
        {
            var logger = new RollingFileApplicationLogger(
                sp.GetRequiredService<TrackdubStoragePaths>().LogFilePath,
                maxArchiveFiles: RollingFileApplicationLogger.SessionMaxArchiveFiles,
                minimumLevel: ApplicationLogLevel.Information,
                rotateOnStartup: true);
            // Engine-capture skips otherwise only reach a debug trace. A skipped capture means
            // TensorRT engines rebuild on every launch, so it must be visible in the operator log.
            Trackdub.Inference.Onnx.OnnxExecutionSessionFactory.DiagnosticsLogger = logger;
            return logger;
        });
        services.TryAddSingleton<LocalModelCacheRecordStore>();
        services.TryAddSingleton<IDiagnosticsBundleExporter, DiagnosticsBundleExporter>();
        services.TryAddSingleton<IModelCacheRegistrar, LocalModelCacheRegistrar>();
        services.TryAddSingleton<IModelCacheRecordLookup, LocalModelCacheRecordLookup>();
        services.TryAddSingleton<IModelVariantRegistrar, LocalModelVariantRegistrar>();
        services.TryAddSingleton<IModelInventoryService, ModelInventoryService>();
        services.TryAddSingleton<StarterPackValidator>();
        services.TryAddSingleton<IStarterPackCatalog>(sp =>
            new StarterPackCatalog(
                sp.GetRequiredService<IAppStoragePaths>(),
                sp.GetRequiredService<StarterPackValidator>(),
                sp.GetRequiredService<BundledModelManifestRegistry>()));
        services.TryAddSingleton<IStarterPackDownloadService, StarterPackDownloadService>();
        services.TryAddSingleton<IStarterPackApplyService, StarterPackApplyService>();
        services.TryAddSingleton<IStarterPackCompatibilityService, StarterPackCompatibilityService>();
        services.TryAddSingleton<IStarterPackPresentationService, StarterPackPresentationService>();
        services.TryAddSingleton<IStarterPackImportExportService, StarterPackImportExportService>();
        services.TryAddSingleton<IStarterPackOptimizationNudgeService, StarterPackOptimizationNudgeService>();
        services.TryAddSingleton<IStarterPackPatchApplier, StarterPackPatchApplier>();
        services.TryAddSingleton<IStarterPackCoordinator, StarterPackCoordinator>();
        services.TryAddSingleton<StarterPackValidator>();
        services.TryAddSingleton<IMigraphxRuntimeReadinessService, MigraphxRuntimeReadinessService>();
        services.TryAddSingleton<IOpenVinoCatalogRuntimeReadinessService, OpenVinoCatalogRuntimeReadinessService>();
        services.TryAddSingleton<IQnnCatalogRuntimeReadinessService, QnnCatalogRuntimeReadinessService>();
        services.TryAddSingleton<IVitisAiCatalogRuntimeReadinessService, VitisAiCatalogRuntimeReadinessService>();
        services.TryAddSingleton<IWindowsMlCertifiedCatalogInstaller, WindowsMlCertifiedCatalogInstaller>();
        services.TryAddSingleton<IOpenVinoCatalogEpInstaller, OpenVinoCatalogEpInstaller>();
        services.TryAddSingleton<IQnnCatalogEpInstaller, QnnCatalogEpInstaller>();
        services.TryAddSingleton<IVitisAiCatalogEpInstaller, VitisAiCatalogEpInstaller>();
        services.TryAddSingleton<StarterPackGpuSetupAdvisor>();
        services.TryAddSingleton<IGpuRuntimeInstallOrchestrator, GpuRuntimeInstallOrchestrator>();
        // The warm queue is injected here so a successful install drives engine warming directly.
        // ModelDownloadOrchestrator is constructed on every download path, so this trigger runs
        // where installs actually happen rather than depending on some other type being resolved.
        services.TryAddSingleton(sp => new ModelDownloadOrchestrator(
            sp.GetRequiredService<BundledModelManifestRegistry>(),
            sp.GetRequiredService<LocalModelCacheRecordStore>(),
            sp.GetRequiredService<IModelDownloaderContract>(),
            sp.GetRequiredService<TrackdubStoragePaths>(),
            sp.GetService<IModelHashVerifier>(),
            sp.GetService<IApplicationLogger>(),
            sp.GetService<IStudioSettingsService>(),
            sp.GetService<IEngineAutoWarmQueue>(),
            sp.GetService<KokoroVoiceCatalogHolder>()));
        services.TryAddSingleton<IModelDownloadOrchestrator>(sp => sp.GetRequiredService<ModelDownloadOrchestrator>());
        services.TryAddSingleton<IModelCacheVerifier>(sp => sp.GetRequiredService<ModelDownloadOrchestrator>());
        services.TryAddSingleton<IFileFingerprintService>(sp =>
            new Sha256FileFingerprintService(sp.GetRequiredService<IApplicationLogger>()));
        services.TryAddSingleton<IFileSystemProbe, PhysicalFileSystemProbe>();
        services.TryAddSingleton<IStudioSettingsService, JsonStudioSettingsService>();
        services.TryAddSingleton<INativeCudaTensorRtWindowsPolicy, StudioSettingsNativeCudaTensorRtWindowsPolicy>();
        services.AddHttpClient<Trackdub.Application.Services.IUpdateService, ReleaseManifestUpdateService>(client =>
        {
            client.Timeout = TimeSpan.FromSeconds(30);
            client.DefaultRequestHeaders.UserAgent.ParseAdd("Trackdub/1.0");
        });
        services.AddHttpClient("TrtRtxEpBundleDownloader", client => client.Timeout = TimeSpan.FromMinutes(30));
        services.TryAddSingleton<TrtRtxEpBundleDownloader>(sp =>
            new TrtRtxEpBundleDownloader(
                sp.GetRequiredService<IHttpClientFactory>().CreateClient("TrtRtxEpBundleDownloader"),
                sp.GetRequiredService<IApplicationLogger>()));
        services.TryAddSingleton<ITrtRtxEpBundleInstaller, TrtRtxEpBundleInstaller>();
        services.TryAddSingleton<ITrtRtxEpInstaller, TrtRtxEpInstaller>();
        services.TryAddSingleton<ITensorRtRtxProviderBootstrap>(sp =>
        {
            IStudioSettingsService settingsService = sp.GetRequiredService<IStudioSettingsService>();
            TrackdubStoragePaths storagePaths = sp.GetRequiredService<TrackdubStoragePaths>();
            ITrtRtxEpBundleInstaller bundleInstaller = sp.GetRequiredService<ITrtRtxEpBundleInstaller>();

            return TensorRtRtxProviderBootstrapFactory.Create(
                async cancellationToken =>
                {
                    StudioSettings settings = await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
                    return settings.TensorRtRtxPluginDirectory;
                },
                cancellationToken =>
                {
                    _ = cancellationToken;
                    if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
                    {
                        return ValueTask.FromResult<string?>(null);
                    }

                    string runtimeIdentifier = OperatingSystem.IsWindows() ? "win-x64" : "linux-x64";
                    string installDirectory = TensorRtRtxProviderConstants.GetDefaultInstallDirectory(
                        storagePaths.UserDataRoot,
                        runtimeIdentifier);
                    return ValueTask.FromResult<string?>(installDirectory);
                },
                async (allowProviderDownloads, cancellationToken) =>
                {
                    if (!allowProviderDownloads)
                    {
                        return new TensorRtRtxBundleEnsureResult(
                            false,
                            null,
                            "Provider downloads are disabled for this probe.");
                    }

                    StudioSettings bundleSettings = await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
                    if (!bundleSettings.NvidiaTensorRtRtxLicenseAccepted)
                    {
                        return new TensorRtRtxBundleEnsureResult(
                            false,
                            null,
                            "NVIDIA TensorRT RTX license not accepted. Accept the license in Model Manager before installing the plugin.");
                    }

                    TrtRtxEpBundleInstallResult result = await bundleInstaller
                        .EnsureBundleAsync(new Progress<string>(_ => { }), cancellationToken)
                        .ConfigureAwait(false);
                    return new TensorRtRtxBundleEnsureResult(
                        result.Succeeded,
                        result.InstallDirectory,
                        result.FailureDetail);
                },
                isLicenseAccepted: async cancellationToken =>
                    (await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false))
                        .NvidiaTensorRtRtxLicenseAccepted);
        });
        services.TryAddSingleton<ITensorRtRtxReadinessProbe>(sp =>
            new CachingTensorRtRtxReadinessProbe(
                new TensorRtRtxReadinessProbe(sp.GetRequiredService<ITensorRtRtxProviderBootstrap>())));
        services.TryAddSingleton<ITensorRtRtxRuntimeReadinessService, TensorRtRtxRuntimeReadinessService>();
        services.TryAddSingleton<ICloudApiKeyProvider, EnvironmentCloudApiKeyProvider>();
        services.TryAddSingleton<CloudEgressConsentService>(sp =>
            new CloudEgressConsentService(sp.GetRequiredService<IStudioSettingsService>()));
        services.TryAddScoped<ICloudEgressConsentService>(sp =>
        {
            StudioSettings? sessionSettings = null;
            if (sp.GetService<TranscriptWorkspaceContext>() is { } context &&
                context.TryGetSettings(out StudioSettings scopedSettings))
            {
                sessionSettings = scopedSettings;
            }

            return new SessionCloudEgressConsentOverlay(
                sp.GetRequiredService<CloudEgressConsentService>(),
                sessionSettings?.CloudEgressConsents);
        });
        services.TryAddSingleton<ICloudCredentialReadiness, CloudCredentialReadinessService>();
        services.TryAddSingleton<IConsentService>(sp =>
            new SqliteConsentService(
                sp.GetRequiredService<TrackdubStoragePaths>(),
                sp.GetRequiredService<IApplicationLogger>()));

        services.TryAddSingleton<IFfmpegHealthCheck>(_ => new FfmpegHealthCheck());
        services.TryAddSingleton<IEspeakNgHealthCheck, EspeakNgHealthCheck>();
        services.TryAddSingleton<IExplicitFfmpegInstaller>(_ => new FfmpegExplicitInstaller());
        services.TryAddSingleton<IDiagnosticsCollector>(sp =>
            new DiagnosticsCollector(
                sp.GetRequiredService<TrackdubStoragePaths>(),
                sp.GetRequiredService<LocalModelCacheRecordStore>(),
                sp.GetService<IDiagnosticsRuntimeInfo>()));

        services.TryAddSingleton(_ => HuggingFaceDownloadOptions.FromEnvironment());
        services.TryAddSingleton<ModelDownloadHttpClient>();
        services.TryAddSingleton<IModelDownloader>(sp =>
            new HuggingFaceModelDownloader(
                sp.GetRequiredService<TrackdubStoragePaths>().ModelCacheDirectory,
                sp.GetRequiredService<IApplicationLogger>(),
                httpClient: sp.GetRequiredService<ModelDownloadHttpClient>().Client,
                downloadOptions: sp.GetRequiredService<HuggingFaceDownloadOptions>()));
        services.TryAddSingleton<IModelContentHashCacheInvalidator, SessionPoolModelContentHashCacheInvalidator>();
        services.TryAddSingleton<IModelDownloaderContract>(sp =>
            new ModelDownloaderAdapter(
                sp.GetRequiredService<IModelDownloader>(),
                httpClient: sp.GetRequiredService<ModelDownloadHttpClient>().Client,
                logger: sp.GetService<IApplicationLogger>(),
                downloadOptions: sp.GetRequiredService<HuggingFaceDownloadOptions>(),
                contentHashCacheInvalidator: sp.GetRequiredService<IModelContentHashCacheInvalidator>()));

        services.TryAddSingleton<PlaybackCapabilityProbe>();
        services.TryAddSingleton<IPlaybackBackendFactory, DefaultPlaybackBackendFactory>();
        services.TryAddSingleton<PlaybackService>();

        services.TryAddScoped<TranscriptWorkspaceContext>();
        services.TryAddScoped<ITranscriptWorkspaceContext>(sp => sp.GetRequiredService<TranscriptWorkspaceContext>());
        services.TryAddScoped<SqliteProjectDatabase>(sp =>
            new SqliteProjectDatabase(sp.GetRequiredService<TranscriptWorkspaceContext>().ProjectRootPath));
        services.TryAddScoped<IArtifactStore>(sp =>
            new FileSystemArtifactStore(
                sp.GetRequiredService<TranscriptWorkspaceContext>().ProjectRootPath,
                sp.GetRequiredService<IApplicationLogger>()));
        services.TryAddScoped<IProjectRepository, SqliteProjectRepository>();
        services.TryAddScoped<IMediaAssetRepository, SqliteMediaAssetRepository>();
        services.TryAddScoped<SqliteProjectStageRunStore>();
        services.TryAddScoped<IProjectStageRunStore, ObservedProjectStageRunStore>();
        services.TryAddScoped<ITranscriptRepository, SqliteTranscriptRepository>();
        services.TryAddScoped<ITranslationRepository, SqliteTranslationRepository>();
        services.TryAddScoped<IAtomicRevisionPersistence, SqliteAtomicRevisionPersistence>();
        services.TryAddScoped<IAtomicRevisionCommitBoundary, AtomicRevisionCommitBoundary>();
        services.TryAddScoped<IGlossaryRepository, SqliteGlossaryRepository>();
        services.TryAddSingleton<SqliteUserGlossaryDatabase>(sp =>
            new SqliteUserGlossaryDatabase(sp.GetRequiredService<IAppStoragePaths>().UserDataRoot));
        services.TryAddSingleton<IGlobalGlossaryRepository, SqliteGlobalGlossaryRepository>();
        services.TryAddScoped<ISpeakerRepository, SqliteSpeakerRepository>();
        services.TryAddScoped<IVoiceAssignmentRepository, SqliteVoiceAssignmentRepository>();
        services.TryAddScoped<ITtsTakeRepository, SqliteTtsTakeRepository>();
        services.TryAddScoped<ITtsCandidateGroupRepository, TtsCandidateGroupRepository>();
        services.TryAddScoped<IVoiceCloneAuditLog, FileSystemVoiceCloneAuditLog>();
        services.TryAddScoped<ISpeakerConsentService, SqliteSpeakerConsentService>();
        services.TryAddScoped<IScopedConnectionProvider>(sp =>
            new ScopedSqliteConnectionProvider(
                Path.Join(
                    sp.GetRequiredService<TranscriptWorkspaceContext>().ProjectRootPath,
                    ProjectArtifactPaths.DatabaseFileName)));

        services.TryAddSingleton<IOliveEnvironmentService, OliveEnvironmentService>();
        services.TryAddSingleton<IOliveRecipesPathProvider, OliveRecipesPathProvider>();
        services.TryAddSingleton<IModelOptimizationService, OliveModelOptimizationService>();

        services.AddHttpClient("TrackdubUpdateService", client =>
        {
            client.Timeout = TimeSpan.FromMinutes(30);
            client.DefaultRequestHeaders.UserAgent.ParseAdd("Trackdub-UpdateService/1.0");
        });
        services.TryAddSingleton<IUpdateService>(sp =>
            new UpdateService(
                sp.GetRequiredService<IHttpClientFactory>().CreateClient("TrackdubUpdateService"),
                sp.GetRequiredService<IApplicationLogger>(),
                sp.GetRequiredService<IAppStoragePaths>()));
    }

    private static void AddApplication(IServiceCollection services)
    {
        services.TryAddSingleton<IMediaProbe>(_ => new FfmpegMediaProbe(ffmpegPath: null, ffprobePath: null));
        services.TryAddSingleton<IExportToolAvailabilityService>(_ => new FfmpegExportToolAvailabilityService(ffmpegPath: null, ffprobePath: null));
        services.TryAddSingleton<IAudioExtractionService>(_ => new FfmpegAudioExtractionService(ffmpegPath: null));
        services.TryAddSingleton<IFfmpegVideoEncoderCapabilities>(_ => new FfmpegVideoEncoderCapabilityService());
        services.TryAddSingleton<ILoudnessNormalizer>(_ => new FfmpegLoudnessNormalizer(ffmpegPath: null));
        services.TryAddSingleton<IExportRenderer>(_ => new FfmpegMuxer(ffmpegPath: null));
        services.TryAddSingleton<IVideoRecomposer>(_ => new FfmpegVideoRecomposer(ffmpegPath: null));
        services.TryAddSingleton<IAudioQualityAnalyzer, PcmAudioQualityAnalyzer>();
        services.TryAddSingleton<ISpeechAudioPreparationPlanner, SpeechAudioPreparationPlanner>();
        services.TryAddSingleton<ISpeechAudioProcessingService>(_ => new FfmpegSpeechAudioProcessingService(ffmpegPath: null));
        // NVIDIA AFX readiness is probe-based: Ready only for a local runtime (settings folder,
        // TRACKDUB_NVIDIA_AFX_RUNTIME_ROOT or a component-store install) that creates and loads an
        // effect on this machine. Downloads stay gated by the packaging checks in the manifest.
        services.TryAddSingleton<INvidiaAfxArchitectureDetector, NvidiaAfxArchitectureDetector>();
        services.TryAddSingleton<INvidiaAfxRuntimeReadinessService>(sp =>
            new NvidiaAfxRuntimeReadinessService(
                sp.GetRequiredService<ComponentStore>(),
                sp.GetRequiredService<INvidiaAfxArchitectureDetector>(),
                Path.Join(AppContext.BaseDirectory, "nvidiaafx-runtime.manifest.json"),
                () => sp.GetRequiredService<IStudioSettingsService>()
                    .LoadAsync(CancellationToken.None)
                    .ConfigureAwait(false)
                    .GetAwaiter()
                    .GetResult()));
        services.AddSingleton<ISpeechAudioEnhancementService>(sp =>
            new NvidiaAfxSpeechAudioEnhancementService(
                sp.GetRequiredService<INvidiaAfxRuntimeReadinessService>(),
                new Trackdub.Composition.DeepFilterNet.ResolvingSpeechAudioEnhancementService(
                    sp.GetService<BundledModelManifestRegistry>(),
                    sp.GetService<IModelCacheInventory>()),
                logger: sp.GetService<IApplicationLogger>()));
        services.TryAddSingleton<NvidiaAfxRuntimeDownloader>(sp =>
            new NvidiaAfxRuntimeDownloader(
                sp.GetRequiredService<ComponentStore>(),
                sp.GetRequiredService<IHttpClientFactory>().CreateClient("NvidiaAfxRuntimeDownloader"),
                sp.GetRequiredService<IApplicationLogger>()));
        services.TryAddSingleton<NvidiaAfxRuntimeInstaller>(sp =>
            new NvidiaAfxRuntimeInstaller(
                sp.GetRequiredService<NvidiaAfxRuntimeDownloader>(),
                sp.GetRequiredService<IStudioSettingsService>(),
                sp.GetRequiredService<INvidiaAfxArchitectureDetector>(),
                Path.Join(AppContext.BaseDirectory, "nvidiaafx-runtime.manifest.json")));
        services.TryAddSingleton<IWaveformSummaryGenerator, WaveformSummaryGenerator>();
        services.TryAddSingleton<IReferenceClipAnalyzer, Pcm16ReferenceClipAnalyzer>();
        services.TryAddSingleton<IReferenceClipTrimmer, Pcm16ReferenceClipTrimmer>();

        services.TryAddScoped<IAudioClipExtractor, Pcm16WaveClipExtractor>();
        services.TryAddScoped<IAudioTimeStretchService>(_ => new AudioTimeStretchService(ffmpegPath: null));
        services.TryAddScoped<ITtsAudioPostProcessor>(sp =>
            new TtsAudioPostProcessor(sp.GetService<IApplicationLogger>()));
        services.TryAddScoped<IPreviewRangeRenderer, PreviewRangeRenderer>();
        services.TryAddScoped<TtsTimingOptions>(sp =>
            CreateTtsTimingOptions(sp.GetRequiredService<TranscriptWorkspaceContext>().Settings.TtsTiming));
        services.TryAddScoped<TtsExecutionOptions>(sp =>
            CreateTtsExecutionOptions(
                sp.GetRequiredService<TranscriptWorkspaceContext>().Settings.TtsMaxConcurrency,
                AcceleratorVramDetector.DetectMaxDedicatedVramMb()));

        services.TryAddScoped<MixPlanBuilder>();
        services.TryAddScoped<TimelineMediaMixPreparer>();
        services.TryAddScoped<MixPlanStore>();
        services.TryAddScoped<SequenceStore>();
        services.TryAddScoped<ProjectMediaIngestService>();
        services.TryAddScoped<TranscriptProjectStateService>();
        services.TryAddScoped<VadGenerationStage>();
        services.TryAddScoped<SpeechEnhancementGenerationStage>();
        services.TryAddScoped<ITranscriptGenerationStage>(sp => sp.GetRequiredService<SpeechEnhancementGenerationStage>());
        services.TryAddScoped<SpeakerDiarizationStage>();
        services.TryAddScoped<AsrGenerationStage>();
        services.TryAddScoped<TextRefinementGenerationStage>();
        services.TryAddScoped<SpeakerAssignmentAndPersistenceStage>();
        services.TryAddScoped<TranscriptGenerationService>();
        services.TryAddScoped<SegmentEditingService>();
        services.TryAddScoped<SpeakerReferenceClipService>();
        services.TryAddScoped<SpeakerAssignmentService>();
        services.TryAddScoped<VoiceAssignmentService>();
        services.TryAddScoped<GlossaryService>();
        services.TryAddEnumerable(ServiceDescriptor.Scoped<IGlossaryLanguageAnalyzer, LuceneJapaneseGlossaryAnalyzer>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<IGlossaryLanguageAnalyzer, LuceneChineseGlossaryAnalyzer>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<IGlossaryLanguageAnalyzer, LuceneArabicGlossaryAnalyzer>());
        services.TryAddScoped<IGlossaryAnalyzerCatalog>(sp =>
            new GlossaryAnalyzerCatalog(sp.GetServices<IGlossaryLanguageAnalyzer>()));
        services.TryAddScoped<IGlossaryTermMatcher, GlossaryTermMatcher>();
        services.TryAddSingleton<IGlossaryTargetTermMatcher, GlossaryTargetTermMatcher>();
        services.TryAddScoped<ITranslatedWordAlignmentService, ProportionalTranslatedWordAlignmentService>();
        services.TryAddScoped<TranslationOrchestrationService>();
        services.TryAddScoped<TtsOrchestrationService>();
        services.TryAddScoped<TtsCandidateSelectionService>();
        services.TryAddScoped<GenerateCandidatesHandler>();
        services.TryAddScoped<TranscriptArtifactWriter>();
        services.TryAddScoped<DurationAnalysisService>();
        services.TryAddScoped(sp =>
            new VadStageHandler(
                sp.GetRequiredService<ISpeechRegionDetector>(),
                sp.GetRequiredService<IProjectStageRunStore>(),
                sp.GetService<IRuntimePlanningPreferences>(),
                sp.GetService<IApplicationLogger>()));
        services.TryAddScoped(sp =>
            new AsrStageHandler(
                sp.GetRequiredService<IAudioTranscriptionEngine>(),
                sp.GetRequiredService<IProjectStageRunStore>(),
                sp.GetService<IRuntimePlanningPreferences>(),
                sp.GetService<IApplicationLogger>()));
        services.TryAddScoped(sp =>
            new TextRefinementStageHandler(
                sp.GetRequiredService<ITextRefinementEngine>(),
                sp.GetRequiredService<IProjectStageRunStore>(),
                sp.GetService<IRuntimePlanningPreferences>(),
                sp.GetService<IApplicationLogger>()));
        services.TryAddScoped<StemSeparationStageHandler>();
        services.TryAddScoped<OverlapRegionDetector>();
        services.TryAddScoped<OverlapRescueStageHandler>();
        services.TryAddScoped<OverlapRescueCandidateTranscriptionService>();
        services.TryAddScoped<OverlapRescueWorkflow>();
        services.TryAddScoped<SpeechAudioPreparationStageHandler>();
        services.TryAddScoped<SpeechAudioEnhancementStageHandler>();
        services.TryAddScoped<StartTtsStageHandler>();
        services.TryAddScoped(sp =>
            new DiarizationStageHandler(
                sp.GetRequiredService<ISpeakerDiarizationEngine>(),
                sp.GetRequiredService<IModelDownloaderContract>(),
                sp.GetRequiredService<IApplicationLogger>(),
                sp.GetRequiredService<IModelCacheRegistrar>(),
                sp.GetRequiredService<TrackdubStoragePaths>().ModelCacheDirectory,
                modelCacheLookup: sp.GetRequiredService<IModelCacheRecordLookup>()));

        services.TryAddScoped<ProjectWorkflow>();
        services.TryAddSingleton<IRuntimeSelectionService, RuntimeSelectionService>();
        services.TryAddSingleton<IPipelinePreFlightChecker, RuntimePlannerPreFlightChecker>();
        services.TryAddSingleton<IPipelineReadinessService>(sp =>
            new PipelineReadinessService(
                sp.GetRequiredService<IRuntimePlanner>(),
                sp.GetRequiredService<ICloudApiKeyProvider>(),
                sp.GetRequiredService<IConsentService>(),
                runtimePlanningPreferences: sp.GetService<IRuntimePlanningPreferences>(),
                cloudEgressConsentService: sp.GetRequiredService<CloudEgressConsentService>(),
                modelRegistry: sp.GetRequiredService<BundledModelManifestRegistry>()));
        services.TryAddSingleton<IStageReadinessOrchestrator, StageReadinessOrchestrator>();
        services.TryAddScoped<IRuntimeModelBootstrapService, RuntimeModelBootstrapService>();
        services.TryAddScoped<RuntimeModelWorkflow>();
        services.TryAddScoped<DiarizationModelWorkflow>();
        services.TryAddScoped<TranscriptWorkflow>();
        services.TryAddScoped<TranslationWorkflow>();
        services.TryAddScoped<SpeakerWorkflow>();
        services.TryAddScoped<VoiceWorkflow>();
        services.TryAddScoped<TtsWorkflow>();
        services.TryAddScoped<LipSyncWorkflow>();
        services.TryAddScoped<LipSynthesisWorkflow>();
        services.TryAddScoped<IAudioPreviewTransport, MediaFoundationAudioPreviewTransport>();
        services.TryAddScoped<TtsDubPreviewCoordinator>();
        services.TryAddScoped<TtsDubPreviewWorkflow>();
        services.TryAddScoped<PreviewMixWorkflow>();
        // Required by ExportStageHandler on headless paths (SDK/CLI/API/Worker); the Avalonia
        // shell additionally registers it earlier so its singleton view model can consume it.
        services.TryAddSingleton<SubtitleExportService>();
        services.TryAddScoped<ExportStageHandler>();
        services.TryAddScoped<ExportWorkflow>();
        services.TryAddScoped<EditingHistoryWorkflow>();
        services.TryAddScoped<PipelineDegradationWriter>();
        services.TryAddScoped<TranscriptWorkspace>();
        services.TryAddSingleton<RuntimeModelSetupCoordinator>();
        services.TryAddSingleton<IPipelineModelSetupInteraction, HeadlessPipelineModelSetupInteraction>();
        services.TryAddSingleton<TranscriptImportModelProvisioner>();
        services.TryAddSingleton<TranscriptWorkspaceCommandService>();
        services.TryAddSingleton<VoicePreviewCache>();

        services.TryAddSingleton<TranscriptWorkspaceFactory>();
        services.TryAddSingleton<ITranscriptWorkspaceSessionFactory>(sp => sp.GetRequiredService<TranscriptWorkspaceFactory>());
        services.TryAddSingleton<ProjectSessionService>();

        // Licensing services
        services.TryAddSingleton<IHardwareFingerprintProvider, HardwareFingerprintProvider>();
        services.TryAddSingleton<LicenseService>();
        services.TryAddSingleton<ILicenseInitializer>(sp => sp.GetRequiredService<LicenseService>());
        services.TryAddSingleton<ILicenseTierProvider>(sp => sp.GetRequiredService<LicenseService>());
        services.TryAddSingleton<ILicenseTokenStore, LicenseFileStore>();
    }

    private static void AddSequenceExport(IServiceCollection services)
    {
        services.TryAddSingleton<ISequenceExportRenderer>(sp =>
            new FfmpegSequenceRenderer(
                sp.GetRequiredService<IMediaProbe>(),
                sp.GetRequiredService<IMediaGpuHintProvider>(),
                ffmpegPath: null));
        services.TryAddScoped<SequenceExportDeliveryService>();
    }

    private static void AddInference(IServiceCollection services)
    {
        services.TryAddSingleton<BundledModelManifestRegistry>(_ => LoadManifestRegistry());
        services.TryAddSingleton<IModelAliasResolver, ModelManifestAliasResolver>();
        services.TryAddSingleton<LocalModelCacheInventory>();
        services.TryAddSingleton<BundledManifestModelCacheInventory>();
        services.TryAddSingleton<IModelCacheInventory>(sp =>
            new CompositeModelCacheInventory(
                sp.GetRequiredService<LocalModelCacheInventory>(),
                sp.GetRequiredService<BundledManifestModelCacheInventory>()));
        services.TryAddSingleton<IHardwareProfileProvider, MachineHardwareProfileProvider>();
        services.TryAddSingleton<IAcceleratorBudgetInfoService, AcceleratorBudgetInfoService>();
        services.TryAddSingleton<IHardwareInfoService, HardwareInfoService>();
        services.TryAddSingleton<IMediaGpuHintProvider, MediaGpuHintProvider>();
        services.TryAddSingleton<IMediaHardwareCapabilitiesService, MediaHardwareCapabilitiesService>();
        services.TryAddSingleton<IInferenceWorkerClient>(_ => new InferenceWorkerClient());
        services.TryAddSingleton<IExecutionProviderDiscovery>(sp =>
            new OnnxExecutionProviderDiscovery(
                sp.GetRequiredService<IOpenVinoAvailabilityProvider>(),
                new LinuxNativeGpuRuntimeProbe(),
                sp.GetRequiredService<INativeCudaTensorRtWindowsPolicy>(),
                sp.GetRequiredService<IMigraphxReadinessProbe>(),
                sp.GetRequiredService<IDnnlReadinessProbe>(),
                sp.GetRequiredService<ITensorRtRtxReadinessProbe>(),
                sp.GetRequiredService<IOpenVinoCatalogReadinessProbe>(),
                sp.GetRequiredService<IQnnCatalogReadinessProbe>(),
                sp.GetRequiredService<IVitisAiCatalogReadinessProbe>(),
                async cancellationToken =>
                    (await sp.GetRequiredService<IStudioSettingsService>()
                        .LoadAsync(cancellationToken)
                        .ConfigureAwait(false))
                    .NvidiaTensorRtRtxLicenseAccepted,
                workerClient: sp.GetService<IInferenceWorkerClient>()));
        services.TryAddSingleton<IExecutionProviderSmokeTester, OnnxExecutionProviderSmokeTester>();
        services.TryAddSingleton<ISmokeVerdictStore>(sp =>
            new FileSmokeVerdictStore(Path.Join(
                sp.GetRequiredService<TrackdubStoragePaths>().UserCacheRoot,
                "smoke-verdicts.json")));
        services.TryAddSingleton<IRuntimePlanner, RuntimePlanner>();
        services.TryAddScoped<IStageWarmupCoordinator>(sp =>
            new RuntimeStageWarmupCoordinator(
                sp.GetRequiredService<IRuntimePlanner>(),
                sp.GetRequiredService<IExecutionProviderSmokeTester>(),
                sp.GetService<IRuntimePlanningPreferences>(),
                sp.GetService<IInferenceWorkerClient>(),
                allowsConcurrentWarmup: Trackdub.Inference.Onnx.Pool.GenAiModelPool.Shared.MaxEntries >= 2,
                smokeVerdictStore: sp.GetService<ISmokeVerdictStore>(),
                hardwareProfileProvider: sp.GetService<IHardwareProfileProvider>()));
        services.TryAddSingleton<IResidentWarmupService>(sp =>
            new ResidentWarmupService(
                sp.GetRequiredService<IServiceScopeFactory>(),
                sp.GetRequiredService<IExecutionProviderDiscovery>(),
                sp.GetRequiredService<IHardwareProfileProvider>()));
        // Post-install engine warming: a freshly installed model has no EP-context artifact, so
        // without this its engines rebuild from source on every launch until something warms it.
        // The queue is injected into ModelDownloadOrchestrator instead of being triggered by an
        // observer: observers require somebody to construct the coordinator, and no production
        // path did, which left an earlier revision a silent no-op. The orchestrator is always
        // constructed to download at all, so its success path is a trigger that cannot be missed.
        services.TryAddSingleton<IEngineAutoWarmQueue>(sp =>
            new Trackdub.Composition.Pipeline.EngineAutoWarmCoordinator(
                sp.GetRequiredService<IResidentWarmupService>(),
                sp.GetRequiredService<BundledModelManifestRegistry>(),
                sp.GetService<IApplicationLogger>()));
        services.TryAddSingleton<EngineAutoWarmInstaller>();
        services.AddHostedService(sp => sp.GetRequiredService<EngineAutoWarmInstaller>());
        // Wire the model cache directory explicitly: the bare-type registration would fall back
        // to the constructor default (no cache), making downloaded models invisible to alias
        // resolution (e.g. the kokoro voice catalog) even when the planner reports them ready.
        services.TryAddSingleton(sp => new BenchmarkModelPathResolver(
            sp.GetService<BundledModelManifestRegistry>(),
            sp.GetRequiredService<TrackdubStoragePaths>().ModelCacheDirectory));
        services.AddHttpClient("translation-catalog", client => client.Timeout = TimeSpan.FromSeconds(30));
        services.TryAddSingleton<ITranslationModelCatalog>(sp => new HuggingFaceTranslationModelCatalog(
            sp.GetRequiredService<IHttpClientFactory>().CreateClient("translation-catalog"),
            Path.Join(sp.GetRequiredService<TrackdubStoragePaths>().UserCacheRoot, "translation-catalog.json")));
        services.TryAddSingleton<ITranslationModelProvisioner, CatalogTranslationModelProvisioner>();
        services.TryAddSingleton<TranslationLanguageRouter>();
        services.TryAddSingleton<ITranslationLanguageRouter>(sp =>
            new CloudAwareTranslationLanguageRouter(
                sp.GetRequiredService<TranslationLanguageRouter>(),
                sp.GetRequiredService<ICloudApiKeyProvider>()));
        services.TryAddSingleton<IGraphemeToPhoneme>(_ => new EspeakNgPhonemizer());
        services.TryAddSingleton<KokoroVoiceCatalogHolder>();
        services.TryAddScoped<IVoiceCatalog>(sp =>
            new StockAndPresetVoiceCatalog(
                sp.GetRequiredService<KokoroVoiceCatalogHolder>().Current,
                Qwen3TtsVoiceCatalog.KnownAvailable()));

        services.TryAddEnumerable(ServiceDescriptor.Scoped<ISpeechRegionDetectorAdapter, SileroVadSpeechRegionDetector>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<IAudioTranscriptionEngineAdapter, WhisperGenAiAudioTranscriptionEngine>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<IAudioTranscriptionEngineAdapter, WhisperOnnxAudioTranscriptionEngine>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<IAudioTranscriptionEngineAdapter, Qwen3AsrOnnxAudioTranscriptionEngine>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<IAudioTranscriptionEngineAdapter, NemotronAsrOnnxAudioTranscriptionEngine>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<IAudioTranscriptionEngineAdapter, ParakeetTdtOnnxAudioTranscriptionEngine>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<ISpeakerDiarizationEngineAdapter, SortFormerDiarizationEngine>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<IStemSeparationEngineAdapter, SpleeterStemSeparationEngine>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<IOverlapRescueEngineAdapter, SepFormerOverlapRescueEngine>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<ITranslationEngineAdapter, OpusMtTranslationEngine>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<ITranslationEngineAdapter, MadladTranslationEngine>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<ITranslationEngineAdapter, PhiGenAiTranslationEngine>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<ITtsEngineAdapter, KokoroTtsEngine>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<ITtsEngineAdapter, ChatterboxVoiceCloneTtsEngine>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<ITtsEngineAdapter, Qwen3TtsEngine>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<ITtsEngineAdapter, CosyVoiceTtsEngine>());
        services.TryAddEnumerable(ServiceDescriptor.Scoped<ITextRefinementEngine, QwenTextRefinementEngine>());
        services.AddHttpClient<GeminiCloudTextRefinementEngine>(c => c.Timeout = TimeSpan.FromSeconds(30));
        services.TryAddEnumerable(ServiceDescriptor.Scoped<ITextRefinementEngine, GeminiCloudTextRefinementEngine>());
        services.TryAddScoped<ITextRefinementEngine>(sp =>
            new RoutedTextRefinementEngine(sp.GetServices<ITextRefinementEngine>()));
        services.TryAddScoped<ILocalAssistant, QwenLocalAssistantEngine>();
        services.AddHttpClient<DeepLCloudTranslationEngine>(c => c.Timeout = TimeSpan.FromSeconds(60));
        services.AddHttpClient<OpenAiCloudTranslationEngine>(c => c.Timeout = TimeSpan.FromSeconds(60));
        services.AddHttpClient<GeminiCloudTranslationEngine>(c => c.Timeout = TimeSpan.FromSeconds(60));
        services.AddHttpClient<OpenAiCloudTranscriptionEngine>(c => c.Timeout = TimeSpan.FromMinutes(5));
        services.AddHttpClient<GeminiCloudTranscriptionEngine>(c => c.Timeout = TimeSpan.FromMinutes(5));
        services.AddHttpClient<ElevenLabsCloudTtsEngine>(c => c.Timeout = TimeSpan.FromSeconds(30));
        services.AddHttpClient<OpenAiCloudTtsEngine>(c => c.Timeout = TimeSpan.FromSeconds(30));
        services.AddHttpClient<GoogleCloudTtsEngine>(c => c.Timeout = TimeSpan.FromSeconds(30));
        services.AddHttpClient<GeminiCloudTtsEngine>(c => c.Timeout = TimeSpan.FromSeconds(30));
        services.AddHttpClient<ElevenLabsCloudDubbingEngine>(c => c.Timeout = TimeSpan.FromMinutes(40));
        services.TryAddTransient<ICloudDubbingEngine>(sp => sp.GetRequiredService<ElevenLabsCloudDubbingEngine>());

        // Forced-alignment adapters. wav2vec2 (phoneme-capable, commercial lane) is registered
        // first; Qwen (word-level only, experimental lane) is the registration-order fallback.
        // RoutedForcedAligner additionally honors ForcedAlignmentOptions.PreferredModelAlias and
        // RequirePhonemeTimings, so lip-sync requests never land on a word-level aligner.
        // Use AddSingleton (not TryAddEnumerable) because factory lambdas share the same
        // ServiceType+ImplementationType key and TryAddEnumerable would silently drop the second.
        services.AddSingleton<IForcedAlignerAdapter>(sp =>
            new Wav2Vec2CtcForcedAligner(
                ResolveForcedAlignerModelRoot(sp, "wav2vec2-lv60-espeak-cv-ft-onnx"),
                sp.GetService<IGraphemeToPhoneme>()));
        services.AddSingleton<IForcedAlignerAdapter>(sp =>
            new QwenForcedAligner(
                ResolveForcedAlignerModelRoot(sp, "qwen3-forced-aligner-0.6b-q4-onnx"),
                sp.GetService<ILogger<QwenForcedAligner>>()));
        services.TryAddSingleton<IForcedAligner>(sp =>
            new RoutedForcedAligner(
                sp.GetServices<IForcedAlignerAdapter>(),
                sp.GetService<IApplicationLogger>()));
        services.TryAddSingleton<IPhonemeTimingPlanner, PhonemeTimingPlanner>();
        services.TryAddSingleton<IPhonemeStretchService, WsolaPhonemeStretchService>();
        services.TryAddScoped<LipSyncStageHandler>(sp =>
            new LipSyncStageHandler(
                sp.GetRequiredService<IForcedAligner>(),
                sp.GetRequiredService<IPhonemeTimingPlanner>(),
                sp.GetRequiredService<IPhonemeStretchService>(),
                sp.GetRequiredService<IArtifactStore>(),
                sp.GetRequiredService<IProjectStageRunStore>(),
                sp.GetRequiredService<ILipSyncSegmentRepository>(),
                sp.GetService<PipelineDegradationWriter>(),
                sp.GetService<IRuntimePlanningPreferences>(),
                sp.GetService<IApplicationLogger>(),
                sp.GetService<IFileFingerprintService>(),
                sp.GetService<IMediaAssetRepository>(),
                sp.GetService<IAudioClipExtractor>()));
        services.TryAddScoped<ILipSyncSegmentRepository, SqliteLipSyncSegmentRepository>();

        // M23 video lip synthesis — LatentSync 1.6 (ByteDance, openrail++, commercial-safe).
        // Engine and face-analysis providers report IsAvailable=false until their model files are
        // present; the stage skips cleanly (SkippedRuntimeUnavailable) and never blocks audio-only export.
        LatentSyncLipSynthesisRegistration.Register(services);
        services.TryAddScoped<LipSynthesisStageHandler>(sp =>
            new LipSynthesisStageHandler(
                sp.GetRequiredService<ILipSynthesisEngine>(),
                sp.GetRequiredService<IFaceDetector>(),
                sp.GetRequiredService<IFaceLandmarkProvider>(),
                sp.GetRequiredService<IFacePoseEstimator>(),
                sp.GetRequiredService<IArtifactStore>(),
                sp.GetRequiredService<IProjectStageRunStore>(),
                sp.GetRequiredService<ILipSynthesisSegmentRepository>(),
                sp.GetService<PipelineDegradationWriter>(),
                sp.GetService<IRuntimePlanningPreferences>(),
                sp.GetService<IApplicationLogger>(),
                sp.GetService<IFileFingerprintService>(),
                sp.GetService<IMediaAssetRepository>()));
        services.TryAddScoped<ILipSynthesisSegmentRepository, SqliteLipSynthesisSegmentRepository>();

        services.TryAddScoped<ISpeechRegionDetector, RoutedSpeechRegionDetector>();
        services.TryAddScoped<RoutedAudioTranscriptionEngine>();
        services.TryAddScoped<IAudioTranscriptionEngine>(sp =>
            new CloudAwareAudioTranscriptionEngine(
                sp.GetRequiredService<RoutedAudioTranscriptionEngine>(),
                sp.GetRequiredService<OpenAiCloudTranscriptionEngine>(),
                sp.GetRequiredService<GeminiCloudTranscriptionEngine>()));
        services.TryAddScoped<ISpeakerDiarizationEngine, RoutedSpeakerDiarizationEngine>();
        services.TryAddScoped<IStemSeparationEngine, RoutedStemSeparationEngine>();
        services.TryAddScoped<IOverlapRescueEngine, RoutedOverlapRescueEngine>();
        services.TryAddScoped<RoutedTranslationEngine>();
        services.TryAddScoped<ITranslationEngine>(sp =>
            new CloudAwareTranslationEngine(
                sp.GetRequiredService<RoutedTranslationEngine>(),
                sp.GetRequiredService<DeepLCloudTranslationEngine>(),
                sp.GetRequiredService<OpenAiCloudTranslationEngine>(),
                sp.GetRequiredService<GeminiCloudTranslationEngine>()));
        services.TryAddScoped<RoutedTtsEngine>();
        services.TryAddScoped<ITtsAcceleratorPlacementResolver>(sp => sp.GetRequiredService<RoutedTtsEngine>());
        services.TryAddScoped<ITtsEngine>(sp =>
            new CloudAwareTtsEngine(
                sp.GetRequiredService<RoutedTtsEngine>(),
                sp.GetRequiredService<ElevenLabsCloudTtsEngine>(),
                sp.GetRequiredService<OpenAiCloudTtsEngine>(),
                sp.GetRequiredService<GoogleCloudTtsEngine>(),
                sp.GetRequiredService<GeminiCloudTtsEngine>()));
    }

    /// <summary>
    /// Resolves the on-disk root directory for a forced-alignment model. When the model is
    /// not downloaded yet, returns the expected model-cache root instead of an empty path so
    /// the adapter's live File.Exists checks flip IsAvailable as soon as the Model Manager
    /// finishes a download — without requiring an app restart.
    /// </summary>
    private static string ResolveForcedAlignerModelRoot(IServiceProvider sp, string modelId)
    {
        BenchmarkModelPathResolver resolver = sp.GetRequiredService<BenchmarkModelPathResolver>();
        try
        {
            BenchmarkModelCandidate candidate = resolver.ResolveSingle(modelId);
            return candidate.RootDirectory
                ?? System.IO.Path.GetDirectoryName(candidate.ModelPath)
                ?? string.Empty;
        }
        catch (FileNotFoundException)
        {
            return System.IO.Path.Join(
                sp.GetRequiredService<TrackdubStoragePaths>().ModelCacheDirectory,
                modelId);
        }
    }

    private static TtsTimingOptions CreateTtsTimingOptions(TtsTimingSettings? settings)
    {
        TtsTimingSettings normalized = settings ?? TtsTimingSettings.Default;
        return TtsTimingOptions.Default with
        {
            EnableRubberbandStretch = normalized.EnableRubberbandStretch,
            RubberbandStretchThreshold = normalized.RubberbandStretchThreshold,
            AutoStretchMaxOverrun = normalized.AutoStretchMaxOverrun ?? TtsTimingOptions.Default.AutoStretchMaxOverrun
        };
    }

    private static TtsExecutionOptions CreateTtsExecutionOptions(
        int? configuredMaxConcurrency,
        long maxAcceleratorVramMb) =>
        new(
            ConfiguredMaxConcurrency: configuredMaxConcurrency is int value && value >= TtsExecutionOptions.MinConcurrency
                ? Math.Min(value, TtsExecutionOptions.AbsoluteMaxConcurrency)
                : null,
            MaxAcceleratorVramMb: maxAcceleratorVramMb);

    [SupportedOSPlatform("linux")]
    private static IExecutionProviderBootstrapper CreateLinuxExecutionProviderBootstrapper(IServiceProvider sp) =>
        new Trackdub.Inference.Onnx.ExecutionProviders.Linux.LinuxExecutionProviderBootstrapper(
            sp.GetRequiredService<IOpenVinoAvailabilityProvider>(),
            new LinuxNativeGpuRuntimeProbe(),
            sp.GetRequiredService<ITensorRtRtxProviderBootstrap>());

    private static void AddHardwareRouting(IServiceCollection services)
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

        // IExecutionProviderBootstrapper — platform-specific EP bootstrap logic, singleton.
        // Also calls OnnxExecutionSessionFactory.Initialize() so the static factory uses the
        // DI-wired bootstrapper (with real IOpenVinoAvailabilityProvider on Linux).
        services.TryAddSingleton<IExecutionProviderBootstrapper>(sp =>
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
        });

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

    private static BundledModelManifestRegistry LoadManifestRegistry()
    {
        if (BundledModelManifestRegistry.TryLoadDefault(out BundledModelManifestRegistry? registry, out string? error) &&
            registry is not null)
        {
            return registry;
        }

        throw new InvalidOperationException(error ?? "Bundled model manifest was not found.");
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

    private static OpenVinoComponentSettings CreateOpenVinoComponentSettings(IApplicationLogger logger)
    {
        if (!IsOpenVinoOverrideEnabled())
        {
            return new OpenVinoComponentSettings
            {
                DownloadUrl = OpenVinoComponentDefaults.DownloadUrl,
                ExpectedSha256Hash = OpenVinoComponentDefaults.ExpectedSha256Hash,
                ExpectedFileSizeBytes = OpenVinoComponentDefaults.ExpectedFileSizeBytes
            };
        }

        string? overrideUrl = TryGetValidatedOpenVinoDownloadUrl(Environment.GetEnvironmentVariable("TRACKDUB_OPENVINO_DOWNLOAD_URL"));
        if (!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TRACKDUB_OPENVINO_DOWNLOAD_URL")) && overrideUrl is null)
            logger.LogWarning("Ignoring TRACKDUB_OPENVINO_DOWNLOAD_URL because it is not a valid absolute HTTPS URL.");

        string? overrideSha256 = TryGetValidatedOpenVinoSha256(Environment.GetEnvironmentVariable("TRACKDUB_OPENVINO_SHA256"));
        if (!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TRACKDUB_OPENVINO_SHA256")) && overrideSha256 is null)
            logger.LogWarning("Ignoring TRACKDUB_OPENVINO_SHA256 because it is not a valid SHA-256 hex value.");

        long? overrideFileSize = TryGetEnvironmentLong("TRACKDUB_OPENVINO_FILE_SIZE_BYTES");
        if (!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TRACKDUB_OPENVINO_FILE_SIZE_BYTES")) && !(overrideFileSize.HasValue && overrideFileSize.Value > 0))
            logger.LogWarning("Ignoring TRACKDUB_OPENVINO_FILE_SIZE_BYTES because it is not a positive integer.");

        return new OpenVinoComponentSettings
        {
            DownloadUrl = overrideUrl ?? OpenVinoComponentDefaults.DownloadUrl,
            ExpectedSha256Hash = overrideSha256 ?? OpenVinoComponentDefaults.ExpectedSha256Hash,
            ExpectedFileSizeBytes = overrideFileSize is { } v && v > 0 ? v : OpenVinoComponentDefaults.ExpectedFileSizeBytes
        };
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
