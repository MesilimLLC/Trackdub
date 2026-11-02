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
    private static void AddInfrastructure(IServiceCollection services)
    {
        AddStorageAndHostServices(services);
        AddReadinessProbes(services);
        AddApplicationLogging(services);
        AddStarterPackServices(services);
        AddModelDownloadServices(services);
        AddSettingsAndUpdateClients(services);
        AddTensorRtRtxServices(services);
        AddCloudConsentServices(services);
        AddHealthAndModelDownloaderServices(services);
        AddPlaybackAndPersistenceServices(services);
        AddOptimizationAndUpdateServices(services);
    }

    private static void AddStorageAndHostServices(IServiceCollection services)
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
    }

    private static void AddReadinessProbes(IServiceCollection services)
    {
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
    }

    private static void AddApplicationLogging(IServiceCollection services)
    {
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
    }

    private static void AddStarterPackServices(IServiceCollection services)
    {
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
    }

    private static void AddModelDownloadServices(IServiceCollection services)
    {
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
    }

    private static void AddSettingsAndUpdateClients(IServiceCollection services)
    {
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
    }

    private static void AddTensorRtRtxServices(IServiceCollection services)
    {
        services.TryAddSingleton<ITensorRtRtxProviderBootstrap>(CreateTensorRtRtxProviderBootstrap);
        services.TryAddSingleton<ITensorRtRtxReadinessProbe>(sp =>
            new CachingTensorRtRtxReadinessProbe(
                new TensorRtRtxReadinessProbe(sp.GetRequiredService<ITensorRtRtxProviderBootstrap>())));
        services.TryAddSingleton<ITensorRtRtxRuntimeReadinessService, TensorRtRtxRuntimeReadinessService>();
    }

    private static ITensorRtRtxProviderBootstrap CreateTensorRtRtxProviderBootstrap(IServiceProvider sp)
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
    }

    private static void AddCloudConsentServices(IServiceCollection services)
    {
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
    }

    private static void AddHealthAndModelDownloaderServices(IServiceCollection services)
    {
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
    }

    private static void AddPlaybackAndPersistenceServices(IServiceCollection services)
    {
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
    }

    private static void AddOptimizationAndUpdateServices(IServiceCollection services)
    {
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
}
