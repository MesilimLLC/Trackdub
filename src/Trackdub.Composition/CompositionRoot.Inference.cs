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
    private static void AddInference(IServiceCollection services)
    {
        AddModelInventory(services);
        AddHardwareAndDiscovery(services);
        AddRuntimePlanning(services);
        AddTranslationAndVoiceServices(services);
        AddEngineAdapters(services);
        AddCloudEngines(services);
        AddForcedAlignmentServices(services);
        AddLipSynthesisServices(services);
        AddRoutedEngines(services);
    }

    private static void AddModelInventory(IServiceCollection services)
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

    private static void AddHardwareAndDiscovery(IServiceCollection services)
    {
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
    }

    private static void AddRuntimePlanning(IServiceCollection services)
    {
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
    }

    private static void AddTranslationAndVoiceServices(IServiceCollection services)
    {
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
    }

    private static void AddEngineAdapters(IServiceCollection services)
    {
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
        services.TryAddScoped<QwenTextRefinementEngine>();
    }

    private static void AddCloudEngines(IServiceCollection services)
    {
        services.AddHttpClient<GeminiCloudTextRefinementEngine>(c => c.Timeout = TimeSpan.FromSeconds(30));
        services.TryAddScoped<ITextRefinementEngine>(sp =>
            new RoutedTextRefinementEngine([
                sp.GetRequiredService<QwenTextRefinementEngine>(),
                sp.GetRequiredService<GeminiCloudTextRefinementEngine>()]));
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
    }

    private static void AddForcedAlignmentServices(IServiceCollection services)
    {
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
    }

    private static void AddLipSynthesisServices(IServiceCollection services)
    {
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
    }

    private static void AddRoutedEngines(IServiceCollection services)
    {
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
}
