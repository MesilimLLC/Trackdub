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
    private static void AddApplication(IServiceCollection services)
    {
        AddMediaServices(services);
        AddSpeechEnhancementServices(services);
        AddTtsAudioServices(services);
        AddTranscriptAndGlossaryServices(services);
        AddStageHandlers(services);
        AddPipelineWorkflows(services);
        AddExportAndWorkspaceServices(services);
        AddLicensingServices(services);
    }

    private static void AddMediaServices(IServiceCollection services)
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
    }

    private static void AddSpeechEnhancementServices(IServiceCollection services)
    {
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
    }

    private static void AddTtsAudioServices(IServiceCollection services)
    {
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
    }

    private static void AddTranscriptAndGlossaryServices(IServiceCollection services)
    {
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
    }

    private static void AddStageHandlers(IServiceCollection services)
    {
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
    }

    private static void AddPipelineWorkflows(IServiceCollection services)
    {
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
    }

    private static void AddExportAndWorkspaceServices(IServiceCollection services)
    {
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
    }

    private static void AddLicensingServices(IServiceCollection services)
    {
        // Licensing services
        services.TryAddSingleton<IHardwareFingerprintProvider, HardwareFingerprintProvider>();
        services.TryAddSingleton<LicenseService>();
        services.TryAddSingleton<ILicenseInitializer>(sp => sp.GetRequiredService<LicenseService>());
        services.TryAddSingleton<ILicenseTierProvider>(sp => sp.GetRequiredService<LicenseService>());
        services.TryAddSingleton<ILicenseTokenStore, LicenseFileStore>();
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
}
