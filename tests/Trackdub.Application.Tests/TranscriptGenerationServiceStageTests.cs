using Trackdub.Application.Transcripts;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Application.Transcripts.Stages;
using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.AudioQuality;
using Trackdub.Domain.Media;
using Trackdub.Domain.Projects;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed class TranscriptGenerationServiceStageTests : IDisposable
{
    private readonly string tempRoot = Path.Join(Path.GetTempPath(), $"trackdub-stage-tests-{Guid.NewGuid():N}");

    [Theory]
    [InlineData("unknown-stage")]
    [InlineData(StageNames.Translation)]
    public async Task GenerateTranscriptStageAsync_throws_for_unsupported_stage(string stageName)
    {
        (TranscriptGenerationService service, _) = CreateMinimalService();
        TranscriptGenerationContext context = CreateContext();

        await Assert.ThrowsAsync<ArgumentException>(() =>
            service.GenerateTranscriptStageAsync(
                context.Project,
                context.MediaAsset,
                context.NormalizedAudioArtifact,
                context.AudioRoutingPlan,
                stageName,
                enableSpeakerDiarization: false,
                InferenceModelPreferences.Empty,
                TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task GenerateTranscriptStageAsync_asr_generates_vad_when_artifact_is_missing()
    {
        var vadDetector = new FakeSpeechRegionDetector();
        vadDetector.SetRegions(new SpeechRegion(0, 0.0d, 1.0d));
        var transcriptionEngine = new CapturingTranscriptionEngine();
        (TranscriptGenerationService service, _) = CreateMinimalService(
            transcriptionEngine,
            vadDetector: vadDetector,
            artifactStore: CreateArtifactStoreWithAudio());
        TranscriptGenerationContext context = CreateContext();

        await service.GenerateTranscriptStageAsync(
            context.Project,
            context.MediaAsset,
            context.NormalizedAudioArtifact,
            context.AudioRoutingPlan,
            StageNames.Asr,
            enableSpeakerDiarization: false,
            InferenceModelPreferences.Empty,
            TestContext.Current.CancellationToken);

        Assert.Equal(1, vadDetector.DetectCallCount);
        Assert.Equal(0d, Assert.Single(transcriptionEngine.ReceivedRegions).StartSeconds);
    }

    [Fact]
    public async Task GenerateTranscriptStageAsync_asr_regenerates_vad_regions_from_another_audio_source()
    {
        var vadDetector = new FakeSpeechRegionDetector();
        vadDetector.SetRegions(new SpeechRegion(0, 0.0d, 1.0d));
        var transcriptionEngine = new CapturingTranscriptionEngine();
        (TranscriptGenerationService service, TranscriptArtifactWriter artifactWriter) = CreateMinimalService(
            transcriptionEngine,
            vadDetector: vadDetector,
            artifactStore: CreateArtifactStoreWithAudio());
        TranscriptGenerationContext context = CreateContext();
        await artifactWriter.WriteSpeechRegionsArtifactAsync(
            new SpeechRegionsArtifactWriteRequest(
                context.Project,
                context.MediaAsset,
                [new SpeechRegion(0, 0.25d, 0.75d)],
                StageRunRecord.Start(context.Project.Id, StageNames.Vad, DateTimeOffset.UtcNow)
                    .Complete(DateTimeOffset.UtcNow),
                context.AudioRoutingPlan.VadAudioArtifact with { Id = Guid.NewGuid() }),
            TestContext.Current.CancellationToken);

        await service.GenerateTranscriptStageAsync(
            context.Project,
            context.MediaAsset,
            context.NormalizedAudioArtifact,
            context.AudioRoutingPlan,
            StageNames.Asr,
            enableSpeakerDiarization: false,
            InferenceModelPreferences.Empty,
            TestContext.Current.CancellationToken);

        Assert.Equal(1, vadDetector.DetectCallCount);
        Assert.Equal(0d, Assert.Single(transcriptionEngine.ReceivedRegions).StartSeconds);
    }

    [Fact]
    public async Task GenerateTranscriptStageAsync_vad_reruns_without_reading_corrupt_cached_regions()
    {
        var detector = new FakeSpeechRegionDetector();
        detector.SetRegions(new SpeechRegion(0, 0.0d, 1.0d));
        FakeArtifactStore artifactStore = CreateArtifactStoreWithAudio();
        (TranscriptGenerationService service, TranscriptArtifactWriter artifactWriter) = CreateMinimalService(
            vadDetector: detector,
            artifactStore: artifactStore);
        TranscriptGenerationContext context = CreateContext();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        StageRunRecord priorVadRun = StageRunRecord
            .Start(context.Project.Id, StageNames.Vad, now)
            .Complete(now);
        await artifactWriter.WriteSpeechRegionsArtifactAsync(
            new SpeechRegionsArtifactWriteRequest(
                context.Project,
                context.MediaAsset,
                [new SpeechRegion(0, 0.0d, 0.5d)],
                priorVadRun,
                context.AudioRoutingPlan.VadAudioArtifact),
            TestContext.Current.CancellationToken);
        artifactStore.Seed(ProjectArtifactPaths.GetSpeechRegionsRelativePath(priorVadRun.Id), [0xFF, 0x00]);

        await service.GenerateTranscriptStageAsync(
            context.Project,
            context.MediaAsset,
            context.NormalizedAudioArtifact,
            context.AudioRoutingPlan,
            StageNames.Vad,
            enableSpeakerDiarization: false,
            InferenceModelPreferences.Empty,
            TestContext.Current.CancellationToken);

        Assert.Equal(1, detector.DetectCallCount);
    }

    [Fact]
    public async Task GenerateTranscriptStageAsync_asr_reruns_vad_when_latest_successful_run_has_no_snapshot()
    {
        var detector = new FakeSpeechRegionDetector();
        detector.SetRegions(new SpeechRegion(0, 0.6d, 0.9d));
        var transcriptionEngine = new CapturingTranscriptionEngine();
        var stageRunStore = new FakeProjectStageRunStore();
        FakeArtifactStore artifactStore = CreateArtifactStoreWithAudio();
        (TranscriptGenerationService service, TranscriptArtifactWriter artifactWriter) = CreateMinimalService(
            transcriptionEngine,
            stageRunStore: stageRunStore,
            vadDetector: detector,
            artifactStore: artifactStore);
        TranscriptGenerationContext context = CreateContext();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        StageRunRecord previousVadRun = StageRunRecord
            .Start(context.Project.Id, StageNames.Vad, now.AddSeconds(-2))
            .Complete(now.AddSeconds(-2));
        StageRunRecord latestVadRun = StageRunRecord
            .Start(context.Project.Id, StageNames.Vad, now.AddSeconds(-1))
            .Complete(now.AddSeconds(-1));
        await stageRunStore.CreateAsync(previousVadRun, TestContext.Current.CancellationToken);
        await stageRunStore.CreateAsync(latestVadRun, TestContext.Current.CancellationToken);
        await artifactWriter.WriteSpeechRegionsArtifactAsync(
            new SpeechRegionsArtifactWriteRequest(
                context.Project,
                context.MediaAsset,
                [new SpeechRegion(0, 0.0d, 0.4d)],
                previousVadRun,
                context.AudioRoutingPlan.VadAudioArtifact),
            TestContext.Current.CancellationToken);

        await service.GenerateTranscriptStageAsync(
            context.Project,
            context.MediaAsset,
            context.NormalizedAudioArtifact,
            context.AudioRoutingPlan,
            StageNames.Asr,
            enableSpeakerDiarization: false,
            InferenceModelPreferences.Empty,
            TestContext.Current.CancellationToken);

        Assert.Equal(1, detector.DetectCallCount);
        Assert.Equal(0.6d, Assert.Single(transcriptionEngine.ReceivedRegions).StartSeconds);
    }

    [Fact]
    public async Task GenerateTranscriptStageAsync_asr_ignores_diarization_without_current_vad_lineage()
    {
        var transcriptionEngine = new CapturingTranscriptionEngine();
        FakeArtifactStore artifactStore = CreateArtifactStoreWithAudio();
        var stageRunStore = new FakeProjectStageRunStore();
        (TranscriptGenerationService service, TranscriptArtifactWriter artifactWriter) = CreateMinimalService(
            transcriptionEngine,
            stageRunStore: stageRunStore,
            artifactStore: artifactStore);
        TranscriptGenerationContext context = CreateContext();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        StageRunRecord vadRun = StageRunRecord
            .Start(context.Project.Id, StageNames.Vad, now)
            .Complete(now);
        await stageRunStore.CreateAsync(vadRun, TestContext.Current.CancellationToken);
        await artifactWriter.WriteSpeechRegionsArtifactAsync(
            new SpeechRegionsArtifactWriteRequest(
                context.Project,
                context.MediaAsset,
                [new SpeechRegion(0, 0.1d, 0.3d)],
                vadRun,
                context.AudioRoutingPlan.VadAudioArtifact),
            TestContext.Current.CancellationToken);
        ProjectSpeaker staleSpeaker = ProjectSpeaker.Create(context.Project.Id, "Stale speaker", now);
        SpeakerTurn staleTurn = SpeakerTurn.Create(
            context.Project.Id,
            staleSpeaker.Id,
            startSeconds: 0.6d,
            endSeconds: 0.9d);
        await artifactWriter.WriteDiarizationArtifactAsync(
            new DiarizationArtifactWriteRequest
            {
                ProjectId = context.Project.Id,
                MediaAsset = context.MediaAsset,
                Speakers = [staleSpeaker],
                Turns = [staleTurn],
                StageRunId = Guid.NewGuid()
            },
            TestContext.Current.CancellationToken);

        await service.GenerateTranscriptStageAsync(
            context.Project,
            context.MediaAsset,
            context.NormalizedAudioArtifact,
            context.AudioRoutingPlan,
            StageNames.Asr,
            enableSpeakerDiarization: false,
            InferenceModelPreferences.Empty,
            TestContext.Current.CancellationToken);

        Assert.Equal(0.1d, Assert.Single(transcriptionEngine.ReceivedRegions).StartSeconds);
    }

    [Fact]
    public async Task GenerateTranscriptStageAsync_text_refinement_rejects_raw_asr_from_another_audio_route()
    {
        FakeArtifactStore artifactStore = CreateArtifactStoreWithAudio();
        (TranscriptGenerationService service, TranscriptArtifactWriter artifactWriter) = CreateMinimalService(
            artifactStore: artifactStore);
        TranscriptGenerationContext context = CreateContext();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        StageRunRecord vadRun = StageRunRecord
            .Start(context.Project.Id, StageNames.Vad, now)
            .Complete(now);
        await artifactWriter.WriteSpeechRegionsArtifactAsync(
            new SpeechRegionsArtifactWriteRequest(
                context.Project,
                context.MediaAsset,
                [new SpeechRegion(0, 0.1d, 0.3d)],
                vadRun,
                context.AudioRoutingPlan.VadAudioArtifact),
            TestContext.Current.CancellationToken);
        var staleAsrRun = StageRunRecord.Start(context.Project.Id, StageNames.Asr, now.AddSeconds(1)).Complete(now.AddSeconds(1));
        await artifactWriter.WriteRawAsrTranscriptArtifactAsync(
            new RawAsrTranscriptArtifactWriteRequest
            {
                ProjectId = context.Project.Id,
                MediaAsset = context.MediaAsset,
                Segments = [new RecognizedTranscriptSegment(0, 0.1d, 0.3d, "stale audio", "en")],
                StageRunId = staleAsrRun.Id,
                Lineage = RawAsrTranscriptArtifactLineage.Create(
                    context.AudioRoutingPlan.AsrAudioArtifact with { Id = Guid.NewGuid() },
                    context.AudioRoutingPlan.VadAudioArtifact,
                    vadRun.Id)
            },
            TestContext.Current.CancellationToken);

        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            service.GenerateTranscriptStageAsync(
                context.Project,
                context.MediaAsset,
                context.NormalizedAudioArtifact,
                context.AudioRoutingPlan,
                StageNames.TextRefinementAsr,
                enableSpeakerDiarization: false,
                new InferenceModelPreferences(EnableAsrTextRefinement: true),
                TestContext.Current.CancellationToken));

        Assert.Contains("Run the ASR stage first", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task GenerateTranscriptStageAsync_disabled_text_refinement_does_not_prepare_vad()
    {
        var detector = new FakeSpeechRegionDetector();
        var stageRunStore = new FakeProjectStageRunStore();
        FakeArtifactStore artifactStore = CreateArtifactStoreWithAudio();
        (TranscriptGenerationService service, _) = CreateMinimalService(
            stageRunStore: stageRunStore,
            vadDetector: detector,
            artifactStore: artifactStore);
        TranscriptGenerationContext context = CreateContext();

        await service.GenerateTranscriptStageAsync(
            context.Project,
            context.MediaAsset,
            context.NormalizedAudioArtifact,
            context.AudioRoutingPlan,
            StageNames.TextRefinementAsr,
            enableSpeakerDiarization: false,
            InferenceModelPreferences.Empty,
            TestContext.Current.CancellationToken);

        Assert.Equal(0, detector.DetectCallCount);
        Assert.Empty(stageRunStore.All);
    }

    [Fact]
    public async Task GenerateTranscriptStageAsync_asr_short_audio_fallback_ignores_existing_diarization_plan()
    {
        var transcriptionEngine = new SegmentReturningTranscriptionEngine();
        var stageRunStore = new FakeProjectStageRunStore();
        (TranscriptGenerationService service, TranscriptArtifactWriter artifactWriter) = CreateMinimalService(
            transcriptionEngine,
            stageRunStore: stageRunStore);
        TranscriptGenerationContext context = CreateContext();
        Guid emptyVadRunId = await SeedEmptyVadAndStaleDiarizationAsync(
            context,
            artifactWriter,
            stageRunStore,
            TestContext.Current.CancellationToken);

        await service.GenerateTranscriptStageAsync(
            context.Project,
            context.MediaAsset,
            context.NormalizedAudioArtifact,
            context.AudioRoutingPlan,
            StageNames.Asr,
            enableSpeakerDiarization: false,
            InferenceModelPreferences.Empty,
            TestContext.Current.CancellationToken);

        SpeechRegion fallbackRegion = Assert.Single(transcriptionEngine.ReceivedRegions);
        Assert.Equal(0, fallbackRegion.Index);
        Assert.Equal(0d, fallbackRegion.StartSeconds);
        Assert.Equal(context.MediaAsset.DurationSeconds, fallbackRegion.EndSeconds);

        await AssertRawAsrLineageUsesShortAudioFallbackAsync(
            context,
            artifactWriter,
            emptyVadRunId,
            TestContext.Current.CancellationToken);
    }

    private static async Task<Guid> SeedEmptyVadAndStaleDiarizationAsync(
        TranscriptGenerationContext context,
        TranscriptArtifactWriter artifactWriter,
        FakeProjectStageRunStore stageRunStore,
        CancellationToken cancellationToken)
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        StageRunRecord emptyVadRun = StageRunRecord
            .Start(context.Project.Id, StageNames.Vad, now)
            .Complete(now);
        await stageRunStore.CreateAsync(emptyVadRun, cancellationToken);
        await artifactWriter.WriteSpeechRegionsArtifactAsync(
            new SpeechRegionsArtifactWriteRequest(
                context.Project,
                context.MediaAsset,
                [],
                emptyVadRun,
                context.AudioRoutingPlan.VadAudioArtifact),
            cancellationToken);

        ProjectSpeaker speaker = ProjectSpeaker.Create(context.Project.Id, "Existing speaker", now);
        SpeakerTurn staleTurn = SpeakerTurn.Create(
            context.Project.Id,
            speaker.Id,
            startSeconds: 0.25d,
            endSeconds: 0.75d);
        await artifactWriter.WriteDiarizationArtifactAsync(
            new DiarizationArtifactWriteRequest
            {
                ProjectId = context.Project.Id,
                MediaAsset = context.MediaAsset,
                Speakers = [speaker],
                Turns = [staleTurn],
                StageRunId = Guid.NewGuid()
            },
            cancellationToken);
        return emptyVadRun.Id;
    }

    private static async Task AssertRawAsrLineageUsesShortAudioFallbackAsync(
        TranscriptGenerationContext context,
        TranscriptArtifactWriter artifactWriter,
        Guid emptyVadRunId,
        CancellationToken cancellationToken)
    {
        RawAsrTranscriptArtifactLineage fallbackLineage = RawAsrTranscriptArtifactLineage.Create(
            context.AudioRoutingPlan.AsrAudioArtifact,
            context.AudioRoutingPlan.VadAudioArtifact,
            null);
        RawAsrTranscriptArtifact? fallbackTranscript = await artifactWriter.TryReadRawAsrTranscriptAsync(
            new RawAsrTranscriptArtifactReadRequest
            {
                ProjectId = context.Project.Id,
                Lineage = fallbackLineage
            },
            cancellationToken);
        Assert.NotNull(fallbackTranscript);

        RawAsrTranscriptArtifact? staleVadTranscript = await artifactWriter.TryReadRawAsrTranscriptAsync(
            new RawAsrTranscriptArtifactReadRequest
            {
                ProjectId = context.Project.Id,
                Lineage = fallbackLineage with { VadStageRunId = emptyVadRunId }
            },
            cancellationToken);
        Assert.Null(staleVadTranscript);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task GenerateTranscriptStageAsync_asr_polish_stage_run_reflects_flag(bool enableAsrTextRefinement)
    {
        var stageRunStore = new FakeProjectStageRunStore();
        var transcriptionEngine = new SegmentReturningTranscriptionEngine();
        (TranscriptGenerationService service, TranscriptArtifactWriter localArtifactWriter) =
            CreateMinimalService(transcriptionEngine: transcriptionEngine, stageRunStore: stageRunStore);
        TranscriptGenerationContext context = CreateContext();
        StageRunRecord vadRun = StageRunRecord
            .Start(context.Project.Id, StageNames.Vad, DateTimeOffset.UtcNow)
            .Complete(DateTimeOffset.UtcNow);
        await stageRunStore.CreateAsync(vadRun, TestContext.Current.CancellationToken);

        await localArtifactWriter.WriteSpeechRegionsArtifactAsync(
            new SpeechRegionsArtifactWriteRequest(
                context.Project,
                context.MediaAsset,
                [new SpeechRegion(0, 0.0d, 1.0d)],
                vadRun,
                context.AudioRoutingPlan.VadAudioArtifact),
            TestContext.Current.CancellationToken);

        await service.GenerateTranscriptStageAsync(
            context.Project,
            context.MediaAsset,
            context.NormalizedAudioArtifact,
            context.AudioRoutingPlan,
            StageNames.Asr,
            enableSpeakerDiarization: false,
            new InferenceModelPreferences(EnableAsrTextRefinement: enableAsrTextRefinement),
            TestContext.Current.CancellationToken);

        StageRunRecord? polishRun = stageRunStore.All
            .FirstOrDefault(run => string.Equals(run.StageName, StageNames.TextRefinementAsr, StringComparison.OrdinalIgnoreCase));
        if (enableAsrTextRefinement)
        {
            Assert.NotNull(polishRun);
            Assert.Equal(StageRunStatus.Completed, polishRun.Status);
        }
        else
        {
            Assert.Null(polishRun);
        }
    }

    private (TranscriptGenerationService Service, TranscriptArtifactWriter ArtifactWriter) CreateMinimalService(
        IAudioTranscriptionEngine? transcriptionEngine = null,
        FakeProjectStageRunStore? stageRunStore = null,
        FakeSpeechRegionDetector? vadDetector = null,
        FakeArtifactStore? artifactStore = null)
    {
        artifactStore ??= new FakeArtifactStore();
        var mediaRepository = new FakeMediaAssetRepository();
        stageRunStore ??= new FakeProjectStageRunStore();
        var artifactWriter = new TranscriptArtifactWriter(
            artifactStore,
            new FakeFileFingerprintService(new FileFingerprint("hash", 1, DateTimeOffset.UtcNow)),
            mediaRepository);
        var transcriptRepository = new FakeTranscriptRepository();
        var asrHandler = new AsrStageHandler(transcriptionEngine ?? new StubAudioTranscriptionEngine(), stageRunStore);
        var speakerAssignmentService = new SpeakerAssignmentService(
            new FakeSpeakerRepository(),
            transcriptRepository,
            new SegmentEditingService(transcriptRepository, new FakeTtsTakeRepository(), artifactWriter),
            artifactStore,
            stageRunStore,
            new FakeDiarizationEngine(),
            new SpeakerReferenceClipService(
                artifactStore,
                new StubAudioClipExtractor(),
                new FakeFileFingerprintService(new FileFingerprint("hash", 1, DateTimeOffset.UtcNow)),
                mediaRepository,
                new FakeVoiceAssignmentRepository(),
                new FakeTtsTakeRepository(),
                new FakeReferenceClipAnalyzer(),
                new FakeReferenceClipTrimmer()),
            artifactWriter,
            new DiarizationStageHandler(
                new FakeDiarizationEngine(),
                new StubModelDownloader()));

        return (new TranscriptGenerationService(
            transcriptRepository,
            artifactStore,
            asrHandler,
            artifactWriter,
            new VadGenerationStage(
                new VadStageHandler(vadDetector ?? new FakeSpeechRegionDetector(), stageRunStore),
                artifactWriter,
                artifactStore),
            new FakeEnhancementStage(),
            new SpeakerDiarizationStage(
                speakerAssignmentService,
                artifactWriter,
                artifactStore,
                stageRunStore),
            new AsrGenerationStage(asrHandler, artifactStore, stageRunStore),
            new TextRefinementGenerationStage(
                new TextRefinementStageHandler(new FakeTextRefinementEngine(), stageRunStore),
                stageRunStore),
            new SpeakerAssignmentAndPersistenceStage(
                speakerAssignmentService,
                transcriptRepository,
                artifactWriter,
                artifactStore,
                stageRunStore),
            new FakePipelinePreFlightChecker(),
            stageRunStore,
            mediaRepository,
            new FakeSpeakerRepository()), artifactWriter);
    }

    private FakeArtifactStore CreateArtifactStoreWithAudio()
    {
        var artifactStore = new FakeArtifactStore(tempRoot);
        artifactStore.Seed("artifacts/audio.wav", FakeWavHelper.MinimalPcm16(durationSeconds: 1.0));
        return artifactStore;
    }

    public void Dispose()
    {
        if (Directory.Exists(tempRoot))
        {
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    private static TranscriptGenerationContext CreateContext()
    {
        Guid projectId = Guid.NewGuid();
        Guid mediaAssetId = Guid.NewGuid();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        var project = new TrackdubProject(projectId, "Demo", now, now);
        var mediaAsset = new MediaAsset(
            mediaAssetId,
            projectId,
            "source.mp4",
            "source.mp4",
            "source-hash",
            100,
            now,
            "mp4",
            1.0d,
            HasAudio: true,
            HasVideo: true,
            now);
        var audioArtifact = new ProjectArtifact(
            Guid.NewGuid(),
            projectId,
            mediaAssetId,
            ArtifactKind.NormalizedAudio,
            "artifacts/audio.wav",
            "audio-hash",
            100,
            1.0d,
            16000,
            1,
            now);

        return new TranscriptGenerationContext(
            project,
            mediaAsset,
            audioArtifact,
            TranscriptAudioRoutingPlan.Raw(audioArtifact, SpeechAudioSourceKind.FullMix),
            enableSpeakerDiarization: false,
            sourceLanguage: "en");
    }

    private sealed class StubAudioTranscriptionEngine : IAudioTranscriptionEngine
    {
        public Task<IReadOnlyList<RecognizedTranscriptSegment>> TranscribeAsync(
            string normalizedAudioPath,
            IReadOnlyList<SpeechRegion> regions,
            CancellationToken cancellationToken) =>
            Task.FromResult<IReadOnlyList<RecognizedTranscriptSegment>>([]);
    }

    private sealed class CapturingTranscriptionEngine : IAudioTranscriptionEngine
    {
        public IReadOnlyList<SpeechRegion> ReceivedRegions { get; private set; } = [];

        public Task<IReadOnlyList<RecognizedTranscriptSegment>> TranscribeAsync(
            string normalizedAudioPath,
            IReadOnlyList<SpeechRegion> regions,
            CancellationToken cancellationToken)
        {
            ReceivedRegions = [.. regions];
            return Task.FromResult<IReadOnlyList<RecognizedTranscriptSegment>>([]);
        }
    }

    private sealed class SegmentReturningTranscriptionEngine : IAudioTranscriptionEngine
    {
        public IReadOnlyList<SpeechRegion> ReceivedRegions { get; private set; } = [];

        public Task<IReadOnlyList<RecognizedTranscriptSegment>> TranscribeAsync(
            string normalizedAudioPath,
            IReadOnlyList<SpeechRegion> regions,
            CancellationToken cancellationToken)
        {
            ReceivedRegions = [.. regions];
            SpeechRegion region = regions.Single();
            return Task.FromResult<IReadOnlyList<RecognizedTranscriptSegment>>(
            [
                new RecognizedTranscriptSegment(
                    region.Index,
                    region.StartSeconds,
                    region.EndSeconds,
                    "Short audio fallback transcript.",
                    "en",
                    [])
            ]);
        }
    }

    private sealed class StubAudioClipExtractor : IAudioClipExtractor
    {
        public Task<AudioClipExtractionResult> ExtractAsync(
            string sourceWavePath,
            double startSeconds,
            double endSeconds,
            string destinationPath,
            CancellationToken cancellationToken) =>
            Task.FromResult(new AudioClipExtractionResult(destinationPath, endSeconds - startSeconds, 48000, 2));

        public Task<AudioClipExtractionResult> ExtractAsync(
            string sourceWavePath,
            IReadOnlyList<AudioClipRange> ranges,
            string destinationPath,
            CancellationToken cancellationToken) =>
            Task.FromResult(new AudioClipExtractionResult(
                destinationPath,
                ranges.Sum(static range => range.EndSeconds - range.StartSeconds),
                48000,
                2));
    }

    private sealed class StubModelDownloader : IModelDownloaderContract
    {
        public Task<bool> DownloadAsync(
            string modelId,
            string fileName,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default,
            string? revision = null) =>
            Task.FromResult(true);

        public Task<bool> DownloadUriAsync(
            Uri sourceUri,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default) =>
            Task.FromResult(true);

        public Task<bool> VerifyHashAsync(
            string filePath,
            string expectedHash,
            CancellationToken cancellationToken = default) =>
            Task.FromResult(true);
    }

    private sealed class FakeEnhancementStage : Trackdub.Application.Transcripts.Pipeline.ITranscriptGenerationStage
    {
        public string StageName => Trackdub.Domain.StageRuns.StageNames.SpeechEnhancement;

        public Task<Trackdub.Application.Transcripts.Pipeline.TranscriptGenerationContext> ExecuteAsync(
            Trackdub.Application.Transcripts.Pipeline.TranscriptGenerationContext context,
            CancellationToken cancellationToken,
            IProgress<Trackdub.Contracts.Pipeline.PipelineProgressEvent>? progress = null) =>
            Task.FromResult(context);
    }
}
