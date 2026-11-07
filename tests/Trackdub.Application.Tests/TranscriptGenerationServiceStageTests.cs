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
    public async Task GenerateTranscriptStageAsync_asr_short_audio_fallback_ignores_existing_diarization_plan()
    {
        var transcriptionEngine = new CapturingTranscriptionEngine();
        (TranscriptGenerationService service, TranscriptArtifactWriter artifactWriter) = CreateMinimalService(transcriptionEngine);
        TranscriptGenerationContext context = CreateContext();

        await artifactWriter.WriteSpeechRegionsArtifactAsync(
            new SpeechRegionsArtifactWriteRequest(
                context.Project,
                context.MediaAsset,
                [],
                StageRunRecord.Start(context.Project.Id, StageNames.Vad, DateTimeOffset.UtcNow)
                    .Complete(DateTimeOffset.UtcNow),
                context.AudioRoutingPlan.VadAudioArtifact),
            TestContext.Current.CancellationToken);
        ProjectSpeaker speaker = ProjectSpeaker.Create(
            context.Project.Id,
            "Existing speaker",
            DateTimeOffset.UtcNow);
        SpeakerTurn staleTurn = SpeakerTurn.Create(
            context.Project.Id,
            speaker.Id,
            startSeconds: 0.25d,
            endSeconds: 0.75d);
        await artifactWriter.WriteDiarizationArtifactAsync(
            context.Project.Id,
            context.MediaAsset,
            [speaker],
            [staleTurn],
            Guid.NewGuid(),
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

        await localArtifactWriter.WriteSpeechRegionsArtifactAsync(
            new SpeechRegionsArtifactWriteRequest(
                context.Project,
                context.MediaAsset,
                [new SpeechRegion(0, 0.0d, 1.0d)],
                StageRunRecord.Start(context.Project.Id, StageNames.Vad, DateTimeOffset.UtcNow)
                    .Complete(DateTimeOffset.UtcNow),
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
