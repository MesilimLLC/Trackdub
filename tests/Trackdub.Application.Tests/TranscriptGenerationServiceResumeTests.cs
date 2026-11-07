using Trackdub.Application.Pipeline;
using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Application.Transcripts.Stages;
using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.Projects;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.AudioQuality;
using Trackdub.Domain.Media;
using Trackdub.Domain.Projects;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed class TranscriptGenerationServiceResumeTests
{
    [Fact]
    public async Task GenerateTranscriptAsync_skips_resumable_vad_and_still_preflights_asr()
    {
        ServiceHarness harness = CreateHarness();
        (TrackdubProject project, MediaAsset mediaAsset, ProjectArtifact audioArtifact, TranscriptAudioRoutingPlan routingPlan) =
            await harness.SeedProjectAsync(TestContext.Current.CancellationToken);
        ProjectArtifact enhancedArtifact = await harness.SeedSpeechEnhancementResumeStateAsync(
            project,
            mediaAsset,
            TestContext.Current.CancellationToken);
        await harness.SeedVadResumeStateAsync(
            project,
            mediaAsset,
            enhancedArtifact,
            [new SpeechRegion(0, 0.0, 2.0)]);

        await harness.Service.GenerateTranscriptAsync(
            project,
            mediaAsset,
            audioArtifact,
            routingPlan,
            enableSpeakerDiarization: false,
            InferenceModelPreferences.Empty,
            TestContext.Current.CancellationToken,
            sourceLanguage: "en");

        AssertAsrWasPreflightedAndExecuted(harness);
        Assert.DoesNotContain(StageNames.Vad, harness.PreFlightChecker.CheckedStageNames);
        Assert.Contains(
            harness.StageRunStore.All,
            run => run.StageName == StageNames.Vad && run.Status == StageRunStatus.Skipped);
        Assert.Single(harness.TranscriptRepository.Revisions);
    }

    [Fact]
    public async Task GenerateTranscriptAsync_preflights_and_reruns_vad_when_audio_content_changes_with_same_artifact_id()
    {
        ServiceHarness harness = CreateHarness();
        (TrackdubProject project, MediaAsset mediaAsset, ProjectArtifact audioArtifact, _) =
            await harness.SeedProjectAsync(TestContext.Current.CancellationToken);
        await harness.SeedVadResumeStateAsync(
            project,
            mediaAsset,
            audioArtifact,
            [new SpeechRegion(0, 0.0, 2.0)]);
        ProjectArtifact changedAudioArtifact = audioArtifact with { Sha256 = "updated-audio-content" };
        TranscriptAudioRoutingPlan changedRoutingPlan = TranscriptAudioRoutingPlan.Raw(
            changedAudioArtifact,
            SpeechAudioSourceKind.FullMix);

        await harness.Service.GenerateTranscriptAsync(
            project,
            mediaAsset,
            audioArtifact,
            changedRoutingPlan,
            enableSpeakerDiarization: false,
            InferenceModelPreferences.Empty,
            TestContext.Current.CancellationToken,
            sourceLanguage: "en");

        Assert.Equal(1, harness.VadDetector.DetectCallCount);
        Assert.Contains(StageNames.Vad, harness.PreFlightChecker.CheckedStageNames);
        ProjectArtifact latestRegions = harness.MediaRepository.Artifacts
            .Where(artifact => artifact.Kind == ArtifactKind.SpeechRegions)
            .MaxBy(artifact => artifact.CreatedAtUtc)!;
        Assert.Equal(
            TranscriptArtifactWriter.GetSpeechRegionsProvenance(changedAudioArtifact),
            latestRegions.Provenance);
    }

    [Fact]
    public async Task GenerateTranscriptAsync_preflights_asr_when_latest_vad_is_newer_than_cached_asr()
    {
        ServiceHarness harness = CreateHarness();
        (TrackdubProject project, MediaAsset mediaAsset, ProjectArtifact audioArtifact, TranscriptAudioRoutingPlan routingPlan) =
            await harness.SeedProjectAsync(TestContext.Current.CancellationToken);
        StageRunRecord oldVadRun = await harness.SeedVadResumeStateAsync(
            project,
            mediaAsset,
            audioArtifact,
            [new SpeechRegion(0, 0.0, 2.0)]);
        await harness.SeedAsrResumeStateAsync(
            project,
            mediaAsset,
            audioArtifact,
            new AsrResumeSeedRequest
            {
                VadStageRunId = oldVadRun.Id,
                Segments = [new RecognizedTranscriptSegment(0, 0.0, 2.0, "cached transcript", "en")],
                CancellationToken = TestContext.Current.CancellationToken
            });
        await harness.SeedVadResumeStateAsync(
            project,
            mediaAsset,
            audioArtifact,
            [new SpeechRegion(0, 0.0, 2.0)]);

        await harness.Service.GenerateTranscriptAsync(
            project,
            mediaAsset,
            audioArtifact,
            routingPlan,
            enableSpeakerDiarization: false,
            InferenceModelPreferences.Empty,
            TestContext.Current.CancellationToken,
            sourceLanguage: "en");

        AssertAsrWasPreflightedAndExecuted(harness);
    }

    [Fact]
    public async Task GenerateTranscriptAsync_skips_resumable_vad_and_asr_without_preflight_for_completed_stages()
    {
        ServiceHarness harness = CreateHarness();
        (TrackdubProject project, MediaAsset mediaAsset, ProjectArtifact audioArtifact, TranscriptAudioRoutingPlan routingPlan) =
            await harness.SeedProjectAsync(TestContext.Current.CancellationToken);
        ProjectArtifact enhancedArtifact = await harness.SeedSpeechEnhancementResumeStateAsync(
            project,
            mediaAsset,
            TestContext.Current.CancellationToken);
        StageRunRecord vadRun = await harness.SeedVadResumeStateAsync(
            project,
            mediaAsset,
            enhancedArtifact,
            [new SpeechRegion(0, 0.0, 2.0)]);
        await harness.SeedAsrResumeStateAsync(
            project,
            mediaAsset,
            audioArtifact,
            new AsrResumeSeedRequest
            {
                VadStageRunId = vadRun.Id,
                VadAudioArtifact = enhancedArtifact,
                Segments = [new RecognizedTranscriptSegment(0, 0.0, 2.0, "hello", "en")],
                CancellationToken = TestContext.Current.CancellationToken
            });

        await harness.Service.GenerateTranscriptAsync(
            project,
            mediaAsset,
            audioArtifact,
            routingPlan,
            enableSpeakerDiarization: false,
            InferenceModelPreferences.Empty,
            TestContext.Current.CancellationToken,
            sourceLanguage: "en");

        Assert.Equal(0, harness.VadDetector.DetectCallCount);
        Assert.Equal(0, harness.AsrEngine.TranscribeCallCount);
        Assert.DoesNotContain(StageNames.Vad, harness.PreFlightChecker.CheckedStageNames);
        Assert.DoesNotContain(StageNames.Asr, harness.PreFlightChecker.CheckedStageNames);
        Assert.Contains(
            harness.StageRunStore.All,
            run => run.StageName == StageNames.Vad && run.Status == StageRunStatus.Skipped);
        Assert.Contains(
            harness.StageRunStore.All,
            run => run.StageName == StageNames.Asr && run.Status == StageRunStatus.Skipped);
        Assert.Equal(2, harness.TranscriptRepository.Revisions.Count);
    }

    [Fact]
    public async Task GenerateTranscriptAsync_preflights_vad_when_speech_enhancement_cannot_be_resumed()
    {
        ServiceHarness harness = CreateHarness();
        (TrackdubProject project, MediaAsset mediaAsset, ProjectArtifact audioArtifact, TranscriptAudioRoutingPlan routingPlan) =
            await harness.SeedProjectAsync(TestContext.Current.CancellationToken);
        await harness.SeedVadResumeStateAsync(
            project,
            mediaAsset,
            audioArtifact,
            [new SpeechRegion(0, 0.0, 2.0)]);

        await harness.Service.GenerateTranscriptAsync(
            project,
            mediaAsset,
            audioArtifact,
            routingPlan,
            enableSpeakerDiarization: false,
            InferenceModelPreferences.Empty,
            TestContext.Current.CancellationToken,
            sourceLanguage: "en");

        Assert.Contains(StageNames.Vad, harness.PreFlightChecker.CheckedStageNames);
        Assert.Equal(0, harness.VadDetector.DetectCallCount);
        Assert.Contains(
            harness.StageRunStore.All,
            run => run.StageName == StageNames.Vad && run.Status == StageRunStatus.Skipped);
    }

    [Fact]
    public async Task GenerateTranscriptAsync_reruns_asr_after_resumed_vad_with_no_regions()
    {
        ServiceHarness harness = CreateHarness();
        (TrackdubProject project, MediaAsset mediaAsset, ProjectArtifact audioArtifact, TranscriptAudioRoutingPlan routingPlan) =
            await harness.SeedProjectAsync(TestContext.Current.CancellationToken);
        StageRunRecord vadRun = await harness.SeedVadResumeStateAsync(project, mediaAsset, audioArtifact, []);
        await harness.SeedAsrResumeStateAsync(
            project,
            mediaAsset,
            audioArtifact,
            new AsrResumeSeedRequest
            {
                VadStageRunId = vadRun.Id,
                Segments = [new RecognizedTranscriptSegment(0, 0.0, 2.0, "old fallback transcript", "en")],
                CancellationToken = TestContext.Current.CancellationToken
            });

        await harness.Service.GenerateTranscriptAsync(
            project,
            mediaAsset,
            audioArtifact,
            routingPlan,
            enableSpeakerDiarization: false,
            InferenceModelPreferences.Empty,
            TestContext.Current.CancellationToken,
            sourceLanguage: "en");

        Assert.Equal(1, harness.AsrEngine.TranscribeCallCount);
        Assert.Contains(StageNames.Asr, harness.PreFlightChecker.CheckedStageNames);
        Assert.Contains(
            harness.StageRunStore.All,
            run => run.StageName == StageNames.Asr && run.Status == StageRunStatus.Completed);
    }

    [Fact]
    public async Task GenerateTranscriptAsync_skips_resumable_speaker_assignment_without_repersisting()
    {
        ServiceHarness harness = CreateHarness();
        (TrackdubProject project, MediaAsset mediaAsset, ProjectArtifact audioArtifact, TranscriptAudioRoutingPlan routingPlan) =
            await harness.SeedProjectAsync(TestContext.Current.CancellationToken);
        StageRunRecord vadRun = await harness.SeedVadResumeStateAsync(
            project,
            mediaAsset,
            audioArtifact,
            [new SpeechRegion(0, 0.0, 2.0)]);
        StageRunRecord asrRun = await harness.SeedAsrResumeStateAsync(
            project,
            mediaAsset,
            audioArtifact,
            new AsrResumeSeedRequest
            {
                VadStageRunId = vadRun.Id,
                Segments = [new RecognizedTranscriptSegment(0, 0.0, 2.0, "hello", "en")],
                CancellationToken = TestContext.Current.CancellationToken
            });
        await harness.SeedSpeakerAssignmentResumeStateAsync(
            project,
            mediaAsset,
            asrRun,
            TestContext.Current.CancellationToken);

        await harness.Service.GenerateTranscriptAsync(
            project,
            mediaAsset,
            audioArtifact,
            routingPlan,
            enableSpeakerDiarization: false,
            InferenceModelPreferences.Empty,
            TestContext.Current.CancellationToken,
            sourceLanguage: "en");

        Assert.Equal(0, harness.VadDetector.DetectCallCount);
        Assert.Equal(0, harness.AsrEngine.TranscribeCallCount);
        Assert.Single(harness.TranscriptRepository.Revisions);
        Assert.Contains(
            harness.StageRunStore.All,
            run => run.StageName == StageNames.Vad && run.Status == StageRunStatus.Skipped);
        Assert.Contains(
            harness.StageRunStore.All,
            run => run.StageName == StageNames.Asr && run.Status == StageRunStatus.Skipped);
        Assert.Contains(
            harness.StageRunStore.All,
            run => run.StageName == StageNames.SpeakerAssignment && run.Status == StageRunStatus.Skipped);
    }

    [Fact]
    public async Task GenerateTranscriptAsync_forceRerun_runs_vad_despite_existing_artifacts()
    {
        ServiceHarness harness = CreateHarness();
        (TrackdubProject project, MediaAsset mediaAsset, ProjectArtifact audioArtifact, TranscriptAudioRoutingPlan routingPlan) =
            await harness.SeedProjectAsync(TestContext.Current.CancellationToken);
        await harness.SeedVadResumeStateAsync(
            project,
            mediaAsset,
            audioArtifact,
            [new SpeechRegion(0, 0.0, 2.0)]);

        await harness.Service.GenerateTranscriptAsync(
            project,
            mediaAsset,
            audioArtifact,
            routingPlan,
            enableSpeakerDiarization: false,
            InferenceModelPreferences.Empty,
            TestContext.Current.CancellationToken,
            sourceLanguage: "en",
            forceRerun: true);

        Assert.Equal(1, harness.VadDetector.DetectCallCount);
        Assert.Contains(StageNames.Vad, harness.PreFlightChecker.CheckedStageNames);
        Assert.DoesNotContain(
            harness.StageRunStore.All,
            run => run.StageName == StageNames.Vad && run.Status == StageRunStatus.Skipped);
    }

    private static void AssertAsrWasPreflightedAndExecuted(ServiceHarness harness)
    {
        Assert.Equal(0, harness.VadDetector.DetectCallCount);
        Assert.Equal(1, harness.AsrEngine.TranscribeCallCount);
        Assert.Contains(StageNames.Asr, harness.PreFlightChecker.CheckedStageNames);
    }

    private static byte[] BuildMinimalWav(int sampleRate, int durationSamples)
    {
        const int bitsPerSample = 16;
        const int channels = 1;
        int byteRate = sampleRate * channels * (bitsPerSample / 8);
        int blockAlign = channels * (bitsPerSample / 8);
        int dataBytes = durationSamples * blockAlign;
        byte[] wav = new byte[44 + dataBytes];
        int pos = 0;

        void WriteBytes(byte[] src)
        {
            Array.Copy(src, 0, wav, pos, src.Length);
            pos += src.Length;
        }

        void WriteAscii(string value) => WriteBytes(System.Text.Encoding.ASCII.GetBytes(value));
        void WriteInt32(int value) => WriteBytes(BitConverter.GetBytes(value));
        void WriteInt16(short value) => WriteBytes(BitConverter.GetBytes(value));

        WriteAscii("RIFF");
        WriteInt32(36 + dataBytes);
        WriteAscii("WAVE");
        WriteAscii("fmt ");
        WriteInt32(16);
        WriteInt16(1);
        WriteInt16((short)channels);
        WriteInt32(sampleRate);
        WriteInt32(byteRate);
        WriteInt16((short)blockAlign);
        WriteInt16(bitsPerSample);
        WriteAscii("data");
        WriteInt32(dataBytes);

        return wav;
    }

    private static ServiceHarness CreateHarness()
    {
        string projectRoot = Path.Join(Path.GetTempPath(), "trackdub-tests", Guid.NewGuid().ToString("N"));
        var artifactStore = new FakeArtifactStore(projectRoot);
        var mediaRepository = new FakeMediaAssetRepository();
        var stageRunStore = new FakeProjectStageRunStore();
        var transcriptRepository = new FakeTranscriptRepository();
        var fingerprintService = new FakeFileFingerprintService(new FileFingerprint("hash", 1, DateTimeOffset.UtcNow));
        var artifactWriter = new TranscriptArtifactWriter(artifactStore, fingerprintService, mediaRepository);
        var vadDetector = new FakeSpeechRegionDetector();
        var asrEngine = new CountingAsrEngine();
        var preFlightChecker = new FakePipelinePreFlightChecker();
        var asrHandler = new AsrStageHandler(asrEngine, stageRunStore);
        var speakerRepository = new FakeSpeakerRepository();
        string modelCacheRoot = Path.Join(projectRoot, "model-cache");

        SpeakerAssignmentService BuildSpeakerAssignmentService() =>
            new(
                speakerRepository,
                transcriptRepository,
                new SegmentEditingService(transcriptRepository, new FakeTtsTakeRepository(), artifactWriter),
                artifactStore,
                stageRunStore,
                new FakeDiarizationEngine(),
                new SpeakerReferenceClipService(
                    artifactStore,
                    new StubAudioClipExtractor(),
                    fingerprintService,
                    mediaRepository,
                    new FakeVoiceAssignmentRepository(),
                    new FakeTtsTakeRepository(),
                    new FakeReferenceClipAnalyzer(),
                    new FakeReferenceClipTrimmer()),
                artifactWriter,
                new DiarizationStageHandler(
                    new FakeDiarizationEngine(),
                    new WritingModelDownloader(),
                    modelCacheRoot: modelCacheRoot,
                    expectedSha256: SortFormerTestFixtures.ExpectedSha256));

        SpeakerAssignmentService speakerAssignmentService = BuildSpeakerAssignmentService();

        TranscriptGenerationService service = new(
            transcriptRepository,
            artifactStore,
            asrHandler,
            artifactWriter,
            new VadGenerationStage(
                new VadStageHandler(vadDetector, stageRunStore),
                artifactWriter,
                artifactStore),
            new NoOpEnhancementStage(),
            new SpeakerDiarizationStage(speakerAssignmentService, artifactWriter, artifactStore, stageRunStore),
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
            preFlightChecker,
            stageRunStore,
            mediaRepository,
            speakerRepository);

        return new ServiceHarness(
            service,
            artifactStore,
            mediaRepository,
            stageRunStore,
            transcriptRepository,
            artifactWriter,
            vadDetector,
            asrEngine,
            preFlightChecker);
    }

    private sealed record AsrResumeSeedRequest
    {
        public required Guid VadStageRunId { get; init; }
        public ProjectArtifact? VadAudioArtifact { get; init; }
        public required IReadOnlyList<RecognizedTranscriptSegment> Segments { get; init; }
        public required CancellationToken CancellationToken { get; init; }
    }

    private sealed class ServiceHarness(
        TranscriptGenerationService service,
        FakeArtifactStore artifactStore,
        FakeMediaAssetRepository mediaRepository,
        FakeProjectStageRunStore stageRunStore,
        FakeTranscriptRepository transcriptRepository,
        TranscriptArtifactWriter artifactWriter,
        FakeSpeechRegionDetector vadDetector,
        CountingAsrEngine asrEngine,
        FakePipelinePreFlightChecker preFlightChecker)
    {
        public TranscriptGenerationService Service { get; } = service;
        public FakeArtifactStore ArtifactStore { get; } = artifactStore;
        public FakeMediaAssetRepository MediaRepository { get; } = mediaRepository;
        public FakeProjectStageRunStore StageRunStore { get; } = stageRunStore;
        public FakeTranscriptRepository TranscriptRepository { get; } = transcriptRepository;
        public FakeSpeechRegionDetector VadDetector { get; } = vadDetector;
        public CountingAsrEngine AsrEngine { get; } = asrEngine;
        public FakePipelinePreFlightChecker PreFlightChecker { get; } = preFlightChecker;

        public async Task<(TrackdubProject Project, MediaAsset MediaAsset, ProjectArtifact AudioArtifact, TranscriptAudioRoutingPlan RoutingPlan)> SeedProjectAsync(
            CancellationToken cancellationToken)
        {
            Guid projectId = Guid.NewGuid();
            Guid mediaAssetId = Guid.NewGuid();
            DateTimeOffset now = DateTimeOffset.UtcNow;
            var project = new TrackdubProject(projectId, "Resume Integration", now, now);
            var mediaAsset = new MediaAsset(
                mediaAssetId,
                projectId,
                "artifacts/audio.wav",
                "audio.wav",
                "audio-hash",
                100,
                now,
                "wav",
                10.0d,
                HasAudio: true,
                HasVideo: false,
                now);
            var audioArtifact = new ProjectArtifact(
                Guid.NewGuid(),
                projectId,
                mediaAssetId,
                ArtifactKind.NormalizedAudio,
                "artifacts/audio.wav",
                "audio-hash",
                100,
                10.0d,
                16000,
                1,
                now);

            ArtifactStore.Seed("artifacts/audio.wav", BuildMinimalWav(sampleRate: 16000, durationSamples: 1600));
            MediaRepository.Seed(mediaAsset);
            await MediaRepository.SaveArtifactAsync(audioArtifact, cancellationToken).ConfigureAwait(false);
            await ArtifactStore.WriteJsonAsync(
                ProjectArtifactPaths.ManifestRelativePath,
                ProjectManifest.FromProject(project),
                cancellationToken).ConfigureAwait(false);

            TranscriptAudioRoutingPlan routingPlan = TranscriptAudioRoutingPlan.Raw(
                audioArtifact,
                SpeechAudioSourceKind.FullMix);

            return (project, mediaAsset, audioArtifact, routingPlan);
        }

        public async Task<StageRunRecord> SeedVadResumeStateAsync(
            TrackdubProject project,
            MediaAsset mediaAsset,
            ProjectArtifact inputAudioArtifact,
            IReadOnlyList<SpeechRegion> regions)
        {
            DateTimeOffset now = DateTimeOffset.UtcNow;
            StageRunRecord vadRun = StageRunRecord.Start(project.Id, StageNames.Vad, now).Complete(now);
            await StageRunStore.CreateAsync(vadRun, CancellationToken.None).ConfigureAwait(false);
            await artifactWriter.WriteSpeechRegionsArtifactAsync(
                new SpeechRegionsArtifactWriteRequest(
                    project,
                    mediaAsset,
                    regions,
                    vadRun,
                    inputAudioArtifact),
                CancellationToken.None).ConfigureAwait(false);
            return vadRun;
        }

        public async Task<ProjectArtifact> SeedSpeechEnhancementResumeStateAsync(
            TrackdubProject project,
            MediaAsset mediaAsset,
            CancellationToken cancellationToken)
        {
            DateTimeOffset now = DateTimeOffset.UtcNow;
            StageRunRecord enhancementRun = StageRunRecord
                .Start(project.Id, StageNames.SpeechEnhancement, now)
                .Complete(now);
            await StageRunStore.CreateAsync(enhancementRun, cancellationToken).ConfigureAwait(false);
            var enhancedArtifact = new ProjectArtifact(
                Guid.NewGuid(),
                project.Id,
                mediaAsset.Id,
                ArtifactKind.SpeechEnhancedAudio,
                "artifacts/audio/speech-enhancement/resume/speech.wav",
                "enhanced-audio-hash",
                100,
                10.0d,
                16000,
                1,
                now,
                StageRunId: enhancementRun.Id,
                Provenance: "generated-test-speech-enhancement");

            ArtifactStore.Seed(
                enhancedArtifact.RelativePath,
                BuildMinimalWav(sampleRate: 16000, durationSamples: 1600));
            await MediaRepository.SaveArtifactAsync(enhancedArtifact, cancellationToken).ConfigureAwait(false);
            return enhancedArtifact;
        }

        public async Task<StageRunRecord> SeedAsrResumeStateAsync(
            TrackdubProject project,
            MediaAsset mediaAsset,
            ProjectArtifact audioArtifact,
            AsrResumeSeedRequest request)
        {
            CancellationToken cancellationToken = request.CancellationToken;
            DateTimeOffset now = DateTimeOffset.UtcNow;
            StageRunRecord asrRun = StageRunRecord.Start(project.Id, StageNames.Asr, now).Complete(now);
            await StageRunStore.CreateAsync(asrRun, cancellationToken).ConfigureAwait(false);
            await artifactWriter.WriteRawAsrTranscriptArtifactAsync(
                new RawAsrTranscriptArtifactWriteRequest
                {
                    ProjectId = project.Id,
                    MediaAsset = mediaAsset,
                    Segments = request.Segments,
                    StageRunId = asrRun.Id,
                    Lineage = RawAsrTranscriptArtifactLineage.Create(
                        audioArtifact,
                        request.VadAudioArtifact ?? audioArtifact,
                        request.VadStageRunId)
                },
                cancellationToken).ConfigureAwait(false);

            TranscriptRevision revision = TranscriptRevision.Create(project.Id, asrRun.Id, 1, now);
            TranscriptSegment[] domainSegments = request.Segments
                .OrderBy(static segment => segment.Index)
                .Select(segment => TranscriptSegment.Create(
                    revision.Id,
                    segment.Index,
                    segment.StartSeconds,
                    segment.EndSeconds,
                    segment.Text,
                    speakerId: null,
                    segment.DetectedLanguage))
                .ToArray();
            await TranscriptRepository.SaveRevisionAsync(revision, domainSegments, cancellationToken)
                .ConfigureAwait(false);
            return asrRun;
        }

        public async Task SeedSpeakerAssignmentResumeStateAsync(
            TrackdubProject project,
            MediaAsset mediaAsset,
            StageRunRecord asrRun,
            CancellationToken cancellationToken)
        {
            TranscriptRevision revision = TranscriptRepository.Revisions.Single();
            IReadOnlyList<TranscriptSegment> segments = await TranscriptRepository
                .GetSegmentsAsync(revision.Id, cancellationToken)
                .ConfigureAwait(false);
            await artifactWriter.WriteTranscriptArtifactAsync(
                project.Id,
                mediaAsset,
                revision,
                segments,
                asrRun.Id,
                "generated-asr",
                cancellationToken).ConfigureAwait(false);

            DateTimeOffset now = DateTimeOffset.UtcNow;
            StageRunRecord speakerAssignmentRun = StageRunRecord
                .Start(project.Id, StageNames.SpeakerAssignment, now)
                .Complete(now);
            await StageRunStore.CreateAsync(speakerAssignmentRun, cancellationToken).ConfigureAwait(false);
        }
    }

    private sealed class NoOpEnhancementStage : ITranscriptGenerationStage
    {
        public string StageName => StageNames.SpeechEnhancement;

        public Task<TranscriptGenerationContext> ExecuteAsync(
            TranscriptGenerationContext context,
            CancellationToken cancellationToken,
            IProgress<PipelineProgressEvent>? progress = null) =>
            Task.FromResult(context);
    }

    private sealed class CountingAsrEngine : IAudioTranscriptionEngine
    {
        public int TranscribeCallCount { get; private set; }

        public Task<IReadOnlyList<RecognizedTranscriptSegment>> TranscribeAsync(
            string normalizedAudioPath,
            IReadOnlyList<SpeechRegion> regions,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            TranscribeCallCount++;
            return Task.FromResult<IReadOnlyList<RecognizedTranscriptSegment>>(
                [new RecognizedTranscriptSegment(0, 0.0, 2.0, "fresh asr", "en")]);
        }

        public Task<IReadOnlyList<RecognizedTranscriptSegment>> TranscribeAsync(
            AudioTranscriptionRequest request,
            CancellationToken cancellationToken) =>
            TranscribeAsync(request.NormalizedAudioPath, request.Regions, cancellationToken);
    }

    private sealed class StubAudioClipExtractor : IAudioClipExtractor
    {
        public Task<AudioClipExtractionResult> ExtractAsync(
            string sourceWavePath,
            double startSeconds,
            double endSeconds,
            string destinationPath,
            CancellationToken cancellationToken) =>
            Task.FromResult(new AudioClipExtractionResult(destinationPath, endSeconds - startSeconds, 16000, 1));

        public Task<AudioClipExtractionResult> ExtractAsync(
            string sourceWavePath,
            IReadOnlyList<AudioClipRange> ranges,
            string destinationPath,
            CancellationToken cancellationToken) =>
            Task.FromResult(new AudioClipExtractionResult(destinationPath, 1.0, 16000, 1));
    }

    private sealed class WritingModelDownloader : IModelDownloaderContract
    {
        public async Task<bool> DownloadAsync(
            string modelId,
            string fileName,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default,
            string? revision = null)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);
            await File.WriteAllBytesAsync(destinationPath, SortFormerTestFixtures.ModelBytes, cancellationToken);
            return true;
        }

        public Task<bool> DownloadUriAsync(
            Uri sourceUri,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();

        public Task<bool> VerifyHashAsync(
            string filePath,
            string expectedHash,
            CancellationToken cancellationToken = default) =>
            Task.FromResult(true);
    }
}
