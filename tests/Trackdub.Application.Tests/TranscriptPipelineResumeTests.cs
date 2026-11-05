using Trackdub.Application.Transcripts;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Application.Transcripts.Stages;
using Trackdub.Application.Projects;
using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.Projects;
using Trackdub.Domain;
using Trackdub.Domain.AudioQuality;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Projects;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed class TranscriptPipelineResumeTests : IDisposable
{
    private readonly string tempRoot = Path.Join(Path.GetTempPath(), $"trackdub-resume-{Guid.NewGuid():N}");

    [Theory]
    [InlineData(false, true, false, false)]
    [InlineData(true, true, false, false)]
    [InlineData(false, false, false, false)]
    [InlineData(false, true, true, false)]
    [InlineData(true, true, false, true)]
    public async Task ExecuteAsync_reuses_vad_only_when_artifacts_and_runtime_are_valid(
        bool forceRerun, bool hasCachedOutput, bool modelChanged, bool detectorFails)
    {
        var artifactStore = new FakeArtifactStore(tempRoot);
        var mediaRepository = new FakeMediaAssetRepository();
        var stageRunStore = new FakeProjectStageRunStore();
        var transcriptRepository = new FakeTranscriptRepository();
        var artifactWriter = new TranscriptArtifactWriter(
            artifactStore,
            new FakeFileFingerprintService(new FileFingerprint("hash", 1, DateTimeOffset.UtcNow)),
            mediaRepository);

        Guid projectId = Guid.NewGuid();
        Guid mediaAssetId = Guid.NewGuid();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        var project = new TrackdubProject(projectId, "Resume", now, now);
        var mediaAsset = new MediaAsset(
            mediaAssetId,
            projectId,
            "source.mp4",
            "source.mp4",
            "source-hash",
            100,
            now,
            "mp4",
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

        StageRunRecord vadRun = StageRunRecord.Start(projectId, StageNames.Vad, now).Complete(now);
        artifactStore.Seed(audioArtifact.RelativePath, FakeWavHelper.MinimalPcm16(durationSeconds: 10.0));
        mediaRepository.Seed(mediaAsset);
        await mediaRepository.SaveArtifactAsync(audioArtifact, CancellationToken.None);
        if (hasCachedOutput)
        {
            await artifactWriter.WriteSpeechRegionsArtifactAsync(
                projectId,
                mediaAsset,
                [new SpeechRegion(0, 0.0, 2.0)],
                vadRun.Id,
                CancellationToken.None);
        }

        var vadDetector = new FakeSpeechRegionDetector();
        vadDetector.SetRegions(new SpeechRegion(0, 3.0, 4.0));
        if (detectorFails)
        {
            vadDetector.SetException(new InvalidOperationException("VAD unavailable"));
        }
        var vadStage = new VadGenerationStage(
            new VadStageHandler(vadDetector, stageRunStore),
            artifactWriter,
            artifactStore);
        var pipeline = new TranscriptGenerationPipeline(
            [new NoOpEnhancementStage(), vadStage],
            artifactStore,
            artifactWriter,
            transcriptRepository,
            stageRunStore: stageRunStore);

        var resumeState = new TranscriptProjectState(
            new OpenProjectResult(
                project,
                mediaAsset,
                null,
                SourceMediaStatus.Available,
                null,
                mediaRepository.Artifacts,
                null),
            CurrentTranscriptRevision: null,
            TranscriptSegments: [],
            Speakers: [],
            SpeakerTurns: [],
            CurrentTranslationRevision: null,
            TranslatedSegments: [],
            IsTranslationStale: false,
            TranscriptLanguage: null,
            StageRuns: [vadRun],
            SupportedTargetLanguages: [],
            SelectedTranslationTargetLanguage: null,
            StaleTranslatedSegmentIndices: new HashSet<int>(),
            WaveformSummary: null,
            AvailableVoices: [],
            VoiceAssignments: [],
            TtsTakes: [],
            TtsSegmentStates: [],
            VoiceAssignmentWarnings: []);

        var context = new TranscriptGenerationContext(
            project,
            mediaAsset,
            audioArtifact,
            TranscriptAudioRoutingPlan.Raw(audioArtifact, SpeechAudioSourceKind.FullMix),
            enableSpeakerDiarization: false,
            sourceLanguage: "en")
        {
            ExecutionSnapshot = modelChanged
                ? new Dictionary<string, string> { [$"Model:{StageNames.Vad}"] = "replacement-vad" }
                : new Dictionary<string, string>(),
            ProjectState = resumeState,
            ProjectRootPath = artifactStore.GetPath("."),
            ForceRerun = forceRerun
        };

        if (detectorFails)
        {
            InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(() =>
                pipeline.ExecuteAsync(context, TestContext.Current.CancellationToken));
            Assert.Equal("VAD unavailable", error.Message);
            Assert.Equal(StageRunStatus.Failed, Assert.Single(stageRunStore.All).Status);
            IReadOnlyList<SpeechRegion>? preserved = await artifactWriter.TryReadSpeechRegionsAsync(
                projectId, TestContext.Current.CancellationToken);
            Assert.Equal(0.0, Assert.Single(preserved!).StartSeconds);
            return;
        }

        TranscriptGenerationContext result = await pipeline.ExecuteAsync(
            context,
            TestContext.Current.CancellationToken);

        bool shouldResume = !forceRerun && hasCachedOutput && !modelChanged;
        Assert.Equal(shouldResume ? 0 : 1, vadDetector.DetectCallCount);
        Assert.Equal(shouldResume ? 0.0 : 3.0, Assert.Single(result.SpeechRegions).StartSeconds);
        StageRunRecord recordedRun = Assert.Single(stageRunStore.All);
        Assert.Equal(StageNames.Vad, recordedRun.StageName);
        Assert.Equal(shouldResume ? StageRunStatus.Skipped : StageRunStatus.Completed, recordedRun.Status);
        Assert.Equal(shouldResume ? vadRun.Id : recordedRun.Id, result.VadStageRunId);
    }

    [Theory]
    [InlineData("studio_voice_high_quality", "nvidia-afx", true, false)]
    [InlineData("deepfilternet3", "deepfilternet", false, false)]
    [InlineData("studio_voice_high_quality", "nvidia-afx", false, true)]
    public async Task ExecuteAsync_hydrates_asr_routing_only_for_native_studio_voice(
        string backendProfile,
        string backend,
        bool expectAsrRouted,
        bool staleSource)
    {
        var artifactStore = new FakeArtifactStore(tempRoot);
        var mediaRepository = new FakeMediaAssetRepository();
        var stageRunStore = new FakeProjectStageRunStore();
        var transcriptRepository = new FakeTranscriptRepository();
        var artifactWriter = new TranscriptArtifactWriter(
            artifactStore,
            new FakeFileFingerprintService(new FileFingerprint("hash", 1, DateTimeOffset.UtcNow)),
            mediaRepository);

        Guid projectId = Guid.NewGuid();
        Guid mediaAssetId = Guid.NewGuid();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        var project = new TrackdubProject(projectId, "Resume", now, now);
        var mediaAsset = new MediaAsset(
            mediaAssetId,
            projectId,
            "source.mp4",
            "source.mp4",
            "source-hash",
            100,
            now,
            "mp4",
            10.0d,
            HasAudio: true,
            HasVideo: false,
            now);
        var normalized = new ProjectArtifact(
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
        StageRunRecord enhancementRun = StageRunRecord
            .Start(projectId, StageNames.SpeechEnhancement, now)
            .Complete(now);
        var enhanced = new ProjectArtifact(
            Guid.NewGuid(),
            projectId,
            mediaAssetId,
            ArtifactKind.SpeechEnhancedAudio,
            "artifacts/audio/speech-enhancement/x/speech.wav",
            "enhanced-hash",
            100,
            10.0d,
            16000,
            1,
            now,
            StageRunId: enhancementRun.Id,
            Provenance: $"generated-{backend}-speech-enhancement:{(staleSource ? Guid.NewGuid() : normalized.Id):D}:profile={backendProfile}");

        artifactStore.Seed(normalized.RelativePath, FakeWavHelper.MinimalPcm16(durationSeconds: 10.0));
        artifactStore.Seed(enhanced.RelativePath, FakeWavHelper.MinimalPcm16(durationSeconds: 10.0));
        mediaRepository.Seed(mediaAsset);
        await mediaRepository.SaveArtifactAsync(normalized, CancellationToken.None);
        await mediaRepository.SaveArtifactAsync(enhanced, CancellationToken.None);

        var resumeState = new TranscriptProjectState(
            new OpenProjectResult(
                project,
                mediaAsset,
                null,
                SourceMediaStatus.Available,
                null,
                mediaRepository.Artifacts,
                null),
            CurrentTranscriptRevision: null,
            TranscriptSegments: [],
            Speakers: [],
            SpeakerTurns: [],
            CurrentTranslationRevision: null,
            TranslatedSegments: [],
            IsTranslationStale: false,
            TranscriptLanguage: null,
            StageRuns: [enhancementRun],
            SupportedTargetLanguages: [],
            SelectedTranslationTargetLanguage: null,
            StaleTranslatedSegmentIndices: new HashSet<int>(),
            WaveformSummary: null,
            AvailableVoices: [],
            VoiceAssignments: [],
            TtsTakes: [],
            TtsSegmentStates: [],
            VoiceAssignmentWarnings: []);

        var context = new TranscriptGenerationContext(
            project,
            mediaAsset,
            normalized,
            TranscriptAudioRoutingPlan.Raw(normalized, SpeechAudioSourceKind.FullMix),
            enableSpeakerDiarization: false,
            sourceLanguage: "en")
        {
            ProjectState = resumeState,
            ProjectRootPath = artifactStore.GetPath(".")
        };

        var pipeline = new TranscriptGenerationPipeline(
            [new NoOpEnhancementStage()],
            artifactStore,
            artifactWriter,
            transcriptRepository,
            stageRunStore: stageRunStore);

        TranscriptGenerationContext result = await pipeline.ExecuteAsync(
            context,
            TestContext.Current.CancellationToken);

        Guid expectedEnhancedStageSourceId = staleSource ? normalized.Id : enhanced.Id;
        Assert.Equal(expectedEnhancedStageSourceId, result.AudioRoutingPlan.VadAudioArtifact.Id);
        Assert.Equal(expectedEnhancedStageSourceId, result.AudioRoutingPlan.DiarizationAudioArtifact.Id);
        Assert.Equal(
            expectAsrRouted ? enhanced.Id : normalized.Id,
            result.AudioRoutingPlan.AsrAudioArtifact.Id);
        Assert.Equal(
            expectAsrRouted ? "nvidia-afx-studio-voice-high-quality" : SpeechAudioProcessingProfileCatalog.NoneProfileId,
            result.AudioRoutingPlan.AsrDecision.ProfileId);
    }

    public void Dispose()
    {
        if (Directory.Exists(tempRoot))
        {
            Directory.Delete(tempRoot, recursive: true);
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
}
