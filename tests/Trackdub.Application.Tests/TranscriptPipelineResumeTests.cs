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

        VadTestInputs inputs = await SeedVadInputsAsync(
            artifactStore, mediaRepository, artifactWriter,
            hasCachedOutput ? CachedVadMode.Present : CachedVadMode.Absent);
        ResumeProject resume = inputs.Resume;
        ProjectArtifact audioArtifact = inputs.AudioArtifact;
        StageRunRecord vadRun = inputs.VadRun;

        FakeSpeechRegionDetector vadDetector = CreateVadDetector(detectorFails ? VadDetectorMode.Fail : VadDetectorMode.Succeed);
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

        TranscriptProjectState resumeState = CreateResumeState(
            new ResumeStateSeed(resume.Project, resume.MediaAsset, mediaRepository.Artifacts, [vadRun]));
        var context = CreateContext(
            new ContextSeed
            {
                Project = resume.Project,
                MediaAsset = resume.MediaAsset,
                AudioArtifact = audioArtifact,
                ProjectState = resumeState,
                ProjectRootPath = artifactStore.GetPath("."),
                ForceRerun = forceRerun,
                ExecutionSnapshot = CreateExecutionSnapshot(modelChanged ? ModelSnapshotMode.Changed : ModelSnapshotMode.Unchanged),
            });

        if (detectorFails)
        {
            await AssertVadFailureAsync(
                new VadFailureContext
                {
                    Pipeline = pipeline,
                    Context = context,
                    ArtifactWriter = artifactWriter,
                    StageRunStore = stageRunStore,
                    ProjectId = resume.Project.Id,
                });
            return;
        }

        await AssertVadResumeAsync(
            new VadResumeExpectation
            {
                Pipeline = pipeline,
                Context = context,
                Detector = vadDetector,
                StageRunStore = stageRunStore,
                VadRun = vadRun,
                ForceRerun = forceRerun,
                HasCachedOutput = hasCachedOutput,
                ModelChanged = modelChanged,
            });
    }

    private static async Task<VadTestInputs> SeedVadInputsAsync(
        FakeArtifactStore artifactStore,
        FakeMediaAssetRepository mediaRepository,
        TranscriptArtifactWriter artifactWriter,
        CachedVadMode cachedVad)
    {
        ResumeProject resume = CreateResumeProject();
        ProjectArtifact audioArtifact = CreateNormalizedAudio(resume);
        StageRunRecord vadRun = StageRunRecord
            .Start(resume.Project.Id, StageNames.Vad, resume.Now)
            .Complete(resume.Now);
        artifactStore.Seed(audioArtifact.RelativePath, FakeWavHelper.MinimalPcm16(durationSeconds: 10.0));
        mediaRepository.Seed(resume.MediaAsset);
        await mediaRepository.SaveArtifactAsync(audioArtifact, CancellationToken.None);
        if (cachedVad == CachedVadMode.Present)
        {
            await WriteCachedVadAsync(artifactWriter, resume, vadRun);
        }

        return new VadTestInputs(resume, audioArtifact, vadRun);
    }

    private static FakeSpeechRegionDetector CreateVadDetector(VadDetectorMode mode)
    {
        var vadDetector = new FakeSpeechRegionDetector();
        vadDetector.SetRegions(new SpeechRegion(0, 3.0, 4.0));
        if (mode == VadDetectorMode.Fail)
        {
            vadDetector.SetException(new InvalidOperationException("VAD unavailable"));
        }

        return vadDetector;
    }

    private static Task WriteCachedVadAsync(
        TranscriptArtifactWriter artifactWriter,
        ResumeProject resume,
        StageRunRecord vadRun) =>
        artifactWriter.WriteSpeechRegionsArtifactAsync(
            resume.Project.Id,
            resume.MediaAsset,
            [new SpeechRegion(0, 0.0, 2.0)],
            vadRun.Id,
            CancellationToken.None);

    private static IReadOnlyDictionary<string, string> CreateExecutionSnapshot(ModelSnapshotMode mode) =>
        mode == ModelSnapshotMode.Changed
            ? new Dictionary<string, string> { [$"Model:{StageNames.Vad}"] = "replacement-vad" }
            : new Dictionary<string, string>();

    private static async Task AssertVadFailureAsync(VadFailureContext context)
    {
        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            context.Pipeline.ExecuteAsync(context.Context, TestContext.Current.CancellationToken));
        Assert.Equal("VAD unavailable", error.Message);
        Assert.Equal(StageRunStatus.Failed, Assert.Single(context.StageRunStore.All).Status);
        IReadOnlyList<SpeechRegion>? preserved = await context.ArtifactWriter.TryReadSpeechRegionsAsync(
            context.ProjectId, TestContext.Current.CancellationToken);
        Assert.Equal(0.0, Assert.Single(preserved!).StartSeconds);
    }

    private static async Task AssertVadResumeAsync(VadResumeExpectation expectation)
    {
        TranscriptGenerationContext result = await expectation.Pipeline.ExecuteAsync(
            expectation.Context,
            TestContext.Current.CancellationToken);

        bool shouldResume = !expectation.ForceRerun && expectation.HasCachedOutput && !expectation.ModelChanged;
        Assert.Equal(shouldResume ? 0 : 1, expectation.Detector.DetectCallCount);
        Assert.Equal(shouldResume ? 0.0 : 3.0, Assert.Single(result.SpeechRegions).StartSeconds);
        StageRunRecord recordedRun = Assert.Single(expectation.StageRunStore.All);
        Assert.Equal(StageNames.Vad, recordedRun.StageName);
        Assert.Equal(shouldResume ? StageRunStatus.Skipped : StageRunStatus.Completed, recordedRun.Status);
        Assert.Equal(shouldResume ? expectation.VadRun.Id : recordedRun.Id, result.VadStageRunId);
    }

    private enum CachedVadMode
    {
        Absent,
        Present,
    }

    private enum VadDetectorMode
    {
        Succeed,
        Fail,
    }

    private enum ModelSnapshotMode
    {
        Unchanged,
        Changed,
    }

    private sealed record VadTestInputs(
        ResumeProject Resume,
        ProjectArtifact AudioArtifact,
        StageRunRecord VadRun);

    private sealed record VadFailureContext
    {
        public required TranscriptGenerationPipeline Pipeline { get; init; }

        public required TranscriptGenerationContext Context { get; init; }

        public required TranscriptArtifactWriter ArtifactWriter { get; init; }

        public required FakeProjectStageRunStore StageRunStore { get; init; }

        public required Guid ProjectId { get; init; }
    }

    private sealed record VadResumeExpectation
    {
        public required TranscriptGenerationPipeline Pipeline { get; init; }

        public required TranscriptGenerationContext Context { get; init; }

        public required FakeSpeechRegionDetector Detector { get; init; }

        public required FakeProjectStageRunStore StageRunStore { get; init; }

        public required StageRunRecord VadRun { get; init; }

        public required bool ForceRerun { get; init; }

        public required bool HasCachedOutput { get; init; }

        public required bool ModelChanged { get; init; }
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

        ResumeProject resume = CreateResumeProject();
        ProjectArtifact normalized = CreateNormalizedAudio(resume);
        StageRunRecord enhancementRun = StageRunRecord
            .Start(resume.Project.Id, StageNames.SpeechEnhancement, resume.Now)
            .Complete(resume.Now);
        var enhanced = new ProjectArtifact(
            Guid.NewGuid(),
            resume.Project.Id,
            resume.MediaAsset.Id,
            ArtifactKind.SpeechEnhancedAudio,
            "artifacts/audio/speech-enhancement/x/speech.wav",
            "enhanced-hash",
            100,
            10.0d,
            16000,
            1,
            resume.Now,
            StageRunId: enhancementRun.Id,
            Provenance: $"generated-{backend}-speech-enhancement:{(staleSource ? Guid.NewGuid() : normalized.Id):D}:profile={backendProfile}");

        artifactStore.Seed(normalized.RelativePath, FakeWavHelper.MinimalPcm16(durationSeconds: 10.0));
        artifactStore.Seed(enhanced.RelativePath, FakeWavHelper.MinimalPcm16(durationSeconds: 10.0));
        mediaRepository.Seed(resume.MediaAsset);
        await mediaRepository.SaveArtifactAsync(normalized, CancellationToken.None);
        await mediaRepository.SaveArtifactAsync(enhanced, CancellationToken.None);

        TranscriptProjectState resumeState = CreateResumeState(
            new ResumeStateSeed(resume.Project, resume.MediaAsset, mediaRepository.Artifacts, [enhancementRun]));
        var context = CreateContext(
            new ContextSeed
            {
                Project = resume.Project,
                MediaAsset = resume.MediaAsset,
                AudioArtifact = normalized,
                ProjectState = resumeState,
                ProjectRootPath = artifactStore.GetPath("."),
            });
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

    private static ResumeProject CreateResumeProject()
    {
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
        return new ResumeProject(project, mediaAsset, now);
    }

    private static ProjectArtifact CreateNormalizedAudio(ResumeProject resume) =>
        new(
            Guid.NewGuid(),
            resume.Project.Id,
            resume.MediaAsset.Id,
            ArtifactKind.NormalizedAudio,
            "artifacts/audio.wav",
            "audio-hash",
            100,
            10.0d,
            16000,
            1,
            resume.Now);

    private static TranscriptProjectState CreateResumeState(ResumeStateSeed seed) =>
        new(
            new OpenProjectResult(
                seed.Project,
                seed.MediaAsset,
                null,
                SourceMediaStatus.Available,
                null,
                seed.Artifacts,
                null),
            CurrentTranscriptRevision: null,
            TranscriptSegments: [],
            Speakers: [],
            SpeakerTurns: [],
            CurrentTranslationRevision: null,
            TranslatedSegments: [],
            IsTranslationStale: false,
            TranscriptLanguage: null,
            StageRuns: seed.StageRuns,
            SupportedTargetLanguages: [],
            SelectedTranslationTargetLanguage: null,
            StaleTranslatedSegmentIndices: new HashSet<int>(),
            WaveformSummary: null,
            AvailableVoices: [],
            VoiceAssignments: [],
            TtsTakes: [],
            TtsSegmentStates: [],
            VoiceAssignmentWarnings: []);

    private static TranscriptGenerationContext CreateContext(ContextSeed seed) =>
        new(
            seed.Project,
            seed.MediaAsset,
            seed.AudioArtifact,
            TranscriptAudioRoutingPlan.Raw(seed.AudioArtifact, SpeechAudioSourceKind.FullMix),
            enableSpeakerDiarization: false,
            sourceLanguage: "en")
        {
            ExecutionSnapshot = seed.ExecutionSnapshot ?? new Dictionary<string, string>(),
            ProjectState = seed.ProjectState,
            ProjectRootPath = seed.ProjectRootPath,
            ForceRerun = seed.ForceRerun,
        };

    private sealed record ResumeProject(
        TrackdubProject Project,
        MediaAsset MediaAsset,
        DateTimeOffset Now);

    private sealed record ResumeStateSeed(
        TrackdubProject Project,
        MediaAsset MediaAsset,
        IReadOnlyList<ProjectArtifact> Artifacts,
        IReadOnlyList<StageRunRecord> StageRuns);

    private sealed record ContextSeed
    {
        public required TrackdubProject Project { get; init; }

        public required MediaAsset MediaAsset { get; init; }

        public required ProjectArtifact AudioArtifact { get; init; }

        public required TranscriptProjectState ProjectState { get; init; }

        public required string ProjectRootPath { get; init; }

        public bool ForceRerun { get; init; }

        public IReadOnlyDictionary<string, string>? ExecutionSnapshot { get; init; }
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
