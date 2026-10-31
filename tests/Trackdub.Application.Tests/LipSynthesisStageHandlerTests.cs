using Trackdub.Application.LipSynthesis;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.LipSynthesis;
using Trackdub.Domain.Media;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed class LipSynthesisStageHandlerTests
{
    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    private static string CreateTempDirectory()
    {
        var path = Path.Join(Path.GetTempPath(), $"lipsynth-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(path);
        return path;
    }

    private static MediaAsset MakeMediaAsset(Guid? projectId = null, bool hasVideo = true) => new(
        Id: Guid.NewGuid(),
        ProjectId: projectId ?? Guid.NewGuid(),
        SourceFilePath: "/tmp/test.mp4",
        SourceFileName: "test.mp4",
        FingerprintSha256: "abc",
        SourceSizeBytes: 0L,
        SourceLastWriteTimeUtc: DateTimeOffset.UtcNow,
        FormatName: "mp4",
        DurationSeconds: 10.0,
        HasAudio: true,
        HasVideo: hasVideo,
        CreatedAtUtc: DateTimeOffset.UtcNow);

    private static LipSynthesisTurn MakeTurn(double start = 1.0, double end = 3.0, string? speakerId = "spk-1") =>
        new(Guid.NewGuid(), TimeSpan.FromSeconds(start), TimeSpan.FromSeconds(end), speakerId);

    private static LipSynthesisStageHandler MakeHandler(
        FakeArtifactStore artifactStore,
        FakeProjectStageRunStore stageRunStore,
        FakeLipSynthesisEngine? engine = null,
        FakeFaceDetector? faceDetector = null,
        FakeFaceLandmarkProvider? landmarks = null,
        FakeFacePoseEstimator? pose = null,
        IMediaAssetRepository? mediaAssetRepository = null) =>
        new(
            lipSynthesisEngine: engine ?? new FakeLipSynthesisEngine(),
            faceDetector: faceDetector ?? new FakeFaceDetector(),
            faceLandmarkProvider: landmarks ?? new FakeFaceLandmarkProvider(),
            facePoseEstimator: pose ?? new FakeFacePoseEstimator(),
            artifactStore: artifactStore,
            stageRunStore: stageRunStore,
            segmentRepository: new FakeLipSynthesisSegmentRepository(),
            mediaAssetRepository: mediaAssetRepository);

    private static string EnsureDubbedAudioFile(string directory)
    {
        string path = Path.Join(directory, "dubbed-driver.wav");
        if (!File.Exists(path))
            File.WriteAllBytes(path, [0]);

        return path;
    }

    private static LipSynthesisStageRequest MakeRequest(
        MediaAsset mediaAsset,
        IReadOnlyList<LipSynthesisTurn> turns,
        string? workspaceDirectory = null,
        bool isEnabled = true,
        bool isLicenseApproved = true,
        bool allowExperimentalExecution = false,
        string? dubbedAudioPath = null) =>
        new(
            ProjectId: mediaAsset.ProjectId,
            MediaAsset: mediaAsset,
            SourceVideoPath: "/tmp/source.mp4",
            DubbedAudioPath: dubbedAudioPath
                ?? (workspaceDirectory is not null
                    ? EnsureDubbedAudioFile(workspaceDirectory)
                    : string.Empty),
            SpeakerTurns: turns,
            IsEnabled: isEnabled,
            IsLicenseApproved: isLicenseApproved,
            AllowExperimentalExecution: allowExperimentalExecution);

    // ---------------------------------------------------------------------------
    // Success
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_AllGuardsPass_SynthesizesAndRegistersPatchedArtifact()
    {
        string directory = CreateTempDirectory();
        try
        {
            var mediaAsset = MakeMediaAsset();
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var mediaRepo = new FakeMediaAssetRepository();
            var engine = new FakeLipSynthesisEngine
            {
                OutputDirectory = Path.Join(directory, "engine-out"),
                Experimental = false
            };

            var turn = MakeTurn();
            var handler = MakeHandler(artifactStore, stageRunStore, engine, mediaAssetRepository: mediaRepo);

            var result = await handler.HandleAsync(
                MakeRequest(mediaAsset, [turn], directory), TestContext.Current.CancellationToken);

            var seg = Assert.Single(result.Segments);
            Assert.Equal(LipSynthesisSegmentStatus.Synthesized, seg.Status);
            Assert.NotNull(seg.PatchedClipRelativePath);
            Assert.False(seg.UsedExperimentalProvider);
            Assert.Equal(StageRunStatus.Completed, result.StageRun.Status);

            // Patched-video artifact registered and present on disk.
            var artifact = Assert.Single(mediaRepo.Artifacts, a => a.Kind == ArtifactKind.LipSynthesisTake);
            Assert.True(File.Exists(artifactStore.GetPath(artifact.RelativePath)));
            Assert.Equal($"lipsynthesis:turn:{turn.SegmentId:N}", artifact.Provenance);
            Assert.Equal(1, engine.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task HandleAsync_ExperimentalEngine_SkipsWithExperimentalGate()
    {
        string directory = CreateTempDirectory();
        try
        {
            var mediaAsset = MakeMediaAsset();
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var engine = new FakeLipSynthesisEngine
            {
                OutputDirectory = Path.Join(directory, "engine-out"),
                Experimental = true
            };

            var handler = MakeHandler(artifactStore, stageRunStore, engine);

            var result = await handler.HandleAsync(
                MakeRequest(mediaAsset, [MakeTurn()], directory), TestContext.Current.CancellationToken);

            var seg = Assert.Single(result.Segments);
            Assert.Equal(LipSynthesisSegmentStatus.SkippedExperimentalGate, seg.Status);
            Assert.Equal(0, engine.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task HandleAsync_ExperimentalEngine_AllowsWhenExplicitOptIn()
    {
        string directory = CreateTempDirectory();
        try
        {
            var mediaAsset = MakeMediaAsset();
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var mediaRepo = new FakeMediaAssetRepository();
            var engine = new FakeLipSynthesisEngine
            {
                OutputDirectory = Path.Join(directory, "engine-out"),
                Experimental = true
            };

            var turn = MakeTurn();
            var handler = MakeHandler(artifactStore, stageRunStore, engine, mediaAssetRepository: mediaRepo);

            var result = await handler.HandleAsync(
                MakeRequest(mediaAsset, [turn], directory, allowExperimentalExecution: true),
                TestContext.Current.CancellationToken);

            var seg = Assert.Single(result.Segments);
            Assert.Equal(LipSynthesisSegmentStatus.Synthesized, seg.Status);
            Assert.True(seg.UsedExperimentalProvider);
            Assert.Equal(1, engine.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Disabled
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenDisabled_SkipsWithoutCallingEngineOrFaceProviders()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var engine = new FakeLipSynthesisEngine();
            var faceDetector = new FakeFaceDetector();
            var handler = MakeHandler(artifactStore, stageRunStore, engine, faceDetector);

            var result = await handler.HandleAsync(
                MakeRequest(MakeMediaAsset(), [MakeTurn()], isEnabled: false),
                TestContext.Current.CancellationToken);

            Assert.Empty(result.Segments);
            Assert.Equal(StageRunStatus.Skipped, result.StageRun.Status);
            Assert.Equal(0, engine.CallCount);
            Assert.Equal(0, faceDetector.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // License gate
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenLicenseNotApproved_SkipsLicenseGatePerTurnAndNeverCallsEngine()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var engine = new FakeLipSynthesisEngine();
            var handler = MakeHandler(artifactStore, stageRunStore, engine);

            var turns = new[] { MakeTurn(), MakeTurn(4.0, 6.0) };
            var result = await handler.HandleAsync(
                MakeRequest(MakeMediaAsset(), turns, directory, isLicenseApproved: false),
                TestContext.Current.CancellationToken);

            Assert.Equal(2, result.Segments.Count);
            Assert.All(result.Segments, s => Assert.Equal(LipSynthesisSegmentStatus.SkippedLicenseGate, s.Status));
            Assert.Equal(StageRunStatus.Skipped, result.StageRun.Status);
            Assert.Equal(0, engine.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task HandleAsync_WhenLicenseNotApproved_BlocksEvenWithAllowExperimentalExecution()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var engine = new FakeLipSynthesisEngine();
            var handler = MakeHandler(artifactStore, stageRunStore, engine);

            var turns = new[] { MakeTurn(), MakeTurn(4.0, 6.0) };
            var result = await handler.HandleAsync(
                MakeRequest(MakeMediaAsset(), turns, directory,
                    isLicenseApproved: false, allowExperimentalExecution: true),
                TestContext.Current.CancellationToken);

            Assert.Equal(2, result.Segments.Count);
            Assert.All(result.Segments, s => Assert.Equal(LipSynthesisSegmentStatus.SkippedLicenseGate, s.Status));
            Assert.Equal(StageRunStatus.Skipped, result.StageRun.Status);
            Assert.Equal(0, engine.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Dubbed mix prerequisite
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenDubbedMixMissing_SkipsWithoutCallingEngine()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var engine = new FakeLipSynthesisEngine();
            var handler = MakeHandler(artifactStore, stageRunStore, engine);

            var result = await handler.HandleAsync(
                MakeRequest(MakeMediaAsset(), [MakeTurn()], dubbedAudioPath: string.Empty),
                TestContext.Current.CancellationToken);

            Assert.Empty(result.Segments);
            Assert.Equal(StageRunStatus.Skipped, result.StageRun.Status);
            Assert.Contains("dubbed-speech driver", result.StageRun.FailureReason, StringComparison.OrdinalIgnoreCase);
            Assert.Equal(0, engine.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Runtime unavailable
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenEngineUnavailable_SkipsRuntimeUnavailable()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var engine = new FakeLipSynthesisEngine { Available = false };
            var handler = MakeHandler(artifactStore, stageRunStore, engine);

            var result = await handler.HandleAsync(
                MakeRequest(MakeMediaAsset(), [MakeTurn()], directory), TestContext.Current.CancellationToken);

            var seg = Assert.Single(result.Segments);
            Assert.Equal(LipSynthesisSegmentStatus.SkippedRuntimeUnavailable, seg.Status);
            Assert.Equal(StageRunStatus.Skipped, result.StageRun.Status);
            Assert.Equal(0, engine.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // No face
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenNoFaceDetected_SkipsNoFaceAndPreservesSource()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var mediaRepo = new FakeMediaAssetRepository();
            var engine = new FakeLipSynthesisEngine();
            var faceDetector = new FakeFaceDetector { FaceFound = false };
            var handler = MakeHandler(artifactStore, stageRunStore, engine, faceDetector, mediaAssetRepository: mediaRepo);

            var result = await handler.HandleAsync(
                MakeRequest(MakeMediaAsset(), [MakeTurn()], directory), TestContext.Current.CancellationToken);

            var seg = Assert.Single(result.Segments);
            Assert.Equal(LipSynthesisSegmentStatus.SkippedNoFace, seg.Status);
            Assert.Null(seg.PatchedClipRelativePath);
            Assert.Equal(StageRunStatus.Skipped, result.StageRun.Status);
            Assert.Equal(0, engine.CallCount);
            // No patched-video artifact registered — original frames preserved.
            Assert.DoesNotContain(mediaRepo.Artifacts, a => a.Kind == ArtifactKind.LipSynthesisTake);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Low confidence
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenFaceConfidenceLow_SkipsLowConfidence()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var engine = new FakeLipSynthesisEngine();
            var faceDetector = new FakeFaceDetector { Confidence = 0.40 };
            var handler = MakeHandler(artifactStore, stageRunStore, engine, faceDetector);

            var result = await handler.HandleAsync(
                MakeRequest(MakeMediaAsset(), [MakeTurn()], directory), TestContext.Current.CancellationToken);

            Assert.Equal(LipSynthesisSegmentStatus.SkippedLowConfidence, Assert.Single(result.Segments).Status);
            Assert.Equal(0, engine.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task HandleAsync_WhenFaceConfidenceMatchesDefaultMinimum_Synthesizes()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var engine = new FakeLipSynthesisEngine { OutputDirectory = Path.Join(directory, "engine-out") };
            var faceDetector = new FakeFaceDetector { Confidence = 0.70 };
            var handler = MakeHandler(artifactStore, stageRunStore, engine, faceDetector);

            var result = await handler.HandleAsync(
                MakeRequest(MakeMediaAsset(), [MakeTurn()], directory), TestContext.Current.CancellationToken);

            Assert.Equal(LipSynthesisSegmentStatus.Synthesized, Assert.Single(result.Segments).Status);
            Assert.Equal(1, engine.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Non-frontal pose
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenPoseNonFrontal_SkipsNonFrontal()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var engine = new FakeLipSynthesisEngine();
            var pose = new FakeFacePoseEstimator { YawDegrees = 45.0 };
            var handler = MakeHandler(artifactStore, stageRunStore, engine, pose: pose);

            var result = await handler.HandleAsync(
                MakeRequest(MakeMediaAsset(), [MakeTurn()], directory), TestContext.Current.CancellationToken);

            Assert.Equal(LipSynthesisSegmentStatus.SkippedNonFrontal, Assert.Single(result.Segments).Status);
            Assert.Equal(0, engine.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Occlusion / unstable crop
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenMouthOccluded_SkipsOccluded()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var engine = new FakeLipSynthesisEngine();
            var landmarks = new FakeFaceLandmarkProvider { MouthOccluded = true };
            var handler = MakeHandler(artifactStore, stageRunStore, engine, landmarks: landmarks);

            var result = await handler.HandleAsync(
                MakeRequest(MakeMediaAsset(), [MakeTurn()], directory), TestContext.Current.CancellationToken);

            Assert.Equal(LipSynthesisSegmentStatus.SkippedOccluded, Assert.Single(result.Segments).Status);
            Assert.Equal(0, engine.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task HandleAsync_WhenLandmarksUnstable_SkipsUnstableCrop()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var landmarks = new FakeFaceLandmarkProvider { IsStable = false };
            var handler = MakeHandler(artifactStore, stageRunStore, landmarks: landmarks);

            var result = await handler.HandleAsync(
                MakeRequest(MakeMediaAsset(), [MakeTurn()], directory), TestContext.Current.CancellationToken);

            Assert.Equal(LipSynthesisSegmentStatus.SkippedUnstableCrop, Assert.Single(result.Segments).Status);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Failure
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenEngineReturnsFailed_SegmentFailedAndStageFails()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var engine = new FakeLipSynthesisEngine { StatusToReturn = LipSynthesisEngineStatus.Failed };
            var handler = MakeHandler(artifactStore, stageRunStore, engine);

            var result = await handler.HandleAsync(
                MakeRequest(MakeMediaAsset(), [MakeTurn()], directory), TestContext.Current.CancellationToken);

            var seg = Assert.Single(result.Segments);
            Assert.Equal(LipSynthesisSegmentStatus.Failed, seg.Status);
            Assert.NotNull(seg.FailureReason);
            Assert.Equal(StageRunStatus.Failed, result.StageRun.Status);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task HandleAsync_WhenEngineThrows_StageRunFailsAndRethrows()
    {
        string directory = CreateTempDirectory();
        try
        {
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var engine = new FakeLipSynthesisEngine { ThrowOnSynthesize = true };
            var handler = MakeHandler(artifactStore, stageRunStore, engine);

            await Assert.ThrowsAsync<InvalidOperationException>(() =>
                handler.HandleAsync(MakeRequest(MakeMediaAsset(), [MakeTurn()], directory), TestContext.Current.CancellationToken));

            Assert.Equal(StageRunStatus.Failed, stageRunStore.All[^1].Status);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Partial: one turn synthesized, one skipped (mixed) → PartiallyCompleted
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_MixedOutcomes_StagePartiallyCompletedAndPreservesSkippedTurns()
    {
        string directory = CreateTempDirectory();
        try
        {
            var mediaAsset = MakeMediaAsset();
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var mediaRepo = new FakeMediaAssetRepository();
            var engine = new FakeLipSynthesisEngine { OutputDirectory = Path.Join(directory, "engine-out") };

            // Detector returns no face for the SECOND call only: first turn synthesizes, second skips.
            var faceDetector = new SequencedFaceDetector([true, false]);
            var handler = new LipSynthesisStageHandler(
                engine, faceDetector, new FakeFaceLandmarkProvider(), new FakeFacePoseEstimator(),
                artifactStore, stageRunStore, new FakeLipSynthesisSegmentRepository(), mediaAssetRepository: mediaRepo);

            var turns = new[] { MakeTurn(1.0, 3.0), MakeTurn(4.0, 6.0) };
            var result = await handler.HandleAsync(
                MakeRequest(mediaAsset, turns, directory), TestContext.Current.CancellationToken);

            Assert.Equal(2, result.Segments.Count);
            Assert.Equal(LipSynthesisSegmentStatus.Synthesized, result.Segments[0].Status);
            Assert.Equal(LipSynthesisSegmentStatus.SkippedNoFace, result.Segments[1].Status);
            Assert.Equal(StageRunStatus.PartiallyCompleted, result.StageRun.Status);

            // Exactly one patched-video artifact (for the synthesized turn); the skipped turn keeps original frames.
            Assert.Single(mediaRepo.Artifacts, a => a.Kind == ArtifactKind.LipSynthesisTake);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    private sealed class SequencedFaceDetector(IReadOnlyList<bool> faceFoundByCall) : IFaceDetector
    {
        private int _index;
        public bool IsAvailable => true;

        public Task<FaceDetectionResult> DetectPrimaryFaceAsync(
            FaceAnalysisRequest request, CancellationToken cancellationToken)
        {
            bool found = _index < faceFoundByCall.Count && faceFoundByCall[_index];
            _index++;
            return Task.FromResult(new FaceDetectionResult(
                FaceFound: found,
                Confidence: found ? 0.95 : 0d,
                PrimaryFace: found ? new FaceRegion(0.25, 0.25, 0.5, 0.5) : null));
        }
    }
}
