using NSubstitute;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.LipSync;
using Trackdub.Domain.Media;
using Trackdub.Domain.Tts;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed class LipSyncStageHandlerTests
{
    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    private static string CreateTempDirectory()
    {
        var path = Path.Join(Path.GetTempPath(), $"lipsync-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(path);
        return path;
    }

    private static MediaAsset MakeMediaAsset(Guid? projectId = null) => new(
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
        HasVideo: true,
        CreatedAtUtc: DateTimeOffset.UtcNow);

    private static LipSyncStageHandler MakeHandler(
        FakeArtifactStore artifactStore,
        FakeProjectStageRunStore stageRunStore,
        FakeForcedAligner? aligner = null,
        FakePhonemeStretchService? stretchService = null,
        FakeAudioClipExtractor? clipExtractor = null,
        IMediaAssetRepository? mediaAssetRepository = null)
    {
        return new LipSyncStageHandler(
            forcedAligner: aligner ?? new FakeForcedAligner(),
            phonemeTimingPlanner: new FakePhonemeTimingPlanner(),
            phonemeStretchService: stretchService ?? new FakePhonemeStretchService(),
            artifactStore: artifactStore,
            stageRunStore: stageRunStore,
            segmentRepository: new FakeLipSyncSegmentRepository(),
            audioClipExtractor: clipExtractor,
            mediaAssetRepository: mediaAssetRepository);
    }

    // ---------------------------------------------------------------------------
    // Success path
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WithEnabledStageAndSuccessfulAlignment_ReturnsAlignedSegments()
    {
        string directory = CreateTempDirectory();
        try
        {
            var projectId = Guid.NewGuid();
            var mediaAsset = MakeMediaAsset(projectId);
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();

            var aligner = new FakeForcedAligner
            {
                StatusToReturn = ForcedAlignmentStatus.Success,
                OverallConfidence = 0.95,
                PhonesToReturn = [new PhonemeTiming("AH", "arpabet", TimeSpan.Zero, TimeSpan.FromMilliseconds(100), 0.9)]
            };
            var stretchService = new FakePhonemeStretchService
            {
                AlignedDurationToReturn = TimeSpan.FromSeconds(2.0)
            };

            var relPath = $"tts/{Guid.NewGuid():N}.wav";
            artifactStore.Seed(relPath);
            var artifactId = Guid.NewGuid();
            var translatedSegmentId = Guid.NewGuid();

            var take = TtsTake.CreateStock(
                projectId: projectId,
                voiceAssignmentId: Guid.NewGuid(),
                translatedSegmentId: translatedSegmentId) with
            { ArtifactId = artifactId };

            var artifact = new ProjectArtifact(
                Id: artifactId,
                ProjectId: projectId,
                MediaAssetId: mediaAsset.Id,
                Kind: ArtifactKind.TtsTake,
                RelativePath: relPath,
                Sha256: "abc",
                SizeBytes: 0L,
                DurationSeconds: null,
                SampleRate: null,
                ChannelCount: null,
                CreatedAtUtc: DateTimeOffset.UtcNow);

            var handler = MakeHandler(artifactStore, stageRunStore, aligner, stretchService);

            var result = await handler.HandleAsync(
                new LipSyncStageRequest(
                    ProjectId: projectId,
                    MediaAsset: mediaAsset,
                    TtsTakes: [take],
                    ExistingArtifacts: [artifact]),
                TestContext.Current.CancellationToken);

            // Stretch is short-circuited to Partial because source phonemes are not yet
            // available (no forced alignment on the source audio). The segment reports
            // Partial with a skip reason; the stretch service is never called.
            Assert.Equal(StageRunStatus.PartiallyCompleted, result.StageRun.Status);
            Assert.Single(result.Segments);
            Assert.Equal(LipSyncSegmentStatus.Partial, result.Segments[0].Status);
            Assert.Equal(translatedSegmentId, result.Segments[0].SegmentId);
            Assert.Null(result.Segments[0].AlignedTtsDuration);
            Assert.NotNull(result.Segments[0].SkipReason);
            Assert.Equal(1, aligner.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Skip: stage disabled
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenStageDisabled_SkipsWithoutCallingAligner()
    {
        string directory = CreateTempDirectory();
        try
        {
            var mediaAsset = MakeMediaAsset();
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var aligner = new FakeForcedAligner();
            var handler = MakeHandler(artifactStore, stageRunStore, aligner);

            var result = await handler.HandleAsync(
                new LipSyncStageRequest(
                    ProjectId: mediaAsset.ProjectId,
                    MediaAsset: mediaAsset,
                    TtsTakes: [TtsTake.CreateStock(mediaAsset.ProjectId, Guid.NewGuid())],
                    ExistingArtifacts: [],
                    IsEnabled: false),
                TestContext.Current.CancellationToken);

            Assert.Equal(StageRunStatus.Skipped, result.StageRun.Status);
            Assert.Empty(result.Segments);
            Assert.Equal(0, aligner.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Skip: take has no artifact
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenTakeHasNoArtifact_SegmentSkippedNoPhonemes()
    {
        string directory = CreateTempDirectory();
        try
        {
            var mediaAsset = MakeMediaAsset();
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var aligner = new FakeForcedAligner();
            var handler = MakeHandler(artifactStore, stageRunStore, aligner);

            // Take has ArtifactId = null → ProcessTakeAsync skips before calling aligner
            var take = TtsTake.CreateStock(
                projectId: mediaAsset.ProjectId,
                voiceAssignmentId: Guid.NewGuid(),
                translatedSegmentId: Guid.NewGuid());

            var result = await handler.HandleAsync(
                new LipSyncStageRequest(
                    ProjectId: mediaAsset.ProjectId,
                    MediaAsset: mediaAsset,
                    TtsTakes: [take],
                    ExistingArtifacts: []),
                TestContext.Current.CancellationToken);

            Assert.Equal(LipSyncSegmentStatus.SkippedNoPhonemes, result.Segments[0].Status);
            Assert.Equal(0, aligner.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Skip: aligner returns Skipped
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenAlignerReturnsSkipped_SegmentIsSkippedLowConfidence()
    {
        string directory = CreateTempDirectory();
        try
        {
            var projectId = Guid.NewGuid();
            var mediaAsset = MakeMediaAsset(projectId);
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();

            var aligner = new FakeForcedAligner
            {
                StatusToReturn = ForcedAlignmentStatus.Skipped,
                SkipReasonToReturn = "Low confidence"
            };

            var relPath = $"tts/{Guid.NewGuid():N}.wav";
            artifactStore.Seed(relPath);
            var artifactId = Guid.NewGuid();

            var take = TtsTake.CreateStock(
                projectId: projectId,
                voiceAssignmentId: Guid.NewGuid(),
                translatedSegmentId: Guid.NewGuid()) with
            { ArtifactId = artifactId };

            var artifact = new ProjectArtifact(
                Id: artifactId, ProjectId: projectId, MediaAssetId: mediaAsset.Id,
                Kind: ArtifactKind.TtsTake, RelativePath: relPath,
                Sha256: "abc", SizeBytes: 0L, DurationSeconds: null,
                SampleRate: null, ChannelCount: null, CreatedAtUtc: DateTimeOffset.UtcNow);

            var handler = MakeHandler(artifactStore, stageRunStore, aligner);

            var result = await handler.HandleAsync(
                new LipSyncStageRequest(
                    ProjectId: projectId,
                    MediaAsset: mediaAsset,
                    TtsTakes: [take],
                    ExistingArtifacts: [artifact]),
                TestContext.Current.CancellationToken);

            Assert.Equal(LipSyncSegmentStatus.SkippedLowConfidence, result.Segments[0].Status);
            Assert.Equal("Low confidence", result.Segments[0].SkipReason);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Skip: stretch service returns null (unsafe ratio)
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenStretchServiceReturnsNull_SegmentSkippedUnsafeStretchRatio()
    {
        string directory = CreateTempDirectory();
        try
        {
            var projectId = Guid.NewGuid();
            var mediaAsset = MakeMediaAsset(projectId);
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();

            var aligner = new FakeForcedAligner
            {
                StatusToReturn = ForcedAlignmentStatus.Success,
                OverallConfidence = 0.95,
                PhonesToReturn = [new PhonemeTiming("AH", "arpabet", TimeSpan.Zero, TimeSpan.FromMilliseconds(100), 0.9)]
            };
            var stretchService = new FakePhonemeStretchService { ReturnNull = true };

            var relPath = $"tts/{Guid.NewGuid():N}.wav";
            artifactStore.Seed(relPath);
            var artifactId = Guid.NewGuid();

            var take = TtsTake.CreateStock(
                projectId: projectId,
                voiceAssignmentId: Guid.NewGuid(),
                translatedSegmentId: Guid.NewGuid()) with
            { ArtifactId = artifactId };

            var artifact = new ProjectArtifact(
                Id: artifactId, ProjectId: projectId, MediaAssetId: mediaAsset.Id,
                Kind: ArtifactKind.TtsTake, RelativePath: relPath,
                Sha256: "abc", SizeBytes: 0L, DurationSeconds: null,
                SampleRate: null, ChannelCount: null, CreatedAtUtc: DateTimeOffset.UtcNow);

            var handler = MakeHandler(artifactStore, stageRunStore, aligner, stretchService);

            var result = await handler.HandleAsync(
                new LipSyncStageRequest(
                    ProjectId: projectId,
                    MediaAsset: mediaAsset,
                    TtsTakes: [take],
                    ExistingArtifacts: [artifact]),
                TestContext.Current.CancellationToken);

            // With source phonemes unavailable, the stretch path is never reached;
            // the segment returns Partial with a skip reason instead.
            Assert.Equal(LipSyncSegmentStatus.Partial, result.Segments[0].Status);
            Assert.NotNull(result.Segments[0].SkipReason);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Failure path: aligner throws
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenAlignerThrows_StageRunFailsAndRethrows()
    {
        string directory = CreateTempDirectory();
        try
        {
            var projectId = Guid.NewGuid();
            var mediaAsset = MakeMediaAsset(projectId);
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();

            var aligner = new FakeForcedAligner { ThrowOnAlign = true };

            var relPath = $"tts/{Guid.NewGuid():N}.wav";
            artifactStore.Seed(relPath);
            var artifactId = Guid.NewGuid();

            var take = TtsTake.CreateStock(
                projectId: projectId,
                voiceAssignmentId: Guid.NewGuid(),
                translatedSegmentId: Guid.NewGuid()) with
            { ArtifactId = artifactId };

            var artifact = new ProjectArtifact(
                Id: artifactId, ProjectId: projectId, MediaAssetId: mediaAsset.Id,
                Kind: ArtifactKind.TtsTake, RelativePath: relPath,
                Sha256: "abc", SizeBytes: 0L, DurationSeconds: null,
                SampleRate: null, ChannelCount: null, CreatedAtUtc: DateTimeOffset.UtcNow);

            var handler = MakeHandler(artifactStore, stageRunStore, aligner);

            await Assert.ThrowsAsync<InvalidOperationException>(() =>
                handler.HandleAsync(
                    new LipSyncStageRequest(
                        ProjectId: projectId,
                        MediaAsset: mediaAsset,
                        TtsTakes: [take],
                        ExistingArtifacts: [artifact]),
                    TestContext.Current.CancellationToken));

            Assert.Equal(StageRunStatus.Failed, stageRunStore.All.Last().Status);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Cancellation path
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenCanceled_StageRunCanceledAndRethrows()
    {
        string directory = CreateTempDirectory();
        try
        {
            using var cts = new CancellationTokenSource();
            await cts.CancelAsync();

            var mediaAsset = MakeMediaAsset();
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var handler = MakeHandler(artifactStore, stageRunStore);

            await Assert.ThrowsAsync<OperationCanceledException>(() =>
                handler.HandleAsync(
                    new LipSyncStageRequest(
                        ProjectId: mediaAsset.ProjectId,
                        MediaAsset: mediaAsset,
                        TtsTakes: [TtsTake.CreateStock(mediaAsset.ProjectId, Guid.NewGuid())],
                        ExistingArtifacts: [],
                        IsEnabled: true),
                    cts.Token));

            Assert.Equal(StageRunStatus.Canceled, stageRunStore.All.Last().Status);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public Task Final_take_extraction_cancellation_records_cancelled(bool cancelCaller)
        => AssertFinalTakeCancellationAsync(duringExtraction: true, cancelCaller: cancelCaller);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public Task Final_take_stretch_cancellation_records_cancelled(bool cancelCaller)
        => AssertFinalTakeCancellationAsync(duringExtraction: false, cancelCaller: cancelCaller);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public Task Final_take_fallback_after_caller_cancellation_records_cancelled(bool duringExtraction)
        => AssertFinalTakeCancellationAsync(duringExtraction, cancelCaller: true, throwCancellation: false);

    private static async Task AssertFinalTakeCancellationAsync(
        bool duringExtraction,
        bool cancelCaller,
        bool throwCancellation = true)
    {
        string directory = CreateTempDirectory();
        try
        {
            using var cts = new CancellationTokenSource();
            var mediaAsset = MakeMediaAsset() with { SourceFilePath = Path.Join(directory, "original.mp4") };
            File.WriteAllBytes(mediaAsset.SourceFilePath, [10, 11, 12]);
            string sourcePath = Path.Join(directory, "source.wav");
            File.WriteAllBytes(sourcePath, [3, 4, 5]);
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var mediaRepo = new FakeMediaAssetRepository();
            var aligner = new FakeForcedAligner
            {
                PhonesToReturn = [new PhonemeTiming("AH", "arpabet", TimeSpan.Zero, TimeSpan.FromMilliseconds(100), 0.9)]
            };
            var takes = Enumerable.Range(0, 2).Select(_ =>
                TtsTake.CreateStock(mediaAsset.ProjectId, Guid.NewGuid(), Guid.NewGuid()) with
                { ArtifactId = Guid.NewGuid() }).ToArray();
            var artifacts = takes.Select(take => new ProjectArtifact(
                Id: take.ArtifactId!.Value, ProjectId: mediaAsset.ProjectId, MediaAssetId: mediaAsset.Id,
                Kind: ArtifactKind.TtsTake, RelativePath: $"tts/{take.Id:N}.wav",
                Sha256: "abc", SizeBytes: 3, DurationSeconds: null,
                SampleRate: null, ChannelCount: null, CreatedAtUtc: DateTimeOffset.UtcNow)).ToArray();
            foreach (var artifact in artifacts)
            {
                artifactStore.Seed(artifact.RelativePath, [1, 2, 3]);
            }

            int extractionCalls = 0;
            int stretchCalls = 0;
            var extractor = new FakeAudioClipExtractor();
            var clipExtractor = Substitute.For<IAudioClipExtractor>();
            clipExtractor.ExtractAsync(Arg.Any<string>(), Arg.Any<double>(), Arg.Any<double>(),
                Arg.Any<string>(), Arg.Any<CancellationToken>()).Returns(call =>
                {
                    if (++extractionCalls == 2 && duringExtraction)
                    {
                        if (cancelCaller) cts.Cancel();
                        if (throwCancellation) throw new OperationCanceledException(cts.Token);
                        throw new InvalidOperationException("Extractor failed after caller cancellation.");
                    }

                    return extractor.ExtractAsync(call.ArgAt<string>(0), call.ArgAt<double>(1),
                        call.ArgAt<double>(2), call.ArgAt<string>(3), call.ArgAt<CancellationToken>(4));
                });
            var stretcher = new FakePhonemeStretchService();
            var stretchService = Substitute.For<IPhonemeStretchService>();
            stretchService.StretchAsync(Arg.Any<string>(), Arg.Any<string>(),
                Arg.Any<IReadOnlyList<PhonemeStretchPlan>>(), Arg.Any<CancellationToken>()).Returns(call =>
                {
                    if (++stretchCalls == 2 && !duringExtraction)
                    {
                        if (cancelCaller) cts.Cancel();
                        if (throwCancellation) throw new OperationCanceledException(cts.Token);
                        return Task.FromResult<PhonemeStretchResult?>(null);
                    }

                    var result = stretcher.StretchAsync(call.ArgAt<string>(0), call.ArgAt<string>(1),
                        call.ArgAt<IReadOnlyList<PhonemeStretchPlan>>(2), call.ArgAt<CancellationToken>(3));
                    File.WriteAllBytes(call.ArgAt<string>(1), [7, 8, 9]);
                    return result;
                });
            var segmentRepository = new FakeLipSyncSegmentRepository();
            var handler = new LipSyncStageHandler(aligner, new FakePhonemeTimingPlanner(),
                stretchService, artifactStore, stageRunStore, segmentRepository,
                mediaAssetRepository: mediaRepo, audioClipExtractor: clipExtractor);
            var request = new LipSyncStageRequest(
                mediaAsset.ProjectId, mediaAsset, takes, artifacts,
                SegmentSourceTimingMap: takes.ToDictionary(take => take.TranslatedSegmentId!.Value,
                    _ => new SegmentSourceTiming(0, 1)),
                SourceSegmentTranscriptMap: takes.ToDictionary(take => take.TranslatedSegmentId!.Value, _ => "hello"),
                SourceAudioPath: sourcePath);

            Task<LipSyncStageResult> runTask = handler.HandleAsync(request, cts.Token);

            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => runTask);
            Assert.False(runTask.IsCompletedSuccessfully);
            Assert.Equal(StageRunStatus.Canceled, Assert.Single(stageRunStore.All).Status);
            var persistedSegment = Assert.Single(await segmentRepository.GetByStageRunAsync(
                Assert.Single(stageRunStore.All).Id, TestContext.Current.CancellationToken));
            Assert.Equal(takes[0].TranslatedSegmentId, persistedSegment.SegmentId);
            Assert.Equal(LipSyncSegmentStatus.Aligned, persistedSegment.Status);
            Assert.Equal(2, extractionCalls);
            Assert.Equal(duringExtraction ? 1 : 2, stretchCalls);
            var accepted = Assert.Single(mediaRepo.Artifacts);
            Assert.Equal(ArtifactKind.LipSyncTake, accepted.Kind);
            Assert.Equal($"lipsync:take:{takes[0].Id:N}", accepted.Provenance);
            Assert.Equal(new byte[] { 7, 8, 9 }, File.ReadAllBytes(artifactStore.GetPath(accepted.RelativePath)));
            foreach (var artifact in artifacts)
            {
                Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(artifactStore.GetPath(artifact.RelativePath)));
            }
            Assert.Equal(new byte[] { 3, 4, 5 }, File.ReadAllBytes(sourcePath));
            Assert.Equal(new byte[] { 10, 11, 12 }, File.ReadAllBytes(mediaAsset.SourceFilePath));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Failed_extraction_leaves_no_temp_segment_file(bool throwCancellation)
    {
        // Track only the paths this handler asked the extractor to write; scanning the shared temp
        // folder races with other tests that create trackdub-src-seg-* files in parallel.
        var extractionOutputPaths = new System.Collections.Concurrent.ConcurrentQueue<string>();
        string directory = CreateTempDirectory();
        try
        {
            using var cts = new CancellationTokenSource();
            var mediaAsset = MakeMediaAsset() with { SourceFilePath = Path.Join(directory, "original.mp4") };
            File.WriteAllBytes(mediaAsset.SourceFilePath, [10, 11, 12]);
            string sourcePath = Path.Join(directory, "source.wav");
            File.WriteAllBytes(sourcePath, [3, 4, 5]);
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var mediaRepo = new FakeMediaAssetRepository();
            var aligner = new FakeForcedAligner
            {
                PhonesToReturn = [new PhonemeTiming("AH", "arpabet", TimeSpan.Zero, TimeSpan.FromMilliseconds(100), 0.9)]
            };
            var takes = Enumerable.Range(0, 1).Select(_ =>
                TtsTake.CreateStock(mediaAsset.ProjectId, Guid.NewGuid(), Guid.NewGuid()) with
                { ArtifactId = Guid.NewGuid() }).ToArray();
            var artifacts = takes.Select(take => new ProjectArtifact(
                Id: take.ArtifactId!.Value, ProjectId: mediaAsset.ProjectId, MediaAssetId: mediaAsset.Id,
                Kind: ArtifactKind.TtsTake, RelativePath: $"tts/{take.Id:N}.wav",
                Sha256: "abc", SizeBytes: 3, DurationSeconds: null,
                SampleRate: null, ChannelCount: null, CreatedAtUtc: DateTimeOffset.UtcNow)).ToArray();
            foreach (var artifact in artifacts)
            {
                artifactStore.Seed(artifact.RelativePath, [1, 2, 3]);
            }
            var clipExtractor = Substitute.For<IAudioClipExtractor>();
            clipExtractor.ExtractAsync(Arg.Any<string>(), Arg.Any<double>(), Arg.Any<double>(),
                Arg.Any<string>(), Arg.Any<CancellationToken>()).Returns<Task>(call =>
                {
                    // Simulate a partially-created extraction, then fail.
                    extractionOutputPaths.Enqueue(call.ArgAt<string>(3));
                    File.WriteAllBytes(call.ArgAt<string>(3), [1]);
                    if (throwCancellation) throw new OperationCanceledException(cts.Token);
                    throw new InvalidOperationException("Extractor failed.");
                });
            var handler = new LipSyncStageHandler(aligner, new FakePhonemeTimingPlanner(),
                new FakePhonemeStretchService(), artifactStore, stageRunStore,
                new FakeLipSyncSegmentRepository(),
                mediaAssetRepository: mediaRepo, audioClipExtractor: clipExtractor);
            var request = new LipSyncStageRequest(
                mediaAsset.ProjectId, mediaAsset, takes, artifacts,
                SegmentSourceTimingMap: takes.ToDictionary(take => take.TranslatedSegmentId!.Value,
                    _ => new SegmentSourceTiming(0, 1)),
                SourceSegmentTranscriptMap: takes.ToDictionary(take => take.TranslatedSegmentId!.Value, _ => "hello"),
                SourceAudioPath: sourcePath);

            if (throwCancellation)
            {
                await Assert.ThrowsAnyAsync<OperationCanceledException>(() => handler.HandleAsync(request, cts.Token));
            }
            else
            {
                await handler.HandleAsync(request, cts.Token);
            }

            Assert.NotEmpty(extractionOutputPaths);
            Assert.All(extractionOutputPaths, path => Assert.False(File.Exists(path), $"Leaked temp segment: {path}"));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public async Task Segment_save_ignores_cancellation_after_artifact_published()
    {
        string directory = CreateTempDirectory();
        try
        {
            using var cts = new CancellationTokenSource();
            var mediaAsset = MakeMediaAsset() with { SourceFilePath = Path.Join(directory, "original.mp4") };
            File.WriteAllBytes(mediaAsset.SourceFilePath, [10, 11, 12]);
            string sourcePath = Path.Join(directory, "source.wav");
            File.WriteAllBytes(sourcePath, [3, 4, 5]);
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var mediaRepo = new FakeMediaAssetRepository();
            var aligner = new FakeForcedAligner
            {
                PhonesToReturn = [new PhonemeTiming("AH", "arpabet", TimeSpan.Zero, TimeSpan.FromMilliseconds(100), 0.9)]
            };
            var takes = Enumerable.Range(0, 1).Select(_ =>
                TtsTake.CreateStock(mediaAsset.ProjectId, Guid.NewGuid(), Guid.NewGuid()) with
                { ArtifactId = Guid.NewGuid() }).ToArray();
            var artifacts = takes.Select(take => new ProjectArtifact(
                Id: take.ArtifactId!.Value, ProjectId: mediaAsset.ProjectId, MediaAssetId: mediaAsset.Id,
                Kind: ArtifactKind.TtsTake, RelativePath: $"tts/{take.Id:N}.wav",
                Sha256: "abc", SizeBytes: 3, DurationSeconds: null,
                SampleRate: null, ChannelCount: null, CreatedAtUtc: DateTimeOffset.UtcNow)).ToArray();
            foreach (var artifact in artifacts)
            {
                artifactStore.Seed(artifact.RelativePath, [1, 2, 3]);
            }
            var clipExtractor = Substitute.For<IAudioClipExtractor>();
            clipExtractor.ExtractAsync(Arg.Any<string>(), Arg.Any<double>(), Arg.Any<double>(),
                Arg.Any<string>(), Arg.Any<CancellationToken>()).Returns<Task>(call =>
                {
                    File.WriteAllBytes(call.ArgAt<string>(3), [7]);
                    return Task.CompletedTask;
                });
            var persisted = new FakeLipSyncSegmentRepository();
            var segmentRepository = new CancelOnSaveSegmentRepository(persisted, cts);
            var handler = new LipSyncStageHandler(aligner, new FakePhonemeTimingPlanner(),
                new FakePhonemeStretchService(), artifactStore, stageRunStore, segmentRepository,
                mediaAssetRepository: mediaRepo, audioClipExtractor: clipExtractor);
            var request = new LipSyncStageRequest(
                mediaAsset.ProjectId, mediaAsset, takes, artifacts,
                SegmentSourceTimingMap: takes.ToDictionary(take => take.TranslatedSegmentId!.Value,
                    _ => new SegmentSourceTiming(0, 1)),
                SourceSegmentTranscriptMap: takes.ToDictionary(take => take.TranslatedSegmentId!.Value, _ => "hello"),
                SourceAudioPath: sourcePath);

            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => handler.HandleAsync(request, cts.Token));

            // The published artifact's row is recorded despite the cancel; the run still cancels.
            Assert.Equal(StageRunStatus.Canceled, Assert.Single(stageRunStore.All).Status);
            var persistedSegment = Assert.Single(await persisted.GetByStageRunAsync(
                Assert.Single(stageRunStore.All).Id, TestContext.Current.CancellationToken));
            Assert.Equal(takes[0].TranslatedSegmentId, persistedSegment.SegmentId);
            Assert.Equal(LipSyncSegmentStatus.Partial, persistedSegment.Status);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    /// <summary>
    /// Simulates a user cancel racing a published artifact: cancels the operation
    /// as the segment save begins, then honors the token actually passed through,
    /// like a real repository would.
    /// </summary>
    private sealed class CancelOnSaveSegmentRepository(
        FakeLipSyncSegmentRepository inner,
        CancellationTokenSource operationCts) : ILipSyncSegmentRepository
    {
        public Task<IReadOnlyList<LipSyncSegment>> GetByProjectAsync(Guid projectId, CancellationToken cancellationToken) =>
            inner.GetByProjectAsync(projectId, cancellationToken);

        public Task<IReadOnlyList<LipSyncSegment>> GetByStageRunAsync(Guid stageRunId, CancellationToken cancellationToken) =>
            inner.GetByStageRunAsync(stageRunId, cancellationToken);

        public Task SaveAllAsync(Guid projectId, Guid stageRunId, IReadOnlyList<LipSyncSegment> segments, CancellationToken cancellationToken)
        {
            operationCts.Cancel();
            cancellationToken.ThrowIfCancellationRequested();
            return inner.SaveAllAsync(projectId, stageRunId, segments, cancellationToken);
        }
    }

    // ---------------------------------------------------------------------------
    // Artifact preservation: original TTS artifact not modified on skip
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenAlignmentSkipped_OriginalTtsArtifactIsPreserved()
    {
        string directory = CreateTempDirectory();
        try
        {
            var projectId = Guid.NewGuid();
            var mediaAsset = MakeMediaAsset(projectId);
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();

            var aligner = new FakeForcedAligner
            {
                StatusToReturn = ForcedAlignmentStatus.Skipped,
                SkipReasonToReturn = "low-confidence"
            };

            var relPath = $"tts/{Guid.NewGuid():N}.wav";
            byte[] originalContent = "RIFF"u8.ToArray(); // "RIFF"
            // Seed with specific content — Seed creates the file on disk with those bytes.
            artifactStore.Seed(relPath, originalContent);
            var absPath = artifactStore.GetPath(relPath);

            var artifactId = Guid.NewGuid();
            var take = TtsTake.CreateStock(
                projectId: projectId,
                voiceAssignmentId: Guid.NewGuid(),
                translatedSegmentId: Guid.NewGuid()) with
            { ArtifactId = artifactId };

            var artifact = new ProjectArtifact(
                Id: artifactId, ProjectId: projectId, MediaAssetId: mediaAsset.Id,
                Kind: ArtifactKind.TtsTake, RelativePath: relPath,
                Sha256: "abc", SizeBytes: 0L, DurationSeconds: null,
                SampleRate: null, ChannelCount: null, CreatedAtUtc: DateTimeOffset.UtcNow);

            var handler = MakeHandler(artifactStore, stageRunStore, aligner);

            await handler.HandleAsync(
                new LipSyncStageRequest(
                    ProjectId: projectId,
                    MediaAsset: mediaAsset,
                    TtsTakes: [take],
                    ExistingArtifacts: [artifact]),
                TestContext.Current.CancellationToken);

            // Original file must remain byte-for-byte identical.
            var actualContent = await File.ReadAllBytesAsync(absPath, TestContext.Current.CancellationToken);
            Assert.Equal(originalContent, actualContent);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Source alignment + stretch path: success (full stretch with source timing)
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WithSourceTimingAndClipExtractor_StretchPathProducesAlignedSegment()
    {
        string directory = CreateTempDirectory();
        try
        {
            var projectId = Guid.NewGuid();
            var mediaAsset = MakeMediaAsset(projectId);
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();

            var phonemes = new List<PhonemeTiming>
            {
                new("AH", "arpabet", TimeSpan.Zero, TimeSpan.FromMilliseconds(100), 0.9),
                new("T",  "arpabet", TimeSpan.FromMilliseconds(100), TimeSpan.FromMilliseconds(200), 0.85)
            };
            var aligner = new FakeForcedAligner
            {
                StatusToReturn = ForcedAlignmentStatus.Success,
                OverallConfidence = 0.95,
                PhonesToReturn = phonemes
            };
            var stretchService = new FakePhonemeStretchService
            {
                AlignedDurationToReturn = TimeSpan.FromSeconds(1.8)
            };
            var clipExtractor = new FakeAudioClipExtractor();

            var relPath = $"tts/{Guid.NewGuid():N}.wav";
            artifactStore.Seed(relPath);
            var artifactId = Guid.NewGuid();
            var translatedSegmentId = Guid.NewGuid();

            var take = TtsTake.CreateStock(
                projectId: projectId,
                voiceAssignmentId: Guid.NewGuid(),
                translatedSegmentId: translatedSegmentId) with
            { ArtifactId = artifactId };

            var artifact = new ProjectArtifact(
                Id: artifactId, ProjectId: projectId, MediaAssetId: mediaAsset.Id,
                Kind: ArtifactKind.TtsTake, RelativePath: relPath,
                Sha256: "abc", SizeBytes: 0L, DurationSeconds: null,
                SampleRate: null, ChannelCount: null, CreatedAtUtc: DateTimeOffset.UtcNow);

            // Build source timing map: segment → 1.0–4.0 s in source audio.
            var sourceTiming = new Dictionary<Guid, SegmentSourceTiming>
            {
                [translatedSegmentId] = new(1.0, 4.0)
            };
            string fakeSourcePath = Path.Join(directory, "source.wav");
            File.WriteAllBytes(fakeSourcePath, []);

            var handler = MakeHandler(artifactStore, stageRunStore, aligner, stretchService, clipExtractor);

            var result = await handler.HandleAsync(
                new LipSyncStageRequest(
                    ProjectId: projectId,
                    MediaAsset: mediaAsset,
                    TtsTakes: [take],
                    ExistingArtifacts: [artifact],
                    SegmentSourceTimingMap: sourceTiming,
                    SourceSegmentTranscriptMap: new Dictionary<Guid, string> { [translatedSegmentId] = "hello world" },
                    SourceAudioPath: fakeSourcePath),
                TestContext.Current.CancellationToken);

            // Clip extractor should have been called for the source segment.
            Assert.Equal(1, clipExtractor.CallCount);
            Assert.Equal(fakeSourcePath, clipExtractor.LastSourcePath);
            Assert.Equal(1.0, clipExtractor.LastStartSeconds);
            Assert.Equal(4.0, clipExtractor.LastEndSeconds);

            // Aligner called twice: once for TTS, once for source.
            Assert.Equal(2, aligner.CallCount);

            // Stretch service called once.
            Assert.Equal(1, stretchService.CallCount);

            // Segment should be Aligned with a filled AlignedTtsDuration.
            Assert.Single(result.Segments);
            var seg = result.Segments[0];
            Assert.Equal(LipSyncSegmentStatus.Aligned, seg.Status);
            Assert.Equal(TimeSpan.FromSeconds(1.8), seg.AlignedTtsDuration);
            Assert.Equal(translatedSegmentId, seg.SegmentId);
            Assert.Null(seg.FailureReason);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Source alignment + stretch path: clip extraction fails → Partial fallback
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task Noncancellation_extraction_error_keeps_partial_fallback()
    {
        string directory = CreateTempDirectory();
        try
        {
            var projectId = Guid.NewGuid();
            var mediaAsset = MakeMediaAsset(projectId);
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();

            var phonemes = new List<PhonemeTiming>
            {
                new("AH", "arpabet", TimeSpan.Zero, TimeSpan.FromMilliseconds(100), 0.9)
            };
            var aligner = new FakeForcedAligner
            {
                StatusToReturn = ForcedAlignmentStatus.Success,
                OverallConfidence = 0.95,
                PhonesToReturn = phonemes
            };
            var clipExtractor = new FakeAudioClipExtractor { ThrowOnExtract = true };

            var relPath = $"tts/{Guid.NewGuid():N}.wav";
            artifactStore.Seed(relPath);
            var artifactId = Guid.NewGuid();
            var translatedSegmentId = Guid.NewGuid();

            var take = TtsTake.CreateStock(
                projectId: projectId,
                voiceAssignmentId: Guid.NewGuid(),
                translatedSegmentId: translatedSegmentId) with
            { ArtifactId = artifactId };

            var artifact = new ProjectArtifact(
                Id: artifactId, ProjectId: projectId, MediaAssetId: mediaAsset.Id,
                Kind: ArtifactKind.TtsTake, RelativePath: relPath,
                Sha256: "abc", SizeBytes: 0L, DurationSeconds: null,
                SampleRate: null, ChannelCount: null, CreatedAtUtc: DateTimeOffset.UtcNow);

            var sourceTiming = new Dictionary<Guid, SegmentSourceTiming>
            {
                [translatedSegmentId] = new(0.0, 2.5)
            };
            string fakeSourcePath = Path.Join(directory, "source.wav");
            File.WriteAllBytes(fakeSourcePath, []);

            var handler = MakeHandler(artifactStore, stageRunStore, aligner, clipExtractor: clipExtractor);

            var result = await handler.HandleAsync(
                new LipSyncStageRequest(
                    ProjectId: projectId,
                    MediaAsset: mediaAsset,
                    TtsTakes: [take],
                    ExistingArtifacts: [artifact],
                    SegmentSourceTimingMap: sourceTiming,
                    SourceSegmentTranscriptMap: new Dictionary<Guid, string> { [translatedSegmentId] = "hello world" },
                    SourceAudioPath: fakeSourcePath),
                TestContext.Current.CancellationToken);

            var seg = result.Segments[0];
            Assert.Equal(LipSyncSegmentStatus.Partial, seg.Status);
            Assert.Equal(StageRunStatus.PartiallyCompleted, result.StageRun.Status);
            Assert.NotNull(seg.SkipReason);
            Assert.Contains("extraction failed", seg.SkipReason, StringComparison.OrdinalIgnoreCase);
            Assert.True(File.Exists(artifactStore.GetPath(relPath)));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Source alignment + stretch path: TTS aligner returns no phonemes → SkippedNoPhonemes
    // (source alignment is never reached, proving the pre-source guard)
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenTtsAlignmentReturnsNoPhonemes_SegmentSkippedBeforeSourceAlignment()
    {
        string directory = CreateTempDirectory();
        try
        {
            var projectId = Guid.NewGuid();
            var mediaAsset = MakeMediaAsset(projectId);
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();

            // Aligner returns success with empty phoneme list.
            var aligner = new FakeForcedAligner
            {
                StatusToReturn = ForcedAlignmentStatus.Success,
                OverallConfidence = 0.92,
                PhonesToReturn = []
            };
            var clipExtractor = new FakeAudioClipExtractor();

            var relPath = $"tts/{Guid.NewGuid():N}.wav";
            artifactStore.Seed(relPath);
            var artifactId = Guid.NewGuid();
            var translatedSegmentId = Guid.NewGuid();

            var take = TtsTake.CreateStock(
                projectId: projectId,
                voiceAssignmentId: Guid.NewGuid(),
                translatedSegmentId: translatedSegmentId) with
            { ArtifactId = artifactId };

            var artifact = new ProjectArtifact(
                Id: artifactId, ProjectId: projectId, MediaAssetId: mediaAsset.Id,
                Kind: ArtifactKind.TtsTake, RelativePath: relPath,
                Sha256: "abc", SizeBytes: 0L, DurationSeconds: null,
                SampleRate: null, ChannelCount: null, CreatedAtUtc: DateTimeOffset.UtcNow);

            var sourceTiming = new Dictionary<Guid, SegmentSourceTiming>
            {
                [translatedSegmentId] = new(0.5, 3.5)
            };
            string fakeSourcePath = Path.Join(directory, "source.wav");
            File.WriteAllBytes(fakeSourcePath, []);

            var handler = MakeHandler(artifactStore, stageRunStore, aligner, clipExtractor: clipExtractor);

            var result = await handler.HandleAsync(
                new LipSyncStageRequest(
                    ProjectId: projectId,
                    MediaAsset: mediaAsset,
                    TtsTakes: [take],
                    ExistingArtifacts: [artifact],
                    SegmentSourceTimingMap: sourceTiming,
                    SourceSegmentTranscriptMap: new Dictionary<Guid, string> { [translatedSegmentId] = "hello world" },
                    SourceAudioPath: fakeSourcePath),
                TestContext.Current.CancellationToken);

            // TTS phonemes are empty → SkippedNoPhonemes before source alignment runs.
            var seg = result.Segments[0];
            Assert.Equal(LipSyncSegmentStatus.SkippedNoPhonemes, seg.Status);

            // Clip extractor must not have been called (source alignment was never reached).
            Assert.Equal(0, clipExtractor.CallCount);

            // Aligner called only once (for TTS take).
            Assert.Equal(1, aligner.CallCount);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Artifact registration: LipSyncTake artifact saved with correct provenance
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenStretchSucceeds_RegistersLipSyncTakeArtifact()
    {
        string directory = CreateTempDirectory();
        try
        {
            var projectId = Guid.NewGuid();
            var mediaAsset = MakeMediaAsset(projectId);
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            var mediaRepo = new FakeMediaAssetRepository();

            var phonemes = new List<PhonemeTiming>
            {
                new("AH", "arpabet", TimeSpan.Zero, TimeSpan.FromMilliseconds(100), 0.9),
                new("T",  "arpabet", TimeSpan.FromMilliseconds(100), TimeSpan.FromMilliseconds(200), 0.85)
            };
            var aligner = new FakeForcedAligner
            {
                StatusToReturn = ForcedAlignmentStatus.Success,
                OverallConfidence = 0.95,
                PhonesToReturn = phonemes
            };
            var stretchService = new FakePhonemeStretchService
            {
                AlignedDurationToReturn = TimeSpan.FromSeconds(1.8)
            };
            var clipExtractor = new FakeAudioClipExtractor();

            var relPath = $"tts/{Guid.NewGuid():N}.wav";
            artifactStore.Seed(relPath);
            var artifactId = Guid.NewGuid();
            var translatedSegmentId = Guid.NewGuid();

            var take = TtsTake.CreateStock(
                projectId: projectId,
                voiceAssignmentId: Guid.NewGuid(),
                translatedSegmentId: translatedSegmentId) with
            { ArtifactId = artifactId };

            var artifact = new ProjectArtifact(
                Id: artifactId, ProjectId: projectId, MediaAssetId: mediaAsset.Id,
                Kind: ArtifactKind.TtsTake, RelativePath: relPath,
                Sha256: "abc", SizeBytes: 0L, DurationSeconds: null,
                SampleRate: null, ChannelCount: null, CreatedAtUtc: DateTimeOffset.UtcNow);

            var sourceTiming = new Dictionary<Guid, SegmentSourceTiming>
            {
                [translatedSegmentId] = new(1.0, 4.0)
            };
            string fakeSourcePath = Path.Join(directory, "source.wav");
            File.WriteAllBytes(fakeSourcePath, []);

            var handler = MakeHandler(
                artifactStore, stageRunStore, aligner, stretchService, clipExtractor,
                mediaAssetRepository: mediaRepo);

            await handler.HandleAsync(
                new LipSyncStageRequest(
                    ProjectId: projectId,
                    MediaAsset: mediaAsset,
                    TtsTakes: [take],
                    ExistingArtifacts: [artifact],
                    SegmentSourceTimingMap: sourceTiming,
                    SourceSegmentTranscriptMap: new Dictionary<Guid, string> { [translatedSegmentId] = "hello world" },
                    SourceAudioPath: fakeSourcePath),
                TestContext.Current.CancellationToken);

            // Exactly one artifact should be registered.
            Assert.Single(mediaRepo.Artifacts);

            var saved = mediaRepo.Artifacts[0];
            Assert.Equal(ArtifactKind.LipSyncTake, saved.Kind);
            Assert.Equal(projectId, saved.ProjectId);
            Assert.Equal(mediaAsset.Id, saved.MediaAssetId);

            // Provenance must encode the TTS take ID so MixPlanBuilder can join on it.
            Assert.Equal($"lipsync:take:{take.Id:N}", saved.Provenance);

            // Duration should reflect what the stretch service reported.
            Assert.Equal(1.8, saved.DurationSeconds);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Artifact registration: no repository provided → no exception, no artifact
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenMediaRepoAbsent_NoArtifactRegisteredAndNoException()
    {
        string directory = CreateTempDirectory();
        try
        {
            var projectId = Guid.NewGuid();
            var mediaAsset = MakeMediaAsset(projectId);
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();
            // No mediaAssetRepository → handler should not throw.

            var phonemes = new List<PhonemeTiming>
            {
                new("AH", "arpabet", TimeSpan.Zero, TimeSpan.FromMilliseconds(100), 0.9),
                new("T",  "arpabet", TimeSpan.FromMilliseconds(100), TimeSpan.FromMilliseconds(200), 0.85)
            };
            var aligner = new FakeForcedAligner
            {
                StatusToReturn = ForcedAlignmentStatus.Success,
                OverallConfidence = 0.95,
                PhonesToReturn = phonemes
            };
            var stretchService = new FakePhonemeStretchService
            {
                AlignedDurationToReturn = TimeSpan.FromSeconds(1.5)
            };
            var clipExtractor = new FakeAudioClipExtractor();

            var relPath = $"tts/{Guid.NewGuid():N}.wav";
            artifactStore.Seed(relPath);
            var artifactId = Guid.NewGuid();
            var translatedSegmentId = Guid.NewGuid();

            var take = TtsTake.CreateStock(
                projectId: projectId,
                voiceAssignmentId: Guid.NewGuid(),
                translatedSegmentId: translatedSegmentId) with
            { ArtifactId = artifactId };

            var artifact = new ProjectArtifact(
                Id: artifactId, ProjectId: projectId, MediaAssetId: mediaAsset.Id,
                Kind: ArtifactKind.TtsTake, RelativePath: relPath,
                Sha256: "abc", SizeBytes: 0L, DurationSeconds: null,
                SampleRate: null, ChannelCount: null, CreatedAtUtc: DateTimeOffset.UtcNow);

            var sourceTiming = new Dictionary<Guid, SegmentSourceTiming>
            {
                [translatedSegmentId] = new(0.5, 3.0)
            };
            string fakeSourcePath = Path.Join(directory, "source.wav");
            File.WriteAllBytes(fakeSourcePath, []);

            var handler = MakeHandler(
                artifactStore, stageRunStore, aligner, stretchService, clipExtractor,
                mediaAssetRepository: null);

            // Should not throw even though no repository is provided.
            var result = await handler.HandleAsync(
                new LipSyncStageRequest(
                    ProjectId: projectId,
                    MediaAsset: mediaAsset,
                    TtsTakes: [take],
                    ExistingArtifacts: [artifact],
                    SegmentSourceTimingMap: sourceTiming,
                    SourceSegmentTranscriptMap: new Dictionary<Guid, string> { [translatedSegmentId] = "hello world" },
                    SourceAudioPath: fakeSourcePath),
                TestContext.Current.CancellationToken);

            // Segment should still be Aligned.
            Assert.Equal(LipSyncSegmentStatus.Aligned, result.Segments[0].Status);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Transcript split: TTS alignment gets translated text, source alignment gets
    // original-language text, and routing options propagate to every aligner call.
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_TtsAlignmentUsesTranslatedText_SourceAlignmentUsesSourceText()
    {
        string directory = CreateTempDirectory();
        try
        {
            var projectId = Guid.NewGuid();
            var mediaAsset = MakeMediaAsset(projectId);
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();

            var phonemes = new List<PhonemeTiming>
            {
                new("AH", "arpabet", TimeSpan.Zero, TimeSpan.FromMilliseconds(100), 0.9)
            };
            var aligner = new FakeForcedAligner
            {
                StatusToReturn = ForcedAlignmentStatus.Success,
                OverallConfidence = 0.95,
                PhonesToReturn = phonemes
            };
            var clipExtractor = new FakeAudioClipExtractor();

            var relPath = $"tts/{Guid.NewGuid():N}.wav";
            artifactStore.Seed(relPath);
            var artifactId = Guid.NewGuid();
            var translatedSegmentId = Guid.NewGuid();

            var take = TtsTake.CreateStock(
                projectId: projectId,
                voiceAssignmentId: Guid.NewGuid(),
                translatedSegmentId: translatedSegmentId) with
            { ArtifactId = artifactId };

            var artifact = new ProjectArtifact(
                Id: artifactId, ProjectId: projectId, MediaAssetId: mediaAsset.Id,
                Kind: ArtifactKind.TtsTake, RelativePath: relPath,
                Sha256: "abc", SizeBytes: 0L, DurationSeconds: null,
                SampleRate: null, ChannelCount: null, CreatedAtUtc: DateTimeOffset.UtcNow);

            var sourceTiming = new Dictionary<Guid, SegmentSourceTiming>
            {
                [translatedSegmentId] = new(1.0, 4.0)
            };
            string fakeSourcePath = Path.Join(directory, "source.wav");
            File.WriteAllBytes(fakeSourcePath, []);

            var handler = MakeHandler(artifactStore, stageRunStore, aligner, clipExtractor: clipExtractor);

            await handler.HandleAsync(
                new LipSyncStageRequest(
                    ProjectId: projectId,
                    MediaAsset: mediaAsset,
                    TtsTakes: [take],
                    ExistingArtifacts: [artifact],
                    PreferredModelAlias: "wav2vec2-lv60-espeak-cv-ft-onnx",
                    SegmentTranscriptMap: new Dictionary<Guid, string> { [translatedSegmentId] = "hola mundo" },
                    SegmentSourceTimingMap: sourceTiming,
                    SourceSegmentTranscriptMap: new Dictionary<Guid, string> { [translatedSegmentId] = "hello world" },
                    SourceAudioPath: fakeSourcePath),
                TestContext.Current.CancellationToken);

            Assert.Equal(2, aligner.Requests.Count);

            // First call aligns the TTS take against the TRANSLATED text.
            var ttsRequest = aligner.Requests[0];
            Assert.Equal("hola mundo", ttsRequest.NormalizedTranscript);

            // Second call aligns the source clip against the ORIGINAL text.
            var sourceRequest = aligner.Requests[1];
            Assert.Equal("hello world", sourceRequest.NormalizedTranscript);
            Assert.StartsWith("src-", sourceRequest.SegmentId, StringComparison.Ordinal);

            // Lip-sync is a phoneme stage: both calls must demand phoneme timings and
            // carry the caller's preferred model alias for routing.
            foreach (var alignmentRequest in aligner.Requests)
            {
                Assert.True(alignmentRequest.Options.RequirePhonemeTimings);
                Assert.Equal("wav2vec2-lv60-espeak-cv-ft-onnx", alignmentRequest.Options.PreferredModelAlias);
            }
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    // ---------------------------------------------------------------------------
    // Transcript split: missing source-language transcript → Partial, no clip
    // extraction, original TTS artifact untouched.
    // ---------------------------------------------------------------------------

    [Fact]
    public async Task HandleAsync_WhenSourceTranscriptMissing_PartialAndOriginalTtsPreserved()
    {
        string directory = CreateTempDirectory();
        try
        {
            var projectId = Guid.NewGuid();
            var mediaAsset = MakeMediaAsset(projectId);
            var artifactStore = new FakeArtifactStore(directory);
            var stageRunStore = new FakeProjectStageRunStore();

            var phonemes = new List<PhonemeTiming>
            {
                new("AH", "arpabet", TimeSpan.Zero, TimeSpan.FromMilliseconds(100), 0.9)
            };
            var aligner = new FakeForcedAligner
            {
                StatusToReturn = ForcedAlignmentStatus.Success,
                OverallConfidence = 0.95,
                PhonesToReturn = phonemes
            };
            var stretchService = new FakePhonemeStretchService();
            var clipExtractor = new FakeAudioClipExtractor();

            var relPath = $"tts/{Guid.NewGuid():N}.wav";
            artifactStore.Seed(relPath);
            var artifactId = Guid.NewGuid();
            var translatedSegmentId = Guid.NewGuid();

            var take = TtsTake.CreateStock(
                projectId: projectId,
                voiceAssignmentId: Guid.NewGuid(),
                translatedSegmentId: translatedSegmentId) with
            { ArtifactId = artifactId };

            var artifact = new ProjectArtifact(
                Id: artifactId, ProjectId: projectId, MediaAssetId: mediaAsset.Id,
                Kind: ArtifactKind.TtsTake, RelativePath: relPath,
                Sha256: "abc", SizeBytes: 0L, DurationSeconds: null,
                SampleRate: null, ChannelCount: null, CreatedAtUtc: DateTimeOffset.UtcNow);

            var sourceTiming = new Dictionary<Guid, SegmentSourceTiming>
            {
                [translatedSegmentId] = new(1.0, 4.0)
            };
            string fakeSourcePath = Path.Join(directory, "source.wav");
            File.WriteAllBytes(fakeSourcePath, []);

            var handler = MakeHandler(artifactStore, stageRunStore, aligner, stretchService, clipExtractor);

            // SourceSegmentTranscriptMap deliberately omitted.
            var result = await handler.HandleAsync(
                new LipSyncStageRequest(
                    ProjectId: projectId,
                    MediaAsset: mediaAsset,
                    TtsTakes: [take],
                    ExistingArtifacts: [artifact],
                    SegmentTranscriptMap: new Dictionary<Guid, string> { [translatedSegmentId] = "hola mundo" },
                    SegmentSourceTimingMap: sourceTiming,
                    SourceAudioPath: fakeSourcePath),
                TestContext.Current.CancellationToken);

            var seg = result.Segments[0];
            Assert.Equal(LipSyncSegmentStatus.Partial, seg.Status);
            Assert.NotNull(seg.SkipReason);
            Assert.Contains("Source-language transcript not available", seg.SkipReason, StringComparison.Ordinal);

            // No source clip extraction, no stretch, single (TTS) alignment call.
            Assert.Equal(0, clipExtractor.CallCount);
            Assert.Equal(0, stretchService.CallCount);
            Assert.Equal(1, aligner.CallCount);

            // Original TTS artifact remains on disk untouched.
            Assert.True(File.Exists(artifactStore.GetPath(relPath)));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }
}
