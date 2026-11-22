using Trackdub.Application.Mixing;
using Trackdub.Contracts;
using Trackdub.Contracts.Projects;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Mixing;
using Trackdub.Domain.Tts;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed class TailOverhangFitterTests
{
    private const string TakeRelativePath = "artifacts/tts/take-0001.wav";

    [Fact]
    public async Task FitAsync_compresses_an_overhanging_take_by_its_overhang_and_clears_the_warning()
    {
        using var temp = new TempDir();
        (FakeArtifactStore store, FakeAudioTimeStretchService stretch) = CreateServices(temp.Path);
        MixSpeechClip clip = Clip(startSeconds: 10d, takeDurationSeconds: 8d);
        MixPlan plan = PlanOf([clip], Overhang(clip, overhangSeconds: 0.5d));
        Guid stageRunId = Guid.NewGuid();

        TailFitResult result = await new TailOverhangFitter(stretch, store)
            .FitAsync(plan, stageRunId, TestContext.Current.CancellationToken);

        double expectedDuration = 8d - 0.5d - TailOverhangFitter.SafetyMarginSeconds;
        MixSpeechClip fitted = Assert.Single(result.Plan.SpeechClips);
        Assert.Empty(result.Plan.Warnings);
        Assert.Equal(expectedDuration, fitted.TakeDurationSeconds!.Value, precision: 9);
        Assert.Equal(
            ProjectArtifactPaths.GetExportTailFitTakeRelativePath(stageRunId, clip.SegmentId),
            fitted.TakeRelativePath);
        Assert.Equal(clip.StartSeconds, fitted.StartSeconds);
        Assert.Equal(8d / expectedDuration, Assert.Single(stretch.StretchRatios), precision: 9);
        AudioTimeStretchRequest request = stretch.LastRequest!;
        Assert.Equal(store.GetPath(TakeRelativePath), request.InputPath);
        Assert.Equal(store.GetPath(fitted.TakeRelativePath!), request.OutputPath);
        Assert.True(request.EnableRubberband);
        Assert.Equal(request.OutputPath, Assert.Single(result.CreatedFilePaths));
        Assert.True(File.Exists(request.OutputPath));
        Assert.True(10d + fitted.TakeDurationSeconds!.Value < 10d + 8d - 0.5d);
    }

    [Fact]
    public async Task FitAsync_fits_a_take_that_needs_exactly_the_maximum_compression()
    {
        using var temp = new TempDir();
        (FakeArtifactStore store, FakeAudioTimeStretchService stretch) = CreateServices(temp.Path);
        // 5.0 s take, target 4.0 s: 5.0 / 4.0 is exactly the 1.25x ceiling.
        double overhang = 5d - 4d - TailOverhangFitter.SafetyMarginSeconds;
        MixSpeechClip clip = Clip(startSeconds: 0d, takeDurationSeconds: 5d);
        MixPlan plan = PlanOf([clip], Overhang(clip, overhang));

        TailFitResult result = await new TailOverhangFitter(stretch, store)
            .FitAsync(plan, Guid.NewGuid(), TestContext.Current.CancellationToken);

        Assert.Empty(result.Plan.Warnings);
        Assert.Equal(TailOverhangFitter.MaxTempoRatio, Assert.Single(stretch.StretchRatios), precision: 9);
    }

    [Fact]
    public async Task FitAsync_leaves_a_take_that_would_need_more_than_the_maximum_compression()
    {
        using var temp = new TempDir();
        (FakeArtifactStore store, FakeAudioTimeStretchService stretch) = CreateServices(temp.Path);
        MixSpeechClip clip = Clip(startSeconds: 0d, takeDurationSeconds: 4d);
        MixPlan plan = PlanOf([clip], Overhang(clip, overhangSeconds: 1d));

        TailFitResult result = await new TailOverhangFitter(stretch, store)
            .FitAsync(plan, Guid.NewGuid(), TestContext.Current.CancellationToken);

        Assert.Same(plan, result.Plan);
        Assert.Single(result.Plan.Warnings);
        Assert.Empty(result.CreatedFilePaths);
        Assert.Empty(stretch.StretchRatios);
    }

    [Fact]
    public async Task FitAsync_leaves_a_take_shorter_than_its_overhang()
    {
        using var temp = new TempDir();
        (FakeArtifactStore store, FakeAudioTimeStretchService stretch) = CreateServices(temp.Path);
        MixSpeechClip clip = Clip(startSeconds: 0d, takeDurationSeconds: 0.4d);
        MixPlan plan = PlanOf([clip], Overhang(clip, overhangSeconds: 0.5d));

        TailFitResult result = await new TailOverhangFitter(stretch, store)
            .FitAsync(plan, Guid.NewGuid(), TestContext.Current.CancellationToken);

        Assert.Same(plan, result.Plan);
        Assert.Empty(stretch.StretchRatios);
    }

    [Fact]
    public async Task FitAsync_ignores_warnings_that_are_not_tail_overhangs()
    {
        using var temp = new TempDir();
        (FakeArtifactStore store, FakeAudioTimeStretchService stretch) = CreateServices(temp.Path);
        MixSpeechClip clip = Clip(startSeconds: 0d, takeDurationSeconds: 5d);
        var missingTake = new MixPlanWarning(0, clip.SegmentId, "No take.", MixPlanWarningCode.MissingTake);
        MixPlan plan = PlanOf([clip], missingTake);

        TailFitResult result = await new TailOverhangFitter(stretch, store)
            .FitAsync(plan, Guid.NewGuid(), TestContext.Current.CancellationToken);

        Assert.Same(plan, result.Plan);
        Assert.Empty(stretch.StretchRatios);
    }

    [Fact]
    public async Task FitAsync_ignores_an_overhang_warning_without_a_measured_overhang()
    {
        using var temp = new TempDir();
        (FakeArtifactStore store, FakeAudioTimeStretchService stretch) = CreateServices(temp.Path);
        MixSpeechClip clip = Clip(startSeconds: 0d, takeDurationSeconds: 5d);
        var warning = new MixPlanWarning(0, clip.SegmentId, "Past the end.", MixPlanWarningCode.TakeExceedsSourceAudio);
        MixPlan plan = PlanOf([clip], warning);

        TailFitResult result = await new TailOverhangFitter(stretch, store)
            .FitAsync(plan, Guid.NewGuid(), TestContext.Current.CancellationToken);

        Assert.Same(plan, result.Plan);
        Assert.Empty(stretch.StretchRatios);
    }

    [Fact]
    public async Task FitAsync_fits_only_the_clip_that_overhangs()
    {
        using var temp = new TempDir();
        (FakeArtifactStore store, FakeAudioTimeStretchService stretch) = CreateServices(temp.Path);
        MixSpeechClip first = Clip(startSeconds: 0d, takeDurationSeconds: 5d, segmentIndex: 0);
        MixSpeechClip last = Clip(startSeconds: 5d, takeDurationSeconds: 5d, segmentIndex: 1);
        MixPlan plan = PlanOf([first, last], Overhang(last, overhangSeconds: 0.3d));

        TailFitResult result = await new TailOverhangFitter(stretch, store)
            .FitAsync(plan, Guid.NewGuid(), TestContext.Current.CancellationToken);

        Assert.Equal(first, result.Plan.SpeechClips[0]);
        Assert.Contains("tail-fit", result.Plan.SpeechClips[1].TakeRelativePath);
        Assert.Single(stretch.StretchRatios);
    }

    [Fact]
    public async Task FitAsync_keeps_the_warning_and_removes_partial_output_when_stretching_fails()
    {
        using var temp = new TempDir();
        var store = new FakeArtifactStore(temp.Path);
        SeedTake(temp.Path);
        var stretch = new ThrowingStretchService(new IOException("ffmpeg exited with code 1."));
        MixSpeechClip clip = Clip(startSeconds: 0d, takeDurationSeconds: 5d);
        MixPlan plan = PlanOf([clip], Overhang(clip, overhangSeconds: 0.3d));

        TailFitResult result = await new TailOverhangFitter(stretch, store)
            .FitAsync(plan, Guid.NewGuid(), TestContext.Current.CancellationToken);

        Assert.Same(plan, result.Plan);
        Assert.Empty(result.CreatedFilePaths);
        Assert.False(File.Exists(stretch.LastOutputPath));
    }

    [Fact]
    public async Task FitAsync_propagates_cancellation()
    {
        using var temp = new TempDir();
        var store = new FakeArtifactStore(temp.Path);
        SeedTake(temp.Path);
        var stretch = new ThrowingStretchService(new OperationCanceledException());
        MixSpeechClip clip = Clip(startSeconds: 0d, takeDurationSeconds: 5d);
        MixPlan plan = PlanOf([clip], Overhang(clip, overhangSeconds: 0.3d));

        await Assert.ThrowsAsync<OperationCanceledException>(() => new TailOverhangFitter(stretch, store)
            .FitAsync(plan, Guid.NewGuid(), TestContext.Current.CancellationToken));
    }

    private static (FakeArtifactStore Store, FakeAudioTimeStretchService Stretch) CreateServices(string root)
    {
        SeedTake(root);
        return (new FakeArtifactStore(root), new FakeAudioTimeStretchService());
    }

    private static void SeedTake(string root)
    {
        string path = Path.Join(root, "artifacts", "tts", "take-0001.wav");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, [1, 2, 3, 4]);
    }

    private static MixSpeechClip Clip(double startSeconds, double takeDurationSeconds, int segmentIndex = 0) =>
        new(
            segmentIndex,
            Guid.NewGuid(),
            startSeconds,
            startSeconds + takeDurationSeconds,
            TakeId: Guid.NewGuid(),
            ArtifactId: Guid.NewGuid(),
            TakeRelativePath,
            takeDurationSeconds,
            IsSilentGap: false,
            WarningMessage: null);

    private static MixPlanWarning Overhang(MixSpeechClip clip, double overhangSeconds) =>
        new(
            clip.SegmentIndex,
            clip.SegmentId,
            "Dubbed take extends past the end of the source audio.",
            MixPlanWarningCode.TakeExceedsSourceAudio)
        {
            OverhangSeconds = overhangSeconds
        };

    private static MixPlan PlanOf(IReadOnlyList<MixSpeechClip> clips, params MixPlanWarning[] warnings) =>
        new(
            Guid.NewGuid(),
            MediaAssetId: null,
            ArtifactKind.NormalizedAudio,
            "artifacts/audio/normalized_audio.wav",
            SourceGainDb: 0d,
            DubbedSpeechGainDb: 0d,
            DuckingGainDb: 0d,
            DuckingLeadSeconds: 0.05d,
            DuckingTailSeconds: 0.18d,
            DateTimeOffset.UtcNow,
            clips,
            DuckingRegions: [],
            warnings);

    private sealed class ThrowingStretchService(Exception exception) : IAudioTimeStretchService
    {
        public string? LastOutputPath { get; private set; }

        public Task<AudioTimeStretchResult> StretchAsync(
            AudioTimeStretchRequest request,
            CancellationToken cancellationToken)
        {
            LastOutputPath = request.OutputPath;
            File.WriteAllBytes(request.OutputPath, [9]);
            throw exception;
        }
    }

    private sealed class TempDir : IDisposable
    {
        public TempDir()
        {
            Path = System.IO.Path.Join(System.IO.Path.GetTempPath(), "trackdub-tailfit-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Path);
        }

        public string Path { get; }

        public void Dispose()
        {
            if (Directory.Exists(Path))
            {
                Directory.Delete(Path, recursive: true);
            }
        }
    }
}
