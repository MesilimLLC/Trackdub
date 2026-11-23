using Trackdub.Application.Mixing;
using Trackdub.Contracts;
using Trackdub.Contracts.Projects;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Mixing;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed class TailOverhangFitterTests : IDisposable
{
    private const string TakeRelativePath = "artifacts/tts/take-0001.wav";

    private readonly string root = Path.Join(Path.GetTempPath(), "trackdub-tailfit-" + Guid.NewGuid().ToString("N"));
    private readonly FakeArtifactStore store;
    private readonly FakeAudioTimeStretchService stretch = new();

    public TailOverhangFitterTests()
    {
        string takePath = Path.Join(root, "artifacts", "tts", "take-0001.wav");
        Directory.CreateDirectory(Path.GetDirectoryName(takePath)!);
        File.WriteAllBytes(takePath, [1, 2, 3, 4]);
        store = new FakeArtifactStore(root);
    }

    public void Dispose()
    {
        if (Directory.Exists(root))
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task FitAsync_compresses_an_overhanging_take_by_its_overhang_and_clears_the_warning()
    {
        MixSpeechClip clip = Clip(startSeconds: 10d, takeDurationSeconds: 8d);
        Guid stageRunId = Guid.NewGuid();

        TailFitResult result = await FitAsync(PlanOf([clip], Overhang(clip, overhangSeconds: 0.5d)), stageRunId);

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
    }

    [Fact]
    public async Task FitAsync_fits_a_take_that_needs_exactly_the_maximum_compression()
    {
        // 5.0 s take, target 4.0 s: 5.0 / 4.0 is exactly the 1.25x ceiling.
        double overhang = 5d - 4d - TailOverhangFitter.SafetyMarginSeconds;
        MixSpeechClip clip = Clip(startSeconds: 0d, takeDurationSeconds: 5d);

        TailFitResult result = await FitAsync(PlanOf([clip], Overhang(clip, overhang)));

        Assert.Empty(result.Plan.Warnings);
        Assert.Equal(TailOverhangFitter.MaxTempoRatio, Assert.Single(stretch.StretchRatios), precision: 9);
    }

    [Theory]
    [InlineData(4d, 1d)]
    [InlineData(0.4d, 0.5d)]
    public async Task FitAsync_leaves_a_take_it_cannot_fit_within_the_maximum_compression(
        double takeDurationSeconds, double overhangSeconds)
    {
        MixSpeechClip clip = Clip(startSeconds: 0d, takeDurationSeconds);
        MixPlan plan = PlanOf([clip], Overhang(clip, overhangSeconds));

        TailFitResult result = await FitAsync(plan);

        AssertUntouched(plan, result);
    }

    [Fact]
    public async Task FitAsync_ignores_warnings_that_are_not_tail_overhangs()
    {
        MixSpeechClip clip = Clip(startSeconds: 0d, takeDurationSeconds: 5d);
        var missingTake = new MixPlanWarning(0, clip.SegmentId, "No take.", MixPlanWarningCode.MissingTake);
        MixPlan plan = PlanOf([clip], missingTake);

        AssertUntouched(plan, await FitAsync(plan));
    }

    [Fact]
    public async Task FitAsync_ignores_an_overhang_warning_without_a_measured_overhang()
    {
        MixSpeechClip clip = Clip(startSeconds: 0d, takeDurationSeconds: 5d);
        var warning = new MixPlanWarning(0, clip.SegmentId, "Past the end.", MixPlanWarningCode.TakeExceedsSourceAudio);
        MixPlan plan = PlanOf([clip], warning);

        AssertUntouched(plan, await FitAsync(plan));
    }

    [Fact]
    public async Task FitAsync_fits_only_the_clip_that_overhangs()
    {
        MixSpeechClip first = Clip(startSeconds: 0d, takeDurationSeconds: 5d, segmentIndex: 0);
        MixSpeechClip last = Clip(startSeconds: 5d, takeDurationSeconds: 5d, segmentIndex: 1);

        TailFitResult result = await FitAsync(PlanOf([first, last], Overhang(last, overhangSeconds: 0.3d)));

        Assert.Equal(first, result.Plan.SpeechClips[0]);
        Assert.Contains("tail-fit", result.Plan.SpeechClips[1].TakeRelativePath);
        Assert.Single(stretch.StretchRatios);
    }

    [Fact]
    public async Task FitAsync_keeps_the_warning_and_removes_partial_output_when_stretching_fails()
    {
        var failing = new ThrowingStretchService(new IOException("ffmpeg exited with code 1."));
        MixSpeechClip clip = Clip(startSeconds: 0d, takeDurationSeconds: 5d);
        MixPlan plan = PlanOf([clip], Overhang(clip, overhangSeconds: 0.3d));

        TailFitResult result = await new TailOverhangFitter(failing, store)
            .FitAsync(plan, Guid.NewGuid(), TestContext.Current.CancellationToken);

        AssertUntouched(plan, result);
        Assert.False(File.Exists(failing.LastOutputPath));
    }

    [Fact]
    public async Task FitAsync_propagates_cancellation_and_removes_every_file_it_wrote()
    {
        // The first take fits; canceling while the second is being stretched must remove both outputs,
        // because the caller only learns about files from a returned result.
        var canceling = new ThrowingStretchService(new OperationCanceledException(), succeedFirst: 1);
        MixSpeechClip first = Clip(startSeconds: 0d, takeDurationSeconds: 5d, segmentIndex: 0);
        MixSpeechClip last = Clip(startSeconds: 5d, takeDurationSeconds: 5d, segmentIndex: 1);
        MixPlan plan = PlanOf([first, last], Overhang(first, 0.3d), Overhang(last, 0.3d));

        await Assert.ThrowsAsync<OperationCanceledException>(() => new TailOverhangFitter(canceling, store)
            .FitAsync(plan, Guid.NewGuid(), TestContext.Current.CancellationToken));

        Assert.Equal(2, canceling.OutputPaths.Count);
        Assert.All(canceling.OutputPaths, path => Assert.False(File.Exists(path), path));
    }

    private Task<TailFitResult> FitAsync(MixPlan plan, Guid? stageRunId = null) =>
        new TailOverhangFitter(stretch, store)
            .FitAsync(plan, stageRunId ?? Guid.NewGuid(), TestContext.Current.CancellationToken);

    private void AssertUntouched(MixPlan plan, TailFitResult result)
    {
        Assert.Same(plan, result.Plan);
        Assert.Single(result.Plan.Warnings);
        Assert.Empty(result.CreatedFilePaths);
        Assert.Empty(stretch.StretchRatios);
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

    private sealed class ThrowingStretchService(Exception exception, int succeedFirst = 0) : IAudioTimeStretchService
    {
        public List<string> OutputPaths { get; } = [];

        public string? LastOutputPath => OutputPaths.Count == 0 ? null : OutputPaths[^1];

        public Task<AudioTimeStretchResult> StretchAsync(
            AudioTimeStretchRequest request,
            CancellationToken cancellationToken)
        {
            OutputPaths.Add(request.OutputPath);
            File.WriteAllBytes(request.OutputPath, [9]);
            if (OutputPaths.Count <= succeedFirst)
            {
                return Task.FromResult(new AudioTimeStretchResult(Trackdub.Domain.Tts.TtsStretchEngine.Atempo, UsedFallback: false));
            }

            throw exception;
        }
    }
}
