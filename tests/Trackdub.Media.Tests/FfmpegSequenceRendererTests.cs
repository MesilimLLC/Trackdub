using Trackdub.Contracts;
using Trackdub.Contracts.Editing;
using Trackdub.Domain.Editing;
using Trackdub.Domain.Media;
using Trackdub.Media.Muxing;
using Trackdub.Media.Process;
using Trackdub.Media.Probe;

namespace Trackdub.Media.Tests;

public sealed class FfmpegSequenceRendererTests
{
    [RequiresFfmpegAndFfprobeFact]
    public async Task RenderAsync_renders_duplicate_source_ranges_with_linked_audio()
    {
        using RealMediaFixture fixture = await RealMediaFixture.CreateAsync(TestContext.Current.CancellationToken);
        Guid projectId = Guid.NewGuid();
        Guid sourceId = Guid.NewGuid();
        Guid firstPlacementId = Guid.NewGuid();
        Guid secondPlacementId = Guid.NewGuid();
        var sequence = new SequenceDocument(
            SequenceDocument.CurrentVersion,
            projectId,
            new SequenceFrame(64, 64),
            [new SequenceSource(sourceId, projectId, Guid.NewGuid(), null, 1.25, 64, 64)],
            [
                new SequencePlacement(firstPlacementId, sourceId, 0, 0.5, new SequenceCrop()),
                new SequencePlacement(secondPlacementId, sourceId, 0.5, 1, new SequenceCrop(1.5, 0.75, 0.5))
            ]);
        string outputPath = Path.Join(fixture.Root, "sequence-output.mp4");
        var plan = new SequenceExportPlan(
            sequence,
            new Dictionary<Guid, SequenceRenderPlacementAssets>
            {
                [firstPlacementId] = new(fixture.SourcePath, fixture.SourcePath),
                [secondPlacementId] = new(fixture.SourcePath, fixture.SourcePath)
            },
            outputPath,
            ExportOutputContainer.Mp4,
            AudioMode: SequenceExportAudioMode.Dubbed,
            VideoEncoder: VideoEncoderPreference.Software,
            OutputFrameRate: 24);
        var renderer = new FfmpegSequenceRenderer(fixture.Probe, fixture.FfmpegPath);

        SequenceExportRenderResult result = await renderer.RenderAsync(plan, TestContext.Current.CancellationToken);
        MediaProbeSnapshot rendered = await fixture.Probe.ProbeAsync(result.OutputPath, TestContext.Current.CancellationToken);

        Assert.True(File.Exists(result.OutputPath));
        Assert.Equal(64, Assert.Single(rendered.VideoStreams).Width);
        Assert.Equal(64, Assert.Single(rendered.VideoStreams).Height);
        Assert.Single(rendered.AudioStreams);
        Assert.InRange(rendered.DurationSeconds, 0.9, 1.1);
    }

    [RequiresFfmpegAndFfprobeFact]
    public async Task RenderAsync_keeps_original_audio_from_each_video_input()
    {
        using RealMediaFixture fixture = await RealMediaFixture.CreateAsync(TestContext.Current.CancellationToken);
        Guid projectId = Guid.NewGuid();
        Guid sourceId = Guid.NewGuid();
        Guid placementId = Guid.NewGuid();
        var sequence = new SequenceDocument(
            SequenceDocument.CurrentVersion,
            projectId,
            new SequenceFrame(64, 64),
            [new SequenceSource(sourceId, projectId, Guid.NewGuid(), null, 1.25, 64, 64)],
            [new SequencePlacement(placementId, sourceId, 0.25, 1.0, new SequenceCrop())]);
        var plan = new SequenceExportPlan(
            sequence,
            new Dictionary<Guid, SequenceRenderPlacementAssets>
            {
                [placementId] = new(fixture.SourcePath, AudioPath: null)
            },
            Path.Join(fixture.Root, "original-audio-sequence.mp4"),
            ExportOutputContainer.Mp4,
            AudioMode: SequenceExportAudioMode.OriginalSource,
            VideoEncoder: VideoEncoderPreference.Software,
            OutputFrameRate: 24);
        var renderer = new FfmpegSequenceRenderer(fixture.Probe, fixture.FfmpegPath);

        SequenceExportRenderResult result = await renderer.RenderAsync(plan, TestContext.Current.CancellationToken);
        MediaProbeSnapshot rendered = await fixture.Probe.ProbeAsync(result.OutputPath, TestContext.Current.CancellationToken);

        Assert.Single(rendered.VideoStreams);
        Assert.Single(rendered.AudioStreams);
        Assert.InRange(rendered.DurationSeconds, 0.7, 0.8);
    }

    [Fact]
    public async Task RenderAsync_publishes_completed_output_and_preserves_existing_output_on_failure()
    {
        string root = CreateTestDirectory();
        try
        {
            var files = new RendererTestFiles(root);
            await File.WriteAllBytesAsync(files.SourcePath, [1, 2, 3], TestContext.Current.CancellationToken);
            await File.WriteAllBytesAsync(files.DubPath, [4, 5, 6], TestContext.Current.CancellationToken);
            await File.WriteAllBytesAsync(files.FfmpegPath, [7], TestContext.Current.CancellationToken);
            await File.WriteAllBytesAsync(files.OutputPath, [8, 9], TestContext.Current.CancellationToken);

            var failingRunner = new RecordingProcessRunner((_, _, _) =>
                Task.FromResult(new ProcessResult(1, string.Empty, "render failed")));
            FfmpegSequenceRenderer failingRenderer = CreateRenderer(failingRunner, files);

            await Assert.ThrowsAsync<InvalidOperationException>(() =>
                failingRenderer.RenderAsync(
                    CreatePlan(files),
                    TestContext.Current.CancellationToken));

            Assert.Equal(new byte[] { 8, 9 }, await File.ReadAllBytesAsync(files.OutputPath, TestContext.Current.CancellationToken));
            Assert.Empty(Directory.GetFiles(root, "*.partial.mp4"));

            var successfulRunner = new RecordingProcessRunner(async (arguments, _, cancellationToken) =>
            {
                string temporaryOutput = arguments[^1];
                await File.WriteAllBytesAsync(temporaryOutput, [10, 11, 12], cancellationToken);
                return new ProcessResult(0, string.Empty, string.Empty);
            });
            FfmpegSequenceRenderer successfulRenderer = CreateRenderer(successfulRunner, files);

            SequenceExportRenderResult result = await successfulRenderer.RenderAsync(
                CreatePlan(files),
                TestContext.Current.CancellationToken);

            Assert.Equal(files.OutputPath, result.OutputPath);
            Assert.Equal(new byte[] { 10, 11, 12 }, await File.ReadAllBytesAsync(files.OutputPath, TestContext.Current.CancellationToken));
            Assert.Empty(Directory.GetFiles(root, "*.partial.mp4"));
            Assert.Single(successfulRunner.Calls);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task RenderAsync_rejects_audio_that_does_not_cover_the_selected_source_range()
    {
        string root = CreateTestDirectory();
        try
        {
            var files = new RendererTestFiles(root);
            await File.WriteAllBytesAsync(files.SourcePath, [1], TestContext.Current.CancellationToken);
            await File.WriteAllBytesAsync(files.DubPath, [2], TestContext.Current.CancellationToken);
            await File.WriteAllBytesAsync(files.FfmpegPath, [3], TestContext.Current.CancellationToken);

            var mediaProbe = new FixedMediaProbe(audioDurationSeconds: 3);
            var runner = new RecordingProcessRunner((_, _, _) =>
                Task.FromResult(new ProcessResult(0, string.Empty, string.Empty)));
            var renderer = new FfmpegSequenceRenderer(
                runner,
                mediaProbe,
                new EmptyVideoEncoderCapabilities(),
                gpuHintProvider: null,
                files.FfmpegPath);

            InvalidDataException exception = await Assert.ThrowsAsync<InvalidDataException>(() =>
                renderer.RenderAsync(
                    CreatePlan(files with { OutputFileName = "output.mp4" }),
                    TestContext.Current.CancellationToken));

            Assert.Contains("ends before the selected source range", exception.Message);
            Assert.Empty(runner.Calls);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task RenderAsync_rejects_original_audio_when_a_source_has_no_audio_stream()
    {
        string root = CreateTestDirectory();
        try
        {
            var files = new RendererTestFiles(root);
            await File.WriteAllBytesAsync(files.SourcePath, [1], TestContext.Current.CancellationToken);
            await File.WriteAllBytesAsync(files.FfmpegPath, [2], TestContext.Current.CancellationToken);

            var runner = new RecordingProcessRunner((_, _, _) =>
                Task.FromResult(new ProcessResult(0, string.Empty, string.Empty)));
            var renderer = new FfmpegSequenceRenderer(
                runner,
                new FixedMediaProbe(audioDurationSeconds: 0, includeAudio: false),
                new EmptyVideoEncoderCapabilities(),
                gpuHintProvider: null,
                files.FfmpegPath);

            InvalidDataException exception = await Assert.ThrowsAsync<InvalidDataException>(() =>
                renderer.RenderAsync(
                    CreatePlan(files with { OutputFileName = "output.mp4" }, includeDub: false) with
                    {
                        AudioMode = SequenceExportAudioMode.OriginalSource
                    },
                    TestContext.Current.CancellationToken));

            Assert.Contains("Original source audio", exception.Message);
            Assert.Empty(runner.Calls);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task RenderAsync_handles_gpu_discovery_failure_and_propagates_cancellation(bool cancel)
    {
        string root = CreateTestDirectory();
        try
        {
            var files = new RendererTestFiles(root);
            await File.WriteAllBytesAsync(files.SourcePath, [1], TestContext.Current.CancellationToken);
            await File.WriteAllBytesAsync(files.DubPath, [2], TestContext.Current.CancellationToken);
            await File.WriteAllBytesAsync(files.FfmpegPath, [3], TestContext.Current.CancellationToken);
            var runner = new RecordingProcessRunner(async (args, _, token) =>
            {
                await File.WriteAllBytesAsync(args[^1], [4], token);
                return new ProcessResult(0, string.Empty, string.Empty);
            });
            var renderer = new FfmpegSequenceRenderer(runner, new FixedMediaProbe(20),
                new EmptyVideoEncoderCapabilities(), new FailingGpuHintProvider(cancel), files.FfmpegPath);
            SequenceExportPlan plan = CreatePlan(files) with { VideoEncoder = VideoEncoderPreference.Auto };

            if (cancel)
            {
                await Assert.ThrowsAnyAsync<OperationCanceledException>(() => renderer.RenderAsync(plan, TestContext.Current.CancellationToken));
                Assert.Empty(runner.Calls);
            }
            else
            {
                await renderer.RenderAsync(plan, TestContext.Current.CancellationToken);
                Assert.Contains("libx264", Assert.Single(runner.Calls));
            }
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private sealed class FailingGpuHintProvider(bool cancel) : IMediaGpuHintProvider
    {
        public Task<MediaGpuHint> GetCurrentAsync(CancellationToken cancellationToken = default) =>
            cancel ? throw new OperationCanceledException() : throw new InvalidOperationException("discovery failed");
    }

    private static FfmpegSequenceRenderer CreateRenderer(RecordingProcessRunner runner, RendererTestFiles files) =>
        new(
            runner,
            new FixedMediaProbe(audioDurationSeconds: 20),
            new EmptyVideoEncoderCapabilities(),
            gpuHintProvider: null,
            files.FfmpegPath);

    private static SequenceExportPlan CreatePlan(RendererTestFiles files, bool includeDub = true)
    {
        Guid projectId = Guid.NewGuid();
        Guid sourceId = Guid.NewGuid();
        Guid placementId = Guid.NewGuid();
        var sequence = new SequenceDocument(
            SequenceDocument.CurrentVersion,
            projectId,
            new SequenceFrame(1920, 1080),
            [new SequenceSource(sourceId, projectId, Guid.NewGuid(), null, 10, 1920, 1080)],
            [new SequencePlacement(placementId, sourceId, 0, 5, new SequenceCrop())]);

        return new SequenceExportPlan(
            sequence,
            new Dictionary<Guid, SequenceRenderPlacementAssets>
            {
                [placementId] = new(files.SourcePath, includeDub ? files.DubPath : null)
            },
            files.OutputPath,
            ExportOutputContainer.Mp4,
            AudioMode: SequenceExportAudioMode.Dubbed,
            VideoEncoder: VideoEncoderPreference.Software);
    }

    private static string CreateTestDirectory()
    {
        string path = Path.Join(Path.GetTempPath(), $"trackdub-sequence-export-{Guid.NewGuid():N}");
        Directory.CreateDirectory(path);
        return path;
    }

    private sealed record RendererTestFiles(string Root)
    {
        public string DubFileName { get; init; } = "dub.wav";

        public string OutputFileName { get; init; } = "delivery.mp4";

        public string SourcePath => Path.Join(Root, "source.mp4");

        public string DubPath => Path.Join(Root, DubFileName);

        public string FfmpegPath => Path.Join(Root, "ffmpeg.exe");

        public string OutputPath => Path.Join(Root, OutputFileName);
    }

    private sealed class RealMediaFixture : IDisposable
    {
        public required string Root { get; init; }

        public required string SourcePath { get; init; }

        public required string FfmpegPath { get; init; }

        public required FfmpegMediaProbe Probe { get; init; }

        public static async Task<RealMediaFixture> CreateAsync(CancellationToken cancellationToken)
        {
            string root = CreateTestDirectory();
            try
            {
                string sourcePath = await MediaFixtureFactory.CreateSampleVideoAsync(root, cancellationToken);
                string ffmpegPath = FfmpegToolResolver.TryResolveExecutableForCurrentProcess(
                    null,
                    "TRACKDUB_FFMPEG_PATH",
                    ["ffmpeg.exe", "ffmpeg"],
                    allowAutoDownload: false)!;
                string ffprobePath = FfmpegToolResolver.TryResolveExecutableForCurrentProcess(
                    null,
                    "TRACKDUB_FFPROBE_PATH",
                    ["ffprobe.exe", "ffprobe"],
                    allowAutoDownload: false)!;
                return new RealMediaFixture
                {
                    Root = root,
                    SourcePath = sourcePath,
                    FfmpegPath = ffmpegPath,
                    Probe = new FfmpegMediaProbe(ffmpegPath, ffprobePath)
                };
            }
            catch
            {
                Directory.Delete(root, recursive: true);
                throw;
            }
        }

        public void Dispose() => Directory.Delete(Root, recursive: true);
    }

    private sealed class RecordingProcessRunner(
        Func<IReadOnlyList<string>, ProcessRunOptions?, CancellationToken, Task<ProcessResult>> runAsync)
        : IProcessRunner
    {
        public List<IReadOnlyList<string>> Calls { get; } = [];

        public Task<ProcessResult> RunAsync(
            string executablePath,
            IReadOnlyList<string> arguments,
            CancellationToken cancellationToken,
            ProcessRunOptions? options = null)
        {
            Calls.Add(arguments);
            return runAsync(arguments, options, cancellationToken);
        }
    }

    private sealed class FixedMediaProbe(double audioDurationSeconds, bool includeAudio = true) : IMediaProbe
    {
        public Task<MediaProbeSnapshot> ProbeAsync(string sourcePath, CancellationToken cancellationToken)
        {
            _ = sourcePath;
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(new MediaProbeSnapshot(
                "mp4",
                "test media",
                20,
                null,
                includeAudio ? [new MediaAudioStream(0, "pcm_s16le", 2, 48_000, audioDurationSeconds)] : [],
                [new MediaVideoStream(0, "h264", 1920, 1080, 30, 20)]));
        }
    }

    private sealed class EmptyVideoEncoderCapabilities : IFfmpegVideoEncoderCapabilities
    {
        public FfmpegVideoEncoderSnapshot GetSnapshot() => FfmpegVideoEncoderSnapshot.Empty;

        public Task<FfmpegVideoEncoderSnapshot> RefreshAsync(CancellationToken cancellationToken = default) =>
            Task.FromResult(FfmpegVideoEncoderSnapshot.Empty);
    }
}
