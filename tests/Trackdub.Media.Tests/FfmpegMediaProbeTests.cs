using Trackdub.Domain.Media;
using Trackdub.Media.Process;
using Trackdub.Media.Probe;

namespace Trackdub.Media.Tests;

public sealed class FfmpegMediaProbeTests
{
    [RequiresFfmpegAndFfprobeFact]
    public async Task ProbeAsync_reads_fixture_media_metadata()
    {
        string tempDirectory = Path.Join(Path.GetTempPath(), "Trackdub.Media.Tests", Guid.NewGuid().ToString("N"));
        string sourcePath = await MediaFixtureFactory.CreateSampleVideoAsync(tempDirectory);
        var probe = new FfmpegMediaProbe();

        var result = await probe.ProbeAsync(sourcePath, TestContext.Current.CancellationToken);

        Assert.Equal("mov,mp4,m4a,3gp,3g2,mj2", result.FormatName);
        Assert.True(result.DurationSeconds > 1d);
        Assert.NotEmpty(result.AudioStreams);
        Assert.NotEmpty(result.VideoStreams);
        Assert.True(result.AudioStreams[0].SampleRate > 0);
        Assert.True(result.VideoStreams[0].Width > 0);
    }

    [Fact]
    public async Task ProbeAsync_reuses_snapshot_for_unchanged_file()
    {
        string root = Directory.CreateTempSubdirectory("trackdub-probe-cache-").FullName;
        string source = Path.Join(root, "source.mp4");
        string ffmpeg = Path.Join(root, ToolName("ffmpeg"));
        string ffprobe = Path.Join(root, ToolName("ffprobe"));
        File.WriteAllBytes(source, [1, 2, 3]);
        File.WriteAllBytes(ffmpeg, [0]);
        File.WriteAllBytes(ffprobe, [0]);
        var runner = new RecordingProbeRunner(ValidPayload);
        var probe = new FfmpegMediaProbe(runner, ffmpeg, ffprobe);
        try
        {
            MediaProbeSnapshot first = await probe.ProbeAsync(source, TestContext.Current.CancellationToken);
            MediaProbeSnapshot second = await probe.ProbeAsync(source, TestContext.Current.CancellationToken);

            Assert.Equal(first, second);
            Assert.Equal(1, runner.Calls);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task ProbeAsync_reprobes_when_source_stat_changes()
    {
        string root = Directory.CreateTempSubdirectory("trackdub-probe-miss-").FullName;
        string source = Path.Join(root, "source.mp4");
        string ffmpeg = Path.Join(root, ToolName("ffmpeg"));
        string ffprobe = Path.Join(root, ToolName("ffprobe"));
        File.WriteAllBytes(source, [1]);
        File.WriteAllBytes(ffmpeg, [0]);
        File.WriteAllBytes(ffprobe, [0]);
        var runner = new RecordingProbeRunner(ValidPayload);
        var probe = new FfmpegMediaProbe(runner, ffmpeg, ffprobe);
        try
        {
            await probe.ProbeAsync(source, TestContext.Current.CancellationToken);
            File.AppendAllText(source, "changed");
            await probe.ProbeAsync(source, TestContext.Current.CancellationToken);

            Assert.Equal(2, runner.Calls);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task ProbeAsync_invalidates_resolved_tool_paths_when_launch_fails()
    {
        string root = Directory.CreateTempSubdirectory("trackdub-probe-launch-").FullName;
        string source = Path.Join(root, "source.mp4");
        File.WriteAllBytes(source, [1, 2, 3]);
        var resolver = new RecordingToolResolver("/fake/ffprobe");
        var runner = new FailingThenSucceedingRunner(ValidPayload);
        var probe = new FfmpegMediaProbe(runner, resolver);
        try
        {
            await Assert.ThrowsAsync<System.ComponentModel.Win32Exception>(
                () => probe.ProbeAsync(source, TestContext.Current.CancellationToken));
            Assert.Equal(1, resolver.InvalidateCalls);

            MediaProbeSnapshot snapshot = await probe.ProbeAsync(source, TestContext.Current.CancellationToken);

            Assert.Equal(2, runner.Calls);
            Assert.Equal("mov,mp4,m4a,3gp,3g2,mj2", snapshot.FormatName);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task ProbeAsync_does_not_share_snapshots_for_case_distinct_paths()
    {
        string root = Directory.CreateTempSubdirectory("trackdub-probe-case-").FullName;
        string upper = Path.Join(root, "Clip.mp4");
        string lower = Path.Join(root, "clip.mp4");
        File.WriteAllBytes(upper, [1, 2, 3]);
        if (File.Exists(lower))
        {
            // Case-insensitive filesystem: both names identify the same file.
            Directory.Delete(root, recursive: true);
            return;
        }

        File.WriteAllBytes(lower, [1, 2, 3]);
        DateTime timestamp = DateTime.UtcNow.AddMinutes(-1);
        File.SetLastWriteTimeUtc(upper, timestamp);
        File.SetLastWriteTimeUtc(lower, timestamp);
        var resolver = new RecordingToolResolver("/fake/ffprobe");
        var runner = new RecordingProbeRunner(ValidPayload);
        var probe = new FfmpegMediaProbe(runner, resolver);
        try
        {
            await probe.ProbeAsync(upper, TestContext.Current.CancellationToken);
            await probe.ProbeAsync(lower, TestContext.Current.CancellationToken);

            Assert.Equal(2, runner.Calls);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static string ToolName(string baseName) =>
        OperatingSystem.IsWindows() ? $"{baseName}.exe" : baseName;

    private const string ValidPayload = """
        {"format":{"format_name":"mov,mp4,m4a,3gp,3g2,mj2","duration":"2.0"},"streams":[{"index":0,"codec_type":"audio","codec_name":"aac","sample_rate":"48000","channels":2},{"index":1,"codec_type":"video","codec_name":"h264","width":640,"height":360,"avg_frame_rate":"30/1"}]}
        """;

    private sealed class RecordingProbeRunner(string output) : IProcessRunner
    {
        public int Calls { get; private set; }
        public Task<ProcessResult> RunAsync(
            string executablePath,
            IReadOnlyList<string> arguments,
            CancellationToken cancellationToken,
            ProcessRunOptions? options = null)
        {
            Calls++;
            return Task.FromResult(new ProcessResult(0, output, string.Empty));
        }
    }

    private sealed class FailingThenSucceedingRunner(string output) : IProcessRunner
    {
        public int Calls { get; private set; }

        public Task<ProcessResult> RunAsync(
            string executablePath,
            IReadOnlyList<string> arguments,
            CancellationToken cancellationToken,
            ProcessRunOptions? options = null)
        {
            Calls++;
            if (Calls == 1)
            {
                throw new System.ComponentModel.Win32Exception("ffprobe could not be launched.");
            }

            return Task.FromResult(new ProcessResult(0, output, string.Empty));
        }
    }

    private sealed class RecordingToolResolver(string ffprobePath) : IFfmpegToolResolver
    {
        public int InvalidateCalls { get; private set; }

        public string ResolveFfmpegPath(bool allowAutoDownload = true) => "ffmpeg";

        public string ResolveFfprobePath(bool allowAutoDownload = true) => ffprobePath;

        public void InvalidateResolvedPaths() => InvalidateCalls++;
    }
}
