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
}
