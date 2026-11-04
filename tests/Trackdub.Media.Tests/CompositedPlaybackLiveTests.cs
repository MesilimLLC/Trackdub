using Trackdub.Domain.Media;
using Trackdub.Media.Playback;

namespace Trackdub.Media.Tests;

/// <summary>
/// Optional live tests against real libmpv / LibVLC natives. Set TRACKDUB_PLAYBACK_INTEGRATION=1
/// and optionally TRACKDUB_PLAYBACK_TEST_MEDIA to a probed MP4 path.
/// </summary>
public sealed class CompositedPlaybackLiveTests
{
    private static bool IsEnabled =>
        string.Equals(
            Environment.GetEnvironmentVariable("TRACKDUB_PLAYBACK_INTEGRATION"),
            "1",
            StringComparison.Ordinal);

    [Fact]
    public async Task LibMpv_open_prepare_delivers_frame_when_runtime_available()
    {
        if (!IsEnabled)
        {
            return;
        }

        string? mediaPath = ResolveTestMediaPath();
        if (mediaPath is null)
        {
            return;
        }

        string? runtimePath = new LibMpvRuntimeLocator().ResolveRuntimeLibraryPath();
        if (runtimePath is null)
        {
            return;
        }

        var backend = new LibMpvCompositedPlaybackBackend(runtimePath);
        var sink = new RecordingFrameSink();
        var source = BuildSource(mediaPath);

        try
        {
            backend.TryAttachFrameSink(sink);
            await backend.OpenAsync(source, TestContext.Current.CancellationToken);

            // The first delivered frame can legitimately be black while decode warms up; keep
            // preparing until a frame with visible signal arrives or the attempt budget runs out.
            bool usable = false;
            for (int attempt = 0; attempt < 20 && !usable; attempt++)
            {
                await backend.PreparePreviewFrameAsync(TestContext.Current.CancellationToken);
                await Task.Delay(250, TestContext.Current.CancellationToken);
                usable = sink.LastFrame is { } last && HasVisibleSignal(last);
            }

            PlaybackSnapshot snapshot = await backend.GetSnapshotAsync(TestContext.Current.CancellationToken);
            Assert.True(snapshot.IsLoaded, snapshot.WarningMessage ?? "backend not loaded");
            Assert.NotNull(sink.LastFrame);
            Assert.True(sink.LastFrame!.Format.Width > 0);
            Assert.True(sink.LastFrame.Format.Height > 0);
            Assert.True(
                usable,
                "libmpv delivered frames but every sampled pixel was zero. " +
                $"hwdec-current={backend.ReadMpvPropertyForDiagnostics("hwdec-current") ?? "(null)"}, " +
                $"video-format={backend.ReadMpvPropertyForDiagnostics("video-format") ?? "(null)"}, " +
                $"width={backend.ReadMpvPropertyForDiagnostics("width") ?? "(null)"}, " +
                $"height={backend.ReadMpvPropertyForDiagnostics("height") ?? "(null)"}, " +
                $"time-pos={backend.ReadMpvPropertyForDiagnostics("time-pos") ?? "(null)"}, " +
                $"warning={snapshot.WarningMessage ?? "(none)"}");
        }
        finally
        {
            backend.Dispose();
        }
    }

    [Fact]
    public async Task PlaybackService_open_with_sink_before_open_delivers_frame_when_runtime_available()
    {
        if (!IsEnabled)
        {
            return;
        }

        string? mediaPath = ResolveTestMediaPath();
        if (mediaPath is null)
        {
            return;
        }

        string? mpvPath = new LibMpvRuntimeLocator().ResolveRuntimeLibraryPath();
        if (mpvPath is null)
        {
            return;
        }

        var probe = new LibMpvPlaybackCapabilityProbeStub();
        var factory = new CompositedPlaybackBackendFactoryStub(mpvPath);
        var service = new PlaybackService(probe, factory);
        var sink = new RecordingFrameSink();
        var source = BuildSource(mediaPath);

        await service.TryAttachFrameSinkAsync(sink, TestContext.Current.CancellationToken);
        PlaybackOpenResult openResult = await service.OpenAsync(source, TestContext.Current.CancellationToken);
        await service.PreparePreviewFrameAsync(TestContext.Current.CancellationToken);
        await Task.Delay(250, TestContext.Current.CancellationToken);

        Assert.True(openResult.IsBackendAvailable, openResult.Snapshot.WarningMessage);
        Assert.NotNull(sink.LastFrame);
    }

    [Fact]
    public async Task PlaybackService_open_attach_after_open_matches_shell_order()
    {
        if (!IsEnabled)
        {
            return;
        }

        string? mediaPath = ResolveTestMediaPath();
        string? mpvPath = new LibMpvRuntimeLocator().ResolveRuntimeLibraryPath();
        if (mediaPath is null || mpvPath is null)
        {
            return;
        }

        var service = new PlaybackService(
            new LibMpvPlaybackCapabilityProbeStub(),
            new CompositedPlaybackBackendFactoryStub(mpvPath));
        var sink = new RecordingFrameSink();
        var source = BuildSource(mediaPath);

        PlaybackOpenResult openResult = await service.OpenAsync(source, TestContext.Current.CancellationToken);
        await service.TryAttachFrameSinkAsync(sink, TestContext.Current.CancellationToken);
        await service.PreparePreviewFrameAsync(TestContext.Current.CancellationToken);
        await Task.Delay(250, TestContext.Current.CancellationToken);

        Assert.True(openResult.IsBackendAvailable, openResult.Snapshot.WarningMessage);
        Assert.NotNull(sink.LastFrame);
    }

    private static string? ResolveTestMediaPath()
    {
        string? fromEnv = Environment.GetEnvironmentVariable("TRACKDUB_PLAYBACK_TEST_MEDIA");
        if (!string.IsNullOrWhiteSpace(fromEnv) && File.Exists(fromEnv))
        {
            return fromEnv;
        }

        string temp = Path.Join(Path.GetTempPath(), "trackdub-playback-test.mp4");
        return File.Exists(temp) ? temp : null;
    }

    private static bool HasVisibleSignal(VideoFrame frame)
    {
        int stride = frame.Format.Stride;
        int height = frame.Format.Height;
        byte[] data = frame.Data;
        for (int y = 0; y < height; y++)
        {
            int rowStart = y * stride;
            int rowEnd = Math.Min(rowStart + stride, data.Length);
            for (int offset = rowStart; offset + 2 < rowEnd; offset += 4)
            {
                if (data[offset] != 0 || data[offset + 1] != 0 || data[offset + 2] != 0)
                {
                    return true;
                }
            }
        }

        return false;
    }

    private static MediaSourceDescriptor BuildSource(string mediaPath) =>
        new(
            mediaPath,
            new MediaProbeSnapshot(
                "mov,mp4,m4a,3gp,3g2,mj2",
                "QuickTime / MOV",
                2.0,
                2048,
                [new MediaAudioStream(0, "aac", 2, 44100, 2.0)],
                [new MediaVideoStream(1, "h264", 320, 240, 24.0, 2.0)]));

    private sealed class CompositedPlaybackBackendFactoryStub(string libMpvPath) : IPlaybackBackendFactory
    {
        public IPlaybackBackend? Create(PlaybackBackendKind backendKind) =>
            backendKind switch
            {
                PlaybackBackendKind.LibMpv => new LibMpvCompositedPlaybackBackend(libMpvPath),
                PlaybackBackendKind.LibVlc => CreateLibVlc(),
                _ => null,
            };

        private static IPlaybackBackend? CreateLibVlc()
        {
            string baseDir = AppContext.BaseDirectory;
            string? appBin = LocateAppOutputDirectory();
            string? runtimePath = new LibVlcRuntimeLocator(appBin ?? baseDir).ResolveRuntimePath()
                ?? new LibVlcRuntimeLocator(baseDir).ResolveRuntimePath();
            return runtimePath is null ? null : new LibVlcCompositedPlaybackBackend(runtimePath);
        }

        private static string? LocateAppOutputDirectory()
        {
            string? current = AppContext.BaseDirectory;
            for (int depth = 0; depth < 12 && !string.IsNullOrWhiteSpace(current); depth++)
            {
                string candidate = Path.Join(current, "src", "Trackdub.App.Avalonia", "bin", "Debug", "net11.0-windows10.0.19041.0");
                if (Directory.Exists(Path.Join(candidate, "libvlc")))
                {
                    return candidate;
                }

                current = Directory.GetParent(current)?.FullName;
            }

            return null;
        }
    }
}
