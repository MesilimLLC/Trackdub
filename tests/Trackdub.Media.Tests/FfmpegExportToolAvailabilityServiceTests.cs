using Trackdub.Media.Process;

namespace Trackdub.Media.Tests;

public sealed class FfmpegExportToolAvailabilityServiceTests
{
    [Fact]
    public void CheckAvailability_resolves_tools_without_auto_download()
    {
        var resolver = new RecordingToolResolver("ffmpeg.exe", "ffprobe.exe");
        var service = new FfmpegExportToolAvailabilityService(resolver);

        var availability = service.CheckAvailability();

        Assert.True(availability.IsAvailable);
        Assert.Equal("ffmpeg.exe", availability.FfmpegPath);
        Assert.Equal("ffprobe.exe", availability.FfprobePath);
        Assert.Equal([false, false], resolver.AllowAutoDownloadRequests);
    }

    [Fact]
    public void CheckAvailability_reports_unavailable_without_auto_download()
    {
        var resolver = new RecordingToolResolver("ffmpeg.exe", "ffprobe.exe")
        {
            Exception = new InvalidOperationException("ffmpeg missing")
        };
        var service = new FfmpegExportToolAvailabilityService(resolver);

        var availability = service.CheckAvailability();

        Assert.False(availability.IsAvailable);
        Assert.Contains("ffmpeg missing", availability.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal([false], resolver.AllowAutoDownloadRequests);
    }

    private sealed class RecordingToolResolver(string ffmpegPath, string ffprobePath) : IFfmpegToolResolver
    {
        private readonly List<bool> allowAutoDownloadRequests = [];

        public Exception? Exception { get; init; }

        public IReadOnlyList<bool> AllowAutoDownloadRequests => allowAutoDownloadRequests;

        public string ResolveFfmpegPath(bool allowAutoDownload = true)
        {
            allowAutoDownloadRequests.Add(allowAutoDownload);
            if (Exception is not null)
            {
                throw Exception;
            }

            return ffmpegPath;
        }

        public string ResolveFfprobePath(bool allowAutoDownload = true)
        {
            allowAutoDownloadRequests.Add(allowAutoDownload);
            if (Exception is not null)
            {
                throw Exception;
            }

            return ffprobePath;
        }

        public void InvalidateResolvedPaths()
        {
        }
    }
}
