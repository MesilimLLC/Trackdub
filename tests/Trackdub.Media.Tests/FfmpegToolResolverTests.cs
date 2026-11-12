using Trackdub.Media.Process;

namespace Trackdub.Media.Tests;

public sealed class FfmpegToolResolverTests : IDisposable
{
    private readonly string root = Directory.CreateTempSubdirectory("trackdub-resolver-").FullName;

    public void Dispose() => Directory.Delete(root, recursive: true);

    private static string ToolName(string baseName) =>
        OperatingSystem.IsWindows() ? $"{baseName}.exe" : baseName;

    private static string? TryResolve(FfmpegToolResolver resolver)
    {
        try
        {
            return resolver.ResolveFfmpegPath(allowAutoDownload: false);
        }
        catch (InvalidOperationException)
        {
            return null;
        }
    }

    [Fact]
    public void ResolveFfmpegPath_reuses_successful_explicit_path()
    {
        string ffmpeg = Path.Join(root, ToolName("ffmpeg"));
        File.WriteAllBytes(ffmpeg, [1]);
        var downloader = new CountingDownloader();
        var resolver = new FfmpegToolResolver(ffmpeg, null, downloader);

        string first = resolver.ResolveFfmpegPath(allowAutoDownload: false);
        File.Delete(ffmpeg);
        string second = resolver.ResolveFfmpegPath(allowAutoDownload: false);

        Assert.Equal(first, second);
        Assert.Equal(0, downloader.EnsureCalls);
    }

    [Fact]
    public void InvalidateResolvedPaths_forces_resolution_again()
    {
        string ffmpeg = Path.Join(root, ToolName("ffmpeg"));
        File.WriteAllBytes(ffmpeg, [1]);
        var resolver = new FfmpegToolResolver(ffmpeg, null, new CountingDownloader());
        string first = resolver.ResolveFfmpegPath(allowAutoDownload: false);
        File.Delete(ffmpeg);

        resolver.InvalidateResolvedPaths();
        string? second = TryResolve(resolver);

        // After invalidation the resolver must not serve the stale cached path, whether it
        // re-resolves from PATH or fails outright.
        Assert.NotEqual(first, second);
    }

    [Fact]
    public void ResolveFfprobePath_reuses_successful_explicit_path()
    {
        string ffmpeg = Path.Join(root, ToolName("ffmpeg"));
        string ffprobe = Path.Join(root, ToolName("ffprobe"));
        File.WriteAllBytes(ffmpeg, [1]);
        File.WriteAllBytes(ffprobe, [2]);
        var resolver = new FfmpegToolResolver(ffmpeg, ffprobe, new CountingDownloader());

        string first = resolver.ResolveFfprobePath(allowAutoDownload: false);
        File.Delete(ffprobe);
        string second = resolver.ResolveFfprobePath(allowAutoDownload: false);

        Assert.Equal(first, second);
    }

    private sealed class CountingDownloader : IFfmpegAutoDownloader
    {
        public int EnsureCalls { get; private set; }
        public string? TryEnsureExecutable(IReadOnlyList<string> fallbacks)
        {
            EnsureCalls++;
            return null;
        }
        public string? TryResolveInstallRoot() => null;
    }
}
