using System.Runtime.InteropServices;

namespace Trackdub.Media.Process;

internal interface IFfmpegToolResolver
{
    string ResolveFfmpegPath(bool allowAutoDownload = true);

    string ResolveFfprobePath(bool allowAutoDownload = true);

    void InvalidateResolvedPaths();
}

internal interface IFfmpegAutoDownloader
{
    string? TryEnsureExecutable(IReadOnlyList<string> fallbacks);

    string? TryResolveInstallRoot();
}

internal sealed class FfmpegToolResolver(
    string? explicitFfmpegPath = null,
    string? explicitFfprobePath = null,
    IFfmpegAutoDownloader? autoDownloader = null)
    : IFfmpegToolResolver
{
    private readonly IFfmpegAutoDownloader autoDownloader = autoDownloader ?? FfmpegAutoDownloader.Shared;
    private string? resolvedFfmpegPath;
    private string? resolvedFfprobePath;

    public string ResolveFfmpegPath(bool allowAutoDownload = true)
    {
        if (Volatile.Read(ref resolvedFfmpegPath) is { } cached)
        {
            return cached;
        }

        string resolved = ResolveExecutable(
            explicitFfmpegPath,
            "TRACKDUB_FFMPEG_PATH",
            GetPlatformExecutableNames("ffmpeg"),
            allowAutoDownload,
            autoDownloader);
        Volatile.Write(ref resolvedFfmpegPath, resolved);
        return resolved;
    }

    private string CacheFfprobePath(string path)
    {
        Volatile.Write(ref resolvedFfprobePath, path);
        return path;
    }

    public void InvalidateResolvedPaths()
    {
        Volatile.Write(ref resolvedFfmpegPath, null);
        Volatile.Write(ref resolvedFfprobePath, null);
    }

    public string ResolveFfprobePath(bool allowAutoDownload = true)
    {
        if (Volatile.Read(ref resolvedFfprobePath) is { } cached)
        {
            return cached;
        }

        string ffmpegPath = ResolveFfmpegPath(allowAutoDownload);
        string? ffprobePath = TryResolveExecutable(
            explicitFfprobePath,
            "TRACKDUB_FFPROBE_PATH",
            GetPlatformExecutableNames("ffprobe"),
            allowAutoDownload,
            autoDownloader);

        if (!string.IsNullOrWhiteSpace(ffprobePath))
        {
            return CacheFfprobePath(ffprobePath);
        }

        string ffmpegDirectory = Path.GetDirectoryName(ffmpegPath)!;

        string? TryFindSiblingFfprobeLocal()
        {
            foreach (string candidate in new[]
                     {
                         Path.Join(ffmpegDirectory, "ffprobe.exe"),
                         Path.Join(ffmpegDirectory, "ffprobe"),
                         Path.Join(Directory.GetParent(ffmpegDirectory)?.FullName ?? ffmpegDirectory, "ffprobe.exe"),
                         Path.Join(Directory.GetParent(ffmpegDirectory)?.FullName ?? ffmpegDirectory, "ffprobe")
                     })
            {
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }

            return null;
        }

        string? TryFindFfprobeBySearchLocal()
        {
            string[] commonRoots =
            [
                ffmpegDirectory,
                AppContext.BaseDirectory,
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86)
            ];

            foreach (string root in commonRoots
                         .Where(static value => !string.IsNullOrWhiteSpace(value))
                         .Distinct(StringComparer.OrdinalIgnoreCase))
            {
                foreach (string candidateName in GetPlatformExecutableNames("ffprobe"))
                {
                    string? discovered = SearchRecursively(root, candidateName);
                    if (!string.IsNullOrWhiteSpace(discovered))
                    {
                        return discovered;
                    }
                }
            }

            return null;
        }

        string? sibling = TryFindSiblingFfprobeLocal();
        if (sibling is not null)
        {
            return CacheFfprobePath(sibling);
        }

        string? discoveredProbe = TryFindFfprobeBySearchLocal();
        if (discoveredProbe is not null)
        {
            return CacheFfprobePath(discoveredProbe);
        }

        throw new InvalidOperationException(
            "Unable to locate ffprobe. Configure TRACKDUB_FFPROBE_PATH or install FFmpeg with ffprobe.");
    }

    private static string ResolveExecutable(
        string? explicitPath,
        string environmentVariable,
        IReadOnlyList<string> fallbacks,
        bool allowAutoDownload,
        IFfmpegAutoDownloader autoDownloader)
    {
        string? resolved = TryResolveExecutable(explicitPath, environmentVariable, fallbacks, allowAutoDownload, autoDownloader);
        if (!string.IsNullOrWhiteSpace(resolved))
        {
            return resolved;
        }

        throw new InvalidOperationException(
            $"Unable to locate '{fallbacks[0]}'. Configure {environmentVariable} or install FFmpeg.");
    }

    private static string? TryResolveExecutable(
        string? explicitPath,
        string environmentVariable,
        IReadOnlyList<string> fallbacks,
        bool allowAutoDownload,
        IFfmpegAutoDownloader autoDownloader)
        => TryResolveExecutableForCurrentProcess(explicitPath, environmentVariable, fallbacks, allowAutoDownload, autoDownloader);

    internal static string? TryResolveExecutableForCurrentProcess(
        string? explicitPath,
        string environmentVariable,
        IReadOnlyList<string> fallbacks,
        bool allowAutoDownload = true,
        IFfmpegAutoDownloader? autoDownloader = null)
    {
        IReadOnlyList<string> platformFallbacks = GetPlatformFallbacks(fallbacks);
        foreach (string? candidate in new[]
                 {
                     explicitPath,
                     Environment.GetEnvironmentVariable(environmentVariable)
                 })
        {
            if (!string.IsNullOrWhiteSpace(candidate) && File.Exists(candidate))
            {
                return Path.GetFullPath(candidate);
            }
        }

        foreach (string fallback in platformFallbacks)
        {
            string? onPath = FindOnPath(fallback);
            if (!string.IsNullOrWhiteSpace(onPath))
            {
                return onPath;
            }
        }

        foreach (string candidate in EnumerateCommonExecutableCandidates(platformFallbacks))
        {
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        // Check if the installer has previously placed binaries in its payload directory.
        // This runs before auto-download so already-installed binaries are found even
        // when allowAutoDownload is false (e.g., health check on Linux/macOS).
        if (autoDownloader is not null)
        {
            string? payloadRoot = autoDownloader.TryResolveInstallRoot();
            if (payloadRoot is not null)
            {
                string? installed = FfmpegAutoDownloader.FindExecutable(payloadRoot, platformFallbacks);
                if (!string.IsNullOrWhiteSpace(installed))
                {
                    return installed;
                }
            }
        }

        if (allowAutoDownload)
        {
            string? downloaded = (autoDownloader ?? FfmpegAutoDownloader.Shared).TryEnsureExecutable(platformFallbacks);
            if (!string.IsNullOrWhiteSpace(downloaded))
            {
                return downloaded;
            }
        }

        return null;
    }

    private static IReadOnlyList<string> GetPlatformExecutableNames(string baseName) =>
        OperatingSystem.IsWindows()
            ? [$"{baseName}.exe", baseName]
            : [baseName];

    private static IReadOnlyList<string> GetPlatformFallbacks(IReadOnlyList<string> fallbacks)
    {
        if (OperatingSystem.IsWindows())
        {
            return fallbacks;
        }

        string[] filtered = fallbacks
            .Where(static fallback => !fallback.EndsWith(".exe", StringComparison.OrdinalIgnoreCase))
            .ToArray();
        return filtered.Length == 0 ? fallbacks : filtered;
    }

    private static IEnumerable<string> EnumerateCommonExecutableCandidates(IReadOnlyList<string> fallbacks)
    {
        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);

        foreach (string seed in new[] { AppContext.BaseDirectory, Environment.CurrentDirectory })
        {
            foreach (string ancestor in EnumerateAncestors(seed))
            {
                foreach (string candidate in EnumerateCandidatePaths(ancestor, fallbacks))
                {
                    if (seen.Add(candidate))
                    {
                        yield return candidate;
                    }
                }
            }
        }

        if (!OperatingSystem.IsWindows())
        {
            yield break;
        }

        foreach (string root in GetWellKnownWindowsRoots())
        {
            foreach (string candidate in EnumerateCandidatePaths(root, fallbacks))
            {
                if (seen.Add(candidate))
                {
                    yield return candidate;
                }
            }
        }
    }

    private static IEnumerable<string> EnumerateCandidatePaths(string root, IReadOnlyList<string> fallbacks)
    {
        if (string.IsNullOrWhiteSpace(root))
        {
            yield break;
        }

        foreach (string fallback in fallbacks)
        {
            string runtimeIdentifier = ResolveRuntimeIdentifier();
            yield return Path.Join(root, fallback);
            yield return Path.Join(root, "bin", fallback);
            yield return Path.Join(root, "ffmpeg", fallback);
            yield return Path.Join(root, "ffmpeg", "bin", fallback);
            yield return Path.Join(root, "tools", "ffmpeg", fallback);
            yield return Path.Join(root, "tools", "ffmpeg", "bin", fallback);
            yield return Path.Join(root, "tools", runtimeIdentifier, fallback);
            yield return Path.Join(root, "tools", runtimeIdentifier, "ffmpeg", fallback);
            yield return Path.Join(root, "native", runtimeIdentifier, fallback);
            yield return Path.Join(root, "native", runtimeIdentifier, "ffmpeg", fallback);
            yield return Path.Join(root, "native", runtimeIdentifier, "ffmpeg", "bin", fallback);
        }
    }

    private static string ResolveRuntimeIdentifier()
    {
        string architecture = RuntimeInformation.OSArchitecture is Architecture.Arm64 ? "arm64" : "x64";
        if (OperatingSystem.IsWindows()) return $"win-{architecture}";
        if (OperatingSystem.IsMacOS()) return $"osx-{architecture}";
        return $"linux-{architecture}";
    }

    private static IEnumerable<string> EnumerateAncestors(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            yield break;
        }

        DirectoryInfo? current;
        try
        {
            current = Directory.Exists(path)
                ? new DirectoryInfo(Path.GetFullPath(path))
                : Directory.GetParent(Path.GetFullPath(path));
        }
        catch (Exception)
        {
            yield break;
        }

        while (current is not null)
        {
            yield return current.FullName;
            current = current.Parent;
        }
    }

    private static IEnumerable<string> GetWellKnownWindowsRoots()
    {
        foreach (string candidate in new[]
                 {
                     Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Microsoft", "WinGet", "Links"),
                     @"C:\ffmpeg",
                     Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "ffmpeg"),
                     Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "ffmpeg"),
                     Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), "chocolatey", "bin")
                 })
        {
            if (!string.IsNullOrWhiteSpace(candidate))
            {
                yield return candidate;
            }
        }
    }

    private static string? FindOnPath(string executableName)
    {
        string? pathEnvironment = Environment.GetEnvironmentVariable("PATH");
        if (string.IsNullOrWhiteSpace(pathEnvironment))
        {
            return null;
        }

        foreach (string pathSegment in pathEnvironment.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            string candidate = Path.Join(pathSegment.Trim(), executableName);
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return null;
    }

    private static string? SearchRecursively(string root, string fileName)
    {
        if (!Directory.Exists(root))
        {
            return null;
        }

        try
        {
            foreach (string file in Directory.EnumerateFiles(root, fileName, SearchOption.AllDirectories))
            {
                return file;
            }
        }
        catch (UnauthorizedAccessException)
        {
        }
        catch (DirectoryNotFoundException)
        {
        }
        catch (IOException)
        {
        }

        return null;
    }
}
