using System.Text.Json;
using System.Text.Json.Serialization;
using Trackdub.Contracts;
using Trackdub.Domain.Media;
using Trackdub.Media.Process;

namespace Trackdub.Media.Probe;

public sealed class FfmpegMediaProbe : IMediaProbe
{
    private const int ProbeMaxStandardOutputCharacters = 128 * 1024 * 1024;
    private static readonly ProcessRunOptions ProbeProcessOptions = new(
        Timeout: TimeSpan.FromSeconds(30),
        MaxStandardOutputCharacters: ProbeMaxStandardOutputCharacters);

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true
    };

    private const int MaxCachedSnapshots = 128;
    private static readonly object SnapshotCacheLock = new();
    private static readonly Dictionary<string, CachedSnapshot> SnapshotCache = new(
        OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
    private sealed record CachedSnapshot(long Length, long WriteTicks, MediaProbeSnapshot Snapshot, long AccessSequence);
    private static long snapshotAccessSequence;

    private readonly record struct ProbeCacheKey(string Value);

    private readonly record struct FfprobeCommand(string FfprobePath, string SourcePath);

    private readonly record struct SourceFingerprint(long Length, long WriteTicks)
    {
        public static SourceFingerprint From(FileInfo info) =>
            new(info.Length, info.LastWriteTimeUtc.Ticks);

        public static bool TryRefresh(FileInfo info, out SourceFingerprint fingerprint)
        {
            try
            {
                info.Refresh();
                if (!info.Exists)
                {
                    fingerprint = default;
                    return false;
                }

                fingerprint = From(info);
                return true;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                fingerprint = default;
                return false;
            }
        }
    }

    private readonly IProcessRunner processRunner;
    private readonly IFfmpegToolResolver toolResolver;

    public FfmpegMediaProbe(string? ffmpegPath = null, string? ffprobePath = null)
        : this(new ProcessRunner(), ffmpegPath, ffprobePath)
    {
    }

    internal FfmpegMediaProbe(IProcessRunner processRunner, string? ffmpegPath = null, string? ffprobePath = null)
        : this(processRunner, new FfmpegToolResolver(ffmpegPath, ffprobePath))
    {
    }

    internal FfmpegMediaProbe(IProcessRunner processRunner, IFfmpegToolResolver toolResolver)
    {
        this.processRunner = processRunner;
        this.toolResolver = toolResolver;
    }

    public async Task<MediaProbeSnapshot> ProbeAsync(string sourcePath, CancellationToken cancellationToken)
    {
        string fullSourcePath = Path.GetFullPath(sourcePath);
        if (!File.Exists(fullSourcePath))
        {
            throw new FileNotFoundException("Source media file was not found.", fullSourcePath);
        }

        var sourceInfo = new FileInfo(fullSourcePath);
        SourceFingerprint fingerprint = SourceFingerprint.From(sourceInfo);
        string ffprobePath = toolResolver.ResolveFfprobePath();
        // Key on the resolved binary as well as the file: two probe instances can be pointed at
        // different ffprobe builds that report different metadata for the same file.
        var cacheKey = new ProbeCacheKey(ffprobePath + '\0' + fullSourcePath);

        if (TryGetCachedSnapshot(cacheKey, fingerprint, out MediaProbeSnapshot? cachedSnapshot))
        {
            return cachedSnapshot;
        }

        var command = new FfprobeCommand(ffprobePath, fullSourcePath);
        ProcessResult result = await RunFfprobeAsync(command, cancellationToken).ConfigureAwait(false);
        FfprobePayload payload = ParsePayload(result);
        var snapshot = CreateSnapshot(payload);

        // Only cache the snapshot under metadata that still describes the probed bytes; if the
        // file changed while ffprobe ran, the key would no longer match its contents.
        if (!SourceFingerprint.TryRefresh(sourceInfo, out SourceFingerprint updated))
        {
            return snapshot;
        }

        if (updated == fingerprint)
        {
            StoreSnapshot(cacheKey, fingerprint, snapshot);
        }

        return snapshot;
    }

    private static bool TryGetCachedSnapshot(
        ProbeCacheKey cacheKey,
        SourceFingerprint fingerprint,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out MediaProbeSnapshot? snapshot)
    {
        lock (SnapshotCacheLock)
        {
            return TryGetCachedSnapshotCore(cacheKey, fingerprint, out snapshot);
        }
    }

    private static bool TryGetCachedSnapshotCore(
        ProbeCacheKey cacheKey,
        SourceFingerprint fingerprint,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out MediaProbeSnapshot? snapshot)
    {
        if (!SnapshotCache.TryGetValue(cacheKey.Value, out CachedSnapshot? cached))
        {
            snapshot = null;
            return false;
        }

        if (cached.Length != fingerprint.Length)
        {
            SnapshotCache.Remove(cacheKey.Value);
            snapshot = null;
            return false;
        }

        if (cached.WriteTicks != fingerprint.WriteTicks)
        {
            SnapshotCache.Remove(cacheKey.Value);
            snapshot = null;
            return false;
        }

        SnapshotCache[cacheKey.Value] = cached with { AccessSequence = ++snapshotAccessSequence };
        snapshot = cached.Snapshot;
        return true;
    }

    private static void StoreSnapshot(
        ProbeCacheKey cacheKey,
        SourceFingerprint fingerprint,
        MediaProbeSnapshot snapshot)
    {
        lock (SnapshotCacheLock)
        {
            EvictIfNeeded(cacheKey);
            SnapshotCache[cacheKey.Value] = new CachedSnapshot(
                fingerprint.Length,
                fingerprint.WriteTicks,
                snapshot,
                ++snapshotAccessSequence);
        }
    }

    private static void EvictIfNeeded(ProbeCacheKey cacheKey)
    {
        if (SnapshotCache.Count < MaxCachedSnapshots)
        {
            return;
        }

        if (SnapshotCache.ContainsKey(cacheKey.Value))
        {
            return;
        }

        string oldest = SnapshotCache.OrderBy(static pair => pair.Value.AccessSequence).First().Key;
        SnapshotCache.Remove(oldest);
    }

    private async Task<ProcessResult> RunFfprobeAsync(
        FfprobeCommand command,
        CancellationToken cancellationToken)
    {
        try
        {
            ProcessResult result = await processRunner.RunAsync(
                command.FfprobePath,
                [
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    command.SourcePath
                ],
                cancellationToken,
                ProbeProcessOptions).ConfigureAwait(false);

            if (result.ExitCode != 0)
            {
                throw new InvalidOperationException(
                    $"ffprobe failed with exit code {result.ExitCode}: {result.StandardError}".Trim());
            }

            return result;
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or FileNotFoundException or DirectoryNotFoundException)
        {
            toolResolver.InvalidateResolvedPaths();
            throw;
        }
    }

    private static FfprobePayload ParsePayload(ProcessResult result)
    {
        FfprobePayload? payload = JsonSerializer.Deserialize<FfprobePayload>(
            result.StandardOutput,
            JsonOptions);

        if (payload?.Format is null)
        {
            throw new InvalidOperationException("ffprobe did not return format metadata.");
        }

        return payload;
    }

    private static MediaProbeSnapshot CreateSnapshot(FfprobePayload payload)
    {
        IReadOnlyList<MediaAudioStream> audioStreams = payload.Streams
            .Where(static stream => string.Equals(stream.CodecType, "audio", StringComparison.OrdinalIgnoreCase))
            .Select(static stream => new MediaAudioStream(
                stream.Index,
                stream.CodecName ?? "unknown",
                stream.Channels ?? 0,
                ParseOrDefault(stream.SampleRate, 0),
                ParseOrDefault(stream.Duration, 0d)))
            .ToArray();

        IReadOnlyList<MediaVideoStream> videoStreams = payload.Streams
            .Where(static stream => string.Equals(stream.CodecType, "video", StringComparison.OrdinalIgnoreCase))
            .Select(static stream => new MediaVideoStream(
                stream.Index,
                stream.CodecName ?? "unknown",
                stream.Width ?? 0,
                stream.Height ?? 0,
                ParseFrameRate(stream.AverageFrameRate),
                ParseOrDefault(stream.Duration, 0d),
                stream.PixelFormat,
                stream.ColorSpace,
                stream.ColorTransfer,
                stream.ColorPrimaries))
            .ToArray();

        IReadOnlyList<MediaSubtitleStream> subtitleStreams = payload.Streams
            .Where(static stream => string.Equals(stream.CodecType, "subtitle", StringComparison.OrdinalIgnoreCase))
            .Select(static stream => new MediaSubtitleStream(
                stream.Index,
                stream.CodecName ?? "unknown",
                stream.Language))
            .ToArray();

        return new MediaProbeSnapshot(
            payload.Format!.FormatName ?? "unknown",
            payload.Format.FormatLongName,
            ParseOrDefault(payload.Format.Duration, 0d),
            ParseNullableLong(payload.Format.BitRate),
            audioStreams,
            videoStreams,
            subtitleStreams);
    }

    private static T ParseOrDefault<T>(string? value, T fallback)
        where T : struct, IParsable<T> =>
        T.TryParse(value, null, out T parsed) ? parsed : fallback;

    private static long? ParseNullableLong(string? value) =>
        long.TryParse(value, out long parsed) ? parsed : null;

    private static double ParseFrameRate(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return 0d;
        }

        string[] parts = value.Split('/');
        if (TryParseFrameRateRatio(parts, out double frameRate))
        {
            return frameRate;
        }

        return ParseOrDefault(value, 0d);
    }

    private static bool TryParseFrameRateRatio(string[] parts, out double frameRate)
    {
        frameRate = 0d;
        if (parts.Length != 2)
        {
            return false;
        }

        if (!double.TryParse(parts[0], out double numerator))
        {
            return false;
        }

        if (!double.TryParse(parts[1], out double denominator))
        {
            return false;
        }

        const double frameRateDenominatorMin = 1e-9;
        if (Math.Abs(denominator) <= frameRateDenominatorMin)
        {
            return false;
        }

        double result = numerator / denominator;
        if (!double.IsFinite(result))
        {
            return false;
        }

        frameRate = result;
        return true;
    }

    private sealed record FfprobePayload(
        [property: JsonPropertyName("format")] FfprobeFormat? Format,
        [property: JsonPropertyName("streams")] IReadOnlyList<FfprobeStream> Streams)
    {
        public FfprobePayload()
            : this(null, [])
        {
        }
    }

    private sealed record FfprobeFormat(
        [property: JsonPropertyName("format_name")] string? FormatName,
        [property: JsonPropertyName("format_long_name")] string? FormatLongName,
        [property: JsonPropertyName("duration")] string? Duration,
        [property: JsonPropertyName("bit_rate")] string? BitRate);

    private sealed record FfprobeStream(
        [property: JsonPropertyName("index")] int Index,
        [property: JsonPropertyName("codec_type")] string? CodecType,
        [property: JsonPropertyName("codec_name")] string? CodecName,
        [property: JsonPropertyName("sample_rate")] string? SampleRate,
        [property: JsonPropertyName("channels")] int? Channels,
        [property: JsonPropertyName("duration")] string? Duration,
        [property: JsonPropertyName("avg_frame_rate")] string? AverageFrameRate,
        [property: JsonPropertyName("width")] int? Width,
        [property: JsonPropertyName("height")] int? Height,
        [property: JsonPropertyName("pix_fmt")] string? PixelFormat,
        [property: JsonPropertyName("color_space")] string? ColorSpace,
        [property: JsonPropertyName("color_transfer")] string? ColorTransfer,
        [property: JsonPropertyName("color_primaries")] string? ColorPrimaries,
        [property: JsonPropertyName("tags")] FfprobeTags? Tags)
    {
        public string? Language => Tags?.Language;
    }

    private sealed record FfprobeTags(
        [property: JsonPropertyName("language")] string? Language);
}
