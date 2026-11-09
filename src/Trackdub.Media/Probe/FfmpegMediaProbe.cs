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
    private static readonly Dictionary<string, CachedSnapshot> SnapshotCache = new(StringComparer.OrdinalIgnoreCase);
    private sealed record CachedSnapshot(long Length, long WriteTicks, MediaProbeSnapshot Snapshot, long AccessSequence);
    private static long snapshotAccessSequence;

    private readonly IProcessRunner processRunner;
    private readonly FfmpegToolResolver toolResolver;

    public FfmpegMediaProbe(string? ffmpegPath = null, string? ffprobePath = null)
        : this(new ProcessRunner(), ffmpegPath, ffprobePath)
    {
    }

    internal FfmpegMediaProbe(IProcessRunner processRunner, string? ffmpegPath = null, string? ffprobePath = null)
    {
        this.processRunner = processRunner;
        toolResolver = new FfmpegToolResolver(ffmpegPath, ffprobePath);
    }

    public async Task<MediaProbeSnapshot> ProbeAsync(string sourcePath, CancellationToken cancellationToken)
    {
        string fullSourcePath = Path.GetFullPath(sourcePath);
        if (!File.Exists(fullSourcePath))
        {
            throw new FileNotFoundException("Source media file was not found.", fullSourcePath);
        }

        var sourceInfo = new FileInfo(fullSourcePath);
        long length = sourceInfo.Length;
        long writeTicks = sourceInfo.LastWriteTimeUtc.Ticks;
        lock (SnapshotCacheLock)
        {
            if (SnapshotCache.TryGetValue(fullSourcePath, out CachedSnapshot? cached) &&
                cached.Length == length)
            {
                if (cached.WriteTicks == writeTicks)
                {
                    SnapshotCache[fullSourcePath] = cached with { AccessSequence = ++snapshotAccessSequence };
                    return cached.Snapshot;
                }
            }
        }

        string ffprobePath = toolResolver.ResolveFfprobePath();
        ProcessResult result;
        try
        {
            result = await processRunner.RunAsync(
                ffprobePath,
                [
                    "-v", "quiet",
                    "-print_format", "json",
                    "-show_format",
                    "-show_streams",
                    fullSourcePath
                ],
                cancellationToken,
                ProbeProcessOptions).ConfigureAwait(false);

            if (result.ExitCode != 0)
            {
                throw new InvalidOperationException(
                    $"ffprobe failed with exit code {result.ExitCode}: {result.StandardError}".Trim());
            }
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or FileNotFoundException or DirectoryNotFoundException)
        {
            toolResolver.InvalidateResolvedPaths();
            throw;
        }

        FfprobePayload payload = ParsePayload(result);
        var snapshot = CreateSnapshot(payload);
        lock (SnapshotCacheLock)
        {
            if (SnapshotCache.Count >= MaxCachedSnapshots && !SnapshotCache.ContainsKey(fullSourcePath))
            {
                string oldest = SnapshotCache.OrderBy(static pair => pair.Value.AccessSequence).First().Key;
                SnapshotCache.Remove(oldest);
            }

            SnapshotCache[fullSourcePath] = new CachedSnapshot(length, writeTicks, snapshot, ++snapshotAccessSequence);
        }

        return snapshot;
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
                ParseInt(stream.SampleRate),
                ParseDouble(stream.Duration)))
            .ToArray();

        IReadOnlyList<MediaVideoStream> videoStreams = payload.Streams
            .Where(static stream => string.Equals(stream.CodecType, "video", StringComparison.OrdinalIgnoreCase))
            .Select(static stream => new MediaVideoStream(
                stream.Index,
                stream.CodecName ?? "unknown",
                stream.Width ?? 0,
                stream.Height ?? 0,
                ParseFrameRate(stream.AverageFrameRate),
                ParseDouble(stream.Duration),
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
            ParseDouble(payload.Format.Duration),
            ParseNullableLong(payload.Format.BitRate),
            audioStreams,
            videoStreams,
            subtitleStreams);
    }

    private static int ParseInt(string? value) =>
        int.TryParse(value, out int parsed) ? parsed : 0;

    private static double ParseDouble(string? value) =>
        double.TryParse(value, out double parsed) ? parsed : 0d;

    private static long? ParseNullableLong(string? value) =>
        long.TryParse(value, out long parsed) ? parsed : null;

    private static double ParseFrameRate(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return 0d;
        }

        const double frameRateDenominatorMin = 1e-9;
        string[] parts = value.Split('/');
        if (parts.Length == 2 &&
            double.TryParse(parts[0], out double numerator) &&
            double.TryParse(parts[1], out double denominator) &&
            Math.Abs(denominator) > frameRateDenominatorMin)
        {
            double result = numerator / denominator;
            return double.IsFinite(result) ? result : 0d;
        }

        return ParseDouble(value);
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
