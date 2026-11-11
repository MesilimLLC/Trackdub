using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace Trackdub.Cloud.Jobs;

/// <summary>CPU-only input validation, shared by upload completion and the one-shot supervisor.</summary>
public static class MediaValidator
{
    public static async Task ValidateAsync(string path, CancellationToken token)
    {
        var info = new ProcessStartInfo("ffprobe") { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, RedirectStandardInput = true };
        foreach (string arg in new[] { "-v", "error", "-show_streams", "-show_format", "-of", "json", Path.GetFullPath(path) }) info.ArgumentList.Add(arg);
        using var process = new Process { StartInfo = info };
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        using var bound = CancellationTokenSource.CreateLinkedTokenSource(token, timeout.Token);
        process.Start(); process.StandardInput.Close();
        try
        {
            Task<string> output = ReadBoundedAsync(process.StandardOutput, bound.Token);
            Task<string> error = ReadBoundedAsync(process.StandardError, bound.Token);
            await Task.WhenAll(process.WaitForExitAsync(bound.Token), output, error).ConfigureAwait(false);
            if (process.ExitCode != 0) throw new JobException("INVALID_INPUT", 400, "Input is not valid media.");
            using var json = JsonDocument.Parse(await output.ConfigureAwait(false));
            var format = json.RootElement.GetProperty("format");
            string[] formats = format.GetProperty("format_name").GetString()!.Split(',');
            double duration = double.Parse(format.GetProperty("duration").GetString()!, CultureInfo.InvariantCulture);
            var streams = json.RootElement.GetProperty("streams").EnumerateArray().ToArray();
            if (!formats.Contains("mp4") || !double.IsFinite(duration) || duration is <= 0 or > CloudLimits.MaxDurationSeconds
                || !streams.Any(s => s.GetProperty("codec_type").GetString() == "audio")
                || !streams.Any(s => s.GetProperty("codec_type").GetString() == "video")
                || streams.Any(s => (s.TryGetProperty("height", out var h) && h.GetInt32() > 1080)
                    || (s.TryGetProperty("width", out var w) && w.GetInt32() > 1920)))
                throw new JobException("INVALID_INPUT", 400, "Expected MP4 audio/video up to ten minutes and 1920x1080.");
        }
        catch (OperationCanceledException) when (timeout.IsCancellationRequested && !token.IsCancellationRequested)
        { throw new JobException("VALIDATION_TIMEOUT", 503, "Media validation timed out. Retry safely."); }
        catch (Exception ex) when (ex is JsonException or KeyNotFoundException or FormatException or IOException)
        { throw new JobException("INVALID_INPUT", 400, "Input metadata is invalid."); }
        finally
        {
            if (!process.HasExited) process.Kill(entireProcessTree: true);
            await process.WaitForExitAsync(CancellationToken.None).ConfigureAwait(false);
        }
    }

    private static async Task<string> ReadBoundedAsync(StreamReader reader, CancellationToken token)
    {
        var text = new StringBuilder(); char[] buffer = new char[4096]; int count;
        while ((count = await reader.ReadAsync(buffer, token).ConfigureAwait(false)) > 0)
        {
            if (text.Length + count > 65536) throw new IOException("Media metadata exceeded the supported size.");
            text.Append(buffer, 0, count);
        }
        return text.ToString();
    }
}
