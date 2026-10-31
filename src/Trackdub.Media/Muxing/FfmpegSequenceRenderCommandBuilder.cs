using System.Globalization;
using System.Text;
using Trackdub.Contracts;
using Trackdub.Contracts.Editing;
using Trackdub.Domain.Editing;
using Trackdub.Media.Process;

namespace Trackdub.Media.Muxing;

internal static class FfmpegSequenceRenderCommandBuilder
{
    public static IReadOnlyList<string> BuildArguments(
        SequenceExportPlan plan,
        VideoEncodeProfile? encodeProfile,
        string outputPath)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        VideoEncodeProfile profile = encodeProfile ?? SoftwareProfile();
        var arguments = new List<string> { "-y", "-hide_banner", "-loglevel", "error" };

        AddHardwareAccelerationArguments(arguments, profile);
        AddInputFiles(arguments, plan);
        AddFilterAndMaps(arguments, plan, profile);
        AddEncodingArguments(arguments, plan, profile);
        AddMetadataArguments(arguments, plan);
        AddContainerArguments(arguments, plan);
        arguments.Add(Path.GetFullPath(outputPath));
        return arguments;
    }

    internal static string BuildFilterGraph(
        SequenceExportPlan plan,
        VideoEncodeProfile? encodeProfile = null)
    {
        ArgumentNullException.ThrowIfNull(plan);
        SequenceFrame frame = plan.Sequence.Frame
            ?? throw new InvalidDataException("Sequence output frame is required.");
        bool hasAudio = HasAudio(plan);
        var filters = new List<string>();
        var videoSettings = new VideoFilterSettings(frame, plan.OutputFrameRate, encodeProfile);

        for (int index = 0; index < plan.Sequence.Placements.Count; index++)
        {
            SequencePlacement placement = plan.Sequence.Placements[index];
            filters.Add(BuildVideoPlacementFilter(placement, index, videoSettings));
            if (hasAudio)
            {
                filters.Add(BuildAudioPlacementFilter(placement, plan, index));
            }
        }

        filters.Add(BuildConcatenationFilter(plan));
        AppendFinalVideoFilters(filters, plan, frame);
        return string.Join(';', filters);
    }

    private static void AddHardwareAccelerationArguments(
        List<string> arguments,
        VideoEncodeProfile profile)
    {
        if (profile.InputHwaccelArguments is { Count: > 0 } hwaccelArguments)
        {
            foreach (string argument in hwaccelArguments)
            {
                arguments.Add(argument);
            }
        }
    }

    private static void AddInputFiles(List<string> arguments, SequenceExportPlan plan)
    {
        foreach (SequencePlacement placement in plan.Sequence.Placements)
        {
            AddInput(arguments, plan.PlacementAssets[placement.Id].SourceMediaPath);
        }

        if (plan.AudioMode is SequenceExportAudioMode.Dubbed)
        {
            foreach (SequencePlacement placement in plan.Sequence.Placements)
            {
                AddInput(arguments, plan.PlacementAssets[placement.Id].AudioPath!);
            }
        }
    }

    private static void AddInput(List<string> arguments, string mediaPath)
    {
        arguments.Add("-i");
        arguments.Add(Path.GetFullPath(mediaPath));
    }

    private static void AddFilterAndMaps(
        List<string> arguments,
        SequenceExportPlan plan,
        VideoEncodeProfile profile)
    {
        arguments.Add("-filter_complex");
        arguments.Add(BuildFilterGraph(plan, profile));
        arguments.AddRange(["-map", "[vout]"]);
        if (HasAudio(plan))
        {
            arguments.AddRange(["-map", "[aout]"]);
        }
    }

    private static void AddEncodingArguments(
        List<string> arguments,
        SequenceExportPlan plan,
        VideoEncodeProfile profile)
    {
        arguments.AddRange(["-c:v", profile.EncoderName]);
        arguments.AddRange(profile.EncoderArguments);
        arguments.AddRange(["-pix_fmt", "yuv420p", "-r", FormatNumber(plan.OutputFrameRate)]);
        AddAudioEncodingArguments(arguments, plan);
    }

    private static void AddAudioEncodingArguments(List<string> arguments, SequenceExportPlan plan)
    {
        if (!HasAudio(plan))
        {
            return;
        }

        arguments.AddRange(plan.Container is ExportOutputContainer.Mkv
            ? ["-c:a", "libopus", "-b:a", "160k"]
            : ["-c:a", "aac", "-b:a", "192k"]);
    }

    private static void AddMetadataArguments(List<string> arguments, SequenceExportPlan plan)
    {
        arguments.AddRange(["-metadata", "DUBBED_BY=Trackdub"]);
        AddLanguageMetadata(arguments, "source_language", plan.SourceLanguage);
        AddLanguageMetadata(arguments, "target_language", plan.TargetLanguage);
    }

    private static void AddLanguageMetadata(List<string> arguments, string key, string? language)
    {
        if (!string.IsNullOrWhiteSpace(language))
        {
            arguments.AddRange(["-metadata", $"{key}={language.Trim().ToLowerInvariant()}"]);
        }
    }

    private static void AddContainerArguments(List<string> arguments, SequenceExportPlan plan)
    {
        if (plan.Container is ExportOutputContainer.Mp4)
        {
            arguments.AddRange(["-movflags", "+faststart"]);
        }
    }

    private static string BuildVideoPlacementFilter(
        SequencePlacement placement,
        int index,
        VideoFilterSettings settings)
    {
        string[] videoFilters = BuildVideoFilters(placement, settings);
        return $"[{index}:v:0]{string.Join(',', videoFilters)}[v{index}]";
    }

    private static string[] BuildVideoFilters(SequencePlacement placement, VideoFilterSettings settings)
    {
        SequenceFrame frame = settings.Frame;
        SequenceCrop crop = placement.Crop;
        string start = FormatNumber(placement.SourceInSeconds);
        string end = FormatNumber(placement.SourceOutSeconds);
        string zoom = FormatNumber(crop.Zoom);
        string focusX = FormatNumber(crop.FocusX);
        string focusY = FormatNumber(crop.FocusY);
        var filters = new List<string>
        {
            "setpts=PTS-STARTPTS",
            $"trim=start={start}:end={end}",
            "setpts=PTS-STARTPTS"
        };

        AddEncoderVideoPrefix(filters, settings.EncodeProfile);
        filters.Add($"scale={frame.Width}:{frame.Height}:force_original_aspect_ratio=increase:force_divisible_by=2");
        if (crop.Zoom > 1d)
        {
            filters.Add($"scale=ceil(iw*{zoom}/2)*2:ceil(ih*{zoom}/2)*2");
        }
        filters.Add($"crop={frame.Width}:{frame.Height}:(iw-{frame.Width})*{focusX}:(ih-{frame.Height})*{focusY}");
        filters.Add("setsar=1");
        filters.Add($"fps={FormatNumber(settings.OutputFrameRate)}");
        filters.Add("format=yuv420p");
        return [.. filters];
    }

    private static void AddEncoderVideoPrefix(List<string> filters, VideoEncodeProfile? encodeProfile)
    {
        string? prefix = encodeProfile?.VideoFilterPrefix;
        if (!string.IsNullOrWhiteSpace(prefix))
        {
            filters.AddRange(prefix.TrimEnd(',').Split(',', StringSplitOptions.RemoveEmptyEntries));
        }
    }

    private static string BuildAudioPlacementFilter(
        SequencePlacement placement,
        SequenceExportPlan plan,
        int index)
    {
        int audioInputIndex = plan.AudioMode is SequenceExportAudioMode.Dubbed
            ? plan.Sequence.Placements.Count + index
            : index;
        string start = FormatNumber(placement.SourceInSeconds);
        string end = FormatNumber(placement.SourceOutSeconds);
        string filters =
            $"asetpts=PTS-STARTPTS,atrim=start={start}:end={end},asetpts=PTS-STARTPTS,aresample=48000," +
            "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo";
        return $"[{audioInputIndex}:a:0]{filters}[a{index}]";
    }

    private static string BuildConcatenationFilter(SequenceExportPlan plan)
    {
        int count = plan.Sequence.Placements.Count;
        bool hasAudio = HasAudio(plan);
        var inputs = new StringBuilder();
        for (int index = 0; index < count; index++)
        {
            inputs.Append($"[v{index}]");
            if (hasAudio)
            {
                inputs.Append($"[a{index}]");
            }
        }

        return $"{inputs}concat=n={count}:v=1:a={(hasAudio ? 1 : 0)}[vconcat]" +
            (hasAudio ? "[aout]" : string.Empty);
    }

    private static void AppendFinalVideoFilters(
        ICollection<string> filters,
        SequenceExportPlan plan,
        SequenceFrame frame)
    {
        string currentLabel = AddSubtitleFilter(filters, plan.BurnInSubtitlePath);
        if (plan.RequiresWatermark)
        {
            filters.Add(
                $"{currentLabel}{FfmpegMuxCommandBuilder.BuildWatermarkFilter(frame.Height, plan.WatermarkFontPath)}[vout]");
        }
        else
        {
            filters.Add($"{currentLabel}null[vout]");
        }
    }

    private static string AddSubtitleFilter(ICollection<string> filters, string? subtitlePath)
    {
        if (string.IsNullOrWhiteSpace(subtitlePath))
        {
            return "[vconcat]";
        }

        filters.Add($"[vconcat]{FfmpegMuxCommandBuilder.BuildSubtitleFilter(subtitlePath)}[vsub]");
        return "[vsub]";
    }

    private static bool HasAudio(SequenceExportPlan plan) =>
        plan.Sequence.Placements.Count > 0 && plan.AudioMode is not SequenceExportAudioMode.None;

    private static VideoEncodeProfile SoftwareProfile() => FfmpegVideoEncoderSelector.Resolve(
        VideoEncoderPreference.Software,
        FfmpegVideoEncoderSnapshot.Empty);

    private static string FormatNumber(double value) =>
        value.ToString("0.#########", CultureInfo.InvariantCulture);

    private sealed record VideoFilterSettings(
        SequenceFrame Frame,
        int OutputFrameRate,
        VideoEncodeProfile? EncodeProfile);
}
