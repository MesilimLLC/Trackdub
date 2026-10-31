using Trackdub.Application.Editing;
using Trackdub.Contracts;
using Trackdub.Contracts.Editing;
using Trackdub.Domain.Editing;

namespace Trackdub.Media.Muxing;

internal static class FfmpegSequencePlanValidator
{
    public static void Validate(SequenceExportPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        SequenceEditor.Validate(plan.Sequence);
        ArgumentException.ThrowIfNullOrWhiteSpace(plan.OutputPath);
        ArgumentNullException.ThrowIfNull(plan.PlacementAssets);

        ValidateSequenceFrame(plan.Sequence);
        ValidateOutputFrameRate(plan.OutputFrameRate);
        ValidateOptions(plan);
        ValidateOutputContainerExtension(plan);
        ValidatePlacementAssets(plan);
        ValidateOutputInputs(plan);
        ValidateSubtitleInput(plan);
        ValidateWatermarkFont(plan);
    }

    private static void ValidateSequenceFrame(SequenceDocument sequence)
    {
        if (sequence.Placements.Count == 0)
        {
            throw new InvalidOperationException("The sequence has no placements to export.");
        }

        SequenceFrame? frame = sequence.Frame;
        if (!HasValidOutputFrame(frame))
        {
            throw new InvalidOperationException("Sequence export requires an even-width and even-height output frame.");
        }
    }

    private static bool HasValidOutputFrame(SequenceFrame? frame) =>
        frame is { Width: > 0, Height: > 0 } && frame.Width % 2 == 0 && frame.Height % 2 == 0;

    private static void ValidateOutputFrameRate(int outputFrameRate)
    {
        if (outputFrameRate is < 1 or > 120)
        {
            throw new ArgumentOutOfRangeException(nameof(outputFrameRate), "Output frame rate must be between 1 and 120 fps.");
        }
    }

    private static void ValidateOptions(SequenceExportPlan plan)
    {
        if (!Enum.IsDefined(plan.Container))
        {
            throw new ArgumentOutOfRangeException(nameof(plan.Container));
        }
        if (!Enum.IsDefined(plan.VideoEncoder))
        {
            throw new ArgumentOutOfRangeException(nameof(plan.VideoEncoder));
        }
        if (!Enum.IsDefined(plan.AudioMode))
        {
            throw new ArgumentOutOfRangeException(nameof(plan.AudioMode));
        }
    }

    private static void ValidateOutputContainerExtension(SequenceExportPlan plan)
    {
        string expectedExtension = plan.Container switch
        {
            ExportOutputContainer.Mp4 => ".mp4",
            ExportOutputContainer.Mkv => ".mkv",
            _ => throw new ArgumentOutOfRangeException(nameof(plan.Container))
        };
        if (!string.Equals(Path.GetExtension(plan.OutputPath), expectedExtension, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                $"The output path extension must match the selected {plan.Container} container.",
                nameof(plan.OutputPath));
        }
    }

    private static void ValidatePlacementAssets(SequenceExportPlan plan)
    {
        if (plan.PlacementAssets.Count != plan.Sequence.Placements.Count ||
            plan.Sequence.Placements.Any(placement => !plan.PlacementAssets.ContainsKey(placement.Id)))
        {
            throw new InvalidDataException("Every sequence placement must have exactly one resolved media input.");
        }
        if (plan.PlacementAssets.Values.Any(static assets => assets is null))
        {
            throw new InvalidDataException("Every sequence placement must have resolved media paths.");
        }

        bool anyAudioPath = plan.PlacementAssets.Values.Any(static assets => assets.AudioPath is not null);
        bool allAudioPaths = plan.PlacementAssets.Values.All(static assets => assets.AudioPath is not null);
        if (plan.AudioMode is SequenceExportAudioMode.Dubbed && !allAudioPaths)
        {
            throw new InvalidDataException("Dubbed audio must be provided for every sequence placement.");
        }
        if (plan.AudioMode is not SequenceExportAudioMode.Dubbed && anyAudioPath)
        {
            throw new InvalidDataException("Audio files can only be supplied when dubbed audio is selected.");
        }
    }

    private static void ValidateOutputInputs(SequenceExportPlan plan)
    {
        string outputPath = Path.GetFullPath(plan.OutputPath);
        foreach (SequencePlacement placement in plan.Sequence.Placements)
        {
            SequenceRenderPlacementAssets assets = plan.PlacementAssets[placement.Id];
            ValidateInputFile(outputPath, assets.SourceMediaPath, "Source video file for sequence placement was not found.");
            if (plan.AudioMode is SequenceExportAudioMode.Dubbed)
            {
                ValidateInputFile(outputPath, assets.AudioPath!, "Audio file for sequence placement was not found.");
            }
        }
    }

    private static void ValidateSubtitleInput(SequenceExportPlan plan)
    {
        if (!string.IsNullOrWhiteSpace(plan.BurnInSubtitlePath))
        {
            ValidateInputFile(
                Path.GetFullPath(plan.OutputPath),
                plan.BurnInSubtitlePath,
                "Sequence burn-in subtitle file was not found.",
                "Sequence output path must be different from the subtitle input.");
        }
    }

    private static void ValidateWatermarkFont(SequenceExportPlan plan)
    {
        if (!plan.RequiresWatermark || string.IsNullOrWhiteSpace(plan.WatermarkFontPath))
        {
            return;
        }

        ValidateInputFile(
            Path.GetFullPath(plan.OutputPath),
            plan.WatermarkFontPath,
            "Sequence watermark font was not found.",
            "Sequence output path must be different from the watermark font.");
    }

    private static void ValidateInputFile(
        string outputPath,
        string inputPath,
        string notFoundMessage,
        string? samePathMessage = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(inputPath);
        string fullInputPath = Path.GetFullPath(inputPath);
        if (FilePathComparison.AreSame(outputPath, fullInputPath))
        {
            throw new InvalidOperationException(samePathMessage ??
                "Sequence output path must be different from every source and audio input.");
        }
        if (!File.Exists(fullInputPath))
        {
            throw new FileNotFoundException(notFoundMessage, fullInputPath);
        }
    }
}
