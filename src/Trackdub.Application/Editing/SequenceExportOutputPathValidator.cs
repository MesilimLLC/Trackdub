using Trackdub.Contracts;
using Trackdub.Contracts.Editing;

namespace Trackdub.Application.Editing;

internal static class SequenceExportOutputPathValidator
{
    private const string OutputPathParameterName = "outputPath";

    public static void Validate(SequenceExportPlan plan, string outputPath)
    {
        var output = new FileInfo(outputPath);
        ValidateContainerExtension(plan.Container, output);
        ValidatePlacementInputs(plan, output);
        ValidateAuxiliaryInputs(plan, output);
    }

    private static void ValidateContainerExtension(ExportOutputContainer container, FileInfo output)
    {
        string expectedExtension = container switch
        {
            ExportOutputContainer.Mp4 => ".mp4",
            ExportOutputContainer.Mkv => ".mkv",
            _ => throw new ArgumentOutOfRangeException(nameof(container))
        };
        if (!string.Equals(output.Extension, expectedExtension, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                $"The output path extension must match the selected {container} container.",
                OutputPathParameterName);
        }
    }

    private static void ValidatePlacementInputs(SequenceExportPlan plan, FileInfo output)
    {
        foreach (SequenceRenderPlacementAssets assets in plan.PlacementAssets.Values)
        {
            EnsureDifferentPath(output, assets.SourceMediaPath, ExportPathConflict.SourceMedia);
            if (plan.AudioMode is SequenceExportAudioMode.Dubbed)
            {
                EnsureDifferentPath(output, assets.AudioPath, ExportPathConflict.Audio);
            }
        }
    }

    private static void ValidateAuxiliaryInputs(SequenceExportPlan plan, FileInfo output)
    {
        EnsureDifferentPath(output, plan.BurnInSubtitlePath, ExportPathConflict.Subtitle);
        EnsureDifferentPath(output, plan.WatermarkFontPath, ExportPathConflict.WatermarkFont);
    }

    private static void EnsureDifferentPath(FileInfo output, string? inputPath, ExportPathConflict conflict)
    {
        if (string.IsNullOrWhiteSpace(inputPath))
        {
            return;
        }

        // Compare resolved filesystem identity, not just normalized strings: an input that is
        // a symlink (or lives under a symlinked directory) can resolve to the requested output,
        // and publishing over it would destroy the source media.
        if (FilePathComparison.AreSame(ResolveRealPath(output), ResolveRealPath(new FileInfo(inputPath))))
        {
            throw new InvalidOperationException(BuildConflictMessage(conflict));
        }
    }

    private static string BuildConflictMessage(ExportPathConflict conflict) => conflict switch
    {
        ExportPathConflict.SourceMedia => "Sequence output path must be different from every source input.",
        ExportPathConflict.Audio => "Sequence output path must be different from every audio input.",
        ExportPathConflict.Subtitle => "Sequence output path must be different from the subtitle input.",
        ExportPathConflict.WatermarkFont => "Sequence output path must be different from the watermark font.",
        _ => throw new ArgumentOutOfRangeException(nameof(conflict))
    };

    private static string ResolveRealPath(FileInfo path)
    {
        string fullPath = path.FullName;
        string root = Path.GetPathRoot(fullPath) ?? string.Empty;
        string[] segments = fullPath[root.Length..].Split(
            [Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar],
            StringSplitOptions.RemoveEmptyEntries);
        string current = root;
        foreach (string segment in segments)
        {
            current = ResolveLink(Path.Join(current, segment));
        }

        return current;
    }

    private static string ResolveLink(string path)
    {
        try
        {
            FileSystemInfo info = Directory.Exists(path) ? new DirectoryInfo(path) : new FileInfo(path);
            if (!info.Exists)
            {
                return path;
            }

            if (info.LinkTarget is null)
            {
                return path;
            }

            FileSystemInfo? target = info.ResolveLinkTarget(returnFinalTarget: true);
            return target is null ? path : Path.GetFullPath(target.FullName);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // A path whose links cannot be resolved may still alias a protected file, so refuse
            // the export rather than compare an unresolved path.
            throw new InvalidDataException($"The output path could not be resolved safely: {path}", ex);
        }
    }

    private enum ExportPathConflict
    {
        SourceMedia,
        Audio,
        Subtitle,
        WatermarkFont
    }
}
