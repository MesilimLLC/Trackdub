using Trackdub.Contracts;
using Trackdub.Contracts.Editing;
using Trackdub.Domain.Editing;
using Trackdub.Domain.Media;

namespace Trackdub.Application.Editing;

/// <summary>Applies sequence scope and tier policy before publishing a rendered sequence.</summary>
public sealed class SequenceExportDeliveryService(
    ISequenceExportRenderer renderer,
    IMediaProbe mediaProbe,
    IExportTierGate? exportTierGate = null)
{
    private readonly ISequenceExportRenderer renderer = renderer ?? throw new ArgumentNullException(nameof(renderer));
    private readonly IMediaProbe mediaProbe = mediaProbe ?? throw new ArgumentNullException(nameof(mediaProbe));

    public async Task<SequenceExportDeliveryResult> ExportAsync(
        SequenceExportPlan plan,
        Guid? selectedPlacementId = null,
        CancellationToken cancellationToken = default)
    {
        PreparedSequenceExport export = Prepare(plan, selectedPlacementId);
        string? blockedReason = CheckDuration(export.PlannedDuration);
        if (!string.IsNullOrWhiteSpace(blockedReason))
        {
            return SequenceExportDeliveryResult.Blocked(blockedReason);
        }

        return await RenderAndPublishAsync(export, cancellationToken).ConfigureAwait(false);
    }

    private PreparedSequenceExport Prepare(SequenceExportPlan plan, Guid? selectedPlacementId)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentException.ThrowIfNullOrWhiteSpace(plan.OutputPath);
        if (selectedPlacementId is not null && !string.IsNullOrWhiteSpace(plan.BurnInSubtitlePath))
        {
            throw new InvalidOperationException(
                "Burn-in subtitles cannot be exported with a selected placement until they are rebased to that placement.");
        }

        SequenceDocument scopedSequence = CreateScope(plan.Sequence, selectedPlacementId);
        IReadOnlyDictionary<Guid, SequenceRenderPlacementAssets> scopedAssets =
            SelectPlacementAssets(plan, scopedSequence);
        string outputPath = Path.GetFullPath(plan.OutputPath);
        SequenceExportOutputPathValidator.Validate(plan, outputPath);

        SequenceExportPlan scopedPlan = plan with
        {
            Sequence = scopedSequence,
            PlacementAssets = scopedAssets,
            OutputPath = outputPath,
            RequiresWatermark = plan.RequiresWatermark || exportTierGate?.RequiresWatermark == true
        };
        TimeSpan plannedDuration = TimeSpan.FromSeconds(SequenceEditor.GetPositions(scopedSequence)[^1].SequenceEndSeconds);
        return new PreparedSequenceExport(scopedPlan, outputPath, plannedDuration);
    }

    private async Task<SequenceExportDeliveryResult> RenderAndPublishAsync(
        PreparedSequenceExport export,
        CancellationToken cancellationToken)
    {
        TemporarySequenceExport temporary = CreateTemporaryOutput(export);

        try
        {
            SequenceExportRenderResult render = await RenderTemporaryAsync(temporary, cancellationToken)
                .ConfigureAwait(false);
            TimeSpan renderedDuration = await ProbeRenderedDurationAsync(temporary, cancellationToken)
                .ConfigureAwait(false);
            string? blockedReason = CheckDuration(renderedDuration);
            if (!string.IsNullOrWhiteSpace(blockedReason))
            {
                return SequenceExportDeliveryResult.Blocked(blockedReason, render.Warnings);
            }

            string? plannedMismatch = CheckRenderedDuration(
                renderedDuration,
                export.PlannedDuration,
                export.Plan.OutputFrameRate);
            if (!string.IsNullOrWhiteSpace(plannedMismatch))
            {
                return SequenceExportDeliveryResult.Blocked(plannedMismatch, render.Warnings);
            }

            cancellationToken.ThrowIfCancellationRequested();
            File.Move(temporary.Path, export.OutputPath, overwrite: true);
            return SequenceExportDeliveryResult.Published(export.Plan, renderedDuration, render.Warnings);
        }
        finally
        {
            temporary.DeleteIfPresent();
        }
    }

    private TemporarySequenceExport CreateTemporaryOutput(PreparedSequenceExport export)
    {
        string directory = Path.GetDirectoryName(export.OutputPath)!;
        Directory.CreateDirectory(directory);
        string fileName = Path.GetFileNameWithoutExtension(export.OutputPath);
        string extension = Path.GetExtension(export.OutputPath);
        string path = Path.Join(directory, $"{fileName}.{Guid.NewGuid():N}.partial{extension}");
        return new TemporarySequenceExport(export, path);
    }

    private async Task<SequenceExportRenderResult> RenderTemporaryAsync(
        TemporarySequenceExport temporary,
        CancellationToken cancellationToken)
    {
        SequenceExportPlan plan = temporary.Export.Plan with { OutputPath = temporary.Path };
        SequenceExportRenderResult result = await renderer.RenderAsync(plan, cancellationToken).ConfigureAwait(false);
        if (!FilePathComparison.AreSame(result.OutputPath, temporary.Path) || !temporary.IsUsable)
        {
            throw new InvalidOperationException("The sequence renderer did not produce the requested temporary output.");
        }

        return result;
    }

    private async Task<TimeSpan> ProbeRenderedDurationAsync(
        TemporarySequenceExport temporary,
        CancellationToken cancellationToken)
    {
        MediaProbeSnapshot probe = await mediaProbe.ProbeAsync(temporary.Path, cancellationToken).ConfigureAwait(false);
        return TimeSpan.FromSeconds(ResolveVideoDurationSeconds(probe));
    }

    private string? CheckDuration(TimeSpan duration) => exportTierGate?.CheckDurationGate(duration);

    private static string? CheckRenderedDuration(
        TimeSpan renderedDuration,
        TimeSpan plannedDuration,
        int outputFrameRate)
    {
        // Allow a couple of frames of rounding, but never publish a visibly truncated render.
        double toleranceSeconds = 2d / Math.Max(outputFrameRate, 1);
        if (renderedDuration.TotalSeconds + toleranceSeconds < plannedDuration.TotalSeconds)
        {
            return $"The rendered output duration is shorter than the planned sequence duration " +
                $"({renderedDuration.TotalSeconds:0.###}s < {plannedDuration.TotalSeconds:0.###}s).";
        }

        return null;
    }

    private static SequenceDocument CreateScope(SequenceDocument sequence, Guid? selectedPlacementId)
    {
        ArgumentNullException.ThrowIfNull(sequence);
        SequenceEditor.Validate(sequence);
        if (selectedPlacementId is not Guid id)
        {
            if (sequence.Placements.Count == 0)
            {
                throw new InvalidOperationException("The sequence has no placements to export.");
            }

            return sequence;
        }

        SequencePlacement selectedPlacement = sequence.Placements
            .SingleOrDefault(placement => placement.Id == id)
            ?? throw new InvalidOperationException("The selected sequence placement was not found.");
        return sequence with { Placements = [selectedPlacement] };
    }

    private static IReadOnlyDictionary<Guid, SequenceRenderPlacementAssets> SelectPlacementAssets(
        SequenceExportPlan plan,
        SequenceDocument scopedSequence)
    {
        ArgumentNullException.ThrowIfNull(plan.PlacementAssets);
        if (plan.PlacementAssets.Count != plan.Sequence.Placements.Count ||
            plan.Sequence.Placements.Any(placement => !plan.PlacementAssets.ContainsKey(placement.Id)))
        {
            throw new InvalidDataException("Every sequence placement must have exactly one resolved media input.");
        }

        var selectedAssets = new Dictionary<Guid, SequenceRenderPlacementAssets>();
        foreach (SequencePlacement placement in scopedSequence.Placements)
        {
            if (!plan.PlacementAssets.TryGetValue(placement.Id, out SequenceRenderPlacementAssets? assets) || assets is null)
            {
                throw new InvalidDataException($"Placement {placement.Id} has no resolved media input.");
            }

            selectedAssets.Add(placement.Id, assets);
        }

        return selectedAssets;
    }

    private static double ResolveVideoDurationSeconds(MediaProbeSnapshot outputProbe)
    {
        ArgumentNullException.ThrowIfNull(outputProbe);
        if (outputProbe.VideoStreams.Count == 0)
        {
            throw new InvalidDataException("The rendered sequence contains no video stream.");
        }

        double duration = outputProbe.VideoStreams
            .Select(static stream => stream.DurationSeconds)
            .Where(double.IsFinite)
            .DefaultIfEmpty(0d)
            .Max();
        if (duration <= 0d)
        {
            duration = outputProbe.DurationSeconds;
        }

        if (!double.IsFinite(duration) || duration <= 0d)
        {
            throw new InvalidDataException("The rendered sequence has no valid video duration.");
        }

        return duration;
    }

    private sealed record PreparedSequenceExport(
        SequenceExportPlan Plan,
        string OutputPath,
        TimeSpan PlannedDuration);

    private sealed record TemporarySequenceExport(PreparedSequenceExport Export, string Path)
    {
        public bool IsUsable => File.Exists(Path) && new FileInfo(Path).Length > 0;

        public void DeleteIfPresent()
        {
            if (File.Exists(Path))
            {
                File.Delete(Path);
            }
        }
    }
}

public sealed record SequenceExportDeliveryResult
{
    private SequenceExportDeliveryResult() { }

    public string? OutputPath { get; private init; }

    public double? DurationSeconds { get; private init; }

    public string? BlockedReason { get; private init; }

    public IReadOnlyList<string> Warnings { get; private init; } = [];

    public bool IsBlocked => !string.IsNullOrWhiteSpace(BlockedReason);

    public static SequenceExportDeliveryResult Published(
        SequenceExportPlan plan,
        TimeSpan duration,
        IReadOnlyList<string> warnings) =>
        new()
        {
            OutputPath = plan.OutputPath,
            DurationSeconds = duration.TotalSeconds,
            Warnings = warnings
        };

    public static SequenceExportDeliveryResult Blocked(
        string reason,
        IReadOnlyList<string>? warnings = null) =>
        new()
        {
            BlockedReason = reason,
            Warnings = warnings ?? []
        };
}
