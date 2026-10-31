using Trackdub.Domain.Editing;

namespace Trackdub.Contracts.Editing;

/// <summary>Renders the linked video and audio tracks of an ordered edit sequence.</summary>
public interface ISequenceExportRenderer
{
    Task<SequenceExportRenderResult> RenderAsync(
        SequenceExportPlan plan,
        CancellationToken cancellationToken);
}

/// <summary>Audio track to use for every sequence placement.</summary>
public enum SequenceExportAudioMode
{
    OriginalSource = 0,
    Dubbed = 1,
    None = 2,
}

/// <summary>
/// Inputs resolved by the owning project workflow. Audio paths are required only for dubbed audio
/// and must be full-source timelines aligned to the source media's zero timestamp.
/// </summary>
public sealed record SequenceRenderPlacementAssets(
    string SourceMediaPath,
    string? AudioPath);

/// <param name="PlacementAssets">One entry per placement, keyed by its stable placement ID.</param>
/// <param name="BurnInSubtitlePath">Optional subtitle file whose cue times are sequence-relative.</param>
public sealed record SequenceExportPlan(
    SequenceDocument Sequence,
    IReadOnlyDictionary<Guid, SequenceRenderPlacementAssets> PlacementAssets,
    string OutputPath,
    ExportOutputContainer Container,
    SequenceExportAudioMode AudioMode,
    VideoEncoderPreference VideoEncoder = VideoEncoderPreference.Auto,
    bool RequiresWatermark = false,
    string? WatermarkFontPath = null,
    string? BurnInSubtitlePath = null,
    string? SourceLanguage = null,
    string? TargetLanguage = null,
    int OutputFrameRate = 30);

public sealed record SequenceExportRenderResult(
    string OutputPath,
    IReadOnlyList<string> Warnings);
