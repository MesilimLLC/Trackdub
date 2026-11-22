using System.Globalization;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts;
using Trackdub.Contracts.Projects;
using Trackdub.Domain.Mixing;

namespace Trackdub.Application.Mixing;

public sealed record TailFitResult(MixPlan Plan, IReadOnlyList<string> CreatedFilePaths);

/// <summary>
/// Fits a take that still runs past the end of the source audio once tail reflow has moved everything
/// it can. The renderer cuts output at the source end, so such a take would lose speech at export.
/// </summary>
/// <remarks>
/// Only the overhanging take is compressed, by exactly its overhang plus a small margin, so a take that
/// is a few seconds too long becomes a few percent faster instead of failing the export. The per-segment
/// auto-stretch limit does not apply here: it bounds how far a take may exceed its own segment, which says
/// nothing about whether the clip has room for it. A take that would need more than
/// <see cref="MaxTempoRatio"/> is left alone and stays a warning, because that much compression is audible
/// and the translation should be shortened instead.
/// </remarks>
public sealed class TailOverhangFitter(
    IAudioTimeStretchService audioTimeStretchService,
    IArtifactStore artifactStore,
    IApplicationLogger? logger = null)
{
    public const double MaxTempoRatio = 1.25d;

    // The overhang check tolerates 10 ms; ffmpeg's tempo filter can land a few ms either side of the
    // requested length, so aim a little short of the source end.
    internal const double SafetyMarginSeconds = 0.05d;

    private readonly IAudioTimeStretchService audioTimeStretchService =
        audioTimeStretchService ?? throw new ArgumentNullException(nameof(audioTimeStretchService));
    private readonly IArtifactStore artifactStore =
        artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));

    private sealed record FittedClip(int Index, MixSpeechClip Clip, string FilePath);

    public async Task<TailFitResult> FitAsync(MixPlan plan, Guid stageRunId, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(plan);

        List<MixSpeechClip> clips = [.. plan.SpeechClips];
        List<MixPlanWarning> remainingWarnings = [];
        List<string> createdFilePaths = [];
        foreach (MixPlanWarning warning in plan.Warnings)
        {
            FittedClip? fitted = await TryFitWarningAsync(clips, warning, stageRunId, cancellationToken)
                .ConfigureAwait(false);
            if (fitted is null)
            {
                remainingWarnings.Add(warning);
                continue;
            }

            clips[fitted.Index] = fitted.Clip;
            createdFilePaths.Add(fitted.FilePath);
        }

        return createdFilePaths.Count == 0
            ? new TailFitResult(plan, createdFilePaths)
            : new TailFitResult(plan with { SpeechClips = clips, Warnings = remainingWarnings }, createdFilePaths);
    }

    private async Task<FittedClip?> TryFitWarningAsync(
        List<MixSpeechClip> clips,
        MixPlanWarning warning,
        Guid stageRunId,
        CancellationToken cancellationToken)
    {
        int index = FindFittableClipIndex(clips, warning);
        return index < 0
            ? null
            : await TryFitAsync(index, clips[index], warning, stageRunId, cancellationToken).ConfigureAwait(false);
    }

    private static int FindFittableClipIndex(List<MixSpeechClip> clips, MixPlanWarning warning) =>
        IsMeasuredTailOverhang(warning)
            ? clips.FindIndex(clip => clip.SegmentId == warning.SegmentId && HasPlacedTake(clip))
            : -1;

    private static bool IsMeasuredTailOverhang(MixPlanWarning warning) =>
        warning.Code == MixPlanWarningCode.TakeExceedsSourceAudio && IsPositiveFinite(warning.OverhangSeconds);

    private static bool HasPlacedTake(MixSpeechClip clip) =>
        !clip.IsSilentGap &&
        !string.IsNullOrWhiteSpace(clip.TakeRelativePath) &&
        IsPositiveFinite(clip.TakeDurationSeconds);

    private static bool IsPositiveFinite(double? value) =>
        value is double number && double.IsFinite(number) && number > 0d;

    private async Task<FittedClip?> TryFitAsync(
        int index,
        MixSpeechClip clip,
        MixPlanWarning warning,
        Guid stageRunId,
        CancellationToken cancellationToken)
    {
        double duration = clip.TakeDurationSeconds!.Value;
        double overhang = warning.OverhangSeconds!.Value;
        double targetDuration = duration - overhang - SafetyMarginSeconds;
        if (!IsWithinFitLimit(duration, targetDuration))
        {
            return null;
        }

        double tempoRatio = duration / targetDuration;
        string outputRelativePath = ProjectArtifactPaths.GetExportTailFitTakeRelativePath(stageRunId, clip.SegmentId);
        string outputPath = artifactStore.GetPath(outputRelativePath);
        if (!await TryStretchAsync(clip, warning, tempoRatio, outputPath, cancellationToken).ConfigureAwait(false))
        {
            return null;
        }

        logger?.LogInformation(string.Create(
            CultureInfo.InvariantCulture,
            $"{warning.SegmentReference}: compressed the take {tempoRatio:0.###}x to remove {warning.OverhangSeconds:0.###} s that ran past the end of the source audio."));
        MixSpeechClip fitted = clip with { TakeRelativePath = outputRelativePath, TakeDurationSeconds = targetDuration };
        return new FittedClip(index, fitted, outputPath);
    }

    private static bool IsWithinFitLimit(double duration, double targetDuration) =>
        targetDuration > 0d && duration / targetDuration <= MaxTempoRatio;

    private async Task<bool> TryStretchAsync(
        MixSpeechClip clip,
        MixPlanWarning warning,
        double tempoRatio,
        string outputPath,
        CancellationToken cancellationToken)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
            await audioTimeStretchService.StretchAsync(
                new AudioTimeStretchRequest(
                    artifactStore.GetPath(clip.TakeRelativePath!),
                    outputPath,
                    tempoRatio,
                    EnableRubberband: true,
                    TtsTimingOptions.Default.RubberbandStretchThreshold),
                cancellationToken).ConfigureAwait(false);
            return true;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Keep the original warning: it names the real problem and the export reports it.
            logger?.LogWarning(
                $"{warning.SegmentReference}: could not fit the take to the end of the source audio: {ex.Message}",
                ex);
            DeleteQuietly(outputPath);
            return false;
        }
    }

    private static void DeleteQuietly(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // Best effort: a partial file that cannot be removed only costs disk space.
        }
    }
}
