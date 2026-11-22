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

    public async Task<TailFitResult> FitAsync(MixPlan plan, Guid stageRunId, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(plan);

        List<MixSpeechClip> clips = [.. plan.SpeechClips];
        List<MixPlanWarning> remainingWarnings = [];
        List<string> createdFilePaths = [];
        foreach (MixPlanWarning warning in plan.Warnings)
        {
            int clipIndex = FindFittableClipIndex(clips, warning);
            string? fittedPath = null;
            if (clipIndex >= 0)
            {
                (MixSpeechClip Clip, string Path)? fitted = await TryFitAsync(
                    clips[clipIndex], warning, stageRunId, cancellationToken).ConfigureAwait(false);
                if (fitted is { } result)
                {
                    clips[clipIndex] = result.Clip;
                    fittedPath = result.Path;
                }
            }

            if (fittedPath is null)
            {
                remainingWarnings.Add(warning);
            }
            else
            {
                createdFilePaths.Add(fittedPath);
            }
        }

        return createdFilePaths.Count == 0
            ? new TailFitResult(plan, createdFilePaths)
            : new TailFitResult(plan with { SpeechClips = clips, Warnings = remainingWarnings }, createdFilePaths);
    }

    private static int FindFittableClipIndex(List<MixSpeechClip> clips, MixPlanWarning warning)
    {
        if (warning.Code != MixPlanWarningCode.TakeExceedsSourceAudio ||
            warning.OverhangSeconds is not double overhang ||
            !double.IsFinite(overhang) ||
            overhang <= 0d)
        {
            return -1;
        }

        return clips.FindIndex(clip =>
            clip.SegmentId == warning.SegmentId &&
            !clip.IsSilentGap &&
            !string.IsNullOrWhiteSpace(clip.TakeRelativePath) &&
            clip.TakeDurationSeconds is double duration &&
            double.IsFinite(duration) &&
            duration > 0d);
    }

    private async Task<(MixSpeechClip Clip, string Path)?> TryFitAsync(
        MixSpeechClip clip,
        MixPlanWarning warning,
        Guid stageRunId,
        CancellationToken cancellationToken)
    {
        double duration = clip.TakeDurationSeconds!.Value;
        double overhang = warning.OverhangSeconds!.Value;
        double targetDuration = duration - overhang - SafetyMarginSeconds;
        if (targetDuration <= 0d || duration / targetDuration > MaxTempoRatio)
        {
            return null;
        }

        double tempoRatio = duration / targetDuration;
        string outputRelativePath = ProjectArtifactPaths.GetExportTailFitTakeRelativePath(stageRunId, clip.SegmentId);
        string outputPath = artifactStore.GetPath(outputRelativePath);
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
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Keep the original warning: it names the real problem and the export reports it.
            logger?.LogWarning(
                $"{warning.SegmentReference}: could not fit the take to the end of the source audio: {ex.Message}",
                ex);
            DeleteQuietly(outputPath);
            return null;
        }

        logger?.LogInformation(string.Create(
            CultureInfo.InvariantCulture,
            $"{warning.SegmentReference}: compressed the take {tempoRatio:0.###}x to remove {overhang:0.###} s that ran past the end of the source audio."));
        return (clip with { TakeRelativePath = outputRelativePath, TakeDurationSeconds = targetDuration }, outputPath);
    }

    private static void DeleteQuietly(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }
}
