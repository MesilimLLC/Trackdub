using Trackdub.Contracts.Editing;
using Trackdub.Contracts;
using Trackdub.Domain.Editing;
using Trackdub.Domain.Media;
using Trackdub.Media.Probe;

namespace Trackdub.Media.Muxing;

internal sealed class FfmpegSequenceMediaRangeValidator(IMediaProbe mediaProbe)
{
    private const double MediaDurationToleranceSeconds = 0.05d;
    private readonly IMediaProbe mediaProbe = mediaProbe ?? throw new ArgumentNullException(nameof(mediaProbe));

    public async Task ValidateAsync(SequenceExportPlan plan, CancellationToken cancellationToken)
    {
        var probeCache = new Dictionary<string, MediaProbeSnapshot>(
            OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
        foreach (SequencePlacement placement in plan.Sequence.Placements)
        {
            cancellationToken.ThrowIfCancellationRequested();
            SequenceRenderPlacementAssets assets = plan.PlacementAssets[placement.Id];
            MediaProbeSnapshot video = await ProbeCachedAsync(assets.SourceMediaPath, probeCache, cancellationToken)
                .ConfigureAwait(false);
            ValidateMediaRange(placement, video, MediaRangeKind.SourceVideo);
            var context = new PlacementMediaContext(plan.AudioMode, placement, assets, video, probeCache);
            await ValidateAudioRangeAsync(context, cancellationToken)
                .ConfigureAwait(false);
        }
    }

    private async Task ValidateAudioRangeAsync(
        PlacementMediaContext context,
        CancellationToken cancellationToken)
    {
        if (context.AudioMode is SequenceExportAudioMode.OriginalSource)
        {
            ValidateMediaRange(context.Placement, context.SourceVideo, MediaRangeKind.OriginalSourceAudio);
        }
        else if (context.AudioMode is SequenceExportAudioMode.Dubbed)
        {
            MediaProbeSnapshot audio = await ProbeCachedAsync(
                context.Assets.AudioPath!, context.ProbeCache, cancellationToken)
                .ConfigureAwait(false);
            ValidateMediaRange(context.Placement, audio, MediaRangeKind.DubbedAudio);
        }
    }

    private static void ValidateMediaRange(
        SequencePlacement placement,
        MediaProbeSnapshot media,
        MediaRangeKind rangeKind)
    {
        bool hasStreams = rangeKind is MediaRangeKind.SourceVideo
            ? media.VideoStreams.Count > 0
            : media.AudioStreams.Count > 0;
        if (!hasStreams)
        {
            throw new InvalidDataException(CreateRangeFailureMessage(placement, rangeKind, missingStream: true));
        }

        double availableDurationSeconds = rangeKind is MediaRangeKind.SourceVideo
            ? ResolveStreamDuration(media.VideoStreams, media.DurationSeconds)
            : ResolveStreamDuration(media.AudioStreams, media.DurationSeconds);
        if (availableDurationSeconds + MediaDurationToleranceSeconds < placement.SourceOutSeconds)
        {
            throw new InvalidDataException(CreateRangeFailureMessage(placement, rangeKind, missingStream: false));
        }
    }

    private static string CreateRangeFailureMessage(
        SequencePlacement placement,
        MediaRangeKind rangeKind,
        bool missingStream) => (rangeKind, missingStream) switch
        {
            (MediaRangeKind.SourceVideo, _) =>
                $"Source video for placement {placement.Id} does not contain the selected source range.",
            (MediaRangeKind.OriginalSourceAudio, _) =>
                $"Original source audio for placement {placement.Id} does not contain the selected source range. " +
                "Choose dubbed audio or turn audio off for this export.",
            (MediaRangeKind.DubbedAudio, true) =>
                $"The selected audio for placement {placement.Id} contains no audio stream.",
            (MediaRangeKind.DubbedAudio, false) =>
                $"The selected audio for placement {placement.Id} ends before the selected source range. " +
                "Regenerate or relink audio before exporting this placement.",
            _ => throw new ArgumentOutOfRangeException(nameof(rangeKind))
        };

    private async Task<MediaProbeSnapshot> ProbeCachedAsync(
        string mediaPath,
        Dictionary<string, MediaProbeSnapshot> probeCache,
        CancellationToken cancellationToken)
    {
        string fullPath = Path.GetFullPath(mediaPath);
        if (probeCache.TryGetValue(fullPath, out MediaProbeSnapshot? cached))
        {
            return cached;
        }

        MediaProbeSnapshot snapshot = await mediaProbe.ProbeAsync(fullPath, cancellationToken).ConfigureAwait(false);
        probeCache.Add(fullPath, snapshot);
        return snapshot;
    }

    private static double ResolveStreamDuration<TStream>(
        IReadOnlyList<TStream> streams,
        double containerDuration) where TStream : class
    {
        if (streams.Count == 0)
        {
            return containerDuration;
        }

        // The render command maps stream 0 of each type (:v:0 / :a:0), so validate that
        // stream's duration rather than the longest stream in the container.
        double duration = streams[0] switch
        {
            MediaAudioStream audio => audio.DurationSeconds,
            MediaVideoStream video => video.DurationSeconds,
            _ => 0d
        };
        return double.IsFinite(duration) && duration > 0d ? duration : containerDuration;
    }

    private sealed record PlacementMediaContext(
        SequenceExportAudioMode AudioMode,
        SequencePlacement Placement,
        SequenceRenderPlacementAssets Assets,
        MediaProbeSnapshot SourceVideo,
        Dictionary<string, MediaProbeSnapshot> ProbeCache);

    private enum MediaRangeKind
    {
        SourceVideo,
        OriginalSourceAudio,
        DubbedAudio
    }
}
