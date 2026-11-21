using Trackdub.Domain.Transcript;

namespace Trackdub.Application.Transcripts;

public enum TtsDurationSeverity
{
    None = 0,
    Green = 1,
    Yellow = 2,
    Red = 3
}

public sealed record DurationAnalysisResult(
    double OriginalDurationSeconds,
    double? TtsDurationSeconds,
    double? DurationDifferenceRatio,
    double? OverrunRatio,
    double? TempoRatio,
    TtsDurationSeverity Severity,
    bool IsStretchable,
    bool AutoStretchEligible,
    bool HasSpeedLimitWarning)
{
    /// <summary>
    /// Tempo the TTS stage applies automatically, or null for none. A take within
    /// <see cref="TtsTimingOptions.AutoStretchMaxOverrun"/> is fitted to its segment exactly. With
    /// <see cref="TtsTimingOptions.AutoStretchToLimit"/>, a longer one is compressed by the limit, so it
    /// overruns less instead of not being touched at all.
    /// </summary>
    public double? AutoStretchTempoRatio { get; init; }
}

public sealed class DurationAnalysisService
{
    public DurationAnalysisResult Analyze(
        TranscriptSegment sourceSegment,
        double? ttsDurationSeconds,
        TtsTimingOptions? options = null)
    {
        TtsTimingOptions normalizedOptions = (options ?? TtsTimingOptions.Default).Normalize();
        double originalDurationSeconds = sourceSegment.EndSeconds - sourceSegment.StartSeconds;
        if (!double.IsFinite(originalDurationSeconds) || originalDurationSeconds <= 0d ||
            ttsDurationSeconds is null ||
            !double.IsFinite(ttsDurationSeconds.Value) ||
            ttsDurationSeconds.Value <= 0d)
        {
            return new DurationAnalysisResult(
                Math.Max(0d, originalDurationSeconds),
                ttsDurationSeconds,
                DurationDifferenceRatio: null,
                OverrunRatio: null,
                TempoRatio: null,
                TtsDurationSeverity.None,
                IsStretchable: false,
                AutoStretchEligible: false,
                HasSpeedLimitWarning: false);
        }

        double durationDifferenceRatio = (ttsDurationSeconds.Value - originalDurationSeconds) / originalDurationSeconds;
        double overrunRatio = Math.Max(0d, durationDifferenceRatio);
        double tempoRatio = ttsDurationSeconds.Value / originalDurationSeconds;
        bool isStretchable = originalDurationSeconds >= normalizedOptions.MinimumStretchableDurationSeconds &&
                             ttsDurationSeconds.Value >= normalizedOptions.MinimumStretchableDurationSeconds &&
                             tempoRatio is >= 0.25d and <= 4.0d;
        bool autoStretchEligible = isStretchable &&
                                   overrunRatio > 0d &&
                                   overrunRatio <= normalizedOptions.AutoStretchMaxOverrun;

        double? autoStretchTempoRatio = autoStretchEligible
            ? tempoRatio
            : normalizedOptions.AutoStretchToLimit &&
              isStretchable &&
              overrunRatio > normalizedOptions.AutoStretchMaxOverrun &&
              normalizedOptions.AutoStretchMaxOverrun > 0d
                ? 1d + normalizedOptions.AutoStretchMaxOverrun
                : null;

        return new DurationAnalysisResult(
            originalDurationSeconds,
            ttsDurationSeconds,
            durationDifferenceRatio,
            overrunRatio,
            tempoRatio,
            ResolveSeverity(overrunRatio, normalizedOptions),
            isStretchable,
            autoStretchEligible,
            tempoRatio > normalizedOptions.SpeedLimitWarningRatio)
        {
            AutoStretchTempoRatio = autoStretchTempoRatio
        };
    }

    private static TtsDurationSeverity ResolveSeverity(double overrunRatio, TtsTimingOptions options)
    {
        if (overrunRatio > options.DurationCriticalThreshold)
        {
            return TtsDurationSeverity.Red;
        }

        if (overrunRatio > options.DurationWarningThreshold)
        {
            return TtsDurationSeverity.Yellow;
        }

        return TtsDurationSeverity.Green;
    }
}
