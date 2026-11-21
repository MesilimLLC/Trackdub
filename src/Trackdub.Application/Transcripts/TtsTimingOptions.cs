namespace Trackdub.Application.Transcripts;

public sealed record TtsTimingOptions(
    double DurationWarningThreshold,
    double DurationCriticalThreshold,
    double AutoStretchMaxOverrun,
    double SpeedLimitWarningRatio,
    double MinimumStretchableDurationSeconds,
    bool EnableRubberbandStretch,
    double RubberbandStretchThreshold)
{
    public static TtsTimingOptions Default { get; } = new(
        DurationWarningThreshold: 0.10d,
        DurationCriticalThreshold: 0.25d,
        AutoStretchMaxOverrun: 0.20d,
        SpeedLimitWarningRatio: 1.50d,
        MinimumStretchableDurationSeconds: 0.50d,
        EnableRubberbandStretch: false,
        RubberbandStretchThreshold: 0.15d);

    /// <summary>
    /// When true, a take that overruns its segment by more than <see cref="AutoStretchMaxOverrun"/> is
    /// still compressed by that limit. Off by default: interactive projects leave such takes untouched
    /// and flag them for a manual stretch, which works from the unstretched take.
    /// </summary>
    public bool AutoStretchToLimit { get; init; }

    public TtsTimingOptions Normalize() =>
        this with
        {
            DurationWarningThreshold = NormalizeThreshold(DurationWarningThreshold, Default.DurationWarningThreshold),
            DurationCriticalThreshold = NormalizeThreshold(DurationCriticalThreshold, Default.DurationCriticalThreshold),
            AutoStretchMaxOverrun = NormalizeThreshold(AutoStretchMaxOverrun, Default.AutoStretchMaxOverrun),
            SpeedLimitWarningRatio = double.IsFinite(SpeedLimitWarningRatio) && SpeedLimitWarningRatio > 1d
                ? SpeedLimitWarningRatio
                : Default.SpeedLimitWarningRatio,
            MinimumStretchableDurationSeconds = double.IsFinite(MinimumStretchableDurationSeconds) && MinimumStretchableDurationSeconds > 0d
                ? MinimumStretchableDurationSeconds
                : Default.MinimumStretchableDurationSeconds,
            RubberbandStretchThreshold = NormalizeThreshold(RubberbandStretchThreshold, Default.RubberbandStretchThreshold)
        };

    private static double NormalizeThreshold(double value, double fallback) =>
        double.IsFinite(value) && value >= 0d && value <= 1d
            ? value
            : fallback;
}
