using Trackdub.Contracts;
using Trackdub.Sdk;

namespace Trackdub.Cli;

/// <summary>
/// Resolves effective TTS timing settings for a CLI pipeline run.
/// Precedence: explicit flags &gt; host settings (settings.json / in-memory overlay) &gt; defaults.
/// </summary>
internal static class CliTtsTimingResolver
{
    public static async Task<(TtsTimingSettings? Settings, int ExitCode)> ResolveAsync(
        TrackdubSessionFactory factory,
        bool? enableRubberbandStretch,
        bool noRubberbandStretch,
        double? rubberbandStretchThreshold,
        double? autoStretchMaxOverrun,
        bool autoStretchToLimit,
        CancellationToken cancellationToken)
    {
        if (rubberbandStretchThreshold is double threshold &&
            (!double.IsFinite(threshold) || threshold is < 0d or > 1d))
        {
            CliErrorReporter.ReportValidationError(
                ErrorCode.InvalidArgument,
                $"Option '--tts-rubberband-threshold' must be between 0 and 1 inclusive (got {rubberbandStretchThreshold}).",
                "--tts-rubberband-threshold");
            return (null, Program.ExitArgumentError);
        }

        if (autoStretchMaxOverrun is double autoStretch &&
            (!double.IsFinite(autoStretch) || autoStretch is < 0d or > 1d))
        {
            CliErrorReporter.ReportValidationError(
                ErrorCode.InvalidArgument,
                $"Option '--tts-auto-stretch-max-overrun' must be between 0 and 1 inclusive (got {autoStretch}).",
                "--tts-auto-stretch-max-overrun");
            return (null, Program.ExitArgumentError);
        }

        bool? enable = enableRubberbandStretch;
        if (noRubberbandStretch)
        {
            if (enable is true)
            {
                CliErrorReporter.ReportValidationError(
                    ErrorCode.InvalidArgument,
                    "Options '--tts-rubberband-stretch' and '--no-tts-rubberband-stretch' cannot both be set.",
                    "--no-tts-rubberband-stretch");
                return (null, Program.ExitArgumentError);
            }

            enable = false;
        }

        TtsTimingSettings baseline = await LoadBaselineAsync(factory, cancellationToken).ConfigureAwait(false);

        if (enable is null && rubberbandStretchThreshold is null && autoStretchMaxOverrun is null && !autoStretchToLimit)
        {
            // No flags: keep host/default settings as-is for this run.
            return (baseline, Program.ExitSuccess);
        }

        var resolved = new TtsTimingSettings(
            EnableRubberbandStretch: enable ?? baseline.EnableRubberbandStretch,
            RubberbandStretchThreshold: rubberbandStretchThreshold ?? baseline.RubberbandStretchThreshold,
            AutoStretchMaxOverrun: autoStretchMaxOverrun ?? baseline.AutoStretchMaxOverrun,
            AutoStretchToLimit: autoStretchToLimit ? true : baseline.AutoStretchToLimit);
        return (resolved, Program.ExitSuccess);
    }

    private static async Task<TtsTimingSettings> LoadBaselineAsync(
        TrackdubSessionFactory factory,
        CancellationToken cancellationToken)
    {
        try
        {
            IStudioSettingsService settingsService = factory.GetRequiredService<IStudioSettingsService>();
            StudioSettings settings = await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
            return settings.TtsTiming ?? TtsTimingSettings.Default;
        }
        catch
        {
            return TtsTimingSettings.Default;
        }
    }
}
