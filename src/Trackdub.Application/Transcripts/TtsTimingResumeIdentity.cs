using System.Globalization;

namespace Trackdub.Application.Transcripts;

internal static class TtsTimingResumeIdentity
{
    internal const string SnapshotKey = "TtsTiming.AutoStretchMaxOverrun";
    private const string FingerprintSuffix = ":auto-stretch-v1=";

    // Only present when AutoStretchToLimit is on, so identities for the default policy are unchanged.
    private const string ToLimitMarker = "+to-limit";

    internal static string Normalize(double? value, bool autoStretchToLimit = false) =>
        string.Concat(
            (TtsTimingOptions.Default with
            {
                AutoStretchMaxOverrun = value ?? TtsTimingOptions.Default.AutoStretchMaxOverrun
            }).Normalize().AutoStretchMaxOverrun.ToString("G17", CultureInfo.InvariantCulture),
            autoStretchToLimit ? ToLimitMarker : string.Empty);

    // Persist timing beside the hash so resume can compare it before entering the handler.
    // Legacy hashes do not prove which limit or policy created the take.
    internal static string Append(string fingerprint, double value, bool autoStretchToLimit = false) =>
        string.Concat(fingerprint, FingerprintSuffix, Normalize(value, autoStretchToLimit));

    // Strips the timing suffix so a legacy (pre-suffix) fingerprint can be looked up. The base
    // hash already encodes AutoStretchMaxOverrun, so a legacy match proves the same limit was used.
    internal static string WithoutSuffix(string fingerprint)
    {
        int suffixIndex = fingerprint.LastIndexOf(FingerprintSuffix, StringComparison.Ordinal);
        return suffixIndex < 0 ? fingerprint : fingerprint[..suffixIndex];
    }

    internal static bool Matches(string? fingerprint, string expectedValue) =>
        fingerprint?.EndsWith(string.Concat(FingerprintSuffix, expectedValue), StringComparison.Ordinal) == true;
}
