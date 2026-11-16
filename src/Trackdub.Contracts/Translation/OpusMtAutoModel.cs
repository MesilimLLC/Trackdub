namespace Trackdub.Contracts.Translation;

/// <summary>
/// Sentinel alias for "Opus MT families: resolve from languages at runtime".
/// Never persisted as a concrete model choice and never displayed in UI (UI shows "Opus MT").
/// </summary>
public static class OpusMtAutoModel
{
    public const string Alias = "opus-mt-auto";

    public static bool IsAutoAlias(string? alias) =>
        string.Equals(alias?.Trim(), Alias, StringComparison.OrdinalIgnoreCase);
}
