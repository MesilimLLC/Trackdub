namespace Trackdub.Inference.Runtime.ModelManifest;

/// <summary>
/// Terms a model must meet before an owner risk acceptance clears the commercial gate
/// (MODEL_LICENSE_POLICY.md, accepted-risk exception).
/// </summary>
public static class RiskAcceptanceEligibility
{
    public static bool IsCleared(BundledModelManifestEntry entry) =>
        entry.CommercialUseVerified ||
        (entry.RiskAccepted && IsEligible(entry.License, entry.CommercialAllowed, entry.RedistributionAllowed, entry.Sha256));

    public static bool IsEligible(string license, bool commercialAllowed, bool redistributionAllowed, string? sha256) =>
        commercialAllowed && redistributionAllowed && IsHashedCustomLicence(license, sha256);

    private static bool IsHashedCustomLicence(string license, string? sha256) =>
        license.Equals("custom", StringComparison.OrdinalIgnoreCase) && !string.IsNullOrWhiteSpace(sha256);
}
