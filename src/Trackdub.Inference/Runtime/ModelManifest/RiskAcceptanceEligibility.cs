namespace Trackdub.Inference.Runtime.ModelManifest;

/// <summary>
/// Terms a model must meet before an owner risk acceptance clears the commercial gate
/// (MODEL_LICENSE_POLICY.md, accepted-risk exception).
/// </summary>
public static class RiskAcceptanceEligibility
{
    /// <summary>Licences that grant commercial use: the named identifiers the policy lists, or <c>custom</c>.</summary>
    public const string EligibleLicenceList = "custom, NVIDIA-Open-Model-License, OpenMDW-1.1 or openrail++";

    private static readonly HashSet<string> EligibleLicences = new(StringComparer.OrdinalIgnoreCase)
    {
        ModelLicenseKind.Custom.ToManifestValue(),
        ModelLicenseKind.NvidiaOpenModelLicense.ToManifestValue(),
        ModelLicenseKind.OpenMdw11.ToManifestValue(),
        ModelLicenseKind.OpenRailPlusPlus.ToManifestValue(),
    };

    public static bool IsCleared(BundledModelManifestEntry entry) =>
        entry.CommercialUseVerified ||
        (entry.RiskAccepted && IsEligible(entry.License, entry.CommercialAllowed, entry.RedistributionAllowed, entry.Sha256));

    public static bool IsCleared(ModelManifest manifest) =>
        manifest.CommercialUseVerified || (manifest.RiskAccepted && IsEligible(manifest));

    public static bool IsEligible(ModelManifest manifest) =>
        IsEligible(manifest.License.ToManifestValue(), manifest.CommercialAllowed, manifest.RedistributionAllowed, manifest.Sha256);

    public static bool IsEligible(string license, bool commercialAllowed, bool redistributionAllowed, string? sha256) =>
        commercialAllowed && redistributionAllowed && IsHashedEligibleLicence(license, sha256);

    private static bool IsHashedEligibleLicence(string license, string? sha256) =>
        EligibleLicences.Contains(license) && IsSha256Digest(sha256);

    private static bool IsSha256Digest(string? sha256) =>
        sha256 is { Length: 64 } && sha256.All(Uri.IsHexDigit);
}
