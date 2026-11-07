namespace Trackdub.Infrastructure.Components.NvidiaAfx;

public sealed record NvidiaAfxRuntimePackage(
    string Architecture,
    string DownloadUrl,
    string Sha256,
    long SizeBytes,
    string RuntimeVersion,
    string LicenseUrl,
    string[] ModelRelativePaths,
    int PartCount = 1);

/// <summary>
/// Licence terms the manifest records for one effect. <see cref="Terms"/> is
/// <see cref="NvidiaAfxEffectTerms.Commercial"/> only when the effect's terms allow production use.
/// </summary>
public sealed record NvidiaAfxEffectLicense(string Selector, string Terms);

public static class NvidiaAfxEffectTerms
{
    public const string Commercial = "commercial";
    public const string Evaluation = "evaluation";
}

public sealed record NvidiaAfxRuntimeManifest(
    string ManifestVersion,
    NvidiaAfxRuntimePackage[] Packages,
    NvidiaAfxEffectLicense[]? Effects = null)
{
    /// <summary>
    /// True only when the manifest explicitly lists the effect with commercial terms. An effect the
    /// manifest does not list, or lists as evaluation-only, is not eligible for production use.
    /// </summary>
    public bool IsCommerciallyLicensed(string selector) =>
        Effects?.Any(effect =>
            string.Equals(effect.Selector, selector, StringComparison.OrdinalIgnoreCase)
            && string.Equals(effect.Terms, NvidiaAfxEffectTerms.Commercial, StringComparison.OrdinalIgnoreCase))
        == true;
}
