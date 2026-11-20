using System.Diagnostics.CodeAnalysis;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.Runtime;

internal static class ModelDownloadManifestFiles
{
    private const string CacheInstalledRevision = "cache-installed";

    public static IReadOnlyList<string> ResolveRequiredFiles(BundledModelManifestEntry entry) =>
        ResolveRequiredFiles(entry, variantAlias: null);

    public static IReadOnlyList<string> ResolveRequiredFiles(
        BundledModelManifestEntry entry,
        string? variantAlias)
    {
        if (string.IsNullOrWhiteSpace(variantAlias))
        {
            var files = new List<string>();
            AddFiles(files, entry.DownloadFiles);

            foreach (BundledModelManifestVariant variant in ResolveDefaultDownloadVariants(entry))
            {
                AddFiles(files, variant.DownloadFiles);
                AddFiles(files, [Path.GetRelativePath(entry.RootDirectory, variant.EntryPath)]);
            }

            AddFiles(files, [Path.GetRelativePath(entry.RootDirectory, entry.DefaultBenchmarkEntryPath)]);

            return files
                .Select(NormalizeRelativePath)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
        }

        return ResolveRequiredFilesForVariant(entry, variantAlias.Trim());
    }

    internal static IReadOnlyList<string> ResolveRequiredFilesForVariant(
        BundledModelManifestEntry entry,
        string variantAlias,
        string? entryRelativePathOverride = null)
    {
        var files = new List<string>();
        AddFiles(files, entry.DownloadFiles);

        BundledModelManifestVariant? variant = entry.Variants.FirstOrDefault(candidate =>
            candidate.Alias.Equals(variantAlias, StringComparison.OrdinalIgnoreCase));
        if (variant is not null)
        {
            AddFiles(files, variant.DownloadFiles);
            string entryPath = string.IsNullOrWhiteSpace(entryRelativePathOverride)
                ? Path.GetRelativePath(entry.RootDirectory, variant.EntryPath)
                : entryRelativePathOverride;
            AddFiles(files, [entryPath]);
        }
        else if (string.IsNullOrWhiteSpace(variantAlias) ||
                 variantAlias.Equals("default", StringComparison.OrdinalIgnoreCase))
        {
            // The synthetic "default" alias (emitted by RuntimePlanFactory for every model,
            // including those with no declared variants) resolves to the benchmark entry,
            // mirroring ResolveSelectedEntryRelativePath. Only genuinely unknown named
            // variants fall through to the throw below.
            string entryPath = string.IsNullOrWhiteSpace(entryRelativePathOverride)
                ? Path.GetRelativePath(entry.RootDirectory, entry.DefaultBenchmarkEntryPath)
                : entryRelativePathOverride;
            AddFiles(files, [entryPath]);
        }
        else
        {
            throw new InvalidOperationException(
                $"Manifest variant alias '{variantAlias}' is not defined for model '{entry.ModelId}'.");
        }

        return files
            .Select(NormalizeRelativePath)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    /// <summary>
    /// The commercial gate for download and model-manager paths: commercial use is allowed and the model is either
    /// verified or carries a loader-validated owner risk acceptance (MODEL_LICENSE_POLICY.md, accepted-risk exception).
    /// </summary>
    public static bool IsCommerciallyCleared(BundledModelManifestEntry entry) =>
        entry.CommercialAllowed && entry.CommercialSafeMode;

    public static string DescribeGateRefusal(BundledModelManifestEntry entry) =>
        $"Model '{entry.ModelId}' is not cleared for commercial use (it needs commercial_allowed with commercial_use_verified or a recorded risk_acceptance); it cannot be downloaded.";

    public static bool CanAutoDownloadAll(BundledModelManifestEntry entry) =>
        IsCommerciallyCleared(entry) &&
        entry.RedistributionAllowed &&
        ResolveRequiredFiles(entry).All(relativePath => CanAutoDownloadFile(entry, relativePath));

    public static bool CanAutoDownloadFile(BundledModelManifestEntry entry, string relativePath) =>
        IsCommerciallyCleared(entry) &&
        (TryResolveExternalDownloadSource(entry, relativePath, out _) || CanDownloadFromHuggingFace(entry));

    public static bool CanDownloadFromHuggingFace(BundledModelManifestEntry entry) =>
        !entry.Revision.Equals(CacheInstalledRevision, StringComparison.OrdinalIgnoreCase) &&
        Uri.TryCreate(entry.SourceUrl, UriKind.Absolute, out Uri? sourceUri) &&
        sourceUri.Host.Equals("huggingface.co", StringComparison.OrdinalIgnoreCase);

    public static string ResolveHuggingFaceModelId(BundledModelManifestEntry entry)
    {
        Uri.TryCreate(entry.SourceUrl, UriKind.Absolute, out Uri? sourceUri);
        return sourceUri!.AbsolutePath.TrimStart('/');
    }

    public static string ResolveDownloadSourceDescription(BundledModelManifestEntry entry, string relativePath) =>
        TryResolveExternalDownloadSource(entry, relativePath, out Uri? sourceUri)
            ? sourceUri.AbsoluteUri
            : CanDownloadFromHuggingFace(entry)
                ? entry.SourceUrl
                : "no configured downloadable source";

    /// <summary>
    /// Resolves the pinned SHA-256 for a required cache file.
    /// Per-file hashes cover optional variants (for example Kokoro fp16/q8f16).
    /// Sidecar files such as tokenizers and voice packs stay required on disk
    /// without a digest unless they are the hash-anchor entry.
    /// </summary>
    public static string? ResolveExpectedSha256(
        BundledModelManifestEntry entry,
        string normalizedRelativePath,
        string hashAnchorRelativePath)
    {
        ArgumentNullException.ThrowIfNull(entry);
        ArgumentException.ThrowIfNullOrWhiteSpace(normalizedRelativePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(hashAnchorRelativePath);

        if (entry.DownloadFileHashes.TryGetValue(normalizedRelativePath, out string? fileHash) &&
            !string.IsNullOrWhiteSpace(fileHash))
        {
            return fileHash;
        }

        return normalizedRelativePath.Equals(hashAnchorRelativePath, StringComparison.OrdinalIgnoreCase)
            ? entry.Sha256
            : null;
    }

    /// <summary>
    /// Cache records store the manifest identity hash (benchmark entry), not the
    /// digest of whichever sidecar happened to be verified last.
    /// </summary>
    public static string ResolveCacheIdentitySha256(
        BundledModelManifestEntry entry,
        string? verifiedFileSha256 = null)
    {
        ArgumentNullException.ThrowIfNull(entry);
        return string.IsNullOrWhiteSpace(entry.Sha256)
            ? verifiedFileSha256 ?? string.Empty
            : entry.Sha256;
    }

    public static bool TryResolveExternalDownloadSource(
        BundledModelManifestEntry entry,
        string relativePath,
        [NotNullWhen(true)] out Uri? sourceUri)
    {
        sourceUri = null;
        string normalizedRelativePath = NormalizeRelativePath(relativePath);
        if (!entry.DownloadFileSources.TryGetValue(normalizedRelativePath, out string? source) ||
            !Uri.TryCreate(source, UriKind.Absolute, out Uri? uri) ||
            uri.Scheme is not ("http" or "https"))
        {
            return false;
        }

        sourceUri = uri;
        return true;
    }

    private static IEnumerable<BundledModelManifestVariant> ResolveDefaultDownloadVariants(BundledModelManifestEntry entry)
    {
        // Only the variant marked as default augments base-model downloads. Model-lab / Olive
        // optimized variants (e.g. trt-rtx-fp16) are local-only and must not be pulled from HF.
        // Prefer the explicit IsDefault flag; fall back to the legacy "default" alias for
        // manifests that have not yet adopted the flag. ModelManifestLoader validates that at
        // most one variant sets IsDefault=true, so the selection is unambiguous.
        BundledModelManifestVariant? defaultVariant =
            entry.Variants.FirstOrDefault(variant => variant.IsDefault)
            ?? entry.Variants.FirstOrDefault(variant =>
                variant.Alias.Equals("default", StringComparison.OrdinalIgnoreCase));
        if (defaultVariant is not null)
        {
            yield return defaultVariant;
        }
    }

    private static void AddFiles(ICollection<string> files, IEnumerable<string> paths)
    {
        foreach (string path in paths)
        {
            if (!string.IsNullOrWhiteSpace(path))
            {
                files.Add(path);
            }
        }
    }

    private static string NormalizeRelativePath(string path)
    {
        string normalized = path.Replace('\\', '/').Trim('/');
        if (string.IsNullOrWhiteSpace(normalized))
        {
            throw new InvalidOperationException("Download file path is empty.");
        }

        if (Path.IsPathRooted(normalized))
        {
            throw new InvalidOperationException("Download file path must be relative.");
        }

        if (normalized.Split('/').Any(part => part is "." or ".." || string.IsNullOrWhiteSpace(part)))
        {
            throw new InvalidOperationException("Download file path must not contain '.' or '..' segments.");
        }

        return normalized;
    }
}
