using Trackdub.Domain.Translation;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

internal sealed record GenAiPivotProvenanceRequest
{
    public required IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> CacheIndex { get; init; }
    public required BundledModelManifestEntry Entry { get; init; }
    public required string ResolvedEntryPath { get; init; }
    public required string TargetLanguage { get; init; }
}

internal static class GenAiPivotProvenanceResolver
{
    public static TranslationModelProvenance Resolve(GenAiPivotProvenanceRequest request)
    {
        BundledModelManifestEntry entry = request.Entry;
        BundledModelManifestVariant? variant = ResolveVariantForEntryPath(request);
        if (variant is null)
        {
            return new TranslationModelProvenance(
                TranslationLanguageRouter.ResolveProvenanceRepository(entry), entry.Revision, entry.Sha256,
                new Dictionary<string, string>(entry.DownloadFileHashes),
                entry.TargetPrefixes?.GetValueOrDefault(request.TargetLanguage), entry.CatalogPolicyVersion);
        }

        string variantEntryRelativePath = Path.GetRelativePath(entry.RootDirectory, variant.EntryPath)
            .Replace('\\', '/')
            .Trim('/');
        // The parent Sha256 may describe the default entry; never attach it to a different variant.
        string variantSha256 = entry.DownloadFileHashes.TryGetValue(variantEntryRelativePath, out string? entryHash)
            ? entryHash
            : string.Empty;
        var variantFileHashes = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (string downloadFile in variant.DownloadFiles)
        {
            string normalizedPath = downloadFile.Replace('\\', '/').Trim('/');
            if (entry.DownloadFileHashes.TryGetValue(normalizedPath, out string? fileHash))
            {
                variantFileHashes[normalizedPath] = fileHash;
            }
        }

        return new TranslationModelProvenance(
            TranslationLanguageRouter.ResolveProvenanceRepository(entry), entry.Revision, variantSha256,
            variantFileHashes,
            entry.TargetPrefixes?.GetValueOrDefault(request.TargetLanguage), entry.CatalogPolicyVersion);
    }

    private static BundledModelManifestVariant? ResolveVariantForEntryPath(GenAiPivotProvenanceRequest request)
    {
        BundledModelManifestEntry entry = request.Entry;
        if (!request.CacheIndex.TryGetValue(entry.ModelId, out IReadOnlyList<LocalModelCacheRecord>? records))
        {
            return null;
        }

        StringComparison pathComparison = OperatingSystem.IsWindows()
            ? StringComparison.OrdinalIgnoreCase
            : StringComparison.Ordinal;
        foreach (LocalModelCacheRecord record in records)
        {
            foreach (BundledModelManifestVariant variant in entry.Variants)
            {
                string relativeVariantPath = Path.GetRelativePath(entry.RootDirectory, variant.EntryPath);
                string candidatePath = Path.GetFullPath(Path.Join(record.RootPath, relativeVariantPath));
                if (candidatePath.Equals(request.ResolvedEntryPath, pathComparison))
                {
                    return variant;
                }
            }
        }

        return null;
    }
}
