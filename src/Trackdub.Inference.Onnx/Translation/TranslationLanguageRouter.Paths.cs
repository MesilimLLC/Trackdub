using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private static IEnumerable<string> EnumerateResolvedEntryPaths(
        IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> cacheIndex,
        BundledModelManifestEntry entry,
        bool includeVariants = false)
    {
        if (!cacheIndex.TryGetValue(entry.ModelId, out IReadOnlyList<LocalModelCacheRecord>? records))
        {
            yield break;
        }

        foreach (LocalModelCacheRecord record in records)
        {
            IEnumerable<string> paths = includeVariants
                ? entry.Variants.Select(variant => variant.EntryPath).Prepend(entry.DefaultBenchmarkEntryPath)
                : [entry.DefaultBenchmarkEntryPath];
            foreach (string candidatePath in paths.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                string relativeEntryPath = Path.GetRelativePath(entry.RootDirectory, candidatePath);
                string resolvedPath = Path.GetFullPath(Path.Join(record.RootPath, relativeEntryPath));
                if (File.Exists(resolvedPath))
                {
                    yield return resolvedPath;
                }
            }
        }
    }

    private static IEnumerable<string> EnumerateResolvedMadladEntryPaths(
        IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> cacheIndex,
        BundledModelManifestEntry entry)
    {
        if (!cacheIndex.TryGetValue(entry.ModelId, out IReadOnlyList<LocalModelCacheRecord>? records))
        {
            yield break;
        }

        foreach (LocalModelCacheRecord record in records)
        {
            foreach (string candidatePath in EnumerateMadladEncoderEntryPaths(entry))
            {
                string relativeEntryPath = Path.GetRelativePath(entry.RootDirectory, candidatePath);
                string resolvedPath = Path.GetFullPath(Path.Join(record.RootPath, relativeEntryPath));
                if (File.Exists(resolvedPath))
                {
                    yield return resolvedPath;
                }
            }
        }
    }

    private static IEnumerable<string> EnumerateMadladEncoderEntryPaths(BundledModelManifestEntry entry)
    {
        yield return entry.DefaultBenchmarkEntryPath;

        foreach (BundledModelManifestVariant variant in entry.Variants)
        {
            string fileName = Path.GetFileName(variant.EntryPath);
            if (fileName.StartsWith("encoder_model", StringComparison.OrdinalIgnoreCase))
            {
                yield return variant.EntryPath;
            }
        }
    }
}
