using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    // Provenance records a Hugging Face repo id, not a URL. The source URL's repo (not ModelId) is the one
    // the recorded revision and hashes belong to; MADLAD's ModelId is google/madlad400-3b-mt but its
    // files come from tonythethompson/madlad400-3b-mt-onnx.
    internal static string ResolveProvenanceRepository(BundledModelManifestEntry entry) =>
        TryGetHuggingFaceRepository(entry.SourceUrl, out string? repository) ? repository! : entry.ModelId;

    private static bool TryGetHuggingFaceRepository(string sourceUrl, out string? repository)
    {
        repository = null;
        if (!Uri.TryCreate(sourceUrl, UriKind.Absolute, out Uri? uri))
        {
            return false;
        }

        if (uri.Scheme != Uri.UriSchemeHttps)
        {
            return false;
        }

        if (!string.Equals(uri.Host, "huggingface.co", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        // AbsolutePath already excludes any query string or fragment.
        string[] segments = uri.AbsolutePath.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (!IsRepositoryPage(segments))
        {
            return false;
        }

        repository = segments[0] + "/" + segments[1];
        return true;
    }

    private static bool IsRepositoryPage(string[] segments)
    {
        if (segments.Length == 2)
        {
            return true;
        }

        if (segments.Length <= 2)
        {
            return false;
        }

        return segments[2] is "tree" or "blob" or "resolve";
    }
}
