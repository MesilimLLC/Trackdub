using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private static string ResolveProviderName(BundledModelManifestEntry entry) =>
        entry.EngineFamily.ToLowerInvariant() switch
        {
            "madlad" => "madlad400",
            var family => family
        };

    private static string ResolveDirectRouteDetail(BundledModelManifestEntry entry) =>
        string.Equals(entry.EngineFamily, "opus-mt", StringComparison.OrdinalIgnoreCase)
            ? "Direct Opus-MT"
            : $"Direct {entry.EngineFamily}";

    private static string ResolvePivotRouteDetail(BundledModelManifestEntry entry, bool directPairKnown)
    {
        string label = ResolvePivotRouteLabel(entry);
        return directPairKnown
            ? $"{label} (direct translation pair missing)"
            : label;
    }

    private static string ResolveUnavailablePivotLabel(IReadOnlyList<BundledModelManifestEntry> pivotEntries) =>
        pivotEntries.Count == 0
            ? "pivot translation"
            : ResolvePivotRouteLabel(pivotEntries[0]);

    private static string ResolvePivotRouteLabel(BundledModelManifestEntry entry) =>
        entry.EngineFamily.ToLowerInvariant() switch
        {
            "madlad" => "MADLAD-400 pivot",
            var family when IsGenAiPivotEngineFamily(family) => "Multilingual pivot",
            _ => "Pivot translation"
        };
}
