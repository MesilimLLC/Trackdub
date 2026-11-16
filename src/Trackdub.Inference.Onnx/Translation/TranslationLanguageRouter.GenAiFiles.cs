namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private static bool HasGenAiSupportingFiles(string modelRootPath) =>
        File.Exists(Path.Join(modelRootPath, "genai_config.json"));

    private static bool IsPlannedEntryPath(string path, string? plannedEntryPath) =>
        !string.IsNullOrWhiteSpace(plannedEntryPath) &&
        Path.GetFullPath(path).Equals(
            Path.GetFullPath(plannedEntryPath),
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
}
