namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private static readonly HashSet<string> GenAiPivotEngineFamilies = new(StringComparer.OrdinalIgnoreCase)
    {
        "phi-genai",
        "qwen-genai",
        "gemma-genai"
    };

    public static bool IsGenAiPivotEngineFamily(string? engineFamily) =>
        !string.IsNullOrWhiteSpace(engineFamily) && GenAiPivotEngineFamilies.Contains(engineFamily.Trim());
}
