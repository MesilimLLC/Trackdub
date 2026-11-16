namespace Trackdub.Composition.Translation;

public sealed partial class CatalogTranslationModelProvisioner
{
    private static bool ShouldSkipPair(string sourceLanguage, string targetLanguage) =>
        string.IsNullOrWhiteSpace(sourceLanguage) ||
        sourceLanguage.Equals("auto", StringComparison.OrdinalIgnoreCase) ||
        sourceLanguage.Equals(targetLanguage, StringComparison.OrdinalIgnoreCase);
}
