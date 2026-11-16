namespace Trackdub.Composition.Translation;

public sealed partial class CatalogTranslationModelProvisioner
{
    private bool IsRegisteredSelection(bool explicitSelection, string? effectiveAlias) =>
        explicitSelection && registry.TryResolve(effectiveAlias!, out _);

    private bool CanUseSelection(bool explicitSelection, bool registeredSelection, string? effectiveAlias)
    {
        if (!explicitSelection)
        {
            return true;
        }

        if (registeredSelection)
        {
            return true;
        }

        return IsCatalogReference(effectiveAlias);
    }

    private static bool IsCatalogReference(string? effectiveAlias) =>
        effectiveAlias?.StartsWith("hf:", StringComparison.Ordinal) == true ||
        effectiveAlias?.StartsWith("onnx-community/opus-mt-", StringComparison.Ordinal) == true ||
        effectiveAlias?.StartsWith("Xenova/opus-mt-", StringComparison.Ordinal) == true;
}
