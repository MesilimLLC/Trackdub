using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.Translation;

public sealed partial class CatalogTranslationModelProvisioner
{
    private async Task ProvisionAsync(
        BundledModelManifestEntry selected,
        VariantSelection selection,
        IProgress<string>? preparationProgress,
        CancellationToken cancellationToken)
    {
        preparationProgress?.Report($"Preparing {selected.ModelId}");
        IReadOnlyList<string?> variantAliases = await ResolveVariantAliasesAsync(selected, selection, cancellationToken).ConfigureAwait(false);
        foreach (string? variantAlias in variantAliases)
        {
            await DownloadVariantAsync(selected, variantAlias, preparationProgress, cancellationToken).ConfigureAwait(false);
        }
    }

    private async Task DownloadVariantAsync(
        BundledModelManifestEntry selected,
        string? variantAlias,
        IProgress<string>? preparationProgress,
        CancellationToken cancellationToken)
    {
        var result = await downloader.DownloadAsync(
            selected.ModelId,
            variantAlias,
            preparationProgress is null ? null : new PreparationDownloadProgress(preparationProgress, selected.ModelId),
            cancellationToken).ConfigureAwait(false);
        if (!result.Success)
        {
            throw new InvalidOperationException(result.FailureReason ?? $"Translation model '{selected.ModelId}' could not be installed.");
        }
    }
}
