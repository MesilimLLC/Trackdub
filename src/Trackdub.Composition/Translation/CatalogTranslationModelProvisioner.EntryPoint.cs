using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.Translation;

public sealed partial class CatalogTranslationModelProvisioner
{
    private async Task<string?> EnsureCoreAsync(EnsureRequest request, CancellationToken cancellationToken)
    {
        if (ShouldSkipPair(request.SourceLanguage, request.TargetLanguage))
        {
            return request.PreferredModelAlias;
        }

        EnsureSelection selection = await ResolveEnsureSelectionAsync(
            request.SourceLanguage,
            request.TargetLanguage,
            request.PreferredModelAlias,
            cancellationToken).ConfigureAwait(false);
        if (selection.Entry is null)
        {
            return request.PreferredModelAlias;
        }

        if (!CanProvisionSelectedModel(selection.Entry, selection.ExplicitSelection))
        {
            return request.PreferredModelAlias;
        }

        await ProvisionAsync(
            selection.Entry,
            new VariantSelection(request.PreferredModelVariantAlias, request.PreferredExecutionProvider),
            request.PreparationProgress,
            cancellationToken).ConfigureAwait(false);
        return selection.Entry.Aliases.FirstOrDefault() ?? selection.Entry.ModelId;
    }
}
