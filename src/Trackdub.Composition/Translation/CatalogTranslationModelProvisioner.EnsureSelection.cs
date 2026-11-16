using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.Translation;

public sealed partial class CatalogTranslationModelProvisioner
{
    private async Task<EnsureSelection> ResolveEnsureSelectionAsync(
        string sourceLanguage,
        string targetLanguage,
        string? preferredModelAlias,
        CancellationToken cancellationToken)
    {
        string? effectiveAlias = ResolveEffectiveAlias(preferredModelAlias);
        bool explicitSelection = !string.IsNullOrWhiteSpace(effectiveAlias);
        bool registeredSelection = IsRegisteredSelection(explicitSelection, effectiveAlias);
        if (!CanUseSelection(explicitSelection, registeredSelection, effectiveAlias))
        {
            return EnsureSelection.None(explicitSelection);
        }

        if (!registeredSelection)
        {
            await RefreshAsync(cancellationToken).ConfigureAwait(false);
        }

        if (!await AreDownloadsEnabledAsync(cancellationToken).ConfigureAwait(false))
        {
            return EnsureSelection.None(explicitSelection);
        }

        // Auto Opus selection must wait for Opus coverage, not just any translation family.
        bool shouldAwaitPending = !registeredSelection &&
            (explicitSelection || !HasEligibleOpusPairCoverage(sourceLanguage, targetLanguage));
        await AwaitPendingRefreshAsync(shouldAwaitPending, cancellationToken).ConfigureAwait(false);

        BundledModelManifestEntry? selected = await SelectEntryAsync(sourceLanguage, targetLanguage, effectiveAlias, cancellationToken).ConfigureAwait(false);
        if (selected is null)
        {
            return EnsureSelection.None(explicitSelection);
        }

        if (!CoversPair(selected, sourceLanguage, targetLanguage))
        {
            return EnsureSelection.None(explicitSelection);
        }

        return new EnsureSelection(selected, explicitSelection);
    }

    private sealed record EnsureSelection(BundledModelManifestEntry? Entry, bool ExplicitSelection)
    {
        public static EnsureSelection None(bool explicitSelection) => new(null, explicitSelection);
    }
}
