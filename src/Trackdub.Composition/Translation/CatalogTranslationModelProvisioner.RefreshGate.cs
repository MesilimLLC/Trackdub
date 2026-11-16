namespace Trackdub.Composition.Translation;

public sealed partial class CatalogTranslationModelProvisioner
{
    private async Task<bool> AreDownloadsEnabledAsync(CancellationToken cancellationToken)
    {
        var settings = await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
        return !settings.OfflineMode && settings.AutoDownloadTranslationModels;
    }

    private async Task AwaitPendingRefreshAsync(bool shouldAwait, CancellationToken cancellationToken)
    {
        if (!shouldAwait)
        {
            return;
        }

        if (refreshTask is not { } pending)
        {
            return;
        }

        await pending.WaitAsync(cancellationToken).ConfigureAwait(false);
    }
}
