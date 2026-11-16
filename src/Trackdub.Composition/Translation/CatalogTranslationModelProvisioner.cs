using Trackdub.Contracts;
using Trackdub.Contracts.Translation;
using Trackdub.Infrastructure.Settings;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Composition.Translation;

/// <summary>Publishes eligible metadata into the shared registry; only the execution path provisions weights.</summary>
public sealed partial class CatalogTranslationModelProvisioner(
    ITranslationModelCatalog catalog,
    BundledModelManifestRegistry registry,
    IStudioSettingsService settingsService,
    IModelDownloadOrchestrator downloader,
    TrackdubStoragePaths storagePaths,
    IModelCacheInventory? cacheInventory = null,
    IHardwareProfileProvider? hardwareProfileProvider = null,
    IExecutionProviderDiscovery? providerDiscovery = null) : ITranslationModelProvisioner
{
    private readonly ITranslationModelCatalog catalog = catalog;
    private readonly BundledModelManifestRegistry registry = registry;
    private readonly IStudioSettingsService settingsService = settingsService;
    private readonly IModelDownloadOrchestrator downloader = downloader;
    private readonly TrackdubStoragePaths storagePaths = storagePaths;
    private readonly IModelCacheInventory? cacheInventory = cacheInventory;
    private readonly IHardwareProfileProvider? hardwareProfileProvider = hardwareProfileProvider;
    private readonly IExecutionProviderDiscovery? providerDiscovery = providerDiscovery;
    private readonly SemaphoreSlim refreshGate = new(1, 1);
    private DateTimeOffset lastRefresh;
    private Task? refreshTask;

    public async Task<string?> EnsureAsync(
        string sourceLanguage,
        string targetLanguage,
        string? preferredModelAlias,
        CancellationToken cancellationToken,
        IProgress<string>? preparationProgress = null,
        string? preferredModelVariantAlias = null,
        string? preferredExecutionProvider = null) =>
        await EnsureCoreAsync(
            new EnsureRequest
            {
                SourceLanguage = sourceLanguage,
                TargetLanguage = targetLanguage,
                PreferredModelAlias = preferredModelAlias,
                PreparationProgress = preparationProgress,
                PreferredModelVariantAlias = preferredModelVariantAlias,
                PreferredExecutionProvider = preferredExecutionProvider
            },
            cancellationToken).ConfigureAwait(false);

    private sealed record EnsureRequest
    {
        public required string SourceLanguage { get; init; }
        public required string TargetLanguage { get; init; }
        public string? PreferredModelAlias { get; init; }
        public IProgress<string>? PreparationProgress { get; init; }
        public string? PreferredModelVariantAlias { get; init; }
        public string? PreferredExecutionProvider { get; init; }
    }

    public async Task RefreshAsync(CancellationToken cancellationToken)
    {
        Publish(catalog.Snapshot);
        var settings = await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
        if (settings.OfflineMode) return;
        await refreshGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (ShouldSkipRefresh()) return;
            lastRefresh = DateTimeOffset.UtcNow;
            refreshTask = RefreshInBackgroundAsync();
        }
        finally { refreshGate.Release(); }
    }

    private bool ShouldSkipRefresh() =>
        refreshTask is { IsCompleted: false } ||
        DateTimeOffset.UtcNow - lastRefresh < TimeSpan.FromHours(24) ||
        DateTimeOffset.UtcNow - catalog.Snapshot.FetchedAtUtc < TimeSpan.FromHours(24);

    private async Task RefreshInBackgroundAsync()
    {
        try { Publish(await catalog.RefreshAsync(false, CancellationToken.None).ConfigureAwait(false)); }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Keep the shared refresh task successful so later translations are not blocked by a fault.
            // Clearing lastRefresh lets the next refresh retry instead of waiting out the 24-hour window.
            await refreshGate.WaitAsync().ConfigureAwait(false);
            try { lastRefresh = DateTimeOffset.MinValue; }
            finally { refreshGate.Release(); }
            Publish(catalog.Snapshot);
        }
    }

    private void Publish(TranslationModelCatalogSnapshot snapshot) =>
        registry.AddCatalogEntries(snapshot.Models.GroupBy(model => model.ModelId)
            .Select(group => CreateEntry(group, snapshot.PolicyVersion)).OfType<BundledModelManifestEntry>());
}
