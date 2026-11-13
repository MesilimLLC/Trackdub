using System.Collections.Immutable;
using Trackdub.Composition.Translation;
using Trackdub.Contracts.Translation;
using Trackdub.Domain;
using Trackdub.Infrastructure.Settings;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.TestDoubles;

namespace Trackdub.Composition.Tests;

public sealed class CatalogTranslationModelProvisionerTests
{
    public enum QwenProviderSelection
    {
        Cuda,
        Cpu
    }

    public enum ExplicitSelectionScenario
    {
        Registered,
        Unregistered
    }

    public enum NoDownloadScenario
    {
        Offline,
        AutomaticDownloadDisabled,
        UnknownSource
    }

    [Fact]
    public async Task ExplicitQwenSelectionDownloadsTheRegisteredModel()
    {
        Assert.True(BundledModelManifestRegistry.TryLoadDefault(out var registry, out var error), error);
        var settings = new FakeStudioSettingsService();
        await settings.SaveAsync(StudioSettings.Default with { OfflineMode = false, AutoDownloadTranslationModels = true }, CancellationToken.None);
        var downloader = new Downloader();
        var service = new CatalogTranslationModelProvisioner(new Catalog(), registry!, settings, downloader,
            new TrackdubStoragePaths(Path.GetTempPath()));

        string? alias = await service.EnsureAsync("en", "fr", "qwen3-4b-translation", CancellationToken.None,
            preferredModelVariantAlias: "cuda-int4");

        Assert.Equal("qwen3-4b-translation", alias);
        Assert.Equal("Qwen/Qwen3-4B", downloader.Selected);
        Assert.Equal("cuda-int4", downloader.Variant);
    }

    [Theory]
    [InlineData(QwenProviderSelection.Cuda)]
    [InlineData(QwenProviderSelection.Cpu)]
    public async Task Qwen_selection_without_a_variant_provisions_the_variant_for_the_requested_provider(
        QwenProviderSelection selection)
    {
        Assert.True(BundledModelManifestRegistry.TryLoadDefault(out var registry, out var error), error);
        var settings = new FakeStudioSettingsService();
        await settings.SaveAsync(StudioSettings.Default with { OfflineMode = false, AutoDownloadTranslationModels = true }, CancellationToken.None);
        var downloader = new Downloader();
        var service = new CatalogTranslationModelProvisioner(new Catalog(), registry!, settings, downloader,
            new TrackdubStoragePaths(Path.GetTempPath()));

        string provider = selection is QwenProviderSelection.Cuda ? "Cuda" : "Cpu";
        string expectedVariant = selection is QwenProviderSelection.Cuda ? "cuda-int4" : "cpu-int4";
        await service.EnsureAsync("en", "fr", "qwen3-4b-translation", CancellationToken.None, preferredExecutionProvider: provider);

        Assert.Equal("Qwen/Qwen3-4B", downloader.Selected);
        Assert.Equal(expectedVariant, downloader.Variant);
    }

    [Theory]
    [InlineData(ExplicitSelectionScenario.Registered)]
    [InlineData(ExplicitSelectionScenario.Unregistered)]
    public async Task ExplicitSelectionsDoNotWaitForUnrelatedCatalogRefresh(ExplicitSelectionScenario scenario)
    {
        Assert.True(BundledModelManifestRegistry.TryLoadDefault(out var registry, out var error), error);
        string alias = scenario is ExplicitSelectionScenario.Registered
            ? registry!.Entries.First(entry => entry.Task == "translation" && entry.EngineFamily != "opus-mt").Aliases.First()
            : "deepl";
        var catalog = new Catalog { Snapshot = TranslationModelCatalogSnapshot.Empty, FailRefresh = true };
        var service = new CatalogTranslationModelProvisioner(catalog, registry!, new FakeStudioSettingsService(),
            new Downloader(), new TrackdubStoragePaths(Path.GetTempPath()));

        Assert.NotNull(await service.EnsureAsync("en", "es", alias, CancellationToken.None));
        Assert.Equal(0, catalog.RefreshCalls);
    }

    [Fact]
    public async Task Cached_metadata_is_shared_and_downloaded_descriptor_stays_pinned()
    {
        Assert.True(BundledModelManifestRegistry.TryLoadDefault(out var registry, out var error), error);
        var catalog = new Catalog();
        var downloader = new Downloader();
        var settings = new FakeStudioSettingsService();
        var service = new CatalogTranslationModelProvisioner(catalog, registry!, settings, downloader, new TrackdubStoragePaths(Path.GetTempPath()));
        string? alias = await service.EnsureAsync("xx", "yy", null, CancellationToken.None);
        Assert.Equal("hf:onnx-community/opus-mt-xx-yy", alias);
        Assert.Equal("onnx-community/opus-mt-xx-yy", downloader.Selected);
        Assert.True(registry!.TryResolve(alias!, out var resolved));
        Assert.Equal(new string('a', 40), resolved!.Entry.Revision);
        Assert.Equal(70, resolved.Entry.ExpectedDownloadBytes);
        Assert.All(resolved.Entry.DownloadFileSources.Values, source => Assert.Contains(new string('a', 40), source));
        catalog.Snapshot = catalog.Snapshot with { Models = [catalog.Snapshot.Models[0] with { Revision = new string('b', 40) }] };
        await service.RefreshAsync(CancellationToken.None);
        Assert.True(registry.TryResolve(alias!, out resolved));
        Assert.Equal(new string('a', 40), resolved!.Entry.Revision);
    }

    [Theory]
    [InlineData(NoDownloadScenario.Offline)]
    [InlineData(NoDownloadScenario.AutomaticDownloadDisabled)]
    [InlineData(NoDownloadScenario.UnknownSource)]
    public async Task Offline_disabled_and_unknown_source_do_not_download(NoDownloadScenario scenario)
    {
        Assert.True(BundledModelManifestRegistry.TryLoadDefault(out var registry, out var error), error);
        (bool offline, bool automatic, string source) = scenario switch
        {
            NoDownloadScenario.Offline => (true, true, "xx"),
            NoDownloadScenario.AutomaticDownloadDisabled => (false, false, "xx"),
            NoDownloadScenario.UnknownSource => (false, true, "auto"),
            _ => throw new ArgumentOutOfRangeException(nameof(scenario), scenario, null)
        };
        var settings = new FakeStudioSettingsService();
        await settings.SaveAsync(StudioSettings.Default with { OfflineMode = offline, AutoDownloadTranslationModels = automatic }, CancellationToken.None);
        var downloader = new Downloader();
        var service = new CatalogTranslationModelProvisioner(new Catalog(), registry!, settings, downloader, new TrackdubStoragePaths(Path.GetTempPath()));
        await service.EnsureAsync(source, "yy", null, CancellationToken.None);
        Assert.Null(downloader.Selected);
    }
}
