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
    [InlineData(false, "hf:onnx-community/opus-mt-xx-yy", "onnx-community/opus-mt-xx-yy")]
    [InlineData(true, OpusMtAutoModel.Alias, null)]
    public async Task OpusAutoSelectionProvisionsOnlyWhenDownloadsAllowed(bool offline, string expectedAlias, string? expectedSelected)
    {
        Assert.True(BundledModelManifestRegistry.TryLoadDefault(out var registry, out var error), error);
        var downloader = new Downloader();
        var settings = new FakeStudioSettingsService();
        await settings.SaveAsync(StudioSettings.Default with { OfflineMode = offline, AutoDownloadTranslationModels = true }, CancellationToken.None);
        var service = new CatalogTranslationModelProvisioner(new Catalog(), registry!, settings, downloader, new TrackdubStoragePaths(Path.GetTempPath()));

        string? alias = await service.EnsureAsync("xx", "yy", OpusMtAutoModel.Alias, CancellationToken.None);

        Assert.Equal(expectedAlias, alias);
        Assert.Equal(expectedSelected, downloader.Selected);
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

    [Fact]
    public async Task EnsureAsync_PrefersUsableInstalledMultiPairOverCorruptCachedSinglePair()
    {
        Assert.True(BundledModelManifestRegistry.TryLoadDefault(out var registry, out var error), error);
        var catalog = new Catalog();
        var settings = new FakeStudioSettingsService();
        await settings.SaveAsync(StudioSettings.Default with { OfflineMode = false, AutoDownloadTranslationModels = true }, CancellationToken.None);
        string tempRoot = Path.Join(Path.GetTempPath(), $"trackdub-provisioner-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempRoot);
        try
        {
            var storagePaths = new TrackdubStoragePaths(tempRoot);
            var bootstrap = new CatalogTranslationModelProvisioner(catalog, registry!, settings, new Downloader(), storagePaths);
            await bootstrap.RefreshAsync(CancellationToken.None);
            Assert.True(registry!.TryResolve("hf:onnx-community/opus-mt-xx-yy", out BundledModelManifestResolution? singleResolved));
            BundledModelManifestEntry single = singleResolved!.Entry;

            string multiRoot = Path.Join(tempRoot, "models", "opus-mt-xx-multi");
            string encoderFull = Path.Join(multiRoot, "onnx", "encoder_model.onnx");
            Directory.CreateDirectory(Path.GetDirectoryName(encoderFull)!);
            await File.WriteAllTextAsync(encoderFull, "placeholder");
            registry.AddCatalogEntries(
            [
                single with
                {
                    ModelId = "onnx-community/opus-mt-xx-multi",
                    Aliases = ["hf:onnx-community/opus-mt-xx-multi"],
                    LanguageCoverage = new ModelLanguageCoverage([], [],
                    [
                        new Trackdub.Inference.Runtime.ModelManifest.ModelLanguagePair("xx", "yy"),
                        new Trackdub.Inference.Runtime.ModelManifest.ModelLanguagePair("xx", "zz")
                    ]),
                    RootDirectory = multiRoot,
                    DefaultBenchmarkEntryPath = encoderFull,
                    SourceUrl = "https://huggingface.co/onnx-community/opus-mt-xx-multi"
                }
            ]);

            string corruptSingleRoot = Path.Join(tempRoot, "corrupt-single");
            Directory.CreateDirectory(corruptSingleRoot);
            var cache = new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord(single.ModelId, corruptSingleRoot, single.Revision, single.Sha256, DateTimeOffset.UtcNow, IntegrityFailed: true),
                new LocalModelCacheRecord("onnx-community/opus-mt-xx-multi", multiRoot, single.Revision, single.Sha256, DateTimeOffset.UtcNow)
            ]);
            var downloader = new Downloader();
            var service = new CatalogTranslationModelProvisioner(catalog, registry, settings, downloader, storagePaths, cache);

            string? alias = await service.EnsureAsync("xx", "yy", OpusMtAutoModel.Alias, CancellationToken.None);

            Assert.Equal("hf:onnx-community/opus-mt-xx-multi", alias);
            Assert.Equal("onnx-community/opus-mt-xx-multi", downloader.Selected);
        }
        finally
        {
            try { Directory.Delete(tempRoot, recursive: true); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }

    [Fact]
    public async Task OpusAuto_WaitsForPendingRefreshWhenOnlyNonOpusCoversPair()
    {
        Assert.True(BundledModelManifestRegistry.TryLoadDefault(out var registry, out var error), error);
        var refreshSnapshot = new TranslationModelCatalogSnapshot(
            DateTimeOffset.UtcNow, "opus-mt-publishers-v1", [Catalog.CreateOpusDescriptor("aa", "bb")], []);
        var catalog = new Catalog
        {
            Snapshot = new TranslationModelCatalogSnapshot(DateTimeOffset.UtcNow.AddDays(-2), "opus-mt-publishers-v1", [], []),
            RefreshResult = refreshSnapshot,
            RefreshDelayMs = 250
        };
        Assert.True(BundledModelManifestRegistry.TryLoadDefault(out var seedRegistry, out _), "default registry required");
        BundledModelManifestEntry nonOpus = seedRegistry!.Entries.First(entry =>
            entry.Task == "translation" && entry.EngineFamily != "opus-mt" && entry.CommercialAllowed);
        registry!.AddCatalogEntries(
        [
            nonOpus with
            {
                ModelId = "test/non-opus-aa-bb",
                Aliases = ["non-opus-aa-bb"],
                LanguageCoverage = new ModelLanguageCoverage([], [],
                [
                    new Trackdub.Inference.Runtime.ModelManifest.ModelLanguagePair("aa", "bb")
                ]),
                EngineFamily = "madlad"
            }
        ]);
        var settings = new FakeStudioSettingsService();
        await settings.SaveAsync(StudioSettings.Default with { OfflineMode = false, AutoDownloadTranslationModels = true }, CancellationToken.None);
        var downloader = new Downloader();
        var service = new CatalogTranslationModelProvisioner(catalog, registry, settings, downloader, new TrackdubStoragePaths(Path.GetTempPath()));

        string? alias = await service.EnsureAsync("aa", "bb", OpusMtAutoModel.Alias, CancellationToken.None);

        Assert.True(catalog.RefreshCalls >= 1);
        Assert.Equal("hf:onnx-community/opus-mt-aa-bb", alias);
        Assert.Equal("onnx-community/opus-mt-aa-bb", downloader.Selected);
    }
}
