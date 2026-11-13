using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Translation;
using Trackdub.Inference.Runtime.ModelManifest;
using static Trackdub.Inference.Tests.TranslationRouterManifestSpecs;
using static Trackdub.Inference.Tests.TranslationLanguageRouterTestSetup;
using static Trackdub.Inference.Tests.TranslationLanguageRouterTestAssertions;

namespace Trackdub.Inference.Tests;

public sealed partial class TranslationLanguageRouterTests
{
    [Theory]
    [InlineData(false, true, true)]
    [InlineData(true, true, false)]
    [InlineData(false, false, false)]
    public async Task Dynamic_uncached_pair_reports_download_required_only_when_policy_allows(bool offline, bool automatic, bool selectable)
    {
        using var workspace = CreateWorkspace();
        var registry = workspace.WriteManifest(CreateTranslationSpec("test/dynamic", ["dynamic"], "dynamic"));
        var original = registry.Entries.Single();
        registry.AddCatalogEntries([original with {
            ModelId = "test/xx-yy", Aliases = ["xx-yy"],
            LanguageCoverage = new ModelLanguageCoverage([], [], [new ModelLanguagePair("xx", "yy")]),
            CommercialAllowed = true, CommercialUseVerified = true, DownloadFiles = ["onnx/encoder_model.onnx"]
        }]);
        var router = CreateRouter(
            registry,
            new RouterSettings(StudioSettings.Default with { OfflineMode = offline, AutoDownloadTranslationModels = automatic }));
        var route = await router.ResolveRouteAsync("xx", "yy", CancellationToken.None);
        Assert.False(route.IsAvailable);
        Assert.Equal(selectable, route.IsSelectable);
        Assert.Equal(selectable ? TranslationPreparationStatus.DownloadRequired : TranslationPreparationStatus.Unavailable, route.Status);
        var target = Assert.Single(await router.GetSupportedTargetLanguagesAsync("xx", CancellationToken.None));
        Assert.Equal("yy", target.LanguageCode);
        Assert.Equal(selectable, target.IsSelectable);
    }

    [Fact]
    public async Task ResolveRouteAsync_UsesOlderCompleteDirectCacheRootWhenNewerRecordIsPartial()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec(
                "Helsinki-NLP/opus-mt-en-es",
                ["opus-en-es", "helsinki-opus-en-es"],
                "manifest-models/opus-en-es"));
        string olderCompleteRoot = workspace.CreateCacheRoot("complete-opus-en-es");
        workspace.WriteOpusCacheFiles(olderCompleteRoot);
        string newerPartialRoot = workspace.CreateCacheRoot("partial-opus-en-es");
        workspace.WriteCacheFile(newerPartialRoot, "onnx/encoder_model.onnx");

        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord("Helsinki-NLP/opus-mt-en-es", newerPartialRoot, "main", "sha-new", DateTimeOffset.UtcNow),
                new LocalModelCacheRecord("Helsinki-NLP/opus-mt-en-es", olderCompleteRoot, "main", "sha-old", DateTimeOffset.UtcNow.AddDays(-1))
            ]));

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "es", CancellationToken.None);

        AssertDirectRoute(route, "Helsinki-NLP/opus-mt-en-es");
        Assert.Equal(Path.Join(olderCompleteRoot, "onnx", "encoder_model.onnx"), route.ResolvedModelEntryPath);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenDirectEntryOmitsCapabilities_UsesLanguagePairsFallback()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec(
                "example/direct-no-capabilities",
                ["direct-no-capabilities"],
                "manifest-models/direct-no-capabilities",
                new ManifestLanguagePairSpec("en", "fr")) with
            {
                Capabilities = []
            });
        var router = CreateCachedRouter(workspace, registry, "example/direct-no-capabilities", writeOpusFiles: true);

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "fr", CancellationToken.None);

        AssertDirectRoute(route, "example/direct-no-capabilities");
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenMadladHasOnlyQuantizedEncoder_SelectsPivotRoute()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec(
                "Helsinki-NLP/opus-mt-en-es",
                ["opus-en-es", "helsinki-opus-en-es"],
                "manifest-models/opus-en-es"),
            CreateMadladSpec());
        var router = CreateMadladRouter(workspace, registry, includeDefaultEncoder: false);

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "fr", CancellationToken.None);

        AssertPivotRoute(route, "google/madlad400-3b-mt");
        Assert.EndsWith("encoder_model_int8.onnx", route.ResolvedModelEntryPath, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task GetSupportedTargetLanguagesAsync_WhenMadladIsMissing_ReportsUnavailablePairsClearly()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec(
                "Helsinki-NLP/opus-mt-en-es",
                ["opus-en-es", "helsinki-opus-en-es"],
                "manifest-models/opus-en-es"));
        var router = CreateCachedRouter(workspace, registry, "Helsinki-NLP/opus-mt-en-es", writeOpusFiles: true);

        IReadOnlyList<TranslationTargetLanguageOption> options = await router.GetSupportedTargetLanguagesAsync(
            "en",
            CancellationToken.None);

        TranslationTargetLanguageOption spanish = Assert.Single(options, option => option.LanguageCode == "es");
        TranslationTargetLanguageOption french = Assert.Single(options, option => option.LanguageCode == "fr");
        TranslationTargetLanguageOption japanese = Assert.Single(options, option => option.LanguageCode == "ja");

        Assert.True(spanish.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Direct, spanish.RoutingKind);
        AssertUnavailableTarget(french);
        AssertUnavailableTarget(japanese);
    }
}
