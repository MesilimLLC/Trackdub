using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.Translation;
using Trackdub.Inference.Onnx.Translation;
using Trackdub.Inference.Runtime.ModelManifest;
using static Trackdub.Inference.Tests.TranslationRouterManifestSpecs;
using static Trackdub.Inference.Tests.TranslationLanguageRouterTestSetup;

namespace Trackdub.Inference.Tests;

public sealed partial class TranslationLanguageRouterTests
{
    [Fact]
    public async Task ResolveRouteAsync_WhenOpusAutoHasNoCoverage_NeverFallsThroughToMadlad()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateMadladSpec());
        TranslationLanguageRouter router = CreateMadladRouter(workspace, registry);

        TranslationRouteSelection route = await router.ResolveRouteAsync(
            "en", "sw", CancellationToken.None, preferredModelAlias: OpusMtAutoModel.Alias);

        Assert.False(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Unavailable, route.RoutingKind);
        Assert.Equal("none", route.ProviderName);
        Assert.Equal(
            "Opus MT has no model for English to Swahili. Choose a different target language or a cloud provider.",
            route.UnavailableReason);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenOpusAutoDownloadsOff_ReportsDownloadsOff()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec("test/opus-mt-en-de", ["opus-en-de"], "manifest-models/en-de", new ManifestLanguagePairSpec("en", "de")));
        TranslationLanguageRouter router = CreateRouter(
            registry,
            new RouterSettings(StudioSettings.Default with { OfflineMode = true, AutoDownloadTranslationModels = true }));

        TranslationRouteSelection route = await router.ResolveRouteAsync(
            "en", "de", CancellationToken.None, preferredModelAlias: OpusMtAutoModel.Alias);

        Assert.False(route.IsAvailable);
        Assert.Equal(
            "Opus MT needs the English to German model, but automatic model downloads are turned off. You can download it in Settings under Models.",
            route.UnavailableReason);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenOpusAutoPairIsInstalled_SelectsDirectRoute()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec(
                "Helsinki-NLP/opus-mt-en-es",
                ["opus-en-es", "helsinki-opus-en-es"],
                "manifest-models/opus-en-es"),
            CreateMadladSpec());
        TranslationLanguageRouter router = CreateDirectAndPivotRouter(workspace, registry);

        TranslationRouteSelection route = await router.ResolveRouteAsync(
            "en", "es", CancellationToken.None, preferredModelAlias: OpusMtAutoModel.Alias);

        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Direct, route.RoutingKind);
        Assert.Equal("Helsinki-NLP/opus-mt-en-es", route.ModelId);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenOpusAutoPairIsDownloadable_ReportsDownloadRequired()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateTranslationSpec("test/dynamic", ["dynamic"], "dynamic"));
        BundledModelManifestEntry original = registry.Entries.Single();
        registry.AddCatalogEntries([original with {
            ModelId = "test/xx-yy", Aliases = ["xx-yy"],
            LanguageCoverage = new ModelLanguageCoverage([], [], [new Trackdub.Inference.Runtime.ModelManifest.ModelLanguagePair("xx", "yy")]),
            CommercialAllowed = true, CommercialUseVerified = true, DownloadFiles = ["onnx/encoder_model.onnx"]
        }]);
        TranslationLanguageRouter router = CreateRouter(
            registry,
            new RouterSettings(StudioSettings.Default with { OfflineMode = false, AutoDownloadTranslationModels = true }));

        TranslationRouteSelection route = await router.ResolveRouteAsync(
            "xx", "yy", CancellationToken.None, preferredModelAlias: OpusMtAutoModel.Alias);

        Assert.Equal(TranslationPreparationStatus.DownloadRequired, route.Status);
        Assert.Equal("test/xx-yy", route.ModelId);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenOpusAutoSourceUnknown_ReportsTranscribeFirst()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateMadladSpec());
        TranslationLanguageRouter router = CreateMadladRouter(workspace, registry);

        TranslationRouteSelection route = await router.ResolveRouteAsync(
            "auto", "de", CancellationToken.None, preferredModelAlias: OpusMtAutoModel.Alias);

        Assert.False(route.IsAvailable);
        Assert.Equal("Transcribe first so the source language is known.", route.UnavailableReason);
        Assert.Equal(TranslationPreparationStatus.SourceUnresolved, route.Status);
    }
}
