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
    [Fact]
    public async Task ResolveRouteAsync_WhenPreferredAliasTargetsDirectOpusPair_SelectsDirectRoute()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec(
                "Helsinki-NLP/opus-mt-en-es",
                ["opus-en-es", "helsinki-opus-en-es"],
                "manifest-models/opus-en-es"),
            CreateMadladSpec());
        var router = CreateDirectAndPivotRouter(workspace, registry);

        TranslationRouteSelection route = await router.ResolveRouteAsync(
            "en", "es", CancellationToken.None, preferredModelAlias: "opus-en-es");

        AssertDirectRoute(route, "Helsinki-NLP/opus-mt-en-es");
        Assert.Equal("opus-mt", route.ProviderName);
        Assert.EndsWith("encoder_model.onnx", route.ResolvedModelEntryPath, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenNonEnglishSpanishSourceUsesMadlad_SelectsPivotRoute()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateMadladSpec());
        var router = CreateMadladRouter(workspace, registry);

        TranslationRouteSelection route = await router.ResolveRouteAsync("el", "en", CancellationToken.None);

        AssertPivotRoute(route, "google/madlad400-3b-mt");
        Assert.Equal("madlad400", route.ProviderName);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenUncachedSinglePairSortsFirst_UsesInstalledMultiPairModel()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec("test/opus-mt-en-ROMANCE", ["en-romance"], "manifest-models/romance") with
            {
                LanguagePairs = [new ManifestLanguagePairSpec("en", "es"), new ManifestLanguagePairSpec("en", "fr")]
            });
        BundledModelManifestEntry multiPair = registry.Entries.Single();
        registry.AddCatalogEntries([multiPair with
        {
            ModelId = "test/opus-mt-en-es",
            Aliases = ["en-es"],
            LanguageCoverage = new ModelLanguageCoverage([], [], [new Trackdub.Inference.Runtime.ModelManifest.ModelLanguagePair("en", "es")]),
            CommercialAllowed = true,
            CommercialUseVerified = true,
            DownloadFiles = ["onnx/encoder_model.onnx"]
        }]);
        string romanceCacheRoot = workspace.CreateCacheRoot("test/opus-mt-en-ROMANCE");
        workspace.WriteOpusCacheFiles(romanceCacheRoot);
        var router = CreateRouter(
            registry,
            new RouterSettings(StudioSettings.Default with { OfflineMode = false, AutoDownloadTranslationModels = true }),
            ("test/opus-mt-en-ROMANCE", romanceCacheRoot));

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "es", CancellationToken.None);
        TranslationRouteSelection explicitAlias = await router.ResolveRouteAsync("en", "es", CancellationToken.None, preferredModelAlias: "en-es");

        Assert.True(route.IsAvailable);
        Assert.Equal("test/opus-mt-en-ROMANCE", route.ModelId);
        Assert.NotEqual(TranslationPreparationStatus.DownloadRequired, route.Status);
        AssertDownloadRequired(explicitAlias, "test/opus-mt-en-es");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ResolveRouteAsync_NeverAutoRoutesARiskAcceptedModelButHonoursItsAlias(bool installed)
    {
        // MODEL_LICENSE_POLICY.md: an accepted-risk model is opt-in and never selected automatically, whether it still has
        // to be downloaded or was installed through an explicit opt-in.
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec("test/opus-mt-en-es", ["en-es"], "manifest-models/opus-en-es"));
        registry.AddCatalogEntries([registry.Entries.Single() with
        {
            ModelId = "test/opus-mt-en-it-risk",
            Aliases = ["en-it-risk"],
            LanguageCoverage = new ModelLanguageCoverage([], [], [new Trackdub.Inference.Runtime.ModelManifest.ModelLanguagePair("en", "it")]),
            CommercialAllowed = true,
            CommercialUseVerified = false,
            RiskAcceptance = new ModelRiskAcceptance("record", ["a", "b", "c", "d"], "owner", "2026-11-19"),
            DownloadFiles = ["onnx/encoder_model.onnx"]
        }]);
        string cacheRoot = workspace.CreateCacheRoot("test/opus-mt-en-it-risk");
        workspace.WriteOpusCacheFiles(cacheRoot);
        (string ModelId, string RootPath)[] cached = installed ? [("test/opus-mt-en-it-risk", cacheRoot)] : [];
        var router = CreateRouter(
            registry,
            new RouterSettings(StudioSettings.Default with { OfflineMode = false, AutoDownloadTranslationModels = true }),
            cached);

        TranslationRouteSelection automatic = await router.ResolveRouteAsync("en", "it", CancellationToken.None);
        TranslationRouteSelection explicitAlias = await router.ResolveRouteAsync("en", "it", CancellationToken.None, preferredModelAlias: "en-it-risk");

        Assert.NotEqual(TranslationPreparationStatus.DownloadRequired, automatic.Status);
        Assert.NotEqual("test/opus-mt-en-it-risk", automatic.ModelId);
        Assert.Equal("test/opus-mt-en-it-risk", explicitAlias.ModelId);
        Assert.Equal(installed, explicitAlias.IsAvailable);
    }

    [Fact]
    public void WriteManifest_WhenDirectTranslationPairsAreMissing_ThrowsValidationException()
    {
        using var workspace = CreateWorkspace();
        ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(() =>
            workspace.WriteManifest(
                CreateTranslationSpec("legacy/opus-model", ["legacy-opus-en-es"], "manifest-models/legacy-opus-en-es") with
                {
                    LanguagePairs = [],
                    BenchmarkEntry = "encoder_model.onnx",
                    Variants = [new ManifestVariantSpec("merged-decoder", "decoder_model_merged.onnx")]
                }));

        Assert.Contains("language_pairs", exception.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("https://huggingface.co/tonythethompson/madlad400-3b-mt-onnx", "tonythethompson/madlad400-3b-mt-onnx")]
    [InlineData("https://huggingface.co/onnx-community/opus-mt-en-es/", "onnx-community/opus-mt-en-es")]
    [InlineData("https://example.invalid/test-model", "test/model")]
    [InlineData("https://huggingface.co/datasets/org/name", "test/model")]
    [InlineData("https://huggingface.co/org/name?download=true", "org/name")]
    [InlineData("https://huggingface.co/org/name#readme", "org/name")]
    [InlineData("https://huggingface.co/org/name/tree/main", "org/name")]
    [InlineData("http://huggingface.co/org/name", "test/model")]
    public void ResolveProvenanceRepository_ReturnsRepoIdFromSourceUrl(string sourceUrl, string expected)
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestEntry entry = workspace.WriteManifest(
            CreateTranslationSpec("test/model", ["model"], "manifest-models/model")).Entries.Single() with
        { SourceUrl = sourceUrl };

        Assert.Equal(expected, TranslationLanguageRouter.ResolveProvenanceRepository(entry));
    }
}
