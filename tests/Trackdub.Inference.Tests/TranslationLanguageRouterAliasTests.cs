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
            LanguageCoverage = new ModelLanguageCoverage([], [], [new ModelLanguagePair("en", "es")]),
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
