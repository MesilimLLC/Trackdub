using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Translation;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;
using static Trackdub.Inference.Tests.TranslationRouterManifestSpecs;
using static Trackdub.Inference.Tests.TranslationLanguageRouterTestSetup;
using static Trackdub.Inference.Tests.TranslationLanguageRouterTestAssertions;

namespace Trackdub.Inference.Tests;

public sealed partial class TranslationLanguageRouterTests
{
    [Fact]
    public async Task UnknownSourceAllowsTargetSelectionWithoutClaimingRunnableRoute()
    {
        using var workspace = CreateWorkspace();
        var registry = workspace.WriteManifest(CreateTranslationSpec("test/dynamic", ["dynamic"], "dynamic"));
        var router = CreateRouter(registry, settings: null);

        var targets = await router.GetSupportedTargetLanguagesAsync("auto", CancellationToken.None);
        Assert.NotEmpty(targets);
        Assert.All(targets, target =>
        {
            Assert.True(target.IsSelectable);
            Assert.False(target.IsAvailable);
            Assert.Equal(TranslationPreparationStatus.SourceUnresolved, target.Status);
        });
        var route = await router.ResolveRouteAsync("auto", "es", CancellationToken.None);
        Assert.False(route.IsAvailable);
        Assert.Equal(TranslationPreparationStatus.SourceUnresolved, route.Status);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenMadladAndDirectOpusPairAreInstalled_PrefersDirectOpusByDefault()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec(
                "Helsinki-NLP/opus-mt-en-es",
                ["opus-en-es", "helsinki-opus-en-es"],
                "manifest-models/opus-en-es"),
            CreateMadladSpec());
        var router = CreateDirectAndPivotRouter(workspace, registry);

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "es", CancellationToken.None);

        AssertDirectRoute(route, "Helsinki-NLP/opus-mt-en-es");
        Assert.Equal("opus-mt", route.ProviderName);
        AssertProvenance(route, registry);
    }

    [Fact]
    public async Task ResolveRouteAsync_UsesManifestLanguagePairsInsteadOfAliasConvention()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec(
                "example/direct-en-fr",
                ["custom-direct-en-fr"],
                "manifest-models/custom-direct-en-fr",
                new ManifestLanguagePairSpec("en", "fr")));
        string directCacheRoot = workspace.CreateCacheRoot("example/direct-en-fr");
        workspace.WriteOpusCacheFiles(directCacheRoot);

        var router = CreateRouter(registry, settings: null, ("example/direct-en-fr", directCacheRoot));

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "fr", CancellationToken.None);

        AssertDirectRoute(route, "example/direct-en-fr");
        Assert.Equal("custom-direct-en-fr", route.PreferredModelAlias);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenPivotEntryUsesExplicitLanguagePairs_SelectsPivotRoute()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreatePivotSpec(
                "example/pivot-language-pairs",
                ["pivot-language-pairs"],
                ["translation", "pivot-translation"],
                [new ManifestLanguagePairSpec("en", "fr")]));
        var router = CreateCachedRouter(workspace, registry, "example/pivot-language-pairs", writeOpusFiles: false);

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "fr", CancellationToken.None);

        AssertPivotRoute(route, "example/pivot-language-pairs");
    }
}
