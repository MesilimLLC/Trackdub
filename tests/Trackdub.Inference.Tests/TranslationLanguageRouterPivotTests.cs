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
    public async Task ResolveRouteAsync_WhenDirectPairIsMissingAndMadladIsInstalled_SelectsPivotRoute()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec(
                "Helsinki-NLP/opus-mt-en-es",
                ["opus-en-es", "helsinki-opus-en-es"],
                "manifest-models/opus-en-es"),
            CreateMadladSpec());
        var router = CreateMadladRouter(workspace, registry);

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "fr", CancellationToken.None);

        AssertPivotRoute(route, "google/madlad400-3b-mt");
        Assert.Equal("madlad400", route.ProviderName);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenPivotEntryOmitsCapabilities_UsesLanguageCoverageFallback()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreatePivotSpec("example/pivot-no-capabilities", ["pivot-no-capabilities"], [], []));
        var router = CreateCachedRouter(workspace, registry, "example/pivot-no-capabilities", writeOpusFiles: false);

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "fr", CancellationToken.None);

        AssertPivotRoute(route, "example/pivot-no-capabilities");
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenPreferredAliasTargetsCapabilityLessPivot_SelectsPreferredPivotRoute()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreatePivotSpec("example/pivot-default", ["pivot-default"], [], []),
            CreatePivotSpec("example/pivot-preferred", ["pivot-preferred"], [], []));
        string defaultCacheRoot = workspace.CreateCacheRoot("example/pivot-default");
        workspace.WriteMadladCacheFiles(defaultCacheRoot);
        string preferredCacheRoot = workspace.CreateCacheRoot("example/pivot-preferred");
        workspace.WriteMadladCacheFiles(preferredCacheRoot);

        var router = CreateRouter(
            registry,
            settings: null,
            ("example/pivot-default", defaultCacheRoot),
            ("example/pivot-preferred", preferredCacheRoot));

        TranslationRouteSelection route = await router.ResolveRouteAsync(
            "en", "fr", CancellationToken.None, preferredModelAlias: "pivot-preferred");

        AssertPivotRoute(route, "example/pivot-preferred");
        Assert.Equal("pivot-preferred", route.PreferredModelAlias);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenPivotOnlyModelDeclaresLanguagePair_DoesNotSelectItAsDirectRoute()
    {
        using var workspace = CreateWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreatePivotSpec(
                "example/pivot-only-with-pairs",
                ["pivot-only-model"],
                ["translation", "pivot-translation"],
                [new ManifestLanguagePairSpec("en", "es")]) with
            {
                RootFolder = "manifest-models/pivot-only"
            });
        var router = CreateCachedRouter(workspace, registry, "example/pivot-only-with-pairs", writeOpusFiles: false);

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "es", CancellationToken.None);

        AssertPivotRoute(route, "example/pivot-only-with-pairs");
    }
}
