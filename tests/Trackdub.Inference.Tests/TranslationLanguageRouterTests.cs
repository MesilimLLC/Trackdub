using System.Text.Json;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Translation;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;
using static Trackdub.Inference.Tests.TranslationRouterManifestSpecs;

namespace Trackdub.Inference.Tests;

public sealed partial class TranslationLanguageRouterTests
{
    [Fact]
    public async Task UnknownSourceAllowsTargetSelectionWithoutClaimingRunnableRoute()
    {
        using var workspace = new TranslationRouterTestWorkspace();
        var registry = workspace.WriteManifest(CreateTranslationSpec("test/dynamic", ["dynamic"], "dynamic"));
        var router = new TranslationLanguageRouter(registry, new InMemoryModelCacheInventory([]));

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

    [Theory]
    [InlineData(false, true, true)]
    [InlineData(true, true, false)]
    [InlineData(false, false, false)]
    public async Task Dynamic_uncached_pair_reports_download_required_only_when_policy_allows(bool offline, bool automatic, bool selectable)
    {
        using var workspace = new TranslationRouterTestWorkspace();
        var registry = workspace.WriteManifest(CreateTranslationSpec("test/dynamic", ["dynamic"], "dynamic"));
        var original = registry.Entries.Single();
        registry.AddCatalogEntries([original with {
            ModelId = "test/xx-yy", Aliases = ["xx-yy"],
            LanguageCoverage = new ModelLanguageCoverage([], [], [new ModelLanguagePair("xx", "yy")]),
            CommercialAllowed = true, CommercialUseVerified = true, DownloadFiles = ["onnx/encoder_model.onnx"]
        }]);
        var router = new TranslationLanguageRouter(registry, new InMemoryModelCacheInventory([]),
            settingsService: new RouterSettings(StudioSettings.Default with { OfflineMode = offline, AutoDownloadTranslationModels = automatic }));
        var route = await router.ResolveRouteAsync("xx", "yy", CancellationToken.None);
        Assert.False(route.IsAvailable);
        Assert.Equal(selectable, route.IsSelectable);
        Assert.Equal(selectable ? TranslationPreparationStatus.DownloadRequired : TranslationPreparationStatus.Unavailable, route.Status);
        var target = Assert.Single(await router.GetSupportedTargetLanguagesAsync("xx", CancellationToken.None));
        Assert.Equal("yy", target.LanguageCode);
        Assert.Equal(selectable, target.IsSelectable);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenMadladAndDirectOpusPairAreInstalled_PrefersDirectOpusByDefault()
    {
        using var workspace = new TranslationRouterTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec(
                "Helsinki-NLP/opus-mt-en-es",
                ["opus-en-es", "helsinki-opus-en-es"],
                "manifest-models/opus-en-es"),
            CreateMadladSpec());
        string opusCacheRoot = workspace.CreateCacheRoot("Helsinki-NLP/opus-mt-en-es");
        workspace.WriteOpusCacheFiles(opusCacheRoot);
        string madladCacheRoot = workspace.CreateCacheRoot("google/madlad400-3b-mt");
        workspace.WriteMadladCacheFiles(madladCacheRoot);

        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord("Helsinki-NLP/opus-mt-en-es", opusCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow),
                new LocalModelCacheRecord("google/madlad400-3b-mt", madladCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ]));

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "es", CancellationToken.None);

        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Direct, route.RoutingKind);
        Assert.Equal("opus-mt", route.ProviderName);
        Assert.Equal("Helsinki-NLP/opus-mt-en-es", route.ModelId);
        Assert.NotNull(route.ModelProvenance);
        Assert.Equal(registry.Entries.Single(entry => entry.ModelId == route.ModelId).Revision, route.ModelProvenance.Revision);
        Assert.Equal(registry.Entries.Single(entry => entry.ModelId == route.ModelId).Sha256, route.ModelProvenance.Sha256);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenPreferredAliasTargetsDirectOpusPair_SelectsDirectRoute()
    {
        using var workspace = new TranslationRouterTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec(
                "Helsinki-NLP/opus-mt-en-es",
                ["opus-en-es", "helsinki-opus-en-es"],
                "manifest-models/opus-en-es"),
            CreateMadladSpec());
        string opusCacheRoot = workspace.CreateCacheRoot("Helsinki-NLP/opus-mt-en-es");
        workspace.WriteOpusCacheFiles(opusCacheRoot);
        string madladCacheRoot = workspace.CreateCacheRoot("google/madlad400-3b-mt");
        workspace.WriteMadladCacheFiles(madladCacheRoot);

        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord("Helsinki-NLP/opus-mt-en-es", opusCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow),
                new LocalModelCacheRecord("google/madlad400-3b-mt", madladCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ]));

        TranslationRouteSelection route = await router.ResolveRouteAsync(
            "en", "es", CancellationToken.None, preferredModelAlias: "opus-en-es");

        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Direct, route.RoutingKind);
        Assert.Equal("opus-mt", route.ProviderName);
        Assert.Equal("Helsinki-NLP/opus-mt-en-es", route.ModelId);
        Assert.EndsWith("encoder_model.onnx", route.ResolvedModelEntryPath, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task ResolveRouteAsync_UsesOlderCompleteDirectCacheRootWhenNewerRecordIsPartial()
    {
        using var workspace = new TranslationRouterTestWorkspace();
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

        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Direct, route.RoutingKind);
        Assert.Equal(Path.Join(olderCompleteRoot, "onnx", "encoder_model.onnx"), route.ResolvedModelEntryPath);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenDirectPairIsMissingAndMadladIsInstalled_SelectsPivotRoute()
    {
        using var workspace = new TranslationRouterTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec(
                "Helsinki-NLP/opus-mt-en-es",
                ["opus-en-es", "helsinki-opus-en-es"],
                "manifest-models/opus-en-es"),
            CreateMadladSpec());
        string madladCacheRoot = workspace.CreateCacheRoot("google/madlad400-3b-mt");
        workspace.WriteMadladCacheFiles(madladCacheRoot);

        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord("google/madlad400-3b-mt", madladCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ]));

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "fr", CancellationToken.None);

        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Pivot, route.RoutingKind);
        Assert.Equal("madlad400", route.ProviderName);
        Assert.Equal("google/madlad400-3b-mt", route.ModelId);
    }

    [Fact]
    public async Task ResolveRouteAsync_UsesManifestLanguagePairsInsteadOfAliasConvention()
    {
        using var workspace = new TranslationRouterTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec(
                "example/direct-en-fr",
                ["custom-direct-en-fr"],
                "manifest-models/custom-direct-en-fr",
                sourceLanguage: "en",
                targetLanguage: "fr"));
        string directCacheRoot = workspace.CreateCacheRoot("example/direct-en-fr");
        workspace.WriteOpusCacheFiles(directCacheRoot);

        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord("example/direct-en-fr", directCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ]));

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "fr", CancellationToken.None);

        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Direct, route.RoutingKind);
        Assert.Equal("example/direct-en-fr", route.ModelId);
        Assert.Equal("custom-direct-en-fr", route.PreferredModelAlias);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenDirectEntryOmitsCapabilities_UsesLanguagePairsFallback()
    {
        using var workspace = new TranslationRouterTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "example/direct-no-capabilities",
                Task: "translation",
                EngineFamily: "opus-mt",
                Capabilities: [],
                LanguagePairs: [new ManifestLanguagePairSpec("en", "fr")],
                SourceLanguages: [],
                TargetLanguages: [],
                Tier: "fast",
                License: "Apache-2.0",
                CommercialAllowed: true,
                RequiresAttribution: true,
                Aliases: ["direct-no-capabilities"],
                RootFolder: "manifest-models/direct-no-capabilities",
                BenchmarkEntry: "onnx/encoder_model.onnx",
                Variants:
                [
                    new ManifestVariantSpec("merged-decoder", "onnx/decoder_model_merged.onnx")
                ]));
        string cacheRoot = workspace.CreateCacheRoot("example/direct-no-capabilities");
        workspace.WriteOpusCacheFiles(cacheRoot);

        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord("example/direct-no-capabilities", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ]));

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "fr", CancellationToken.None);

        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Direct, route.RoutingKind);
        Assert.Equal("example/direct-no-capabilities", route.ModelId);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenPivotEntryOmitsCapabilities_UsesLanguageCoverageFallback()
    {
        using var workspace = new TranslationRouterTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "example/pivot-no-capabilities",
                Task: "translation",
                EngineFamily: "madlad",
                Capabilities: [],
                LanguagePairs: [],
                SourceLanguages: ["multi"],
                TargetLanguages: ["multi"],
                Tier: "quality",
                License: "Apache-2.0",
                CommercialAllowed: true,
                RequiresAttribution: true,
                Aliases: ["pivot-no-capabilities"],
                RootFolder: "manifest-models/pivot-no-capabilities",
                BenchmarkEntry: "encoder_model.onnx",
                Variants:
                [
                    new ManifestVariantSpec("int8", "encoder_model_int8.onnx")
                ]));
        string cacheRoot = workspace.CreateCacheRoot("example/pivot-no-capabilities");
        workspace.WriteMadladCacheFiles(cacheRoot);

        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord("example/pivot-no-capabilities", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ]));

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "fr", CancellationToken.None);

        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Pivot, route.RoutingKind);
        Assert.Equal("example/pivot-no-capabilities", route.ModelId);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenPreferredAliasTargetsCapabilityLessPivot_SelectsPreferredPivotRoute()
    {
        using var workspace = new TranslationRouterTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "example/pivot-default",
                Task: "translation",
                EngineFamily: "madlad",
                Capabilities: [],
                LanguagePairs: [],
                SourceLanguages: ["multi"],
                TargetLanguages: ["multi"],
                Tier: "quality",
                License: "Apache-2.0",
                CommercialAllowed: true,
                RequiresAttribution: true,
                Aliases: ["pivot-default"],
                RootFolder: "manifest-models/pivot-default",
                BenchmarkEntry: "encoder_model.onnx",
                Variants:
                [
                    new ManifestVariantSpec("int8", "encoder_model_int8.onnx")
                ]),
            new ManifestSpec(
                ModelId: "example/pivot-preferred",
                Task: "translation",
                EngineFamily: "madlad",
                Capabilities: [],
                LanguagePairs: [],
                SourceLanguages: ["multi"],
                TargetLanguages: ["multi"],
                Tier: "quality",
                License: "Apache-2.0",
                CommercialAllowed: true,
                RequiresAttribution: true,
                Aliases: ["pivot-preferred"],
                RootFolder: "manifest-models/pivot-preferred",
                BenchmarkEntry: "encoder_model.onnx",
                Variants:
                [
                    new ManifestVariantSpec("int8", "encoder_model_int8.onnx")
                ]));
        string defaultCacheRoot = workspace.CreateCacheRoot("example/pivot-default");
        workspace.WriteMadladCacheFiles(defaultCacheRoot);
        string preferredCacheRoot = workspace.CreateCacheRoot("example/pivot-preferred");
        workspace.WriteMadladCacheFiles(preferredCacheRoot);

        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord("example/pivot-default", defaultCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow),
                new LocalModelCacheRecord("example/pivot-preferred", preferredCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ]));

        TranslationRouteSelection route = await router.ResolveRouteAsync(
            "en",
            "fr",
            CancellationToken.None,
            preferredModelAlias: "pivot-preferred");

        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Pivot, route.RoutingKind);
        Assert.Equal("example/pivot-preferred", route.ModelId);
        Assert.Equal("pivot-preferred", route.PreferredModelAlias);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenPivotEntryUsesExplicitLanguagePairs_SelectsPivotRoute()
    {
        using var workspace = new TranslationRouterTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "example/pivot-language-pairs",
                Task: "translation",
                EngineFamily: "madlad",
                Capabilities: ["translation", "pivot-translation"],
                LanguagePairs: [new ManifestLanguagePairSpec("en", "fr")],
                SourceLanguages: [],
                TargetLanguages: [],
                Tier: "quality",
                License: "Apache-2.0",
                CommercialAllowed: true,
                RequiresAttribution: true,
                Aliases: ["pivot-language-pairs"],
                RootFolder: "manifest-models/pivot-language-pairs",
                BenchmarkEntry: "encoder_model.onnx",
                Variants:
                [
                    new ManifestVariantSpec("int8", "encoder_model_int8.onnx")
                ]));
        string cacheRoot = workspace.CreateCacheRoot("example/pivot-language-pairs");
        workspace.WriteMadladCacheFiles(cacheRoot);

        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord("example/pivot-language-pairs", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ]));

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "fr", CancellationToken.None);

        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Pivot, route.RoutingKind);
        Assert.Equal("example/pivot-language-pairs", route.ModelId);
    }

    [Fact]
    public void WriteManifest_WhenDirectTranslationPairsAreMissing_ThrowsValidationException()
    {
        using var workspace = new TranslationRouterTestWorkspace();
        ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(() =>
            workspace.WriteManifest(
                new ManifestSpec(
                    ModelId: "legacy/opus-model",
                    Task: "translation",
                    EngineFamily: "opus-mt",
                    Capabilities: ["translation", "direct-translation"],
                    LanguagePairs: [],
                    SourceLanguages: [],
                    TargetLanguages: [],
                    Tier: "fast",
                    License: "Apache-2.0",
                    CommercialAllowed: true,
                    RequiresAttribution: true,
                    Aliases: ["legacy-opus-en-es"],
                    RootFolder: "manifest-models/legacy-opus-en-es",
                    BenchmarkEntry: "encoder_model.onnx",
                    Variants:
                    [
                        new ManifestVariantSpec("merged-decoder", "decoder_model_merged.onnx")
                    ])));

        Assert.Contains("language_pairs", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenNonEnglishSpanishSourceUsesMadlad_SelectsPivotRoute()
    {
        using var workspace = new TranslationRouterTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(CreateMadladSpec());
        string madladCacheRoot = workspace.CreateCacheRoot("google/madlad400-3b-mt");
        workspace.WriteMadladCacheFiles(madladCacheRoot);

        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord("google/madlad400-3b-mt", madladCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ]));

        TranslationRouteSelection route = await router.ResolveRouteAsync("el", "en", CancellationToken.None);

        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Pivot, route.RoutingKind);
        Assert.Equal("madlad400", route.ProviderName);
        Assert.Equal("google/madlad400-3b-mt", route.ModelId);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenMadladHasOnlyQuantizedEncoder_SelectsPivotRoute()
    {
        using var workspace = new TranslationRouterTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec(
                "Helsinki-NLP/opus-mt-en-es",
                ["opus-en-es", "helsinki-opus-en-es"],
                "manifest-models/opus-en-es"),
            CreateMadladSpec());
        string madladCacheRoot = workspace.CreateCacheRoot("google/madlad400-3b-mt");
        workspace.WriteMadladCacheFiles(madladCacheRoot, includeDefaultEncoder: false);

        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord("google/madlad400-3b-mt", madladCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ]));

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "fr", CancellationToken.None);

        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Pivot, route.RoutingKind);
        Assert.EndsWith("encoder_model_int8.onnx", route.ResolvedModelEntryPath, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task GetSupportedTargetLanguagesAsync_WhenMadladIsMissing_ReportsUnavailablePairsClearly()
    {
        using var workspace = new TranslationRouterTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateTranslationSpec(
                "Helsinki-NLP/opus-mt-en-es",
                ["opus-en-es", "helsinki-opus-en-es"],
                "manifest-models/opus-en-es"));
        string opusCacheRoot = workspace.CreateCacheRoot("Helsinki-NLP/opus-mt-en-es");
        workspace.WriteOpusCacheFiles(opusCacheRoot);

        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord("Helsinki-NLP/opus-mt-en-es", opusCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ]));

        IReadOnlyList<TranslationTargetLanguageOption> options = await router.GetSupportedTargetLanguagesAsync(
            "en",
            CancellationToken.None);

        TranslationTargetLanguageOption spanish = Assert.Single(options, option => option.LanguageCode == "es");
        TranslationTargetLanguageOption french = Assert.Single(options, option => option.LanguageCode == "fr");
        TranslationTargetLanguageOption japanese = Assert.Single(options, option => option.LanguageCode == "ja");

        Assert.True(spanish.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Direct, spanish.RoutingKind);
        Assert.False(french.IsAvailable);
        Assert.Contains("pivot translation is unavailable", french.Detail, StringComparison.Ordinal);
        Assert.False(japanese.IsAvailable);
        Assert.Contains("pivot translation is unavailable", japanese.Detail, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenPivotOnlyModelDeclaresLanguagePair_DoesNotSelectItAsDirectRoute()
    {
        using var workspace = new TranslationRouterTestWorkspace();
        // Create a pivot-only translation model that also declares a language pair
        // (it should NOT be selected as a direct route)
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            new ManifestSpec(
                ModelId: "example/pivot-only-with-pairs",
                Task: "translation",
                EngineFamily: "madlad",
                Capabilities: ["translation", "pivot-translation"],
                LanguagePairs: [new ManifestLanguagePairSpec("en", "es")],
                SourceLanguages: ["multi"],
                TargetLanguages: ["multi"],
                Tier: "quality",
                License: "Apache-2.0",
                CommercialAllowed: true,
                RequiresAttribution: true,
                Aliases: ["pivot-only-model"],
                RootFolder: "manifest-models/pivot-only",
                BenchmarkEntry: "encoder_model.onnx",
                Variants:
                [
                    new ManifestVariantSpec("int8", "encoder_model_int8.onnx")
                ]));
        string cacheRoot = workspace.CreateCacheRoot("example/pivot-only-with-pairs");
        workspace.WriteMadladCacheFiles(cacheRoot);

        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord("example/pivot-only-with-pairs", cacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ]));

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "es", CancellationToken.None);

        // Should route as pivot, not direct
        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Pivot, route.RoutingKind);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenUncachedSinglePairSortsFirst_UsesInstalledMultiPairModel()
    {
        using var workspace = new TranslationRouterTestWorkspace();
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
        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord("test/opus-mt-en-ROMANCE", romanceCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ]),
            settingsService: new RouterSettings(StudioSettings.Default with { OfflineMode = false, AutoDownloadTranslationModels = true }));

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "es", CancellationToken.None);
        TranslationRouteSelection explicitAlias = await router.ResolveRouteAsync("en", "es", CancellationToken.None, preferredModelAlias: "en-es");

        Assert.True(route.IsAvailable);
        Assert.Equal("test/opus-mt-en-ROMANCE", route.ModelId);
        Assert.NotEqual(TranslationPreparationStatus.DownloadRequired, route.Status);
        Assert.Equal(TranslationPreparationStatus.DownloadRequired, explicitAlias.Status);
        Assert.Equal("test/opus-mt-en-es", explicitAlias.ModelId);
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
        using var workspace = new TranslationRouterTestWorkspace();
        BundledModelManifestEntry entry = workspace.WriteManifest(
            CreateTranslationSpec("test/model", ["model"], "manifest-models/model")).Entries.Single() with
        { SourceUrl = sourceUrl };

        Assert.Equal(expected, TranslationLanguageRouter.ResolveProvenanceRepository(entry));
    }
}
