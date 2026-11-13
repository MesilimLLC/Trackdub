using Trackdub.Domain;
using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Onnx.Translation;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;
using static Trackdub.Inference.Tests.TranslationRouterManifestSpecs;

namespace Trackdub.Inference.Tests;

public sealed partial class TranslationLanguageRouterTests
{
    [Fact]
    public async Task QwenSelection_ResolvesInstalledCudaVariantWithoutCpuFiles()
    {
        Assert.True(BundledModelManifestRegistry.TryLoadDefault(out var registry, out var error), error);
        using var workspace = new TranslationRouterTestWorkspace();
        string root = workspace.CreateCacheRoot("Qwen/Qwen3-4B");
        workspace.WriteGenAiCacheFiles(Path.Join(root, "cuda-int4"));
        var router = new TranslationLanguageRouter(registry!, new InMemoryModelCacheInventory(
            [new LocalModelCacheRecord("Qwen/Qwen3-4B", root, "main", ValidSha256, DateTimeOffset.UtcNow)]));

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "fr", CancellationToken.None, "qwen3-4b-translation");

        Assert.True(route.IsAvailable);
        Assert.EndsWith(Path.Join("cuda-int4", "genai_config.json"), route.ResolvedModelEntryPath, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task QwenSelection_ReportsDownloadRequiredWhenNotInstalled()
    {
        Assert.True(BundledModelManifestRegistry.TryLoadDefault(out var registry, out var error), error);
        var router = new TranslationLanguageRouter(registry!, new InMemoryModelCacheInventory([]));
        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "fr", CancellationToken.None, "qwen3-4b-translation");

        Assert.False(route.IsAvailable);
        Assert.Equal(TranslationPreparationStatus.DownloadRequired, route.Status);
        Assert.Equal("Qwen/Qwen3-4B", route.ModelId);
    }

    [Theory]
    [InlineData("phi-genai")]
    [InlineData("qwen-genai")]
    [InlineData("gemma-genai")]
    public void IsGenAiPivotEngineFamily_AcceptsSupportedFamilies(string engineFamily) =>
        Assert.True(TranslationLanguageRouter.IsGenAiPivotEngineFamily(engineFamily));

    [Theory]
    [InlineData("madlad")]
    [InlineData("gemma-4-translation")]
    [InlineData("qwen-instruct")]
    [InlineData(null)]
    public void IsGenAiPivotEngineFamily_RejectsOtherFamilies(string? engineFamily) =>
        Assert.False(TranslationLanguageRouter.IsGenAiPivotEngineFamily(engineFamily));

    [Theory]
    [InlineData("phi-genai")]
    [InlineData("qwen-genai")]
    [InlineData("gemma-genai")]
    public async Task ResolveRouteAsync_WhenPreferredAliasTargetsGenAiPivot_SelectsPreferredGenAiRoute(string family)
    {
        using var workspace = new TranslationRouterTestWorkspace();
        BundledModelManifestRegistry registry = workspace.WriteManifest(
            CreateMadladSpec(),
            CreatePhiGenAiSpec() with { EngineFamily = family });
        string madladCacheRoot = workspace.CreateCacheRoot("google/madlad400-3b-mt");
        workspace.WriteMadladCacheFiles(madladCacheRoot);
        string phiCacheRoot = workspace.CreateCacheRoot("microsoft/Phi-3.5-mini-instruct-onnx");
        workspace.WriteGenAiCacheFiles(phiCacheRoot);

        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
            [
                new LocalModelCacheRecord("google/madlad400-3b-mt", madladCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow),
                new LocalModelCacheRecord("microsoft/Phi-3.5-mini-instruct-onnx", phiCacheRoot, "main", ValidSha256, DateTimeOffset.UtcNow)
            ]));

        TranslationRouteSelection route = await router.ResolveRouteAsync(
            "en",
            "fr",
            CancellationToken.None,
            preferredModelAlias: "phi-genai-pivot");

        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Pivot, route.RoutingKind);
        Assert.Equal(family, route.ProviderName);
        Assert.Equal(family, route.EngineFamily);
        Assert.Equal("microsoft/Phi-3.5-mini-instruct-onnx", route.ModelId);
        Assert.Equal("phi-genai-pivot", route.PreferredModelAlias);
    }

    [Fact]
    public async Task ResolveRouteAsync_WhenGenAiPivotSelectsVariant_UsesVariantProvenance()
    {
        const string modelId = "microsoft/Phi-3.5-mini-instruct-onnx";
        const string repositoryRevision = "0123456789abcdef0123456789abcdef01234567";
        const string defaultEntryHash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        const string variantEntryHash = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        const string variantWeightsHash = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
        var fixture = new VariantProvenanceFixture(
            repositoryRevision,
            defaultEntryHash,
            variantEntryHash,
            variantWeightsHash);
        using var workspace = new TranslationRouterTestWorkspace();
        string modelRoot = workspace.CreateCacheRoot(modelId);
        workspace.WriteCacheFile(modelRoot, "cuda-int4/genai_config.json", "{}");
        workspace.WriteCacheFile(modelRoot, "cuda-int4/model.onnx", "weights");
        BundledModelManifestRegistry registry = WriteVariantProvenanceManifest(
            workspace.RootPath,
            modelId,
            fixture);
        var router = new TranslationLanguageRouter(
            registry,
            new InMemoryModelCacheInventory(
                [new LocalModelCacheRecord(modelId, modelRoot, repositoryRevision, defaultEntryHash, DateTimeOffset.UtcNow)]));

        TranslationRouteSelection route = await router.ResolveRouteAsync("en", "fr", CancellationToken.None);

        Assert.True(route.IsAvailable);
        Assert.EndsWith(
            Path.Join("cuda-int4", "genai_config.json"),
            route.ResolvedModelEntryPath,
            StringComparison.OrdinalIgnoreCase);
        Assert.NotNull(route.ModelProvenance);
        Assert.Equal(repositoryRevision, route.ModelProvenance.Revision);
        Assert.Equal(variantEntryHash, route.ModelProvenance.Sha256);
        Assert.Equal(
            new[] { "cuda-int4/genai_config.json", "cuda-int4/model.onnx" },
            route.ModelProvenance.FileHashes.Keys.OrderBy(static path => path, StringComparer.Ordinal).ToArray());
    }

    private static BundledModelManifestRegistry WriteVariantProvenanceManifest(
        string workspaceRoot,
        string modelId,
        VariantProvenanceFixture fixture)
    {
        string manifestPath = Path.Join(workspaceRoot, "bundled-models.manifest.json");
        string manifestJson = $$"""
            {
              "models": [
                {
                  "model_id": "{{modelId}}",
                  "task": "translation",
                  "engine_family": "phi-genai",
                  "capabilities": ["translation", "pivot-translation"],
                  "language_coverage": {
                    "source_languages": ["multi"],
                    "target_languages": ["multi"],
                    "language_pairs": []
                  },
                  "tier": "quality",
                  "license": "MIT",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": true,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_use_verified": true,
                  "source_url": "https://huggingface.co/{{modelId}}",
                  "revision": "{{fixture.Revision}}",
                  "sha256": "{{fixture.DefaultEntryHash}}",
                  "download_files": ["default/genai_config.json", "cuda-int4/genai_config.json", "cuda-int4/model.onnx"],
                  "download_file_hashes": {
                    "default/genai_config.json": "{{fixture.DefaultEntryHash}}",
                    "cuda-int4/genai_config.json": "{{fixture.VariantEntryHash}}",
                    "cuda-int4/model.onnx": "{{fixture.VariantWeightsHash}}"
                  },
                  "aliases": ["phi-genai-pivot"],
                  "root_path": "./manifest-models/phi-genai",
                  "benchmark_entry": "default/genai_config.json",
                  "variants": [
                    {
                      "alias": "cuda-int4",
                      "entry_path": "cuda-int4/genai_config.json",
                      "download_files": ["cuda-int4/genai_config.json", "cuda-int4/model.onnx"]
                    }
                  ]
                }
              ]
            }
            """;
        File.WriteAllText(manifestPath, manifestJson);
        return BundledModelManifestRegistry.Load(manifestPath);
    }

    private sealed record VariantProvenanceFixture(
        string Revision,
        string DefaultEntryHash,
        string VariantEntryHash,
        string VariantWeightsHash);
}
