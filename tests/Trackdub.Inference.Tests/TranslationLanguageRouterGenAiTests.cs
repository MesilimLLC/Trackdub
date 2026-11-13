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
    [InlineData("qwen-instruct")]
    public void IsGenAiPivotEngineFamily_AcceptsSupportedFamilies(string engineFamily) =>
        Assert.True(TranslationLanguageRouter.IsGenAiPivotEngineFamily(engineFamily));

    [Theory]
    [InlineData("madlad")]
    [InlineData("gemma-4-translation")]
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
}
