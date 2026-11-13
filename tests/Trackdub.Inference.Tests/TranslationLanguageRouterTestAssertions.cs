using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Onnx.Translation;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Tests;

internal static class TranslationLanguageRouterTestAssertions
{
    internal static void AssertDirectRoute(TranslationRouteSelection route, string modelId)
    {
        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Direct, route.RoutingKind);
        Assert.Equal(modelId, route.ModelId);
    }

    internal static void AssertPivotRoute(TranslationRouteSelection route, string modelId)
    {
        Assert.True(route.IsAvailable);
        Assert.Equal(TranslationRoutingKind.Pivot, route.RoutingKind);
        Assert.Equal(modelId, route.ModelId);
    }

    internal static void AssertProvenance(TranslationRouteSelection route, BundledModelManifestRegistry registry)
    {
        Assert.NotNull(route.ModelProvenance);
        BundledModelManifestEntry entry = registry.Entries.Single(candidate => candidate.ModelId == route.ModelId);
        Assert.Equal(entry.Revision, route.ModelProvenance.Revision);
        Assert.Equal(entry.Sha256, route.ModelProvenance.Sha256);
    }

    internal static void AssertUnavailableTarget(TranslationTargetLanguageOption option)
    {
        Assert.False(option.IsAvailable);
        Assert.Contains("pivot translation is unavailable", option.Detail, StringComparison.Ordinal);
    }

    internal static void AssertDownloadRequired(TranslationRouteSelection route, string modelId)
    {
        Assert.Equal(TranslationPreparationStatus.DownloadRequired, route.Status);
        Assert.Equal(modelId, route.ModelId);
    }
}
