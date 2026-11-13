using Trackdub.Inference.Onnx;
using Trackdub.Domain;

namespace Trackdub.Inference.Tests;

public sealed class OnnxExecutionSessionFactoryCatalogPlacementTests
{
    [Fact]
    public void ResolveCatalogEffectiveProviderSelection_reports_cpu_when_placement_cannot_be_established()
    {
        OnnxExecutionSessionFactory.EffectiveProviderResolution resolution =
            OnnxExecutionSessionFactory.ResolveCatalogEffectiveProviderSelection([]);

        Assert.Equal(ExecutionProviderKind.Cpu, resolution.Provider);
        Assert.Contains("placement could not be established", resolution.FallbackReason, StringComparison.Ordinal);
    }
}
