using Trackdub.Domain;
using Trackdub.Inference.Onnx.DeepFilterNet;

namespace Trackdub.Inference.Tests;

public sealed class RiskAcceptanceGateSiteTests
{
    [Theory]
    [InlineData(false, false, false)]
    [InlineData(true, false, true)]
    [InlineData(false, true, true)]
    public void InventoryEntry_ClearsOnVerificationOrRiskAcceptance(bool verified, bool riskAccepted, bool expected)
    {
        ModelInventoryEntry entry = new(
            "example/model", "Example", "separation", "example", "custom", true, verified,
            ModelCacheState.Missing, null, null, null, RiskAccepted: riskAccepted);

        Assert.Equal(expected, entry.CommercialClearance);
    }

    [Theory]
    [InlineData(true, false, false, false)]
    [InlineData(true, false, true, true)]
    [InlineData(true, true, false, true)]
    [InlineData(false, false, true, false)]
    public void DeepFilterNetPaths_RequireCommercialAllowedAndClearance(
        bool commercialAllowed, bool verified, bool riskAccepted, bool expected)
    {
        DeepFilterNetModelPaths paths = new("root", "enc", "erb", "df", commercialAllowed, verified)
        {
            RiskAccepted = riskAccepted,
        };

        Assert.Equal(expected, paths.IsCommercialSafe);
    }
}
