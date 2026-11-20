using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Tests;

public sealed class ModelManifestRiskAcceptanceTests
{
    private const string ValidSha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void LoadCatalog_GrantsClearanceOnlyWithRiskAcceptance(bool withRisk)
    {
        ModelManifest manifest = Assert.Single(Load(RiskManifest("custom", verified: false, ValidSha256, withRisk)).Models);

        Assert.Equal(withRisk, manifest.RiskAccepted);
        Assert.Equal(withRisk, manifest.CommercialClearance);
        Assert.Equal(withRisk, manifest.CommercialSafeMode);
        Assert.False(manifest.CommercialUseVerified);
    }

    [Theory]
    [InlineData("custom", true, ValidSha256, "risk_acceptance")]
    [InlineData("mit", false, ValidSha256, "custom")]
    [InlineData("custom", false, "", "sha256")]
    public void LoadCatalog_RejectsInvalidRiskAcceptance(string license, bool verified, string sha256, string expectedText)
    {
        ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
            () => Load(RiskManifest(license, verified, sha256, withRisk: true)));

        Assert.Contains(expectedText, exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    private static ModelManifestCatalog Load(string json)
    {
        string manifestPath = Path.Join(Path.GetTempPath(), $"{Guid.NewGuid():N}.json");
        File.WriteAllText(manifestPath, json);
        try
        {
            return ModelManifestLoader.LoadCatalog(manifestPath);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    private static string RiskManifest(string license, bool verified, string sha256, bool withRisk)
    {
        string risk = withRisk
            ? """
              ,
                  "risk_acceptance": {
                    "decision_record": "owner decision 2026-11-19",
                    "accepted_by": "owner",
                    "accepted_on": "2026-11-19",
                    "accepted_risks": [ "licence indemnity", "unverified training data" ]
                  }
              """
            : string.Empty;
        return $$"""
            {
              "models": [
                {
                  "model_id": "example/custom-model",
                  "task": "separation",
                  "engine_family": "example-engine",
                  "license": "{{license}}",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_use_verified": {{(verified ? "true" : "false")}},
                  "source_url": "https://example.invalid/model",
                  "revision": "rev",
                  "sha256": "{{sha256}}",
                  "variants": []{{risk}}
                }
              ]
            }
            """;
    }
}
