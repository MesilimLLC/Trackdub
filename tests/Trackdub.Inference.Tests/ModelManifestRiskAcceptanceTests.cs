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
    [InlineData("custom", false, "not-a-hash", "sha256")]
    public void LoadCatalog_RejectsInvalidRiskAcceptance(string license, bool verified, string sha256, string expectedText)
    {
        ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
            () => Load(RiskManifest(license, verified, sha256, withRisk: true)));

        Assert.Contains(expectedText, exception.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void LoadCatalog_NamesRiskAcceptancePathForMissingField()
    {
        string json = RiskManifest("custom", verified: false, ValidSha256, withRisk: true)
            .Replace("\"decision_record\": \"owner decision 2026-11-19\",", string.Empty, StringComparison.Ordinal);

        ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(() => Load(json));

        Assert.Contains("$.models[0].risk_acceptance", exception.Message, StringComparison.Ordinal);
        Assert.Contains("decision_record", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void LoadCatalog_RejectsIncompleteAcceptedRisks()
    {
        string json = RiskManifest("custom", verified: false, ValidSha256, withRisk: true)
            .Replace("\"unverified training-data provenance\"", "\"something else\"", StringComparison.Ordinal);

        ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(() => Load(json));

        Assert.Contains("unverified training-data provenance", exception.Message, StringComparison.Ordinal);
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
                    "owner": "owner",
                    "date": "2026-11-19",
                    "accepted_risks": [
                      "licence indemnity to the licensor",
                      "termination and deletion on licence end",
                      "Licensor's right to change terms",
                      "unverified training-data provenance"
                    ]
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
