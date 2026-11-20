using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Tests;

public sealed class ModelRiskAcceptanceTests
{
    private const string PinnedSha = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    private const string CompleteBlock = """
        "risk_acceptance": {
          "decision_record": "docs/legal/MODEL_LICENSE_POLICY.md",
          "accepted_risks": [ "licence indemnity to the licensor", "unverified training-data provenance" ],
          "owner": "owner-name",
          "date": "2026-11-19"
        },
        """;

    private sealed record Entry
    {
        public string License { get; init; } = "custom";
        public bool CommercialAllowed { get; init; } = true;
        public bool RedistributionAllowed { get; init; } = true;
        public bool Verified { get; init; }
        public string Sha { get; init; } = PinnedSha;
        public string Block { get; init; } = CompleteBlock;
    }

    [Fact]
    public void Unverified_model_with_risk_acceptance_is_commercial_safe()
    {
        ModelManifest manifest = Load(new Entry());

        Assert.False(manifest.CommercialUseVerified);
        Assert.Equal("owner-name", manifest.RiskAcceptance?.Owner);
        Assert.True(manifest.CommercialSafeMode);
    }

    [Fact]
    public void Custom_licence_alone_is_not_commercial_safe()
    {
        ModelManifest manifest = Load(new Entry { Block = string.Empty });

        Assert.Null(manifest.RiskAcceptance);
        Assert.False(manifest.CommercialSafeMode);
    }

    [Fact]
    public void Non_commercial_licence_is_rejected() =>
        AssertRejected(new Entry { License = "non-commercial", CommercialAllowed = false }, "non-commercial or unknown");

    [Fact]
    public void Unknown_licence_is_rejected() =>
        AssertRejected(new Entry { License = "unknown" }, "non-commercial or unknown");

    [Fact]
    public void Commercial_use_not_allowed_is_rejected() =>
        AssertRejected(new Entry { CommercialAllowed = false }, "'commercial_allowed'");

    [Fact]
    public void Redistribution_not_allowed_is_rejected() =>
        AssertRejected(new Entry { RedistributionAllowed = false }, "'redistribution_allowed'");

    [Fact]
    public void Verified_model_does_not_need_the_exception() =>
        AssertRejected(new Entry { Verified = true }, "'commercial_use_verified' is true");

    [Fact]
    public void Missing_artifact_hash_is_rejected() =>
        AssertRejected(new Entry { Sha = string.Empty }, "'sha256'");

    [Theory]
    [InlineData("""{ "decision_record": "", "accepted_risks": [ "x" ], "owner": "o", "date": "2026-11-19" }""")]
    [InlineData("""{ "decision_record": "r", "accepted_risks": [], "owner": "o", "date": "2026-11-19" }""")]
    [InlineData("""{ "decision_record": "r", "accepted_risks": [ "x" ], "owner": "", "date": "2026-11-19" }""")]
    [InlineData("""{ "decision_record": "r", "accepted_risks": [ "x" ], "owner": "o", "date": "19/11/2026" }""")]
    public void Incomplete_block_is_rejected(string block) =>
        AssertRejected(new Entry { Block = $"\"risk_acceptance\": {block}," }, "risk_acceptance");

    [Fact]
    public void Merge_comparison_accepts_identical_records_and_rejects_changed_or_missing_ones()
    {
        ModelRiskAcceptance baseRecord = new("record", ["a", "b"], "owner", "2026-11-19");

        Assert.True(ModelRiskAcceptanceReader.SameAcceptance(null, null));
        Assert.True(ModelRiskAcceptanceReader.SameAcceptance(baseRecord, new("record", ["a", "b"], "owner", "2026-11-19")));
        Assert.False(ModelRiskAcceptanceReader.SameAcceptance(baseRecord, null));
        Assert.False(ModelRiskAcceptanceReader.SameAcceptance(baseRecord, baseRecord with { Owner = "someone" }));
        Assert.False(ModelRiskAcceptanceReader.SameAcceptance(baseRecord, baseRecord with { Date = "2026-11-20" }));
        Assert.False(ModelRiskAcceptanceReader.SameAcceptance(baseRecord, baseRecord with { DecisionRecord = "other" }));
        Assert.False(ModelRiskAcceptanceReader.SameAcceptance(baseRecord, baseRecord with { AcceptedRisks = ["a"] }));
    }

    private static void AssertRejected(Entry entry, string expected)
    {
        string path = WriteManifest(entry);
        try
        {
            var error = Assert.Throws<ModelManifestValidationException>(() => ModelManifestLoader.LoadCatalog(path));

            Assert.Contains(expected, error.Message);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static ModelManifest Load(Entry entry)
    {
        string path = WriteManifest(entry);
        try
        {
            return Assert.Single(ModelManifestLoader.LoadCatalog(path).Models);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static string WriteManifest(Entry entry)
    {
        string json = $$"""
            {
              "model_id": "example/accepted-risk",
              "task": "separation",
              "engine_family": "sam-audio",
              "license": "{{entry.License}}",
              "commercial_allowed": {{Json(entry.CommercialAllowed)}},
              "redistribution_allowed": {{Json(entry.RedistributionAllowed)}},
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_use_verified": {{Json(entry.Verified)}},
              {{entry.Block}}
              "source_url": "https://example.invalid/model",
              "revision": "main",
              "sha256": "{{entry.Sha}}",
              "download_files": [ "model.onnx" ],
              "variants": [ { "alias": "default", "entry_path": "model.onnx", "is_default": true } ]
            }
            """;
        string path = Path.Join(Path.GetTempPath(), $"{Guid.NewGuid():N}.json");
        File.WriteAllText(path, json);
        return path;
    }

    private static string Json(bool value) => value ? "true" : "false";
}
