using System.Text.Json;

namespace Trackdub.Inference.Runtime.ModelManifest;

// Owner-accepted risk for commercial licences with unverified provenance (MODEL_LICENSE_POLICY.md, accepted-risk exception).
public static partial class ModelManifestLoader
{
    private readonly record struct RiskTerms(bool CommercialUseVerified, bool Eligible);

    // The four risks MODEL_LICENSE_POLICY.md requires the record to list, matched case-insensitively after trimming.
    private static readonly string[] RequiredAcceptedRisks =
    [
        "licence indemnity to the licensor",
        "termination and deletion on licence end",
        "licensor's right to change terms",
        "unverified training-data provenance",
    ];

    private static ModelManifest ReadManifestWithRisk(JsonElement element, string path, string sourceName)
    {
        ModelManifest manifest = ReadManifest(element, path, sourceName);
        RiskTerms terms = new(manifest.CommercialUseVerified, IsRiskAcceptanceEligible(manifest));
        ModelRiskAcceptance? riskAcceptance = ReadRiskAcceptance(element, path, sourceName, terms);
        return riskAcceptance is null ? manifest : manifest with { RiskAcceptance = riskAcceptance };
    }

    private static bool IsRiskAcceptanceEligible(ModelManifest manifest) =>
        RiskAcceptanceEligibility.IsEligible(manifest);

    private static void EnsureRiskAcceptanceAllowed(RiskTerms terms, string path, string sourceName)
    {
        if (terms.CommercialUseVerified)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' sets both 'commercial_use_verified' and 'risk_acceptance'; a verified model needs no risk acceptance.");
        }

        if (!terms.Eligible)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' sets 'risk_acceptance', which needs a licence of {RiskAcceptanceEligibility.EligibleLicenceList}, commercial_allowed and redistribution_allowed true, and a valid sha256.");
        }
    }

    private static ModelRiskAcceptance? ReadRiskAcceptance(
        JsonElement element, string path, string sourceName, RiskTerms terms)
    {
        if (!element.TryGetProperty("risk_acceptance", out JsonElement value))
        {
            return null;
        }

        EnsureRiskAcceptanceAllowed(terms, path, sourceName);

        if (value.ValueKind != JsonValueKind.Object)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' field 'risk_acceptance' must be an object.");
        }

        string riskPath = $"{path}.risk_acceptance";
        IReadOnlyList<string> risks = ReadStringArray(value, "accepted_risks", riskPath, sourceName);
        string[] missing = MissingAcceptedRisks(risks);
        if (missing.Length > 0)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{riskPath}.accepted_risks' is missing required risks: {string.Join("; ", missing)}.");
        }

        return new ModelRiskAcceptance(
            ReadRequiredString(value, "decision_record", riskPath, sourceName),
            ReadRequiredString(value, "owner", riskPath, sourceName),
            ReadRequiredString(value, "date", riskPath, sourceName),
            risks);
    }

    private static string[] MissingAcceptedRisks(IReadOnlyList<string> risks) =>
        RequiredAcceptedRisks
            .Where(required => !risks.Any(risk => string.Equals(risk.Trim(), required, StringComparison.OrdinalIgnoreCase)))
            .ToArray();
}
