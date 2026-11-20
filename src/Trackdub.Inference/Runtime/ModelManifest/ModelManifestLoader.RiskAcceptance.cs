using System.Text.Json;

namespace Trackdub.Inference.Runtime.ModelManifest;

// Owner-accepted risk for custom commercial licences (MODEL_LICENSE_POLICY.md, accepted-risk exception).
public static partial class ModelManifestLoader
{
    private readonly record struct RiskTerms(bool CommercialUseVerified, bool Eligible);

    private static ModelManifest ReadManifestWithRisk(JsonElement element, string path, string sourceName)
    {
        ModelManifest manifest = ReadManifest(element, path, sourceName);
        RiskTerms terms = new(manifest.CommercialUseVerified, IsRiskAcceptanceEligible(manifest));
        ModelRiskAcceptance? riskAcceptance = ReadRiskAcceptance(element, path, sourceName, terms);
        return riskAcceptance is null ? manifest : manifest with { RiskAcceptance = riskAcceptance };
    }

    private static bool IsRiskAcceptanceEligible(ModelManifest manifest) =>
        manifest.License == ModelLicenseKind.Custom &&
        manifest.CommercialAllowed &&
        manifest.RedistributionAllowed &&
        !string.IsNullOrWhiteSpace(manifest.Sha256);

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
                $"Manifest '{sourceName}' entry '{path}' sets 'risk_acceptance', which needs a 'custom' licence, commercial_allowed and redistribution_allowed true, and a sha256.");
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

        IReadOnlyList<string> risks = ReadStringArray(value, "accepted_risks", path, sourceName);
        if (risks.Count == 0)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' risk_acceptance must list at least one accepted risk.");
        }

        return new ModelRiskAcceptance(
            ReadRequiredString(value, "decision_record", path, sourceName),
            ReadRequiredString(value, "accepted_by", path, sourceName),
            ReadRequiredString(value, "accepted_on", path, sourceName),
            risks);
    }
}
