using System.Globalization;
using System.Text.Json;

namespace Trackdub.Inference.Runtime.ModelManifest;

/// <summary>Where a manifest entry came from, for error messages.</summary>
internal readonly record struct ManifestLocation(string Path, string Source)
{
    public ModelManifestValidationException Invalid(string detail) =>
        new($"Manifest '{Source}' entry '{Path}' field 'risk_acceptance' {detail}");
}

/// <summary>
/// Reads and validates the optional <c>risk_acceptance</c> block of a manifest entry. The block is the
/// owner-recorded exception described in docs/legal/MODEL_LICENSE_POLICY.md; it never replaces
/// <c>commercial_use_verified</c>.
/// </summary>
internal static class ModelRiskAcceptanceReader
{
    private static readonly string[] RequiredText = ["decision_record", "owner", "date"];

    public static bool IsCommercialSafe(bool commercialUseVerified, ModelRiskAcceptance? riskAcceptance) =>
        commercialUseVerified || riskAcceptance is not null;

    public static ModelManifest Attach(ModelManifest manifest, JsonElement element, ManifestLocation location)
    {
        if (!element.TryGetProperty("risk_acceptance", out JsonElement block) || block.ValueKind is JsonValueKind.Null)
        {
            return manifest;
        }

        ModelRiskAcceptance acceptance = Read(block, location);
        string? problem = FindEligibilityProblem(manifest);
        return problem is null
            ? manifest with { RiskAcceptance = acceptance }
            : throw location.Invalid($"is set but {problem}.");
    }

    public static void EnsureSameRiskAcceptance(
        BundledModelManifestEntry existing,
        BundledModelManifestEntry incoming,
        string incomingManifestPath)
    {
        if ((existing.RiskAcceptance is null) != (incoming.RiskAcceptance is null))
        {
            throw new InvalidOperationException(
                $"Generated manifest '{incomingManifestPath}' cannot merge model '{incoming.ModelId}' because its risk_acceptance does not match the base entry.");
        }
    }

    private static ModelRiskAcceptance Read(JsonElement block, ManifestLocation location)
    {
        if (block.ValueKind is not JsonValueKind.Object)
        {
            throw location.Invalid("must be an object.");
        }

        string[] text = RequiredText.Select(name => ReadString(block, name)).ToArray();
        string[] risks = ReadRisks(block);
        bool complete = Array.TrueForAll(text, static value => value.Length > 0) && risks.Length > 0 && IsIsoDate(text[2]);
        return complete
            ? new ModelRiskAcceptance(text[0], risks, text[1], text[2])
            : throw location.Invalid(
                "requires 'decision_record', 'owner', a 'date' in yyyy-MM-dd form and a non-empty 'accepted_risks' list.");
    }

    private static string? FindEligibilityProblem(ModelManifest manifest)
    {
        (bool Failed, string Message)[] checks =
        [
            (manifest.License is ModelLicenseKind.NonCommercial or ModelLicenseKind.CcByNc40 or ModelLicenseKind.Unknown,
                "the licence is non-commercial or unknown"),
            (!manifest.CommercialAllowed, "'commercial_allowed' is not true"),
            (!manifest.RedistributionAllowed, "'redistribution_allowed' is not true"),
            (manifest.CommercialUseVerified, "'commercial_use_verified' is true, so the exception is not needed"),
            (!IsSha256(manifest.Sha256), "'sha256' does not pin the shipped artifact")
        ];

        return checks.FirstOrDefault(static check => check.Failed).Message;
    }

    private static string ReadString(JsonElement block, string name) =>
        block.TryGetProperty(name, out JsonElement value) && value.ValueKind is JsonValueKind.String
            ? value.GetString()?.Trim() ?? string.Empty
            : string.Empty;

    private static string[] ReadRisks(JsonElement block)
    {
        if (!block.TryGetProperty("accepted_risks", out JsonElement list) || list.ValueKind is not JsonValueKind.Array)
        {
            return [];
        }

        string[] risks = list.EnumerateArray()
            .Select(static item => item.ValueKind is JsonValueKind.String ? item.GetString()?.Trim() ?? string.Empty : string.Empty)
            .ToArray();
        return Array.Exists(risks, static risk => risk.Length == 0) ? [] : risks;
    }

    private static bool IsIsoDate(string value) =>
        DateOnly.TryParseExact(value, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out _);

    private static bool IsSha256(string value) =>
        value.Length == 64 && value.All(static c => char.IsAsciiHexDigit(c));
}
