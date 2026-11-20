using Trackdub.Domain;

namespace Trackdub.Application.LipSynthesis;

/// <summary>
/// Resolves lip-synthesis inventory gates from bundled model inventory entries.
/// Keeps manifest/licensing rules in Application instead of the Avalonia shell.
/// </summary>
public static class LipSynthesisInventoryGate
{
    public static ModelInventoryEntry? ResolveEntry(
        IReadOnlyList<ModelInventoryEntry> inventory,
        string? preferredModelAlias)
    {
        if (!string.IsNullOrWhiteSpace(preferredModelAlias))
        {
            ModelInventoryEntry? byAlias = inventory.FirstOrDefault(entry =>
                entry.Aliases.Any(alias => string.Equals(alias, preferredModelAlias, StringComparison.OrdinalIgnoreCase)) ||
                string.Equals(entry.ModelId, preferredModelAlias, StringComparison.OrdinalIgnoreCase));
            if (byAlias is not null)
            {
                return byAlias;
            }
        }

        return inventory.FirstOrDefault(entry =>
            string.Equals(entry.Task, "lip-synthesis", StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>License allows commercial use in the active mode.</summary>
    public static bool IsLicenseApproved(ModelInventoryEntry? entry) =>
        entry is { CommercialAllowed: true };

    /// <summary>Engine maturity is not fully verified for commercial shipping yet.</summary>
    /// <remarks>
    /// Keyed to verification, not to CommercialClearance: a risk-accepted entry keeps the experimental opt-in, because
    /// the engine (LatentSyncOnnxLipSynthesisEngine) still reports itself experimental for non-commercial lanes and
    /// this inventory has no lane to check.
    /// </remarks>
    public static bool IsExperimentalEngine(ModelInventoryEntry? entry) =>
        entry is not { CommercialAllowed: true, CommercialUseVerified: true };

    /// <summary>UI stage Run is explicit opt-in for experimental engines.</summary>
    public static bool AllowExperimentalExecution(ModelInventoryEntry? entry) =>
        IsExperimentalEngine(entry);
}
