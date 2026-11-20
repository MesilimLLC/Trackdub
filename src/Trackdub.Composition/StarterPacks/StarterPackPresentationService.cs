using Trackdub.Contracts;
using Trackdub.Contracts.Licensing;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.StarterPacks;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.StarterPacks;

public sealed class StarterPackPresentationService(
    IStarterPackCatalog catalog,
    IModelInventoryService inventory,
    IStudioSettingsService settingsService,
    IConsentService consentService,
    IHardwareProfilerService hardwareProfiler,
    IStarterPackCompatibilityService compatibility,
    ICloudCredentialReadiness cloudCredentialReadiness,
    BundledModelManifestRegistry manifestRegistry,
    StarterPackGpuSetupAdvisor gpuSetupAdvisor) : IStarterPackPresentationService
{
    private static readonly string[] BundledLocalPackIds = ["basic", "balanced", "premium"];

    public async Task<IReadOnlyList<StarterPackSummary>> ListSummariesAsync(CancellationToken cancellationToken = default)
    {
        StudioSettings settings = await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
        string? recommendedPackId = await GetRecommendedPackIdAsync(cancellationToken).ConfigureAwait(false);
        IReadOnlyList<StarterPackDefinition> packs = await catalog.ListDefinitionsAsync(cancellationToken).ConfigureAwait(false);
        IReadOnlyList<ModelInventoryEntry> inventoryEntries = await inventory
            .GetAllAsync(cancellationToken)
            .ConfigureAwait(false);

        var summaries = new List<StarterPackSummary>();
        foreach (StarterPackDefinition pack in packs)
        {
            string profileId = StarterPackResolver.ResolveDefaultProfileId(pack);
            StarterPackCompatibilityReport? compatibilityReport = await EvaluateCompatibilityIfLocalAsync(
                pack,
                profileId,
                cancellationToken).ConfigureAwait(false);
            CloudCredentialReadinessReport? cloudReadiness = await TryEvaluateCloudReadinessAsync(
                pack,
                cancellationToken).ConfigureAwait(false);
            StarterPackGpuSetupHint? gpuSetup = await gpuSetupAdvisor
                .ResolveAsync(compatibilityReport, cancellationToken)
                .ConfigureAwait(false);
            summaries.Add(BuildSummary(
                pack,
                profileId,
                settings,
                recommendedPackId,
                inventoryEntries,
                manifestRegistry,
                consentService.IsVoiceCloningConsentGranted,
                compatibilityReport,
                cloudReadiness,
                gpuSetup));
        }

        return summaries;
    }

    public async Task<StarterPackSummary> GetSummaryAsync(
        string packId,
        string profileId,
        CancellationToken cancellationToken = default)
    {
        StudioSettings settings = await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
        string? recommendedPackId = await GetRecommendedPackIdAsync(cancellationToken).ConfigureAwait(false);
        StarterPackDefinition pack = await catalog.GetAsync(packId, cancellationToken).ConfigureAwait(false);
        IReadOnlyList<ModelInventoryEntry> inventoryEntries = await inventory
            .GetAllAsync(cancellationToken)
            .ConfigureAwait(false);
        StarterPackCompatibilityReport? compatibilityReport = await EvaluateCompatibilityIfLocalAsync(
            pack,
            profileId,
            cancellationToken).ConfigureAwait(false);
        CloudCredentialReadinessReport? cloudReadiness = await TryEvaluateCloudReadinessAsync(
            pack,
            cancellationToken).ConfigureAwait(false);
        StarterPackGpuSetupHint? gpuSetup = await gpuSetupAdvisor
            .ResolveAsync(compatibilityReport, cancellationToken)
            .ConfigureAwait(false);

        return BuildSummary(
            pack,
            profileId,
            settings,
            recommendedPackId,
            inventoryEntries,
            manifestRegistry,
            consentService.IsVoiceCloningConsentGranted,
            compatibilityReport,
            cloudReadiness,
            gpuSetup);
    }

    public async Task<string?> GetRecommendedPackIdAsync(CancellationToken cancellationToken = default)
    {
        StudioSettings settings = await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
        string tier = hardwareProfiler.ResolveEffectiveModelTierPreference(settings);
        string tierPackId = MapTierToPackId(tier) ?? "balanced";

        IReadOnlyList<string> runnablePackIds = await GetRunnablePackIdsAsync(cancellationToken).ConfigureAwait(false);
        if (runnablePackIds.Contains(tierPackId, StringComparer.OrdinalIgnoreCase))
        {
            return tierPackId;
        }

        return runnablePackIds.Count > 0 ? runnablePackIds[0] : tierPackId;
    }

    public async Task<IReadOnlyList<string>> GetRunnablePackIdsAsync(CancellationToken cancellationToken = default)
    {
        IReadOnlyList<StarterPackDefinition> packs = await catalog
            .ListDefinitionsAsync(cancellationToken)
            .ConfigureAwait(false);

        var runnable = new List<string>();
        foreach (StarterPackDefinition pack in packs.Where(p =>
                     BundledLocalPackIds.Contains(p.Id, StringComparer.OrdinalIgnoreCase)))
        {
            string profileId = StarterPackResolver.ResolveDefaultProfileId(pack);
            StarterPackCompatibilityReport report = await compatibility
                .EvaluateAsync(
                    pack.Id,
                    profileId,
                    hardwareProfile: null,
                    cancellationToken,
                    skipProviderSmokeTest: true)
                .ConfigureAwait(false);
            if (report.AllStagesRunnable)
            {
                runnable.Add(pack.Id);
            }
        }

        return runnable;
    }

    public async Task<bool> RequiresVoiceCloningConsentAsync(
        string packId,
        string profileId,
        CancellationToken cancellationToken = default)
    {
        StarterPackDefinition pack = await catalog.GetAsync(packId, cancellationToken).ConfigureAwait(false);
        IReadOnlyList<string> requiredModelIds = StarterPackResolver.GetRequiredModelIds(pack, profileId);
        return RequiresVoiceCloningConsent(requiredModelIds, manifestRegistry);
    }

    public static StarterPackSummary BuildSummary(
        StarterPackDefinition pack,
        string profileId,
        StudioSettings settings,
        string? recommendedPackId,
        IReadOnlyList<ModelInventoryEntry> inventoryEntries,
        BundledModelManifestRegistry manifestRegistry,
        bool voiceCloningConsentGranted,
        StarterPackCompatibilityReport? compatibilityReport = null,
        CloudCredentialReadinessReport? cloudReadiness = null,
        StarterPackGpuSetupHint? gpuSetup = null)
    {
        bool isPureCloudPack = pack.PackKind == StarterPackKind.Cloud;
        bool requiresCloudCredentials = isPureCloudPack ||
            pack.CloudDefaults is not null ||
            (pack.Apply?.CloudStages?.Count ?? 0) > 0;
        IReadOnlyList<string> requiredModelIds = StarterPackResolver.GetRequiredModelIds(pack, profileId);
        int installedCount = isPureCloudPack
            ? 0
            : requiredModelIds.Count(modelId =>
            {
                ModelInventoryEntry? entry = inventoryEntries.FirstOrDefault(candidate =>
                    string.Equals(candidate.ModelId, modelId, StringComparison.OrdinalIgnoreCase));
                return entry?.State is ModelCacheState.Ready or ModelCacheState.Installed;
            });

        bool hasCommercialGap = isPureCloudPack
            ? false
            : HasModelManifestSafetyGap(requiredModelIds, manifestRegistry);
        bool consentRequired = isPureCloudPack
            ? false
            : RequiresVoiceCloningConsent(requiredModelIds, manifestRegistry);
        bool consentSatisfied = !consentRequired || voiceCloningConsentGranted;
        bool cloudCredentialsReady = !requiresCloudCredentials || (cloudReadiness?.IsReady ?? false);
        bool installedComplete = isPureCloudPack || installedCount == requiredModelIds.Count;
        bool isRunnable = isPureCloudPack || (compatibilityReport?.AllStagesRunnable ?? false);
        bool canApply = installedComplete && !hasCommercialGap && consentSatisfied && isRunnable && cloudCredentialsReady;

        string? blockedReason = null;
        if (!cloudCredentialsReady)
        {
            blockedReason = cloudReadiness?.BlockedReason
                ?? "Configure API keys in Cloud Models before applying this pack.";
        }
        else if (!installedComplete)
        {
            blockedReason = $"Download pack first ({installedCount}/{requiredModelIds.Count} installed).";
        }
        else if (hasCommercialGap)
        {
            blockedReason = BuildModelManifestSafetyBlockMessage(requiredModelIds, manifestRegistry);
        }
        else if (consentRequired)
        {
            blockedReason = "Voice cloning consent is required before applying this pack.";
        }
        else if (!isRunnable)
        {
            blockedReason = "One or more required stages are not runnable on this machine.";
        }

        IReadOnlyList<string>? compatibilityWarnings = BuildCompatibilityWarnings(compatibilityReport, gpuSetup);

        bool applied = string.Equals(settings.AppliedStarterPackId, pack.Id, StringComparison.OrdinalIgnoreCase);
        bool recommended = pack.PackOrigin != StarterPackOrigin.User &&
            string.Equals(recommendedPackId, pack.Id, StringComparison.OrdinalIgnoreCase);

        string statusLabel = applied
            ? "applied"
            : hasCommercialGap
                ? "license review needed"
                : recommended
                    ? "recommended"
                    : string.Empty;

        return new StarterPackSummary(
            pack.Id,
            pack.DisplayName,
            pack.TierPreference,
            pack.Profiles.Select(profile => profile.Id).ToList(),
            requiredModelIds.Count,
            installedCount,
            canApply,
            hasCommercialGap,
            consentRequired,
            recommended,
            applied,
            blockedReason,
            statusLabel,
            compatibilityReport?.CompatibilityStatus,
            isRunnable,
            compatibilityWarnings,
            pack.PackKind,
            cloudCredentialsReady,
            pack.PackOrigin == StarterPackOrigin.User,
            gpuSetup);
    }

    internal static IReadOnlyList<string>? BuildCompatibilityWarnings(
        StarterPackCompatibilityReport? compatibilityReport,
        StarterPackGpuSetupHint? gpuSetup)
    {
        if (gpuSetup is { GpuFallbackStageCount: > 0 })
        {
            return null;
        }

        if (gpuSetup is { RequiresGpuVariantOptimization: true })
        {
            return ["Some models need GPU-optimized variants. Use Optimize in Model Manager after GPU runtime is ready."];
        }

        if (compatibilityReport is null)
        {
            return null;
        }

        var lines = new List<string>();

        // Positive inventory: what each model will run. Avoids a wall of
        // "path unavailable" lines when the pack is still runnable.
        List<string> runtimeLines = compatibilityReport.Stages
            .Where(static stage => stage.Runnable)
            .Select(static stage => stage.DescribeResolvedRuntime())
            .Where(static line => !string.IsNullOrWhiteSpace(line))
            .ToList();
        if (runtimeLines.Count > 0)
        {
            lines.Add("Runtime plan");
            lines.AddRange(runtimeLines);
        }

        // True exceptions only — real blockers, not every preferred-path miss.
        lines.AddRange(compatibilityReport.Stages
            .Where(static stage => stage is { Runnable: false } or { FallbackApplied: true })
            .Select(static stage => stage.DescribeFallback())
            .Where(static message => !string.IsNullOrWhiteSpace(message)));

        return lines.Count == 0 ? null : lines;
    }

    private async Task<StarterPackCompatibilityReport?> EvaluateCompatibilityIfLocalAsync(
        StarterPackDefinition pack,
        string profileId,
        CancellationToken cancellationToken)
    {
        if (pack.PackKind == StarterPackKind.Cloud)
        {
            return null;
        }

        return await compatibility
            .EvaluateAsync(
                pack.Id,
                profileId,
                hardwareProfile: null,
                cancellationToken,
                skipProviderSmokeTest: true)
            .ConfigureAwait(false);
    }

    private async Task<CloudCredentialReadinessReport?> TryEvaluateCloudReadinessAsync(
        StarterPackDefinition pack,
        CancellationToken cancellationToken)
    {
        StarterPackCloudDefaults? cloudDefaults = pack.CloudDefaults
            ?? (pack.Apply is null ? null : StarterPackDataDrivenApplyMapper.ToCloudDefaults(pack.Apply));
        if (cloudDefaults is null)
        {
            return null;
        }

        try
        {
            return await cloudCredentialReadiness
                .EvaluateAsync(cloudDefaults, cancellationToken)
                .ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            return new CloudCredentialReadinessReport(
                false,
                [],
                $"Cloud credential check failed: {ex.Message}");
        }
    }

    public static string? MapTierToPackId(string tier) =>
        tier.Trim().ToLowerInvariant() switch
        {
            "fast" => "basic",
            "balanced" => "balanced",
            "quality" => "premium",
            _ => "balanced"
        };

    private static bool HasModelManifestSafetyGap(
        IReadOnlyList<string> requiredModelIds,
        BundledModelManifestRegistry manifestRegistry) =>
        requiredModelIds.Any(modelId =>
        {
            BundledModelManifestEntry? entry = manifestRegistry.Entries
                .FirstOrDefault(candidate => candidate.ModelId.Equals(modelId, StringComparison.OrdinalIgnoreCase));
            return entry is null || !entry.CommercialClearance;
        });

    private static string BuildModelManifestSafetyBlockMessage(
        IReadOnlyList<string> requiredModelIds,
        BundledModelManifestRegistry manifestRegistry)
    {
        foreach (string modelId in requiredModelIds)
        {
            BundledModelManifestEntry? entry = manifestRegistry.Entries
                .FirstOrDefault(candidate => candidate.ModelId.Equals(modelId, StringComparison.OrdinalIgnoreCase));
            if (entry is null)
            {
                return $"Required model '{modelId}' is not in the bundled manifest.";
            }

            if (!entry.CommercialClearance)
            {
                string alias = entry.Aliases.FirstOrDefault() ?? modelId;
                return $"License review needed: {alias} is not commercial-use verified.";
            }
        }

        return "License review needed: a required model is not commercial-use verified.";
    }

    private static bool RequiresVoiceCloningConsent(
        IReadOnlyList<string> requiredModelIds,
        BundledModelManifestRegistry manifestRegistry) =>
        requiredModelIds.Any(modelId =>
        {
            BundledModelManifestEntry? entry = manifestRegistry.Entries
                .FirstOrDefault(candidate => candidate.ModelId.Equals(modelId, StringComparison.OrdinalIgnoreCase));
            return entry is { RequiresUserConsent: true } or { VoiceCloning: true };
        });
}
