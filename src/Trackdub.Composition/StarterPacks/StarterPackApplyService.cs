using Trackdub.Application.Runtime;
using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.StarterPacks;
using Trackdub.Domain;
using Trackdub.Domain.StageRuns;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.StarterPacks;

public sealed class StarterPackApplyService(
    IStarterPackCatalog catalog,
    IStudioSettingsService studioSettingsService,
    IModelInventoryService inventoryService,
    BundledModelManifestRegistry manifestRegistry,
    IHardwareProfilerService hardwareProfilerService,
    IConsentService consentService,
    IStarterPackCompatibilityService compatibilityService,
    ICloudCredentialReadiness cloudCredentialReadiness) : IStarterPackApplyService
{
    private readonly IStarterPackCatalog catalog = catalog ?? throw new ArgumentNullException(nameof(catalog));
    private readonly IStudioSettingsService studioSettingsService =
        studioSettingsService ?? throw new ArgumentNullException(nameof(studioSettingsService));
    private readonly IModelInventoryService inventoryService =
        inventoryService ?? throw new ArgumentNullException(nameof(inventoryService));
    private readonly BundledModelManifestRegistry manifestRegistry =
        manifestRegistry ?? throw new ArgumentNullException(nameof(manifestRegistry));
    private readonly IHardwareProfilerService hardwareProfilerService =
        hardwareProfilerService ?? throw new ArgumentNullException(nameof(hardwareProfilerService));
    private readonly IConsentService consentService =
        consentService ?? throw new ArgumentNullException(nameof(consentService));
    private readonly IStarterPackCompatibilityService compatibilityService =
        compatibilityService ?? throw new ArgumentNullException(nameof(compatibilityService));
    private readonly ICloudCredentialReadiness cloudCredentialReadiness =
        cloudCredentialReadiness ?? throw new ArgumentNullException(nameof(cloudCredentialReadiness));

    public async Task<StarterPackApplyResult> ApplyAsync(
        string packId,
        string profileId,
        StarterPackHardwareProfile? hardwareProfile = null,
        bool acceptVoiceCloningConsent = false,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(packId);
        ArgumentException.ThrowIfNullOrWhiteSpace(profileId);

        StarterPackDefinition pack = await catalog.GetAsync(packId, cancellationToken).ConfigureAwait(false);
        StarterPackProfileDefinition profile = StarterPackResolver.ResolveProfile(pack, profileId);
        StarterPackApplySettings applySettings = ResolveApplySettings(pack, profile.Id);

        string? validationError = await ValidateApplyAsync(
            pack,
            profile,
            applySettings,
            acceptVoiceCloningConsent,
            cancellationToken).ConfigureAwait(false);

        StarterPackHardwareProfile resolvedHardwareProfile = hardwareProfile ??
            (pack.PackKind == StarterPackKind.Cloud
                ? StarterPackHardwareProfile.BalancedGpu
                : await ResolveHardwareProfileAsync(cancellationToken).ConfigureAwait(false));

        if (validationError is not null)
        {
            return new StarterPackApplyResult(packId, profileId, StarterPackStageMapping.ToHardwareProfileKey(resolvedHardwareProfile), false, validationError);
        }

        StarterPackCompatibilityReport compatibilityReport = pack.PackKind == StarterPackKind.Cloud
            ? new StarterPackCompatibilityReport(
                packId,
                profileId,
                StarterPackStageMapping.ToHardwareProfileKey(resolvedHardwareProfile),
                [],
                AllStagesRunnable: true,
                AnyFallbackApplied: false)
            : await compatibilityService
                .EvaluateAsync(packId, profileId, resolvedHardwareProfile, cancellationToken)
                .ConfigureAwait(false);
        if (!compatibilityReport.AllStagesRunnable)
        {
            return new StarterPackApplyResult(
                packId,
                profileId,
                compatibilityReport.HardwareProfileKey,
                false,
                "One or more required stages are not runnable on this machine.");
        }

        IReadOnlyList<StageFallbackRecord> fallbacks = compatibilityReport.Stages
            .Where(stage => stage.FallbackApplied)
            .Select(stage => new StageFallbackRecord(
                stage.Stage,
                stage.Alias,
                stage.RequestedVariant,
                stage.RequestedExecutionProvider,
                stage.ResolvedVariant,
                stage.ResolvedExecutionProvider,
                stage.FallbackReason))
            .ToList();

        StudioSettings current = await studioSettingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
        StudioSettings updated = BuildUpdatedSettings(
            current,
            pack,
            profile,
            applySettings,
            resolvedHardwareProfile,
            compatibilityReport);

        await studioSettingsService.SaveAsync(updated, cancellationToken).ConfigureAwait(false);

        if (RequiresVoiceCloningConsent(pack, profile) && acceptVoiceCloningConsent)
        {
            consentService.GrantVoiceCloningConsent();
        }

        return new StarterPackApplyResult(
            packId,
            profileId,
            StarterPackStageMapping.ToHardwareProfileKey(resolvedHardwareProfile),
            true,
            Fallbacks: fallbacks.Count > 0 ? fallbacks : null);
    }

    public static StudioSettings BuildUpdatedSettings(
        StudioSettings current,
        StarterPackDefinition pack,
        StarterPackProfileDefinition profile,
        StarterPackApplySettings applySettings,
        StarterPackHardwareProfile hardwareProfile,
        StarterPackCompatibilityReport? compatibilityReport = null)
    {
        string hardwareKey = StarterPackStageMapping.ToHardwareProfileKey(hardwareProfile);
        IReadOnlyList<StarterPackModelDefinition> activeModels =
            StarterPackResolver.GetActiveModelsForProfile(pack, profile.Id);

        var stageAliases = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (current.StageModelAliases is not null)
        {
            foreach ((string key, string value) in current.StageModelAliases)
            {
                if (!string.IsNullOrWhiteSpace(key) && !string.IsNullOrWhiteSpace(value))
                {
                    stageAliases[key] = value.Trim();
                }
            }
        }

        var variantOverrides = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (current.ModelVariantOverrides is not null)
        {
            foreach ((string key, string value) in current.ModelVariantOverrides)
            {
                if (!string.IsNullOrWhiteSpace(key) && !string.IsNullOrWhiteSpace(value))
                {
                    variantOverrides[key] = value.Trim();
                }
            }
        }

        var hardwareOverrides = new Dictionary<string, ExecutionProviderKind>(StringComparer.OrdinalIgnoreCase);
        if (current.HardwareOverrides is not null)
        {
            foreach ((string key, ExecutionProviderKind value) in current.HardwareOverrides)
            {
                if (!string.IsNullOrWhiteSpace(key))
                {
                    hardwareOverrides[key] = value;
                }
            }
        }

        foreach (StarterPackModelDefinition model in activeModels)
        {
            string stageName = StarterPackStageMapping.ToStageName(model.Stage);
            stageAliases[stageName] = model.Alias;

            StageCompatibilityEntry? compatibilityEntry = compatibilityReport?.Stages
                .FirstOrDefault(stage => string.Equals(stage.Stage, stageName, StringComparison.OrdinalIgnoreCase));

            string variant;
            string packExecutionProvider;
            if (compatibilityEntry is not null)
            {
                variant = compatibilityEntry.ResolvedVariant;
                // For fallback stages the requested provider was rejected as unusable and a
                // different one was selected; pin the resolved provider so subsequent runs
                // request what actually works instead of re-requesting the unavailable one.
                packExecutionProvider = compatibilityEntry.FallbackApplied
                    ? compatibilityEntry.ResolvedExecutionProvider
                    : compatibilityEntry.RequestedExecutionProvider;
            }
            else if (model.RuntimeDefaults.TryGetValue(hardwareKey, out StarterPackRuntimeDefaults? runtimeDefaults))
            {
                variant = runtimeDefaults.Variant;
                packExecutionProvider = runtimeDefaults.ExecutionProvider;
            }
            else
            {
                continue;
            }

            string variantKey = ModelVariantOverrideKeys.Build(stageName, model.Alias);
            variantOverrides[variantKey] = variant;
            variantOverrides[stageName] = variant;

            if (!TryResolveHardwareOverrideKey(stageName, applySettings.AsrModelOverride, out string hardwareOverrideKey))
            {
                continue;
            }

            // auto clears leftover pins so the planner owns GPU EP at run time.
            if (TryResolveExecutionProvider(packExecutionProvider, out ExecutionProviderKind provider))
            {
                hardwareOverrides[hardwareOverrideKey] = provider;
            }
            else
            {
                hardwareOverrides.Remove(hardwareOverrideKey);
            }
        }

        if (pack.PackKind == StarterPackKind.Cloud)
        {
            hardwareOverrides.Clear();
            ClearCloudStageOverrides(stageAliases, variantOverrides);
            ApplyCloudStageAliases(stageAliases, applySettings);
        }
        else if (pack.Apply is not null)
        {
            if (pack.Apply.StageAliases is not null)
            {
                foreach ((string key, string value) in pack.Apply.StageAliases)
                {
                    if (!string.IsNullOrWhiteSpace(key) && !string.IsNullOrWhiteSpace(value))
                    {
                        stageAliases[key.Trim()] = value.Trim();
                    }
                }
            }

            if (pack.PackKind == StarterPackKind.Hybrid)
            {
                ApplyCloudStageAliases(stageAliases, applySettings);
            }
        }

        string tierPreference = pack.Apply?.TierPreference ?? pack.TierPreference;

        return current with
        {
            ModelTierPreference = tierPreference,
            AppliedStarterPackId = pack.Id,
            AppliedStarterPackProfileId = profile.Id,
            AsrModelOverride = applySettings.AsrModelOverride,
            TranslationModelOverride = applySettings.TranslationModelOverride,
            TtsModelOverride = applySettings.TtsModelOverride,
            StageModelAliases = stageAliases,
            ModelVariantOverrides = variantOverrides,
            HardwareOverrides = hardwareOverrides
        };
    }

    private async Task<string?> ValidateApplyAsync(
        StarterPackDefinition pack,
        StarterPackProfileDefinition profile,
        StarterPackApplySettings applySettings,
        bool acceptVoiceCloningConsent,
        CancellationToken cancellationToken)
    {
        StarterPackCloudDefaults? cloudDefaults = pack.CloudDefaults
            ?? (pack.Apply is null ? null : StarterPackDataDrivenApplyMapper.ToCloudDefaults(pack.Apply));
        if (cloudDefaults is not null)
        {
            CloudCredentialReadinessReport readiness = await cloudCredentialReadiness
                .EvaluateAsync(cloudDefaults, cancellationToken)
                .ConfigureAwait(false);
            if (!readiness.IsReady)
            {
                return readiness.BlockedReason;
            }
        }

        if (pack.PackKind == StarterPackKind.Cloud && pack.Models.Count == 0)
        {
            return null;
        }

        IReadOnlyList<string> requiredModelIds = StarterPackResolver.GetRequiredModelIds(pack, profile.Id);
        IReadOnlyList<ModelInventoryEntry> inventory = await inventoryService
            .GetAllAsync(cancellationToken)
            .ConfigureAwait(false);

        foreach (string modelId in requiredModelIds)
        {
            ModelInventoryEntry? entry = inventory
                .FirstOrDefault(candidate => string.Equals(candidate.ModelId, modelId, StringComparison.OrdinalIgnoreCase));
            if (entry is null ||
                entry.State is not (ModelCacheState.Ready or ModelCacheState.Installed))
            {
                int installed = requiredModelIds.Count(id =>
                {
                    ModelInventoryEntry? candidate = inventory.FirstOrDefault(e =>
                        string.Equals(e.ModelId, id, StringComparison.OrdinalIgnoreCase));
                    return candidate?.State is ModelCacheState.Ready or ModelCacheState.Installed;
                });

                return $"Download pack first ({installed}/{requiredModelIds.Count} installed).";
            }

            BundledModelManifestEntry? manifestEntry = manifestRegistry.Entries
                .FirstOrDefault(candidate => candidate.ModelId.Equals(modelId, StringComparison.OrdinalIgnoreCase));
            if (manifestEntry is null)
            {
                return $"Required model '{modelId}' is not in the bundled manifest.";
            }

            if (!manifestEntry.CommercialClearance)
            {
                string alias = manifestEntry.Aliases.FirstOrDefault() ?? modelId;
                return $"License review needed: {alias} is not commercial-use verified.";
            }
        }

        if (RequiresVoiceCloningConsent(pack, profile))
        {
            bool consentGranted = consentService.IsVoiceCloningConsentGranted || acceptVoiceCloningConsent;
            if (!consentGranted)
            {
                return "Voice cloning consent is required before applying this pack.";
            }
        }

        return null;
    }

    private static StarterPackApplySettings ResolveApplySettings(StarterPackDefinition pack, string profileId)
    {
        if (pack.PackKind == StarterPackKind.Cloud)
        {
            StarterPackCloudDefaults? cloudDefaults = pack.CloudDefaults
                ?? (pack.Apply is null ? null : StarterPackDataDrivenApplyMapper.ToCloudDefaults(pack.Apply));
            if (cloudDefaults is null)
            {
                throw new InvalidOperationException($"Cloud pack '{pack.Id}' is missing cloud_defaults or apply.cloud_stages.");
            }

            return StarterPackCloudDefaultsMapper.ToApplySettings(cloudDefaults);
        }

        if (pack.Apply is not null)
        {
            return StarterPackDataDrivenApplyMapper.ToApplySettings(pack.Apply);
        }

        if (StarterPackShippingIds.IsShippingPack(pack.Id))
        {
            return StarterPackApplyContract.Resolve(pack.Id, profileId);
        }

        throw new InvalidOperationException($"Starter pack '{pack.Id}' has no apply contract or apply block.");
    }

    private static readonly string[] CloudStageNames = ["asr", "tts", "translation"];

    private static void ClearCloudStageOverrides(
        Dictionary<string, string> stageAliases,
        Dictionary<string, string> variantOverrides)
    {
        foreach (string stage in CloudStageNames)
        {
            stageAliases.Remove(stage);
        }

        foreach (string key in variantOverrides.Keys.ToList())
        {
            if (CloudStageNames.Contains(key, StringComparer.OrdinalIgnoreCase))
            {
                variantOverrides.Remove(key);
                continue;
            }

            if (ModelVariantOverrideKeys.TryParse(key, out string stageKey, out _) &&
                CloudStageNames.Contains(stageKey, StringComparer.OrdinalIgnoreCase))
            {
                variantOverrides.Remove(key);
            }
        }
    }

    private static void ApplyCloudStageAliases(
        Dictionary<string, string> stageAliases,
        StarterPackApplySettings applySettings)
    {
        string? asrAlias = AsrModelOverrideSettings.ResolveModelAlias(applySettings.AsrModelOverride);
        if (!string.IsNullOrWhiteSpace(asrAlias))
        {
            stageAliases["asr"] = asrAlias;
        }

        string? ttsAlias = TtsModelOverrideSettings.ResolveModelAlias(applySettings.TtsModelOverride);
        if (!string.IsNullOrWhiteSpace(ttsAlias))
        {
            stageAliases["tts"] = ttsAlias;
        }

        string? translationAlias = TranslationModelOverrideSettings.ResolveModelAlias(applySettings.TranslationModelOverride);
        if (!string.IsNullOrWhiteSpace(translationAlias))
        {
            stageAliases["translation"] = translationAlias;
        }
    }

    private bool RequiresVoiceCloningConsent(StarterPackDefinition pack, StarterPackProfileDefinition profile)
    {
        foreach (string modelId in StarterPackResolver.GetRequiredModelIds(pack, profile.Id))
        {
            BundledModelManifestEntry? manifestEntry = manifestRegistry.Entries
                .FirstOrDefault(candidate => candidate.ModelId.Equals(modelId, StringComparison.OrdinalIgnoreCase));
            if (manifestEntry is { RequiresUserConsent: true } or { VoiceCloning: true })
            {
                return true;
            }
        }

        return false;
    }

    private async Task<StarterPackHardwareProfile> ResolveHardwareProfileAsync(CancellationToken cancellationToken) =>
        await StarterPackHardwareResolver
            .ResolveHardwareProfileAsync(hardwareProfilerService, cancellationToken)
            .ConfigureAwait(false);

    private static bool TryResolveHardwareOverrideKey(
        string stageName,
        AsrModelOverride asrModelOverride,
        out string hardwareKey)
    {
        if (string.Equals(stageName, StageNames.Vad, StringComparison.OrdinalIgnoreCase))
        {
            hardwareKey = "Vad";
            return true;
        }

        return HardwareOverrideCatalog.TryResolvePipelineHardwareOverrideKey(stageName, asrModelOverride, out hardwareKey);
    }

    private static bool TryResolveExecutionProvider(string token, out ExecutionProviderKind provider)
    {
        provider = default;
        if (string.IsNullOrWhiteSpace(token) ||
            string.Equals(token, "auto", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        if (!ExecutionProviderTokens.TryParse(token, out ExecutionProviderKind parsed))
        {
            throw new InvalidOperationException($"Unknown execution provider token '{token}'.");
        }

        provider = ExecutionProviderTokens.ResolvePlatformPin(parsed);
        return true;
    }
}
