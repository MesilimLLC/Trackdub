using Trackdub.Contracts;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.Migraphx;
using Trackdub.Inference.Runtime.ModelManifest;
using System.Collections.Concurrent;

namespace Trackdub.Inference.Runtime.Planning;

public sealed class RuntimePlanner : IRuntimePlanner
{
    private readonly IHardwareProfileProvider hardwareProfileProvider;
    private readonly IExecutionProviderDiscovery executionProviderDiscovery;
    private readonly RuntimePlannerRankingStrategy rankingStrategy;
    private readonly RuntimePlannerCacheIndexBuilder cacheIndexBuilder;
    private readonly RuntimePlanFactory planFactory;
    private readonly IReadOnlyDictionary<RuntimeStage, StageRuntimeRequirements> stageRequirements;
    private readonly IPipelineDeviceExclusionProvider? deviceExclusionProvider;
    private readonly IDeviceEnumerator? deviceEnumerator;

    // --- Hardware / provider cache (process-lifetime; no TTL) ---
    private HardwareProfile? _cachedHardwareProfile;
    private IReadOnlyList<ExecutionProviderAvailability>? _cachedProviderAvailabilities;
    private readonly SemaphoreSlim _hardwareCacheLock = new(1, 1);

    // --- Plan result cache ---
    private readonly ConcurrentDictionary<PlanCacheKey, StageRuntimePlan> _planCache = new();

    public RuntimePlanner(
        BundledModelManifestRegistry manifestRegistry,
        IHardwareProfileProvider hardwareProfileProvider,
        IExecutionProviderDiscovery executionProviderDiscovery,
        IExecutionProviderSmokeTester executionProviderSmokeTester,
        IModelCacheInventory modelCacheInventory,
        IReadOnlyDictionary<RuntimeStage, StageRuntimeRequirements>? stageRequirements = null,
        IPipelineDeviceExclusionProvider? deviceExclusionProvider = null,
        IDeviceEnumerator? deviceEnumerator = null,
        ISmokeVerdictStore? smokeVerdictStore = null)
    {
        ArgumentNullException.ThrowIfNull(manifestRegistry);
        this.hardwareProfileProvider = hardwareProfileProvider ?? throw new ArgumentNullException(nameof(hardwareProfileProvider));
        this.executionProviderDiscovery = executionProviderDiscovery ?? throw new ArgumentNullException(nameof(executionProviderDiscovery));
        ArgumentNullException.ThrowIfNull(executionProviderSmokeTester);
        ArgumentNullException.ThrowIfNull(modelCacheInventory);
        rankingStrategy = new RuntimePlannerRankingStrategy(manifestRegistry);
        cacheIndexBuilder = new RuntimePlannerCacheIndexBuilder(modelCacheInventory);
        planFactory = new RuntimePlanFactory(executionProviderSmokeTester, smokeVerdictStore);
        this.stageRequirements = stageRequirements ?? StageRuntimeRequirementsCatalog.All;
        this.deviceExclusionProvider = deviceExclusionProvider;
        this.deviceEnumerator = deviceEnumerator;
    }

    public async Task<StageRuntimePlan> PlanAsync(
        StageRuntimePlanningRequest request,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        // Cold-start attribution: planning runs per stage (often with plan-cache hits) and is
        // a named phase of the pipeline-start overhead this run evidence exists to explain.
        using IDisposable planPhase = BenchmarkPhaseCapture.Start("runtime-plan");

        // Resolve the effective device exclusion set: explicit request parameter takes
        // precedence, otherwise fall back to the provider's current run exclusions.
        DeviceExclusionSet? effectiveExclusions = request.DeviceExclusions
            ?? deviceExclusionProvider?.CurrentExclusions;

        if (!stageRequirements.TryGetValue(request.Stage, out StageRuntimeRequirements? requirements))
        {
            throw new InvalidOperationException($"Runtime stage '{request.Stage}' is not configured for runtime planning.");
        }

        // Load hardware profile and base provider availabilities from cache (or populate on first call).
        (HardwareProfile hardwareProfile, IReadOnlyList<ExecutionProviderAvailability> baseProviderAvailabilities) =
            await GetOrLoadHardwareAsync(cancellationToken).ConfigureAwait(false);

        // Fetch the device list whenever an enumerator is available (internally cached by
        // IDeviceEnumerator) so plans carry DeviceIndex even when no exclusions exist.
        // Preflight (EnsureModelsAvailableAsync) plans before BeginRun, when CurrentExclusions
        // is still null; without the device list that plan is cached with DeviceIndex = null
        // and the run's empty-exclusion lookup hits the same key, leaving the
        // DeviceFallbackSessionCreator catch filter unreachable. Exclusion filtering and the
        // snapshot below stay gated on effectiveExclusions.
        IReadOnlyList<DeviceEntry>? devices = null;
        if (deviceEnumerator is not null)
        {
            devices = await deviceEnumerator.GetDevicesAsync(cancellationToken).ConfigureAwait(false);
        }

        // Apply device exclusions freshly each call against the cached base provider list.
        IReadOnlyList<ExecutionProviderAvailability> providerAvailabilities =
            effectiveExclusions is not null && devices is not null
                ? ApplyDeviceExclusions(baseProviderAvailabilities, devices, effectiveExclusions)
                : baseProviderAvailabilities;

        // Build the exclusion snapshot for use in the plan cache key.
        string exclusionSnapshot = BuildExclusionSnapshot(effectiveExclusions, devices);

        // Check the plan cache before performing the expensive ranking/resolution work.
        PlanCacheKey cacheKey = new(
            Stage: request.Stage,
            ModelAlias: request.NormalizedPreferredModelAlias, RequireModelAlias: request.RequirePreferredModelAlias,
            VariantAlias: request.NormalizedPreferredModelVariantAlias,
            PreferredEp: request.PreferredExecutionProvider,
            RequireEp: request.RequirePreferredExecutionProvider,
            SourceLanguage: request.SourceLanguage,
            TargetLanguage: request.TargetLanguage,
            ModelTier: request.PreferredModelTier,
            ExclusionSetSnapshot: exclusionSnapshot,
            NvidiaGpuArchitecture: hardwareProfile.NvidiaGpuArchitecture,
            SkipProviderSmokeTest: request.SkipProviderSmokeTest);

        if (_planCache.TryGetValue(cacheKey, out StageRuntimePlan? cachedPlan))
        {
            return StampRequirePreferred(cachedPlan, request.RequirePreferredExecutionProvider);
        }

        IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> cacheIndex = await cacheIndexBuilder.BuildAsync(cancellationToken).ConfigureAwait(false);
        bool preferMigraphxOnAmdGpu = MigraphxProviderOrdering.ShouldPreferMigraphxOnAmdGpu(hardwareProfile);

        // File existence cache scoped to this planning call: avoids redundant File.Exists calls
        // across the two TryResolvePlanFromEntriesAsync passes, but stays fresh across calls.
        var fileExistenceCache = new ConcurrentDictionary<string, bool>(StringComparer.OrdinalIgnoreCase);

        RankedManifestEntry[] rankedEntries = rankingStrategy.RankEntries(request, requirements);
        if (rankedEntries.Length == 0)
        {
            return StampRequirePreferred(
                planFactory.CreateBlockedPlan(
                    request.Stage,
                    new RuntimePlanFallback(
                        RuntimePlanFallbackCode.NoCompatibleVariant,
                        $"No {requirements.RequiredTask.ToManifestValue()} model is registered in the bundled model manifest.")),
                request.RequirePreferredExecutionProvider);
        }

        if (request.RequirePreferredModelAlias &&
            request.NormalizedPreferredModelAlias is null)
        {
            return StampRequirePreferred(
                planFactory.CreateBlockedPlan(
                    request.Stage,
                    new RuntimePlanFallback(
                        RuntimePlanFallbackCode.NoCompatibleVariant,
                        "Model override requires a model alias, but no alias was provided.")),
                request.RequirePreferredExecutionProvider);
        }

        if (request.RequirePreferredModelAlias)
        {
            rankedEntries = RuntimePlannerRankingStrategy.FilterPreferredModelAliasEntries(request, rankedEntries);
            if (rankedEntries.Length == 0)
            {
                return StampRequirePreferred(
                    planFactory.CreateBlockedPlan(
                        request.Stage,
                        new RuntimePlanFallback(
                            RuntimePlanFallbackCode.NoCompatibleVariant,
                            $"Model override requires model alias '{request.NormalizedPreferredModelAlias}', but no {requirements.RequiredTask.ToManifestValue()} model with that alias is registered.")),
                    request.RequirePreferredExecutionProvider);
            }
        }

        RankedManifestEntry[] planEntries = RuntimePlannerRankingStrategy.FilterTopRankedEntriesIfRequired(
            requirements,
            rankedEntries);

        StageRuntimePlan? resolvedPlan = await TryResolvePlanFromEntriesAsync(
            request,
            planEntries,
            requirements,
            hardwareProfile,
            providerAvailabilities,
            cacheIndex,
            fileExistenceCache,
            preferMigraphxOnAmdGpu,
            devices,
            effectiveExclusions,
            cancellationToken).ConfigureAwait(false);
        if (resolvedPlan is not null)
        {
            resolvedPlan = StampRequirePreferred(resolvedPlan, request.RequirePreferredExecutionProvider);
            TryCachePlan(cacheKey, resolvedPlan);
            return resolvedPlan;
        }

        if (request.NormalizedPreferredModelVariantAlias is string preferredVariantAlias)
        {
            StageRuntimePlanningRequest requestWithoutPreferredVariant = request with
            {
                PreferredModelVariantAlias = null
            };

            // Build a new cache key for the fallback request (no variant alias).
            PlanCacheKey fallbackCacheKey = cacheKey with { VariantAlias = null };

            if (_planCache.TryGetValue(fallbackCacheKey, out StageRuntimePlan? cachedFallbackPlan))
            {
                return StampRequirePreferred(
                    AppendPreferredVariantFallbackWarning(cachedFallbackPlan, preferredVariantAlias),
                    request.RequirePreferredExecutionProvider);
            }

            StageRuntimePlan? fallbackPlan = await TryResolvePlanFromEntriesAsync(
                requestWithoutPreferredVariant,
                planEntries,
                requirements,
                hardwareProfile,
                providerAvailabilities,
                cacheIndex,
                fileExistenceCache,
                preferMigraphxOnAmdGpu,
                devices,
                effectiveExclusions,
                cancellationToken).ConfigureAwait(false);
            if (fallbackPlan is not null)
            {
                fallbackPlan = StampRequirePreferred(fallbackPlan, request.RequirePreferredExecutionProvider);
                TryCachePlan(fallbackCacheKey, fallbackPlan);
                return AppendPreferredVariantFallbackWarning(fallbackPlan, preferredVariantAlias);
            }

            return StampRequirePreferred(
                planFactory.CreateBlockedPlan(
                    request.Stage,
                    new RuntimePlanFallback(
                        RuntimePlanFallbackCode.NoCompatibleVariant,
                        $"Selected optimized variant '{preferredVariantAlias}' is not available for the requested model and provider. Re-optimize the model or clear the variant selection.")),
                request.RequirePreferredExecutionProvider);
        }

        return StampRequirePreferred(
            planFactory.CreateBlockedPlan(
                request.Stage,
                new RuntimePlanFallback(
                    RuntimePlanFallbackCode.NoCompatibleVariant,
                    request.RequirePreferredExecutionProvider && request.PreferredExecutionProvider is ExecutionProviderKind requiredProvider
                        ? $"No compatible {requirements.RequiredTask.ToManifestValue()} variant could be planned for required execution provider {requiredProvider}."
                        : $"No compatible {requirements.RequiredTask.ToManifestValue()} variant could be planned for the current provider policy.")),
            request.RequirePreferredExecutionProvider);
    }

    private static StageRuntimePlan StampRequirePreferred(StageRuntimePlan plan, bool requirePreferredExecutionProvider) =>
        plan.RequirePreferredExecutionProvider == requirePreferredExecutionProvider
            ? plan
            : plan with { RequirePreferredExecutionProvider = requirePreferredExecutionProvider };

    /// <inheritdoc />
    public void InvalidatePlanCache()
    {
        _planCache.Clear();
        _cachedHardwareProfile = null;
        _cachedProviderAvailabilities = null;
        cacheIndexBuilder.Invalidate();
    }

    /// <summary>
    /// Returns cached hardware profile and base provider availabilities, loading them on the first call.
    /// Thread-safe: only one concurrent load is allowed; subsequent callers wait for the first to complete.
    /// </summary>
    private async Task<(HardwareProfile HardwareProfile, IReadOnlyList<ExecutionProviderAvailability> ProviderAvailabilities)>
        GetOrLoadHardwareAsync(CancellationToken cancellationToken)
    {
        // Fast path: both values are already cached.
        if (_cachedHardwareProfile is HardwareProfile cachedProfile &&
            _cachedProviderAvailabilities is IReadOnlyList<ExecutionProviderAvailability> cachedAvailabilities)
        {
            return (cachedProfile, cachedAvailabilities);
        }

        await _hardwareCacheLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            // Double-check inside the lock in case another thread just populated the cache.
            if (_cachedHardwareProfile is HardwareProfile lockedProfile &&
                _cachedProviderAvailabilities is IReadOnlyList<ExecutionProviderAvailability> lockedAvailabilities)
            {
                return (lockedProfile, lockedAvailabilities);
            }

            HardwareProfile profile = await hardwareProfileProvider.GetCurrentAsync(cancellationToken).ConfigureAwait(false);
            IReadOnlyList<ExecutionProviderAvailability> availabilities = await executionProviderDiscovery.DiscoverAsync(
                profile,
                cancellationToken).ConfigureAwait(false);

            _cachedHardwareProfile = profile;
            _cachedProviderAvailabilities = availabilities;

            return (profile, availabilities);
        }
        finally
        {
            _hardwareCacheLock.Release();
        }
    }

    /// <summary>
    /// Builds a stable, sorted string snapshot of the currently excluded device indices,
    /// suitable for use as a plan cache key component.
    /// Returns an empty string when there are no exclusions or no device list is available.
    /// </summary>
    private static string BuildExclusionSnapshot(
        DeviceExclusionSet? exclusions,
        IReadOnlyList<DeviceEntry>? devices)
    {
        if (exclusions is null || devices is null)
        {
            return string.Empty;
        }

        // Collect indices of excluded devices in ascending order for a stable key.
        var excludedIndices = new List<int>();
        foreach (DeviceEntry device in devices)
        {
            if (exclusions.IsExcluded(device.DeviceIndex))
            {
                excludedIndices.Add(device.DeviceIndex);
            }
        }

        if (excludedIndices.Count == 0)
        {
            return string.Empty;
        }

        excludedIndices.Sort();
        return string.Join(',', excludedIndices);
    }

    /// <summary>
    /// Stores a plan in the cache only if it is in a runnable (Ready or Verified) status.
    /// DownloadRequired and Blocked plans are intentionally not cached.
    /// </summary>
    private void TryCachePlan(PlanCacheKey key, StageRuntimePlan plan)
    {
        if (plan.Status.IsRunnable())
        {
            _planCache.TryAdd(key, plan);
        }
    }

    private async Task<StageRuntimePlan?> TryResolvePlanFromEntriesAsync(
        StageRuntimePlanningRequest request,
        RankedManifestEntry[] planEntries,
        StageRuntimeRequirements requirements,
        HardwareProfile hardwareProfile,
        IReadOnlyList<ExecutionProviderAvailability> providerAvailabilities,
        IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> cacheIndex,
        ConcurrentDictionary<string, bool> fileExistenceCache,
        bool preferMigraphxOnAmdGpu,
        IReadOnlyList<DeviceEntry>? devices,
        DeviceExclusionSet? effectiveExclusions,
        CancellationToken cancellationToken)
    {
        foreach (RankedManifestEntry candidate in planEntries)
        {
            StageRuntimePlan? readyPlan = await planFactory.TryCreateReadyPlanAsync(
                request.Stage,
                requirements,
                candidate,
                hardwareProfile,
                providerAvailabilities,
                cacheIndex,
                fileExistenceCache,
                request.PreferredExecutionProvider,
                request.RequirePreferredExecutionProvider,
                preferMigraphxOnAmdGpu,
                request.NormalizedPreferredModelVariantAlias,
                request.SkipProviderSmokeTest,
                cancellationToken,
                devices,
                effectiveExclusions).ConfigureAwait(false);

            if (readyPlan is not null)
            {
                return readyPlan;
            }
        }

        foreach (RankedManifestEntry candidate in planEntries)
        {
            StageRuntimePlan? downloadPlan = planFactory.TryCreateDownloadRequiredPlan(
                request.Stage,
                requirements,
                candidate,
                hardwareProfile,
                providerAvailabilities,
                cacheIndex,
                fileExistenceCache,
                request.PreferredExecutionProvider,
                request.RequirePreferredExecutionProvider,
                preferMigraphxOnAmdGpu,
                request.NormalizedPreferredModelVariantAlias,
                devices,
                effectiveExclusions);

            if (downloadPlan is not null)
            {
                return downloadPlan;
            }
        }

        return null;
    }

    private static StageRuntimePlan AppendPreferredVariantFallbackWarning(
        StageRuntimePlan plan,
        string unavailableVariantAlias) =>
        plan with
        {
            Warnings =
            [
                ..plan.Warnings,
                new RuntimePlanWarning(
                    RuntimePlanWarningCode.PreferredOptimizedVariantUnavailable,
                    $"Selected optimized variant '{unavailableVariantAlias}' is not available for the requested model and provider. Using the base model instead.")
            ]
        };

    internal static IReadOnlyList<ExecutionProviderAvailability> ApplyDeviceExclusions(
        IReadOnlyList<ExecutionProviderAvailability> availabilities,
        IReadOnlyList<DeviceEntry> devices,
        DeviceExclusionSet exclusions)
    {
        // Build a HashSet of excluded device indices for O(1) lookups.
        var excludedIndices = new HashSet<int>();
        foreach (DeviceEntry device in devices)
        {
            if (exclusions.IsExcluded(device.DeviceIndex))
            {
                excludedIndices.Add(device.DeviceIndex);
            }
        }

        // Pre-group devices by provider so we can check eligibility without scanning all devices per provider.
        // Store concrete List values so Adds do not need a brittle abstract-to-concrete cast.
        var devicesByProvider = new Dictionary<ExecutionProviderKind, List<DeviceEntry>>();
        foreach (DeviceEntry device in devices)
        {
            foreach (ExecutionProviderKind provider in device.SupportedProviders)
            {
                if (!devicesByProvider.TryGetValue(provider, out var list))
                {
                    list = [];
                    devicesByProvider[provider] = list;
                }

                list.Add(device);
            }
        }

        var filtered = new List<ExecutionProviderAvailability>(availabilities.Count);

        foreach (ExecutionProviderAvailability availability in availabilities)
        {
            bool hasEligibleDevice = devicesByProvider.TryGetValue(availability.Provider, out var providerDevices)
                && providerDevices.Any(device => !excludedIndices.Contains(device.DeviceIndex));

            // The Linux OpenVINO catalog append chooses a device by type (NPU, GPU, CPU), not by
            // the planner's device index, so it could still land on an excluded device. Suppress
            // the route while any OpenVINO device is excluded.
            if (hasEligibleDevice
                && availability.Provider is ExecutionProviderKind.OpenVino
                && providerDevices!.Any(device => excludedIndices.Contains(device.DeviceIndex)))
            {
                filtered.Add(new ExecutionProviderAvailability(
                    availability.Provider,
                    false,
                    "OpenVino cannot be pinned to a device; unavailable while one of its devices is excluded for this run."));
                continue;
            }

            if (hasEligibleDevice)
            {
                filtered.Add(availability);
                continue;
            }

            filtered.Add(new ExecutionProviderAvailability(
                availability.Provider,
                false,
                $"{availability.Provider} has no eligible devices after applying current run exclusions."));
        }

        return filtered;
    }
}

/// <summary>
/// Immutable cache key for <see cref="StageRuntimePlan"/> results.
/// <para>
/// The <see cref="ExclusionSetSnapshot"/> encodes the active device exclusion state as
/// a sorted comma-separated string of excluded <see cref="DeviceEntry.DeviceIndex"/> values.
/// This is stable within a single process lifetime because device indices are assigned
/// by <see cref="IDeviceEnumerator"/> and do not change during a run.
/// </para>
/// <para>
/// If the hardware topology changes (GPU hotplug, driver reinstall, enumeration reordering),
/// <see cref="InvalidatePlanCache"/> must be called to prevent stale cache hits from matching
/// entries computed under a different device ordering or GPU architecture.
/// </para>
/// </summary>
// RequireModelAlias is part of the key: a soft-alias request may fall through to a different
// model, and that plan must never be served to a request that requires the alias.
internal sealed record PlanCacheKey(
    RuntimeStage Stage,
    string? ModelAlias,
    bool RequireModelAlias,
    string? VariantAlias,
    ExecutionProviderKind? PreferredEp,
    bool RequireEp,
    string? SourceLanguage,
    string? TargetLanguage,
    string? ModelTier,
    string ExclusionSetSnapshot,
    NvidiaGpuArchitectureBucket NvidiaGpuArchitecture,
    bool SkipProviderSmokeTest);
