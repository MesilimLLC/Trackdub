using Trackdub.Contracts;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.Migraphx;
using Trackdub.Inference.Runtime.ModelManifest;
using System.Collections.Concurrent;

namespace Trackdub.Inference.Runtime.Planning;

public sealed partial class RuntimePlanner : IRuntimePlanner
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
        StageRuntimePlan plan = await PlanCoreAsync(request, cancellationToken).ConfigureAwait(false);
        return plan with { HasExplicitProviderPreference = request.PreferredExecutionProvider.HasValue };
    }

    private async Task<StageRuntimePlan> PlanCoreAsync(
        StageRuntimePlanningRequest request,
        CancellationToken cancellationToken)
    {
        // Cold-start attribution: planning runs per stage (often with plan-cache hits) and is
        // a named phase of the pipeline-start overhead this run evidence exists to explain.
        using IDisposable planPhase = BenchmarkPhaseCapture.Start("runtime-plan");

        if (!stageRequirements.TryGetValue(request.Stage, out StageRuntimeRequirements? requirements))
        {
            throw new InvalidOperationException($"Runtime stage '{request.Stage}' is not configured for runtime planning.");
        }

        PlanningSnapshot snapshot = await CaptureSnapshotAsync(request, cancellationToken).ConfigureAwait(false);

        // Check the plan cache before performing the expensive ranking/resolution work.
        if (_planCache.TryGetValue(snapshot.CacheKey, out StageRuntimePlan? cachedPlan))
        {
            return StampRequirePreferred(cachedPlan, request.RequirePreferredExecutionProvider);
        }

        IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> cacheIndex = await cacheIndexBuilder.BuildAsync(cancellationToken).ConfigureAwait(false);
        (RankedManifestEntry[] planEntries, StageRuntimePlan? blockedPlan) = SelectPlanEntries(request, requirements);
        if (blockedPlan is not null)
        {
            return blockedPlan;
        }

        var context = new PlanningContext(
            request,
            requirements,
            snapshot.HardwareProfile,
            snapshot.ProviderAvailabilities,
            cacheIndex,
            // File existence cache scoped to this planning call: avoids redundant File.Exists calls
            // across the two TryResolvePlanFromEntriesAsync passes, but stays fresh across calls.
            new ConcurrentDictionary<string, bool>(StringComparer.OrdinalIgnoreCase),
            MigraphxProviderOrdering.ShouldPreferMigraphxOnAmdGpu(snapshot.HardwareProfile),
            snapshot.Devices,
            snapshot.EffectiveExclusions,
            planEntries);

        StageRuntimePlan? resolvedPlan = await TryResolvePlanFromEntriesAsync(context, cancellationToken).ConfigureAwait(false);
        if (resolvedPlan is not null)
        {
            // A blocked preferred-variant plan is replaced by a base-variant fallback, which
            // ResolveWithoutPreferredVariantAsync caches under its own key; the preferred-variant
            // key must keep resolving freshly once that variant becomes available.
            bool preferredVariantResolved = resolvedPlan.Status != StageRuntimePlanStatus.Blocked;
            resolvedPlan = await RuntimePlannerPreferredVariantFallback.ResolveAsync(
                new PreferredVariantFallbackRequest
                {
                    ResolvedPlan = resolvedPlan,
                    PreferredVariantAlias = context.Request.NormalizedPreferredModelVariantAlias,
                    RequirePreferredExecutionProvider = request.RequirePreferredExecutionProvider,
                    ResolveFallbackAsync = (alias, token) => ResolveWithoutPreferredVariantAsync(
                        context, snapshot.CacheKey, alias, token)
                },
                cancellationToken).ConfigureAwait(false);
            if (preferredVariantResolved)
            {
                TryCachePlan(snapshot.CacheKey, resolvedPlan);
            }

            return resolvedPlan;
        }

        return request.NormalizedPreferredModelVariantAlias is string preferredVariantAlias
            ? await ResolveWithoutPreferredVariantAsync(context, snapshot.CacheKey, preferredVariantAlias, cancellationToken).ConfigureAwait(false)
            : BlockedNoCompatibleVariant(request, DescribeNoCompatibleVariant(request, requirements));
    }

    private async Task<PlanningSnapshot> CaptureSnapshotAsync(
        StageRuntimePlanningRequest request,
        CancellationToken cancellationToken)
    {
        // Resolve the effective device exclusion set: explicit request parameter takes
        // precedence, otherwise fall back to the provider's current run exclusions.
        DeviceExclusionSet? effectiveExclusions = request.DeviceExclusions
            ?? deviceExclusionProvider?.CurrentExclusions;

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

        PlanCacheKey cacheKey = new(
            Stage: request.Stage,
            ModelAlias: request.NormalizedPreferredModelAlias, RequireModelAlias: request.RequirePreferredModelAlias,
            VariantAlias: request.NormalizedPreferredModelVariantAlias,
            PreferredEp: request.PreferredExecutionProvider,
            RequireEp: request.RequirePreferredExecutionProvider,
            SourceLanguage: request.SourceLanguage,
            TargetLanguage: request.TargetLanguage,
            ModelTier: request.PreferredModelTier,
            PreferredEngineFamily: request.PreferredEngineFamily,
            ExclusionSetSnapshot: exclusionSnapshot,
            NvidiaGpuArchitecture: hardwareProfile.NvidiaGpuArchitecture,
            SkipProviderSmokeTest: request.SkipProviderSmokeTest);
        return new PlanningSnapshot(hardwareProfile, providerAvailabilities, devices, effectiveExclusions, cacheKey);
    }

    /// <summary>
    /// Ranks the manifest entries that may be planned for the request. A non-null blocked plan means
    /// no entry can be planned and the plan is final.
    /// </summary>
    private (RankedManifestEntry[] PlanEntries, StageRuntimePlan? BlockedPlan) SelectPlanEntries(
        StageRuntimePlanningRequest request,
        StageRuntimeRequirements requirements)
    {
        RankedManifestEntry[] rankedEntries = rankingStrategy.RankEntries(request, requirements);
        if (rankedEntries.Length == 0)
        {
            return ([], BlockedNoCompatibleVariant(
                request,
                $"No {requirements.RequiredTask.ToManifestValue()} model is registered in the bundled model manifest."));
        }

        if (request.RequirePreferredModelAlias && request.NormalizedPreferredModelAlias is null)
        {
            return ([], BlockedNoCompatibleVariant(
                request,
                "Model override requires a model alias, but no alias was provided."));
        }

        if (request.RequirePreferredModelAlias)
        {
            rankedEntries = RuntimePlannerRankingStrategy.FilterPreferredModelAliasEntries(request, rankedEntries);
            if (rankedEntries.Length == 0)
            {
                return ([], BlockedNoCompatibleVariant(
                    request,
                    $"Model override requires model alias '{request.NormalizedPreferredModelAlias}', but no {requirements.RequiredTask.ToManifestValue()} model with that alias is registered."));
            }
        }

        return (RuntimePlannerRankingStrategy.FilterTopRankedEntriesIfRequired(requirements, rankedEntries), null);
    }

    private async Task<StageRuntimePlan> ResolveWithoutPreferredVariantAsync(
        PlanningContext context,
        PlanCacheKey cacheKey,
        string preferredVariantAlias,
        CancellationToken cancellationToken)
    {
        StageRuntimePlanningRequest request = context.Request;

        // Build a new cache key for the fallback request (no variant alias).
        PlanCacheKey fallbackCacheKey = cacheKey with { VariantAlias = null };

        if (_planCache.TryGetValue(fallbackCacheKey, out StageRuntimePlan? cachedFallbackPlan))
        {
            return StampRequirePreferred(
                AppendPreferredVariantFallbackWarning(cachedFallbackPlan, preferredVariantAlias),
                request.RequirePreferredExecutionProvider);
        }

        StageRuntimePlan? fallbackPlan = await TryResolvePlanFromEntriesAsync(
            context with { Request = request with { PreferredModelVariantAlias = null } },
            cancellationToken).ConfigureAwait(false);
        if (fallbackPlan is null)
        {
            return BlockedNoCompatibleVariant(
                request,
                $"Selected optimized variant '{preferredVariantAlias}' is not available for the requested model and provider. Re-optimize the model or clear the variant selection.");
        }

        fallbackPlan = StampRequirePreferred(fallbackPlan, request.RequirePreferredExecutionProvider);
        TryCachePlan(fallbackCacheKey, fallbackPlan);
        return AppendPreferredVariantFallbackWarning(fallbackPlan, preferredVariantAlias);
    }

    private static string DescribeNoCompatibleVariant(
        StageRuntimePlanningRequest request,
        StageRuntimeRequirements requirements) =>
        request.RequirePreferredExecutionProvider && request.PreferredExecutionProvider is ExecutionProviderKind requiredProvider
            ? $"No compatible {requirements.RequiredTask.ToManifestValue()} variant could be planned for required execution provider {requiredProvider}."
            : $"No compatible {requirements.RequiredTask.ToManifestValue()} variant could be planned for the current provider policy.";

    private StageRuntimePlan BlockedNoCompatibleVariant(StageRuntimePlanningRequest request, string message) =>
        StampRequirePreferred(
            planFactory.CreateBlockedPlan(
                request.Stage,
                new RuntimePlanFallback(RuntimePlanFallbackCode.NoCompatibleVariant, message)),
            request.RequirePreferredExecutionProvider);

    internal static StageRuntimePlan StampRequirePreferred(StageRuntimePlan plan, bool requirePreferredExecutionProvider) =>
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

    private sealed record PlanningSnapshot(
        HardwareProfile HardwareProfile,
        IReadOnlyList<ExecutionProviderAvailability> ProviderAvailabilities,
        IReadOnlyList<DeviceEntry>? Devices,
        DeviceExclusionSet? EffectiveExclusions,
        PlanCacheKey CacheKey);

    private sealed record PlanningContext(
        StageRuntimePlanningRequest Request,
        StageRuntimeRequirements Requirements,
        HardwareProfile HardwareProfile,
        IReadOnlyList<ExecutionProviderAvailability> ProviderAvailabilities,
        IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> CacheIndex,
        ConcurrentDictionary<string, bool> FileExistenceCache,
        bool PreferMigraphxOnAmdGpu,
        IReadOnlyList<DeviceEntry>? Devices,
        DeviceExclusionSet? EffectiveExclusions,
        RankedManifestEntry[] PlanEntries);

    /// <summary>
    /// Remembers the first blocked candidate plan without ending the search: a later candidate
    /// may still be runnable, and the blocked plan is only the answer when none is.
    /// </summary>
    private sealed class PlanSearch
    {
        public StageRuntimePlan? BlockedPlan { get; private set; }

        /// <summary>True when the candidate ends the search.</summary>
        public bool TryAccept(StageRuntimePlan? candidate)
        {
            if (candidate?.Status == StageRuntimePlanStatus.Blocked)
            {
                BlockedPlan ??= candidate;
                return false;
            }

            return candidate is not null;
        }
    }

    private async Task<StageRuntimePlan?> TryResolvePlanFromEntriesAsync(
        PlanningContext context,
        CancellationToken cancellationToken)
    {
        StageRuntimePlanningRequest request = context.Request;
        var search = new PlanSearch();
        foreach (RankedManifestEntry candidate in context.PlanEntries)
        {
            StageRuntimePlan? readyPlan = await planFactory.TryCreateReadyPlanAsync(
                request.Stage,
                context.Requirements,
                candidate,
                context.HardwareProfile,
                context.ProviderAvailabilities,
                context.CacheIndex,
                context.FileExistenceCache,
                request.PreferredExecutionProvider,
                request.RequirePreferredExecutionProvider,
                context.PreferMigraphxOnAmdGpu,
                request.NormalizedPreferredModelVariantAlias,
                request.SkipProviderSmokeTest,
                cancellationToken,
                context.Devices,
                context.EffectiveExclusions).ConfigureAwait(false);

            if (search.TryAccept(readyPlan))
            {
                return readyPlan;
            }
        }

        foreach (RankedManifestEntry candidate in context.PlanEntries)
        {
            StageRuntimePlan? downloadPlan = planFactory.TryCreateDownloadRequiredPlan(
                request.Stage,
                context.Requirements,
                candidate,
                context.HardwareProfile,
                context.ProviderAvailabilities,
                context.CacheIndex,
                context.FileExistenceCache,
                request.PreferredExecutionProvider,
                request.RequirePreferredExecutionProvider,
                context.PreferMigraphxOnAmdGpu,
                request.NormalizedPreferredModelVariantAlias,
                context.Devices,
                context.EffectiveExclusions);

            if (search.TryAccept(downloadPlan))
            {
                return downloadPlan;
            }
        }

        return search.BlockedPlan;
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
    string? PreferredEngineFamily,
    string ExclusionSetSnapshot,
    NvidiaGpuArchitectureBucket NvidiaGpuArchitecture,
    bool SkipProviderSmokeTest);
