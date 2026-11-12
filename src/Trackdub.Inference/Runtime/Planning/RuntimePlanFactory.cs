using System.Collections.Concurrent;
using Trackdub.Contracts;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.Migraphx;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.TensorRtRtx;

namespace Trackdub.Inference.Runtime.Planning;

internal sealed class RuntimePlanFactory
{
    private readonly ISmokeVerdictStore smokeVerdictStore;

    // The smoke tester parameter is retained for constructor compatibility with existing
    // hosts/tests; planning no longer smokes (see TryCreateReadyPlanAsync), so it is unused.
    public RuntimePlanFactory(
        IExecutionProviderSmokeTester executionProviderSmokeTester,
        ISmokeVerdictStore? smokeVerdictStore = null)
    {
        ArgumentNullException.ThrowIfNull(executionProviderSmokeTester);
        this.smokeVerdictStore = smokeVerdictStore ?? NullSmokeVerdictStore.Instance;
    }

    public async Task<StageRuntimePlan?> TryCreateReadyPlanAsync(
        RuntimeStage stage,
        StageRuntimeRequirements requirements,
        RankedManifestEntry candidate,
        HardwareProfile hardwareProfile,
        IReadOnlyList<ExecutionProviderAvailability> providerAvailabilities,
        IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> cacheIndex,
        ConcurrentDictionary<string, bool> fileExistenceCache,
        ExecutionProviderKind? preferredExecutionProvider,
        bool requirePreferredExecutionProvider,
        bool preferMigraphxOnAmdGpu,
        string? preferredModelVariantAlias,
        bool skipProviderSmokeTest,
        CancellationToken cancellationToken,
        IReadOnlyList<DeviceEntry>? devices = null,
        DeviceExclusionSet? exclusions = null)
    {
        cacheIndex.TryGetValue(candidate.Entry.ModelId, out IReadOnlyList<LocalModelCacheRecord>? cacheRecords);

        IReadOnlyList<ExecutionProviderKind> orderedProviders = GetOrderedProviders(
            requirements,
            candidate.Entry,
            preferredExecutionProvider,
            requirePreferredExecutionProvider,
            preferMigraphxOnAmdGpu);
        StageRuntimePlan? earlyExit = TryGetBlockedEarlyExit(stage, orderedProviders, candidate, preferredExecutionProvider, requirePreferredExecutionProvider);
        if (earlyExit is not null) return earlyExit;

        bool preferredForbiddenForEngine = IsPreferredProviderForbiddenForEngine(
            requirements,
            candidate.Entry,
            preferredExecutionProvider);

        RuntimePlanFallback? providerFallback = null;
        cancellationToken.ThrowIfCancellationRequested();
        foreach (ExecutionProviderKind provider in orderedProviders)
        {
            StageRuntimePlan? guardResult = EvaluateProviderGuard(provider, candidate, providerAvailabilities,
                preferredExecutionProvider, requirePreferredExecutionProvider && !preferredForbiddenForEngine, stage,
                out ProviderGuardOutcome outcome, out RuntimePlanFallback? fallbackUpdate);
            if (outcome == ProviderGuardOutcome.Return) return guardResult;
            if (outcome == ProviderGuardOutcome.Skip)
            {
                if (fallbackUpdate is not null)
                {
                    providerFallback = MergeProviderFallback(providerFallback, fallbackUpdate);
                }
                continue;
            }

            foreach (VariantCandidate variant in EnumerateVariants(
                         candidate.Entry,
                         provider,
                         requirements,
                         hardwareProfile,
                         cacheRecords,
                         preferredModelVariantAlias))
            {
                if (!TryResolveCachedEntryPath(
                        candidate.Entry,
                        cacheRecords,
                        variant,
                        fileExistenceCache,
                        out string? entryPath,
                        out string? rootPath,
                        out RuntimeModelIntegrityStatus modelIntegrityStatus,
                        out _,
                        out string? modelSha256))
                {
                    continue;
                }

                if (provider is ExecutionProviderKind.Cpu || skipProviderSmokeTest)
                {
                    // CPU is file-only Ready. Listing/inventory SkipProviderSmokeTest is the same
                    // gate without compiling graphs: files + EP present, not Verified.
                    return CreatePlan(
                        stage,
                        StageRuntimePlanStatus.Ready,
                        candidate,
                        provider,
                        variant.Alias,
                        entryPath,
                        modelIntegrityStatus,
                        providerFallback,
                        includeCpuFallbackWarning: providerFallback is not null && provider is ExecutionProviderKind.Cpu,
                        isLocalOptimizedVariant: variant.IsLocalOptimizedVariant,
                        modelRootPath: rootPath,
                        modelEntryRelativePath: variant.RelativeEntryPath,
                        requiredModelRelativePaths: variant.RequiredRelativePaths,
                        preferredExecutionProviderSkippedForEngine: preferredForbiddenForEngine
                            ? preferredExecutionProvider
                            : null,
                        modelRevisionHash: modelSha256,
                        devices: devices,
                        exclusions: exclusions);
                }

                // A local optimized variant is a different graph file from its source model;
                // a source-sha verdict cannot prove it, so it never consults the verdict store.
                SmokeVerdictKey? verdictKey = variant.IsLocalOptimizedVariant
                    ? null
                    : TryBuildVerdictKey(modelSha256, provider, hardwareProfile);
                if (verdictKey is not null && smokeVerdictStore.IsVerified(verdictKey))
                {
                    // Proven on a previous launch under the same model/EP/environment identity.
                    // File and manifest integrity were already checked above.
                    return CreatePlan(
                        stage,
                        StageRuntimePlanStatus.Verified,
                        candidate,
                        provider,
                        variant.Alias,
                        entryPath,
                        modelIntegrityStatus,
                        fallback: null,
                        includeCpuFallbackWarning: false,
                        isLocalOptimizedVariant: variant.IsLocalOptimizedVariant,
                        modelRootPath: rootPath,
                        modelEntryRelativePath: variant.RelativeEntryPath,
                        requiredModelRelativePaths: variant.RequiredRelativePaths,
                        preferredExecutionProviderSkippedForEngine: preferredForbiddenForEngine
                            ? preferredExecutionProvider
                            : null,
                        modelRevisionHash: modelSha256,
                        devices: devices,
                        exclusions: exclusions);
                }

                // No persisted verdict: planning must not create throwaway smoke sessions per
                // provider candidate (C4). Return the honest file-only Ready status already
                // used for unproven plans (CPU / smoke-skipped listing) — never Verified. The
                // single proving create is deferred to RuntimeStageWarmupCoordinator, whose
                // smoke leases the same pooled session the stage later reuses, or to the
                // stage's own pooled session creation at execution time.
                return CreatePlan(
                    stage,
                    StageRuntimePlanStatus.Ready,
                    candidate,
                    provider,
                    variant.Alias,
                    entryPath,
                    modelIntegrityStatus,
                    providerFallback,
                    includeCpuFallbackWarning: false,
                    isLocalOptimizedVariant: variant.IsLocalOptimizedVariant,
                    modelRootPath: rootPath,
                    modelEntryRelativePath: variant.RelativeEntryPath,
                    requiredModelRelativePaths: variant.RequiredRelativePaths,
                    preferredExecutionProviderSkippedForEngine: preferredForbiddenForEngine
                        ? preferredExecutionProvider
                        : null,
                    modelRevisionHash: modelSha256,
                    devices: devices,
                    exclusions: exclusions);
            }
        }

        return null;
    }

    public StageRuntimePlan? TryCreateDownloadRequiredPlan(
        RuntimeStage stage,
        StageRuntimeRequirements requirements,
        RankedManifestEntry candidate,
        HardwareProfile hardwareProfile,
        IReadOnlyList<ExecutionProviderAvailability> providerAvailabilities,
        IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> cacheIndex,
        ConcurrentDictionary<string, bool> fileExistenceCache,
        ExecutionProviderKind? preferredExecutionProvider,
        bool requirePreferredExecutionProvider,
        bool preferMigraphxOnAmdGpu,
        string? preferredModelVariantAlias,
        IReadOnlyList<DeviceEntry>? devices = null,
        DeviceExclusionSet? exclusions = null)
    {
        cacheIndex.TryGetValue(candidate.Entry.ModelId, out IReadOnlyList<LocalModelCacheRecord>? cacheRecords);

        IReadOnlyList<ExecutionProviderKind> orderedProviders = GetOrderedProviders(
            requirements,
            candidate.Entry,
            preferredExecutionProvider,
            requirePreferredExecutionProvider,
            preferMigraphxOnAmdGpu);
        StageRuntimePlan? earlyExit = TryGetBlockedEarlyExit(stage, orderedProviders, candidate, preferredExecutionProvider, requirePreferredExecutionProvider);
        if (earlyExit is not null) return earlyExit;

        bool preferredForbiddenForEngine = IsPreferredProviderForbiddenForEngine(
            requirements,
            candidate.Entry,
            preferredExecutionProvider);

        foreach (ExecutionProviderKind provider in orderedProviders)
        {
            StageRuntimePlan? guardResult = EvaluateProviderGuard(provider, candidate, providerAvailabilities,
                preferredExecutionProvider, requirePreferredExecutionProvider && !preferredForbiddenForEngine, stage,
                out ProviderGuardOutcome outcome, out _);
            if (outcome == ProviderGuardOutcome.Return) return guardResult;
            if (outcome == ProviderGuardOutcome.Skip) continue;

            foreach (VariantCandidate variant in EnumerateVariants(
                         candidate.Entry,
                         provider,
                         requirements,
                         hardwareProfile,
                         cacheRecords,
                         preferredModelVariantAlias))
            {
                if (TryResolveCachedEntryPath(
                        candidate.Entry,
                        cacheRecords,
                        variant,
                        fileExistenceCache,
                        out _,
                        out _,
                        out _,
                        out RuntimePlanFallback? integrityFallback,
                        out _))
                {
                    continue;
                }

                return CreatePlan(
                    stage,
                    StageRuntimePlanStatus.DownloadRequired,
                    candidate,
                    provider,
                    variant.Alias,
                    null,
                    RuntimeModelIntegrityStatus.Unknown,
                    integrityFallback ?? new RuntimePlanFallback(
                        RuntimePlanFallbackCode.ModelNotCached,
                        $"Machine-local cache does not contain '{variant.RelativeEntryPath}' for model '{candidate.Entry.ModelId}'."),
                    includeCpuFallbackWarning: false,
                    isLocalOptimizedVariant: variant.IsLocalOptimizedVariant,
                    modelRootPath: variant.LocalRootPath,
                    modelEntryRelativePath: variant.RelativeEntryPath,
                    requiredModelRelativePaths: variant.RequiredRelativePaths,
                    preferredExecutionProviderSkippedForEngine: preferredForbiddenForEngine
                        ? preferredExecutionProvider
                        : null,
                    devices: devices,
                    exclusions: exclusions);
            }
        }

        return null;
    }

    public StageRuntimePlan CreateBlockedPlan(
        RuntimeStage stage,
        RuntimePlanFallback fallback)
    {
        return new StageRuntimePlan
        {
            Stage = stage,
            Status = StageRuntimePlanStatus.Blocked,
            Fallback = fallback,
            Warnings = []
        };
    }

    private StageRuntimePlan? TryGetBlockedEarlyExit(
        RuntimeStage stage,
        IReadOnlyList<ExecutionProviderKind> orderedProviders,
        RankedManifestEntry candidate,
        ExecutionProviderKind? preferredExecutionProvider,
        bool requirePreferredExecutionProvider)
    {
        if (orderedProviders.Count == 0 &&
            requirePreferredExecutionProvider &&
            preferredExecutionProvider is ExecutionProviderKind requiredProvider)
        {
            return CreateBlockedPlan(
                stage,
                new RuntimePlanFallback(
                    RuntimePlanFallbackCode.NoCompatibleVariant,
                    $"{requiredProvider} is not allowed for {candidate.Entry.EngineFamily} in stage {stage}."));
        }
        return null;
    }

    private static RuntimePlanFallback MergeProviderFallback(
        RuntimePlanFallback? current,
        RuntimePlanFallback next)
    {
        if (current is null)
        {
            return next;
        }

        // A later smoke failure is more specific than an earlier catalog EP that was simply
        // not present on this machine (common once TensorRT is excluded from a stage).
        if (current.Code == RuntimePlanFallbackCode.ProviderUnavailable &&
            next.Code == RuntimePlanFallbackCode.ProviderSmokeTestFailed)
        {
            return next;
        }

        return current;
    }

    private StageRuntimePlan? EvaluateProviderGuard(
        ExecutionProviderKind provider,
        RankedManifestEntry candidate,
        IReadOnlyList<ExecutionProviderAvailability> providerAvailabilities,
        ExecutionProviderKind? preferredExecutionProvider,
        bool requirePreferredExecutionProvider,
        RuntimeStage stage,
        out ProviderGuardOutcome outcome,
        out RuntimePlanFallback? providerFallbackUpdate)
    {
        providerFallbackUpdate = null;
        outcome = ProviderGuardOutcome.Proceed;

        if (provider is ExecutionProviderKind.Migraphx && !MigraphxModelSupport.SupportsEntry(candidate.Entry))
        {
            if (requirePreferredExecutionProvider && preferredExecutionProvider is ExecutionProviderKind.Migraphx)
            {
                outcome = ProviderGuardOutcome.Return;
                return CreateBlockedPlan(stage,
                    new RuntimePlanFallback(RuntimePlanFallbackCode.NoCompatibleVariant,
                        $"Model '{candidate.Entry.ModelId}' does not support required execution provider {ExecutionProviderKind.Migraphx}."));
            }
            outcome = ProviderGuardOutcome.Skip;
            return null;
        }

        ExecutionProviderAvailability availability = GetAvailability(providerAvailabilities, provider);
        if (availability.IsAvailable &&
            availability.EngineFamilies is { } hostedFamilies &&
            !hostedFamilies.Contains(candidate.Entry.EngineFamily, StringComparer.OrdinalIgnoreCase))
        {
            availability = availability with
            {
                IsAvailable = false,
                Detail = $"{provider} is only available to {string.Join(", ", hostedFamilies)} on this machine; " +
                         $"'{candidate.Entry.EngineFamily}' runs in-process. {availability.Detail}",
            };
        }

        if (provider is not ExecutionProviderKind.Cpu && !availability.IsAvailable)
        {
            if (requirePreferredExecutionProvider)
            {
                outcome = ProviderGuardOutcome.Return;
                return CreateBlockedPlan(stage,
                    new RuntimePlanFallback(RuntimePlanFallbackCode.ProviderUnavailable,
                        availability.Detail ?? $"{provider} is not available on this machine."));
            }
            providerFallbackUpdate = new RuntimePlanFallback(RuntimePlanFallbackCode.ProviderUnavailable,
                availability.Detail ?? $"{provider} is not available on this machine.");
            outcome = ProviderGuardOutcome.Skip;
            return null;
        }

        return null;
    }

    private static StageRuntimePlan CreatePlan(
        RuntimeStage stage,
        StageRuntimePlanStatus status,
        RankedManifestEntry candidate,
        ExecutionProviderKind provider,
        string variant,
        string? modelEntryPath,
        RuntimeModelIntegrityStatus modelIntegrityStatus,
        RuntimePlanFallback? fallback,
        bool includeCpuFallbackWarning,
        bool isLocalOptimizedVariant = false,
        string? modelRootPath = null,
        string? modelEntryRelativePath = null,
        IReadOnlyList<string>? requiredModelRelativePaths = null,
        ExecutionProviderKind? preferredExecutionProviderSkippedForEngine = null,
        string? modelRevisionHash = null,
        IReadOnlyList<DeviceEntry>? devices = null,
        DeviceExclusionSet? exclusions = null)
    {
        (int? deviceIndex, string? deviceAdapterDescription) = ResolvePlanDevice(provider, devices, exclusions);
        return new StageRuntimePlan
        {
            Stage = stage,
            Status = status,
            ModelId = candidate.Entry.ModelId,
            ModelAlias = ResolvePrimaryAlias(candidate.Entry),
            EngineFamily = candidate.Entry.EngineFamily,
            ModelTier = candidate.Entry.Tier,
            Variant = variant,
            ExecutionProvider = provider,
            ModelIntegrityStatus = modelIntegrityStatus,
            ModelEntryPath = modelEntryPath,
            ModelRootPath = modelRootPath,
            ModelEntryRelativePath = modelEntryRelativePath,
            RequiredModelRelativePaths = requiredModelRelativePaths ?? [],
            IsLocalOptimizedVariant = isLocalOptimizedVariant,
            ModelRevisionHash = modelRevisionHash,
            DeviceIndex = deviceIndex,
            DeviceAdapterDescription = deviceAdapterDescription,
            Fallback = fallback,
            Warnings = BuildWarnings(
                candidate.Entry,
                provider,
                includeCpuFallbackWarning,
                modelIntegrityStatus,
                preferredExecutionProviderSkippedForEngine)
        };
    }

    /// <summary>
    /// Selects the device the planned execution provider will run on: the lowest-index
    /// device advertising the provider that is not in the run's exclusion set.
    /// CPU-only providers (<see cref="ExecutionProviderKind.Cpu"/> and
    /// <see cref="ExecutionProviderKind.Dnnl"/>) additionally require
    /// <see cref="DeviceKind.Cpu"/>: on Linux an AMD GPU advertises
    /// <see cref="ExecutionProviderKind.Cpu"/> at index 0 while the CPU entry is appended
    /// last, and stamping the GPU index would misattribute a CPU OOM to the GPU on fallback.
    /// Returns no device when the device list is unavailable (planning outside a run)
    /// or no eligible device remains; the device-fallback catch filter then lets the
    /// failure propagate instead of recording a bogus exclusion.
    /// </summary>
    /// <remarks>
    /// The stamp is the planner-ranked device used for exclusion routing, not a session
    /// binding: pooled session creation keys on <c>deviceId: null</c> and appends the
    /// provider with device id 0 (CUDA/DirectML), so on multi-GPU hosts a replan that
    /// excludes the stamped index can still bind the same physical device. Wiring the
    /// planned index into session creation (per-provider device-id mapping, pool-key
    /// sharding) is intentionally out of scope here; see
    /// <see cref="StageRuntimePlan.DeviceIndex"/>.
    /// </remarks>
    private static (int? DeviceIndex, string? AdapterDescription) ResolvePlanDevice(
        ExecutionProviderKind provider,
        IReadOnlyList<DeviceEntry>? devices,
        DeviceExclusionSet? exclusions)
    {
        if (devices is null)
        {
            return (null, null);
        }

        foreach (DeviceEntry device in devices.OrderBy(static candidate => candidate.DeviceIndex))
        {
            if (!device.SupportedProviders.Contains(provider))
            {
                continue;
            }

            // CPU-only providers must land on the CPU entry: accelerators may also
            // advertise Cpu (Linux AMD GPUs do at index 0), and stamping one would
            // misattribute a CPU failure to the accelerator on device fallback.
            if (provider is ExecutionProviderKind.Cpu or ExecutionProviderKind.Dnnl
                && device.Kind != DeviceKind.Cpu)
            {
                continue;
            }

            if (exclusions is not null && exclusions.IsExcluded(device.DeviceIndex))
            {
                continue;
            }

            return (device.DeviceIndex, device.AdapterDescription);
        }

        return (null, null);
    }

    private static IReadOnlyList<RuntimePlanWarning> BuildWarnings(
        BundledModelManifestEntry entry,
        ExecutionProviderKind provider,
        bool includeCpuFallbackWarning,
        RuntimeModelIntegrityStatus modelIntegrityStatus,
        ExecutionProviderKind? preferredExecutionProviderSkippedForEngine = null)
    {
        var warnings = new List<RuntimePlanWarning>();

        if (includeCpuFallbackWarning)
        {
            warnings.Add(new RuntimePlanWarning(RuntimePlanWarningCode.CpuFallback));
        }

        if (preferredExecutionProviderSkippedForEngine is ExecutionProviderKind skippedProvider)
        {
            warnings.Add(new RuntimePlanWarning(
                RuntimePlanWarningCode.PreferredExecutionProviderNotAllowedForEngine,
                $"{skippedProvider} is not allowed for {entry.EngineFamily}; planned {provider} instead."));
        }

        if (modelIntegrityStatus is RuntimeModelIntegrityStatus.Skipped)
        {
            warnings.Add(new RuntimePlanWarning(
                RuntimePlanWarningCode.ModelIntegrityNotVerified,
                "Manifest sha256 is missing, so cached model integrity was not verified."));
        }

        if (entry.RequiresAttribution)
        {
            warnings.Add(new RuntimePlanWarning(RuntimePlanWarningCode.AttributionRequired));
        }

        if (entry.RequiresUserConsent)
        {
            warnings.Add(new RuntimePlanWarning(RuntimePlanWarningCode.UserConsentRequired));
        }

        if (!RuntimeProviderTokenCompatibility.IsExpectedRuntimeCompatible(entry.ExpectedRuntime, provider))
        {
            warnings.Add(new RuntimePlanWarning(
                RuntimePlanWarningCode.ExpectedRuntimeMismatch,
                $"Manifest expected_runtime '{entry.ExpectedRuntime}' does not list selected provider '{RuntimeProviderTokenCompatibility.ToManifestToken(provider)}'."));
        }

        return warnings;
    }

    private static IReadOnlyList<ExecutionProviderKind> GetOrderedProviders(
        StageRuntimeRequirements requirements,
        BundledModelManifestEntry entry,
        ExecutionProviderKind? preferredExecutionProvider,
        bool requirePreferredExecutionProvider,
        bool preferMigraphxOnAmdGpu)
    {
        IReadOnlyList<ExecutionProviderKind> allowedProviders = requirements.AllowedProvidersThisMilestone;
        if (requirements.AllowedProvidersByEngineFamily is not null &&
            requirements.AllowedProvidersByEngineFamily.TryGetValue(entry.EngineFamily, out IReadOnlyList<ExecutionProviderKind>? engineFamilyProviders))
        {
            allowedProviders = engineFamilyProviders;
        }

        if (requirePreferredExecutionProvider && preferredExecutionProvider is ExecutionProviderKind requiredProvider)
        {
            if (allowedProviders.Contains(requiredProvider))
            {
                return [requiredProvider];
            }

            // A required provider is a hard constraint, including for engine family restrictions.
            return [];
        }

        // Family overrides may intentionally change priority (for example, DirectML
        // before an expensive TensorRT RTX compile for Qwen ASR).
        var availableProviders = allowedProviders
            .Where(provider => Milestone5PlanningPolicy.SupportedProvidersThisMilestone.Contains(provider));

        IReadOnlyList<ExecutionProviderKind> ordered = availableProviders.ToArray();

        if (preferredExecutionProvider is ExecutionProviderKind preferred
            && allowedProviders.Contains(preferred))
        {
            ordered = availableProviders
                .OrderByDescending(p => p == preferred)
                .ToArray();
        }

        return MigraphxProviderOrdering.ApplyAmdMigraphxFirst(ordered, preferMigraphxOnAmdGpu);
    }

    private static bool IsPreferredProviderForbiddenForEngine(
        StageRuntimeRequirements requirements,
        BundledModelManifestEntry entry,
        ExecutionProviderKind? preferredExecutionProvider)
    {
        if (preferredExecutionProvider is not ExecutionProviderKind preferred)
        {
            return false;
        }

        IReadOnlyList<ExecutionProviderKind> allowedProviders = requirements.AllowedProvidersThisMilestone;
        if (requirements.AllowedProvidersByEngineFamily is not null &&
            requirements.AllowedProvidersByEngineFamily.TryGetValue(entry.EngineFamily, out IReadOnlyList<ExecutionProviderKind>? engineFamilyProviders))
        {
            allowedProviders = engineFamilyProviders;
        }

        return !allowedProviders.Contains(preferred);
    }

    private static ExecutionProviderAvailability GetAvailability(
        IReadOnlyList<ExecutionProviderAvailability> providerAvailabilities,
        ExecutionProviderKind provider)
    {
        ExecutionProviderAvailability? availability = providerAvailabilities
            .FirstOrDefault(candidate => candidate.Provider == provider);

        if (availability is not null)
        {
            return availability;
        }

        return provider switch
        {
            ExecutionProviderKind.Cpu => new ExecutionProviderAvailability(provider, true),
            _ => new ExecutionProviderAvailability(provider, false, $"{provider} was not reported by execution provider discovery.")
        };
    }

    private static IReadOnlyList<VariantCandidate> EnumerateVariants(
        BundledModelManifestEntry entry,
        ExecutionProviderKind provider,
        StageRuntimeRequirements requirements,
        HardwareProfile hardwareProfile,
        IReadOnlyList<LocalModelCacheRecord>? cacheRecords,
        string? preferredModelVariantAlias)
    {
        string? normalizedPreferredVariantAlias = string.IsNullOrWhiteSpace(preferredModelVariantAlias)
            ? null
            : preferredModelVariantAlias.Trim();
        IReadOnlyList<string> preferredAliases = provider switch
        {
            ExecutionProviderKind.DirectMl or ExecutionProviderKind.TensorRTRtx or ExecutionProviderKind.Migraphx
                or ExecutionProviderKind.Cuda or ExecutionProviderKind.TensorRt or ExecutionProviderKind.OpenVino
                or ExecutionProviderKind.OpenVinoCatalog or ExecutionProviderKind.Qnn or ExecutionProviderKind.VitisAi
                or ExecutionProviderKind.CoreMl =>
                GpuVariantPreferencePolicy.GetPreferredGpuVariantAliases(requirements, hardwareProfile, provider, entry),
            _ => requirements.PreferredCpuVariants
        };

        var candidates = new List<VariantCandidate>();
        var seenCandidates = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var variantsByAlias = entry.Variants.ToDictionary(variant => variant.Alias, StringComparer.OrdinalIgnoreCase);

        if (normalizedPreferredVariantAlias is not null)
        {
            AddLocalVariantCandidates(candidates, seenCandidates, cacheRecords, normalizedPreferredVariantAlias, provider);
            if (variantsByAlias.TryGetValue(normalizedPreferredVariantAlias, out BundledModelManifestVariant? preferredManifestVariant) &&
                VariantManifestReadiness.IsManifestVariantEligibleForPlanning(entry, preferredManifestVariant, provider))
            {
                AddVariant(
                    candidates,
                    seenCandidates,
                    entry,
                    normalizedPreferredVariantAlias,
                    preferredManifestVariant.EntryPath,
                    preferredManifestVariant.DownloadFiles);
            }

            if (normalizedPreferredVariantAlias.Equals("default", StringComparison.OrdinalIgnoreCase))
            {
                string preferredDefaultRelativeEntryPath = Path.GetRelativePath(entry.RootDirectory, entry.DefaultBenchmarkEntryPath);
                BundledModelManifestVariant? preferredDefaultVariant = entry.Variants
                    .FirstOrDefault(variant =>
                        Path.GetRelativePath(entry.RootDirectory, variant.EntryPath)
                            .Equals(preferredDefaultRelativeEntryPath, StringComparison.OrdinalIgnoreCase));
                if (preferredDefaultVariant is null ||
                    VariantManifestReadiness.IsManifestVariantEligibleForPlanning(entry, preferredDefaultVariant, provider))
                {
                    IReadOnlyList<string> preferredDefaultVariantDownloadFiles = preferredDefaultVariant?.DownloadFiles ?? [];
                    AddVariant(candidates, seenCandidates, entry, "default", entry.DefaultBenchmarkEntryPath, preferredDefaultVariantDownloadFiles);
                }
            }

            return candidates;
        }

        foreach (string alias in preferredAliases)
        {
            if (!variantsByAlias.TryGetValue(alias, out BundledModelManifestVariant? variant))
            {
                continue;
            }

            if (!VariantManifestReadiness.IsManifestVariantEligibleForPlanning(entry, variant, provider))
            {
                continue;
            }

            AddVariant(candidates, seenCandidates, entry, alias, variant.EntryPath, variant.DownloadFiles);
        }

        string defaultRelativeEntryPath = Path.GetRelativePath(entry.RootDirectory, entry.DefaultBenchmarkEntryPath);
        BundledModelManifestVariant? defaultVariant = entry.Variants
            .FirstOrDefault(variant =>
                Path.GetRelativePath(entry.RootDirectory, variant.EntryPath)
                    .Equals(defaultRelativeEntryPath, StringComparison.OrdinalIgnoreCase));
        if (defaultVariant is null ||
            VariantManifestReadiness.IsManifestVariantEligibleForPlanning(entry, defaultVariant, provider))
        {
            IReadOnlyList<string> defaultVariantDownloadFiles = defaultVariant?.DownloadFiles ?? [];
            AddVariant(candidates, seenCandidates, entry, "default", entry.DefaultBenchmarkEntryPath, defaultVariantDownloadFiles);
        }

        foreach (BundledModelManifestVariant variant in entry.Variants.OrderBy(variant => variant.Alias, StringComparer.OrdinalIgnoreCase))
        {
            if (!VariantManifestReadiness.IsManifestVariantEligibleForPlanning(entry, variant, provider))
            {
                continue;
            }

            AddVariant(candidates, seenCandidates, entry, variant.Alias, variant.EntryPath, variant.DownloadFiles);
        }

        return candidates;
    }

    private static void AddLocalVariantCandidates(
        ICollection<VariantCandidate> candidates,
        ISet<string> seenCandidates,
        IReadOnlyList<LocalModelCacheRecord>? cacheRecords,
        string preferredModelVariantAlias,
        ExecutionProviderKind provider)
    {
        if (cacheRecords is null)
        {
            return;
        }

        foreach (LocalModelCacheRecord cacheRecord in cacheRecords)
        {
            foreach (LocalModelVariantRecord variant in cacheRecord.Variants)
            {
                if (!variant.Alias.Equals(preferredModelVariantAlias, StringComparison.OrdinalIgnoreCase) ||
                    variant.ExecutionProvider != provider)
                {
                    continue;
                }

                AddLocalVariantCandidate(candidates, seenCandidates, cacheRecord, variant);
            }
        }
    }

    private static void AddLocalVariantCandidate(
        ICollection<VariantCandidate> candidates,
        ISet<string> seenCandidates,
        LocalModelCacheRecord cacheRecord,
        LocalModelVariantRecord variant)
    {
        string? invalidReason = null;
        if (!TryResolveRootUnderBase(cacheRecord.RootPath, variant.RootPath, out string? variantRoot, out string? rootInvalidReason))
        {
            invalidReason = rootInvalidReason;
        }

        if (!TryNormalizeLocalRelativePath(variant.EntryRelativePath, out string entryRelativePath))
        {
            invalidReason ??= $"Selected optimized variant '{variant.Alias}' has an invalid entry path '{variant.EntryRelativePath}'.";
            entryRelativePath = variant.EntryRelativePath;
        }

        var requiredPaths = new List<string> { entryRelativePath };
        var requiredPathsSet = new HashSet<string>(requiredPaths, StringComparer.OrdinalIgnoreCase);
        foreach (string componentRelativePath in variant.ComponentRelativePaths)
        {
            if (!TryNormalizeLocalRelativePath(componentRelativePath, out string normalizedComponentPath))
            {
                invalidReason ??= $"Selected optimized variant '{variant.Alias}' has an invalid component path '{componentRelativePath}'.";
                normalizedComponentPath = componentRelativePath;
            }

            if (requiredPathsSet.Add(normalizedComponentPath))
            {
                requiredPaths.Add(normalizedComponentPath);
            }
        }

        string candidateKey = $"local|{variantRoot ?? variant.RootPath}|{entryRelativePath}";
        if (!seenCandidates.Add(candidateKey))
        {
            return;
        }

        candidates.Add(new VariantCandidate(
            variant.Alias,
            entryRelativePath,
            requiredPaths,
            IsLocalOptimizedVariant: true,
            LocalRootPath: variantRoot,
            LocalIntegrityFailed: variant.IntegrityFailed,
            InvalidReason: invalidReason,
            SourceModelSha256: !string.IsNullOrWhiteSpace(variant.SourceModelSha256)
                ? variant.SourceModelSha256
                : cacheRecord.Sha256));
    }

    private static void AddVariant(
        ICollection<VariantCandidate> candidates,
        ISet<string> seenPaths,
        BundledModelManifestEntry entry,
        string alias,
        string absoluteEntryPath,
        IReadOnlyList<string> variantDownloadFiles)
    {
        string relativeEntryPath = Path.GetRelativePath(entry.RootDirectory, absoluteEntryPath);
        if (!seenPaths.Add($"manifest|{relativeEntryPath}"))
        {
            return;
        }

        candidates.Add(new VariantCandidate(
            alias,
            relativeEntryPath,
            BuildRequiredRelativePaths(relativeEntryPath, entry.DownloadFiles, variantDownloadFiles)));
    }

    private static IReadOnlyList<string> BuildRequiredRelativePaths(
        string relativeEntryPath,
        IReadOnlyList<string> entryDownloadFiles,
        IReadOnlyList<string> variantDownloadFiles)
    {
        var requiredFiles = new List<string>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        AddRequiredFile(relativeEntryPath);
        foreach (string file in entryDownloadFiles)
        {
            AddRequiredFile(file);
        }

        foreach (string file in variantDownloadFiles)
        {
            AddRequiredFile(file);
        }

        return requiredFiles;

        void AddRequiredFile(string relativePath)
        {
            string normalizedPath = relativePath.Replace('\\', Path.DirectorySeparatorChar).Replace('/', Path.DirectorySeparatorChar);
            if (seen.Add(normalizedPath))
            {
                requiredFiles.Add(normalizedPath);
            }
        }
    }

    private static bool TryResolveCachedEntryPath(
        BundledModelManifestEntry entry,
        IReadOnlyList<LocalModelCacheRecord>? cacheRecords,
        VariantCandidate variant,
        ConcurrentDictionary<string, bool> fileExistenceCache,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out string? entryPath,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out string? rootPath,
        out RuntimeModelIntegrityStatus integrityStatus,
        out RuntimePlanFallback? integrityFallback,
        out string? modelSha256)
    {
        entryPath = null;
        rootPath = null;
        integrityStatus = RuntimeModelIntegrityStatus.Unknown;
        integrityFallback = null;
        modelSha256 = null;
        if (cacheRecords is null)
        {
            return false;
        }

        if (variant.IsLocalOptimizedVariant)
        {
            return TryResolveLocalOptimizedVariant(
                variant,
                fileExistenceCache,
                out entryPath,
                out rootPath,
                out integrityStatus,
                out integrityFallback,
                out modelSha256);
        }

        foreach (LocalModelCacheRecord cacheRecord in cacheRecords)
        {
            if (cacheRecord.IntegrityFailed)
            {
                integrityFallback ??= new RuntimePlanFallback(
                    RuntimePlanFallbackCode.ModelIntegrityMismatch,
                    $"Cached model '{entry.ModelId}' is marked as integrity-failed. Re-download the model.");
                continue;
            }

            string candidatePath = Path.GetFullPath(Path.Join(cacheRecord.RootPath, variant.RelativeEntryPath));
            if (RequiredFilesExist(cacheRecord.RootPath, variant.RequiredRelativePaths, fileExistenceCache))
            {
                if (HasManifestHashMismatch(entry, cacheRecord, out string? detail))
                {
                    integrityFallback ??= new RuntimePlanFallback(
                        RuntimePlanFallbackCode.ModelIntegrityMismatch,
                        detail);
                    continue;
                }

                entryPath = candidatePath;
                rootPath = cacheRecord.RootPath;
                integrityStatus = ResolveIntegrityStatus(entry, cacheRecord);
                modelSha256 = ResolveModelSha256(entry, cacheRecord, variant);
                return true;
            }
        }

        return false;
    }

    private static bool TryResolveLocalOptimizedVariant(
        VariantCandidate variant,
        ConcurrentDictionary<string, bool> fileExistenceCache,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out string? entryPath,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out string? rootPath,
        out RuntimeModelIntegrityStatus integrityStatus,
        out RuntimePlanFallback? integrityFallback,
        out string? modelSha256)
    {
        entryPath = null;
        rootPath = null;
        integrityStatus = RuntimeModelIntegrityStatus.Unknown;
        integrityFallback = null;
        modelSha256 = variant.SourceModelSha256;

        if (!string.IsNullOrWhiteSpace(variant.InvalidReason))
        {
            integrityFallback = new RuntimePlanFallback(RuntimePlanFallbackCode.ModelNotCached, variant.InvalidReason);
            return false;
        }

        if (variant.LocalIntegrityFailed)
        {
            integrityFallback = new RuntimePlanFallback(
                RuntimePlanFallbackCode.ModelIntegrityMismatch,
                $"Selected optimized variant '{variant.Alias}' is marked as failed. Re-optimize the model or clear the variant selection.");
            return false;
        }

        if (string.IsNullOrWhiteSpace(variant.LocalRootPath))
        {
            integrityFallback = new RuntimePlanFallback(
                RuntimePlanFallbackCode.ModelNotCached,
                $"Selected optimized variant '{variant.Alias}' has no registered local root path.");
            return false;
        }

        foreach (string relativePath in variant.RequiredRelativePaths)
        {
            if (!TryResolvePathUnderRoot(variant.LocalRootPath, relativePath, out string? requiredPath))
            {
                integrityFallback = new RuntimePlanFallback(
                    RuntimePlanFallbackCode.ModelNotCached,
                    $"Selected optimized variant '{variant.Alias}' has an invalid registered path '{relativePath}'.");
                return false;
            }

            if (!FileExists(fileExistenceCache, requiredPath))
            {
                integrityFallback = new RuntimePlanFallback(
                    RuntimePlanFallbackCode.ModelNotCached,
                    $"Selected optimized variant '{variant.Alias}' is missing required local file '{relativePath}'. Re-optimize the model or clear the variant selection.");
                return false;
            }
        }

        if (!TryResolvePathUnderRoot(variant.LocalRootPath, variant.RelativeEntryPath, out entryPath))
        {
            integrityFallback = new RuntimePlanFallback(
                RuntimePlanFallbackCode.ModelNotCached,
                $"Selected optimized variant '{variant.Alias}' has an invalid entry path '{variant.RelativeEntryPath}'.");
            return false;
        }

        rootPath = variant.LocalRootPath;
        return true;
    }

    private static RuntimeModelIntegrityStatus ResolveIntegrityStatus(
        BundledModelManifestEntry entry,
        LocalModelCacheRecord cacheRecord) =>
        string.IsNullOrWhiteSpace(entry.Sha256) || string.IsNullOrWhiteSpace(cacheRecord.Sha256)
            ? RuntimeModelIntegrityStatus.Skipped
            : RuntimeModelIntegrityStatus.Verified;

    private static string? ResolveModelSha256(
        BundledModelManifestEntry entry,
        LocalModelCacheRecord cacheRecord,
        VariantCandidate variant)
    {
        // Bundled variants (e.g. Silero fp16 vs int8) are distinct graph files with their own
        // hash in the manifest's per-file table. Falling back to the shared model-level hash
        // would key the smoke verdict by model instead of by graph, letting a verdict for one
        // variant wrongly verify a different, never-tested variant.
        string normalizedRelativeEntryPath = variant.RelativeEntryPath.Replace('\\', '/');
        if (entry.DownloadFileHashes.TryGetValue(normalizedRelativeEntryPath, out string? fileHash) &&
            !string.IsNullOrWhiteSpace(fileHash))
        {
            return fileHash;
        }

        return !string.IsNullOrWhiteSpace(cacheRecord.Sha256)
            ? cacheRecord.Sha256
            : !string.IsNullOrWhiteSpace(entry.Sha256)
                ? entry.Sha256
                : null;
    }

    private static SmokeVerdictKey? TryBuildVerdictKey(
        string? modelSha256,
        ExecutionProviderKind provider,
        HardwareProfile hardwareProfile) =>
        // The persisted key does not yet include the Windows ML device policy. A verdict
        // from a different policy cannot establish this session's actual placement.
        string.IsNullOrWhiteSpace(modelSha256) ||
        (OperatingSystem.IsWindows() && provider is ExecutionProviderKind.Qnn or ExecutionProviderKind.VitisAi or ExecutionProviderKind.OpenVinoCatalog or ExecutionProviderKind.Migraphx)
            ? null
            : new SmokeVerdictKey(
                modelSha256,
                provider,
                hardwareProfile.NvidiaGpuArchitecture,
                hardwareProfile.GpuDriverVersion,
                TensorRtRtxProviderConstants.BundledFingerprintVersion,
                hardwareProfile.OrtRuntimeVersion);

    private static bool HasManifestHashMismatch(
        BundledModelManifestEntry entry,
        LocalModelCacheRecord cacheRecord,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out string? detail)
    {
        detail = null;
        if (string.IsNullOrWhiteSpace(entry.Sha256) ||
            string.IsNullOrWhiteSpace(cacheRecord.Sha256))
        {
            return false;
        }

        if (entry.Sha256.Equals(cacheRecord.Sha256, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        detail = $"Cached model '{entry.ModelId}' sha256 '{cacheRecord.Sha256}' does not match manifest sha256 '{entry.Sha256}'.";
        return true;
    }

    private static bool RequiredFilesExist(
        string rootPath,
        IReadOnlyList<string> requiredRelativePaths,
        ConcurrentDictionary<string, bool> fileExistenceCache)
    {
        foreach (string relativePath in requiredRelativePaths)
        {
            string requiredPath = Path.GetFullPath(Path.Join(rootPath, relativePath));
            if (!FileExists(fileExistenceCache, requiredPath))
            {
                return false;
            }
        }

        return true;
    }

    private static bool TryResolveRootUnderBase(
        string baseRootPath,
        string variantRootPath,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out string? normalizedVariantRoot,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(false)] out string? invalidReason)
    {
        normalizedVariantRoot = null;
        invalidReason = null;
        if (string.IsNullOrWhiteSpace(baseRootPath) || string.IsNullOrWhiteSpace(variantRootPath))
        {
            invalidReason = "Selected optimized variant has an invalid registered root path.";
            return false;
        }

        string baseRoot = Path.GetFullPath(baseRootPath);
        string variantRoot = Path.GetFullPath(variantRootPath);
        if (!IsSameOrUnderRoot(baseRoot, variantRoot))
        {
            invalidReason = $"Selected optimized variant root '{variantRootPath}' is outside the base model cache root.";
            return false;
        }

        normalizedVariantRoot = variantRoot;
        return true;
    }

    private static bool TryNormalizeLocalRelativePath(
        string relativePath,
        out string normalizedPath)
    {
        normalizedPath = relativePath;
        if (string.IsNullOrWhiteSpace(relativePath) || Path.IsPathRooted(relativePath))
        {
            return false;
        }

        string[] parts = relativePath
            .Trim()
            .Replace('\\', '/')
            .Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length == 0 ||
            parts.Any(part => part.Equals(".", StringComparison.Ordinal) || part.Equals("..", StringComparison.Ordinal)))
        {
            return false;
        }

        normalizedPath = string.Join("/", parts);
        return true;
    }

    private static bool TryResolvePathUnderRoot(
        string rootPath,
        string relativePath,
        [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out string? resolvedPath)
    {
        resolvedPath = null;
        if (!TryNormalizeLocalRelativePath(relativePath, out string normalizedRelativePath))
        {
            return false;
        }

        string root = Path.GetFullPath(rootPath);
        string candidatePath = Path.GetFullPath(Path.Join(root, Path.Join(normalizedRelativePath.Split('/'))));
        if (!IsSameOrUnderRoot(root, candidatePath))
        {
            return false;
        }

        resolvedPath = candidatePath;
        return true;
    }

    private static bool IsSameOrUnderRoot(string rootPath, string candidatePath)
    {
        string root = Path.GetFullPath(rootPath)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string candidate = Path.GetFullPath(candidatePath)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        StringComparison comparison = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows)
            ? StringComparison.OrdinalIgnoreCase
            : StringComparison.Ordinal;
        return candidate.Equals(root, comparison) ||
            candidate.StartsWith(root + Path.DirectorySeparatorChar, comparison);
    }

    private static bool FileExists(ConcurrentDictionary<string, bool> fileExistenceCache, string path) =>
        fileExistenceCache.GetOrAdd(path, static candidatePath => File.Exists(candidatePath));

    private static string ResolvePrimaryAlias(BundledModelManifestEntry entry) =>
        entry.Aliases.Count > 0 ? entry.Aliases[0] : entry.ModelId;

    private enum ProviderGuardOutcome { Proceed, Skip, Return }

    private sealed record VariantCandidate(
        string Alias,
        string RelativeEntryPath,
        IReadOnlyList<string> RequiredRelativePaths,
        bool IsLocalOptimizedVariant = false,
        string? LocalRootPath = null,
        bool LocalIntegrityFailed = false,
        string? InvalidReason = null,
        string? SourceModelSha256 = null);
}
