using Trackdub.Contracts;
using Trackdub.Application.Runtime;
using Trackdub.Domain;
using Trackdub.Infrastructure.Persistence.Repositories;
using Trackdub.Infrastructure.Settings;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.Runtime;

public sealed class ModelInventoryService(
    BundledModelManifestRegistry manifestRegistry,
    LocalModelCacheRecordStore cacheStore,
    TrackdubStoragePaths storagePaths,
    IRuntimeSelectionService? runtimeSelectionService = null)
    : IModelInventoryService
{
    private readonly BundledModelManifestRegistry manifestRegistry = manifestRegistry ?? throw new ArgumentNullException(nameof(manifestRegistry));
    private readonly LocalModelCacheRecordStore cacheStore = cacheStore ?? throw new ArgumentNullException(nameof(cacheStore));
    private readonly IRuntimeSelectionService? runtimeSelectionService = runtimeSelectionService;
    private readonly string configuredModelCacheDirectory = Path.GetFullPath(
        (storagePaths ?? throw new ArgumentNullException(nameof(storagePaths))).ModelCacheDirectory);

    public async Task<IReadOnlyList<ModelInventoryEntry>> GetAllAsync(CancellationToken cancellationToken = default)
    {
        IReadOnlyList<LocalModelCacheRecord> cacheRecords = await cacheStore.LoadAsync(cancellationToken).ConfigureAwait(false);
        IReadOnlyList<ProviderCapability> providerCapabilities = await GetProviderCapabilitiesAsync(cancellationToken).ConfigureAwait(false);
        var cacheIndex = cacheRecords
            .GroupBy(r => r.ModelId, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.ToArray(), StringComparer.OrdinalIgnoreCase);

        return manifestRegistry.Entries
            .Where(static entry => !entry.Deprecated)
            .Select(entry => BuildEntry(entry, cacheIndex, providerCapabilities))
            .ToList();
    }

    public async Task<ModelInventoryEntry?> GetByModelIdAsync(string modelId, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);

        BundledModelManifestEntry? entry = manifestRegistry.Entries
            .FirstOrDefault(e => e.ModelId.Equals(modelId, StringComparison.OrdinalIgnoreCase) && !e.Deprecated);
        if (entry is null)
            return null;

        IReadOnlyList<LocalModelCacheRecord> cacheRecords = await cacheStore.LoadAsync(cancellationToken).ConfigureAwait(false);
        IReadOnlyList<ProviderCapability> providerCapabilities = await GetProviderCapabilitiesAsync(cancellationToken).ConfigureAwait(false);
        var cacheIndex = cacheRecords
            .GroupBy(r => r.ModelId, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.ToArray(), StringComparer.OrdinalIgnoreCase);
        return BuildEntry(entry, cacheIndex, providerCapabilities);
    }

    private ModelInventoryEntry BuildEntry(
        BundledModelManifestEntry manifest,
        Dictionary<string, LocalModelCacheRecord[]> cacheIndex,
        IReadOnlyList<ProviderCapability> providerCapabilities)
    {
        cacheIndex.TryGetValue(manifest.ModelId, out LocalModelCacheRecord[]? cacheRecords);
        LocalModelCacheRecord? cacheRecord = SelectBestCacheRecord(manifest, cacheRecords, configuredModelCacheDirectory);

        ModelCacheState state = DetermineState(manifest, cacheRecord);
        bool canAutoDownload = ModelDownloadManifestFiles.CanAutoDownloadAll(manifest);
        string? failureReason = state == ModelCacheState.Corrupt
            ? cacheRecord!.IntegrityFailed || HasManifestHashMismatch(manifest, cacheRecord)
                ? "Model failed integrity verification; use repair or re-download."
                : "Model files missing or corrupted on disk."
            : state == ModelCacheState.Missing && !canAutoDownload
                ? "No downloadable source configured for this model; install or import the model files into the local cache."
                : null;

        long? fileSize = cacheRecord is not null && state is ModelCacheState.Installed or ModelCacheState.Ready
            ? TryGetFileSize(ResolveCachedBenchmarkEntryPath(manifest, cacheRecord))
            : null;

        string displayName = !string.IsNullOrWhiteSpace(manifest.DisplayName)
            ? manifest.DisplayName
            : manifest.Aliases.Count > 0
                ? manifest.Aliases[0]
                : manifest.ModelId;

        string? modelRootPath = state is ModelCacheState.Installed or ModelCacheState.Ready
            ? cacheRecord?.RootPath
            : null;
        ModelOptimizationAvailability optimizationAvailability = BuildOptimizationAvailability(
            manifest,
            state,
            modelRootPath,
            providerCapabilities);

        return new ModelInventoryEntry(
            ModelId: manifest.ModelId,
            DisplayName: displayName,
            Task: manifest.Task,
            EngineFamily: manifest.EngineFamily,
            License: manifest.License,
            CommercialAllowed: manifest.CommercialAllowed,
            CommercialUseVerified: manifest.CommercialUseVerified,
            RiskAccepted: manifest.RiskAccepted,
            State: state,
            FileSizeBytes: fileSize,
            CachedAtUtc: cacheRecord?.CachedAtUtc,
            FailureReason: failureReason,
            LanguageCoverageDisplay: ResolveLanguageCoverageDisplay(manifest),
            ExpectedRuntime: manifest.ExpectedRuntime,
            ExpectedRuntimeHint: ModelExpectedRuntimeFormatter.FormatHint(manifest.ExpectedRuntime),
            CanAutoDownload: canAutoDownload,
            ModelRootPath: modelRootPath,
            IsOliveOptimizable: optimizationAvailability.CanOptimize,
            OptimizationAvailability: optimizationAvailability,
            OptimizedVariants: BuildOptimizedVariants(cacheRecord),
            Aliases: manifest.Aliases,
            LanguagePairs: BuildLanguagePairs(manifest));
    }

    private static IReadOnlyList<Trackdub.Domain.ModelLanguagePair> BuildLanguagePairs(BundledModelManifestEntry manifest)
    {
        var pairs = new HashSet<(string Source, string Target)>(ValueTupleComparer.Instance);
        foreach (Trackdub.Inference.Runtime.ModelManifest.ModelLanguagePair pair in manifest.LanguageCoverage.LanguagePairs)
        {
            pairs.Add((pair.SourceLanguage, pair.TargetLanguage));
        }

        pairs.UnionWith(ExpandTargetPrefixes(manifest));
        return pairs
            .Select(pair => new Trackdub.Domain.ModelLanguagePair(pair.Source, pair.Target))
            .ToArray();
    }

    private static IEnumerable<(string Source, string Target)> ExpandTargetPrefixes(BundledModelManifestEntry manifest)
    {
        if (manifest.TargetPrefixes is not { Count: > 0 })
        {
            return [];
        }

        return manifest.LanguageCoverage.SourceLanguages
            .SelectMany(
                source => manifest.TargetPrefixes.Keys,
                (source, target) => (Source: source, Target: target))
            .Where(pair => !string.IsNullOrWhiteSpace(pair.Source) && !string.IsNullOrWhiteSpace(pair.Target));
    }

    private sealed class ValueTupleComparer : IEqualityComparer<(string Source, string Target)>
    {
        public static readonly ValueTupleComparer Instance = new();

        public bool Equals((string Source, string Target) x, (string Source, string Target) y) =>
            string.Equals(x.Source, y.Source, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(x.Target, y.Target, StringComparison.OrdinalIgnoreCase);

        public int GetHashCode((string Source, string Target) obj) =>
            HashCode.Combine(
                StringComparer.OrdinalIgnoreCase.GetHashCode(obj.Source ?? string.Empty),
                StringComparer.OrdinalIgnoreCase.GetHashCode(obj.Target ?? string.Empty));
    }

    private async Task<IReadOnlyList<ProviderCapability>> GetProviderCapabilitiesAsync(CancellationToken cancellationToken)
    {
        if (runtimeSelectionService is null)
        {
            return
            [
                new ProviderCapability
                {
                    Provider = ExecutionProviderKind.Cpu,
                    DeviceDetected = true,
                    ProviderLoadable = true
                }
            ];
        }

        return await runtimeSelectionService.GetCapabilitiesAsync(cancellationToken).ConfigureAwait(false);
    }

    private static IReadOnlyList<ModelOptimizedVariantInfo> BuildOptimizedVariants(
        LocalModelCacheRecord? cacheRecord)
    {
        if (cacheRecord is null || cacheRecord.Variants.Count == 0)
        {
            return [];
        }

        return cacheRecord.Variants
            .Select(BuildOptimizedVariantInfo)
            .OrderBy(variant => variant.Alias, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static ModelOptimizedVariantInfo BuildOptimizedVariantInfo(LocalModelVariantRecord variant)
    {
        ModelCacheState state = DetermineOptimizedVariantState(variant, out string? failureReason);
        return new ModelOptimizedVariantInfo(
            Alias: variant.Alias,
            OptimizerId: variant.OptimizerId,
            ExecutionProvider: variant.ExecutionProvider,
            Precision: variant.Precision,
            State: state,
            CreatedAtUtc: variant.CreatedAtUtc,
            RootPath: variant.RootPath,
            EntryRelativePath: variant.EntryRelativePath,
            ComponentRelativePaths: variant.ComponentRelativePaths,
            FailureReason: failureReason,
            Provenance: variant.Provenance);
    }

    private static ModelCacheState DetermineOptimizedVariantState(
        LocalModelVariantRecord variant,
        out string? failureReason)
    {
        failureReason = null;
        if (variant.IntegrityFailed)
        {
            failureReason = "Optimized variant failed integrity verification.";
            return ModelCacheState.Corrupt;
        }

        if (string.IsNullOrWhiteSpace(variant.RootPath) || !Directory.Exists(variant.RootPath))
        {
            failureReason = "Optimized variant path is missing.";
            return ModelCacheState.Corrupt;
        }

        if (!TryResolveOptimizedVariantPath(variant.RootPath, variant.EntryRelativePath, out string entryPath, out string? entryError))
        {
            failureReason = entryError;
            return ModelCacheState.Corrupt;
        }

        if (!File.Exists(entryPath))
        {
            failureReason = $"Optimized variant entry missing: {variant.EntryRelativePath}.";
            return ModelCacheState.Corrupt;
        }

        foreach (string componentRelativePath in variant.ComponentRelativePaths)
        {
            if (!TryResolveOptimizedVariantPath(variant.RootPath, componentRelativePath, out string componentPath, out string? componentError))
            {
                failureReason = componentError;
                return ModelCacheState.Corrupt;
            }

            if (!File.Exists(componentPath))
            {
                failureReason = $"Optimized variant component missing: {componentRelativePath}.";
                return ModelCacheState.Corrupt;
            }
        }

        return ModelCacheState.Ready;
    }

    private static bool TryResolveOptimizedVariantPath(
        string variantRootPath,
        string relativePath,
        out string fullPath,
        out string? error)
    {
        fullPath = string.Empty;
        error = null;

        if (string.IsNullOrWhiteSpace(relativePath) || IsRootedLikePath(relativePath))
        {
            error = $"Optimized variant path is invalid: {relativePath}.";
            return false;
        }

        string normalized = relativePath.Replace('\\', '/');
        if (normalized.Split('/').Any(segment => segment is "." or ".." || string.IsNullOrWhiteSpace(segment)))
        {
            error = $"Optimized variant path is invalid: {relativePath}.";
            return false;
        }

        bool isOnnx = normalized.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase);
        bool isGenAiConfig = Path.GetFileName(normalized).Equals("genai_config.json", StringComparison.OrdinalIgnoreCase);
        if (!isOnnx && !isGenAiConfig)
        {
            error = $"Optimized variant path must reference an ONNX file or genai_config.json: {relativePath}.";
            return false;
        }

        string root = Path.GetFullPath(variantRootPath);
        string candidate = Path.GetFullPath(Path.Join(root, normalized.Replace('/', Path.DirectorySeparatorChar)));
        if (!ModelDownloadPathGuard.IsStrictSubpathOrEqual(candidate, root))
        {
            error = $"Optimized variant path is invalid: {relativePath}.";
            return false;
        }

        fullPath = candidate;
        return true;
    }

    private static ModelOptimizationAvailability BuildOptimizationAvailability(
        BundledModelManifestEntry manifest,
        ModelCacheState state,
        string? modelRootPath,
        IReadOnlyList<ProviderCapability> providerCapabilities)
    {
        ModelOliveOptimizationProfile? profile = manifest.OliveOptimizationProfile;
        if (profile is null)
        {
            return ModelOptimizationAvailability.None;
        }

        string entryRelativePath = ResolveOptimizationEntryRelativePath(manifest, profile);
        bool isGenAiBuilder = profile.Mode.Equals("ort-genai-builder", StringComparison.OrdinalIgnoreCase);
        bool isExistingOnnx = profile.Mode.Equals("existing-onnx-components", StringComparison.OrdinalIgnoreCase);

        if (!isGenAiBuilder && !isExistingOnnx)
        {
            return Blocked(profile, entryRelativePath, "Optimization profile mode is not supported by this app version.");
        }

        if (state is not (ModelCacheState.Installed or ModelCacheState.Ready))
        {
            return Blocked(profile, entryRelativePath, "Download or repair the model before optimizing.");
        }

        if (string.IsNullOrWhiteSpace(modelRootPath))
        {
            return Blocked(profile, entryRelativePath, "Model cache path is unavailable.");
        }

        if (isGenAiBuilder)
        {
            string genAiConfigPath = manifest.DefaultBenchmarkEntryPath.EndsWith("genai_config.json", StringComparison.OrdinalIgnoreCase)
                ? ModelDownloadPathGuard.ResolveCachedManifestPath(manifest, modelRootPath, manifest.DefaultBenchmarkEntryPath)
                : Path.Join(modelRootPath, "genai_config.json");
            if (!File.Exists(genAiConfigPath))
            {
                return Blocked(profile, entryRelativePath, "genai_config.json not found — model may not be fully downloaded.");
            }
        }
        else
        {
            foreach (string componentPath in profile.Components)
            {
                if (!TryResolveRelativeComponentPath(modelRootPath, componentPath, out string fullPath, out string? error))
                {
                    return Blocked(profile, entryRelativePath, error ?? $"Optimization component path is invalid: {componentPath}.");
                }

                if (!File.Exists(fullPath))
                {
                    return Blocked(profile, entryRelativePath, $"Optimization component missing: {componentPath}.");
                }
            }
        }

        ExecutionProviderKind[] availableProviders = profile.SupportedProviders
            .Select(MapOptimizationProvider)
            .Where(provider => ProviderCapabilityAvailable(provider, providerCapabilities))
            .Distinct()
            .ToArray();

        if (availableProviders.Length == 0)
        {
            return Blocked(profile, entryRelativePath, "No supported Olive provider is available on this machine.");
        }

        return new ModelOptimizationAvailability(
            HasProfile: true,
            CanOptimize: true,
            ComponentRelativePaths: profile.Components,
            AvailableProviders: availableProviders,
            UnavailableReason: null,
            EntryRelativePath: entryRelativePath,
            Mode: profile.Mode,
            BaseVariantAliases: manifest.Variants.Select(v => v.Alias).ToArray(),
            SupportedPrecisions: NormalizePrecisions(profile.SupportedPrecisions),
            DeclaredOpset: manifest.Variants.FirstOrDefault(variant =>
                    variant.EntryPath.Equals(manifest.DefaultBenchmarkEntryPath, StringComparison.OrdinalIgnoreCase))
                ?.Opset,
            OpsetPolicies: profile.OpsetPolicies.Select(MapOpsetPolicy).ToArray(),
            RequireOpsetMetadata: profile.RequireOpsetMetadata,
            RecipeBindings: profile.RecipeBindings.Select(MapRecipeBinding).ToArray(),
            FallbackPolicy: MapFallbackPolicy(profile.FallbackPolicy));
    }

    private static ModelOptimizationAvailability Blocked(
        ModelOliveOptimizationProfile profile,
        string entryRelativePath,
        string reason) =>
        new(
            HasProfile: true,
            CanOptimize: false,
            ComponentRelativePaths: profile.Components,
            AvailableProviders: [],
            UnavailableReason: reason,
            EntryRelativePath: entryRelativePath,
            BaseVariantAliases: [],
            SupportedPrecisions: NormalizePrecisions(profile.SupportedPrecisions),
            DeclaredOpset: null,
            OpsetPolicies: profile.OpsetPolicies.Select(MapOpsetPolicy).ToArray(),
            RequireOpsetMetadata: profile.RequireOpsetMetadata,
            RecipeBindings: profile.RecipeBindings.Select(MapRecipeBinding).ToArray());

    private static ModelOptimizationRecipeBinding MapRecipeBinding(OliveRecipeBinding binding) =>
        new(
            binding.ConfigRelativePath,
            binding.Provider,
            binding.Precision,
            binding.Operations.Select(MapOptimizationOperation).ToArray(),
            MapExpectedOutput(binding.ExpectedOutput),
            binding.FallbackPolicy is null ? null : MapFallbackPolicy(binding.FallbackPolicy.Value),
            binding.QuantizationMethod,
            binding.RequiresCalibrationData,
            binding.ScriptRelativePath,
            binding.ScriptSha256,
            binding.Evaluator,
            binding.SplitCount,
            binding.CostModelRelativePath,
            binding.AdapterRelativePath,
            binding.AdapterMode,
            binding.OutputManifestRelativePath,
            binding.Component);

    private static ModelOptimizationOperation MapOptimizationOperation(OliveOptimizationOperation op) =>
        op switch
        {
            OliveOptimizationOperation.OnnxExport => ModelOptimizationOperation.OnnxExport,
            OliveOptimizationOperation.QnnConversion => ModelOptimizationOperation.QnnConversion,
            OliveOptimizationOperation.OpenVinoConversion => ModelOptimizationOperation.OpenVinoConversion,
            OliveOptimizationOperation.Compression => ModelOptimizationOperation.Compression,
            OliveOptimizationOperation.ProviderOptimization => ModelOptimizationOperation.ProviderOptimization,
            OliveOptimizationOperation.GenAiPackaging => ModelOptimizationOperation.GenAiPackaging,
            OliveOptimizationOperation.ModelSplitting => ModelOptimizationOperation.ModelSplitting,
            OliveOptimizationOperation.Evaluation => ModelOptimizationOperation.Evaluation,
            OliveOptimizationOperation.AdapterHandling => ModelOptimizationOperation.AdapterHandling,
            OliveOptimizationOperation.Registration => ModelOptimizationOperation.Registration,
            _ => throw new ArgumentOutOfRangeException(nameof(op), op, null)
        };

    private static ModelOptimizationExpectedOutput MapExpectedOutput(OliveRecipeExpectedOutput output) =>
        output switch
        {
            OliveRecipeExpectedOutput.OnnxComponents => ModelOptimizationExpectedOutput.OnnxComponents,
            OliveRecipeExpectedOutput.OrtGenAi => ModelOptimizationExpectedOutput.OrtGenAi,
            OliveRecipeExpectedOutput.QnnModelLibrary => ModelOptimizationExpectedOutput.QnnModelLibrary,
            OliveRecipeExpectedOutput.OpenVinoModel => ModelOptimizationExpectedOutput.OpenVinoModel,
            OliveRecipeExpectedOutput.SplitOnnxComponents => ModelOptimizationExpectedOutput.SplitOnnxComponents,
            OliveRecipeExpectedOutput.AdapterPackage => ModelOptimizationExpectedOutput.AdapterPackage,
            _ => throw new ArgumentOutOfRangeException(nameof(output), output, null)
        };

    private static ModelOptimizationFallbackPolicy MapFallbackPolicy(OliveRecipeFallbackPolicy policy) =>
        policy switch
        {
            OliveRecipeFallbackPolicy.None => ModelOptimizationFallbackPolicy.None,
            OliveRecipeFallbackPolicy.AutoOptAllowed => ModelOptimizationFallbackPolicy.AutoOptAllowed,
            OliveRecipeFallbackPolicy.BaseVariantAllowed => ModelOptimizationFallbackPolicy.BaseVariantAllowed,
            OliveRecipeFallbackPolicy.CpuRuntimeAllowed => ModelOptimizationFallbackPolicy.CpuRuntimeAllowed,
            _ => throw new ArgumentOutOfRangeException(nameof(policy), policy, null)
        };

    private static string ResolveOptimizationEntryRelativePath(
        BundledModelManifestEntry manifest,
        ModelOliveOptimizationProfile profile)
    {
        string entry = string.IsNullOrWhiteSpace(manifest.DefaultBenchmarkEntryPath)
            ? profile.Components[0]
            : Path.GetRelativePath(manifest.RootDirectory, manifest.DefaultBenchmarkEntryPath);
        return entry.Replace('\\', '/');
    }

    private static ExecutionProviderKind MapOptimizationProvider(OliveOptimizationProvider provider) =>
        provider switch
        {
            OliveOptimizationProvider.Dml => ExecutionProviderKind.DirectMl,
            OliveOptimizationProvider.Cuda => ExecutionProviderKind.Cuda,
            OliveOptimizationProvider.TensorRt => ExecutionProviderKind.TensorRt,
            OliveOptimizationProvider.TensorRtRtx => ExecutionProviderKind.TensorRTRtx,
            OliveOptimizationProvider.Migraphx or OliveOptimizationProvider.Rocm => ExecutionProviderKind.Migraphx,
            OliveOptimizationProvider.Qnn => ExecutionProviderKind.Qnn,
            OliveOptimizationProvider.OpenVino => ExecutionProviderKind.OpenVinoCatalog,
            OliveOptimizationProvider.VitisAi => ExecutionProviderKind.VitisAi,
            _ => ExecutionProviderKind.Cpu
        };

    private static ModelOptimizationOpsetPolicy MapOpsetPolicy(OliveOpsetPolicy policy) =>
        new(
            policy.Provider is null ? null : MapOptimizationProvider(policy.Provider.Value),
            NormalizePrecision(policy.Precision),
            policy.MinimumOpset);

    private static IReadOnlyList<string> NormalizePrecisions(IReadOnlyList<string> precisions) =>
        precisions
            .Select(NormalizePrecision)
            .OfType<string>()
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

    private static string? NormalizePrecision(string? precision)
    {
        if (string.IsNullOrWhiteSpace(precision))
        {
            return null;
        }

        return precision.Trim().ToLowerInvariant();
    }

    private static bool ProviderCapabilityAvailable(
        ExecutionProviderKind provider,
        IReadOnlyList<ProviderCapability> providerCapabilities)
    {
        ProviderCapability? capability = providerCapabilities.FirstOrDefault(capability => capability.Provider == provider);
        if (capability is null)
        {
            return provider == ExecutionProviderKind.Cpu;
        }

        return provider switch
        {
            ExecutionProviderKind.Cpu => capability.DeviceDetected,
            ExecutionProviderKind.Dnnl => capability.DeviceDetected && capability.ProviderLoadable,
            ExecutionProviderKind.DirectMl =>
                capability.DeviceDetected,
            ExecutionProviderKind.Cuda or ExecutionProviderKind.TensorRt =>
                capability.DeviceDetected && capability.ProviderLoadable,
            ExecutionProviderKind.TensorRTRtx =>
                capability.DeviceDetected && capability.ProviderLoadable,
            ExecutionProviderKind.Migraphx =>
                capability.DeviceDetected && capability.ProviderLoadable,
            ExecutionProviderKind.Qnn or ExecutionProviderKind.OpenVinoCatalog or ExecutionProviderKind.VitisAi =>
                capability.DeviceDetected && capability.ProviderLoadable,
            _ => false
        };
    }

    private static bool TryResolveRelativeComponentPath(
        string modelRootPath,
        string componentRelativePath,
        out string fullPath,
        out string? error)
    {
        fullPath = string.Empty;
        error = null;

        if (string.IsNullOrWhiteSpace(componentRelativePath) || IsRootedLikePath(componentRelativePath))
        {
            error = $"Optimization component path is invalid: {componentRelativePath}.";
            return false;
        }

        string normalized = componentRelativePath.Replace('\\', '/');
        if (normalized.Split('/').Any(segment => segment is "." or ".." || string.IsNullOrWhiteSpace(segment)))
        {
            error = $"Optimization component path is invalid: {componentRelativePath}.";
            return false;
        }

        string root = Path.GetFullPath(modelRootPath);
        string candidate = Path.GetFullPath(Path.Join(root, normalized.Replace('/', Path.DirectorySeparatorChar)));
        if (!ModelDownloadPathGuard.IsStrictSubpathOrEqual(candidate, root))
        {
            error = $"Optimization component path is invalid: {componentRelativePath}.";
            return false;
        }

        fullPath = candidate;
        return true;
    }

    private static bool IsRootedLikePath(string path)
    {
        string normalized = path.Replace('\\', '/');
        return Path.IsPathRooted(path) ||
            normalized.StartsWith('/') ||
            (normalized.Length >= 2 && char.IsLetter(normalized[0]) && normalized[1] == ':');
    }

    private static string? ResolveLanguageCoverageDisplay(BundledModelManifestEntry manifest)
    {
        if (!manifest.Task.Equals("translation", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        ModelLanguageCoverage coverage = manifest.LanguageCoverage;
        if (IsMultilingualCoverage(coverage))
        {
            return "Language scope: Multilingual";
        }

        if (coverage.LanguagePairs.Count == 1)
        {
            Trackdub.Inference.Runtime.ModelManifest.ModelLanguagePair pair = coverage.LanguagePairs[0];
            return $"Language scope: {FormatLanguage(pair.SourceLanguage)} -> {FormatLanguage(pair.TargetLanguage)} only";
        }

        if (coverage.LanguagePairs.Count > 1)
        {
            string? sharedSourceDisplay = TryFormatSharedSourceLanguagePairs(coverage.LanguagePairs);
            return sharedSourceDisplay ?? $"Language scope: {coverage.LanguagePairs.Count} direct language pairs only";
        }

        if (coverage.SourceLanguages.Count > 0 || coverage.TargetLanguages.Count > 0)
        {
            string source = FormatLanguageList(coverage.SourceLanguages);
            string target = FormatLanguageList(coverage.TargetLanguages);
            return $"Language scope: {source} -> {target} only";
        }

        return "Language scope: Not declared";
    }

    private static bool IsMultilingualCoverage(ModelLanguageCoverage coverage) =>
        ContainsMultilingualMarker(coverage.SourceLanguages) ||
        ContainsMultilingualMarker(coverage.TargetLanguages);

    private static bool ContainsMultilingualMarker(IEnumerable<string> languages) =>
        languages.Any(language =>
            language.Equals("multi", StringComparison.OrdinalIgnoreCase) ||
            language.Equals("multilingual", StringComparison.OrdinalIgnoreCase));

    private static string FormatLanguageList(IReadOnlyList<string> languages) =>
        languages.Count switch
        {
            0 => "unspecified",
            1 => FormatLanguage(languages[0]),
            _ => string.Join(", ", languages.Select(FormatLanguage))
        };

    private static string FormatLanguage(string languageCode) =>
        languageCode.Trim().ToLowerInvariant() switch
        {
            "auto" => "Auto-detect",
            "multi" or "multilingual" => "Multilingual",
            "en" => "English",
            "es" => "Spanish",
            "fr" => "French",
            "de" => "German",
            "it" => "Italian",
            "pt" => "Portuguese",
            "ro" => "Romanian",
            "" => "unspecified",
            var code => code.ToUpperInvariant()
        };

    private static string? TryFormatSharedSourceLanguagePairs(IReadOnlyList<Trackdub.Inference.Runtime.ModelManifest.ModelLanguagePair> languagePairs)
    {
        string? sourceLanguage = languagePairs[0].SourceLanguage;
        if (languagePairs.Any(pair => !pair.SourceLanguage.Equals(sourceLanguage, StringComparison.OrdinalIgnoreCase)))
        {
            return null;
        }

        string[] targetLanguages = languagePairs
            .Select(pair => pair.TargetLanguage)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Select(FormatLanguage)
            .ToArray();

        return targetLanguages.Length > 1
            ? $"Language scope: {FormatLanguage(sourceLanguage)} -> {string.Join(", ", targetLanguages)} only"
            : null;
    }

    internal static ModelCacheState DetermineState(BundledModelManifestEntry manifest, LocalModelCacheRecord? cacheRecord)
    {
        if (cacheRecord is null)
            return ModelCacheState.Missing;

        if (cacheRecord.IntegrityFailed || HasManifestHashMismatch(manifest, cacheRecord))
            return ModelCacheState.Corrupt;

        string modelRootDirectory = cacheRecord.RootPath;
        string benchmarkEntryPath;
        try
        {
            benchmarkEntryPath = ResolveCachedBenchmarkEntryPath(manifest, cacheRecord);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException or InvalidOperationException)
        {
            // Malformed legacy roots (e.g. embedded null characters) must not block the
            // selection of other valid installations in the same index.
            return ModelCacheState.Corrupt;
        }

        if (!Directory.Exists(modelRootDirectory) && !File.Exists(benchmarkEntryPath))
            return ModelCacheState.Corrupt;

        if (Directory.Exists(modelRootDirectory))
        {
            try
            {
                if (!Directory.EnumerateFileSystemEntries(modelRootDirectory).Any())
                    return ModelCacheState.Corrupt;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return ModelCacheState.Corrupt;
            }

            // Verify the primary model file is actually present
            if (!File.Exists(benchmarkEntryPath))
                return ModelCacheState.Corrupt;
        }

        return ModelCacheState.Installed;
    }

    internal static LocalModelCacheRecord? SelectBestCacheRecord(
        BundledModelManifestEntry manifest,
        IReadOnlyList<LocalModelCacheRecord>? cacheRecords,
        string configuredModelCacheDirectory)
    {
        if (cacheRecords is null || cacheRecords.Count == 0)
            return null;

        IEnumerable<LocalModelCacheRecord> ordered = cacheRecords
            .OrderByDescending(record => IsUnderConfiguredCache(record.RootPath, configuredModelCacheDirectory) ? 1 : 0)
            .ThenByDescending(record =>
                DetermineState(manifest, record) == ModelCacheState.Installed ? 1 : 0)
            .ThenByDescending(record => record.CachedAtUtc)
            .ThenBy(record => record.RootPath, OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);

        return ordered.FirstOrDefault(record =>
                DetermineState(manifest, record) == ModelCacheState.Installed)
            ?? ordered.FirstOrDefault();
    }

    private static bool IsUnderConfiguredCache(string rootPath, string configuredModelCacheDirectory)
    {
        try
        {
            return ModelDownloadPathGuard.IsModelRootUnderConfiguredCache(rootPath, configuredModelCacheDirectory, out _);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
        {
            // A malformed legacy root must not abort selection of other valid records.
            return false;
        }
    }

    private static bool HasManifestHashMismatch(BundledModelManifestEntry manifest, LocalModelCacheRecord record) =>
        !string.IsNullOrWhiteSpace(manifest.Sha256) && !string.IsNullOrWhiteSpace(record.Sha256) &&
        !string.Equals(manifest.Sha256, record.Sha256, StringComparison.OrdinalIgnoreCase);

    private static string ResolveCachedBenchmarkEntryPath(
        BundledModelManifestEntry manifest,
        LocalModelCacheRecord cacheRecord) =>
        ModelDownloadPathGuard.ResolveCachedManifestPath(
            manifest,
            cacheRecord.RootPath,
            manifest.DefaultBenchmarkEntryPath);

    private static long? TryGetFileSize(string path)
    {
        try
        {
            return File.Exists(path) ? new FileInfo(path).Length : null;
        }
        catch
        {
            return null;
        }
    }
}
