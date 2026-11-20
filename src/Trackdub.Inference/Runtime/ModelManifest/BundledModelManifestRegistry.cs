namespace Trackdub.Inference.Runtime.ModelManifest;

public sealed class BundledModelManifestRegistry
{
    private IReadOnlyDictionary<string, BundledModelManifestEntry> aliasIndex;
    private readonly object catalogGate = new();
    private IReadOnlyList<BundledModelManifestEntry> entries = [];

    private BundledModelManifestRegistry(
        string manifestPath,
        IReadOnlyList<BundledModelManifestEntry> entries,
        IReadOnlyDictionary<string, BundledModelManifestEntry> aliasIndex)
    {
        ManifestPath = manifestPath;
        this.entries = entries;
        this.aliasIndex = aliasIndex;
    }

    public string ManifestPath { get; }

    public IReadOnlyList<BundledModelManifestEntry> Entries => Volatile.Read(ref entries);

    /// <summary>Adds validated catalog entries without changing any already published model revision.</summary>
    public void AddCatalogEntries(IEnumerable<BundledModelManifestEntry> additions)
    {
        lock (catalogGate)
        {
            var merged = Entries.ToList();
            var aliases = new Dictionary<string, BundledModelManifestEntry>(aliasIndex, StringComparer.OrdinalIgnoreCase);
            foreach (var entry in additions)
            {
                if (merged.Any(existing => existing.ModelId.Equals(entry.ModelId, StringComparison.OrdinalIgnoreCase)) ||
                    entry.Aliases.Any(aliases.ContainsKey)) continue;
                merged.Add(entry);
                foreach (string alias in entry.Aliases) aliases.Add(alias, entry);
            }
            Volatile.Write(ref aliasIndex, aliases);
            Volatile.Write(ref entries, merged.ToArray());
        }
    }

    public static bool TryLoadDefault(out BundledModelManifestRegistry? registry, out string? error)
    {
        string? manifestPath = LocateDefaultManifestPath();
        if (manifestPath is null)
        {
            registry = null;
            error = "Bundled model manifest was not found.";
            return false;
        }

        try
        {
            registry = LoadWithDefaultFragments(manifestPath);
            error = null;
            return true;
        }
        catch (Exception ex) when (ex is IOException or ModelManifestValidationException or InvalidOperationException)
        {
            registry = null;
            error = AppendSourceTreeToolHint(manifestPath, ex.Message);
            return false;
        }
    }

    public static BundledModelManifestRegistry Load(string manifestPath)
    {
        return LoadMany([Path.GetFullPath(manifestPath)]);
    }

    public static BundledModelManifestRegistry LoadWithFragments(string manifestPath, string fragmentDirectory)
    {
        string fullManifestPath = Path.GetFullPath(manifestPath);
        var manifestPaths = new List<string> { fullManifestPath };
        if (Directory.Exists(fragmentDirectory))
        {
            manifestPaths.AddRange(Directory
                .EnumerateFiles(fragmentDirectory, "*.manifest.json", SearchOption.TopDirectoryOnly)
                .Order(StringComparer.OrdinalIgnoreCase)
                .Select(Path.GetFullPath));
        }

        return LoadMany(manifestPaths);
    }

    private static BundledModelManifestRegistry LoadWithDefaultFragments(string manifestPath)
    {
        string fullManifestPath = Path.GetFullPath(manifestPath);
        string? fragmentDirectory = LocateDefaultFragmentDirectory(fullManifestPath);
        return fragmentDirectory is null
            ? Load(fullManifestPath)
            : LoadWithFragments(fullManifestPath, fragmentDirectory);
    }

    private static BundledModelManifestRegistry LoadMany(IReadOnlyList<string> manifestPaths)
    {
        if (manifestPaths.Count == 0)
        {
            throw new InvalidOperationException("At least one bundled model manifest path is required.");
        }

        var mergedEntries = new List<BundledModelManifestEntry>();
        foreach (string manifestPath in manifestPaths)
        {
            ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);
            foreach (ModelManifest model in catalog.Models)
            {
                BundledModelManifestEntry entry = NormalizeEntry(model, manifestPath);
                int existingIndex = mergedEntries.FindIndex(existing =>
                    existing.ModelId.Equals(entry.ModelId, StringComparison.OrdinalIgnoreCase) &&
                    existing.RootDirectory.Equals(entry.RootDirectory, StringComparison.OrdinalIgnoreCase));
                if (existingIndex >= 0)
                {
                    // A fragment that omits "deprecated" is silent, not an assertion of "not
                    // deprecated"; inherit the base entry's value instead of rejecting the merge.
                    if (model.Deprecated is null)
                    {
                        entry = entry with { Deprecated = mergedEntries[existingIndex].Deprecated };
                    }
                    mergedEntries[existingIndex] = MergeEntry(mergedEntries[existingIndex], entry, manifestPath);
                }
                else
                {
                    mergedEntries.Add(entry);
                }
            }
        }

        BundledModelManifestEntry[] entries = mergedEntries.ToArray();

        var aliasIndex = new Dictionary<string, BundledModelManifestEntry>(StringComparer.OrdinalIgnoreCase);
        foreach (BundledModelManifestEntry entry in entries)
        {
            if (entry.Aliases.Count == 0)
            {
                throw new InvalidOperationException($"Model '{entry.ModelId}' in bundled model manifests did not define any aliases.");
            }

            foreach (string alias in entry.Aliases)
            {
                if (!aliasIndex.TryAdd(alias, entry))
                {
                    throw new InvalidOperationException($"Alias '{alias}' is defined more than once across bundled model manifests.");
                }
            }
        }

        return new BundledModelManifestRegistry(manifestPaths[0], entries, aliasIndex);
    }

    internal static BundledModelManifestRegistry CreateForTests(
        string manifestPath,
        IReadOnlyList<BundledModelManifestEntry> entries)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(manifestPath);
        ArgumentNullException.ThrowIfNull(entries);

        var aliasIndex = new Dictionary<string, BundledModelManifestEntry>(StringComparer.OrdinalIgnoreCase);
        foreach (BundledModelManifestEntry entry in entries)
        {
            if (entry.Aliases.Count == 0)
            {
                throw new InvalidOperationException($"Model '{entry.ModelId}' did not define any aliases.");
            }

            foreach (string alias in entry.Aliases)
            {
                if (!aliasIndex.TryAdd(alias, entry))
                {
                    throw new InvalidOperationException($"Alias '{alias}' is defined more than once.");
                }
            }
        }

        return new BundledModelManifestRegistry(manifestPath, entries, aliasIndex);
    }

    public bool TryResolve(string reference, out BundledModelManifestResolution? resolution)
    {
        resolution = null;
        if (string.IsNullOrWhiteSpace(reference))
        {
            return false;
        }

        string trimmedReference = reference.Trim();
        string alias = trimmedReference;
        string? variantAlias = null;

        int variantSeparatorIndex = trimmedReference.IndexOf('@');
        if (variantSeparatorIndex >= 0)
        {
            alias = trimmedReference[..variantSeparatorIndex];
            variantAlias = trimmedReference[(variantSeparatorIndex + 1)..];
        }

        if (!aliasIndex.TryGetValue(alias, out BundledModelManifestEntry? entry))
        {
            entry = Entries.FirstOrDefault(
                candidate => candidate.ModelId.Equals(alias, StringComparison.OrdinalIgnoreCase));
            if (entry is null)
            {
                return false;
            }

            alias = entry.Aliases[0];
        }

        string resolvedEntryPath = entry.DefaultBenchmarkEntryPath;
        string resolvedVariantAlias = entry.Variants.Any(v => v.Alias.Equals("default", StringComparison.OrdinalIgnoreCase))
            ? "default"
            : string.Empty;
        if (!string.IsNullOrWhiteSpace(variantAlias))
        {
            BundledModelManifestVariant? variant = entry.Variants
                .FirstOrDefault(candidate => candidate.Alias.Equals(variantAlias, StringComparison.OrdinalIgnoreCase));

            if (variant is null)
            {
                throw new FileNotFoundException(
                    $"Model alias '{alias}' does not define benchmark variant '{variantAlias}'.",
                    trimmedReference);
            }

            resolvedEntryPath = variant.EntryPath;
            resolvedVariantAlias = variant.Alias;
        }

        resolution = new BundledModelManifestResolution(
            entry,
            trimmedReference,
            alias,
            resolvedVariantAlias,
            resolvedEntryPath);
        return true;
    }

    private static BundledModelManifestEntry NormalizeEntry(ModelManifest model, string manifestPath)
    {
        string manifestDirectory = Path.GetDirectoryName(manifestPath)
            ?? throw new InvalidOperationException($"Manifest path '{manifestPath}' did not have a parent directory.");

        if (string.IsNullOrWhiteSpace(model.RootPath))
        {
            throw new InvalidOperationException($"Model '{model.ModelId}' in '{manifestPath}' did not define root_path.");
        }

        if (string.IsNullOrWhiteSpace(model.BenchmarkEntry))
        {
            throw new InvalidOperationException($"Model '{model.ModelId}' in '{manifestPath}' did not define benchmark_entry.");
        }

        string rootDirectory = ResolveRootDirectory(manifestDirectory, model);
        string defaultBenchmarkEntryPath = Path.GetFullPath(Path.Join(rootDirectory, model.BenchmarkEntry));
        BundledModelManifestVariant[] variants = model.Variants
            .Select(variant => new BundledModelManifestVariant(
                variant.Alias,
                Path.GetFullPath(Path.Join(rootDirectory, variant.EntryPath)),
                variant.DownloadFiles,
                variant.DisplayName,
                variant.Description,
                variant.IsDefault,
                variant.SupportedProviders,
                variant.Opset))
            .ToArray();

        return new BundledModelManifestEntry(
            ModelId: model.ModelId,
            Task: model.Task.ToManifestValue(),
            EngineFamily: model.EngineFamily,
            Capabilities: model.Capabilities,
            LanguageCoverage: model.LanguageCoverage,
            Tier: model.Tier,
            Lane: model.Lane,
            License: model.License.ToManifestValue(),
            CommercialAllowed: model.CommercialAllowed,
            RedistributionAllowed: model.RedistributionAllowed,
            RequiresAttribution: model.RequiresAttribution,
            RequiresUserConsent: model.RequiresUserConsent,
            VoiceCloning: model.VoiceCloning,
            CommercialUseVerified: model.CommercialUseVerified,
            SourceUrl: model.SourceUrl,
            Revision: model.Revision,
            Sha256: model.Sha256,
            DownloadFiles: model.DownloadFiles,
            DownloadFileSources: NormalizeDownloadFileSources(model.DownloadFileSources),
            DownloadFileHashes: NormalizeDownloadFileHashes(model.DownloadFileHashes),
            Aliases: model.Aliases,
            RootDirectory: rootDirectory,
            DefaultBenchmarkEntryPath: defaultBenchmarkEntryPath,
            Variants: variants,
            EstimatedVramMb: model.EstimatedVramMb,
            MinVramMb: model.MinVramMb,
            SupportsPartialOffload: model.SupportsPartialOffload,
            DisplayName: model.DisplayName,
            OliveOptimizable: model.OliveOptimizable,
            OliveOptimizationProfile: model.Optimization?.Olive,
            ProviderId: model.ProviderId,
            ExpectedRuntime: model.ExpectedRuntime,
            Deprecated: model.Deprecated ?? false,
            DeprecatedReason: model.DeprecatedReason)
        {
            RiskAcceptance = model.RiskAcceptance
        };
    }

    private static bool HasMatchingGovernance(BundledModelManifestEntry existing, BundledModelManifestEntry incoming) =>
        existing.Task.Equals(incoming.Task, StringComparison.OrdinalIgnoreCase) &&
        existing.EngineFamily.Equals(incoming.EngineFamily, StringComparison.OrdinalIgnoreCase) &&
        existing.Lane == incoming.Lane &&
        HasMatchingLicensing(existing, incoming) &&
        existing.VoiceCloning == incoming.VoiceCloning &&
        existing.Deprecated == incoming.Deprecated;

    private static bool HasMatchingLicensing(BundledModelManifestEntry existing, BundledModelManifestEntry incoming) =>
        existing.License.Equals(incoming.License, StringComparison.OrdinalIgnoreCase) &&
        existing.CommercialAllowed == incoming.CommercialAllowed &&
        existing.RedistributionAllowed == incoming.RedistributionAllowed &&
        existing.RequiresAttribution == incoming.RequiresAttribution &&
        existing.RequiresUserConsent == incoming.RequiresUserConsent;

    private static BundledModelManifestEntry MergeEntry(
        BundledModelManifestEntry existing,
        BundledModelManifestEntry incoming,
        string incomingManifestPath)
    {
        if (!HasMatchingGovernance(existing, incoming))
        {
            throw new InvalidOperationException(
                $"Generated manifest '{incomingManifestPath}' cannot merge model '{incoming.ModelId}' because its governance metadata does not match the base entry.");
        }

        ModelRiskAcceptanceReader.EnsureSameRiskAcceptance(existing, incoming, incomingManifestPath);

        if (!existing.RootDirectory.Equals(incoming.RootDirectory, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                $"Generated manifest '{incomingManifestPath}' cannot merge model '{incoming.ModelId}' because root_path resolves to '{incoming.RootDirectory}' instead of '{existing.RootDirectory}'.");
        }

        if (HasConflictingManifestMetadata(existing.ProviderId, incoming.ProviderId))
        {
            throw new InvalidOperationException(
                $"Generated manifest '{incomingManifestPath}' cannot merge model '{incoming.ModelId}' because provider_id '{incoming.ProviderId}' does not match the base entry '{existing.ProviderId}'.");
        }

        if (HasConflictingManifestMetadata(existing.ExpectedRuntime, incoming.ExpectedRuntime))
        {
            throw new InvalidOperationException(
                $"Generated manifest '{incomingManifestPath}' cannot merge model '{incoming.ModelId}' because expected_runtime '{incoming.ExpectedRuntime}' does not match the base entry '{existing.ExpectedRuntime}'.");
        }

        string[] aliases = existing.Aliases
            .Concat(incoming.Aliases)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        string[] downloadFiles = existing.DownloadFiles
            .Concat(incoming.DownloadFiles)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        Dictionary<string, string> downloadFileSources = MergeDownloadFileSources(
            existing.DownloadFileSources,
            incoming.DownloadFileSources,
            incoming.ModelId,
            incomingManifestPath);
        Dictionary<string, string> downloadFileHashes = MergeDownloadFileHashes(
            existing.DownloadFileHashes,
            incoming.DownloadFileHashes,
            incoming.ModelId,
            incomingManifestPath);
        BundledModelManifestVariant[] variants = MergeVariants(existing, incoming, incomingManifestPath);

        return existing with
        {
            DownloadFiles = downloadFiles,
            DownloadFileSources = downloadFileSources,
            DownloadFileHashes = downloadFileHashes,
            Aliases = aliases,
            Variants = variants,
            OliveOptimizationProfile = MergeOptimizationProfile(existing, incoming, incomingManifestPath),
            ProviderId = !string.IsNullOrWhiteSpace(existing.ProviderId) ? existing.ProviderId : incoming.ProviderId,
            ExpectedRuntime = !string.IsNullOrWhiteSpace(existing.ExpectedRuntime) ? existing.ExpectedRuntime : incoming.ExpectedRuntime,
            EstimatedVramMb = existing.EstimatedVramMb != 0 ? existing.EstimatedVramMb : incoming.EstimatedVramMb,
            MinVramMb = existing.MinVramMb != 0 ? existing.MinVramMb : incoming.MinVramMb,
            SupportsPartialOffload = existing.SupportsPartialOffload || incoming.SupportsPartialOffload
        };
    }

    private static bool HasConflictingManifestMetadata(string? existingValue, string? incomingValue) =>
        !string.IsNullOrWhiteSpace(existingValue) &&
        !string.IsNullOrWhiteSpace(incomingValue) &&
        !existingValue.Equals(incomingValue, StringComparison.OrdinalIgnoreCase);

    private static ModelOliveOptimizationProfile? MergeOptimizationProfile(
        BundledModelManifestEntry existing,
        BundledModelManifestEntry incoming,
        string incomingManifestPath)
    {
        if (existing.OliveOptimizationProfile is null)
        {
            return incoming.OliveOptimizationProfile;
        }

        if (incoming.OliveOptimizationProfile is null)
        {
            return existing.OliveOptimizationProfile;
        }

        if (OptimizationProfilesEqual(existing.OliveOptimizationProfile, incoming.OliveOptimizationProfile))
        {
            return existing.OliveOptimizationProfile;
        }

        throw new InvalidOperationException(
            $"Generated manifest '{incomingManifestPath}' cannot merge model '{incoming.ModelId}' because its Olive optimization profile conflicts with the base entry.");
    }

    private static bool OptimizationProfilesEqual(
        ModelOliveOptimizationProfile left,
        ModelOliveOptimizationProfile right) =>
        left.Mode.Equals(right.Mode, StringComparison.OrdinalIgnoreCase) &&
        left.Components.SequenceEqual(right.Components, StringComparer.OrdinalIgnoreCase) &&
        left.SupportedProviders.SequenceEqual(right.SupportedProviders) &&
        left.SupportedPrecisions.SequenceEqual(right.SupportedPrecisions, StringComparer.OrdinalIgnoreCase) &&
        left.RequireOpsetMetadata == right.RequireOpsetMetadata &&
        left.OpsetPolicies.SequenceEqual(right.OpsetPolicies) &&
        left.RecipeBindings.SequenceEqual(right.RecipeBindings);

    private static Dictionary<string, string> MergeDownloadFileSources(
        IReadOnlyDictionary<string, string> existing,
        IReadOnlyDictionary<string, string> incoming,
        string modelId,
        string incomingManifestPath)
    {
        var merged = new Dictionary<string, string>(existing, StringComparer.OrdinalIgnoreCase);
        foreach ((string relativePath, string sourceUri) in incoming)
        {
            if (merged.TryGetValue(relativePath, out string? existingSourceUri) &&
                !existingSourceUri.Equals(sourceUri, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"Generated manifest '{incomingManifestPath}' cannot merge model '{modelId}' because download source '{relativePath}' conflicts with the base entry.");
            }

            merged[relativePath] = sourceUri;
        }

        return merged;
    }

    private static Dictionary<string, string> MergeDownloadFileHashes(
        IReadOnlyDictionary<string, string> existing,
        IReadOnlyDictionary<string, string> incoming,
        string modelId,
        string incomingManifestPath)
    {
        var merged = new Dictionary<string, string>(existing, StringComparer.OrdinalIgnoreCase);
        foreach ((string relativePath, string sha256) in incoming)
        {
            if (merged.TryGetValue(relativePath, out string? existingSha256) &&
                !existingSha256.Equals(sha256, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException(
                    $"Generated manifest '{incomingManifestPath}' cannot merge model '{modelId}' because download hash '{relativePath}' conflicts with the base entry.");
            }

            merged[relativePath] = sha256;
        }

        return merged;
    }

    private static BundledModelManifestVariant[] MergeVariants(
        BundledModelManifestEntry existing,
        BundledModelManifestEntry incoming,
        string incomingManifestPath)
    {
        var variants = new Dictionary<string, BundledModelManifestVariant>(StringComparer.OrdinalIgnoreCase);
        foreach (BundledModelManifestVariant variant in existing.Variants)
        {
            variants.Add(variant.Alias, variant);
        }

        foreach (BundledModelManifestVariant variant in incoming.Variants)
        {
            if (variants.ContainsKey(variant.Alias))
            {
                throw new InvalidOperationException(
                    $"Generated manifest '{incomingManifestPath}' cannot merge model '{incoming.ModelId}' because variant alias '{variant.Alias}' already exists.");
            }

            variants.Add(variant.Alias, variant);
        }

        return variants.Values.ToArray();
    }

    internal static string AppendSourceTreeToolHint(string manifestPath, string message)
    {
        string normalized = manifestPath.Replace('\\', '/');
        if (!normalized.Contains("/src/Trackdub.Inference/Runtime/ModelManifest/", StringComparison.OrdinalIgnoreCase))
        {
            return message;
        }

        return message
            + " This is the repo source-tree manifest, resolved from the current directory."
            + " A globally installed trackdub tool must be rebuilt from this checkout,"
            + " or run `dotnet run --project src/Trackdub.Cli` instead.";
    }

    private static string? LocateDefaultManifestPath()
    {
        foreach (string seed in new[] { Environment.CurrentDirectory, AppContext.BaseDirectory })
        {
            foreach (string ancestor in EnumerateAncestors(seed))
            {
                string candidate = Path.Join(
                    ancestor,
                    "src",
                    "Trackdub.Inference",
                    "Runtime",
                    "ModelManifest",
                    "bundled-models.manifest.json");

                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }
        }

        return null;
    }

    private static string? LocateDefaultFragmentDirectory(string manifestPath)
    {
        string? current = Path.GetDirectoryName(manifestPath);
        while (current is not null)
        {
            string candidate = Path.Join(current, "models", "manifest-fragments");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }

            if (File.Exists(Path.Join(current, "Trackdub.slnx")) ||
                Directory.Exists(Path.Join(current, ".git")))
            {
                return null;
            }

            current = Path.GetDirectoryName(current);
        }

        return null;
    }

    private static string ResolveRootDirectory(string manifestDirectory, ModelManifest model)
    {
        if (string.IsNullOrWhiteSpace(model.RootPath))
        {
            return manifestDirectory;
        }

        // Default behavior: relative to manifest as defined in json (usually ../../../../models/...).
        // model.RootPath may be an absolute override outside the configured cache; Path.Combine's
        // rooted-path reset is required here, so this call is intentionally not Path.Join.
#pragma warning disable RS0030
        string defaultPath = Path.GetFullPath(Path.Combine(manifestDirectory, model.RootPath));
#pragma warning restore RS0030
        if (Directory.Exists(defaultPath))
        {
            return defaultPath;
        }

        // Robust fallback: Find the repository model root by searching for 'models'.
        string? current = manifestDirectory;
        while (current is not null)
        {
            string candidateModels = Path.Join(current, "models");
            if (Directory.Exists(candidateModels))
            {
                string modelPathWithinModels = GetModelPathWithinModels(model.RootPath);
                if (!string.IsNullOrWhiteSpace(modelPathWithinModels))
                {
                    string specificModelPath = Path.Join(candidateModels, modelPathWithinModels);
                    if (Directory.Exists(specificModelPath))
                    {
                        return Path.GetFullPath(specificModelPath);
                    }
                }
            }

            if (Directory.Exists(Path.Join(current, ".git")))
            {
                break;
            }

            current = Path.GetDirectoryName(current);
        }

        return defaultPath;
    }

    private static string GetModelPathWithinModels(string rootPath)
    {
        string[] pathSegments = rootPath
            .Replace('\\', '/')
            .Trim('/')
            .Split('/', StringSplitOptions.RemoveEmptyEntries);

        if (pathSegments.Length == 0)
        {
            return string.Empty;
        }

        int modelsSegmentIndex = Array.FindLastIndex(
            pathSegments,
            segment => string.Equals(segment, "models", StringComparison.OrdinalIgnoreCase));

        if (modelsSegmentIndex >= 0 && modelsSegmentIndex + 1 < pathSegments.Length)
        {
            return Path.Join(pathSegments[(modelsSegmentIndex + 1)..]);
        }

        return pathSegments[^1];
    }

    private static IReadOnlyDictionary<string, string> NormalizeDownloadFileSources(
        IReadOnlyDictionary<string, string> downloadFileSources)
    {
        var normalized = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach ((string relativePath, string sourceUri) in downloadFileSources)
        {
            string normalizedRelativePath = relativePath.Replace('\\', '/').Trim('/');
            if (string.IsNullOrWhiteSpace(normalizedRelativePath) ||
                Path.IsPathRooted(normalizedRelativePath) ||
                normalizedRelativePath.Split('/').Any(part => part is "." or ".." || string.IsNullOrWhiteSpace(part)))
            {
                throw new InvalidOperationException($"Manifest download_file_sources key '{relativePath}' must be a safe relative path.");
            }

            if (!Uri.TryCreate(sourceUri, UriKind.Absolute, out Uri? uri) ||
                uri.Scheme is not ("http" or "https"))
            {
                throw new InvalidOperationException($"Manifest download source for '{relativePath}' must be an absolute HTTP(S) URI.");
            }

            normalized.Add(normalizedRelativePath, uri.AbsoluteUri);
        }

        return normalized;
    }

    private static IReadOnlyDictionary<string, string> NormalizeDownloadFileHashes(
        IReadOnlyDictionary<string, string> downloadFileHashes)
    {
        var normalized = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach ((string relativePath, string sha256) in downloadFileHashes)
        {
            string normalizedRelativePath = relativePath.Replace('\\', '/').Trim('/');
            if (string.IsNullOrWhiteSpace(normalizedRelativePath) ||
                Path.IsPathRooted(normalizedRelativePath) ||
                normalizedRelativePath.Split('/').Any(part => part is "." or ".." || string.IsNullOrWhiteSpace(part)))
            {
                throw new InvalidOperationException($"Manifest download_file_hashes key '{relativePath}' must be a safe relative path.");
            }

            normalized.Add(normalizedRelativePath, sha256.Trim().ToLowerInvariant());
        }

        return normalized;
    }

    private static IEnumerable<string> EnumerateAncestors(string path)
    {
        DirectoryInfo? current = new DirectoryInfo(Path.GetFullPath(path));
        while (current is not null)
        {
            yield return current.FullName;
            current = current.Parent;
        }
    }
}

public sealed record BundledModelManifestEntry(
    string ModelId,
    string Task,
    string EngineFamily,
    IReadOnlyList<string> Capabilities,
    ModelLanguageCoverage LanguageCoverage,
    string Tier,
    ModelLane Lane,
    string License,
    bool CommercialAllowed,
    bool RedistributionAllowed,
    bool RequiresAttribution,
    bool RequiresUserConsent,
    bool VoiceCloning,
    bool CommercialUseVerified,
    string SourceUrl,
    string Revision,
    string Sha256,
    IReadOnlyList<string> DownloadFiles,
    IReadOnlyDictionary<string, string> DownloadFileSources,
    IReadOnlyDictionary<string, string> DownloadFileHashes,
    IReadOnlyList<string> Aliases,
    string RootDirectory,
    string DefaultBenchmarkEntryPath,
    IReadOnlyList<BundledModelManifestVariant> Variants,
    int EstimatedVramMb = 0,
    int MinVramMb = 0,
    bool SupportsPartialOffload = false,
    string? DisplayName = null,
    bool OliveOptimizable = false,
    ModelOliveOptimizationProfile? OliveOptimizationProfile = null,
    string? ProviderId = null,
    string? ExpectedRuntime = null,
    bool Deprecated = false,
    string? DeprecatedReason = null,
    long? ExpectedDownloadBytes = null,
    string? CatalogPolicyVersion = null,
    IReadOnlyDictionary<string, string?>? TargetPrefixes = null)
{
    public bool CommercialSafeMode => ModelRiskAcceptanceReader.IsCommercialSafe(CommercialUseVerified, RiskAcceptance);

    public ModelRiskAcceptance? RiskAcceptance { get; init; }
}

public sealed record BundledModelManifestVariant(
    string Alias,
    string EntryPath,
    IReadOnlyList<string> DownloadFiles,
    string? DisplayName = null,
    string? Description = null,
    bool IsDefault = false,
    IReadOnlyList<string>? SupportedProviders = null,
    int? Opset = null);

public sealed record BundledModelManifestResolution(
    BundledModelManifestEntry Entry,
    string RequestedReference,
    string Alias,
    string VariantAlias,
    string EntryPath);
