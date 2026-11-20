using System.Text.Json;

namespace Trackdub.Inference.Runtime.ModelManifest;

public static class ModelManifestLoader
{
    public static ModelManifestCatalog LoadCatalog(string manifestPath)
    {
        string fullManifestPath = Path.GetFullPath(manifestPath);
        using FileStream stream = File.OpenRead(fullManifestPath);
        using JsonDocument document = JsonDocument.Parse(stream);
        return LoadCatalog(document.RootElement, fullManifestPath);
    }

    public static ModelManifestCatalog LoadCatalog(JsonElement rootElement, string sourceName)
    {
        if (rootElement.ValueKind is JsonValueKind.Object &&
            rootElement.TryGetProperty("models", out JsonElement modelsElement))
        {
            return new ModelManifestCatalog(ReadManifestArray(modelsElement, sourceName));
        }

        return new ModelManifestCatalog([ModelRiskAcceptanceReader.Attach(ReadManifest(rootElement, "$", sourceName), rootElement, new("$", sourceName))]);
    }

    private static IReadOnlyList<ModelManifest> ReadManifestArray(JsonElement modelsElement, string sourceName)
    {
        if (modelsElement.ValueKind is not JsonValueKind.Array)
        {
            throw new ModelManifestValidationException($"Manifest '{sourceName}' field '$.models' must be an array.");
        }

        var models = new List<ModelManifest>();
        int index = 0;
        foreach (JsonElement element in modelsElement.EnumerateArray())
        {
            models.Add(ModelRiskAcceptanceReader.Attach(ReadManifest(element, $"$.models[{index}]", sourceName), element, new($"$.models[{index}]", sourceName)));
            index++;
        }

        if (models.Count == 0)
        {
            throw new ModelManifestValidationException($"Manifest '{sourceName}' did not contain any model entries.");
        }

        return models;
    }

    private static ModelManifest ReadManifest(JsonElement element, string path, string sourceName)
    {
        if (element.ValueKind is not JsonValueKind.Object)
        {
            throw new ModelManifestValidationException($"Manifest '{sourceName}' entry '{path}' must be an object.");
        }

        string modelId = ReadRequiredString(element, "model_id", path, sourceName);
        ModelTask task = ParseTask(ReadRequiredString(element, "task", path, sourceName), path, sourceName);
        string engineFamily = ReadRequiredString(element, "engine_family", path, sourceName);
        IReadOnlyList<string> capabilities = ReadStringArray(element, "capabilities", path, sourceName);
        ModelLanguageCoverage languageCoverage = ReadLanguageCoverage(element, path, sourceName);
        string tier = ReadOptionalString(element, "tier");
        if (string.IsNullOrWhiteSpace(tier))
        {
            tier = "balanced";
        }

        ModelLicenseKind license = ParseLicense(ReadRequiredString(element, "license", path, sourceName), path, sourceName);
        bool commercialAllowed = ReadRequiredBoolean(element, "commercial_allowed", path, sourceName);
        bool redistributionAllowed = ReadRequiredBoolean(element, "redistribution_allowed", path, sourceName);
        bool requiresAttribution = ReadRequiredBoolean(element, "requires_attribution", path, sourceName);
        bool requiresUserConsent = ReadRequiredBoolean(element, "requires_user_consent", path, sourceName);
        bool voiceCloning = ReadRequiredBoolean(element, "voice_cloning", path, sourceName);
        bool commercialUseVerified = ReadOptionalBooleanAlias(
            element,
            "commercial_use_verified",
            "commercial_safe_mode",
            path,
            sourceName,
            defaultValue: false);
        ModelLane lane = ReadModelLane(element, path, sourceName, license, commercialAllowed, commercialUseVerified);
        string sourceUrl = ReadOptionalString(element, "source_url");
        string revision = ReadOptionalString(element, "revision");
        string sha256 = ReadOptionalString(element, "sha256");
        IReadOnlyList<string> downloadFiles = ReadStringArray(element, "download_files", path, sourceName);
        IReadOnlyDictionary<string, string> downloadFileSources = ReadStringMap(element, "download_file_sources", path, sourceName);
        IReadOnlyDictionary<string, string> downloadFileHashes = ReadStringMap(element, "download_file_hashes", path, sourceName);
        string? rootPath = ReadOptionalNullableString(element, "root_path", path, sourceName);
        string? benchmarkEntry = ReadOptionalNullableString(element, "benchmark_entry", path, sourceName);
        bool oliveOptimizable = ReadOptionalBoolean(element, "olive_optimizable", path, sourceName, defaultValue: false);
        int estimatedVramMb = ReadOptionalInt32(element, "estimated_vram_mb", path, sourceName) ?? 0;
        int minVramMb = ReadOptionalInt32(element, "min_vram_mb", path, sourceName) ?? 0;
        bool supportsPartialOffload = ReadOptionalBoolean(element, "supports_partial_offload", path, sourceName, defaultValue: false);
        ModelOptimizationManifest? optimization = ReadOptimization(element, path, sourceName);
        string? displayName = ReadOptionalNullableString(element, "display_name", path, sourceName);
        string? providerId = ReadOptionalNullableString(element, "provider_id", path, sourceName);
        string? expectedRuntime = ReadOptionalNullableString(element, "expected_runtime", path, sourceName);
        bool? deprecated = ReadOptionalNullableBoolean(element, "deprecated", path, sourceName);
        string? deprecatedReason = ReadOptionalNullableString(element, "deprecated_reason", path, sourceName);
        IReadOnlyList<string> aliases = ReadAliases(element, path, sourceName);
        IReadOnlyList<ModelVariantManifest> variants = ReadVariants(element, path, sourceName);
        ValidateVariants(variants, path, sourceName);
        HashVerificationPolicy hashVerificationPolicy = ReadHashVerificationPolicy(element, path, sourceName);
        ValidateDownloadFileHashes(downloadFileHashes, downloadFiles, variants, benchmarkEntry, hashVerificationPolicy, path, sourceName);

        if (voiceCloning && !requiresUserConsent)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' must set 'requires_user_consent' when 'voice_cloning' is true.");
        }

        if ((license is ModelLicenseKind.NonCommercial or ModelLicenseKind.CcByNc40) && commercialAllowed)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' cannot set 'commercial_allowed' to true when the license is non-commercial.");
        }

        if (lane is ModelLane.NonCommercial && commercialAllowed)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' cannot use lane 'non-commercial' when commercial use is allowed.");
        }

        if (!string.IsNullOrWhiteSpace(benchmarkEntry) && string.IsNullOrWhiteSpace(rootPath))
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' must define 'root_path' when 'benchmark_entry' is present.");
        }

        if (capabilities.Any(c => c.Equals("direct-translation", StringComparison.OrdinalIgnoreCase)) &&
            languageCoverage.LanguagePairs.Count == 0)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' declares 'direct-translation' capability but has no 'language_coverage.language_pairs'.");
        }

        if (hashVerificationPolicy.Mode is HashVerificationMode.Required &&
            string.IsNullOrWhiteSpace(sha256) &&
            downloadFileHashes.Count == 0)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' sets hash_verification.mode to 'required' but does not define sha256 or download_file_hashes.");
        }

        if (commercialUseVerified && !HasCommercialUseHashEvidence(sha256, downloadFileHashes, benchmarkEntry))
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' sets 'commercial_use_verified' to true but does not define sha256 or download_file_hashes covering 'benchmark_entry'.");
        }

        return new ModelManifest(
            modelId,
            task,
            engineFamily,
            capabilities,
            languageCoverage,
            tier,
            lane,
            license,
            commercialAllowed,
            redistributionAllowed,
            requiresAttribution,
            requiresUserConsent,
            voiceCloning,
            commercialUseVerified,
            sourceUrl,
            revision,
            sha256,
            downloadFiles,
            downloadFileSources,
            variants,
            aliases,
            rootPath,
            benchmarkEntry,
            hashVerificationPolicy,
            displayName,
            oliveOptimizable,
            optimization,
            providerId,
            expectedRuntime,
            downloadFileHashes,
            estimatedVramMb,
            minVramMb,
            supportsPartialOffload,
            deprecated,
            deprecatedReason);
    }

    private static ModelLane ReadModelLane(
        JsonElement element,
        string path,
        string sourceName,
        ModelLicenseKind license,
        bool commercialAllowed,
        bool commercialUseVerified)
    {
        if (element.TryGetProperty("lane", out JsonElement laneProperty))
        {
            if (laneProperty.ValueKind is not JsonValueKind.String)
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.lane' must be a string.");
            }

            string laneText = laneProperty.GetString()?.Trim() ?? string.Empty;
            if (!string.IsNullOrWhiteSpace(laneText))
            {
                return ParseLane(laneText, path, sourceName);
            }
        }

        if (!commercialAllowed || license is ModelLicenseKind.NonCommercial or ModelLicenseKind.CcByNc40)
        {
            return ModelLane.NonCommercial;
        }

        return commercialUseVerified ? ModelLane.Commercial : ModelLane.Experimental;
    }

    private static IReadOnlyList<string> ReadStringArray(
        JsonElement element,
        string propertyName,
        string path,
        string sourceName)
    {
        if (!element.TryGetProperty(propertyName, out JsonElement arrayElement))
        {
            return [];
        }

        if (arrayElement.ValueKind is not JsonValueKind.Array)
        {
            throw new ModelManifestValidationException($"Manifest '{sourceName}' field '{path}.{propertyName}' must be an array.");
        }

        var values = new List<string>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        int index = 0;
        foreach (JsonElement valueElement in arrayElement.EnumerateArray())
        {
            if (valueElement.ValueKind is not JsonValueKind.String)
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.{propertyName}[{index}]' must be a string.");
            }

            string value = valueElement.GetString()?.Trim() ?? string.Empty;
            if (string.IsNullOrWhiteSpace(value))
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.{propertyName}[{index}]' cannot be empty.");
            }

            if (seen.Add(value))
            {
                values.Add(value);
            }
            else
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.{propertyName}' contains duplicate value '{value}'.");
            }

            index++;
        }

        return values;
    }

    private static IReadOnlyDictionary<string, string> ReadStringMap(
        JsonElement element,
        string propertyName,
        string path,
        string sourceName)
    {
        if (!element.TryGetProperty(propertyName, out JsonElement mapElement))
        {
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        }

        if (mapElement.ValueKind is not JsonValueKind.Object)
        {
            throw new ModelManifestValidationException($"Manifest '{sourceName}' field '{path}.{propertyName}' must be an object.");
        }

        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (JsonProperty property in mapElement.EnumerateObject())
        {
            string key = property.Name.Trim();
            if (string.IsNullOrWhiteSpace(key))
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.{propertyName}' contains an empty key.");
            }

            if (property.Value.ValueKind is not JsonValueKind.String)
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.{propertyName}.{key}' must be a string.");
            }

            string value = property.Value.GetString()?.Trim() ?? string.Empty;
            if (string.IsNullOrWhiteSpace(value))
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.{propertyName}.{key}' cannot be empty.");
            }

            if (!values.TryAdd(key, value))
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.{propertyName}' contains duplicate key '{key}'.");
            }
        }

        return values;
    }

    private static void ValidateDownloadFileHashes(
        IReadOnlyDictionary<string, string> downloadFileHashes,
        IReadOnlyList<string> downloadFiles,
        IReadOnlyList<ModelVariantManifest> variants,
        string? benchmarkEntry,
        HashVerificationPolicy hashVerificationPolicy,
        string path,
        string sourceName)
    {
        if (downloadFileHashes.Count == 0)
        {
            return;
        }

        var normalizedHashes = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach ((string relativePath, string hash) in downloadFileHashes)
        {
            string normalizedPath = NormalizeSafeRelativePath(relativePath, "download_file_hashes", path, sourceName);
            if (!IsValidSha256(hash))
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' entry '{path}' download_file_hashes['{relativePath}'] must be a SHA-256 hex digest.");
            }

            normalizedHashes.Add(normalizedPath);
        }

        if (hashVerificationPolicy.Mode is not HashVerificationMode.Required)
        {
            return;
        }

        foreach (string requiredPath in EnumerateHashRequiredPaths(downloadFiles, variants, benchmarkEntry))
        {
            string normalizedRequiredPath = NormalizeSafeRelativePath(requiredPath, "download_file_hashes", path, sourceName);
            if (!normalizedHashes.Contains(normalizedRequiredPath))
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' entry '{path}' requires download_file_hashes['{normalizedRequiredPath}'] because hash verification is required.");
            }
        }
    }

    private static IEnumerable<string> EnumerateHashRequiredPaths(
        IReadOnlyList<string> downloadFiles,
        IReadOnlyList<ModelVariantManifest> variants,
        string? benchmarkEntry)
    {
        foreach (string downloadFile in downloadFiles)
        {
            yield return downloadFile;
        }

        foreach (ModelVariantManifest variant in variants.Where(static variant =>
                     variant.IsDefault ||
                     variant.Alias.Equals("default", StringComparison.OrdinalIgnoreCase)))
        {
            yield return variant.EntryPath;
            foreach (string downloadFile in variant.DownloadFiles)
            {
                yield return downloadFile;
            }
        }

        if (!string.IsNullOrWhiteSpace(benchmarkEntry))
        {
            yield return benchmarkEntry;
        }
    }

    private static string NormalizeSafeRelativePath(
        string relativePath,
        string fieldName,
        string path,
        string sourceName)
    {
        string normalizedPath = relativePath.Replace('\\', '/').Trim('/');
        if (string.IsNullOrWhiteSpace(normalizedPath) ||
            Path.IsPathRooted(normalizedPath) ||
            normalizedPath.Split('/').Any(part => part is "." or ".." || string.IsNullOrWhiteSpace(part)))
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' field '{fieldName}' contains unsafe relative path '{relativePath}'.");
        }

        return normalizedPath;
    }

    private static bool IsValidSha256(string hash) =>
        hash.Length == 64 && hash.All(static c =>
            c is >= '0' and <= '9' or >= 'a' and <= 'f' or >= 'A' and <= 'F');

    private static bool HasCommercialUseHashEvidence(
        string sha256,
        IReadOnlyDictionary<string, string> downloadFileHashes,
        string? benchmarkEntry)
    {
        if (!string.IsNullOrWhiteSpace(sha256) && IsValidSha256(sha256))
        {
            return true;
        }

        if (string.IsNullOrWhiteSpace(benchmarkEntry) || downloadFileHashes.Count == 0)
        {
            return false;
        }

        string normalizedBenchmarkEntry = benchmarkEntry.Replace('\\', '/').Trim('/');
        foreach ((string relativePath, string hash) in downloadFileHashes)
        {
            if (!IsValidSha256(hash))
            {
                continue;
            }

            string normalizedPath = relativePath.Replace('\\', '/').Trim('/');
            if (normalizedPath.Equals(normalizedBenchmarkEntry, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private static ModelLanguageCoverage ReadLanguageCoverage(
        JsonElement element,
        string path,
        string sourceName)
    {
        if (!element.TryGetProperty("language_coverage", out JsonElement coverageElement))
        {
            return ModelLanguageCoverage.Empty;
        }

        if (coverageElement.ValueKind is not JsonValueKind.Object)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.language_coverage' must be an object.");
        }

        IReadOnlyList<string> sourceLanguages = ReadStringArray(
            coverageElement,
            "source_languages",
            $"{path}.language_coverage",
            sourceName);
        IReadOnlyList<string> targetLanguages = ReadStringArray(
            coverageElement,
            "target_languages",
            $"{path}.language_coverage",
            sourceName);
        IReadOnlyList<ModelLanguagePair> languagePairs = ReadLanguagePairs(
            coverageElement,
            $"{path}.language_coverage",
            sourceName);

        return new ModelLanguageCoverage(sourceLanguages, targetLanguages, languagePairs);
    }

    private static IReadOnlyList<ModelLanguagePair> ReadLanguagePairs(
        JsonElement coverageElement,
        string path,
        string sourceName)
    {
        if (!coverageElement.TryGetProperty("language_pairs", out JsonElement pairsElement))
        {
            return [];
        }

        if (pairsElement.ValueKind is not JsonValueKind.Array)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.language_pairs' must be an array.");
        }

        var pairs = new List<ModelLanguagePair>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        int index = 0;
        foreach (JsonElement pairElement in pairsElement.EnumerateArray())
        {
            if (pairElement.ValueKind is not JsonValueKind.Object)
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.language_pairs[{index}]' must be an object.");
            }

            string sourceLanguage = ReadRequiredString(pairElement, "source", $"{path}.language_pairs[{index}]", sourceName)
                .ToLowerInvariant();
            string targetLanguage = ReadRequiredString(pairElement, "target", $"{path}.language_pairs[{index}]", sourceName)
                .ToLowerInvariant();
            string key = $"{sourceLanguage}->{targetLanguage}";
            if (seen.Add(key))
            {
                pairs.Add(new ModelLanguagePair(sourceLanguage, targetLanguage));
            }

            index++;
        }

        return pairs;
    }

    private static IReadOnlyList<string> ReadAliases(JsonElement element, string path, string sourceName)
    {
        if (!element.TryGetProperty("aliases", out JsonElement aliasesElement))
        {
            return [];
        }

        if (aliasesElement.ValueKind is not JsonValueKind.Array)
        {
            throw new ModelManifestValidationException($"Manifest '{sourceName}' field '{path}.aliases' must be an array.");
        }

        var aliases = new List<string>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        int index = 0;
        foreach (JsonElement aliasElement in aliasesElement.EnumerateArray())
        {
            if (aliasElement.ValueKind is not JsonValueKind.String)
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.aliases[{index}]' must be a string.");
            }

            string alias = aliasElement.GetString()?.Trim() ?? string.Empty;
            if (string.IsNullOrWhiteSpace(alias))
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.aliases[{index}]' cannot be empty.");
            }

            if (!seen.Add(alias))
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.aliases' contains duplicate alias '{alias}'.");
            }

            aliases.Add(alias);
            index++;
        }

        return aliases;
    }

    private static IReadOnlyList<ModelVariantManifest> ReadVariants(JsonElement element, string path, string sourceName)
    {
        if (!element.TryGetProperty("variants", out JsonElement variantsElement))
        {
            return [];
        }

        if (variantsElement.ValueKind is not JsonValueKind.Array)
        {
            throw new ModelManifestValidationException($"Manifest '{sourceName}' field '{path}.variants' must be an array.");
        }

        var variants = new List<ModelVariantManifest>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        int index = 0;
        foreach (JsonElement variantElement in variantsElement.EnumerateArray())
        {
            if (variantElement.ValueKind is not JsonValueKind.Object)
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.variants[{index}]' must be an object.");
            }

            string alias = ReadRequiredString(variantElement, "alias", $"{path}.variants[{index}]", sourceName);
            string entryPath = ReadRequiredString(variantElement, "entry_path", $"{path}.variants[{index}]", sourceName);
            string sha256 = ReadOptionalString(variantElement, "sha256");
            IReadOnlyList<string> downloadFiles = ReadStringArray(variantElement, "download_files", $"{path}.variants[{index}]", sourceName);
            string? displayName = ReadOptionalNullableString(variantElement, "display_name", $"{path}.variants[{index}]", sourceName);
            string? description = ReadOptionalNullableString(variantElement, "description", $"{path}.variants[{index}]", sourceName);
            bool isDefault = ReadOptionalBoolean(variantElement, "is_default", $"{path}.variants[{index}]", sourceName, defaultValue: false);
            IReadOnlyList<string> supportedProviders = ReadStringArray(variantElement, "supported_providers", $"{path}.variants[{index}]", sourceName);
            int? opset = ReadOptionalInt32(variantElement, "opset", $"{path}.variants[{index}]", sourceName);
            int? variantEstimatedVramMb = ReadOptionalInt32(variantElement, "estimated_vram_mb", $"{path}.variants[{index}]", sourceName);
            int? variantMinVramMb = ReadOptionalInt32(variantElement, "min_vram_mb", $"{path}.variants[{index}]", sourceName);

            if (!seen.Add(alias))
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.variants' contains duplicate variant alias '{alias}'.");
            }

            variants.Add(new ModelVariantManifest(
                alias,
                entryPath,
                sha256,
                downloadFiles,
                displayName,
                description,
                isDefault,
                supportedProviders,
                opset,
                variantEstimatedVramMb,
                variantMinVramMb));
            index++;
        }

        return variants;
    }

    private static void ValidateVariants(
        IReadOnlyList<ModelVariantManifest> variants,
        string path,
        string sourceName)
    {
        if (variants.Count == 0)
        {
            return;
        }

        int defaultCount = variants.Count(variant => variant.IsDefault);
        if (defaultCount > 1)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.variants' has multiple defaults. Only one variant may set is_default=true.");
        }

        foreach (ModelVariantManifest variant in variants)
        {
            if (IsRootedLikePath(variant.EntryPath) ||
                variant.EntryPath.Split('/', '\\').Any(segment => segment is "." or ".." || string.IsNullOrWhiteSpace(segment)))
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' variant '{variant.Alias}' in '{path}.variants' must use a safe relative entry_path.");
            }
        }
    }

    private static HashVerificationPolicy ReadHashVerificationPolicy(JsonElement element, string path, string sourceName)
    {
        if (!element.TryGetProperty("hash_verification", out JsonElement policyElement))
        {
            return new HashVerificationPolicy(HashVerificationMode.VerifyIfShaPresent, "SHA-256");
        }

        if (policyElement.ValueKind is not JsonValueKind.Object)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.hash_verification' must be an object.");
        }

        string modeText = ReadRequiredString(policyElement, "mode", $"{path}.hash_verification", sourceName);
        HashVerificationMode mode = ParseHashVerificationMode(modeText, path, sourceName);
        string algorithm = ReadOptionalString(policyElement, "algorithm");
        if (string.IsNullOrWhiteSpace(algorithm))
        {
            algorithm = "SHA-256";
        }

        return new HashVerificationPolicy(mode, algorithm);
    }

    private static ModelOptimizationManifest? ReadOptimization(JsonElement element, string path, string sourceName)
    {
        if (!element.TryGetProperty("optimization", out JsonElement optimizationElement))
        {
            return null;
        }

        if (optimizationElement.ValueKind is not JsonValueKind.Object)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.optimization' must be an object.");
        }

        if (!optimizationElement.TryGetProperty("olive", out JsonElement oliveElement))
        {
            return null;
        }

        return new ModelOptimizationManifest(ReadOliveOptimizationProfile(oliveElement, $"{path}.optimization.olive", sourceName));
    }

    private static ModelOliveOptimizationProfile ReadOliveOptimizationProfile(
        JsonElement element,
        string path,
        string sourceName)
    {
        if (element.ValueKind is not JsonValueKind.Object)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}' must be an object.");
        }

        string mode = ReadRequiredString(element, "mode", path, sourceName);
        IReadOnlyList<string> components = ReadOptimizationComponents(element, path, sourceName, mode);
        IReadOnlyList<OliveOptimizationProvider> providers = ReadOliveProviders(element, path, sourceName);
        IReadOnlyList<string> supportedPrecisions = ReadStringArray(element, "supported_precisions", path, sourceName);
        IReadOnlyList<OliveOpsetPolicy> opsetPolicies = ReadOliveOpsetPolicies(element, path, sourceName);
        bool requireOpsetMetadata = ReadOptionalBoolean(element, "require_opset_metadata", path, sourceName, defaultValue: false);
        IReadOnlyList<OliveRecipeBinding> recipeBindings = ReadOliveRecipeBindings(element, path, sourceName, components);
        OliveRecipeFallbackPolicy fallbackPolicy = ReadOliveFallbackPolicy(
            element, "fallback_policy", path, sourceName, defaultValue: OliveRecipeFallbackPolicy.None);

        return new ModelOliveOptimizationProfile(
            mode,
            components,
            providers,
            supportedPrecisions,
            opsetPolicies,
            requireOpsetMetadata,
            recipeBindings,
            fallbackPolicy);
    }

    private static IReadOnlyList<OliveRecipeBinding> ReadOliveRecipeBindings(
        JsonElement element,
        string path,
        string sourceName,
        IReadOnlyList<string> components)
    {
        if (!element.TryGetProperty("recipe_bindings", out JsonElement bindingsElement))
        {
            return [];
        }

        if (bindingsElement.ValueKind is not JsonValueKind.Array)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.recipe_bindings' must be an array.");
        }

        var bindings = new List<OliveRecipeBinding>();
        var index = 0;
        foreach (JsonElement bindingElement in bindingsElement.EnumerateArray())
        {
            string bindingPath = $"{path}.recipe_bindings[{index}]";
            if (bindingElement.ValueKind is not JsonValueKind.Object)
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{bindingPath}' must be an object.");
            }

            string configRelativePath = ReadRequiredString(
                bindingElement,
                "config_relative_path",
                bindingPath,
                sourceName);

            string? provider = null;
            if (bindingElement.TryGetProperty("provider", out JsonElement providerElement) &&
                providerElement.ValueKind is JsonValueKind.String)
            {
                string providerText = providerElement.GetString() ?? string.Empty;
                OliveOptimizationProvider parsedProvider = ParseOliveProvider(providerText, $"{bindingPath}.provider", sourceName);
                provider = NormalizeOliveProviderKey(parsedProvider);
            }

            string? precision = ReadOptionalNullableString(
                bindingElement,
                "precision",
                bindingPath,
                sourceName);

            IReadOnlyList<OliveOptimizationOperation> operations =
                ReadOliveOperations(bindingElement, $"{bindingPath}.operations", sourceName);

            if (operations.Count == 0)
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{bindingPath}.operations' must contain at least one operation.");
            }

            OliveRecipeExpectedOutput expectedOutput = ReadOliveExpectedOutput(
                bindingElement, "expected_output", bindingPath, sourceName,
                defaultValue: OliveRecipeExpectedOutput.OnnxComponents);

            OliveRecipeFallbackPolicy? bindingFallbackPolicy = bindingElement.TryGetProperty("fallback_policy", out _)
                ? ReadOliveFallbackPolicy(bindingElement, "fallback_policy", bindingPath, sourceName, OliveRecipeFallbackPolicy.None)
                : null;

            string? quantizationMethod = ReadOptionalNullableString(bindingElement, "quantization_method", bindingPath, sourceName);
            bool requiresCalibrationData = ReadOptionalBoolean(bindingElement, "requires_calibration_data", bindingPath, sourceName, defaultValue: false);
            string? scriptRelativePath = ReadOptionalNullableString(bindingElement, "script_relative_path", bindingPath, sourceName);
            string? scriptSha256 = ReadOptionalNullableString(bindingElement, "script_sha256", bindingPath, sourceName);
            string? evaluator = ReadOptionalNullableString(bindingElement, "evaluator", bindingPath, sourceName);
            int? splitCount = ReadOptionalInt32(bindingElement, "split_count", bindingPath, sourceName);
            string? costModelRelativePath = ReadOptionalNullableString(bindingElement, "cost_model_relative_path", bindingPath, sourceName);
            string? adapterRelativePath = ReadOptionalNullableString(bindingElement, "adapter_relative_path", bindingPath, sourceName);
            string? adapterMode = ReadOptionalNullableString(bindingElement, "adapter_mode", bindingPath, sourceName);
            string? outputManifestRelativePath = ReadOptionalNullableString(bindingElement, "output_manifest_relative_path", bindingPath, sourceName);
            string? component = ReadOptionalNullableString(bindingElement, "component", bindingPath, sourceName);
            // The resolver matches a component-scoped binding only by exact name, so a typo would
            // leave the binding silently unused; reject it here instead.
            if (component is not null && !components.Contains(component, StringComparer.OrdinalIgnoreCase))
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{bindingPath}.component' value '{component}' is not one of '{path}.components'.");
            }

            bindings.Add(new OliveRecipeBinding(
                provider,
                precision,
                configRelativePath,
                operations,
                expectedOutput,
                bindingFallbackPolicy,
                quantizationMethod,
                requiresCalibrationData,
                scriptRelativePath,
                scriptSha256,
                evaluator,
                splitCount,
                costModelRelativePath,
                adapterRelativePath,
                adapterMode,
                outputManifestRelativePath,
                component));
            index++;
        }

        return bindings;
    }

    private static IReadOnlyList<OliveOpsetPolicy> ReadOliveOpsetPolicies(
        JsonElement element,
        string path,
        string sourceName)
    {
        if (!element.TryGetProperty("opset_policies", out JsonElement policiesElement))
        {
            return [];
        }

        if (policiesElement.ValueKind is not JsonValueKind.Array)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.opset_policies' must be an array.");
        }

        var policies = new List<OliveOpsetPolicy>();
        int index = 0;
        foreach (JsonElement policyElement in policiesElement.EnumerateArray())
        {
            if (policyElement.ValueKind is not JsonValueKind.Object)
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.opset_policies[{index}]' must be an object.");
            }

            OliveOptimizationProvider? provider = null;
            if (policyElement.TryGetProperty("provider", out JsonElement providerElement) &&
                providerElement.ValueKind is JsonValueKind.String)
            {
                string providerText = providerElement.GetString() ?? string.Empty;
                provider = ParseOliveProvider(providerText, $"{path}.opset_policies[{index}].provider", sourceName);
            }

            string? precision = ReadOptionalNullableString(
                policyElement,
                "precision",
                $"{path}.opset_policies[{index}]",
                sourceName);
            int minimumOpset = ReadRequiredInt32(
                policyElement,
                "minimum_opset",
                $"{path}.opset_policies[{index}]",
                sourceName);
            policies.Add(new OliveOpsetPolicy(provider, precision, minimumOpset));
            index++;
        }

        return policies;
    }

    private static IReadOnlyList<string> ReadOptimizationComponents(
        JsonElement element,
        string path,
        string sourceName,
        string mode)
    {
        if (!element.TryGetProperty("components", out JsonElement arrayElement))
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' is missing required field 'components'.");
        }

        if (arrayElement.ValueKind is not JsonValueKind.Array)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.components' must be an array.");
        }

        var components = new List<string>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        int index = 0;
        foreach (JsonElement valueElement in arrayElement.EnumerateArray())
        {
            if (valueElement.ValueKind is not JsonValueKind.String)
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.components[{index}]' must be a string.");
            }

            string component = NormalizeOptimizationComponentPath(
                valueElement.GetString() ?? string.Empty,
                $"{path}.components[{index}]",
                sourceName,
                mode);

            if (!seen.Add(component))
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.components' contains duplicate value '{component}'.");
            }

            components.Add(component);
            index++;
        }

        if (components.Count == 0)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.components' must contain at least one ONNX component.");
        }

        return components;
    }

    private static string NormalizeOptimizationComponentPath(
        string value,
        string path,
        string sourceName,
        string mode)
    {
        string trimmed = value.Trim().Replace('\\', '/');
        string normalized = trimmed.Trim('/');
        if (string.IsNullOrWhiteSpace(normalized) ||
            IsRootedLikePath(trimmed) ||
            normalized.Split('/').Any(segment => segment is "." or ".." || string.IsNullOrWhiteSpace(segment)))
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}' must be a safe relative path.");
        }

        bool requiresOnnx = !mode.Equals("ort-genai-builder", StringComparison.OrdinalIgnoreCase);
        if (requiresOnnx && !normalized.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}' must reference an ONNX component.");
        }

        return normalized;
    }

    private static bool IsRootedLikePath(string path) =>
        Path.IsPathRooted(path) ||
        path.StartsWith('/') ||
        path.StartsWith('\\') ||
        (path.Length >= 2 && char.IsLetter(path[0]) && path[1] == ':');

    private static IReadOnlyList<OliveOptimizationProvider> ReadOliveProviders(
        JsonElement element,
        string path,
        string sourceName)
    {
        if (!element.TryGetProperty("supported_providers", out JsonElement arrayElement))
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' is missing required field 'supported_providers'.");
        }

        if (arrayElement.ValueKind is not JsonValueKind.Array)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.supported_providers' must be an array.");
        }

        var providers = new List<OliveOptimizationProvider>();
        var seen = new HashSet<OliveOptimizationProvider>();
        int index = 0;
        foreach (JsonElement valueElement in arrayElement.EnumerateArray())
        {
            if (valueElement.ValueKind is not JsonValueKind.String)
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.supported_providers[{index}]' must be a string.");
            }

            string providerText = valueElement.GetString()?.Trim() ?? string.Empty;
            OliveOptimizationProvider provider = ParseOliveProvider(providerText, $"{path}.supported_providers[{index}]", sourceName);
            if (!seen.Add(provider))
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}.supported_providers' contains duplicate value '{providerText}'.");
            }

            providers.Add(provider);
            index++;
        }

        if (providers.Count == 0)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.supported_providers' must contain at least one provider.");
        }

        return providers;
    }

    private static string ReadRequiredString(JsonElement element, string propertyName, string path, string sourceName)
    {
        if (!element.TryGetProperty(propertyName, out JsonElement property))
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' is missing required field '{propertyName}'.");
        }

        if (property.ValueKind is not JsonValueKind.String)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.{propertyName}' must be a string.");
        }

        string value = property.GetString()?.Trim() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.{propertyName}' cannot be empty.");
        }

        return value;
    }

    private static string ReadOptionalString(JsonElement element, string propertyName)
    {
        if (!element.TryGetProperty(propertyName, out JsonElement property))
        {
            return string.Empty;
        }

        return property.ValueKind is JsonValueKind.String
            ? property.GetString()?.Trim() ?? string.Empty
            : string.Empty;
    }

    private static string? ReadOptionalNullableString(
        JsonElement element,
        string propertyName,
        string path,
        string sourceName)
    {
        if (!element.TryGetProperty(propertyName, out JsonElement property))
        {
            return null;
        }

        if (property.ValueKind is JsonValueKind.Null)
        {
            return null;
        }

        return property.ValueKind is JsonValueKind.String
            ? property.GetString()?.Trim()
            : throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.{propertyName}' must be a string or null.");
    }

    private static bool ReadOptionalBoolean(
        JsonElement element,
        string propertyName,
        string path,
        string sourceName,
        bool defaultValue = false)
    {
        if (!element.TryGetProperty(propertyName, out JsonElement property))
        {
            return defaultValue;
        }

        return ReadBooleanElement(property, propertyName, path, sourceName);
    }

    private static bool ReadRequiredBoolean(JsonElement element, string propertyName, string path, string sourceName)
    {
        if (!element.TryGetProperty(propertyName, out JsonElement property))
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' is missing required field '{propertyName}'.");
        }

        if (property.ValueKind is not JsonValueKind.True and not JsonValueKind.False)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.{propertyName}' must be a boolean.");
        }

        return property.GetBoolean();
    }

    private static int ReadRequiredInt32(JsonElement element, string propertyName, string path, string sourceName)
    {
        if (!element.TryGetProperty(propertyName, out JsonElement property))
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' entry '{path}' is missing required field '{propertyName}'.");
        }

        if (property.ValueKind is not JsonValueKind.Number || !property.TryGetInt32(out int value))
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.{propertyName}' must be an integer.");
        }

        return value;
    }

    private static int? ReadOptionalInt32(
        JsonElement element,
        string propertyName,
        string path,
        string sourceName)
    {
        if (!element.TryGetProperty(propertyName, out JsonElement property))
        {
            return null;
        }

        if (property.ValueKind is JsonValueKind.Null)
        {
            return null;
        }

        if (property.ValueKind is JsonValueKind.Number && property.TryGetInt32(out int value))
        {
            return value;
        }

        throw new ModelManifestValidationException(
            $"Manifest '{sourceName}' field '{path}.{propertyName}' must be an integer or null.");
    }

    private static bool? ReadOptionalNullableBoolean(
        JsonElement element,
        string propertyName,
        string path,
        string sourceName)
    {
        if (!element.TryGetProperty(propertyName, out JsonElement property))
        {
            return null;
        }

        return ReadBooleanElement(property, propertyName, path, sourceName);
    }

    private static bool ReadOptionalBooleanAlias(
        JsonElement element,
        string preferredPropertyName,
        string legacyPropertyName,
        string path,
        string sourceName,
        bool defaultValue = false)
    {
        if (element.TryGetProperty(preferredPropertyName, out JsonElement preferredProperty))
        {
            return ReadBooleanElement(preferredProperty, preferredPropertyName, path, sourceName);
        }

        if (element.TryGetProperty(legacyPropertyName, out JsonElement legacyProperty))
        {
            return ReadBooleanElement(legacyProperty, legacyPropertyName, path, sourceName);
        }

        return defaultValue;
    }

    private static bool ReadBooleanElement(
        JsonElement property,
        string propertyName,
        string path,
        string sourceName)
    {
        if (property.ValueKind is not JsonValueKind.True and not JsonValueKind.False)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.{propertyName}' must be a boolean.");
        }

        return property.GetBoolean();
    }

    private static ModelTask ParseTask(string value, string path, string sourceName) =>
        TryParse(
            () => ModelManifestText.ParseTask(value),
            () => new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.task' value '{value}' is invalid. Expected asr, translation, tts, diarization, vad, separation, speech-enhancement, forced-alignment, text-refinement, overlap-rescue, lip-synthesis, face-detection, or face-landmarks."));

    private static ModelLicenseKind ParseLicense(string value, string path, string sourceName) =>
        TryParse(
            () => ModelManifestText.ParseLicense(value),
            () => new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.license' value '{value}' is invalid. Expected MIT, Apache-2.0, CC-BY-4.0, CC-BY-NC-4.0, NVIDIA-Open-Model-License, OpenMDW-1.1, openrail++, custom, unknown, non-commercial, or noncommercial."));

    private static ModelLane ParseLane(string value, string path, string sourceName) =>
        TryParse(
            () => ModelManifestText.ParseLane(value),
            () => new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.lane' value '{value}' is invalid. Expected commercial, non-commercial, or experimental."));

    private static T TryParse<T>(Func<T> parser, Func<Exception> errorFactory)
    {
        try
        {
            return parser();
        }
        catch (ArgumentException)
        {
            throw errorFactory();
        }
    }

    private static HashVerificationMode ParseHashVerificationMode(string value, string path, string sourceName) =>
        value.ToLowerInvariant() switch
        {
            "none" => HashVerificationMode.None,
            "verify-if-sha-present" => HashVerificationMode.VerifyIfShaPresent,
            "required" => HashVerificationMode.Required,
            _ => throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}.hash_verification.mode' value '{value}' is invalid. Expected none, verify-if-sha-present, or required.")
        };

    private static string NormalizeOliveProviderKey(OliveOptimizationProvider provider) =>
        provider switch
        {
            OliveOptimizationProvider.Cpu => "cpu",
            OliveOptimizationProvider.Dml => "dml",
            OliveOptimizationProvider.Cuda => "cuda",
            OliveOptimizationProvider.TensorRt => "tensorrt",
            OliveOptimizationProvider.TensorRtRtx => "trt-rtx",
            OliveOptimizationProvider.Migraphx => "migraphx",
            OliveOptimizationProvider.Rocm => "rocm",
            OliveOptimizationProvider.VitisAi => "vitisai",
            OliveOptimizationProvider.Qnn => "qnn",
            OliveOptimizationProvider.OpenVino => "openvino",
            _ => throw new ModelManifestValidationException(
                $"Unsupported optimization provider '{provider}'.")
        };

    private static OliveOptimizationProvider ParseOliveProvider(string value, string path, string sourceName) =>
        value.ToLowerInvariant() switch
        {
            "cpu" => OliveOptimizationProvider.Cpu,
            "dml" or "directml" => OliveOptimizationProvider.Dml,
            "cuda" => OliveOptimizationProvider.Cuda,
            "tensorrt" => OliveOptimizationProvider.TensorRt,
            "trt-rtx" or "tensorrt-rtx" => OliveOptimizationProvider.TensorRtRtx,
            "migraphx" => OliveOptimizationProvider.Migraphx,
            "rocm" => OliveOptimizationProvider.Rocm,
            "vitisai" or "vitis-ai" => OliveOptimizationProvider.VitisAi,
            "qnn" => OliveOptimizationProvider.Qnn,
            "openvino" => OliveOptimizationProvider.OpenVino,
            _ => throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}' value '{value}' is invalid. Expected cpu, dml, cuda, tensorrt, trt-rtx, migraphx, rocm, vitisai, qnn, or openvino.")
        };

    private static IReadOnlyList<OliveOptimizationOperation> ReadOliveOperations(
        JsonElement element,
        string path,
        string sourceName)
    {
        if (!element.TryGetProperty("operations", out JsonElement operationsElement))
        {
            return [];
        }

        if (operationsElement.ValueKind is not JsonValueKind.Array)
        {
            throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}' must be an array.");
        }

        var operations = new List<OliveOptimizationOperation>();
        var seen = new HashSet<OliveOptimizationOperation>();
        int index = 0;
        foreach (JsonElement operationElement in operationsElement.EnumerateArray())
        {
            if (operationElement.ValueKind is not JsonValueKind.String)
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}[{index}]' must be a string.");
            }

            string value = operationElement.GetString() ?? string.Empty;
            OliveOptimizationOperation operation = ParseOliveOperation(value, $"{path}[{index}]", sourceName);
            if (!seen.Add(operation))
            {
                throw new ModelManifestValidationException(
                    $"Manifest '{sourceName}' field '{path}' contains duplicate value '{value}'.");
            }

            operations.Add(operation);
            index++;
        }

        return operations;
    }

    private static OliveOptimizationOperation ParseOliveOperation(string value, string path, string sourceName) =>
        value.ToLowerInvariant() switch
        {
            "onnx_export" => OliveOptimizationOperation.OnnxExport,
            "qnn_conversion" => OliveOptimizationOperation.QnnConversion,
            "openvino_conversion" => OliveOptimizationOperation.OpenVinoConversion,
            "compression" => OliveOptimizationOperation.Compression,
            "provider_optimization" => OliveOptimizationOperation.ProviderOptimization,
            "genai_packaging" => OliveOptimizationOperation.GenAiPackaging,
            "model_splitting" => OliveOptimizationOperation.ModelSplitting,
            "evaluation" => OliveOptimizationOperation.Evaluation,
            "adapter_handling" => OliveOptimizationOperation.AdapterHandling,
            "registration" => OliveOptimizationOperation.Registration,
            _ => throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}' value '{value}' is invalid.")
        };

    private static OliveRecipeExpectedOutput ReadOliveExpectedOutput(
        JsonElement element,
        string propertyName,
        string path,
        string sourceName,
        OliveRecipeExpectedOutput defaultValue)
    {
        string? value = ReadOptionalNullableString(element, propertyName, path, sourceName);
        return string.IsNullOrWhiteSpace(value)
            ? defaultValue
            : ParseOliveExpectedOutput(value, $"{path}.{propertyName}", sourceName);
    }

    private static OliveRecipeExpectedOutput ParseOliveExpectedOutput(string value, string path, string sourceName) =>
        value.ToLowerInvariant() switch
        {
            "onnx_components" => OliveRecipeExpectedOutput.OnnxComponents,
            "ort_genai" => OliveRecipeExpectedOutput.OrtGenAi,
            "qnn_model_library" => OliveRecipeExpectedOutput.QnnModelLibrary,
            "openvino_model" => OliveRecipeExpectedOutput.OpenVinoModel,
            "split_onnx_components" => OliveRecipeExpectedOutput.SplitOnnxComponents,
            "adapter_package" => OliveRecipeExpectedOutput.AdapterPackage,
            _ => throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}' value '{value}' is invalid.")
        };

    private static OliveRecipeFallbackPolicy ReadOliveFallbackPolicy(
        JsonElement element,
        string propertyName,
        string path,
        string sourceName,
        OliveRecipeFallbackPolicy defaultValue)
    {
        string? value = ReadOptionalNullableString(element, propertyName, path, sourceName);
        return string.IsNullOrWhiteSpace(value)
            ? defaultValue
            : ParseOliveFallbackPolicy(value, $"{path}.{propertyName}", sourceName);
    }

    private static OliveRecipeFallbackPolicy ParseOliveFallbackPolicy(string value, string path, string sourceName) =>
        value.ToLowerInvariant() switch
        {
            "none" => OliveRecipeFallbackPolicy.None,
            "auto_opt_allowed" => OliveRecipeFallbackPolicy.AutoOptAllowed,
            "base_variant_allowed" => OliveRecipeFallbackPolicy.BaseVariantAllowed,
            "cpu_runtime_allowed" => OliveRecipeFallbackPolicy.CpuRuntimeAllowed,
            _ => throw new ModelManifestValidationException(
                $"Manifest '{sourceName}' field '{path}' value '{value}' is invalid.")
        };
}
