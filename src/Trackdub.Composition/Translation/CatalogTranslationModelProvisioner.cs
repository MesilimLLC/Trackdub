using Trackdub.Contracts;
using Trackdub.Contracts.Translation;
using Trackdub.Contracts.Licensing;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Infrastructure.Settings;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Composition.Translation;

/// <summary>Publishes eligible metadata into the shared registry; only the execution path provisions weights.</summary>
public sealed class CatalogTranslationModelProvisioner(
    ITranslationModelCatalog catalog,
    BundledModelManifestRegistry registry,
    IStudioSettingsService settingsService,
    IModelDownloadOrchestrator downloader,
    TrackdubStoragePaths storagePaths,
    IModelCacheInventory? cacheInventory = null) : ITranslationModelProvisioner
{
    private readonly SemaphoreSlim refreshGate = new(1, 1);
    private DateTimeOffset lastRefresh;

    private Task? refreshTask;

    public async Task RefreshAsync(CancellationToken cancellationToken)
    {
        Publish(catalog.Snapshot);
        var settings = await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
        if (settings.OfflineMode) return;
        await refreshGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (refreshTask is { IsCompleted: false } || DateTimeOffset.UtcNow - lastRefresh < TimeSpan.FromHours(24) ||
                DateTimeOffset.UtcNow - catalog.Snapshot.FetchedAtUtc < TimeSpan.FromHours(24)) return;
            lastRefresh = DateTimeOffset.UtcNow;
            refreshTask = RefreshInBackgroundAsync();
        }
        finally { refreshGate.Release(); }
    }

    private async Task RefreshInBackgroundAsync()
    {
        try { Publish(await catalog.RefreshAsync(false, CancellationToken.None).ConfigureAwait(false)); }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Keep the shared refresh task successful so later translations are not blocked by a fault.
            // Clearing lastRefresh lets the next refresh retry instead of waiting out the 24-hour window.
            await refreshGate.WaitAsync().ConfigureAwait(false);
            try { lastRefresh = DateTimeOffset.MinValue; }
            finally { refreshGate.Release(); }
            Publish(catalog.Snapshot);
        }
    }

    private void Publish(TranslationModelCatalogSnapshot snapshot) =>
        registry.AddCatalogEntries(snapshot.Models.GroupBy(model => model.ModelId)
            .Select(group => CreateEntry(group, snapshot.PolicyVersion)).OfType<BundledModelManifestEntry>());

    public async Task<string?> EnsureAsync(string sourceLanguage, string targetLanguage, string? preferredModelAlias, CancellationToken cancellationToken,
        IProgress<string>? preparationProgress = null, string? preferredModelVariantAlias = null, string? preferredExecutionProvider = null)
    {
        if (string.IsNullOrWhiteSpace(sourceLanguage) || sourceLanguage.Equals("auto", StringComparison.OrdinalIgnoreCase) ||
            sourceLanguage.Equals(targetLanguage, StringComparison.OrdinalIgnoreCase)) return preferredModelAlias;
        bool explicitSelection = !string.IsNullOrWhiteSpace(preferredModelAlias);
        bool registeredSelection = explicitSelection && registry.TryResolve(preferredModelAlias!, out _);
        bool catalogReference = preferredModelAlias?.StartsWith("hf:", StringComparison.Ordinal) == true ||
            preferredModelAlias?.StartsWith("onnx-community/opus-mt-", StringComparison.Ordinal) == true ||
            preferredModelAlias?.StartsWith("Xenova/opus-mt-", StringComparison.Ordinal) == true;
        if (explicitSelection && !registeredSelection && !catalogReference) return preferredModelAlias;
        if (!registeredSelection) await RefreshAsync(cancellationToken).ConfigureAwait(false);
        var settings = await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
        if (settings.OfflineMode || !settings.AutoDownloadTranslationModels) return preferredModelAlias;
        if (!registeredSelection && (explicitSelection || !registry.Entries.Any(entry => entry.Task == "translation" &&
            entry.LanguageCoverage.LanguagePairs.Any(pair => pair.SourceLanguage.Equals(sourceLanguage, StringComparison.OrdinalIgnoreCase) &&
                pair.TargetLanguage.Equals(targetLanguage, StringComparison.OrdinalIgnoreCase)))) && refreshTask is { } pending)
            await pending.WaitAsync(cancellationToken).ConfigureAwait(false);
        // Existing entries/revisions win over refreshed metadata, so a chosen descriptor cannot change underneath a run.
        var installed = cacheInventory is null ? new HashSet<string>(StringComparer.OrdinalIgnoreCase) :
            (await cacheInventory.LoadAsync(cancellationToken).ConfigureAwait(false)).Select(record => record.ModelId).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var candidates = registry.Entries.Where(entry => entry.Task == "translation" && entry.CommercialAllowed && entry.CommercialUseVerified)
            .OrderBy(entry => entry.LanguageCoverage.LanguagePairs.Count == 1 ? 0 : 1)
            .ThenBy(entry => installed.Contains(entry.ModelId) ? 0 : 1)
            .ThenBy(entry => entry.SourceUrl.Contains("/onnx-community/", StringComparison.Ordinal) ? 0 : 1)
            .ThenBy(entry => entry.ModelId, StringComparer.Ordinal);
        BundledModelManifestEntry? selected = !string.IsNullOrWhiteSpace(preferredModelAlias)
            ? candidates.FirstOrDefault(entry => entry.ModelId.Equals(preferredModelAlias, StringComparison.OrdinalIgnoreCase) ||
                entry.Aliases.Contains(preferredModelAlias, StringComparer.OrdinalIgnoreCase))
            : candidates.FirstOrDefault(entry => entry.EngineFamily == "opus-mt" && entry.LanguageCoverage.LanguagePairs.Any(pair =>
                pair.SourceLanguage.Equals(sourceLanguage, StringComparison.OrdinalIgnoreCase) && pair.TargetLanguage.Equals(targetLanguage, StringComparison.OrdinalIgnoreCase)));
        if (selected is null) return preferredModelAlias;
        bool coversPair = selected.LanguageCoverage.LanguagePairs.Any(pair =>
            pair.SourceLanguage.Equals(sourceLanguage, StringComparison.OrdinalIgnoreCase) && pair.TargetLanguage.Equals(targetLanguage, StringComparison.OrdinalIgnoreCase)) ||
            (Covers(selected.LanguageCoverage.SourceLanguages, sourceLanguage) && Covers(selected.LanguageCoverage.TargetLanguages, targetLanguage));
        if (!coversPair) return preferredModelAlias;
        // Explicit Qwen selections may request a specific CPU or CUDA variant.
        if (!CanProvisionSelectedModel(selected, explicitSelection))
            return preferredModelAlias;
        preparationProgress?.Report($"Preparing {selected.ModelId}");
        // Without an explicit variant, the default CPU variant would be downloaded even for a CUDA plan.
        string? variantAlias = string.IsNullOrWhiteSpace(preferredModelVariantAlias)
            ? ResolveVariantForProvider(selected, preferredExecutionProvider)
            : preferredModelVariantAlias;
        var result = await downloader.DownloadAsync(selected.ModelId, variantAlias,
            preparationProgress is null ? null : new PreparationDownloadProgress(preparationProgress, selected.ModelId),
            cancellationToken).ConfigureAwait(false);
        if (!result.Success) throw new InvalidOperationException(result.FailureReason ?? $"Translation model '{selected.ModelId}' could not be installed.");
        return selected.Aliases.FirstOrDefault() ?? selected.ModelId;
    }

    private static bool CanProvisionSelectedModel(BundledModelManifestEntry selected, bool explicitSelection) =>
        selected.EngineFamily == "opus-mt" ||
        (explicitSelection && selected.EngineFamily == "qwen-genai");

    private static string? ResolveVariantForProvider(BundledModelManifestEntry selected, string? executionProvider) =>
        string.IsNullOrWhiteSpace(executionProvider) ? null
        : selected.Variants.FirstOrDefault(variant =>
            variant.SupportedProviders?.Contains(executionProvider.Trim(), StringComparer.OrdinalIgnoreCase) == true)?.Alias;

    private static bool Covers(IReadOnlyList<string> languages, string language) =>
        languages.Any(candidate => candidate.Equals(language, StringComparison.OrdinalIgnoreCase) || candidate is "multi" or "any" or "*");

    private sealed class PreparationDownloadProgress(IProgress<string> progress, string modelId) : IProgress<ModelDownloadProgress>
    {
        public void Report(ModelDownloadProgress value) => progress.Report(
            value.PercentComplete < 100
                ? $"Downloading {modelId}: {value.PercentComplete}%"
                : $"Verifying {modelId}");
    }

    private BundledModelManifestEntry? CreateEntry(IGrouping<string, TranslationModelDescriptor> group, string policyVersion)
    {
        var model = group.First();
        // This allowlist is the catalog's explicit commercial license policy. Metadata without evidence is rejected.
        if (model.License is not ("apache-2.0" or "mit" or "cc-by-4.0") || string.IsNullOrWhiteSpace(model.LicenseEvidence) ||
            model.Revision.Length != 40 || !model.Revision.All(Uri.IsHexDigit) || model.Files.IsEmpty ||
            !(model.Repository.StartsWith("onnx-community/opus-mt-", StringComparison.Ordinal) || model.Repository.StartsWith("Xenova/opus-mt-", StringComparison.Ordinal)) ||
            model.Repository.Split('/').Length != 2 || model.Repository.Any(character => !(char.IsAsciiLetterOrDigit(character) || character is '/' or '-' or '_' or '.')) ||
            model.Files.Any(file => file.SizeBytes < 0 || file.Sha256.Length != 64 || !file.Sha256.All(Uri.IsHexDigit) ||
                Path.IsPathRooted(file.Path) || file.Path.Contains('\\') || file.Path.Contains(':') || file.Path.Split('/').Any(segment => segment is ".." or "."))) return null;
        string[] required = ["onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx", "config.json", "tokenizer_config.json", "vocab.json", "source.spm", "target.spm"];
        if (required.Any(path => !model.Files.Any(file => file.Path == path)) || model.Files.Select(file => file.Path).Distinct().Count() != model.Files.Length) return null;
        string root = Path.Join(storagePaths.ModelCacheDirectory, model.Repository);
        string? encoder = model.Files.FirstOrDefault(file => file.Path.EndsWith("encoder_model.onnx", StringComparison.Ordinal))?.Path;
        if (encoder is null) return null;
        string[] files = model.Files.Select(file => file.Path).ToArray();
        return new BundledModelManifestEntry(model.Repository, "translation", "opus-mt", ["translation", "direct-translation"],
            new ModelLanguageCoverage([], [], group.Select(pair => new ModelLanguagePair(pair.SourceLanguage, pair.TargetLanguage)).ToArray()),
            "fast", ModelLane.Commercial, model.License, true, true, model.License == "cc-by-4.0", false, false, true,
            $"https://huggingface.co/{model.Repository}", model.Revision, model.Files.First(file => file.Path == encoder).Sha256,
            files, model.Files.ToDictionary(file => file.Path, file => $"https://huggingface.co/{model.Repository}/resolve/{model.Revision}/{file.Path}"),
            model.Files.ToDictionary(file => file.Path, file => file.Sha256), [model.ModelId, model.Repository], root, Path.Join(root, encoder), [],
            ExpectedDownloadBytes: model.Files.Sum(file => file.SizeBytes),
            CatalogPolicyVersion: policyVersion,
            TargetPrefixes: group.GroupBy(pair => pair.TargetLanguage).ToDictionary(pair => pair.Key, pair => pair.First().TargetPrefix));
    }
}
