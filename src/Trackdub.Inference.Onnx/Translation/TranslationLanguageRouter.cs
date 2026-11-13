using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts;
using Trackdub.Contracts.Translation;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.Translation;

public sealed class TranslationLanguageRouter(
    BundledModelManifestRegistry manifestRegistry,
    IModelCacheInventory modelCacheInventory,
    IRuntimePlanner? runtimePlanner = null,
    ITranslationModelProvisioner? modelProvisioner = null,
    IStudioSettingsService? settingsService = null)
    : ITranslationLanguageRouter
{
    private static readonly HashSet<string> GenAiPivotEngineFamilies = new(StringComparer.OrdinalIgnoreCase)
    {
        "phi-genai",
        "qwen-genai",
        "gemma-genai"
    };

    private readonly BundledModelManifestRegistry manifestRegistry = manifestRegistry ?? throw new ArgumentNullException(nameof(manifestRegistry));
    private readonly IModelCacheInventory modelCacheInventory = modelCacheInventory ?? throw new ArgumentNullException(nameof(modelCacheInventory));

    // Routing through IRuntimePlanner lets the router share the same planning rules
    // (model integrity, smoke tests, EP availability) as the rest of the pipeline.
    // Optional only so existing test doubles that build the router without DI continue to compile;
    // the production composition root always supplies a planner.
    private readonly IRuntimePlanner? runtimePlanner = runtimePlanner;

    public static bool IsGenAiPivotEngineFamily(string? engineFamily) =>
        !string.IsNullOrWhiteSpace(engineFamily) &&
        GenAiPivotEngineFamilies.Contains(engineFamily.Trim());

    public async Task<IReadOnlyList<TranslationTargetLanguageOption>> GetSupportedTargetLanguagesAsync(
        string sourceLanguage,
        CancellationToken cancellationToken)
    {
        TranslationRoutingContext context = await BuildContextAsync(cancellationToken).ConfigureAwait(false);
        if (string.IsNullOrWhiteSpace(sourceLanguage) || sourceLanguage.Equals("auto", StringComparison.OrdinalIgnoreCase) || sourceLanguage.Equals("und", StringComparison.OrdinalIgnoreCase))
        {
            return TranslationLanguageCoverageMatrix.AllLanguages
                .Concat(context.DirectEntriesByPair.Keys.Select(pair => new TranslationLanguageDefinition(pair.TargetLanguage, pair.TargetLanguage, pair.TargetLanguage)))
                .DistinctBy(language => language.Code)
                .Select(language => new TranslationTargetLanguageOption(language.Code, language.DisplayName,
                    TranslationRoutingKind.Unavailable, false, "Translation model selection waits for source language detection.",
                    TranslationPreparationStatus.SourceUnresolved))
                .ToArray();
        }
        return TranslationLanguageCoverageMatrix.GetTargets(sourceLanguage)
            .Concat(context.DirectEntriesByPair.Keys.Where(pair => pair.SourceLanguage.Equals(sourceLanguage, StringComparison.OrdinalIgnoreCase))
                .Select(pair => new TranslationLanguageDefinition(pair.TargetLanguage, pair.TargetLanguage, pair.TargetLanguage)))
            .DistinctBy(language => language.Code)
            .Select(language =>
            {
                TranslationRouteSelection route = ResolveRoute(
                    context,
                    sourceLanguage,
                    language.Code,
                    language.DisplayName);
                return new TranslationTargetLanguageOption(
                    language.Code,
                    language.DisplayName,
                    route.RoutingKind,
                    route.IsAvailable,
                    route.IsAvailable
                        ? route.RouteDetail
                        : route.UnavailableReason ?? route.RouteDetail,
                    route.Status, route.ExpectedDownloadBytes);
            })
            .ToArray();
    }

    public async Task<TranslationRouteSelection> ResolveRouteAsync(
        string sourceLanguage,
        string targetLanguage,
        CancellationToken cancellationToken,
        string? preferredModelAlias = null)
    {
        TranslationRoutingContext context = await BuildContextAsync(cancellationToken).ConfigureAwait(false);
        return ResolveRoute(context, sourceLanguage, targetLanguage, displayName: null, preferredModelAlias);
    }

    private async Task<TranslationRoutingContext> BuildContextAsync(
        CancellationToken cancellationToken)
    {
        if (modelProvisioner is not null) await modelProvisioner.RefreshAsync(cancellationToken).ConfigureAwait(false);
        IReadOnlyList<LocalModelCacheRecord> cacheRecords = await modelCacheInventory.LoadAsync(cancellationToken).ConfigureAwait(false);
        IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> cacheIndex = cacheRecords
            .GroupBy(record => record.ModelId, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(
                group => group.Key,
                group => (IReadOnlyList<LocalModelCacheRecord>)group
                    .OrderByDescending(record => record.CachedAtUtc)
                    .ToArray(),
                StringComparer.OrdinalIgnoreCase);

        var manifestEntries = manifestRegistry.Entries;
        Dictionary<(string SourceLanguage, string TargetLanguage), IReadOnlyList<BundledModelManifestEntry>> directEntriesByPair = manifestEntries
            .Where(static entry => string.Equals(entry.Task, "translation", StringComparison.OrdinalIgnoreCase) &&
                                   IsDirectTranslationEntry(entry))
            .SelectMany(entry => EnumerateDirectPairs(entry).Select(pair => (Entry: entry, Pair: pair)))
            .GroupBy(candidate => candidate.Pair)
            .ToDictionary(
                group => group.Key,
                group => (IReadOnlyList<BundledModelManifestEntry>)group.Select(candidate => candidate.Entry)
                    .OrderBy(entry => entry.LanguageCoverage.LanguagePairs.Count == 1 ? 0 : 1)
                    .ThenBy(entry => cacheIndex.ContainsKey(entry.ModelId) ? 0 : 1)
                    .ThenBy(entry => entry.SourceUrl.Contains("/onnx-community/", StringComparison.Ordinal) ? 0 : 1)
                    .ThenBy(entry => entry.ModelId, StringComparer.Ordinal).ToArray());

        BundledModelManifestEntry[] pivotEntries = manifestEntries.Where(entry =>
            string.Equals(entry.Task, "translation", StringComparison.OrdinalIgnoreCase) &&
            IsPivotTranslationEntry(entry))
            .ToArray();

        IReadOnlyDictionary<string, bool> planAvailabilityByModelId = await BuildPlanAvailabilityAsync(
            directEntriesByPair.Values.SelectMany(static entries => entries).Concat(pivotEntries),
            cancellationToken).ConfigureAwait(false);

        var settings = settingsService is null ? null : await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
        return new TranslationRoutingContext(
            cacheIndex,
            directEntriesByPair,
            pivotEntries,
            planAvailabilityByModelId,
            settings is null || (!settings.OfflineMode && settings.AutoDownloadTranslationModels));
    }

    /// <summary>
    /// Asks the shared <see cref="IRuntimePlanner"/> whether each candidate translation model
    /// is currently runnable. The router only marks a route as available when the planner
    /// agrees, so model integrity checks, EP availability, and smoke tests are honored
    /// uniformly with the rest of the pipeline.
    /// When no planner is configured (legacy test wiring) the router falls back to the
    /// inline cache+file checks below.
    /// </summary>
    private async Task<IReadOnlyDictionary<string, bool>> BuildPlanAvailabilityAsync(
        IEnumerable<BundledModelManifestEntry> entries,
        CancellationToken cancellationToken)
    {
        if (runtimePlanner is null)
        {
            return new Dictionary<string, bool>(StringComparer.OrdinalIgnoreCase);
        }

        var availability = new Dictionary<string, bool>(StringComparer.OrdinalIgnoreCase);
        foreach (BundledModelManifestEntry entry in entries.DistinctBy(e => e.ModelId, StringComparer.OrdinalIgnoreCase))
        {
            string? primaryAlias = entry.Aliases.FirstOrDefault();
            try
            {
                StageRuntimePlan plan = await runtimePlanner.PlanAsync(
                    new StageRuntimePlanningRequest(
                        Stage: RuntimeStage.Translation,
                        PreferredModelAlias: primaryAlias,
                        PreferredEngineFamily: entry.EngineFamily,
                        RequirePreferredModelAlias: !string.IsNullOrWhiteSpace(primaryAlias)),
                    cancellationToken).ConfigureAwait(false);
                availability[entry.ModelId] = plan.IsRunnable();
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                throw;
            }
            catch
            {
                // Planner failure is treated as "not runnable" — a strictly more conservative
                // outcome than silently masking an underlying problem with the planner stack.
                availability[entry.ModelId] = false;
            }
        }

        return availability;
    }

    private bool IsPlanRunnableOrUngated(TranslationRoutingContext context, BundledModelManifestEntry entry)
    {
        // When no planner is configured, fall back to the inline file checks each TryResolve
        // method already performs. With a planner present, route availability requires the
        // planner to confirm the model is in a runnable status (Ready or Verified).
        if (runtimePlanner is null)
        {
            return true;
        }

        return context.PlanAvailabilityByModelId.TryGetValue(entry.ModelId, out bool isRunnable) && isRunnable;
    }

    private TranslationRouteSelection ResolveRoute(
        TranslationRoutingContext context,
        string sourceLanguage,
        string targetLanguage,
        string? displayName,
        string? preferredModelAlias = null)
    {
        string normalizedSourceLanguage = NormalizeLanguageCode(sourceLanguage)
            ?? throw new InvalidOperationException("Source language is required.");
        string normalizedTargetLanguage = NormalizeLanguageCode(targetLanguage)
            ?? throw new InvalidOperationException("Target language is required.");

        if (normalizedSourceLanguage is "auto" or "und")
        {
            return new TranslationRouteSelection(normalizedSourceLanguage, normalizedTargetLanguage,
                TranslationRoutingKind.Unavailable, false, "none", "Waiting for source language detection",
                PreparationStatus: TranslationPreparationStatus.SourceUnresolved);
        }

        TranslationLanguageCoverageMatrix.TryGetLanguage(normalizedTargetLanguage, out TranslationLanguageDefinition? targetDefinition);
        if (!context.DirectEntriesByPair.ContainsKey((normalizedSourceLanguage, normalizedTargetLanguage)) &&
            (targetDefinition is null ||
            !TranslationLanguageCoverageMatrix.GetTargets(normalizedSourceLanguage)
                .Any(candidate => string.Equals(candidate.Code, normalizedTargetLanguage, StringComparison.Ordinal))))
        {
            return new TranslationRouteSelection(
                normalizedSourceLanguage,
                normalizedTargetLanguage,
                TranslationRoutingKind.Unavailable,
                IsAvailable: false,
                ProviderName: "none",
                RouteDetail: "Unavailable",
                UnavailableReason: $"No approved translation route is known for {normalizedSourceLanguage} -> {normalizedTargetLanguage}.");
        }

        string resolvedDisplayName = displayName ?? targetDefinition?.DisplayName ?? normalizedTargetLanguage;
        if (TryResolvePreferredRoute(
            context,
            normalizedSourceLanguage,
            normalizedTargetLanguage,
            preferredModelAlias,
            out TranslationRouteSelection? preferredRoute))
        {
            return preferredRoute!;
        }

        // An installed direct model (e.g. a multi-pair opus-mt-en-ROMANCE) must win over an uninstalled
        // single-pair entry that sorts ahead of it; only an explicit alias forces that alias's download.
        if (string.IsNullOrWhiteSpace(preferredModelAlias) &&
            TryResolveDirectRoute(context, normalizedSourceLanguage, normalizedTargetLanguage, out TranslationRouteSelection? directRoute))
        {
            return directRoute!;
        }

        var downloadable = GetDirectEntriesForPair(context, normalizedSourceLanguage, normalizedTargetLanguage)
            .FirstOrDefault(entry => entry.CommercialAllowed && entry.CommercialUseVerified && entry.DownloadFiles.Count > 0 &&
                (string.IsNullOrWhiteSpace(preferredModelAlias) || HasAlias(entry, preferredModelAlias)));
        if (context.CanDownload && downloadable is not null && !context.CacheIndex.ContainsKey(downloadable.ModelId))
            return new TranslationRouteSelection(normalizedSourceLanguage, normalizedTargetLanguage,
                TranslationRoutingKind.Direct, false, ResolveProviderName(downloadable), "Download required",
                ModelId: downloadable.ModelId, PreferredModelAlias: downloadable.Aliases.FirstOrDefault(),
                EngineFamily: downloadable.EngineFamily, PreparationStatus: TranslationPreparationStatus.DownloadRequired,
                ExpectedDownloadBytes: downloadable.ExpectedDownloadBytes);

        if (!string.IsNullOrWhiteSpace(preferredModelAlias))
        {
            return CreateUnavailablePreferredRoute(normalizedSourceLanguage, normalizedTargetLanguage, preferredModelAlias);
        }

        if (TryResolveMadladPivotRoute(context, normalizedSourceLanguage, normalizedTargetLanguage, out TranslationRouteSelection? madladRoute))
        {
            return madladRoute!;
        }

        if (TryResolvePivotRoute(context, normalizedSourceLanguage, normalizedTargetLanguage, out TranslationRouteSelection? pivotRoute))
        {
            return pivotRoute!;
        }

        if (TryResolveGenAiPivotRoute(context, normalizedSourceLanguage, normalizedTargetLanguage, out TranslationRouteSelection? genAiPivotRoute))
        {
            return genAiPivotRoute!;
        }

        string pivotLabel = ResolveUnavailablePivotLabel(context.PivotEntries);
        string missingReason = context.DirectEntriesByPair.ContainsKey((normalizedSourceLanguage, normalizedTargetLanguage))
            ? $"Direct translation is not installed for {resolvedDisplayName}, and {pivotLabel} is unavailable."
            : $"{pivotLabel} is unavailable for {resolvedDisplayName}.";

        return new TranslationRouteSelection(
            normalizedSourceLanguage,
            normalizedTargetLanguage,
            TranslationRoutingKind.Unavailable,
            IsAvailable: false,
            ProviderName: "none",
            RouteDetail: "Unavailable",
            UnavailableReason: missingReason);
    }

    private static bool HasAlias(BundledModelManifestEntry entry, string alias) =>
        entry.Aliases.Any(candidate => candidate.Equals(alias.Trim(), StringComparison.OrdinalIgnoreCase));

    private bool TryResolvePreferredRoute(
        TranslationRoutingContext context,
        string sourceLanguage,
        string targetLanguage,
        string? preferredModelAlias,
        out TranslationRouteSelection? route)
    {
        route = null;
        if (string.IsNullOrWhiteSpace(preferredModelAlias))
        {
            return false;
        }

        foreach (BundledModelManifestEntry entry in GetDirectEntriesForPair(context, sourceLanguage, targetLanguage)
                     .Concat(GetPivotEntriesForPair(context, sourceLanguage, targetLanguage))
                     .Where(entry => HasAlias(entry, preferredModelAlias)))
        {
            if (HasDirectPair(entry, sourceLanguage, targetLanguage) &&
                TryResolveDirectRoute(context, sourceLanguage, targetLanguage, entry, out route))
            {
                return true;
            }

            if (IsPivotTranslationEntry(entry) &&
                TryResolvePivotRoute(context, sourceLanguage, targetLanguage, entry, out route))
            {
                return true;
            }

            if (IsGenAiPivotEngineFamily(entry.EngineFamily) &&
                TryResolveGenAiPivotRoute(context, sourceLanguage, targetLanguage, entry, out route))
            {
                return true;
            }
        }

        return TryResolvePreferredGenAiDownload(context, sourceLanguage, targetLanguage, preferredModelAlias, out route);
    }

    private static bool TryResolvePreferredGenAiDownload(
        TranslationRoutingContext context,
        string sourceLanguage,
        string targetLanguage,
        string preferredModelAlias,
        out TranslationRouteSelection? route)
    {
        route = null;
        if (!context.CanDownload)
        {
            return false;
        }

        BundledModelManifestEntry? entry = GetPivotEntriesForPair(context, sourceLanguage, targetLanguage)
            .FirstOrDefault(candidate => IsUninstalledPreferredGenAiDownload(context, candidate, preferredModelAlias));
        if (entry is null)
        {
            return false;
        }

        route = new TranslationRouteSelection(sourceLanguage, targetLanguage,
            TranslationRoutingKind.Pivot, false, entry.EngineFamily, "Download required",
            ModelId: entry.ModelId, PreferredModelAlias: entry.Aliases.FirstOrDefault(),
            EngineFamily: entry.EngineFamily, PreparationStatus: TranslationPreparationStatus.DownloadRequired,
            ExpectedDownloadBytes: entry.ExpectedDownloadBytes);
        return true;
    }

    private static bool IsUninstalledPreferredGenAiDownload(
        TranslationRoutingContext context,
        BundledModelManifestEntry candidate,
        string preferredModelAlias) =>
        IsPreferredGenAiEntry(candidate, preferredModelAlias) && IsUninstalledCommercialDownload(context, candidate);

    private static bool IsPreferredGenAiEntry(BundledModelManifestEntry candidate, string preferredModelAlias) =>
        HasAlias(candidate, preferredModelAlias) && IsGenAiPivotEngineFamily(candidate.EngineFamily);

    private static bool IsUninstalledCommercialDownload(TranslationRoutingContext context, BundledModelManifestEntry candidate)
    {
        if (context.CacheIndex.ContainsKey(candidate.ModelId))
        {
            return false;
        }

        bool commerciallyUsable = candidate.CommercialAllowed && candidate.CommercialUseVerified;
        bool hasDownloadFiles = candidate.DownloadFiles.Count > 0 || candidate.Variants.Any(variant => variant.DownloadFiles.Count > 0);
        return commerciallyUsable && hasDownloadFiles;
    }

    private static TranslationRouteSelection CreateUnavailablePreferredRoute(
        string sourceLanguage,
        string targetLanguage,
        string preferredModelAlias)
    {
        bool gemmaRuntimeUnavailable = preferredModelAlias.Equals("gemma-4-e4b-translation", StringComparison.OrdinalIgnoreCase);
        string reason = gemmaRuntimeUnavailable
            ? "Gemma 4 E4B translation is not available until its runtime and memory requirements have been validated."
            : $"Selected translation model '{preferredModelAlias}' is not available for this language pair.";
        return new TranslationRouteSelection(sourceLanguage, targetLanguage,
            TranslationRoutingKind.Unavailable, false, gemmaRuntimeUnavailable ? "gemma-genai" : "none", "Unavailable",
            UnavailableReason: reason);
    }

    private bool TryResolveMadladPivotRoute(
        TranslationRoutingContext context,
        string sourceLanguage,
        string targetLanguage,
        out TranslationRouteSelection? route)
    {
        route = null;
        foreach (BundledModelManifestEntry entry in GetPivotEntriesForPair(context, sourceLanguage, targetLanguage)
                     .Where(static entry => string.Equals(entry.EngineFamily, "madlad", StringComparison.OrdinalIgnoreCase)))
        {
            if (TryResolvePivotRoute(context, sourceLanguage, targetLanguage, entry, out route))
            {
                return true;
            }
        }

        return false;
    }

    private bool TryResolveDirectRoute(
        TranslationRoutingContext context,
        string sourceLanguage,
        string targetLanguage,
        out TranslationRouteSelection? route)
    {
        route = null;
        foreach (BundledModelManifestEntry entry in GetDirectEntriesForPair(context, sourceLanguage, targetLanguage))
        {
            if (TryResolveDirectRoute(context, sourceLanguage, targetLanguage, entry, out route))
            {
                return true;
            }
        }

        return false;
    }

    private bool TryResolveDirectRoute(
        TranslationRoutingContext context,
        string sourceLanguage,
        string targetLanguage,
        BundledModelManifestEntry entry,
        out TranslationRouteSelection? route)
    {
        route = null;
        if (!IsPlanRunnableOrUngated(context, entry))
        {
            return false;
        }

        foreach (string entryPath in EnumerateResolvedEntryPaths(context.CacheIndex, entry))
        {
            string modelRootPath = ResolveOpusModelRootPath(entryPath);
            if (!HasOpusSupportingFiles(modelRootPath))
            {
                continue;
            }

            route = new TranslationRouteSelection(
                sourceLanguage,
                targetLanguage,
                TranslationRoutingKind.Direct,
                IsAvailable: true,
                ProviderName: ResolveProviderName(entry),
                RouteDetail: ResolveDirectRouteDetail(entry),
                ModelId: entry.ModelId,
                PreferredModelAlias: entry.Aliases.FirstOrDefault(),
                ResolvedModelEntryPath: entryPath,
                EngineFamily: entry.EngineFamily,
                ModelProvenance: new Trackdub.Domain.Translation.TranslationModelProvenance(
                    ResolveProvenanceRepository(entry), entry.Revision, entry.Sha256,
                    new Dictionary<string, string>(entry.DownloadFileHashes),
                    entry.TargetPrefixes?.GetValueOrDefault(targetLanguage), entry.CatalogPolicyVersion));
            return true;
        }

        return false;
    }

    private bool TryResolvePivotRoute(
        TranslationRoutingContext context,
        string sourceLanguage,
        string targetLanguage,
        out TranslationRouteSelection? route)
    {
        route = null;
        foreach (BundledModelManifestEntry entry in GetPivotEntriesForPair(context, sourceLanguage, targetLanguage))
        {
            if (TryResolvePivotRoute(context, sourceLanguage, targetLanguage, entry, out route))
            {
                return true;
            }
        }

        return false;
    }

    private bool TryResolvePivotRoute(
        TranslationRoutingContext context,
        string sourceLanguage,
        string targetLanguage,
        BundledModelManifestEntry entry,
        out TranslationRouteSelection? route)
    {
        route = null;
        if (!IsPlanRunnableOrUngated(context, entry))
        {
            return false;
        }

        foreach (string entryPath in EnumerateResolvedMadladEntryPaths(context.CacheIndex, entry))
        {
            string modelRootPath = Path.GetDirectoryName(entryPath)
                ?? throw new InvalidOperationException("MADLAD model root path could not be resolved.");
            if (!HasMadladSupportingFiles(modelRootPath))
            {
                continue;
            }

            string routeDetail = ResolvePivotRouteDetail(
                entry,
                context.DirectEntriesByPair.ContainsKey((sourceLanguage, targetLanguage)));

            route = new TranslationRouteSelection(
                sourceLanguage,
                targetLanguage,
                TranslationRoutingKind.Pivot,
                IsAvailable: true,
                ProviderName: ResolveProviderName(entry),
                RouteDetail: routeDetail,
                ModelId: entry.ModelId,
                PreferredModelAlias: entry.Aliases.FirstOrDefault(),
                ResolvedModelEntryPath: entryPath,
                EngineFamily: entry.EngineFamily,
                ModelProvenance: new Trackdub.Domain.Translation.TranslationModelProvenance(
                    ResolveProvenanceRepository(entry), entry.Revision, entry.Sha256,
                    new Dictionary<string, string>(entry.DownloadFileHashes),
                    entry.TargetPrefixes?.GetValueOrDefault(targetLanguage), entry.CatalogPolicyVersion));
            return true;
        }

        return false;
    }

    private bool TryResolveGenAiPivotRoute(
        TranslationRoutingContext context,
        string sourceLanguage,
        string targetLanguage,
        out TranslationRouteSelection? route)
    {
        route = null;
        foreach (BundledModelManifestEntry entry in GetPivotEntriesForPair(context, sourceLanguage, targetLanguage)
            .Where(static e => IsGenAiPivotEngineFamily(e.EngineFamily)))
        {
            if (TryResolveGenAiPivotRoute(context, sourceLanguage, targetLanguage, entry, out route))
            {
                return true;
            }
        }

        return false;
    }

    private bool TryResolveGenAiPivotRoute(
        TranslationRoutingContext context,
        string sourceLanguage,
        string targetLanguage,
        BundledModelManifestEntry entry,
        out TranslationRouteSelection? route)
    {
        route = null;
        if (!IsPlanRunnableOrUngated(context, entry))
        {
            return false;
        }

        foreach (string entryPath in EnumerateResolvedEntryPaths(context.CacheIndex, entry, includeVariants: true))
        {
            string? modelRootPath = Path.GetDirectoryName(entryPath);
            if (modelRootPath is null || !HasGenAiSupportingFiles(modelRootPath))
            {
                continue;
            }

            route = new TranslationRouteSelection(
                sourceLanguage,
                targetLanguage,
                TranslationRoutingKind.Pivot,
                IsAvailable: true,
                ProviderName: entry.EngineFamily,
                RouteDetail: ResolvePivotRouteLabel(entry),
                ModelId: entry.ModelId,
                PreferredModelAlias: entry.Aliases.FirstOrDefault(),
                ResolvedModelEntryPath: entryPath,
                EngineFamily: entry.EngineFamily,
                ModelProvenance: GenAiPivotProvenanceResolver.Resolve(new GenAiPivotProvenanceRequest
                {
                    CacheIndex = context.CacheIndex,
                    Entry = entry,
                    ResolvedEntryPath = entryPath,
                    TargetLanguage = targetLanguage
                }));
            return true;
        }

        return false;
    }

    private static bool HasGenAiSupportingFiles(string modelRootPath) =>
        File.Exists(Path.Join(modelRootPath, "genai_config.json"));

    // Provenance records a Hugging Face repo id, not a URL. The source URL's repo (not ModelId) is the one
    // the recorded revision and hashes belong to; MADLAD's ModelId is google/madlad400-3b-mt but its
    // files come from tonythethompson/madlad400-3b-mt-onnx.
    internal static string ResolveProvenanceRepository(BundledModelManifestEntry entry)
    {
        if (Uri.TryCreate(entry.SourceUrl, UriKind.Absolute, out Uri? uri)
            && uri.Scheme == Uri.UriSchemeHttps
            && string.Equals(uri.Host, "huggingface.co", StringComparison.OrdinalIgnoreCase))
        {
            // AbsolutePath already excludes any query string or fragment.
            string[] segments = uri.AbsolutePath.Split('/', StringSplitOptions.RemoveEmptyEntries);
            bool isRepositoryPage = segments.Length == 2
                || (segments.Length > 2 && segments[2] is "tree" or "blob" or "resolve");
            if (isRepositoryPage)
            {
                return segments[0] + "/" + segments[1];
            }
        }

        return entry.ModelId;
    }

    private static IReadOnlyList<BundledModelManifestEntry> GetDirectEntriesForPair(
        TranslationRoutingContext context,
        string sourceLanguage,
        string targetLanguage) =>
        context.DirectEntriesByPair.TryGetValue((sourceLanguage, targetLanguage), out IReadOnlyList<BundledModelManifestEntry>? entries)
            ? entries
            : [];

    private static IEnumerable<BundledModelManifestEntry> GetPivotEntriesForPair(
        TranslationRoutingContext context,
        string sourceLanguage,
        string targetLanguage) =>
        context.PivotEntries.Where(entry => CoversLanguagePair(entry.LanguageCoverage, sourceLanguage, targetLanguage));

    private static IEnumerable<string> EnumerateResolvedEntryPaths(
        IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> cacheIndex,
        BundledModelManifestEntry entry,
        bool includeVariants = false)
    {
        if (!cacheIndex.TryGetValue(entry.ModelId, out IReadOnlyList<LocalModelCacheRecord>? records))
        {
            yield break;
        }

        foreach (LocalModelCacheRecord record in records)
        {
            IEnumerable<string> paths = includeVariants
                ? entry.Variants.Select(variant => variant.EntryPath).Prepend(entry.DefaultBenchmarkEntryPath)
                : [entry.DefaultBenchmarkEntryPath];
            foreach (string candidatePath in paths.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                string relativeEntryPath = Path.GetRelativePath(entry.RootDirectory, candidatePath);
                string resolvedPath = Path.GetFullPath(Path.Join(record.RootPath, relativeEntryPath));
                if (File.Exists(resolvedPath))
                {
                    yield return resolvedPath;
                }
            }
        }
    }

    private static IEnumerable<string> EnumerateResolvedMadladEntryPaths(
        IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> cacheIndex,
        BundledModelManifestEntry entry)
    {
        if (!cacheIndex.TryGetValue(entry.ModelId, out IReadOnlyList<LocalModelCacheRecord>? records))
        {
            yield break;
        }

        foreach (LocalModelCacheRecord record in records)
        {
            foreach (string candidatePath in EnumerateMadladEncoderEntryPaths(entry))
            {
                string relativeEntryPath = Path.GetRelativePath(entry.RootDirectory, candidatePath);
                string resolvedPath = Path.GetFullPath(Path.Join(record.RootPath, relativeEntryPath));
                if (File.Exists(resolvedPath))
                {
                    yield return resolvedPath;
                }
            }
        }
    }

    private static IEnumerable<string> EnumerateMadladEncoderEntryPaths(BundledModelManifestEntry entry)
    {
        yield return entry.DefaultBenchmarkEntryPath;

        foreach (BundledModelManifestVariant variant in entry.Variants)
        {
            string fileName = Path.GetFileName(variant.EntryPath);
            if (fileName.StartsWith("encoder_model", StringComparison.OrdinalIgnoreCase))
            {
                yield return variant.EntryPath;
            }
        }
    }

    private static bool HasOpusSupportingFiles(string modelRootPath)
    {
        string onnxDirectory = Path.Join(modelRootPath, "onnx");
        return (File.Exists(Path.Join(modelRootPath, "decoder_model.onnx")) ||
                File.Exists(Path.Join(modelRootPath, "decoder_model_merged.onnx")) ||
                File.Exists(Path.Join(onnxDirectory, "decoder_model.onnx")) ||
                File.Exists(Path.Join(onnxDirectory, "decoder_model_merged.onnx"))) &&
               File.Exists(Path.Join(modelRootPath, "vocab.json")) &&
               (File.Exists(Path.Join(modelRootPath, "source.spm")) || File.Exists(Path.Join(modelRootPath, "source.model"))) &&
               (File.Exists(Path.Join(modelRootPath, "target.spm")) || File.Exists(Path.Join(modelRootPath, "target.model")));
    }

    private static string ResolveOpusModelRootPath(string entryPath)
    {
        string entryDirectory = Path.GetDirectoryName(entryPath)
            ?? throw new InvalidOperationException("Direct Opus model root path could not be resolved.");
        if (HasOpusTokenizerFiles(entryDirectory))
        {
            return entryDirectory;
        }

        string? parentDirectory = Path.GetDirectoryName(entryDirectory);
        return parentDirectory is not null && HasOpusTokenizerFiles(parentDirectory)
            ? parentDirectory
            : entryDirectory;
    }

    private static bool HasOpusTokenizerFiles(string modelRootPath) =>
        File.Exists(Path.Join(modelRootPath, "vocab.json")) &&
        (File.Exists(Path.Join(modelRootPath, "source.spm")) || File.Exists(Path.Join(modelRootPath, "source.model"))) &&
        (File.Exists(Path.Join(modelRootPath, "target.spm")) || File.Exists(Path.Join(modelRootPath, "target.model")));

    private static bool HasMadladSupportingFiles(string modelRootPath)
    {
        return (File.Exists(Path.Join(modelRootPath, "decoder_model.onnx")) ||
                File.Exists(Path.Join(modelRootPath, "decoder_model_merged.onnx")) ||
                File.Exists(Path.Join(modelRootPath, "decoder_model_quantized.onnx")) ||
                File.Exists(Path.Join(modelRootPath, "decoder_model_int8.onnx"))) &&
               (File.Exists(Path.Join(modelRootPath, "spiece.model")) ||
                File.Exists(Path.Join(modelRootPath, "tokenizer.model")) ||
                File.Exists(Path.Join(modelRootPath, "sentencepiece.model")));
    }

    private static (string SourceLanguage, string TargetLanguage)? NormalizePair(ModelLanguagePair pair)
    {
        string? sourceLanguage = NormalizeLanguageCode(pair.SourceLanguage);
        string? targetLanguage = NormalizeLanguageCode(pair.TargetLanguage);
        return sourceLanguage is null || targetLanguage is null
            ? null
            : (sourceLanguage, targetLanguage);
    }

    private static IEnumerable<(string SourceLanguage, string TargetLanguage)> EnumerateDirectPairs(BundledModelManifestEntry entry)
    {
        HashSet<(string SourceLanguage, string TargetLanguage)> pairs = [];
        foreach ((string SourceLanguage, string TargetLanguage)? explicitPair in entry.LanguageCoverage.LanguagePairs.Select(NormalizePair))
        {
            if (explicitPair is { } value && pairs.Add(value))
            {
                yield return value;
            }
        }
    }

    private static bool HasDirectPair(
        BundledModelManifestEntry entry,
        string sourceLanguage,
        string targetLanguage) =>
        entry.LanguageCoverage.LanguagePairs.Any(pair =>
            string.Equals(pair.SourceLanguage, sourceLanguage, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(pair.TargetLanguage, targetLanguage, StringComparison.OrdinalIgnoreCase));

    private static bool CoversLanguagePair(
        ModelLanguageCoverage coverage,
        string sourceLanguage,
        string targetLanguage)
    {
        if (coverage.LanguagePairs.Select(NormalizePair).Any(pair => pair == (sourceLanguage, targetLanguage)))
        {
            return true;
        }

        return CoversLanguage(coverage.SourceLanguages, sourceLanguage) &&
               CoversLanguage(coverage.TargetLanguages, targetLanguage);
    }

    private static bool CoversLanguage(IReadOnlyList<string> supportedLanguages, string language) =>
        supportedLanguages.Any(candidate =>
            string.Equals(candidate, language, StringComparison.OrdinalIgnoreCase) ||
            string.Equals(candidate, "multi", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(candidate, "any", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(candidate, "*", StringComparison.OrdinalIgnoreCase));

    private static bool HasCapability(BundledModelManifestEntry entry, string capability) =>
        entry.Capabilities.Any(candidate => string.Equals(candidate, capability, StringComparison.OrdinalIgnoreCase));

    private static bool IsDirectTranslationEntry(BundledModelManifestEntry entry)
    {
        if (HasCapability(entry, "direct-translation"))
        {
            return true;
        }

        if (HasCapability(entry, "pivot-translation"))
        {
            return false;
        }

        return entry.LanguageCoverage.LanguagePairs.Count > 0;
    }

    private static bool IsPivotTranslationEntry(BundledModelManifestEntry entry)
    {
        if (HasCapability(entry, "pivot-translation"))
        {
            return true;
        }

        if (HasCapability(entry, "direct-translation"))
        {
            return false;
        }

        return entry.LanguageCoverage.SourceLanguages.Count > 0 ||
               entry.LanguageCoverage.TargetLanguages.Count > 0;
    }

    private static string ResolveProviderName(BundledModelManifestEntry entry) =>
        entry.EngineFamily.ToLowerInvariant() switch
        {
            "madlad" => "madlad400",
            var family => family
        };

    private static string ResolveDirectRouteDetail(BundledModelManifestEntry entry) =>
        string.Equals(entry.EngineFamily, "opus-mt", StringComparison.OrdinalIgnoreCase)
            ? "Direct Opus-MT"
            : $"Direct {entry.EngineFamily}";

    private static string ResolvePivotRouteDetail(BundledModelManifestEntry entry, bool directPairKnown)
    {
        string label = ResolvePivotRouteLabel(entry);
        return directPairKnown
            ? $"{label} (direct translation pair missing)"
            : label;
    }

    private static string ResolveUnavailablePivotLabel(IReadOnlyList<BundledModelManifestEntry> pivotEntries) =>
        pivotEntries.Count == 0
            ? "pivot translation"
            : ResolvePivotRouteLabel(pivotEntries[0]);

    private static string ResolvePivotRouteLabel(BundledModelManifestEntry entry) =>
        entry.EngineFamily.ToLowerInvariant() switch
        {
            "madlad" => "MADLAD-400 pivot",
            var family when IsGenAiPivotEngineFamily(family) => "Multilingual pivot",
            _ => "Pivot translation"
        };

    private static string? NormalizeLanguageCode(string? languageCode) =>
        string.IsNullOrWhiteSpace(languageCode)
            ? null
            : languageCode.Trim().ToLowerInvariant();

    private sealed record TranslationRoutingContext(
        IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> CacheIndex,
        IReadOnlyDictionary<(string SourceLanguage, string TargetLanguage), IReadOnlyList<BundledModelManifestEntry>> DirectEntriesByPair,
        IReadOnlyList<BundledModelManifestEntry> PivotEntries,
        IReadOnlyDictionary<string, bool> PlanAvailabilityByModelId,
        bool CanDownload);
}
