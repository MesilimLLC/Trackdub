using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Runtime.Planning;

internal sealed class RuntimePlannerRankingStrategy(BundledModelManifestRegistry manifestRegistry)
{
    private readonly BundledModelManifestRegistry manifestRegistry = manifestRegistry ?? throw new ArgumentNullException(nameof(manifestRegistry));

    public RankedManifestEntry[] RankEntries(
        StageRuntimePlanningRequest request,
        StageRuntimeRequirements requirements)
    {
        return manifestRegistry.Entries
            .Where(entry => string.Equals(entry.Task, requirements.RequiredTask.ToManifestValue(), StringComparison.OrdinalIgnoreCase))
            .Where(entry => IsEngineFamilyAllowed(entry, requirements, request))
            .Where(entry => HasRequiredCapabilities(entry, requirements))
            .Where(entry => IsLanguageCompatible(entry, request))
            .Select(entry => new RankedManifestEntry(
                entry,
                GetSelectionRank(entry, request, requirements)))
            .OrderBy(candidate => candidate.Rank)
            .ThenBy(candidate => candidate.Entry.Aliases.Count > 0 ? candidate.Entry.Aliases[0] : candidate.Entry.ModelId, StringComparer.OrdinalIgnoreCase)
            .ThenBy(candidate => candidate.Entry.ModelId, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static bool IsEngineFamilyAllowed(
        BundledModelManifestEntry entry,
        StageRuntimeRequirements requirements,
        StageRuntimePlanningRequest request)
    {
        if (requirements.AllowedEngineFamilies is null ||
            requirements.AllowedEngineFamilies.Count == 0 ||
            requirements.AllowedEngineFamilies.Any(engineFamily =>
                entry.EngineFamily.Equals(engineFamily, StringComparison.OrdinalIgnoreCase)))
        {
            return true;
        }

        // Explicit preferred alias may still target a model outside the auto-planning
        // allowlist (e.g. diagnostic Nemotron). Auto/quality ranking must not.
        return !string.IsNullOrWhiteSpace(request.NormalizedPreferredModelAlias) &&
            ManifestEntryMatchesPreferredAlias(entry, request.NormalizedPreferredModelAlias);
    }

    private static bool HasRequiredCapabilities(
        BundledModelManifestEntry entry,
        StageRuntimeRequirements requirements) =>
        requirements.RequiredCapabilities is null ||
        requirements.RequiredCapabilities.Count == 0 ||
        requirements.RequiredCapabilities.All(requiredCapability =>
            entry.Capabilities.Any(capability =>
                capability.Equals(requiredCapability, StringComparison.OrdinalIgnoreCase)));

    private static bool IsLanguageCompatible(
        BundledModelManifestEntry entry,
        StageRuntimePlanningRequest request)
    {
        if (string.Equals(entry.Task, ModelTask.Asr.ToManifestValue(), StringComparison.OrdinalIgnoreCase))
        {
            return IsAsrLanguageCompatible(entry, request);
        }

        if (!string.Equals(entry.Task, ModelTask.Translation.ToManifestValue(), StringComparison.OrdinalIgnoreCase) ||
            string.IsNullOrWhiteSpace(request.SourceLanguage) ||
            string.IsNullOrWhiteSpace(request.TargetLanguage))
        {
            return true;
        }

        string sourceLanguage = request.SourceLanguage.Trim();
        string targetLanguage = request.TargetLanguage.Trim();
        if (entry.LanguageCoverage.LanguagePairs.Count > 0)
        {
            return entry.LanguageCoverage.LanguagePairs.Any(pair =>
                string.Equals(pair.SourceLanguage, sourceLanguage, StringComparison.OrdinalIgnoreCase) &&
                string.Equals(pair.TargetLanguage, targetLanguage, StringComparison.OrdinalIgnoreCase));
        }

        if (entry.LanguageCoverage.SourceLanguages.Count == 0 &&
            entry.LanguageCoverage.TargetLanguages.Count == 0)
        {
            return true;
        }

        return CoversLanguage(entry.LanguageCoverage.SourceLanguages, sourceLanguage) &&
               CoversLanguage(entry.LanguageCoverage.TargetLanguages, targetLanguage);
    }

    private static bool IsAsrLanguageCompatible(
        BundledModelManifestEntry entry,
        StageRuntimePlanningRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.SourceLanguage))
        {
            return DeclaresAsrLanguageDetection(entry);
        }

        string sourceLanguage = NormalizeAsrSourceLanguage(request.SourceLanguage.Trim()) ?? request.SourceLanguage.Trim();
        if (sourceLanguage.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            return DeclaresAsrLanguageDetection(entry);
        }

        return entry.LanguageCoverage.SourceLanguages.Count == 0 ||
               CoversLanguage(entry.LanguageCoverage.SourceLanguages, sourceLanguage);
    }

    private static bool DeclaresAsrLanguageDetection(BundledModelManifestEntry entry) =>
        entry.Capabilities.Any(capability =>
            capability.Equals("language-detection", StringComparison.OrdinalIgnoreCase)) ||
        entry.LanguageCoverage.SourceLanguages.Any(language =>
            language.Equals("auto", StringComparison.OrdinalIgnoreCase));

    private static string? NormalizeAsrSourceLanguage(string? languageCode)
    {
        if (string.IsNullOrWhiteSpace(languageCode))
        {
            return null;
        }

        string normalized = languageCode.Trim().ToLowerInvariant().Replace('_', '-');
        int separatorIndex = normalized.IndexOf('-', StringComparison.Ordinal);
        if (separatorIndex > 0)
        {
            normalized = normalized[..separatorIndex];
        }

        return normalized.Length is >= 2 and <= 8 &&
               normalized.All(static character => character is >= 'a' and <= 'z')
            ? normalized
            : null;
    }

    private static bool CoversLanguage(
        IReadOnlyList<string> languages,
        string language) =>
        languages.Any(candidate =>
            string.Equals(candidate, language, StringComparison.OrdinalIgnoreCase) ||
            string.Equals(candidate, "multi", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(candidate, "auto", StringComparison.OrdinalIgnoreCase));

    public static RankedManifestEntry[] FilterPreferredModelAliasEntries(
        StageRuntimePlanningRequest request,
        IReadOnlyList<RankedManifestEntry> rankedEntries)
    {
        if (!request.RequirePreferredModelAlias)
        {
            return rankedEntries.ToArray();
        }

        if (request.NormalizedPreferredModelAlias is not string preferredModelAlias)
        {
            return [];
        }

        return rankedEntries
            .Where(candidate => ManifestEntryMatchesPreferredAlias(candidate.Entry, preferredModelAlias))
            .ToArray();
    }

    /// <summary>
    /// Drops entries the machine cannot run automatically: the declared <c>min_vram_mb</c> is not met (no GPU, or less
    /// dedicated VRAM), or the engine family is verified on NVIDIA only and the GPU is not NVIDIA. An automatic choice then
    /// falls back to the next-ranked model. A request that names a model by alias is honoured as asked, and if no entry
    /// would remain the ranking is kept rather than leaving the stage with nothing to plan.
    /// </summary>
    public static RankedManifestEntry[] FilterEntriesTheHardwareCanRun(
        StageRuntimePlanningRequest request,
        StageRuntimeRequirements requirements,
        HardwareProfile hardware,
        RankedManifestEntry[] rankedEntries)
    {
        if (request.NormalizedPreferredModelAlias is not null)
        {
            return rankedEntries;
        }

        RankedManifestEntry[] runnable =
            [.. rankedEntries.Where(candidate => CanRunAutomatically(candidate.Entry, requirements, hardware))];
        return runnable.Length > 0 ? runnable : rankedEntries;
    }

    private static bool CanRunAutomatically(BundledModelManifestEntry entry, StageRuntimeRequirements requirements, HardwareProfile hardware) =>
        MeetsMinimumVram(entry, hardware) && (!RequiresNvidia(entry, requirements) || hardware.NvidiaGpuArchitecture != NvidiaGpuArchitectureBucket.Unknown);

    private static bool MeetsMinimumVram(BundledModelManifestEntry entry, HardwareProfile hardware) =>
        entry.MinVramMb <= 0 || (hardware.HasGpu && hardware.DedicatedVramMb >= entry.MinVramMb);

    private static bool RequiresNvidia(BundledModelManifestEntry entry, StageRuntimeRequirements requirements) =>
        requirements.NvidiaOnlyAutomaticEngineFamilies?.Contains(entry.EngineFamily, StringComparer.OrdinalIgnoreCase) == true;

    public static RankedManifestEntry[] FilterTopRankedEntriesIfRequired(
        StageRuntimeRequirements requirements,
        IReadOnlyList<RankedManifestEntry> rankedEntries)
    {
        if (!requirements.PreferTopRankedModelUntilReady || rankedEntries.Count == 0)
        {
            return rankedEntries.ToArray();
        }

        int topRank = rankedEntries.Min(candidate => candidate.Rank);
        return rankedEntries
            .Where(candidate => candidate.Rank == topRank)
            .ToArray();
    }

    private static int GetSelectionRank(
        BundledModelManifestEntry entry,
        StageRuntimePlanningRequest request,
        StageRuntimeRequirements requirements)
    {
        int rank = 0;

        if (!string.IsNullOrWhiteSpace(request.NormalizedPreferredModelAlias))
        {
            if (ManifestEntryMatchesPreferredAlias(entry, request.NormalizedPreferredModelAlias))
            {
                return 0;
            }

            rank += 10_000;
        }

        if (!string.IsNullOrWhiteSpace(request.PreferredEngineFamily) &&
            !entry.EngineFamily.Equals(request.PreferredEngineFamily.Trim(), StringComparison.OrdinalIgnoreCase))
        {
            rank += 1_000;
        }

        if (!string.IsNullOrWhiteSpace(request.PreferredModelTier) &&
            !entry.Tier.Equals(request.PreferredModelTier.Trim(), StringComparison.OrdinalIgnoreCase))
        {
            rank += 100;
        }

        int bestAliasRank = int.MaxValue / 2;
        for (int index = 0; index < requirements.PreferredModelAliases.Count; index++)
        {
            string preferredAlias = requirements.PreferredModelAliases[index];
            if (entry.Aliases.Any(alias => alias.Equals(preferredAlias, StringComparison.OrdinalIgnoreCase)))
            {
                bestAliasRank = Math.Min(bestAliasRank, index + 1);
            }
        }

        return rank + bestAliasRank;
    }

    private static bool ManifestEntryMatchesPreferredAlias(
        BundledModelManifestEntry entry,
        string preferredModelAlias) =>
        entry.Aliases.Any(alias => alias.Equals(preferredModelAlias, StringComparison.OrdinalIgnoreCase)) ||
        entry.ModelId.Equals(preferredModelAlias, StringComparison.OrdinalIgnoreCase);
}

internal sealed record RankedManifestEntry(
    BundledModelManifestEntry Entry,
    int Rank);
