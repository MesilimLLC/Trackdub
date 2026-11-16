using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private sealed record PlanAvailability(bool IsRunnable, string? PlannedEntryPath);

    private sealed record TranslationRoutingContext
    {
        public required IReadOnlyDictionary<string, IReadOnlyList<LocalModelCacheRecord>> CacheIndex { get; init; }

        public required IReadOnlyDictionary<(string SourceLanguage, string TargetLanguage), IReadOnlyList<BundledModelManifestEntry>> DirectEntriesByPair { get; init; }

        public required IReadOnlyList<BundledModelManifestEntry> PivotEntries { get; init; }

        public required IReadOnlyDictionary<string, PlanAvailability> PlanAvailabilityByModelId { get; init; }

        public bool CanDownload { get; init; }
    }

    private async Task<bool> ResolveCanDownloadAsync(CancellationToken cancellationToken)
    {
        if (settingsService is null)
        {
            return true;
        }

        var settings = await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
        return !settings.OfflineMode && settings.AutoDownloadTranslationModels;
    }
}
