using Trackdub.Domain;

namespace Trackdub.Composition.Tests;

internal sealed class Downloader : IModelDownloadOrchestrator
{
    public string? Selected { get; private set; }
    public string? Variant { get; private set; }
    public IObservable<ModelStateChange> StateChanges => throw new NotSupportedException();
    public Task<ModelDownloadResult> DownloadAsync(string modelId, IProgress<ModelDownloadProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        Selected = modelId;
        return Task.FromResult(new ModelDownloadResult(modelId, true, ModelCacheState.Installed, null));
    }
    public Task<ModelDownloadResult> DownloadAsync(string modelId, string? variantAlias, IProgress<ModelDownloadProgress>? progress, CancellationToken cancellationToken = default)
    {
        Variant = variantAlias;
        return DownloadAsync(modelId, progress, cancellationToken);
    }
    public Task<ModelDownloadResult> RepairAsync(string modelId, IProgress<ModelDownloadProgress>? progress = null, CancellationToken cancellationToken = default) => throw new NotSupportedException();
    public Task<bool> UninstallAsync(string modelId, CancellationToken cancellationToken = default) => throw new NotSupportedException();
}
