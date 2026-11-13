using Trackdub.Contracts;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Tests;

internal sealed class RouterSettings(StudioSettings settings) : IStudioSettingsService
{
    public Task<StudioSettings> LoadAsync(CancellationToken cancellationToken) => Task.FromResult(settings);
    public Task SaveAsync(StudioSettings value, CancellationToken cancellationToken) => throw new NotSupportedException();
    public Task<StudioSettings> TouchRecentProjectAsync(string projectPath, string projectName, CancellationToken cancellationToken) => throw new NotSupportedException();
}

internal sealed class InMemoryModelCacheInventory(IReadOnlyList<LocalModelCacheRecord> records)
    : IModelCacheInventory
{
    public Task<IReadOnlyList<LocalModelCacheRecord>> LoadAsync(CancellationToken cancellationToken = default) =>
        Task.FromResult(records);
}
