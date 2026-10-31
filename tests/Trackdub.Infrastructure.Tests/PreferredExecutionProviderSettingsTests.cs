using Trackdub.Contracts;
using Trackdub.Domain;
using Trackdub.Infrastructure.Settings;

namespace Trackdub.Infrastructure.Tests;

public sealed class PreferredExecutionProviderSettingsTests
{
    [Fact]
    public async Task Save_and_load_round_trip_preferred_execution_provider()
    {
        string tempRoot = Path.Join(Path.GetTempPath(), "Trackdub.Infrastructure.Tests", Guid.NewGuid().ToString("N"));
        var storagePaths = new TrackdubStoragePaths(tempRoot);
        var service = new JsonStudioSettingsService(storagePaths);
        try
        {
            await service.SaveAsync(
                StudioSettings.Default with { PreferredExecutionProvider = ExecutionProviderKind.Cuda },
                TestContext.Current.CancellationToken);

            StudioSettings loaded = await service.LoadAsync(TestContext.Current.CancellationToken);

            Assert.Equal(ExecutionProviderKind.Cuda, loaded.PreferredExecutionProvider);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public async Task Load_without_preferred_provider_uses_automatic_selection()
    {
        string tempRoot = Path.Join(Path.GetTempPath(), "Trackdub.Infrastructure.Tests", Guid.NewGuid().ToString("N"));
        var storagePaths = new TrackdubStoragePaths(tempRoot);
        var service = new JsonStudioSettingsService(storagePaths);
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(storagePaths.SettingsPath)!);
            await File.WriteAllTextAsync(
                storagePaths.SettingsPath,
                "{\"defaultSourceLanguage\":\"es\",\"defaultTargetLanguage\":\"fr\",\"recentProjects\":[]}",
                TestContext.Current.CancellationToken);

            StudioSettings loaded = await service.LoadAsync(TestContext.Current.CancellationToken);

            Assert.Null(loaded.PreferredExecutionProvider);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }
}
