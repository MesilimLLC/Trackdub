using Trackdub.Contracts;
using Trackdub.Infrastructure.Settings;

namespace Trackdub.Infrastructure.Tests;

public sealed class JsonStudioSettingsDefaultDirectoriesTests
{
    [Fact]
    public async Task Save_and_load_round_trip_default_project_export_and_model_cache_directories()
    {
        string tempRoot = Path.Join(Path.GetTempPath(), "Trackdub.Infrastructure.Tests", Guid.NewGuid().ToString("N"));
        var storagePaths = new TrackdubStoragePaths(tempRoot);
        var service = new JsonStudioSettingsService(storagePaths);
        StudioSettings settings = StudioSettings.Default with
        {
            DefaultProjectDirectory = @"D:\Projects",
            DefaultExportDirectory = @"D:\Exports",
            DefaultModelCacheDirectory = Path.Join(tempRoot, "models")
        };

        try
        {
            await service.SaveAsync(settings, TestContext.Current.CancellationToken);
            StudioSettings loaded = await service.LoadAsync(TestContext.Current.CancellationToken);

            Assert.Equal(settings.DefaultProjectDirectory, loaded.DefaultProjectDirectory);
            Assert.Equal(settings.DefaultExportDirectory, loaded.DefaultExportDirectory);
            Assert.Equal(settings.DefaultModelCacheDirectory, loaded.DefaultModelCacheDirectory);
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
