using Trackdub.Domain;
using Trackdub.Infrastructure.Settings;

namespace Trackdub.Infrastructure.Tests;

public sealed class JsonStudioModelCachePersistenceTests : IDisposable
{
    private const string RootVariable = "TRACKDUB_TEST_MODEL_CACHE_ROOT";
    private readonly string root = Path.Join(Path.GetTempPath(), "Trackdub.ModelCachePersistence", Guid.NewGuid().ToString("N"));

    [Fact]
    public async Task Relative_saved_model_cache_directory_is_persisted_as_unset()
    {
        var service = new JsonStudioSettingsService(new TrackdubStoragePaths(root));
        StudioSettings settings = StudioSettings.Default with { DefaultModelCacheDirectory = "relative/cache" };

        await service.SaveAsync(settings, TestContext.Current.CancellationToken);
        StudioSettings loaded = await service.LoadAsync(TestContext.Current.CancellationToken);

        Assert.Null(loaded.DefaultModelCacheDirectory);
    }

    [Fact]
    public async Task Environment_variable_model_cache_directory_is_persisted_in_absolute_form()
    {
        string previous = Environment.GetEnvironmentVariable(RootVariable) ?? string.Empty;
        Environment.SetEnvironmentVariable(RootVariable, Path.Join(root, "from-env"));
        try
        {
            var service = new JsonStudioSettingsService(new TrackdubStoragePaths(root));
            StudioSettings settings = StudioSettings.Default with
            {
                DefaultModelCacheDirectory = $"%{RootVariable}%/models",
            };

            await service.SaveAsync(settings, TestContext.Current.CancellationToken);
            StudioSettings loaded = await service.LoadAsync(TestContext.Current.CancellationToken);

            Assert.Equal(Path.GetFullPath(Path.Join(root, "from-env", "models")), loaded.DefaultModelCacheDirectory);
        }
        finally
        {
            Environment.SetEnvironmentVariable(RootVariable, previous.Length == 0 ? null : previous);
        }
    }

    [Fact]
    public async Task Failed_save_removes_its_temp_file()
    {
        var paths = new TrackdubStoragePaths(root);
        Directory.CreateDirectory(paths.RootDirectory);
        string blocked = Path.Join(root, "blocked");
        File.WriteAllText(blocked, "not a directory");
        var service = new JsonStudioSettingsService(paths);
        StudioSettings settings = StudioSettings.Default with { DefaultModelCacheDirectory = blocked };

        await Assert.ThrowsAnyAsync<IOException>(() => service.SaveAsync(settings, TestContext.Current.CancellationToken));

        Assert.Empty(Directory.GetFiles(paths.RootDirectory, "settings.json.*.tmp"));
    }

    public void Dispose()
    {
        if (Directory.Exists(root))
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
