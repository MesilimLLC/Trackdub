using System.Text.Json;
using Trackdub.Infrastructure.Settings;
using Trackdub.Infrastructure.Persistence.Repositories;
using Trackdub.Domain;

namespace Trackdub.Infrastructure.Tests;

public sealed class TrackdubStorageModelCacheSettingsTests
{
    [Fact]
    public void Resolve_uses_saved_model_cache_directory()
    {
        string root = CreateTestRoot();
        string localAppData = Path.Join(root, "local-app-data");
        string userData = Path.Join(localAppData, TrackdubStoragePathResolver.ProductDirectoryName);
        string configuredModelCache = Path.Join(root, "large-drive", "models");
        Directory.CreateDirectory(userData);
        File.WriteAllText(
            Path.Join(userData, "settings.json"),
            JsonSerializer.Serialize(new { defaultModelCacheDirectory = configuredModelCache }));

        var context = new TrackdubStoragePathResolutionContext(
            Path.Join(root, "app"),
            localAppData,
            Path.Join(root, "common-app-data"),
            new Dictionary<string, string?>());

        TrackdubStorageOptions options = TrackdubStoragePathResolver.Resolve(context);
        var storagePaths = new TrackdubStoragePaths(options);

        Assert.Equal(Path.GetFullPath(configuredModelCache), options.ExplicitModelCacheDirectory);
        Assert.Equal(Path.GetFullPath(configuredModelCache), storagePaths.ModelCacheDirectory);
        Assert.Equal(Path.Join(Path.GetFullPath(configuredModelCache), "model-cache-records.json"), storagePaths.ModelCacheIndexPath);
    }

    [Fact]
    public void Resolve_ignores_invalid_settings_json_and_uses_default_model_cache_directory()
    {
        string root = CreateTestRoot();
        string localAppData = Path.Join(root, "local-app-data");
        string userData = Path.Join(localAppData, TrackdubStoragePathResolver.ProductDirectoryName);
        Directory.CreateDirectory(userData);
        File.WriteAllText(Path.Join(userData, "settings.json"), "{ invalid json");

        var context = new TrackdubStoragePathResolutionContext(
            Path.Join(root, "app"),
            localAppData,
            Path.Join(root, "common-app-data"),
            new Dictionary<string, string?>());

        TrackdubStorageOptions options = TrackdubStoragePathResolver.Resolve(context);
        var storagePaths = new TrackdubStoragePaths(options);

        Assert.Null(options.ExplicitModelCacheDirectory);
        Assert.Equal(Path.Join(Path.GetFullPath(userData), "model-cache"), storagePaths.ModelCacheDirectory);
    }

    [Fact]
    public async Task Migrate_migrates_each_changed_directory_without_replaying_old_records()
    {
        string root = CreateTestRoot();
        string localAppData = Path.Join(root, "local");
        string userData = Path.Join(localAppData, TrackdubStoragePathResolver.ProductDirectoryName);
        string first = Path.Join(root, "first");
        string second = Path.Join(root, "second");
        Directory.CreateDirectory(userData);
        var context = new TrackdubStoragePathResolutionContext(
            Path.Join(root, "app"), localAppData, Path.Join(root, "common"), new Dictionary<string, string?>());
        var initial = new LocalModelCacheRecordStore(new TrackdubStoragePaths(ResolveAndMigrate(context)));
        var record = new LocalModelCacheRecord("model", Path.Join(root, "model"), "revision", new string('a', 64), DateTimeOffset.UnixEpoch);
        await initial.MutateAsync(_ => [record], TestContext.Current.CancellationToken);

        void SetDirectory(string? path) => File.WriteAllText(Path.Join(userData, "settings.json"),
            JsonSerializer.Serialize(new { defaultModelCacheDirectory = path }));

        SetDirectory(first);
        var firstStore = new LocalModelCacheRecordStore(new TrackdubStoragePaths(ResolveAndMigrate(context)));
        Assert.Single(await firstStore.LoadAsync(TestContext.Current.CancellationToken));
        await firstStore.MutateAsync(_ => [], TestContext.Current.CancellationToken);
        Assert.Empty(await new LocalModelCacheRecordStore(new TrackdubStoragePaths(ResolveAndMigrate(context)))
            .LoadAsync(TestContext.Current.CancellationToken));

        var nextRecord = record with { ModelId = "next" };
        await firstStore.MutateAsync(_ => [nextRecord], TestContext.Current.CancellationToken);
        SetDirectory(second);
        var secondStore = new LocalModelCacheRecordStore(new TrackdubStoragePaths(ResolveAndMigrate(context)));
        Assert.Equal("next", Assert.Single(await secondStore.LoadAsync(TestContext.Current.CancellationToken)).ModelId);

        SetDirectory(null);
        var defaultStore = new LocalModelCacheRecordStore(new TrackdubStoragePaths(ResolveAndMigrate(context)));
        Assert.Contains(await defaultStore.LoadAsync(TestContext.Current.CancellationToken), item => item.ModelId == "next");
    }

    [Fact]
    public async Task Migrate_migrates_when_cache_root_changes()
    {
        string root = CreateTestRoot();
        string userData = Path.Join(root, "data");
        var environment = new Dictionary<string, string?>
        {
            [TrackdubStoragePathResolver.DataRootEnvironmentVariable] = userData,
            [TrackdubStoragePathResolver.CacheRootEnvironmentVariable] = Path.Join(root, "old-cache")
        };
        var context = new TrackdubStoragePathResolutionContext(
            Path.Join(root, "app"), Path.Join(root, "local"), Path.Join(root, "common"), environment);
        var initial = new LocalModelCacheRecordStore(new TrackdubStoragePaths(ResolveAndMigrate(context)));
        await initial.MutateAsync(_ => [new LocalModelCacheRecord("model", root, "revision", new string('a', 64), DateTimeOffset.UnixEpoch)], TestContext.Current.CancellationToken);
        environment[TrackdubStoragePathResolver.CacheRootEnvironmentVariable] = Path.Join(root, "new-cache");
        var migrated = new LocalModelCacheRecordStore(new TrackdubStoragePaths(ResolveAndMigrate(context)));
        Assert.Single(await migrated.LoadAsync(TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task Migrate_does_not_advance_directory_when_copy_fails()
    {
        string root = CreateTestRoot();
        string localAppData = Path.Join(root, "local");
        string userData = Path.Join(localAppData, TrackdubStoragePathResolver.ProductDirectoryName);
        Directory.CreateDirectory(userData);
        var context = new TrackdubStoragePathResolutionContext(
            Path.Join(root, "app"), localAppData, Path.Join(root, "common"), new Dictionary<string, string?>());
        var initial = new LocalModelCacheRecordStore(new TrackdubStoragePaths(ResolveAndMigrate(context)));
        await initial.MutateAsync(_ => [new LocalModelCacheRecord("model", root, "revision", new string('a', 64), DateTimeOffset.UnixEpoch)], TestContext.Current.CancellationToken);
        string destination = Path.Join(root, "blocked");
        File.WriteAllText(destination, "blocked");
        File.WriteAllText(Path.Join(userData, "settings.json"), JsonSerializer.Serialize(new { defaultModelCacheDirectory = destination }));
        Assert.ThrowsAny<IOException>(() => ResolveAndMigrate(context));
        File.Delete(destination);
        var migrated = new LocalModelCacheRecordStore(new TrackdubStoragePaths(ResolveAndMigrate(context)));
        Assert.Single(await migrated.LoadAsync(TestContext.Current.CancellationToken));
    }

    [Theory]
    [InlineData("")]
    [InlineData("relative/cache")]
    public void Migrate_repairs_invalid_saved_directory(string saved)
    {
        string root = CreateTestRoot();
        string data = Path.Join(root, "data");
        Directory.CreateDirectory(data);
        string state = Path.Join(data, "model-cache-directory.txt");
        File.WriteAllText(state, saved);
        var options = new TrackdubStorageOptions(data, Path.Join(root, "cache"), null, false,
            ExplicitModelCacheDirectory: Path.Join(root, "configured"));

        TrackdubStoragePathResolver.MigrateModelCacheIfChanged(options);

        Assert.Equal(options.ExplicitModelCacheDirectory, File.ReadAllText(state));
    }

    [Fact]
    public void Resolve_does_not_create_state_or_locks()
    {
        string root = CreateTestRoot();
        string data = Path.Join(root, "data");
        var context = new TrackdubStoragePathResolutionContext(Path.Join(root, "app"), root, root,
            new Dictionary<string, string?> { [TrackdubStoragePathResolver.DataRootEnvironmentVariable] = data });

        TrackdubStorageOptions options = TrackdubStoragePathResolver.Resolve(context);

        Assert.Equal(data, options.UserDataRoot);
        Assert.False(Directory.Exists(data));
    }

    [Fact]
    public async Task Saving_settings_migrates_records_to_selected_directory()
    {
        string root = CreateTestRoot();
        var options = new TrackdubStorageOptions(Path.Join(root, "data"), Path.Join(root, "cache"), null, false);
        var paths = new TrackdubStoragePaths(options);
        TrackdubStoragePathResolver.MigrateModelCacheIfChanged(options);
        var source = new LocalModelCacheRecordStore(paths);
        await source.MutateAsync(_ => [new LocalModelCacheRecord("model", root, "revision", new string('a', 64), DateTimeOffset.UnixEpoch)], TestContext.Current.CancellationToken);
        string destination = Path.Join(root, "selected");
        var settings = new JsonStudioSettingsService(paths);

        await settings.SaveAsync(Trackdub.Contracts.StudioSettings.Default with { DefaultModelCacheDirectory = destination }, TestContext.Current.CancellationToken);

        var migrated = new LocalModelCacheRecordStore(new TrackdubStoragePaths(options with { ExplicitModelCacheDirectory = destination }));
        Assert.Single(await migrated.LoadAsync(TestContext.Current.CancellationToken));
        Assert.Equal(destination, File.ReadAllText(Path.Join(options.UserDataRoot, "model-cache-directory.txt")));
    }

    private static TrackdubStorageOptions ResolveAndMigrate(TrackdubStoragePathResolutionContext context)
    {
        TrackdubStorageOptions options = TrackdubStoragePathResolver.Resolve(context);
        TrackdubStoragePathResolver.MigrateModelCacheIfChanged(options);
        return options;
    }

    private static string CreateTestRoot()
    {
        string root = Path.Join(Path.GetTempPath(), "Trackdub.StorageModelCacheSettings.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        return root;
    }
}
