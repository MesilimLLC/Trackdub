using Trackdub.Domain;
using Trackdub.Infrastructure.Persistence.Repositories;
using Trackdub.Infrastructure.Settings;

namespace Trackdub.Infrastructure.Tests;

public sealed class LocalModelCacheRecordStoreMigrationTests : IDisposable
{
    private readonly string root = Path.Join(
        Path.GetTempPath(),
        "Trackdub.LocalModelCacheRecordStoreMigration.Tests",
        Guid.NewGuid().ToString("N"));

    [Fact]
    public async Task Copy_records_to_new_directory_merges_indexes_and_keeps_source_intact()
    {
        TrackdubStoragePaths sourcePaths = CreateStoragePaths(Path.Join(root, "old-cache"));
        TrackdubStoragePaths destinationPaths = CreateStoragePaths(Path.Join(root, "new-cache"));
        var sourceStore = new LocalModelCacheRecordStore(sourcePaths);
        var destinationStore = new LocalModelCacheRecordStore(destinationPaths);
        var existingRecord = new LocalModelCacheRecord(
            "test/source-model",
            Path.Join(root, "old-cache", "test", "source-model"),
            "source-revision",
            new string('a', 64),
            DateTimeOffset.UnixEpoch);
        var destinationRecord = new LocalModelCacheRecord(
            "test/destination-model",
            Path.Join(root, "new-cache", "test", "destination-model"),
            "destination-revision",
            new string('b', 64),
            DateTimeOffset.UnixEpoch.AddDays(1));
        await sourceStore.MutateAsync(_ => [existingRecord], TestContext.Current.CancellationToken);
        await destinationStore.MutateAsync(_ => [destinationRecord], TestContext.Current.CancellationToken);

        await sourceStore.CopyRecordsToDirectoryAsync(
            new DirectoryInfo(destinationPaths.ModelCacheDirectory),
            TestContext.Current.CancellationToken);

        IReadOnlyList<LocalModelCacheRecord> copiedRecords = await destinationStore.LoadAsync(TestContext.Current.CancellationToken);
        Assert.Equal(2, copiedRecords.Count);
        Assert.Contains(copiedRecords, record => record.ModelId == existingRecord.ModelId && record.RootPath == existingRecord.RootPath);
        Assert.Contains(copiedRecords, record => record.ModelId == destinationRecord.ModelId && record.RootPath == destinationRecord.RootPath);
        Assert.True(File.Exists(sourcePaths.ModelCacheIndexPath));
        Assert.False(Directory.Exists(existingRecord.RootPath));

        var downloadedAfterChange = new LocalModelCacheRecord(
            "test/next-download",
            Path.Join(root, "old-cache", "test", "next-download"),
            "new-revision",
            new string('c', 64),
            DateTimeOffset.UnixEpoch.AddDays(2));
        await sourceStore.MutateAsync(
            records => [.. records, downloadedAfterChange],
            TestContext.Current.CancellationToken);

        IReadOnlyList<LocalModelCacheRecord> updatedDestination = await destinationStore.LoadAsync(TestContext.Current.CancellationToken);
        Assert.Contains(updatedDestination, record => record.ModelId == downloadedAfterChange.ModelId);
    }

    [Fact]
    public async Task Copy_empty_index_to_new_directory_mirrors_records_added_before_restart()
    {
        TrackdubStoragePaths sourcePaths = CreateStoragePaths(Path.Join(root, "empty-old-cache"));
        TrackdubStoragePaths destinationPaths = CreateStoragePaths(Path.Join(root, "empty-new-cache"));
        var sourceStore = new LocalModelCacheRecordStore(sourcePaths);
        var destinationStore = new LocalModelCacheRecordStore(destinationPaths);

        await sourceStore.CopyRecordsToDirectoryAsync(
            new DirectoryInfo(destinationPaths.ModelCacheDirectory),
            TestContext.Current.CancellationToken);

        var addedBeforeRestart = new LocalModelCacheRecord(
            "test/downloaded-before-restart",
            Path.Join(root, "empty-old-cache", "test", "downloaded-before-restart"),
            "revision",
            new string('d', 64),
            DateTimeOffset.UnixEpoch);
        await sourceStore.MutateAsync(
            _ => [addedBeforeRestart],
            TestContext.Current.CancellationToken);

        IReadOnlyList<LocalModelCacheRecord> destinationRecords = await destinationStore.LoadAsync(
            TestContext.Current.CancellationToken);
        Assert.Contains(destinationRecords, record => record.ModelId == addedBeforeRestart.ModelId);
    }

    [Fact]
    public async Task Copy_waits_for_destination_mutation_and_preserves_both_records()
    {
        var source = new LocalModelCacheRecordStore(CreateStoragePaths(Path.Join(root, "source")));
        var destination = new LocalModelCacheRecordStore(CreateStoragePaths(Path.Join(root, "destination")));
        var sourceRecord = Record("source");
        var destinationRecord = Record("destination");
        await source.MutateAsync(_ => [sourceRecord], TestContext.Current.CancellationToken);

        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        Task mutation = Task.Run(() => destination.MutateAsync(records =>
        {
            entered.Set();
            release.Wait(TestContext.Current.CancellationToken);
            return [.. records, destinationRecord];
        }, TestContext.Current.CancellationToken));
        try
        {
            Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));
            Task copy = source.CopyRecordsToDirectoryAsync(new DirectoryInfo(Path.Join(root, "destination")), TestContext.Current.CancellationToken);
            Assert.False(copy.IsCompleted);
            release.Set();
            await Task.WhenAll(mutation, copy);
        }
        finally
        {
            release.Set();
            await mutation;
        }

        Assert.Equal(2, (await destination.LoadAsync(TestContext.Current.CancellationToken)).Count);

        var laterRecord = Record("later");
        await destination.MutateAsync(records => [.. records, laterRecord], TestContext.Current.CancellationToken);
        await source.MutateAsync(records => records.Where(record => record.ModelId != sourceRecord.ModelId).ToArray(), TestContext.Current.CancellationToken);
        IReadOnlyList<LocalModelCacheRecord> result = await destination.LoadAsync(TestContext.Current.CancellationToken);
        Assert.DoesNotContain(result, record => record.ModelId == sourceRecord.ModelId);
        Assert.Contains(result, record => record.ModelId == destinationRecord.ModelId);
        Assert.Contains(result, record => record.ModelId == laterRecord.ModelId);
    }

    [Fact]
    public async Task Destination_deletion_survives_later_source_mutation()
    {
        var source = new LocalModelCacheRecordStore(CreateStoragePaths(Path.Join(root, "source-delete")));
        var destination = new LocalModelCacheRecordStore(CreateStoragePaths(Path.Join(root, "destination-delete")));
        var removed = Record("removed");
        await source.MutateAsync(_ => [removed], TestContext.Current.CancellationToken);
        await source.CopyRecordsToDirectoryAsync(new DirectoryInfo(Path.Join(root, "destination-delete")), TestContext.Current.CancellationToken);
        await destination.MutateAsync(_ => [], TestContext.Current.CancellationToken);
        var added = Record("added");
        await source.MutateAsync(records => [.. records, added], TestContext.Current.CancellationToken);
        Assert.Equal("added", Assert.Single(await destination.LoadAsync(TestContext.Current.CancellationToken)).ModelId);
        Assert.Equal("added", Assert.Single(await source.LoadAsync(TestContext.Current.CancellationToken)).ModelId);
    }

    [Fact]
    public async Task Waiting_for_index_lock_observes_cancellation()
    {
        TrackdubStoragePaths paths = CreateStoragePaths(Path.Join(root, "locked"));
        Directory.CreateDirectory(paths.ModelCacheDirectory);
        using var fileLock = new FileStream(paths.ModelCacheIndexPath + ".lock", FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        using var cancellation = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
        var store = new LocalModelCacheRecordStore(paths);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => store.MutateAsync(_ => [Record("locked")], cancellation.Token));
    }

    private LocalModelCacheRecord Record(string name) => new(
        name, Path.Join(root, name), "revision", new string('a', 64), DateTimeOffset.UnixEpoch);

    private TrackdubStoragePaths CreateStoragePaths(string modelCacheDirectory) => new(new TrackdubStorageOptions(
        Path.Join(root, "user-data"),
        Path.Join(root, "user-cache"),
        SharedAssetRoot: null,
        IsPortable: false,
        ExplicitModelCacheDirectory: modelCacheDirectory));

    public void Dispose()
    {
        if (Directory.Exists(root))
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
