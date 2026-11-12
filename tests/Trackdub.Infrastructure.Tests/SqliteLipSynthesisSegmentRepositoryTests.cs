using Microsoft.Data.Sqlite;
using Trackdub.Domain.LipSynthesis;
using Trackdub.Domain.Projects;
using Trackdub.Infrastructure.Persistence.Sqlite;

namespace Trackdub.Infrastructure.Tests;

public sealed class SqliteLipSynthesisSegmentRepositoryTests
{
    [Fact]
    public async Task MigrateAsync_CreatesLipSynthesisSegmentsTable()
    {
        string projectRoot = CreateProjectRoot("LipSynthesisMigration");
        try
        {
            var database = new SqliteProjectDatabase(projectRoot);
            await database.InitializeAsync(TestContext.Current.CancellationToken);

            await using SqliteConnection connection = await OpenConnectionAsync(database.DatabasePath);
            Assert.True(await TableExistsAsync(connection, "LipSynthesisSegments"));
        }
        finally
        {
            DeleteDirectory(projectRoot);
        }
    }

    [Fact]
    public async Task SaveAllAsync_ThenGetByProjectAsync_RoundTrips()
    {
        string projectRoot = CreateProjectRoot("LipSynthesisByProject");
        try
        {
            var database = new SqliteProjectDatabase(projectRoot);
            var repository = new SqliteLipSynthesisSegmentRepository(database);
            Guid projectId = Guid.NewGuid();
            Guid stageRunId = Guid.NewGuid();
            DateTimeOffset now = DateTimeOffset.UtcNow;

            await new SqliteProjectRepository(database).InitializeAsync(
                new TrackdubProject(projectId, "Test", now, now), TestContext.Current.CancellationToken);

            var synthesized = new LipSynthesisSegment(
                SegmentId: Guid.NewGuid(),
                Status: LipSynthesisSegmentStatus.Synthesized,
                SpeakerId: "spk-1",
                TurnStart: TimeSpan.FromSeconds(1.0),
                TurnEnd: TimeSpan.FromSeconds(3.0),
                FaceConfidence: 0.92,
                PatchedClipRelativePath: "artifacts/lip-synthesis/run/seg.mp4",
                SkipReason: null,
                FailureReason: null,
                ProviderId: "latentsync-onnx",
                ModelId: "ByteDance/LatentSync-1.6",
                UsedExperimentalProvider: true,
                CreatedAtUtc: now);

            var skipped = new LipSynthesisSegment(
                SegmentId: Guid.NewGuid(),
                Status: LipSynthesisSegmentStatus.SkippedNoFace,
                SpeakerId: null,
                TurnStart: TimeSpan.FromSeconds(4.0),
                TurnEnd: TimeSpan.FromSeconds(5.0),
                FaceConfidence: null,
                PatchedClipRelativePath: null,
                SkipReason: "No usable face detected in the turn.",
                FailureReason: null,
                ProviderId: null,
                ModelId: null,
                UsedExperimentalProvider: false,
                CreatedAtUtc: now);

            await repository.SaveAllAsync(projectId, stageRunId, [synthesized, skipped], TestContext.Current.CancellationToken);

            IReadOnlyList<LipSynthesisSegment> retrieved =
                await repository.GetByProjectAsync(projectId, TestContext.Current.CancellationToken);

            Assert.Equal(2, retrieved.Count);
            LipSynthesisSegment reloaded1 = Assert.Single(retrieved, s => s.SegmentId == synthesized.SegmentId);
            Assert.Equal(LipSynthesisSegmentStatus.Synthesized, reloaded1.Status);
            Assert.Equal("spk-1", reloaded1.SpeakerId);
            Assert.Equal(0.92, reloaded1.FaceConfidence);
            Assert.Equal("artifacts/lip-synthesis/run/seg.mp4", reloaded1.PatchedClipRelativePath);
            Assert.True(reloaded1.UsedExperimentalProvider);
            Assert.Equal(TimeSpan.FromSeconds(1.0), reloaded1.TurnStart);
            Assert.Equal(TimeSpan.FromSeconds(3.0), reloaded1.TurnEnd);

            LipSynthesisSegment reloaded2 = Assert.Single(retrieved, s => s.SegmentId == skipped.SegmentId);
            Assert.Equal(LipSynthesisSegmentStatus.SkippedNoFace, reloaded2.Status);
            Assert.Equal("No usable face detected in the turn.", reloaded2.SkipReason);
            Assert.Null(reloaded2.PatchedClipRelativePath);
            Assert.Null(reloaded2.FaceConfidence);
            Assert.False(reloaded2.UsedExperimentalProvider);
        }
        finally
        {
            DeleteDirectory(projectRoot);
        }
    }

    [Fact]
    public async Task SaveAllAsync_ThenGetByStageRunAsync_RoundTrips()
    {
        string projectRoot = CreateProjectRoot("LipSynthesisByStageRun");
        try
        {
            var database = new SqliteProjectDatabase(projectRoot);
            var repository = new SqliteLipSynthesisSegmentRepository(database);
            Guid projectId = Guid.NewGuid();
            Guid stageRunId = Guid.NewGuid();
            Guid otherStageRunId = Guid.NewGuid();
            DateTimeOffset now = DateTimeOffset.UtcNow;

            await new SqliteProjectRepository(database).InitializeAsync(
                new TrackdubProject(projectId, "Test", now, now), TestContext.Current.CancellationToken);

            var s1 = MakeSegment(LipSynthesisSegmentStatus.Synthesized, now);
            var s2 = MakeSegment(LipSynthesisSegmentStatus.SkippedNonFrontal, now);
            var otherRun = MakeSegment(LipSynthesisSegmentStatus.Failed, now);

            await repository.SaveAllAsync(projectId, stageRunId, [s1, s2], TestContext.Current.CancellationToken);
            await repository.SaveAllAsync(projectId, otherStageRunId, [otherRun], TestContext.Current.CancellationToken);

            IReadOnlyList<LipSynthesisSegment> retrieved =
                await repository.GetByStageRunAsync(stageRunId, TestContext.Current.CancellationToken);

            Assert.Equal(2, retrieved.Count);
            Assert.Contains(retrieved, s => s.SegmentId == s1.SegmentId);
            Assert.Contains(retrieved, s => s.SegmentId == s2.SegmentId);
            Assert.DoesNotContain(retrieved, s => s.SegmentId == otherRun.SegmentId);
        }
        finally
        {
            DeleteDirectory(projectRoot);
        }
    }

    [Fact]
    public async Task SaveAllAsync_WithSameSegmentAcrossStageRuns_PreservesBothRuns()
    {
        string projectRoot = CreateProjectRoot("LipSynthesisStageRunHistory");
        try
        {
            var database = new SqliteProjectDatabase(projectRoot);
            var repository = new SqliteLipSynthesisSegmentRepository(database);
            Guid projectId = Guid.NewGuid();
            Guid firstStageRunId = Guid.NewGuid();
            Guid secondStageRunId = Guid.NewGuid();
            Guid segmentId = Guid.NewGuid();
            DateTimeOffset now = DateTimeOffset.UtcNow;

            await new SqliteProjectRepository(database).InitializeAsync(
                new TrackdubProject(projectId, "Test", now, now), TestContext.Current.CancellationToken);

            var firstRun = new LipSynthesisSegment(
                segmentId,
                LipSynthesisSegmentStatus.Synthesized,
                "spk-1",
                TimeSpan.FromSeconds(1.0),
                TimeSpan.FromSeconds(2.0),
                0.9,
                "artifacts/lip-synthesis/run-1/seg.mp4",
                null,
                null,
                "prov",
                "mod",
                false,
                now);

            var secondRun = firstRun with
            {
                Status = LipSynthesisSegmentStatus.SkippedNoFace,
                PatchedClipRelativePath = null,
                SkipReason = "No face on rerun.",
                CreatedAtUtc = now.AddMinutes(1)
            };

            await repository.SaveAllAsync(projectId, firstStageRunId, [firstRun], TestContext.Current.CancellationToken);
            await repository.SaveAllAsync(projectId, secondStageRunId, [secondRun], TestContext.Current.CancellationToken);

            IReadOnlyList<LipSynthesisSegment> firstRunRows =
                await repository.GetByStageRunAsync(firstStageRunId, TestContext.Current.CancellationToken);
            LipSynthesisSegment reloadedFirst = Assert.Single(firstRunRows);
            Assert.Equal(LipSynthesisSegmentStatus.Synthesized, reloadedFirst.Status);
            Assert.Equal("artifacts/lip-synthesis/run-1/seg.mp4", reloadedFirst.PatchedClipRelativePath);

            IReadOnlyList<LipSynthesisSegment> secondRunRows =
                await repository.GetByStageRunAsync(secondStageRunId, TestContext.Current.CancellationToken);
            LipSynthesisSegment reloadedSecond = Assert.Single(secondRunRows);
            Assert.Equal(LipSynthesisSegmentStatus.SkippedNoFace, reloadedSecond.Status);
            Assert.Equal("No face on rerun.", reloadedSecond.SkipReason);

            Assert.Equal(2, (await repository.GetByProjectAsync(projectId, TestContext.Current.CancellationToken)).Count);
        }
        finally
        {
            DeleteDirectory(projectRoot);
        }
    }

    [Fact]
    public async Task SaveAllAsync_WithDuplicateId_Replaces()
    {
        string projectRoot = CreateProjectRoot("LipSynthesisDuplicate");
        try
        {
            var database = new SqliteProjectDatabase(projectRoot);
            var repository = new SqliteLipSynthesisSegmentRepository(database);
            Guid projectId = Guid.NewGuid();
            Guid stageRunId = Guid.NewGuid();
            DateTimeOffset now = DateTimeOffset.UtcNow;

            await new SqliteProjectRepository(database).InitializeAsync(
                new TrackdubProject(projectId, "Test", now, now), TestContext.Current.CancellationToken);

            var original = MakeSegment(LipSynthesisSegmentStatus.Synthesized, now) with
            {
                PatchedClipRelativePath = "artifacts/lip-synthesis/run/orig.mp4",
                UsedExperimentalProvider = true
            };

            var replacement = original with
            {
                Status = LipSynthesisSegmentStatus.Failed,
                PatchedClipRelativePath = null,
                FailureReason = "Re-run failed.",
                UsedExperimentalProvider = false
            };

            await repository.SaveAllAsync(projectId, stageRunId, [original], TestContext.Current.CancellationToken);
            await repository.SaveAllAsync(projectId, stageRunId, [replacement], TestContext.Current.CancellationToken);

            LipSynthesisSegment reloaded = Assert.Single(
                await repository.GetByProjectAsync(projectId, TestContext.Current.CancellationToken));
            Assert.Equal(original.SegmentId, reloaded.SegmentId);
            Assert.Equal(LipSynthesisSegmentStatus.Failed, reloaded.Status);
            Assert.Equal("Re-run failed.", reloaded.FailureReason);
            Assert.Null(reloaded.PatchedClipRelativePath);
            Assert.False(reloaded.UsedExperimentalProvider);
        }
        finally
        {
            DeleteDirectory(projectRoot);
        }
    }

    private static LipSynthesisSegment MakeSegment(LipSynthesisSegmentStatus status, DateTimeOffset now) =>
        new(
            SegmentId: Guid.NewGuid(),
            Status: status,
            SpeakerId: "spk",
            TurnStart: TimeSpan.FromSeconds(1.0),
            TurnEnd: TimeSpan.FromSeconds(2.0),
            FaceConfidence: status == LipSynthesisSegmentStatus.Synthesized ? 0.9 : null,
            PatchedClipRelativePath: null,
            SkipReason: null,
            FailureReason: status == LipSynthesisSegmentStatus.Failed ? "fail" : null,
            ProviderId: "prov",
            ModelId: "mod",
            UsedExperimentalProvider: false,
            CreatedAtUtc: now);

    private static string CreateProjectRoot(string projectName) =>
        Path.Join(
            Path.GetTempPath(),
            "Trackdub.Infrastructure.Tests",
            Guid.NewGuid().ToString("N"),
            $"{projectName}.trackdub");

    private static async Task<SqliteConnection> OpenConnectionAsync(string databasePath)
    {
        var builder = new SqliteConnectionStringBuilder
        {
            DataSource = databasePath,
            ForeignKeys = true,
            Pooling = false
        };
        var connection = new SqliteConnection(builder.ConnectionString);
        await connection.OpenAsync();
        return connection;
    }

    private static async Task<bool> TableExistsAsync(SqliteConnection connection, string tableName)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
              AND name = $name
            LIMIT 1;
            """;
        command.Parameters.AddWithValue("$name", tableName);
        return await command.ExecuteScalarAsync() is not null;
    }

    private static void DeleteDirectory(string path)
    {
        if (Directory.Exists(path))
        {
            Directory.Delete(path, recursive: true);
        }
    }
}
