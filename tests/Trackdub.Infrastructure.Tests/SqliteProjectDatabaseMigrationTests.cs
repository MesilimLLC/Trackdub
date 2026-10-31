using Microsoft.Data.Sqlite;
using Trackdub.Infrastructure.Persistence.Sqlite;

namespace Trackdub.Infrastructure.Tests;

public sealed class SqliteProjectDatabaseMigrationTests
{
    [Fact]
    public async Task InitializeAsync_records_project_schema_versions_in_order()
    {
        string projectRoot = CreateProjectRoot("VersionedSchema");
        try
        {
            var database = new SqliteProjectDatabase(projectRoot);

            await database.InitializeAsync(TestContext.Current.CancellationToken);
            await database.InitializeAsync(TestContext.Current.CancellationToken);

            await using SqliteConnection connection = await OpenConnectionAsync(database.DatabasePath);
            (int Version, string Name)[] appliedVersions = (await ReadProjectSchemaVersionsAsync(connection)).ToArray();
            (int Version, string Name)[] declaredVersions = SqliteProjectSchemaMigrations.All
                .Select(migration => (migration.Version, migration.Name))
                .ToArray();

            for (int i = 1; i < declaredVersions.Length; i++)
            {
                Assert.True(
                    declaredVersions[i].Version > declaredVersions[i - 1].Version,
                    $"Project migration versions must be strictly increasing: version {declaredVersions[i].Version} follows {declaredVersions[i - 1].Version}.");
            }

            Assert.Equal(declaredVersions.Length, declaredVersions.Select(version => version.Version).Distinct().Count());
            Assert.Contains((20, "create-glossary-entries"), declaredVersions);
            Assert.Equal(declaredVersions, appliedVersions);
        }
        finally
        {
            DeleteDirectory(projectRoot);
        }
    }

    [Fact]
    public async Task Translation_origin_migration_marks_existing_rows_unknown()
    {
        string projectRoot = CreateProjectRoot("LegacyTranslationOrigin");
        var database = new SqliteProjectDatabase(projectRoot);
        try
        {
            await database.InitializeAsync(TestContext.Current.CancellationToken);
            await using (SqliteConnection connection = await OpenConnectionAsync(database.DatabasePath))
            {
                await ExecuteNonQueryAsync(connection, """
                    PRAGMA foreign_keys = OFF;
                    ALTER TABLE translated_segments DROP COLUMN origin;
                    DELETE FROM project_schema_versions WHERE version = 31;
                    INSERT INTO translated_segments
                        (id, translation_revision_id, segment_index, start_seconds, end_seconds, text, source_segment_hash)
                    VALUES ('legacy', 'revision', 0, 0, 1, 'edited text', 'hash');
                    """);
            }

            var reopened = new SqliteProjectDatabase(projectRoot);
            await reopened.InitializeAsync(TestContext.Current.CancellationToken);
            await using SqliteConnection check = await OpenConnectionAsync(reopened.DatabasePath);
            await using SqliteCommand command = check.CreateCommand();
            command.CommandText = "SELECT origin FROM translated_segments WHERE id = 'legacy';";
            Assert.Equal(0L, await command.ExecuteScalarAsync(TestContext.Current.CancellationToken));
        }
        finally
        {
            DeleteDirectory(projectRoot);
        }
    }

    [Fact]
    public async Task InitializeAsync_throws_ProjectDatabaseSchemaVersionException_when_schema_is_newer_than_build()
    {
        string projectRoot = CreateProjectRoot("FutureSchema");
        var database = new SqliteProjectDatabase(projectRoot);
        Directory.CreateDirectory(Path.GetDirectoryName(database.DatabasePath)!);

        try
        {
            // Seed the version table with a schema version far in the future.
            await using (SqliteConnection connection = await OpenConnectionAsync(database.DatabasePath))
            {
                await ExecuteNonQueryAsync(
                    connection,
                    """
                    CREATE TABLE project_schema_versions (
                        version INTEGER NOT NULL PRIMARY KEY,
                        name TEXT NOT NULL,
                        applied_at_utc TEXT NOT NULL
                    );
                    INSERT INTO project_schema_versions (version, name, applied_at_utc)
                    VALUES (9999, 'future-migration', '2099-01-01T00:00:00+00:00');
                    """);
            }

            ProjectDatabaseSchemaVersionException ex = await Assert.ThrowsAsync<ProjectDatabaseSchemaVersionException>(
                () => database.InitializeAsync(TestContext.Current.CancellationToken));

            Assert.Equal(9999, ex.SchemaVersion);
            Assert.Equal(SqliteProjectSchemaMigrations.All.Max(m => m.Version), ex.MaxSupportedVersion);
        }
        finally
        {
            DeleteDirectory(projectRoot);
        }
    }

    [Fact]
    public async Task InitializeAsync_creates_backup_before_applying_pending_migrations()
    {
        string projectRoot = CreateProjectRoot("BackupOnMigration");
        var database = new SqliteProjectDatabase(projectRoot);

        try
        {
            // First initialization: build a fully-migrated database.
            await database.InitializeAsync(TestContext.Current.CancellationToken);

            // Remove a known idempotent migration version to simulate pending migrations safely.
            int lastVersion = SqliteProjectSchemaMigrations.KnownIdempotentMigrationVersion;
            Assert.Contains(SqliteProjectSchemaMigrations.All, m => m.Version == lastVersion);
            await using (SqliteConnection connection = await OpenConnectionAsync(database.DatabasePath))
            {
                await ExecuteNonQueryAsync(
                    connection,
                    $"DELETE FROM project_schema_versions WHERE version = {SqliteProjectSchemaMigrations.KnownIdempotentMigrationVersion};");
            }

            // Second initialization should detect the pending migration and create a backup.
            await database.InitializeAsync(TestContext.Current.CancellationToken);

            string dbDir = Path.GetDirectoryName(database.DatabasePath)!;
            string dbFileName = Path.GetFileName(database.DatabasePath);
            string[] backupFiles = Directory.GetFiles(dbDir, $"{dbFileName}.*.bak");

            Assert.Single(backupFiles);

            // Running again when already up-to-date must NOT create another backup.
            await database.InitializeAsync(TestContext.Current.CancellationToken);
            backupFiles = Directory.GetFiles(dbDir, $"{dbFileName}.*.bak");
            Assert.Single(backupFiles);
        }
        finally
        {
            DeleteDirectory(projectRoot);
        }
    }

    [Fact]
    public async Task InitializeAsync_throws_ProjectDatabaseCorruptedException_for_corrupted_database()
    {
        string projectRoot = CreateProjectRoot("CorruptedDb");
        var database = new SqliteProjectDatabase(projectRoot);
        Directory.CreateDirectory(Path.GetDirectoryName(database.DatabasePath)!);

        try
        {
            // Write garbage bytes to the database file to simulate corruption.
            await File.WriteAllBytesAsync(database.DatabasePath, [0x53, 0x51, 0x4C, 0x69, 0x74, 0x65, 0xFF, 0xFF], TestContext.Current.CancellationToken);

            ProjectDatabaseCorruptedException ex = await Assert.ThrowsAsync<ProjectDatabaseCorruptedException>(
                () => database.InitializeAsync(TestContext.Current.CancellationToken));

            Assert.Equal(database.DatabasePath, ex.DatabasePath);
            Assert.False(string.IsNullOrEmpty(ex.IntegrityCheckResult));
            Assert.NotEqual("ok", ex.IntegrityCheckResult, StringComparer.OrdinalIgnoreCase);
            // Original SqliteException must be preserved as the inner exception for diagnostics.
            Assert.IsType<Microsoft.Data.Sqlite.SqliteException>(ex.InnerException);
        }
        finally
        {
            DeleteDirectory(projectRoot);
        }
    }

    [Fact]
    public async Task InitializeAsync_does_not_create_backup_for_fresh_database()
    {
        string projectRoot = CreateProjectRoot("FreshDb");
        var database = new SqliteProjectDatabase(projectRoot);

        try
        {
            // A brand-new database (no file yet) should not have a backup created.
            await database.InitializeAsync(TestContext.Current.CancellationToken);

            string dbDir = Path.GetDirectoryName(database.DatabasePath)!;
            string dbFileName = Path.GetFileName(database.DatabasePath);
            string[] backupFiles = Directory.GetFiles(dbDir, $"{dbFileName}.*.bak");

            Assert.Empty(backupFiles);
        }
        finally
        {
            DeleteDirectory(projectRoot);
        }
    }

    [Fact]
    public async Task InitializeAsync_upgrades_legacy_project_database_missing_schema_columns()
    {
        string projectRoot = CreateProjectRoot("LegacySchema");
        var database = new SqliteProjectDatabase(projectRoot);
        Directory.CreateDirectory(Path.GetDirectoryName(database.DatabasePath)!);

        try
        {
            await using (SqliteConnection connection = await OpenConnectionAsync(database.DatabasePath))
            {
                await ExecuteNonQueryAsync(
                    connection,
                    """
                    CREATE TABLE projects (
                        id TEXT NOT NULL PRIMARY KEY,
                        name TEXT NOT NULL,
                        created_at_utc TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL
                    );

                    CREATE TABLE media_assets (
                        id TEXT NOT NULL PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        source_file_name TEXT NOT NULL,
                        fingerprint_sha256 TEXT NOT NULL,
                        source_size_bytes INTEGER NOT NULL,
                        source_last_write_time_utc TEXT NOT NULL,
                        format_name TEXT NOT NULL,
                        duration_seconds REAL NOT NULL,
                        has_audio INTEGER NOT NULL,
                        has_video INTEGER NOT NULL,
                        created_at_utc TEXT NOT NULL
                    );

                    CREATE TABLE artifacts (
                        id TEXT NOT NULL PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        media_asset_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        sha256 TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        duration_seconds REAL NULL,
                        sample_rate INTEGER NULL,
                        channel_count INTEGER NULL,
                        created_at_utc TEXT NOT NULL
                    );

                    CREATE TABLE StageRuns (
                        Id TEXT NOT NULL PRIMARY KEY,
                        ProjectId TEXT NOT NULL,
                        StageName TEXT NOT NULL,
                        Status TEXT NOT NULL,
                        StartedAtUtc TEXT NOT NULL,
                        CompletedAtUtc TEXT NULL,
                        FailureReason TEXT NULL
                    );

                    CREATE TABLE transcript_revisions (
                        id TEXT NOT NULL PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        stage_run_id TEXT NULL,
                        revision_number INTEGER NOT NULL,
                        created_at_utc TEXT NOT NULL
                    );

                    CREATE TABLE transcript_segments (
                        id TEXT NOT NULL PRIMARY KEY,
                        transcript_revision_id TEXT NOT NULL,
                        segment_index INTEGER NOT NULL,
                        start_seconds REAL NOT NULL,
                        end_seconds REAL NOT NULL,
                        text TEXT NOT NULL
                    );

                    CREATE TABLE words (
                        id TEXT NOT NULL PRIMARY KEY,
                        transcript_segment_id TEXT NOT NULL,
                        word_index INTEGER NOT NULL,
                        start_seconds REAL NOT NULL,
                        end_seconds REAL NOT NULL,
                        text TEXT NOT NULL
                    );

                    CREATE TABLE speakers (
                        id TEXT NOT NULL PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        display_name TEXT NOT NULL,
                        created_at_utc TEXT NOT NULL
                    );

                    CREATE TABLE speaker_turns (
                        id TEXT NOT NULL PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        speaker_id TEXT NOT NULL,
                        stage_run_id TEXT NULL,
                        start_seconds REAL NOT NULL,
                        end_seconds REAL NOT NULL
                    );

                    CREATE TABLE translation_revisions (
                        id TEXT NOT NULL PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        stage_run_id TEXT NULL,
                        source_transcript_revision_id TEXT NOT NULL,
                        target_language TEXT NOT NULL,
                        revision_number INTEGER NOT NULL,
                        created_at_utc TEXT NOT NULL
                    );

                    CREATE TABLE translated_segments (
                        id TEXT NOT NULL PRIMARY KEY,
                        translation_revision_id TEXT NOT NULL,
                        segment_index INTEGER NOT NULL,
                        start_seconds REAL NOT NULL,
                        end_seconds REAL NOT NULL,
                        text TEXT NOT NULL
                    );

                    CREATE TABLE voice_assignments (
                        id TEXT NOT NULL PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        speaker_id TEXT NOT NULL,
                        voice_model_id TEXT NOT NULL,
                        voice_variant TEXT NULL,
                        requires_consent INTEGER NOT NULL,
                        created_at_utc TEXT NOT NULL
                    );

                    CREATE UNIQUE INDEX ix_voice_assignments_project_speaker
                        ON voice_assignments (project_id, speaker_id);

                    CREATE TABLE tts_takes (
                        id TEXT NOT NULL PRIMARY KEY,
                        project_id TEXT NOT NULL,
                        voice_assignment_id TEXT NOT NULL,
                        translated_segment_id TEXT NULL,
                        artifact_id TEXT NULL,
                        stage_run_id TEXT NULL,
                        status TEXT NOT NULL,
                        is_stale INTEGER NOT NULL DEFAULT 0,
                        duration_samples INTEGER NULL,
                        sample_rate INTEGER NULL,
                        provider TEXT NULL,
                        created_at_utc TEXT NOT NULL
                    );
                    """);
            }

            await database.InitializeAsync(TestContext.Current.CancellationToken);

            await using SqliteConnection verificationConnection = await OpenConnectionAsync(database.DatabasePath);
            Assert.Contains("source_file_path", await ReadColumnNamesAsync(verificationConnection, "media_assets"));
            Assert.Contains("stage_run_id", await ReadColumnNamesAsync(verificationConnection, "artifacts"));
            Assert.Contains("provenance", await ReadColumnNamesAsync(verificationConnection, "artifacts"));
            Assert.Contains("RequestedProvider", await ReadColumnNamesAsync(verificationConnection, "StageRuns"));
            Assert.Contains("BootstrapDetail", await ReadColumnNamesAsync(verificationConnection, "StageRuns"));
            Assert.Contains("translation_provider", await ReadColumnNamesAsync(verificationConnection, "translation_revisions"));
            Assert.Contains("execution_provider", await ReadColumnNamesAsync(verificationConnection, "translation_revisions"));
            Assert.Contains("source_segment_hash", await ReadColumnNamesAsync(verificationConnection, "translated_segments"));
            Assert.Contains("speaker_id", await ReadColumnNamesAsync(verificationConnection, "transcript_segments"));
            Assert.Contains("detected_language", await ReadColumnNamesAsync(verificationConnection, "transcript_segments"));
            Assert.Contains("speaker_assignment_origin", await ReadColumnNamesAsync(verificationConnection, "transcript_segments"));
            Assert.Contains("confidence", await ReadColumnNamesAsync(verificationConnection, "words"));
            Assert.Contains("has_overlap", await ReadColumnNamesAsync(verificationConnection, "speaker_turns"));
            Assert.Contains("is_fallback", await ReadColumnNamesAsync(verificationConnection, "voice_assignments"));
            Assert.Contains("segment_index", await ReadColumnNamesAsync(verificationConnection, "tts_takes"));
            Assert.Contains("stretch_engine", await ReadColumnNamesAsync(verificationConnection, "tts_takes"));
            Assert.Contains("candidate_group_id", await ReadColumnNamesAsync(verificationConnection, "tts_takes"));
            Assert.Contains("candidate_index", await ReadColumnNamesAsync(verificationConnection, "tts_takes"));
            Assert.Contains("candidate_variant", await ReadColumnNamesAsync(verificationConnection, "tts_takes"));
            Assert.Contains("selected_candidate_id", await ReadColumnNamesAsync(verificationConnection, "tts_candidate_groups"));
            Assert.True(await IndexExistsAsync(verificationConnection, "ix_voice_assignments_project_speaker_user"));
            Assert.True(await IndexExistsAsync(verificationConnection, "ix_tts_takes_candidate_group"));
            Assert.True(await IndexExistsAsync(verificationConnection, "ix_tts_candidate_groups_project_segment"));
            Assert.False(await IndexExistsAsync(verificationConnection, "ix_voice_assignments_project_speaker"));

            (int Version, string Name)[] appliedVersions = (await ReadProjectSchemaVersionsAsync(verificationConnection)).ToArray();
            Assert.Equal(SqliteProjectSchemaMigrations.All.Count, appliedVersions.Length);
        }
        finally
        {
            DeleteDirectory(projectRoot);
        }
    }

    private static string CreateProjectRoot(string projectName)
    {
        return Path.Join(
            Path.GetTempPath(),
            "Trackdub.Infrastructure.Tests",
            Guid.NewGuid().ToString("N"),
            $"{projectName}.trackdub");
    }

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

    private static async Task ExecuteNonQueryAsync(SqliteConnection connection, string sql)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = sql;
        await command.ExecuteNonQueryAsync();
    }

    private static async Task<IReadOnlyList<(int Version, string Name)>> ReadProjectSchemaVersionsAsync(SqliteConnection connection)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            SELECT version, name
            FROM project_schema_versions
            ORDER BY version;
            """;

        var versions = new List<(int Version, string Name)>();
        await using SqliteDataReader reader = await command.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            versions.Add((reader.GetInt32(0), reader.GetString(1)));
        }

        return versions;
    }

    private static async Task<IReadOnlyList<string>> ReadColumnNamesAsync(SqliteConnection connection, string tableName)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText = $"PRAGMA table_info([{tableName}]);";

        var columns = new List<string>();
        await using SqliteDataReader reader = await command.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            columns.Add(reader.GetString(1));
        }

        return columns;
    }

    private static async Task<bool> IndexExistsAsync(SqliteConnection connection, string indexName)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'index'
              AND name = $name
            LIMIT 1;
            """;
        command.Parameters.AddWithValue("$name", indexName);

        object? result = await command.ExecuteScalarAsync();
        return result is not null;
    }

    private static void DeleteDirectory(string path)
    {
        if (Directory.Exists(path))
        {
            Directory.Delete(path, recursive: true);
        }
    }
}
