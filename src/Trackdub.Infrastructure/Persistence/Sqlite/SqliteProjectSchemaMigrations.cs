using System.Globalization;
using Microsoft.Data.Sqlite;

namespace Trackdub.Infrastructure.Persistence.Sqlite;

internal static class SqliteProjectSchemaMigrations
{
    // Keep this pinned to a migration that remains safe to re-apply in tests
    // (CREATE TABLE/INDEX IF NOT EXISTS, DROP INDEX IF EXISTS only). If this migration is ever
    // made non-idempotent, update this constant to point at another safe migration.
    internal const int KnownIdempotentMigrationVersion = 23;

    public static IReadOnlyList<SqliteProjectMigration> All { get; } =
    [
        new(1, "create-current-project-tables", CreateCurrentProjectTablesAsync),
        new(2, "add-voice-assignment-fallback-column", AddVoiceAssignmentFallbackColumnAsync),
        new(3, "rebuild-voice-assignment-indexes", RebuildVoiceAssignmentIndexesAsync),
        new(4, "add-media-asset-source-file-path", AddMediaAssetSourceFilePathAsync),
        new(5, "add-artifact-stage-run-id", AddArtifactStageRunIdAsync),
        new(6, "add-artifact-provenance", AddArtifactProvenanceAsync),
        new(7, "add-stage-run-runtime-context", AddStageRunRuntimeContextAsync),
        new(8, "add-translation-revision-runtime-context", AddTranslationRevisionRuntimeContextAsync),
        new(9, "add-translated-segment-source-hash", AddTranslatedSegmentSourceHashAsync),
        new(10, "add-transcript-segment-speaker-language", AddTranscriptSegmentSpeakerLanguageAsync),
        new(11, "ensure-words-table", EnsureWordsTableAsync),
        new(12, "add-word-confidence", AddWordConfidenceAsync),
        new(13, "add-speaker-turn-metadata", AddSpeakerTurnMetadataAsync),
        new(14, "add-tts-take-metadata", AddTtsTakeMetadataAsync),
        new(15, "create-current-project-indexes", CreateCurrentProjectIndexesAsync),
        new(16, "add-voice-cloning-tts-metadata", AddVoiceCloningTtsMetadataAsync),
        new(17, "add-artifact-degradation-columns", AddArtifactDegradationColumnsAsync),
        new(18, "add-tts-take-input-fingerprint", AddTtsTakeInputFingerprintAsync),
        new(19, "add-stage-run-extended-runtime-info", AddStageRunExtendedRuntimeInfoAsync),
        new(20, "create-glossary-entries", CreateGlossaryEntriesAsync),
        new(21, "create-voice-clone-consents", CreateVoiceCloneConsentsAsync),
        new(22, "ensure-translated-words-table", EnsureTranslatedWordsTableAsync),
        new(23, "rebuild-translated-words-unique-index", RebuildTranslatedWordsUniqueIndexAsync),
        new(24, "add-tts-candidate-metadata", AddTtsCandidateMetadataAsync),
        new(25, "create-lip-sync-segments", CreateLipSyncSegmentsAsync),
        new(26, "create-lip-synthesis-segments", CreateLipSynthesisSegmentsAsync),
        new(27, "rebuild-lip-synthesis-segments-stage-run-key", RebuildLipSynthesisSegmentsStageRunKeyAsync),
        new(28, "add-translation-model-provenance", AddTranslationModelProvenanceAsync),
        new(29, "add-speaker-assignment-origin", AddSpeakerAssignmentOriginAsync),
        new(30, "preserve-merged-speaker-history", PreserveMergedSpeakerHistoryAsync),
        new(31, "add-translation-segment-origin", AddTranslationSegmentOriginAsync)
    ];

    private static Task AddTranslationSegmentOriginAsync(
        SqliteConnection connection, SqliteTransaction transaction, CancellationToken cancellationToken) =>
        EnsureColumnAsync(connection, transaction, "translated_segments", "origin", "INTEGER NOT NULL DEFAULT 0", cancellationToken);

    private static Task PreserveMergedSpeakerHistoryAsync(
        SqliteConnection connection, SqliteTransaction transaction, CancellationToken cancellationToken) =>
        EnsureColumnAsync(connection, transaction, "speakers", "merged_into_speaker_id", "TEXT NULL", cancellationToken);

    public static async Task MigrateAsync(SqliteConnection connection, CancellationToken cancellationToken)
    {
        await EnsureVersionTableAsync(connection, cancellationToken).ConfigureAwait(false);

        HashSet<int> appliedVersions = await LoadAppliedVersionsAsync(connection, cancellationToken).ConfigureAwait(false);
        int newestKnownVersion = All.Max(migration => migration.Version);
        int newestAppliedVersion = appliedVersions.Count == 0 ? 0 : appliedVersions.Max();
        if (newestAppliedVersion > newestKnownVersion)
        {
            throw new ProjectDatabaseSchemaVersionException(newestAppliedVersion, newestKnownVersion);
        }

        foreach (SqliteProjectMigration migration in All.OrderBy(migration => migration.Version))
        {
            if (appliedVersions.Contains(migration.Version))
            {
                continue;
            }

            await using SqliteTransaction transaction = connection.BeginTransaction();
            try
            {
                await migration.ApplyAsync(connection, transaction, cancellationToken).ConfigureAwait(false);
                await RecordAppliedMigrationAsync(connection, transaction, migration, cancellationToken).ConfigureAwait(false);
                await transaction.CommitAsync(cancellationToken).ConfigureAwait(false);
            }
            catch
            {
                await transaction.RollbackAsync(cancellationToken).ConfigureAwait(false);
                throw;
            }
        }
    }

    /// <summary>
    /// Returns <see langword="true"/> when the database has schema migrations that have not yet been applied.
    /// This is a read-only check — it does not create or modify any tables.
    /// </summary>
    internal static async Task<bool> HasPendingMigrationsAsync(SqliteConnection connection, CancellationToken cancellationToken)
    {
        // If the version tracking table doesn't exist this is a legacy DB or a brand-new file; migrations are needed.
        await using SqliteCommand existsCommand = connection.CreateCommand();
        existsCommand.CommandText =
            """
            SELECT name FROM sqlite_master WHERE type='table' AND name='project_schema_versions';
            """;
        object? tableName = await existsCommand.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false);
        if (tableName is null)
        {
            return true;
        }

        // Check every known migration version, not just the maximum, so that gaps in the
        // applied set (e.g. a mid-range migration removed during testing or recovery) are
        // also detected as pending.
        HashSet<int> appliedVersions = await LoadAppliedVersionsAsync(connection, cancellationToken).ConfigureAwait(false);
        return All.Any(migration => !appliedVersions.Contains(migration.Version));
    }

    private static Task CreateCurrentProjectTablesAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return ExecuteNonQueryAsync(
            connection,
            transaction,
            static command => command.CommandText =
                """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT NOT NULL PRIMARY KEY,
                name TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS media_assets (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL,
                source_file_path TEXT NULL,
                source_file_name TEXT NOT NULL,
                fingerprint_sha256 TEXT NOT NULL,
                source_size_bytes INTEGER NOT NULL,
                source_last_write_time_utc TEXT NOT NULL,
                format_name TEXT NOT NULL,
                duration_seconds REAL NOT NULL,
                has_audio INTEGER NOT NULL,
                has_video INTEGER NOT NULL,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS artifacts (
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
                stage_run_id TEXT NULL,
                provenance TEXT NULL,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (media_asset_id) REFERENCES media_assets(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS StageRuns (
                Id TEXT NOT NULL PRIMARY KEY,
                ProjectId TEXT NOT NULL,
                StageName TEXT NOT NULL,
                Status TEXT NOT NULL,
                StartedAtUtc TEXT NOT NULL,
                CompletedAtUtc TEXT NULL,
                FailureReason TEXT NULL,
                FOREIGN KEY (ProjectId) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS transcript_revisions (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL,
                stage_run_id TEXT NULL,
                revision_number INTEGER NOT NULL,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (stage_run_id) REFERENCES StageRuns(Id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS transcript_segments (
                id TEXT NOT NULL PRIMARY KEY,
                transcript_revision_id TEXT NOT NULL,
                segment_index INTEGER NOT NULL,
                start_seconds REAL NOT NULL,
                end_seconds REAL NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY (transcript_revision_id) REFERENCES transcript_revisions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS speakers (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS speaker_turns (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL,
                speaker_id TEXT NOT NULL,
                stage_run_id TEXT NULL,
                start_seconds REAL NOT NULL,
                end_seconds REAL NOT NULL,
                confidence REAL NULL,
                has_overlap INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE CASCADE,
                FOREIGN KEY (stage_run_id) REFERENCES StageRuns(Id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS translation_revisions (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL,
                stage_run_id TEXT NULL,
                source_transcript_revision_id TEXT NOT NULL,
                target_language TEXT NOT NULL,
                translation_provider TEXT NULL,
                model_id TEXT NULL,
                execution_provider TEXT NULL,
                revision_number INTEGER NOT NULL,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (stage_run_id) REFERENCES StageRuns(Id) ON DELETE SET NULL,
                FOREIGN KEY (source_transcript_revision_id) REFERENCES transcript_revisions(id) ON DELETE RESTRICT
            );

            CREATE TABLE IF NOT EXISTS translated_segments (
                id TEXT NOT NULL PRIMARY KEY,
                translation_revision_id TEXT NOT NULL,
                segment_index INTEGER NOT NULL,
                start_seconds REAL NOT NULL,
                end_seconds REAL NOT NULL,
                text TEXT NOT NULL,
                source_segment_hash TEXT NULL,
                FOREIGN KEY (translation_revision_id) REFERENCES translation_revisions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS voice_assignments (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL,
                speaker_id TEXT NOT NULL,
                voice_model_id TEXT NOT NULL,
                voice_variant TEXT NULL,
                requires_consent INTEGER NOT NULL,
                is_fallback INTEGER NOT NULL DEFAULT 0,
                reference_clip_artifact_id TEXT NULL,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS tts_takes (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL,
                voice_assignment_id TEXT NOT NULL,
                translated_segment_id TEXT NULL,
                segment_index INTEGER NOT NULL DEFAULT 0,
                translated_text_hash TEXT NULL,
                artifact_id TEXT NULL,
                stage_run_id TEXT NULL,
                status TEXT NOT NULL,
                is_stale INTEGER NOT NULL DEFAULT 0,
                duration_samples INTEGER NULL,
                sample_rate INTEGER NULL,
                provider TEXT NULL,
                model_id TEXT NULL,
                voice_id TEXT NULL,
                kind TEXT NOT NULL DEFAULT 'Stock',
                reference_clip_artifact_id TEXT NULL,
                duration_overrun_ratio REAL NULL,
                pre_stretch_duration_seconds REAL NULL,
                stretch_ratio_applied REAL NULL,
                stretch_mode TEXT NOT NULL DEFAULT 'None',
                stretch_engine TEXT NOT NULL DEFAULT 'None',
                created_at_utc TEXT NOT NULL,
                candidate_group_id TEXT NULL,
                candidate_index INTEGER NULL,
                candidate_variant INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (voice_assignment_id) REFERENCES voice_assignments(id) ON DELETE CASCADE,
                FOREIGN KEY (translated_segment_id) REFERENCES translated_segments(id) ON DELETE SET NULL,
                FOREIGN KEY (artifact_id) REFERENCES artifacts(id) ON DELETE SET NULL,
                FOREIGN KEY (stage_run_id) REFERENCES StageRuns(Id) ON DELETE SET NULL
            );
            """,
            cancellationToken);
    }

    private static Task AddVoiceAssignmentFallbackColumnAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return EnsureColumnAsync(
            connection,
            transaction,
            "voice_assignments",
            "is_fallback",
            "INTEGER NOT NULL DEFAULT 0",
            cancellationToken);
    }

    private static Task RebuildVoiceAssignmentIndexesAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return ExecuteNonQueryAsync(
            connection,
            transaction,
            static command => command.CommandText =
                """
            DROP INDEX IF EXISTS ix_voice_assignments_project_speaker;
            CREATE UNIQUE INDEX IF NOT EXISTS ix_voice_assignments_project_speaker_user
                ON voice_assignments (project_id, speaker_id)
                WHERE is_fallback = 0;
            """,
            cancellationToken);
    }

    private static Task AddMediaAssetSourceFilePathAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return EnsureColumnAsync(
            connection,
            transaction,
            "media_assets",
            "source_file_path",
            "TEXT NULL",
            cancellationToken);
    }

    private static Task AddArtifactStageRunIdAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return EnsureColumnAsync(
            connection,
            transaction,
            "artifacts",
            "stage_run_id",
            "TEXT NULL",
            cancellationToken);
    }

    private static Task AddArtifactProvenanceAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return EnsureColumnAsync(
            connection,
            transaction,
            "artifacts",
            "provenance",
            "TEXT NULL",
            cancellationToken);
    }

    private static Task AddStageRunRuntimeContextAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return EnsureColumnsAsync(
            connection,
            transaction,
            "StageRuns",
            cancellationToken,
            new("RequestedProvider", "TEXT NULL"),
            new("SelectedProvider", "TEXT NULL"),
            new("RuntimeModelId", "TEXT NULL"),
            new("RuntimeModelAlias", "TEXT NULL"),
            new("RuntimeModelVariant", "TEXT NULL"),
            new("BootstrapDetail", "TEXT NULL"));
    }

    private static Task AddTranslationModelProvenanceAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken) =>
        EnsureColumnAsync(connection, transaction, "translation_revisions", "model_provenance_json", "TEXT NULL", cancellationToken);

    private static Task AddTranslationRevisionRuntimeContextAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return EnsureColumnsAsync(
            connection,
            transaction,
            "translation_revisions",
            cancellationToken,
            new("translation_provider", "TEXT NULL"),
            new("model_id", "TEXT NULL"),
            new("execution_provider", "TEXT NULL"));
    }

    private static Task AddTranslatedSegmentSourceHashAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return EnsureColumnAsync(
            connection,
            transaction,
            "translated_segments",
            "source_segment_hash",
            "TEXT NULL",
            cancellationToken);
    }

    private static Task AddTranscriptSegmentSpeakerLanguageAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return EnsureColumnsAsync(
            connection,
            transaction,
            "transcript_segments",
            cancellationToken,
            new("speaker_id", "TEXT NULL"),
            new("detected_language", "TEXT NULL"));
    }

    private static Task AddSpeakerAssignmentOriginAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken) =>
        EnsureColumnsAsync(
            connection,
            transaction,
            "transcript_segments",
            cancellationToken,
            new ColumnDefinition("speaker_assignment_origin", "INTEGER NOT NULL DEFAULT 0"));

    private static Task EnsureWordsTableAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return ExecuteNonQueryAsync(
            connection,
            transaction,
            static command => command.CommandText =
                """
            CREATE TABLE IF NOT EXISTS words (
                id TEXT NOT NULL PRIMARY KEY,
                transcript_segment_id TEXT NOT NULL,
                word_index INTEGER NOT NULL,
                start_seconds REAL NOT NULL,
                end_seconds REAL NOT NULL,
                text TEXT NOT NULL,
                confidence REAL NULL,
                FOREIGN KEY (transcript_segment_id) REFERENCES transcript_segments(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_words_transcript_segment_id
                ON words (transcript_segment_id, word_index);
            """,
            cancellationToken);
    }

    private static Task AddWordConfidenceAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return EnsureColumnAsync(
            connection,
            transaction,
            "words",
            "confidence",
            "REAL NULL",
            cancellationToken);
    }

    private static Task EnsureTranslatedWordsTableAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return ExecuteNonQueryAsync(
            connection,
            transaction,
            static command => command.CommandText =
                """
            CREATE TABLE IF NOT EXISTS translated_words (
                id TEXT NOT NULL PRIMARY KEY,
                translated_segment_id TEXT NOT NULL,
                word_index INTEGER NOT NULL,
                start_seconds REAL NOT NULL,
                end_seconds REAL NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY (translated_segment_id) REFERENCES translated_segments(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_translated_words_segment_id
                ON translated_words (translated_segment_id, word_index);
            """,
            cancellationToken);
    }

    private static Task RebuildTranslatedWordsUniqueIndexAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return ExecuteNonQueryAsync(
            connection,
            transaction,
            static command => command.CommandText =
                """
            DROP INDEX IF EXISTS ix_translated_words_segment_id;
            CREATE UNIQUE INDEX IF NOT EXISTS ix_translated_words_segment_id
                ON translated_words (translated_segment_id, word_index);
            """,
            cancellationToken);
    }

    private static Task AddSpeakerTurnMetadataAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return EnsureColumnsAsync(
            connection,
            transaction,
            "speaker_turns",
            cancellationToken,
            new("confidence", "REAL NULL"),
            new("has_overlap", "INTEGER NOT NULL DEFAULT 0"));
    }

    private static Task AddTtsTakeMetadataAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return EnsureColumnsAsync(
            connection,
            transaction,
            "tts_takes",
            cancellationToken,
            new("segment_index", "INTEGER NOT NULL DEFAULT 0"),
            new("translated_text_hash", "TEXT NULL"),
            new("model_id", "TEXT NULL"),
            new("voice_id", "TEXT NULL"),
            new("duration_overrun_ratio", "REAL NULL"),
            new("pre_stretch_duration_seconds", "REAL NULL"),
            new("stretch_ratio_applied", "REAL NULL"),
            new("stretch_mode", "TEXT NOT NULL DEFAULT 'None'"),
            new("stretch_engine", "TEXT NOT NULL DEFAULT 'None'"));
    }

    private static Task CreateCurrentProjectIndexesAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return ExecuteNonQueryAsync(
            connection,
            transaction,
            static command => command.CommandText =
                """
            CREATE UNIQUE INDEX IF NOT EXISTS ix_artifacts_project_relative_path
                ON artifacts (project_id, relative_path);
            CREATE INDEX IF NOT EXISTS ix_stage_runs_project_id
                ON StageRuns (ProjectId, StartedAtUtc);
            CREATE INDEX IF NOT EXISTS ix_transcript_revisions_project_id
                ON transcript_revisions (project_id, revision_number);
            CREATE INDEX IF NOT EXISTS ix_transcript_segments_revision_id
                ON transcript_segments (transcript_revision_id, segment_index);
            CREATE INDEX IF NOT EXISTS ix_speakers_project_id
                ON speakers (project_id, created_at_utc);
            CREATE INDEX IF NOT EXISTS ix_speaker_turns_project_id
                ON speaker_turns (project_id, start_seconds);
            CREATE INDEX IF NOT EXISTS ix_speaker_turns_speaker_id
                ON speaker_turns (speaker_id, start_seconds);
            CREATE INDEX IF NOT EXISTS ix_translation_revisions_project_language
                ON translation_revisions (project_id, target_language, revision_number);
            CREATE INDEX IF NOT EXISTS ix_translated_segments_revision_id
                ON translated_segments (translation_revision_id, segment_index);
            CREATE INDEX IF NOT EXISTS ix_tts_takes_project_segment
                ON tts_takes (project_id, segment_index, created_at_utc);
            CREATE INDEX IF NOT EXISTS ix_tts_takes_voice_assignment
                ON tts_takes (voice_assignment_id, is_stale);
            """,
            cancellationToken);
    }

    private static async Task AddVoiceCloningTtsMetadataAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        await EnsureColumnsAsync(
            connection,
            transaction,
            "voice_assignments",
            cancellationToken,
            new ColumnDefinition("reference_clip_artifact_id", "TEXT NULL")).ConfigureAwait(false);
        await EnsureColumnsAsync(
            connection,
            transaction,
            "tts_takes",
            cancellationToken,
            new ColumnDefinition("kind", "TEXT NOT NULL DEFAULT 'Stock'"),
            new ColumnDefinition("reference_clip_artifact_id", "TEXT NULL")).ConfigureAwait(false);
    }

    private static async Task AddTtsTakeInputFingerprintAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        await EnsureColumnsAsync(
            connection,
            transaction,
            "tts_takes",
            cancellationToken,
            new ColumnDefinition("input_fingerprint", "TEXT NULL")).ConfigureAwait(false);

        // Add a covering index so fingerprint cache-hit lookups (GetByFingerprintAsync)
        // remain O(log n) as the tts_takes table grows. Without this, every lookup
        // degrades to a full-table scan.
        await ExecuteNonQueryAsync(
            connection,
            transaction,
            static command => command.CommandText =
                """
            CREATE INDEX IF NOT EXISTS ix_tts_takes_input_fingerprint
                ON tts_takes (project_id, input_fingerprint, status, is_stale, created_at_utc);
            """,
            cancellationToken).ConfigureAwait(false);
    }

    private static async Task AddTtsCandidateMetadataAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        await EnsureColumnsAsync(
            connection,
            transaction,
            "tts_takes",
            cancellationToken,
            new ColumnDefinition("candidate_group_id", "TEXT NULL"),
            new ColumnDefinition("candidate_index", "INTEGER NULL"),
            new ColumnDefinition("candidate_variant", "INTEGER NOT NULL DEFAULT 0")).ConfigureAwait(false);

        await ExecuteNonQueryAsync(
            connection,
            transaction,
            static command => command.CommandText =
                """
            CREATE TABLE IF NOT EXISTS tts_candidate_groups (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL,
                translated_segment_id TEXT NOT NULL,
                segment_index INTEGER NOT NULL,
                selected_candidate_id TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                UNIQUE (translated_segment_id),
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (translated_segment_id) REFERENCES translated_segments(id) ON DELETE CASCADE,
                FOREIGN KEY (selected_candidate_id) REFERENCES tts_takes(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS ix_tts_candidate_groups_project_segment
                ON tts_candidate_groups (project_id, segment_index);

            CREATE INDEX IF NOT EXISTS ix_tts_takes_candidate_group
                ON tts_takes (candidate_group_id, candidate_index);
            """,
            cancellationToken).ConfigureAwait(false);
    }

    private static Task AddArtifactDegradationColumnsAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return EnsureColumnsAsync(
            connection,
            transaction,
            "artifacts",
            cancellationToken,
            new ColumnDefinition("degradation_code", "TEXT NULL"),
            new ColumnDefinition("degradation_stage", "TEXT NULL"));
    }

    private static async Task EnsureVersionTableAsync(SqliteConnection connection, CancellationToken cancellationToken)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            CREATE TABLE IF NOT EXISTS project_schema_versions (
                version INTEGER NOT NULL PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at_utc TEXT NOT NULL
            );
            """;
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    private static async Task<HashSet<int>> LoadAppliedVersionsAsync(
        SqliteConnection connection,
        CancellationToken cancellationToken)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            SELECT version
            FROM project_schema_versions
            ORDER BY version;
            """;

        var versions = new HashSet<int>();
        await using SqliteDataReader reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            versions.Add(reader.GetInt32(0));
        }

        return versions;
    }

    private static async Task RecordAppliedMigrationAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        SqliteProjectMigration migration,
        CancellationToken cancellationToken)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandText =
            """
            INSERT INTO project_schema_versions (version, name, applied_at_utc)
            VALUES ($version, $name, $appliedAtUtc);
            """;
        command.Parameters.AddWithValue("$version", migration.Version);
        command.Parameters.AddWithValue("$name", migration.Name);
        command.Parameters.AddWithValue(
            "$appliedAtUtc",
            DateTimeOffset.UtcNow.ToString("O", CultureInfo.InvariantCulture));

        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    private static async Task EnsureColumnsAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        string tableName,
        CancellationToken cancellationToken,
        params ColumnDefinition[] columns)
    {
        foreach (ColumnDefinition column in columns)
        {
            await EnsureColumnAsync(
                connection,
                transaction,
                tableName,
                column.Name,
                column.Sql,
                cancellationToken).ConfigureAwait(false);
        }
    }

    private static async Task EnsureColumnAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        string tableName,
        string columnName,
        string columnDefinition,
        CancellationToken cancellationToken)
    {
        if (await TableColumnExistsAsync(connection, transaction, tableName, columnName, cancellationToken).ConfigureAwait(false))
        {
            return;
        }

        await using SqliteCommand command = connection.CreateCommand();
        command.Transaction = transaction;
        // Table and column identifiers are internal migration constants, validated before use.
        command.CommandText = SqliteIdentifierHelper.BuildAlterTableAddColumn(
            tableName,
            columnName,
            columnDefinition);
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    private static async Task<bool> TableColumnExistsAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        string tableName,
        string columnName,
        CancellationToken cancellationToken)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.Transaction = transaction;
        command.CommandText = SqliteIdentifierHelper.BuildPragmaTableInfo(tableName);

        await using SqliteDataReader reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            if (string.Equals(reader.GetString(1), columnName, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private static async Task ExecuteNonQueryAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        Action<SqliteCommand> configureCommand,
        CancellationToken cancellationToken)
    {
        await using SqliteCommand command = connection.CreateCommand();
        command.Transaction = transaction;
        configureCommand(command);
        await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
    }

    private static Task AddStageRunExtendedRuntimeInfoAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return EnsureColumnsAsync(
            connection,
            transaction,
            "StageRuns",
            cancellationToken,
            new ColumnDefinition("DeviceTarget", "TEXT NULL"),
            new ColumnDefinition("FallbackReason", "TEXT NULL"),
            new ColumnDefinition("SmokeEvidenceId", "TEXT NULL"),
            new ColumnDefinition("BenchmarkEvidenceId", "TEXT NULL"));
    }

    private static Task CreateGlossaryEntriesAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return ExecuteNonQueryAsync(
            connection,
            transaction,
            static command => command.CommandText =
                """
            CREATE TABLE IF NOT EXISTS glossary_entries (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL,
                source_language TEXT NOT NULL,
                target_language TEXT NOT NULL,
                source_term TEXT NOT NULL,
                target_term TEXT NOT NULL,
                is_case_sensitive INTEGER NOT NULL,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS ix_glossary_entries_project_language
                ON glossary_entries (project_id, source_language, target_language, source_term, target_term);
            """,
            cancellationToken);
    }

    private static Task CreateVoiceCloneConsentsAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return ExecuteNonQueryAsync(
            connection,
            transaction,
            static command => command.CommandText =
                """
            CREATE TABLE IF NOT EXISTS voice_clone_consents (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL,
                speaker_id TEXT NOT NULL,
                granted_at_utc TEXT NOT NULL,
                consent_version TEXT NOT NULL DEFAULT 'v1',
                is_third_party INTEGER NOT NULL DEFAULT 0,
                notes TEXT NULL,
                expires_at_utc TEXT NULL,
                revoked_at_utc TEXT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
                FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE CASCADE
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ix_voice_clone_consents_speaker
                ON voice_clone_consents (project_id, speaker_id);
            """,
            cancellationToken);
    }

    private static Task CreateLipSyncSegmentsAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return ExecuteNonQueryAsync(
            connection,
            transaction,
            static command => command.CommandText =
                """
            CREATE TABLE IF NOT EXISTS LipSyncSegments (
                Id TEXT NOT NULL PRIMARY KEY,
                ProjectId TEXT NOT NULL,
                TranslatedSegmentId TEXT NOT NULL,
                StageRunId TEXT NOT NULL,
                Status TEXT NOT NULL,
                SourceAlignmentId TEXT NULL,
                TtsAlignmentId TEXT NULL,
                SourceDurationSeconds REAL NOT NULL,
                TtsDurationSeconds REAL NOT NULL,
                AlignedTtsDurationSeconds REAL NULL,
                PlanConfidence REAL NULL,
                SkipReason TEXT NULL,
                FailureReason TEXT NULL,
                ProviderId TEXT NULL,
                ModelId TEXT NULL,
                CreatedAtUtc TEXT NOT NULL,
                FOREIGN KEY (ProjectId) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS ix_LipSyncSegments_ProjectId
                ON LipSyncSegments (ProjectId);

            CREATE INDEX IF NOT EXISTS ix_LipSyncSegments_StageRunId
                ON LipSyncSegments (StageRunId);
            """,
            cancellationToken);
    }

    private static Task CreateLipSynthesisSegmentsAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return ExecuteNonQueryAsync(
            connection,
            transaction,
            static command => command.CommandText =
                """
            CREATE TABLE IF NOT EXISTS LipSynthesisSegments (
                Id TEXT NOT NULL PRIMARY KEY,
                ProjectId TEXT NOT NULL,
                StageRunId TEXT NOT NULL,
                Status TEXT NOT NULL,
                SpeakerId TEXT NULL,
                TurnStartSeconds REAL NOT NULL,
                TurnEndSeconds REAL NOT NULL,
                FaceConfidence REAL NULL,
                PatchedClipRelativePath TEXT NULL,
                SkipReason TEXT NULL,
                FailureReason TEXT NULL,
                ProviderId TEXT NULL,
                ModelId TEXT NULL,
                UsedExperimentalProvider INTEGER NOT NULL,
                CreatedAtUtc TEXT NOT NULL,
                FOREIGN KEY (ProjectId) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS ix_LipSynthesisSegments_ProjectId
                ON LipSynthesisSegments (ProjectId);

            CREATE INDEX IF NOT EXISTS ix_LipSynthesisSegments_StageRunId
                ON LipSynthesisSegments (StageRunId);
            """,
            cancellationToken);
    }

    private static Task RebuildLipSynthesisSegmentsStageRunKeyAsync(
        SqliteConnection connection,
        SqliteTransaction transaction,
        CancellationToken cancellationToken)
    {
        return ExecuteNonQueryAsync(
            connection,
            transaction,
            static command => command.CommandText =
                """
            CREATE TABLE LipSynthesisSegments__v27 (
                SegmentId TEXT NOT NULL,
                ProjectId TEXT NOT NULL,
                StageRunId TEXT NOT NULL,
                Status TEXT NOT NULL,
                SpeakerId TEXT NULL,
                TurnStartSeconds REAL NOT NULL,
                TurnEndSeconds REAL NOT NULL,
                FaceConfidence REAL NULL,
                PatchedClipRelativePath TEXT NULL,
                SkipReason TEXT NULL,
                FailureReason TEXT NULL,
                ProviderId TEXT NULL,
                ModelId TEXT NULL,
                UsedExperimentalProvider INTEGER NOT NULL,
                CreatedAtUtc TEXT NOT NULL,
                PRIMARY KEY (StageRunId, SegmentId),
                FOREIGN KEY (ProjectId) REFERENCES projects(id) ON DELETE CASCADE
            );

            INSERT INTO LipSynthesisSegments__v27 (
                SegmentId,
                ProjectId,
                StageRunId,
                Status,
                SpeakerId,
                TurnStartSeconds,
                TurnEndSeconds,
                FaceConfidence,
                PatchedClipRelativePath,
                SkipReason,
                FailureReason,
                ProviderId,
                ModelId,
                UsedExperimentalProvider,
                CreatedAtUtc)
            SELECT
                Id,
                ProjectId,
                StageRunId,
                Status,
                SpeakerId,
                TurnStartSeconds,
                TurnEndSeconds,
                FaceConfidence,
                PatchedClipRelativePath,
                SkipReason,
                FailureReason,
                ProviderId,
                ModelId,
                UsedExperimentalProvider,
                CreatedAtUtc
            FROM LipSynthesisSegments;

            DROP TABLE LipSynthesisSegments;

            ALTER TABLE LipSynthesisSegments__v27 RENAME TO LipSynthesisSegments;

            CREATE INDEX IF NOT EXISTS ix_LipSynthesisSegments_ProjectId
                ON LipSynthesisSegments (ProjectId);

            CREATE INDEX IF NOT EXISTS ix_LipSynthesisSegments_StageRunId
                ON LipSynthesisSegments (StageRunId);
            """,
            cancellationToken);
    }

    private sealed record ColumnDefinition(string Name, string Sql);
}

internal sealed record SqliteProjectMigration(
    int Version,
    string Name,
    Func<SqliteConnection, SqliteTransaction, CancellationToken, Task> ApplyAsync);
