using System.Text.Json;
using Trackdub.Contracts;
using Trackdub.Contracts.Transcripts;
using Trackdub.Domain.Translation;
using DomainTranslatedSegment = Trackdub.Domain.Translation.TranslatedSegment;
using Microsoft.Data.Sqlite;

namespace Trackdub.Infrastructure.Persistence.Sqlite;

public sealed class SqliteTranslationRepository(
    SqliteProjectDatabase database,
    IScopedConnectionProvider? scopedConnectionProvider = null)
    : ITranslationRepository
{
    public async Task<TranslationRevision?> GetCurrentRevisionAsync(
        Guid projectId,
        string targetLanguage,
        CancellationToken cancellationToken)
    {
        await database.InitializeAsync(cancellationToken).ConfigureAwait(false);
        await using SqliteConnectionLease connectionLease = await OpenConnectionAsync(cancellationToken).ConfigureAwait(false);
        SqliteConnection connection = connectionLease.Connection;
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            SELECT id,
                   project_id,
                   stage_run_id,
                   source_transcript_revision_id,
                   target_language,
                   translation_provider,
                   model_id,
                   execution_provider,
                   revision_number,
                   created_at_utc,
                   model_provenance_json
            FROM translation_revisions
            WHERE project_id = $projectId
              AND target_language = $targetLanguage
            ORDER BY revision_number DESC
            LIMIT 1;
            """;
        command.Parameters.AddWithValue("$projectId", projectId.ToString("D"));
        command.Parameters.AddWithValue("$targetLanguage", NormalizeLanguageCode(targetLanguage));

        await using SqliteDataReader reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        if (!await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            return null;
        }

        return ReadRevision(reader);
    }

    public async Task<IReadOnlyList<DomainTranslatedSegment>> GetSegmentsAsync(
        Guid translationRevisionId,
        CancellationToken cancellationToken)
    {
        await database.InitializeAsync(cancellationToken).ConfigureAwait(false);
        await using SqliteConnectionLease connectionLease = await OpenConnectionAsync(cancellationToken).ConfigureAwait(false);
        SqliteConnection connection = connectionLease.Connection;
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            SELECT id,
                   translation_revision_id,
                   segment_index,
                   start_seconds,
                   end_seconds,
                   text,
                   source_segment_hash,
                   origin
            FROM translated_segments
            WHERE translation_revision_id = $translationRevisionId
            ORDER BY segment_index;
            """;
        command.Parameters.AddWithValue("$translationRevisionId", translationRevisionId.ToString("D"));

        var results = new List<DomainTranslatedSegment>();
        await using SqliteDataReader reader = await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
        while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            results.Add(new DomainTranslatedSegment(
                Guid.Parse(reader.GetString(0)),
                Guid.Parse(reader.GetString(1)),
                reader.GetInt32(2),
                reader.GetDouble(3),
                reader.GetDouble(4),
                reader.GetString(5),
                reader.IsDBNull(6) ? null : reader.GetString(6))
            {
                Origin = Enum.IsDefined(typeof(TranslationSegmentOrigin), reader.GetInt32(7))
                    ? (TranslationSegmentOrigin)reader.GetInt32(7)
                    : TranslationSegmentOrigin.Unknown,
            });
        }

        if (results.Count == 0)
        {
            return results;
        }

        IReadOnlyDictionary<Guid, IReadOnlyList<TranslatedWord>> wordsBySegmentId =
            await LoadWordsAsync(connection, results.Select(segment => segment.Id).ToArray(), cancellationToken)
                .ConfigureAwait(false);

        return results
            .Select(segment => segment with
            {
                Words = wordsBySegmentId.TryGetValue(segment.Id, out IReadOnlyList<TranslatedWord>? words)
                    ? NormalizeLoadedWords(words)
                    : []
            })
            .ToArray();
    }

    public async Task<int> GetNextRevisionNumberAsync(
        Guid projectId,
        string targetLanguage,
        CancellationToken cancellationToken)
    {
        await database.InitializeAsync(cancellationToken).ConfigureAwait(false);
        await using SqliteConnectionLease connectionLease = await OpenConnectionAsync(cancellationToken).ConfigureAwait(false);
        SqliteConnection connection = connectionLease.Connection;
        await using SqliteCommand command = connection.CreateCommand();
        command.CommandText =
            """
            SELECT COALESCE(MAX(revision_number), 0)
            FROM translation_revisions
            WHERE project_id = $projectId
              AND target_language = $targetLanguage;
            """;
        command.Parameters.AddWithValue("$projectId", projectId.ToString("D"));
        command.Parameters.AddWithValue("$targetLanguage", NormalizeLanguageCode(targetLanguage));

        object? result = await command.ExecuteScalarAsync(cancellationToken).ConfigureAwait(false);
        int currentValue = Convert.ToInt32(result, System.Globalization.CultureInfo.InvariantCulture);
        return currentValue + 1;
    }

    public async Task SaveRevisionAsync(
        TranslationRevision revision,
        IReadOnlyList<DomainTranslatedSegment> segments,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(revision);
        ArgumentNullException.ThrowIfNull(segments);

        await database.InitializeAsync(cancellationToken).ConfigureAwait(false);
        await using SqliteConnectionLease connectionLease = await OpenConnectionAsync(cancellationToken).ConfigureAwait(false);
        SqliteConnection connection = connectionLease.Connection;
        await using SqliteTransaction transaction = (SqliteTransaction)await connection.BeginTransactionAsync(cancellationToken).ConfigureAwait(false);

        await SaveRevisionAsync(connection, transaction, revision, segments, cancellationToken).ConfigureAwait(false);
        await transaction.CommitAsync(cancellationToken).ConfigureAwait(false);
    }

    internal static async Task SaveRevisionAsync(
        SqliteConnection connection, SqliteTransaction transaction, TranslationRevision revision,
        IReadOnlyList<DomainTranslatedSegment> segments, CancellationToken cancellationToken,
        bool guardExpectedHeads = false, Guid? expectedPrevious = null)
    {
        await using (SqliteCommand revisionCommand = connection.CreateCommand())
        {
            revisionCommand.Transaction = transaction;
            revisionCommand.CommandText =
                """
                INSERT INTO translation_revisions (
                    id,
                    project_id,
                    stage_run_id,
                    source_transcript_revision_id,
                    target_language,
                    translation_provider,
                    model_id,
                    execution_provider,
                    revision_number,
                    created_at_utc,
                    model_provenance_json)
                SELECT
                    $id,
                    $projectId,
                    $stageRunId,
                    $sourceTranscriptRevisionId,
                    $targetLanguage,
                    $translationProvider,
                    $modelId,
                    $executionProvider,
                    $revisionNumber,
                    $createdAtUtc,
                    $modelProvenance
                WHERE $guard = 0 OR (
                    (SELECT id FROM transcript_revisions WHERE project_id = $projectId ORDER BY revision_number DESC LIMIT 1) = $sourceTranscriptRevisionId
                    AND (SELECT id FROM translation_revisions WHERE project_id = $projectId AND target_language = $targetLanguage ORDER BY revision_number DESC LIMIT 1) IS $expectedPrevious);
                """;
            revisionCommand.Parameters.AddWithValue("$id", revision.Id.ToString("D"));
            revisionCommand.Parameters.AddWithValue("$projectId", revision.ProjectId.ToString("D"));
            revisionCommand.Parameters.AddWithValue("$stageRunId", revision.StageRunId?.ToString("D") ?? (object)DBNull.Value);
            revisionCommand.Parameters.AddWithValue("$sourceTranscriptRevisionId", revision.SourceTranscriptRevisionId.ToString("D"));
            revisionCommand.Parameters.AddWithValue("$targetLanguage", revision.TargetLanguage);
            revisionCommand.Parameters.AddWithValue("$translationProvider", revision.TranslationProvider ?? (object)DBNull.Value);
            revisionCommand.Parameters.AddWithValue("$modelId", revision.ModelId ?? (object)DBNull.Value);
            revisionCommand.Parameters.AddWithValue("$executionProvider", revision.ExecutionProvider ?? (object)DBNull.Value);
            revisionCommand.Parameters.AddWithValue("$modelProvenance", revision.ModelProvenance is null ? (object)DBNull.Value : JsonSerializer.Serialize(revision.ModelProvenance));
            revisionCommand.Parameters.AddWithValue("$revisionNumber", revision.RevisionNumber);
            revisionCommand.Parameters.AddWithValue("$createdAtUtc", revision.CreatedAtUtc.UtcDateTime);
            revisionCommand.Parameters.AddWithValue("$guard", guardExpectedHeads ? 1 : 0);
            revisionCommand.Parameters.AddWithValue("$expectedPrevious", expectedPrevious?.ToString("D") ?? (object)DBNull.Value);
            if (await revisionCommand.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false) != 1)
            {
                throw new InvalidDataException("Revision publication conflicted with the expected revision heads.");
            }
        }

        await using SqliteCommand segmentCommand = connection.CreateCommand();
        segmentCommand.Transaction = transaction;
        segmentCommand.CommandText =
            """
            INSERT INTO translated_segments (
                id,
                translation_revision_id,
                segment_index,
                start_seconds,
                end_seconds,
                text,
                source_segment_hash,
                origin)
            VALUES (
                $id,
                $translationRevisionId,
                $segmentIndex,
                $startSeconds,
                $endSeconds,
                $text,
                $sourceSegmentHash,
                $origin);
            """;
        SqliteParameter segmentIdParameter = segmentCommand.Parameters.Add("$id", SqliteType.Text);
        SqliteParameter translationRevisionIdParameter = segmentCommand.Parameters.Add("$translationRevisionId", SqliteType.Text);
        SqliteParameter segmentIndexParameter = segmentCommand.Parameters.Add("$segmentIndex", SqliteType.Integer);
        SqliteParameter startSecondsParameter = segmentCommand.Parameters.Add("$startSeconds", SqliteType.Real);
        SqliteParameter endSecondsParameter = segmentCommand.Parameters.Add("$endSeconds", SqliteType.Real);
        SqliteParameter textParameter = segmentCommand.Parameters.Add("$text", SqliteType.Text);
        SqliteParameter sourceSegmentHashParameter = segmentCommand.Parameters.Add("$sourceSegmentHash", SqliteType.Text);

        SqliteParameter originParameter = segmentCommand.Parameters.Add("$origin", SqliteType.Integer);

        await using SqliteCommand wordCommand = connection.CreateCommand();
        wordCommand.Transaction = transaction;
        wordCommand.CommandText =
            """
                INSERT INTO translated_words (
                    id,
                    translated_segment_id,
                    word_index,
                    start_seconds,
                    end_seconds,
                    text)
                VALUES (
                    $id,
                    $translatedSegmentId,
                    $wordIndex,
                    $startSeconds,
                    $endSeconds,
                    $text);
            """;
        SqliteParameter wordIdParameter = wordCommand.Parameters.Add("$id", SqliteType.Text);
        SqliteParameter wordSegmentIdParameter = wordCommand.Parameters.Add("$translatedSegmentId", SqliteType.Text);
        SqliteParameter wordIndexParameter = wordCommand.Parameters.Add("$wordIndex", SqliteType.Integer);
        SqliteParameter wordStartSecondsParameter = wordCommand.Parameters.Add("$startSeconds", SqliteType.Real);
        SqliteParameter wordEndSecondsParameter = wordCommand.Parameters.Add("$endSeconds", SqliteType.Real);
        SqliteParameter wordTextParameter = wordCommand.Parameters.Add("$text", SqliteType.Text);

        foreach (DomainTranslatedSegment segment in segments.OrderBy(segment => segment.SegmentIndex))
        {
            segmentIdParameter.Value = segment.Id.ToString("D");
            translationRevisionIdParameter.Value = segment.TranslationRevisionId.ToString("D");
            segmentIndexParameter.Value = segment.SegmentIndex;
            startSecondsParameter.Value = segment.StartSeconds;
            endSecondsParameter.Value = segment.EndSeconds;
            textParameter.Value = segment.Text;
            sourceSegmentHashParameter.Value = segment.SourceSegmentHash ?? (object)DBNull.Value;
            originParameter.Value = (int)segment.Origin;
            await segmentCommand.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);

            foreach (TranslatedWord word in segment.Words.OrderBy(static word => word.WordIndex))
            {
                wordIdParameter.Value = Guid.NewGuid().ToString("D");
                wordSegmentIdParameter.Value = segment.Id.ToString("D");
                wordIndexParameter.Value = word.WordIndex;
                wordStartSecondsParameter.Value = word.StartSeconds;
                wordEndSecondsParameter.Value = word.EndSeconds;
                wordTextParameter.Value = word.Text;
                await wordCommand.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
            }
        }
    }

    private static TranslationRevision ReadRevision(SqliteDataReader reader) =>
        new(
            Guid.Parse(reader.GetString(0)),
            Guid.Parse(reader.GetString(1)),
            reader.IsDBNull(2) ? null : Guid.Parse(reader.GetString(2)),
            Guid.Parse(reader.GetString(3)),
            reader.GetString(4),
            reader.IsDBNull(5) ? null : reader.GetString(5),
            reader.IsDBNull(6) ? null : reader.GetString(6),
            reader.IsDBNull(7) ? null : reader.GetString(7),
            reader.GetInt32(8),
            new DateTimeOffset(DateTime.SpecifyKind(reader.GetDateTime(9), DateTimeKind.Utc)),
            reader.IsDBNull(10) ? null : JsonSerializer.Deserialize<TranslationModelProvenance>(reader.GetString(10)));

    private static string NormalizeLanguageCode(string targetLanguage)
    {
        if (string.IsNullOrWhiteSpace(targetLanguage))
        {
            throw new ArgumentException("Target language is required.", nameof(targetLanguage));
        }

        return targetLanguage.Trim().ToLowerInvariant();
    }

    private static async Task<IReadOnlyDictionary<Guid, IReadOnlyList<TranslatedWord>>> LoadWordsAsync(
        SqliteConnection connection,
        IReadOnlyList<Guid> segmentIds,
        CancellationToken cancellationToken)
    {
        if (segmentIds.Count == 0)
        {
            return new Dictionary<Guid, IReadOnlyList<TranslatedWord>>();
        }

        // Query translated_words using chunked IN (...) clauses instead of a shared temp table.
        // Each chunk uses its own parameters and command, keeping calls fully isolated with no
        // risk of concurrent interference. Chunk size stays well under SQLite's parameter limit.
        const int chunkSize = 500;
        var wordsBySegmentId = new Dictionary<Guid, List<TranslatedWord>>();

        for (int chunkStart = 0; chunkStart < segmentIds.Count; chunkStart += chunkSize)
        {
            int chunkEnd = Math.Min(chunkStart + chunkSize, segmentIds.Count);
            int count = chunkEnd - chunkStart;

            await using SqliteCommand command = connection.CreateCommand();
            var placeholders = new string[count];
            for (int i = 0; i < count; i++)
            {
                string paramName = $"$id{i}";
                placeholders[i] = paramName;
                command.Parameters.Add(paramName, SqliteType.Text).Value =
                    segmentIds[chunkStart + i].ToString("D");
            }

            command.CommandText = string.Concat(
                SelectTranslatedWordsBySegmentIdsPrefix,
                string.Join(", ", placeholders),
                SelectTranslatedWordsBySegmentIdsSuffix);

            await using SqliteDataReader reader =
                await command.ExecuteReaderAsync(cancellationToken).ConfigureAwait(false);
            while (await reader.ReadAsync(cancellationToken).ConfigureAwait(false))
            {
                Guid segmentId = Guid.Parse(reader.GetString(0));
                if (!wordsBySegmentId.TryGetValue(segmentId, out List<TranslatedWord>? words))
                {
                    words = [];
                    wordsBySegmentId[segmentId] = words;
                }

                words.Add(TranslatedWord.Create(
                    reader.GetInt32(1),
                    reader.GetDouble(2),
                    reader.GetDouble(3),
                    reader.GetString(4)));
            }
        }

        return wordsBySegmentId.ToDictionary(
            entry => entry.Key,
            entry => (IReadOnlyList<TranslatedWord>)entry.Value.ToArray());
    }

    private const string SelectTranslatedWordsBySegmentIdsPrefix =
        """
        SELECT w.translated_segment_id,
               w.word_index,
               w.start_seconds,
               w.end_seconds,
               w.text
        FROM translated_words w
        WHERE w.translated_segment_id IN (
        """;

    private const string SelectTranslatedWordsBySegmentIdsSuffix =
        """
        )
        ORDER BY w.translated_segment_id, w.word_index;
        """;

    /// <summary>
    /// Normalizes words loaded from the database by ordering them by <c>word_index</c> and
    /// rebuilding sequential indices starting at zero, matching the normalization applied by
    /// <see cref="TranslatedSegment"/>'s constructor. This prevents gaps, duplicates, or
    /// out-of-order rows in the database from leaking inconsistent sequences into the domain.
    /// </summary>
    private static IReadOnlyList<TranslatedWord> NormalizeLoadedWords(IReadOnlyList<TranslatedWord> words) =>
        words.Count == 0
            ? []
            : words
                .OrderBy(static word => word.WordIndex)
                .Select(static (word, index) => TranslatedWord.Create(
                    index,
                    word.StartSeconds,
                    word.EndSeconds,
                    word.Text))
                .ToArray();

    private Task<SqliteConnectionLease> OpenConnectionAsync(CancellationToken cancellationToken) =>
        SqliteConnectionLease.OpenAsync(database, scopedConnectionProvider, cancellationToken);
}
