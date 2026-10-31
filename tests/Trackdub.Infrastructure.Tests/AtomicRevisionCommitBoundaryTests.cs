using System.Data;
using System.Runtime.CompilerServices;
using Microsoft.Data.Sqlite;
using Trackdub.Application.Transcripts;
using Trackdub.Application.Mixing;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Projects;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Trackdub.Domain.Tts;
using Trackdub.Infrastructure.FileSystem;
using Trackdub.Infrastructure.Persistence.Sqlite;
using Trackdub.TestDoubles;

namespace Trackdub.Infrastructure.Tests;

public sealed partial class AtomicRevisionCommitBoundaryTests
{
    [Fact]
    public async Task Streaming_wrong_revision_reaches_real_boundary_after_valid_items_and_changes_no_output()
    {
        await using var f = await Fixture.CreateAsync();
        await f.SeedTranslationAsync();
        var before = await f.OutputSnapshotAsync();
        var engine = new StreamEngine { WrongRevisionAt = 2, EmptyTextAt = 1 };
        var recorder = new RecordingBoundary(f.Boundary);
        await Assert.ThrowsAsync<InvalidDataException>(() => f.GenerateAsync(engine, recorder));
        Assert.Equal(3, engine.Emitted);
        var stream = Assert.IsType<TranslationStreamCommit>(Assert.Single(recorder.Requests).Stream);
        Assert.Equal(3, stream.Items.Count);
        Assert.Equal(f.Transcript.Id, stream.Items[0].Identity.RevisionId);
        Assert.Equal(f.Transcript.Id, stream.Items[1].Identity.RevisionId);
        Assert.NotEqual(f.Transcript.Id, stream.Items[2].Identity.RevisionId);
        Assert.Equal(before, await f.OutputSnapshotAsync());
        Assert.Empty(Directory.EnumerateFiles(Path.Join(f.Root, "temp")));
    }

    [Theory]
    [InlineData("success")]
    [InlineData("incomplete")]
    [InlineData("fault")]
    [InlineData("source-change")]
    [InlineData("canceled")]
    public async Task Production_streaming_path_publishes_only_complete_current_output(string mode)
    {
        await using var f = await Fixture.CreateAsync();
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(f.Ct);
        var engine = new StreamEngine { Limit = mode == "incomplete" ? 2 : 3, FaultAfter = mode == "fault" ? 2 : null };
        if (mode == "source-change") engine.BeforeLast = () => f.ReplaceSourceAsync();
        if (mode == "canceled") engine.BeforeLast = () => { cts.Cancel(); return Task.CompletedTask; };
        if (mode == "success")
        {
            await f.GenerateAsync(engine, f.Boundary);
            var revision = await f.Translations.GetCurrentRevisionAsync(f.Project.Id, "es", f.Ct);
            Assert.NotNull(revision);
            Assert.Equal(3, (await f.Translations.GetSegmentsAsync(revision.Id, f.Ct)).Count);
            var artifact = Assert.Single(await f.Media.GetArtifactsAsync(f.Project.Id, f.Ct));
            Assert.True(File.Exists(f.Store.GetPath(artifact.RelativePath)));
            Assert.Contains(revision.Id.ToString("N"), artifact.RelativePath);
        }
        else
        {
            await Assert.ThrowsAnyAsync<Exception>(() => f.GenerateAsync(engine, f.Boundary, cts.Token));
            Assert.Null(await f.Translations.GetCurrentRevisionAsync(f.Project.Id, "es", f.Ct));
            Assert.Empty(await f.Media.GetArtifactsAsync(f.Project.Id, f.Ct));
            Assert.Empty(Directory.EnumerateFiles(Path.Join(f.Root, "temp")));
            Assert.Empty(Directory.EnumerateFiles(Path.Join(f.Root, "artifacts"), "*", SearchOption.AllDirectories));
        }
    }

    [Fact]
    public async Task Translation_changed_during_streaming_inference_rejects_expected_previous_head()
    {
        await using var f = await Fixture.CreateAsync();
        string? concurrentOutput = null;
        var engine = new StreamEngine
        {
            BeforeLast = async () => { await f.SeedTranslationAsync(); concurrentOutput = await f.OutputSnapshotAsync(); }
        };
        var recorder = new RecordingBoundary(f.Boundary);
        await Assert.ThrowsAsync<InvalidDataException>(() => f.GenerateAsync(engine, recorder));
        Assert.Equal(3, Assert.Single(recorder.Requests).Stream!.Items.Count);
        Assert.Equal(concurrentOutput, await f.OutputSnapshotAsync());
    }

    [Fact]
    public async Task Write_transaction_blocks_competing_writer_between_validation_and_publication()
    {
        await using var f = await Fixture.CreateAsync();
        await using var competing = await f.Database.OpenConnectionAsync(f.Ct);
        competing.DefaultTimeout = 1;
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        f.Persistence.BeforePublicationAsync = async ct => { entered.SetResult(); await release.Task.WaitAsync(ct); };
        Task<TranslationRevision> publication = Task.Run(() => f.CommitRevisionAsync(), f.Ct);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(15), f.Ct);
        try
        {
            var error = await Assert.ThrowsAsync<SqliteException>(() => Task.Run(() =>
            {
                using var transaction = competing.BeginTransaction(IsolationLevel.Serializable, deferred: false);
            }, f.Ct));
            Assert.Equal(5, error.SqliteErrorCode);
            await using var read = competing.CreateCommand();
            read.CommandText = "SELECT COUNT(*) FROM translation_revisions;";
            Assert.Equal(0L, await read.ExecuteScalarAsync(f.Ct));
        }
        finally { release.TrySetResult(); }
        await publication;
        Assert.NotNull(await f.Translations.GetCurrentRevisionAsync(f.Project.Id, "es", f.Ct));
    }

    [Fact]
    public async Task Writer_that_commits_first_causes_expected_translation_head_conflict()
    {
        await using var f = await Fixture.CreateAsync();
        var request = f.NewRevisionRequest();
        await f.SeedTranslationAsync();
        var before = await f.OutputSnapshotAsync();
        await Assert.ThrowsAsync<InvalidDataException>(() => f.Boundary.CommitRevisionAsync(request, f.PrepareRevisionAsync, f.Ct));
        Assert.Equal(before, await f.OutputSnapshotAsync());
    }

    [Fact]
    public async Task Writer_that_commits_source_first_causes_expected_source_head_conflict()
    {
        await using var f = await Fixture.CreateAsync();
        var request = f.NewRevisionRequest();
        await f.ReplaceSourceAsync();
        var before = await f.OutputSnapshotAsync();
        await Assert.ThrowsAsync<InvalidDataException>(() => f.Boundary.CommitRevisionAsync(request, f.PrepareRevisionAsync, f.Ct));
        Assert.Equal(before, await f.OutputSnapshotAsync());
    }

    [Fact]
    public async Task Artifact_preparation_does_not_hold_writer_and_a_writer_during_preparation_is_rejected()
    {
        await using var f = await Fixture.CreateAsync();
        string? concurrentOutput = null;
        await Assert.ThrowsAsync<InvalidDataException>(() => f.Boundary.CommitRevisionAsync(f.NewRevisionRequest(),
            async (r, ct) =>
            {
                // This independent writer completes before BEGIN IMMEDIATE. The expected-head
                // validation must still reject the staged revision after preparation finishes.
                await f.SeedTranslationAsync();
                concurrentOutput = await f.OutputSnapshotAsync();
                return await f.PrepareRevisionAsync(r, ct);
            }, f.Ct));
        Assert.Equal(concurrentOutput, await f.OutputSnapshotAsync());
        Assert.Empty(Directory.EnumerateFiles(Path.Join(f.Root, "temp")));
    }

    [Fact]
    public async Task Sql_failure_after_revision_insert_rolls_back_all_rows_and_temporary_artifact()
    {
        await using var f = await Fixture.CreateAsync();
        await using (var connection = await f.Database.OpenConnectionAsync(f.Ct))
        {
            await using var trigger = connection.CreateCommand();
            trigger.CommandText = "CREATE TRIGGER reject_segment BEFORE INSERT ON translated_segments BEGIN SELECT RAISE(ABORT, 'injected failure'); END;";
            await trigger.ExecuteNonQueryAsync(f.Ct);
        }
        await Assert.ThrowsAsync<SqliteException>(() => f.CommitRevisionAsync());
        Assert.Null(await f.Translations.GetCurrentRevisionAsync(f.Project.Id, "es", f.Ct));
        Assert.Empty(await f.Media.GetArtifactsAsync(f.Project.Id, f.Ct));
        Assert.Empty(Directory.EnumerateFiles(Path.Join(f.Root, "temp")));
        Assert.Empty(Directory.EnumerateFiles(Path.Join(f.Root, "artifacts"), "*", SearchOption.AllDirectories));
    }

    [Theory]
    [InlineData("collision")]
    [InlineData("cancel-before-promotion")]
    [InlineData("cancel-after-promotion")]
    [InlineData("commit-failure")]
    [InlineData("commit-ack-lost")]
    [InlineData("content-replaced")]
    public async Task Artifact_cleanup_respects_ownership_and_durable_publication(string failure)
    {
        await using var f = await Fixture.CreateAsync();
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(f.Ct);
        PreparedCommitArtifact? prepared = null;
        Func<TranslationRevision, CancellationToken, Task<PreparedCommitArtifact>> prepare = async (r, ct) =>
        {
            prepared = await f.PrepareRevisionAsync(r, ct);
            return prepared;
        };
        f.Persistence.BeforePublicationAsync = _ =>
        {
            if (failure == "collision")
            {
                Directory.CreateDirectory(Path.GetDirectoryName(prepared!.Handle.FinalPath)!);
                File.WriteAllText(prepared.Handle.FinalPath, "pre-existing");
            }
            if (failure == "cancel-before-promotion") cts.Cancel();
            return Task.CompletedTask;
        };
        f.Persistence.CommitTransactionAsync = async (tx, ct) =>
        {
            if (failure == "cancel-after-promotion") { cts.Cancel(); ct.ThrowIfCancellationRequested(); }
            if (failure == "commit-failure") throw new IOException("injected commit failure");
            if (failure == "content-replaced") { await File.WriteAllTextAsync(prepared!.Handle.FinalPath, "replacement", ct); throw new IOException("injected failure"); }
            await tx.CommitAsync(ct);
            if (failure == "commit-ack-lost") { cts.Cancel(); throw new IOException("commit acknowledgement lost"); }
        };
        Task<TranslationRevision> operation = f.Boundary.CommitRevisionAsync(f.NewRevisionRequest(), prepare, cts.Token);
        if (failure == "commit-ack-lost") await operation;
        else await Assert.ThrowsAnyAsync<Exception>(() => operation);
        Assert.NotNull(prepared);
        bool shouldPreserve = failure is "collision" or "commit-ack-lost" or "content-replaced";
        Assert.Equal(shouldPreserve, File.Exists(prepared.Handle.FinalPath));
        Assert.False(File.Exists(prepared.Handle.TemporaryPath));
        Assert.Equal(failure == "commit-ack-lost", (await f.Translations.GetCurrentRevisionAsync(f.Project.Id, "es", f.Ct)) is not null);
        if (failure == "collision") Assert.Equal("pre-existing", await File.ReadAllTextAsync(prepared.Handle.FinalPath, f.Ct));
        if (failure == "content-replaced") Assert.Equal("replacement", await File.ReadAllTextAsync(prepared.Handle.FinalPath, f.Ct));
    }

    [Fact]
    public async Task Unavailable_recovery_connection_preserves_owned_files_and_reports_unresolved_publication()
    {
        await using var f = await Fixture.CreateAsync();
        PreparedCommitArtifact? prepared = null;
        f.Persistence.CommitTransactionAsync = (_, _) => throw new IOException("commit outcome unavailable");
        f.Persistence.OpenRecoveryConnectionAsync = _ => throw new IOException("database unavailable");
        var error = await Assert.ThrowsAsync<IOException>(() => f.Boundary.CommitRevisionAsync(f.NewRevisionRequest(),
            async (r, ct) => prepared = await f.PrepareRevisionAsync(r, ct), f.Ct));
        Assert.Contains("could not be resolved", error.Message, StringComparison.Ordinal);
        Assert.True(prepared!.Receipt.CreatedByAttempt);
        Assert.True(File.Exists(prepared.Handle.FinalPath));
        Assert.False(File.Exists(prepared.Handle.TemporaryPath));
        Assert.Null(await f.Translations.GetCurrentRevisionAsync(f.Project.Id, "es", f.Ct));
        Assert.Empty(await f.Media.GetArtifactsAsync(f.Project.Id, f.Ct));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Cleanup_preserves_preexisting_or_shared_database_referenced_paths(bool preexistingFile)
    {
        await using var f = await Fixture.CreateAsync();
        const string sharedPath = "artifacts/translation/shared-content.json";
        await using var shared = await f.PrepareBytesAsync(ArtifactKind.TranslationRevision, sharedPath);
        // A path referenced by durable metadata is protected, including legacy Windows separators.
        await f.Media.SaveArtifactAsync(shared.Artifact with { RelativePath = sharedPath.Replace('/', '\\') }, f.Ct);
        if (preexistingFile) await f.Store.CommitNewAsync(shared.Handle, shared.Receipt, f.Ct);
        var before = await f.OutputSnapshotAsync();
        PreparedCommitArtifact? attempt = null;
        f.Persistence.CommitTransactionAsync = (_, _) => throw new IOException("injected commit failure");
        await Assert.ThrowsAsync<IOException>(() => f.Boundary.CommitRevisionAsync(f.NewRevisionRequest(),
            async (_, _) => attempt = await f.PrepareBytesAsync(ArtifactKind.TranslationRevision, sharedPath), f.Ct));
        Assert.Equal(!preexistingFile, attempt!.Receipt.CreatedByAttempt);
        Assert.True(File.Exists(attempt.Handle.FinalPath));
        Assert.Null(await f.Translations.GetCurrentRevisionAsync(f.Project.Id, "es", f.Ct));
        Assert.Equal(shared.Artifact.Id, Assert.Single(await f.Media.GetArtifactsAsync(f.Project.Id, f.Ct)).Id);
        if (preexistingFile) Assert.Equal(before, await f.OutputSnapshotAsync());
    }

    [Fact]
    public async Task Promotion_failure_rolls_back_metadata_and_cleans_temporary_content()
    {
        await using var f = await Fixture.CreateAsync();
        PreparedCommitArtifact? prepared = null;
        f.Persistence.BeforePublicationAsync = _ => { Directory.CreateDirectory(prepared!.Handle.FinalPath); return Task.CompletedTask; };
        await Assert.ThrowsAsync<IOException>(() => f.Boundary.CommitRevisionAsync(f.NewRevisionRequest(),
            async (r, ct) => prepared = await f.PrepareRevisionAsync(r, ct), f.Ct));
        Assert.False(prepared!.Receipt.CreatedByAttempt);
        Assert.False(File.Exists(prepared.Handle.TemporaryPath));
        Assert.Null(await f.Translations.GetCurrentRevisionAsync(f.Project.Id, "es", f.Ct));
        Assert.Empty(await f.Media.GetArtifactsAsync(f.Project.Id, f.Ct));
    }

    [Fact]
    public async Task Stale_marking_preserves_unchanged_and_other_language_takes_and_handles_removed_added_timing_and_hash_changes()
    {
        await using var f = await Fixture.CreateAsync();
        await f.SeedTranslationAsync(segmentCount: 7);
        var alreadyStale = (await f.Takes.GetByProjectAsync(f.Project.Id, f.Ct)).Single(t => t.SegmentIndex == 6);
        await f.Takes.SaveAsync(alreadyStale with { IsStale = true, Status = TtsTakeStatus.Stale }, f.Ct);
        var unlinked = TtsTake.CreateStock(f.Project.Id, f.Assignment.Id, segmentIndex: 1);
        await f.Takes.SaveAsync(unlinked, f.Ct);
        var original = (await f.Takes.GetByProjectAsync(f.Project.Id, f.Ct)).Where(t => t.TranslatedSegmentId is not null).ToArray();
        var other = await f.SeedTranslationAsync("fr", segmentCount: 7);
        var otherTakes = (await f.Takes.GetByProjectAsync(f.Project.Id, f.Ct)).Where(t => t.Id != unlinked.Id && !original.Any(o => o.Id == t.Id)).ToArray();
        var old = await f.Translations.GetCurrentRevisionAsync(f.Project.Id, "es", f.Ct);
        var oldSegments = await f.Translations.GetSegmentsAsync(old!.Id, f.Ct);
        var revision = TranslationRevision.Create(f.Project.Id, null, f.Transcript.Id, "es", 1, DateTimeOffset.UtcNow);
        var next = oldSegments.Where(s => s.SegmentIndex != 5).Select(s => s with { Id = Guid.NewGuid(), TranslationRevisionId = revision.Id }).ToList();
        next[1] = next[1] with { Text = "changed" };
        next[2] = next[2] with { StartSeconds = 2.1 };
        next[3] = next[3] with { SourceSegmentHash = "new-hash" };
        next[4] = next[4] with { SourceSegmentHash = null };
        next.Add(TranslatedSegment.Create(revision.Id, 7, 7, 8, "added"));
        await f.Boundary.CommitRevisionAsync(new(revision, old.Id, next), (r, ct) => f.PrepareRevisionAsync(r, ct, next), f.Ct);
        foreach (var take in original)
        {
            var persisted = await f.Takes.GetAsync(take.Id, f.Ct);
            if (take.SegmentIndex is 0 or 6) Assert.Equal(take, persisted);
            else Assert.True(persisted!.IsStale);
            Assert.True(File.Exists(f.Store.GetPath((await f.Media.GetArtifactsAsync(f.Project.Id, f.Ct)).Single(a => a.Id == take.ArtifactId).RelativePath)));
        }
        foreach (var take in otherTakes) Assert.Equal(take, await f.Takes.GetAsync(take.Id, f.Ct));
        Assert.Equal(unlinked, await f.Takes.GetAsync(unlinked.Id, f.Ct));
        Assert.NotNull(other);
    }

    [Fact]
    public async Task Unchanged_take_with_historical_segment_link_remains_playable_after_revision_publication()
    {
        await using var f = await Fixture.CreateAsync();
        var old = await f.SeedTranslationAsync();
        var oldSegments = await f.Translations.GetSegmentsAsync(old.Id, f.Ct);
        var prior = (await f.Takes.GetByProjectAsync(f.Project.Id, f.Ct)).Single(t => t.SegmentIndex == 0);
        var revision = TranslationRevision.Create(f.Project.Id, null, f.Transcript.Id, "es", 1, DateTimeOffset.UtcNow);
        var next = oldSegments.Select(s => s with { Id = Guid.NewGuid(), TranslationRevisionId = revision.Id }).ToArray();
        await f.Boundary.CommitRevisionAsync(new(revision, old.Id, next), (r, ct) => f.PrepareRevisionAsync(r, ct, next), f.Ct);
        await using var sourceAudio = await f.PrepareBytesAsync(ArtifactKind.NormalizedAudio);
        await f.Store.CommitNewAsync(sourceAudio.Handle, sourceAudio.Receipt, f.Ct);
        await f.Media.SaveArtifactAsync(sourceAudio.Artifact, f.Ct);
        var plan = new MixPlanBuilder(f.Store).Build(new(f.Project.Id, f.Asset.Id,
            await f.Media.GetArtifactsAsync(f.Project.Id, f.Ct), f.SourceSegments, next, await f.Takes.GetByProjectAsync(f.Project.Id, f.Ct), StageRuns: null));
        var clip = Assert.Single(plan.SpeechClips, c => c.SegmentIndex == 0);
        Assert.Equal(prior.Id, clip.TakeId);
        Assert.Equal(prior.ArtifactId, clip.ArtifactId);
        Assert.False(clip.IsSilentGap);
        Assert.Equal(prior, await f.Takes.GetAsync(prior.Id, f.Ct));
    }

    [Theory]
    [InlineData("success")]
    [InlineData("uncommitted")]
    [InlineData("wrong-content")]
    [InlineData("source-changed")]
    [InlineData("translation-changed")]
    [InlineData("sql-failure")]
    public async Task Take_boundary_guards_exact_persisted_input_and_publishes_artifact_and_take_together(string mode)
    {
        await using var f = await Fixture.CreateAsync();
        var revision = await f.SeedTranslationAsync();
        var segment = (await f.Translations.GetSegmentsAsync(revision.Id, f.Ct))[0];
        if (mode == "uncommitted") segment = segment with { TranslationRevisionId = Guid.NewGuid() };
        if (mode == "wrong-content") segment = segment with { Text = "not persisted" };
        var input = new AtomicTakeInput(f.Project.Id, "es", f.Transcript.Id, segment.TranslationRevisionId, segment);
        if (mode is "success" or "source-changed" or "translation-changed") await f.Boundary.ValidateTakeInputAsync(input, f.Ct);
        if (mode == "source-changed") await f.ReplaceSourceAsync();
        if (mode == "translation-changed") await f.SeedTranslationAsync();
        var before = await f.OutputSnapshotAsync();
        if (mode == "sql-failure")
        {
            await using var connection = await f.Database.OpenConnectionAsync(f.Ct);
            await using var trigger = connection.CreateCommand();
            trigger.CommandText = "CREATE TRIGGER reject_take BEFORE INSERT ON tts_takes BEGIN SELECT RAISE(ABORT, 'injected take failure'); END;";
            await trigger.ExecuteNonQueryAsync(f.Ct);
        }
        var prepared = await f.PrepareBytesAsync(ArtifactKind.TtsTake);
        var take = TtsTake.CreateStock(f.Project.Id, f.Assignment.Id, segment.Id, segment.SegmentIndex,
            TtsTextHash.Compute(segment.SegmentIndex, segment.Text)).Complete(prepared.Artifact.Id, null, 24000, 24000, "fake", "fake", "af_heart", null);
        var request = new AtomicTakeCommitRequest(input, take);
        if (mode == "success")
        {
            await f.Boundary.CommitTakeAsync(request, prepared, f.Ct);
            Assert.NotNull(await f.Takes.GetAsync(take.Id, f.Ct));
            Assert.True(File.Exists(prepared.Handle.FinalPath));
        }
        else
        {
            if (mode == "sql-failure") await Assert.ThrowsAsync<SqliteException>(() => f.Boundary.CommitTakeAsync(request, prepared, f.Ct));
            else await Assert.ThrowsAsync<InvalidDataException>(() => f.Boundary.CommitTakeAsync(request, prepared, f.Ct));
            Assert.Equal(before, await f.OutputSnapshotAsync());
            Assert.False(File.Exists(prepared.Handle.FinalPath));
            Assert.False(File.Exists(prepared.Handle.TemporaryPath));
        }
    }

    [Theory]
    [InlineData("commit-failure")]
    [InlineData("commit-ack-lost")]
    [InlineData("cancel-after-promotion")]
    [InlineData("pending-take")]
    public async Task Take_cleanup_and_commit_ambiguity_follow_durable_take_and_artifact_references(string mode)
    {
        await using var f = await Fixture.CreateAsync();
        var revision = await f.SeedTranslationAsync();
        var segment = (await f.Translations.GetSegmentsAsync(revision.Id, f.Ct))[0];
        var before = await f.OutputSnapshotAsync();
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(f.Ct);
        var prepared = await f.PrepareBytesAsync(ArtifactKind.TtsTake);
        var take = TtsTake.CreateStock(f.Project.Id, f.Assignment.Id, segment.Id, 0, TtsTextHash.Compute(0, segment.Text))
            .Complete(prepared.Artifact.Id, null, 24000, 24000, "test", "test", "ef_dora", null);
        if (mode == "pending-take") take = take with { Status = TtsTakeStatus.Pending };
        f.Persistence.CommitTransactionAsync = async (tx, ct) =>
        {
            if (mode == "cancel-after-promotion") { cts.Cancel(); ct.ThrowIfCancellationRequested(); }
            if (mode == "commit-failure") throw new IOException("commit failed");
            await tx.CommitAsync(ct);
            cts.Cancel();
            throw new IOException("acknowledgement lost after durable take commit");
        };
        Task operation = f.Boundary.CommitTakeAsync(new(new(f.Project.Id, "es", f.Transcript.Id, revision.Id, segment), take), prepared, cts.Token);
        if (mode == "commit-ack-lost")
        {
            await operation;
            Assert.Equal(take, await f.Takes.GetAsync(take.Id, f.Ct));
            Assert.Contains(await f.Media.GetArtifactsAsync(f.Project.Id, f.Ct), a => a.Id == prepared.Artifact.Id);
            Assert.True(File.Exists(prepared.Handle.FinalPath));
        }
        else
        {
            await Assert.ThrowsAnyAsync<Exception>(() => operation);
            Assert.Equal(before, await f.OutputSnapshotAsync());
            Assert.False(File.Exists(prepared.Handle.FinalPath));
        }
        Assert.False(File.Exists(prepared.Handle.TemporaryPath));
    }

    [Theory]
    [InlineData("success")]
    [InlineData("source-changed")]
    [InlineData("translation-changed")]
    [InlineData("canceled")]
    public async Task Production_tts_path_rechecks_revision_after_inference_before_publication(string mode)
    {
        await using var f = await Fixture.CreateAsync();
        var revision = await f.SeedTranslationAsync();
        var segment = (await f.Translations.GetSegmentsAsync(revision.Id, f.Ct))[0];
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(f.Ct);
        string? inferenceOutput = null;
        var engine = new TtsEngine(async () =>
        {
            if (mode == "source-changed") await f.ReplaceSourceAsync();
            if (mode == "translation-changed") await f.SeedTranslationAsync();
            if (mode == "canceled") cts.Cancel();
            inferenceOutput = await f.OutputSnapshotAsync();
        });
        var recorder = new RecordingBoundary(f.Boundary);
        using var handler = new StartTtsStageHandler(engine,
            new FakeVoiceCatalog([new("ef_dora", "es", "female", "Dora")]), f.Store, f.Fingerprints, f.Media,
            f.Takes, new SqliteProjectStageRunStore(f.Database), commitBoundary: recorder);
        var sourceSnapshot = f.SourceSegments.ToArray();
        var request = new StartTtsStageRequest(f.Project.Id, f.Asset, f.Assignment.SpeakerId, "es", f.Assignment,
            sourceSnapshot, [segment]);
        if (mode == "success")
        {
            var result = await handler.HandleAsync(request, cts.Token);
            var take = Assert.Single(result.Takes);
            Assert.Equal(StageRunStatus.Completed, result.StageRun.Status);
            Assert.Equal(take, await f.Takes.GetAsync(take.Id, f.Ct));
            var artifact = (await f.Media.GetArtifactsAsync(f.Project.Id, f.Ct)).Single(a => a.Id == take.ArtifactId);
            Assert.Contains(take.Id.ToString("N"), artifact.RelativePath);
            Assert.True(File.Exists(f.Store.GetPath(artifact.RelativePath)));
        }
        else
        {
            await Assert.ThrowsAnyAsync<Exception>(() => handler.HandleAsync(request, cts.Token));
            Assert.Equal(inferenceOutput, await f.OutputSnapshotAsync());
            Assert.Equal(mode == "canceled" ? 0 : 1, recorder.TakeRequests.Count);
        }
        Assert.Equal(1, engine.Calls);
        Assert.Empty(Directory.EnumerateFiles(Path.Join(f.Root, "temp")));
    }

    private sealed class TtsEngine(Func<Task> duringInference) : ITtsEngine
    {
        public int Calls { get; private set; }
        public async Task<TtsSynthesisResult> SynthesizeAsync(TtsSynthesisRequest request, CancellationToken ct)
        {
            Calls++;
            await duringInference();
            ct.ThrowIfCancellationRequested();
            return new([1, 2, 3], 24000, 24000, "test", request.Voice.VoiceId, "test");
        }
    }

    private sealed class RecordingBoundary(IAtomicRevisionCommitBoundary inner) : IAtomicRevisionCommitBoundary
    {
        public List<AtomicRevisionCommitRequest> Requests { get; } = [];
        public List<AtomicTakeCommitRequest> TakeRequests { get; } = [];
        public Task<TranslationRevision> CommitRevisionAsync(AtomicRevisionCommitRequest request,
            Func<TranslationRevision, CancellationToken, Task<PreparedCommitArtifact>> prepareArtifact, CancellationToken ct)
        { Requests.Add(request); return inner.CommitRevisionAsync(request, prepareArtifact, ct); }
        public Task ValidateTakeInputAsync(AtomicTakeInput input, CancellationToken ct) => inner.ValidateTakeInputAsync(input, ct);
        public Task CommitTakeAsync(AtomicTakeCommitRequest request, PreparedCommitArtifact artifact, CancellationToken ct)
        { TakeRequests.Add(request); return inner.CommitTakeAsync(request, artifact, ct); }
    }

    private sealed class StreamEngine : IStreamingTranslationEngine
    {
        public int? WrongRevisionAt { get; init; }
        public int? EmptyTextAt { get; init; }
        public int Limit { get; init; } = 3;
        public int? FaultAfter { get; init; }
        public Func<Task>? BeforeLast { get; set; }
        public int Emitted { get; private set; }
        public Task<IReadOnlyList<TranslatedTextSegment>> TranslateAsync(TranslationRequest request, CancellationToken ct) => throw new NotSupportedException();
        public async IAsyncEnumerable<PipelineStreamItem<TranslatedTextSegment>> TranslateStreamAsync(TranslationRequest request,
            Guid runId, string snapshotId, Guid sourceRevisionId, [EnumeratorCancellation] CancellationToken ct)
        {
            foreach (var segment in request.Segments.Take(Limit))
            {
                ct.ThrowIfCancellationRequested();
                if (Emitted == FaultAfter) throw new IOException("stream fault");
                if (Emitted == 2 && BeforeLast is { } before) await before();
                var payload = new TranslatedTextSegment(segment.Index, segment.StartSeconds, segment.EndSeconds,
                    Emitted == EmptyTextAt ? string.Empty : "Hola " + segment.Index);
                var item = PipelineStreamItemFactory.CreateTranslation(payload, runId, snapshotId, sourceRevisionId, Emitted);
                if (Emitted == WrongRevisionAt) item = item with { Identity = item.Identity with { RevisionId = Guid.NewGuid() } };
                Emitted++;
                yield return item;
            }
        }
    }

    private sealed class Fixture : IAsyncDisposable
    {
        public CancellationToken Ct { get; } = TestContext.Current.CancellationToken;
        public string Root { get; } = Path.Join(Path.GetTempPath(), "Trackdub.Infrastructure.Tests", "atomic-" + Guid.NewGuid().ToString("N"));
        public SqliteProjectDatabase Database { get; }
        public FileSystemArtifactStore Store { get; }
        public Sha256FileFingerprintService Fingerprints { get; } = new();
        public SqliteTranslationRepository Translations { get; }
        public SqliteTranscriptRepository Transcripts { get; }
        public SqliteTtsTakeRepository Takes { get; }
        public SqliteMediaAssetRepository Media { get; }
        public SqliteAtomicRevisionPersistence Persistence { get; }
        public IAtomicRevisionCommitBoundary Boundary { get; }
        public TrackdubProject Project { get; } = new(Guid.NewGuid(), "Atomic", DateTimeOffset.UtcNow, DateTimeOffset.UtcNow);
        public TranscriptRevision Transcript { get; private set; } = null!;
        public TranscriptSegment[] SourceSegments { get; private set; } = [];
        public MediaAsset Asset { get; private set; } = null!;
        public VoiceAssignment Assignment { get; private set; } = null!;
        private Fixture()
        {
            Database = new(Root); Store = new(Root); Translations = new(Database); Transcripts = new(Database);
            Takes = new(Database); Media = new(Database); Persistence = new(Database, Store); Boundary = new AtomicRevisionCommitBoundary(Persistence);
        }
        public static async Task<Fixture> CreateAsync()
        {
            var f = new Fixture();
            await new SqliteProjectRepository(f.Database).InitializeAsync(f.Project, f.Ct);
            await f.Store.EnsureLayoutAsync(f.Ct);
            var speaker = await new SqliteSpeakerRepository(f.Database).EnsureDefaultSpeakerAsync(f.Project.Id, f.Ct);
            f.Assignment = VoiceAssignment.Create(f.Project.Id, speaker.Id, "kokoro-onnx", "ef_dora");
            await new SqliteVoiceAssignmentRepository(f.Database).SaveAsync(f.Assignment, f.Ct);
            f.Asset = new(Guid.NewGuid(), f.Project.Id, "source.mp4", "source.mp4", "source", 100, DateTimeOffset.UtcNow, "mp4", 3, true, true, DateTimeOffset.UtcNow);
            await f.Media.SaveAsync(f.Asset, f.Ct);
            await f.ReplaceSourceAsync();
            return f;
        }
        public async Task ReplaceSourceAsync()
        {
            Transcript = TranscriptRevision.Create(Project.Id, null, await Transcripts.GetNextRevisionNumberAsync(Project.Id, Ct), DateTimeOffset.UtcNow);
            SourceSegments = Enumerable.Range(0, 3).Select(i => TranscriptSegment.Create(Transcript.Id, i, i, i + 1, "Hello " + i, Assignment.SpeakerId)).ToArray();
            await Transcripts.SaveRevisionAsync(Transcript, SourceSegments, Ct);
        }
        public AtomicRevisionCommitRequest NewRevisionRequest()
        {
            var revision = TranslationRevision.Create(Project.Id, null, Transcript.Id, "es", 1, DateTimeOffset.UtcNow);
            return new(revision, null, SourceSegments.Select(s => TranslatedSegment.Create(revision.Id, s.SegmentIndex, s.StartSeconds, s.EndSeconds, "Hola " + s.SegmentIndex, "hash-" + s.SegmentIndex)).ToArray());
        }
        public Task<TranslationRevision> CommitRevisionAsync()
        {
            var request = NewRevisionRequest();
            return Boundary.CommitRevisionAsync(request, (r, ct) => PrepareRevisionAsync(r, ct, request.Segments), Ct);
        }
        public Task<PreparedCommitArtifact> PrepareRevisionAsync(TranslationRevision revision, CancellationToken ct) =>
            PrepareRevisionAsync(revision, ct, NewRevisionRequest().Segments.Select(s => s with { TranslationRevisionId = revision.Id }).ToArray());
        public Task<PreparedCommitArtifact> PrepareRevisionAsync(TranslationRevision revision, CancellationToken ct, IReadOnlyList<TranslatedSegment> segments) =>
            new TranscriptArtifactWriter(Store, Fingerprints, Media).PrepareTranslationArtifactAsync(Project.Id, Asset, revision, segments, null, "generated-translation", ct);
        public async Task<PreparedCommitArtifact> PrepareBytesAsync(ArtifactKind kind, string? relativePath = null)
        {
            string path = relativePath ?? $"artifacts/tts/{Guid.NewGuid():N}.wav";
            var handle = Store.CreateWriteHandle(path);
            await File.WriteAllBytesAsync(handle.TemporaryPath, [1, 2, 3], Ct);
            var fingerprint = await Fingerprints.ComputeAsync(handle.TemporaryPath, Ct);
            return new(new(Guid.NewGuid(), Project.Id, Asset.Id, kind, path, fingerprint.Sha256, fingerprint.SizeBytes, 1, 24000, 1, DateTimeOffset.UtcNow, null, "test"), handle);
        }
        public async Task<TranslationRevision> SeedTranslationAsync(string language = "es", int segmentCount = 3)
        {
            var revision = TranslationRevision.Create(Project.Id, null, Transcript.Id, language, await Translations.GetNextRevisionNumberAsync(Project.Id, language, Ct), DateTimeOffset.UtcNow);
            var segments = Enumerable.Range(0, segmentCount).Select(i => TranslatedSegment.Create(revision.Id, i, i, i + 1, "Hola " + i, "hash-" + i)).ToArray();
            await Translations.SaveRevisionAsync(revision, segments, Ct);
            foreach (var segment in segments)
            {
                await using var prepared = await PrepareBytesAsync(ArtifactKind.TtsTake);
                await Store.CommitNewAsync(prepared.Handle, prepared.Receipt, Ct);
                await Media.SaveArtifactAsync(prepared.Artifact, Ct);
                var take = TtsTake.CreateStock(Project.Id, Assignment.Id, segment.Id, segment.SegmentIndex, TtsTextHash.Compute(segment.SegmentIndex, segment.Text))
                    .Complete(prepared.Artifact.Id, null, 24000, 24000, "test", "test", "test", null) with
                { PreStretchDurationSeconds = 1.2, StretchRatioApplied = 1.2 };
                await Takes.SaveAsync(take, Ct);
            }
            return revision;
        }
        public async Task<string> OutputSnapshotAsync()
        {
            await using var connection = await Database.OpenConnectionAsync(Ct);
            var values = new List<string>();
            foreach (string table in new[] { "translation_revisions", "translated_segments", "artifacts", "tts_takes" })
            {
                await using var command = connection.CreateCommand(); command.CommandText = $"SELECT * FROM {table} ORDER BY id;";
                await using var reader = await command.ExecuteReaderAsync(Ct);
                while (await reader.ReadAsync(Ct)) values.Add(table + ":" + string.Join("|", Enumerable.Range(0, reader.FieldCount).Select(reader.GetValue)));
            }
            foreach (string path in Directory.EnumerateFiles(Path.Join(Root, "artifacts"), "*", SearchOption.AllDirectories).Order())
                values.Add(path + ":" + (await Fingerprints.ComputeAsync(path, Ct)).Sha256);
            return string.Join("\n", values);
        }
        public Task GenerateAsync(ITranslationEngine engine, IAtomicRevisionCommitBoundary boundary, CancellationToken? cancellationToken = null,
            TranslationRunScope scope = TranslationRunScope.All)
        {
            var state = new TranscriptProjectState(new OpenProjectResult(Project, Asset, null, SourceMediaStatus.Available, null, [], "en"),
                Transcript, SourceSegments, [], [], null, [], false, "en", [], [], "es", new HashSet<int>(), null, [], [], [], [], []);
            var service = new TranslationOrchestrationService(Translations, new GlossaryService(new SqliteGlossaryRepository(Database)),
                new GlossaryTermMatcher(), new FakeTranslationLanguageRouter(), engine, Takes, new SqliteProjectStageRunStore(Database),
                Store, new TranscriptArtifactWriter(Store, Fingerprints, Media),
                degradationWriter: new PipelineDegradationWriter(Store, Fingerprints, Media), commitBoundary: boundary);
            return service.GenerateTranslationAsync(state, new("auto", "es", EnableSegmentStreaming: true, Scope: scope), cancellationToken ?? Ct);
        }
        public ValueTask DisposeAsync()
        {
            string path = Path.GetFullPath(Root);
            string expectedBase = Path.Join(Path.GetTempPath(), "Trackdub.Infrastructure.Tests") + Path.DirectorySeparatorChar;
            if (!path.StartsWith(expectedBase, StringComparison.OrdinalIgnoreCase)) throw new InvalidOperationException("Unexpected fixture cleanup path.");
            if (Directory.Exists(path)) Directory.Delete(path, recursive: true);
            return ValueTask.CompletedTask;
        }
    }
}
