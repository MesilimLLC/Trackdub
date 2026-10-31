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
    public enum ScopedTranslationOutcome
    {
        Success,
        Canceled,
        TranslationChanged,
    }

    public static TheoryData<ScopedTranslationOutcome> ScopedTranslationOutcomes => new()
    {
        new(ScopedTranslationOutcome.Success),
        new(ScopedTranslationOutcome.Canceled),
        new(ScopedTranslationOutcome.TranslationChanged),
    };

    [Theory]
    [MemberData(nameof(ScopedTranslationOutcomes))]
    public async Task Scoped_translation_uses_real_atomic_publication_and_preserves_concurrent_output(
        ScopedTranslationOutcome outcome)
    {
        await using var fixture = await Fixture.CreateAsync();
        TranslationRevision previous = await fixture.SeedTranslationAsync();
        await using (var connection = await fixture.Database.OpenConnectionAsync(fixture.Ct))
        await using (var command = connection.CreateCommand())
        {
            command.CommandText = "UPDATE translated_segments SET origin = 1 WHERE translation_revision_id = $revision";
            command.Parameters.AddWithValue("$revision", previous.Id.ToString("D"));
            await command.ExecuteNonQueryAsync(fixture.Ct);
        }

        string? expected = await fixture.OutputSnapshotAsync();
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(fixture.Ct);
        var engine = new StreamEngine();
        if (outcome == ScopedTranslationOutcome.Canceled)
        {
            engine.BeforeLast = () =>
            {
                cancellation.Cancel();
                return Task.CompletedTask;
            };
        }
        else if (outcome == ScopedTranslationOutcome.TranslationChanged)
        {
            engine.BeforeLast = async () =>
            {
                await fixture.SeedTranslationAsync();
                expected = await fixture.OutputSnapshotAsync();
            };
        }

        if (outcome == ScopedTranslationOutcome.Success)
        {
            await fixture.GenerateAsync(engine, fixture.Boundary, cancellation.Token, TranslationRunScope.MissingOrOutdated);
            TranslationRevision revision = (await fixture.Translations.GetCurrentRevisionAsync(fixture.Project.Id, "es", fixture.Ct))!;
            Assert.NotEqual(previous.Id, revision.Id);
            IReadOnlyList<TranslatedSegment> segments = await fixture.Translations.GetSegmentsAsync(revision.Id, fixture.Ct);
            Assert.Equal(3, segments.Count);
            Assert.All(segments, segment => Assert.Equal(TranslationSegmentOrigin.Generated, segment.Origin));
        }
        else
        {
            await Assert.ThrowsAnyAsync<Exception>(() => fixture.GenerateAsync(
                engine,
                fixture.Boundary,
                cancellation.Token,
                TranslationRunScope.MissingOrOutdated));
            Assert.Equal(expected, await fixture.OutputSnapshotAsync());
        }

        Assert.Empty(Directory.EnumerateFiles(Path.Join(fixture.Root, "temp")));
    }
}
