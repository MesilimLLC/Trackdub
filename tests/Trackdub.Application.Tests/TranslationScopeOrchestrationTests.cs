using Trackdub.Contracts;
using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Projects;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Trackdub.Domain.Tts;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed partial class OrchestrationServiceTests
{
    public static TheoryData<TranslationScopeScenario> TranslationScopeScenarios => new()
    {
        new(new TranslationScopeScenario(2, TranslationSegmentOrigin.Manual)),
        new(new TranslationScopeScenario(1, TranslationSegmentOrigin.Unknown)),
    };

    public sealed record TranslationScopeScenario(
        int SeededSegmentCount,
        TranslationSegmentOrigin PreservedOrigin);

    [Theory]
    [MemberData(nameof(TranslationScopeScenarios))]
    public async Task Scoped_translation_preserves_existing_text_and_updates_or_adds_missing_text(
        TranslationScopeScenario scenario)
    {
        var engine = new FakeTranslationEngine((_, segment) => $"new:{segment.Index}");
        TranslationServiceContext context = CreateTranslationServiceContext(engine);
        TranscriptProjectState state = CreateTranslatedProjectStateWithWords();
        TranslatedSegment[] previous = state.TranslatedSegments
            .Take(scenario.SeededSegmentCount)
            .Select((segment, index) => segment with
            {
                Origin = index == 0 ? scenario.PreservedOrigin : TranslationSegmentOrigin.Generated,
            })
            .ToArray();
        context.TranslationRepository.Seed(state.CurrentTranslationRevision!, previous);

        await context.Service.GenerateTranslationAsync(state,
            new GenerateTranslationRequest("en", "es", Scope: TranslationRunScope.MissingOrOutdated),
            TestContext.Current.CancellationToken);

        TranslationRevision revision = Assert.Single(context.TranslationRepository.Revisions, item => item.RevisionNumber == 2);
        IReadOnlyList<TranslatedSegment> result = await context.TranslationRepository.GetSegmentsAsync(revision.Id, TestContext.Current.CancellationToken);
        Assert.Collection(result.OrderBy(segment => segment.SegmentIndex),
            preserved =>
            {
                Assert.Equal(previous[0].Text, preserved.Text);
                Assert.Equal(scenario.PreservedOrigin, preserved.Origin);
                Assert.Equal(2, preserved.Words.Count);
            },
            generated =>
            {
                Assert.Equal("new:1", generated.Text);
                Assert.Equal(TranslationSegmentOrigin.Generated, generated.Origin);
            });
    }

    [Fact]
    public async Task Scoped_translation_preserves_revision_when_engine_returns_empty_text()
    {
        var engine = new FakeTranslationEngine((_, _) => "");
        TranslationServiceContext context = CreateTranslationServiceContext(engine);
        TranscriptProjectState state = CreateTranslatedProjectStateWithWords();
        TranslatedSegment[] previous = state.TranslatedSegments
            .Select(segment => segment with { Origin = TranslationSegmentOrigin.Generated }).ToArray();
        context.TranslationRepository.Seed(state.CurrentTranslationRevision!, previous);

        await Assert.ThrowsAsync<InvalidDataException>(() => context.Service.GenerateTranslationAsync(state,
            new GenerateTranslationRequest("en", "es", Scope: TranslationRunScope.MissingOrOutdated),
            TestContext.Current.CancellationToken));

        Assert.Single(context.TranslationRepository.Revisions);
        IReadOnlyList<TranslatedSegment> result = await context.TranslationRepository.GetSegmentsAsync(state.CurrentTranslationRevision!.Id, TestContext.Current.CancellationToken);
        Assert.Equal(previous.Select(segment => segment.Text), result.Select(segment => segment.Text));
    }

    [Fact]
    public async Task Scoped_translation_skips_when_only_legacy_text_is_outdated()
    {
        var engine = new FakeTranslationEngine((_, segment) => $"new:{segment.Index}");
        TranslationServiceContext context = CreateTranslationServiceContext(engine);
        TranscriptProjectState state = CreateTranslatedProjectStateWithWords();
        context.TranslationRepository.Seed(state.CurrentTranslationRevision!, state.TranslatedSegments);

        await context.Service.GenerateTranslationAsync(state,
            new GenerateTranslationRequest("en", "es", Scope: TranslationRunScope.MissingOrOutdated),
            TestContext.Current.CancellationToken);

        Assert.Single(context.TranslationRepository.Revisions);
        StageRunRecord skippedRun = Assert.Single(context.StageRunStore.All);
        Assert.Equal(StageNames.Translation, skippedRun.StageName);
        Assert.Equal(StageRunStatus.Skipped, skippedRun.Status);
        Assert.Equal(StageSkipReasonCodes.ExistingArtifactsValid, skippedRun.FailureReason);
    }

    [Fact]
    public async Task Scoped_translation_keeps_generated_segment_whose_source_hash_is_current()
    {
        var engine = new FakeTranslationEngine((_, segment) => $"new:{segment.Index}");
        TranslationServiceContext context = CreateTranslationServiceContext(engine);
        TranscriptProjectState state = CreateTranslatedProjectStateWithWords();
        string currentHash = TranscriptWorkflowUtilities.ComputeSourceSegmentHash(state.TranscriptSegments[0]);
        TranslatedSegment upToDate = state.TranslatedSegments[0] with
        {
            Origin = TranslationSegmentOrigin.Generated,
            SourceSegmentHash = currentHash,
        };
        TranslatedSegment outdated = state.TranslatedSegments[1] with
        {
            Origin = TranslationSegmentOrigin.Generated,
            SourceSegmentHash = "stale",
        };
        context.TranslationRepository.Seed(state.CurrentTranslationRevision!, [upToDate, outdated]);

        await context.Service.GenerateTranslationAsync(state,
            new GenerateTranslationRequest("en", "es", Scope: TranslationRunScope.MissingOrOutdated),
            TestContext.Current.CancellationToken);

        TranslationRevision revision = Assert.Single(context.TranslationRepository.Revisions, item => item.RevisionNumber == 2);
        IReadOnlyList<TranslatedSegment> result = await context.TranslationRepository
            .GetSegmentsAsync(revision.Id, TestContext.Current.CancellationToken);
        Assert.Collection(
            result.OrderBy(static segment => segment.SegmentIndex),
            kept =>
            {
                Assert.Equal(upToDate.Text, kept.Text);
                Assert.Equal(TranslationSegmentOrigin.Generated, kept.Origin);
                Assert.Equal(currentHash, kept.SourceSegmentHash);
            },
            regenerated =>
            {
                Assert.Equal("new:1", regenerated.Text);
                Assert.Equal(TranslationSegmentOrigin.Generated, regenerated.Origin);
            });
    }

    [Fact]
    public async Task SaveTranslationEdits_marks_changed_segments_manual_and_preserves_other_origins()
    {
        TranslationServiceContext context = CreateTranslationServiceContext(new FakeTranslationEngine());
        TranscriptProjectState currentState = CreateTranslatedProjectStateWithWords();

        context.TranslationRepository.Seed(currentState.CurrentTranslationRevision!, currentState.TranslatedSegments);

        await context.Service.SaveTranslationEditsAsync(
            currentState,
            new SaveTranslationEditsRequest(
                currentState.CurrentTranslationRevision!.Id,
                "es",
                [new EditedTranslatedSegment(0, "Linea editada.")]),
            TestContext.Current.CancellationToken);

        TranslationRevision savedRevision = Assert.Single(context.TranslationRepository.Revisions, revision => revision.RevisionNumber == 2);
        IReadOnlyList<TranslatedSegment> savedSegments = await context.TranslationRepository
            .GetSegmentsAsync(savedRevision.Id, TestContext.Current.CancellationToken);

        Assert.Collection(
            savedSegments.OrderBy(static segment => segment.SegmentIndex),
            changedSegment =>
            {
                Assert.Equal("Linea editada.", changedSegment.Text);
                Assert.Equal(TranslationSegmentOrigin.Manual, changedSegment.Origin);
                Assert.Empty(changedSegment.Words);
            },
            unchangedSegment =>
            {
                Assert.Equal("Segunda linea.", unchangedSegment.Text);
                Assert.Equal(TranslationSegmentOrigin.Unknown, unchangedSegment.Origin);
                Assert.Collection(
                    unchangedSegment.Words,
                    first => Assert.Equal("Segunda", first.Text),
                    second => Assert.Equal("linea", second.Text));
            });
    }
}
