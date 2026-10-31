using Trackdub.Contracts;
using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.Projects;
using Trackdub.Domain;
using Trackdub.Domain.Media;
using Trackdub.Domain.Projects;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed class TranslationOrchestrationServiceTests
{
    [Fact]
    [Trait("Bug", "B-2")]
    public async Task RetranslateSegmentAsync_WhenEngineThrowsOce_MarksStageAsCanceled_NotFailed()
    {
        // Arrange
        var stageRunStore = new FakeProjectStageRunStore();
        var translationRepository = new FakeTranslationRepository();
        var glossaryRepository = new FakeGlossaryRepository();
        var ttsTakeRepository = new FakeTtsTakeRepository();
        var artifactStore = new FakeArtifactStore();
        var fileFingerprintService = new FakeFileFingerprintService();
        var mediaAssetRepository = new FakeMediaAssetRepository();
        var artifactWriter = new TranscriptArtifactWriter(
            artifactStore,
            fileFingerprintService,
            mediaAssetRepository);

        // The engine cancels the caller's token, then throws — a real user
        // cancellation arriving mid-call.
        using var cts = new CancellationTokenSource();
        var cancellingEngine = new CancellingTranslationEngine(cts);

        var service = new TranslationOrchestrationService(
            translationRepository,
            new GlossaryService(glossaryRepository),
            new GlossaryTermMatcher(),
            new FakeTranslationLanguageRouter(),
            cancellingEngine,
            ttsTakeRepository,
            stageRunStore,
            artifactStore,
            artifactWriter,
            commitBoundary: TestAtomicCommitBoundary.Create(translationRepository, ttsTakeRepository, artifactStore, mediaAssetRepository));

        TranscriptProjectState state = BuildStateWithTranslationRevision();

        RetranslateSegmentRequest request = new(
            state.CurrentTranslationRevision!.Id,
            state.TranscriptSegments[0].Id,
            SourceLanguage: "en",
            TargetLanguage: "es");

        // Act
        OperationCanceledException? caught = null;
        try
        {
            await service.RetranslateSegmentAsync(
                state,
                request,
                cts.Token);
        }
        catch (OperationCanceledException ex)
        {
            caught = ex;
        }

        // Assert: the exception must propagate as OperationCanceledException
        Assert.NotNull(caught);

        // Assert: the stage run must be marked Canceled, not Failed
        StageRunRecord stageRun = Assert.Single(stageRunStore.All);
        Assert.Equal(StageNames.Translation, stageRun.StageName);
        Assert.Equal(StageRunStatus.Canceled, stageRun.Status);
    }

    [Fact]
    public async Task RetranslateSegmentAsync_WhenEngineTimesOutWithoutUserCancel_MarksStageAsFailed()
    {
        // A provider-side cancellation (e.g. HttpClient timeout) is not a user
        // cancel: the stage run must record Failed, not Canceled.
        var stageRunStore = new FakeProjectStageRunStore();
        var artifactStore = new FakeArtifactStore();
        var service = new TranslationOrchestrationService(
            new FakeTranslationRepository(),
            new GlossaryService(new FakeGlossaryRepository()),
            new GlossaryTermMatcher(),
            new FakeTranslationLanguageRouter(),
            new TimeoutTranslationEngine(),
            new FakeTtsTakeRepository(),
            stageRunStore,
            artifactStore,
            new TranscriptArtifactWriter(
                artifactStore,
                new FakeFileFingerprintService(),
                new FakeMediaAssetRepository()),
            commitBoundary: TestAtomicCommitBoundary.Create(new FakeTranslationRepository(), new FakeTtsTakeRepository(), artifactStore, new FakeMediaAssetRepository()));

        TranscriptProjectState state = BuildStateWithTranslationRevision();
        RetranslateSegmentRequest request = new(
            state.CurrentTranslationRevision!.Id,
            state.TranscriptSegments[0].Id,
            SourceLanguage: "en",
            TargetLanguage: "es");

        await Assert.ThrowsAsync<TaskCanceledException>(() =>
            service.RetranslateSegmentAsync(
                state,
                request,
                TestContext.Current.CancellationToken));

        StageRunRecord stageRun = Assert.Single(stageRunStore.All);
        Assert.Equal(StageNames.Translation, stageRun.StageName);
        Assert.Equal(StageRunStatus.Failed, stageRun.Status);
    }

    [Fact]
    public async Task RetranslateSegmentAsync_WhenEngineThrowsBareOceWithoutCancel_MarksStageAsFailed()
    {
        // A bare OperationCanceledException (not TaskCanceledException) without
        // canceling the caller's token is a provider-side failure: must record Failed.
        var stageRunStore = new FakeProjectStageRunStore();
        var artifactStore = new FakeArtifactStore();
        var service = new TranslationOrchestrationService(
            new FakeTranslationRepository(),
            new GlossaryService(new FakeGlossaryRepository()),
            new GlossaryTermMatcher(),
            new FakeTranslationLanguageRouter(),
            new BareOceTranslationEngine(),
            new FakeTtsTakeRepository(),
            stageRunStore,
            artifactStore,
            new TranscriptArtifactWriter(
                artifactStore,
                new FakeFileFingerprintService(),
                new FakeMediaAssetRepository()),
            commitBoundary: TestAtomicCommitBoundary.Create(new FakeTranslationRepository(), new FakeTtsTakeRepository(), artifactStore, new FakeMediaAssetRepository()));

        TranscriptProjectState state = BuildStateWithTranslationRevision();
        RetranslateSegmentRequest request = new(
            state.CurrentTranslationRevision!.Id,
            state.TranscriptSegments[0].Id,
            SourceLanguage: "en",
            TargetLanguage: "es");

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            service.RetranslateSegmentAsync(state, request, TestContext.Current.CancellationToken));

        StageRunRecord stageRun = Assert.Single(stageRunStore.All);
        Assert.Equal(StageNames.Translation, stageRun.StageName);
        Assert.Equal(StageRunStatus.Failed, stageRun.Status);
    }

    [Fact]
    public async Task RetranslateSegmentAsync_WhenTargetLanguageDiffersFromCurrentRevision_ThrowsWithoutStartingStage()
    {
        var stageRunStore = new FakeProjectStageRunStore();
        var artifactStore = new FakeArtifactStore();
        var service = new TranslationOrchestrationService(
            new FakeTranslationRepository(),
            new GlossaryService(new FakeGlossaryRepository()),
            new GlossaryTermMatcher(),
            new FakeTranslationLanguageRouter(),
            new FakeTranslationEngine(),
            new FakeTtsTakeRepository(),
            stageRunStore,
            artifactStore,
            new TranscriptArtifactWriter(
                artifactStore,
                new FakeFileFingerprintService(),
                new FakeMediaAssetRepository()),
            commitBoundary: TestAtomicCommitBoundary.Create(new FakeTranslationRepository(), new FakeTtsTakeRepository(), artifactStore, new FakeMediaAssetRepository()));

        TranscriptProjectState state = BuildStateWithTranslationRevision();
        RetranslateSegmentRequest request = new(
            state.CurrentTranslationRevision!.Id,
            state.TranscriptSegments[0].Id,
            SourceLanguage: "en",
            TargetLanguage: "fr");

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            service.RetranslateSegmentAsync(state, request, TestContext.Current.CancellationToken));

        Assert.Empty(stageRunStore.All);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WhenEngineThrowsOceUnderCancelledToken_MarksStageAsCanceled()
    {
        using var cts = new CancellationTokenSource();
        TranslationHarness harness = CreateTranslationHarness(
            transcriptLanguage: "en",
            segmentDetectedLanguage: "en",
            engine: new CancellingTranslationEngine(cts));

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            harness.Service.GenerateTranslationAsync(
                harness.State,
                new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "es"),
                cts.Token));

        StageRunRecord stageRun = Assert.Single(harness.StageRunStore.All);
        Assert.Equal(StageRunStatus.Canceled, stageRun.Status);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WhenEngineTimesOutWithoutUserCancel_MarksStageAsFailed()
    {
        TranslationHarness harness = CreateTranslationHarness(
            transcriptLanguage: "en",
            segmentDetectedLanguage: "en",
            engine: new TimeoutTranslationEngine());

        await Assert.ThrowsAsync<TaskCanceledException>(() =>
            harness.Service.GenerateTranslationAsync(
                harness.State,
                new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "es"),
                TestContext.Current.CancellationToken));

        StageRunRecord stageRun = Assert.Single(harness.StageRunStore.All);
        Assert.Equal(StageRunStatus.Failed, stageRun.Status);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WithAutoSource_FollowsPersistedTranscriptLanguage()
    {
        TranslationHarness harness = CreateTranslationHarness(transcriptLanguage: "en", segmentDetectedLanguage: null);

        await harness.Service.GenerateTranslationAsync(
            harness.State,
            new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "es"),
            TestContext.Current.CancellationToken);

        Assert.Equal("en", harness.CapturedSourceLanguage());
        StageRunRecord stageRun = Assert.Single(harness.StageRunStore.All);
        Assert.Equal(StageRunStatus.Completed, stageRun.Status);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WithAutoSource_FallsBackToDetectedSegmentLanguage_AndPersistsIt()
    {
        TranslationHarness harness = CreateTranslationHarness(transcriptLanguage: null, segmentDetectedLanguage: "en");

        await harness.Service.GenerateTranslationAsync(
            harness.State,
            new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "es"),
            TestContext.Current.CancellationToken);

        Assert.Equal("en", harness.CapturedSourceLanguage());
        ProjectManifest? manifest = await harness.ArtifactStore.ReadJsonAsync<ProjectManifest>(
            ProjectArtifactPaths.ManifestRelativePath,
            TestContext.Current.CancellationToken);
        Assert.Equal("en", manifest?.TranscriptLanguage);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WithAutoSource_AndNoLanguageAnywhere_ThrowsActionableError()
    {
        TranslationHarness harness = CreateTranslationHarness(transcriptLanguage: null, segmentDetectedLanguage: null);

        InvalidOperationException ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            harness.Service.GenerateTranslationAsync(
                harness.State,
                new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "es"),
                TestContext.Current.CancellationToken));

        Assert.Contains("Source language is unknown", ex.Message, StringComparison.Ordinal);
        Assert.Empty(harness.StageRunStore.All);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WithEmptyTranscriptSegments_ThrowsInsteadOfNoOpSuccess()
    {
        TranslationHarness harness = CreateTranslationHarness(transcriptLanguage: "en", segmentDetectedLanguage: "en");
        TranscriptProjectState emptyState = harness.State with { TranscriptSegments = [] };

        InvalidOperationException ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            harness.Service.GenerateTranslationAsync(
                emptyState,
                new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "es"),
                TestContext.Current.CancellationToken));

        Assert.Contains("no segments to translate", ex.Message, StringComparison.Ordinal);
        Assert.Empty(harness.StageRunStore.All);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WithExplicitSource_MismatchingTranscriptLanguage_Throws()
    {
        TranslationHarness harness = CreateTranslationHarness(transcriptLanguage: "en", segmentDetectedLanguage: "en");

        InvalidOperationException ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            harness.Service.GenerateTranslationAsync(
                harness.State,
                new GenerateTranslationRequest(SourceLanguage: "fr", TargetLanguage: "es"),
                TestContext.Current.CancellationToken));

        Assert.Contains("does not match the transcript language", ex.Message, StringComparison.Ordinal);
        Assert.Empty(harness.StageRunStore.All);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WithExplicitSource_AdoptsItWhenNoTranscriptLanguageIsSet()
    {
        TranslationHarness harness = CreateTranslationHarness(transcriptLanguage: null, segmentDetectedLanguage: null);

        await harness.Service.GenerateTranslationAsync(
            harness.State,
            new GenerateTranslationRequest(SourceLanguage: "en", TargetLanguage: "es"),
            TestContext.Current.CancellationToken);

        Assert.Equal("en", harness.CapturedSourceLanguage());
        ProjectManifest? manifest = await harness.ArtifactStore.ReadJsonAsync<ProjectManifest>(
            ProjectArtifactPaths.ManifestRelativePath,
            TestContext.Current.CancellationToken);
        Assert.Equal("en", manifest?.TranscriptLanguage);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WhenArtifactPersistFails_MarksStageFailedNotCompleted()
    {
        TranslationHarness harness = CreateTranslationHarness(
            transcriptLanguage: "en",
            segmentDetectedLanguage: "en");
        await harness.ArtifactStore.WriteJsonAsync(
            ProjectArtifactPaths.ManifestRelativePath,
            ProjectManifest.FromProject(harness.State.ProjectState.Project, "en")
                .WithUiSettings(new ProjectUiSettings(SelectedTranslationTargetLanguage: "es")),
            TestContext.Current.CancellationToken);
        harness.ArtifactStore.FailingJsonWriteFileName = "translation-revision-0001.json";

        await Assert.ThrowsAsync<IOException>(() =>
            harness.Service.GenerateTranslationAsync(
                harness.State,
                new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "fr"),
                TestContext.Current.CancellationToken));

        StageRunRecord stageRun = Assert.Single(harness.StageRunStore.All);
        Assert.Equal(StageRunStatus.Failed, stageRun.Status);
        ProjectManifest? manifest = await harness.ArtifactStore.ReadJsonAsync<ProjectManifest>(
            ProjectArtifactPaths.ManifestRelativePath, TestContext.Current.CancellationToken);
        Assert.Equal("es", manifest?.UiSettings?.SelectedTranslationTargetLanguage);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WhenTargetSelectionPersistFails_MarksStageFailedNotCompleted()
    {
        TranslationHarness harness = CreateTranslationHarness(transcriptLanguage: "en", segmentDetectedLanguage: "en");
        harness.ArtifactStore.FailingJsonWriteFileName = Path.GetFileName(ProjectArtifactPaths.ManifestRelativePath);

        await Assert.ThrowsAsync<IOException>(() => harness.Service.GenerateTranslationAsync(
            harness.State,
            new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "fr"),
            TestContext.Current.CancellationToken));

        Assert.NotNull(await harness.TranslationRepository.GetCurrentRevisionAsync(
            harness.State.ProjectState.Project.Id, "fr", TestContext.Current.CancellationToken));
        Assert.Equal(StageRunStatus.Failed, Assert.Single(harness.StageRunStore.All).Status);
    }

    [Fact]
    public async Task GenerateTranslationAsync_StreamCapableEngine_DefaultFlagKeepsBatchPath()
    {
        // An engine implementing the optional streaming interface must NOT stream unless
        // the request opts in — batch is authoritative by default.
        var engine = new StreamingFakeTranslationEngine();
        TranslationHarness harness = CreateTranslationHarness(
            transcriptLanguage: "en",
            segmentDetectedLanguage: "en",
            segmentCount: 3,
            engine: engine);

        await harness.Service.GenerateTranslationAsync(
            harness.State,
            new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "es"),
            TestContext.Current.CancellationToken);

        Assert.Equal(1, engine.BatchCalls);
        Assert.Equal(0, engine.StreamCalls);

        TranslationRevision revision = Assert.Single(harness.TranslationRepository.Revisions);
        Assert.NotNull(revision.ModelProvenance);
        Assert.Equal("policy-v1", revision.ModelProvenance.CatalogPolicyVersion);
        IReadOnlyList<TranslatedSegment> segments =
            await harness.TranslationRepository.GetSegmentsAsync(
                revision.Id, TestContext.Current.CancellationToken);
        Assert.Equal(3, segments.Count);
        StageRunRecord stageRun = Assert.Single(harness.StageRunStore.All);
        Assert.Equal(StageRunStatus.Completed, stageRun.Status);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WithStreamingEngine_CommitsSingleEquivalentRevision()
    {
        var engine = new StreamingFakeTranslationEngine();
        TranslationHarness harness = CreateTranslationHarness(
            transcriptLanguage: "en",
            segmentDetectedLanguage: "en",
            segmentCount: 3,
            engine: engine);

        await harness.Service.GenerateTranslationAsync(
            harness.State,
            new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "es", EnableSegmentStreaming: true),
            TestContext.Current.CancellationToken);

        Assert.Equal(1, engine.StreamCalls);
        Assert.Equal(0, engine.BatchCalls);

        TranslationRevision revision = Assert.Single(harness.TranslationRepository.Revisions);
        Assert.Equal(1, revision.RevisionNumber); // repository seeding is separate from state
        Assert.Equal("es", revision.TargetLanguage);
        Assert.NotNull(revision.ModelProvenance);
        Assert.Equal("policy-v1", revision.ModelProvenance.CatalogPolicyVersion);
        Assert.Equal(">>spa<<", revision.ModelProvenance.TargetPrefix);

        IReadOnlyList<TranslatedSegment> segments =
            await harness.TranslationRepository.GetSegmentsAsync(
                revision.Id, TestContext.Current.CancellationToken);
        Assert.Equal(3, segments.Count);
        Assert.Equal([0, 1, 2], segments.Select(s => s.SegmentIndex).ToArray());
        Assert.All(segments, segment => Assert.StartsWith("[es] ", segment.Text, StringComparison.Ordinal));

        StageRunRecord stageRun = Assert.Single(harness.StageRunStore.All);
        Assert.Equal(StageRunStatus.Completed, stageRun.Status);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WithStreamingEngine_NotifiesObserverForEverySegmentAsItArrives()
    {
        var engine = new StreamingFakeTranslationEngine();
        TranslationHarness harness = CreateTranslationHarness(
            transcriptLanguage: "en",
            segmentDetectedLanguage: "en",
            segmentCount: 3,
            engine: engine);
        var observer = new RecordingSegmentObserver();

        await harness.Service.GenerateTranslationAsync(
            harness.State,
            new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "es", EnableSegmentStreaming: true),
            new TranslationOrchestrationService.TranslationRunProgress(Progress: null, observer),
            TestContext.Current.CancellationToken);

        Assert.Equal([0, 1, 2], observer.Segments.Select(segment => segment.Index).ToArray());
        TranslationRevision revision = Assert.Single(harness.TranslationRepository.Revisions);
        IReadOnlyList<TranslatedSegment> committed =
            await harness.TranslationRepository.GetSegmentsAsync(revision.Id, TestContext.Current.CancellationToken);
        Assert.Equal(
            committed.OrderBy(segment => segment.SegmentIndex).Select(segment => segment.Text).ToArray(),
            observer.Segments.Select(segment => segment.Text).ToArray());
    }

    private sealed class RecordingSegmentObserver : ITranslatedSegmentObserver
    {
        public List<(int Index, string Text)> Segments { get; } = [];

        public void OnSegmentTranslated(int segmentIndex, string text) => Segments.Add((segmentIndex, text));
    }

    [Fact]
    public async Task GenerateTranslationAsync_WithStreamingFault_CommitsNoRevisionAndStageFailed()
    {
        var engine = new StreamingFakeTranslationEngine
        {
            FaultAfterItems = 1,
            Fault = new InvalidOperationException("stream exploded"),
        };
        TranslationHarness harness = CreateTranslationHarness(
            transcriptLanguage: "en",
            segmentDetectedLanguage: "en",
            segmentCount: 3,
            engine: engine);

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            harness.Service.GenerateTranslationAsync(
                harness.State,
                new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "es", EnableSegmentStreaming: true),
                TestContext.Current.CancellationToken));

        Assert.Empty(harness.TranslationRepository.Revisions); // no partial commit
        StageRunRecord stageRun = Assert.Single(harness.StageRunStore.All);
        Assert.Equal(StageRunStatus.Failed, stageRun.Status);
        Assert.Contains("stream exploded", stageRun.FailureReason, StringComparison.Ordinal);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WithMalformedStreamItem_CommitsNoRevision()
    {
        var engine = new StreamingFakeTranslationEngine
        {
            // Second item replays the first identity — duplicate sequence must be rejected.
            RewriteItem = (item, seq) =>
                seq == 1
                    ? item with { Identity = item.Identity with { Sequence = 0 } }
                    : item,
        };
        TranslationHarness harness = CreateTranslationHarness(
            transcriptLanguage: "en",
            segmentDetectedLanguage: "en",
            segmentCount: 3,
            engine: engine);

        await Assert.ThrowsAsync<InvalidDataException>(() =>
            harness.Service.GenerateTranslationAsync(
                harness.State,
                new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "es", EnableSegmentStreaming: true),
                TestContext.Current.CancellationToken));

        Assert.Empty(harness.TranslationRepository.Revisions);
        StageRunRecord stageRun = Assert.Single(harness.StageRunStore.All);
        Assert.Equal(StageRunStatus.Failed, stageRun.Status);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WithUnrequestedStreamSegment_CommitsNoRevision()
    {
        var engine = new StreamingFakeTranslationEngine
        {
            RewriteItem = (item, seq) =>
                seq == 1
                    ? item with
                    {
                        Identity = item.Identity with { SegmentIndex = 99 },
                        Payload = item.Payload with { Index = 99 },
                    }
                    : item,
        };
        TranslationHarness harness = CreateTranslationHarness(
            transcriptLanguage: "en",
            segmentDetectedLanguage: "en",
            segmentCount: 3,
            engine: engine);

        await Assert.ThrowsAsync<InvalidDataException>(() =>
            harness.Service.GenerateTranslationAsync(
                harness.State,
                new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "es", EnableSegmentStreaming: true),
                TestContext.Current.CancellationToken));

        Assert.Empty(harness.TranslationRepository.Revisions);
        Assert.Equal(StageRunStatus.Failed, Assert.Single(harness.StageRunStore.All).Status);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WithIncompleteStream_CommitsNoRevision()
    {
        var engine = new StreamingFakeTranslationEngine { StopAfterItems = 2 };
        TranslationHarness harness = CreateTranslationHarness(
            transcriptLanguage: "en",
            segmentDetectedLanguage: "en",
            segmentCount: 3,
            engine: engine);

        await Assert.ThrowsAsync<InvalidDataException>(() =>
            harness.Service.GenerateTranslationAsync(
                harness.State,
                new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "es", EnableSegmentStreaming: true),
                TestContext.Current.CancellationToken));

        Assert.Empty(harness.TranslationRepository.Revisions);
        Assert.Equal(StageRunStatus.Failed, Assert.Single(harness.StageRunStore.All).Status);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WithStreamingWrongRevision_CommitsNoRevision()
    {
        var engine = new StreamingFakeTranslationEngine
        {
            RewriteItem = (item, _) =>
                item with { Identity = item.Identity with { RevisionId = Guid.NewGuid() } },
        };
        TranslationHarness harness = CreateTranslationHarness(
            transcriptLanguage: "en",
            segmentDetectedLanguage: "en",
            segmentCount: 3,
            engine: engine);

        await Assert.ThrowsAsync<InvalidDataException>(() =>
            harness.Service.GenerateTranslationAsync(
                harness.State,
                new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "es", EnableSegmentStreaming: true),
                TestContext.Current.CancellationToken));

        Assert.Empty(harness.TranslationRepository.Revisions);
    }

    [Fact]
    public async Task GenerateTranslationAsync_WithStreamingCancellation_MarksStageCanceled()
    {
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var engine = new StreamingFakeTranslationEngine { CancelAfterItems = 1, CancelSource = cts };
        TranslationHarness harness = CreateTranslationHarness(
            transcriptLanguage: "en",
            segmentDetectedLanguage: "en",
            segmentCount: 3,
            engine: engine);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            harness.Service.GenerateTranslationAsync(
                harness.State,
                new GenerateTranslationRequest(SourceLanguage: "auto", TargetLanguage: "es", EnableSegmentStreaming: true),
                cts.Token));

        Assert.Empty(harness.TranslationRepository.Revisions);
        StageRunRecord stageRun = Assert.Single(harness.StageRunStore.All);
        Assert.Equal(StageRunStatus.Canceled, stageRun.Status);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private sealed record TranslationHarness(
        TranslationOrchestrationService Service,
        TranscriptProjectState State,
        FakeProjectStageRunStore StageRunStore,
        FakeArtifactStore ArtifactStore,
        Func<string?> CapturedSourceLanguage,
        FakeTranslationRepository TranslationRepository);

    private static TranslationHarness CreateTranslationHarness(
        string? transcriptLanguage,
        string? segmentDetectedLanguage,
        ITranslationEngine? engine = null,
        int segmentCount = 1)
    {
        var stageRunStore = new FakeProjectStageRunStore();
        var translationRepository = new FakeTranslationRepository();
        var glossaryRepository = new FakeGlossaryRepository();
        var ttsTakeRepository = new FakeTtsTakeRepository();
        var artifactStore = new FakeArtifactStore();
        var fileFingerprintService = new FakeFileFingerprintService();
        var mediaAssetRepository = new FakeMediaAssetRepository();
        var artifactWriter = new TranscriptArtifactWriter(
            artifactStore,
            fileFingerprintService,
            mediaAssetRepository);

        string? capturedSourceLanguage = null;
        engine ??= new FakeTranslationEngine(
            (request, segment) =>
            {
                capturedSourceLanguage = request.SourceLanguage;
                return $"[es] {segment.Text}";
            });

        var service = new TranslationOrchestrationService(
            translationRepository,
            new GlossaryService(glossaryRepository),
            new GlossaryTermMatcher(),
            new FakeTranslationLanguageRouter(),
            engine,
            ttsTakeRepository,
            stageRunStore,
            artifactStore,
            artifactWriter,
            commitBoundary: TestAtomicCommitBoundary.Create(translationRepository, ttsTakeRepository, artifactStore, mediaAssetRepository));

        TranscriptProjectState state = BuildStateWithTranslationRevision(transcriptLanguage, segmentDetectedLanguage, segmentCount);
        return new TranslationHarness(service, state, stageRunStore, artifactStore, () => capturedSourceLanguage, translationRepository);
    }

    private static TranscriptProjectState BuildStateWithTranslationRevision(
        string? transcriptLanguage = "en",
        string? segmentDetectedLanguage = "en",
        int segmentCount = 1)
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        Guid projectId = Guid.NewGuid();
        var project = new TrackdubProject(projectId, "Test Project", now, now);
        var mediaAsset = new MediaAsset(
            Guid.NewGuid(),
            projectId,
            "source.mp4",
            "source.mp4",
            "source-hash",
            100,
            now,
            "mp4",
            4.0d,
            HasAudio: true,
            HasVideo: true,
            now);
        var projectState = new OpenProjectResult(
            project,
            mediaAsset,
            null,
            SourceMediaStatus.Available,
            null,
            [],
            transcriptLanguage);

        TranscriptRevision transcriptRevision = TranscriptRevision.Create(
            projectId,
            stageRunId: null,
            revisionNumber: 1,
            now);

        var speaker = new ProjectSpeaker(Guid.NewGuid(), projectId, "Speaker 1", now);

        TranscriptSegment[] transcriptSegments = Enumerable.Range(0, segmentCount)
            .Select(index => TranscriptSegment.Create(
                transcriptRevision.Id,
                index,
                index * 2.0d,
                (index + 1) * 2.0d,
                $"Hello there {index}.",
                speaker.Id,
                segmentDetectedLanguage))
            .ToArray();

        TranslationRevision translationRevision = TranslationRevision.Create(
            projectId,
            stageRunId: null,
            transcriptRevision.Id,
            "es",
            revisionNumber: 1,
            now,
            translationProvider: "fake",
            modelId: "fake-model");

        TranslatedSegment translatedSegment = TranslatedSegment.Create(
            translationRevision.Id,
            0,
            0.0d,
            2.0d,
            "Hola.");

        return new TranscriptProjectState(
            projectState,
            transcriptRevision,
            transcriptSegments,
            [speaker],
            [],
            translationRevision,
            [translatedSegment],
            IsTranslationStale: false,
            TranscriptLanguage: transcriptLanguage,
            StageRuns: [],
            SupportedTargetLanguages: [],
            SelectedTranslationTargetLanguage: "es",
            StaleTranslatedSegmentIndices: new HashSet<int>(),
            WaveformSummary: null,
            AvailableVoices: [],
            VoiceAssignments: [],
            TtsTakes: [],
            TtsSegmentStates: [],
            VoiceAssignmentWarnings: []);
    }

    // -------------------------------------------------------------------------
    // Private test doubles
    // -------------------------------------------------------------------------

    /// <summary>
    /// A translation engine that cancels the caller's token and then throws
    /// <summary>
    /// Streaming translation engine fake: emits one identity-correct stream item per input
    /// segment via <see cref="PipelineStreamItemFactory"/>, with injectable fault/cancel/
    /// identity-rewrite hooks for contract-violation tests.
    /// </summary>
    private sealed class StreamingFakeTranslationEngine : IStreamingTranslationEngine, ITranslationExecutionMetadataReporter
    {
        public TranslationExecutionMetadata? LastExecutionMetadata { get; } = new(
            "opus-mt", "catalog-model", "catalog-alias", "cpu", TranslationRoutingKind.Direct,
            new Trackdub.Domain.Translation.TranslationModelProvenance("test/model", new string('a', 40), new string('b', 64),
                new Dictionary<string, string> { ["model.onnx"] = new string('b', 64) }, ">>spa<<", "policy-v1"));

        public int StreamCalls { get; private set; }

        public int BatchCalls { get; private set; }

        public int? FaultAfterItems { get; set; }

        public int? StopAfterItems { get; set; }

        public Exception? Fault { get; set; }

        public int? CancelAfterItems { get; set; }

        public CancellationTokenSource? CancelSource { get; set; }

        public Func<PipelineStreamItem<TranslatedTextSegment>, long, PipelineStreamItem<TranslatedTextSegment>>? RewriteItem { get; set; }

        public Task<IReadOnlyList<TranslatedTextSegment>> TranslateAsync(
            TranslationRequest request,
            CancellationToken cancellationToken)
        {
            BatchCalls++;
            return Task.FromResult<IReadOnlyList<TranslatedTextSegment>>(
                request.Segments
                    .OrderBy(static s => s.Index)
                    .Select(segment => new TranslatedTextSegment(
                        segment.Index, segment.StartSeconds, segment.EndSeconds,
                        $"[es] {segment.Text}"))
                    .ToArray());
        }

        public async IAsyncEnumerable<PipelineStreamItem<TranslatedTextSegment>> TranslateStreamAsync(
            TranslationRequest request,
            Guid runId,
            string snapshotId,
            Guid sourceRevisionId,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
        {
            StreamCalls++;
            long sequence = 0;
            foreach (TranslationInputSegment segment in request.Segments.OrderBy(static s => s.Index))
            {
                if (CancelAfterItems is { } cancelAfter && sequence == cancelAfter)
                {
                    CancelSource?.Cancel();
                    cancellationToken.ThrowIfCancellationRequested();
                }

                if (FaultAfterItems is { } faultAfter && sequence == faultAfter)
                {
                    throw Fault ?? new InvalidOperationException("stream fault");
                }

                cancellationToken.ThrowIfCancellationRequested();
                await Task.Yield();
                var translated = new TranslatedTextSegment(
                    segment.Index, segment.StartSeconds, segment.EndSeconds,
                    $"[es] {segment.Text}");
                PipelineStreamItem<TranslatedTextSegment> item =
                    PipelineStreamItemFactory.CreateTranslation(
                        translated, runId, snapshotId, sourceRevisionId, sequence);
                yield return RewriteItem?.Invoke(item, sequence) ?? item;
                sequence++;
                if (StopAfterItems is { } stopAfter && sequence >= stopAfter)
                {
                    yield break;
                }
            }
        }
    }

    /// <see cref="OperationCanceledException"/>, simulating user cancellation
    /// arriving mid-run.
    /// </summary>
    private sealed class CancellingTranslationEngine(CancellationTokenSource cts) : ITranslationEngine
    {
        public Task<IReadOnlyList<TranslatedTextSegment>> TranslateAsync(
            TranslationRequest request,
            CancellationToken cancellationToken)
        {
            cts.Cancel();
            throw new OperationCanceledException("Translation was cancelled.");
        }
    }

    /// <summary>
    /// A translation engine that throws <see cref="TaskCanceledException"/>
    /// without touching the caller's token, simulating a cloud-provider timeout.
    /// </summary>
    private sealed class TimeoutTranslationEngine : ITranslationEngine
    {
        public Task<IReadOnlyList<TranslatedTextSegment>> TranslateAsync(
            TranslationRequest request,
            CancellationToken cancellationToken) =>
            throw new TaskCanceledException("The request timed out.");
    }

    /// <summary>
    /// A translation engine that throws a bare <see cref="OperationCanceledException"/>
    /// without canceling the caller's token, simulating a provider-internal failure.
    /// </summary>
    private sealed class BareOceTranslationEngine : ITranslationEngine
    {
        public Task<IReadOnlyList<TranslatedTextSegment>> TranslateAsync(
            TranslationRequest request,
            CancellationToken cancellationToken) =>
            throw new OperationCanceledException("Provider-side operation was cancelled.");
    }
}
