using Trackdub.Application.Dubbing;
using Trackdub.Application.LipSynthesis;
using Trackdub.Contracts;
using Trackdub.Contracts.Dubbing;
using Trackdub.Contracts.Licensing;
using Trackdub.Application.Mixing;
using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts;
using Trackdub.Application.Transcripts.Stages;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.AudioQuality;
using Trackdub.Domain.Media;
using Trackdub.Domain.Mixing;
using Trackdub.Domain.Projects;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Trackdub.Domain.Tts;
using Trackdub.TestDoubles;
using System.Security.Cryptography;

#pragma warning disable CS0618

namespace Trackdub.Application.Tests;

public partial class TranscriptProjectServiceTests
{
    [Fact]
    public async Task ProjectWorkflow_CreateAsync_returns_refreshed_state()
    {
        var (_, state) = await CreateWorkspaceProjectAsync();

        Assert.NotNull(state.CurrentTranscriptRevision);
        Assert.Equal(1, state.CurrentTranscriptRevision!.RevisionNumber);
        Assert.Equal(2, state.TranscriptSegments.Count);
    }

    [Fact]
    public async Task ProjectWorkflow_RelocateSourceAsync_preserves_selected_translation_target()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        FakeServiceScope scope = CreateScope(tempDirectory);
        await scope.Workspace.Project.CreateAsync(
            new CreateTranscriptProjectRequest("Transcript Demo", sourcePath),
            TestContext.Current.CancellationToken);
        await scope.Workspace.Translation.SetTranscriptLanguageAsync(
            new SetTranscriptLanguageRequest("en"),
            TestContext.Current.CancellationToken);
        await scope.Workspace.Translation.GenerateTranslationAsync(
            new GenerateTranslationRequest("en", "fr"),
            TestContext.Current.CancellationToken);

        TranscriptProjectState relocated = await scope.Workspace.Project.RelocateSourceAsync(
            new RelocateTranscriptSourceRequest(sourcePath, "fr"),
            TestContext.Current.CancellationToken);

        Assert.Equal("fr", relocated.SelectedTranslationTargetLanguage);
        Assert.NotNull(relocated.CurrentTranslationRevision);
        Assert.Equal("fr", relocated.CurrentTranslationRevision!.TargetLanguage);
        Assert.Equal(2, relocated.TranslatedSegments.Count);
    }

    [Fact]
    public async Task ProjectWorkflow_RenameProjectAsync_persists_project_name_and_manifest()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        FakeServiceScope scope = CreateScope(tempDirectory);
        await scope.Workspace.Project.CreateAsync(
            new CreateTranscriptProjectRequest("Transcript Demo", sourcePath),
            TestContext.Current.CancellationToken);
        await scope.Workspace.Translation.SetTranscriptLanguageAsync(
            new SetTranscriptLanguageRequest("en"),
            TestContext.Current.CancellationToken);
        await scope.Workspace.Translation.GenerateTranslationAsync(
            new GenerateTranslationRequest("en", "fr"),
            TestContext.Current.CancellationToken);

        TranscriptProjectState renamed = await scope.Workspace.Project.RenameProjectAsync(
            new RenameProjectRequest("Renamed Demo", "fr"),
            TestContext.Current.CancellationToken);
        ProjectManifest? manifest = await scope.ArtifactStore.ReadJsonAsync<ProjectManifest>(
            ProjectArtifactPaths.ManifestRelativePath,
            TestContext.Current.CancellationToken);

        Assert.Equal("Renamed Demo", renamed.ProjectState.Project.Name);
        Assert.Equal("fr", renamed.SelectedTranslationTargetLanguage);
        Assert.NotNull(renamed.CurrentTranslationRevision);
        Assert.NotNull(manifest);
        Assert.Equal("Renamed Demo", manifest!.Name);

        TranscriptProjectState current = await scope.Workspace.Project.OpenAsync(TestContext.Current.CancellationToken);
        TranscriptProjectState renamedAgain = await scope.Workspace.Project.RenameProjectAsync(
            new RenameProjectRequest("Renamed Demo Again"),
            TestContext.Current.CancellationToken);

        Assert.Equal("Renamed Demo Again", renamedAgain.ProjectState.Project.Name);
        Assert.Equal(current.SelectedTranslationTargetLanguage, renamedAgain.SelectedTranslationTargetLanguage);
    }

    [Fact]
    public async Task TranscriptWorkflow_SaveEditsAsync_returns_refreshed_state()
    {
        var (scope, created) = await CreateWorkspaceProjectAsync();

        TranscriptProjectState saved = await scope.Workspace.Transcript.SaveEditsAsync(
            new SaveTranscriptEditsRequest(
                created.CurrentTranscriptRevision!.Id,
                [new EditedTranscriptSegment(created.TranscriptSegments[0].Id, "Workspace edited text.", created.TranscriptSegments[0].SpeakerId)]),
            TestContext.Current.CancellationToken);

        Assert.Equal(2, saved.CurrentTranscriptRevision!.RevisionNumber);
        Assert.Equal("Workspace edited text.", saved.TranscriptSegments[0].Text);
    }

    [Fact]
    public async Task TranslationWorkflow_GenerateTranslationAsync_returns_refreshed_state()
    {
        var (scope, _) = await CreateWorkspaceProjectAsync();

        await scope.Workspace.Translation.SetTranscriptLanguageAsync(new SetTranscriptLanguageRequest("en"), TestContext.Current.CancellationToken);
        TranscriptProjectState translated = await scope.Workspace.Translation.GenerateTranslationAsync(
            new GenerateTranslationRequest("en", "es"),
            TestContext.Current.CancellationToken);

        Assert.NotNull(translated.CurrentTranslationRevision);
        Assert.Equal("es", translated.SelectedTranslationTargetLanguage);
        Assert.Equal(2, translated.TranslatedSegments.Count);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Translation_to_tts_export_and_lip_synthesis_preserves_nondefault_target(bool hasPreviousTranslation)
    {
        var (scope, created) = await CreateWorkspaceProjectAsync(
            ttsEngine: new FakeTtsEngine { DurationSamples = 168000 },
            loudnessNormalizer: new CopyingLoudnessNormalizer());
        CancellationToken token = TestContext.Current.CancellationToken;
        await scope.Workspace.Translation.SetTranscriptLanguageAsync(new SetTranscriptLanguageRequest("en"), token);
        TranslationRevision? previousRevision = null;
        if (hasPreviousTranslation)
        {
            TranscriptProjectState previous = await scope.Workspace.GenerateTranslationAsync(
                new GenerateTranslationRequest("en", "es"), token);
            previousRevision = previous.CurrentTranslationRevision;
            await scope.Workspace.Translation.SelectTranslationTargetAsync(new SetTranslationTargetRequest("es"), token);
        }

        using var session = new WorkflowSession(scope.Workspace, scope.ArtifactStore.GetPath(string.Empty));
        var options = new DubbingSessionOptions
        {
            SourceMediaPath = "sample.mp4",
            TargetLanguageCode = "fr",
            VoiceAssignmentOverrides = created.Speakers.ToDictionary(speaker => speaker.Id.ToString(), _ => "af_heart"),
            ExportOutputPath = Path.Join(scope.ArtifactStore.GetPath(string.Empty), "delivery", "dubbed.mp4"),
        };
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto, IsDevBuild: false, new Dictionary<string, ExecutionProviderKind>());

        DubbingPipelineEngine.StageWorkflowResult translation = await DubbingPipelineEngine.RunStageWorkflowAsync(
            session, options, StageNames.Translation, selections, progress: null, token);
        Assert.Equal(StageStatus.Succeeded, translation.Status);
        Assert.NotNull(await scope.TranslationRepository.GetCurrentRevisionAsync(
            (await scope.Workspace.Project.OpenAsync(token)).ProjectState.Project.Id, "fr", token));

        DubbingPipelineEngine.StageWorkflowResult tts = await DubbingPipelineEngine.RunStageWorkflowAsync(
            session, options, StageNames.Tts, selections, progress: null, token);
        Assert.Equal(StageStatus.Succeeded, tts.Status);
        TranslationRevision? frenchRevision = await scope.TranslationRepository.GetCurrentRevisionAsync(
            (await scope.Workspace.Project.OpenAsync(token)).ProjectState.Project.Id, "fr", token);
        Assert.NotNull(frenchRevision);
        IReadOnlyList<TranslatedSegment> frenchSegments = await scope.TranslationRepository.GetSegmentsAsync(frenchRevision!.Id, token);
        Assert.Equal(frenchSegments.Count, scope.TtsEngine.SynthesizeCallCount);
        Assert.Equal(frenchSegments.Count, scope.TtsEngine.Requests.Count);
        Assert.All(scope.TtsEngine.Requests, request => Assert.Equal("fr", request.LanguageCode));
        Assert.Equal(
            frenchSegments.Select(segment => segment.Text).OrderBy(text => text),
            scope.TtsEngine.Requests.Select(request => request.Text).OrderBy(text => text));
        Dictionary<int, Guid> frenchSegmentIdsByIndex = frenchSegments.ToDictionary(segment => segment.SegmentIndex, segment => segment.Id);
        foreach (TtsTake take in scope.TtsTakeRepository.Takes)
        {
            Assert.Equal(TtsTakeStatus.Completed, take.Status);
            Assert.False(take.IsStale);
            Assert.Equal(frenchSegmentIdsByIndex[take.SegmentIndex], take.TranslatedSegmentId);
        }

        TranscriptProjectState synthesized = await AssertLipSynthesisUsesCurrentMixPlanDriverAsync(scope, token);

        DubbingPipelineEngine.StageWorkflowResult export = await DubbingPipelineEngine.RunStageWorkflowAsync(
            session, options, StageNames.Export, selections, progress: null, token);
        Assert.Equal(StageStatus.Succeeded, export.Status);
        Assert.True(File.Exists(options.ExportOutputPath));
        // Drop all in-memory JSON so reopen rehydrates the manifest from disk,
        // simulating a process restart. Repositories stay in-memory; only the
        // manifest selection round-trip is disk-backed here.
        scope.ArtifactStore.SimulateProcessRestart();
        TranscriptProjectState reopened = await scope.CreateReopenedStateService().OpenAsync(null, token);
        Assert.Equal("fr", reopened.SelectedTranslationTargetLanguage);
        Assert.Equal("fr", reopened.CurrentTranslationRevision?.TargetLanguage);
        Assert.Equal(2, reopened.TranslatedSegments.Count);
        ProjectArtifact exportManifestArtifact = Assert.Single(reopened.ProjectState.Artifacts,
            artifact => artifact.Kind == ArtifactKind.ExportManifest);
        ExportManifest? exportManifest = await scope.ArtifactStore.ReadJsonAsync<ExportManifest>(
            exportManifestArtifact.RelativePath, token);
        Assert.Equal("fr", exportManifest?.TargetLanguage);
        Assert.NotNull(exportManifest);
        Assert.All(exportManifest!.Segments, segment =>
        {
            Assert.NotNull(segment.TtsTakeId);
            TtsTake take = Assert.Single(scope.TtsTakeRepository.Takes, candidate => candidate.Id == segment.TtsTakeId);
            Assert.Equal(TtsTakeStatus.Completed, take.Status);
            Assert.False(take.IsStale);
            Assert.Equal(frenchSegmentIdsByIndex[segment.SegmentIndex], take.TranslatedSegmentId);
        });
        ProjectManifest? manifest = await scope.ArtifactStore.ReadJsonAsync<ProjectManifest>(
            ProjectArtifactPaths.ManifestRelativePath, token);
        Assert.Equal("fr", manifest?.UiSettings?.SelectedTranslationTargetLanguage);
        string subtitles = await File.ReadAllTextAsync(Path.ChangeExtension(options.ExportOutputPath!, ".srt"), token);
        Assert.All(reopened.TranslatedSegments, segment => Assert.Contains(segment.Text, subtitles));
        if (previousRevision is not null)
        {
            Assert.Equal(previousRevision.Id, (await scope.TranslationRepository.GetCurrentRevisionAsync(
                reopened.ProjectState.Project.Id, "es", token))?.Id);
        }

        Assert.Equal(reopened.CurrentTranslationRevision!.Id, synthesized.CurrentTranslationRevision?.Id);
    }

    private sealed class CopyingLoudnessNormalizer : ILoudnessNormalizer
    {
        public Task<LoudnessAnalysisResult> AnalyzeAsync(LoudnessAnalysisRequest request, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(new LoudnessAnalysisResult(request.InputPath, ExportLoudnessTargets.OnlineLufs, Warnings: []));
        }

        public async Task<LoudnessNormalizationResult> NormalizeAsync(
            LoudnessNormalizationRequest request, CancellationToken cancellationToken)
        {
            await using FileStream input = File.OpenRead(request.InputPath);
            await using FileStream output = File.Create(request.OutputPath);
            await input.CopyToAsync(output, cancellationToken);
            return new LoudnessNormalizationResult(request.OutputPath, request.TargetLufs, request.TargetLufs, Warnings: []);
        }
    }

    private sealed class WorkflowSession(TranscriptWorkspace workspace, string projectRootPath) : IDubbingSession
    {
        public string ProjectRootPath => projectRootPath;
        public TranscriptWorkspace Workspace => workspace;
        public IServiceProvider Services { get; } = new WorkflowServiceProvider();
        public void Dispose() => workspace.Dispose();
        public ValueTask DisposeAsync()
        {
            Dispose();
            return ValueTask.CompletedTask;
        }
    }

    private sealed class WorkflowServiceProvider : IServiceProvider
    {
        public object? GetService(Type serviceType) => null;
    }


    [Fact]
    public async Task SpeakerWorkflow_RenameSpeakerAsync_returns_refreshed_state()
    {
        var (scope, created) = await CreateWorkspaceProjectAsync();

        ProjectSpeaker speaker = created.Speakers[0];
        TranscriptProjectState renamed = await scope.Workspace.Speakers.RenameSpeakerAsync(
            new RenameSpeakerRequest(speaker.Id, "Host"),
            TestContext.Current.CancellationToken);

        Assert.Equal("Host", Assert.Single(renamed.Speakers, candidate => candidate.Id == speaker.Id).DisplayName);
    }

    [Fact]
    public async Task VoiceWorkflow_AssignVoiceToSpeakerAsync_returns_refreshed_state()
    {
        var (scope, _) = await CreateWorkspaceProjectAsync();
        await scope.Workspace.Translation.SetTranscriptLanguageAsync(new SetTranscriptLanguageRequest("en"), TestContext.Current.CancellationToken);
        TranscriptProjectState translated = await scope.Workspace.Translation.GenerateTranslationAsync(
            new GenerateTranslationRequest("en", "es"),
            TestContext.Current.CancellationToken);

        TranscriptProjectState assigned = await scope.Workspace.Voices.AssignVoiceToSpeakerAsync(
            new AssignVoiceToSpeakerRequest(translated.Speakers[0].Id, "af_heart"),
            TestContext.Current.CancellationToken);

        Assert.Contains(assigned.VoiceAssignments, assignment =>
            assignment.SpeakerId == translated.Speakers[0].Id &&
            assignment.VoiceVariant == "af_heart");
    }

    [Fact]
    public async Task TtsWorkflow_GenerateTtsForSpeakerAsync_returns_refreshed_state()
    {
        var (scope, _) = await CreateWorkspaceProjectAsync(ttsEngine: new FakeTtsEngine { DurationSamples = 168000 });
        await scope.Workspace.Translation.SetTranscriptLanguageAsync(new SetTranscriptLanguageRequest("en"), TestContext.Current.CancellationToken);
        TranscriptProjectState translated = await scope.Workspace.Translation.GenerateTranslationAsync(
            new GenerateTranslationRequest("en", "es"),
            TestContext.Current.CancellationToken);
        await scope.Workspace.Voices.AssignVoiceToSpeakerAsync(
            new AssignVoiceToSpeakerRequest(translated.Speakers[0].Id, "af_heart"),
            TestContext.Current.CancellationToken);

        TranscriptProjectState tts = await scope.Workspace.Tts.GenerateTtsForSpeakerAsync(
            new GenerateTtsForSpeakerRequest(translated.Speakers[0].Id),
            TestContext.Current.CancellationToken);

        Assert.Single(tts.TtsTakes);
        Assert.Contains(tts.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.TtsTake);
    }

    [Fact]
    public async Task TtsWorkflow_GenerateTtsForSpeakerAsync_rejects_overlapping_runs()
    {
        var ttsEngine = new BlockingTtsEngine();
        var (scope, _) = await CreateWorkspaceProjectAsync(ttsEngine: ttsEngine);
        await scope.Workspace.Translation.SetTranscriptLanguageAsync(new SetTranscriptLanguageRequest("en"), TestContext.Current.CancellationToken);
        TranscriptProjectState translated = await scope.Workspace.Translation.GenerateTranslationAsync(
            new GenerateTranslationRequest("en", "es"),
            TestContext.Current.CancellationToken);
        await scope.Workspace.Voices.AssignVoiceToSpeakerAsync(
            new AssignVoiceToSpeakerRequest(translated.Speakers[0].Id, "af_heart"),
            TestContext.Current.CancellationToken);

        Task<TranscriptProjectState> firstRun = scope.Workspace.Tts.GenerateTtsForSpeakerAsync(
            new GenerateTtsForSpeakerRequest(translated.Speakers[0].Id),
            TestContext.Current.CancellationToken);
        await ttsEngine.WaitForSynthesisStartedAsync();

        Task<TranscriptProjectState> secondRun = scope.Workspace.Tts.GenerateTtsForSpeakerAsync(
            new GenerateTtsForSpeakerRequest(translated.Speakers[0].Id),
            TestContext.Current.CancellationToken);

        try
        {
            Task completed = await Task.WhenAny(secondRun, Task.Delay(TimeSpan.FromMilliseconds(250), TestContext.Current.CancellationToken));
            Assert.Same(secondRun, completed);
            InvalidOperationException exception = await Assert.ThrowsAsync<InvalidOperationException>(() => secondRun);
            Assert.Contains("already running", exception.Message);
        }
        finally
        {
            ttsEngine.Release();
            await firstRun;
            try
            {
                await secondRun;
            }
            catch (InvalidOperationException)
            {
            }
        }
    }

    [Fact]
    public async Task EditingHistoryWorkflow_RestoreEditingStateAsync_returns_refreshed_state()
    {
        var (scope, created) = await CreateWorkspaceProjectAsync(enableSpeakerDiarization: false);
        TranscriptProjectState split = await scope.Workspace.Transcript.SplitSegmentAsync(
            new SplitTranscriptSegmentRequest(created.CurrentTranscriptRevision!.Id, created.TranscriptSegments[0].Id, 2.9),
            TestContext.Current.CancellationToken);

        TranscriptProjectState restored = await scope.Workspace.EditingHistory.RestoreEditingStateAsync(
            new RestoreEditingStateRequest(
                created.SelectedTranslationTargetLanguage,
                created.TranscriptSegments,
                created.CurrentTranslationRevision is null ? null : created.TranslatedSegments,
                created.Speakers.ToDictionary(speaker => speaker.Id, speaker => speaker.DisplayName),
                created.VoiceAssignments),
            TestContext.Current.CancellationToken);

        Assert.Equal(3, split.TranscriptSegments.Count);
        Assert.Equal(2, restored.TranscriptSegments.Count);
        Assert.Equal(3, restored.CurrentTranscriptRevision!.RevisionNumber);
    }

    [Fact]
    public void DiarizationModelWorkflow_reports_required_model_status()
    {
        string tempDirectory = CreateTempDirectory();
        string modelCacheRoot = Path.Join(tempDirectory, "model-cache");
        var handler = new DiarizationStageHandler(
            new RecordingDiarizationEngine(),
            new RecordingModelDownloader(),
            modelCacheRoot: modelCacheRoot,
            expectedSha256: SortFormerTestFixtures.ExpectedSha256);
        FakeServiceScope scope = CreateScope(
            tempDirectory,
            diarizationStageHandler: handler);

        RequiredDiarizationModelStatus? status = scope.Workspace.DiarizationModels.GetRequiredDiarizationModelStatus();

        Assert.NotNull(status);
        Assert.False(status.IsAvailable);
        Assert.True(status.CanAutoDownload);
    }
}
