using Trackdub.Contracts;
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
    [Theory]
    [InlineData(StageRunStatus.Failed)]
    [InlineData(StageRunStatus.Canceled)]
    [InlineData(StageRunStatus.Running)]
    public async Task Failed_producer_is_not_selected(StageRunStatus status)
    {
        await AssertPriorStemPairRemainsSelectedAsync(status, includeAmbiance: true);
    }

    [Fact]
    public async Task Stem_pair_comes_from_one_completed_run()
    {
        await AssertPriorStemPairRemainsSelectedAsync(StageRunStatus.Completed, includeAmbiance: false);
    }

    private async Task AssertPriorStemPairRemainsSelectedAsync(StageRunStatus status, bool includeAmbiance)
    {
        string directory = CreateTempDirectory();
        string sourcePath = Path.Join(directory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);
        FakeServiceScope scope = CreateScope(directory);
        TranscriptProjectState prior = await CreateWithStemsAsync(scope, new("Stem generations", sourcePath));
        ProjectArtifact vocals = Assert.Single(prior.ProjectState.Artifacts, a => a.Kind == ArtifactKind.Vocals);
        ProjectArtifact ambiance = Assert.Single(prior.ProjectState.Artifacts, a => a.Kind == ArtifactKind.Ambiance);
        DateTimeOffset now = DateTimeOffset.UtcNow.AddMinutes(1);
        StageRunRecord orphanRun = StageRunRecord.Start(vocals.ProjectId, StageNames.Separation, now) with
        {
            Status = status,
            CompletedAtUtc = status == StageRunStatus.Running ? null : now.AddSeconds(1)
        };
        await scope.StageRunStore.CreateAsync(orphanRun, TestContext.Current.CancellationToken);
        scope.MediaAssetRepository.Artifacts.Add(vocals with
        {
            Id = Guid.NewGuid(),
            StageRunId = orphanRun.Id,
            CreatedAtUtc = now,
            RelativePath = "artifacts/stems/orphan/vocals.wav"
        });
        if (includeAmbiance)
        {
            scope.MediaAssetRepository.Artifacts.Add(ambiance with
            {
                Id = Guid.NewGuid(),
                StageRunId = orphanRun.Id,
                CreatedAtUtc = now,
                RelativePath = "artifacts/stems/orphan/ambiance.wav"
            });
        }

        TranscriptProjectState reopened = await scope.Service.OpenAsync(TestContext.Current.CancellationToken);
        StemAudioRoute route = TranscriptWorkflowUtilities.BuildStemAudioRoute(
            reopened.ProjectState.Artifacts, reopened.StageRuns.Where(r => r.StageName == StageNames.Separation).ToArray());
        ProjectArtifact selectedVocals = Assert.Single(reopened.ProjectState.Artifacts,
            a => a.RelativePath == route.AsrAudioRelativePath);
        ProjectArtifact selectedAmbiance = Assert.Single(reopened.ProjectState.Artifacts,
            a => a.RelativePath == reopened.MixSourceAudioRelativePath);
        Assert.Equal(selectedVocals.StageRunId, selectedAmbiance.StageRunId);
        Assert.Equal(vocals.StageRunId, selectedAmbiance.StageRunId);
        Assert.Equal(ambiance.RelativePath, reopened.MixSourceAudioRelativePath);
    }

    [Fact]
    public async Task CreateAsync_with_stem_separation_keeps_transcript_stages_on_the_full_mix()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        var transcriptionEngine = new RecordingAudioTranscriptionEngine();
        var diarizationEngine = new RecordingDiarizationEngine();
        FakeServiceScope scope = CreateScope(
            tempDirectory,
            diarizationEngine: diarizationEngine,
            transcriptionEngine: transcriptionEngine);

        TranscriptProjectState result = await CreateWithStemsAsync(
            scope,
            new CreateTranscriptProjectRequest(
                "Transcript Demo",
                sourcePath,
                EnableSpeakerDiarization: true));

        ProjectArtifact vocals = Assert.Single(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.Vocals);
        ProjectArtifact ambiance = Assert.Single(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.Ambiance);
        ProjectArtifact music = Assert.Single(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.Music);
        ProjectArtifact sfx = Assert.Single(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.SoundEffects);
        ProjectArtifact analysis = Assert.Single(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.AudioQualityAnalysis);
        Assert.Equal(12.0d, vocals.DurationSeconds);
        Assert.Equal(12.0d, ambiance.DurationSeconds);
        Assert.Equal(12.0d, music.DurationSeconds);
        Assert.Equal(12.0d, sfx.DurationSeconds);
        Assert.Equal("hash-vocals.wav", vocals.Sha256);
        Assert.Equal("hash-ambiance.wav", ambiance.Sha256);
        Assert.Equal("hash-music.wav", music.Sha256);
        Assert.Equal("hash-sfx.wav", sfx.Sha256);
        string vocalsPath = scope.ArtifactStore.GetPath(vocals.RelativePath);
        Assert.NotEqual(vocals.RelativePath, result.AsrAudioRelativePath);
        Assert.Equal(ambiance.RelativePath, result.MixSourceAudioRelativePath);
        Assert.NotEqual(vocalsPath, transcriptionEngine.LastAudioPath);
        Assert.NotEqual(vocalsPath, scope.SpeechRegionDetector.LastNormalizedAudioPath);
        Assert.NotEqual(vocalsPath, diarizationEngine.LastAudioPath);
        Assert.Single(scope.AudioQualityAnalyzer.Requests);
        Assert.NotNull(analysis);
        Assert.Equal(1, scope.StemSeparationEngine.CallCount);
        Assert.Contains(result.StageRuns, stageRun => stageRun.StageName == "separation" && stageRun.Status == StageRunStatus.Completed);
        Assert.Contains(result.StageRuns, stageRun => stageRun.StageName == StageNames.AudioPreparation && stageRun.Status == StageRunStatus.Completed);
        Assert.Null(result.StemSeparationWarning);
    }

    [Fact]
    public async Task CreateAsync_with_vocal_stem_does_not_analyze_it_for_transcript_preparation()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        var analyzer = new FakeAudioQualityAnalyzer();
        analyzer.QueueResult(new AudioQualityAnalysisResult(
            "full-mix.wav",
            CreateAudioQualityMetrics(SpeechAudioSourceKind.FullMix),
            AudioQualityAnalysisThresholds.ForSource(SpeechAudioSourceKind.FullMix),
            [],
            []));
        var transcriptionEngine = new RecordingAudioTranscriptionEngine();
        FakeServiceScope scope = CreateScope(
            tempDirectory,
            transcriptionEngine: transcriptionEngine,
            audioQualityAnalyzer: analyzer);

        TranscriptProjectState result = await CreateWithStemsAsync(
            scope,
            new CreateTranscriptProjectRequest(
                "Transcript Demo",
                sourcePath,
                EnableSpeakerDiarization: false));

        Assert.Contains(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.Vocals);
        Assert.Equal(ProjectArtifactPaths.NormalizedAudioRelativePath, result.AsrAudioRelativePath);
        Assert.Equal(scope.ArtifactStore.GetPath(ProjectArtifactPaths.NormalizedAudioRelativePath), transcriptionEngine.LastAudioPath);
        Assert.Equal(SpeechAudioSourceKind.FullMix, Assert.Single(analyzer.Requests).SourceKind);
    }

    [Fact]
    public async Task CreateAsync_without_stem_separation_uses_full_mix_audio()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        var transcriptionEngine = new RecordingAudioTranscriptionEngine();
        FakeServiceScope scope = CreateScope(tempDirectory, transcriptionEngine: transcriptionEngine);

        TranscriptProjectState result = await scope.Service.CreateAsync(
            new CreateTranscriptProjectRequest("Transcript Demo", sourcePath),
            TestContext.Current.CancellationToken);

        ProjectArtifact analysis = Assert.Single(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.AudioQualityAnalysis);
        Assert.DoesNotContain(result.ProjectState.Artifacts, artifact => artifact.Kind is ArtifactKind.Vocals or ArtifactKind.Ambiance);
        Assert.Equal(ProjectArtifactPaths.NormalizedAudioRelativePath, result.AsrAudioRelativePath);
        Assert.Equal(ProjectArtifactPaths.NormalizedAudioRelativePath, result.MixSourceAudioRelativePath);
        Assert.Equal(scope.ArtifactStore.GetPath(ProjectArtifactPaths.NormalizedAudioRelativePath), transcriptionEngine.LastAudioPath);
        Assert.Equal(scope.ArtifactStore.GetPath(ProjectArtifactPaths.NormalizedAudioRelativePath), scope.SpeechRegionDetector.LastNormalizedAudioPath);
        Assert.NotNull(analysis);
        Assert.Equal(0, scope.StemSeparationEngine.CallCount);
        Assert.Single(scope.AudioQualityAnalyzer.Requests);
        Assert.Null(result.StemSeparationWarning);
    }

    [Fact]
    public async Task RunStemSeparationAsync_preserves_prior_stems_and_selects_new_completed_run()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        FakeServiceScope scope = CreateScope(tempDirectory);
        TranscriptProjectState created = await CreateWithStemsAsync(
            scope,
            new CreateTranscriptProjectRequest(
                "Transcript Demo",
                sourcePath,
                EnableSpeakerDiarization: false));
        ProjectArtifact initialVocals = Assert.Single(created.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.Vocals);
        ProjectArtifact initialAmbiance = Assert.Single(created.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.Ambiance);
        ProjectArtifact initialMusic = Assert.Single(created.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.Music);
        ProjectArtifact initialSfx = Assert.Single(created.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.SoundEffects);

        TranscriptProjectState rerun = await scope.Service.RunStemSeparationAsync(TestContext.Current.CancellationToken);

        ProjectArtifact rerunVocals = Assert.IsType<ProjectArtifact>(TranscriptWorkflowUtilities.GetLatestAcceptedVocalStem(rerun.ProjectState.Artifacts, rerun.StageRuns));
        ProjectArtifact rerunAmbiance = Assert.IsType<ProjectArtifact>(TranscriptWorkflowUtilities.GetLatestAcceptedAmbianceStem(rerun.ProjectState.Artifacts, rerun.StageRuns));
        ProjectArtifact rerunMusic = Assert.Single(rerun.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.Music && artifact.StageRunId == rerunVocals.StageRunId);
        ProjectArtifact rerunSfx = Assert.Single(rerun.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.SoundEffects && artifact.StageRunId == rerunVocals.StageRunId);
        Assert.NotEqual(initialVocals.Id, rerunVocals.Id);
        Assert.Contains(initialVocals, rerun.ProjectState.Artifacts);
        Assert.NotEqual(initialAmbiance.Id, rerunAmbiance.Id);
        Assert.Contains(initialAmbiance, rerun.ProjectState.Artifacts);
        Assert.NotEqual(initialMusic.Id, rerunMusic.Id);
        Assert.Contains(initialMusic, rerun.ProjectState.Artifacts);
        Assert.NotEqual(initialSfx.Id, rerunSfx.Id);
        Assert.Contains(initialSfx, rerun.ProjectState.Artifacts);
        Assert.NotEqual(initialVocals.RelativePath, rerunVocals.RelativePath);
        Assert.NotEqual(initialAmbiance.RelativePath, rerunAmbiance.RelativePath);
        Assert.NotEqual(initialMusic.RelativePath, rerunMusic.RelativePath);
        Assert.NotEqual(initialSfx.RelativePath, rerunSfx.RelativePath);
        Assert.Equal(rerunVocals.StageRunId, rerunAmbiance.StageRunId);
        Assert.Equal(rerunAmbiance.RelativePath, rerun.MixSourceAudioRelativePath);
        Assert.NotEqual(rerunVocals.RelativePath, rerun.AsrAudioRelativePath);
        Assert.Equal(created.CurrentTranscriptRevision!.Id, rerun.CurrentTranscriptRevision!.Id);
        Assert.Equal(2, scope.StemSeparationEngine.CallCount);
    }

    [Fact]
    public async Task RunTranscriptStageAsync_passes_requested_source_language_to_asr()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        var transcriptionEngine = new RecordingAudioTranscriptionEngine();
        FakeServiceScope scope = CreateScope(tempDirectory, transcriptionEngine: transcriptionEngine);
        await scope.Service.CreateAsync(
            new CreateTranscriptProjectRequest(
                "Transcript Demo",
                sourcePath,
                EnableSpeakerDiarization: false),
            TestContext.Current.CancellationToken);

        await scope.Workspace.RunTranscriptStageAsync(
            StageNames.Asr,
            new TranscriptRunOptions
            {
                EnableSpeakerDiarization = false,
                SourceLanguage = "fr"
            },
            TestContext.Current.CancellationToken);

        Assert.Equal("fr", transcriptionEngine.LastSourceLanguage);
    }

    [Fact]
    public async Task RunStemSeparationAsync_never_regenerates_the_transcript()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        FakeServiceScope scope = CreateScope(tempDirectory);
        TranscriptProjectState created = await CreateWithStemsAsync(
            scope,
            new CreateTranscriptProjectRequest(
                "Transcript Demo",
                sourcePath,
                EnableSpeakerDiarization: false));

        TranscriptProjectState rerun = await scope.Service.RunStemSeparationAsync(TestContext.Current.CancellationToken);

        Assert.Equal(created.CurrentTranscriptRevision!.Id, rerun.CurrentTranscriptRevision!.Id);
        Assert.Equal(created.CurrentTranscriptRevision.RevisionNumber, rerun.CurrentTranscriptRevision.RevisionNumber);
        Assert.Equal(2, scope.StemSeparationEngine.CallCount);
    }

    [Fact]
    public async Task RunStemSeparationAsync_preserves_existing_diarized_assignments()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        var diarizationEngine = new RecordingDiarizationEngine(
        [
            new DiarizedSpeakerTurn("spk_0", 0d, 5.8d, Confidence: 0.9d, HasOverlap: false),
            new DiarizedSpeakerTurn("spk_1", 6d, 11.8d, Confidence: 0.8d, HasOverlap: false)
        ]);
        FakeServiceScope scope = CreateScope(tempDirectory, diarizationEngine: diarizationEngine);
        TranscriptProjectState created = await CreateWithStemsAsync(
            scope,
            new CreateTranscriptProjectRequest(
                "Transcript Demo",
                sourcePath,
                EnableSpeakerDiarization: true));
        Guid[] createdSpeakerIdsBySegment = created.TranscriptSegments
            .OrderBy(segment => segment.SegmentIndex)
            .Select(segment => segment.SpeakerId ?? Guid.Empty)
            .ToArray();

        TranscriptProjectState rerun = await scope.Service.RunStemSeparationAsync(TestContext.Current.CancellationToken);

        Assert.NotEmpty(created.SpeakerTurns);
        Assert.Equal(1, diarizationEngine.CallCount);
        Assert.Equal(created.CurrentTranscriptRevision!.Id, rerun.CurrentTranscriptRevision!.Id);
        Assert.Equal(created.CurrentTranscriptRevision.RevisionNumber, rerun.CurrentTranscriptRevision.RevisionNumber);
        Assert.Equal(
            createdSpeakerIdsBySegment,
            rerun.TranscriptSegments
                .OrderBy(segment => segment.SegmentIndex)
                .Select(segment => segment.SpeakerId ?? Guid.Empty)
                .ToArray());
        Assert.Equal(2, scope.StemSeparationEngine.CallCount);
    }

    [Fact]
    public async Task RunStemSeparationAsync_when_separation_fails_records_failure_and_keeps_full_mix_route()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        var transcriptionEngine = new RecordingAudioTranscriptionEngine();
        var stemEngine = new FakeStemSeparationEngine { ThrowOnSeparate = true };
        FakeServiceScope scope = CreateScope(
            tempDirectory,
            transcriptionEngine: transcriptionEngine,
            stemSeparationEngine: stemEngine);

        await scope.Service.CreateAsync(
            new CreateTranscriptProjectRequest(
                "Transcript Demo",
                sourcePath,
                EnableSpeakerDiarization: true),
            TestContext.Current.CancellationToken);
        await Assert.ThrowsAnyAsync<Exception>(
            () => scope.Service.RunStemSeparationAsync(TestContext.Current.CancellationToken));
        TranscriptProjectState result = await scope.Service.OpenAsync(TestContext.Current.CancellationToken);

        Assert.DoesNotContain(result.ProjectState.Artifacts, artifact => artifact.Kind is ArtifactKind.Vocals or ArtifactKind.Ambiance);
        ProjectArtifact analysis = Assert.Single(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.AudioQualityAnalysis);
        Assert.Equal(ProjectArtifactPaths.NormalizedAudioRelativePath, result.AsrAudioRelativePath);
        Assert.Equal(scope.ArtifactStore.GetPath(ProjectArtifactPaths.NormalizedAudioRelativePath), transcriptionEngine.LastAudioPath);
        Assert.NotNull(analysis);
        Assert.Contains(result.StageRuns, stageRun => stageRun.StageName == "separation" && stageRun.Status == StageRunStatus.Failed);
        Assert.Contains(result.StageRuns, stageRun => stageRun.StageName == StageNames.AudioPreparation && stageRun.Status == StageRunStatus.Completed);
        Assert.Equal("Dialogue isolation model unavailable; no clean ambiance track was generated.", result.StemSeparationWarning);
        ProjectArtifact degradation = Assert.Single(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.PipelineDegradation);
        Assert.Equal("DIALOGUE_ISOLATION_UNAVAILABLE", degradation.DegradationCode);
    }

    [Fact]
    public async Task OpenAsync_with_demucs_v4_stems_warns_as_legacy_and_uses_normalized_audio_routes()
    {
        string tempDirectory = CreateTempDirectory();
        string sourcePath = Path.Join(tempDirectory, "sample.mp4");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);

        var stemEngine = new FakeStemSeparationEngine
        {
            EngineFamily = "demucs-v4",
            Model = "demucs-v4",
            RawStemNames = ["drums", "bass", "other", "vocals"],
            WriteSoundEffects = false
        };
        FakeServiceScope scope = CreateScope(tempDirectory, stemSeparationEngine: stemEngine);

        TranscriptProjectState result = await CreateWithStemsAsync(
            scope,
            new CreateTranscriptProjectRequest(
                "Transcript Demo",
                sourcePath,
                EnableSpeakerDiarization: false));

        ProjectArtifact vocals = Assert.Single(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.Vocals);
        ProjectArtifact ambiance = Assert.Single(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.Ambiance);
        ProjectArtifact music = Assert.Single(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.Music);

        Assert.Contains("older/non-current separator", result.StemSeparationWarning, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(ProjectArtifactPaths.NormalizedAudioRelativePath, result.AsrAudioRelativePath);
        Assert.Equal(ProjectArtifactPaths.NormalizedAudioRelativePath, result.MixSourceAudioRelativePath);
        Assert.Contains("/demucs-v4/", vocals.RelativePath, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("/demucs-v4/", ambiance.RelativePath, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("/demucs-v4/", music.RelativePath, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(result.ProjectState.Artifacts, artifact => artifact.Kind == ArtifactKind.SoundEffects);
        Assert.Contains("generated-demucs-v4-vocals", vocals.Provenance, StringComparison.Ordinal);
        Assert.Contains("engine_family=demucs-v4", vocals.Provenance, StringComparison.Ordinal);
    }
}
