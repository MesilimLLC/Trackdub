using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.Projects;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Media;
using Trackdub.Domain.Projects;
using Trackdub.Domain.Tts;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.Transcript;

namespace Trackdub.Application.Tests;

public sealed class TranscriptWorkflowUtilitiesTests
{
    [Fact]
    public void Single_speaker_default_does_not_override_a_manual_unassignment()
    {
        var speaker = ProjectSpeaker.Create(Guid.NewGuid(), "Speaker 1", DateTimeOffset.UtcNow);
        var segment = TranscriptSegment.Create(Guid.NewGuid(), 0, 0, 1, "Hello",
            speakerAssignmentOrigin: SpeakerAssignmentOrigin.Manual);

        var actual = Assert.Single(TranscriptWorkflowUtilities.ApplySingleSpeakerDefaultAssignments([segment], [speaker]));
        Assert.Null(actual.SpeakerId);
        Assert.Equal(SpeakerAssignmentOrigin.Manual, actual.SpeakerAssignmentOrigin);
    }

    [Theory]
    [InlineData(StageRunStatus.Completed, true, true, true, true)]
    [InlineData(StageRunStatus.Failed, true, true, true, false)]
    [InlineData(StageRunStatus.Canceled, true, true, true, false)]
    [InlineData(StageRunStatus.Running, true, true, true, false)]
    [InlineData(StageRunStatus.Completed, false, true, true, false)]
    [InlineData(StageRunStatus.Completed, true, false, true, false)]
    [InlineData(StageRunStatus.Completed, true, true, false, false)]
    public void Stem_selection_requires_producer_evidence_and_both_required_outputs(
        StageRunStatus status, bool includeProducer, bool includeAmbiance, bool sameMedia, bool accepted)
    {
        MediaAsset media = CreateMediaAsset();
        StageRunRecord run = StageRunRecord.Start(media.ProjectId, StageNames.Separation, DateTimeOffset.UtcNow) with { Status = status };
        ProjectArtifact vocals = new(Guid.NewGuid(), media.ProjectId, media.Id, ArtifactKind.Vocals,
            "stems/vocals.wav", "vocals", 100, 1d, 44100, 1, DateTimeOffset.UtcNow,
            StageRunId: run.Id, Provenance: "generated-spleeter-vocals");
        ProjectArtifact ambiance = vocals with
        {
            Id = Guid.NewGuid(),
            Kind = ArtifactKind.Ambiance,
            RelativePath = "stems/ambiance.wav",
            MediaAssetId = sameMedia ? media.Id : Guid.NewGuid()
        };
        ProjectArtifact[] artifacts = includeAmbiance ? [vocals, ambiance] : [vocals];
        StageRunRecord[]? runs = includeProducer ? [run] : null;

        Assert.Equal(accepted ? vocals : null, TranscriptWorkflowUtilities.GetLatestAcceptedVocalStem(artifacts, runs));
        Assert.Equal(accepted ? ambiance : null, TranscriptWorkflowUtilities.GetLatestAcceptedAmbianceStem(artifacts, runs));
    }

    [Fact]
    public void Optional_stems_are_selected_only_from_the_accepted_required_stem_run()
    {
        MediaAsset media = CreateMediaAsset();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        StageRunRecord oldRun = StageRunRecord.Start(media.ProjectId, StageNames.Separation, now.AddMinutes(-2)).Complete(now.AddMinutes(-1));
        StageRunRecord newRun = StageRunRecord.Start(media.ProjectId, StageNames.Separation, now).Complete(now.AddSeconds(1));
        ProjectArtifact vocals = new(Guid.NewGuid(), media.ProjectId, media.Id, ArtifactKind.Vocals,
            "old/vocals.wav", "vocals", 100, 1d, 44100, 1, now.AddMinutes(-1),
            StageRunId: oldRun.Id, Provenance: "generated-spleeter-vocals");
        ProjectArtifact ambiance = vocals with { Id = Guid.NewGuid(), Kind = ArtifactKind.Ambiance, RelativePath = "old/ambiance.wav" };
        ProjectArtifact oldMusic = vocals with { Id = Guid.NewGuid(), Kind = ArtifactKind.Music, RelativePath = "old/music.wav" };
        ProjectArtifact oldSfx = vocals with { Id = Guid.NewGuid(), Kind = ArtifactKind.SoundEffects, RelativePath = "old/sfx.wav" };
        ProjectArtifact newVocals = vocals with { Id = Guid.NewGuid(), StageRunId = newRun.Id, RelativePath = "new/vocals.wav" };
        ProjectArtifact newAmbiance = ambiance with { Id = Guid.NewGuid(), StageRunId = newRun.Id, RelativePath = "new/ambiance.wav" };
        ProjectArtifact newMusic = oldMusic with { Id = Guid.NewGuid(), StageRunId = newRun.Id, RelativePath = "new/music.wav" };
        ProjectArtifact[] artifacts = [vocals, ambiance, oldMusic, oldSfx, newVocals, newAmbiance];

        IReadOnlyList<ProjectArtifact> selected = TranscriptWorkflowUtilities.GetAcceptedStemArtifacts(artifacts, [oldRun, newRun]);

        Assert.Equal(2, selected.Count);
        Assert.All(selected, artifact => Assert.Equal(newRun.Id, artifact.StageRunId));
        selected = TranscriptWorkflowUtilities.GetAcceptedStemArtifacts([.. artifacts, newMusic], [oldRun, newRun]);
        Assert.Contains(newMusic, selected);
        Assert.DoesNotContain(oldMusic, selected);
        Assert.DoesNotContain(oldSfx, selected);
    }

    [Fact]
    public void ResolveDetectedTranscriptLanguage_ignores_punctuation_only_detection()
    {
        RecognizedTranscriptSegment[] segments =
        [
            new(0, 0.0d, 2.0d, "。 。 。 。 。", DetectedLanguage: "zh")
        ];

        Assert.Null(TranscriptWorkflowUtilities.ResolveDetectedTranscriptLanguage(segments));
    }

    [Fact]
    public void ResolveDetectedTranscriptLanguage_ignores_cjk_detection_for_mostly_latin_text()
    {
        RecognizedTranscriptSegment[] segments =
        [
            new(0, 0.0d, 3.0d, "拉斯, las veremos en cine, television y video.", DetectedLanguage: "zh")
        ];

        Assert.Null(TranscriptWorkflowUtilities.ResolveDetectedTranscriptLanguage(segments));
    }

    [Fact]
    public void ResolveDetectedTranscriptLanguage_keeps_spanish_detection_for_latin_text()
    {
        RecognizedTranscriptSegment[] segments =
        [
            new(0, 0.0d, 3.0d, "Las veremos en cine, television y video.", DetectedLanguage: "es")
        ];

        Assert.Equal("es", TranscriptWorkflowUtilities.ResolveDetectedTranscriptLanguage(segments));
    }

    [Fact]
    public void BuildStemAudioRoute_accepts_current_spleeter_stems_for_asr_and_mix()
    {
        MediaAsset mediaAsset = CreateMediaAsset();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        ProjectArtifact normalized = new(
            Guid.NewGuid(),
            mediaAsset.ProjectId,
            mediaAsset.Id,
            ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath,
            "sha-normalized",
            100,
            mediaAsset.DurationSeconds,
            44100,
            2,
            now);
        ProjectArtifact vocals = new(
            Guid.NewGuid(),
            mediaAsset.ProjectId,
            mediaAsset.Id,
            ArtifactKind.Vocals,
            "artifacts/stems/spleeter/vocals.wav",
            "sha-vocals",
            100,
            mediaAsset.DurationSeconds,
            44100,
            1,
            now.AddSeconds(1),
            Provenance: "generated-spleeter-vocals;engine_family=spleeter;model=spleeter");
        ProjectArtifact ambiance = new(
            Guid.NewGuid(),
            mediaAsset.ProjectId,
            mediaAsset.Id,
            ArtifactKind.Ambiance,
            "artifacts/stems/spleeter/ambiance.wav",
            "sha-ambiance",
            100,
            mediaAsset.DurationSeconds,
            44100,
            1,
            now.AddSeconds(2),
            Provenance: "generated-spleeter-ambiance;engine_family=spleeter;model=spleeter");

        StageRunRecord run = StageRunRecord.Start(mediaAsset.ProjectId, StageNames.Separation, now)
            .Complete(now.AddSeconds(3));
        vocals = vocals with { StageRunId = run.Id };
        ambiance = ambiance with { StageRunId = run.Id };
        StemAudioRoute route = TranscriptWorkflowUtilities.BuildStemAudioRoute(
            [normalized, vocals, ambiance], [run]);

        Assert.Equal(vocals.RelativePath, route.AsrAudioRelativePath);
        Assert.Equal(ambiance.RelativePath, route.MixSourceAudioRelativePath);
        Assert.Null(route.WarningMessage);
    }

    [Fact]
    public void BuildStemAudioRoute_warns_when_only_legacy_hush_stems_exist()
    {
        MediaAsset mediaAsset = CreateMediaAsset();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        ProjectArtifact vocals = new(
            Guid.NewGuid(),
            mediaAsset.ProjectId,
            mediaAsset.Id,
            ArtifactKind.Vocals,
            "artifacts/stems/hush-dialogue/vocals.wav",
            "sha-vocals",
            100,
            mediaAsset.DurationSeconds,
            44100,
            1,
            now,
            Provenance: "generated-hush-dialogue-vocals;engine_family=hush-dialogue");

        StemAudioRoute route = TranscriptWorkflowUtilities.BuildStemAudioRoute([vocals], stageRuns: null);

        Assert.Equal(ProjectArtifactPaths.NormalizedAudioRelativePath, route.AsrAudioRelativePath);
        Assert.Equal(ProjectArtifactPaths.NormalizedAudioRelativePath, route.MixSourceAudioRelativePath);
        Assert.Contains("older/non-current separator", route.WarningMessage, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ResolveLipSynthesisDriverAudioArtifact_returns_current_export_audio()
    {
        DateTimeOffset takeCreated = DateTimeOffset.UtcNow.AddMinutes(-10);
        DateTimeOffset exportCreated = DateTimeOffset.UtcNow.AddMinutes(-5);
        MediaAsset mediaAsset = CreateMediaAsset();
        TtsTake completedTake = TtsTake.CreateStock(mediaAsset.ProjectId, Guid.NewGuid())
            .Complete(Guid.NewGuid(), Guid.NewGuid(), 44_100, 44_100, "fake", "model", "voice", null)
            with
        { CreatedAtUtc = takeCreated };
        ProjectArtifact exportAudio = CreateArtifact(
            mediaAsset,
            ArtifactKind.ExportAudio,
            "artifacts/export/audio.wav",
            exportCreated);
        TranscriptProjectState state = CreateProjectState(mediaAsset, [exportAudio], [completedTake]);

        ProjectArtifact? driver = TranscriptWorkflowUtilities.ResolveLipSynthesisDriverAudioArtifact(state);

        Assert.NotNull(driver);
        Assert.Equal(exportAudio.RelativePath, driver.RelativePath);
    }

    [Fact]
    public void ResolveLipSynthesisDriverAudioArtifact_returns_null_when_export_predates_latest_take()
    {
        DateTimeOffset exportCreated = DateTimeOffset.UtcNow.AddMinutes(-10);
        DateTimeOffset takeCreated = DateTimeOffset.UtcNow.AddMinutes(-5);
        MediaAsset mediaAsset = CreateMediaAsset();
        TtsTake completedTake = TtsTake.CreateStock(mediaAsset.ProjectId, Guid.NewGuid())
            .Complete(Guid.NewGuid(), Guid.NewGuid(), 44_100, 44_100, "fake", "model", "voice", null)
            with
        { CreatedAtUtc = takeCreated };
        ProjectArtifact exportAudio = CreateArtifact(
            mediaAsset,
            ArtifactKind.ExportAudio,
            "artifacts/export/audio.wav",
            exportCreated);
        TranscriptProjectState state = CreateProjectState(mediaAsset, [exportAudio], [completedTake]);

        Assert.Null(TranscriptWorkflowUtilities.ResolveLipSynthesisDriverAudioArtifact(state));
    }

    [Fact]
    public void ResolveLipSynthesisDriverAudioArtifact_returns_null_when_any_take_is_stale()
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        MediaAsset mediaAsset = CreateMediaAsset();
        TtsTake staleTake = TtsTake.CreateStock(mediaAsset.ProjectId, Guid.NewGuid()).MarkStale();
        ProjectArtifact exportAudio = CreateArtifact(
            mediaAsset,
            ArtifactKind.ExportAudio,
            "artifacts/export/audio.wav",
            now);
        TranscriptProjectState state = CreateProjectState(mediaAsset, [exportAudio], [staleTake]);

        Assert.Null(TranscriptWorkflowUtilities.ResolveLipSynthesisDriverAudioArtifact(state));
    }

    [Fact]
    public void BuildLipSynthesisExportCompositingWarning_returns_message_when_take_artifacts_are_not_composable()
    {
        MediaAsset mediaAsset = CreateMediaAsset();
        ProjectArtifact lipTake = CreateArtifact(
            mediaAsset,
            ArtifactKind.LipSynthesisTake,
            "artifacts/lipsynthesis/turn.mp4",
            DateTimeOffset.UtcNow);
        TranscriptProjectState state = CreateProjectState(mediaAsset, [lipTake], []);

        string? warning = TranscriptWorkflowUtilities.BuildLipSynthesisExportCompositingWarning(state);

        Assert.NotNull(warning);
        Assert.Contains("could not be composited", warning, StringComparison.OrdinalIgnoreCase);
    }

    private static TranscriptProjectState CreateProjectState(
        MediaAsset mediaAsset,
        IReadOnlyList<ProjectArtifact> artifacts,
        IReadOnlyList<TtsTake> ttsTakes)
    {
        var project = new TrackdubProject(mediaAsset.ProjectId, "Test", DateTimeOffset.UtcNow, DateTimeOffset.UtcNow);
        var projectState = new OpenProjectResult(
            project,
            mediaAsset,
            null,
            SourceMediaStatus.Available,
            null,
            artifacts,
            "en");

        return new TranscriptProjectState(
            projectState,
            CurrentTranscriptRevision: null,
            TranscriptSegments: [],
            Speakers: [],
            SpeakerTurns: [],
            CurrentTranslationRevision: null,
            TranslatedSegments: [],
            IsTranslationStale: false,
            TranscriptLanguage: "en",
            StageRuns: [],
            SupportedTargetLanguages: [],
            SelectedTranslationTargetLanguage: null,
            StaleTranslatedSegmentIndices: new HashSet<int>(),
            WaveformSummary: null,
            AvailableVoices: [],
            VoiceAssignments: [],
            TtsTakes: ttsTakes,
            TtsSegmentStates: [],
            VoiceAssignmentWarnings: []);
    }

    private static ProjectArtifact CreateArtifact(
        MediaAsset mediaAsset,
        ArtifactKind kind,
        string relativePath,
        DateTimeOffset createdAtUtc) =>
        new(
            Guid.NewGuid(),
            mediaAsset.ProjectId,
            mediaAsset.Id,
            kind,
            relativePath,
            "sha",
            100,
            mediaAsset.DurationSeconds,
            44_100,
            2,
            createdAtUtc);

    private static MediaAsset CreateMediaAsset()
    {
        Guid projectId = Guid.NewGuid();
        Guid mediaAssetId = Guid.NewGuid();
        return new MediaAsset(
            mediaAssetId,
            projectId,
            @"C:\media\sample.mp4",
            "sample.mp4",
            "media-hash",
            1024,
            DateTimeOffset.UtcNow,
            "mp4",
            12d,
            HasAudio: true,
            HasVideo: true,
            DateTimeOffset.UtcNow);
    }
}
