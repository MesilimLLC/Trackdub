using Trackdub.Application.Transcripts;
using Trackdub.Contracts;
using Trackdub.Contracts.Projects;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.AudioQuality;
using Trackdub.Domain.Media;
using Trackdub.Domain.Projects;
using Trackdub.Domain.StageRuns;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

/// <summary>
/// Pins the preparation layer of the pipeline-lifecycle stack: a canceled or failed
/// audio-preparation run must leave no registered artifacts behind, consumers and the
/// resume gate must only accept output from a completed preparation producer, and a
/// later canceled run must not shadow an earlier completed one.
/// </summary>
public sealed class SpeechAudioPreparationProvenanceTests
{
    [Fact]
    public async Task HandleAsync_cancellation_after_processed_registration_removes_artifacts_and_records_canceled()
    {
        MediaAsset mediaAsset = CreateMediaAsset();
        ProjectArtifact normalized = CreateArtifact(mediaAsset, ArtifactKind.NormalizedAudio, ProjectArtifactPaths.NormalizedAudioRelativePath);
        var analyzer = new FakeAudioQualityAnalyzer();
        analyzer.QueueResult(CreateAnalysis([AudioQualityDefectKind.Rumble]));
        var innerRepository = new FakeMediaAssetRepository();
        innerRepository.Seed(mediaAsset);
        using var cancellation = new CancellationTokenSource();
        var repository = new CancellingMediaRepository(innerRepository, cancellation);
        var stageRunStore = new FakeProjectStageRunStore();
        var handler = new SpeechAudioPreparationStageHandler(
            analyzer,
            new SpeechAudioPreparationPlanner(),
            new FakeSpeechAudioProcessingService(),
            new FakeArtifactStore(),
            new FakeFileFingerprintService(),
            repository,
            stageRunStore);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            handler.HandleAsync(
                new SpeechAudioPreparationStageRequest(mediaAsset.ProjectId, mediaAsset, normalized, [normalized]),
                cancellation.Token));

        Assert.DoesNotContain(innerRepository.Artifacts, artifact => artifact.Kind == ArtifactKind.SpeechProcessedAudio);
        Assert.DoesNotContain(innerRepository.Artifacts, artifact => artifact.Kind == ArtifactKind.AudioQualityAnalysis);
        Assert.Equal(StageRunStatus.Canceled, Assert.Single(stageRunStore.All).Status);
    }

    [Fact]
    public async Task HandleAsync_failure_after_processed_registration_removes_artifacts_and_records_failed()
    {
        MediaAsset mediaAsset = CreateMediaAsset();
        ProjectArtifact normalized = CreateArtifact(mediaAsset, ArtifactKind.NormalizedAudio, ProjectArtifactPaths.NormalizedAudioRelativePath);
        var analyzer = new FakeAudioQualityAnalyzer();
        analyzer.QueueResult(CreateAnalysis([AudioQualityDefectKind.Rumble]));
        var mediaRepository = new FakeMediaAssetRepository();
        mediaRepository.Seed(mediaAsset);
        var stageRunStore = new FakeProjectStageRunStore();
        var handler = new SpeechAudioPreparationStageHandler(
            analyzer,
            new SpeechAudioPreparationPlanner(),
            new FakeSpeechAudioProcessingService { ThrowOnCallNumber = 2 },
            new FakeArtifactStore(),
            new FakeFileFingerprintService(),
            mediaRepository,
            stageRunStore);

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            handler.HandleAsync(
                new SpeechAudioPreparationStageRequest(mediaAsset.ProjectId, mediaAsset, normalized, [normalized]),
                TestContext.Current.CancellationToken));

        Assert.DoesNotContain(mediaRepository.Artifacts, artifact => artifact.Kind == ArtifactKind.SpeechProcessedAudio);
        Assert.DoesNotContain(mediaRepository.Artifacts, artifact => artifact.Kind == ArtifactKind.AudioQualityAnalysis);
        Assert.Equal(StageRunStatus.Failed, Assert.Single(stageRunStore.All).Status);
    }

    [Fact]
    public void GetAcceptedSpeechProcessedAudio_ignores_canceled_producer()
    {
        Guid projectId = Guid.NewGuid();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        StageRunRecord completedRun = StageRunRecord
            .Start(projectId, StageNames.AudioPreparation, now)
            .Complete(now.AddMinutes(1));
        StageRunRecord canceledRun = StageRunRecord
            .Start(projectId, StageNames.AudioPreparation, now.AddMinutes(2))
            .Cancel(now.AddMinutes(3), "canceled");
        ProjectArtifact completedArtifact = CreateProcessedArtifact(projectId, completedRun.Id, now.AddMinutes(1));
        ProjectArtifact canceledArtifact = CreateProcessedArtifact(projectId, canceledRun.Id, now.AddMinutes(3));

        ProjectArtifact? accepted = TranscriptWorkflowUtilities.GetAcceptedSpeechProcessedAudio(
            [completedArtifact, canceledArtifact],
            [completedRun, canceledRun]);

        Assert.Equal(completedArtifact.Id, accepted?.Id);
    }

    [Fact]
    public void GetAcceptedSpeechProcessedAudio_returns_null_without_completed_producer_evidence()
    {
        Guid projectId = Guid.NewGuid();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        StageRunRecord canceledRun = StageRunRecord
            .Start(projectId, StageNames.AudioPreparation, now)
            .Cancel(now.AddMinutes(1), "canceled");
        ProjectArtifact canceledArtifact = CreateProcessedArtifact(projectId, canceledRun.Id, now.AddMinutes(1));

        Assert.Null(TranscriptWorkflowUtilities.GetAcceptedSpeechProcessedAudio([canceledArtifact], [canceledRun]));
        Assert.Null(TranscriptWorkflowUtilities.GetAcceptedSpeechProcessedAudio([canceledArtifact], stageRuns: null));
    }

    [Fact]
    public void GetAcceptedSpeechProcessedAudio_returns_null_when_latest_completed_producer_has_no_output()
    {
        Guid projectId = Guid.NewGuid();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        StageRunRecord earlierRun = StageRunRecord
            .Start(projectId, StageNames.AudioPreparation, now)
            .Complete(now.AddMinutes(1));
        StageRunRecord latestRun = StageRunRecord
            .Start(projectId, StageNames.AudioPreparation, now.AddMinutes(2))
            .Complete(now.AddMinutes(3));
        ProjectArtifact earlierArtifact = CreateProcessedArtifact(projectId, earlierRun.Id, now.AddMinutes(1));

        ProjectArtifact? accepted = TranscriptWorkflowUtilities.GetAcceptedSpeechProcessedAudio(
            [earlierArtifact],
            [earlierRun, latestRun]);

        Assert.Null(accepted);
    }

    [Fact]
    public void GetAcceptedSpeechEnhancedAudio_ignores_failed_producer()
    {
        Guid projectId = Guid.NewGuid();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        StageRunRecord completedRun = StageRunRecord
            .Start(projectId, StageNames.SpeechEnhancement, now)
            .Complete(now.AddMinutes(1));
        StageRunRecord failedRun = StageRunRecord
            .Start(projectId, StageNames.SpeechEnhancement, now.AddMinutes(2))
            .Fail(now.AddMinutes(3), "failed");
        ProjectArtifact completedArtifact = CreateEnhancedArtifact(projectId, completedRun.Id, now.AddMinutes(1));
        ProjectArtifact failedArtifact = CreateEnhancedArtifact(projectId, failedRun.Id, now.AddMinutes(3));

        ProjectArtifact? accepted = TranscriptWorkflowUtilities.GetAcceptedSpeechEnhancedAudio(
            [completedArtifact, failedArtifact],
            [completedRun, failedRun]);

        Assert.Equal(completedArtifact.Id, accepted?.Id);
        Assert.Null(TranscriptWorkflowUtilities.GetAcceptedSpeechEnhancedAudio([failedArtifact], [failedRun]));
        Assert.Null(TranscriptWorkflowUtilities.GetAcceptedSpeechEnhancedAudio([failedArtifact], stageRuns: null));
    }

    [Fact]
    public void ResolveAsrAudioArtifact_does_not_fall_back_to_a_failed_enhancement_producer() =>
        AssertResolvedAsrSource(EnhancementOutcome.Failed);

    [Fact]
    public void ResolveAsrAudioArtifact_prefers_native_studio_voice_enhancement() =>
        AssertResolvedAsrSource(EnhancementOutcome.Completed);

    private static void AssertResolvedAsrSource(EnhancementOutcome outcome)
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        Guid projectId = Guid.NewGuid();
        Guid mediaAssetId = Guid.NewGuid();
        StageRunRecord run = StageRunRecord
            .Start(projectId, StageNames.SpeechEnhancement, now);
        run = outcome == EnhancementOutcome.Completed
            ? run.Complete(now.AddMinutes(1))
            : run.Fail(now.AddMinutes(1), "failed");
        ProjectArtifact normalized = new(
            Guid.NewGuid(), projectId, mediaAssetId, ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath, "hash", 100, 12.0d, 48000, 1, now);
        ProjectArtifact enhanced = CreateEnhancedArtifact(projectId, run.Id, now.AddMinutes(1)) with
        {
            ProjectId = projectId,
            MediaAssetId = mediaAssetId,
            Provenance = outcome == EnhancementOutcome.Completed
                ? $"generated-nvidia-afx-speech-enhancement:{normalized.Id:D}:profile=studio_voice_high_quality"
                : "speech-enhancement",
        };

        ProjectArtifact? resolved = TranscriptWorkflowUtilities.ResolveAsrAudioArtifact(
            [normalized, enhanced],
            [run]);

        Assert.Equal(
            outcome == EnhancementOutcome.Completed ? enhanced.Id : normalized.Id,
            resolved?.Id);
    }

    private enum EnhancementOutcome
    {
        Failed,
        Completed,
    }

    [Fact]
    public void ResolveAsrAudioArtifact_uses_earlier_completed_run_when_latest_preparation_run_canceled()
    {
        Guid projectId = Guid.NewGuid();
        Guid mediaAssetId = Guid.NewGuid();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        StageRunRecord completedRun = StageRunRecord
            .Start(projectId, StageNames.AudioPreparation, now)
            .Complete(now.AddMinutes(1));
        StageRunRecord canceledRun = StageRunRecord
            .Start(projectId, StageNames.AudioPreparation, now.AddMinutes(2))
            .Cancel(now.AddMinutes(3), "canceled");
        var normalized = new ProjectArtifact(
            Guid.NewGuid(), projectId, mediaAssetId, ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath, "hash", 100, 12.0d, 48000, 1, now);
        var analysis = new ProjectArtifact(
            Guid.NewGuid(), projectId, mediaAssetId, ArtifactKind.AudioQualityAnalysis,
            ProjectArtifactPaths.GetAudioQualityAnalysisRelativePath(completedRun.Id), "hash", 100, null, null, null,
            now.AddMinutes(1),
            StageRunId: completedRun.Id,
            Provenance: "audio-quality-analysis:policy=v1;catalog=v1;selectedSource=FullMix");
        var processed = new ProjectArtifact(
            Guid.NewGuid(), projectId, mediaAssetId, ArtifactKind.SpeechProcessedAudio,
            ProjectArtifactPaths.GetSpeechProcessedAudioRelativePath(completedRun.Id, "asr"), "hash", 100, 12.0d, 48000, 1,
            now.AddMinutes(1),
            StageRunId: completedRun.Id,
            Provenance: "speech-processing:source=x;stage=asr;profile=p;version=1;hash=h;analysis=y");

        ProjectArtifact? resolved = TranscriptWorkflowUtilities.ResolveAsrAudioArtifact(
            [normalized, analysis, processed],
            [completedRun, canceledRun]);

        Assert.Equal(processed.Id, resolved?.Id);
    }

    [Fact]
    public async Task CanResumeStage_returns_false_when_processed_file_is_missing()
    {
        PreparationResumeFixture fixture = await PreparationResumeFixture.CreateAsync(includeProcessedRow: true, seedProcessedFile: false);

        bool canResume = StageArtifactResumeEvaluator.CanResumeStage(
            fixture.State,
            fixture.ArtifactStore,
            StageNames.AudioPreparation,
            new Dictionary<string, string>(),
            "project-root");

        Assert.False(canResume);
    }

    [Fact]
    public async Task CanResumeStageAsync_returns_false_when_audit_expects_a_missing_processed_row()
    {
        PreparationResumeFixture fixture = await PreparationResumeFixture.CreateAsync(includeProcessedRow: false, seedProcessedFile: false);

        bool canResume = await StageArtifactResumeEvaluator.CanResumeStageAsync(
            fixture.State,
            fixture.ArtifactStore,
            StageNames.AudioPreparation,
            new Dictionary<string, string>(),
            "project-root",
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.False(canResume);
    }

    [Fact]
    public async Task CanResumeStageAsync_returns_true_for_a_coherent_completed_producer()
    {
        PreparationResumeFixture fixture = await PreparationResumeFixture.CreateAsync(includeProcessedRow: true, seedProcessedFile: true);

        bool canResume = await StageArtifactResumeEvaluator.CanResumeStageAsync(
            fixture.State,
            fixture.ArtifactStore,
            StageNames.AudioPreparation,
            new Dictionary<string, string>(),
            "project-root",
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(canResume);
    }

    private static ProjectArtifact CreateProcessedArtifact(Guid projectId, Guid stageRunId, DateTimeOffset createdAtUtc) =>
        new(
            Guid.NewGuid(),
            projectId,
            Guid.NewGuid(),
            ArtifactKind.SpeechProcessedAudio,
            ProjectArtifactPaths.GetSpeechProcessedAudioRelativePath(stageRunId, "asr"),
            "hash",
            100,
            12.0d,
            48000,
            1,
            createdAtUtc,
            StageRunId: stageRunId,
            Provenance: "speech-processing:source=x;stage=asr;profile=p;version=1;hash=h;analysis=y");

    private static ProjectArtifact CreateEnhancedArtifact(Guid projectId, Guid stageRunId, DateTimeOffset createdAtUtc) =>
        new(
            Guid.NewGuid(),
            projectId,
            Guid.NewGuid(),
            ArtifactKind.SpeechEnhancedAudio,
            ProjectArtifactPaths.GetSpeechEnhancedAudioRelativePath(stageRunId),
            "hash",
            100,
            12.0d,
            48000,
            1,
            createdAtUtc,
            StageRunId: stageRunId,
            Provenance: "speech-enhancement");

    private static AudioQualityAnalysisResult CreateAnalysis(IReadOnlyList<AudioQualityDefectKind> defects) =>
        new(
            "virtual.wav",
            new AudioQualityMetrics(
                DurationSeconds: 12.0d,
                PeakDbfs: -6.0d,
                RmsDbfs: -24.0d,
                ActiveRmsDbfs: -20.0d,
                Lufs: null,
                AudioQualityAnalysisConfidence.High,
                SpeechAudioSourceKind.FullMix,
                ClippedSamplePercent: 0.0d,
                NearSilencePercent: 0.0d,
                DcOffset: 0.0d,
                RumbleRatioDb: -30.0d,
                HissRatioDb: -10.0d,
                SpeechBandRatioDb: -3.0d,
                CrestFactorDb: 18.0d,
                DynamicRangeDb: 12.0d,
                NoiseFloorDbfs: -50.0d,
                SnrDb: 30.0d,
                AudioSnrConfidence.Reliable),
            AudioQualityAnalysisThresholds.ForSource(SpeechAudioSourceKind.FullMix),
            defects,
            []);

    private static MediaAsset CreateMediaAsset()
    {
        Guid projectId = Guid.NewGuid();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        return new MediaAsset(
            Guid.NewGuid(),
            projectId,
            "source.mp4",
            "source.mp4",
            "source-hash",
            100,
            now,
            "mp4",
            12.0d,
            HasAudio: true,
            HasVideo: true,
            now);
    }

    private static ProjectArtifact CreateArtifact(MediaAsset mediaAsset, ArtifactKind kind, string relativePath) =>
        new(
            Guid.NewGuid(),
            mediaAsset.ProjectId,
            mediaAsset.Id,
            kind,
            relativePath,
            "hash",
            100,
            mediaAsset.DurationSeconds,
            48000,
            1,
            DateTimeOffset.UtcNow);

    private sealed class CancellingMediaRepository(
        FakeMediaAssetRepository inner,
        CancellationTokenSource cancellation) : IMediaAssetRepository
    {
        public Task SaveAsync(MediaAsset asset, CancellationToken token) => inner.SaveAsync(asset, token);
        public Task UpdateSourcePathAsync(Guid id, string path, string name, CancellationToken token) =>
            inner.UpdateSourcePathAsync(id, path, name, token);
        [Obsolete("Single-asset compatibility member.")]
        public Task<MediaAsset?> GetPrimaryAsync(Guid projectId, CancellationToken token) =>
            inner.GetPrimaryAsync(projectId, token);
        public Task<IReadOnlyList<MediaAsset>> GetAllAsync(Guid projectId, CancellationToken token) =>
            inner.GetAllAsync(projectId, token);
        public async Task SaveArtifactAsync(ProjectArtifact artifact, CancellationToken token)
        {
            await inner.SaveArtifactAsync(artifact, token);
            if (artifact.Kind == ArtifactKind.SpeechProcessedAudio)
            {
                cancellation.Cancel();
            }
        }
        public Task DeleteArtifactAsync(Guid id, CancellationToken token) => inner.DeleteArtifactAsync(id, token);
        public Task<IReadOnlyList<ProjectArtifact>> GetArtifactsAsync(Guid projectId, CancellationToken token) =>
            inner.GetArtifactsAsync(projectId, token);
        public Task<ProjectArtifact?> GetArtifactByIdAsync(Guid id, CancellationToken token) =>
            inner.GetArtifactByIdAsync(id, token);
    }

    private sealed record PreparationResumeFixture(
        TranscriptProjectState State,
        FakeArtifactStore ArtifactStore)
    {
        public static async Task<PreparationResumeFixture> CreateAsync(bool includeProcessedRow, bool seedProcessedFile)
        {
            DateTimeOffset now = DateTimeOffset.UtcNow;
            Guid projectId = Guid.NewGuid();
            Guid mediaAssetId = Guid.NewGuid();
            var project = new TrackdubProject(projectId, "Preparation", now, now);
            var mediaAsset = new MediaAsset(
                mediaAssetId,
                projectId,
                "source.mp4",
                "source.mp4",
                "source-hash",
                100,
                now,
                "mp4",
                12.0d,
                HasAudio: true,
                HasVideo: true,
                now);
            StageRunRecord run = StageRunRecord
                .Start(projectId, StageNames.AudioPreparation, now)
                .Complete(now.AddMinutes(1));
            string analysisRelativePath = ProjectArtifactPaths.GetAudioQualityAnalysisRelativePath(run.Id);
            var analysisArtifact = new ProjectArtifact(
                Guid.NewGuid(), projectId, mediaAssetId, ArtifactKind.AudioQualityAnalysis,
                analysisRelativePath, "hash", 100, null, null, null, now.AddMinutes(1),
                StageRunId: run.Id,
                Provenance: "audio-quality-analysis:policy=v1;catalog=v1;selectedSource=FullMix");
            ProjectArtifact processedArtifact = CreateProcessedArtifact(projectId, run.Id, now.AddMinutes(1)) with
            {
                ProjectId = projectId,
                MediaAssetId = mediaAssetId,
            };

            var artifactStore = new FakeArtifactStore();
            var audit = new SpeechAudioPreparationAudit(
                projectId,
                mediaAssetId,
                "v1",
                "v1",
                new SpeechAudioSourceAudit(
                    Guid.NewGuid(),
                    ProjectArtifactPaths.NormalizedAudioRelativePath,
                    "hash",
                    ArtifactKind.NormalizedAudio,
                    CreateAnalysis([])),
                SpeechAudioSourceKind.FullMix,
                [
                    new SpeechAudioDecisionAudit(
                        SpeechPipelineStageKind.Asr,
                        "profile",
                        1,
                        "hash",
                        "chain",
                        RequiresProcessing: true,
                        [],
                        processedArtifact.Id,
                        processedArtifact.RelativePath,
                        FallbackReason: null,
                        ProcessedAnalysis: null),
                ]);
            await artifactStore.WriteJsonAsync(analysisRelativePath, audit, CancellationToken.None);
            if (seedProcessedFile)
            {
                artifactStore.Seed(processedArtifact.RelativePath, [1, 2, 3]);
            }

            var artifacts = new List<ProjectArtifact> { analysisArtifact };
            if (includeProcessedRow)
            {
                artifacts.Add(processedArtifact);
            }

            var openResult = new OpenProjectResult(
                project,
                mediaAsset,
                new SourceMediaReference(
                    mediaAsset.SourceFilePath,
                    mediaAsset.SourceFileName,
                    new FileFingerprint(mediaAsset.FingerprintSha256, mediaAsset.SourceSizeBytes, mediaAsset.SourceLastWriteTimeUtc),
                    new MediaProbeSnapshot(
                        "mp4",
                        "MP4",
                        12.0d,
                        BitRate: null,
                        AudioStreams: [new MediaAudioStream(0, "aac", Channels: 2, SampleRate: 48000, 12.0d)],
                        VideoStreams: [],
                        SubtitleStreams: []),
                    DateTimeOffset.UtcNow),
                SourceMediaStatus.Available,
                SourceStatusMessage: null,
                Artifacts: artifacts,
                TranscriptLanguage: "en");
            var state = new TranscriptProjectState(
                openResult,
                CurrentTranscriptRevision: null,
                TranscriptSegments: [],
                Speakers: [],
                SpeakerTurns: [],
                CurrentTranslationRevision: null,
                TranslatedSegments: [],
                IsTranslationStale: false,
                TranscriptLanguage: "en",
                StageRuns: [run],
                SupportedTargetLanguages: [],
                SelectedTranslationTargetLanguage: null,
                StaleTranslatedSegmentIndices: new HashSet<int>(),
                WaveformSummary: null,
                AvailableVoices: [],
                VoiceAssignments: [],
                TtsTakes: [],
                TtsSegmentStates: [],
                VoiceAssignmentWarnings: []);

            return new PreparationResumeFixture(state, artifactStore);
        }
    }
}
