using Trackdub.Contracts;
using Trackdub.Application.Mixing;
using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Mixing;
using Trackdub.Domain.Projects;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Trackdub.Domain.Tts;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed class ExportStageHandlerTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Rollback_failure_preserves_original_completion_cause_and_lifecycle(bool canceled)
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        byte[] priorVideo = [9, 8, 7];
        File.WriteAllBytes(outputPath, priorVideo);
        string manifestPath = Path.ChangeExtension(outputPath, ".export-manifest.json");
        File.WriteAllBytes(manifestPath, [6, 5, 4]);
        Exception original = canceled
            ? new OperationCanceledException("Original completion cancellation.", new CancellationToken(canceled: true))
            : new IOException("Original completion failure.");
        string? blockedRestorePath = null;
        var stageStore = new CompletionFailureStageRunStore(stageRun =>
        {
            blockedRestorePath = Path.Join(Path.GetDirectoryName(outputPath), $".lesson-dub.export-manifest.json.{stageRun.Id:N}.tmp");
            Directory.CreateDirectory(blockedRestorePath);
            throw original;
        });
        var artifactStore = new FakeArtifactStore(temp.Path);
        ExportStageHandler handler = CreateHandler(artifactStore, new FakeMixRenderer(), new FakeLoudnessNormalizer(),
            new FakeExportRenderer(), new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(), stageStore);
        try
        {
            if (canceled)
            {
                OperationCanceledException exception = await Assert.ThrowsAsync<OperationCanceledException>(() =>
                    handler.ExportAsync(context.State, new ExportStageRequest(context.Project.Id, outputPath, []), TestContext.Current.CancellationToken));
                Assert.Same(original, exception);
            }
            else
            {
                ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() =>
                    handler.ExportAsync(context.State, new ExportStageRequest(context.Project.Id, outputPath, []), TestContext.Current.CancellationToken));
                Assert.Same(original, exception.InnerException);
            }
            StageRunRecord run = Assert.Single(stageStore.Inner.All);
            Assert.Equal(canceled ? StageRunStatus.Canceled : StageRunStatus.Failed, run.Status);
            ExportFailureReport? report = await artifactStore.ReadJsonAsync<ExportFailureReport>(
                ProjectArtifactPaths.GetExportFailureReportRelativePath(run.Id), TestContext.Current.CancellationToken);
            Assert.NotNull(report);
            Assert.Contains(report.Causes, cause => cause.Code == "export-recovery-error");
            Assert.Contains(report.Causes, cause => cause.Message == original.Message);
            Assert.Single(Directory.EnumerateFiles(Path.GetDirectoryName(outputPath)!, "*.export-delivery.json"));
            Assert.NotEmpty(Directory.EnumerateFiles(Path.GetDirectoryName(outputPath)!, "*.bak"));
        }
        finally
        {
            if (blockedRestorePath is not null) Directory.Delete(blockedRestorePath);
        }
        ExportDeliveryTransaction.Recover(outputPath);
        Assert.Equal(priorVideo, File.ReadAllBytes(outputPath));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Recovery_rejection_records_failed_attempt_and_structured_diagnostics(bool activeWriter)
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        File.WriteAllBytes(outputPath, [9, 8, 7]);
        string replacement = Path.Join(temp.Path, "replacement.mp4");
        File.WriteAllBytes(replacement, [1, 2, 3]);
        using var transaction = new ExportDeliveryTransaction([(outputPath, replacement)], Guid.NewGuid());
        string journalPath = Path.Join(Path.GetDirectoryName(outputPath), ".lesson-dub.export-delivery.json");
        if (activeWriter)
        {
            await transaction.PrepareAsync(TestContext.Current.CancellationToken);
            transaction.Apply();
        }
        else File.WriteAllText(journalPath, "{\"Version\":2}");
        byte[] journalBytes = File.ReadAllBytes(journalPath);
        byte[] videoBytes = File.ReadAllBytes(outputPath);
        var stageStore = new FakeProjectStageRunStore();
        ExportStageHandler handler = CreateHandler(new FakeArtifactStore(temp.Path), new FakeMixRenderer(), new FakeLoudnessNormalizer(),
            new FakeExportRenderer(), new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(), stageStore);
        try
        {
            ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() => handler.ExportAsync(
                context.State, new ExportStageRequest(context.Project.Id, outputPath, []), TestContext.Current.CancellationToken));
            Assert.Contains(exception.Report.Causes, cause => cause.Code == "export-recovery-error");
            Assert.Equal(StageRunStatus.Failed, Assert.Single(stageStore.All).Status);
            Assert.True(File.Exists(GetFailureReportPath(outputPath)));
            Assert.Equal(journalBytes, File.ReadAllBytes(journalPath));
            Assert.Equal(videoBytes, File.ReadAllBytes(outputPath));
        }
        finally
        {
            if (activeWriter) transaction.Rollback();
        }
    }

    [Fact]
    public async Task Completion_failure_restores_prior_delivery_and_sidecars()
    {
        await AssertPriorDeliverySurvivesCompletionFailureAsync(cancel: false);
    }

    [Fact]
    public async Task Cancellation_after_swap_restores_prior_delivery()
    {
        await AssertPriorDeliverySurvivesCompletionFailureAsync(cancel: true);
    }

    private static async Task AssertPriorDeliverySurvivesCompletionFailureAsync(bool cancel)
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");
        string manifestPath = Path.ChangeExtension(outputPath, ".export-manifest.json");
        string subtitlePath = Path.ChangeExtension(outputPath, ".srt");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        byte[] priorVideoBytes = [9, 8, 7, 6];
        byte[] priorManifestBytes = [5, 4, 3, 2];
        byte[] priorSubtitleBytes = [1, 8, 1];
        File.WriteAllBytes(outputPath, priorVideoBytes);
        File.WriteAllBytes(manifestPath, priorManifestBytes);
        File.WriteAllBytes(subtitlePath, priorSubtitleBytes);
        bool failCompletion = true;
        var store = new CompletionFailureStageRunStore(_ =>
        {
            if (!failCompletion) return;
            Assert.NotEqual(priorVideoBytes, File.ReadAllBytes(outputPath));
            Assert.NotEqual(priorManifestBytes, File.ReadAllBytes(manifestPath));
            Assert.NotEqual(priorSubtitleBytes, File.ReadAllBytes(subtitlePath));
            if (cancel) throw new OperationCanceledException("Canceled after delivery swap.");
            throw new IOException("Completion database write failed.");
        });
        ExportStageHandler handler = CreateHandler(new FakeArtifactStore(temp.Path), new FakeMixRenderer(),
            new FakeLoudnessNormalizer(), new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(), store);
        var request = new ExportStageRequest(context.Project.Id, outputPath, [ExportSubtitleFormat.Srt]);

        // Retry must preserve the prior accepted bytes on every unsuccessful attempt.
        for (int attempt = 0; attempt < 2; attempt++)
        {
            if (cancel)
                await Assert.ThrowsAnyAsync<OperationCanceledException>(() => handler.ExportAsync(context.State, request, TestContext.Current.CancellationToken));
            else
            {
                ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() => handler.ExportAsync(context.State, request, TestContext.Current.CancellationToken));
                Assert.IsType<IOException>(exception.InnerException);
            }
            Assert.Equal(priorVideoBytes, File.ReadAllBytes(outputPath));
            Assert.Equal(priorManifestBytes, File.ReadAllBytes(manifestPath));
            Assert.Equal(priorSubtitleBytes, File.ReadAllBytes(subtitlePath));
            Assert.DoesNotContain(store.Inner.All, run => run.Status == StageRunStatus.Completed);
        }
        failCompletion = false;
        ExportStageResult accepted = await handler.ExportAsync(context.State, request, TestContext.Current.CancellationToken);
        byte[] acceptedVideo = File.ReadAllBytes(outputPath);
        byte[] acceptedManifest = File.ReadAllBytes(manifestPath);
        byte[] acceptedSubtitles = File.ReadAllBytes(subtitlePath);
        ExportDeliveryTransaction.Recover(outputPath);
        Assert.NotEqual(priorVideoBytes, acceptedVideo);
        Assert.NotEqual(priorManifestBytes, acceptedManifest);
        Assert.NotEqual(priorSubtitleBytes, acceptedSubtitles);
        Assert.Equal(acceptedVideo, File.ReadAllBytes(outputPath));
        Assert.Equal(acceptedManifest, File.ReadAllBytes(manifestPath));
        Assert.Equal(acceptedSubtitles, File.ReadAllBytes(subtitlePath));
        Assert.Equal(accepted.StageRun.Id, Assert.Single(store.Inner.All, run => run.Status == StageRunStatus.Completed).Id);
    }

    private sealed class CompletionFailureStageRunStore(Action<StageRunRecord> onCompletion) : IProjectStageRunStore
    {
        public FakeProjectStageRunStore Inner { get; } = new();
        public Task CreateAsync(StageRunRecord stageRun, CancellationToken cancellationToken) => Inner.CreateAsync(stageRun, cancellationToken);
        public Task<IReadOnlyList<StageRunRecord>> ListByProjectAsync(Guid projectId, CancellationToken cancellationToken) => Inner.ListByProjectAsync(projectId, cancellationToken);
        public Task UpdateAsync(StageRunRecord stageRun, CancellationToken cancellationToken)
        {
            if (stageRun.Status == StageRunStatus.Completed) onCompletion(stageRun);
            return Inner.UpdateAsync(stageRun, cancellationToken);
        }
    }

    [Fact]
    public void ExportStageRequest_can_carry_match_original_loudness_and_pan_restoration()
    {
        var request = new ExportStageRequest(
            Guid.NewGuid(),
            "output.mp4",
            [],
            MatchOriginalLoudness: true,
            RestoreOriginalPan: true);

        Assert.True(request.MatchOriginalLoudness);
        Assert.True(request.RestoreOriginalPan);
    }

    [Fact]
    public async Task ExportAsync_returns_blocked_result_when_free_tier_duration_exceeds_gate()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true, durationSeconds: 600d);
        var tierGate = new FakeExportTierGate(
            requiresWatermark: false,
            durationBlockReason: "Free tier limits export to 5 minutes.");
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer(),
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(600d) },
            new FakeMediaAssetRepository(),
            new FakeProjectStageRunStore(),
            exportTierGate: tierGate);

        ExportStageResult result = await handler.ExportAsync(
            context.State,
            new ExportStageRequest(context.Project.Id, Path.Join(temp.Path, "out.mp4"), []),
            TestContext.Current.CancellationToken);

        Assert.True(result.IsBlocked);
        Assert.Contains("5 minutes", result.BlockedReason, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ExportAsync_passes_watermark_requirements_to_export_plan_for_free_tier()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var capturingRenderer = new CapturingExportRenderer();
        var tierGate = new FakeExportTierGate(requiresWatermark: true);
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer(),
            capturingRenderer,
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds, height: 1080) },
            new FakeMediaAssetRepository(),
            new FakeProjectStageRunStore(),
            exportTierGate: tierGate);

        await handler.ExportAsync(
            context.State,
            new ExportStageRequest(
                context.Project.Id,
                Path.Join(temp.Path, "delivery", "lesson-dub.mp4"),
                [],
                Container: ExportOutputContainer.Mp4),
            TestContext.Current.CancellationToken);

        Assert.NotNull(capturingRenderer.LastPlan);
        Assert.True(capturingRenderer.LastPlan.RequiresWatermark);
        Assert.Equal(1080, capturingRenderer.LastPlan.OutputHeight);
    }

    [Fact]
    public async Task ExportAsync_renders_full_mix_normalizes_muxes_and_writes_sidecars()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var mixRenderer = new FakeMixRenderer();
        var loudnessNormalizer = new FakeLoudnessNormalizer { AchievedLufs = -13.8d };
        var exportRenderer = new FakeExportRenderer();
        var mediaProbe = new FakeMediaProbe
        {
            Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds)
        };
        var mediaAssetRepository = new FakeMediaAssetRepository();
        var artifactStore = new FakeArtifactStore(temp.Path);
        ExportStageHandler handler = CreateHandler(
            artifactStore,
            mixRenderer,
            loudnessNormalizer,
            exportRenderer,
            mediaProbe,
            mediaAssetRepository,
            new FakeProjectStageRunStore());
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        await File.WriteAllTextAsync(GetFailureReportPath(outputPath), "stale failure", TestContext.Current.CancellationToken);

        ExportStageResult result = await handler.ExportAsync(
            context.State,
            new ExportStageRequest(
                context.Project.Id,
                outputPath,
                [ExportSubtitleFormat.Srt, ExportSubtitleFormat.Vtt],
                BurnInSubtitles: true,
                TargetLufs: -14d,
                Container: ExportOutputContainer.Mp4,
                RestoreOriginalPan: true),
            TestContext.Current.CancellationToken);

        Assert.True(File.Exists(result.OutputPath));
        Assert.True(File.Exists(Path.ChangeExtension(outputPath, ".srt")));
        Assert.True(File.Exists(Path.ChangeExtension(outputPath, ".vtt")));
        Assert.True(File.Exists(result.ManifestPath));
        Assert.False(File.Exists(GetFailureReportPath(outputPath)));
        Assert.False(File.Exists(artifactStore.GetPath(ProjectArtifactPaths.GetExportSubtitleRelativePath(result.StageRun.Id, ".srt"))));
        Assert.False(File.Exists(artifactStore.GetPath(ProjectArtifactPaths.GetExportSubtitleRelativePath(result.StageRun.Id, ".vtt"))));
        Assert.False(File.Exists(artifactStore.GetPath(ProjectArtifactPaths.GetExportSubtitleRelativePath(result.StageRun.Id, ".burnin.ass"))));
        byte[] srtBytes = File.ReadAllBytes(Path.ChangeExtension(outputPath, ".srt"));
        Assert.False(srtBytes.Take(3).SequenceEqual(new byte[] { 0xEF, 0xBB, 0xBF }));
        PreviewRangeRenderRequest mixRequest = Assert.Single(mixRenderer.Calls);
        Assert.Equal(0d, mixRequest.StartSeconds);
        Assert.Equal(context.MediaAsset.DurationSeconds, mixRequest.EndSeconds);
        Assert.True(mixRequest.MixPlan.RestoreOriginalPan);
        Assert.Equal(-14d, Assert.Single(loudnessNormalizer.Calls).TargetLufs);
        ExportPlan exportPlan = Assert.Single(exportRenderer.Calls);
        Assert.Equal(ExportOutputContainer.Mp4, exportPlan.Container);
        Assert.NotNull(exportPlan.BurnInSubtitlePath);
        Assert.Equal("en", exportPlan.SourceLanguage);
        Assert.Equal("es", exportPlan.TargetLanguage);
        Assert.Contains(mediaAssetRepository.Artifacts, artifact => artifact.Kind == ArtifactKind.ExportManifest);
        Assert.Contains(mediaAssetRepository.Artifacts, artifact => artifact.Kind == ArtifactKind.ExportAudio);
        Assert.Contains(mediaAssetRepository.Artifacts, artifact => artifact.Kind == ArtifactKind.ExportVideo);

        ExportManifest? manifest = await artifactStore.ReadJsonAsync<ExportManifest>(
            ProjectArtifactPaths.GetExportManifestRelativePath(result.StageRun.Id),
            TestContext.Current.CancellationToken);

        Assert.NotNull(manifest);
        Assert.Equal(context.Project.Id, manifest!.ProjectId);
        Assert.Equal(result.StageRun.Id, manifest.ExportStageRunId);
        Assert.Equal(-14d, manifest.Loudness?.TargetLufs);
        Assert.Equal(-13.8d, manifest.Loudness?.AchievedLufs);
        Assert.Contains("kokoro", manifest.ModelIds);
        Assert.Contains("af_heart", manifest.TtsVoices);
        Assert.Contains(context.UpstreamStageRun.Id, manifest.StageRunIds);
        Assert.Contains(result.StageRun.Id, manifest.StageRunIds);
        Assert.Contains(manifest.Outputs, output => output.Kind == "video" && output.Path == Path.GetFileName(outputPath));
        Assert.Contains(
            manifest.Outputs,
            output => output.Kind == "audio" &&
                      output.PathBase == ExportManifestOutputPathBases.Artifact &&
                      output.Path.StartsWith("artifacts/", StringComparison.Ordinal));
        Assert.DoesNotContain(manifest.Outputs, output => string.IsNullOrWhiteSpace(output.PathBase));
        Assert.DoesNotContain(manifest.Outputs, output => Path.IsPathRooted(output.Path));
        Assert.Equal(StageRunStatus.Completed, result.StageRun.Status);
    }

    [Fact]
    public async Task ExportAsync_recomposes_lip_synthesis_takes_before_mux()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var mixRenderer = new FakeMixRenderer();
        var loudnessNormalizer = new FakeLoudnessNormalizer { AchievedLufs = -13.8d };
        var exportRenderer = new FakeExportRenderer();
        var videoRecomposer = new FakeVideoRecomposer();
        var mediaProbe = new FakeMediaProbe
        {
            Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds)
        };
        var mediaAssetRepository = new FakeMediaAssetRepository();
        var artifactStore = new FakeArtifactStore(temp.Path);

        Guid turnId = Guid.NewGuid();
        Guid speakerId = Guid.NewGuid();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        StageRunRecord lipRun = StageRunRecord.Start(context.Project.Id, StageNames.LipSynthesis, now).Complete(now);
        const string relativeClip = "artifacts/lipsynthesis/turn.mp4";
        artifactStore.Seed(relativeClip, [0, 0, 0, 12]);

        var lipArtifact = new ProjectArtifact(
            Guid.NewGuid(),
            context.Project.Id,
            context.MediaAsset.Id,
            ArtifactKind.LipSynthesisTake,
            relativeClip,
            "hash",
            12,
            2.0d,
            null,
            null,
            now,
            StageRunId: lipRun.Id,
            Provenance: $"lipsynthesis:turn:{turnId:N}");

        TranscriptProjectState state = context.State with
        {
            SpeakerTurns = [new SpeakerTurn(turnId, context.Project.Id, speakerId, 1.0d, 3.0d)],
            StageRuns = [.. context.State.StageRuns, lipRun],
            ProjectState = context.State.ProjectState with
            {
                Artifacts = [.. context.State.ProjectState.Artifacts, lipArtifact]
            }
        };

        ExportStageHandler handler = CreateHandler(
            artifactStore,
            mixRenderer,
            loudnessNormalizer,
            exportRenderer,
            mediaProbe,
            mediaAssetRepository,
            new FakeProjectStageRunStore(),
            videoRecomposer: videoRecomposer);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        await handler.ExportAsync(
            state,
            new ExportStageRequest(
                context.Project.Id,
                outputPath,
                [],
                BurnInSubtitles: false,
                TargetLufs: -14d,
                Container: ExportOutputContainer.Mp4),
            TestContext.Current.CancellationToken);

        (ResolvedVideoRecompositionPlan Plan, string OutputPath) recomposeCall = Assert.Single(videoRecomposer.Calls);
        Assert.Single(recomposeCall.Plan.PatchedTurns);
        ExportPlan exportPlan = Assert.Single(exportRenderer.Calls);
        Assert.Equal(recomposeCall.OutputPath, exportPlan.SourceMediaPath);
    }

    [Fact]
    public async Task ExportAsync_falls_back_to_original_video_when_lip_recomposition_fails()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var mixRenderer = new FakeMixRenderer();
        var loudnessNormalizer = new FakeLoudnessNormalizer { AchievedLufs = -13.8d };
        var exportRenderer = new FakeExportRenderer();
        var videoRecomposer = new ThrowingVideoRecomposer();
        var mediaProbe = new FakeMediaProbe
        {
            Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds)
        };
        var mediaAssetRepository = new FakeMediaAssetRepository();
        var artifactStore = new FakeArtifactStore(temp.Path);

        Guid turnId = Guid.NewGuid();
        Guid speakerId = Guid.NewGuid();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        StageRunRecord lipRun = StageRunRecord.Start(context.Project.Id, StageNames.LipSynthesis, now).Complete(now);
        const string relativeClip = "artifacts/lipsynthesis/turn.mp4";
        artifactStore.Seed(relativeClip, [0, 0, 0, 12]);

        var lipArtifact = new ProjectArtifact(
            Guid.NewGuid(),
            context.Project.Id,
            context.MediaAsset.Id,
            ArtifactKind.LipSynthesisTake,
            relativeClip,
            "hash",
            12,
            2.0d,
            null,
            null,
            now,
            StageRunId: lipRun.Id,
            Provenance: $"lipsynthesis:turn:{turnId:N}");

        TranscriptProjectState state = context.State with
        {
            SpeakerTurns = [new SpeakerTurn(turnId, context.Project.Id, speakerId, 1.0d, 3.0d)],
            StageRuns = [.. context.State.StageRuns, lipRun],
            ProjectState = context.State.ProjectState with
            {
                Artifacts = [.. context.State.ProjectState.Artifacts, lipArtifact]
            }
        };

        ExportStageHandler handler = CreateHandler(
            artifactStore,
            mixRenderer,
            loudnessNormalizer,
            exportRenderer,
            mediaProbe,
            mediaAssetRepository,
            new FakeProjectStageRunStore(),
            videoRecomposer: videoRecomposer);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        ExportStageResult result = await handler.ExportAsync(
            state,
            new ExportStageRequest(
                context.Project.Id,
                outputPath,
                [],
                BurnInSubtitles: false,
                TargetLufs: -14d,
                Container: ExportOutputContainer.Mp4),
            TestContext.Current.CancellationToken);

        Assert.Equal(StageRunStatus.Completed, result.StageRun.Status);
        ExportPlan exportPlan = Assert.Single(exportRenderer.Calls);
        Assert.Equal(context.MediaAsset.SourceFilePath, exportPlan.SourceMediaPath);
        Assert.Contains(result.Warnings, warning => warning.Contains("could not be composited", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task ExportAsync_when_matching_original_loudness_analyzes_source_and_raw_mix_then_normalizes_to_matched_target()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var mixRenderer = new FakeMixRenderer();
        var loudnessNormalizer = new FakeLoudnessNormalizer { AchievedLufs = -17.9d };
        loudnessNormalizer.EnqueueAnalysisResult(-18d);
        loudnessNormalizer.EnqueueAnalysisResult(-23d);
        var artifactStore = new FakeArtifactStore(temp.Path);
        ExportStageHandler handler = CreateHandler(
            artifactStore,
            mixRenderer,
            loudnessNormalizer,
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            new FakeProjectStageRunStore());
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageResult result = await handler.ExportAsync(
            context.State,
            new ExportStageRequest(
                context.Project.Id,
                outputPath,
                [],
                TargetLufs: -14d,
                MatchOriginalLoudness: true),
            TestContext.Current.CancellationToken);

        Assert.Equal(context.MediaAsset.SourceFilePath, loudnessNormalizer.AnalysisCalls[0].InputPath);
        Assert.Equal(Assert.Single(mixRenderer.Calls).OutputPath, loudnessNormalizer.AnalysisCalls[1].InputPath);
        LoudnessNormalizationRequest normalizeRequest = Assert.Single(loudnessNormalizer.Calls);
        Assert.Equal(loudnessNormalizer.AnalysisCalls[1].InputPath, normalizeRequest.InputPath);
        Assert.Equal(-18d, normalizeRequest.TargetLufs);
        Assert.Empty(result.Warnings);

        ExportManifest? manifest = await artifactStore.ReadJsonAsync<ExportManifest>(
            ProjectArtifactPaths.GetExportManifestRelativePath(result.StageRun.Id),
            TestContext.Current.CancellationToken);
        Assert.NotNull(manifest);
        Assert.Equal(-18d, manifest!.Loudness?.TargetLufs);
        Assert.Empty(manifest.Warnings);
    }

    [Fact]
    public async Task ExportAsync_when_matching_original_loudness_caps_upward_boost_and_warns()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var loudnessNormalizer = new FakeLoudnessNormalizer { AchievedLufs = -21d };
        loudnessNormalizer.EnqueueAnalysisResult(-12d);
        loudnessNormalizer.EnqueueAnalysisResult(-30d);
        var artifactStore = new FakeArtifactStore(temp.Path);
        ExportStageHandler handler = CreateHandler(
            artifactStore,
            new FakeMixRenderer(),
            loudnessNormalizer,
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            new FakeProjectStageRunStore());
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageResult result = await handler.ExportAsync(
            context.State,
            new ExportStageRequest(
                context.Project.Id,
                outputPath,
                [],
                TargetLufs: -14d,
                MatchOriginalLoudness: true),
            TestContext.Current.CancellationToken);

        Assert.Equal(-21d, Assert.Single(loudnessNormalizer.Calls).TargetLufs);
        string warning = Assert.Single(result.Warnings);
        Assert.Contains("9 dB", warning, StringComparison.OrdinalIgnoreCase);

        ExportManifest? manifest = await artifactStore.ReadJsonAsync<ExportManifest>(
            ProjectArtifactPaths.GetExportManifestRelativePath(result.StageRun.Id),
            TestContext.Current.CancellationToken);
        Assert.NotNull(manifest);
        Assert.Equal(-21d, manifest!.Loudness?.TargetLufs);
        Assert.Contains(manifest.Warnings, value => value.Contains("9 dB", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task ExportAsync_when_matching_original_loudness_falls_back_to_request_target_when_analysis_is_non_finite()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var loudnessNormalizer = new FakeLoudnessNormalizer { AchievedLufs = -14d };
        loudnessNormalizer.EnqueueAnalysisResult(-18d);
        loudnessNormalizer.EnqueueAnalysisResult(double.NegativeInfinity);
        var artifactStore = new FakeArtifactStore(temp.Path);
        ExportStageHandler handler = CreateHandler(
            artifactStore,
            new FakeMixRenderer(),
            loudnessNormalizer,
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            new FakeProjectStageRunStore());
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageResult result = await handler.ExportAsync(
            context.State,
            new ExportStageRequest(
                context.Project.Id,
                outputPath,
                [],
                TargetLufs: -14d,
                MatchOriginalLoudness: true),
            TestContext.Current.CancellationToken);

        Assert.Equal(-14d, Assert.Single(loudnessNormalizer.Calls).TargetLufs);
        string warning = Assert.Single(result.Warnings);
        Assert.Contains("match original loudness", warning, StringComparison.OrdinalIgnoreCase);

        ExportManifest? manifest = await artifactStore.ReadJsonAsync<ExportManifest>(
            ProjectArtifactPaths.GetExportManifestRelativePath(result.StageRun.Id),
            TestContext.Current.CancellationToken);
        Assert.NotNull(manifest);
        Assert.Equal(-14d, manifest!.Loudness?.TargetLufs);
        Assert.Contains(manifest.Warnings, value => value.Contains("match original loudness", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task ExportAsync_when_matching_original_loudness_falls_back_to_request_target_when_analysis_fails()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var loudnessNormalizer = new FakeLoudnessNormalizer { AchievedLufs = -23d };
        loudnessNormalizer.EnqueueAnalysisFailure(new InvalidOperationException("ffmpeg analysis failed."));
        var artifactStore = new FakeArtifactStore(temp.Path);
        ExportStageHandler handler = CreateHandler(
            artifactStore,
            new FakeMixRenderer(),
            loudnessNormalizer,
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            new FakeProjectStageRunStore());
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageResult result = await handler.ExportAsync(
            context.State,
            new ExportStageRequest(
                context.Project.Id,
                outputPath,
                [],
                TargetLufs: -23d,
                MatchOriginalLoudness: true),
            TestContext.Current.CancellationToken);

        Assert.Equal(context.MediaAsset.SourceFilePath, Assert.Single(loudnessNormalizer.AnalysisCalls).InputPath);
        Assert.Equal(-23d, Assert.Single(loudnessNormalizer.Calls).TargetLufs);
        Assert.Contains(result.Warnings, value => value.Contains("ffmpeg analysis failed", StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task ExportAsync_records_only_contributing_stage_runs_in_manifest()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        StageRunRecord unrelatedRun = StageRunRecord
            .Start(context.Project.Id, StageNames.Translation, DateTimeOffset.UtcNow)
            .Fail(DateTimeOffset.UtcNow.AddSeconds(1), "Older failed translation.");
        TranscriptProjectState state = context.State with
        {
            StageRuns = [.. context.State.StageRuns, unrelatedRun]
        };
        var artifactStore = new FakeArtifactStore(temp.Path);
        ExportStageHandler handler = CreateHandler(
            artifactStore,
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer { AchievedLufs = -13.8d },
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            new FakeProjectStageRunStore());
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageResult result = await handler.ExportAsync(
            state,
            new ExportStageRequest(context.Project.Id, outputPath, []),
            TestContext.Current.CancellationToken);

        ExportManifest? manifest = await artifactStore.ReadJsonAsync<ExportManifest>(
            ProjectArtifactPaths.GetExportManifestRelativePath(result.StageRun.Id),
            TestContext.Current.CancellationToken);

        Assert.NotNull(manifest);
        Assert.Contains(context.UpstreamStageRun.Id, manifest!.StageRunIds);
        Assert.Contains(result.StageRun.Id, manifest.StageRunIds);
        Assert.DoesNotContain(unrelatedRun.Id, manifest.StageRunIds);
    }

    [Fact]
    public async Task ExportAsync_records_stereo_export_audio_artifact_when_original_mix_is_stereo()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true, normalizedAudioChannelCount: 2);
        var artifactStore = new FakeArtifactStore(temp.Path);
        var mediaAssetRepository = new FakeMediaAssetRepository();
        var mixRenderer = new FakeMixRenderer();
        ExportStageHandler handler = CreateHandler(
            artifactStore,
            mixRenderer,
            new FakeLoudnessNormalizer { AchievedLufs = -13.8d },
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            mediaAssetRepository,
            new FakeProjectStageRunStore());
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        await handler.ExportAsync(
            context.State,
            new ExportStageRequest(context.Project.Id, outputPath, []),
            TestContext.Current.CancellationToken);

        Assert.Equal(2, mixRenderer.LastMixPlan!.OutputChannelCount);
        ProjectArtifact audioArtifact = Assert.Single(
            mediaAssetRepository.Artifacts,
            artifact => artifact.Kind == ArtifactKind.ExportAudio);
        Assert.Equal(2, audioArtifact.ChannelCount);
    }

    [Fact]
    public async Task ExportAsync_updates_manifest_artifact_fingerprint_after_final_manifest_rewrite()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var artifactStore = new FakeArtifactStore(temp.Path);
        var mediaAssetRepository = new FakeMediaAssetRepository();
        var fingerprintService = new ManifestSequenceFingerprintService();
        ExportStageHandler handler = CreateHandler(
            artifactStore,
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer { AchievedLufs = -13.8d },
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            mediaAssetRepository,
            new FakeProjectStageRunStore(),
            fingerprintService);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageResult result = await handler.ExportAsync(
            context.State,
            new ExportStageRequest(context.Project.Id, outputPath, []),
            TestContext.Current.CancellationToken);

        ProjectArtifact manifestArtifact = Assert.Single(
            mediaAssetRepository.Artifacts,
            artifact => artifact.Kind == ArtifactKind.ExportManifest);
        Assert.Equal("final-manifest-hash", manifestArtifact.Sha256);
        Assert.Equal(20, manifestArtifact.SizeBytes);
        Assert.Equal(2, fingerprintService.ManifestCallCount);
        Assert.Equal(StageRunStatus.Completed, result.StageRun.Status);
    }

    [Fact]
    public async Task ExportAsync_excludes_stale_translated_segments_from_subtitles()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        TranslatedSegment staleSegment = TranslatedSegment.Create(
            context.State.CurrentTranslationRevision!.Id,
            5,
            5.0d,
            6.0d,
            "Stale subtitle.");
        TranscriptProjectState state = context.State with
        {
            TranslatedSegments = [.. context.State.TranslatedSegments, staleSegment],
            IsTranslationStale = true,
            StaleTranslatedSegmentIndices = new HashSet<int> { staleSegment.SegmentIndex }
        };
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer { AchievedLufs = -13.8d },
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            new FakeProjectStageRunStore());
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageResult result = await handler.ExportAsync(
            state,
            new ExportStageRequest(
                context.Project.Id,
                outputPath,
                [ExportSubtitleFormat.Srt],
                SubtitleSource: ExportSubtitleSource.Translated),
            TestContext.Current.CancellationToken);

        string srt = await File.ReadAllTextAsync(Path.ChangeExtension(result.OutputPath, ".srt"), TestContext.Current.CancellationToken);
        Assert.Contains("Hola", srt, StringComparison.Ordinal);
        Assert.DoesNotContain("Stale subtitle.", srt, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ExportAsync_excludes_translated_segments_marked_stale_from_subtitles()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        DateTimeOffset now = DateTimeOffset.UtcNow;
        Guid speakerId = context.State.TranscriptSegments[0].SpeakerId!.Value;
        TranscriptSegment freshTranscriptSegment = TranscriptSegment.Create(
            context.State.CurrentTranscriptRevision!.Id,
            1,
            2.0d,
            3.0d,
            "Goodbye",
            speakerId,
            "en");
        TranslatedSegment freshTranslatedSegment = TranslatedSegment.Create(
            context.State.CurrentTranslationRevision!.Id,
            1,
            2.0d,
            3.0d,
            "Adios");
        ProjectArtifact freshTakeArtifact = CreateArtifact(
            context.Project.Id,
            context.MediaAsset.Id,
            ArtifactKind.TtsTake,
            "artifacts/tts/take-0002.wav",
            durationSeconds: 1.0d,
            now);
        TtsTake freshTake = TtsTake
            .Create(context.Project.Id, Guid.NewGuid(), freshTranslatedSegment.Id, segmentIndex: 1)
            .Complete(
                freshTakeArtifact.Id,
                stageRunId: Guid.NewGuid(),
                durationSamples: 48000,
                sampleRate: 48000,
                provider: "fake",
                modelId: "kokoro",
                voiceId: "af_heart",
                durationOverrunRatio: null);
        TranscriptProjectState state = context.State with
        {
            ProjectState = context.State.ProjectState with
            {
                Artifacts = [.. context.State.ProjectState.Artifacts, freshTakeArtifact]
            },
            TranscriptSegments = [.. context.State.TranscriptSegments, freshTranscriptSegment],
            TranslatedSegments = [.. context.State.TranslatedSegments, freshTranslatedSegment],
            TtsTakes = [.. context.State.TtsTakes, freshTake],
            IsTranslationStale = true,
            StaleTranslatedSegmentIndices = new HashSet<int> { 0 }
        };
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer { AchievedLufs = -13.8d },
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            new FakeProjectStageRunStore());
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageResult result = await handler.ExportAsync(
            state,
            new ExportStageRequest(
                context.Project.Id,
                outputPath,
                [ExportSubtitleFormat.Srt],
                SubtitleSource: ExportSubtitleSource.Translated),
            TestContext.Current.CancellationToken);

        string srt = await File.ReadAllTextAsync(Path.ChangeExtension(result.OutputPath, ".srt"), TestContext.Current.CancellationToken);
        Assert.Contains("Adios", srt, StringComparison.Ordinal);
        Assert.DoesNotContain("Hola", srt, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ExportAsync_writes_bilingual_subtitles_when_requested()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer { AchievedLufs = -13.8d },
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            new FakeProjectStageRunStore());
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageResult result = await handler.ExportAsync(
            context.State,
            new ExportStageRequest(
                context.Project.Id,
                outputPath,
                [ExportSubtitleFormat.Srt],
                SubtitleSource: ExportSubtitleSource.Bilingual),
            TestContext.Current.CancellationToken);

        string srt = await File.ReadAllTextAsync(Path.ChangeExtension(result.OutputPath, ".srt"), TestContext.Current.CancellationToken);
        Assert.Contains($"Hello{Environment.NewLine}Hola", srt, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ExportAsync_reports_missing_takes_before_starting_loudness_or_mux()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: false);
        var mixRenderer = new FakeMixRenderer();
        var loudnessNormalizer = new FakeLoudnessNormalizer();
        var exportRenderer = new FakeExportRenderer();
        var stageRunStore = new FakeProjectStageRunStore();
        var mediaAssetRepository = new FakeMediaAssetRepository();
        var artifactStore = new FakeArtifactStore(temp.Path);
        ExportStageHandler handler = CreateHandler(
            artifactStore,
            mixRenderer,
            loudnessNormalizer,
            exportRenderer,
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            mediaAssetRepository,
            stageRunStore);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() =>
            handler.ExportAsync(
                context.State,
                new ExportStageRequest(context.Project.Id, outputPath, []),
                TestContext.Current.CancellationToken));

        ExportFailureCause cause = Assert.Single(exception.Report.Causes);
        Assert.Equal("missing-take", cause.Code);
        Assert.Equal(0, cause.SegmentIndex);
        Assert.Empty(mixRenderer.Calls);
        Assert.Empty(loudnessNormalizer.Calls);
        Assert.Empty(exportRenderer.Calls);
        Assert.Empty(mediaAssetRepository.Artifacts);
        Assert.False(File.Exists(artifactStore.GetPath(ProjectArtifactPaths.GetExportManifestRelativePath(exception.Report.StageRunId))));
        Assert.True(File.Exists(GetFailureReportPath(outputPath)));
        Assert.Equal(StageRunStatus.Failed, Assert.Single(stageRunStore.All).Status);
    }

    [Fact]
    public async Task ExportAsync_marks_stage_failed_when_failure_report_write_fails()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: false);
        var artifactStore = new FakeArtifactStore(temp.Path)
        {
            FailingJsonWriteFileName = "export-failure.json"
        };
        var stageRunStore = new FakeProjectStageRunStore();
        ExportStageHandler handler = CreateHandler(
            artifactStore,
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer(),
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            stageRunStore);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() =>
            handler.ExportAsync(
                context.State,
                new ExportStageRequest(context.Project.Id, outputPath, []),
                TestContext.Current.CancellationToken));

        Assert.Equal("missing-take", Assert.Single(exception.Report.Causes).Code);
        Assert.IsType<IOException>(exception.InnerException);
        Assert.Equal(StageRunStatus.Failed, Assert.Single(stageRunStore.All).Status);
    }

    [Fact]
    public async Task ExportAsync_reports_take_exceeding_source_audio_before_rendering()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true, durationSeconds: 6.0d);
        // Make the take overrun the source: even moved back from 1.0s to 0, a 7.5s take ends 1.5s
        // past the 6.0s source.
        var state = context.State;
        var artifacts = state.ProjectState.Artifacts
            .Select(artifact => artifact.Kind == ArtifactKind.TtsTake
                ? artifact with { DurationSeconds = 7.5d }
                : artifact)
            .ToList();
        state = state with { ProjectState = state.ProjectState with { Artifacts = artifacts } };
        var mixRenderer = new FakeMixRenderer();
        var loudnessNormalizer = new FakeLoudnessNormalizer();
        var exportRenderer = new FakeExportRenderer();
        var stageRunStore = new FakeProjectStageRunStore();
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            mixRenderer,
            loudnessNormalizer,
            exportRenderer,
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(6.0d) },
            new FakeMediaAssetRepository(),
            stageRunStore);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() =>
            handler.ExportAsync(
                state,
                new ExportStageRequest(context.Project.Id, outputPath, []),
                TestContext.Current.CancellationToken));

        ExportFailureCause cause = Assert.Single(exception.Report.Causes);
        Assert.Equal("take-exceeds-source-audio", cause.Code);
        Assert.Contains("past the end", cause.Message);
        Assert.Empty(mixRenderer.Calls);
        Assert.Empty(loudnessNormalizer.Calls);
        Assert.Empty(exportRenderer.Calls);
        Assert.Equal(StageRunStatus.Failed, Assert.Single(stageRunStore.All).Status);
    }

    [Fact]
    public async Task ExportAsync_fits_a_take_that_overhangs_the_source_audio_instead_of_failing()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true, durationSeconds: 6.0d);
        // A 6.5s take cannot start before 0, so it ends 0.5s past the 6.0s source.
        TranscriptProjectState state = WithTakeDuration(context.State, 6.5d);
        SeedTakeFile(temp.Path);
        var artifactStore = new FakeArtifactStore(temp.Path);
        var stretchService = new FakeAudioTimeStretchService();
        var mixRenderer = new FakeMixRenderer();
        ExportStageHandler handler = CreateHandler(
            artifactStore,
            mixRenderer,
            new FakeLoudnessNormalizer(),
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(6.0d) },
            new FakeMediaAssetRepository(),
            new FakeProjectStageRunStore(),
            tailOverhangFitter: new TailOverhangFitter(stretchService, artifactStore));
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageResult result = await handler.ExportAsync(
            state,
            new ExportStageRequest(context.Project.Id, outputPath, []),
            TestContext.Current.CancellationToken);

        Assert.Equal(StageRunStatus.Completed, result.StageRun.Status);
        double expectedDuration = 6.5d - 0.5d - TailOverhangFitter.SafetyMarginSeconds;
        Assert.Equal(6.5d / expectedDuration, Assert.Single(stretchService.StretchRatios), precision: 9);
        MixSpeechClip clip = Assert.Single(
            Assert.Single(mixRenderer.Calls).MixPlan.SpeechClips, speechClip => !speechClip.IsSilentGap);
        Assert.Contains("/tail-fit/", clip.TakeRelativePath);
        Assert.Equal(6.0d - TailOverhangFitter.SafetyMarginSeconds, clip.StartSeconds + clip.TakeDurationSeconds!.Value, precision: 9);
        Assert.True(File.Exists(artifactStore.GetPath(clip.TakeRelativePath!)));
        MixPlan savedPlan = (await new MixPlanStore(artifactStore).LoadAsync(TestContext.Current.CancellationToken))!;
        Assert.Empty(savedPlan.Warnings);
        Assert.Equal(clip.TakeRelativePath, Assert.Single(savedPlan.SpeechClips, speechClip => !speechClip.IsSilentGap).TakeRelativePath);
    }

    [Fact]
    public async Task ExportAsync_leaves_the_saved_mix_plan_on_the_original_take_when_a_fitted_export_fails()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true, durationSeconds: 6.0d);
        TranscriptProjectState state = WithTakeDuration(context.State, 6.5d);
        SeedTakeFile(temp.Path);
        var artifactStore = new FakeArtifactStore(temp.Path);
        var mixRenderer = new FakeMixRenderer();
        ExportStageHandler handler = CreateHandler(
            artifactStore,
            mixRenderer,
            new FakeLoudnessNormalizer(),
            new ThrowingExportRenderer(new IOException("Export render failed.")),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(6.0d) },
            new FakeMediaAssetRepository(),
            new FakeProjectStageRunStore(),
            tailOverhangFitter: new TailOverhangFitter(new FakeAudioTimeStretchService(), artifactStore));
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        await Assert.ThrowsAsync<ExportStageException>(() => handler.ExportAsync(
            state,
            new ExportStageRequest(context.Project.Id, outputPath, []),
            TestContext.Current.CancellationToken));

        MixSpeechClip fittedClip = Assert.Single(
            Assert.Single(mixRenderer.Calls).MixPlan.SpeechClips, speechClip => !speechClip.IsSilentGap);
        Assert.Contains("/tail-fit/", fittedClip.TakeRelativePath);
        Assert.False(File.Exists(artifactStore.GetPath(fittedClip.TakeRelativePath!)));
        MixPlan savedPlan = (await new MixPlanStore(artifactStore).LoadAsync(TestContext.Current.CancellationToken))!;
        Assert.Equal(
            "artifacts/tts/take-0001.wav",
            Assert.Single(savedPlan.SpeechClips, speechClip => !speechClip.IsSilentGap).TakeRelativePath);
    }

    [Fact]
    public async Task ExportAsync_still_reports_a_take_that_would_need_more_compression_than_the_fit_limit()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true, durationSeconds: 6.0d);
        // 7.5s against 6.0s needs 7.5 / 5.95 = 1.26x, just over the 1.25x limit.
        TranscriptProjectState state = WithTakeDuration(context.State, 7.5d);
        SeedTakeFile(temp.Path);
        var artifactStore = new FakeArtifactStore(temp.Path);
        var stretchService = new FakeAudioTimeStretchService();
        var mixRenderer = new FakeMixRenderer();
        ExportStageHandler handler = CreateHandler(
            artifactStore,
            mixRenderer,
            new FakeLoudnessNormalizer(),
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(6.0d) },
            new FakeMediaAssetRepository(),
            new FakeProjectStageRunStore(),
            tailOverhangFitter: new TailOverhangFitter(stretchService, artifactStore));
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() =>
            handler.ExportAsync(
                state,
                new ExportStageRequest(context.Project.Id, outputPath, []),
                TestContext.Current.CancellationToken));

        Assert.Equal("take-exceeds-source-audio", Assert.Single(exception.Report.Causes).Code);
        Assert.Empty(stretchService.StretchRatios);
        Assert.Empty(mixRenderer.Calls);
    }

    private static TranscriptProjectState WithTakeDuration(TranscriptProjectState state, double takeDurationSeconds) =>
        state with
        {
            ProjectState = state.ProjectState with
            {
                Artifacts = state.ProjectState.Artifacts
                    .Select(artifact => artifact.Kind == ArtifactKind.TtsTake
                        ? artifact with { DurationSeconds = takeDurationSeconds }
                        : artifact)
                    .ToList()
            }
        };

    private static void SeedTakeFile(string rootPath)
    {
        string path = Path.Join(rootPath, "artifacts", "tts", "take-0001.wav");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, [1, 2, 3, 4]);
    }

    [Theory]
    [InlineData(null, 7.5d, "take-exceeds-source-audio")]
    [InlineData(0d, 7.5d, "take-exceeds-source-audio")]
    [InlineData(null, 0d, "export-error")]
    public async Task ExportAsync_probes_missing_take_duration_before_rendering(
        double? storedDuration, double probedDuration, string failureCode)
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        TranscriptProjectState state = context.State with
        {
            ProjectState = context.State.ProjectState with
            {
                Artifacts = context.State.ProjectState.Artifacts.Select(artifact => artifact.Kind == ArtifactKind.TtsTake
                    ? artifact with { DurationSeconds = storedDuration }
                    : artifact).ToArray()
            },
            TtsTakes = context.State.TtsTakes.Select(take => take with { DurationSamples = null, SampleRate = null }).ToArray()
        };
        var probe = new FakeMediaProbe { Snapshot = CreateProbeSnapshot(probedDuration) };
        var renderer = new FakeMixRenderer();
        ExportStageHandler handler = CreateHandler(new FakeArtifactStore(temp.Path), renderer,
            new FakeLoudnessNormalizer(), new FakeExportRenderer(), probe,
            new FakeMediaAssetRepository(), new FakeProjectStageRunStore());

        ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() => handler.ExportAsync(
            state, new ExportStageRequest(context.Project.Id, Path.Join(temp.Path, "delivery", "dub.mp4"), []),
            TestContext.Current.CancellationToken));

        Assert.Equal(failureCode, Assert.Single(exception.Report.Causes).Code);
        Assert.EndsWith("take-0001.wav", Assert.Single(probe.Calls));
        Assert.Empty(renderer.Calls);
        Assert.Equal(storedDuration, state.ProjectState.Artifacts.Single(artifact => artifact.Kind == ArtifactKind.TtsTake).DurationSeconds);
    }

    [Fact]
    public async Task ExportAsync_probes_missing_source_duration_before_rendering()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        TranscriptProjectState state = context.State with
        {
            ProjectState = context.State.ProjectState with
            {
                Artifacts = context.State.ProjectState.Artifacts.Select(artifact => artifact.Kind == ArtifactKind.NormalizedAudio
                    ? artifact with { DurationSeconds = null }
                    : artifact).ToArray()
            }
        };
        // The probed source is 0.5s: the 1.0s take overruns it even when moved back to 0.
        var probe = new FakeMediaProbe { Snapshot = CreateProbeSnapshot(0.5d) };
        var renderer = new FakeMixRenderer();
        ExportStageHandler handler = CreateHandler(new FakeArtifactStore(temp.Path), renderer,
            new FakeLoudnessNormalizer(), new FakeExportRenderer(), probe,
            new FakeMediaAssetRepository(), new FakeProjectStageRunStore());

        ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() => handler.ExportAsync(
            state, new ExportStageRequest(context.Project.Id, Path.Join(temp.Path, "delivery", "dub.mp4"), []),
            TestContext.Current.CancellationToken));

        Assert.Equal("take-exceeds-source-audio", Assert.Single(exception.Report.Causes).Code);
        Assert.EndsWith("normalized_audio.wav", Assert.Single(probe.Calls));
        Assert.Empty(renderer.Calls);
    }

    [Fact]
    public async Task ExportAsync_marks_stage_failed_when_unexpected_error_report_write_fails()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var artifactStore = new FakeArtifactStore(temp.Path)
        {
            FailingJsonWriteFileName = "export-failure.json"
        };
        var stageRunStore = new FakeProjectStageRunStore();
        ExportStageHandler handler = CreateHandler(
            artifactStore,
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer(),
            new ThrowingExportRenderer(new InvalidOperationException("Mux failed.")),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            stageRunStore);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() =>
            handler.ExportAsync(
                context.State,
                new ExportStageRequest(context.Project.Id, outputPath, []),
                TestContext.Current.CancellationToken));

        ExportFailureCause cause = Assert.Single(exception.Report.Causes);
        Assert.Equal("export-error", cause.Code);
        Assert.Equal("Mux failed.", cause.Message);
        Assert.IsType<InvalidOperationException>(exception.InnerException);
        Assert.Equal(StageRunStatus.Failed, Assert.Single(stageRunStore.All).Status);
    }

    [Fact]
    public async Task ExportAsync_cleans_delivery_sidecars_and_unlinked_subtitle_artifacts_when_verification_fails()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var artifactStore = new FakeArtifactStore(temp.Path);
        var stageRunStore = new FakeProjectStageRunStore();
        var mediaAssetRepository = new FakeMediaAssetRepository();
        ExportStageHandler handler = CreateHandler(
            artifactStore,
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer(),
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds + 1d) },
            mediaAssetRepository,
            stageRunStore);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() =>
            handler.ExportAsync(
                context.State,
                new ExportStageRequest(
                    context.Project.Id,
                    outputPath,
                    [ExportSubtitleFormat.Srt]),
                TestContext.Current.CancellationToken));

        ExportFailureCause cause = Assert.Single(exception.Report.Causes);
        Assert.Equal("duration-tolerance-exceeded", cause.Code);
        Assert.False(File.Exists(outputPath));
        Assert.False(File.Exists(Path.ChangeExtension(outputPath, ".srt")));
        Assert.True(File.Exists(GetFailureReportPath(outputPath)));
        Assert.DoesNotContain(mediaAssetRepository.Artifacts, artifact => artifact.Kind == ArtifactKind.ExportAudio);
        Assert.False(File.Exists(artifactStore.GetPath(ProjectArtifactPaths.GetExportSubtitleRelativePath(
            exception.Report.StageRunId,
            ".srt"))));
        Assert.False(File.Exists(artifactStore.GetPath(ProjectArtifactPaths.GetExportAudioRelativePath(
            exception.Report.StageRunId))));
        Assert.False(File.Exists(artifactStore.GetPath(ProjectArtifactPaths.GetExportVideoRelativePath(
            exception.Report.StageRunId,
            ".mp4"))));
        Assert.Equal(StageRunStatus.Failed, Assert.Single(stageRunStore.All).Status);
    }

    [Fact]
    public async Task ExportAsync_preserves_existing_delivery_output_when_sidecar_commit_fails()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var stageRunStore = new FakeProjectStageRunStore();
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer(),
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            stageRunStore);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        byte[] originalOutput = [9, 8, 7, 6];
        await File.WriteAllBytesAsync(outputPath, originalOutput, TestContext.Current.CancellationToken);
        string sidecarPath = Path.ChangeExtension(outputPath, ".srt");
        Directory.CreateDirectory(sidecarPath);

        ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() =>
            handler.ExportAsync(
                context.State,
                new ExportStageRequest(
                    context.Project.Id,
                    outputPath,
                    [ExportSubtitleFormat.Srt]),
                TestContext.Current.CancellationToken));

        Assert.Equal("export-error", Assert.Single(exception.Report.Causes).Code);
        Assert.Equal(originalOutput, File.ReadAllBytes(outputPath));
        Assert.True(Directory.Exists(sidecarPath));
        Assert.Equal(StageRunStatus.Failed, Assert.Single(stageRunStore.All).Status);
    }

    [Fact]
    public async Task ExportAsync_reports_missing_output_video_stream()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var mixRenderer = new FakeMixRenderer();
        var loudnessNormalizer = new FakeLoudnessNormalizer();
        var exportRenderer = new FakeExportRenderer();
        var stageRunStore = new FakeProjectStageRunStore();
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            mixRenderer,
            loudnessNormalizer,
            exportRenderer,
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds, hasVideo: false) },
            new FakeMediaAssetRepository(),
            stageRunStore);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() =>
            handler.ExportAsync(
                context.State,
                new ExportStageRequest(context.Project.Id, outputPath, []),
                TestContext.Current.CancellationToken));

        ExportFailureCause cause = Assert.Single(exception.Report.Causes);
        Assert.Equal("missing-video-stream", cause.Code);
        Assert.False(File.Exists(outputPath));
        Assert.Equal(StageRunStatus.Failed, Assert.Single(stageRunStore.All).Status);
    }

    [Fact]
    public async Task ExportAsync_reports_missing_source_media_before_rendering_or_muxing()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        File.Delete(context.MediaAsset.SourceFilePath);
        var mixRenderer = new FakeMixRenderer();
        var loudnessNormalizer = new FakeLoudnessNormalizer();
        var exportRenderer = new FakeExportRenderer();
        var stageRunStore = new FakeProjectStageRunStore();
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            mixRenderer,
            loudnessNormalizer,
            exportRenderer,
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            stageRunStore);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() =>
            handler.ExportAsync(
                context.State,
                new ExportStageRequest(context.Project.Id, outputPath, []),
                TestContext.Current.CancellationToken));

        Assert.Equal("missing-source-media", Assert.Single(exception.Report.Causes).Code);
        Assert.Empty(mixRenderer.Calls);
        Assert.Empty(loudnessNormalizer.Calls);
        Assert.Empty(exportRenderer.Calls);
        Assert.Equal(StageRunStatus.Failed, Assert.Single(stageRunStore.All).Status);
    }

    [Fact]
    public async Task ExportAsync_reports_unavailable_ffmpeg_before_rendering_or_muxing()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        TranscriptProjectState state = context.State with
        {
            ExportTools = ExportToolAvailability.Unavailable("ffmpeg missing")
        };
        var mixRenderer = new FakeMixRenderer();
        var loudnessNormalizer = new FakeLoudnessNormalizer();
        var exportRenderer = new FakeExportRenderer();
        var stageRunStore = new FakeProjectStageRunStore();
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            mixRenderer,
            loudnessNormalizer,
            exportRenderer,
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            stageRunStore);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() =>
            handler.ExportAsync(
                state,
                new ExportStageRequest(context.Project.Id, outputPath, []),
                TestContext.Current.CancellationToken));

        ExportFailureCause cause = Assert.Single(exception.Report.Causes);
        Assert.Equal("ffmpeg-unavailable", cause.Code);
        Assert.Contains("ffmpeg missing", cause.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Empty(mixRenderer.Calls);
        Assert.Empty(loudnessNormalizer.Calls);
        Assert.Empty(exportRenderer.Calls);
        Assert.Equal(StageRunStatus.Failed, Assert.Single(stageRunStore.All).Status);
    }

    [Fact]
    public async Task ExportAsync_reports_non_finite_output_duration()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var stageRunStore = new FakeProjectStageRunStore();
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer(),
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(double.NaN) },
            new FakeMediaAssetRepository(),
            stageRunStore);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() =>
            handler.ExportAsync(
                context.State,
                new ExportStageRequest(context.Project.Id, outputPath, []),
                TestContext.Current.CancellationToken));

        ExportFailureCause cause = Assert.Single(exception.Report.Causes);
        Assert.Equal("invalid-duration", cause.Code);
        Assert.Equal(StageRunStatus.Failed, Assert.Single(stageRunStore.All).Status);
    }

    [Fact]
    public async Task ExportAsync_reports_missing_video_before_rendering_or_muxing()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true, hasVideo: false);
        var mixRenderer = new FakeMixRenderer();
        var loudnessNormalizer = new FakeLoudnessNormalizer();
        var exportRenderer = new FakeExportRenderer();
        var stageRunStore = new FakeProjectStageRunStore();
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            mixRenderer,
            loudnessNormalizer,
            exportRenderer,
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds, hasVideo: false) },
            new FakeMediaAssetRepository(),
            stageRunStore);
        string outputPath = Path.Join(temp.Path, "delivery", "lesson-dub.mp4");

        ExportStageException exception = await Assert.ThrowsAsync<ExportStageException>(() =>
            handler.ExportAsync(
                context.State,
                new ExportStageRequest(context.Project.Id, outputPath, []),
                TestContext.Current.CancellationToken));

        ExportFailureCause cause = Assert.Single(exception.Report.Causes);
        Assert.Equal("missing-video-stream", cause.Code);
        Assert.Empty(mixRenderer.Calls);
        Assert.Empty(loudnessNormalizer.Calls);
        Assert.Empty(exportRenderer.Calls);
        Assert.Equal(StageRunStatus.Failed, Assert.Single(stageRunStore.All).Status);
    }

    [Fact]
    public async Task ExportAsync_rejects_container_extension_mismatch_before_starting_stage_run()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var stageRunStore = new FakeProjectStageRunStore();
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer(),
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            stageRunStore);

        InvalidOperationException exception = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            handler.ExportAsync(
                context.State,
                new ExportStageRequest(
                    context.Project.Id,
                    Path.Join(temp.Path, "delivery", "lesson-dub.mp4"),
                    [],
                    Container: ExportOutputContainer.Mkv),
                TestContext.Current.CancellationToken));

        Assert.Contains(".mkv", exception.Message);
        Assert.Empty(stageRunStore.All);
    }

    [Fact]
    public async Task ExportAsync_rejects_source_media_path_before_starting_stage_run()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var stageRunStore = new FakeProjectStageRunStore();
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer(),
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            stageRunStore);

        InvalidOperationException exception = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            handler.ExportAsync(
                context.State,
                new ExportStageRequest(
                    context.Project.Id,
                    context.MediaAsset.SourceFilePath,
                    []),
                TestContext.Current.CancellationToken));

        Assert.Contains("source media", exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Empty(stageRunStore.All);
    }

    [Fact]
    public async Task ExportAsync_rejects_unknown_container_before_starting_stage_run()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var stageRunStore = new FakeProjectStageRunStore();
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer(),
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            stageRunStore);

        InvalidOperationException exception = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            handler.ExportAsync(
                context.State,
                new ExportStageRequest(
                    context.Project.Id,
                    Path.Join(temp.Path, "delivery", "lesson-dub.mp4"),
                    [],
                    Container: (ExportOutputContainer)99),
                TestContext.Current.CancellationToken));

        Assert.Contains("container", exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Empty(stageRunStore.All);
    }

    [Fact]
    public async Task ExportAsync_rejects_unknown_subtitle_source_before_starting_stage_run()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var stageRunStore = new FakeProjectStageRunStore();
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer(),
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            stageRunStore);

        InvalidOperationException exception = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            handler.ExportAsync(
                context.State,
                new ExportStageRequest(
                    context.Project.Id,
                    Path.Join(temp.Path, "delivery", "lesson-dub.mp4"),
                    [],
                    SubtitleSource: (ExportSubtitleSource)99),
                TestContext.Current.CancellationToken));

        Assert.Contains("subtitle source", exception.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Empty(stageRunStore.All);
    }

    [Fact]
    public async Task ExportAsync_rejects_null_subtitle_formats_before_starting_stage_run()
    {
        using var temp = new TempDirectory();
        TestExportContext context = CreateContext(temp.Path, includeCompletedTake: true);
        var stageRunStore = new FakeProjectStageRunStore();
        ExportStageHandler handler = CreateHandler(
            new FakeArtifactStore(temp.Path),
            new FakeMixRenderer(),
            new FakeLoudnessNormalizer(),
            new FakeExportRenderer(),
            new FakeMediaProbe { Snapshot = CreateProbeSnapshot(context.MediaAsset.DurationSeconds) },
            new FakeMediaAssetRepository(),
            stageRunStore);

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            handler.ExportAsync(
                context.State,
                new ExportStageRequest(
                    context.Project.Id,
                    Path.Join(temp.Path, "delivery", "lesson-dub.mp4"),
                    null!),
                TestContext.Current.CancellationToken));

        Assert.Empty(stageRunStore.All);
    }

    private static ExportStageHandler CreateHandler(
        IArtifactStore artifactStore,
        IPreviewRangeRenderer mixRenderer,
        ILoudnessNormalizer loudnessNormalizer,
        IExportRenderer exportRenderer,
        IMediaProbe mediaProbe,
        IMediaAssetRepository mediaAssetRepository,
        IProjectStageRunStore stageRunStore,
        IFileFingerprintService? fileFingerprintService = null,
        IVideoRecomposer? videoRecomposer = null,
        IExportTierGate? exportTierGate = null,
        TailOverhangFitter? tailOverhangFitter = null) =>
        new(
            new MixPlanBuilder(),
            new MixPlanStore(artifactStore),
            mixRenderer,
            artifactStore,
            fileFingerprintService ?? new FakeFileFingerprintService(),
            mediaAssetRepository,
            stageRunStore,
            loudnessNormalizer,
            exportRenderer,
            mediaProbe,
            new SubtitleExportService(),
            videoRecomposer ?? new FakeVideoRecomposer(),
            exportTierGate,
            tailOverhangFitter: tailOverhangFitter);

    private static TestExportContext CreateContext(
        string rootPath,
        bool includeCompletedTake,
        bool hasVideo = true,
        int normalizedAudioChannelCount = 1,
        double durationSeconds = 6.0d)
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        Guid projectId = Guid.NewGuid();
        Guid mediaAssetId = Guid.NewGuid();
        Guid transcriptRevisionId = Guid.NewGuid();
        Guid translationRevisionId = Guid.NewGuid();
        Guid speakerId = Guid.NewGuid();
        Guid voiceAssignmentId = Guid.NewGuid();
        var project = new TrackdubProject(projectId, "Export", now, now);
        string sourcePath = Path.Join(rootPath, "source.mp4");
        Directory.CreateDirectory(Path.GetDirectoryName(sourcePath)!);
        File.WriteAllBytes(sourcePath, [0, 0, 0, 24, 102, 116, 121, 112]);
        var mediaAsset = new MediaAsset(
            mediaAssetId,
            projectId,
            sourcePath,
            Path.GetFileName(sourcePath),
            "source-hash",
            100,
            now,
            "mp4",
            durationSeconds,
            HasAudio: true,
            HasVideo: hasVideo,
            now);
        TranscriptSegment transcriptSegment = TranscriptSegment.Create(
            transcriptRevisionId,
            0,
            1.0d,
            2.0d,
            "Hello",
            speakerId,
            "en");
        TranslatedSegment translatedSegment = TranslatedSegment.Create(
            translationRevisionId,
            0,
            1.0d,
            2.0d,
            "Hola");
        ProjectArtifact sourceArtifact = CreateArtifact(
            projectId,
            mediaAssetId,
            ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath,
            durationSeconds: durationSeconds,
            now,
            channelCount: normalizedAudioChannelCount);
        var artifacts = new List<ProjectArtifact> { sourceArtifact };
        var takes = new List<TtsTake>();
        StageRunRecord upstreamRun = StageRunRecord
            .Start(projectId, StageNames.Tts, now)
            .WithRuntimeInfo("fake", "fake", modelId: "kokoro", modelAlias: "kokoro")
            .Complete(now.AddSeconds(1));

        if (includeCompletedTake)
        {
            ProjectArtifact takeArtifact = CreateArtifact(
                projectId,
                mediaAssetId,
                ArtifactKind.TtsTake,
                "artifacts/tts/take-0001.wav",
                durationSeconds: 1.0d,
                now.AddSeconds(1));
            artifacts.Add(takeArtifact);
            TtsTake take = TtsTake.Create(projectId, voiceAssignmentId, translatedSegment.Id, segmentIndex: 0)
                .Complete(
                    takeArtifact.Id,
                    stageRunId: upstreamRun.Id,
                    durationSamples: 48000,
                    sampleRate: 48000,
                    provider: "fake",
                    modelId: "kokoro",
                    voiceId: "af_heart",
                    durationOverrunRatio: null);
            takes.Add(take);
        }

        TranscriptProjectState state = CreateState(
            project,
            mediaAsset,
            artifacts,
            [transcriptSegment],
            [translatedSegment],
            takes,
            [upstreamRun]);
        return new TestExportContext(project, mediaAsset, upstreamRun, state);
    }

    private static TranscriptProjectState CreateState(
        TrackdubProject project,
        MediaAsset mediaAsset,
        IReadOnlyList<ProjectArtifact> artifacts,
        IReadOnlyList<TranscriptSegment> transcriptSegments,
        IReadOnlyList<TranslatedSegment> translatedSegments,
        IReadOnlyList<TtsTake> ttsTakes,
        IReadOnlyList<StageRunRecord> stageRuns)
    {
        var sourceReference = new SourceMediaReference(
            mediaAsset.SourceFilePath,
            mediaAsset.SourceFileName,
            new FileFingerprint(mediaAsset.FingerprintSha256, mediaAsset.SourceSizeBytes, mediaAsset.SourceLastWriteTimeUtc),
            CreateProbeSnapshot(mediaAsset.DurationSeconds, mediaAsset.HasVideo),
            DateTimeOffset.UtcNow);
        var openResult = new OpenProjectResult(
            project,
            mediaAsset,
            sourceReference,
            SourceMediaStatus.Available,
            SourceStatusMessage: null,
            artifacts,
            TranscriptLanguage: "en");
        TranscriptRevision transcriptRevision = TranscriptRevision.Create(
            project.Id,
            stageRunId: null,
            revisionNumber: 1,
            DateTimeOffset.UtcNow);
        TranslationRevision translationRevision = TranslationRevision.Create(
            project.Id,
            stageRunId: null,
            transcriptRevision.Id,
            "es",
            revisionNumber: 1,
            DateTimeOffset.UtcNow);
        return new TranscriptProjectState(
            openResult,
            transcriptRevision,
            transcriptSegments,
            Speakers: [],
            SpeakerTurns: [],
            translationRevision,
            translatedSegments,
            IsTranslationStale: false,
            TranscriptLanguage: "en",
            stageRuns,
            SupportedTargetLanguages: [],
            SelectedTranslationTargetLanguage: "es",
            StaleTranslatedSegmentIndices: new HashSet<int>(),
            WaveformSummary: null,
            AvailableVoices: [],
            VoiceAssignments: [],
            ttsTakes,
            TtsSegmentStates: [],
            VoiceAssignmentWarnings: []);
    }

    private static ProjectArtifact CreateArtifact(
        Guid projectId,
        Guid mediaAssetId,
        ArtifactKind kind,
        string relativePath,
        double durationSeconds,
        DateTimeOffset createdAtUtc,
        int channelCount = 1) =>
        new(
            Guid.NewGuid(),
            projectId,
            mediaAssetId,
            kind,
            relativePath,
            $"{kind.ToString().ToLowerInvariant()}-hash",
            100,
            durationSeconds,
            48000,
            channelCount,
            createdAtUtc);

    private static MediaProbeSnapshot CreateProbeSnapshot(double durationSeconds, bool hasVideo = true, int height = 1080) =>
        new(
            "mp4",
            "MP4",
            durationSeconds,
            BitRate: null,
            AudioStreams: [new MediaAudioStream(0, "aac", Channels: 2, SampleRate: 48000, durationSeconds)],
            VideoStreams: hasVideo
                ? [new MediaVideoStream(1, "h264", Width: 1920, Height: height, FrameRate: 24.0d, durationSeconds)]
                : [],
            SubtitleStreams: []);

    private static string GetFailureReportPath(string outputPath)
    {
        string directory = Path.GetDirectoryName(Path.GetFullPath(outputPath))!;
        string fileName = Path.GetFileNameWithoutExtension(outputPath);
        return Path.Join(directory, $"{fileName}.export-failure.json");
    }

    private sealed record TestExportContext(
        TrackdubProject Project,
        MediaAsset MediaAsset,
        StageRunRecord UpstreamStageRun,
        TranscriptProjectState State);

    private sealed class ManifestSequenceFingerprintService : IFileFingerprintService
    {
        public int ManifestCallCount { get; private set; }

        public Task<FileFingerprint> ComputeAsync(string path, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (Path.GetFileName(path).Equals("export-manifest.json", StringComparison.OrdinalIgnoreCase))
            {
                ManifestCallCount++;
                return Task.FromResult(ManifestCallCount == 1
                    ? new FileFingerprint("initial-manifest-hash", 10, DateTimeOffset.UnixEpoch)
                    : new FileFingerprint("final-manifest-hash", 20, DateTimeOffset.UnixEpoch.AddSeconds(1)));
            }

            return Task.FromResult(new FileFingerprint($"hash-{Path.GetFileName(path)}", 100, DateTimeOffset.UnixEpoch));
        }
    }

    private sealed class ThrowingExportRenderer(Exception exception) : IExportRenderer
    {
        public Task<ExportRenderResult> RenderAsync(ExportPlan plan, CancellationToken cancellationToken) =>
            Task.FromException<ExportRenderResult>(exception);
    }

    private sealed class FakeExportTierGate(bool requiresWatermark, string? durationBlockReason = null) : IExportTierGate
    {
        public bool RequiresWatermark { get; } = requiresWatermark;

        public string? CheckDurationGate(TimeSpan sourceDuration) => durationBlockReason;
    }

    private sealed class CapturingExportRenderer : IExportRenderer
    {
        public ExportPlan? LastPlan { get; private set; }

        public Task<ExportRenderResult> RenderAsync(ExportPlan plan, CancellationToken cancellationToken)
        {
            LastPlan = plan;
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(plan.OutputPath))!);
            File.WriteAllText(plan.OutputPath, "video");
            return Task.FromResult(new ExportRenderResult(plan.OutputPath, []));
        }
    }

    private sealed class ThrowingVideoRecomposer : IVideoRecomposer
    {
        public Task<VideoRecompositionResult> RecomposeAsync(
            ResolvedVideoRecompositionPlan plan,
            string outputVideoPath,
            CancellationToken cancellationToken) =>
            throw new InvalidOperationException("ffmpeg filter_complex failed");
    }

    private sealed class TempDirectory : IDisposable
    {
        public TempDirectory()
        {
            Path = System.IO.Path.Join(System.IO.Path.GetTempPath(), $"trackdub-export-tests-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Path);
        }

        public string Path { get; }

        public void Dispose()
        {
            try
            {
                if (Directory.Exists(Path))
                {
                    Directory.Delete(Path, recursive: true);
                }
            }
            catch (IOException)
            {
            }
            catch (UnauthorizedAccessException)
            {
            }
        }
    }
}
