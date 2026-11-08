using System.Text;
using Trackdub.Application.Logging;
using Trackdub.Application.LipSynthesis;
using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Application.Mixing;
using Trackdub.Contracts.Projects;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Mixing;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Translation;
using Trackdub.Domain.Tts;

namespace Trackdub.Application.Transcripts;

public sealed class ExportStageHandler(
    MixPlanBuilder mixPlanBuilder,
    MixPlanStore mixPlanStore,
    IPreviewRangeRenderer mixRenderer,
    IArtifactStore artifactStore,
    IFileFingerprintService fileFingerprintService,
    IMediaAssetRepository mediaAssetRepository,
    IProjectStageRunStore stageRunStore,
    ILoudnessNormalizer loudnessNormalizer,
    IExportRenderer exportRenderer,
    IMediaProbe mediaProbe,
    SubtitleExportService subtitleExportService,
    IVideoRecomposer videoRecomposer,
    IExportTierGate? exportTierGate = null,
    IApplicationLogger? logger = null,
    ITtsCandidateGroupRepository? candidateGroupRepository = null,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null,
    TimelineMediaMixPreparer? timelineMediaMixPreparer = null)
{
    private const double DurationToleranceSeconds = 0.5d;
    private const double MaxMatchOriginalUpwardBoostDb = 9d;
    private const string RecoveryFailureDataKey = "Trackdub.ExportDeliveryRecoveryFailure";
    private static readonly Encoding SubtitleEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);

    private readonly MixPlanBuilder mixPlanBuilder =
        mixPlanBuilder ?? throw new ArgumentNullException(nameof(mixPlanBuilder));
    private readonly MixPlanStore mixPlanStore =
        mixPlanStore ?? throw new ArgumentNullException(nameof(mixPlanStore));
    private readonly IPreviewRangeRenderer mixRenderer =
        mixRenderer ?? throw new ArgumentNullException(nameof(mixRenderer));
    private readonly IArtifactStore artifactStore =
        artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));
    private readonly IFileFingerprintService fileFingerprintService =
        fileFingerprintService ?? throw new ArgumentNullException(nameof(fileFingerprintService));
    private readonly IMediaAssetRepository mediaAssetRepository =
        mediaAssetRepository ?? throw new ArgumentNullException(nameof(mediaAssetRepository));
    private readonly IProjectStageRunStore stageRunStore =
        stageRunStore ?? throw new ArgumentNullException(nameof(stageRunStore));
    private readonly ILoudnessNormalizer loudnessNormalizer =
        loudnessNormalizer ?? throw new ArgumentNullException(nameof(loudnessNormalizer));
    private readonly IExportRenderer exportRenderer =
        exportRenderer ?? throw new ArgumentNullException(nameof(exportRenderer));
    private readonly IMediaProbe mediaProbe =
        mediaProbe ?? throw new ArgumentNullException(nameof(mediaProbe));
    private readonly SubtitleExportService subtitleExportService =
        subtitleExportService ?? throw new ArgumentNullException(nameof(subtitleExportService));
    private readonly IVideoRecomposer videoRecomposer =
        videoRecomposer ?? throw new ArgumentNullException(nameof(videoRecomposer));
    private readonly IApplicationLogger? logger = logger;
    private readonly IExportTierGate? exportTierGate = exportTierGate;
    private readonly ITtsCandidateGroupRepository? candidateGroupRepository = candidateGroupRepository;
    private readonly IRuntimePlanningPreferences? runtimePlanningPreferences = runtimePlanningPreferences;

    public async Task<ExportStageResult> ExportAsync(
        TranscriptProjectState currentState,
        ExportStageRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(currentState);
        ArgumentNullException.ThrowIfNull(request);

        MediaAsset mediaAsset = TranscriptWorkflowUtilities.GetRequiredMediaAsset(currentState);
        ValidateRequest(currentState, mediaAsset, request);

        // Tier gate: block export early if duration exceeds the tier limit (fail fast before encoding).
        if (exportTierGate is not null)
        {
            string? durationBlockReason = exportTierGate.CheckDurationGate(TimeSpan.FromSeconds(mediaAsset.DurationSeconds));
            if (durationBlockReason is not null)
            {
                return ExportStageResult.Blocked(durationBlockReason);
            }
        }

        string outputPath = Path.GetFullPath(request.OutputPath);
        if (FilePathComparison.AreSame(outputPath, mediaAsset.SourceFilePath))
        {
            throw new InvalidOperationException("Export output path must be different from the source media path.");
        }

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        StageRunRecord stageRun = await StageRunHelper
            .StartAsync(stageRunStore, request.ProjectId, StageNames.Export, cancellationToken)
            .ConfigureAwait(false);

        string manifestRelativePath = ProjectArtifactPaths.GetExportManifestRelativePath(stageRun.Id);
        string failureRelativePath = ProjectArtifactPaths.GetExportFailureReportRelativePath(stageRun.Id);
        var transientArtifactPaths = new List<string>();
        var rollbackArtifactPaths = new List<string>();
        var registeredExportArtifactIds = new List<Guid>();

        // exportSucceeded gates the finally-cleanup: set to true only after WriteManifestAndFinalizeAsync
        // returns without throwing.  Centralising cleanup in finally ensures it always runs on any failure
        // path — including future catch arms that might be added without remembering to duplicate the calls.
        bool exportSucceeded = false;
        bool recoveringDelivery = true;
        try
        {
            ExportDeliveryTransaction.Recover(outputPath);
            recoveringDelivery = false;
            ProjectArtifact manifestArtifact = await InitializeExportAsync(
                currentState, mediaAsset, request, stageRun, manifestRelativePath,
                rollbackArtifactPaths, registeredExportArtifactIds, cancellationToken).ConfigureAwait(false);

            await RunPreflightChecksAsync(
                currentState, mediaAsset, stageRun, failureRelativePath, outputPath, cancellationToken).ConfigureAwait(false);

            MixPlan mixPlan = await BuildAndSaveMixPlanAsync(
                currentState, mediaAsset, request, stageRun, failureRelativePath, outputPath, cancellationToken).ConfigureAwait(false);

            RenderedSubtitleArtifacts subtitles = await CheckSubtitleCuesAsync(
                currentState, request, stageRun, failureRelativePath, outputPath, cancellationToken).ConfigureAwait(false);
            transientArtifactPaths.AddRange(subtitles.ArtifactPaths);

            var (audioFinalPath, audioArtifact, audioRenderResult) = await RenderDubAudioArtifactAsync(
                currentState, mediaAsset, request, stageRun, mixPlan, cancellationToken).ConfigureAwait(false);
            rollbackArtifactPaths.Add(audioFinalPath);
            registeredExportArtifactIds.Add(audioArtifact.Id);

            var (videoFinalPath, videoArtifact, videoRenderResult) = await RenderExportVideoAsync(
                currentState, mediaAsset, request, stageRun,
                audioFinalPath, subtitles, failureRelativePath, outputPath,
                rollbackArtifactPaths, registeredExportArtifactIds, cancellationToken).ConfigureAwait(false);

            ExportStageResult result = await WriteManifestAndFinalizeAsync(
                currentState, request, stageRun, manifestRelativePath, manifestArtifact,
                mixPlan, subtitles, audioArtifact, audioRenderResult,
                videoFinalPath, videoArtifact, videoRenderResult,
                outputPath, transientArtifactPaths,
                rollbackArtifactPaths, cancellationToken).ConfigureAwait(false);

            exportSucceeded = true;
            return result;
        }
        catch (ExportStageException)
        {
            throw;
        }
        catch (OperationCanceledException ex)
        {
            Exception? recoveryFailure = ex.Data[RecoveryFailureDataKey] as Exception;
            if (recoveryFailure is not null)
            {
                ExportFailureReport report = new(request.ProjectId, stageRun.Id, DateTimeOffset.UtcNow,
                    [new ExportFailureCause("export-canceled", ex.Message),
                     new ExportFailureCause("export-recovery-error", recoveryFailure.Message)]);
                await TryWriteFailureReportAsync(failureRelativePath, outputPath, report).ConfigureAwait(false);
            }
            await StageRunHelper
                .CancelAsync(stageRunStore, stageRun, runtimeReporter: null,
                    recoveryFailure is null ? "Export canceled." : $"Export canceled. Delivery recovery failed: {recoveryFailure.Message}",
                    CancellationToken.None, runtimePlanningPreferences, logger)
                .ConfigureAwait(false);
            throw;
        }
        catch (Exception ex)
        {
            List<ExportFailureCause> causes = [new(recoveringDelivery ? "export-recovery-error" : "export-error", ex.Message)];
            if (ex.Data[RecoveryFailureDataKey] is Exception recoveryFailure)
                causes.Add(new ExportFailureCause("export-recovery-error", recoveryFailure.Message));
            ExportFailureReport report = new(
                request.ProjectId,
                stageRun.Id,
                DateTimeOffset.UtcNow,
                causes);
            // Preserve the diagnostic report even when the caller's token is already canceled.
            await TryWriteFailureReportAsync(failureRelativePath, outputPath, report).ConfigureAwait(false);

            await StageRunHelper
                .FailAsync(stageRunStore, stageRun, runtimeReporter: null, ex.Message, CancellationToken.None, runtimePlanningPreferences, logger)
                .ConfigureAwait(false);
            throw new ExportStageException(ex.Message, report, GetSidecarFailureReportPath(outputPath), ex);
        }
        finally
        {
            // Cleanup runs unconditionally on failure; on success the lists are already cleared/committed.
            if (!exportSucceeded)
            {
                await CleanupRegisteredExportArtifactsAsync(registeredExportArtifactIds, CancellationToken.None).ConfigureAwait(false);
                CleanupPartialExportOutputs(transientArtifactPaths.Concat(rollbackArtifactPaths));
            }
        }
    }

    private async Task<ProjectArtifact> InitializeExportAsync(
        TranscriptProjectState currentState,
        MediaAsset mediaAsset,
        ExportStageRequest request,
        StageRunRecord stageRun,
        string manifestRelativePath,
        List<string> rollbackArtifactPaths,
        List<Guid> registeredExportArtifactIds,
        CancellationToken cancellationToken)
    {
        ExportManifest initialManifest = BuildManifest(
            currentState,
            request,
            stageRun,
            outputs: [],
            targetLufs: ExportLoudnessTargets.NormalizeTargetLufs(request.TargetLufs),
            achievedLufs: null,
            warnings: [],
            mixPlan: null);
        await artifactStore.WriteJsonAsync(manifestRelativePath, initialManifest, cancellationToken).ConfigureAwait(false);
        rollbackArtifactPaths.Add(artifactStore.GetPath(manifestRelativePath));
        ProjectArtifact manifestArtifact = await RegisterArtifactAsync(
            currentState,
            mediaAsset,
            stageRun,
            ArtifactKind.ExportManifest,
            manifestRelativePath,
            durationSeconds: null,
            sampleRate: null,
            channelCount: null,
            "export-manifest",
            cancellationToken).ConfigureAwait(false);
        registeredExportArtifactIds.Add(manifestArtifact.Id);
        return manifestArtifact;
    }

    private async Task RunPreflightChecksAsync(
        TranscriptProjectState currentState,
        MediaAsset mediaAsset,
        StageRunRecord stageRun,
        string failureRelativePath,
        string outputPath,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<ExportFailureCause> preflightCauses = BuildPreflightFailureCauses(currentState, mediaAsset);
        if (preflightCauses.Count > 0)
        {
            await FailWithReportAsync(
                currentState,
                stageRun,
                failureRelativePath,
                outputPath,
                preflightCauses,
                cancellationToken).ConfigureAwait(false);
        }
    }

    private async Task<MixPlan> BuildAndSaveMixPlanAsync(
        TranscriptProjectState currentState,
        MediaAsset mediaAsset,
        ExportStageRequest request,
        StageRunRecord stageRun,
        string failureRelativePath,
        string outputPath,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<TtsCandidateGroup>? candidateGroups = candidateGroupRepository is not null
            ? await candidateGroupRepository.GetByProjectAsync(request.ProjectId, cancellationToken).ConfigureAwait(false)
            : null;
        var buildRequest = new MixPlanBuildRequest(
            request.ProjectId,
            mediaAsset.Id,
            currentState.ProjectState.Artifacts,
            currentState.TranscriptSegments,
            currentState.TranslatedSegments,
            currentState.TtsTakes,
            currentState.StageRuns,
            request.SourceGainDb, request.DubbedSpeechGainDb, request.DuckingGainDb,
            RestoreOriginalPan: request.RestoreOriginalPan,
            ApplyTimbrePolish: request.ApplyTimbrePolish,
            CandidateGroups: candidateGroups,
            MediaDurationSeconds: mediaAsset.DurationSeconds,
            MediaClips: await TimelineMediaMixPreparer.PrepareAsync(
                timelineMediaMixPreparer, currentState, cancellationToken).ConfigureAwait(false),
            BackgroundMediaGainDb: request.BackgroundMediaGainDb);
        MixPlan mixPlan = mixPlanBuilder.Build(buildRequest);
        if (mixPlan.Warnings.Count == 0)
        {
            var durationsByPath = new Dictionary<string, double>(StringComparer.Ordinal);
            ProjectArtifact sourceArtifact = currentState.ProjectState.Artifacts
                .Where(artifact => artifact.RelativePath == mixPlan.SourceAudioRelativePath)
                .OrderByDescending(artifact => artifact.CreatedAtUtc).First();
            if (!HasUsableDuration(sourceArtifact.DurationSeconds))
            {
                durationsByPath[sourceArtifact.RelativePath] = await ProbeAudioDurationAsync(
                    sourceArtifact.RelativePath, cancellationToken).ConfigureAwait(false);
            }
            foreach (MixSpeechClip clip in mixPlan.SpeechClips.Where(static clip => !clip.IsSilentGap))
            {
                if (!HasUsableDuration(clip.TakeDurationSeconds) && clip.TakeRelativePath is string relativePath &&
                    !durationsByPath.ContainsKey(relativePath))
                {
                    durationsByPath[relativePath] = await ProbeAudioDurationAsync(
                        relativePath, cancellationToken).ConfigureAwait(false);
                }
            }
            if (durationsByPath.Count > 0)
            {
                mixPlan = mixPlanBuilder.Build(buildRequest with
                {
                    Artifacts = currentState.ProjectState.Artifacts.Select(artifact =>
                        durationsByPath.TryGetValue(artifact.RelativePath, out double duration)
                            ? artifact with { DurationSeconds = duration }
                            : artifact).ToArray()
                });
            }
        }
        await mixPlanStore.SaveAsync(mixPlan, cancellationToken).ConfigureAwait(false);
        if (mixPlan.Warnings.Count > 0)
        {
            await FailWithReportAsync(
                currentState,
                stageRun,
                failureRelativePath,
                outputPath,
                BuildTakeFailureCauses(mixPlan),
                cancellationToken).ConfigureAwait(false);
        }
        return mixPlan;
    }

    private async Task<double> ProbeAudioDurationAsync(string relativePath, CancellationToken cancellationToken)
    {
        MediaProbeSnapshot probe = await mediaProbe.ProbeAsync(
            artifactStore.GetPath(relativePath), cancellationToken).ConfigureAwait(false);
        if (!HasUsableDuration(probe.DurationSeconds))
        {
            throw new InvalidOperationException($"Cannot establish audio duration for '{relativePath}' before export.");
        }
        return probe.DurationSeconds;
    }

    private static bool HasUsableDuration(double? duration) =>
        duration is double value && double.IsFinite(value) && value > 0d;

    private async Task<RenderedSubtitleArtifacts> CheckSubtitleCuesAsync(
        TranscriptProjectState currentState,
        ExportStageRequest request,
        StageRunRecord stageRun,
        string failureRelativePath,
        string outputPath,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<SubtitleCue> cues = subtitleExportService.NormalizeForExport(BuildSubtitleCues(currentState, request));
        if ((request.SubtitleFormats.Count > 0 || request.BurnInSubtitles) && cues.Count == 0)
        {
            await FailWithReportAsync(
                currentState,
                stageRun,
                failureRelativePath,
                outputPath,
                [new ExportFailureCause("missing-subtitles", "No subtitle cues are available for the selected subtitle source.")],
                cancellationToken).ConfigureAwait(false);
        }
        return await WriteSubtitleArtifactsAsync(
            stageRun,
            request,
            cues,
            outputPath,
            cancellationToken).ConfigureAwait(false);
    }

    private async Task<(string AudioFinalPath, ProjectArtifact AudioArtifact, RenderedDubAudio AudioRenderResult)> RenderDubAudioArtifactAsync(
        TranscriptProjectState currentState,
        MediaAsset mediaAsset,
        ExportStageRequest request,
        StageRunRecord stageRun,
        MixPlan mixPlan,
        CancellationToken cancellationToken)
    {
        string exportAudioRelativePath = ProjectArtifactPaths.GetExportAudioRelativePath(stageRun.Id);
        await using ArtifactWriteHandle audioHandle = artifactStore.CreateWriteHandle(exportAudioRelativePath);
        RenderedDubAudio audioRenderResult = await RenderNormalizedDubAudioAsync(
            mixPlan,
            mediaAsset,
            request,
            audioHandle.TemporaryPath,
            cancellationToken).ConfigureAwait(false);
        await artifactStore.CommitAsync(audioHandle, cancellationToken).ConfigureAwait(false);
        ProjectArtifact audioArtifact = await RegisterArtifactAsync(
            currentState,
            mediaAsset,
            stageRun,
            ArtifactKind.ExportAudio,
            exportAudioRelativePath,
            audioRenderResult.RenderResult.DurationSeconds,
            audioRenderResult.RenderResult.SampleRate,
            audioRenderResult.RenderResult.ChannelCount,
            "export-audio",
            cancellationToken).ConfigureAwait(false);
        return (audioHandle.FinalPath, audioArtifact, audioRenderResult);
    }

    private async Task<(string VideoFinalPath, ProjectArtifact VideoArtifact, ExportRenderResult RenderResult)> RenderExportVideoAsync(
        TranscriptProjectState currentState,
        MediaAsset mediaAsset,
        ExportStageRequest request,
        StageRunRecord stageRun,
        string audioFinalPath,
        RenderedSubtitleArtifacts subtitles,
        string failureRelativePath,
        string outputPath,
        List<string> rollbackArtifactPaths,
        List<Guid> registeredExportArtifactIds,
        CancellationToken cancellationToken)
    {
        string exportVideoRelativePath = ProjectArtifactPaths.GetExportVideoRelativePath(stageRun.Id, GetContainerExtension(request.Container));
        await using ArtifactWriteHandle videoHandle = artifactStore.CreateWriteHandle(exportVideoRelativePath);

        string sourceVideoPath = mediaAsset.SourceFilePath;
        string? recomposedVideoPath = null;
        var renderWarnings = new List<string>();
        ResolvedVideoRecompositionPlan? recompositionPlan = LipSynthesisExportRecomposition
            .TryBuildResolvedPlan(currentState, artifactStore, sourceVideoPath);
        if (recompositionPlan is not null)
        {
            recomposedVideoPath = Path.Join(
                Path.GetTempPath(),
                $"trackdub-lipsynth-export-{stageRun.Id:N}.mp4");
            try
            {
                VideoRecompositionResult recompositionResult = await videoRecomposer
                    .RecomposeAsync(recompositionPlan, recomposedVideoPath, cancellationToken)
                    .ConfigureAwait(false);
                sourceVideoPath = recompositionResult.OutputPath;
                renderWarnings.AddRange(recompositionResult.Warnings);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                renderWarnings.Add(
                    "Lip-synthesis clips could not be composited into export video; using original source footage.");
                TryDeleteIntermediateOutput(
                    recomposedVideoPath,
                    mediaAsset.SourceFilePath,
                    audioFinalPath,
                    videoHandle.TemporaryPath);
                recomposedVideoPath = null;
                sourceVideoPath = mediaAsset.SourceFilePath;
            }
        }

        bool requiresWatermark = exportTierGate?.RequiresWatermark == true;
        int outputHeight = 0;
        if (requiresWatermark)
        {
            MediaProbeSnapshot sourceProbe = await mediaProbe
                .ProbeAsync(sourceVideoPath, cancellationToken)
                .ConfigureAwait(false);
            outputHeight = sourceProbe.VideoStreams.FirstOrDefault()?.Height ?? 0;
        }

        ExportRenderResult renderResult = await exportRenderer
            .RenderAsync(
                new ExportPlan(
                    sourceVideoPath,
                    audioFinalPath,
                    videoHandle.TemporaryPath,
                    request.Container,
                    subtitles.BurnInSubtitlePath,
                    currentState.TranscriptLanguage,
                    currentState.CurrentTranslationRevision?.TargetLanguage ?? currentState.SelectedTranslationTargetLanguage,
                    request.VideoEncoder,
                    RequiresWatermark: requiresWatermark,
                    OutputHeight: outputHeight),
                cancellationToken)
            .ConfigureAwait(false);
        if (renderWarnings.Count > 0)
        {
            renderResult = renderResult with
            {
                Warnings = renderResult.Warnings.Concat(renderWarnings).ToArray()
            };
        }

        if (recomposedVideoPath is not null)
        {
            TryDeleteIntermediateOutput(
                recomposedVideoPath,
                mediaAsset.SourceFilePath,
                audioFinalPath,
                videoHandle.TemporaryPath);
        }
        if (!FilePathComparison.AreSame(renderResult.OutputPath, videoHandle.TemporaryPath))
        {
            await CopyFileAsync(renderResult.OutputPath, videoHandle.TemporaryPath, cancellationToken).ConfigureAwait(false);
            TryDeleteIntermediateOutput(renderResult.OutputPath, mediaAsset.SourceFilePath, audioFinalPath);
        }
        await artifactStore.CommitAsync(videoHandle, cancellationToken).ConfigureAwait(false);
        // Register the committed file for rollback before verifying so cleanup works if verification fails.
        rollbackArtifactPaths.Add(videoHandle.FinalPath);
        MediaProbeSnapshot outputProbe = await mediaProbe.ProbeAsync(videoHandle.FinalPath, cancellationToken).ConfigureAwait(false);
        IReadOnlyList<ExportFailureCause> verificationCauses = VerifyOutput(mediaAsset, outputProbe);
        if (verificationCauses.Count > 0)
        {
            await FailWithReportAsync(
                currentState,
                stageRun,
                failureRelativePath,
                outputPath,
                verificationCauses,
                cancellationToken).ConfigureAwait(false);
        }
        ProjectArtifact videoArtifact = await RegisterArtifactAsync(
            currentState,
            mediaAsset,
            stageRun,
            ArtifactKind.ExportVideo,
            exportVideoRelativePath,
            outputProbe.DurationSeconds,
            sampleRate: null,
            channelCount: outputProbe.AudioStreams.FirstOrDefault()?.Channels,
            "export-video",
            cancellationToken).ConfigureAwait(false);
        registeredExportArtifactIds.Add(videoArtifact.Id);
        return (videoHandle.FinalPath, videoArtifact, renderResult);
    }

    private async Task<ExportStageResult> WriteManifestAndFinalizeAsync(
        TranscriptProjectState currentState,
        ExportStageRequest request,
        StageRunRecord stageRun,
        string manifestRelativePath,
        ProjectArtifact manifestArtifact,
        MixPlan mixPlan,
        RenderedSubtitleArtifacts subtitles,
        ProjectArtifact audioArtifact,
        RenderedDubAudio audioRenderResult,
        string videoFinalPath,
        ProjectArtifact videoArtifact,
        ExportRenderResult videoRenderResult,
        string outputPath,
        List<string> transientArtifactPaths,
        List<string> rollbackArtifactPaths,
        CancellationToken cancellationToken)
    {
        List<ExportManifestOutput> outputs =
        [
            new("manifest", GetDeliveryRelativePath(outputPath, GetSidecarManifestPath(outputPath))),
            new("audio", audioArtifact.RelativePath, ExportManifestOutputPathBases.Artifact),
            new("video", GetDeliveryRelativePath(outputPath, outputPath)),
            .. subtitles.SidecarOutputPaths.Select(path => new ExportManifestOutput("subtitle", GetDeliveryRelativePath(outputPath, path)))
        ];
        List<string> warnings = [.. mixPlan.Warnings.Select(static warning => $"{warning.SegmentReference}: {warning.Message}")];
        warnings.AddRange(audioRenderResult.Warnings);
        warnings.AddRange(videoRenderResult.Warnings);
        string? lipSynthesisCompositingWarning = TranscriptWorkflowUtilities
            .BuildLipSynthesisExportCompositingWarning(currentState, artifactStore);
        if (lipSynthesisCompositingWarning is not null)
        {
            warnings.Add(lipSynthesisCompositingWarning);
        }
        ExportManifest finalManifest = BuildManifest(
            currentState,
            request,
            stageRun,
            outputs,
            audioRenderResult.TargetLufs,
            audioRenderResult.AchievedLufs,
            warnings,
            mixPlan);
        await artifactStore.WriteJsonAsync(manifestRelativePath, finalManifest, cancellationToken).ConfigureAwait(false);
        FileFingerprint manifestFingerprint = await fileFingerprintService
            .ComputeAsync(artifactStore.GetPath(manifestRelativePath), cancellationToken)
            .ConfigureAwait(false);
        await mediaAssetRepository.SaveArtifactAsync(
            manifestArtifact with
            {
                Sha256 = manifestFingerprint.Sha256,
                SizeBytes = manifestFingerprint.SizeBytes,
                CreatedAtUtc = DateTimeOffset.UtcNow
            },
            cancellationToken).ConfigureAwait(false);
        List<(string DestinationPath, string ReplacementPath)> replacements = [(outputPath, videoFinalPath)];
        replacements.AddRange(subtitles.Sidecars.Select(sidecar => (sidecar.OutputPath, sidecar.ArtifactPath)));
        replacements.Add((GetSidecarManifestPath(outputPath), artifactStore.GetPath(manifestRelativePath)));
        using var delivery = new ExportDeliveryTransaction(replacements, stageRun.Id);
        StageRunRecord completed;
        try
        {
            await delivery.PrepareAsync(cancellationToken).ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();
            delivery.Apply();
            cancellationToken.ThrowIfCancellationRequested();
            completed = await StageRunHelper
                .CompleteAsync(stageRunStore, stageRun, runtimeReporter: null, cancellationToken, runtimePlanningPreferences)
                .ConfigureAwait(false);
            delivery.Accept();
        }
        catch (Exception originalFailure)
        {
            try
            {
                delivery.Rollback();
            }
            catch (Exception recoveryFailure)
            {
                // Keep cancellation/root cause intact. The outer lifecycle handler records
                // both failures, while the unaccepted journal retains recovery evidence.
                originalFailure.Data[RecoveryFailureDataKey] = recoveryFailure;
                logger?.LogWarning($"Failed to recover export delivery '{outputPath}'. Recovery journal retained.", recoveryFailure);
            }
            throw;
        }
        TryDelete(GetSidecarFailureReportPath(outputPath));
        CleanupTransientExportArtifacts(transientArtifactPaths);
        rollbackArtifactPaths.Clear();
        return new ExportStageResult(
            completed,
            outputPath,
            GetSidecarManifestPath(outputPath),
            audioArtifact.RelativePath,
            videoArtifact.RelativePath,
            subtitles.SidecarOutputPaths,
            warnings);
    }

    private async Task<RenderedDubAudio> RenderNormalizedDubAudioAsync(
        MixPlan mixPlan,
        MediaAsset mediaAsset,
        ExportStageRequest request,
        string normalizedOutputPath,
        CancellationToken cancellationToken)
    {
        string rawPath = Path.Join(
            Path.GetDirectoryName(Path.GetFullPath(normalizedOutputPath))!,
            $"raw-{Guid.NewGuid():N}.wav");
        try
        {
            PreviewRangeRenderResult renderResult = await mixRenderer
                .RenderAsync(
                    new PreviewRangeRenderRequest(
                        mixPlan,
                        StartSeconds: 0d,
                        EndSeconds: mediaAsset.DurationSeconds,
                        rawPath),
                    cancellationToken)
                .ConfigureAwait(false);

            LoudnessTargetResolution target = await ResolveExportTargetLufsAsync(
                mediaAsset.SourceFilePath,
                rawPath,
                request,
                cancellationToken).ConfigureAwait(false);

            LoudnessNormalizationResult loudness = await loudnessNormalizer
                .NormalizeAsync(
                    new LoudnessNormalizationRequest(rawPath, normalizedOutputPath, target.TargetLufs),
                    cancellationToken)
                .ConfigureAwait(false);

            if (!FilePathComparison.AreSame(loudness.OutputPath, normalizedOutputPath))
            {
                await CopyFileAsync(loudness.OutputPath, normalizedOutputPath, cancellationToken).ConfigureAwait(false);
                TryDelete(loudness.OutputPath);
            }

            return new RenderedDubAudio(
                renderResult with { OutputPath = normalizedOutputPath },
                target.TargetLufs,
                loudness.AchievedLufs,
                [.. target.Warnings, .. loudness.Warnings]);
        }
        finally
        {
            TryDelete(rawPath);
        }
    }

    private async Task<LoudnessTargetResolution> ResolveExportTargetLufsAsync(
        string sourceMediaPath,
        string rawMixPath,
        ExportStageRequest request,
        CancellationToken cancellationToken)
    {
        double fallbackTargetLufs = ExportLoudnessTargets.NormalizeTargetLufs(request.TargetLufs);
        if (!request.MatchOriginalLoudness)
        {
            return new LoudnessTargetResolution(fallbackTargetLufs, Warnings: []);
        }

        try
        {
            LoudnessAnalysisResult sourceAnalysis = await loudnessNormalizer
                .AnalyzeAsync(new LoudnessAnalysisRequest(sourceMediaPath), cancellationToken)
                .ConfigureAwait(false);
            LoudnessAnalysisResult rawMixAnalysis = await loudnessNormalizer
                .AnalyzeAsync(new LoudnessAnalysisRequest(rawMixPath), cancellationToken)
                .ConfigureAwait(false);

            if (!double.IsFinite(sourceAnalysis.IntegratedLufs) || !double.IsFinite(rawMixAnalysis.IntegratedLufs))
            {
                return new LoudnessTargetResolution(
                    fallbackTargetLufs,
                    ["Could not match original loudness because loudness analysis returned a non-finite value; using the requested LUFS target."]);
            }

            var warnings = new List<string>();
            warnings.AddRange(sourceAnalysis.Warnings);
            warnings.AddRange(rawMixAnalysis.Warnings);

            double targetLufs = ExportLoudnessTargets.NormalizeTargetLufs(sourceAnalysis.IntegratedLufs);
            double upwardBoostDb = targetLufs - rawMixAnalysis.IntegratedLufs;
            if (upwardBoostDb > MaxMatchOriginalUpwardBoostDb)
            {
                double cappedTargetLufs = ExportLoudnessTargets.NormalizeTargetLufs(
                    rawMixAnalysis.IntegratedLufs + MaxMatchOriginalUpwardBoostDb);
                warnings.Add(
                    $"Match original loudness boost was capped at {MaxMatchOriginalUpwardBoostDb:0.#} dB; using {cappedTargetLufs:0.##} LUFS instead of {targetLufs:0.##} LUFS.");
                targetLufs = cappedTargetLufs;
            }

            return new LoudnessTargetResolution(targetLufs, warnings);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            return new LoudnessTargetResolution(
                fallbackTargetLufs,
                [$"Could not match original loudness because analysis failed ({ex.Message}); using the requested LUFS target."]);
        }
    }

    private async Task<RenderedSubtitleArtifacts> WriteSubtitleArtifactsAsync(
        StageRunRecord stageRun,
        ExportStageRequest request,
        IReadOnlyList<SubtitleCue> cues,
        string outputPath,
        CancellationToken cancellationToken)
    {
        var sidecarOutputPaths = new List<string>();
        var sidecars = new List<SubtitleSidecarOutput>();
        var artifactPaths = new List<string>();
        string? burnInSubtitlePath = null;
        foreach (ExportSubtitleFormat format in request.SubtitleFormats.Distinct())
        {
            string relativePath = ProjectArtifactPaths.GetExportSubtitleRelativePath(stageRun.Id, GetSubtitleExtension(format));
            string artifactPath = await WriteSubtitleArtifactAsync(relativePath, cues, format, cancellationToken).ConfigureAwait(false);
            string outputSidecarPath = GetSubtitleSidecarPath(outputPath, format);
            sidecars.Add(new SubtitleSidecarOutput(artifactPath, outputSidecarPath));
            artifactPaths.Add(artifactPath);
            sidecarOutputPaths.Add(outputSidecarPath);

            if (request.BurnInSubtitles && format is ExportSubtitleFormat.Ass)
            {
                burnInSubtitlePath = artifactPath;
            }
        }

        if (request.BurnInSubtitles && burnInSubtitlePath is null)
        {
            string relativePath = ProjectArtifactPaths.GetExportSubtitleRelativePath(stageRun.Id, ".burnin.ass");
            burnInSubtitlePath = await WriteSubtitleArtifactAsync(relativePath, cues, ExportSubtitleFormat.Ass, cancellationToken)
                .ConfigureAwait(false);
            artifactPaths.Add(burnInSubtitlePath);
        }

        return new RenderedSubtitleArtifacts(sidecarOutputPaths, sidecars, artifactPaths, burnInSubtitlePath);
    }

    private async Task<string> WriteSubtitleArtifactAsync(
        string relativePath,
        IReadOnlyList<SubtitleCue> cues,
        ExportSubtitleFormat format,
        CancellationToken cancellationToken)
    {
        await using ArtifactWriteHandle handle = artifactStore.CreateWriteHandle(relativePath);
        string contents = subtitleExportService.ExportNormalized(cues, format);
        await File.WriteAllTextAsync(handle.TemporaryPath, contents, SubtitleEncoding, cancellationToken).ConfigureAwait(false);
        await artifactStore.CommitAsync(handle, cancellationToken).ConfigureAwait(false);
        return handle.FinalPath;
    }

    private SubtitleCue[] BuildSubtitleCues(TranscriptProjectState currentState, ExportStageRequest request) =>
        request.SubtitleSource switch
        {
            ExportSubtitleSource.Translated => subtitleExportService.BuildTranslatedCues(GetRenderedTranslatedSegments(currentState)).ToArray(),
            ExportSubtitleSource.Transcript => subtitleExportService.BuildTranscriptCues(currentState.TranscriptSegments).ToArray(),
            ExportSubtitleSource.Bilingual => subtitleExportService
                .BuildBilingualCues(currentState.TranscriptSegments, GetRenderedTranslatedSegments(currentState))
                .ToArray(),
            _ => throw new ArgumentOutOfRangeException(nameof(request.SubtitleSource), "Unsupported export subtitle source.")
        };

    private static TranslatedSegment[] GetRenderedTranslatedSegments(TranscriptProjectState currentState)
    {
        HashSet<int> renderedSegmentIndices = currentState.TranscriptSegments
            .Select(static segment => segment.SegmentIndex)
            .ToHashSet();
        HashSet<int>? staleTranslatedSegmentIndices = currentState.IsTranslationStale
            ? currentState.StaleTranslatedSegmentIndices.ToHashSet()
            : null;
        return currentState.TranslatedSegments
            .Where(segment =>
                renderedSegmentIndices.Contains(segment.SegmentIndex) &&
                (staleTranslatedSegmentIndices is null || !staleTranslatedSegmentIndices.Contains(segment.SegmentIndex)))
            .ToArray();
    }

    private static IReadOnlyList<ExportFailureCause> BuildTakeFailureCauses(MixPlan mixPlan) =>
        mixPlan.Warnings
            .Select(static warning => new ExportFailureCause(
                ResolveTakeFailureCode(warning.Code),
                warning.Message,
                warning.SegmentId,
                warning.SegmentIndex))
            .ToArray();

    private static string ResolveTakeFailureCode(MixPlanWarningCode code) =>
        code switch
        {
            MixPlanWarningCode.MissingTake => "missing-take",
            MixPlanWarningCode.StaleTake => "stale-take",
            MixPlanWarningCode.MissingTakeArtifact => "missing-take-artifact",
            MixPlanWarningCode.LipSyncArtifactMissing => "lip-sync-artifact-missing",
            MixPlanWarningCode.TakeExceedsSourceAudio => "take-exceeds-source-audio",
            _ => "invalid-take"
        };

    private static IReadOnlyList<ExportFailureCause> VerifyOutput(MediaAsset mediaAsset, MediaProbeSnapshot outputProbe)
    {
        var causes = new List<ExportFailureCause>();
        if (outputProbe.AudioStreams.Count == 0)
        {
            causes.Add(new ExportFailureCause("missing-audio-stream", "Exported output does not contain an audio stream."));
        }

        if (outputProbe.VideoStreams.Count == 0)
        {
            causes.Add(new ExportFailureCause("missing-video-stream", "Exported output does not contain a video stream."));
        }

        if (!double.IsFinite(outputProbe.DurationSeconds) || !double.IsFinite(mediaAsset.DurationSeconds))
        {
            causes.Add(new ExportFailureCause(
                "invalid-duration",
                "Exported output duration could not be compared with the source duration."));
            return causes;
        }

        double durationDelta = Math.Abs(outputProbe.DurationSeconds - mediaAsset.DurationSeconds);
        if (durationDelta > DurationToleranceSeconds)
        {
            causes.Add(new ExportFailureCause(
                "duration-tolerance-exceeded",
                $"Exported output duration differs from source by {durationDelta:F3} seconds."));
        }

        return causes;
    }

    private async Task FailWithReportAsync(
        TranscriptProjectState currentState,
        StageRunRecord stageRun,
        string failureRelativePath,
        string outputPath,
        IReadOnlyList<ExportFailureCause> causes,
        CancellationToken cancellationToken)
    {
        ExportFailureReport report = new(
            currentState.ProjectState.Project.Id,
            stageRun.Id,
            DateTimeOffset.UtcNow,
            causes);
        string reason = string.Join("; ", causes.Select(static cause => cause.Message));
        Exception? reportWriteException = null;
        try
        {
            await WriteFailureReportAsync(failureRelativePath, outputPath, report, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            reportWriteException = ex;
        }

        await StageRunHelper
            .FailAsync(stageRunStore, stageRun, runtimeReporter: null, reason, CancellationToken.None, runtimePlanningPreferences)
            .ConfigureAwait(false);
        throw new ExportStageException(reason, report, GetSidecarFailureReportPath(outputPath), reportWriteException);
    }

    private async Task TryWriteFailureReportAsync(string relativePath, string outputPath, ExportFailureReport report)
    {
        try
        {
            await WriteFailureReportAsync(relativePath, outputPath, report, CancellationToken.None).ConfigureAwait(false);
        }
        catch (Exception reportWriteException)
        {
            logger?.LogWarning(
                $"Failed to write export failure report. {reportWriteException.GetType().Name}: {reportWriteException.Message}");
        }
    }

    private async Task WriteFailureReportAsync(
        string failureRelativePath,
        string outputPath,
        ExportFailureReport report,
        CancellationToken cancellationToken)
    {
        await artifactStore.WriteJsonAsync(failureRelativePath, report, cancellationToken).ConfigureAwait(false);
        await CopyArtifactIfExistsAsync(failureRelativePath, GetSidecarFailureReportPath(outputPath), cancellationToken)
            .ConfigureAwait(false);
    }

    private async Task<ProjectArtifact> RegisterArtifactAsync(
        TranscriptProjectState currentState,
        MediaAsset mediaAsset,
        StageRunRecord stageRun,
        ArtifactKind kind,
        string relativePath,
        double? durationSeconds,
        int? sampleRate,
        int? channelCount,
        string provenance,
        CancellationToken cancellationToken)
    {
        string fullPath = artifactStore.GetPath(relativePath);
        FileFingerprint fingerprint = await fileFingerprintService.ComputeAsync(fullPath, cancellationToken).ConfigureAwait(false);
        var artifact = new ProjectArtifact(
            Guid.NewGuid(),
            currentState.ProjectState.Project.Id,
            mediaAsset.Id,
            kind,
            relativePath,
            fingerprint.Sha256,
            fingerprint.SizeBytes,
            durationSeconds,
            sampleRate,
            channelCount,
            DateTimeOffset.UtcNow,
            stageRun.Id,
            provenance);
        await mediaAssetRepository.SaveArtifactAsync(artifact, cancellationToken).ConfigureAwait(false);
        return artifact;
    }

    private static ExportManifest BuildManifest(
        TranscriptProjectState currentState,
        ExportStageRequest request,
        StageRunRecord stageRun,
        IReadOnlyList<ExportManifestOutput> outputs,
        double targetLufs,
        double? achievedLufs,
        IReadOnlyList<string> warnings,
        MixPlan? mixPlan) =>
        ExportManifestBuilder.Build(new ExportManifestBuildRequest(
            currentState.ProjectState.Project.Id,
            currentState.TranslatedSegments,
            currentState.TtsTakes,
            GetContributingStageRuns(currentState, mixPlan),
            stageRun.Id,
            currentState.TranscriptLanguage,
            currentState.CurrentTranslationRevision?.TargetLanguage ?? currentState.SelectedTranslationTargetLanguage,
            request.Container,
            targetLufs,
            achievedLufs,
            outputs,
            warnings,
            RenderedSegmentIndices: currentState.TranscriptSegments.Select(static segment => segment.SegmentIndex).ToArray(),
            Gating: BuildExportGating(request, mixPlan?.SourceAudioKind ?? ArtifactKind.NormalizedAudio),
            HasTimelineMedia: TimelineMediaMixPreparer.GetPlacements(currentState).Count > 0));

    // Persist the export-gating flag values for this run so StageArtifactResumeEvaluator can
    // compare them against a later run's snapshot. Produced by the same ExportResumeGating helper
    // the execution snapshot uses, so capture and comparison cannot diverge.
    private static ExportManifestGating BuildExportGating(ExportStageRequest request, ArtifactKind sourceAudioKind) =>
        new(ExportResumeGating.Build(
            request.Container,
            sourceAudioKind,
            request.ApplyTimbrePolish,
            request.RestoreOriginalPan,
            request.MatchOriginalLoudness,
            request.BurnInSubtitles,
            request.SubtitleSource,
            // Persist the RAW requested formats so the token matches the snapshot side exactly:
            // null (pipeline default) -> "default" on both sides, avoiding the spurious rerun that
            // resolving to [Srt] would cause for every default-subtitle export.
            ExportResumeGating.SubtitleFormatsTokenFromRawOptions(request.RawSubtitleFormats),
            request.VideoEncoder,
            request.TargetLufs,
            ExportStageRequestBuilder.ToMixGainSettings(request)));

    private static StageRunRecord[] GetContributingStageRuns(TranscriptProjectState currentState, MixPlan? mixPlan)
    {
        var contributingStageRunIds = new HashSet<Guid>();
        AddStageRunId(currentState.CurrentTranscriptRevision?.StageRunId, contributingStageRunIds);
        AddStageRunId(currentState.CurrentTranslationRevision?.StageRunId, contributingStageRunIds);

        // Provenance must reflect the takes that the MixPlan actually selected for rendering,
        // not "newest take per segment". Selection is governed by MixPlan.SpeechClips (which in
        // turn honor candidate-group selection) so its TakeIds are the authoritative source.
        if (mixPlan is not null)
        {
            Dictionary<Guid, TtsTake> takesById = currentState.TtsTakes.ToDictionary(static take => take.Id);
            foreach (MixSpeechClip clip in mixPlan.SpeechClips)
            {
                if (clip.TakeId is Guid takeId && takesById.TryGetValue(takeId, out TtsTake? take))
                {
                    AddStageRunId(take.StageRunId, contributingStageRunIds);
                }
            }
        }

        return currentState.StageRuns
            .Where(stageRun => contributingStageRunIds.Contains(stageRun.Id))
            .ToArray();
    }

    private static void AddStageRunId(Guid? stageRunId, ISet<Guid> stageRunIds)
    {
        if (stageRunId is Guid id && id != Guid.Empty)
        {
            stageRunIds.Add(id);
        }
    }

    private static void ValidateRequest(TranscriptProjectState currentState, MediaAsset mediaAsset, ExportStageRequest request)
    {
        if (request.ProjectId == Guid.Empty ||
            request.ProjectId != currentState.ProjectState.Project.Id)
        {
            throw new InvalidOperationException("Export request does not match the loaded project.");
        }

        ArgumentException.ThrowIfNullOrWhiteSpace(request.OutputPath);
        ArgumentNullException.ThrowIfNull(request.SubtitleFormats);
        if (Directory.Exists(request.OutputPath))
        {
            throw new InvalidOperationException("Export output path must be a file path, not a directory.");
        }

        if (request.SubtitleSource is not ExportSubtitleSource.Translated and not ExportSubtitleSource.Transcript and not ExportSubtitleSource.Bilingual)
        {
            throw new InvalidOperationException("Export subtitle source is not supported.");
        }

        if (request.Container is not ExportOutputContainer.Mp4 and not ExportOutputContainer.Mkv)
        {
            throw new InvalidOperationException("Export output container is not supported.");
        }

        string requestedExtension = Path.GetExtension(request.OutputPath);
        string expectedExtension = GetContainerExtension(request.Container);
        if (!string.Equals(requestedExtension, expectedExtension, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException(
                $"Export output extension must match the selected container ({expectedExtension}).");
        }

        if (currentState.CurrentTranscriptRevision is null)
        {
            throw new InvalidOperationException("Export requires a transcript revision.");
        }

        if (string.IsNullOrWhiteSpace(mediaAsset.SourceFilePath))
        {
            throw new InvalidOperationException("Export requires a source media file path.");
        }
    }

    private static IReadOnlyList<ExportFailureCause> BuildPreflightFailureCauses(
        TranscriptProjectState currentState,
        MediaAsset mediaAsset)
    {
        var causes = new List<ExportFailureCause>();
        if (!File.Exists(mediaAsset.SourceFilePath))
        {
            causes.Add(new ExportFailureCause(
                "missing-source-media",
                "Export requires the source media file to be available on disk."));
        }

        if (!mediaAsset.HasVideo)
        {
            causes.Add(new ExportFailureCause(
                "missing-video-stream",
                "Export requires source media with a video stream."));
        }

        if (currentState.ExportTools is { IsAvailable: false } exportTools)
        {
            causes.Add(new ExportFailureCause(
                "ffmpeg-unavailable",
                exportTools.Message ?? "Export requires FFmpeg and ffprobe to be installed or configured."));
        }

        return causes;
    }

    private static string GetContainerExtension(ExportOutputContainer container) =>
        container switch
        {
            ExportOutputContainer.Mkv => ".mkv",
            ExportOutputContainer.Mp4 => ".mp4",
            _ => throw new ArgumentOutOfRangeException(nameof(container), "Unsupported export container.")
        };

    private static string GetSubtitleExtension(ExportSubtitleFormat format) =>
        format switch
        {
            ExportSubtitleFormat.Vtt => ".vtt",
            ExportSubtitleFormat.Ass => ".ass",
            _ => ".srt"
        };

    private static string GetSubtitleSidecarPath(string outputPath, ExportSubtitleFormat format) =>
        Path.ChangeExtension(outputPath, GetSubtitleExtension(format));

    private static string GetSidecarManifestPath(string outputPath)
    {
        string directory = Path.GetDirectoryName(Path.GetFullPath(outputPath))!;
        string fileName = Path.GetFileNameWithoutExtension(outputPath);
        return Path.Join(directory, $"{fileName}.export-manifest.json");
    }

    private static string GetSidecarFailureReportPath(string outputPath)
    {
        string directory = Path.GetDirectoryName(Path.GetFullPath(outputPath))!;
        string fileName = Path.GetFileNameWithoutExtension(outputPath);
        return Path.Join(directory, $"{fileName}.export-failure.json");
    }

    private static string GetDeliveryRelativePath(string outputPath, string sidecarPath) =>
        Path.GetRelativePath(
            Path.GetDirectoryName(Path.GetFullPath(outputPath))!,
            Path.GetFullPath(sidecarPath));

    private async Task CopyArtifactIfExistsAsync(
        string relativePath,
        string destinationPath,
        CancellationToken cancellationToken)
    {
        string sourcePath = artifactStore.GetPath(relativePath);
        if (!File.Exists(sourcePath))
        {
            return;
        }

        DeliveryReplacement replacement = await PrepareDeliveryReplacementAsync(
            sourcePath,
            destinationPath,
            cancellationToken).ConfigureAwait(false);
        try
        {
            File.Move(replacement.TemporaryPath, replacement.DestinationPath, overwrite: true);
        }
        finally
        {
            TryDelete(replacement.TemporaryPath);
        }
    }

    private static async Task CopyFileAsync(string sourcePath, string destinationPath, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(destinationPath))!);
        await using FileStream source = File.Open(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        await using FileStream destination = File.Create(destinationPath);
        await source.CopyToAsync(destination, cancellationToken).ConfigureAwait(false);
    }

    private async Task<DeliveryReplacement> PrepareDeliveryReplacementAsync(
        string sourcePath,
        string destinationPath,
        CancellationToken cancellationToken)
    {
        string tempPath = GetAdjacentTemporaryPath(destinationPath, ".tmp");
        try
        {
            await CopyFileAsync(sourcePath, tempPath, cancellationToken).ConfigureAwait(false);
            return new DeliveryReplacement(tempPath, Path.GetFullPath(destinationPath));
        }
        catch
        {
            TryDelete(tempPath);
            throw;
        }
    }

    private void CleanupPartialExportOutputs(
        IEnumerable<string> transientArtifactPaths)
    {
        foreach (string path in transientArtifactPaths)
        {
            TryDelete(path);
        }
    }

    private void CleanupTransientExportArtifacts(ICollection<string> transientArtifactPaths)
    {
        foreach (string path in transientArtifactPaths)
        {
            TryDelete(path);
        }

        transientArtifactPaths.Clear();
    }

    private static string GetAdjacentTemporaryPath(string destinationPath, string suffix)
    {
        string fullPath = Path.GetFullPath(destinationPath);
        string directory = Path.GetDirectoryName(fullPath)!;
        string fileName = Path.GetFileName(fullPath);
        Directory.CreateDirectory(directory);
        return Path.Join(directory, $".{fileName}.{Guid.NewGuid():N}{suffix}");
    }

    private async Task CleanupRegisteredExportArtifactsAsync(
        IEnumerable<Guid> registeredExportArtifactIds,
        CancellationToken cancellationToken)
    {
        foreach (Guid artifactId in registeredExportArtifactIds)
        {
            try
            {
                await mediaAssetRepository.DeleteArtifactAsync(artifactId, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (LogCleanupFailure("Failed to delete export artifact registration '{ArtifactId}'.", artifactId, ex))
            {
            }
        }
    }

    private void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (Exception ex) when (LogCleanupFailure("Failed to delete export cleanup file '{Path}'.", path, ex))
        {
        }
    }

    private void TryDeleteIntermediateOutput(string path, params string[] reservedPaths)
    {
        foreach (string reservedPath in reservedPaths)
        {
            if (FilePathComparison.AreSame(path, reservedPath))
            {
                return;
            }
        }

        TryDelete(path);
    }

    private bool LogCleanupFailure(string message, Exception exception)
    {
        logger?.LogWarning(exception, message);
        return true;
    }

    private bool LogCleanupFailure<T>(string message, T argument, Exception exception)
    {
        logger?.LogWarning(exception, message, argument);
        return true;
    }

    private sealed record RenderedSubtitleArtifacts(
        IReadOnlyList<string> SidecarOutputPaths,
        IReadOnlyList<SubtitleSidecarOutput> Sidecars,
        IReadOnlyList<string> ArtifactPaths,
        string? BurnInSubtitlePath);

    private sealed record SubtitleSidecarOutput(
        string ArtifactPath,
        string OutputPath);

    private sealed record RenderedDubAudio(
        PreviewRangeRenderResult RenderResult,
        double TargetLufs,
        double? AchievedLufs,
        IReadOnlyList<string> Warnings);

    private sealed record LoudnessTargetResolution(
        double TargetLufs,
        IReadOnlyList<string> Warnings);

    private sealed record DeliveryReplacement(
        string TemporaryPath,
        string DestinationPath);
}
