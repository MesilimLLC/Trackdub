using Trackdub.Application.Artifacts;
using Trackdub.Application.Logging;
using Trackdub.Contracts;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.LipSynthesis;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.Projects;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.LipSynthesis;
using Trackdub.Domain.Media;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.LipSynthesis;

/// <summary>One speaker turn (a time window in the original video) to consider for synthesis.</summary>
public sealed record LipSynthesisTurn(
    Guid SegmentId,
    TimeSpan Start,
    TimeSpan End,
    string? SpeakerId);

public sealed record LipSynthesisStageRequest(
    Guid ProjectId,
    MediaAsset MediaAsset,
    /// <summary>Absolute path to the ORIGINAL video — the authority. Never overwritten.</summary>
    string SourceVideoPath,
    /// <summary>Absolute path to the dubbed audio (improved M22 take when available, else post-atempo).</summary>
    string DubbedAudioPath,
    IReadOnlyList<LipSynthesisTurn> SpeakerTurns,
    bool IsEnabled = true,
    /// <summary>
    /// Whether the selected synthesis provider is license-approved for the active commercial mode.
    /// Resolved upstream from the manifest gate; false yields a clean SkippedLicenseGate per turn.
    /// </summary>
    bool IsLicenseApproved = true,
    /// <summary>
    /// When true, allows an experimental-lane engine to run after explicit user opt-in.
    /// </summary>
    bool AllowExperimentalExecution = false,
    string? PreferredModelAlias = null,
    LipSynthesisOptions? Options = null);

public sealed record LipSynthesisStageResult(
    StageRunRecord StageRun,
    IReadOnlyList<LipSynthesisSegment> Segments);

/// <summary>
/// M23 video lip-synthesis stage. Repairs mouth motion in the ORIGINAL footage per speaker turn,
/// writing a patched per-turn clip and never overwriting the source video. Each turn passes face
/// quality guards (no-face, low-confidence, non-frontal, occlusion, unstable-crop) before the
/// engine runs; skipped turns preserve the original frames. Stage-level gates (disabled, license,
/// runtime-unavailable) must never block audio-only export.
/// </summary>
public sealed class LipSynthesisStageHandler(
    ILipSynthesisEngine lipSynthesisEngine,
    IFaceDetector faceDetector,
    IFaceLandmarkProvider faceLandmarkProvider,
    IFacePoseEstimator facePoseEstimator,
    IArtifactStore artifactStore,
    IProjectStageRunStore stageRunStore,
    ILipSynthesisSegmentRepository segmentRepository,
    PipelineDegradationWriter? degradationWriter = null,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null,
    IApplicationLogger? logger = null,
    IFileFingerprintService? fileFingerprintService = null,
    IMediaAssetRepository? mediaAssetRepository = null)
{
    private readonly ILipSynthesisSegmentRepository _segmentRepository = segmentRepository ?? throw new ArgumentNullException(nameof(segmentRepository));
    private static readonly LipSynthesisOptions DefaultOptions = new();

    public async Task<LipSynthesisStageResult> HandleAsync(
        LipSynthesisStageRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);

        StageRunRecord stageRun = await StageRunHelper
            .StartAsync(stageRunStore, request.ProjectId, StageNames.LipSynthesis, cancellationToken)
            .ConfigureAwait(false);

        try
        {
            if (!request.IsEnabled)
            {
                stageRun = await SkipStageAsync(stageRun, "Lip-synthesis stage is disabled for this run.",
                    "LipSynthesisDisabled", request).ConfigureAwait(false);
                return new LipSynthesisStageResult(stageRun, []);
            }

            if (request.SpeakerTurns.Count == 0)
            {
                stageRun = await SkipStageAsync(stageRun, "No speaker turns provided; lip-synthesis prerequisite not met.",
                    "LipSynthesisNoTurns", request).ConfigureAwait(false);
                return new LipSynthesisStageResult(stageRun, []);
            }

            if (string.IsNullOrWhiteSpace(request.DubbedAudioPath) || !File.Exists(request.DubbedAudioPath))
            {
                stageRun = await SkipStageAsync(stageRun,
                    "A rendered dubbed-speech driver is required for lip repair. Generate current dubbed audio and identify speakers before running repair.",
                    "LipSynthesisNoDubbedMix", request).ConfigureAwait(false);
                return new LipSynthesisStageResult(stageRun, []);
            }

            // License gate is independent — a non-approved model is always blocked regardless of
            // experimental opt-in. AllowExperimentalExecution controls engine maturity, not licensing.
            if (!request.IsLicenseApproved)
            {
                var gated = BuildUniformSkip(request, LipSynthesisSegmentStatus.SkippedLicenseGate,
                    "Synthesis provider is not license-approved for the active commercial mode.");
                await _segmentRepository.SaveAllAsync(request.ProjectId, stageRun.Id, gated, cancellationToken)
                    .ConfigureAwait(false);
                cancellationToken.ThrowIfCancellationRequested();
                stageRun = await SkipStageAsync(stageRun, "Lip-synthesis blocked by license gate.",
                    "LipSynthesisLicenseGate", request).ConfigureAwait(false);
                return new LipSynthesisStageResult(stageRun, gated);
            }

            // Experimental engines (manifest lane != commercial, commercial_use_verified=false,
            // or commercial_allowed=false) must not run until the user explicitly opts in.
            if (lipSynthesisEngine.IsExperimental && !request.AllowExperimentalExecution)
            {
                var gated = BuildUniformSkip(request, LipSynthesisSegmentStatus.SkippedExperimentalGate,
                    "Synthesis engine is marked experimental. Run again after downloading models and confirming experimental execution in stage options.");
                await _segmentRepository.SaveAllAsync(request.ProjectId, stageRun.Id, gated, cancellationToken)
                    .ConfigureAwait(false);
                cancellationToken.ThrowIfCancellationRequested();
                stageRun = await SkipStageAsync(stageRun,
                    "Lip-synthesis blocked: engine is experimental.",
                    "LipSynthesisExperimentalGate", request).ConfigureAwait(false);
                return new LipSynthesisStageResult(stageRun, gated);
            }

            if (!lipSynthesisEngine.IsAvailable || !faceDetector.IsAvailable
                || !faceLandmarkProvider.IsAvailable || !facePoseEstimator.IsAvailable)
            {
                var gated = BuildUniformSkip(request, LipSynthesisSegmentStatus.SkippedRuntimeUnavailable,
                    "Synthesis engine or a face-analysis provider is not installed/available.");
                await _segmentRepository.SaveAllAsync(request.ProjectId, stageRun.Id, gated, cancellationToken)
                    .ConfigureAwait(false);
                cancellationToken.ThrowIfCancellationRequested();
                stageRun = await SkipStageAsync(stageRun, "Lip-synthesis runtime unavailable.",
                    "LipSynthesisRuntimeUnavailable", request).ConfigureAwait(false);
                return new LipSynthesisStageResult(stageRun, gated);
            }

            LipSynthesisOptions options = request.Options ?? DefaultOptions;
            var segments = new List<LipSynthesisSegment>(request.SpeakerTurns.Count);
            foreach (LipSynthesisTurn turn in request.SpeakerTurns)
            {
                cancellationToken.ThrowIfCancellationRequested();
                LipSynthesisSegment segment = await ProcessTurnAsync(turn, options, stageRun.Id, request, cancellationToken)
                    .ConfigureAwait(false);
                cancellationToken.ThrowIfCancellationRequested();
                // Artifact is already published; record its row even if cancellation just arrived.
                await _segmentRepository.SaveAllAsync(request.ProjectId, stageRun.Id, [segment], CancellationToken.None)
                    .ConfigureAwait(false);
                segments.Add(segment);
            }

            cancellationToken.ThrowIfCancellationRequested();
            stageRun = await FinalizeStageAsync(stageRun, segments).ConfigureAwait(false);
            return new LipSynthesisStageResult(stageRun, segments);
        }
        catch (OperationCanceledException)
        {
            await StageRunHelper
                .CancelAsync(stageRunStore, stageRun, lipSynthesisEngine as IStageRuntimeExecutionReporter,
                    "LipSynthesis canceled.", CancellationToken.None, runtimePlanningPreferences, logger)
                .ConfigureAwait(false);
            throw;
        }
        catch (Exception ex)
        {
            await StageRunHelper
                .FailAsync(stageRunStore, stageRun, lipSynthesisEngine as IStageRuntimeExecutionReporter,
                    ex.Message, CancellationToken.None, runtimePlanningPreferences, logger)
                .ConfigureAwait(false);

            await WriteDegradationAsync(request, stageRun.Id, "LipSynthesisUnhandledFailure",
                "Unhandled exception in lip-synthesis stage.", ex.Message,
                "Check synthesis provider, face analysis providers, and video paths.").ConfigureAwait(false);
            throw;
        }
    }

    private async Task<LipSynthesisSegment> ProcessTurnAsync(
        LipSynthesisTurn turn,
        LipSynthesisOptions options,
        Guid stageRunId,
        LipSynthesisStageRequest request,
        CancellationToken cancellationToken)
    {
        var analysisRequest = new FaceAnalysisRequest(request.SourceVideoPath, turn.Start, turn.End, turn.SpeakerId);

        FaceDetectionResult detection = await faceDetector
            .DetectPrimaryFaceAsync(analysisRequest, cancellationToken).ConfigureAwait(false);

        if (!detection.FaceFound || detection.PrimaryFace is null)
            return Skip(turn, LipSynthesisSegmentStatus.SkippedNoFace, "No usable face detected in the turn.", null);

        if (detection.Confidence < options.MinFaceConfidence)
            return Skip(turn, LipSynthesisSegmentStatus.SkippedLowConfidence,
                $"Face confidence {detection.Confidence:F2} below threshold {options.MinFaceConfidence:F2}.",
                detection.Confidence);

        FacePoseEstimate pose = await facePoseEstimator
            .EstimatePoseAsync(analysisRequest, cancellationToken).ConfigureAwait(false);

        if (Math.Abs(pose.YawDegrees) > options.MaxYawDegrees || Math.Abs(pose.PitchDegrees) > options.MaxPitchDegrees)
            return Skip(turn, LipSynthesisSegmentStatus.SkippedNonFrontal,
                $"Head pose (yaw {pose.YawDegrees:F0}°, pitch {pose.PitchDegrees:F0}°) exceeds frontal limits.",
                detection.Confidence);

        FaceLandmarkResult landmarks = await faceLandmarkProvider
            .DetectLandmarksAsync(analysisRequest, cancellationToken).ConfigureAwait(false);

        if (!landmarks.LandmarksFound || !landmarks.IsStable)
            return Skip(turn, LipSynthesisSegmentStatus.SkippedUnstableCrop,
                "Facial landmarks missing or unstable across the turn.", detection.Confidence);

        if (landmarks.MouthOccluded)
            return Skip(turn, LipSynthesisSegmentStatus.SkippedOccluded,
                "Mouth region is occluded in the turn.", detection.Confidence);

        // All quality guards passed — attempt synthesis.
        LipSynthesisResult result = await lipSynthesisEngine
            .SynthesizeTurnAsync(
                new LipSynthesisRequest(
                    OriginalVideoPath: request.SourceVideoPath,
                    DubbedAudioPath: request.DubbedAudioPath,
                    SegmentId: turn.SegmentId,
                    TurnStart: turn.Start,
                    TurnEnd: turn.End,
                    SpeakerId: turn.SpeakerId,
                    Options: options,
                    PreferredModelAlias: request.PreferredModelAlias),
                cancellationToken)
            .ConfigureAwait(false);

        if (result.Status == LipSynthesisEngineStatus.Failed)
            return new LipSynthesisSegment(turn.SegmentId, LipSynthesisSegmentStatus.Failed, turn.SpeakerId,
                turn.Start, turn.End, detection.Confidence, null, null,
                result.FailureReason ?? "Synthesis engine returned Failed.",
                result.ProviderId, result.ModelId, lipSynthesisEngine.IsExperimental, DateTimeOffset.UtcNow);

        if (result.Status == LipSynthesisEngineStatus.Skipped || result.PatchedClipPath is null)
            // Engine declined synthesis on quality grounds the coarse guards did not catch; preserve frames.
            return new LipSynthesisSegment(turn.SegmentId, LipSynthesisSegmentStatus.SkippedLowConfidence, turn.SpeakerId,
                turn.Start, turn.End, detection.Confidence, null,
                result.SkipReason ?? "Synthesis engine skipped the turn.", null,
                result.ProviderId, result.ModelId, lipSynthesisEngine.IsExperimental, DateTimeOffset.UtcNow);

        // Persist the patched clip as a LipSynthesisTake artifact. Source video is never modified.
        string relativeOutputPath = ProjectArtifactPaths.GetLipSynthesisTakeRelativePath(turn.SegmentId, stageRunId);
        await using (var tx = new ArtifactWriteTransaction(artifactStore.CreateWriteHandle(relativeOutputPath)))
        {
            File.Copy(result.PatchedClipPath, tx.TemporaryPath, overwrite: true);
            await tx.CommitAsync(artifactStore, cancellationToken).ConfigureAwait(false);
            try { File.Delete(result.PatchedClipPath); } catch { }
        }

        await RegisterPatchedArtifactAsync(request, turn, stageRunId, relativeOutputPath, cancellationToken)
            .ConfigureAwait(false);

        return new LipSynthesisSegment(turn.SegmentId, LipSynthesisSegmentStatus.Synthesized, turn.SpeakerId,
            turn.Start, turn.End, detection.Confidence, relativeOutputPath, null, null,
            result.ProviderId, result.ModelId, lipSynthesisEngine.IsExperimental, DateTimeOffset.UtcNow);
    }

    private async Task RegisterPatchedArtifactAsync(
        LipSynthesisStageRequest request,
        LipSynthesisTurn turn,
        Guid stageRunId,
        string relativeOutputPath,
        CancellationToken cancellationToken)
    {
        if (mediaAssetRepository is null)
        {
            await WriteDegradationAsync(request, stageRunId, "LipSynthesisArtifactNotRegistered",
                "Patched clip written but its metadata could not be registered; preview/export will fall back to the original video.",
                $"SegmentId={turn.SegmentId}, ArtifactPath={relativeOutputPath}",
                "Ensure IMediaAssetRepository is registered in the DI container.").ConfigureAwait(false);
            logger?.LogWarning(
                "LipSynthesisTake artifact metadata not registered; IMediaAssetRepository unavailable. SegmentId={SegmentId}",
                turn.SegmentId);
            return;
        }

        string absOutputPath = artifactStore.GetPath(relativeOutputPath);
        string sha256 = "unknown";
        long sizeBytes = 0L;
        if (fileFingerprintService is not null)
        {
            FileFingerprint fp = await fileFingerprintService.ComputeAsync(absOutputPath, cancellationToken).ConfigureAwait(false);
            sha256 = fp.Sha256;
            sizeBytes = fp.SizeBytes;
        }

        await mediaAssetRepository.SaveArtifactAsync(
            new ProjectArtifact(
                Id: Guid.NewGuid(),
                ProjectId: request.ProjectId,
                MediaAssetId: request.MediaAsset.Id,
                Kind: ArtifactKind.LipSynthesisTake,
                RelativePath: relativeOutputPath,
                Sha256: sha256,
                SizeBytes: sizeBytes,
                DurationSeconds: (turn.End - turn.Start).TotalSeconds,
                SampleRate: null,
                ChannelCount: null,
                CreatedAtUtc: DateTimeOffset.UtcNow,
                StageRunId: stageRunId,
                Provenance: $"lipsynthesis:turn:{turn.SegmentId:N}"),
            cancellationToken).ConfigureAwait(false);
    }

    private async Task<StageRunRecord> FinalizeStageAsync(StageRunRecord stageRun, List<LipSynthesisSegment> segments)
    {
        bool anySynthesized = segments.Any(s => s.Status == LipSynthesisSegmentStatus.Synthesized);
        bool anyFailed = segments.Any(s => s.Status == LipSynthesisSegmentStatus.Failed);
        bool allSkipped = segments.All(s =>
            s.Status is not (LipSynthesisSegmentStatus.Synthesized or LipSynthesisSegmentStatus.Failed));

        var reporter = lipSynthesisEngine as IStageRuntimeExecutionReporter;

        if (allSkipped)
            return await StageRunHelper
                .SkipAsync(stageRunStore, stageRun, reporter,
                    "All speaker turns skipped during lip-synthesis.", CancellationToken.None,
                    runtimePlanningPreferences, logger)
                .ConfigureAwait(false);

        if (anySynthesized && !anyFailed && segments.All(s => s.Status == LipSynthesisSegmentStatus.Synthesized))
            return await StageRunHelper
                .CompleteAsync(stageRunStore, stageRun, reporter, CancellationToken.None, runtimePlanningPreferences)
                .ConfigureAwait(false);

        if (anyFailed && !anySynthesized)
            return await StageRunHelper
                .FailAsync(stageRunStore, stageRun, reporter,
                    $"All {segments.Count(s => s.Status == LipSynthesisSegmentStatus.Failed)} synthesized turns failed.",
                    CancellationToken.None, runtimePlanningPreferences, logger)
                .ConfigureAwait(false);

        int synthesized = segments.Count(s => s.Status == LipSynthesisSegmentStatus.Synthesized);
        int failed = segments.Count(s => s.Status == LipSynthesisSegmentStatus.Failed);
        int skipped = segments.Count - synthesized - failed;
        return await StageRunHelper
            .PartiallyCompleteAsync(stageRunStore, stageRun, reporter,
                $"{synthesized} synthesized, {skipped} skipped, {failed} failed.", CancellationToken.None,
                runtimePlanningPreferences, logger)
            .ConfigureAwait(false);
    }

    private async Task<StageRunRecord> SkipStageAsync(
        StageRunRecord stageRun, string skipReason, string degradationCode, LipSynthesisStageRequest request)
    {
        StageRunRecord skipped = await StageRunHelper
            .SkipAsync(stageRunStore, stageRun, lipSynthesisEngine as IStageRuntimeExecutionReporter,
                skipReason, CancellationToken.None, runtimePlanningPreferences, logger)
            .ConfigureAwait(false);

        await WriteDegradationAsync(request, skipped.Id, degradationCode, skipReason, null, null).ConfigureAwait(false);
        return skipped;
    }

    private async Task WriteDegradationAsync(
        LipSynthesisStageRequest request, Guid stageRunId, string code, string message, string? detail, string? recommendedAction)
    {
        if (degradationWriter is null)
            return;

        await degradationWriter.WriteAsync(
            new PipelineDegradationRecord(
                Stage: StageNames.LipSynthesis,
                Code: code,
                Message: message,
                Detail: detail,
                SelectedFallback: "original-video",
                RecommendedAction: recommendedAction,
                OccurredAtUtc: DateTimeOffset.UtcNow,
                StageRunId: stageRunId),
            request.ProjectId, request.MediaAsset.Id, CancellationToken.None)
            .ConfigureAwait(false);
    }

    private static IReadOnlyList<LipSynthesisSegment> BuildUniformSkip(
        LipSynthesisStageRequest request, LipSynthesisSegmentStatus status, string skipReason) =>
        request.SpeakerTurns
            .Select(turn => new LipSynthesisSegment(
                turn.SegmentId, status, turn.SpeakerId, turn.Start, turn.End, null, null,
                skipReason, null, null, null, false, DateTimeOffset.UtcNow))
            .ToList();

    private LipSynthesisSegment Skip(
        LipSynthesisTurn turn, LipSynthesisSegmentStatus status, string skipReason, double? faceConfidence) =>
        new(turn.SegmentId, status, turn.SpeakerId, turn.Start, turn.End, faceConfidence, null,
            skipReason, null, lipSynthesisEngine.ProviderId, lipSynthesisEngine.ModelId,
            lipSynthesisEngine.IsExperimental, DateTimeOffset.UtcNow);
}
