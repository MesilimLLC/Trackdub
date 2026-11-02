using System.Globalization;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.Extensions.DependencyInjection;
using Trackdub.Application.LipSync;
using Trackdub.Application.LipSynthesis;
using Trackdub.Application.Pipeline;
using Trackdub.Application.Transcripts;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Contracts;
using Trackdub.Contracts.Dubbing;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Contracts.Licensing;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.Transcripts;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Pipeline;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Tts;

namespace Trackdub.Application.Dubbing;

public sealed partial class DubbingPipelineEngine
{
    /// <summary>Session-scoped inputs shared by every stage workflow runner.</summary>
    internal sealed record StageExecutionContext
    {
        public required IDubbingSession Session { get; init; }
        public required DubbingSessionOptions Options { get; init; }
        public required RuntimeModelSelections RuntimeSelections { get; init; }
        public IProgress<PipelineProgressEvent>? Progress { get; init; }
        public TtsStreamingPrefetch? TtsPrefetch { get; init; }
        public TranscriptWorkspace Workspace => Session.Workspace;
        public InferenceModelPreferences ModelPreferences =>
            RuntimeModelRequestFactory.CreateModelPreferences(RuntimeSelections);
    }

    /// <summary>
    /// Dispatches execution to the appropriate workspace workflow method for the given stage.
    /// Returns artifact paths and any degradation records produced by the stage.
    /// </summary>
    internal static async Task<StageWorkflowResult> RunStageWorkflowAsync(
        StageExecutionContext context,
        string stageName,
        CancellationToken cancellationToken)
    {
        // Stage attribution for cold-start evidence: everything this workflow runs —
        // including plan-time smoke inference (which scopes itself to the smoke phase) —
        // is attributed to this stage until the workflow returns.
        ColdStartCapture.MarkFirstStageStart(stageName);
        using IDisposable stagePhase = ColdStartCapture.EnterPhase(ColdStartPhase.Stage, stageName);

        if (StageWorkflowHandlers.TryGetValue(stageName, out var workflow))
        {
            return await workflow(context, cancellationToken).ConfigureAwait(false);
        }

        // A catalog stage with no registered workflow is a defect, not a no-op;
        // report failure rather than silently succeeding.
        return new StageWorkflowResult(
            [],
            [$"Stage '{stageName}' has no registered workflow."],
            StageStatus.Failed,
            "STAGE_UNSUPPORTED");
    }

    private static readonly IReadOnlyDictionary<string, Func<StageExecutionContext, CancellationToken, Task<StageWorkflowResult>>>
        StageWorkflowHandlers =
            new Dictionary<string, Func<StageExecutionContext, CancellationToken, Task<StageWorkflowResult>>>(StringComparer.OrdinalIgnoreCase)
            {
                [StageNames.Separation] = RunSeparationStageAsync,
                [StageNames.Vad] = RunVadStageAsync,
                [StageNames.Asr] = RunAsrStageAsync,
                [StageNames.Diarization] = RunDiarizationStageAsync,
                [StageNames.AudioPreparation] = RunAudioPreparationStageAsync,
                [StageNames.TextRefinementAsr] = RunTextRefinementStageAsync,
                [StageNames.OverlapRescue] = RunOverlapRescueStageAsync,
                [StageNames.Translation] = RunTranslationStageAsync,
                [StageNames.Tts] = RunTtsStageAsync,
                [StageNames.Export] = RunExportStageAsync,
                [StageNames.LipSync] = RunLipSyncStageAsync,
                [StageNames.LipSynthesis] = RunLipSynthesisStageAsync,
            };

    /// <summary>
    /// Runs stem separation for voice-clone references and the mix bed.
    /// </summary>
    private static async Task<StageWorkflowResult> RunSeparationStageAsync(
        StageExecutionContext context,
        CancellationToken cancellationToken)
    {
        if (!context.Options.EnableStemSeparation)
        {
            return new StageWorkflowResult(
                [],
                ["Stem separation is disabled for this run."],
                StageStatus.Skipped,
                StageSkipReasonCodes.DisabledByOption);
        }

        DateTimeOffset stageWorkStartedUtc = DateTimeOffset.UtcNow;
        TranscriptProjectState separationState = await context.Workspace.RunStemSeparationAsync(
            cancellationToken,
            preferredModelAlias: context.RuntimeSelections.SeparationModelAlias,
            modelPreferences: context.ModelPreferences).ConfigureAwait(false);

        return BuildStageWorkflowResultFromStageRun(
            separationState,
            StageNames.Separation,
            stageWorkStartedUtc);
    }

    private static async Task<StageWorkflowResult> RunVadStageAsync(
        StageExecutionContext context,
        CancellationToken cancellationToken)
    {
        DateTimeOffset stageWorkStartedUtc = DateTimeOffset.UtcNow;
        TranscriptProjectState vadState = await context.Workspace.RunTranscriptStageAsync(
            StageNames.Vad,
            TranscriptRunOptions.ForDubbingStage(
                context.Options, StageNames.Vad, context.ModelPreferences, context.Progress),
            cancellationToken).ConfigureAwait(false);

        // When enabled, speech enhancement runs while routing audio for the first transcript stage;
        // surface its failure before the VAD Completed event so it shows in progress and outcome.
        IReadOnlyList<string>? enhancementDegradations =
            ExtractSpeechEnhancementDegradations(vadState, stageWorkStartedUtc);
        if (enhancementDegradations is { Count: > 0 })
        {
            ReportProgress(context.Progress,
                new StageProgressEvent(StageNames.SpeechEnhancement, PipelineProgressEventKind.Failed,
                    enhancementDegradations[0]));
        }

        return BuildStageWorkflowResultFromStageRun(vadState, StageNames.Vad, stageWorkStartedUtc, enhancementDegradations);
    }

    private static async Task<StageWorkflowResult> RunAsrStageAsync(
        StageExecutionContext context,
        CancellationToken cancellationToken)
    {
        TranscriptProjectState asrOpen = await context.Workspace.Project
            .OpenAsync(cancellationToken).ConfigureAwait(false);
        if (asrOpen.CurrentTranscriptRevision is not null &&
            asrOpen.TranscriptSegments.Count > 0)
        {
            // Interactive re-run: re-transcribe the existing segments in place
            // instead of regenerating a fresh revision from VAD regions.
            IReadOnlyList<Guid> segmentIds = asrOpen.TranscriptSegments
                .Select(static segment => segment.Id)
                .ToArray();
            DateTimeOffset retranscribeStartedUtc = DateTimeOffset.UtcNow;
            TranscriptProjectState retranscribed = await context.Workspace.RetranscribeSegmentsAsync(
                RuntimeModelSetupCoordinator.CreateRetranscribeRequest(
                    context.RuntimeSelections,
                    asrOpen.CurrentTranscriptRevision.Id,
                    segmentIds),
                cancellationToken).ConfigureAwait(false);
            return BuildStageWorkflowResultFromStageRun(retranscribed, StageNames.Asr, retranscribeStartedUtc);
        }

        DateTimeOffset stageWorkStartedUtc = DateTimeOffset.UtcNow;
        TranscriptProjectState asrState = await context.Workspace.RunTranscriptStageAsync(
            StageNames.Asr,
            TranscriptRunOptions.ForDubbingStage(
                context.Options, StageNames.Asr, context.ModelPreferences, context.Progress),
            cancellationToken).ConfigureAwait(false);
        return BuildStageWorkflowResultFromStageRun(asrState, StageNames.Asr, stageWorkStartedUtc);
    }

    private static async Task<StageWorkflowResult> RunDiarizationStageAsync(
        StageExecutionContext context,
        CancellationToken cancellationToken)
    {
        TranscriptProjectState diarizationOpen = await context.Workspace.Project
            .OpenAsync(cancellationToken).ConfigureAwait(false);
        if (diarizationOpen.CurrentTranscriptRevision is not null &&
            diarizationOpen.TranscriptSegments.Count > 0)
        {
            // Interactive re-run: re-diarize and re-assign speaker ids on the
            // existing transcript segments, matching the desktop host's
            // "Identify speakers" semantics.
            DateTimeOffset rediarizeStartedUtc = DateTimeOffset.UtcNow;
            TranscriptProjectState rediarized = await context.Workspace.RerunDiarizationAsync(
                RuntimeModelSetupCoordinator.CreateRerunDiarizationRequest(context.RuntimeSelections),
                cancellationToken).ConfigureAwait(false);
            return BuildStageWorkflowResultFromStageRun(rediarized, StageNames.Diarization, rediarizeStartedUtc);
        }

        if (!context.Options.EnableSpeakerDiarization)
        {
            // The single-stage diarization path records no run when the option is off
            // (the generation stage only builds a default region plan, which is then
            // discarded). Report the skip explicitly instead of running work whose
            // outcome cannot be observed.
            return new StageWorkflowResult(
                [],
                ["Speaker diarization is disabled for this run."],
                StageStatus.Skipped,
                StageSkipReasonCodes.DisabledByOption);
        }

        DateTimeOffset stageWorkStartedUtc = DateTimeOffset.UtcNow;
        TranscriptProjectState diarizationState = await context.Workspace.RunTranscriptStageAsync(
            StageNames.Diarization,
            TranscriptRunOptions.ForDubbingStage(
                context.Options, StageNames.Diarization, context.ModelPreferences, context.Progress),
            cancellationToken).ConfigureAwait(false);
        return BuildStageWorkflowResultFromStageRun(diarizationState, StageNames.Diarization, stageWorkStartedUtc);
    }

    private static async Task<StageWorkflowResult> RunAudioPreparationStageAsync(
        StageExecutionContext context,
        CancellationToken cancellationToken)
    {
        DateTimeOffset stageWorkStartedUtc = DateTimeOffset.UtcNow;
        TranscriptProjectState audioPrepState = await context.Workspace
            .RunSpeechAudioPreparationAsync(cancellationToken, context.Progress)
            .ConfigureAwait(false);
        // A failed speech-enhancement pass inside preparation degrades the cleanup
        // stage even when preparation itself completes on the unenhanced audio.
        return BuildStageWorkflowResultFromStageRun(
            audioPrepState,
            StageNames.AudioPreparation,
            stageWorkStartedUtc,
            ExtractSpeechEnhancementDegradations(audioPrepState, stageWorkStartedUtc));
    }

    private static async Task<StageWorkflowResult> RunTextRefinementStageAsync(
        StageExecutionContext context,
        CancellationToken cancellationToken)
    {
        DateTimeOffset stageWorkStartedUtc = DateTimeOffset.UtcNow;
        TranscriptProjectState refinementState = await context.Workspace.RunTranscriptStageAsync(
            StageNames.TextRefinementAsr,
            TranscriptRunOptions.ForDubbingStage(
                context.Options, StageNames.TextRefinementAsr, context.ModelPreferences, context.Progress),
            cancellationToken).ConfigureAwait(false);
        return BuildStageWorkflowResultFromStageRun(refinementState, StageNames.TextRefinementAsr, stageWorkStartedUtc);
    }

    private static async Task<StageWorkflowResult> RunOverlapRescueStageAsync(
        StageExecutionContext context,
        CancellationToken cancellationToken)
    {
        TranscriptProjectState preRescueState = await context.Workspace.Project
            .OpenAsync(cancellationToken).ConfigureAwait(false);
        StageRunRecord? diarizationRun = GetLatestStageRun(preRescueState, StageNames.Diarization);
        if (diarizationRun is null ||
            diarizationRun.Status is not (StageRunStatus.Completed
                or StageRunStatus.PartiallyCompleted
                or StageRunStatus.Skipped))
        {
            return new StageWorkflowResult(
                [],
                ["Overlap rescue requires a completed diarization run first."],
                StageStatus.Skipped,
                StageSkipReasonCodes.PrerequisiteFailed);
        }

        var rescueProgress = new Progress<OverlapRescueProgress>(update =>
            ReportProgress(
                context.Progress,
                new StageProgressEvent(
                    StageNames.OverlapRescue,
                    PipelineProgressEventKind.Progress,
                    update.IsPersistingArtifacts
                        ? "Overlap rescue: saving source candidates."
                        : string.Format(
                            CultureInfo.InvariantCulture,
                            "Overlap rescue region {0}/{1}",
                            update.CompletedRegions,
                            update.TotalRegions))));

        DateTimeOffset rescueStartedUtc = DateTimeOffset.UtcNow;
        TranscriptProjectState rescueState = await context.Workspace.RunOverlapRescueAsync(
            cancellationToken,
            rescueProgress,
            preferredModelAlias: context.RuntimeSelections.OverlapRescueModelAlias,
            modelPreferences: context.ModelPreferences,
            retranscribeCandidates: context.Options.RetranscribeOverlapCandidates).ConfigureAwait(false);
        return BuildStageWorkflowResultFromStageRun(rescueState, StageNames.OverlapRescue, rescueStartedUtc);
    }

    private static async Task<StageWorkflowResult> RunTranslationStageAsync(
        StageExecutionContext context,
        CancellationToken cancellationToken)
    {
        TranscriptProjectState translationState = await context.Workspace.Project.OpenAsync(cancellationToken).ConfigureAwait(false);
        if (translationState.TranscriptSegments.Count == 0)
        {
            return BuildNoTranscriptSegmentsSkip(StageNames.Translation);
        }

        string sourceLanguage = context.Options.SourceLanguageCode ?? "auto";
        DateTimeOffset stageWorkStartedUtc = DateTimeOffset.UtcNow;
        RuntimeExecutionProviderSelection translationExecutionProvider =
            RuntimeModelSetupCoordinator.CreateExecutionProviderSelection(
                context.RuntimeSelections,
                RuntimeStage.Translation);
        TranscriptProjectState translatedState = await context.Workspace.GenerateTranslationAsync(
            new GenerateTranslationRequest(
                SourceLanguage: sourceLanguage,
                TargetLanguage: context.Options.TargetLanguageCode,
                PreferredModelAlias: context.RuntimeSelections.GetSelectionIntent(RuntimeStage.Translation) == RuntimeModelSelectionIntent.Automatic
                    ? null
                    : context.RuntimeSelections.TranslationModelAlias,
                PreferredExecutionProvider: translationExecutionProvider.PreferredExecutionProvider,
                RequirePreferredExecutionProvider: translationExecutionProvider.RequirePreferredExecutionProvider,
                PreferredModelVariantAlias: RuntimeModelSetupCoordinator.ResolvePreferredModelVariantAlias(
                    context.RuntimeSelections,
                    RuntimeStage.Translation),
                EnableSegmentStreaming: context.Options.EnableTranslationSegmentStreaming, Scope: context.Options.TranslationScope),
            cancellationToken,
            context.Progress,
            context.TtsPrefetch).ConfigureAwait(false);
        return BuildStageWorkflowResultFromStageRun(translatedState, StageNames.Translation, stageWorkStartedUtc);
    }

    private static async Task<StageWorkflowResult> RunTtsStageAsync(
        StageExecutionContext context,
        CancellationToken cancellationToken)
    {
        TranscriptWorkspace workspace = context.Workspace;
        TranscriptProjectState ttsState = await workspace.Project.OpenAsync(cancellationToken).ConfigureAwait(false);
        if (ttsState.TranscriptSegments.Count == 0)
        {
            return BuildNoTranscriptSegmentsSkip(StageNames.Tts);
        }

        if (UnattendedVoiceSelection.RequestsVoiceCloning(context.Options))
        {
            TryResolveService<IConsentService>(context.Session)?.GrantVoiceCloningConsent();
            ReportProgress(
                context.Progress,
                new StageProgressEvent(
                    StageNames.Tts,
                    PipelineProgressEventKind.Progress,
                    context.Options.VoiceAssignmentOverrides is { Count: > 0 }
                        ? "Cloning speakers without a voice override from source audio."
                        : "Cloning each speaker from source audio."));
        }

        GenerateTtsForAllSpeakersRequest ttsRequest = BuildPipelineTtsRequest(
            ttsState, context.Options, context.RuntimeSelections);
        if (ttsRequest.VoiceIdsBySpeakerId is { Count: > 0 })
        {
            ReportProgress(
                context.Progress,
                new StageProgressEvent(
                    StageNames.Tts,
                    PipelineProgressEventKind.Progress,
                    $"Applying voice override to {ttsRequest.VoiceIdsBySpeakerId.Count} speaker(s)."));
        }

        if (ttsRequest.FallbackVoiceIdsBySpeakerId is { Count: > 0 })
        {
            ReportProgress(
                context.Progress,
                new StageProgressEvent(
                    StageNames.Tts,
                    PipelineProgressEventKind.Progress,
                    $"Auto-assigning a fallback voice to {ttsRequest.FallbackVoiceIdsBySpeakerId.Count} speaker(s) without a voice assignment (unattended run)."));
        }

        DateTimeOffset stageWorkStartedUtc = DateTimeOffset.UtcNow;
        TranscriptProjectState ttsResult = await workspace.GenerateTtsForAllSpeakersAsync(
            ttsRequest,
            cancellationToken,
            context.Progress,
            context.TtsPrefetch).ConfigureAwait(false);
        return BuildStageWorkflowResultFromStageRun(ttsResult, StageNames.Tts, stageWorkStartedUtc);
    }

    private static GenerateTtsForAllSpeakersRequest BuildPipelineTtsRequest(
        TranscriptProjectState state,
        DubbingSessionOptions options,
        RuntimeModelSelections runtimeSelections)
    {
        RuntimeExecutionProviderSelection ttsExecutionProvider =
            RuntimeModelSetupCoordinator.CreateExecutionProviderSelection(
                runtimeSelections,
                RuntimeStage.Tts);
        return UnattendedVoiceSelection.BuildUnattendedTtsRequest(
            state,
            options,
            runtimeSelections.TtsModelAlias) with
        {
            PreferredExecutionProvider = ttsExecutionProvider.PreferredExecutionProvider,
            RequirePreferredExecutionProvider = ttsExecutionProvider.RequirePreferredExecutionProvider,
            PreferredModelVariantAlias = RuntimeModelSetupCoordinator.ResolvePreferredModelVariantAlias(
                runtimeSelections,
                RuntimeStage.Tts),
        };
    }

    private static async Task<StageWorkflowResult> RunExportStageAsync(
        StageExecutionContext context,
        CancellationToken cancellationToken)
    {
        TranscriptWorkspace workspace = context.Workspace;
        TranscriptProjectState state = await workspace.Project.OpenAsync(cancellationToken).ConfigureAwait(false);
        bool hasTranscriptSegments = state.TranscriptSegments.Count > 0;
        if (!hasTranscriptSegments)
        {
            ReportProgress(
                context.Progress,
                new StageProgressEvent(
                    StageNames.Export,
                    PipelineProgressEventKind.Progress,
                    "No transcript segments were detected; exporting source media without subtitles or dubbed speech."));
        }

        ExportStageResult exportResult = await workspace.ExportAsync(
            state,
            BuildExportStageRequest(context, state, hasTranscriptSegments),
            cancellationToken).ConfigureAwait(false);
        if (exportResult.IsBlocked)
        {
            throw new InvalidOperationException(exportResult.BlockedReason ?? "Export blocked by tier gate.");
        }

        (StageStatus exportStatus, string? exportReasonCode, IReadOnlyList<string>? exportDegradations) =
            MapStageRunToSdkOutcome(exportResult.StageRun);
        IReadOnlyList<string>? degradations = exportResult.Warnings is { Count: > 0 } warnings
            ? (exportDegradations is { Count: > 0 } existing ? warnings.Concat(existing).ToArray() : warnings)
            : exportDegradations;

        return new StageWorkflowResult(
            [exportResult.OutputPath, exportResult.ExportVideoRelativePath],
            degradations,
            exportStatus,
            exportReasonCode);
    }

    private static ExportStageRequest BuildExportStageRequest(
        StageExecutionContext context,
        TranscriptProjectState state,
        bool hasTranscriptSegments)
    {
        DubbingSessionOptions options = context.Options;
        ExportOutputContainer container = ResolveExportContainer(options.ExportFormat);
        string outputPath = !string.IsNullOrWhiteSpace(options.ExportOutputPath)
            ? options.ExportOutputPath
            : ResolveExportOutputPath(context.Session.ProjectRootPath, container);

        return new ExportStageRequest(
            ProjectId: state.ProjectState.Project.Id,
            OutputPath: outputPath,
            SubtitleFormats: ResolveSubtitleFormats(options.SubtitleFormats, hasTranscriptSegments),
            SubtitleSource: ResolveSubtitleSource(options.SubtitleSource),
            BurnInSubtitles: options.BurnInSubtitles,
            TargetLufs: options.ExportTargetLufs ?? ExportLoudnessTargets.OnlineLufs,
            Container: container,
            SourceGainDb: options.ExportSourceGainDb ?? 0d,
            DubbedSpeechGainDb: options.ExportDubbedSpeechGainDb ?? 0d,
            DuckingGainDb: options.ExportDuckingGainDb,
            ApplyTimbrePolish: options.ApplyTimbrePolish,
            RestoreOriginalPan: options.RestoreOriginalPan,
            MatchOriginalLoudness: options.MatchOriginalLoudness,
            VideoEncoder: options.VideoEncoder,
            // Carry the raw requested formats (including the null/empty distinction)
            // so the export-resume gate persists and compares the same token the
            // snapshot records, instead of the transcript-state-resolved formats.

            RawSubtitleFormats: options.SubtitleFormats);
    }

    private static async Task<StageWorkflowResult> RunLipSyncStageAsync(
        StageExecutionContext context,
        CancellationToken cancellationToken)
    {
        if (context.Workspace.LipSync is null)
        {
            throw new InvalidOperationException(
                "Lip-sync workflow is not available in this session. Ensure CompositionRoot registered LipSyncWorkflow.");
        }

        DateTimeOffset lipSyncStartedUtc = DateTimeOffset.UtcNow;
        TranscriptProjectState lipSyncState = await context.Workspace.RunLipSyncAsync(
            new LipSyncAlignAllRequest(
                PreferredModelAlias: context.RuntimeSelections.LipSyncModelAlias),
            cancellationToken).ConfigureAwait(false);
        return BuildStageWorkflowResultFromStageRun(lipSyncState, StageNames.LipSync, lipSyncStartedUtc);
    }

    private static async Task<StageWorkflowResult> RunLipSynthesisStageAsync(
        StageExecutionContext context,
        CancellationToken cancellationToken)
    {
        TranscriptWorkspace workspace = context.Workspace;
        if (workspace.LipSynthesis is null)
        {
            throw new InvalidOperationException(
                "Lip-synthesis workflow is not available in this session. Ensure CompositionRoot registered LipSynthesisWorkflow.");
        }

        (bool isLicenseApproved, bool allowExperimentalExecution) =
            await ResolveLipSynthesisExecutionGatesAsync(
                context.Session,
                context.RuntimeSelections.LipSynthesisModelAlias,
                cancellationToken).ConfigureAwait(false);

        DateTimeOffset lipSynthesisStartedUtc = DateTimeOffset.UtcNow;
        TranscriptProjectState lipSynthesisState = await workspace.RunLipSynthesisAsync(
            new LipSynthesisRunRequest(
                IsLicenseApproved: isLicenseApproved,
                AllowExperimentalExecution: allowExperimentalExecution,
                PreferredModelAlias: context.RuntimeSelections.LipSynthesisModelAlias),
            cancellationToken).ConfigureAwait(false);
        return BuildStageWorkflowResultFromStageRun(lipSynthesisState, StageNames.LipSynthesis, lipSynthesisStartedUtc);
    }

    private static StageWorkflowResult BuildStageWorkflowResultFromStageRun(
        TranscriptProjectState state,
        string stageName,
        DateTimeOffset notBeforeUtc,
        IReadOnlyList<string>? additionalDegradations = null)
    {
        (StageStatus status, string? reasonCode, IReadOnlyList<string>? degradations) =
            MapStageRunToSdkOutcome(GetLatestStageRun(state, stageName, notBeforeUtc));

        if (additionalDegradations is { Count: > 0 })
        {
            degradations = degradations is { Count: > 0 }
                ? additionalDegradations.Concat(degradations).ToArray()
                : additionalDegradations;
            // Degraded output from a nominally successful run is a partial success,
            // not a clean one.
            if (status == StageStatus.Succeeded)
            {
                status = StageStatus.PartiallySucceeded;
            }
        }

        return new StageWorkflowResult([], degradations, status, reasonCode);
    }

    private static StageWorkflowResult BuildNoTranscriptSegmentsSkip(string stageName) =>
        new(
            [],
            ["No transcript segments were detected; this stage has no work to perform."],
            StageStatus.Skipped,
            StageSkipReasonCodes.NoTranscriptSegments);

    /// <summary>
    /// Returns the most recent run for a stage. When <paramref name="notBeforeUtc"/> is
    /// supplied, only runs started at or after that instant qualify — used to bind a
    /// stage outcome to the run produced by the current invocation instead of
    /// reporting a stale row left by an earlier run.
    /// </summary>
    internal static StageRunRecord? GetLatestStageRun(
        TranscriptProjectState state,
        string stageName,
        DateTimeOffset? notBeforeUtc = null) =>
        state.StageRuns
            .Where(r => string.Equals(r.StageName, stageName, StringComparison.OrdinalIgnoreCase)
                && (notBeforeUtc is null || r.StartedAtUtc >= notBeforeUtc))
            .OrderByDescending(static r => r.StartedAtUtc)
            .FirstOrDefault();

    internal static (StageStatus Status, string? ReasonCode, IReadOnlyList<string>? Degradations)
        MapStageRunToSdkOutcome(StageRunRecord? stageRun)
    {
        if (stageRun is null)
        {
            return (StageStatus.Failed, "STAGE_RUN_MISSING", ["Stage completed without recording a stage run."]);
        }

        return stageRun.Status switch
        {
            StageRunStatus.Completed =>
                (StageStatus.Succeeded, null, null),
            StageRunStatus.Skipped =>
                (StageStatus.Skipped,
                    stageRun.FailureReason ?? "STAGE_SKIPPED",
                    null),
            StageRunStatus.PartiallyCompleted =>
                (StageStatus.PartiallySucceeded,
                    null,
                    stageRun.FailureReason is not null ? [stageRun.FailureReason] : null),
            StageRunStatus.Running =>
                (StageStatus.Failed,
                    "STAGE_INCOMPLETE",
                    ["Stage run did not reach a terminal state."]),
            _ =>
                (StageStatus.Failed,
                    stageRun.FailureReason ?? ResolveFailedStageFallbackReasonCode(stageRun.Status),
                    null),
        };
    }

    private static string ResolveFailedStageFallbackReasonCode(StageRunStatus status) =>
        status == StageRunStatus.Canceled ? "CANCELLED" : "STAGE_FAILED";

    // Speech-enhancement runs as an internal sub-step of RunStemSeparationAsync and is caught
    // there (fallback to unenhanced audio). Surface the failure so callers can include it in
    // StageOutcome.DegradationRecords and progress events rather than silently discarding it.
    internal static IReadOnlyList<string>? ExtractSpeechEnhancementDegradations(
        TranscriptProjectState state,
        DateTimeOffset? notBeforeUtc = null)
    {
        // Order first so we evaluate the *most recent* speech-enhancement attempt.
        // Filtering to Failed before ordering would surface stale failures even when
        // a later run succeeded, producing a false degraded outcome after a successful rerun.
        StageRunRecord? latestRun = state.StageRuns
            .Where(r =>
                string.Equals(r.StageName, StageNames.SpeechEnhancement, StringComparison.OrdinalIgnoreCase)
                && (notBeforeUtc is null || r.StartedAtUtc >= notBeforeUtc))
            .OrderByDescending(static r => r.StartedAtUtc)
            .FirstOrDefault();

        if (latestRun is null || latestRun.Status != StageRunStatus.Failed)
        {
            return null;
        }

        string message = string.IsNullOrWhiteSpace(latestRun.FailureReason)
            ? "speech-enhancement failed; falling back to unenhanced audio"
            : $"speech-enhancement failed (falling back to unenhanced audio): {latestRun.FailureReason}";

        return [message];
    }
}
