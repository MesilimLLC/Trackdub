using Trackdub.Application.LipSync;
using Trackdub.Application.LipSynthesis;
using Trackdub.Contracts.Projects;
using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Transcripts;

public sealed partial class TranscriptWorkspace(
    ProjectWorkflow project,
    RuntimeModelWorkflow runtimeModels,
    DiarizationModelWorkflow diarizationModels,
    TranscriptWorkflow transcript,
    TranslationWorkflow translation,
    SpeakerWorkflow speakers,
    VoiceWorkflow voices,
    TtsWorkflow tts,
    TtsDubPreviewWorkflow dubPreview,
    PreviewMixWorkflow preview,
    ExportWorkflow export,
    EditingHistoryWorkflow editingHistory,
    TranscriptImportModelProvisioner? importModelProvisioner = null,
    IApplicationLogger? logger = null,
    LipSyncWorkflow? lipSync = null,
    LipSynthesisWorkflow? lipSynthesis = null)
    : IDisposable
{
    private readonly object disposalSync = new();
    private readonly SemaphoreSlim _pipelineGuard = new(1, 1);
    private readonly CancellationTokenSource workspaceCancellation = new();
    private bool disposed;

    public TranscriptWorkspace(
        ProjectWorkflow project,
        DiarizationModelWorkflow diarizationModels,
        TranscriptWorkflow transcript,
        TranslationWorkflow translation,
        SpeakerWorkflow speakers,
        VoiceWorkflow voices,
        TtsWorkflow tts,
        TtsDubPreviewWorkflow dubPreview,
        PreviewMixWorkflow preview,
        ExportWorkflow export,
        EditingHistoryWorkflow editingHistory,
        TranscriptImportModelProvisioner? importModelProvisioner = null,
        IApplicationLogger? logger = null,
        LipSyncWorkflow? lipSync = null,
        LipSynthesisWorkflow? lipSynthesis = null)
        : this(
            project,
            new RuntimeModelWorkflow(),
            diarizationModels,
            transcript,
            translation,
            speakers,
            voices,
            tts,
            dubPreview,
            preview,
            export,
            editingHistory,
            importModelProvisioner,
            logger,
            lipSync,
            lipSynthesis)
    {
    }

    private readonly TranscriptImportModelProvisioner? importModelProvisioner = importModelProvisioner;

    public ProjectWorkflow Project { get; } = project ?? throw new ArgumentNullException(nameof(project));

    public RuntimeModelWorkflow RuntimeModels { get; } = runtimeModels ?? throw new ArgumentNullException(nameof(runtimeModels));

    public DiarizationModelWorkflow DiarizationModels { get; } = diarizationModels ?? throw new ArgumentNullException(nameof(diarizationModels));

    public TranscriptWorkflow Transcript { get; } = transcript ?? throw new ArgumentNullException(nameof(transcript));

    public TranslationWorkflow Translation { get; } = translation ?? throw new ArgumentNullException(nameof(translation));

    public SpeakerWorkflow Speakers { get; } = speakers ?? throw new ArgumentNullException(nameof(speakers));

    public VoiceWorkflow Voices { get; } = voices ?? throw new ArgumentNullException(nameof(voices));

    public TtsWorkflow Tts { get; } = tts ?? throw new ArgumentNullException(nameof(tts));

    public TtsDubPreviewWorkflow DubPreview { get; } = dubPreview ?? throw new ArgumentNullException(nameof(dubPreview));

    public PreviewMixWorkflow Preview { get; } = preview ?? throw new ArgumentNullException(nameof(preview));

    public ExportWorkflow Export { get; } = export ?? throw new ArgumentNullException(nameof(export));

    public EditingHistoryWorkflow EditingHistory { get; } = editingHistory ?? throw new ArgumentNullException(nameof(editingHistory));

    /// <summary>
    /// Lip-sync alignment workflow. Null when the composition did not register <see cref="LipSyncWorkflow"/>.
    /// </summary>
    public LipSyncWorkflow? LipSync { get; } = lipSync;

    /// <summary>
    /// M23 video lip-synthesis workflow. Null when the composition did not register <see cref="LipSynthesisWorkflow"/>.
    /// </summary>
    public LipSynthesisWorkflow? LipSynthesis { get; } = lipSynthesis;

    public void Dispose()
    {
        lock (disposalSync)
        {
            if (disposed)
            {
                return;
            }

            disposed = true;
            workspaceCancellation.Cancel();
        }

        _pipelineGuard.Dispose();
        workspaceCancellation.Dispose();
        Tts.Dispose();
        LipSynthesis?.Dispose();
    }

    public Task<TranscriptProjectState> CreateProjectAsync(
        CreateTranscriptProjectRequest request,
        CancellationToken cancellationToken) =>
        RunPipelineAsync(
            nameof(CreateProjectAsync),
            async ct =>
            {
                await EnsureImportModelsForTranscriptionAsync(
                        request.ModelPreferences,
                        ct,
                        request.SourceLanguage)
                    .ConfigureAwait(false);
                return await Project.CreateAsync(request, ct).ConfigureAwait(false);
            },
            cancellationToken);

    public Task<TranscriptProjectState> CreateMediaSpineAsync(
        CreateTranscriptProjectRequest request,
        CancellationToken cancellationToken) =>
        RunPipelineAsync(
            nameof(CreateMediaSpineAsync),
            ct => Project.CreateMediaSpineAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> EnsureNormalizedProjectAudioAsync(
        CancellationToken cancellationToken,
        int? ffmpegThreadBudget = null) =>
        RunPipelineAsync(
            nameof(EnsureNormalizedProjectAudioAsync),
            ct => Project.EnsureNormalizedProjectAudioAsync(ct, ffmpegThreadBudget),
            cancellationToken);

    private Task EnsureImportModelsForTranscriptionAsync(
        InferenceModelPreferences? modelPreferences,
        CancellationToken cancellationToken,
        string? sourceLanguage = null)
    {
        if (importModelProvisioner is null)
        {
            return Task.CompletedTask;
        }

        return importModelProvisioner.EnsureImportModelsAsync(
            this,
            modelPreferences,
            cancellationToken,
            sourceLanguage: sourceLanguage);
    }

    private Task EnsureDiarizationModelsForTranscriptionAsync(
        InferenceModelPreferences? modelPreferences,
        CancellationToken cancellationToken)
    {
        if (importModelProvisioner is null)
        {
            return Task.CompletedTask;
        }

        return importModelProvisioner.EnsureDiarizationModelAsync(this, modelPreferences, cancellationToken);
    }

    public Task<TranscriptProjectState> SaveProjectUiSettingsAsync(
        ProjectUiSettings settings,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Project.SaveUiSettingsAsync(settings, ct),
            cancellationToken);

    public Task<TranscriptProjectState> RelocateSourceAsync(
        RelocateTranscriptSourceRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Project.RelocateSourceAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> RenameProjectAsync(
        RenameProjectRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Project.RenameProjectAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> RunStemSeparationAsync(
        CancellationToken cancellationToken,
        IProgress<StemSeparationProgress>? progress = null,
        string? preferredModelAlias = null,
        InferenceModelPreferences? modelPreferences = null) =>
        RunPipelineAsync(
            nameof(RunStemSeparationAsync),
            ct => Project.RunStemSeparationAsync(
                ct,
                progress,
                preferredModelAlias,
                modelPreferences),
            cancellationToken);

    public Task<TranscriptProjectState> RunOverlapRescueAsync(
        CancellationToken cancellationToken,
        IProgress<OverlapRescueProgress>? progress = null,
        string? preferredModelAlias = null,
        InferenceModelPreferences? modelPreferences = null,
        bool retranscribeCandidates = false) =>
        RunPipelineAsync(
            nameof(RunOverlapRescueAsync),
            ct => Project.RunOverlapRescueAsync(
                ct,
                progress,
                preferredModelAlias,
                modelPreferences,
                retranscribeCandidates),
            cancellationToken);

    public Task<TranscriptProjectState> RunSpeechAudioPreparationAsync(
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress = null) =>
        RunPipelineAsync(
            nameof(RunSpeechAudioPreparationAsync),
            ct => Project.RunSpeechAudioPreparationAsync(ct, progress),
            cancellationToken);

    internal async Task<T> RunPipelineAsync<T>(
        string operationName,
        Func<CancellationToken, Task<T>> operation,
        CancellationToken cancellationToken)
    {
        CancellationTokenSource linkedCancellation = CreateLinkedCancellation(cancellationToken);
        bool acquired = false;

        try
        {
            acquired = await _pipelineGuard
                .WaitAsync(TimeSpan.Zero, linkedCancellation.Token)
                .ConfigureAwait(false);

            if (!acquired)
            {
                logger?.LogWarning(
                    $"[pipeline_guard_busy] Operation rejected because another pipeline run is in progress: {operationName}");
                throw new InvalidOperationException(
                    $"A pipeline operation is already running for this project. Cannot start '{operationName}'.");
            }

            return await operation(linkedCancellation.Token).ConfigureAwait(false);
        }
        finally
        {
            linkedCancellation.Dispose();
            if (acquired)
            {
                try
                {
                    _pipelineGuard.Release();
                }
                catch (ObjectDisposedException) when (disposed)
                {
                    // Disposal cancels in-flight work before releasing resources. If the
                    // operation observes cancellation after disposal, there is no live guard
                    // left to release.
                }
            }
        }
    }

    internal Task RunPipelineAsync(
        string operationName,
        Func<CancellationToken, Task> operation,
        CancellationToken cancellationToken) =>
        RunPipelineAsync<object?>(
            operationName,
            async ct => { await operation(ct).ConfigureAwait(false); return null; },
            cancellationToken);

    /// <summary>
    /// Runs a lightweight, non-inference operation (metadata edits, renames, UI settings, speaker
    /// assignments) without acquiring <see cref="_pipelineGuard"/>. These operations do not invoke
    /// AI inference stages and must not be blocked while a long-running pipeline stage is in progress
    /// (e.g. renaming a project while stem separation runs should succeed immediately).
    /// The workspace cancellation token is still linked so the operation is aborted on disposal.
    /// </summary>
    internal async Task<T> RunLightWriteAsync<T>(
        Func<CancellationToken, Task<T>> operation,
        CancellationToken cancellationToken)
    {
        CancellationTokenSource linkedCancellation = CreateLinkedCancellation(cancellationToken);
        try
        {
            return await operation(linkedCancellation.Token).ConfigureAwait(false);
        }
        finally
        {
            linkedCancellation.Dispose();
        }
    }

    internal Task RunLightWriteAsync(
        Func<CancellationToken, Task> operation,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync<object?>(
            async ct => { await operation(ct).ConfigureAwait(false); return null; },
            cancellationToken);

    private CancellationTokenSource CreateLinkedCancellation(CancellationToken cancellationToken)
    {
        lock (disposalSync)
        {
            if (disposed)
            {
                throw new ObjectDisposedException(nameof(TranscriptWorkspace));
            }

            return CancellationTokenSource.CreateLinkedTokenSource(
                cancellationToken,
                workspaceCancellation.Token);
        }
    }

    // Translation.* guarded entry points

    public Task<TranscriptProjectState> GenerateTranslationAsync(
        GenerateTranslationRequest request,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress = null) =>
        GenerateTranslationAsync(request, cancellationToken, progress, segmentObserver: null);

    internal Task<TranscriptProjectState> GenerateTranslationAsync(
        GenerateTranslationRequest request,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress,
        ITranslatedSegmentObserver? segmentObserver) =>
        RunPipelineAsync(
            nameof(GenerateTranslationAsync),
            ct => Translation.GenerateTranslationAsync(request, ct, progress, segmentObserver),
            cancellationToken);

    public Task<TranscriptProjectState> RetranslateSegmentAsync(
        RetranslateSegmentRequest request,
        CancellationToken cancellationToken) =>
        RunPipelineAsync(
            nameof(RetranslateSegmentAsync),
            ct => Translation.RetranslateSegmentAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> SaveTranslationEditsAsync(
        SaveTranslationEditsRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Translation.SaveTranslationEditsAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> SetTranscriptLanguageAsync(
        SetTranscriptLanguageRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Translation.SetTranscriptLanguageAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> SelectTranslationTargetAsync(
        SetTranslationTargetRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Translation.SelectTranslationTargetAsync(request, ct),
            cancellationToken);

    // Tts.* guarded entry points (Requirement 5.3)

    public Task<TranscriptProjectState> GenerateTtsForSpeakerAsync(
        GenerateTtsForSpeakerRequest request,
        CancellationToken cancellationToken) =>
        RunPipelineAsync(
            nameof(GenerateTtsForSpeakerAsync),
            ct => Tts.GenerateTtsForSpeakerAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> GenerateTtsForSegmentAsync(
        GenerateTtsForSegmentRequest request,
        CancellationToken cancellationToken) =>
        RunPipelineAsync(
            nameof(GenerateTtsForSegmentAsync),
            ct => Tts.GenerateTtsForSegmentAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> GenerateTtsForAllSpeakersAsync(
        GenerateTtsForAllSpeakersRequest request,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress = null) =>
        GenerateTtsForAllSpeakersAsync(request, cancellationToken, progress, prefetch: null);

    internal Task<TranscriptProjectState> GenerateTtsForAllSpeakersAsync(
        GenerateTtsForAllSpeakersRequest request,
        CancellationToken cancellationToken,
        IProgress<PipelineProgressEvent>? progress,
        TtsStreamingPrefetch? prefetch) =>
        RunPipelineAsync(
            nameof(GenerateTtsForAllSpeakersAsync),
            ct => Tts.GenerateTtsForAllSpeakersAsync(request, ct, progress, prefetch),
            cancellationToken);

    /// <summary>
    /// Resolves every speaker's TTS plan and returns a prefetch that renders segments while the
    /// translation stream runs. The prefetch lives until <paramref name="runCancellationToken"/>
    /// is cancelled or it is disposed.
    /// </summary>
    internal Task<TtsStreamingPrefetch> CreateTtsStreamingPrefetchAsync(
        GenerateTtsForAllSpeakersRequest request,
        string targetLanguage,
        CancellationToken runCancellationToken) =>
        RunPipelineAsync(
            nameof(CreateTtsStreamingPrefetchAsync),
            ct => Tts.CreateStreamingPrefetchAsync(request, targetLanguage, logger, ct, runCancellationToken),
            runCancellationToken);

    public Task<TranscriptProjectState> RegenerateStaleTtsForSpeakerAsync(
        RegenerateStaleTtsForSpeakerRequest request,
        CancellationToken cancellationToken) =>
        RunPipelineAsync(
            nameof(RegenerateStaleTtsForSpeakerAsync),
            ct => Tts.RegenerateStaleTtsForSpeakerAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> StretchTtsTakeAsync(
        StretchTtsTakeRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Tts.StretchTtsTakeAsync(request, ct),
            cancellationToken);

    // LipSync.* guarded entry points

    /// <summary>
    /// Run the lip-sync forced-alignment pass for all TTS takes. User-triggered only.
    /// Returns the refreshed project state.
    /// </summary>
    public Task<TranscriptProjectState> RunLipSyncAsync(
        LipSyncAlignAllRequest request,
        CancellationToken cancellationToken)
    {
        if (LipSync is null)
            throw new InvalidOperationException("Lip-sync workflow is not available in this workspace.");

        LipSyncWorkflow lipSyncWorkflow = LipSync;
        return RunPipelineAsync(
            nameof(RunLipSyncAsync),
            ct => lipSyncWorkflow.AlignAllAsync(request, ct),
            cancellationToken);
    }

    public Task<TranscriptProjectState> RunLipSynthesisAsync(
        LipSynthesisRunRequest request,
        CancellationToken cancellationToken)
    {
        if (LipSynthesis is null)
            throw new InvalidOperationException("Lip-synthesis workflow is not available in this workspace.");

        LipSynthesisWorkflow lipSynthesisWorkflow = LipSynthesis;
        return RunPipelineAsync(
            nameof(RunLipSynthesisAsync),
            ct => lipSynthesisWorkflow.SynthesizeAllAsync(request, ct),
            cancellationToken);
    }

    // Preview.* guarded entry points (Requirement 5.5)

    public Task<PreviewMixStageResult> GeneratePreviewAsync(
        TranscriptProjectState state,
        PreviewMixStageRequest request,
        CancellationToken cancellationToken) =>
        RunPipelineAsync(
            nameof(GeneratePreviewAsync),
            ct => Preview.GeneratePreviewAsync(state, request, ct),
            cancellationToken);

    // Export.* guarded entry points (Requirement 5.6)

    public Task<ExportStageResult> ExportAsync(
        TranscriptProjectState state,
        ExportStageRequest request,
        CancellationToken cancellationToken) =>
        RunPipelineAsync(
            nameof(ExportAsync),
            ct => Export.ExportAsync(state, request, ct),
            cancellationToken);

    // Transcript.* guarded entry points (Requirements 5.1, 5.7)
    // Segment editing operations are light writes — they do not invoke inference and must not be
    // blocked by an in-progress pipeline run.

    public Task<TranscriptProjectState> SaveTranscriptEditsAsync(
        SaveTranscriptEditsRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Transcript.SaveEditsAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> SplitSegmentAsync(
        SplitTranscriptSegmentRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Transcript.SplitSegmentAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> MergeSegmentsAsync(
        MergeTranscriptSegmentsRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Transcript.MergeSegmentsAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> MergeSegmentRunAsync(
        MergeTranscriptSegmentRunRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Transcript.MergeSegmentRunAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> TrimSegmentAsync(
        TrimTranscriptSegmentRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Transcript.TrimSegmentAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> DeleteSegmentAsync(
        DeleteTranscriptSegmentRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Transcript.DeleteSegmentAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> RetranscribeSegmentsAsync(
        RetranscribeTranscriptSegmentsRequest request,
        CancellationToken cancellationToken) =>
        RunPipelineAsync(
            nameof(RetranscribeSegmentsAsync),
            ct => Transcript.RetranscribeSegmentsAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> RerunDiarizationAsync(
        RerunDiarizationRequest request,
        CancellationToken cancellationToken) =>
        RunPipelineAsync(
            nameof(RerunDiarizationAsync),
            ct => Speakers.RerunDiarizationAsync(request, ct),
            cancellationToken);

    // Speaker metadata edits are light writes — no inference, no pipeline gate needed.
    public Task<TranscriptProjectState> RenameSpeakerAsync(
        RenameSpeakerRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Speakers.RenameSpeakerAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> MergeSpeakersAsync(
        MergeSpeakersRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Speakers.MergeSpeakersAsync(request, ct),
            cancellationToken);

    public Task<IReadOnlyList<Domain.Speakers.MergedSpeakerEntry>> ListMergedSpeakersAsync(
        CancellationToken cancellationToken) =>
        Speakers.ListMergedSpeakersAsync(cancellationToken);

    public Task<TranscriptProjectState> DeleteEmptySpeakerAsync(
        DeleteEmptySpeakerRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Speakers.DeleteEmptySpeakerAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> AssignSpeakerToSegmentAsync(
        AssignSpeakerToSegmentRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Speakers.AssignSpeakerToSegmentAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> AssignSpeakerToSegmentsAsync(
        AssignSpeakerToSegmentsRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Speakers.AssignSpeakerToSegmentsAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> CreateSpeakerFromSegmentsAsync(
        CreateSpeakerFromSegmentsRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Speakers.CreateSpeakerFromSegmentsAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> SplitSpeakerTurnAsync(
        SplitSpeakerTurnRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Speakers.SplitSpeakerTurnAsync(request, ct),
            cancellationToken);

    // Reference clip extraction/import write audio files but do not invoke inference models.
    public Task<TranscriptProjectState> ExtractReferenceClipAsync(
        ExtractReferenceClipRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Speakers.ExtractReferenceClipAsync(request, ct),
            cancellationToken);

    public Task<TranscriptProjectState> ImportReferenceClipAsync(
        ImportReferenceClipRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Speakers.ImportReferenceClipAsync(request, ct),
            cancellationToken);

    // Voice assignment is a metadata write; TTS generation is triggered separately via pipeline.
    public Task<TranscriptProjectState> AssignVoiceToSpeakerAsync(
        AssignVoiceToSpeakerRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => Voices.AssignVoiceToSpeakerAsync(request, ct),
            cancellationToken);

    public Task<Domain.Tts.VoiceAssignment?> GetVoiceAssignmentHistoryAsync(
        Guid speakerId,
        CancellationToken cancellationToken) =>
        Voices.GetVoiceAssignmentHistoryAsync(speakerId, cancellationToken);

    // Restoring an editing snapshot is a write-only operation; no inference involved.
    public Task<TranscriptProjectState> RestoreEditingStateAsync(
        RestoreEditingStateRequest request,
        CancellationToken cancellationToken) =>
        RunLightWriteAsync(
            ct => EditingHistory.RestoreEditingStateAsync(request, ct),
            cancellationToken);
}
