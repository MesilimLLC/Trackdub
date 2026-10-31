using Trackdub.Domain;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.Transcript;
using Trackdub.Contracts;
using Trackdub.Contracts.Projects;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain.Translation;
using Trackdub.Domain.Tts;
using Trackdub.Domain.Mixing;

namespace Trackdub.Application.Transcripts;

public sealed record CreateTranscriptProjectRequest(
    string ProjectName,
    string SourceMediaPath,
    bool EnableSpeakerDiarization = true,
    InferenceModelPreferences? ModelPreferences = null,
    string? SourceLanguage = null);

public sealed record InferenceModelPreferences(
    string? VadModelAlias = null,
    string? AsrModelAlias = null,
    string? DiarizationModelAlias = null,
    string? SeparationModelAlias = null,
    string? OverlapRescueModelAlias = null,
    string? TranslationModelAlias = null,
    string? TtsModelAlias = null,
    string? TextRefinementModelAlias = null,
    string? LipSyncModelAlias = null,
    string? LipSynthesisModelAlias = null,
    bool RequireAsrModelAlias = false,
    bool RequireTextRefinementModelAlias = false,
    bool EnableAsrTextRefinement = false,
    IReadOnlyDictionary<RuntimeStage, ExecutionProviderKind>? PreferredExecutionProviders = null,
    IReadOnlySet<RuntimeStage>? RequiredExecutionProviderStages = null,
    IReadOnlyDictionary<RuntimeStage, string>? PreferredModelVariantAliases = null,
    ExecutionProviderKind? GlobalPreferredExecutionProvider = null,
    IReadOnlySet<RuntimeStage>? ExplicitExecutionProviderStages = null)
{
    public static InferenceModelPreferences Empty { get; } = new();

    public ExecutionProviderKind? GetPreferredExecutionProvider(RuntimeStage stage) =>
        PreferredExecutionProviders is not null &&
        PreferredExecutionProviders.TryGetValue(stage, out ExecutionProviderKind provider)
            ? provider
            : null;

    public bool RequiresPreferredExecutionProvider(RuntimeStage stage) =>
        RequiredExecutionProviderStages?.Contains(stage) == true;

    public string? GetPreferredModelVariantAlias(RuntimeStage stage) =>
        PreferredModelVariantAliases is not null &&
        PreferredModelVariantAliases.TryGetValue(stage, out string? alias) &&
        !string.IsNullOrWhiteSpace(alias)
            ? alias
            : null;
}

public sealed record RequiredDiarizationModelStatus(
    string ModelId,
    string ExpectedFileName,
    string ModelPath,
    string SourceUrl,
    bool IsAvailable,
    bool CanAutoDownload,
    bool RequiresOnnxExport,
    string HelpText);

public sealed record SaveTranscriptEditsRequest(
    Guid TranscriptRevisionId,
    IReadOnlyList<EditedTranscriptSegment> Segments);

public sealed record SetTranscriptLanguageRequest(
    string? TranscriptLanguage,
    string? SelectedTranslationTargetLanguage = null);

public sealed record GenerateTranslationRequest(
    string SourceLanguage,
    string TargetLanguage,
    string? PreferredModelAlias = null,
    ExecutionProviderKind? PreferredExecutionProvider = null,
    bool RequirePreferredExecutionProvider = false,
    string? PreferredModelVariantAlias = null,
    bool EnableSegmentStreaming = false,
    TranslationRunScope Scope = TranslationRunScope.All);

public sealed record RetranslateSegmentRequest(
    Guid TranslationRevisionId,
    Guid SegmentId,
    string SourceLanguage,
    string TargetLanguage,
    string? PreferredModelAlias = null,
    ExecutionProviderKind? PreferredExecutionProvider = null,
    bool RequirePreferredExecutionProvider = false,
    string? PreferredModelVariantAlias = null);

public sealed record SaveTranslationEditsRequest(
    Guid TranslationRevisionId,
    string TargetLanguage,
    IReadOnlyList<EditedTranslatedSegment> Segments);

public sealed record SetTranslationTargetRequest(
    string? TargetLanguage);

public sealed record EditedTranscriptSegment(
    Guid SegmentId,
    string Text,
    Guid? SpeakerId = null);

public sealed record EditedTranslatedSegment(
    int SegmentIndex,
    string Text);

public sealed record RelocateTranscriptSourceRequest(
    string NewSourceMediaPath,
    string? SelectedTranslationTargetLanguage = null);

public sealed record SplitTranscriptSegmentRequest(
    Guid TranscriptRevisionId,
    Guid SegmentId,
    double SplitSeconds);

public sealed record MergeTranscriptSegmentsRequest(
    Guid TranscriptRevisionId,
    Guid FirstSegmentId,
    Guid SecondSegmentId);

public sealed record MergeTranscriptSegmentRunRequest(
    Guid TranscriptRevisionId,
    IReadOnlyList<Guid> SegmentIds);

public sealed record TrimTranscriptSegmentRequest(
    Guid TranscriptRevisionId,
    Guid SegmentId,
    double StartSeconds,
    double EndSeconds);

public sealed record DeleteTranscriptSegmentRequest(
    Guid TranscriptRevisionId,
    Guid SegmentId);

public sealed record RetranscribeTranscriptSegmentsRequest(
    Guid TranscriptRevisionId,
    IReadOnlyList<Guid> SegmentIds,
    string? PreferredModelAlias = null,
    bool RequirePreferredModelAlias = false,
    ExecutionProviderKind? PreferredExecutionProvider = null,
    bool RequirePreferredExecutionProvider = false,
    string? PreferredModelVariantAlias = null);

public sealed record RerunDiarizationRequest(
    string? PreferredModelAlias = null,
    ExecutionProviderKind? PreferredExecutionProvider = null,
    bool RequirePreferredExecutionProvider = false,
    string? PreferredModelVariantAlias = null);

public sealed record RenameSpeakerRequest(
    Guid SpeakerId,
    string DisplayName);

public sealed record MergeSpeakersRequest(
    Guid SourceSpeakerId,
    Guid TargetSpeakerId);

public sealed record DeleteEmptySpeakerRequest(
    Guid SpeakerId);

public sealed record AssignVoiceToSpeakerRequest(
    Guid SpeakerId,
    string VoiceId,
    string VoiceModelId = "kokoro-onnx");

public sealed record GenerateTtsForSpeakerRequest(
    Guid SpeakerId,
    string? PreferredModelAlias = null,
    bool UseReferenceClipForVoiceCloning = false,
    ExecutionProviderKind? PreferredExecutionProvider = null,
    bool RequirePreferredExecutionProvider = false,
    string? PreferredModelVariantAlias = null);

public sealed record GenerateTtsForSegmentRequest(
    Guid TranscriptRevisionId,
    Guid SegmentId,
    string? PreferredModelAlias = null,
    bool UseReferenceClipForVoiceCloning = false,
    ExecutionProviderKind? PreferredExecutionProvider = null,
    bool RequirePreferredExecutionProvider = false,
    string? PreferredModelVariantAlias = null);

public sealed record GenerateTtsForAllSpeakersRequest(
    IReadOnlyDictionary<Guid, string>? FallbackVoiceIdsBySpeakerId = null,
    string? PreferredModelAlias = null,
    IReadOnlyDictionary<Guid, bool>? UseReferenceClipForVoiceCloningBySpeakerId = null,
    ExecutionProviderKind? PreferredExecutionProvider = null,
    bool RequirePreferredExecutionProvider = false,
    string? PreferredModelVariantAlias = null,
    IReadOnlyDictionary<Guid, string>? VoiceIdsBySpeakerId = null);

public sealed record PreviewVoiceRequest(
    string VoiceId,
    string LanguageCode,
    string SampleText,
    string? PreferredModelAlias = null,
    ExecutionProviderKind? PreferredExecutionProvider = null,
    bool RequirePreferredExecutionProvider = false,
    string? PreferredModelVariantAlias = null);

public sealed record PreviewVoiceResult(
    byte[] WavBytes,
    int SampleRate,
    string ModelId,
    string VoiceId,
    string Provider);

public sealed record RestoreEditingStateRequest(
    string? SelectedTranslationTargetLanguage,
    IReadOnlyList<TranscriptSegment> TranscriptSegments,
    IReadOnlyList<TranslatedSegment>? TranslatedSegments,
    IReadOnlyDictionary<Guid, string> SpeakerDisplayNames,
    IReadOnlyList<VoiceAssignment> VoiceAssignments);

public sealed record RegenerateStaleTtsForSpeakerRequest(
    Guid SpeakerId,
    string? PreferredModelAlias = null,
    bool UseReferenceClipForVoiceCloning = false,
    ExecutionProviderKind? PreferredExecutionProvider = null,
    bool RequirePreferredExecutionProvider = false,
    string? PreferredModelVariantAlias = null);

public sealed record StretchTtsTakeRequest(
    Guid TakeId);

public sealed record AssignSpeakerToSegmentRequest(
    Guid TranscriptRevisionId,
    Guid SegmentId,
    Guid SpeakerId);

public sealed record AssignSpeakerToSegmentsRequest(
    Guid TranscriptRevisionId,
    IReadOnlyList<Guid> SegmentIds,
    Guid SpeakerId);

public sealed record CreateSpeakerFromSegmentsRequest(
    Guid TranscriptRevisionId,
    IReadOnlyList<Guid> SegmentIds);

public sealed record SplitSpeakerTurnRequest(
    Guid SpeakerTurnId,
    double SplitSeconds);

public sealed record ExtractReferenceClipRequest(
    Guid SpeakerId,
    Guid? SpeakerTurnId = null);

public sealed record ImportReferenceClipRequest(
    Guid SpeakerId,
    string SourcePath);

public sealed record VoiceAssignmentWarning(
    Guid SpeakerId,
    string VoiceId,
    string Message);

public sealed record TtsSegmentState(
    int SegmentIndex,
    Guid? TakeId,
    string? ArtifactRelativePath,
    TtsTakeStatus? Status,
    bool IsStale,
    double? DurationSeconds,
    double? DurationOverrunRatio,
    bool HasDurationWarning,
    string? WarningMessage,
    double? OriginalDurationSeconds = null,
    double? PreStretchDurationSeconds = null,
    double? StretchRatioApplied = null,
    TtsStretchMode StretchMode = TtsStretchMode.None,
    TtsStretchEngine StretchEngine = TtsStretchEngine.None,
    TtsDurationSeverity DurationSeverity = TtsDurationSeverity.None,
    bool HasSpeedLimitWarning = false,
    bool CanManualStretch = false,
    string? Provider = null,
    string? ModelId = null,
    string? ModelAlias = null,
    string? ModelVariant = null);

/// <summary>
/// Projection of a <see cref="Trackdub.Domain.LipSync.LipSyncSegment"/> for display in the UI.
/// Keyed by <see cref="SegmentIndex"/> so it can be joined to <see cref="TranscriptSegment"/>.
/// </summary>
public sealed record LipSyncSegmentState(
    int SegmentIndex,
    Trackdub.Domain.LipSync.LipSyncSegmentStatus Status,
    TimeSpan? AlignedTtsDuration,
    double? PlanConfidence,
    string? SkipReason,
    string? FailureReason,
    string? ProviderId,
    string? ModelId);

/// <summary>
/// Per-transcript-segment projection of M23 lip-synthesis outcomes overlapping the segment window.
/// </summary>
public sealed record LipSynthesisSegmentUiState(
    int SegmentIndex,
    Trackdub.Domain.LipSynthesis.LipSynthesisSegmentStatus Status,
    double? FaceConfidence,
    string? SkipReason,
    string? FailureReason,
    string? ProviderId,
    string? ModelId,
    bool UsedExperimentalProvider);

public sealed record TranscriptProjectState(
    OpenProjectResult ProjectState,
    TranscriptRevision? CurrentTranscriptRevision,
    IReadOnlyList<TranscriptSegment> TranscriptSegments,
    IReadOnlyList<ProjectSpeaker> Speakers,
    IReadOnlyList<SpeakerTurn> SpeakerTurns,
    TranslationRevision? CurrentTranslationRevision,
    IReadOnlyList<TranslatedSegment> TranslatedSegments,
    bool IsTranslationStale,
    string? TranscriptLanguage,
    IReadOnlyList<StageRunRecord> StageRuns,
    IReadOnlyList<TranslationTargetLanguageOption> SupportedTargetLanguages,
    string? SelectedTranslationTargetLanguage,
    IReadOnlySet<int> StaleTranslatedSegmentIndices,
    WaveformSummary? WaveformSummary,
    IReadOnlyList<VoiceCatalogEntry> AvailableVoices,
    IReadOnlyList<VoiceAssignment> VoiceAssignments,
    IReadOnlyList<TtsTake> TtsTakes,
    IReadOnlyList<TtsSegmentState> TtsSegmentStates,
    IReadOnlyList<VoiceAssignmentWarning> VoiceAssignmentWarnings,
    string AsrAudioRelativePath = ProjectArtifactPaths.NormalizedAudioRelativePath,
    string MixSourceAudioRelativePath = ProjectArtifactPaths.NormalizedAudioRelativePath,
    string? StemSeparationWarning = null,
    MixPlan? CurrentMixPlan = null,
    ExportToolAvailability? ExportTools = null,
    ProjectUiSettings? ProjectUiSettings = null,
    IReadOnlyList<TtsCandidateGroup>? TtsCandidateGroups = null,
    IReadOnlyList<LipSyncSegmentState>? LipSyncSegmentStates = null,
    IReadOnlyList<LipSynthesisSegmentUiState>? LipSynthesisSegmentStates = null);
