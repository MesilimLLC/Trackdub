using System.Security.Cryptography;
using System.Text;
using Trackdub.Contracts;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.AudioQuality;

namespace Trackdub.Application.Transcripts.Pipeline;

public sealed record TranscriptAudioRoutingPlan(
    ProjectArtifact VadAudioArtifact,
    ProjectArtifact AsrAudioArtifact,
    ProjectArtifact DiarizationAudioArtifact,
    SpeechAudioSourceKind SourceKind,
    ProjectArtifact? AnalysisArtifact,
    SpeechAudioStageDecision VadDecision,
    SpeechAudioStageDecision AsrDecision,
    SpeechAudioStageDecision DiarizationDecision)
{
    public static TranscriptAudioRoutingPlan Raw(ProjectArtifact sourceArtifact, SpeechAudioSourceKind sourceKind) =>
        new(
            sourceArtifact,
            sourceArtifact,
            sourceArtifact,
            sourceKind,
            AnalysisArtifact: null,
            CreateRawDecision(SpeechPipelineStageKind.Vad, sourceKind),
            CreateRawDecision(SpeechPipelineStageKind.Asr, sourceKind),
            CreateRawDecision(SpeechPipelineStageKind.Diarization, sourceKind));

    /// <summary>
    /// Routes ASR to the unprocessed full mix. Speech enhancement (DeepFilterNet or the ffmpeg
    /// chain) and stem separation measurably degraded Parakeet-TDT and at best matched Qwen3-ASR,
    /// while VAD and diarization keep the enhanced audio.
    /// </summary>
    public TranscriptAudioRoutingPlan WithUnprocessedAsrSource(ProjectArtifact normalizedAudioArtifact) =>
        this with
        {
            AsrAudioArtifact = normalizedAudioArtifact,
            AsrDecision = CreateRawDecision(SpeechPipelineStageKind.Asr, SourceKind),
        };

    private const string StudioVoiceProfileId = "nvidia-afx-studio-voice-high-quality";
    private const string StudioVoiceProvenancePrefix = "generated-nvidia-afx-speech-enhancement:";
    private static readonly string StudioVoiceProvenanceSuffix =
        $":profile={NvidiaAfxIntegration.StudioVoiceHighQualitySelector}";
    private static readonly string _studioVoiceProfileHash =
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(StudioVoiceProfileId)))
            .ToLowerInvariant();

    public TranscriptAudioRoutingPlan WithStudioVoiceAsrSource(ProjectArtifact enhancedAudioArtifact)
    {
        ArgumentNullException.ThrowIfNull(enhancedAudioArtifact);

        return this with
        {
            AsrAudioArtifact = enhancedAudioArtifact,
            AsrDecision = AsrDecision with
            {
                ProfileId = StudioVoiceProfileId,
                ProfileVersion = 1,
                CatalogVersion = "nvidia-afx-profile-v1",
                FilterChain = NvidiaAfxIntegration.StudioVoiceHighQualitySelector,
                ProfileHash = _studioVoiceProfileHash,
                RequiresProcessing = true,
                OutputArtifactId = enhancedAudioArtifact.Id,
                OutputRelativePath = enhancedAudioArtifact.RelativePath,
            },
        };
    }

    public TranscriptAudioRoutingPlan WithSpeechEnhancementAsrSource(
        ProjectArtifact enhancedAudioArtifact,
        SpeechAudioEnhancementBackend? backend,
        string? backendProfile) =>
        backend == SpeechAudioEnhancementBackend.NvidiaAfx
        && string.Equals(
            backendProfile,
            NvidiaAfxIntegration.StudioVoiceHighQualitySelector,
            StringComparison.Ordinal)
            ? WithStudioVoiceAsrSource(enhancedAudioArtifact)
            : this;

    /// <summary>
    /// True when the artifact was produced by the native Studio Voice effect. When
    /// <paramref name="sourceArtifactIds"/> is supplied, the persisted source lineage must also match
    /// one of the current source artifacts, so a stale enhancement from an earlier source is never
    /// routed to ASR after the source changes.
    /// </summary>
    public static bool IsStudioVoiceEnhancementArtifact(
        ProjectArtifact artifact,
        IReadOnlyCollection<Guid>? sourceArtifactIds = null)
    {
        ArgumentNullException.ThrowIfNull(artifact);
        if (artifact.Provenance is not { } provenance
            || !provenance.StartsWith(StudioVoiceProvenancePrefix, StringComparison.Ordinal)
            || !provenance.EndsWith(StudioVoiceProvenanceSuffix, StringComparison.Ordinal))
        {
            return false;
        }

        if (sourceArtifactIds is null || sourceArtifactIds.Count == 0)
        {
            return true;
        }

        return sourceArtifactIds.Any(sourceArtifactId =>
            string.Equals(
                provenance,
                $"{StudioVoiceProvenancePrefix}{sourceArtifactId:D}{StudioVoiceProvenanceSuffix}",
                StringComparison.Ordinal));
    }

    public TranscriptAudioRoutingPlan WithStudioVoiceAsrSourceIfApplicable(
        ProjectArtifact enhancedAudioArtifact,
        IReadOnlyCollection<Guid>? sourceArtifactIds = null) =>
        IsStudioVoiceEnhancementArtifact(enhancedAudioArtifact, sourceArtifactIds)
            ? WithStudioVoiceAsrSource(enhancedAudioArtifact)
            : this;

    private static SpeechAudioStageDecision CreateRawDecision(
        SpeechPipelineStageKind stage,
        SpeechAudioSourceKind sourceKind)
    {
        SpeechAudioFilterSelection selection = SpeechAudioProcessingProfileCatalog.BuildFilterSelection(
            SpeechAudioProcessingProfileCatalog.NoneProfileId,
            []);
        return new SpeechAudioStageDecision(
            stage,
            sourceKind,
            selection.ProfileId,
            selection.ProfileVersion,
            selection.CatalogVersion,
            selection.FilterChain,
            selection.ProfileHash,
            RequiresProcessing: false,
            TriggeredDefects: []);
    }
}
