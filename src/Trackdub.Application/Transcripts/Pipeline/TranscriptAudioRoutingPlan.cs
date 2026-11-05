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

    private static readonly string _studioVoiceProfileHash =
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes("nvidia-afx-studio-voice-high-quality")))
            .ToLowerInvariant();

    public TranscriptAudioRoutingPlan WithStudioVoiceAsrSource(ProjectArtifact enhancedAudioArtifact)
    {
        ArgumentNullException.ThrowIfNull(enhancedAudioArtifact);
        const string profileId = "nvidia-afx-studio-voice-high-quality";
        const string effectSelector = "studio_voice_high_quality";
        string profileHash = _studioVoiceProfileHash;

        return this with
        {
            AsrAudioArtifact = enhancedAudioArtifact,
            AsrDecision = AsrDecision with
            {
                ProfileId = profileId,
                ProfileVersion = 1,
                CatalogVersion = "nvidia-afx-profile-v1",
                FilterChain = effectSelector,
                ProfileHash = profileHash,
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

    public static bool IsStudioVoiceEnhancementArtifact(ProjectArtifact artifact)
    {
        ArgumentNullException.ThrowIfNull(artifact);
        const string provenancePrefix = "generated-nvidia-afx-speech-enhancement:";
        string provenanceSuffix = $":profile={NvidiaAfxIntegration.StudioVoiceHighQualitySelector}";
        return artifact.Provenance is { } provenance
            && provenance.StartsWith(provenancePrefix, StringComparison.Ordinal)
            && provenance.EndsWith(provenanceSuffix, StringComparison.Ordinal);
    }

    public TranscriptAudioRoutingPlan WithStudioVoiceAsrSourceIfApplicable(ProjectArtifact enhancedAudioArtifact) =>
        IsStudioVoiceEnhancementArtifact(enhancedAudioArtifact)
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
