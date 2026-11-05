using Trackdub.Contracts;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.AudioQuality;
using Trackdub.Domain.Media;

namespace Trackdub.Application.Tests;

public sealed partial class SpeechAudioPreparationTests
{
    [Fact]
    public void WithStudioVoiceAsrSource_routes_asr_to_the_enhancement_artifact()
    {
        MediaAsset mediaAsset = CreateMediaAsset();
        ProjectArtifact original = CreateArtifact(mediaAsset, ArtifactKind.NormalizedAudio, "media/normalized_audio.wav");
        ProjectArtifact enhanced = CreateArtifact(mediaAsset, ArtifactKind.SpeechEnhancedAudio, "artifacts/audio/speech-enhancement/x/speech.wav");

        TranscriptAudioRoutingPlan plan = TranscriptAudioRoutingPlan.Raw(original, SpeechAudioSourceKind.FullMix)
            .WithStudioVoiceAsrSource(enhanced);

        Assert.Same(enhanced, plan.AsrAudioArtifact);
        Assert.Equal("nvidia-afx-studio-voice-high-quality", plan.AsrDecision.ProfileId);
        Assert.True(plan.AsrDecision.RequiresProcessing);
        Assert.Equal(enhanced.Id, plan.AsrDecision.OutputArtifactId);
        Assert.Equal(enhanced.RelativePath, plan.AsrDecision.OutputRelativePath);
        Assert.Same(original, plan.VadAudioArtifact);
    }

    [Theory]
    [InlineData(SpeechAudioEnhancementBackend.Ffmpeg, "studio_voice_high_quality")]
    [InlineData(SpeechAudioEnhancementBackend.DeepFilterNet, "studio_voice_high_quality")]
    [InlineData(SpeechAudioEnhancementBackend.NvidiaAfx, "dereverb_denoiser")]
    public void WithSpeechEnhancementAsrSource_keeps_original_when_studio_voice_did_not_run(
        SpeechAudioEnhancementBackend backend,
        string profile)
    {
        MediaAsset mediaAsset = CreateMediaAsset();
        ProjectArtifact original = CreateArtifact(mediaAsset, ArtifactKind.NormalizedAudio, "media/normalized_audio.wav");
        ProjectArtifact enhanced = CreateArtifact(mediaAsset, ArtifactKind.SpeechEnhancedAudio, "artifacts/audio/speech-enhancement/x/speech.wav");
        TranscriptAudioRoutingPlan plan = TranscriptAudioRoutingPlan.Raw(original, SpeechAudioSourceKind.FullMix);

        TranscriptAudioRoutingPlan routed = plan.WithSpeechEnhancementAsrSource(enhanced, backend, profile);

        Assert.Same(original, routed.AsrAudioArtifact);
        Assert.Equal(SpeechAudioProcessingProfileCatalog.NoneProfileId, routed.AsrDecision.ProfileId);
    }

    [Fact]
    public void IsStudioVoiceEnhancementArtifact_uses_persisted_backend_and_profile_provenance()
    {
        MediaAsset mediaAsset = CreateMediaAsset();
        ProjectArtifact original = CreateArtifact(mediaAsset, ArtifactKind.NormalizedAudio, "media/normalized_audio.wav");
        ProjectArtifact studioVoice = CreateArtifact(mediaAsset, ArtifactKind.SpeechEnhancedAudio, "artifacts/audio/speech-enhancement/x/speech.wav") with
        {
            Provenance = "generated-nvidia-afx-speech-enhancement:source-id:profile=studio_voice_high_quality"
        };
        ProjectArtifact fallback = studioVoice with
        {
            Provenance = "generated-deepfilternet-speech-enhancement:source-id:profile=deepfilternet3"
        };

        Assert.True(TranscriptAudioRoutingPlan.IsStudioVoiceEnhancementArtifact(studioVoice));
        Assert.False(TranscriptAudioRoutingPlan.IsStudioVoiceEnhancementArtifact(fallback));
        Assert.False(TranscriptAudioRoutingPlan.IsStudioVoiceEnhancementArtifact(original));
    }
}
