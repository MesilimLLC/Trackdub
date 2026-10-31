using Trackdub.Contracts;
using Trackdub.Contracts.Dubbing;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Transcripts;

/// <summary>
/// Immutable inputs shared by transcript creation and individual transcript stages.
/// </summary>
public sealed record TranscriptRunOptions
{
    public bool EnableSpeakerDiarization { get; init; } = true;

    public InferenceModelPreferences? ModelPreferences { get; init; }

    public IProgress<PipelineProgressEvent>? Progress { get; init; }

    public string? SourceLanguage { get; init; }

    public bool PrepareSpeechAudio { get; init; } = true;

    internal static TranscriptRunOptions ForDubbingStage(
        DubbingSessionOptions sessionOptions,
        string stageName,
        InferenceModelPreferences modelPreferences,
        IProgress<PipelineProgressEvent>? progress) =>
        new()
        {
            EnableSpeakerDiarization = sessionOptions.EnableSpeakerDiarization
                && stageName is not StageNames.Vad and not StageNames.Asr,
            ModelPreferences = modelPreferences,
            Progress = progress,
            SourceLanguage = TranscriptWorkflowUtilities.NormalizeTranscriptLanguageCode(sessionOptions.SourceLanguageCode),
            PrepareSpeechAudio = sessionOptions.EnableSpeechAudioPreparation
        };
}
