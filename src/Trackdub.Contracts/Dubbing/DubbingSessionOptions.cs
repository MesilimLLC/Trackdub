using Trackdub.Contracts;
using Trackdub.Domain.Translation;

namespace Trackdub.Contracts.Dubbing;

using Trackdub.Contracts.ApplicationContracts;

/// <summary>
/// Immutable configuration snapshot capturing all inputs required to execute
/// a full or partial dubbing pipeline run.
/// </summary>
public sealed record DubbingSessionOptions
{
    /// <summary>
    /// Path to the source media file (required).
    /// </summary>
    public required string SourceMediaPath { get; init; }

    /// <summary>
    /// Directory where the project is created or found.
    /// When null, derived from the source media path.
    /// </summary>
    public string? ProjectOutputDirectory { get; init; }

    /// <summary>
    /// BCP-47 language code for the source language.
    /// When null, the ASR stage auto-detects the source language.
    /// </summary>
    public string? SourceLanguageCode { get; init; }

    /// <summary>
    /// BCP-47 language code for the target language (required).
    /// </summary>
    public required string TargetLanguageCode { get; init; }

    /// <summary>
    /// Stage-specific model overrides. Keys are stage names, values are model aliases.
    /// When null, default model selection applies.
    /// </summary>
    public IReadOnlyDictionary<string, string>? ModelPreferences { get; init; }

    /// <summary>Explicit stage variant aliases; overrides persisted local selection for this run.</summary>
    public IReadOnlyDictionary<string, string>? ModelVariantPreferences { get; init; }

    /// <summary>
    /// Stage-specific execution-provider pins. Keys are canonical stage names
    /// (<see cref="Trackdub.Domain.StageRuns.StageNames"/> members that map to a
    /// <see cref="Trackdub.Domain.RuntimeStage"/>), values are provider labels
    /// understood by <see cref="Trackdub.Domain.ExecutionProviderTokens"/>.
    /// When null, the planner chooses providers as usual.
    /// </summary>
    public IReadOnlyDictionary<string, string>? ExecutionProviderPreferences { get; init; }

    /// <summary>
    /// When true, <see cref="ExecutionProviderPreferences"/> pins are required: a stage
    /// whose pinned provider cannot be used fails instead of silently falling back.
    /// Defaults to false (pins act as preferences).
    /// </summary>
    public bool RequireExecutionProviderPreferences { get; init; }

    /// <summary>
    /// Preferred export container format. Supported values are "mp4" and "mkv".
    /// When null, the export stage uses the product default.
    /// </summary>
    public string? ExportFormat { get; init; }

    /// <summary>
    /// Absolute output path for the Export stage. When null, the engine derives
    /// a path inside the project directory from the container format.
    /// Interactive hosts pass the user's chosen destination.
    /// </summary>
    public string? ExportOutputPath { get; init; }

    /// <summary>
    /// Target loudness (LUFS) for the exported mix. When null, the export stage
    /// uses the product default loudness target.
    /// </summary>
    public double? ExportTargetLufs { get; init; }

    /// <summary>
    /// Gain applied to the source/background bed in the exported mix (dB).
    /// </summary>
    public double? ExportSourceGainDb { get; init; }

    /// <summary>
    /// Gain applied to dubbed speech in the exported mix (dB).
    /// </summary>
    public double? ExportDubbedSpeechGainDb { get; init; }

    /// <summary>
    /// Optional ducking gain applied to the source bed under dubbed speech (dB).
    /// </summary>
    public double? ExportDuckingGainDb { get; init; }

    /// <summary>
    /// Optional voice assignment overrides per speaker.
    /// Keys are speaker identifiers, values are voice identifiers.
    /// </summary>
    public IReadOnlyDictionary<string, string>? VoiceAssignmentOverrides { get; init; }

    /// <summary>
    /// Optional subset of pipeline stages to execute.
    /// When null, all applicable stages execute in standard order.
    /// </summary>
    public IReadOnlyList<string>? StageFilter { get; init; }

    /// <summary>
    /// When true, optional Qwen ASR text polish runs after transcription.
    /// Defaults to false.
    /// </summary>
    public bool EnableAsrTextRefinement { get; init; }

    /// <summary>
    /// When true, the Separation stage executes when included in the run.
    /// Defaults to true to preserve the historical headless behavior; desktop
    /// callers pass their stem-separation shell flag.
    /// </summary>
    public bool EnableStemSeparation { get; init; } = true;

    /// <summary>
    /// When true, the Diarization stage produces speaker turns during transcription.
    /// Defaults to true to preserve the historical headless behavior.
    /// </summary>
    public bool EnableSpeakerDiarization { get; init; } = true;

    /// <summary>
    /// When true, transcript stages may enhance and prepare source audio before analysis.
    /// When false, they use normalized source audio directly. Defaults to true for callers
    /// that do not specify a preference.
    /// </summary>
    public bool EnableSpeechAudioPreparation { get; init; } = true;

    /// <summary>
    /// When true, the OverlapRescue stage retranscribes rescued regions and merges
    /// candidates into the transcript. Defaults to false (artifacts only).
    /// </summary>
    public bool RetranscribeOverlapCandidates { get; init; }

    /// <summary>
    /// When true, speakers without a voice assignment silently receive a fallback
    /// voice (unattended/headless behavior). When false, TTS fails honestly when
    /// assignments are missing. Defaults to true to preserve headless behavior.
    /// </summary>
    public bool AutoAssignFallbackVoices { get; init; } = true;

    /// <summary>
    /// When true, re-executes all stages regardless of existing artifacts.
    /// Defaults to false.
    /// </summary>
    public bool ForceRerun { get; init; }

    /// <summary>
    /// When true, TTS clones each speaker from source audio instead of assigning a stock voicepack.
    /// Headless runs treat this flag as session voice-cloning consent.
    /// Defaults to false.
    /// </summary>
    public bool UseVoiceCloning { get; init; }

    /// <summary>
    /// Experimental opt-in: enables bounded per-segment streaming inside the translation
    /// stage only. Does not overlap TTS and never exposes a partial translation revision —
    /// persistence still commits atomically at stage end. Default false keeps the batch path.
    /// </summary>
    public bool EnableTranslationSegmentStreaming { get; init; }

    /// <summary>
    /// Which translated clips the translation stage should generate. Defaults to all clips
    /// for existing CLI and automation callers.
    /// </summary>
    public TranslationRunScope TranslationScope { get; init; } = TranslationRunScope.All;

    /// <summary>
    /// Per-speaker voice-clone map: speaker id -> clone from reference audio (true)
    /// or use the assigned/stock voice (false). When set, this map wins over the
    /// blanket <see cref="UseVoiceCloning"/> behavior for TTS. Interactive hosts
    /// pass their per-speaker voice mode selections.
    /// </summary>
    public IReadOnlyDictionary<Guid, bool>? VoiceCloneBySpeakerId { get; init; }

    /// <summary>
    /// When true, applies room-tone convolution to dubbed speech to match the acoustic environment.
    /// Defaults to true.
    /// </summary>
    public bool ApplyTimbrePolish { get; init; } = true;

    /// <summary>
    /// When true, restores the original stereo pan position of each speaker in the dubbed mix.
    /// Defaults to false.
    /// </summary>
    public bool RestoreOriginalPan { get; init; }

    /// <summary>
    /// When true, measures source loudness and normalizes the dubbed mix to match it.
    /// Defaults to false.
    /// </summary>
    public bool MatchOriginalLoudness { get; init; }

    /// <summary>
    /// Subtitle formats to include in the export. Null uses the pipeline default (SRT).
    /// Pass an empty list to suppress all subtitles.
    /// Accepted values: "srt", "vtt", "ass".
    /// </summary>
    public IReadOnlyList<string>? SubtitleFormats { get; init; }

    /// <summary>
    /// Which transcript to use for subtitle text.
    /// Accepted values: "translated" (default), "transcript", "bilingual".
    /// </summary>
    public string? SubtitleSource { get; init; }

    /// <summary>
    /// When true, burns subtitles into the exported video.
    /// Defaults to false.
    /// </summary>
    public bool BurnInSubtitles { get; init; }

    /// <summary>
    /// Preferred video encoder for the export. Defaults to Auto.
    /// </summary>
    public VideoEncoderPreference VideoEncoder { get; init; }

    /// <summary>
    /// Optional per-run TTS timing reconciliation overrides (Rubberband stretch on/off
    /// and mismatch threshold). When null, the session host settings apply
    /// (desktop Settings / <c>settings.json</c> / <see cref="TtsTimingSettings.Default"/>).
    /// </summary>
    public TtsTimingSettings? TtsTiming { get; init; }

    /// <summary>
    /// Optional Windows ML catalog device policy for this run. When null, host settings apply.
    /// </summary>
    public WindowsMlExecutionDevicePolicy? WindowsMlExecutionDevicePolicy { get; init; }

    /// <summary>
    /// Optional explicit cloud egress consent keys for headless/CI runs (e.g. "audio:openai", "text:deepl").
    /// </summary>
    public IReadOnlyList<string>? ExplicitCloudEgressConsents { get; init; }
}
