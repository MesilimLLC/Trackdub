using System.CommandLine;
using Trackdub.Contracts;

namespace Trackdub.Cli.Commands;

internal sealed record PipelineCommandOptions(
    Option<string?> Media,
    Option<string?> TargetLanguage,
    Option<string?> SourceLanguage,
    Option<string?> Output,
    Option<string[]> Model,
    Option<string[]> Variant,
    Option<string?> ExportFormat,
    Option<string?> FromStage,
    Option<string[]> Only,
    Option<bool> LipSync,
    Option<bool> RepairLips,
    Option<bool> ForceRerun,
    Option<bool?> EnableAsrTextRefinement,
    Option<bool> VoiceClone,
    Option<bool?> TimbrePolish,
    Option<bool> NoTimbrePolish,
    Option<bool?> RestorePan,
    Option<bool?> MatchLoudness,
    Option<string[]> Voice,
    Option<string[]> SubtitleFormat,
    Option<string?> SubtitleSource,
    Option<bool> BurnInSubtitles,
    Option<string?> VideoEncoder,
    Option<string?> Preset,
    Option<string?> InputDir,
    Option<string?> InputGlob,
    Option<bool> Recursive,
    Option<bool> ContinueOnError,
    Option<bool?> TtsRubberbandStretch,
    Option<bool> NoTtsRubberbandStretch,
    Option<double?> TtsRubberbandThreshold,
    Option<double?> TtsAutoStretchMaxOverrun,
    Option<string?> ResultFile,
    Option<string?> ProgressFile);

internal static class RunPipelineCommandOptions
{
    public static PipelineCommandOptions Create(string[] acceptedStageNames)
    {
        var media = CreateMediaOption();
        var targetLanguage = CreateTargetLanguageOption();
        var sourceLanguage = CreateSourceLanguageOption();
        var output = CreateOutputOption();
        var model = CreateModelOption();
        var variant = CreateVariantOption();
        var exportFormat = CreateExportFormatOption();
        var fromStage = CreateFromStageOption(acceptedStageNames);
        var only = CreateOnlyOption();
        var lipSync = new Option<bool>("--lip-sync")
        {
            Description = "Align dubbed speech to source mouth cadence (stage: lip-sync)",
        };
        var repairLips = new Option<bool>("--repair-lips")
        {
            Description = "Repair video mouth motion after mixing audio (stage: lip-synthesis)",
        };
        var forceRerun = CreateForceRerunOption();
        var refinement = CreateRefinementOption();
        var voiceClone = CreateVoiceCloneOption();
        var timbre = CreateTimbreOptions(out Option<bool> noTimbre);
        var restorePan = CreateRestorePanOption();
        var matchLoudness = CreateMatchLoudnessOption();
        var voice = CreateVoiceOption();
        var subtitleFormat = CreateSubtitleFormatOption();
        var subtitleSource = CreateSubtitleSourceOption();
        var burnIn = CreateBurnInOption();
        var videoEncoder = CreateVideoEncoderOption();
        var preset = CreatePresetOption();
        var batch = CreateBatchOptions(out Option<string?> inputGlob, out Option<bool> recursive, out Option<bool> continueOnError);
        var ttsRubberband = CreateTtsRubberbandOptions(out Option<bool> noTtsRubberband, out Option<double?> ttsRubberbandThreshold, out Option<double?> ttsAutoStretchMaxOverrun);

        return new PipelineCommandOptions(
            media, targetLanguage, sourceLanguage, output, model, variant, exportFormat,
            fromStage, only, lipSync, repairLips, forceRerun, refinement, voiceClone, timbre, noTimbre,
            restorePan, matchLoudness, voice, subtitleFormat, subtitleSource,
            burnIn, videoEncoder, preset, batch, inputGlob, recursive, continueOnError,
            ttsRubberband, noTtsRubberband, ttsRubberbandThreshold, ttsAutoStretchMaxOverrun,
            new Option<string?>("--result-file") { Description = "Write a strict atomic machine result to a new file" },
            new Option<string?>("--progress-file") { Description = "Write dedicated JSONL progress to a new file" });
    }

    public static void AddTo(Command command, PipelineCommandOptions options)
    {
        command.Add(options.ResultFile);
        command.Add(options.ProgressFile);
        command.Add(options.Media);
        command.Add(options.TargetLanguage);
        command.Add(options.SourceLanguage);
        command.Add(options.Output);
        command.Add(options.Model);
        command.Add(options.Variant);
        command.Add(options.ExportFormat);
        command.Add(options.FromStage);
        command.Add(options.Only);
        command.Add(options.LipSync);
        command.Add(options.RepairLips);
        command.Add(options.ForceRerun);
        command.Add(options.EnableAsrTextRefinement);
        command.Add(options.VoiceClone);
        command.Add(options.TimbrePolish);
        command.Add(options.NoTimbrePolish);
        command.Add(options.RestorePan);
        command.Add(options.MatchLoudness);
        command.Add(options.Voice);
        command.Add(options.SubtitleFormat);
        command.Add(options.SubtitleSource);
        command.Add(options.BurnInSubtitles);
        command.Add(options.VideoEncoder);
        command.Add(options.Preset);
        command.Add(options.InputDir);
        command.Add(options.InputGlob);
        command.Add(options.Recursive);
        command.Add(options.ContinueOnError);
        command.Add(options.TtsRubberbandStretch);
        command.Add(options.NoTtsRubberbandStretch);
        command.Add(options.TtsRubberbandThreshold);
        command.Add(options.TtsAutoStretchMaxOverrun);
    }

    private static Option<string?> CreateMediaOption() => new("--media")
    {
        Description = "Path to the source media file",
    };

    private static Option<string?> CreateTargetLanguageOption() => new("--target-language")
    {
        Description = "Target language BCP-47 code (e.g., es, fr, de)",
    };

    private static Option<string?> CreateSourceLanguageOption() => new("--source-language")
    {
        Description = "Source language BCP-47 code. When omitted, ASR auto-detects the source language",
    };

    private static Option<string?> CreateOutputOption() => new("--output")
    {
        Description = "Output directory for the project. Defaults to <media-stem>.trackdub adjacent to source media",
    };

    private static Option<string[]> CreateModelOption()
    {
        var model = new Option<string[]>("--model")
        {
            Description = "Stage-specific model override in format stage:alias (repeatable)",
            AllowMultipleArgumentsPerToken = true,
        };
        model.Arity = System.CommandLine.ArgumentArity.ZeroOrMore;
        return model;
    }

    private static Option<string[]> CreateVariantOption()
    {
        var variant = new Option<string[]>("--variant")
        {
            Description = "Stage-specific model variant in format stage:variant, e.g. translation:trt-fp16 (repeatable). "
                + "Overrides the variant a --preset selects; the variant must be installed for the chosen model",
            AllowMultipleArgumentsPerToken = true,
            HelpName = "stage:variant",
        };
        variant.Arity = System.CommandLine.ArgumentArity.ZeroOrMore;
        return variant;
    }

    private static Option<string?> CreateExportFormatOption()
    {
        var exportFormat = new Option<string?>("--export-format")
        {
            Description = "Export container format (mp4 or mkv)",
        };
        exportFormat.AcceptOnlyFromAmong("mp4", "mkv");
        return exportFormat;
    }

    private static Option<string?> CreateFromStageOption(string[] acceptedStageNames)
    {
        var fromStage = new Option<string?>("--from-stage")
        {
            Description = "Run from this stage through export in canonical order",
        };
        fromStage.AcceptOnlyFromAmong(acceptedStageNames);
        return fromStage;
    }

    private static Option<string[]> CreateOnlyOption()
    {
        var only = new Option<string[]>("--only")
        {
            Description = "Run only the listed stages (repeatable)",
            AllowMultipleArgumentsPerToken = true,
        };
        only.Arity = System.CommandLine.ArgumentArity.ZeroOrMore;
        return only;
    }

    private static Option<bool> CreateForceRerunOption() => new("--force-rerun")
    {
        Description = "Re-run stages even when valid artifacts already exist",
        DefaultValueFactory = _ => false,
    };

    private static Option<bool?> CreateRefinementOption() => new("--enable-asr-text-refinement")
    {
        Description = "Run optional Qwen ASR text polish after transcription",
    };

    private static Option<bool> CreateVoiceCloneOption() => new("--voice-clone")
    {
        Description = "Clone each speaker from source audio instead of a stock voicepack. Grants session voice-cloning consent for this run.",
        DefaultValueFactory = _ => false,
    };

    private static Option<bool?> CreateTimbreOptions(out Option<bool> noTimbre)
    {
        var timbrePolish = new Option<bool?>("--timbre-polish")
        {
            Description = "Apply room-tone convolution to dubbed speech to match the acoustic environment (default: true)",
        };
        noTimbre = new Option<bool>("--no-timbre-polish")
        {
            Description = "Disable room-tone convolution (negates the default-on --timbre-polish)",
        };
        return timbrePolish;
    }

    private static Option<bool?> CreateRestorePanOption() => new("--restore-pan")
    {
        Description = "Restore the original stereo pan position of each speaker in the dubbed mix (default: false)",
    };

    private static Option<bool?> CreateMatchLoudnessOption() => new("--match-loudness")
    {
        Description = "Measure source loudness and normalize the dubbed mix to match it (default: false)",
    };

    private static Option<string[]> CreateVoiceOption()
    {
        var voice = new Option<string[]>("--voice")
        {
            Description = "Assign a specific voice to a speaker in format SPEAKER_ID:voice_id (repeatable, e.g., --voice SPEAKER_00:af_bella)",
            AllowMultipleArgumentsPerToken = true,
        };
        voice.Arity = System.CommandLine.ArgumentArity.ZeroOrMore;
        return voice;
    }

    private static Option<string[]> CreateSubtitleFormatOption()
    {
        var subtitleFormat = new Option<string[]>("--subtitle-format")
        {
            Description = "Subtitle format(s) to include: srt, vtt, ass, or none (repeatable; default: srt)",
            AllowMultipleArgumentsPerToken = true,
        };
        subtitleFormat.Arity = System.CommandLine.ArgumentArity.ZeroOrMore;
        subtitleFormat.AcceptOnlyFromAmong("srt", "vtt", "ass", "none");
        return subtitleFormat;
    }

    private static Option<string?> CreateSubtitleSourceOption()
    {
        var subtitleSource = new Option<string?>("--subtitle-source")
        {
            Description = "Which transcript to use for subtitles: translated (default), transcript, bilingual",
        };
        subtitleSource.AcceptOnlyFromAmong("translated", "transcript", "bilingual");
        return subtitleSource;
    }

    private static Option<bool> CreateBurnInOption() => new("--burn-in")
    {
        Description = "Burn subtitles into the exported video (default: false)",
        DefaultValueFactory = _ => false,
    };

    private static Option<string?> CreateVideoEncoderOption()
    {
        var videoEncoder = new Option<string?>("--video-encoder")
        {
            Description = $"Video encoder preference: {string.Join(", ", VideoEncoderPreferenceSettings.AutoKey, VideoEncoderPreferenceSettings.SoftwareKey, VideoEncoderPreferenceSettings.NvencKey, VideoEncoderPreferenceSettings.QsvKey, VideoEncoderPreferenceSettings.AmfKey, VideoEncoderPreferenceSettings.VideoToolboxKey, VideoEncoderPreferenceSettings.VaapiKey)} (default: auto)",
        };
        videoEncoder.AcceptOnlyFromAmong(
            VideoEncoderPreferenceSettings.AutoKey,
            VideoEncoderPreferenceSettings.SoftwareKey,
            VideoEncoderPreferenceSettings.NvencKey,
            VideoEncoderPreferenceSettings.QsvKey,
            VideoEncoderPreferenceSettings.AmfKey,
            VideoEncoderPreferenceSettings.VideoToolboxKey,
            VideoEncoderPreferenceSettings.VaapiKey);
        return videoEncoder;
    }

    private static Option<string?> CreatePresetOption() => new("--preset")
    {
        Description = "Named preset to load pipeline settings from",
    };

    private static Option<string?> CreateBatchOptions(
        out Option<string?> inputGlob,
        out Option<bool> recursive,
        out Option<bool> continueOnError)
    {
        var inputDir = new Option<string?>("--input-dir")
        {
            Description = "Directory path for batch processing of all supported media files",
        };
        inputGlob = new Option<string?>("--input-glob")
        {
            Description = "Glob pattern for batch processing (resolved relative to current directory)",
        };
        recursive = new Option<bool>("--recursive")
        {
            Description = "Discover media files recursively (only valid with --input-dir)",
            DefaultValueFactory = _ => false,
        };
        continueOnError = new Option<bool>("--continue-on-error")
        {
            Description = "Continue processing remaining files after a failure in batch mode",
            DefaultValueFactory = _ => false,
        };
        return inputDir;
    }

    private static Option<bool?> CreateTtsRubberbandOptions(
        out Option<bool> noTtsRubberband,
        out Option<double?> ttsRubberbandThreshold,
        out Option<double?> ttsAutoStretchMaxOverrun)
    {
        var ttsRubberbandStretch = new Option<bool?>("--tts-rubberband-stretch")
        {
            Description = "Force-enable Rubberband time-stretch when fitting dubbed TTS to source segment pace. Optional; overrides host TTS timing settings for this run.",
        };
        noTtsRubberband = new Option<bool>("--no-tts-rubberband-stretch")
        {
            Description = "Force-disable Rubberband stretch (use ffmpeg atempo). Optional; overrides host TTS timing settings for this run.",
            DefaultValueFactory = _ => false,
        };
        ttsRubberbandThreshold = new Option<double?>("--tts-rubberband-threshold")
        {
            Description = "Rubberband stretch mismatch threshold in 0..1 (fraction of source duration). Optional; overrides host settings when Rubberband stretch is on.",
        };
        ttsAutoStretchMaxOverrun = new Option<double?>("--tts-auto-stretch-max-overrun")
        {
            Description = "Maximum take overhang (fraction of source duration, 0..1) that auto time-stretch is allowed to fix. Takes overrunning by more are flagged for manual stretching instead. Optional; overrides host TTS timing settings for this run.",
        };
        return ttsRubberbandStretch;
    }
}
