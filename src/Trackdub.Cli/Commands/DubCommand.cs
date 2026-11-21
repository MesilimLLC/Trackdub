using System.CommandLine;
using System.CommandLine.Parsing;

using Trackdub.Cli.Interactive;
using Trackdub.Cli.Handlers;
using Trackdub.Contracts;
using Trackdub.Contracts.Dubbing;
using Trackdub.Sdk;

namespace Trackdub.Cli.Commands;

/// <summary>
/// Defines the <c>dub</c> CLI command that executes a full dubbing pipeline
/// from media ingest through export.
/// </summary>
internal static class DubCommand
{
    /// <summary>
    /// Creates the <c>dub</c> command with all options and the execution handler.
    /// </summary>
    public static Command Create(Func<bool>? isSetupInteractive = null)
    {
        Func<bool> setupInteractive = isSetupInteractive ?? (() => !Console.IsInputRedirected);

        var mediaOption = new Option<string?>("--media")
        {
            Description = "Path to the source media file",
        };

        var targetLanguageOption = new Option<string?>("--target-language")
        {
            Description = "Target language BCP-47 code (e.g., es, fr, de)",
        };

        var sourceLanguageOption = new Option<string?>("--source-language")
        {
            Description = "Source language BCP-47 code. When omitted, ASR auto-detects the source language",
        };

        var outputOption = new Option<string?>("--output")
        {
            Description = "Output directory for the project. Defaults to <media-stem>.trackdub/ adjacent to source media",
        };

        var modelOption = new Option<string[]>("--model")
        {
            Description = "Stage-specific model override in format stage:alias (repeatable, e.g., --model asr:large-v3 --model tts:xtts-v2)",
            AllowMultipleArgumentsPerToken = true,
        };
        modelOption.Arity = ArgumentArity.ZeroOrMore;

        var exportFormatOption = new Option<string?>("--export-format")
        {
            Description = "Export container format (mp4 or mkv)",
        };
        exportFormatOption.AcceptOnlyFromAmong("mp4", "mkv");

        var enableAsrTextRefinementOption = new Option<bool?>("--enable-asr-text-refinement")
        {
            Description = "Run optional Qwen ASR text polish after transcription",
        };

        var voiceCloneOption = new Option<bool>("--voice-clone")
        {
            Description = "Clone each speaker from source audio instead of a stock voicepack. Grants session voice-cloning consent for this run.",
            DefaultValueFactory = _ => false,
        };

        var timbrePolishOption = new Option<bool?>("--timbre-polish")
        {
            Description = "Apply room-tone convolution to dubbed speech to match the acoustic environment (default: true)",
        };

        var noTimbrePolishOption = new Option<bool>("--no-timbre-polish")
        {
            Description = "Disable room-tone convolution (negates the default-on --timbre-polish)",
        };

        var restorePanOption = new Option<bool?>("--restore-pan")
        {
            Description = "Restore the original stereo pan position of each speaker in the dubbed mix (default: false)",
        };

        var matchLoudnessOption = new Option<bool?>("--match-loudness")
        {
            Description = "Measure source loudness and normalize the dubbed mix to match it (default: false)",
        };

        var voiceOption = new Option<string[]>("--voice")
        {
            Description = "Assign a specific voice to a speaker in format SPEAKER_ID:voice_id (repeatable, e.g., --voice SPEAKER_00:af_bella)",
            AllowMultipleArgumentsPerToken = true,
        };
        voiceOption.Arity = ArgumentArity.ZeroOrMore;

        var subtitleFormatOption = new Option<string[]>("--subtitle-format")
        {
            Description = "Subtitle format(s) to include: srt, vtt, ass, or none (repeatable; default: srt)",
            AllowMultipleArgumentsPerToken = true,
        };
        subtitleFormatOption.Arity = ArgumentArity.ZeroOrMore;
        subtitleFormatOption.AcceptOnlyFromAmong("srt", "vtt", "ass", "none");

        var subtitleSourceOption = new Option<string?>("--subtitle-source")
        {
            Description = "Which transcript to use for subtitles: translated (default), transcript, bilingual",
        };
        subtitleSourceOption.AcceptOnlyFromAmong("translated", "transcript", "bilingual");

        var burnInSubtitlesOption = new Option<bool>("--burn-in")
        {
            Description = "Burn subtitles into the exported video (default: false)",
            DefaultValueFactory = _ => false,
        };

        var videoEncoderOption = new Option<string?>("--video-encoder")
        {
            Description = $"Video encoder preference: {string.Join(", ", VideoEncoderPreferenceSettings.AutoKey, VideoEncoderPreferenceSettings.SoftwareKey, VideoEncoderPreferenceSettings.NvencKey, VideoEncoderPreferenceSettings.QsvKey, VideoEncoderPreferenceSettings.AmfKey, VideoEncoderPreferenceSettings.VideoToolboxKey, VideoEncoderPreferenceSettings.VaapiKey)} (default: auto)",
        };
        videoEncoderOption.AcceptOnlyFromAmong(
            VideoEncoderPreferenceSettings.AutoKey,
            VideoEncoderPreferenceSettings.SoftwareKey,
            VideoEncoderPreferenceSettings.NvencKey,
            VideoEncoderPreferenceSettings.QsvKey,
            VideoEncoderPreferenceSettings.AmfKey,
            VideoEncoderPreferenceSettings.VideoToolboxKey,
            VideoEncoderPreferenceSettings.VaapiKey);

        var presetOption = new Option<string?>("--preset")
        {
            Description = "Named preset to load pipeline settings from",
        };

        var inputDirOption = new Option<string?>("--input-dir")
        {
            Description = "Directory path for batch processing of all supported media files",
        };

        var inputGlobOption = new Option<string?>("--input-glob")
        {
            Description = "Glob pattern for batch processing (resolved relative to current working directory)",
        };

        var recursiveOption = new Option<bool>("--recursive")
        {
            Description = "Discover media files recursively in subdirectories (only with --input-dir)",
            DefaultValueFactory = _ => false,
        };

        var continueOnErrorOption = new Option<bool>("--continue-on-error")
        {
            Description = "Continue processing remaining files after a failure in batch mode",
            DefaultValueFactory = _ => false,
        };

        var ttsRubberbandStretchOption = new Option<bool?>("--tts-rubberband-stretch")
        {
            Description = "Force-enable Rubberband time-stretch when fitting dubbed TTS to source segment pace. Optional; overrides host TTS timing settings for this run.",
        };

        var noTtsRubberbandStretchOption = new Option<bool>("--no-tts-rubberband-stretch")
        {
            Description = "Force-disable Rubberband stretch (use ffmpeg atempo). Optional; overrides host TTS timing settings for this run.",
            DefaultValueFactory = _ => false,
        };

        var ttsRubberbandThresholdOption = new Option<double?>("--tts-rubberband-threshold")
        {
            Description = "Rubberband stretch mismatch threshold in 0..1 (fraction of source duration). Optional; overrides host settings when Rubberband stretch is on.",
        };

        var ttsAutoStretchMaxOverrunOption = new Option<double?>("--tts-auto-stretch-max-overrun")
        {
            Description = "Maximum take overhang (fraction of source duration, 0..1) that auto time-stretch is allowed to fix. Takes overrunning by more are flagged for manual stretching instead. Optional; overrides host TTS timing settings for this run.",
        };

        var command = new Command("dub", """
            Execute a full dubbing pipeline from media ingest through export.

            Examples:
              trackdub dub --media ./video.mp4 --target-language es
              trackdub dub --media ./video.mp4 --target-language fr --export-format mkv
              trackdub dub --media ./video.mp4 --target-language de --model asr:whisper-small --model tts:kokoro-onnx
              trackdub dub --media ./video.mp4 --target-language en --voice-clone
              trackdub dub --media ./video.mp4 --target-language en --voice-clone --tts-rubberband-stretch --tts-rubberband-threshold 0.2
              trackdub dub --preset my-preset --input-dir ./videos
              trackdub dub --preset my-preset --input-glob "**/*.mp4"
            """)
        {
            mediaOption,
            targetLanguageOption,
            sourceLanguageOption,
            outputOption,
            modelOption,
            exportFormatOption,
            enableAsrTextRefinementOption,
            voiceCloneOption,
            timbrePolishOption,
            noTimbrePolishOption,
            restorePanOption,
            matchLoudnessOption,
            voiceOption,
            subtitleFormatOption,
            subtitleSourceOption,
            burnInSubtitlesOption,
            videoEncoderOption,
            presetOption,
            inputDirOption,
            inputGlobOption,
            recursiveOption,
            continueOnErrorOption,
            ttsRubberbandStretchOption,
            noTtsRubberbandStretchOption,
            ttsRubberbandThresholdOption,
            ttsAutoStretchMaxOverrunOption,
        };

        command.SetAction(async (ParseResult parseResult, CancellationToken cancellationToken) =>
        {
            string? mediaPath = UserPathText.NormalizeOptional(parseResult.GetValue(mediaOption));
            string? targetLanguage = parseResult.GetValue(targetLanguageOption);
            string? sourceLanguage = parseResult.GetValue(sourceLanguageOption);
            string? outputDirectory = UserPathText.NormalizeOptional(parseResult.GetValue(outputOption));
            string[] modelOverrides = parseResult.GetValue(modelOption) ?? [];
            string? exportFormat = parseResult.GetValue(exportFormatOption);
            bool? enableAsrTextRefinement = parseResult.GetValue(enableAsrTextRefinementOption);
            bool voiceClone = parseResult.GetValue(voiceCloneOption);
            bool timbrePolish = !parseResult.GetValue(noTimbrePolishOption) && (parseResult.GetValue(timbrePolishOption) ?? true);
            bool restorePan = parseResult.GetValue(restorePanOption) ?? false;
            bool matchLoudness = parseResult.GetValue(matchLoudnessOption) ?? false;
            string[] voiceOverrideTokens = parseResult.GetValue(voiceOption) ?? [];
            string[] subtitleFormatTokens = parseResult.GetValue(subtitleFormatOption) ?? [];
            string? subtitleSource = parseResult.GetValue(subtitleSourceOption);
            bool burnInSubtitles = parseResult.GetValue(burnInSubtitlesOption);
            string? videoEncoderKey = parseResult.GetValue(videoEncoderOption);
            string? presetName = parseResult.GetValue(presetOption);
            string? inputDir = UserPathText.NormalizeOptional(parseResult.GetValue(inputDirOption));
            string? inputGlob = parseResult.GetValue(inputGlobOption);
            bool recursive = parseResult.GetValue(recursiveOption);
            bool continueOnError = parseResult.GetValue(continueOnErrorOption);
            bool? ttsRubberbandStretch = parseResult.GetValue(ttsRubberbandStretchOption);
            bool noTtsRubberbandStretch = parseResult.GetValue(noTtsRubberbandStretchOption);
            double? ttsRubberbandThreshold = parseResult.GetValue(ttsRubberbandThresholdOption);
            double? ttsAutoStretchMaxOverrun = parseResult.GetValue(ttsAutoStretchMaxOverrunOption);

            Dictionary<string, string>? voiceOverrides = CliModelOverrides.ParseVoiceOverrides(voiceOverrideTokens);
            if (voiceOverrides is null)
            {
                return Program.ExitArgumentError;
            }

            IReadOnlyList<string>? subtitleFormats = subtitleFormatTokens.Length == 0
                ? null
                : subtitleFormatTokens.Any(f => f.Equals("none", StringComparison.OrdinalIgnoreCase))
                    ? (IReadOnlyList<string>)[]
                    : subtitleFormatTokens;

            bool isBatchMode = inputDir is not null || inputGlob is not null;

            if (!CliBatchCommandHelpers.TryValidateBatchInputOptions(
                    mediaPath,
                    inputDir,
                    inputGlob,
                    recursive,
                    out int batchValidationExitCode))
            {
                return batchValidationExitCode;
            }

            // --- Preset resolution ---

            (PipelinePreset? preset, int presetExitCode) = await CliBatchCommandHelpers.TryLoadPresetAsync(
                presetName,
                parseResult,
                cancellationToken).ConfigureAwait(false);
            if (presetExitCode != Program.ExitSuccess)
            {
                return presetExitCode;
            }

            // --- Merge preset values with explicit flags (explicit > preset > default) ---

            if (preset is not null)
            {
                CliBatchCommandHelpers.ApplyPresetPipelineDefaults(
                    preset,
                    ref targetLanguage,
                    ref sourceLanguage,
                    ref exportFormat,
                    ref enableAsrTextRefinement);
                modelOverrides = CliBatchCommandHelpers.ResolveModelOverrides(modelOverrides, preset);
            }

            // Resolve execution provider / device policy (explicit CLI > preset > default)
            CliParseHelpers.ResolvePresetExecutionPreferences(
                parseResult, preset, out string? resolvedExecutionProvider, out string? resolvedDevicePolicy);
            string? modelDirectory = CliParseHelpers.GetGlobalOptionValue<string?>(parseResult, "model-directory");

            // --- Batch processing path ---

            if (isBatchMode)
            {
                // In batch mode, target-language is required (from preset or explicit)
                if (string.IsNullOrWhiteSpace(targetLanguage))
                {
                    CliErrorReporter.ReportValidationError(
                        ErrorCode.InvalidArgument,
                        "Option '--target-language' is required for batch processing. Provide it explicitly or via a preset.",
                        "--target-language");
                    return Program.ExitArgumentError;
                }

                // Discover files
                if (!CliBatchCommandHelpers.TryDiscoverBatchMediaFiles(
                        inputDir,
                        inputGlob,
                        recursive,
                        out IReadOnlyList<string> mediaFiles,
                        out int discoveryExitCode))
                {
                    return discoveryExitCode;
                }

                // Parse model overrides
                Dictionary<string, string>? modelPreferences = CliModelOverrides.Parse(modelOverrides);
                if (modelPreferences is null)
                {
                    return Program.ExitArgumentError;
                }

                // Build template DubbingSessionOptions (SourceMediaPath is a placeholder; BatchProcessor overrides per-file)
                var templateOptions = PipelineOptionBuilder.BuildBatchSessionOptions(
                    targetLanguage,
                    sourceLanguage,
                    outputDirectory,
                    modelPreferences,
                    exportFormat,
                    enableAsrTextRefinement ?? false,
                    voiceClone,
                    timbrePolish,
                    restorePan,
                    matchLoudness,
                    voiceOverrides,
                    subtitleFormats,
                    subtitleSource,
                    burnInSubtitles,
                    videoEncoderKey,
                    ttsRubberbandStretch: ttsRubberbandStretch,
                    noTtsRubberbandStretch: noTtsRubberbandStretch,
                    ttsRubberbandThreshold: ttsRubberbandThreshold,
                    ttsAutoStretchMaxOverrun: ttsAutoStretchMaxOverrun);

                // Build BatchOptions
                var batchOptions = new BatchOptions
                {
                    ContinueOnError = continueOnError,
                    OutputRoot = outputDirectory is not null ? Path.GetFullPath(outputDirectory) : null,
                };

                // Build factory for batch execution
                TrackdubSessionFactory? factory = CliParseHelpers.TryBuildFactory(
                    parseResult, modelDirectory, resolvedExecutionProvider, resolvedDevicePolicy, out int buildExitCode);
                if (factory is null)
                {
                    return buildExitCode;
                }

                string progressFormat = CliParseHelpers.GetGlobalOptionValue<string>(parseResult, "progress") ?? "text";

                using (factory)
                {
                    (TtsTimingSettings? resolvedTiming, int timingExitCode) = await CliTtsTimingResolver.ResolveAsync(
                        factory,
                        ttsRubberbandStretch,
                        noTtsRubberbandStretch,
                        ttsRubberbandThreshold,
                        ttsAutoStretchMaxOverrun,
                        autoStretchToLimit: false,
                        cancellationToken).ConfigureAwait(false);
                    if (timingExitCode != Program.ExitSuccess)
                    {
                        return timingExitCode;
                    }

                    templateOptions = templateOptions with { TtsTiming = resolvedTiming };

                    return await BatchHandler.ExecuteAsync(
                        factory,
                        mediaFiles,
                        templateOptions,
                        batchOptions,
                        presetName,
                        progressFormat,
                        Console.Out,
                        cancellationToken).ConfigureAwait(false);
                }
            }

            // --- Single-file path (existing behavior) ---

            var setupRequest = new DubSetupRequest(
                mediaPath,
                targetLanguage,
                sourceLanguage,
                outputDirectory,
                modelOverrides,
                exportFormat);

            if (DubSetupWizard.RequiresSetup(setupRequest))
            {
                if (!setupInteractive())
                {
                    ReportMissingSetupValues(setupRequest);
                    return Program.ExitArgumentError;
                }

                DubSetupRequest? completedSetup = await DubSetupWizard
                    .CompleteAsync(setupRequest, new SpectreDubSetupPromptAdapter(), cancellationToken)
                    .ConfigureAwait(false);

                if (completedSetup is null)
                {
                    CliErrorReporter.ReportError(ErrorCode.Cancelled, "Interactive setup was cancelled before required inputs were collected.");
                    return Program.ExitArgumentError;
                }

                setupRequest = completedSetup;
                mediaPath = setupRequest.MediaPath;
                targetLanguage = setupRequest.TargetLanguage;
                sourceLanguage = setupRequest.SourceLanguage;
                outputDirectory = setupRequest.OutputDirectory;
                modelOverrides = setupRequest.ModelOverrides;
                exportFormat = setupRequest.ExportFormat;
            }

            if (string.IsNullOrWhiteSpace(mediaPath) || string.IsNullOrWhiteSpace(targetLanguage))
            {
                ReportMissingSetupValues(setupRequest);
                return Program.ExitArgumentError;
            }

            string resolvedMediaPath = Path.GetFullPath(mediaPath);
            if (!File.Exists(resolvedMediaPath))
            {
                CliErrorReporter.ReportValidationError(
                    ErrorCode.MediaNotFound,
                    $"Media file not found: {resolvedMediaPath}",
                    "--media");
                return Program.ExitArgumentError;
            }

            string resolvedOutputDirectory = outputDirectory is not null
                ? Path.GetFullPath(outputDirectory)
                : Path.Join(
                    Path.GetDirectoryName(resolvedMediaPath) ?? ".",
                    Path.GetFileNameWithoutExtension(resolvedMediaPath) + ".trackdub");

            Dictionary<string, string>? singleModelPreferences = CliModelOverrides.Parse(modelOverrides);
            if (singleModelPreferences is null)
            {
                return Program.ExitArgumentError;
            }

            TrackdubSessionFactory? singleFactory = CliParseHelpers.TryBuildFactory(
                parseResult, modelDirectory, resolvedExecutionProvider, resolvedDevicePolicy, out int singleBuildExitCode);
            if (singleFactory is null)
            {
                return singleBuildExitCode;
            }

            string singleProgressFormat = CliParseHelpers.GetGlobalOptionValue<string>(parseResult, "progress") ?? "text";

            using (singleFactory)
            {
                return await CliProgressRunner.ExecuteAsync(
                    singleProgressFormat,
                    async (progress, ct) => await RunPipelineHandler.ExecuteAsync(
                        singleFactory,
                        PipelineOptionBuilder.BuildSingleFileRequest(
                            resolvedMediaPath,
                            resolvedOutputDirectory,
                            sourceLanguage,
                            targetLanguage,
                            singleModelPreferences,
                            exportFormat,
                            enableAsrTextRefinement ?? false,
                            voiceClone,
                            timbrePolish,
                            restorePan,
                            matchLoudness,
                            voiceOverrides,
                            subtitleFormats,
                            subtitleSource,
                            burnInSubtitles,
                            videoEncoderKey,
                            ttsRubberbandStretch: ttsRubberbandStretch,
                            noTtsRubberbandStretch: noTtsRubberbandStretch,
                            ttsRubberbandThreshold: ttsRubberbandThreshold,
                            ttsAutoStretchMaxOverrun: ttsAutoStretchMaxOverrun),
                        progress,
                        Console.Out,
                        ct).ConfigureAwait(false),
                    cancellationToken).ConfigureAwait(false);
            }
        });

        return command;
    }

    private static void ReportMissingSetupValues(DubSetupRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.MediaPath))
        {
            CliErrorReporter.ReportValidationError(
                ErrorCode.InvalidArgument,
                "Option '--media' is required. Run this command in an interactive terminal for guided setup, or pass --media explicitly.",
                "--media");
        }

        if (string.IsNullOrWhiteSpace(request.TargetLanguage))
        {
            CliErrorReporter.ReportValidationError(
                ErrorCode.InvalidArgument,
                "Option '--target-language' is required. Run this command in an interactive terminal for guided setup, or pass --target-language explicitly.",
                "--target-language");
        }
    }
}
