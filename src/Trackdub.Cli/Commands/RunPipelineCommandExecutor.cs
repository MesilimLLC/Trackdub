using System.CommandLine;
using System.CommandLine.Parsing;

using Trackdub.Cli.Handlers;
using Trackdub.Cli.Interactive;
using Trackdub.Contracts;
using Trackdub.Contracts.Dubbing;
using Trackdub.Sdk;

namespace Trackdub.Cli.Commands;

internal sealed class PipelineRunState
{
    public string? MediaPath;
    public string? TargetLanguage;
    public string? SourceLanguage;
    public string? OutputDirectory;
    public string[] ModelOverrides = [];
    public string? ExportFormat;
    public string? FromStage;
    public string[] OnlyStages = [];
    public bool LipSync;
    public bool RepairLips;
    public bool ForceRerun;
    public bool? EnableAsrTextRefinement;
    public bool VoiceClone;
    public bool TimbrePolish = true;
    public bool RestorePan;
    public bool MatchLoudness;
    public string[] VoiceOverrideTokens = [];
    public string[] SubtitleFormatTokens = [];
    public string? SubtitleSource;
    public bool BurnInSubtitles;
    public string? VideoEncoderKey;
    public string? PresetName;
    public string? InputDir;
    public string? InputGlob;
    public bool Recursive;
    public bool ContinueOnError;
    public string? ResultFile;
    public string? ProgressFile;
    public Dictionary<string, string>? VoiceOverrides;
    public IReadOnlyList<string>? SubtitleFormats;
    public bool IsBatchMode;
    public bool? TtsRubberbandStretch;
    public bool NoTtsRubberbandStretch;
    public double? TtsRubberbandThreshold;
    public double? TtsAutoStretchMaxOverrun;
}

internal sealed record BatchExecutionContext(
    PipelinePreset? Preset,
    string? ExecutionProvider,
    string? DevicePolicy,
    string? ModelDirectory);

internal static class RunPipelineCommandExecutor
{
    public static Task<int> ExecuteAsync(
        ParseResult parseResult,
        PipelineCommandOptions options,
        Func<bool> setupInteractive,
        CancellationToken cancellationToken)
    {
        PipelineRunState state = ParseInputs(parseResult, options);
        if (state.IsBatchMode && (state.ResultFile is not null || state.ProgressFile is not null))
        {
            CliErrorReporter.ReportValidationError(ErrorCode.InvalidArgument, "Machine files require a single pipeline run.", "--result-file");
            return Task.FromResult(Program.ExitArgumentError);
        }
        if (!TryValidateInputs(state, out int validationExitCode))
        {
            return Task.FromResult(validationExitCode);
        }

        return state.IsBatchMode
            ? ExecuteBatchAsync(parseResult, state, cancellationToken)
            : ExecuteSingleAsync(parseResult, state, setupInteractive, cancellationToken);
    }

    private static PipelineRunState ParseInputs(ParseResult parseResult, PipelineCommandOptions options)
    {
        var state = new PipelineRunState();
        ParseMediaInputs(parseResult, options, state);
        ParseAudioInputs(parseResult, options, state);
        ParseBatchInputs(parseResult, options, state);
        state.VoiceOverrides = CliModelOverrides.ParseVoiceOverrides(state.VoiceOverrideTokens);
        state.SubtitleFormats = ResolveSubtitleFormats(state.SubtitleFormatTokens);
        state.IsBatchMode = state.InputDir is not null || state.InputGlob is not null;
        state.ResultFile = parseResult.GetValue(options.ResultFile);
        state.ProgressFile = parseResult.GetValue(options.ProgressFile);
        return state;
    }

    private static void ParseMediaInputs(ParseResult parseResult, PipelineCommandOptions options, PipelineRunState state)
    {
        state.MediaPath = UserPathText.NormalizeOptional(parseResult.GetValue(options.Media));
        state.TargetLanguage = parseResult.GetValue(options.TargetLanguage);
        state.SourceLanguage = parseResult.GetValue(options.SourceLanguage);
        state.OutputDirectory = UserPathText.NormalizeOptional(parseResult.GetValue(options.Output));
        state.ModelOverrides = parseResult.GetValue(options.Model) ?? [];
        state.ExportFormat = parseResult.GetValue(options.ExportFormat);
        state.FromStage = parseResult.GetValue(options.FromStage);
        state.OnlyStages = parseResult.GetValue(options.Only) ?? [];
        state.LipSync = parseResult.GetValue(options.LipSync);
        state.RepairLips = parseResult.GetValue(options.RepairLips);
        state.ForceRerun = parseResult.GetValue(options.ForceRerun);
        state.EnableAsrTextRefinement = parseResult.GetValue(options.EnableAsrTextRefinement);
        state.VoiceClone = parseResult.GetValue(options.VoiceClone);
    }

    private static void ParseAudioInputs(ParseResult parseResult, PipelineCommandOptions options, PipelineRunState state)
    {
        state.TimbrePolish = !parseResult.GetValue(options.NoTimbrePolish) && (parseResult.GetValue(options.TimbrePolish) ?? true);
        state.RestorePan = parseResult.GetValue(options.RestorePan) ?? false;
        state.MatchLoudness = parseResult.GetValue(options.MatchLoudness) ?? false;
        state.VoiceOverrideTokens = parseResult.GetValue(options.Voice) ?? [];
        state.SubtitleFormatTokens = parseResult.GetValue(options.SubtitleFormat) ?? [];
        state.SubtitleSource = parseResult.GetValue(options.SubtitleSource);
        state.BurnInSubtitles = parseResult.GetValue(options.BurnInSubtitles);
        state.VideoEncoderKey = parseResult.GetValue(options.VideoEncoder);
        state.PresetName = parseResult.GetValue(options.Preset);
        state.TtsRubberbandStretch = parseResult.GetValue(options.TtsRubberbandStretch);
        state.NoTtsRubberbandStretch = parseResult.GetValue(options.NoTtsRubberbandStretch);
        state.TtsRubberbandThreshold = parseResult.GetValue(options.TtsRubberbandThreshold);
        state.TtsAutoStretchMaxOverrun = parseResult.GetValue(options.TtsAutoStretchMaxOverrun);
    }

    private static void ParseBatchInputs(ParseResult parseResult, PipelineCommandOptions options, PipelineRunState state)
    {
        state.InputDir = UserPathText.NormalizeOptional(parseResult.GetValue(options.InputDir));
        state.InputGlob = parseResult.GetValue(options.InputGlob);
        state.Recursive = parseResult.GetValue(options.Recursive);
        state.ContinueOnError = parseResult.GetValue(options.ContinueOnError);
    }

    private static IReadOnlyList<string>? ResolveSubtitleFormats(string[] tokens)
    {
        if (tokens.Length == 0)
        {
            return null;
        }

        return tokens.Any(f => f.Equals("none", StringComparison.OrdinalIgnoreCase))
            ? (IReadOnlyList<string>)[]
            : tokens;
    }

    private static bool TryValidateInputs(PipelineRunState state, out int exitCode)
    {
        if (!CliBatchCommandHelpers.TryValidateBatchInputOptions(
                state.MediaPath, state.InputDir, state.InputGlob, state.Recursive, out exitCode))
        {
            return false;
        }

        if (!CliBatchCommandHelpers.TryValidatePresetName(state.PresetName, out exitCode))
        {
            return false;
        }

        if (state.VoiceOverrides is null)
        {
            exitCode = Program.ExitArgumentError;
            return false;
        }

        exitCode = Program.ExitSuccess;
        return true;
    }

    private static async Task<int> ExecuteBatchAsync(
        ParseResult parseResult,
        PipelineRunState state,
        CancellationToken cancellationToken)
    {
        TrackdubSessionFactory? presetFactory = CliParseHelpers.TryBuildFactoryForPresetLoad(parseResult, out int buildExitCode);
        if (presetFactory is null)
        {
            return buildExitCode;
        }

        BatchExecutionContext context;
        using (presetFactory)
        {
            (PipelinePreset? preset, int loadExitCode) = await LoadBatchPresetAsync(
                state.PresetName, presetFactory, cancellationToken).ConfigureAwait(false);
            if (loadExitCode != Program.ExitSuccess)
            {
                return loadExitCode;
            }

            CliParseHelpers.ResolvePresetExecutionPreferences(
                parseResult, preset, out string? provider, out string? policy);
            string? modelDirectory = CliParseHelpers.GetGlobalOptionValue<string?>(parseResult, "model-directory");
            context = new BatchExecutionContext(preset, provider, policy, modelDirectory);
        }

        TrackdubSessionFactory? execFactory = CliParseHelpers.TryBuildFactory(
            parseResult, context.ModelDirectory, context.ExecutionProvider, context.DevicePolicy, out int execExitCode);
        if (execFactory is null)
        {
            return execExitCode;
        }

        using (execFactory)
        {
            return await RunBatchWithFactoryAsync(parseResult, state, context, execFactory, cancellationToken)
                .ConfigureAwait(false);
        }
    }

    private static Task<(PipelinePreset? Preset, int ExitCode)> LoadBatchPresetAsync(
        string? presetName,
        TrackdubSessionFactory factory,
        CancellationToken cancellationToken)
    {
        if (presetName is null)
        {
            return Task.FromResult<(PipelinePreset?, int)>((null, Program.ExitSuccess));
        }

        if (PocPresets.TryResolve(presetName, out PipelinePreset? builtIn, out _, out _))
        {
            return Task.FromResult<(PipelinePreset?, int)>((builtIn, Program.ExitSuccess));
        }

        return CliBatchCommandHelpers.TryLoadPresetAsync(presetName, factory, cancellationToken);
    }

    private static async Task<int> RunBatchWithFactoryAsync(
        ParseResult parseResult,
        PipelineRunState state,
        BatchExecutionContext context,
        TrackdubSessionFactory factory,
        CancellationToken cancellationToken)
    {
        string? targetLanguage = state.TargetLanguage ?? context.Preset?.TargetLanguage;
        string? sourceLanguage = state.SourceLanguage ?? context.Preset?.SourceLanguage;
        string? exportFormat = state.ExportFormat ?? context.Preset?.ExportFormat;
        bool refinement = state.EnableAsrTextRefinement ?? context.Preset?.EnableAsrTextRefinement ?? false;
        string[] modelOverrides = CliBatchCommandHelpers.ResolveModelOverrides(state.ModelOverrides, context.Preset);

        Dictionary<string, string>? modelPreferences = CliModelOverrides.Parse(modelOverrides);
        if (modelPreferences is null)
        {
            return Program.ExitArgumentError;
        }

        if (string.IsNullOrWhiteSpace(targetLanguage))
        {
            CliErrorReporter.ReportValidationError(
                ErrorCode.InvalidArgument,
                "Option '--target-language' is required for batch processing. Provide it explicitly or via a preset.",
                "--target-language");
            return Program.ExitArgumentError;
        }

        if (!CliBatchCommandHelpers.TryDiscoverBatchMediaFiles(
                state.InputDir, state.InputGlob, state.Recursive,
                out IReadOnlyList<string> mediaFiles, out int discoveryExitCode))
        {
            return discoveryExitCode;
        }

        IReadOnlyList<string>? stageFilter = CliStageFilter.Build(state.FromStage, state.OnlyStages, state.LipSync, state.RepairLips);
        if (stageFilter is { Count: 0 })
        {
            return Program.ExitArgumentError;
        }

        return await ExecuteBatchFilesAsync(
            parseResult, state, factory, targetLanguage, sourceLanguage, exportFormat,
            refinement, modelPreferences, mediaFiles, stageFilter, cancellationToken)
            .ConfigureAwait(false);
    }

    private static async Task<int> ExecuteBatchFilesAsync(
        ParseResult parseResult,
        PipelineRunState state,
        TrackdubSessionFactory factory,
        string targetLanguage,
        string? sourceLanguage,
        string? exportFormat,
        bool refinement,
        Dictionary<string, string>? modelPreferences,
        IReadOnlyList<string> mediaFiles,
        IReadOnlyList<string>? stageFilter,
        CancellationToken cancellationToken)
    {
        var templateOptions = PipelineOptionBuilder.BuildBatchSessionOptions(
            targetLanguage, sourceLanguage, null, modelPreferences, exportFormat,
            refinement, state.VoiceClone, state.TimbrePolish, state.RestorePan,
            state.MatchLoudness, state.VoiceOverrides, state.SubtitleFormats,
            state.SubtitleSource, state.BurnInSubtitles, state.VideoEncoderKey,
            stageFilter, state.ForceRerun,
            state.TtsRubberbandStretch, state.NoTtsRubberbandStretch, state.TtsRubberbandThreshold,
            state.TtsAutoStretchMaxOverrun);
        if (PocPresets.TryResolve(
                state.PresetName,
                out _,
                out IReadOnlyDictionary<string, string>? presetProviders,
                out IReadOnlyDictionary<string, string>? presetVariants))
        {
            templateOptions = templateOptions with
            {
                ExecutionProviderPreferences = presetProviders,
                ModelVariantPreferences = presetVariants,
                RequireExecutionProviderPreferences = true,
            };
        }

        var batchOptions = new BatchOptions
        {
            ContinueOnError = state.ContinueOnError,
            OutputRoot = state.OutputDirectory is not null ? Path.GetFullPath(state.OutputDirectory) : null,
        };

        string progressFormat = CliParseHelpers.GetGlobalOptionValue<string>(parseResult, "progress") ?? "text";

        (TtsTimingSettings? resolvedTiming, int timingExitCode) = await CliTtsTimingResolver.ResolveAsync(
            factory,
            state.TtsRubberbandStretch,
            state.NoTtsRubberbandStretch,
            state.TtsRubberbandThreshold,
            state.TtsAutoStretchMaxOverrun,
            cancellationToken).ConfigureAwait(false);
        if (timingExitCode != Program.ExitSuccess)
        {
            return timingExitCode;
        }

        templateOptions = templateOptions with { TtsTiming = resolvedTiming };

        return await BatchHandler.ExecuteAsync(
            factory, mediaFiles, templateOptions, batchOptions,
            state.PresetName, progressFormat, Console.Out, cancellationToken).ConfigureAwait(false);
    }

    private static async Task<int> ExecuteSingleAsync(
        ParseResult parseResult,
        PipelineRunState state,
        Func<bool> setupInteractive,
        CancellationToken cancellationToken)
    {
        (bool applied, PipelinePreset? preset) = await TryApplySinglePresetAsync(state, parseResult, cancellationToken)
            .ConfigureAwait(false);
        if (!applied)
        {
            return Program.ExitArgumentError;
        }

        IReadOnlyList<string>? stageFilter = CliStageFilter.Build(state.FromStage, state.OnlyStages, state.LipSync, state.RepairLips);
        if (stageFilter is { Count: 0 })
        {
            return Program.ExitArgumentError;
        }

        if (!await TryCompleteSetupAsync(state, setupInteractive, cancellationToken).ConfigureAwait(false))
        {
            return Program.ExitArgumentError;
        }

        return await RunSinglePipelineAsync(parseResult, state, preset, stageFilter, cancellationToken)
            .ConfigureAwait(false);
    }

    private static async Task<(bool Applied, PipelinePreset? Preset)> TryApplySinglePresetAsync(
        PipelineRunState state,
        ParseResult parseResult,
        CancellationToken cancellationToken)
    {
        if (state.PresetName is null)
        {
            return (true, null);
        }

        (PipelinePreset? preset, int exitCode) = PocPresets.TryResolve(state.PresetName, out PipelinePreset? builtIn, out _, out _)
            ? (builtIn, Program.ExitSuccess)
            : await CliBatchCommandHelpers.TryLoadPresetAsync(state.PresetName, parseResult, cancellationToken).ConfigureAwait(false);
        if (exitCode != Program.ExitSuccess)
        {
            return (false, null);
        }

        if (preset is not null)
        {
            string? target = state.TargetLanguage;
            string? source = state.SourceLanguage;
            string? export = state.ExportFormat;
            bool? refinement = state.EnableAsrTextRefinement;
            CliBatchCommandHelpers.ApplyPresetPipelineDefaults(preset, ref target, ref source, ref export, ref refinement);
            state.TargetLanguage = target;
            state.SourceLanguage = source;
            state.ExportFormat = export;
            state.EnableAsrTextRefinement = refinement;
            state.ModelOverrides = CliBatchCommandHelpers.ResolveModelOverrides(state.ModelOverrides, preset);
        }

        return (true, preset);
    }

    private static async Task<bool> TryCompleteSetupAsync(
        PipelineRunState state,
        Func<bool> setupInteractive,
        CancellationToken cancellationToken)
    {
        bool requiresSourceMedia = await RequiresSourceMediaAsync(state).ConfigureAwait(false);
        string? resolvedOutput = ResolveOutputDirectory(state.OutputDirectory);
        bool projectResume = IsProjectResume(requiresSourceMedia, resolvedOutput);

        var setupRequest = new DubSetupRequest(
            state.MediaPath, state.TargetLanguage, state.SourceLanguage,
            state.OutputDirectory, state.ModelOverrides, state.ExportFormat);

        if (projectResume || !DubSetupWizard.RequiresSetup(setupRequest))
        {
            return true;
        }

        if (!setupInteractive())
        {
            ReportMissingSetupValues(setupRequest);
            return false;
        }

        DubSetupRequest? completed = await DubSetupWizard
            .CompleteAsync(setupRequest, new SpectreDubSetupPromptAdapter(), cancellationToken)
            .ConfigureAwait(false);

        if (completed is null)
        {
            CliErrorReporter.ReportError(ErrorCode.Cancelled, "Interactive setup was cancelled before required inputs were collected.");
            return false;
        }

        ApplySetupResult(state, completed);
        return true;
    }

    private static Task<bool> RequiresSourceMediaAsync(PipelineRunState state)
    {
        IReadOnlyList<string>? filter = CliStageFilter.Build(state.FromStage, state.OnlyStages, state.LipSync, state.RepairLips);
        bool requires = filter is null || filter.Any(TrackdubPipelineStages.RequiresSourceMedia);
        return Task.FromResult(requires);
    }

    private static string? ResolveOutputDirectory(string? outputDirectory)
    {
        return outputDirectory is not null ? Path.GetFullPath(outputDirectory) : null;
    }

    private static bool IsProjectResume(bool requiresSourceMedia, string? resolvedOutputDirectory)
    {
        return !requiresSourceMedia
            && resolvedOutputDirectory is not null
            && TrackdubProjectPaths.ContainsDatabase(resolvedOutputDirectory);
    }

    private static void ApplySetupResult(PipelineRunState state, DubSetupRequest completed)
    {
        state.MediaPath = completed.MediaPath;
        state.TargetLanguage = completed.TargetLanguage;
        state.SourceLanguage = completed.SourceLanguage;
        state.OutputDirectory = completed.OutputDirectory;
        state.ModelOverrides = completed.ModelOverrides;
        state.ExportFormat = completed.ExportFormat;
    }

    private static async Task<int> RunSinglePipelineAsync(
        ParseResult parseResult,
        PipelineRunState state,
        PipelinePreset? preset,
        IReadOnlyList<string>? stageFilter,
        CancellationToken cancellationToken)
    {
        Dictionary<string, string>? modelPreferences = CliModelOverrides.Parse(state.ModelOverrides);
        if (modelPreferences is null)
        {
            return Program.ExitArgumentError;
        }

        CliParseHelpers.ResolvePresetExecutionPreferences(
            parseResult, preset, out string? provider, out string? policy);
        string? modelDirectory = CliParseHelpers.GetGlobalOptionValue<string?>(parseResult, "model-directory");

        TrackdubSessionFactory? factory = CliParseHelpers.TryBuildFactory(
            parseResult, modelDirectory, provider, policy, out int buildExitCode);
        if (factory is null)
        {
            return buildExitCode;
        }

        using (factory)
        {
            return await RunSingleWithFactoryAsync(
                    parseResult, state, factory, modelPreferences, stageFilter, cancellationToken)
                .ConfigureAwait(false);
        }
    }

    private static async Task<int> RunSingleWithFactoryAsync(
        ParseResult parseResult,
        PipelineRunState state,
        TrackdubSessionFactory factory,
        Dictionary<string, string>? modelPreferences,
        IReadOnlyList<string>? stageFilter,
        CancellationToken cancellationToken)
    {
        string? resolvedOutputDirectory = ResolveOutputDirectory(state.OutputDirectory);
        TrackdubProjectContext? projectContext = await TryOpenProjectAsync(
            factory, resolvedOutputDirectory, cancellationToken).ConfigureAwait(false);

        bool requiresSourceMedia = stageFilter is null || stageFilter.Any(TrackdubPipelineStages.RequiresSourceMedia);
        bool projectResume = IsProjectResume(requiresSourceMedia, resolvedOutputDirectory);

        if (!TryResolveSingleMediaAndLanguage(state, projectContext, projectResume, out int resolveExitCode))
        {
            return resolveExitCode;
        }

        if (!TryResolveSinglePaths(state, projectContext, requiresSourceMedia, ref resolvedOutputDirectory, out string resolvedMediaPath, out resolveExitCode))
        {
            return resolveExitCode;
        }

        if (string.IsNullOrWhiteSpace(state.TargetLanguage))
        {
            CliErrorReporter.ReportValidationError(
                ErrorCode.InvalidArgument,
                "Option '--target-language' is required. Run this command in an interactive terminal for guided setup, or pass --target-language explicitly.",
                "--target-language");
            return Program.ExitArgumentError;
        }

        return await ExecuteSingleRequestAsync(
            parseResult, state, factory, modelPreferences,
            resolvedMediaPath, resolvedOutputDirectory!, stageFilter, cancellationToken)
            .ConfigureAwait(false);
    }

    private static Task<TrackdubProjectContext?> TryOpenProjectAsync(
        TrackdubSessionFactory factory,
        string? resolvedOutputDirectory,
        CancellationToken cancellationToken)
    {
        if (resolvedOutputDirectory is null || !TrackdubProjectPaths.ContainsDatabase(resolvedOutputDirectory))
        {
            return Task.FromResult<TrackdubProjectContext?>(null);
        }

        return TrackdubProjectContextResolver.TryOpenAsync(factory, resolvedOutputDirectory, cancellationToken);
    }

    private static bool TryResolveSingleMediaAndLanguage(
        PipelineRunState state,
        TrackdubProjectContext? projectContext,
        bool projectResume,
        out int exitCode)
    {
        exitCode = Program.ExitSuccess;
        if (projectResume)
        {
            if (projectContext is null)
            {
                CliErrorReporter.ReportValidationError(
                    ErrorCode.InvalidArgument,
                    $"Existing project not found or unreadable: {Path.GetFullPath(state.OutputDirectory!)}",
                    "--output");
                exitCode = Program.ExitArgumentError;
                return false;
            }

            if (string.IsNullOrWhiteSpace(state.MediaPath))
            {
                state.MediaPath = projectContext.SourceMediaPath ?? string.Empty;
            }

            if (string.IsNullOrWhiteSpace(state.TargetLanguage))
            {
                state.TargetLanguage = projectContext.TargetLanguageCode;
            }

            return true;
        }

        if (string.IsNullOrWhiteSpace(state.MediaPath) || string.IsNullOrWhiteSpace(state.TargetLanguage))
        {
            ReportMissingSetupValues(new DubSetupRequest(
                state.MediaPath, state.TargetLanguage, state.SourceLanguage,
                state.OutputDirectory, state.ModelOverrides, state.ExportFormat));
            exitCode = Program.ExitArgumentError;
            return false;
        }

        return true;
    }

    private static bool TryResolveSinglePaths(
        PipelineRunState state,
        TrackdubProjectContext? projectContext,
        bool requiresSourceMedia,
        ref string? resolvedOutputDirectory,
        out string resolvedMediaPath,
        out int exitCode)
    {
        resolvedMediaPath = string.IsNullOrWhiteSpace(state.MediaPath)
            ? string.Empty
            : Path.GetFullPath(state.MediaPath);

        if (requiresSourceMedia)
        {
            if (!File.Exists(resolvedMediaPath) && !string.IsNullOrWhiteSpace(projectContext?.SourceMediaPath))
            {
                resolvedMediaPath = Path.GetFullPath(projectContext.SourceMediaPath);
            }

            if (!File.Exists(resolvedMediaPath))
            {
                CliErrorReporter.ReportValidationError(
                    ErrorCode.MediaNotFound,
                    $"Media file not found: {resolvedMediaPath}",
                    "--media");
                exitCode = Program.ExitArgumentError;
                return false;
            }
        }

        if (resolvedOutputDirectory is null)
        {
            if (string.IsNullOrWhiteSpace(resolvedMediaPath))
            {
                CliErrorReporter.ReportValidationError(
                    ErrorCode.InvalidArgument,
                    "Option '--output' is required when resuming a project without source media.",
                    "--output");
                exitCode = Program.ExitArgumentError;
                return false;
            }

            resolvedOutputDirectory = Path.Join(
                Path.GetDirectoryName(resolvedMediaPath) ?? ".",
                Path.GetFileNameWithoutExtension(resolvedMediaPath) + ".trackdub");
        }

        exitCode = Program.ExitSuccess;
        return true;
    }

    private static Task<int> ExecuteSingleRequestAsync(
        ParseResult parseResult,
        PipelineRunState state,
        TrackdubSessionFactory factory,
        Dictionary<string, string>? modelPreferences,
        string resolvedMediaPath,
        string resolvedOutputDirectory,
        IReadOnlyList<string>? stageFilter,
        CancellationToken cancellationToken)
    {
        string progressFormat = CliParseHelpers.GetGlobalOptionValue<string>(parseResult, "progress") ?? "text";
        PocPresets.TryResolve(
            state.PresetName,
            out _,
            out IReadOnlyDictionary<string, string>? presetProviders,
            out IReadOnlyDictionary<string, string>? presetVariants);
        return CliProgressRunner.ExecuteAsync(
            progressFormat,
            (progress, ct) => RunPipelineHandler.ExecuteAsync(
                factory,
                PipelineOptionBuilder.BuildSingleFileRequest(
                    resolvedMediaPath, resolvedOutputDirectory, state.SourceLanguage,
                    state.TargetLanguage!, modelPreferences, state.ExportFormat,
                    state.EnableAsrTextRefinement ?? false, state.VoiceClone,
                    state.TimbrePolish, state.RestorePan, state.MatchLoudness,
                    state.VoiceOverrides, state.SubtitleFormats, state.SubtitleSource,
                    state.BurnInSubtitles, state.VideoEncoderKey, stageFilter, state.ForceRerun,
                    state.TtsRubberbandStretch, state.NoTtsRubberbandStretch, state.TtsRubberbandThreshold,
                    state.TtsAutoStretchMaxOverrun) with
                {
                    ResultFile = state.ResultFile,
                    ProgressFile = state.ProgressFile,
                    ExecutionProviderPreferences = presetProviders,
                    ModelVariantPreferences = presetVariants,
                },
                progress, Console.Out, ct),
            cancellationToken);
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
