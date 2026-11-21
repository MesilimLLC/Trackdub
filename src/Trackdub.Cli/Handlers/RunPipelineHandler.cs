using System.Text.Json;

using Trackdub.Contracts;
using Trackdub.Contracts.Dubbing;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.Jobs;
using Trackdub.Domain.StageRuns;
using Trackdub.Sdk;

namespace Trackdub.Cli.Handlers;

/// <summary>
/// Shared full-pipeline execution for <c>dub</c> and <c>run pipeline</c>.
/// </summary>
internal static class RunPipelineHandler
{
    public static async Task<int> ExecuteAsync(
        TrackdubSessionFactory factory,
        RunPipelineRequest request,
        IProgress<PipelineProgressEvent>? progress,
        TextWriter output,
        CancellationToken cancellationToken)
    {
        if (request.ResultFile is null && request.ProgressFile is null)
            return await ExecuteCoreAsync(factory, request, progress, output, cancellationToken).ConfigureAwait(false);
        try
        {
            if (request.ResultFile is not null && File.Exists(request.ResultFile))
                throw new IOException("Result file already exists.");
            using var sink = request.ProgressFile is null ? null : new CliProgressFileSink(request.ProgressFile, progress);
            return await ExecuteCoreAsync(factory, request, sink ?? progress, output, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            await WriteOutcomeAsync(request, new(1, "cancelled", null, "CANCELLED", "Pipeline cancelled.", [])).ConfigureAwait(false);
            return Program.ExitCancelled;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            CliErrorReporter.ReportError(ErrorCode.StageFailed, $"Machine output could not be written: {ex.Message}");
            return Program.ExitPipelineFailure;
        }
    }

    private static async Task<int> ExecuteCoreAsync(
        TrackdubSessionFactory factory,
        RunPipelineRequest request,
        IProgress<PipelineProgressEvent>? progress,
        TextWriter output,
        CancellationToken cancellationToken)
    {
        var engine = new TrackdubDubbingEngine(factory);

        (TtsTimingSettings? ttsTiming, int timingExitCode) = await CliTtsTimingResolver.ResolveAsync(
            factory,
            request.TtsRubberbandStretch,
            request.NoTtsRubberbandStretch,
            request.TtsRubberbandThreshold,
            request.TtsAutoStretchMaxOverrun,
            request.TtsAutoStretchToLimit,
            cancellationToken).ConfigureAwait(false);
        if (timingExitCode != Program.ExitSuccess)
        {
            return timingExitCode;
        }

        string projectOutputDirectory = ResolveProjectDirectory(request);

        using ProjectLock? projectLock = CliProjectLock.TryAcquire(projectOutputDirectory);
        if (projectLock is null)
        {
            return Program.ExitPipelineFailure;
        }

        var dubbingOptions = new DubbingSessionOptions
        {
            SourceMediaPath = request.SourceMediaPath,
            ProjectOutputDirectory = request.ProjectOutputDirectory,
            SourceLanguageCode = request.SourceLanguageCode,
            TargetLanguageCode = request.TargetLanguageCode,
            ModelPreferences = request.ModelPreferences,
            ModelVariantPreferences = request.ModelVariantPreferences,
            ExecutionProviderPreferences = request.ExecutionProviderPreferences,
            RequireExecutionProviderPreferences = request.ExecutionProviderPreferences is { Count: > 0 },
            ExportFormat = request.ExportFormat,
            StageFilter = request.StageFilter,
            ForceRerun = request.ForceRerun,
            EnableAsrTextRefinement = request.EnableAsrTextRefinement,
            UseVoiceCloning = request.UseVoiceCloning,
            ApplyTimbrePolish = request.ApplyTimbrePolish,
            RestoreOriginalPan = request.RestoreOriginalPan,
            MatchOriginalLoudness = request.MatchOriginalLoudness,
            VoiceAssignmentOverrides = request.VoiceAssignmentOverrides,
            SubtitleFormats = request.SubtitleFormats,
            SubtitleSource = request.SubtitleSource,
            BurnInSubtitles = request.BurnInSubtitles,
            VideoEncoder = request.VideoEncoder,
            TtsTiming = ttsTiming,
        };

        DubbingRunResult result;
        try
        {
            result = await engine.ExecuteAsync(dubbingOptions, progress, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            await WriteOutcomeAsync(request, new(1, "cancelled", null, "CANCELLED", "Pipeline cancelled.", [])).ConfigureAwait(false);
            CliErrorReporter.ReportError(ErrorCode.Cancelled, "Pipeline execution was cancelled.");
            return Program.ExitCancelled;
        }

        var manifestWriter = new RunManifestWriter();
        await manifestWriter.WriteAsync(result, projectOutputDirectory, cancellationToken).ConfigureAwait(false);
        string manifestPath = Path.Join(projectOutputDirectory, "run-manifest.json");

        string? exportedFilePath = result.StageOutcomes
            .Where(o => string.Equals(o.StageName, "Export", StringComparison.OrdinalIgnoreCase))
            .Where(o => o.Status is StageStatus.Succeeded or StageStatus.PartiallySucceeded)
            .SelectMany(o => o.ArtifactPaths)
            .FirstOrDefault();

        if (result.OverallStatus == DubbingRunStatus.Succeeded
            || (result.OverallStatus == DubbingRunStatus.PartialSuccess
                && IsGoalAchieved(result, request.StageFilter, exportedFilePath)))
        {
            var artifacts = new List<WorkerArtifact>();
            if (request.ResultFile is not null)
            {
                if (exportedFilePath is not null)
                    artifacts.Add(await CliMachineResultWriter.DescribeAsync(projectOutputDirectory, exportedFilePath, "video", cancellationToken).ConfigureAwait(false));
                foreach (string subtitle in result.StageOutcomes.SelectMany(s => s.ArtifactPaths)
                    .Where(p => Path.GetExtension(p) is ".srt" or ".vtt").Distinct(StringComparer.Ordinal))
                    artifacts.Add(await CliMachineResultWriter.DescribeAsync(projectOutputDirectory, subtitle, "subtitle", cancellationToken).ConfigureAwait(false));
                await WriteOutcomeAsync(request, new(1, "succeeded", result, null, null, artifacts)).ConfigureAwait(false);
            }
            var payload = new RunPipelineOutput
            {
                ExportedFilePath = exportedFilePath,
                ManifestPath = manifestPath,
                Status = result.OverallStatus.ToString(),
            };

            string json = JsonSerializer.Serialize(payload, CliJsonOptions.Default);
            await output.WriteLineAsync(json).ConfigureAwait(false);
            return Program.ExitSuccess;
        }

        await WriteOutcomeAsync(request, new(1, "failed", result, "PIPELINE_FAILED", "Pipeline did not achieve its goal.", [])).ConfigureAwait(false);
        foreach (StageOutcome failedStage in result.StageOutcomes.Where(o => o.Status == StageStatus.Failed))
        {
            CliErrorReporter.ReportStageFailure(
                ErrorCode.StageFailed,
                failedStage.StageName,
                failedStage.ReasonCode ?? "Stage execution failed",
                failedStage.ArtifactPaths);
        }

        if (!result.StageOutcomes.Any(o => o.Status == StageStatus.Failed))
        {
            foreach (StageOutcome skippedStage in result.StageOutcomes.Where(o => o.Status == StageStatus.Skipped))
            {
                CliErrorReporter.ReportStageFailure(
                    ErrorCode.StagePrerequisiteMissing,
                    skippedStage.StageName,
                    skippedStage.ReasonCode ?? "Stage skipped",
                    artifactPaths: null);
            }
        }

        if (result.PreFlightFailures is { Count: > 0 })
        {
            foreach (string failure in result.PreFlightFailures)
            {
                CliErrorReporter.ReportError(ErrorCode.PreFlightFailed, failure);
            }
        }

        var failurePayload = new RunPipelineOutput
        {
            ManifestPath = manifestPath,
            Status = result.OverallStatus.ToString(),
        };

        string failureJson = JsonSerializer.Serialize(failurePayload, CliJsonOptions.Default);
        await output.WriteLineAsync(failureJson).ConfigureAwait(false);
        return Program.ExitPipelineFailure;
    }

    internal sealed record RunPipelineRequest
    {
        public string? ResultFile { get; init; }
        public string? ProgressFile { get; init; }
        public IReadOnlyDictionary<string, string>? ExecutionProviderPreferences { get; init; }
        public IReadOnlyDictionary<string, string>? ModelVariantPreferences { get; init; }
        public required string SourceMediaPath { get; init; }
        public string? ProjectOutputDirectory { get; init; }
        public string? SourceLanguageCode { get; init; }
        public required string TargetLanguageCode { get; init; }
        public IReadOnlyDictionary<string, string>? ModelPreferences { get; init; }
        public string? ExportFormat { get; init; }
        public IReadOnlyList<string>? StageFilter { get; init; }
        public bool ForceRerun { get; init; }
        public bool EnableAsrTextRefinement { get; init; }
        public bool UseVoiceCloning { get; init; }
        public bool ApplyTimbrePolish { get; init; } = true;
        public bool RestoreOriginalPan { get; init; }
        public bool MatchOriginalLoudness { get; init; }
        public IReadOnlyDictionary<string, string>? VoiceAssignmentOverrides { get; init; }
        public IReadOnlyList<string>? SubtitleFormats { get; init; }
        public string? SubtitleSource { get; init; }
        public bool BurnInSubtitles { get; init; }
        public VideoEncoderPreference VideoEncoder { get; init; }
        public bool? TtsRubberbandStretch { get; init; }
        public bool NoTtsRubberbandStretch { get; init; }
        public double? TtsRubberbandThreshold { get; init; }
        public double? TtsAutoStretchMaxOverrun { get; init; }
        public bool TtsAutoStretchToLimit { get; init; }
    }

    private static async Task WriteOutcomeAsync(RunPipelineRequest request, WorkerRunOutcome outcome)
    {
        if (request.ResultFile is null) return;
        using var cleanup = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        await CliMachineResultWriter.WriteAsync(outcome, request.ResultFile, cleanup.Token).ConfigureAwait(false);
    }

    private sealed class RunPipelineOutput
    {
        public string? ExportedFilePath { get; init; }
        public string? ManifestPath { get; init; }
        public string? Status { get; init; }
    }

    internal static string ResolveProjectDirectory(RunPipelineRequest request) =>
        TrackdubProjectPaths.ResolveProjectDirectory(
            request.SourceMediaPath,
            request.ProjectOutputDirectory);

    internal static bool IsGoalAchieved(
        DubbingRunResult result,
        IReadOnlyList<string>? stageFilter,
        string? exportedFilePath)
    {
        bool exportRequested = stageFilter is null
            || stageFilter.Any(s => string.Equals(s, "Export", StringComparison.OrdinalIgnoreCase));
        if (exportRequested && exportedFilePath is null)
        {
            return false;
        }

        return result.StageOutcomes
            .Where(o => stageFilter is null
                || stageFilter.Any(f => string.Equals(f, o.StageName, StringComparison.OrdinalIgnoreCase)))
            .All(o => o.Status is StageStatus.Succeeded or StageStatus.PartiallySucceeded
                || (o.Status == StageStatus.Skipped && StageSkipReasonCodes.IsBenignSkip(o.ReasonCode)));
    }
}
