using System.Globalization;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.Extensions.DependencyInjection;
using Trackdub.Application.LipSync;
using Trackdub.Application.LipSynthesis;
using Trackdub.Application.Pipeline;
using Trackdub.Application.Transcripts;
using Trackdub.Application.Transcripts.Pipeline;
using Trackdub.Contracts;
using Trackdub.Contracts.Dubbing;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Contracts.Licensing;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.Transcripts;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Pipeline;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Tts;

namespace Trackdub.Application.Dubbing;

public sealed partial class DubbingPipelineEngine
{
    /// <summary>Inputs the artifact-resume gate reads for a stage.</summary>
    private sealed record ArtifactResumeContext(
        IDubbingSession Session,
        DubbingSessionOptions Options,
        IReadOnlyDictionary<string, string> ExecutionSnapshot);

    internal static ExportOutputContainer ResolveExportContainer(string? exportFormat) =>
        ExportStageRequestBuilder.ResolveContainer(exportFormat);

    internal static string ResolveExportOutputPath(string projectRootPath, ExportOutputContainer container)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(projectRootPath);
        string extension = container == ExportOutputContainer.Mkv ? ".mkv" : ".mp4";
        return Path.Join(projectRootPath, "exports", "dubbed" + extension);
    }

    private static readonly IReadOnlyDictionary<string, ExportSubtitleFormat> SubtitleFormatByName =
        new Dictionary<string, ExportSubtitleFormat>(StringComparer.OrdinalIgnoreCase)
        {
            ["srt"] = ExportSubtitleFormat.Srt,
            ["vtt"] = ExportSubtitleFormat.Vtt,
            ["ass"] = ExportSubtitleFormat.Ass,
        };

    private static IReadOnlyList<ExportSubtitleFormat> ResolveSubtitleFormats(
        IReadOnlyList<string>? formats, bool hasTranscriptSegments)
    {
        if (formats is null)
            return hasTranscriptSegments ? [ExportSubtitleFormat.Srt] : [];
        var result = new List<ExportSubtitleFormat>(formats.Count);
        foreach (string? f in formats)
        {
            if (!string.IsNullOrWhiteSpace(f) && SubtitleFormatByName.TryGetValue(f, out ExportSubtitleFormat format))
            {
                result.Add(format);
            }
        }
        return result;
    }

    internal static ExportSubtitleSource ResolveSubtitleSource(string? source) =>
        source?.Trim().ToLowerInvariant() switch
        {
            "transcript" => ExportSubtitleSource.Transcript,
            "bilingual" => ExportSubtitleSource.Bilingual,
            _ => ExportSubtitleSource.Translated,
        };

    /// <summary>
    /// Determines the ordered list of stages to execute, respecting an optional filter.
    /// Unfiltered runs use the speech/export default order. Filtered runs resolve against
    /// the extended stage order so lip-sync / lip-synthesis can be opted in.
    /// </summary>
    internal static string[] ResolveStageOrder(IReadOnlyList<string>? stageFilter)
    {
        if (stageFilter is null || stageFilter.Count == 0)
        {
            return [.. DubbingPipelineStages.DefaultStageOrder];
        }

        HashSet<string> filterSet = new(stageFilter, StringComparer.OrdinalIgnoreCase);
        return DubbingPipelineStages.ExtendedStageOrder.Where(s => filterSet.Contains(s)).ToArray();
    }

    /// <summary>
    /// Checks whether a stage has valid existing artifacts from a prior run
    /// that match the current execution snapshot.
    /// </summary>
    private static async Task<bool> HasValidExistingArtifactsAsync(
        ArtifactResumeContext context,
        string stageName,
        CancellationToken cancellationToken)
    {
        IArtifactStore? artifactStore = context.Session.Services.GetService<IArtifactStore>();
        if (artifactStore is null)
        {
            return false;
        }

        TranscriptProjectState? state = await TryOpenProjectStateAsync(context.Session, cancellationToken)
            .ConfigureAwait(false);
        if (state is null)
        {
            return false;
        }

        return await StageArtifactResumeEvaluator.CanResumeStageAsync(
            state,
            artifactStore,
            stageName,
            context.ExecutionSnapshot,
            context.Session.ProjectRootPath,
            context.Options.TargetLanguageCode,
            ResolveExportResumePath(context, stageName),
            cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Opens the project for the artifact-resume gate. Any miss that means "no usable
    /// prior state" (I/O, access, missing project, schema-mismatch SQLite) is a clean
    /// non-resume, not a failure.
    /// </summary>
    private static async Task<TranscriptProjectState?> TryOpenProjectStateAsync(
        IDubbingSession session,
        CancellationToken cancellationToken)
    {
        try
        {
            return await session.Workspace.Project.OpenAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (IsArtifactResumeSafeMiss(ex))
        {
            return null;
        }
    }

    private static bool IsArtifactResumeSafeMiss(Exception ex) =>
        ex is IOException
            or UnauthorizedAccessException
            or System.Security.SecurityException
            or InvalidOperationException
        || string.Equals(
            ex.GetType().FullName,
            "Microsoft.Data.Sqlite.SqliteException",
            StringComparison.Ordinal);

    /// <summary>
    /// The resume check must look at the path the next run would actually write to:
    /// an explicit ExportOutputPath takes precedence over the project-default exports/
    /// location, and only that effective path proves the prior export is still usable.
    /// Path.GetFullPath mirrors ExportStageHandler's normalization (relative explicit
    /// paths resolve against the current directory, same as at execution time).
    /// </summary>
    private static string? ResolveExportResumePath(ArtifactResumeContext context, string stageName)
    {
        if (!string.Equals(stageName, StageNames.Export, StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        string effectiveOutputPath = string.IsNullOrWhiteSpace(context.Options.ExportOutputPath)
            ? ResolveExportOutputPath(
                context.Session.ProjectRootPath,
                ResolveExportContainer(context.Options.ExportFormat))
            : context.Options.ExportOutputPath;
        try
        {
            return Path.GetFullPath(effectiveOutputPath);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
        {
            return effectiveOutputPath;
        }
    }

    /// <summary>
    /// Skip reason codes that represent intentional resume/prerequisite gating rather than
    /// a failed attempt to run a requested stage.
    /// </summary>
    internal static bool IsBenignSkipReasonCode(string? reasonCode) =>
        StageSkipReasonCodes.IsBenignSkip(reasonCode);

    internal static bool ShouldRunPostLipSynthesisExport(
        IReadOnlyList<string> stagesToRun,
        IReadOnlyList<StageOutcome> outcomes)
    {
        if (!stagesToRun.Any(static stage =>
                string.Equals(stage, StageNames.LipSynthesis, StringComparison.OrdinalIgnoreCase)))
        {
            return false;
        }

        StageOutcome? lipSynthesisOutcome = outcomes.LastOrDefault(static outcome =>
            string.Equals(outcome.StageName, StageNames.LipSynthesis, StringComparison.OrdinalIgnoreCase));

        return lipSynthesisOutcome?.Status
            is StageStatus.Succeeded or StageStatus.PartiallySucceeded;
    }

    private static async Task<(bool IsLicenseApproved, bool AllowExperimentalExecution)>
        ResolveLipSynthesisExecutionGatesAsync(
            IDubbingSession session,
            string? preferredModelAlias,
            CancellationToken cancellationToken)
    {
        IModelInventoryService? inventoryService = TryResolveService<IModelInventoryService>(session);
        if (inventoryService is null)
        {
            return (false, false);
        }

        IReadOnlyList<ModelInventoryEntry> inventory =
            await inventoryService.GetAllAsync(cancellationToken).ConfigureAwait(false);
        ModelInventoryEntry? entry =
            LipSynthesisInventoryGate.ResolveEntry(inventory, preferredModelAlias);

        return (
            LipSynthesisInventoryGate.IsLicenseApproved(entry),
            LipSynthesisInventoryGate.AllowExperimentalExecution(entry));
    }

    /// <summary>
    /// Determines the overall run status from individual stage outcomes.
    /// </summary>
    internal static DubbingRunStatus DetermineOverallStatus(List<StageOutcome> outcomes)
    {
        if (outcomes.Count == 0)
        {
            return DubbingRunStatus.Succeeded;
        }

        // A failed (or non-benignly skipped, e.g. warmup-failed) prerequisite stage
        // blocks every subsequent stage in the runner, so no
        // deliverable can exist. Report Failed even when earlier stages succeeded:
        // PartialSuccess would let callers count a deliverable-free run as success (#327).
        if (HasBlockingPrerequisite(outcomes))
        {
            return DubbingRunStatus.Failed;
        }

        if (AllSucceeded(outcomes))
        {
            return DubbingRunStatus.Succeeded;
        }

        bool anySucceededOrPartial = HasAnySuccessOrPartial(outcomes);
        if (HasAnyFailed(outcomes))
        {
            return PartialOrFailed(anySucceededOrPartial);
        }

        if (HasNonBenignSkip(outcomes))
        {
            return PartialOrFailed(anySucceededOrPartial);
        }

        // A stage that completed with degraded output makes the run a partial success:
        // the artifacts exist, but the result is not what a clean run would produce.
        return HasAnyPartial(outcomes)
            ? DubbingRunStatus.PartialSuccess
            : DubbingRunStatus.Succeeded;
    }

    private static DubbingRunStatus PartialOrFailed(bool anySucceededOrPartial) =>
        anySucceededOrPartial ? DubbingRunStatus.PartialSuccess : DubbingRunStatus.Failed;

    private static bool HasBlockingPrerequisite(List<StageOutcome> outcomes) =>
        outcomes.Any(static o => IsBlockingPrerequisiteOutcome(o));

    private static bool AllSucceeded(List<StageOutcome> outcomes) =>
        outcomes.All(static o => o.Status == StageStatus.Succeeded);

    private static bool HasAnySuccessOrPartial(List<StageOutcome> outcomes) =>
        outcomes.Any(static o => o.Status is StageStatus.Succeeded or StageStatus.PartiallySucceeded);

    private static bool HasAnyFailed(List<StageOutcome> outcomes) =>
        outcomes.Any(static o => o.Status == StageStatus.Failed);

    private static bool HasNonBenignSkip(List<StageOutcome> outcomes) =>
        outcomes.Any(IsNonBenignSkip);

    private static bool HasAnyPartial(List<StageOutcome> outcomes) =>
        outcomes.Any(static o => o.Status == StageStatus.PartiallySucceeded);

    private static bool IsNonBenignSkip(StageOutcome outcome) =>
        outcome.Status == StageStatus.Skipped
        && !IsBenignSkipReasonCode(outcome.ReasonCode);

    /// <summary>
    /// Reports a progress event to the optional progress reporter.
    /// </summary>
    private static void ReportProgress(
        IProgress<PipelineProgressEvent>? progress,
        StageProgressEvent stageEvent)
    {
        progress?.Report(new PipelineProgressEvent(
            StageName: stageEvent.StageName,
            EventKind: stageEvent.Kind,
            Percentage: stageEvent.Kind == PipelineProgressEventKind.Completed ? 100 : 0,
            Message: stageEvent.Message,
            ElapsedDuration: stageEvent.Elapsed));
    }

    /// <summary>
    /// Builds a <see cref="StageOutcome"/> for a skipped stage.
    /// </summary>
    private static StageOutcome BuildSkippedOutcome(string stageName, string reasonCode)
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        return new StageOutcome
        {
            StageName = stageName,
            Status = StageStatus.Skipped,
            StartTime = now,
            EndTime = now,
            ArtifactPaths = [],
            ReasonCode = reasonCode,
        };
    }

    /// <summary>
    /// Builds a terminal <see cref="DubbingRunResult"/> for validation or pre-flight failures.
    /// </summary>
    private static DubbingRunResult BuildErrorResult(
        RunScope scope,
        DubbingRunStatus status,
        IReadOnlyList<string>? preFlightFailures = null)
    {
        return new DubbingRunResult
        {
            RunId = scope.RunId,
            CorrelationId = scope.RunId,
            StartTime = scope.RunStart,
            EndTime = DateTimeOffset.UtcNow,
            OverallStatus = status,
            StageOutcomes = scope.StageOutcomes.AsReadOnly(),
            PreFlightFailures = preFlightFailures,
            ExecutionSnapshot = scope.ExecutionSnapshot,
        };
    }

}
