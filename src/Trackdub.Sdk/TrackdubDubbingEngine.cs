using Trackdub.Application.Dubbing;
using Trackdub.Contracts.Dubbing;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Tts;

namespace Trackdub.Sdk;

/// <summary>
/// Primary entry point for SDK consumers to execute the dubbing pipeline.
/// Thin wrapper over <see cref="DubbingPipelineEngine"/> in Trackdub.Application.
/// </summary>
public sealed class TrackdubDubbingEngine : IDubbingPipelineEngine, ITransientFaultReporting
{
    private readonly DubbingPipelineEngine _engine;

    /// <summary>
    /// Creates a new <see cref="TrackdubDubbingEngine"/> backed by the given session factory.
    /// DI overrides should be applied via <see cref="TrackdubBuilder.ConfigureServices"/> before Build().
    /// </summary>
    public TrackdubDubbingEngine(TrackdubSessionFactory sessionFactory)
    {
        ArgumentNullException.ThrowIfNull(sessionFactory);
        _engine = new DubbingPipelineEngine(sessionFactory);
    }

    /// <summary>
    /// Executes the dubbing pipeline according to the provided options.
    /// </summary>
    public Task<DubbingRunResult> ExecuteAsync(
        DubbingSessionOptions options,
        IProgress<PipelineProgressEvent>? progress = null,
        CancellationToken cancellationToken = default) =>
        _engine.ExecuteAsync(options, progress, cancellationToken);

    /// <summary>
    /// Forwards transient-fault telemetry from the inner <see cref="DubbingPipelineEngine"/>.
    /// SDK consumers (UI shell, telemetry monitors) see the same stream as the engine-level
    /// ITransientFaultReporting surface. See
    /// <c>docs/internal/pipeline-readiness-spec.md</c> section 4.4.
    /// </summary>
    public IAsyncEnumerable<PipelineTransientFault> TransientFaultsAsync(
        CancellationToken cancellationToken = default) =>
        _engine.TransientFaultsAsync(cancellationToken);

    // ── Test/internal helpers forwarded to Application engine ──────────────

    internal static string? NormalizeAsrSourceLanguageCode(string? sourceLanguageCode) =>
        DubbingPipelineEngine.NormalizeAsrSourceLanguageCode(sourceLanguageCode);

    internal static bool IsBenignSkipReasonCode(string? reasonCode) =>
        DubbingPipelineEngine.IsBenignSkipReasonCode(reasonCode);

    internal static bool ShouldRunPostLipSynthesisExport(
        IReadOnlyList<string> stagesToRun,
        IReadOnlyList<StageOutcome> outcomes) =>
        DubbingPipelineEngine.ShouldRunPostLipSynthesisExport(stagesToRun, outcomes);

    internal static DubbingRunStatus DetermineOverallStatus(List<StageOutcome> outcomes) =>
        DubbingPipelineEngine.DetermineOverallStatus(outcomes);

    internal static Dictionary<Guid, string>? BuildUnattendedFallbackVoiceIds(
        TranscriptProjectState state,
        string? targetLanguageCode,
        string? preferredTtsModelAlias = null) =>
        UnattendedVoiceSelection.BuildUnattendedFallbackVoiceIds(state, targetLanguageCode, preferredTtsModelAlias);

    internal static DubbingSessionOptions ApplyVoiceCloningDefaults(DubbingSessionOptions options) =>
        UnattendedVoiceSelection.ApplyVoiceCloningDefaults(options);

    internal static GenerateTtsForAllSpeakersRequest BuildUnattendedTtsRequest(
        TranscriptProjectState state,
        DubbingSessionOptions options,
        string? ttsModelAlias) =>
        UnattendedVoiceSelection.BuildUnattendedTtsRequest(state, options, ttsModelAlias);

    internal static ExportOutputContainer ResolveExportContainer(string? exportFormat) =>
        DubbingPipelineEngine.ResolveExportContainer(exportFormat);

    internal static string ResolveExportOutputPath(string projectRootPath, ExportOutputContainer container) =>
        DubbingPipelineEngine.ResolveExportOutputPath(projectRootPath, container);

    internal static IReadOnlyList<string>? ExtractSpeechEnhancementDegradations(
        TranscriptProjectState state) =>
        DubbingPipelineEngine.ExtractSpeechEnhancementDegradations(state);

    internal static void MergeRuntimeModelSelectionsIntoSnapshot(
        Dictionary<string, string> snapshot,
        RuntimeModelSelections selections,
        IModelAliasResolver? modelAliasResolver = null) =>
        DubbingExecutionSnapshot.MergeRuntimeModelSelectionsIntoSnapshot(snapshot, selections, modelAliasResolver);

    internal static (StageStatus Status, string? ReasonCode, IReadOnlyList<string>? Degradations)
        MapStageRunToSdkOutcome(StageRunRecord? stageRun) =>
        DubbingPipelineEngine.MapStageRunToSdkOutcome(stageRun);

    internal static bool ShouldSkipModelPreFlight(
        string stageName,
        IReadOnlyDictionary<string, string>? modelPreferences) =>
        DubbingPipelineEngine.ShouldSkipModelPreFlight(stageName, modelPreferences);

    internal static string[] ResolveStageOrder(IReadOnlyList<string>? stageFilter) =>
        DubbingPipelineEngine.ResolveStageOrder(stageFilter);

    internal static Dictionary<string, string> CaptureExecutionSnapshot(DubbingSessionOptions options) =>
        DubbingExecutionSnapshot.CaptureExecutionSnapshot(options);
}
