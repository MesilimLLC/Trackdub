using System.Globalization;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts;
using Trackdub.Contracts.Dubbing;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Translation;

namespace Trackdub.Application.Dubbing;

/// <summary>
/// The per-run execution snapshot: an immutable record of provider/model/voice/export decisions
/// that artifact resume compares against, so a cached stage is reused only when it was produced
/// under the same decisions.
/// </summary>
internal static class DubbingExecutionSnapshot
{
    private static readonly (string StageName, RuntimeStage Stage, Func<RuntimeModelSelections, string?> ModelAlias)[]
        ModelSnapshotStages =
        [
            (StageNames.Asr, RuntimeStage.Asr, static selections => selections.AsrModelAlias),
            (StageNames.Diarization, RuntimeStage.Diarization, static selections => selections.DiarizationModelAlias),
            (StageNames.Separation, RuntimeStage.Separation, static selections => selections.SeparationModelAlias),
            (StageNames.OverlapRescue, RuntimeStage.OverlapRescue, static selections => selections.OverlapRescueModelAlias),
            (StageNames.Translation, RuntimeStage.Translation, static selections => selections.TranslationModelAlias),
            (StageNames.Tts, RuntimeStage.Tts, static selections => selections.TtsModelAlias),
            (StageNames.TextRefinementAsr, RuntimeStage.TextRefinement, static selections => selections.TextRefinementModelAlias),
            (StageNames.LipSync, RuntimeStage.LipSync, static selections => selections.LipSyncModelAlias),
            (StageNames.LipSynthesis, RuntimeStage.LipSynthesis, static selections => selections.LipSynthesisModelAlias),
        ];

    private static readonly (string StageName, RuntimeStage Stage)[] ProviderSnapshotStages =
    [
        (StageNames.Vad, RuntimeStage.Vad),
        (StageNames.Asr, RuntimeStage.Asr),
        (StageNames.Diarization, RuntimeStage.Diarization),
        (StageNames.Separation, RuntimeStage.Separation),
        (StageNames.OverlapRescue, RuntimeStage.OverlapRescue),
        (StageNames.Translation, RuntimeStage.Translation),
        (StageNames.Tts, RuntimeStage.Tts),
        (StageNames.AudioPreparation, RuntimeStage.SpeechEnhancement),
        (StageNames.TextRefinementAsr, RuntimeStage.TextRefinement),
        (StageNames.LipSync, RuntimeStage.LipSync),
        (StageNames.LipSynthesis, RuntimeStage.LipSynthesis),
    ];

    /// <summary>
    /// Captures an immutable snapshot of provider/model/voice decisions at run start.
    /// </summary>
    internal static Dictionary<string, string> CaptureExecutionSnapshot(DubbingSessionOptions options)
    {
        var snapshot = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["SourceMediaPath"] = options.SourceMediaPath,
            ["TargetLanguageCode"] = options.TargetLanguageCode,
            ["ForceRerun"] = options.ForceRerun.ToString(),
            ["EnableAsrTextRefinement"] = options.EnableAsrTextRefinement.ToString(),
            ["EnableSpeechAudioPreparation"] = options.EnableSpeechAudioPreparation.ToString(),
            ["UseVoiceCloning"] = options.UseVoiceCloning.ToString(),
        };

        // All is the historical behavior, so omit it from the snapshot and keep existing
        // translation artifacts resumable after upgrading to the scope-aware contract.
        if (options.TranslationScope != TranslationRunScope.All)
        {
            snapshot["TranslationScope"] = options.TranslationScope.ToString();
        }

        AddTtsTimingEntries(snapshot, options.TtsTiming);

        // The project may not exist yet, so export gating starts from normalized audio;
        // UpdateExecutionSnapshotWithSourceAudioKind applies the actual kind at export time.
        AddExportResumeGatingEntries(snapshot, options, ArtifactKind.NormalizedAudio);

        if (options.SourceLanguageCode is not null)
        {
            snapshot["SourceLanguage"] = options.SourceLanguageCode;
        }

        AddEntries(snapshot, options.ModelPreferences?
            .Select(static pair => ($"Model:{pair.Key}", pair.Value)));
        AddEntries(snapshot, options.VoiceCloneBySpeakerId?
            .Select(static pair => ($"VoiceClone:{pair.Key:D}", pair.Value.ToString())));
        AddEntries(snapshot, options.VoiceAssignmentOverrides?
            .Select(static pair => ($"Voice:{pair.Key}", pair.Value)));

        return snapshot;
    }

    /// <summary>
    /// Merges resolved runtime model aliases (settings, starter packs, explicit preferences)
    /// into the execution snapshot so artifact resume checks compare against models that
    /// will actually execute.
    /// </summary>
    internal static void MergeRuntimeModelSelectionsIntoSnapshot(
        Dictionary<string, string> snapshot,
        RuntimeModelSelections selections,
        IModelAliasResolver? modelAliasResolver = null)
    {
        ArgumentNullException.ThrowIfNull(snapshot);
        ArgumentNullException.ThrowIfNull(selections);

        RuntimeModelRequestOptions options = RuntimeModelRequestFactory.CreateOptions(selections);

        foreach ((string stageName, RuntimeStage stage, Func<RuntimeModelSelections, string?> modelAlias) in
                 ModelSnapshotStages.Where(entry => IsModelStageActive(entry.Stage, selections)))
        {
            var entry = new ModelStageEntry(stageName, stage, modelAlias(selections));
            SetModelAlias(snapshot, entry);
            SetModelVariant(snapshot, options, entry);
            AddModelId(snapshot, entry, modelAliasResolver);
        }

        // Hardware/provider overrides change which execution provider a stage runs on; a
        // prior run's artifacts were produced under the provider recorded on its stage row,
        // so resume must compare them rather than reuse artifacts across provider changes.
        foreach ((string stageName, RuntimeStage stage) in ProviderSnapshotStages)
        {
            SetExecutionProvider(snapshot, options, stageName, stage);
        }
    }

    /// <summary>
    /// Re-applies the export resume gating once the export stage knows whether it mixes from the
    /// ambiance stem or the normalized audio.
    /// </summary>
    internal static void UpdateExecutionSnapshotWithSourceAudioKind(
        Dictionary<string, string> snapshot,
        DubbingSessionOptions options,
        ArtifactKind sourceAudioKind) =>
        AddExportResumeGatingEntries(snapshot, options, sourceAudioKind);

    private static bool IsModelStageActive(RuntimeStage stage, RuntimeModelSelections selections) =>
        stage != RuntimeStage.TextRefinement || selections.EnableAsrTextRefinement;

    private static void AddEntries(
        Dictionary<string, string> snapshot,
        IEnumerable<(string Key, string Value)>? entries)
    {
        foreach ((string key, string value) in entries ?? [])
        {
            snapshot[key] = value;
        }
    }

    private static void AddTtsTimingEntries(Dictionary<string, string> snapshot, TtsTimingSettings? ttsTiming)
    {
        if (ttsTiming is not null)
        {
            snapshot["TtsTiming.EnableRubberbandStretch"] = ttsTiming.EnableRubberbandStretch.ToString();
            snapshot["TtsTiming.RubberbandStretchThreshold"] =
                ttsTiming.RubberbandStretchThreshold.ToString("G17", CultureInfo.InvariantCulture);
            if (ttsTiming.AutoStretchMaxOverrun is double autoStretchMaxOverrun)
            {
                snapshot["TtsTiming.AutoStretchMaxOverrun"] =
                    autoStretchMaxOverrun.ToString("G17", CultureInfo.InvariantCulture);
            }
        }

        snapshot[TtsTimingResumeIdentity.SnapshotKey] =
            TtsTimingResumeIdentity.Normalize(ttsTiming?.AutoStretchMaxOverrun);
    }

    /// <summary>
    /// Audio/subtitle/encoder flags (and the pre-existing ExportFormat) gate the Export
    /// stage's artifact resume: any change here must invalidate a cached export so it reruns
    /// without requiring --force-rerun. These values are produced by ExportResumeGating so
    /// capture (here) and comparison (the persisted ExportManifest, read back by
    /// StageArtifactResumeEvaluator) share one normalization and cannot drift.
    /// </summary>
    private static void AddExportResumeGatingEntries(
        Dictionary<string, string> snapshot,
        DubbingSessionOptions options,
        ArtifactKind sourceAudioKind)
    {
        foreach ((string key, string value) in ExportResumeGating.Build(
            DubbingPipelineEngine.ResolveExportContainer(options.ExportFormat),
            sourceAudioKind,
            options.ApplyTimbrePolish,
            options.RestoreOriginalPan,
            options.MatchOriginalLoudness,
            options.BurnInSubtitles,
            DubbingPipelineEngine.ResolveSubtitleSource(options.SubtitleSource),
            ExportResumeGating.SubtitleFormatsTokenFromRawOptions(options.SubtitleFormats),
            options.VideoEncoder,
            options.ExportTargetLufs ?? ExportLoudnessTargets.OnlineLufs,
            options.ExportSourceGainDb ?? 0d,
            options.ExportDubbedSpeechGainDb ?? 0d,
            options.ExportDuckingGainDb))
        {
            snapshot[key] = value;
        }
    }

    private static void SetExecutionProvider(
        Dictionary<string, string> snapshot,
        RuntimeModelRequestOptions options,
        string stageName,
        RuntimeStage stage)
    {
        ExecutionProviderKind? preferredProvider =
            RuntimeModelRequestFactory.ResolvePreferredExecutionProvider(options, stage);
        if (preferredProvider is not null)
        {
            snapshot[$"Provider:{stageName}"] =
                RuntimeModelRequestFactory.FormatExecutionProviderLabel(preferredProvider.Value);
        }
    }

    /// <summary>One pipeline stage's resolved model alias, as recorded in the snapshot.</summary>
    private readonly record struct ModelStageEntry(string StageName, RuntimeStage Stage, string? ModelAlias);

    private static void AddModelId(
        Dictionary<string, string> snapshot,
        ModelStageEntry entry,
        IModelAliasResolver? modelAliasResolver)
    {
        if (modelAliasResolver is null || string.IsNullOrWhiteSpace(entry.ModelAlias))
        {
            return;
        }

        if (modelAliasResolver.TryResolveModelId(entry.ModelAlias, out string? modelId) &&
            !string.IsNullOrWhiteSpace(modelId))
        {
            snapshot[$"ModelId:{entry.StageName}"] = modelId;
        }
    }

    private static void SetModelAlias(Dictionary<string, string> snapshot, ModelStageEntry entry)
    {
        if (!string.IsNullOrWhiteSpace(entry.ModelAlias))
        {
            snapshot[$"Model:{entry.StageName}"] = entry.ModelAlias;
        }
    }

    private static void SetModelVariant(
        Dictionary<string, string> snapshot,
        RuntimeModelRequestOptions options,
        ModelStageEntry entry)
    {
        string? variant = RuntimeModelRequestFactory.ResolvePreferredModelVariantAlias(
            options,
            entry.Stage,
            entry.ModelAlias);
        if (!string.IsNullOrWhiteSpace(variant))
        {
            snapshot[$"ModelVariant:{entry.StageName}"] = variant;
        }
    }
}
