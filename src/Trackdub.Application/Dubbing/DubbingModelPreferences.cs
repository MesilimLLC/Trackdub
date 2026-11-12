using Trackdub.Application.Transcripts;
using Trackdub.Contracts;
using Trackdub.Contracts.Dubbing;
using Trackdub.Domain;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Dubbing;

/// <summary>
/// Turns the per-run model, variant and execution-provider preferences in
/// <see cref="DubbingSessionOptions"/> into <see cref="InferenceModelPreferences"/>, and overlays them onto
/// host- or settings-derived <see cref="RuntimeModelSelections"/>.
/// </summary>
internal static class DubbingModelPreferences
{
    /// <summary>
    /// Overlays per-stage execution-provider pins from <paramref name="preferences"/> onto the
    /// resolved <paramref name="selections"/> hardware overrides. Pinned stages become required
    /// providers only when <see cref="DubbingSessionOptions.RequireExecutionProviderPreferences"/>
    /// is set; otherwise they act as preferences and existing host/settings overrides still apply.
    /// </summary>
    internal static RuntimeModelSelections ApplyExecutionProviderPins(
        RuntimeModelSelections selections,
        DubbingSessionOptions options,
        InferenceModelPreferences? preferences)
    {
        if (preferences?.PreferredExecutionProviders is not { Count: > 0 })
        {
            return selections;
        }

        // CreateSelectionsFromPreferences maps the stage-keyed provider pins onto the
        // HardwareOverrides key vocabulary (including the ASR engine disambiguation).
        IReadOnlyDictionary<string, ExecutionProviderKind> pinOverrides =
            RuntimeModelRequestFactory.CreateSelectionsFromPreferences(
                preferences,
                selections.AsrModelOverride,
                selections.IsDevBuild).HardwareOverrides;
        var overrides = new Dictionary<string, ExecutionProviderKind>(
            selections.HardwareOverrides,
            StringComparer.OrdinalIgnoreCase);
        foreach ((string key, ExecutionProviderKind provider) in pinOverrides)
        {
            overrides[key] = provider;
        }

        return selections with
        {
            HardwareOverrides = overrides,
            RequirePreferredExecutionProviders = selections.RequirePreferredExecutionProviders
                || options.RequireExecutionProviderPreferences,
        };
    }

    /// <summary>
    /// Overlays non-null per-stage model aliases from explicit
    /// <see cref="InferenceModelPreferences"/> onto a host-provided selection set.
    /// Fields the request did not pin keep the host's selection.
    /// </summary>
    internal static RuntimeModelSelections ApplyModelPreferenceAliases(
        RuntimeModelSelections selections,
        InferenceModelPreferences? preferences)
    {
        if (preferences is null)
        {
            return selections;
        }

        RuntimeModelSelections requested = RuntimeModelRequestFactory.CreateSelectionsFromPreferences(preferences);
        return selections with
        {
            SelectionIntents = MergeSelectionIntents(selections, requested),
            VadModelAlias = Prefer(preferences.VadModelAlias, selections.VadModelAlias),
            DiarizationModelAlias = Prefer(preferences.DiarizationModelAlias, selections.DiarizationModelAlias),
            SeparationModelAlias = Prefer(preferences.SeparationModelAlias, selections.SeparationModelAlias),
            OverlapRescueModelAlias = Prefer(preferences.OverlapRescueModelAlias, selections.OverlapRescueModelAlias),
            AsrModelAlias = Prefer(preferences.AsrModelAlias, selections.AsrModelAlias),
            TranslationModelAlias = Prefer(preferences.TranslationModelAlias, selections.TranslationModelAlias),
            TtsModelAlias = Prefer(preferences.TtsModelAlias, selections.TtsModelAlias),
            TextRefinementModelAlias = Prefer(preferences.TextRefinementModelAlias, selections.TextRefinementModelAlias),
            LipSyncModelAlias = Prefer(preferences.LipSyncModelAlias, selections.LipSyncModelAlias),
            LipSynthesisModelAlias = Prefer(preferences.LipSynthesisModelAlias, selections.LipSynthesisModelAlias),
            ModelVariantOverrides = MergeModelVariantOverrides(selections.ModelVariantOverrides, requested.ModelVariantOverrides),
        };
    }

    private static string? Prefer(string? requested, string? current) => requested ?? current;

    /// <summary>A stage the request pinned explicitly stays explicit; other stages keep the host's intent.</summary>
    private static IReadOnlyDictionary<RuntimeStage, RuntimeModelSelectionIntent> MergeSelectionIntents(
        RuntimeModelSelections selections,
        RuntimeModelSelections requested) =>
        Enum.GetValues<RuntimeStage>().ToDictionary(
            stage => stage,
            stage => requested.GetSelectionIntent(stage) == RuntimeModelSelectionIntent.Explicit
                ? RuntimeModelSelectionIntent.Explicit
                : selections.GetSelectionIntent(stage));

    /// <summary>
    /// Requested variant overrides replace the host's overrides for the same stage; the host's
    /// overrides for other stages are kept. Without requested overrides the host's set is unchanged.
    /// </summary>
    private static IReadOnlyDictionary<string, string>? MergeModelVariantOverrides(
        IReadOnlyDictionary<string, string>? current,
        IReadOnlyDictionary<string, string>? requested)
    {
        if (requested is not { Count: > 0 })
        {
            return current;
        }

        return new Dictionary<string, string>(current ?? new Dictionary<string, string>(), StringComparer.OrdinalIgnoreCase)
            .Where(pair => !ModelVariantOverrideKeys.TryParse(pair.Key, out string stageKey, out _)
                || !requested.ContainsKey(stageKey))
            .Concat(requested)
            .GroupBy(p => p.Key, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(g => g.Key, g => g.Last().Value, StringComparer.OrdinalIgnoreCase);
    }

    private static bool HasAnyModelPreference(DubbingSessionOptions options) =>
        options.ModelPreferences is { Count: > 0 } ||
        options.ExecutionProviderPreferences is { Count: > 0 } ||
        options.ModelVariantPreferences is { Count: > 0 } ||
        options.EnableAsrTextRefinement;

    private static Dictionary<RuntimeStage, string>? BuildVariantAliases(
        IReadOnlyDictionary<string, string>? variants)
    {
        if (variants is null)
        {
            return null;
        }

        var aliases = new Dictionary<RuntimeStage, string>();
        foreach ((string stageName, string alias) in variants)
        {
            RuntimeStage stage = DubbingPipelineEngine.MapStageNameToRuntimeStage(stageName)
                ?? throw new ArgumentException("Unknown variant stage.", nameof(variants));
            if (!aliases.TryAdd(stage, alias))
            {
                throw new ArgumentException(
                    $"Variant stages for runtime stage '{stage}' conflict: '{stageName}' duplicates an earlier entry.",
                    nameof(variants));
            }
        }

        return aliases;
    }

    /// <summary>
    /// Builds <see cref="InferenceModelPreferences"/> from the dubbing session options.
    /// Per-stage provider pins in <see cref="DubbingSessionOptions.ExecutionProviderPreferences"/>
    /// are mapped from canonical <see cref="StageNames"/> keys onto runtime stages and marked
    /// required; a stage name with no runtime mapping or an unparseable provider label is an
    /// argument error, never silently ignored.
    /// </summary>
    internal static InferenceModelPreferences? BuildModelPreferences(DubbingSessionOptions options)
    {
        if (!HasAnyModelPreference(options))
        {
            return null;
        }

        IReadOnlyDictionary<string, string> modelPreferences = options.ModelPreferences
            ?? new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        IReadOnlyDictionary<RuntimeStage, ExecutionProviderKind>? providerPins =
            ParseExecutionProviderPreferences(options);

        return new InferenceModelPreferences(
            VadModelAlias: modelPreferences.GetValueOrDefault(StageNames.Vad),
            AsrModelAlias: modelPreferences.GetValueOrDefault(StageNames.Asr),
            DiarizationModelAlias: modelPreferences.GetValueOrDefault(StageNames.Diarization),
            SeparationModelAlias: modelPreferences.GetValueOrDefault(StageNames.Separation),
            OverlapRescueModelAlias: modelPreferences.GetValueOrDefault(StageNames.OverlapRescue),
            TranslationModelAlias: modelPreferences.GetValueOrDefault(StageNames.Translation),
            TtsModelAlias: modelPreferences.GetValueOrDefault(StageNames.Tts),
            TextRefinementModelAlias: modelPreferences.GetValueOrDefault(StageNames.TextRefinementAsr),
            LipSyncModelAlias: modelPreferences.GetValueOrDefault(StageNames.LipSync),
            LipSynthesisModelAlias: modelPreferences.GetValueOrDefault(StageNames.LipSynthesis),
            EnableAsrTextRefinement: options.EnableAsrTextRefinement,
            PreferredExecutionProviders: providerPins,
            PreferredModelVariantAliases: BuildVariantAliases(options.ModelVariantPreferences),
            RequiredExecutionProviderStages:
                providerPins is { Count: > 0 } && options.RequireExecutionProviderPreferences
                    ? new HashSet<RuntimeStage>(providerPins.Keys)
                    : null);
    }

    /// <summary>
    /// Parses <see cref="DubbingSessionOptions.ExecutionProviderPreferences"/> into a
    /// runtime-stage keyed map. Returns null when no pins were supplied.
    /// </summary>
    private static IReadOnlyDictionary<RuntimeStage, ExecutionProviderKind>? ParseExecutionProviderPreferences(
        DubbingSessionOptions options)
    {
        if (options.ExecutionProviderPreferences is not { Count: > 0 } pins)
        {
            return null;
        }

        var providers = new Dictionary<RuntimeStage, ExecutionProviderKind>();
        foreach ((string stageName, string label) in pins)
        {
            if (DubbingPipelineEngine.MapStageNameToRuntimeStage(stageName) is not { } stage)
            {
                throw new ArgumentException(
                    $"Execution provider pin names unknown or non-runtime stage '{stageName}'.",
                    nameof(options));
            }

            if (!ExecutionProviderTokens.TryParse(label, out ExecutionProviderKind provider))
            {
                throw new ArgumentException(
                    $"Execution provider pin for stage '{stageName}' has invalid provider label '{label}'. " +
                    $"Expected one of: {ExecutionProviderTokens.FormatSupportedCliTags()}.",
                    nameof(options));
            }

            providers[stage] = ExecutionProviderTokens.ResolvePlatformPin(provider);
        }

        return providers;
    }
}
