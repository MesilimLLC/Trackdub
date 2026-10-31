using Trackdub.Contracts;
using Trackdub.Contracts.Licensing;
using Trackdub.Domain;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Transcripts;

public sealed record RuntimeModelRequestOptions(
    AsrModelOverride AsrModelOverride,
    bool IsDevBuild,
    IReadOnlyDictionary<string, ExecutionProviderKind> HardwareOverrides,
    string? DiarizationModelAlias = null,
    string? SeparationModelAlias = null,
    string? OverlapRescueModelAlias = null,
    string? AsrModelAlias = null,
    string? TranslationModelAlias = null,
    string? TtsModelAlias = null,
    string? TextRefinementModelAlias = null,
    string? LipSyncModelAlias = null,
    string? LipSynthesisModelAlias = null,
    bool EnableAsrTextRefinement = false,
    TranslationModelOverride TranslationModelOverride = TranslationModelOverride.Auto,
    TtsModelOverride TtsModelOverride = TtsModelOverride.Auto,
    SeparationModelOverride SeparationModelOverride = SeparationModelOverride.Auto,
    IReadOnlyDictionary<string, string>? ModelVariantOverrides = null,
    bool RequirePreferredExecutionProviders = false,
    string? VadModelAlias = null,
    string? SpeechEnhancementModelAlias = null,
    ExecutionProviderKind? PreferredExecutionProvider = null,
    IReadOnlySet<RuntimeStage>? RequiredExecutionProviderStages = null);

public sealed record RuntimeModelSelections(
    AsrModelOverride AsrModelOverride,
    bool IsDevBuild,
    IReadOnlyDictionary<string, ExecutionProviderKind> HardwareOverrides,
    string? DiarizationModelAlias = null,
    string? SeparationModelAlias = null,
    string? OverlapRescueModelAlias = null,
    string? AsrModelAlias = null,
    string? TranslationModelAlias = null,
    string? TtsModelAlias = null,
    string? TextRefinementModelAlias = null,
    string? LipSyncModelAlias = null,
    string? LipSynthesisModelAlias = null,
    bool EnableAsrTextRefinement = false,
    TranslationModelOverride TranslationModelOverride = TranslationModelOverride.Auto,
    TtsModelOverride TtsModelOverride = TtsModelOverride.Auto,
    SeparationModelOverride SeparationModelOverride = SeparationModelOverride.Auto,
    IReadOnlyDictionary<string, string>? ModelVariantOverrides = null,
    bool RequirePreferredExecutionProviders = false,
    string? VadModelAlias = null,
    IReadOnlyDictionary<RuntimeStage, RuntimeModelSelectionIntent>? SelectionIntents = null,
    string? SpeechEnhancementModelAlias = null,
    ExecutionProviderKind? PreferredExecutionProvider = null,
    IReadOnlySet<RuntimeStage>? RequiredExecutionProviderStages = null)
{
    public RuntimeModelSelectionIntent GetSelectionIntent(RuntimeStage stage)
    {
        if (SelectionIntents?.TryGetValue(stage, out var intent) == true) return intent;
        string? alias = stage switch
        {
            RuntimeStage.Asr => AsrModelAlias,
            RuntimeStage.Translation => TranslationModelAlias,
            RuntimeStage.Tts => TtsModelAlias,
            RuntimeStage.Separation => SeparationModelAlias,
            RuntimeStage.Diarization => DiarizationModelAlias,
            RuntimeStage.Vad => VadModelAlias,
            RuntimeStage.OverlapRescue => OverlapRescueModelAlias,
            RuntimeStage.TextRefinement => TextRefinementModelAlias,
            RuntimeStage.LipSync => LipSyncModelAlias,
            RuntimeStage.LipSynthesis => LipSynthesisModelAlias,
            RuntimeStage.SpeechEnhancement => SpeechEnhancementModelAlias,
            _ => null
        };
        bool configuredOverride = stage switch
        {
            RuntimeStage.Asr => AsrModelOverride != AsrModelOverride.Auto,
            RuntimeStage.Translation => TranslationModelOverride != TranslationModelOverride.Auto,
            RuntimeStage.Tts => TtsModelOverride != TtsModelOverride.Auto,
            RuntimeStage.Separation => SeparationModelOverride != SeparationModelOverride.Auto,
            _ => false
        };
        return configuredOverride || !string.IsNullOrWhiteSpace(alias)
            ? RuntimeModelSelectionIntent.Explicit : RuntimeModelSelectionIntent.Automatic;
    }
}

public enum RuntimeModelSelectionIntent { Automatic, Explicit }

public static class RuntimeModelRequestFactory
{
    public static RuntimeModelSelections CreateSelectionsFromPreferences(
        InferenceModelPreferences? preferences,
        AsrModelOverride asrModelOverride = AsrModelOverride.Auto,
        bool isDevBuild = false)
    {
        preferences ??= InferenceModelPreferences.Empty;

        return CaptureSelectionIntents(new RuntimeModelSelections(
            asrModelOverride,
            isDevBuild,
            CreateHardwareOverrides(preferences, asrModelOverride),
            DiarizationModelAlias: preferences.DiarizationModelAlias,
            SeparationModelAlias: preferences.SeparationModelAlias,
            OverlapRescueModelAlias: preferences.OverlapRescueModelAlias,
            AsrModelAlias: preferences.AsrModelAlias,
            TranslationModelAlias: preferences.TranslationModelAlias,
            TtsModelAlias: preferences.TtsModelAlias,
            TextRefinementModelAlias: preferences.TextRefinementModelAlias,
            LipSyncModelAlias: preferences.LipSyncModelAlias,
            LipSynthesisModelAlias: preferences.LipSynthesisModelAlias,
            EnableAsrTextRefinement: preferences.EnableAsrTextRefinement,
            ModelVariantOverrides: CreateModelVariantOverridesFromPreferences(preferences),
            VadModelAlias: preferences.VadModelAlias,
            PreferredExecutionProvider: NormalizePreferredExecutionProvider(preferences.GlobalPreferredExecutionProvider),
            RequiredExecutionProviderStages: preferences.RequiredExecutionProviderStages));
    }

    public static RuntimeModelSelections CreateSelectionsFromSettings(
        StudioSettings settings,
        InferenceModelPreferences? explicitPreferences = null,
        bool isDevBuild = false)
    {
        ArgumentNullException.ThrowIfNull(settings);

        return CaptureSettingsSelectionIntents(settings, explicitPreferences, new RuntimeModelSelections(
            settings.AsrModelOverride,
            isDevBuild,
            settings.HardwareOverrides ?? new Dictionary<string, ExecutionProviderKind>(),
            DiarizationModelAlias: ResolveDiarizationAlias(settings, explicitPreferences),
            SeparationModelAlias: ResolveSeparationAlias(settings, explicitPreferences),
            OverlapRescueModelAlias: ResolveExplicitAlias(explicitPreferences?.OverlapRescueModelAlias),
            AsrModelAlias: ResolveAsrAlias(settings, explicitPreferences),
            TranslationModelAlias: ResolveTranslationAlias(settings, explicitPreferences),
            TtsModelAlias: ResolveTtsAlias(settings, explicitPreferences),
            TextRefinementModelAlias: ResolveTextRefinementAlias(settings, explicitPreferences),
            LipSyncModelAlias: ResolveLipSyncAlias(settings, explicitPreferences),
            LipSynthesisModelAlias: ResolveLipSynthesisAlias(settings, explicitPreferences),
            EnableAsrTextRefinement: explicitPreferences?.EnableAsrTextRefinement ?? false,
            TranslationModelOverride: settings.TranslationModelOverride,
            TtsModelOverride: settings.TtsModelOverride,
            SeparationModelOverride: settings.SeparationModelOverride,
            ModelVariantOverrides: settings.ModelVariantOverrides,
            RequirePreferredExecutionProviders: settings.RequirePreferredExecutionProviders,
            VadModelAlias: ResolveVadAlias(settings, explicitPreferences),
            PreferredExecutionProvider: NormalizePreferredExecutionProvider(settings.PreferredExecutionProvider)));
    }

    private static RuntimeModelSelections CaptureSettingsSelectionIntents(StudioSettings settings,
        InferenceModelPreferences? preferences, RuntimeModelSelections selections)
    {
        selections = CaptureSelectionIntents(selections);
        var intents = selections.SelectionIntents!.ToDictionary(pair => pair.Key, pair => pair.Value);
        foreach (RuntimeStage stage in Enum.GetValues<RuntimeStage>())
        {
            string key = stage switch
            {
                RuntimeStage.Asr => StageNames.Asr,
                RuntimeStage.Translation => StageNames.Translation,
                RuntimeStage.Tts => StageNames.Tts,
                RuntimeStage.Separation => StageNames.Separation,
                RuntimeStage.Vad => StageNames.Vad,
                RuntimeStage.Diarization => StageNames.Diarization,
                RuntimeStage.TextRefinement => StageNames.TextRefinementAsr,
                RuntimeStage.LipSync => StageNames.LipSync,
                RuntimeStage.LipSynthesis => StageNames.LipSynthesis,
                RuntimeStage.OverlapRescue => StageNames.OverlapRescue,
                _ => stage.ToString()
            };
            string? callerAlias = stage switch
            {
                RuntimeStage.Asr => preferences?.AsrModelAlias,
                RuntimeStage.Translation => preferences?.TranslationModelAlias,
                RuntimeStage.Tts => preferences?.TtsModelAlias,
                RuntimeStage.Separation => preferences?.SeparationModelAlias,
                RuntimeStage.Vad => preferences?.VadModelAlias,
                RuntimeStage.Diarization => preferences?.DiarizationModelAlias,
                RuntimeStage.TextRefinement => preferences?.TextRefinementModelAlias,
                RuntimeStage.LipSync => preferences?.LipSyncModelAlias,
                RuntimeStage.LipSynthesis => preferences?.LipSynthesisModelAlias,
                RuntimeStage.OverlapRescue => preferences?.OverlapRescueModelAlias,
                _ => null
            };
            string? resolvedAlias = stage switch
            {
                RuntimeStage.Asr => selections.AsrModelAlias,
                RuntimeStage.Translation => selections.TranslationModelAlias,
                RuntimeStage.Tts => selections.TtsModelAlias,
                RuntimeStage.Separation => selections.SeparationModelAlias,
                RuntimeStage.Vad => selections.VadModelAlias,
                RuntimeStage.Diarization => selections.DiarizationModelAlias,
                RuntimeStage.TextRefinement => selections.TextRefinementModelAlias,
                RuntimeStage.LipSync => selections.LipSyncModelAlias,
                RuntimeStage.LipSynthesis => selections.LipSynthesisModelAlias,
                RuntimeStage.OverlapRescue => selections.OverlapRescueModelAlias,
                _ => null
            };
            if (string.IsNullOrWhiteSpace(callerAlias) && !TryGetStageAlias(settings.StageModelAliases, key, out _)
                && settings.AutomaticModelAliases?.TryGetValue(key, out string? defaultAlias) == true
                && string.Equals(defaultAlias, resolvedAlias, StringComparison.OrdinalIgnoreCase))
                intents[stage] = RuntimeModelSelectionIntent.Automatic;
        }
        return selections with { SelectionIntents = intents };
    }

    private static RuntimeModelSelections CaptureSelectionIntents(RuntimeModelSelections selections) =>
        selections with { SelectionIntents = Enum.GetValues<RuntimeStage>().ToDictionary(stage => stage, selections.GetSelectionIntent) };

    private static string? ResolveAsrAlias(StudioSettings settings, InferenceModelPreferences? explicitPreferences)
    {
        string? explicitAlias = ResolveExplicitAlias(explicitPreferences?.AsrModelAlias);
        if (explicitAlias is not null)
        {
            return explicitAlias;
        }

        if (TryGetStageAlias(settings.StageModelAliases, StageNames.Asr, out string? packAlias))
        {
            return packAlias;
        }

        return ShouldUseOverrideEnumFallback(settings)
            ? AsrModelOverrideSettings.ResolveModelAlias(settings.AsrModelOverride)
            : null;
    }

    private static string? ResolveTranslationAlias(StudioSettings settings, InferenceModelPreferences? explicitPreferences)
    {
        string? explicitAlias = ResolveExplicitAlias(explicitPreferences?.TranslationModelAlias);
        if (explicitAlias is not null)
        {
            return explicitAlias;
        }

        if (TryGetStageAlias(settings.StageModelAliases, StageNames.Translation, out string? packAlias))
        {
            return packAlias;
        }

        return ShouldUseOverrideEnumFallback(settings)
            ? TranslationModelOverrideSettings.ResolveModelAlias(settings.TranslationModelOverride)
            : null;
    }

    private static string? ResolveTtsAlias(StudioSettings settings, InferenceModelPreferences? explicitPreferences)
    {
        string? explicitAlias = ResolveExplicitAlias(explicitPreferences?.TtsModelAlias);
        if (explicitAlias is not null)
        {
            return explicitAlias;
        }

        if (TryGetStageAlias(settings.StageModelAliases, StageNames.Tts, out string? packAlias))
        {
            return packAlias;
        }

        return ShouldUseOverrideEnumFallback(settings)
            ? TtsModelOverrideSettings.ResolveModelAlias(settings.TtsModelOverride)
            : null;
    }

    private static string? ResolveTextRefinementAlias(
        StudioSettings settings,
        InferenceModelPreferences? explicitPreferences)
    {
        string? explicitAlias = ResolveExplicitAlias(explicitPreferences?.TextRefinementModelAlias);
        if (explicitAlias is not null)
        {
            return explicitAlias;
        }

        return TryGetStageAlias(settings.StageModelAliases, StageNames.TextRefinementAsr, out string? packAlias)
            ? packAlias
            : null;
    }

    private static string? ResolveLipSyncAlias(StudioSettings settings, InferenceModelPreferences? explicitPreferences)
    {
        string? explicitAlias = ResolveExplicitAlias(explicitPreferences?.LipSyncModelAlias);
        if (explicitAlias is not null)
        {
            return explicitAlias;
        }

        return TryGetStageAlias(settings.StageModelAliases, StageNames.LipSync, out string? packAlias)
            ? packAlias
            : null;
    }

    private static string? ResolveLipSynthesisAlias(StudioSettings settings, InferenceModelPreferences? explicitPreferences)
    {
        string? explicitAlias = ResolveExplicitAlias(explicitPreferences?.LipSynthesisModelAlias);
        if (explicitAlias is not null)
        {
            return explicitAlias;
        }

        return TryGetStageAlias(settings.StageModelAliases, StageNames.LipSynthesis, out string? packAlias)
            ? packAlias
            : null;
    }

    private static string? ResolveVadAlias(
        StudioSettings settings,
        InferenceModelPreferences? explicitPreferences)
    {
        string? explicitAlias = ResolveExplicitAlias(explicitPreferences?.VadModelAlias);
        if (explicitAlias is not null)
        {
            return explicitAlias;
        }

        return TryGetStageAlias(settings.StageModelAliases, StageNames.Vad, out string? packAlias)
            ? packAlias
            : null;
    }

    private static string? ResolveDiarizationAlias(StudioSettings settings, InferenceModelPreferences? explicitPreferences)
    {
        string? explicitAlias = ResolveExplicitAlias(explicitPreferences?.DiarizationModelAlias);
        if (explicitAlias is not null)
        {
            return explicitAlias;
        }

        return TryGetStageAlias(settings.StageModelAliases, StageNames.Diarization, out string? packAlias)
            ? packAlias
            : null;
    }

    private static string? ResolveSeparationAlias(StudioSettings settings, InferenceModelPreferences? explicitPreferences)
    {
        string? explicitAlias = ResolveExplicitAlias(explicitPreferences?.SeparationModelAlias);
        if (explicitAlias is not null)
        {
            return explicitAlias;
        }

        if (TryGetStageAlias(settings.StageModelAliases, StageNames.Separation, out string? packAlias))
        {
            return packAlias;
        }

        return ShouldUseOverrideEnumFallback(settings)
            ? SeparationModelOverrideSettings.ResolveModelAlias(settings.SeparationModelOverride)
            : null;
    }

    private static bool ShouldUseOverrideEnumFallback(StudioSettings settings) =>
        string.IsNullOrWhiteSpace(settings.AppliedStarterPackId) &&
        (settings.StageModelAliases is null || settings.StageModelAliases.Count == 0);

    private static string? ResolveExplicitAlias(string? alias) =>
        string.IsNullOrWhiteSpace(alias) ? null : alias.Trim();

    private static bool TryGetStageAlias(
        IReadOnlyDictionary<string, string>? aliases,
        string stageName,
        out string? alias)
    {
        alias = null;
        if (aliases is null || aliases.Count == 0)
        {
            return false;
        }

        if (aliases.TryGetValue(stageName, out string? value) && !string.IsNullOrWhiteSpace(value))
        {
            alias = value.Trim();
            return true;
        }

        return false;
    }

    public static RuntimeModelRequestOptions CreateOptions(RuntimeModelSelections selections)
    {
        ArgumentNullException.ThrowIfNull(selections);

        return new RuntimeModelRequestOptions(
            selections.AsrModelOverride,
            selections.IsDevBuild,
            selections.HardwareOverrides,
            selections.DiarizationModelAlias,
            selections.SeparationModelAlias,
            selections.OverlapRescueModelAlias,
            selections.AsrModelAlias,
            selections.TranslationModelAlias,
            selections.TtsModelAlias,
            selections.TextRefinementModelAlias,
            selections.LipSyncModelAlias,
            selections.LipSynthesisModelAlias,
            selections.EnableAsrTextRefinement,
            selections.TranslationModelOverride,
            selections.TtsModelOverride,
            selections.SeparationModelOverride,
            selections.ModelVariantOverrides,
            selections.RequirePreferredExecutionProviders,
            selections.VadModelAlias,
            selections.SpeechEnhancementModelAlias,
            selections.PreferredExecutionProvider,
            selections.RequiredExecutionProviderStages);
    }

    public static InferenceModelPreferences CreateModelPreferences(RuntimeModelSelections selections) =>
        CreateModelPreferences(CreateOptions(selections));

    public static InferenceModelPreferences CreateModelPreferences(RuntimeModelRequestOptions options) =>
        new(
            VadModelAlias: options.VadModelAlias,
            AsrModelAlias: ResolveExplicitAsrModelAlias(options),
            DiarizationModelAlias: ResolveDiarizationModelAlias(options),
            SeparationModelAlias: ResolveSeparationModelAlias(options),
            OverlapRescueModelAlias: ResolveOverlapRescueModelAlias(options),
            TranslationModelAlias: ResolveTranslationModelAlias(options),
            TtsModelAlias: ResolveTtsModelAlias(options),
            TextRefinementModelAlias: ResolveTextRefinementModelAlias(options),
            LipSyncModelAlias: ResolveLipSyncModelAlias(options),
            LipSynthesisModelAlias: ResolveLipSynthesisModelAlias(options),
            EnableAsrTextRefinement: options.EnableAsrTextRefinement,
            RequireAsrModelAlias: RequiresExplicitAsrModelAlias(options),
            PreferredExecutionProviders: CreatePreferredExecutionProviders(options),
            RequiredExecutionProviderStages: CreateRequiredExecutionProviderStages(options),
            PreferredModelVariantAliases: CreatePreferredModelVariantAliases(options),
            GlobalPreferredExecutionProvider: options.PreferredExecutionProvider,
            ExplicitExecutionProviderStages: Enum.GetValues<RuntimeStage>()
                .Where(stage => ResolveStageExecutionProviderOverride(options, stage) is not null)
                .ToHashSet());

    public static RuntimeModelRequest CreateAsrRequest(
        RuntimeModelRequestOptions options,
        string? sourceLanguageCode = null)
    {
        string? preferredModelAlias = ResolveExplicitAsrModelAlias(options);
        return new RuntimeModelRequest(
            RuntimeStage.Asr,
            preferredModelAlias,
            SourceLanguage: sourceLanguageCode,
            RequirePreferredModelAlias: RequiresExplicitAsrModelAlias(options),
            PreferredExecutionProvider: ResolvePreferredExecutionProvider(options, RuntimeStage.Asr),
            RequirePreferredExecutionProvider: IsPreferredExecutionProviderRequired(options, RuntimeStage.Asr),
            PreferredModelVariantAlias: ResolvePreferredModelVariantAlias(options, RuntimeStage.Asr, preferredModelAlias));
    }

    public static RuntimeModelRequest CreateDiarizationRequest(RuntimeModelRequestOptions options)
    {
        string? preferredModelAlias = ResolveDiarizationModelAlias(options);
        return new RuntimeModelRequest(
            RuntimeStage.Diarization,
            preferredModelAlias,
            RequirePreferredModelAlias: IsDiarizationModelAliasRequired(options),
            PreferredExecutionProvider: ResolvePreferredExecutionProvider(options, RuntimeStage.Diarization),
            RequirePreferredExecutionProvider: IsPreferredExecutionProviderRequired(options, RuntimeStage.Diarization),
            PreferredModelVariantAlias: ResolvePreferredModelVariantAlias(options, RuntimeStage.Diarization, preferredModelAlias));
    }

    public static IReadOnlyList<RuntimeModelRequest> CreateImportRequests(
        RuntimeModelRequestOptions options,
        bool enableStemSeparation,
        string? sourceLanguageCode = null)
    {
        var requests = new List<RuntimeModelRequest>
        {
            CreateStageRequest(options, RuntimeStage.Vad),
            CreateAsrRequest(options, sourceLanguageCode)
        };

        if (enableStemSeparation)
        {
            requests.Add(CreateSeparationRequest(options));
        }

        return requests;
    }

    public static IReadOnlyList<RuntimeModelRequest> CreateStemRerunRequests(
        RuntimeModelRequestOptions options,
        IReadOnlyList<RuntimeStage> stages,
        string? sourceLanguageCode = null) =>
        stages.Select(stage => stage switch
        {
            RuntimeStage.Asr => CreateAsrRequest(options, sourceLanguageCode),
            RuntimeStage.Diarization => CreateDiarizationRequest(options),
            RuntimeStage.Separation => CreateSeparationRequest(options),
            _ => CreateStageRequest(options, stage)
        }).ToArray();

    public static RuntimeModelRequest CreateTranslationRequest(
        RuntimeModelRequestOptions options,
        string sourceLanguageCode,
        string targetLanguageCode)
    {
        string? preferredModelAlias = ResolveTranslationModelAlias(options);
        return new RuntimeModelRequest(
            RuntimeStage.Translation,
            PreferredModelAlias: preferredModelAlias,
            RequirePreferredModelAlias: IsTranslationModelAliasRequired(options),
            SourceLanguage: sourceLanguageCode,
            TargetLanguage: targetLanguageCode,
            PreferredExecutionProvider: ResolvePreferredExecutionProvider(options, RuntimeStage.Translation),
            RequirePreferredExecutionProvider: IsPreferredExecutionProviderRequired(options, RuntimeStage.Translation),
            PreferredModelVariantAlias: ResolvePreferredModelVariantAlias(options, RuntimeStage.Translation, preferredModelAlias));
    }

    public static RuntimeModelRequest CreateTtsRequest(
        RuntimeModelRequestOptions options,
        bool requiresVoiceClone,
        string preferredTier = "balanced")
    {
        string? explicitAlias = ResolveTtsModelAlias(options);
        if (string.IsNullOrWhiteSpace(explicitAlias) && options.TtsModelOverride == TtsModelOverride.Qwen3Tts)
        {
            explicitAlias = requiresVoiceClone
                ? Qwen3TtsDefaults.ResolveBaseAlias(preferredTier)
                : Qwen3TtsDefaults.ResolveCustomVoiceAlias(preferredTier);
        }

        if (!string.IsNullOrWhiteSpace(explicitAlias))
        {
            return new RuntimeModelRequest(
                RuntimeStage.Tts,
                explicitAlias,
                RequirePreferredModelAlias: true,
                PreferredExecutionProvider: ResolvePreferredExecutionProvider(options, RuntimeStage.Tts),
                RequirePreferredExecutionProvider: IsPreferredExecutionProviderRequired(options, RuntimeStage.Tts),
                PreferredModelVariantAlias: ResolvePreferredModelVariantAlias(options, RuntimeStage.Tts, explicitAlias));
        }

        return requiresVoiceClone
            ? new RuntimeModelRequest(
                RuntimeStage.Tts,
                VoiceCloningDefaults.ChatterboxPrimaryAlias,
                RequirePreferredModelAlias: true,
                PreferredExecutionProvider: ResolvePreferredExecutionProvider(options, RuntimeStage.Tts),
                RequirePreferredExecutionProvider: IsPreferredExecutionProviderRequired(options, RuntimeStage.Tts),
                PreferredModelVariantAlias: ResolvePreferredModelVariantAlias(options, RuntimeStage.Tts, VoiceCloningDefaults.ChatterboxPrimaryAlias))
            : new RuntimeModelRequest(
                RuntimeStage.Tts,
                StockTtsDefaults.KokoroPrimaryAlias,
                RequirePreferredModelAlias: true,
                PreferredExecutionProvider: ResolvePreferredExecutionProvider(options, RuntimeStage.Tts),
                RequirePreferredExecutionProvider: IsPreferredExecutionProviderRequired(options, RuntimeStage.Tts),
                PreferredModelVariantAlias: ResolvePreferredModelVariantAlias(options, RuntimeStage.Tts, StockTtsDefaults.KokoroPrimaryAlias));
    }

    public static RetranscribeTranscriptSegmentsRequest CreateRetranscribeRequest(
        RuntimeModelRequestOptions options,
        Guid transcriptRevisionId,
        IReadOnlyList<Guid> segmentIds) =>
        new(
            transcriptRevisionId,
            segmentIds,
            ResolveExplicitAsrModelAlias(options),
            RequiresExplicitAsrModelAlias(options),
            PreferredExecutionProvider: ResolvePreferredExecutionProvider(options, RuntimeStage.Asr),
            RequirePreferredExecutionProvider: IsPreferredExecutionProviderRequired(options, RuntimeStage.Asr),
            PreferredModelVariantAlias: ResolvePreferredModelVariantAlias(options, RuntimeStage.Asr, ResolveExplicitAsrModelAlias(options)));

    public static RerunDiarizationRequest CreateRerunDiarizationRequest(RuntimeModelRequestOptions options) =>
        new(
            ResolveDiarizationModelAlias(options),
            PreferredExecutionProvider: ResolvePreferredExecutionProvider(options, RuntimeStage.Diarization),
            RequirePreferredExecutionProvider: IsPreferredExecutionProviderRequired(options, RuntimeStage.Diarization),
            PreferredModelVariantAlias: ResolvePreferredModelVariantAlias(options, RuntimeStage.Diarization, ResolveDiarizationModelAlias(options)));

    /// <summary>
    /// Stable lowercase provider token shared by the execution snapshot and the
    /// StageRunRecord runtime info comparison. Mirrors the labels persisted by the ONNX
    /// session layer so a requested provider can be compared against a prior run.
    /// </summary>
    public static string FormatExecutionProviderLabel(ExecutionProviderKind provider) =>
        provider switch
        {
            ExecutionProviderKind.Cpu => "cpu",
            ExecutionProviderKind.DirectMl => "dml",
            ExecutionProviderKind.TensorRTRtx => "tensorrt-rtx",
            ExecutionProviderKind.OpenVino => "openvino",
            ExecutionProviderKind.CoreMl => "coreml",
            ExecutionProviderKind.Cuda => "cuda",
            ExecutionProviderKind.TensorRt => "tensorrt",
            ExecutionProviderKind.Migraphx => "migraphx",
            ExecutionProviderKind.Dnnl => "dnnl",
            ExecutionProviderKind.Qnn => "qnn",
            ExecutionProviderKind.OpenVinoCatalog => "openvino-catalog",
            ExecutionProviderKind.VitisAi => "vitisai",
            _ => provider.ToString().ToLowerInvariant()
        };

    public static ExecutionProviderKind? ResolvePreferredExecutionProvider(
        RuntimeModelRequestOptions options,
        RuntimeStage stage)
    {
        ExecutionProviderKind? stagePreference = ResolveStageExecutionProviderOverride(options, stage);
        return NormalizePreferredExecutionProvider(stagePreference ?? options.PreferredExecutionProvider);
    }

    // A saved value outside ExecutionProviderKind means automatic selection, not a forbidden provider.
    private static ExecutionProviderKind? NormalizePreferredExecutionProvider(ExecutionProviderKind? provider) =>
        provider is { } kind && Enum.IsDefined(kind) ? kind : null;

    private static ExecutionProviderKind? ResolveStageExecutionProviderOverride(
        RuntimeModelRequestOptions options,
        RuntimeStage stage)
    {
        string key = stage.ToString();
        if (stage == RuntimeStage.Asr)
        {
            if (NormalizeAsrModelOverride(options) == AsrModelOverride.GenAi)
            {
                key = "AsrGenAi";
            }
            else if (NormalizeAsrModelOverride(options) == AsrModelOverride.Nemotron35)
            {
                key = "AsrNemotron";
            }
            else if (NormalizeAsrModelOverride(options) == AsrModelOverride.OnnxRuntime)
            {
                key = "AsrOnnxRuntime";
            }
        }

        return options.HardwareOverrides.TryGetValue(key, out ExecutionProviderKind provider)
            ? NormalizePreferredExecutionProvider(provider)
            : null;
    }

    public static string? ResolvePreferredModelVariantAlias(
        RuntimeModelRequestOptions options,
        RuntimeStage stage,
        string? preferredModelAlias = null)
    {
        if (options.ModelVariantOverrides is null || options.ModelVariantOverrides.Count == 0)
        {
            return null;
        }

        preferredModelAlias ??= ResolveModelAliasForVariantLookup(options, stage);

        string stageKey = ResolveStageKey(stage);
        if (!string.IsNullOrWhiteSpace(preferredModelAlias))
        {
            string compositeKey = ModelVariantOverrideKeys.Build(stageKey, preferredModelAlias);
            if (options.ModelVariantOverrides.TryGetValue(compositeKey, out string? modelScopedAlias) &&
                !string.IsNullOrWhiteSpace(modelScopedAlias))
            {
                return modelScopedAlias.Trim();
            }
        }

        return options.ModelVariantOverrides.TryGetValue(stageKey, out string? stageAlias) && !string.IsNullOrWhiteSpace(stageAlias)
            ? stageAlias.Trim()
            : null;
    }

    private static string? ResolveModelAliasForVariantLookup(RuntimeModelRequestOptions options, RuntimeStage stage)
    {
        string? alias = stage switch
        {
            RuntimeStage.Asr => ResolveExplicitAsrModelAlias(options),
            RuntimeStage.Diarization => ResolveDiarizationModelAlias(options),
            RuntimeStage.Separation => ResolveSeparationModelAlias(options),
            RuntimeStage.OverlapRescue => ResolveOverlapRescueModelAlias(options),
            RuntimeStage.Translation => ResolveTranslationModelAlias(options),
            RuntimeStage.Tts => ResolveTtsModelAlias(options),
            RuntimeStage.TextRefinement => ResolveTextRefinementModelAlias(options),
            RuntimeStage.LipSync => ResolveLipSyncModelAlias(options),
            RuntimeStage.LipSynthesis => ResolveLipSynthesisModelAlias(options),
            RuntimeStage.SpeechEnhancement => ResolveSpeechEnhancementModelAlias(options),
            _ => null
        };

        return !string.IsNullOrWhiteSpace(alias)
            ? alias.Trim()
            : TryResolveModelAliasFromScopedVariantKeys(options, ResolveStageKey(stage));
    }

    private static string? TryResolveModelAliasFromScopedVariantKeys(
        RuntimeModelRequestOptions options,
        string stageKey)
    {
        if (options.ModelVariantOverrides is null || options.ModelVariantOverrides.Count == 0)
        {
            return null;
        }

        string? resolvedAlias = null;
        foreach ((string key, _) in options.ModelVariantOverrides)
        {
            if (!ModelVariantOverrideKeys.TryParse(key, out string parsedStageKey, out string modelAlias) ||
                !string.Equals(parsedStageKey, stageKey, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (resolvedAlias is not null &&
                !string.Equals(resolvedAlias, modelAlias, StringComparison.OrdinalIgnoreCase))
            {
                return null;
            }

            resolvedAlias = modelAlias;
        }

        return resolvedAlias;
    }

    public static RuntimeModelRequest CreateSeparationRequest(RuntimeModelRequestOptions options)
    {
        string? preferredModelAlias = ResolveSeparationModelAlias(options);
        return new RuntimeModelRequest(
            RuntimeStage.Separation,
            PreferredModelAlias: preferredModelAlias,
            RequirePreferredModelAlias: IsSeparationModelAliasRequired(options),
            PreferredExecutionProvider: ResolvePreferredExecutionProvider(options, RuntimeStage.Separation),
            RequirePreferredExecutionProvider: IsPreferredExecutionProviderRequired(options, RuntimeStage.Separation),
            PreferredModelVariantAlias: ResolvePreferredModelVariantAlias(options, RuntimeStage.Separation, preferredModelAlias));
    }

    public static RuntimeModelRequest CreateOverlapRescueRequest(RuntimeModelRequestOptions options)
    {
        string? preferredModelAlias = ResolveOverlapRescueModelAlias(options);
        return new RuntimeModelRequest(
            RuntimeStage.OverlapRescue,
            PreferredModelAlias: preferredModelAlias,
            RequirePreferredModelAlias: IsOverlapRescueModelAliasRequired(options),
            PreferredExecutionProvider: ResolvePreferredExecutionProvider(options, RuntimeStage.OverlapRescue),
            RequirePreferredExecutionProvider: IsPreferredExecutionProviderRequired(options, RuntimeStage.OverlapRescue),
            PreferredModelVariantAlias: ResolvePreferredModelVariantAlias(options, RuntimeStage.OverlapRescue, preferredModelAlias));
    }

    public static RuntimeModelRequest CreateStageRequest(
        RuntimeModelRequestOptions options,
        RuntimeStage stage) =>
        new(
            stage,
            PreferredModelAlias: stage == RuntimeStage.SpeechEnhancement
                ? ResolveSpeechEnhancementModelAlias(options)
                : null,
            RequirePreferredModelAlias: stage == RuntimeStage.SpeechEnhancement
                && ResolveSpeechEnhancementModelAlias(options) is not null,
            PreferredExecutionProvider: ResolvePreferredExecutionProvider(options, stage),
            RequirePreferredExecutionProvider: IsPreferredExecutionProviderRequired(options, stage),
            PreferredModelVariantAlias: ResolvePreferredModelVariantAlias(options, stage));

    public static RuntimeModelRequest CreateLipSyncRequest(
        RuntimeModelRequestOptions options,
        string? preferredModelAlias = null)
    {
        string? alias = ResolveLipSyncModelAlias(options, preferredModelAlias);
        return new RuntimeModelRequest(
            RuntimeStage.LipSync,
            PreferredModelAlias: alias,
            RequirePreferredModelAlias: !string.IsNullOrWhiteSpace(alias),
            PreferredExecutionProvider: ResolvePreferredExecutionProvider(options, RuntimeStage.LipSync),
            RequirePreferredExecutionProvider: IsPreferredExecutionProviderRequired(options, RuntimeStage.LipSync),
            PreferredModelVariantAlias: ResolvePreferredModelVariantAlias(options, RuntimeStage.LipSync, alias));
    }

    public static RuntimeModelRequest CreateLipSynthesisRequest(
        RuntimeModelRequestOptions options,
        string? preferredModelAlias = null)
    {
        string? alias = ResolveLipSynthesisModelAlias(options, preferredModelAlias);
        return new RuntimeModelRequest(
            RuntimeStage.LipSynthesis,
            PreferredModelAlias: alias,
            RequirePreferredModelAlias: !string.IsNullOrWhiteSpace(alias),
            PreferredExecutionProvider: ResolvePreferredExecutionProvider(options, RuntimeStage.LipSynthesis),
            RequirePreferredExecutionProvider: IsPreferredExecutionProviderRequired(options, RuntimeStage.LipSynthesis),
            PreferredModelVariantAlias: ResolvePreferredModelVariantAlias(options, RuntimeStage.LipSynthesis, alias));
    }

    private static IReadOnlyDictionary<RuntimeStage, ExecutionProviderKind> CreatePreferredExecutionProviders(
        RuntimeModelRequestOptions options)
    {
        var providers = new Dictionary<RuntimeStage, ExecutionProviderKind>();
        foreach (RuntimeStage stage in Enum.GetValues<RuntimeStage>())
        {
            ExecutionProviderKind? provider = ResolvePreferredExecutionProvider(options, stage);
            if (provider is not null)
            {
                providers[stage] = provider.Value;
            }
        }

        return providers;
    }

    private static IReadOnlySet<RuntimeStage> CreateRequiredExecutionProviderStages(RuntimeModelRequestOptions options)
    {
        var stages = new HashSet<RuntimeStage>();
        foreach (RuntimeStage stage in Enum.GetValues<RuntimeStage>())
        {
            if (IsPreferredExecutionProviderRequired(options, stage))
            {
                stages.Add(stage);
            }
        }

        return stages;
    }

    private static IReadOnlyDictionary<RuntimeStage, string> CreatePreferredModelVariantAliases(
        RuntimeModelRequestOptions options)
    {
        var aliases = new Dictionary<RuntimeStage, string>();
        foreach (RuntimeStage stage in Enum.GetValues<RuntimeStage>())
        {
            string? alias = ResolvePreferredModelVariantAlias(options, stage);
            if (!string.IsNullOrWhiteSpace(alias))
            {
                aliases[stage] = alias;
            }
        }

        return aliases;
    }

    private static string ResolveStageKey(RuntimeStage stage) =>
        stage switch
        {
            RuntimeStage.Vad => StageNames.Vad,
            RuntimeStage.Asr => StageNames.Asr,
            RuntimeStage.Diarization => StageNames.Diarization,
            RuntimeStage.Separation => StageNames.Separation,
            RuntimeStage.OverlapRescue => StageNames.OverlapRescue,
            RuntimeStage.Translation => StageNames.Translation,
            RuntimeStage.Tts => StageNames.Tts,
            RuntimeStage.SpeechEnhancement => StageNames.SpeechEnhancement,
            RuntimeStage.TextRefinement => StageNames.TextRefinementAsr,
            RuntimeStage.LipSync => StageNames.LipSync,
            RuntimeStage.LipSynthesis => StageNames.LipSynthesis,
            _ => stage.ToString().Trim().ToLowerInvariant()
        };

    public static bool IsPreferredExecutionProviderRequired(
        RuntimeModelRequestOptions options,
        RuntimeStage stage) =>
        ResolveStageExecutionProviderOverride(options, stage) is not null
        && (options.IsDevBuild || options.RequirePreferredExecutionProviders ||
            options.RequiredExecutionProviderStages?.Contains(stage) == true);

    private static bool RequiresExplicitAsrModelAlias(RuntimeModelRequestOptions options) =>
        !string.IsNullOrWhiteSpace(options.AsrModelAlias) ||
        AsrModelOverrideSettings.RequiresModelAlias(NormalizeAsrModelOverride(options));

    private static string? ResolveExplicitAsrModelAlias(RuntimeModelRequestOptions options)
    {
        if (!string.IsNullOrWhiteSpace(options.AsrModelAlias))
        {
            return options.AsrModelAlias.Trim();
        }

        return AsrModelOverrideSettings.ResolveModelAlias(NormalizeAsrModelOverride(options));
    }

    private static string? ResolveSeparationModelAlias(RuntimeModelRequestOptions options)
    {
        if (!string.IsNullOrWhiteSpace(options.SeparationModelAlias))
        {
            return options.SeparationModelAlias.Trim();
        }

        return SeparationModelOverrideSettings.ResolveModelAlias(options.SeparationModelOverride);
    }

    private static bool IsSeparationModelAliasRequired(RuntimeModelRequestOptions options) =>
        ResolveSeparationModelAlias(options) is not null;

    private static string? ResolveOverlapRescueModelAlias(RuntimeModelRequestOptions options) =>
        string.IsNullOrWhiteSpace(options.OverlapRescueModelAlias)
            ? null
            : options.OverlapRescueModelAlias.Trim();

    private static bool IsOverlapRescueModelAliasRequired(RuntimeModelRequestOptions options) =>
        ResolveOverlapRescueModelAlias(options) is not null;

    private static string? ResolveTranslationModelAlias(RuntimeModelRequestOptions options)
    {
        if (!string.IsNullOrWhiteSpace(options.TranslationModelAlias))
        {
            return options.TranslationModelAlias.Trim();
        }

        return TranslationModelOverrideSettings.ResolveModelAlias(options.TranslationModelOverride);
    }

    private static bool IsTranslationModelAliasRequired(RuntimeModelRequestOptions options) =>
        ResolveTranslationModelAlias(options) is not null;

    private static string? ResolveTtsModelAlias(RuntimeModelRequestOptions options)
    {
        if (!string.IsNullOrWhiteSpace(options.TtsModelAlias))
        {
            return options.TtsModelAlias.Trim();
        }

        return TtsModelOverrideSettings.ResolveModelAlias(options.TtsModelOverride);
    }

    private static string? ResolveSpeechEnhancementModelAlias(RuntimeModelRequestOptions options) =>
        string.IsNullOrWhiteSpace(options.SpeechEnhancementModelAlias)
            ? null
            : options.SpeechEnhancementModelAlias.Trim();

    private static string? ResolveTextRefinementModelAlias(RuntimeModelRequestOptions options) =>
        string.IsNullOrWhiteSpace(options.TextRefinementModelAlias)
            ? null
            : options.TextRefinementModelAlias.Trim();

    private static string? ResolveLipSyncModelAlias(
        RuntimeModelRequestOptions options,
        string? preferredModelAlias = null)
    {
        if (!string.IsNullOrWhiteSpace(preferredModelAlias))
        {
            return preferredModelAlias.Trim();
        }

        return string.IsNullOrWhiteSpace(options.LipSyncModelAlias)
            ? null
            : options.LipSyncModelAlias.Trim();
    }

    private static string? ResolveLipSynthesisModelAlias(
        RuntimeModelRequestOptions options,
        string? preferredModelAlias = null)
    {
        if (!string.IsNullOrWhiteSpace(preferredModelAlias))
        {
            return preferredModelAlias.Trim();
        }

        return string.IsNullOrWhiteSpace(options.LipSynthesisModelAlias)
            ? null
            : options.LipSynthesisModelAlias.Trim();
    }

    private static bool IsTextRefinementModelAliasRequired(RuntimeModelRequestOptions options) =>
        ResolveTextRefinementModelAlias(options) is not null;

    public static RuntimeModelRequest CreateTextRefinementRequest(RuntimeModelRequestOptions options) =>
        new(
            RuntimeStage.TextRefinement,
            PreferredModelAlias: ResolveTextRefinementModelAlias(options),
            RequirePreferredModelAlias: IsTextRefinementModelAliasRequired(options),
            PreferredExecutionProvider: ResolvePreferredExecutionProvider(options, RuntimeStage.TextRefinement),
            RequirePreferredExecutionProvider: IsPreferredExecutionProviderRequired(options, RuntimeStage.TextRefinement),
            PreferredModelVariantAlias: ResolvePreferredModelVariantAlias(options, RuntimeStage.TextRefinement, ResolveTextRefinementModelAlias(options)));

    private static string? ResolveDiarizationModelAlias(RuntimeModelRequestOptions options) =>
        string.IsNullOrWhiteSpace(options.DiarizationModelAlias)
            ? null
            : options.DiarizationModelAlias.Trim();

    private static bool IsDiarizationModelAliasRequired(RuntimeModelRequestOptions options) =>
        ResolveDiarizationModelAlias(options) is not null;

    private static AsrModelOverride NormalizeAsrModelOverride(RuntimeModelRequestOptions options) =>
        options.AsrModelOverride is AsrModelOverride.Auto
            or AsrModelOverride.GenAi
            or AsrModelOverride.OnnxRuntime
            or AsrModelOverride.Nemotron35
            ? options.AsrModelOverride
            : AsrModelOverride.Auto;

    private static IReadOnlyDictionary<string, string>? CreateModelVariantOverridesFromPreferences(
        InferenceModelPreferences preferences)
    {
        if (preferences.PreferredModelVariantAliases is not { Count: > 0 } variantAliases)
        {
            return null;
        }

        var overrides = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach ((RuntimeStage stage, string alias) in variantAliases)
        {
            overrides[ResolveStageKey(stage)] = alias;
        }

        return overrides;
    }

    private static IReadOnlyDictionary<string, ExecutionProviderKind> CreateHardwareOverrides(
        InferenceModelPreferences preferences,
        AsrModelOverride asrModelOverride)
    {
        if (preferences.PreferredExecutionProviders is not { Count: > 0 } providers)
        {
            return new Dictionary<string, ExecutionProviderKind>();
        }

        var overrides = new Dictionary<string, ExecutionProviderKind>(StringComparer.OrdinalIgnoreCase);
        foreach ((RuntimeStage stage, ExecutionProviderKind provider) in providers)
        {
            if (preferences.ExplicitExecutionProviderStages is not null &&
                !preferences.ExplicitExecutionProviderStages.Contains(stage))
            {
                continue;
            }

            string key = stage switch
            {
                RuntimeStage.Asr => asrModelOverride switch
                {
                    AsrModelOverride.GenAi => "AsrGenAi",
                    AsrModelOverride.OnnxRuntime => "AsrOnnxRuntime",
                    AsrModelOverride.Nemotron35 => "AsrNemotron",
                    _ => "Asr",
                },
                _ => stage.ToString(),
            };
            overrides[key] = provider;
        }

        return overrides;
    }
}
