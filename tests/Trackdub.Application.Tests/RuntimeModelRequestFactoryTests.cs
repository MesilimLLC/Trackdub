using Trackdub.Contracts;
using Trackdub.Contracts.Licensing;
using Trackdub.Application.Transcripts;
using Trackdub.Domain;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Tests;

public sealed class RuntimeModelRequestFactoryTests
{
    [Fact]
    public void CreateModelPreferences_uses_asr_override_in_all_builds()
    {
        var devOptions = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.GenAi,
            IsDevBuild: true,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>());
        var releaseOptions = devOptions with { IsDevBuild = false };

        InferenceModelPreferences devPreferences = RuntimeModelRequestFactory.CreateModelPreferences(devOptions);
        InferenceModelPreferences releasePreferences = RuntimeModelRequestFactory.CreateModelPreferences(releaseOptions);

        Assert.Equal(AsrModelOverrideSettings.GenAiModelAlias, devPreferences.AsrModelAlias);
        Assert.True(devPreferences.RequireAsrModelAlias);
        Assert.Equal(AsrModelOverrideSettings.GenAiModelAlias, releasePreferences.AsrModelAlias);
        Assert.True(releasePreferences.RequireAsrModelAlias);
    }

    [Fact]
    public void CreateModelPreferences_maps_provider_overrides_and_requires_them_only_for_dev_builds()
    {
        var devOptions = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.GenAi,
            IsDevBuild: true,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>
            {
                ["Vad"] = ExecutionProviderKind.Cpu,
                ["AsrGenAi"] = ExecutionProviderKind.DirectMl,
                ["Separation"] = ExecutionProviderKind.TensorRTRtx
            });
        var releaseOptions = devOptions with { IsDevBuild = false };

        InferenceModelPreferences devPreferences = RuntimeModelRequestFactory.CreateModelPreferences(devOptions);
        InferenceModelPreferences releasePreferences = RuntimeModelRequestFactory.CreateModelPreferences(releaseOptions);

        Assert.Equal(ExecutionProviderKind.Cpu, devPreferences.GetPreferredExecutionProvider(RuntimeStage.Vad));
        Assert.Equal(ExecutionProviderKind.DirectMl, devPreferences.GetPreferredExecutionProvider(RuntimeStage.Asr));
        Assert.Equal(ExecutionProviderKind.TensorRTRtx, devPreferences.GetPreferredExecutionProvider(RuntimeStage.Separation));
        Assert.True(devPreferences.RequiresPreferredExecutionProvider(RuntimeStage.Asr));
        Assert.True(devPreferences.RequiresPreferredExecutionProvider(RuntimeStage.Separation));
        Assert.Equal(ExecutionProviderKind.DirectMl, releasePreferences.GetPreferredExecutionProvider(RuntimeStage.Asr));
        Assert.False(releasePreferences.RequiresPreferredExecutionProvider(RuntimeStage.Asr));
    }

    [Fact]
    public void CreateModelPreferences_requires_provider_overrides_when_headless_pin_flag_set()
    {
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: false,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>
            {
                ["Asr"] = ExecutionProviderKind.Qnn,
                ["Tts"] = ExecutionProviderKind.TensorRTRtx
            },
            RequirePreferredExecutionProviders: true);

        InferenceModelPreferences preferences = RuntimeModelRequestFactory.CreateModelPreferences(options);

        Assert.True(preferences.RequiresPreferredExecutionProvider(RuntimeStage.Asr));
        Assert.True(preferences.RequiresPreferredExecutionProvider(RuntimeStage.Tts));
        Assert.Equal(ExecutionProviderKind.Qnn, preferences.GetPreferredExecutionProvider(RuntimeStage.Asr));
    }

    [Fact]
    public void Provider_preferences_round_trip_preserves_global_fallback_and_explicit_stage()
    {
        var options = new RuntimeModelRequestOptions(AsrModelOverride.Auto, true,
            new Dictionary<string, ExecutionProviderKind> { ["Asr"] = ExecutionProviderKind.Cpu },
            PreferredExecutionProvider: ExecutionProviderKind.DirectMl);
        InferenceModelPreferences preferences = RuntimeModelRequestFactory.CreateModelPreferences(options);
        RuntimeModelSelections selections = RuntimeModelRequestFactory.CreateSelectionsFromPreferences(preferences, isDevBuild: true);
        InferenceModelPreferences result = RuntimeModelRequestFactory.CreateModelPreferences(selections);

        Assert.Equal(ExecutionProviderKind.DirectMl, result.GetPreferredExecutionProvider(RuntimeStage.Tts));
        Assert.False(result.RequiresPreferredExecutionProvider(RuntimeStage.Tts));
        Assert.Equal(ExecutionProviderKind.Cpu, result.GetPreferredExecutionProvider(RuntimeStage.Asr));
        Assert.True(result.RequiresPreferredExecutionProvider(RuntimeStage.Asr));
        Assert.Single(selections.HardwareOverrides);

        RuntimeModelSelections releaseSelections = RuntimeModelRequestFactory.CreateSelectionsFromPreferences(preferences);
        InferenceModelPreferences releaseResult = RuntimeModelRequestFactory.CreateModelPreferences(releaseSelections);
        Assert.True(releaseResult.RequiresPreferredExecutionProvider(RuntimeStage.Asr));
        Assert.False(releaseResult.RequiresPreferredExecutionProvider(RuntimeStage.Tts));
    }

    [Fact]
    public void CreateSelectionsFromPreferences_maps_model_aliases()
    {
        var preferences = new InferenceModelPreferences(
            VadModelAlias: "silero-custom",
            AsrModelAlias: "whisper-large",
            DiarizationModelAlias: "pyannote");

        RuntimeModelSelections selections =
            RuntimeModelRequestFactory.CreateSelectionsFromPreferences(preferences);

        InferenceModelPreferences mapped = RuntimeModelRequestFactory.CreateModelPreferences(selections);

        Assert.Equal("silero-custom", mapped.VadModelAlias);
        Assert.Equal("whisper-large", mapped.AsrModelAlias);
        Assert.Equal("pyannote", mapped.DiarizationModelAlias);
    }

    [Theory]
    [InlineData(AsrModelOverride.Auto, "Asr")]
    [InlineData(AsrModelOverride.GenAi, "AsrGenAi")]
    [InlineData(AsrModelOverride.OnnxRuntime, "AsrOnnxRuntime")]
    [InlineData(AsrModelOverride.Nemotron35, "AsrNemotron")]
    public void CreateSelectionsFromPreferences_maps_asr_provider_from_effective_override(
        AsrModelOverride asrModelOverride,
        string expectedHardwareKey)
    {
        var preferences = new InferenceModelPreferences(
            AsrModelAlias: AsrModelOverrideSettings.GenAiModelAlias,
            PreferredExecutionProviders: new Dictionary<RuntimeStage, ExecutionProviderKind>
            {
                [RuntimeStage.Asr] = ExecutionProviderKind.Cpu,
            });

        RuntimeModelSelections selections = RuntimeModelRequestFactory.CreateSelectionsFromPreferences(
            preferences,
            asrModelOverride,
            isDevBuild: true);

        Assert.True(selections.IsDevBuild);
        KeyValuePair<string, ExecutionProviderKind> hardwareOverride = Assert.Single(selections.HardwareOverrides);
        Assert.Equal(expectedHardwareKey, hardwareOverride.Key);
        Assert.Equal(ExecutionProviderKind.Cpu, hardwareOverride.Value);
    }

    [Fact]
    public void CreateSelectionsFromPreferences_maps_lip_aliases()
    {
        var preferences = new InferenceModelPreferences(
            LipSyncModelAlias: "latentsync",
            LipSynthesisModelAlias: "musetalk");

        RuntimeModelSelections selections =
            RuntimeModelRequestFactory.CreateSelectionsFromPreferences(preferences);

        Assert.Equal("latentsync", selections.LipSyncModelAlias);
        Assert.Equal("musetalk", selections.LipSynthesisModelAlias);
    }

    [Fact]
    public void CreateModelPreferences_maps_lip_aliases_round_trip()
    {
        var selections = new RuntimeModelSelections(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: false,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>(),
            LipSyncModelAlias: "wav2vec2-lv60-espeak-cv-ft-onnx",
            LipSynthesisModelAlias: "ByteDance/LatentSync-1.6");

        InferenceModelPreferences preferences = RuntimeModelRequestFactory.CreateModelPreferences(selections);

        Assert.Equal("wav2vec2-lv60-espeak-cv-ft-onnx", preferences.LipSyncModelAlias);
        Assert.Equal("ByteDance/LatentSync-1.6", preferences.LipSynthesisModelAlias);
    }

    [Fact]
    public void CreateSelectionsFromSettings_applies_global_provider_before_stage_overrides()
    {
        StudioSettings settings = StudioSettings.Default with
        {
            AsrModelOverride = AsrModelOverride.GenAi,
            PreferredExecutionProvider = ExecutionProviderKind.DirectMl,
            HardwareOverrides = new Dictionary<string, ExecutionProviderKind>
            {
                ["AsrGenAi"] = ExecutionProviderKind.Cpu,
                ["Tts"] = ExecutionProviderKind.TensorRTRtx,
            },
        };

        RuntimeModelSelections selections = RuntimeModelRequestFactory.CreateSelectionsFromSettings(settings, isDevBuild: true);
        AssertSelectionPreferences(selections);

        InferenceModelPreferences preferences = RuntimeModelRequestFactory.CreateModelPreferences(selections);
        AssertGlobalPreferenceIsSoft(preferences);
        AssertStageOverridesRemainRequired(preferences);
    }

    [Fact]
    public void CreateSelectionsFromSettings_maps_undefined_global_provider_to_null()
    {
        StudioSettings settings = StudioSettings.Default with
        {
            PreferredExecutionProvider = (ExecutionProviderKind)int.MaxValue,
        };

        RuntimeModelSelections selections = RuntimeModelRequestFactory.CreateSelectionsFromSettings(settings, isDevBuild: true);

        Assert.Null(selections.PreferredExecutionProvider);
    }

    [Fact]
    public void CreateDiarizationRequest_maps_undefined_stage_override_to_automatic_selection()
    {
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: true,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>
            {
                [nameof(RuntimeStage.Diarization)] = (ExecutionProviderKind)int.MaxValue,
            });

        RuntimeModelRequest request = RuntimeModelRequestFactory.CreateDiarizationRequest(options);

        Assert.Null(request.PreferredExecutionProvider);
        Assert.False(request.RequirePreferredExecutionProvider);
    }

    [Fact]
    public void CreateSelectionsFromSettings_uses_stage_aliases_from_starter_pack()
    {
        StudioSettings settings = StudioSettings.Default with
        {
            AppliedStarterPackId = "lip-pack",
            StageModelAliases = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                [StageNames.Vad] = "silero-vad-custom",
                [StageNames.LipSync] = "wav2vec2-lv60-espeak-cv-ft-onnx",
                [StageNames.LipSynthesis] = "ByteDance/LatentSync-1.6",
            },
        };

        RuntimeModelSelections selections = RuntimeModelRequestFactory.CreateSelectionsFromSettings(settings);

        Assert.Equal("silero-vad-custom", selections.VadModelAlias);
        Assert.Equal("wav2vec2-lv60-espeak-cv-ft-onnx", selections.LipSyncModelAlias);
        Assert.Equal("ByteDance/LatentSync-1.6", selections.LipSynthesisModelAlias);
    }

    [Fact]
    public void CreateOptions_maps_ui_runtime_selections_to_request_options()
    {
        var selections = new RuntimeModelSelections(
            AsrModelOverride: AsrModelOverride.OnnxRuntime,
            IsDevBuild: true,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>
            {
                ["AsrOnnxRuntime"] = ExecutionProviderKind.DirectMl
            });

        RuntimeModelRequestOptions options = RuntimeModelRequestFactory.CreateOptions(selections);

        Assert.Equal(AsrModelOverride.OnnxRuntime, options.AsrModelOverride);
        Assert.True(options.IsDevBuild);
        Assert.Equal(ExecutionProviderKind.DirectMl, options.HardwareOverrides["AsrOnnxRuntime"]);
    }

    [Fact]
    public void CreateOptions_maps_asr_text_refinement_toggle_and_model_alias()
    {
        var selections = new RuntimeModelSelections(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: false,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>(),
            TextRefinementModelAlias: "qwen-polisher",
            EnableAsrTextRefinement: true);

        RuntimeModelRequestOptions options = RuntimeModelRequestFactory.CreateOptions(selections);
        InferenceModelPreferences preferences = RuntimeModelRequestFactory.CreateModelPreferences(options);

        Assert.True(options.EnableAsrTextRefinement);
        Assert.Equal("qwen-polisher", options.TextRefinementModelAlias);
        Assert.True(preferences.EnableAsrTextRefinement);
        Assert.Equal("qwen-polisher", preferences.TextRefinementModelAlias);
    }

    [Fact]
    public void CreateTranslationRequest_maps_selected_stage_variant_override()
    {
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: true,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>(),
            ModelVariantOverrides: new Dictionary<string, string>
            {
                ["translation"] = "olive-cpu-fp32"
            });

        RuntimeModelRequest request = RuntimeModelRequestFactory.CreateTranslationRequest(options, "en", "es");
        InferenceModelPreferences preferences = RuntimeModelRequestFactory.CreateModelPreferences(options);

        Assert.Equal(RuntimeStage.Translation, request.Stage);
        Assert.Equal("olive-cpu-fp32", request.PreferredModelVariantAlias);
        Assert.Equal("olive-cpu-fp32", preferences.GetPreferredModelVariantAlias(RuntimeStage.Translation));
    }

    [Fact]
    public void CreateTranslationRequest_prefers_model_scoped_variant_override_over_stage_override()
    {
        const string modelAlias = TranslationModelOverrideSettings.MadladModelAlias;
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: true,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>(),
            TranslationModelOverride: TranslationModelOverride.Madlad,
            ModelVariantOverrides: new Dictionary<string, string>
            {
                ["translation"] = "olive-cpu-fp32",
                [ModelVariantOverrideKeys.Build("translation", modelAlias)] = "olive-cpu-int8"
            });

        RuntimeModelRequest request = RuntimeModelRequestFactory.CreateTranslationRequest(options, "en", "es");

        Assert.Equal("olive-cpu-int8", request.PreferredModelVariantAlias);
    }

    [Fact]
    public void CreateTranslationRequest_maps_deepl_cloud_override_without_local_model_variant()
    {
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: false,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>(),
            TranslationModelOverride: TranslationModelOverride.DeepL);

        RuntimeModelRequest request = RuntimeModelRequestFactory.CreateTranslationRequest(options, "en", "es");
        InferenceModelPreferences preferences = RuntimeModelRequestFactory.CreateModelPreferences(options);

        Assert.Equal(RuntimeStage.Translation, request.Stage);
        Assert.Equal(TranslationModelOverrideSettings.DeepLModelAlias, request.PreferredModelAlias);
        Assert.True(request.RequirePreferredModelAlias);
        Assert.Null(request.PreferredModelVariantAlias);
        Assert.Equal(TranslationModelOverrideSettings.DeepLModelAlias, preferences.TranslationModelAlias);
    }

    [Fact]
    public void CreateAsrRequest_maps_override_specific_provider_key()
    {
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.OnnxRuntime,
            IsDevBuild: true,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>
            {
                ["Asr"] = ExecutionProviderKind.Cpu,
                ["AsrOnnxRuntime"] = ExecutionProviderKind.DirectMl
            });

        RuntimeModelRequest request = RuntimeModelRequestFactory.CreateAsrRequest(options, sourceLanguageCode: "es");

        Assert.Equal(RuntimeStage.Asr, request.Stage);
        Assert.Equal("es", request.SourceLanguage);
        Assert.Equal(AsrModelOverrideSettings.OnnxRuntimeModelAlias, request.PreferredModelAlias);
        Assert.True(request.RequirePreferredModelAlias);
        Assert.Equal(ExecutionProviderKind.DirectMl, request.PreferredExecutionProvider);
        Assert.True(request.RequirePreferredExecutionProvider);
    }

    [Fact]
    public void CreateAsrRequest_maps_nemotron_override_and_provider_key()
    {
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Nemotron35,
            IsDevBuild: true,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>
            {
                ["Asr"] = ExecutionProviderKind.Cpu,
                ["AsrNemotron"] = ExecutionProviderKind.DirectMl
            });

        RuntimeModelRequest request = RuntimeModelRequestFactory.CreateAsrRequest(options);

        Assert.Equal(RuntimeStage.Asr, request.Stage);
        Assert.Equal(AsrModelOverrideSettings.Nemotron35ModelAlias, request.PreferredModelAlias);
        Assert.True(request.RequirePreferredModelAlias);
        Assert.Equal(ExecutionProviderKind.DirectMl, request.PreferredExecutionProvider);
        Assert.True(request.RequirePreferredExecutionProvider);
    }

    [Fact]
    public void CreateImportRequests_includes_optional_separation_only_when_enabled()
    {
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: true,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>
            {
                ["Vad"] = ExecutionProviderKind.Cpu,
                ["Separation"] = ExecutionProviderKind.TensorRTRtx
            });

        RuntimeModelRequest[] requests = RuntimeModelRequestFactory
            .CreateImportRequests(options, enableStemSeparation: true, sourceLanguageCode: "fr")
            .ToArray();

        Assert.Equal([RuntimeStage.Vad, RuntimeStage.Asr, RuntimeStage.Separation], requests.Select(request => request.Stage));
        Assert.Equal(ExecutionProviderKind.Cpu, requests[0].PreferredExecutionProvider);
        Assert.True(requests[0].RequirePreferredExecutionProvider);
        Assert.Equal("fr", requests[1].SourceLanguage);
        Assert.Equal(ExecutionProviderKind.TensorRTRtx, requests[2].PreferredExecutionProvider);
        Assert.True(requests[2].RequirePreferredExecutionProvider);

        RuntimeModelRequest[] noSeparation = RuntimeModelRequestFactory
            .CreateImportRequests(options, enableStemSeparation: false)
            .ToArray();
        Assert.Equal([RuntimeStage.Vad, RuntimeStage.Asr], noSeparation.Select(request => request.Stage));
    }

    [Fact]
    public void CreateDiarizationRequest_maps_preferred_diarization_model_alias()
    {
        const string diarizationAlias = "diar-streaming-sortformer-4spk-v2.1";
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: true,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>(),
            DiarizationModelAlias: diarizationAlias);

        RuntimeModelRequest request = RuntimeModelRequestFactory.CreateDiarizationRequest(options);
        InferenceModelPreferences preferences = RuntimeModelRequestFactory.CreateModelPreferences(options);
        RerunDiarizationRequest rerunRequest = RuntimeModelRequestFactory.CreateRerunDiarizationRequest(options);

        Assert.Equal(RuntimeStage.Diarization, request.Stage);
        Assert.Equal(diarizationAlias, request.PreferredModelAlias);
        Assert.True(request.RequirePreferredModelAlias);
        Assert.Equal(diarizationAlias, preferences.DiarizationModelAlias);
        Assert.Equal(diarizationAlias, rerunRequest.PreferredModelAlias);
    }

    [Fact]
    public void CreateStemRerunRequests_builds_requests_from_view_model_stage_plan()
    {
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: true,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>());

        const string diarizationAlias = "sortformer-4spk";
        var optionsWithDiarization = options with { DiarizationModelAlias = diarizationAlias };

        RuntimeModelRequest[] requests = RuntimeModelRequestFactory
            .CreateStemRerunRequests(
                optionsWithDiarization,
                [RuntimeStage.Separation, RuntimeStage.Vad, RuntimeStage.Asr, RuntimeStage.Diarization])
            .ToArray();

        Assert.Equal(
            [RuntimeStage.Separation, RuntimeStage.Vad, RuntimeStage.Asr, RuntimeStage.Diarization],
            requests.Select(request => request.Stage));
        Assert.Equal(diarizationAlias, requests.Single(request => request.Stage == RuntimeStage.Diarization).PreferredModelAlias);
        Assert.True(requests.Single(request => request.Stage == RuntimeStage.Diarization).RequirePreferredModelAlias);
    }

    [Fact]
    public void CreateTtsRequest_requires_chatterbox_alias_for_voice_clone()
    {
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: true,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>());

        RuntimeModelRequest request = RuntimeModelRequestFactory.CreateTtsRequest(options, requiresVoiceClone: true);

        Assert.Equal(RuntimeStage.Tts, request.Stage);
        Assert.Equal(VoiceCloningDefaults.ChatterboxPrimaryAlias, request.PreferredModelAlias);
        Assert.True(request.RequirePreferredModelAlias);
    }

    [Fact]
    public void CreateTtsRequest_requires_kokoro_alias_for_stock_synthetic_voice()
    {
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: true,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>());

        RuntimeModelRequest request = RuntimeModelRequestFactory.CreateTtsRequest(options, requiresVoiceClone: false);

        Assert.Equal(RuntimeStage.Tts, request.Stage);
        Assert.Equal("kokoro-onnx", request.PreferredModelAlias);
        Assert.True(request.RequirePreferredModelAlias);
    }

    [Fact]
    public void CreateStageRequest_builds_lip_sync_request()
    {
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: true,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>
            {
                ["LipSync"] = ExecutionProviderKind.Cpu
            });

        RuntimeModelRequest request = RuntimeModelRequestFactory.CreateStageRequest(options, RuntimeStage.LipSync);

        Assert.Equal(RuntimeStage.LipSync, request.Stage);
        Assert.Equal(ExecutionProviderKind.Cpu, request.PreferredExecutionProvider);
        Assert.True(request.RequirePreferredExecutionProvider);
    }

    [Fact]
    public void CreateLipSynthesisRequest_uses_preferred_alias_and_stage()
    {
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: false,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>());

        RuntimeModelRequest request = RuntimeModelRequestFactory.CreateLipSynthesisRequest(
            options,
            "ByteDance/LatentSync-1.6");

        Assert.Equal(RuntimeStage.LipSynthesis, request.Stage);
        Assert.Equal("ByteDance/LatentSync-1.6", request.PreferredModelAlias);
        Assert.True(request.RequirePreferredModelAlias);
    }

    [Fact]
    public void CreateLipSyncRequest_reads_alias_from_options_when_not_passed_explicitly()
    {
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: false,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>(),
            LipSyncModelAlias: "wav2vec2-lv60-espeak-cv-ft-onnx");

        RuntimeModelRequest request = RuntimeModelRequestFactory.CreateLipSyncRequest(options);

        Assert.Equal(RuntimeStage.LipSync, request.Stage);
        Assert.Equal("wav2vec2-lv60-espeak-cv-ft-onnx", request.PreferredModelAlias);
        Assert.True(request.RequirePreferredModelAlias);
    }

    [Fact]
    public void CreateLipSynthesisRequest_reads_alias_from_options_when_not_passed_explicitly()
    {
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Auto,
            IsDevBuild: false,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>(),
            LipSynthesisModelAlias: "ByteDance/LatentSync-1.6");

        RuntimeModelRequest request = RuntimeModelRequestFactory.CreateLipSynthesisRequest(options);

        Assert.Equal(RuntimeStage.LipSynthesis, request.Stage);
        Assert.Equal("ByteDance/LatentSync-1.6", request.PreferredModelAlias);
        Assert.True(request.RequirePreferredModelAlias);
    }

    [Fact]
    public void CreateTtsRequest_uses_cosyvoice_alias_when_override_is_explicit()
    {
        var options = new RuntimeModelRequestOptions(
            AsrModelOverride: AsrModelOverride.Auto,
            TtsModelOverride: TtsModelOverride.CosyVoice,
            IsDevBuild: true,
            HardwareOverrides: new Dictionary<string, ExecutionProviderKind>());

        RuntimeModelRequest request = RuntimeModelRequestFactory.CreateTtsRequest(options, requiresVoiceClone: true);

        Assert.Equal(RuntimeStage.Tts, request.Stage);
        Assert.Equal("cosyvoice-300m", request.PreferredModelAlias);
        Assert.True(request.RequirePreferredModelAlias);
    }

    private static void AssertSelectionPreferences(RuntimeModelSelections selections)
    {
        Assert.Equal(ExecutionProviderKind.DirectMl, selections.PreferredExecutionProvider);
        Assert.Equal(ExecutionProviderKind.Cpu, selections.HardwareOverrides["AsrGenAi"]);
        Assert.Equal(ExecutionProviderKind.TensorRTRtx, selections.HardwareOverrides["Tts"]);
    }

    private static void AssertGlobalPreferenceIsSoft(InferenceModelPreferences preferences)
    {
        Assert.Equal(ExecutionProviderKind.DirectMl, preferences.GetPreferredExecutionProvider(RuntimeStage.Vad));
        Assert.Equal(ExecutionProviderKind.DirectMl, preferences.GetPreferredExecutionProvider(RuntimeStage.Translation));
        AssertGlobalPreferenceDoesNotRequireProvider(preferences);
    }

    private static void AssertGlobalPreferenceDoesNotRequireProvider(InferenceModelPreferences preferences)
    {
        Assert.False(preferences.RequiresPreferredExecutionProvider(RuntimeStage.Vad));
        Assert.False(preferences.RequiresPreferredExecutionProvider(RuntimeStage.Translation));
    }

    private static void AssertStageOverridesRemainRequired(InferenceModelPreferences preferences)
    {
        Assert.True(preferences.RequiresPreferredExecutionProvider(RuntimeStage.Asr));
        Assert.True(preferences.RequiresPreferredExecutionProvider(RuntimeStage.Tts));
    }
}
