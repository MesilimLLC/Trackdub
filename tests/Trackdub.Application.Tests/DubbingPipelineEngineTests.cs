using Trackdub.Application.Dubbing;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Contracts.Dubbing;
using Trackdub.Domain;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Tests;

public sealed class DubbingPipelineEngineTests
{
    [Fact]
    public void ExplicitVariantWinsOverPersistedSelection()
    {
        var selections = new RuntimeModelSelections(AsrModelOverride.Auto, false, new Dictionary<string, ExecutionProviderKind>(),
            ModelVariantOverrides: new Dictionary<string, string> { ["translation"] = "int4-kv", [ModelVariantOverrideKeys.Build("translation", "madlad400")] = "int4-kv" });
        var preferences = new InferenceModelPreferences(PreferredModelVariantAliases:
            new Dictionary<RuntimeStage, string> { [RuntimeStage.Translation] = "quantized" });
        var merged = DubbingModelPreferences.ApplyModelPreferenceAliases(selections, preferences);
        Assert.Equal("quantized", merged.ModelVariantOverrides!["translation"]);
        Assert.False(merged.ModelVariantOverrides.ContainsKey(ModelVariantOverrideKeys.Build("translation", "madlad400")));
    }
    [Fact]
    public async Task RunStageWorkflowAsync_unknown_stage_fails_with_unsupported_reason()
    {
        var session = new FakeSession();
        var options = new DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "es",
        };
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            new Dictionary<string, ExecutionProviderKind>());

        DubbingPipelineEngine.StageWorkflowResult result =
            await DubbingPipelineEngine.RunStageWorkflowAsync(
                new DubbingPipelineEngine.StageExecutionContext
                {
                    Session = session,
                    Options = options,
                    RuntimeSelections = selections,
                },
                "bogus-stage",
                CancellationToken.None);

        Assert.Equal(StageStatus.Failed, result.Status);
        Assert.Equal("STAGE_UNSUPPORTED", result.ReasonCode);
    }

    [Fact]
    public void ExtendedStageOrder_contains_shared_utility_stages_in_pipeline_position()
    {
        IReadOnlyList<string> order = DubbingPipelineStages.ExtendedStageOrder;

        Assert.Contains(StageNames.AudioPreparation, order);
        Assert.Contains(StageNames.TextRefinementAsr, order);
        Assert.Contains(StageNames.OverlapRescue, order);

        Assert.True(IndexOf(order, StageNames.AudioPreparation) < IndexOf(order, StageNames.Vad));
        Assert.True(IndexOf(order, StageNames.OverlapRescue) > IndexOf(order, StageNames.Diarization));
        Assert.True(IndexOf(order, StageNames.TextRefinementAsr) > IndexOf(order, StageNames.Asr));
        Assert.True(IndexOf(order, StageNames.TextRefinementAsr) < IndexOf(order, StageNames.Translation));
    }

    [Fact]
    public void ApplyModelPreferenceAliases_pins_explicit_alias_over_ui_selection()
    {
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            new Dictionary<string, ExecutionProviderKind>(),
            VadModelAlias: "silero-ui-pick",
            AsrModelAlias: "whisper-ui-pick",
            TtsModelAlias: "kokoro-ui-pick");
        var preferences = new InferenceModelPreferences(
            VadModelAlias: "silero-request-pin",
            TtsModelAlias: "chatterbox-clone-pin");

        RuntimeModelSelections merged = DubbingModelPreferences.ApplyModelPreferenceAliases(selections, preferences);

        Assert.Equal("silero-request-pin", merged.VadModelAlias);
        Assert.Equal("chatterbox-clone-pin", merged.TtsModelAlias);
        Assert.Equal("whisper-ui-pick", merged.AsrModelAlias);
    }

    [Fact]
    public void ApplyModelPreferenceAliases_keeps_vad_selection_without_explicit_alias()
    {
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            new Dictionary<string, ExecutionProviderKind>(),
            VadModelAlias: "silero-ui-pick");
        var preferences = new InferenceModelPreferences(TtsModelAlias: "kokoro-request-pin");

        RuntimeModelSelections merged = DubbingModelPreferences.ApplyModelPreferenceAliases(selections, preferences);

        Assert.Equal("silero-ui-pick", merged.VadModelAlias);
    }

    [Fact]
    public void ApplyModelPreferenceAliases_with_null_preferences_keeps_host_selections()
    {
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            new Dictionary<string, ExecutionProviderKind>(),
            VadModelAlias: "silero-ui-pick",
            TtsModelAlias: "kokoro-ui-pick");

        RuntimeModelSelections merged = DubbingModelPreferences.ApplyModelPreferenceAliases(selections, null);

        Assert.Equal("silero-ui-pick", merged.VadModelAlias);
        Assert.Equal("kokoro-ui-pick", merged.TtsModelAlias);
    }

    [Fact]
    public void PrerequisiteStages_does_not_contain_Separation()
    {
        Assert.DoesNotContain(StageNames.Separation, DubbingPipelineStages.PrerequisiteStages);
    }

    [Fact]
    public void PrerequisiteStages_contains_expected_core_stages()
    {
        var expected = new[]
        {
            StageNames.Vad,
            StageNames.Asr,
            StageNames.Translation,
            StageNames.Tts,
        };

        foreach (string stage in expected)
        {
            Assert.Contains(stage, DubbingPipelineStages.PrerequisiteStages);
        }
    }

    [Fact]
    public void DubbingRunResult_CorrelationId_defaults_to_empty()
    {
        var result = new DubbingRunResult
        {
            RunId = Guid.NewGuid(),
            StartTime = DateTimeOffset.UtcNow,
            EndTime = DateTimeOffset.UtcNow,
            OverallStatus = DubbingRunStatus.Succeeded,
            StageOutcomes = [],
        };

        Assert.Equal(Guid.Empty, result.CorrelationId);
    }

    [Fact]
    public void DubbingRunResult_CorrelationId_can_be_set_independently_of_RunId()
    {
        var runId = Guid.NewGuid();
        var correlationId = Guid.NewGuid();
        var result = new DubbingRunResult
        {
            RunId = runId,
            CorrelationId = correlationId,
            StartTime = DateTimeOffset.UtcNow,
            EndTime = DateTimeOffset.UtcNow,
            OverallStatus = DubbingRunStatus.Succeeded,
            StageOutcomes = [],
        };

        Assert.Equal(runId, result.RunId);
        Assert.Equal(correlationId, result.CorrelationId);
        Assert.NotEqual(result.RunId, result.CorrelationId);
    }

    [Fact]
    public void BuildModelPreferences_maps_provider_pins_to_runtime_stages()
    {
        var options = new DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "es",
            ExecutionProviderPreferences = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["ASR"] = "cpu",
                [StageNames.Tts] = "directml",
            },
            RequireExecutionProviderPreferences = true,
        };

        InferenceModelPreferences? preferences = DubbingModelPreferences.BuildModelPreferences(options);

        Assert.NotNull(preferences);
        Assert.Equal(ExecutionProviderKind.Cpu, preferences!.GetPreferredExecutionProvider(RuntimeStage.Asr));
        Assert.Equal(ExecutionProviderKind.DirectMl, preferences.GetPreferredExecutionProvider(RuntimeStage.Tts));
        Assert.True(preferences.RequiresPreferredExecutionProvider(RuntimeStage.Asr));
        Assert.True(preferences.RequiresPreferredExecutionProvider(RuntimeStage.Tts));
        Assert.False(preferences.RequiresPreferredExecutionProvider(RuntimeStage.Vad));
        Assert.NotNull(preferences.RequiredExecutionProviderStages);
        Assert.True(preferences.RequiredExecutionProviderStages!.SetEquals(
            [RuntimeStage.Asr, RuntimeStage.Tts]));
    }

    [Fact]
    public void BuildModelPreferences_provider_pins_are_preferences_only_without_require_flag()
    {
        var options = new DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "es",
            ExecutionProviderPreferences = new Dictionary<string, string>
            {
                [StageNames.Asr] = "cpu",
            },
        };

        InferenceModelPreferences? preferences = DubbingModelPreferences.BuildModelPreferences(options);

        Assert.NotNull(preferences);
        Assert.Equal(ExecutionProviderKind.Cpu, preferences!.GetPreferredExecutionProvider(RuntimeStage.Asr));
        Assert.Null(preferences.RequiredExecutionProviderStages);
        Assert.False(preferences.RequiresPreferredExecutionProvider(RuntimeStage.Asr));
    }

    [Theory]
    [InlineData(StageNames.Export)]
    [InlineData(StageNames.SpeakerAssignment)]
    [InlineData("not-a-stage")]
    public void BuildModelPreferences_rejects_non_runtime_stage_provider_pin(string stage)
    {
        var options = new DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "es",
            ExecutionProviderPreferences = new Dictionary<string, string>
            {
                [stage] = "cpu",
            },
        };

        Assert.Throws<ArgumentException>(() => DubbingModelPreferences.BuildModelPreferences(options));
    }

    [Fact]
    public void BuildModelPreferences_rejects_invalid_provider_label()
    {
        var options = new DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "es",
            ExecutionProviderPreferences = new Dictionary<string, string>
            {
                [StageNames.Asr] = "definitely-not-a-provider",
            },
        };

        Assert.Throws<ArgumentException>(() => DubbingModelPreferences.BuildModelPreferences(options));
    }

    private static int IndexOf(IReadOnlyList<string> order, string stageName)
    {
        for (int i = 0; i < order.Count; i++)
        {
            if (string.Equals(order[i], stageName, StringComparison.OrdinalIgnoreCase))
            {
                return i;
            }
        }

        return -1;
    }

    [Fact]
    public async Task ExecuteAsync_attaches_cold_start_evidence_to_the_run_result()
    {
        // The run fails validation (no known stage in the filter) before any I/O; the
        // evidence must still be attached (reporting "no inference completed") rather than
        // omitted, because a cold run that never reached inference is itself the finding.
        var engine = new DubbingPipelineEngine(new FakeSessionFactory());
        var options = new DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "es",
            StageFilter = ["bogus-stage"],
        };

        DubbingRunResult result = await engine.ExecuteAsync(
            options, progress: null, TestContext.Current.CancellationToken);

        Assert.NotNull(result.ColdStartTiming);
        Assert.NotNull(result.PreFlightFailures);
        Assert.True(result.ColdStartTiming.RunElapsedMs >= 0);
        Assert.Null(result.ColdStartTiming.FirstInferenceCompletionElapsedMs);
    }

    [Fact]
    public async Task RunStageWorkflowAsync_attributes_completions_to_the_stage()
    {
        // A skipped stage performs no inference, so nothing may be recorded as stage
        // inference; the stage start marker must still name the stage.
        var capture = new ColdStartCapture();
        var session = new FakeSession();
        var options = new DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "es",
            EnableStemSeparation = false,
        };

        DubbingPipelineEngine.StageWorkflowResult result;
        using (ColdStartCapture.Activate(capture))
        {
            result = await DubbingPipelineEngine.RunStageWorkflowAsync(
                new DubbingPipelineEngine.StageExecutionContext
                {
                    Session = session,
                    Options = options,
                    RuntimeSelections = new RuntimeModelSelections(
                        AsrModelOverride.Auto,
                        IsDevBuild: false,
                        new Dictionary<string, ExecutionProviderKind>()),
                },
                StageNames.Separation,
                TestContext.Current.CancellationToken);
        }

        Assert.Equal(StageStatus.Skipped, result.Status);
        Assert.Equal(StageNames.Separation, capture.BuildTiming().FirstStageStartedStageName);
        Assert.Null(capture.BuildTiming().FirstStageInferenceCompletionElapsedMs);
    }

    private sealed class FakeSessionFactory : IDubbingSessionFactory
    {
        public IDubbingSession CreateSession(string projectRootPath, StudioSettings? settings = null) =>
            throw new InvalidOperationException(
                "A validation-failure run must end before any session is created.");

        public void Dispose()
        {
        }
    }

    private sealed class FakeSession : IDubbingSession
    {
        public string ProjectRootPath => ".";

        // Only consumed by recognized stages; the unknown-stage arm never dereferences it.
        public TranscriptWorkspace Workspace => null!;

        public IServiceProvider Services { get; } = new EmptyServiceProvider();

        public void Dispose()
        {
        }

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    private sealed class EmptyServiceProvider : IServiceProvider
    {
        public object? GetService(Type serviceType) => null;
    }
}
