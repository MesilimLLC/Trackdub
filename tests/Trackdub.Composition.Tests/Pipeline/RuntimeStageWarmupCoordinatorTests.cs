using Microsoft.Extensions.DependencyInjection;
using Trackdub.Application.Pipeline;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Pipeline;
using Trackdub.Composition;
using Trackdub.Composition.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Worker;
using Trackdub.Inference.Runtime.Planning;
using Trackdub.TestDoubles;
using Xunit;

namespace Trackdub.Composition.Tests.Pipeline;

public class RuntimeStageWarmupCoordinatorTests
{
    private static RuntimeModelSelections Selections(
        IReadOnlyDictionary<string, ExecutionProviderKind>? hardwareOverrides = null,
        string? translationModelAlias = null) =>
        new(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            HardwareOverrides: hardwareOverrides ?? new Dictionary<string, ExecutionProviderKind>(),
            TranslationModelAlias: translationModelAlias);

    private static StageRuntimePlan RunnablePlan(StageRuntimePlanningRequest request) =>
        new()
        {
            Stage = request.Stage,
            Status = StageRuntimePlanStatus.Verified,
            ModelId = "model-x",
            ModelAlias = "alias-x",
            EngineFamily = "family-x",
            Variant = "fp16",
            ExecutionProvider = ExecutionProviderKind.DirectMl,
            ModelRootPath = "/models/x",
            ModelEntryPath = "/models/x/graph.onnx",
            ModelRevisionHash = "deadbeef",
        };

    [Fact]
    public async Task WarmAsync_Translation_RoutesSelectionsThroughPlannerIntoOneSmoke()
    {
        StageRuntimePlanningRequest? capturedPlan = null;
        var planner = new FakeRuntimePlanner
        {
            PlanHandler = req =>
            {
                capturedPlan = req;
                return RunnablePlan(req);
            },
        };
        ExecutionProviderSmokeTestRequest? capturedSmoke = null;
        var smoke = new FakeExecutionProviderSmokeTester(req =>
        {
            capturedSmoke = req;
            return new ExecutionProviderSmokeTestResult(true);
        });
        var capture = new BenchmarkPhaseCapture();
        using IDisposable? activation = BenchmarkPhaseCapture.Activate(capture);
        var coordinator = new RuntimeStageWarmupCoordinator(planner, smoke);

        StageWarmupResult result = await coordinator.WarmAsync(
            new StageWarmupRequest(
                RuntimeStage.Translation,
                Selections(
                    new Dictionary<string, ExecutionProviderKind>
                    {
                        ["Translation"] = ExecutionProviderKind.DirectMl,
                    },
                    translationModelAlias: "opus-mt-en-es"),
                SourceLanguageCode: "pt-BR",
                TargetLanguageCode: "es-MX"),
            CancellationToken.None);

        Assert.True(result.Attempted);
        Assert.True(result.Succeeded);
        Assert.NotNull(capturedPlan);
        Assert.Equal(RuntimeStage.Translation, capturedPlan!.Stage);
        Assert.Equal("opus-mt-en-es", capturedPlan.PreferredModelAlias);
        Assert.True(capturedPlan.RequirePreferredModelAlias);
        Assert.Equal("pt-BR", capturedPlan.SourceLanguage);
        Assert.Equal("es-MX", capturedPlan.TargetLanguage);
        Assert.Equal(ExecutionProviderKind.DirectMl, capturedPlan.PreferredExecutionProvider);
        // Warmup planning is file-only: it must never run the planner's
        // authoritative (evicting) smoke on a plan-cache miss.
        Assert.True(capturedPlan.SkipProviderSmokeTest);

        Assert.NotNull(capturedSmoke);
        Assert.Equal(RuntimeStage.Translation, capturedSmoke!.Stage);
        Assert.Equal("model-x", capturedSmoke.ModelId);
        Assert.Equal("alias-x", capturedSmoke.ModelAlias);
        Assert.Equal("family-x", capturedSmoke.EngineFamily);
        Assert.Equal("fp16", capturedSmoke.Variant);
        Assert.Equal(ExecutionProviderKind.DirectMl, capturedSmoke.ExecutionProvider);
        Assert.Equal("/models/x", capturedSmoke.ModelRootPath);
        Assert.Equal("/models/x/graph.onnx", capturedSmoke.EntryPath);
        Assert.Equal("deadbeef", capturedSmoke.ModelRevisionHash);
        // Warmup smokes are optimization-only: they must never evict a resident model.
        Assert.True(capturedSmoke.NonEvictingWarmup);

        IReadOnlyDictionary<string, long> counters = capture.SnapshotCounters();
        Assert.Equal(1L, counters["stageWarmupAttempt"]);
        Assert.Equal(1L, counters["stageWarmupSuccess"]);
        Assert.False(counters.ContainsKey("stageWarmupFailure"));
    }

    [Fact]
    public async Task WarmAsync_NonRunnablePlan_AttemptsNothingAndNeverSmokes()
    {
        var planner = new FakeRuntimePlanner
        {
            PlanHandler = req => new StageRuntimePlan
            {
                Stage = req.Stage,
                Status = StageRuntimePlanStatus.DownloadRequired,
                Fallback = new RuntimePlanFallback(RuntimePlanFallbackCode.ModelNotCached, "cache miss"),
            },
        };
        int smokeCalls = 0;
        var smoke = new FakeExecutionProviderSmokeTester(_ =>
        {
            smokeCalls++;
            return new ExecutionProviderSmokeTestResult(true);
        });
        var coordinator = new RuntimeStageWarmupCoordinator(planner, smoke);

        StageWarmupResult result = await coordinator.WarmAsync(
            new StageWarmupRequest(RuntimeStage.Vad, Selections()),
            CancellationToken.None);

        Assert.False(result.Attempted);
        Assert.False(result.Succeeded);
        Assert.Contains("no runnable plan", result.Detail);
        Assert.Equal(0, smokeCalls);
    }

    [Fact]
    public async Task WarmAsync_SmokeNotAttempted_ReportsSkippedNotFailed()
    {
        var planner = new FakeRuntimePlanner { PlanHandler = RunnablePlan };
        var smoke = new FakeExecutionProviderSmokeTester(
            _ => new ExecutionProviderSmokeTestResult(
                false, "GenAI warmup skipped: would evict a resident model")
            {
                NotAttempted = true,
            });
        var capture = new BenchmarkPhaseCapture();
        using IDisposable? activation = BenchmarkPhaseCapture.Activate(capture);
        var coordinator = new RuntimeStageWarmupCoordinator(planner, smoke);

        StageWarmupResult result = await coordinator.WarmAsync(
            new StageWarmupRequest(RuntimeStage.TextRefinement, Selections()),
            CancellationToken.None);

        // Honest non-attempt: not a failure, and no readiness is implied either way.
        Assert.False(result.Attempted);
        Assert.False(result.Succeeded);
        Assert.Contains("would evict a resident model", result.Detail);

        IReadOnlyDictionary<string, long> counters = capture.SnapshotCounters();
        Assert.Equal(1L, counters["stageWarmupSkipped"]);
        Assert.False(counters.ContainsKey("stageWarmupFailure"));
        Assert.False(counters.ContainsKey("stageWarmupSuccess"));
    }

    [Fact]
    public async Task WarmAsync_SmokeFailure_ReturnsFailedResultWithoutThrowing()
    {
        var planner = new FakeRuntimePlanner { PlanHandler = RunnablePlan };
        var smoke = new FakeExecutionProviderSmokeTester(
            _ => new ExecutionProviderSmokeTestResult(false, "provider failed"));
        var coordinator = new RuntimeStageWarmupCoordinator(planner, smoke);

        StageWarmupResult result = await coordinator.WarmAsync(
            new StageWarmupRequest(RuntimeStage.Asr, Selections()),
            CancellationToken.None);

        Assert.True(result.Attempted);
        Assert.False(result.Succeeded);
        Assert.Equal("provider failed", result.Detail);
    }

    [Fact]
    public async Task WarmAsync_SmokeFailure_NonGenAiRoute_ReplansWithAuthoritativeSmokeAndWarmsFallback()
    {
        var requests = new List<StageRuntimePlanningRequest>();
        var planner = new FakeRuntimePlanner
        {
            PlanHandler = req =>
            {
                requests.Add(req);
                StageRuntimePlan plan = RunnablePlan(req);
                return req.SkipProviderSmokeTest
                    ? plan
                    : plan with { ExecutionProvider = ExecutionProviderKind.Cpu };
            },
        };
        var smokedProviders = new List<ExecutionProviderKind>();
        var smoke = new FakeExecutionProviderSmokeTester(req =>
        {
            smokedProviders.Add(req.ExecutionProvider);
            return new ExecutionProviderSmokeTestResult(req.ExecutionProvider != ExecutionProviderKind.DirectMl, "directml broken");
        });
        var coordinator = new RuntimeStageWarmupCoordinator(planner, smoke);

        StageWarmupResult result = await coordinator.WarmAsync(
            new StageWarmupRequest(RuntimeStage.Asr, Selections()),
            CancellationToken.None);

        Assert.True(result.Succeeded);
        Assert.Equal(2, requests.Count);
        Assert.True(requests[0].SkipProviderSmokeTest);
        Assert.False(requests[1].SkipProviderSmokeTest);
        Assert.Equal([ExecutionProviderKind.DirectMl, ExecutionProviderKind.Cpu], smokedProviders);
    }

    [Fact]
    public async Task WarmAsync_SmokeFailure_GenAiRoute_NeverReplansWithAuthoritativeSmoke()
    {
        // The no-replan guard is load-bearing: re-planning a GenAI route with the
        // authoritative smoke would take the evicting GenAI lease. A failed (not
        // skipped) GenAI smoke must stay a single plan + single smoke.
        int planCalls = 0;
        var planner = new FakeRuntimePlanner
        {
            PlanHandler = req =>
            {
                planCalls++;
                return RunnablePlan(req);
            },
        };
        int smokeCalls = 0;
        var smoke = new FakeExecutionProviderSmokeTester(_ =>
        {
            smokeCalls++;
            return new ExecutionProviderSmokeTestResult(false, "genai load failed");
        });
        var coordinator = new RuntimeStageWarmupCoordinator(planner, smoke);

        StageWarmupResult result = await coordinator.WarmAsync(
            new StageWarmupRequest(RuntimeStage.TextRefinement, Selections()),
            CancellationToken.None);

        Assert.True(result.Attempted);
        Assert.False(result.Succeeded);
        Assert.Equal("genai load failed", result.Detail);
        Assert.Equal(1, planCalls);
        Assert.Equal(1, smokeCalls);
    }

    [Theory]
    [InlineData(RuntimeStage.Asr, "whisper-genai")]
    [InlineData(RuntimeStage.Translation, "phi-genai")]
    public async Task WarmAsync_SmokeFailure_GenAiEngineFamily_NeverReplansWithAuthoritativeSmoke(
        RuntimeStage stage, string engineFamily)
    {
        // Same no-replan guard, exercised through the engine-family disjuncts of
        // IsGenAiRouted rather than the TextRefinement stage disjunct: deleting
        // either family clause must turn this red.
        int planCalls = 0;
        var planner = new FakeRuntimePlanner
        {
            PlanHandler = req =>
            {
                planCalls++;
                return RunnablePlan(req) with { EngineFamily = engineFamily };
            },
        };
        int smokeCalls = 0;
        var smoke = new FakeExecutionProviderSmokeTester(_ =>
        {
            smokeCalls++;
            return new ExecutionProviderSmokeTestResult(false, "genai load failed");
        });
        var coordinator = new RuntimeStageWarmupCoordinator(planner, smoke);

        StageWarmupResult result = await coordinator.WarmAsync(
            new StageWarmupRequest(stage, Selections()),
            CancellationToken.None);

        Assert.True(result.Attempted);
        Assert.False(result.Succeeded);
        Assert.Equal(1, planCalls);
        Assert.Equal(1, smokeCalls);
    }

    [Fact]
    public async Task WarmAsync_WorkerWarmOnWrongProvider_ReportsFailureAndSkipsSmoke()
    {
        var planner = new FakeRuntimePlanner
        {
            PlanHandler = _ => new StageRuntimePlan
            {
                Stage = RuntimeStage.Tts,
                Status = StageRuntimePlanStatus.Verified,
                ModelId = "kokoro-model",
                ModelAlias = "kokoro-onnx",
                EngineFamily = "kokoro",
                Variant = "fp16",
                ExecutionProvider = ExecutionProviderKind.Cuda,
                ModelRootPath = "/models/kokoro",
                ModelEntryPath = "/models/kokoro/model.onnx",
            },
        };
        int smokeCalls = 0;
        var smoke = new FakeExecutionProviderSmokeTester(_ =>
        {
            smokeCalls++;
            return new ExecutionProviderSmokeTestResult(true);
        });
        var worker = new FakeWorkerClient([]) { WarmProvider = "cpu" };
        var coordinator = new RuntimeStageWarmupCoordinator(
            planner, smoke, inferenceWorkerClient: worker, hostCudaProviderListed: () => false);

        StageWarmupResult result = await coordinator.WarmAsync(
            new StageWarmupRequest(RuntimeStage.Tts, Selections()), CancellationToken.None);

        Assert.True(result.Attempted);
        Assert.False(result.Succeeded);
        Assert.Contains("instead of cuda", result.Detail);
        Assert.Equal(0, smokeCalls);
    }

    [Fact]
    public async Task WarmAsync_PlannerException_IsConvertedToFailedResult()
    {
        var planner = new FakeRuntimePlanner
        {
            PlanHandlerAsync = (_, _) =>
                Task.FromException<StageRuntimePlan>(new InvalidOperationException("planner blew up")),
        };
        var smoke = new FakeExecutionProviderSmokeTester();
        var coordinator = new RuntimeStageWarmupCoordinator(planner, smoke);

        StageWarmupResult result = await coordinator.WarmAsync(
            new StageWarmupRequest(RuntimeStage.Asr, Selections()),
            CancellationToken.None);

        Assert.True(result.Attempted);
        Assert.False(result.Succeeded);
        Assert.Equal("planner blew up", result.Detail);
    }

    [Fact]
    public async Task WarmAsync_Cancellation_Propagates()
    {
        var planner = new FakeRuntimePlanner
        {
            PlanHandlerAsync = (_, ct) =>
                Task.FromCanceled<StageRuntimePlan>(ct),
        };
        var coordinator = new RuntimeStageWarmupCoordinator(
            planner, new FakeExecutionProviderSmokeTester());

        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => coordinator.WarmAsync(
                new StageWarmupRequest(RuntimeStage.Asr, Selections()), cts.Token));
    }

    [Fact]
    public async Task WarmAsync_TtsVoiceClone_UsesCloneAliasRoute()
    {
        StageRuntimePlanningRequest? capturedPlan = null;
        var planner = new FakeRuntimePlanner
        {
            PlanHandler = req =>
            {
                capturedPlan = req;
                return RunnablePlan(req);
            },
        };
        var coordinator = new RuntimeStageWarmupCoordinator(
            planner, new FakeExecutionProviderSmokeTester());

        await coordinator.WarmAsync(
            new StageWarmupRequest(RuntimeStage.Tts, Selections(), RequiresVoiceClone: true),
            CancellationToken.None);

        Assert.NotNull(capturedPlan);
        Assert.Equal(RuntimeStage.Tts, capturedPlan!.Stage);
        Assert.Equal(VoiceCloningDefaults.ChatterboxPrimaryAlias, capturedPlan.PreferredModelAlias);
        Assert.True(capturedPlan.RequirePreferredModelAlias);
    }

    [Fact]
    public async Task WarmAsync_PassedSmoke_RecordsVerdictForNextLaunch()
    {
        var planner = new FakeRuntimePlanner { PlanHandler = RunnablePlan };
        var smoke = new FakeExecutionProviderSmokeTester(
            _ => new ExecutionProviderSmokeTestResult(true));
        var verdictStore = new RecordingVerdictStore();
        var coordinator = new RuntimeStageWarmupCoordinator(
            planner, smoke, smokeVerdictStore: verdictStore,
            hardwareProfileProvider: new FakeHardwareProfileProvider());

        StageWarmupResult result = await coordinator.WarmAsync(
            new StageWarmupRequest(RuntimeStage.Asr, Selections()),
            CancellationToken.None);

        Assert.True(result.Succeeded);
        SmokeVerdictKey key = Assert.Single(verdictStore.Recorded);
        Assert.Equal("deadbeef", key.ModelSha256);
        Assert.Equal(ExecutionProviderKind.DirectMl, key.ExecutionProvider);
        Assert.Equal("1.24.0-test", key.OrtRuntimeVersion);
    }

    [Fact]
    public async Task WarmAsync_FailedSmoke_RecordsNoVerdictAndReturnsFailureWithReason()
    {
        var planner = new FakeRuntimePlanner { PlanHandler = RunnablePlan };
        var smoke = new FakeExecutionProviderSmokeTester(
            _ => new ExecutionProviderSmokeTestResult(false, "provider failed"));
        var verdictStore = new RecordingVerdictStore();
        var coordinator = new RuntimeStageWarmupCoordinator(
            planner, smoke, smokeVerdictStore: verdictStore,
            hardwareProfileProvider: new FakeHardwareProfileProvider());

        StageWarmupResult result = await coordinator.WarmAsync(
            new StageWarmupRequest(RuntimeStage.Asr, Selections()),
            CancellationToken.None);

        Assert.True(result.Attempted);
        Assert.False(result.Succeeded);
        Assert.Equal("provider failed", result.Detail);
        Assert.Empty(verdictStore.Recorded);
    }

    private sealed class RecordingVerdictStore : ISmokeVerdictStore
    {
        public List<SmokeVerdictKey> Recorded { get; } = [];
        public bool IsVerified(SmokeVerdictKey key) => Recorded.Contains(key);
        public void RecordVerified(SmokeVerdictKey key) => Recorded.Add(key);
        public void Clear() => Recorded.Clear();
    }

    private sealed class FakeHardwareProfileProvider : IHardwareProfileProvider
    {
        public Task<HardwareProfile> GetCurrentAsync(CancellationToken cancellationToken = default) =>
            Task.FromResult(new HardwareProfile(
                "windows", "x64", HasGpu: true, GpuDescription: "Test GPU",
                NvidiaGpuArchitecture: NvidiaGpuArchitectureBucket.Ada,
                GpuDriverVersion: "555.1",
                OrtRuntimeVersion: "1.24.0-test"));
    }

    [Fact]
    public async Task WarmAsync_WorkerBackedCudaTts_WarmsWorkerBeforeSmoke()
    {
        var planner = new FakeRuntimePlanner();
        planner.PlanHandler = _ => new StageRuntimePlan
        {
            Stage = RuntimeStage.Tts,
            Status = StageRuntimePlanStatus.Verified,
            ModelId = "kokoro-model",
            ModelAlias = "kokoro-onnx",
            EngineFamily = "kokoro",
            Variant = "fp16",
            ExecutionProvider = ExecutionProviderKind.Cuda,
            ModelRootPath = "/models/kokoro",
            ModelEntryPath = "/models/kokoro/model.onnx",
        };
        var order = new List<string>();
        var smoke = new FakeExecutionProviderSmokeTester(_ =>
        {
            order.Add("smoke");
            return new ExecutionProviderSmokeTestResult(true);
        });
        var worker = new FakeWorkerClient(order);
        var coordinator = new RuntimeStageWarmupCoordinator(
            planner, smoke, inferenceWorkerClient: worker, hostCudaProviderListed: () => false);

        StageWarmupResult result = await coordinator.WarmAsync(
            new StageWarmupRequest(RuntimeStage.Tts, Selections()), CancellationToken.None);

        Assert.True(result.Succeeded);
        Assert.Equal(1, worker.WarmCalls);
        Assert.Equal(["warm", "smoke"], order);
    }

    [Fact]
    public async Task WarmAsync_WorkerWarmFailure_ReportsReasonAndSkipsSmoke()
    {
        var planner = new FakeRuntimePlanner
        {
            PlanHandler = _ => new StageRuntimePlan
            {
                Stage = RuntimeStage.Tts,
                Status = StageRuntimePlanStatus.Verified,
                ModelId = "kokoro-model",
                ModelAlias = "kokoro-onnx",
                EngineFamily = "kokoro",
                Variant = "fp16",
                ExecutionProvider = ExecutionProviderKind.Cuda,
                ModelRootPath = "/models/kokoro",
                ModelEntryPath = "/models/kokoro/model.onnx",
            },
        };
        int smokeCalls = 0;
        var smoke = new FakeExecutionProviderSmokeTester(_ =>
        {
            smokeCalls++;
            return new ExecutionProviderSmokeTestResult(true);
        });
        var worker = new FakeWorkerClient([])
        {
            WarmFailure = new InferenceWorkerException("worker session create failed"),
        };
        var coordinator = new RuntimeStageWarmupCoordinator(
            planner, smoke, inferenceWorkerClient: worker, hostCudaProviderListed: () => false);

        StageWarmupResult result = await coordinator.WarmAsync(
            new StageWarmupRequest(RuntimeStage.Tts, Selections()), CancellationToken.None);

        Assert.True(result.Attempted);
        Assert.False(result.Succeeded);
        Assert.Contains("worker session create failed", result.Detail);
        Assert.Contains("retry construction inline", result.Detail);
        Assert.Equal(0, smokeCalls);
    }

    [Fact]
    public async Task WarmAsync_NonWorkerPlan_NeverWarmsWorker()
    {
        var planner = new FakeRuntimePlanner { PlanHandler = RunnablePlan };
        var worker = new FakeWorkerClient([]);
        var coordinator = new RuntimeStageWarmupCoordinator(
            planner, new FakeExecutionProviderSmokeTester(), inferenceWorkerClient: worker,
            hostCudaProviderListed: () => false);

        await coordinator.WarmAsync(
            new StageWarmupRequest(RuntimeStage.Translation, Selections()), CancellationToken.None);

        Assert.Equal(0, worker.WarmCalls);
    }

    private sealed class FakeWorkerClient(List<string> order) : IInferenceWorkerClient
    {
        public bool IsInstalled => true;
        public int WarmCalls { get; private set; }
        public Exception? WarmFailure { get; set; }
        public string WarmProvider { get; set; } = "cuda";

        public Task<WorkerHelloResult?> TryGetHelloAsync(CancellationToken cancellationToken) =>
            Task.FromResult<WorkerHelloResult?>(null);

        public Task<WorkerTranslationResult> TranslateAsync(
            TranslationRequest request, StageRuntimePlan plan, CancellationToken cancellationToken,
            bool nonEvictingWarmup = false) =>
            throw new NotSupportedException("Coordinator tests do not translate through the worker.");

        public Task<TtsSynthesisResult> SynthesizeTtsAsync(
            TtsSynthesisRequest request, StageRuntimePlan plan, CancellationToken cancellationToken) =>
            throw new NotSupportedException("Coordinator tests never synthesize through the worker.");

        public Task<WorkerWarmResult> WarmTtsAsync(StageRuntimePlan plan, CancellationToken cancellationToken)
        {
            WarmCalls++;
            order.Add("warm");
            return WarmFailure is not null
                ? Task.FromException<WorkerWarmResult>(WarmFailure)
                : Task.FromResult(new WorkerWarmResult(plan.ModelId, WarmProvider));
        }
    }

    [Fact]
    public void CompositionRoot_ResolvesStageWarmupCoordinator_InScope()
    {
        var services = new ServiceCollection();
        services.AddTrackdub();
        using ServiceProvider provider = services.BuildServiceProvider(
            new ServiceProviderOptions { ValidateOnBuild = true, ValidateScopes = true });
        using IServiceScope scope = provider.CreateScope();

        IStageWarmupCoordinator coordinator =
            scope.ServiceProvider.GetRequiredService<IStageWarmupCoordinator>();
        Assert.IsType<RuntimeStageWarmupCoordinator>(coordinator);
    }
}
