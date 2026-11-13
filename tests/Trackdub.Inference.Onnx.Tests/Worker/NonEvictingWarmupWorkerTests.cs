using System.Text.Json;
using System.Text.Json.Nodes;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Onnx.Worker;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.Tests.Worker;

public sealed class NonEvictingWarmupWorkerTests
{
    private static readonly TranslationRequest HelloRequest =
        new("en", "fr", [new TranslationInputSegment(0, 0, 1, "Hello.")]);

    [Fact]
    public void Translation_request_round_trips_the_non_evicting_warmup_flag()
    {
        var request = new WorkerTranslationRequest(HelloRequest, WorkerPlan.From(QwenCudaPlan()), NonEvictingWarmup: true);

        var restored = InferenceWorkerProtocol.FromPayload<WorkerTranslationRequest>(
            InferenceWorkerProtocol.ToPayload(request));

        Assert.True(restored.NonEvictingWarmup);
    }

    [Fact]
    public void Translation_request_from_a_peer_without_the_flag_is_evicting()
    {
        var request = new WorkerTranslationRequest(HelloRequest, WorkerPlan.From(QwenCudaPlan()), NonEvictingWarmup: true);
        JsonNode node = JsonSerializer.SerializeToNode(request, InferenceWorkerProtocol.JsonOptions)!;
        Assert.True(node.AsObject().Remove("nonEvictingWarmup"));

        var restored = node.Deserialize<WorkerTranslationRequest>(InferenceWorkerProtocol.JsonOptions)!;

        Assert.False(restored.NonEvictingWarmup);
    }

    [Fact]
    public void Translation_result_round_trips_the_not_attempted_shape()
    {
        var result = new WorkerTranslationResult([], null, NotAttempted: true, Detail: "would evict");

        var restored = InferenceWorkerProtocol.FromPayload<WorkerTranslationResult>(
            InferenceWorkerProtocol.ToPayload(result));

        Assert.True(restored.NotAttempted);
        Assert.Equal("would evict", restored.Detail);
        Assert.Empty(restored.Segments);
    }

    [Fact]
    public void Translation_result_from_a_peer_without_the_shape_was_attempted()
    {
        var result = new WorkerTranslationResult([], null, NotAttempted: true, Detail: "would evict");
        JsonNode node = JsonSerializer.SerializeToNode(result, InferenceWorkerProtocol.JsonOptions)!;
        Assert.True(node.AsObject().Remove("notAttempted"));
        Assert.True(node.AsObject().Remove("detail"));

        var restored = node.Deserialize<WorkerTranslationResult>(InferenceWorkerProtocol.JsonOptions)!;

        Assert.False(restored.NotAttempted);
        Assert.Null(restored.Detail);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Smoke_test_forwards_the_warmup_mode_to_the_worker(bool nonEvictingWarmup)
    {
        var worker = new FakeWorkerClient(new WorkerTranslationResult([], null));
        var tester = new OnnxExecutionProviderSmokeTester(worker);

        await tester.SmokeTestAsync(
            QwenCudaSmokeRequest() with { NonEvictingWarmup = nonEvictingWarmup }, TestContext.Current.CancellationToken);

        Assert.Equal(nonEvictingWarmup, worker.LastNonEvictingWarmup);
    }

    [Fact]
    public async Task Smoke_test_reports_not_attempted_when_the_worker_skips_a_warmup()
    {
        var worker = new FakeWorkerClient(new WorkerTranslationResult([], null, NotAttempted: true, Detail: "would evict"));
        var tester = new OnnxExecutionProviderSmokeTester(worker);

        ExecutionProviderSmokeTestResult result = await tester.SmokeTestAsync(
            QwenCudaSmokeRequest() with { NonEvictingWarmup = true }, TestContext.Current.CancellationToken);

        Assert.False(result.Passed);
        Assert.True(result.NotAttempted);
        Assert.Equal("would evict", result.Detail);
    }

    [Fact]
    public async Task Smoke_test_without_a_produced_translation_fails_rather_than_skips()
    {
        var worker = new FakeWorkerClient(new WorkerTranslationResult([], null));
        var tester = new OnnxExecutionProviderSmokeTester(worker);

        ExecutionProviderSmokeTestResult result = await tester.SmokeTestAsync(
            QwenCudaSmokeRequest(), TestContext.Current.CancellationToken);

        Assert.False(result.Passed);
        Assert.False(result.NotAttempted);
    }

    private static ExecutionProviderSmokeTestRequest QwenCudaSmokeRequest() =>
        new(RuntimeStage.Translation, "Qwen/Qwen3-4B", "qwen3-4b-translation", "qwen-genai", "cuda-int4",
            ExecutionProviderKind.Cuda, "/models/qwen", "/models/qwen/cuda-int4/genai_config.json");

    private static StageRuntimePlan QwenCudaPlan() =>
        new()
        {
            Stage = RuntimeStage.Translation,
            Status = StageRuntimePlanStatus.Ready,
            ModelId = "Qwen/Qwen3-4B",
            ModelAlias = "qwen3-4b-translation",
            EngineFamily = "qwen-genai",
            Variant = "cuda-int4",
            ExecutionProvider = ExecutionProviderKind.Cuda,
            RequirePreferredExecutionProvider = true,
        };

    private sealed class FakeWorkerClient(WorkerTranslationResult translation) : IInferenceWorkerClient
    {
        public bool? LastNonEvictingWarmup { get; private set; }

        public bool IsInstalled => true;

        public Task<WorkerHelloResult?> TryGetHelloAsync(CancellationToken cancellationToken) =>
            Task.FromResult<WorkerHelloResult?>(null);

        public Task<WorkerTranslationResult> TranslateAsync(
            TranslationRequest request,
            StageRuntimePlan plan,
            CancellationToken cancellationToken,
            bool nonEvictingWarmup = false)
        {
            LastNonEvictingWarmup = nonEvictingWarmup;
            return Task.FromResult(translation);
        }

        public Task<TtsSynthesisResult> SynthesizeTtsAsync(
            TtsSynthesisRequest request, StageRuntimePlan plan, CancellationToken cancellationToken) =>
            throw new NotSupportedException();

        public Task<WorkerWarmResult> WarmTtsAsync(StageRuntimePlan plan, CancellationToken cancellationToken) =>
            throw new NotSupportedException();
    }
}
