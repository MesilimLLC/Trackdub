using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Worker;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Tests;

internal sealed class RecordingWorkerClient : IInferenceWorkerClient
{
    public bool IsInstalled => true;

    public List<bool> NonEvictingWarmupValues { get; } = [];

    public Task<WorkerHelloResult?> TryGetHelloAsync(CancellationToken cancellationToken) =>
        Task.FromResult<WorkerHelloResult?>(null);

    public Task<WorkerTranslationResult> TranslateAsync(
        TranslationRequest request,
        StageRuntimePlan plan,
        CancellationToken cancellationToken,
        bool nonEvictingWarmup = false)
    {
        NonEvictingWarmupValues.Add(nonEvictingWarmup);
        return Task.FromResult(new WorkerTranslationResult([], null, NotAttempted: true, Detail: "would evict"));
    }

    public Task<TtsSynthesisResult> SynthesizeTtsAsync(
        TtsSynthesisRequest request,
        StageRuntimePlan plan,
        CancellationToken cancellationToken) =>
        throw new NotSupportedException();

    public Task<WorkerWarmResult> WarmTtsAsync(
        StageRuntimePlan plan,
        CancellationToken cancellationToken) =>
        throw new NotSupportedException();
}
