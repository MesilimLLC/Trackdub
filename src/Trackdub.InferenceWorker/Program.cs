using System.Collections.Concurrent;
using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Onnx;
using Trackdub.Inference.Onnx.Kokoro;
using Trackdub.Inference.Onnx.Phi;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Onnx.Runtime;
using Trackdub.Inference.Onnx.Worker;
using Trackdub.Inference.Runtime.Planning;

// ORT 1.30 inference worker (ADR-0017). Standard output carries protocol messages only; all
// diagnostics go to standard error, which the application keeps as the crash tail.
Stream input = Console.OpenStandardInput();
Stream output = Console.OpenStandardOutput();

// Bind stock ORT before the first managed ORT call. The dependency graph also contains
// Windows ML assets, so default native probing can select those. The worker can still
// answer Hello when the loaded ORT has no GenAI provenance; the GenAI path verifies its
// native pair immediately before loading GenAI.
string? ortUnavailableDetail = null;
try
{
    GenAiNativeRuntimeSelection.EnsureOrtLoaded(requireVerified: false);
}
catch (Exception ex) when (ex is InvalidOperationException or DllNotFoundException or BadImageFormatException or EntryPointNotFoundException)
{
    ortUnavailableDetail = ex.Message;
}

// Opt-in EcoQoS (MS efficiency guidance, CPU workloads): the worker is a background helper
// process, so operators may prefer efficiency-oriented CPU scheduling for it. The variable is
// inherited from the host process env; default stays OS-managed. The outcome is verified by
// read-back and reported — never claimed without proof.
if (string.Equals(
        Environment.GetEnvironmentVariable(Trackdub.Inference.Onnx.Runtime.WindowsEcoQos.EnvironmentVariable),
        "1", StringComparison.Ordinal))
{
    (bool ecoQosApplied, string? ecoQosError) =
        Trackdub.Inference.Onnx.Runtime.WindowsEcoQos.EnableExecutionSpeedThrottling();
    Console.Error.WriteLine(ecoQosApplied
        ? $"[worker] EcoQoS enabled (verified: {Trackdub.Inference.Onnx.Runtime.WindowsEcoQos.IsExecutionSpeedThrottled})."
        : $"[worker] EcoQoS requested but NOT applied: {ecoQosError}");
}
var writeGate = new SemaphoreSlim(1, 1);
var inFlight = new ConcurrentDictionary<long, CancellationTokenSource>();
var handlers = new ConcurrentDictionary<long, Task>();
// One host per worker lifetime: the Kokoro engine is constructed at most once, and a
// failed construction/warm is never cached as a failure — the next request retries.
var kokoroHost = new KokoroWorkerHost(() => new KokoroTtsEngine(
    new PlanSuppliedByApplication(),
    BenchmarkModelPathResolver.CreateDefault(),
    new EspeakNgPhonemizer()));

async Task RespondAsync(WorkerMessage message)
{
    await writeGate.WaitAsync().ConfigureAwait(false);
    try
    {
        await InferenceWorkerProtocol.WriteAsync(output, message, CancellationToken.None).ConfigureAwait(false);
    }
    finally
    {
        writeGate.Release();
    }
}

async Task<WorkerMessage> CreateResponseAsync(WorkerMessage request, CancellationToken cancellationToken) =>
    request.Kind switch
    {
        InferenceWorkerProtocol.Hello => new WorkerMessage(request.Id, InferenceWorkerProtocol.Result,
            InferenceWorkerProtocol.ToPayload(DescribeWorker())),
        InferenceWorkerProtocol.TtsSynthesize => new WorkerMessage(request.Id, InferenceWorkerProtocol.Result,
            InferenceWorkerProtocol.ToPayload(await kokoroHost.SynthesizeAsync(
                InferenceWorkerProtocol.FromPayload<WorkerTtsRequest>(request.Payload),
                cancellationToken).ConfigureAwait(false))),
        InferenceWorkerProtocol.TtsWarm => new WorkerMessage(request.Id, InferenceWorkerProtocol.Result,
            InferenceWorkerProtocol.ToPayload(await kokoroHost.WarmAsync(
                InferenceWorkerProtocol.FromPayload<WorkerWarmRequest>(request.Payload).Plan,
                cancellationToken).ConfigureAwait(false))),
        InferenceWorkerProtocol.Translation => new WorkerMessage(request.Id, InferenceWorkerProtocol.Result,
            InferenceWorkerProtocol.ToPayload(await TranslateAsync(
                InferenceWorkerProtocol.FromPayload<WorkerTranslationRequest>(request.Payload),
                cancellationToken).ConfigureAwait(false))),
        _ => throw new NotSupportedException($"Unknown inference worker request '{request.Kind}'."),
    };

async Task HandleAsync(WorkerMessage request, CancellationToken cancellationToken)
{
    try
    {
        WorkerMessage response = await CreateResponseAsync(request, cancellationToken).ConfigureAwait(false);
        await RespondAsync(response).ConfigureAwait(false);
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"[worker] {request.Kind} #{request.Id} failed: {ex}");
        await RespondAsync(new WorkerMessage(request.Id, InferenceWorkerProtocol.Error,
            Error: ex.Message, ErrorType: ex is OperationCanceledException ? nameof(OperationCanceledException) : ex.GetType().Name))
            .ConfigureAwait(false);
    }
    finally
    {
        if (inFlight.TryRemove(request.Id, out CancellationTokenSource? source))
        {
            source.Dispose();
        }
    }
}

static async Task<WorkerTranslationResult> TranslateAsync(WorkerTranslationRequest request, CancellationToken cancellationToken)
{
    if (!SupportsTranslationRequest(request))
    {
        throw new NotSupportedException("The inference worker supports CUDA translation for qwen-genai.");
    }

    var engine = new PhiGenAiTranslationEngine(
        new SuppliedTranslationPlan(request.Plan.ToStageRuntimePlan()),
        BenchmarkModelPathResolver.CreateDefault(),
        engineFamily: PhiGenAiTranslationEngine.QwenEngineFamilyName,
        inferenceWorkerProcess: true)
    {
        NonEvictingWarmup = request.NonEvictingWarmup,
    };
    try
    {
        var segments = await engine.TranslateAsync(request.Request, cancellationToken).ConfigureAwait(false);
        return new WorkerTranslationResult(segments, engine.LastExecutionSummary);
    }
    catch (GenAiWarmupSkippedException ex)
    {
        return new WorkerTranslationResult([], null, NotAttempted: true, Detail: ex.Message);
    }
}

static bool SupportsTranslationRequest(WorkerTranslationRequest request) =>
    string.Equals(request.Plan.EngineFamily, PhiGenAiTranslationEngine.QwenEngineFamilyName, StringComparison.OrdinalIgnoreCase)
    && request.Plan.Stage == Trackdub.Domain.RuntimeStage.Translation
    && request.Plan.ExecutionProvider == Trackdub.Domain.ExecutionProviderKind.Cuda;

WorkerHelloResult DescribeWorker()
{
    if (ortUnavailableDetail is string unavailableDetail)
    {
        return new WorkerHelloResult(
            InferenceWorkerProtocol.Version,
            InferenceWorkerProtocol.BuildStamp,
            "unavailable",
            CudaAvailable: false,
            CudaDetail: $"Native ONNX Runtime is unavailable: {unavailableDetail}");
    }

    bool cudaListed = CudaOrtProbe.IsCudaProviderListed();
    bool cuDnn = CudaOrtProbe.IsCuDnn9Available();
    string detail = (cudaListed, cuDnn) switch
    {
        (true, true) => "Native ORT CUDA execution provider and cuDNN 9 are available.",
        (false, _) => "This worker's ONNX Runtime build does not list CUDAExecutionProvider.",
        (true, false) => "cuDNN 9 (cudnn64_9.dll) for CUDA 13 was not found on PATH.",
    };
    return new WorkerHelloResult(
        InferenceWorkerProtocol.Version,
        InferenceWorkerProtocol.BuildStamp,
        OrtEnv.Instance().GetVersionString(),
        cudaListed && cuDnn,
        detail);
}

async Task CancelQuietlyAsync(CancellationTokenSource source)
{
    try
    {
        await source.CancelAsync().ConfigureAwait(false);
    }
    catch (ObjectDisposedException)
    {
        // The request finished and disposed its source between lookup and cancel.
    }
}

// Cancel what is still running, let it finish (its sessions are released by the engine), then
// release the engine's native ONNX Runtime sessions before the process exits.
async Task DrainAsync()
{
    foreach (CancellationTokenSource source in inFlight.Values)
    {
        await CancelQuietlyAsync(source).ConfigureAwait(false);
    }

    try
    {
        await Task.WhenAll(handlers.Values).WaitAsync(TimeSpan.FromSeconds(10)).ConfigureAwait(false);
    }
    catch (TimeoutException)
    {
        Console.Error.WriteLine("[worker] In-flight requests did not stop within 10 s; exiting anyway.");
        return;
    }
    catch (Exception ex)
    {
        // Handlers report their own failures; a broken pipe while answering must not skip cleanup.
        Console.Error.WriteLine($"[worker] An in-flight request failed during shutdown: {ex.Message}");
    }

    kokoroHost.Dispose();
}

while (await InferenceWorkerProtocol.ReadAsync(input, CancellationToken.None).ConfigureAwait(false) is { } message)
{
    switch (message.Kind)
    {
        case InferenceWorkerProtocol.Shutdown:
            await DrainAsync().ConfigureAwait(false);
            return 0;
        case InferenceWorkerProtocol.Cancel:
            long target = InferenceWorkerProtocol.FromPayload<long>(message.Payload);
            if (inFlight.TryGetValue(target, out CancellationTokenSource? running))
            {
                await CancelQuietlyAsync(running).ConfigureAwait(false);
            }

            break;
        default:
            var cancellation = new CancellationTokenSource();
            inFlight[message.Id] = cancellation;
            Task handler = Task.Run(() => HandleAsync(message, cancellation.Token));
            handlers[message.Id] = handler;
            _ = handler.ContinueWith(done => handlers.TryRemove(message.Id, out _), TaskScheduler.Default);
            break;
    }
}

// The application closed the pipe without a shutdown message.
await DrainAsync().ConfigureAwait(false);
return 0;

/// <summary>The worker never plans: the application's planner already chose model and provider.</summary>
internal sealed class PlanSuppliedByApplication : IRuntimePlanner
{
    public Task<StageRuntimePlan> PlanAsync(StageRuntimePlanningRequest request, CancellationToken cancellationToken = default) =>
        throw new InvalidOperationException("The inference worker only runs plans supplied by the application.");
}

internal sealed class SuppliedTranslationPlan(StageRuntimePlan plan) : IRuntimePlanner
{
    public Task<StageRuntimePlan> PlanAsync(StageRuntimePlanningRequest request, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(plan);
    }
}
