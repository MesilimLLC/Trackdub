using Microsoft.ML.OnnxRuntimeGenAI;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.Planning;
using Trackdub.Inference.Onnx.Runtime;
using Trackdub.Inference.Onnx.ExecutionProviders;
using Trackdub.Inference.Onnx.Qwen3Asr;
using Trackdub.Inference.Onnx.NemotronAsr;
using Trackdub.Inference.Onnx.OpusMt;
using Trackdub.Inference.Onnx.ParakeetTdt;
using Trackdub.Inference.Onnx.CosyVoice;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.SortFormer;
using Trackdub.Inference.Onnx.Spleeter;
using Trackdub.Inference.Onnx.Whisper;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Inference.Onnx.Worker;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Contracts.Pipeline;

namespace Trackdub.Inference.Onnx.Runtime.Planning;

public sealed partial class OnnxExecutionProviderSmokeTester : IExecutionProviderSmokeTester
{
    private readonly GenAiModelPool genAiModelPool;

    public OnnxExecutionProviderSmokeTester()
        : this(GenAiModelPool.Shared)
    {
    }

    internal OnnxExecutionProviderSmokeTester(GenAiModelPool genAiModelPool)
    {
        this.genAiModelPool = genAiModelPool;
    }

    /// <summary>
    /// Smoke-tests engine families hosted by the ORT 1.30 inference worker on CUDA through that
    /// worker, since this process's ONNX Runtime cannot create CUDA sessions (ADR-0017).
    /// </summary>
    public OnnxExecutionProviderSmokeTester(IInferenceWorkerClient workerClient)
        : this(GenAiModelPool.Shared)
    {
        this.workerClient = workerClient ?? throw new ArgumentNullException(nameof(workerClient));
    }

    private readonly IInferenceWorkerClient? workerClient;

    private async Task<ExecutionProviderSmokeTestResult> SmokeTestThroughWorkerAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        var plan = new StageRuntimePlan
        {
            Stage = request.Stage,
            Status = StageRuntimePlanStatus.Ready,
            ModelId = request.ModelId,
            ModelAlias = request.ModelAlias,
            EngineFamily = request.EngineFamily,
            Variant = request.Variant,
            ExecutionProvider = request.ExecutionProvider,
            RequirePreferredExecutionProvider = true,
            ModelEntryPath = request.EntryPath,
            ModelRootPath = request.ModelRootPath,
            ModelRevisionHash = request.ModelRevisionHash,
        };
        try
        {
            return await RunWorkerSmokeAsync(request, plan, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is InferenceWorkerException or InvalidOperationException or IOException)
        {
            return new ExecutionProviderSmokeTestResult(false, $"Inference worker smoke failed: {ex.Message}");
        }
    }

    private Task<ExecutionProviderSmokeTestResult> RunWorkerSmokeAsync(
        ExecutionProviderSmokeTestRequest request,
        StageRuntimePlan plan,
        CancellationToken cancellationToken) =>
        request.Stage == RuntimeStage.Translation &&
        string.Equals(request.EngineFamily, "qwen-genai", StringComparison.OrdinalIgnoreCase)
            ? SmokeTestQwenTranslationWorkerAsync(plan, request.NonEvictingWarmup, cancellationToken)
            : SmokeTestTtsWorkerAsync(request, plan, cancellationToken);

    private async Task<ExecutionProviderSmokeTestResult> SmokeTestQwenTranslationWorkerAsync(
        StageRuntimePlan plan,
        bool nonEvictingWarmup,
        CancellationToken cancellationToken)
    {
        WorkerTranslationResult translated = await workerClient!.TranslateAsync(
            new TranslationRequest("en", "fr", [new TranslationInputSegment(0, 0, 1, "Hello.")]),
            plan,
            cancellationToken,
            nonEvictingWarmup).ConfigureAwait(false);
        if (translated.NotAttempted)
        {
            return new ExecutionProviderSmokeTestResult(
                false,
                translated.Detail ?? "Inference worker skipped the warmup because loading would evict a resident model.")
            {
                NotAttempted = true,
            };
        }

        bool onCuda = string.Equals(translated.ExecutionSummary?.SelectedProvider, "cuda", StringComparison.OrdinalIgnoreCase);
        bool producedText = translated.Segments.Count == 1 && !string.IsNullOrWhiteSpace(translated.Segments[0].Text);
        bool succeeded = onCuda && producedText;
        return new ExecutionProviderSmokeTestResult(succeeded,
            succeeded ? "Inference worker translated on CUDA." : "Inference worker did not produce a CUDA translation.");
    }

    private async Task<ExecutionProviderSmokeTestResult> SmokeTestTtsWorkerAsync(
        ExecutionProviderSmokeTestRequest request,
        StageRuntimePlan plan,
        CancellationToken cancellationToken)
    {
        TtsSynthesisResult result = await workerClient!.SynthesizeTtsAsync(
            new TtsSynthesisRequest("Hello.", "en", new VoiceCatalogEntry("af_heart", "en-us", "female", "Heart")),
            plan,
            cancellationToken).ConfigureAwait(false);
        bool onRequestedProvider = string.Equals(
            result.Provider,
            FormatProviderLabel(request.ExecutionProvider),
            StringComparison.OrdinalIgnoreCase);
        return new ExecutionProviderSmokeTestResult(
            onRequestedProvider && result.DurationSamples > 0,
            onRequestedProvider
                ? $"Inference worker synthesized {result.DurationSamples} samples on {result.Provider}."
                : $"Inference worker ran on '{result.Provider}' instead of the requested provider.");
    }

    public async Task<ExecutionProviderSmokeTestResult> SmokeTestAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();

        // Smoke and warmup inference is real inference that warms pooled sessions; attribute
        // its completions to the smoke phase so cold-start evidence separates it from the
        // first real stage inference.
        using IDisposable smokePhase = ColdStartCapture.EnterPhase(ColdStartPhase.Smoke);
        using IDisposable smokeSpan = BenchmarkPhaseCapture.Start("ep-smoke");

        if (ShouldRouteCudaSmokeToWorker(request))
        {
            return await SmokeTestThroughWorkerAsync(request, cancellationToken).ConfigureAwait(false);
        }

        try
        {
            ThrowIfFatalSmokeFamily(request);

            ExecutionProviderSmokeTestResult? bootstrapFailure = await EnsureBootstrapFulfillsRequestAsync(
                request, cancellationToken).ConfigureAwait(false);
            if (bootstrapFailure is not null)
            {
                return bootstrapFailure;
            }

            request = request with
            {
                AllowPolicyProviderSelection = await OnnxExecutionSessionFactory.CanPolicySelectProviderAsync(
                    request.ExecutionProvider, request.HasExplicitProviderPreference, cancellationToken).ConfigureAwait(false),
            };

            return await DispatchSmokeAsync(request, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (GenAiWarmupSkippedException ex)
        {
            // Honest non-attempt: warmup declined to evict a resident model. No provider
            // verdict was produced, so this must not read as a smoke failure.
            return new ExecutionProviderSmokeTestResult(false, ex.Message) { NotAttempted = true };
        }
        catch (Exception ex)
        {
            return new ExecutionProviderSmokeTestResult(false, ex.Message);
        }
    }

    private bool ShouldRouteCudaSmokeToWorker(ExecutionProviderSmokeTestRequest request)
    {
        if (request.ExecutionProvider is not ExecutionProviderKind.Cuda)
        {
            return false;
        }

        if (workerClient is null)
        {
            return false;
        }

        return OnnxExecutionProviderDiscovery.WorkerCudaEngineFamilies
            .Contains(request.EngineFamily, StringComparer.OrdinalIgnoreCase);
    }

    // Register/validate the requested EP before any probe session. When the bootstrapper
    // cannot keep the requested provider selected (e.g. TRT RTX plugin missing and
    // falling back to DirectML), the pair is unproven — fail fast with that detail
    // instead of creating a session that silently lands on a different provider.
    private static async Task<ExecutionProviderSmokeTestResult?> EnsureBootstrapFulfillsRequestAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        ExecutionProviderBootstrapResult bootstrap = await OnnxExecutionSessionFactory
            .BootstrapForSmokeAsync(request.ExecutionProvider, cancellationToken)
            .ConfigureAwait(false);
        if (bootstrap.IsRequestFulfilled)
        {
            return null;
        }

        return new ExecutionProviderSmokeTestResult(
            false,
            $"Smoke bootstrap could not select {request.ExecutionProvider} "
            + $"(selected {bootstrap.SelectedProvider}). {bootstrap.Detail}");
    }

    private async Task<ExecutionProviderSmokeTestResult> DispatchSmokeAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        Func<CancellationToken, Task>? handler = ResolveStageHandler(request);
        if (handler is null)
        {
            return new ExecutionProviderSmokeTestResult(
                false,
                $"Smoke testing is not implemented for runtime stage '{request.Stage}'.");
        }

        await handler(cancellationToken).ConfigureAwait(false);
        return new ExecutionProviderSmokeTestResult(true);
    }

    private Func<CancellationToken, Task>? ResolveStageHandler(ExecutionProviderSmokeTestRequest request)
    {
        var handlers = new Dictionary<RuntimeStage, Func<CancellationToken, Task>>
        {
            [RuntimeStage.Vad] = ct => SmokeTestVadAsync(request, ct),
            [RuntimeStage.Asr] = ct => SmokeTestAsrAsync(request, ct),
            [RuntimeStage.Separation] = ct => SmokeTestSeparationAsync(request, ct),
            [RuntimeStage.Translation] = ct => SmokeTestTranslationAsync(request, ct),
            [RuntimeStage.Diarization] = ct => SmokeTestDiarizationAsync(request, ct),
            [RuntimeStage.Tts] = ct => SmokeTestTtsAsync(request, ct),
            [RuntimeStage.TextRefinement] = ct => SmokeTestTextRefinementGenAiAsync(request, ct),
            [RuntimeStage.SpeechEnhancement] = ct => SmokeTestGenericSessionAsync(request, ct),
            [RuntimeStage.OverlapRescue] = ct => SmokeTestGenericSessionAsync(request, ct),
            [RuntimeStage.LipSync] = ct => SmokeTestGenericSessionAsync(request, ct),
            [RuntimeStage.LipSynthesis] = ct => SmokeTestGenericSessionAsync(request, ct),
        };

        return handlers.TryGetValue(request.Stage, out Func<CancellationToken, Task>? handler)
            ? handler
            : null;
    }

    private static async Task SmokeTestVadAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        // Keyed exactly like SileroVadSpeechRegionDetector, so the session this smoke verifies
        // is the one the VAD stage leases next instead of being built a second time.
        try
        {
            using OnnxExecutionSessionFactory.SingleSessionLease sessionLease = await OnnxExecutionSessionFactory
                .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = SileroVadPoolFamily, Provider = request.ExecutionProvider, HasExplicitProviderPreference = request.HasExplicitProviderPreference, CancellationToken = cancellationToken, AllowTrtInitFallback = true }, request.EntryPath)
                .ConfigureAwait(false);
            try
            {
                EnsureSelectedProviderMatchesRequested(request.ExecutionProvider, sessionLease.SelectedProvider, request.AllowPolicyProviderSelection);
            }
            catch (InvalidOperationException ex)
            {
                throw new InvalidOperationException(
                    $"{ex.Message} Session bootstrap: {sessionLease.BootstrapDetail}",
                    ex);
            }

            using var input = CreateVadInputs();
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _ = sessionLease.Session.RunWithRetry(input.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: request.ExecutionProvider);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            // Do not leave a session that failed verification for the stage to reuse.
            await InferenceSessionPool.Shared.EvictModelAsync(SileroVadPoolFamily).ConfigureAwait(false);
            throw;
        }
    }

    private const string SileroVadPoolFamily = "silero-vad";

    private static async Task SmokeTestGenericSessionAsync(
        ExecutionProviderSmokeTestRequest request,
        CancellationToken cancellationToken)
    {
        string poolFamily = ResolveGenericPoolFamily(request);
        using OnnxExecutionSessionFactory.SingleSessionLease sessionLease = await OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = poolFamily, Provider = request.ExecutionProvider, HasExplicitProviderPreference = request.HasExplicitProviderPreference, CancellationToken = cancellationToken, AllowTrtInitFallback = true }, request.EntryPath)
            .ConfigureAwait(false);
        EnsureSelectedProviderMatchesRequested(request.ExecutionProvider, sessionLease.SelectedProvider, request.AllowPolicyProviderSelection);

        using var inputs = CreateMetadataDrivenInputs(sessionLease.Session.InputMetadata);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> _ = sessionLease.Session.RunWithRetry(inputs.Values, maxAttempts: 1, cancellationToken: cancellationToken, provider: request.ExecutionProvider);
    }

    /// <summary>
    /// Resolves the pool family for generic single-graph smoke targets so the key matches
    /// the production engine's CreatePooledSingleAsync family (DeepFilterNet, SepFormer, LatentSync).
    /// </summary>
    private static string ResolveGenericPoolFamily(ExecutionProviderSmokeTestRequest request)
    {
        string name = $"{request.EngineFamily} {request.ModelId} {Path.GetFileNameWithoutExtension(request.EntryPath)}";
        if (IsDeepFilterNetTarget(request, name))
        {
            string? graphFamily = ResolveDeepFilterNetPoolFamily(name);
            if (graphFamily is not null)
            {
                return graphFamily;
            }
        }

        if (request.Stage is RuntimeStage.OverlapRescue)
        {
            return name.Contains("osd", StringComparison.OrdinalIgnoreCase)
                ? "sepformer-osd"
                : "sepformer";
        }

        if (request.Stage is RuntimeStage.LipSync or RuntimeStage.LipSynthesis)
        {
            return string.IsNullOrWhiteSpace(request.EngineFamily) ? "latentsync" : request.EngineFamily;
        }

        return string.IsNullOrWhiteSpace(request.EngineFamily) ? request.Stage.ToString().ToLowerInvariant() : request.EngineFamily;
    }

    private static bool IsDeepFilterNetTarget(ExecutionProviderSmokeTestRequest request, string name)
    {
        if (request.Stage is RuntimeStage.SpeechEnhancement)
        {
            return true;
        }

        return name.Contains("deepfilternet", StringComparison.OrdinalIgnoreCase);
    }

    private static string? ResolveDeepFilterNetPoolFamily(string name)
    {
        if (name.Contains("erb", StringComparison.OrdinalIgnoreCase))
        {
            return "deepfilternet3-erb-dec";
        }

        if (name.Contains("df_dec", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("df-dec", StringComparison.OrdinalIgnoreCase))
        {
            return "deepfilternet3-df-dec";
        }

        return name.Contains("enc", StringComparison.OrdinalIgnoreCase)
            ? "deepfilternet3-enc"
            : null;
    }

    private static InputSet CreateVadInputs()
    {
        IReadOnlyList<NamedOnnxValue> values =
        [
            NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(new float[512], [1, 512])),
            NamedOnnxValue.CreateFromTensor("state", new DenseTensor<float>(new float[2 * 128], [2, 1, 128])),
            NamedOnnxValue.CreateFromTensor("sr", new DenseTensor<long>(new long[] { 16000 }, [1]))
        ];

        return new InputSet(values);
    }

    private static int[] ResolveMetadataDims(
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata,
        string inputName,
        int[] fallback)
    {
        if (!inputMetadata.TryGetValue(inputName, out NodeMetadata? metadata) ||
            metadata.Dimensions.Length != fallback.Length)
        {
            return fallback;
        }

        var dims = new int[fallback.Length];
        for (int index = 0; index < dims.Length; index++)
        {
            dims[index] = metadata.Dimensions[index] > 0 ? metadata.Dimensions[index] : fallback[index];
        }

        return dims;
    }

    private static InputSet CreateMetadataDrivenInputs(IReadOnlyDictionary<string, NodeMetadata> inputMetadata)
    {
        var values = new List<NamedOnnxValue>(inputMetadata.Count);
        foreach ((string inputName, NodeMetadata metadata) in inputMetadata)
        {
            values.Add(CreateSmokeTensorInput(inputName, metadata));
        }

        return new InputSet(values);
    }

    private static readonly IReadOnlyDictionary<TensorElementType, Func<string, int, int[], NamedOnnxValue>> SmokeTensorFactories =
        new Dictionary<TensorElementType, Func<string, int, int[], NamedOnnxValue>>
        {
            [TensorElementType.Float] = static (name, count, dims) =>
                NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(new float[count], dims)),
            [TensorElementType.Float16] = static (name, count, dims) =>
                NamedOnnxValue.CreateFromTensor(name, new DenseTensor<Float16>(new Float16[count], dims)),
            [TensorElementType.Int32] = static (name, count, dims) =>
                NamedOnnxValue.CreateFromTensor(name, new DenseTensor<int>(new int[count], dims)),
            [TensorElementType.Int64] = static (name, count, dims) =>
                NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(new long[count], dims)),
            [TensorElementType.Bool] = static (name, count, dims) =>
                NamedOnnxValue.CreateFromTensor(name, new DenseTensor<bool>(new bool[count], dims)),
        };

    private static NamedOnnxValue CreateSmokeTensorInput(string inputName, NodeMetadata metadata)
    {
        if (!metadata.IsTensor)
        {
            throw new NotSupportedException($"Smoke test input '{inputName}' is not a tensor input.");
        }

        int[] dimensions = metadata.Dimensions.Select(static dimension => dimension > 0 ? dimension : 1).ToArray();
        int elementCount = dimensions.Aggregate(1, static (product, dimension) => checked(product * dimension));

        if (SmokeTensorFactories.TryGetValue(metadata.ElementDataType, out Func<string, int, int[], NamedOnnxValue>? factory))
        {
            return factory(inputName, elementCount, dimensions);
        }

        throw new NotSupportedException(
            $"Smoke test input '{inputName}' uses unsupported tensor element type '{metadata.ElementDataType}'.");
    }

    private static void EnsureSelectedProviderMatchesRequested(
        ExecutionProviderKind requestedProvider,
        string selectedProvider,
        bool allowPolicyProviderSelection = false)
    {
        string requestedProviderLabel = FormatProviderLabel(requestedProvider);
        if (allowPolicyProviderSelection || string.Equals(selectedProvider, requestedProviderLabel, StringComparison.OrdinalIgnoreCase))
        {
            return;
        }

        throw new InvalidOperationException(
            $"Smoke test requested provider '{requestedProviderLabel}' but session creation selected effective provider '{selectedProvider}'.");
    }

    private static string FormatProviderLabel(ExecutionProviderKind provider) =>
        ProviderNames.FormatProviderLabel(provider);

    private sealed class InputSet(IReadOnlyList<NamedOnnxValue> values) : IDisposable
    {
        public IReadOnlyList<NamedOnnxValue> Values { get; } = values;

        public void Dispose()
        {
            foreach (IDisposable value in Values.OfType<IDisposable>())
            {
                value.Dispose();
            }
        }
    }
}
