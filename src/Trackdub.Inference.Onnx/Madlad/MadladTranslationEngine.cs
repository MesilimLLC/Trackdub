using System.Runtime.CompilerServices;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.Runtime.Routing;
using Trackdub.Inference.Onnx.Translation;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Runtime.Planning;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Trackdub.Inference.Onnx.Madlad;

public sealed class MadladTranslationEngine(IRuntimePlanner runtimePlanner,
    BenchmarkModelPathResolver modelPathResolver,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null)
    : IStreamingTranslationEngineAdapter, IStageRuntimeExecutionReporter
{
    public const string EngineFamilyName = "madlad";

    private readonly IRuntimePlanner runtimePlanner = runtimePlanner ?? throw new ArgumentNullException(nameof(runtimePlanner));
    private readonly BenchmarkModelPathResolver modelPathResolver = modelPathResolver ?? throw new ArgumentNullException(nameof(modelPathResolver));

    public StageRuntimeExecutionSummary? LastExecutionSummary { get; private set; }

    public string EngineFamily => EngineFamilyName;

    public async Task<IReadOnlyList<TranslatedTextSegment>> TranslateAsync(
        TranslationRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.Segments);

        StageRuntimePlanningRequest planningRequest = await StageRuntimePlanningRequestFactory.ApplyPreferredModelTierAsync(
            new StageRuntimePlanningRequest(
                RuntimeStage.Translation,
                PreferredModelAlias: request.PreferredModelAlias,
                SourceLanguage: request.SourceLanguage,
                TargetLanguage: request.TargetLanguage,
                PreferredExecutionProvider: ExecutionProviderRequest.ParsePreferredExecutionProvider(
                    request.PreferredExecutionProvider,
                    request.RequirePreferredExecutionProvider),
                RequirePreferredExecutionProvider: request.RequirePreferredExecutionProvider,
                PreferredModelVariantAlias: request.PreferredModelVariantAlias, RequirePreferredModelAlias: RequiresRoutedModelAlias(request)),
            runtimePlanningPreferences,
            cancellationToken).ConfigureAwait(false);

        StageRuntimePlan plan = await runtimePlanner.PlanAsync(planningRequest, cancellationToken).ConfigureAwait(false);
        EnsurePlanReady(plan, RuntimeStage.Translation);

        string encoderModelPath = ResolveEncoderModelPath(plan, request.ResolvedModelEntryPath);
        string decoderModelPath = ResolveDecoderModelPath(plan, encoderModelPath);
        string modelRootPath = ResolveModelRootPath(encoderModelPath);
        MadladTokenizerDecoder tokenizer = await MadladTokenizerDecoder.LoadAsync(modelRootPath).ConfigureAwait(false);
        string targetLanguageTag = ResolveTargetLanguageTag(request.TargetLanguage);

        if (request.Segments.Count == 0)
        {
            LastExecutionSummary = CreatePlannedOnlySummary(plan, "Translation skipped because the transcript did not contain any segments.");
            return [];
        }

        // GPU-first with CPU fallback: the 3B MADLAD encoder+decoder bundle (~6.4GB for the
        // quantized export at 2x, ~8.1GB for the external-data fp16 export on TensorRT RTX)
        // can exceed the accelerator admission budget on smaller GPUs, or stall behind GPU
        // memory the pool cannot evict. On those admission failures, re-plan pinned to
        // CPU and retry instead of failing the stage. An explicitly required GPU pin
        // (RequirePreferredExecutionProvider) is honored — no silent fallback.
        StageRuntimePlan effectivePlan = plan;
        string effectiveEncoderModelPath = encoderModelPath;
        string effectiveDecoderModelPath = decoderModelPath;
        OnnxExecutionSessionFactory.OpusSessionLease? sessionLease = null;
        try
        {
            try
            {
                sessionLease = await AcquireLeaseAsync(
                    effectivePlan, effectiveEncoderModelPath, effectiveDecoderModelPath, cancellationToken)
                    .ConfigureAwait(false);
            }
            catch (Exception ex) when (IsAcceleratorSessionFailure(ex, effectivePlan) && CanFallBackToCpu(request, effectivePlan))
            {
                effectivePlan = await ReplanForCpuAsync(request with { PreferredModelAlias = effectivePlan.ModelAlias ?? request.PreferredModelAlias }, ex, cancellationToken).ConfigureAwait(false);
                effectiveEncoderModelPath = ResolveEncoderModelPath(effectivePlan, null);
                effectiveDecoderModelPath = ResolveDecoderModelPath(effectivePlan, effectiveEncoderModelPath);
                tokenizer = await MadladTokenizerDecoder.LoadAsync(ResolveModelRootPath(effectiveEncoderModelPath)).ConfigureAwait(false);
                sessionLease = await OnnxExecutionSessionFactory
                    .CreatePooledOpusAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "madlad", Provider = ExecutionProviderKind.Cpu, CancellationToken = cancellationToken }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = effectiveEncoderModelPath, Decoder = effectiveDecoderModelPath })
                    .ConfigureAwait(false);
            }

            var translatedSegments = new List<TranslatedTextSegment>(request.Segments.Count);
            foreach (TranslationInputSegment segment in request.Segments.OrderBy(static segment => segment.Index))
            {
                cancellationToken.ThrowIfCancellationRequested();
                string translatedText = await TranslateSegmentAsync(
                    sessionLease,
                    tokenizer,
                    targetLanguageTag,
                    segment.Text,
                    cancellationToken).ConfigureAwait(false);

                translatedSegments.Add(new TranslatedTextSegment(
                    segment.Index,
                    segment.StartSeconds,
                    segment.EndSeconds,
                    translatedText));
            }

            LastExecutionSummary = CreateExecutionSummary(effectivePlan, sessionLease);
            return translatedSegments;
        }
        finally
        {
            sessionLease?.Dispose();
        }
    }

    /// <summary>
    /// Emits each finalized translated segment, acquiring and disposing the pooled session
    /// bundle around only that segment so backpressure never pins native resources.
    /// </summary>
    public async IAsyncEnumerable<PipelineStreamItem<TranslatedTextSegment>> TranslateStreamAsync(
        TranslationRequest request,
        Guid runId,
        string snapshotId,
        Guid sourceRevisionId,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.Segments);
        PipelineStreamItemFactory.ValidateTranslationStreamContext(runId, snapshotId, sourceRevisionId);

        StageRuntimePlanningRequest planningRequest = await StageRuntimePlanningRequestFactory.ApplyPreferredModelTierAsync(
            new StageRuntimePlanningRequest(
                RuntimeStage.Translation,
                PreferredModelAlias: request.PreferredModelAlias,
                SourceLanguage: request.SourceLanguage,
                TargetLanguage: request.TargetLanguage,
                PreferredExecutionProvider: ExecutionProviderRequest.ParsePreferredExecutionProvider(
                    request.PreferredExecutionProvider,
                    request.RequirePreferredExecutionProvider),
                RequirePreferredExecutionProvider: request.RequirePreferredExecutionProvider,
                PreferredModelVariantAlias: request.PreferredModelVariantAlias, RequirePreferredModelAlias: RequiresRoutedModelAlias(request)),
            runtimePlanningPreferences,
            cancellationToken).ConfigureAwait(false);

        StageRuntimePlan plan = await runtimePlanner.PlanAsync(planningRequest, cancellationToken).ConfigureAwait(false);
        EnsurePlanReady(plan, RuntimeStage.Translation);

        string encoderModelPath = ResolveEncoderModelPath(plan, request.ResolvedModelEntryPath);
        string decoderModelPath = ResolveDecoderModelPath(plan, encoderModelPath);
        string modelRootPath = ResolveModelRootPath(encoderModelPath);
        MadladTokenizerDecoder tokenizer = await MadladTokenizerDecoder.LoadAsync(modelRootPath).ConfigureAwait(false);
        string targetLanguageTag = ResolveTargetLanguageTag(request.TargetLanguage);

        if (request.Segments.Count == 0)
        {
            LastExecutionSummary = CreatePlannedOnlySummary(plan, "Translation skipped because the transcript did not contain any segments.");
            yield break;
        }

        long sequence = 0;
        StageRuntimePlan effectivePlan = plan;
        string effectiveEncoderModelPath = encoderModelPath;
        string effectiveDecoderModelPath = decoderModelPath;
        foreach (TranslationInputSegment segment in request.Segments.OrderBy(static segment => segment.Index))
        {
            cancellationToken.ThrowIfCancellationRequested();

            TranslatedTextSegment translated;
            try
            {
                using (OnnxExecutionSessionFactory.OpusSessionLease sessionLease = await AcquireLeaseAsync(
                    effectivePlan, effectiveEncoderModelPath, effectiveDecoderModelPath, cancellationToken)
                    .ConfigureAwait(false))
                {
                    string translatedText = await TranslateSegmentAsync(
                        sessionLease,
                        tokenizer,
                        targetLanguageTag,
                        segment.Text,
                        cancellationToken).ConfigureAwait(false);
                    translated = new TranslatedTextSegment(
                        segment.Index, segment.StartSeconds, segment.EndSeconds, translatedText);
                    LastExecutionSummary = CreateExecutionSummary(effectivePlan, sessionLease);
                }
            }
            catch (Exception ex) when (IsAcceleratorSessionFailure(ex, effectivePlan) && CanFallBackToCpu(request, effectivePlan))
            {
                effectivePlan = await ReplanForCpuAsync(request with { PreferredModelAlias = effectivePlan.ModelAlias ?? request.PreferredModelAlias }, ex, cancellationToken).ConfigureAwait(false);
                effectiveEncoderModelPath = ResolveEncoderModelPath(effectivePlan, null);
                effectiveDecoderModelPath = ResolveDecoderModelPath(effectivePlan, effectiveEncoderModelPath);
                tokenizer = await MadladTokenizerDecoder.LoadAsync(ResolveModelRootPath(effectiveEncoderModelPath)).ConfigureAwait(false);
                using (OnnxExecutionSessionFactory.OpusSessionLease sessionLease = await OnnxExecutionSessionFactory
                    .CreatePooledOpusAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "madlad", Provider = ExecutionProviderKind.Cpu, CancellationToken = cancellationToken }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = effectiveEncoderModelPath, Decoder = effectiveDecoderModelPath })
                    .ConfigureAwait(false))
                {
                    string translatedText = await TranslateSegmentAsync(
                        sessionLease,
                        tokenizer,
                        targetLanguageTag,
                        segment.Text,
                        cancellationToken).ConfigureAwait(false);
                    translated = new TranslatedTextSegment(
                        segment.Index, segment.StartSeconds, segment.EndSeconds, translatedText);
                    LastExecutionSummary = CreateExecutionSummary(effectivePlan, sessionLease);
                }
            }

            yield return PipelineStreamItemFactory.CreateTranslation(
                translated, runId, snapshotId, sourceRevisionId, sequence++);
        }
    }

    private static async Task<OnnxExecutionSessionFactory.OpusSessionLease> AcquireLeaseAsync(
        StageRuntimePlan plan,
        string encoderModelPath,
        string decoderModelPath,
        CancellationToken cancellationToken)
    {
        // Classic TensorRT takes the same bounded profiles as TensorRT-RTX so production loads the
        // engines that the classic warm path built (EpContextTrtProfiles.Resolve).
        bool boundedTensorRtProfiles = plan.ExecutionProvider is ExecutionProviderKind.TensorRTRtx or ExecutionProviderKind.TensorRt;
        bool classicTensorRt = plan.ExecutionProvider is ExecutionProviderKind.TensorRt;
        return await OnnxExecutionSessionFactory
            .CreatePooledOpusAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "madlad", Provider = plan.ExecutionProvider!.Value, CancellationToken = cancellationToken, AdditionalTrtEncoderOptions = boundedTensorRtProfiles ? (classicTensorRt ? MadladTrtProfiles.ClassicEncoder : MadladTrtProfiles.Encoder) : null, AdditionalTrtDecoderOptions = boundedTensorRtProfiles ? (classicTensorRt ? MadladTrtProfiles.ClassicDecoder : MadladTrtProfiles.Decoder) : null }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = encoderModelPath, Decoder = decoderModelPath })
            .ConfigureAwait(false);
    }

    private static bool IsAdmissionBudgetFailure(Exception ex) =>
        ex is InvalidOperationException &&
        ex.Message.Contains("admission budget", StringComparison.OrdinalIgnoreCase);

    // TensorRT RTX builds its engines while the session is created, so too little free VRAM
    // surfaces there as an ONNX Runtime error rather than as an admission-budget refusal.
    private static bool IsAcceleratorSessionFailure(Exception ex, StageRuntimePlan plan) =>
        IsAdmissionBudgetFailure(ex) ||
        (plan.ExecutionProvider is ExecutionProviderKind.TensorRTRtx && ex is OnnxRuntimeException);

    private static bool CanFallBackToCpu(TranslationRequest request, StageRuntimePlan plan) =>
        !request.RequirePreferredExecutionProvider &&
        plan.ExecutionProvider is not null &&
        plan.ExecutionProvider != ExecutionProviderKind.Cpu;

    internal async Task<StageRuntimePlan> ReplanForCpuAsync(
        TranslationRequest request,
        Exception acceleratorFailure,
        CancellationToken cancellationToken)
    {
        StageRuntimePlanningRequest cpuRequest = await StageRuntimePlanningRequestFactory.ApplyPreferredModelTierAsync(
            new StageRuntimePlanningRequest(
                RuntimeStage.Translation,
                PreferredModelAlias: request.PreferredModelAlias,
                SourceLanguage: request.SourceLanguage,
                TargetLanguage: request.TargetLanguage,
                PreferredExecutionProvider: ExecutionProviderKind.Cpu,
                RequirePreferredExecutionProvider: true,
                PreferredModelVariantAlias: null,
                RequirePreferredModelAlias: RequiresRoutedModelAlias(request)),
            runtimePlanningPreferences,
            cancellationToken).ConfigureAwait(false);

        StageRuntimePlan cpuPlan = await runtimePlanner.PlanAsync(cpuRequest, cancellationToken).ConfigureAwait(false);
        if (cpuPlan.Status == StageRuntimePlanStatus.DownloadRequired)
        {
            throw new InvalidOperationException(
                "MADLAD accelerator initialization failed and CPU fallback requires a downloaded CPU-compatible model. "
                + "Download the MADLAD quantized or int4-kv variant in Model Manager, then retry. "
                + "The trt-fp16 bundle cannot run on CPU.", acceleratorFailure);
        }

        EnsurePlanReady(cpuPlan, RuntimeStage.Translation);
        return cpuPlan;
    }


    // The router has already chosen this MADLAD model. A soft alias lets the planner fall through
    // to another language-pair model (e.g. OPUS-MT) when the requested variant is not ready, and
    // the engine would then load a non-MADLAD model directory.
    private static bool RequiresRoutedModelAlias(TranslationRequest request) =>
        !string.IsNullOrWhiteSpace(request.PreferredModelAlias);

    internal static async Task SmokeTestAsync(
        ExecutionProviderSmokeTestRequest request, string encoderPath, string decoderPath,
        CancellationToken cancellationToken)
    {
        using OnnxExecutionSessionFactory.OpusSessionLease lease = await AcquireLeaseAsync(
            new StageRuntimePlan { ExecutionProvider = request.ExecutionProvider },
            encoderPath, decoderPath, cancellationToken).ConfigureAwait(false);
        if (lease.SelectedProviderKind != request.ExecutionProvider)
        {
            throw new InvalidOperationException($"MADLAD smoke requested {request.ExecutionProvider} but selected {lease.SelectedProvider}.");
        }

        MadladTokenizerDecoder tokenizer = await MadladTokenizerDecoder.LoadAsync(ResolveModelRootPath(encoderPath)).ConfigureAwait(false);
        // Exercise production padding and bucketed decoding on the same pooled CUDA-graph sessions.
        await TranslateSegmentAsync(lease, tokenizer, "<2en>", "Hello world.", cancellationToken, maxSteps: 2).ConfigureAwait(false);
    }

    private static Task<string> TranslateSegmentAsync(
        OnnxExecutionSessionFactory.OpusSessionLease sessionLease,
        MadladTokenizerDecoder tokenizer,
        string targetLanguageTag,
        string text,
        CancellationToken cancellationToken,
        int? maxSteps = null)
    {
        long[] inputIds = tokenizer.EncodeSourceText(text, targetLanguageTag);
        long[] attentionMask = Enumerable.Repeat(1L, inputIds.Length).ToArray();
        // TensorRT RTX compiles kernels per input shape, so feed it a few padded lengths; the
        // attention mask hides the padding from the encoder and from cross-attention.
        bool staticShapes = sessionLease.SelectedProviderKind is ExecutionProviderKind.TensorRTRtx;
        if (staticShapes)
        {
            MadladTrtProfiles.ValidateLengths(inputIds.Length, maxSteps ?? Math.Max(8, tokenizer.MaxGenerationLength));
            int paddedLength = Seq2SeqGreedyDecoder.BucketLength(inputIds.Length);
            long[] paddedIds = new long[paddedLength];
            inputIds.CopyTo(paddedIds);
            Array.Fill(paddedIds, (long)tokenizer.PadTokenId, inputIds.Length, paddedLength - inputIds.Length);
            inputIds = paddedIds;
            Array.Resize(ref attentionMask, paddedLength);
        }

        using var encoderInputs = CreateEncoderInputs(
            sessionLease.EncoderSession.InputMetadata,
            inputIds,
            attentionMask);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults =
            sessionLease.EncoderSession.RunWithRetry(encoderInputs.Values, cancellationToken: cancellationToken, provider: sessionLease.SelectedProviderKind);
        Tensor<float> encoderHiddenStates = encoderResults
            .Single(static result => result.Name == "last_hidden_state")
            .AsTensor<float>();

        List<long> generatedTokens = Seq2SeqGreedyDecoder.Decode(
            sessionLease.DecoderSession,
            encoderHiddenStates,
            attentionMask,
            tokenizer.DecoderStartTokenId,
            tokenizer.EndOfSentenceTokenId,
            tokenizer.PadTokenId,
            maxSteps ?? Math.Max(8, tokenizer.MaxGenerationLength),
            provider: sessionLease.SelectedProviderKind,
            cancellationToken,
            staticShapes);
        string translatedText = tokenizer.DecodeTargetText(generatedTokens);

        return Task.FromResult(string.IsNullOrWhiteSpace(translatedText)
            ? string.Empty
            : translatedText);
    }

    private static InputSet CreateEncoderInputs(
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata,
        IReadOnlyList<long> inputIds,
        IReadOnlyList<long> attentionMask)
    {
        var values = new List<NamedOnnxValue>(inputMetadata.Count);
        foreach ((string inputName, _) in inputMetadata)
        {
            values.Add(inputName switch
            {
                "input_ids" => NamedOnnxValue.CreateFromTensor(
                    "input_ids",
                    new DenseTensor<long>(inputIds.ToArray(), [1, inputIds.Count])),
                "attention_mask" => NamedOnnxValue.CreateFromTensor(
                    "attention_mask",
                    new DenseTensor<long>(attentionMask.ToArray(), [1, attentionMask.Count])),
                _ => throw new NotSupportedException($"MADLAD encoder input '{inputName}' is not supported.")
            });
        }

        return new InputSet(values);
    }

    private static void EnsurePlanReady(StageRuntimePlan plan, RuntimeStage stage)
    {
        if (plan.IsRunnable() &&
            plan.ExecutionProvider is not null &&
            !string.IsNullOrWhiteSpace(plan.ModelAlias))
        {
            return;
        }

        throw new InvalidOperationException(
            plan.Fallback?.Detail ??
            $"Runtime planner did not produce a ready {stage} plan.");
    }

    private string ResolveEncoderModelPath(StageRuntimePlan plan, string? resolvedModelEntryPath)
    {
        if (!string.IsNullOrWhiteSpace(plan.ModelEntryPath))
        {
            return Path.GetFullPath(plan.ModelEntryPath);
        }

        if (!string.IsNullOrWhiteSpace(resolvedModelEntryPath))
        {
            return Path.GetFullPath(resolvedModelEntryPath);
        }

        if (!string.IsNullOrWhiteSpace(plan.Variant))
        {
            BenchmarkModelCandidate variantCandidate = modelPathResolver.ResolveSingle(plan.ModelAlias!, plan.Variant);
            string fileName = Path.GetFileName(variantCandidate.ModelPath);
            if (fileName.StartsWith("encoder_model", StringComparison.OrdinalIgnoreCase))
            {
                return variantCandidate.ModelPath;
            }
        }

        BenchmarkModelCandidate candidate = modelPathResolver.ResolveSingle(plan.ModelAlias!);
        return candidate.ModelPath;
    }

    private string ResolveDecoderModelPath(StageRuntimePlan plan, string encoderModelPath)
    {
        string modelRootPath = ResolveModelRootPath(encoderModelPath);
        // TensorRT RTX decodes with static shapes, which re-run the plain decoder; the merged
        // decoder's cache branch would change shape every step.
        string[] decoderFileNames = plan.ExecutionProvider is ExecutionProviderKind.TensorRTRtx
            ? ["decoder_model.onnx", "decoder_model_merged.onnx", "decoder_model_quantized.onnx", "decoder_model_int8.onnx"]
            : ["decoder_model_merged.onnx", "decoder_model_quantized.onnx", "decoder_model_int8.onnx", "decoder_model.onnx"];
        foreach (string fileName in decoderFileNames)
        {
            string candidatePath = Path.Join(modelRootPath, fileName);
            if (File.Exists(candidatePath))
            {
                return Path.GetFullPath(candidatePath);
            }
        }

        if (!string.IsNullOrWhiteSpace(plan.Variant))
        {
            BenchmarkModelCandidate candidate = modelPathResolver.ResolveSingle(plan.ModelAlias!, plan.Variant);
            string fileName = Path.GetFileName(candidate.ModelPath);
            if (fileName.StartsWith("decoder_model", StringComparison.OrdinalIgnoreCase))
            {
                return candidate.ModelPath;
            }
        }

        foreach (string fileName in decoderFileNames)
        {
            string candidatePath = Path.Join(modelRootPath, fileName);
            if (File.Exists(candidatePath))
            {
                return Path.GetFullPath(candidatePath);
            }
        }

        throw new FileNotFoundException("The MADLAD decoder model was not found next to the encoder model.", encoderModelPath);
    }

    private static string ResolveModelRootPath(string encoderModelPath)
    {
        string? onnxDirectory = Path.GetDirectoryName(encoderModelPath);
        return onnxDirectory is null
            ? throw new InvalidOperationException("The MADLAD model root path could not be resolved.")
            : onnxDirectory;
    }

    private static string ResolveTargetLanguageTag(string targetLanguage)
    {
        if (!TranslationLanguageCoverageMatrix.TryGetLanguage(targetLanguage, out TranslationLanguageDefinition? definition))
        {
            throw new InvalidOperationException($"MADLAD does not know the target language tag for '{targetLanguage}'.");
        }

        return $"<2{definition!.MadladTag}>";
    }

    private static StageRuntimeExecutionSummary CreateExecutionSummary(
        StageRuntimePlan plan,
        OnnxExecutionSessionFactory.OpusSessionLease sessionLease) =>
        new(
            sessionLease.RequestedProvider,
            sessionLease.SelectedProvider,
            plan.ModelId,
            plan.ModelAlias,
            plan.Variant,
            sessionLease.BootstrapDetail);

    private static StageRuntimeExecutionSummary CreatePlannedOnlySummary(
        StageRuntimePlan plan,
        string bootstrapDetail) =>
        new(
            "auto",
            plan.ExecutionProvider is ExecutionProviderKind.DirectMl ? "dml" : "cpu",
            plan.ModelId,
            plan.ModelAlias,
            plan.Variant,
            bootstrapDetail);

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
