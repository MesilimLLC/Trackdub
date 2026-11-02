using System.Runtime.CompilerServices;
using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.Runtime.Routing;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Runtime.Planning;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Trackdub.Inference.Onnx.OpusMt;

public sealed class OpusMtTranslationEngine(IRuntimePlanner runtimePlanner,
    BenchmarkModelPathResolver modelPathResolver,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null,
    IApplicationLogger? applicationLogger = null)
    : IStreamingTranslationEngineAdapter, IStageRuntimeExecutionReporter, ITranslationExecutionMetadataReporter
{
    public const string EngineFamilyName = "opus-mt";

    private readonly IRuntimePlanner runtimePlanner = runtimePlanner ?? throw new ArgumentNullException(nameof(runtimePlanner));
    private readonly BenchmarkModelPathResolver modelPathResolver = modelPathResolver ?? throw new ArgumentNullException(nameof(modelPathResolver));
    private readonly IApplicationLogger? applicationLogger = applicationLogger;

    public StageRuntimeExecutionSummary? LastExecutionSummary { get; private set; }

    public TranslationExecutionMetadata? LastExecutionMetadata { get; private set; }

    public string EngineFamily => EngineFamilyName;

    public async Task<IReadOnlyList<TranslatedTextSegment>> TranslateAsync(
        TranslationRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.Segments);

        StageRuntimePlan plan = await runtimePlanner.PlanAsync(
            await StageRuntimePlanningRequestFactory.ApplyPreferredModelTierAsync(new StageRuntimePlanningRequest(
                RuntimeStage.Translation,
                PreferredModelAlias: request.PreferredModelAlias,
                SourceLanguage: request.SourceLanguage,
                TargetLanguage: request.TargetLanguage,
                PreferredExecutionProvider: ExecutionProviderRequest.ParsePreferredExecutionProvider(
                    request.PreferredExecutionProvider,
                    request.RequirePreferredExecutionProvider),
                RequirePreferredExecutionProvider: request.RequirePreferredExecutionProvider,
                PreferredModelVariantAlias: request.PreferredModelVariantAlias),
            runtimePlanningPreferences,
            cancellationToken),
            cancellationToken).ConfigureAwait(false);
        EnsurePlanReady(plan, RuntimeStage.Translation);

        string encoderModelPath = ResolveEncoderModelPath(plan, request.ResolvedModelEntryPath);
        string decoderModelPath = ResolveDecoderModelPath(plan, encoderModelPath);
        string modelRootPath = ResolveModelRootPath(encoderModelPath);
        OpusTokenizerDecoder tokenizer = await LoadTokenizerAsync(modelRootPath).ConfigureAwait(false);
        string? targetPrefix = tokenizer.ResolveTargetLanguagePrefix(request.TargetLanguage, request.ModelProvenance?.TargetPrefix);
        if (request.ModelProvenance is { CatalogPolicyVersion: not null } provenance &&
            !string.Equals(provenance.TargetPrefix, targetPrefix, StringComparison.Ordinal))
        {
            throw new InvalidOperationException("The translation catalog target prefix does not match the installed tokenizer.");
        }
        LastExecutionMetadata = new TranslationExecutionMetadata("opus-mt", plan.ModelId, plan.ModelAlias,
            plan.ExecutionProvider?.ToString(), TranslationRoutingKind.Direct,
            request.ModelProvenance is { } identity ? identity with { TargetPrefix = targetPrefix } : null);

        if (tokenizer.RequiresTargetLanguagePrefix && targetPrefix is null)
        {
            throw new InvalidOperationException(
                $"The selected translation model does not support target language '{request.TargetLanguage}'. " +
                $"Choose a model that covers this language pair.");
        }

        if (request.Segments.Count == 0)
        {
            LastExecutionSummary = CreatePlannedOnlySummary(plan, "Translation skipped because the transcript did not contain any segments.");
            return [];
        }

        using OnnxExecutionSessionFactory.OpusSessionLease sessionLease = await OnnxExecutionSessionFactory
            .CreatePooledOpusAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "opus-mt", Provider = plan.ExecutionProvider!.Value, CancellationToken = cancellationToken }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = encoderModelPath, Decoder = decoderModelPath })
            .ConfigureAwait(false);

        var translatedSegments = new List<TranslatedTextSegment>(request.Segments.Count);
        foreach (TranslationInputSegment segment in request.Segments.OrderBy(static segment => segment.Index))
        {
            cancellationToken.ThrowIfCancellationRequested();
            string translatedText = await TranslateSegmentAsync(
                sessionLease,
                tokenizer,
                segment.Text,
                targetPrefix,
                cancellationToken).ConfigureAwait(false);

            translatedSegments.Add(new TranslatedTextSegment(
                segment.Index,
                segment.StartSeconds,
                segment.EndSeconds,
                translatedText));
        }

        LastExecutionSummary = CreateExecutionSummary(plan, sessionLease);
        return translatedSegments;
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

        StageRuntimePlan plan = await runtimePlanner.PlanAsync(
            await StageRuntimePlanningRequestFactory.ApplyPreferredModelTierAsync(new StageRuntimePlanningRequest(
                RuntimeStage.Translation,
                PreferredModelAlias: request.PreferredModelAlias,
                SourceLanguage: request.SourceLanguage,
                TargetLanguage: request.TargetLanguage,
                PreferredExecutionProvider: ExecutionProviderRequest.ParsePreferredExecutionProvider(
                    request.PreferredExecutionProvider,
                    request.RequirePreferredExecutionProvider),
                RequirePreferredExecutionProvider: request.RequirePreferredExecutionProvider,
                PreferredModelVariantAlias: request.PreferredModelVariantAlias),
            runtimePlanningPreferences,
            cancellationToken),
            cancellationToken).ConfigureAwait(false);
        EnsurePlanReady(plan, RuntimeStage.Translation);

        string encoderModelPath = ResolveEncoderModelPath(plan, request.ResolvedModelEntryPath);
        string decoderModelPath = ResolveDecoderModelPath(plan, encoderModelPath);
        string modelRootPath = ResolveModelRootPath(encoderModelPath);
        OpusTokenizerDecoder tokenizer = await LoadTokenizerAsync(modelRootPath).ConfigureAwait(false);
        string? targetPrefix = tokenizer.ResolveTargetLanguagePrefix(request.TargetLanguage, request.ModelProvenance?.TargetPrefix);
        if (request.ModelProvenance is { CatalogPolicyVersion: not null } provenance &&
            !string.Equals(provenance.TargetPrefix, targetPrefix, StringComparison.Ordinal))
        {
            throw new InvalidOperationException("The translation catalog target prefix does not match the installed tokenizer.");
        }
        LastExecutionMetadata = new TranslationExecutionMetadata("opus-mt", plan.ModelId, plan.ModelAlias,
            plan.ExecutionProvider?.ToString(), TranslationRoutingKind.Direct,
            request.ModelProvenance is { } identity ? identity with { TargetPrefix = targetPrefix } : null);

        if (tokenizer.RequiresTargetLanguagePrefix && targetPrefix is null)
        {
            throw new InvalidOperationException(
                $"The selected translation model does not support target language '{request.TargetLanguage}'. " +
                $"Choose a model that covers this language pair.");
        }

        if (request.Segments.Count == 0)
        {
            LastExecutionSummary = CreatePlannedOnlySummary(plan, "Translation skipped because the transcript did not contain any segments.");
            yield break;
        }

        long sequence = 0;
        foreach (TranslationInputSegment segment in request.Segments.OrderBy(static segment => segment.Index))
        {
            cancellationToken.ThrowIfCancellationRequested();

            TranslatedTextSegment translated;
            using (OnnxExecutionSessionFactory.OpusSessionLease sessionLease = await OnnxExecutionSessionFactory
                .CreatePooledOpusAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "opus-mt", Provider = plan.ExecutionProvider!.Value, CancellationToken = cancellationToken }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = encoderModelPath, Decoder = decoderModelPath })
                .ConfigureAwait(false))
            {
                string translatedText = await TranslateSegmentAsync(
                    sessionLease,
                    tokenizer,
                    segment.Text,
                    targetPrefix,
                    cancellationToken).ConfigureAwait(false);
                translated = new TranslatedTextSegment(
                    segment.Index, segment.StartSeconds, segment.EndSeconds, translatedText);
                LastExecutionSummary = CreateExecutionSummary(plan, sessionLease);
            }

            yield return PipelineStreamItemFactory.CreateTranslation(
                translated, runId, snapshotId, sourceRevisionId, sequence++);
        }
    }

    /// <summary>
    /// Loads the Opus tokenizer and records provenance for troubleshooting: the resolved special
    /// token ids and whether the optional Marian config files were present. The desktop model cache
    /// only guarantees the manifest's ONNX/tokenizer files, so a missing <c>config.json</c> is the
    /// normal shape and the ids are derived from the vocabulary instead.
    /// </summary>
    private async Task<OpusTokenizerDecoder> LoadTokenizerAsync(string modelRootPath)
    {
        OpusTokenizerDecoder tokenizer = await OpusTokenizerDecoder.LoadAsync(modelRootPath).ConfigureAwait(false);
        applicationLogger?.LogInformation(
            $"Opus tokenizer provenance: root='{modelRootPath}'; " +
            $"config.json={(tokenizer.ConfigFilePresent ? "present" : "absent")}; " +
            $"generation_config.json={(tokenizer.GenerationConfigFilePresent ? "present" : "absent")}; " +
            $"vocab={tokenizer.VocabularySize}; decoder_start={tokenizer.DecoderStartTokenId}; " +
            $"pad={tokenizer.PadTokenId}; eos={tokenizer.EndOfSentenceTokenId}; " +
            $"max_generation_length={tokenizer.MaxGenerationLength}.");
        return tokenizer;
    }

    private static async Task<string> TranslateSegmentAsync(
        OnnxExecutionSessionFactory.OpusSessionLease sessionLease,
        OpusTokenizerDecoder tokenizer,
        string text,
        string? targetLanguagePrefix,
        CancellationToken cancellationToken)
    {
        long[] inputIds = tokenizer.EncodeSourceText(text, targetLanguagePrefix);
        long[] attentionMask = Enumerable.Repeat(1L, inputIds.Length).ToArray();

        using var encoderInputs = CreateEncoderInputs(
            sessionLease.EncoderSession.InputMetadata,
            inputIds,
            attentionMask);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults =
            sessionLease.EncoderSession.RunWithRetry(encoderInputs.Values, cancellationToken: cancellationToken);
        Tensor<float> encoderHiddenStates = encoderResults
            .Single(static result => result.Name == "last_hidden_state")
            .AsTensor<float>();

        List<long> generatedTokens = await GreedyDecodeAsync(
            sessionLease.DecoderSession,
            tokenizer,
            encoderHiddenStates,
            attentionMask,
            cancellationToken).ConfigureAwait(false);
        string translatedText = tokenizer.DecodeTargetText(generatedTokens);

        return string.IsNullOrWhiteSpace(translatedText)
            ? string.Empty
            : translatedText;
    }

    private static Task<List<long>> GreedyDecodeAsync(
        InferenceSession decoderSession,
        OpusTokenizerDecoder tokenizer,
        Tensor<float> encoderHiddenStates,
        IReadOnlyList<long> attentionMask,
        CancellationToken cancellationToken)
    {
        var generatedTokens = new List<long> { tokenizer.DecoderStartTokenId };
        int maxSteps = Math.Max(8, tokenizer.MaxGenerationLength);

        for (int step = 0; step < maxSteps; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            using var decoderInputs = CreateDecoderInputs(
                decoderSession.InputMetadata,
                encoderHiddenStates,
                attentionMask,
                generatedTokens);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults = decoderSession.RunWithRetry(decoderInputs.Values, cancellationToken: cancellationToken);
            Tensor<float> logits = decoderResults
                .Single(static result => result.Name == "logits")
                .AsTensor<float>();

            int sequenceLength = logits.Dimensions[1];
            int vocabularySize = logits.Dimensions[2];
            int nextToken = SelectNextToken(
                logits,
                sequenceLength - 1,
                vocabularySize,
                tokenizer.PadTokenId);
            if (nextToken == tokenizer.EndOfSentenceTokenId || nextToken < 0)
            {
                break;
            }

            generatedTokens.Add(nextToken);
        }

        return Task.FromResult(generatedTokens.Skip(1).ToList());
    }

    private static int SelectNextToken(
        Tensor<float> logits,
        int timeIndex,
        int vocabularySize,
        int padTokenId)
    {
        int bestToken = -1;
        float bestValue = float.NegativeInfinity;
        for (int tokenIndex = 0; tokenIndex < vocabularySize; tokenIndex++)
        {
            if (tokenIndex == padTokenId)
            {
                continue;
            }

            float value = logits[0, timeIndex, tokenIndex];
            if (value > bestValue)
            {
                bestValue = value;
                bestToken = tokenIndex;
            }
        }

        return bestToken;
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
                _ => throw new NotSupportedException($"Opus encoder input '{inputName}' is not supported.")
            });
        }

        return new InputSet(values);
    }

    private static InputSet CreateDecoderInputs(
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata,
        Tensor<float> encoderHiddenStates,
        IReadOnlyList<long> attentionMask,
        IReadOnlyList<long> generatedTokens)
    {
        var values = new List<NamedOnnxValue>(inputMetadata.Count);
        DenseTensor<long> inputIdsTensor = new(generatedTokens.ToArray(), [1, generatedTokens.Count]);
        DenseTensor<long> attentionMaskTensor = new(attentionMask.ToArray(), [1, attentionMask.Count]);

        foreach ((string inputName, _) in inputMetadata)
        {
            values.Add(inputName switch
            {
                "input_ids" => NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                "encoder_hidden_states" => NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                "attention_mask" => NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
                "encoder_attention_mask" => NamedOnnxValue.CreateFromTensor("encoder_attention_mask", attentionMaskTensor),
                "use_cache_branch" => NamedOnnxValue.CreateFromTensor("use_cache_branch", new DenseTensor<bool>(new[] { false }, new[] { 1 })),
                _ when inputName.StartsWith("past_key_values.", StringComparison.Ordinal) =>
                    NamedOnnxValue.CreateFromTensor(inputName, CreateEmptyPastTensor(inputMetadata[inputName])),
                _ => throw new NotSupportedException($"Opus decoder input '{inputName}' is not supported.")
            });
        }

        return new InputSet(values);
    }

    private static DenseTensor<float> CreateEmptyPastTensor(NodeMetadata metadata)
    {
        int[] sourceDims = metadata.Dimensions;
        int[] dims = new int[sourceDims.Length];
        for (int i = 0; i < sourceDims.Length; i++)
            dims[i] = sourceDims[i] > 0 ? sourceDims[i] : 1;
        dims[0] = 1;                       // batch = 1
        if (dims.Length > 2) dims[2] = 0; // sequence = 0 (empty cache)
        return new(Array.Empty<float>(), dims);
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
        if (!string.IsNullOrWhiteSpace(resolvedModelEntryPath))
        {
            return Path.GetFullPath(resolvedModelEntryPath);
        }

        BenchmarkModelCandidate candidate = modelPathResolver.ResolveSingle(plan.ModelAlias!);
        return candidate.ModelPath;
    }

    private string ResolveDecoderModelPath(StageRuntimePlan plan, string encoderModelPath)
    {
        string modelRootPath = ResolveModelRootPath(encoderModelPath);
        string encoderDirectory = Path.GetDirectoryName(encoderModelPath)
            ?? throw new InvalidOperationException("The Opus encoder model directory could not be resolved.");
        foreach (string directory in new[] { encoderDirectory, modelRootPath }.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            foreach (string fileName in new[] { "decoder_model_merged.onnx", "decoder_model.onnx", "decoder_model_int8.onnx" })
            {
                string candidatePath = Path.Join(directory, fileName);
                if (File.Exists(candidatePath))
                {
                    return Path.GetFullPath(candidatePath);
                }
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

        string decoderModelPath = Path.Join(modelRootPath, "decoder_model.onnx");
        if (File.Exists(decoderModelPath))
        {
            return Path.GetFullPath(decoderModelPath);
        }

        string mergedDecoderModelPath = Path.Join(modelRootPath, "decoder_model_merged.onnx");
        if (File.Exists(mergedDecoderModelPath))
        {
            return Path.GetFullPath(mergedDecoderModelPath);
        }

        throw new FileNotFoundException("The Opus decoder model was not found next to the encoder model.", decoderModelPath);
    }

    private static string ResolveModelRootPath(string encoderModelPath)
    {
        string? onnxDirectory = Path.GetDirectoryName(encoderModelPath);
        if (onnxDirectory is null)
        {
            throw new InvalidOperationException("The Opus model root path could not be resolved.");
        }

        if (HasTokenizerFiles(onnxDirectory))
        {
            return onnxDirectory;
        }

        string? parentDirectory = Path.GetDirectoryName(onnxDirectory);
        return parentDirectory is not null && HasTokenizerFiles(parentDirectory)
            ? parentDirectory
            : onnxDirectory;
    }

    private static bool HasTokenizerFiles(string directory) =>
        File.Exists(Path.Join(directory, "vocab.json")) &&
        (File.Exists(Path.Join(directory, "source.spm")) || File.Exists(Path.Join(directory, "source.model"))) &&
        (File.Exists(Path.Join(directory, "target.spm")) || File.Exists(Path.Join(directory, "target.model")));

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
