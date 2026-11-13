using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.ML.OnnxRuntimeGenAI;
using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.Runtime;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Onnx.Runtime.Routing;
using Trackdub.Inference.Onnx.Worker;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.Phi;

public enum GenAiTranslationPromptFormat
{
    Phi,
    Qwen3,
    QwenChatMl,
    Gemma4,
    Gemma3
}

public sealed class PhiGenAiTranslationEngine(
    IRuntimePlanner runtimePlanner,
    BenchmarkModelPathResolver modelPathResolver,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null,
    IApplicationLogger? applicationLogger = null,
    string engineFamily = PhiGenAiTranslationEngine.EngineFamilyName,
    IInferenceWorkerClient? workerClient = null,
    bool inferenceWorkerProcess = false)
    : IStreamingTranslationEngineAdapter, IStageRuntimeExecutionReporter, ITranslationExecutionMetadataReporter
{
    public const string EngineFamilyName = "phi-genai";
    public const string QwenEngineFamilyName = "qwen-genai";
    public const string GemmaEngineFamilyName = "gemma-genai";

    /// <summary>Cold-start evidence source label for the GenAI token-generation choke point.</summary>
    internal const string GenerateSource = "genai:phi";

    private const string GenAiConfigFileName = "genai_config.json";
    private const string HfConfigFileName = "config.json";
    private const int MaxNewTokens = 512;
    private const double RepetitionPenalty = 1.1;
    private const string PromptDefangingCharacter = "\u200B";

    private static readonly (string OpenTag, string CloseTag)[] ThinkingBlockTags =
    [
        ("<think>", "</think>"),
        ("<|thought>", "<thought|>"),
        ("<|channel>thought\n", "<channel|>")
    ];

    private static readonly string[] EosTokenSuffixes =
    [
        "<|end|>",
        "<|endoftext|>",
        "<|assistant|>",
        "<|im_end|>",
        "<end_of_turn>",
        "<turn|>",
        "<eos>"
    ];

    private static readonly IReadOnlyDictionary<GenAiTranslationPromptFormat, string[]> PromptControlTokens =
        new Dictionary<GenAiTranslationPromptFormat, string[]>
        {
            [GenAiTranslationPromptFormat.Qwen3] = ["<|im_start|>", "<|im_end|>", "<|im_sep|>"],
            [GenAiTranslationPromptFormat.QwenChatMl] = ["<|im_start|>", "<|im_end|>", "<|im_sep|>"],
            [GenAiTranslationPromptFormat.Gemma3] = ["<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>"],
            [GenAiTranslationPromptFormat.Gemma4] = ["<|turn>", "<turn|>", "<bos>", "<eos>"],
            [GenAiTranslationPromptFormat.Phi] = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|endoftext|>"]
        };

    private static readonly IReadOnlyDictionary<string, string> TargetLanguageNames =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["ar"] = "Arabic",
            ["da"] = "Danish",
            ["de"] = "German",
            ["el"] = "Greek",
            ["en"] = "English",
            ["es"] = "Spanish",
            ["fi"] = "Finnish",
            ["fr"] = "French",
            ["he"] = "Hebrew",
            ["hi"] = "Hindi",
            ["it"] = "Italian",
            ["ja"] = "Japanese",
            ["ko"] = "Korean",
            ["ms"] = "Malay",
            ["nl"] = "Dutch",
            ["no"] = "Norwegian",
            ["pl"] = "Polish",
            ["pt"] = "Portuguese",
            ["ru"] = "Russian",
            ["sv"] = "Swedish",
            ["sw"] = "Swahili",
            ["tr"] = "Turkish",
            ["zh"] = "Chinese",
            ["zh-hans"] = "Simplified Chinese",
            ["zh-hant"] = "Traditional Chinese"
        };

    private readonly IRuntimePlanner runtimePlanner = runtimePlanner ?? throw new ArgumentNullException(nameof(runtimePlanner));
    private readonly BenchmarkModelPathResolver modelPathResolver = modelPathResolver ?? throw new ArgumentNullException(nameof(modelPathResolver));
    private readonly IApplicationLogger? applicationLogger = applicationLogger;
    private readonly string engineFamily = string.IsNullOrWhiteSpace(engineFamily) ? EngineFamilyName : engineFamily.Trim();
    private readonly bool inferenceWorkerProcess = inferenceWorkerProcess;

    public StageRuntimeExecutionSummary? LastExecutionSummary { get; private set; }

    public TranslationExecutionMetadata? LastExecutionMetadata { get; private set; }

    public string EngineFamily => engineFamily;

    /// <summary>
    /// Set by the inference worker for an optimization-only warmup: the model lease must not evict a
    /// resident model, and the request is skipped with <see cref="GenAiWarmupSkippedException"/> when it would.
    /// </summary>
    internal bool NonEvictingWarmup { get; init; }

    public async Task<IReadOnlyList<TranslatedTextSegment>> TranslateAsync(
        TranslationRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.Segments);

        StageRuntimePlan plan = await PlanTranslationAsync(request, cancellationToken).ConfigureAwait(false);
        EnsurePlanReady(plan);
        LastExecutionSummary = null;
        SetExecutionMetadata(request, plan);

        if (request.Segments.Count == 0)
        {
            LastExecutionSummary = CreateExecutionSummary(plan, "Translation skipped: no segments.", ran: false);
            SetExecutionMetadata(request, plan);
            return [];
        }

        if (UseCudaWorker(plan))
        {
            WorkerTranslationResult result = await workerClient!.TranslateAsync(
                request, plan, cancellationToken, nonEvictingWarmup: NonEvictingWarmup).ConfigureAwait(false);
            LastExecutionSummary = result.ExecutionSummary;
            SetExecutionMetadata(request, plan);
            return result.Segments;
        }

        string modelRootPath = PlannedRuntimeModelResolver.ResolveModelRootPath(plan, modelPathResolver);
        LogModelProvenance(plan, modelRootPath);
        EnsureGenAiModelRoot(modelRootPath);

        GenAiModelKey modelKey = await GenAiModelKey.CreateAsync(
            modelRootPath, plan.ExecutionProvider!.Value, plan.ModelId, plan.Variant, plan.DeviceIndex,
            plan.ModelRevisionHash, cancellationToken).ConfigureAwait(false);
        using GenAiModelLease modelLease = await AcquireModelLeaseAsync(modelKey, cancellationToken).ConfigureAwait(false);
        using IDisposable? executionAdmission = await CpuExecutionAdmission.Shared
            .AcquireAsync(modelKey.Provider, cancellationToken).ConfigureAwait(false);
        Model model = modelLease.Model;
        Tokenizer tokenizer = modelLease.Tokenizer;

        var translatedSegments = new List<TranslatedTextSegment>(request.Segments.Count);

        foreach (TranslationInputSegment segment in request.Segments.OrderBy(static s => s.Index))
        {
            cancellationToken.ThrowIfCancellationRequested();
            TranslationPromptContext promptContext = CreatePromptContext(request, plan, segment);
            string translatedText = TranslateSegment(
                model,
                tokenizer,
                segment.Text,
                promptContext,
                cancellationToken);
            translatedSegments.Add(new TranslatedTextSegment(
                segment.Index,
                segment.StartSeconds,
                segment.EndSeconds,
                translatedText));
        }

        LastExecutionSummary = CreateExecutionSummary(plan, $"ONNX Runtime GenAI {engineFamily} text generation.");
        SetExecutionMetadata(request, plan);
        return translatedSegments;
    }

    /// <summary>
    /// Emits each finalized translated segment, releasing the exclusive GenAI model lease
    /// and tokenizer before every yield so backpressure never pins native resources.
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

        StageRuntimePlan plan = await PlanTranslationAsync(request, cancellationToken).ConfigureAwait(false);
        EnsurePlanReady(plan);
        LastExecutionSummary = null;
        SetExecutionMetadata(request, plan);

        if (request.Segments.Count == 0)
        {
            LastExecutionSummary = CreateExecutionSummary(plan, "Translation skipped: no segments.", ran: false);
            SetExecutionMetadata(request, plan);
            yield break;
        }

        if (UseCudaWorker(plan))
        {
            var streamContext = new TranslationStreamContext(runId, snapshotId, sourceRevisionId);
            await foreach (PipelineStreamItem<TranslatedTextSegment> item in TranslateOnWorkerAsync(
                request, plan, streamContext, cancellationToken))
                yield return item;
            yield break;
        }

        string modelRootPath = PlannedRuntimeModelResolver.ResolveModelRootPath(plan, modelPathResolver);
        LogModelProvenance(plan, modelRootPath);
        EnsureGenAiModelRoot(modelRootPath);

        GenAiModelKey modelKey = await GenAiModelKey.CreateAsync(
            modelRootPath, plan.ExecutionProvider!.Value, plan.ModelId, plan.Variant, plan.DeviceIndex,
            plan.ModelRevisionHash, cancellationToken).ConfigureAwait(false);
        long sequence = 0;

        foreach (TranslationInputSegment segment in request.Segments.OrderBy(static s => s.Index))
        {
            cancellationToken.ThrowIfCancellationRequested();
            TranslationPromptContext promptContext = CreatePromptContext(request, plan, segment);

            TranslatedTextSegment translated;
            using (GenAiModelLease modelLease = await GenAiModelPool.Shared
                .GetLeaseAsync(modelKey, cancellationToken).ConfigureAwait(false))
            using (await CpuExecutionAdmission.Shared
                .AcquireAsync(modelKey.Provider, cancellationToken).ConfigureAwait(false))
            {
                Model model = modelLease.Model;
                Tokenizer tokenizer = modelLease.Tokenizer;
                string translatedText = TranslateSegment(
                    model,
                    tokenizer,
                    segment.Text,
                    promptContext,
                    cancellationToken);
                translated = new TranslatedTextSegment(
                    segment.Index,
                    segment.StartSeconds,
                    segment.EndSeconds,
                    translatedText);
            }

            LastExecutionSummary = CreateExecutionSummary(plan, $"ONNX Runtime GenAI {engineFamily} text generation.");
            SetExecutionMetadata(request, plan);
            yield return PipelineStreamItemFactory.CreateTranslation(
                translated, runId, snapshotId, sourceRevisionId, sequence++);
        }
    }

    private async Task<GenAiModelLease> AcquireModelLeaseAsync(GenAiModelKey modelKey, CancellationToken cancellationToken)
    {
        if (!NonEvictingWarmup)
        {
            return await GenAiModelPool.Shared.GetLeaseAsync(modelKey, cancellationToken).ConfigureAwait(false);
        }

        return await GenAiModelPool.Shared.TryGetWarmupLeaseAsync(modelKey, cancellationToken).ConfigureAwait(false)
            ?? throw new GenAiWarmupSkippedException(
                $"GenAI warmup skipped for '{modelKey}': loading it would evict a resident model " +
                "(GenAI pool at capacity or admission budget full). The stage will load it inline.");
    }

    private async IAsyncEnumerable<PipelineStreamItem<TranslatedTextSegment>> TranslateOnWorkerAsync(
        TranslationRequest request,
        StageRuntimePlan plan,
        TranslationStreamContext streamContext,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        // One worker call (chunked by the client) pays engine construction and model-key hashing once.
        WorkerTranslationResult result = await workerClient!.TranslateAsync(
            request, plan, cancellationToken, nonEvictingWarmup: NonEvictingWarmup).ConfigureAwait(false);
        LastExecutionSummary = result.ExecutionSummary;
        SetExecutionMetadata(request, plan);
        long sequence = 0;
        foreach (TranslatedTextSegment translated in result.Segments)
        {
            yield return PipelineStreamItemFactory.CreateTranslation(
                translated, streamContext.RunId, streamContext.SnapshotId,
                streamContext.SourceRevisionId, sequence++);
        }
    }

    private readonly record struct TranslationStreamContext(
        Guid RunId,
        string SnapshotId,
        Guid SourceRevisionId);

    private bool UseCudaWorker(StageRuntimePlan plan)
    {
        bool isWindowsCudaQwenHostPlan = OperatingSystem.IsWindows() &&
            plan.ExecutionProvider == ExecutionProviderKind.Cuda &&
            string.Equals(engineFamily, QwenEngineFamilyName, StringComparison.OrdinalIgnoreCase) &&
            !inferenceWorkerProcess;
        if (!isWindowsCudaQwenHostPlan)
        {
            return false;
        }

        if (workerClient is { IsInstalled: true })
        {
            return true;
        }

        throw new InvalidOperationException(
            "CUDA translation requires the inference worker, which is not installed.");
    }

    private void SetExecutionMetadata(TranslationRequest request, StageRuntimePlan plan) =>
        LastExecutionMetadata = new TranslationExecutionMetadata(
            engineFamily,
            LastExecutionSummary?.ModelId ?? plan.ModelId,
            LastExecutionSummary?.ModelAlias ?? plan.ModelAlias,
            LastExecutionSummary?.SelectedProvider ?? plan.ExecutionProvider?.ToString(),
            TranslationRoutingKind.Pivot,
            request.ModelProvenance);

    private async Task<StageRuntimePlan> PlanTranslationAsync(
        TranslationRequest request,
        CancellationToken cancellationToken)
    {
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
                PreferredModelVariantAlias: request.PreferredModelVariantAlias),
            runtimePlanningPreferences,
            cancellationToken).ConfigureAwait(false);
        return await runtimePlanner.PlanAsync(planningRequest, cancellationToken).ConfigureAwait(false);
    }

    private TranslationPromptContext CreatePromptContext(
        TranslationRequest request,
        StageRuntimePlan plan,
        TranslationInputSegment segment) =>
        new(
            ResolveTargetLanguageName(request.TargetLanguage),
            ResolvePromptFormat(plan.ModelId, plan.ModelAlias, engineFamily),
            SelectApplicableGlossaryHints(segment, request.GlossaryHints));

    private static string TranslateSegment(
        Model model,
        Tokenizer tokenizer,
        string sourceText,
        TranslationPromptContext promptContext,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(sourceText))
        {
            return string.Empty;
        }

        string prompt = BuildTranslationPrompt(
            sourceText, promptContext.TargetLanguageName, promptContext.PromptFormat, promptContext.GlossaryHints);
        using Sequences inputTokens = tokenizer.Encode(prompt);
        if (inputTokens.NumSequences == 0)
        {
            return string.Empty;
        }

        using GeneratorParams generatorParams = new(model);
        generatorParams.SetSearchOption("max_length", (double)(inputTokens[0].Length + MaxNewTokens));
        generatorParams.SetSearchOption("repetition_penalty", RepetitionPenalty);

        using Generator generator = new(model, generatorParams);
        generator.AppendTokenSequences(inputTokens);

        // Capture prompt length before generation to decode only new tokens.
        int promptLength = (int)generator.GetSequence(0).Length;

        bool firstTokenGenerated = false;
        while (!generator.IsDone())
        {
            cancellationToken.ThrowIfCancellationRequested();
            generator.GenerateNextToken();
            if (!firstTokenGenerated)
            {
                firstTokenGenerated = true;
                ColdStartCapture.MarkInferenceCompleted(GenerateSource);
            }
        }

        ReadOnlySpan<int> fullSequence = generator.GetSequence(0);
        string decoded = tokenizer.Decode(fullSequence[promptLength..]);
        return CleanDecodedTranslation(decoded);
    }

    internal static string CleanDecodedTranslation(string decoded)
    {
        if (string.IsNullOrWhiteSpace(decoded))
        {
            return string.Empty;
        }

        string result = StripThinkingBlocks(decoded);
        foreach (string suffix in EosTokenSuffixes)
        {
            int suffixIndex = result.IndexOf(suffix, StringComparison.Ordinal);
            if (suffixIndex >= 0)
            {
                result = result[..suffixIndex];
            }
        }

        return result.Trim();
    }

    private static string StripThinkingBlocks(string text)
    {
        string result = text;
        foreach ((string openTag, string closeTag) in ThinkingBlockTags)
        {
            result = StripThinkingBlock(result, openTag, closeTag);
        }

        return result;
    }

    private static string StripThinkingBlock(string text, string openTag, string closeTag)
    {
        while (true)
        {
            int closeIndex = text.IndexOf(closeTag, StringComparison.Ordinal);
            if (closeIndex < 0)
            {
                int openIndex = text.IndexOf(openTag, StringComparison.Ordinal);
                return openIndex >= 0 ? text[..openIndex] : text;
            }

            int matchingOpenIndex = text.LastIndexOf(openTag, closeIndex, StringComparison.Ordinal);
            text = matchingOpenIndex >= 0
                ? string.Concat(text.AsSpan(0, matchingOpenIndex), text.AsSpan(closeIndex + closeTag.Length))
                : string.Concat(text.AsSpan(0, closeIndex), text.AsSpan(closeIndex + closeTag.Length));
        }
    }

    internal static GenAiTranslationPromptFormat ResolvePromptFormat(
        string? modelId,
        string? modelAlias,
        string? engineFamily = null)
    {
        string combined = $"{modelId} {modelAlias} {engineFamily}";

        if (combined.Contains("gemma-4", StringComparison.OrdinalIgnoreCase) ||
            combined.Contains("gemma4", StringComparison.OrdinalIgnoreCase))
        {
            return GenAiTranslationPromptFormat.Gemma4;
        }

        if (combined.Contains("gemma", StringComparison.OrdinalIgnoreCase))
        {
            return GenAiTranslationPromptFormat.Gemma3;
        }

        if (combined.Contains("qwen3", StringComparison.OrdinalIgnoreCase) ||
            combined.Contains("qwen-3", StringComparison.OrdinalIgnoreCase))
        {
            return GenAiTranslationPromptFormat.Qwen3;
        }

        if (combined.Contains("qwen", StringComparison.OrdinalIgnoreCase))
        {
            return GenAiTranslationPromptFormat.QwenChatMl;
        }

        return GenAiTranslationPromptFormat.Phi;
    }

    internal static IReadOnlyList<TranslationGlossaryHint> SelectApplicableGlossaryHints(
        TranslationInputSegment segment,
        IReadOnlyList<TranslationGlossaryHint>? glossaryHints)
    {
        if (glossaryHints is null || glossaryHints.Count == 0 || string.IsNullOrWhiteSpace(segment.Text))
        {
            return [];
        }

        var applicable = new List<TranslationGlossaryHint>();
        var seenPairs = new HashSet<(string SourceTerm, string TargetTerm)>();

        foreach (TranslationGlossaryHint hint in glossaryHints)
        {
            if (MatchesSegment(segment, hint) && seenPairs.Add((hint.SourceTerm.Trim(), hint.TargetTerm.Trim())))
            {
                applicable.Add(hint);
            }
        }

        return applicable;
    }

    private static bool MatchesSegment(TranslationInputSegment segment, TranslationGlossaryHint hint)
    {
        if (string.IsNullOrWhiteSpace(hint.SourceTerm) || string.IsNullOrWhiteSpace(hint.TargetTerm))
        {
            return false;
        }

        if (hint.SourceMatches is { Count: > 0 })
        {
            return hint.SourceMatches.Any(match => match.SegmentIndex == segment.Index);
        }

        return segment.Text.Contains(
            hint.SourceTerm,
            hint.IsCaseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);
    }

    private const int MaxGlossaryHintsInInstruction = 32;
    private const int MaxGlossaryTermLength = 100;

    internal static string BuildSystemInstruction(
        string targetLanguageName,
        IReadOnlyList<TranslationGlossaryHint>? glossaryHints = null,
        GenAiTranslationPromptFormat promptFormat = GenAiTranslationPromptFormat.Phi)
    {
        var builder = new StringBuilder();
        builder.Append($"Translate the input to {DefangPromptControlTokens(targetLanguageName, promptFormat)}. Output only the translation, no explanation.");

        if (glossaryHints is { Count: > 0 })
        {
            builder.Append(" Adhere strictly to the following glossary terms: ");
            builder.Append(string.Join(
                ", ",
                glossaryHints
                    .Take(MaxGlossaryHintsInInstruction)
                    .Select(g => $"{BoundGlossaryTerm(g.SourceTerm, promptFormat)} -> {BoundGlossaryTerm(g.TargetTerm, promptFormat)}")));
            builder.Append('.');
        }

        return builder.ToString();
    }

    private static string BoundGlossaryTerm(string term, GenAiTranslationPromptFormat promptFormat)
    {
        string flattened = term.Trim().ReplaceLineEndings(" ");
        string bounded = flattened.Length <= MaxGlossaryTermLength
            ? flattened
            : flattened[..MaxGlossaryTermLength];
        return DefangPromptControlTokens(bounded, promptFormat);
    }

    internal static string BuildTranslationPrompt(
        string sourceText,
        string targetLanguageName,
        GenAiTranslationPromptFormat promptFormat = GenAiTranslationPromptFormat.Phi,
        IReadOnlyList<TranslationGlossaryHint>? glossaryHints = null)
    {
        string systemInstruction = BuildSystemInstruction(targetLanguageName, glossaryHints, promptFormat);
        string safeSourceText = DefangPromptControlTokens(sourceText, promptFormat);

        return promptFormat switch
        {
            GenAiTranslationPromptFormat.Qwen3 =>
                $"<|im_start|>system\n{systemInstruction}<|im_end|>\n" +
                $"<|im_start|>user\n{safeSourceText}<|im_end|>\n" +
                "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            GenAiTranslationPromptFormat.QwenChatMl =>
                $"<|im_start|>system\n{systemInstruction}<|im_end|>\n" +
                $"<|im_start|>user\n{safeSourceText}<|im_end|>\n" +
                "<|im_start|>assistant\n",
            GenAiTranslationPromptFormat.Gemma4 =>
                $"<bos><|turn>system\n{systemInstruction}<turn|>\n" +
                $"<|turn>user\n{safeSourceText}<turn|>\n" +
                "<|turn>model\n",
            GenAiTranslationPromptFormat.Gemma3 =>
                $"<start_of_turn>user\n{systemInstruction}\n\n{safeSourceText}<end_of_turn>\n" +
                "<start_of_turn>model\n",
            _ =>
                $"<|system|>\n{systemInstruction}<|end|>\n" +
                $"<|user|>\n{safeSourceText}<|end|>\n" +
                "<|assistant|>\n"
        };
    }

    private static string DefangPromptControlTokens(
        string text,
        GenAiTranslationPromptFormat promptFormat)
    {
        if (!PromptControlTokens.TryGetValue(promptFormat, out string[]? controlTokens))
        {
            controlTokens = PromptControlTokens[GenAiTranslationPromptFormat.Phi];
        }

        foreach (string token in controlTokens)
        {
            text = text.Replace(token, token.Insert(1, PromptDefangingCharacter), StringComparison.Ordinal);
        }

        return text;
    }

    internal static string ResolveTargetLanguageName(string targetLanguageCode)
    {
        string code = targetLanguageCode.Trim();
        return TargetLanguageNames.TryGetValue(code, out string? name) ? name : code;
    }

    private sealed record TranslationPromptContext(
        string TargetLanguageName,
        GenAiTranslationPromptFormat PromptFormat,
        IReadOnlyList<TranslationGlossaryHint> GlossaryHints);

    private static void EnsurePlanReady(StageRuntimePlan plan)
    {
        if (plan.IsRunnable() && plan.ExecutionProvider is not null && !string.IsNullOrWhiteSpace(plan.ModelAlias))
        {
            return;
        }

        throw new InvalidOperationException(
            plan.Fallback?.Detail ?? "Runtime planner did not produce a ready translation plan for Phi GenAI.");
    }

    private static void EnsureGenAiModelRoot(string modelRootPath)
    {
        string configPath = Path.Join(modelRootPath, GenAiConfigFileName);
        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException(
                "Phi GenAI model root does not contain genai_config.json.", configPath);
        }
    }

    /// <summary>
    /// Records the resolved GenAI model provenance before the native model is loaded: the root
    /// ORT GenAI will read, whether the files it requires are present, and which planner-visible
    /// identity (provider/model/variant/revision) the run is attributed to. Mirrors the OPUS
    /// tokenizer provenance line so a cache-shaped model root is diagnosable from trackdub.log.
    /// <c>genai_config.json</c> is the only config ORT GenAI needs; the Hugging Face
    /// <c>config.json</c> is reported because the desktop model cache does not guarantee it.
    /// </summary>
    private void LogModelProvenance(StageRuntimePlan plan, string modelRootPath) =>
        applicationLogger?.LogInformation(
            $"{(engineFamily == EngineFamilyName ? "Phi" : engineFamily)} GenAI model provenance: root='{modelRootPath}'; " +
            $"genai_config.json={(File.Exists(Path.Join(modelRootPath, GenAiConfigFileName)) ? "present" : "absent")}; " +
            $"config.json={(File.Exists(Path.Join(modelRootPath, HfConfigFileName)) ? "present" : "absent")}; " +
            $"provider={plan.ExecutionProvider!.Value.ToString().ToLowerInvariant()}; " +
            $"model={plan.ModelId}; variant={plan.Variant}; " +
            $"revision={plan.ModelRevisionHash ?? "unresolved"}.");

    private static StageRuntimeExecutionSummary CreateExecutionSummary(
        StageRuntimePlan plan,
        string bootstrapDetail,
        bool ran = true) =>
        new(
            "auto",
            ran ? plan.ExecutionProvider!.Value.ToString().ToLowerInvariant() : "not-run",
            plan.ModelId,
            plan.ModelAlias,
            plan.Variant,
            plan.Fallback is null ? bootstrapDetail : $"{bootstrapDetail} {plan.Fallback.Detail}");
}
