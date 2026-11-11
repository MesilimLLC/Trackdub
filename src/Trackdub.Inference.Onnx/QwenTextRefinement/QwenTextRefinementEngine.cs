using Microsoft.ML.OnnxRuntimeGenAI;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Runtime.Routing;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.Runtime;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.QwenTextRefinement;

public sealed class QwenTextRefinementEngine(
    IRuntimePlanner runtimePlanner,
    BenchmarkModelPathResolver modelPathResolver,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null)
    : ITextRefinementEngine, IStageRuntimeExecutionReporter
{
    public const string EngineFamilyName = "qwen-instruct";

    /// <summary>Cold-start evidence source label for the GenAI token-generation choke point.</summary>
    internal const string GenerateSource = "genai:qwen-instruct";

    private const string GenAiConfigFileName = "genai_config.json";
    private const int MaxNewTokens = 256;
    private const double RepetitionPenalty = 1.05;

    private static readonly string[] EosTokenSuffixes =
        ["<|im_end|>", "<|endoftext|>", "<|assistant|>"];

    private readonly IRuntimePlanner runtimePlanner = runtimePlanner ?? throw new ArgumentNullException(nameof(runtimePlanner));
    private readonly BenchmarkModelPathResolver modelPathResolver = modelPathResolver ?? throw new ArgumentNullException(nameof(modelPathResolver));

    public StageRuntimeExecutionSummary? LastExecutionSummary { get; private set; }

    public string EngineFamily => EngineFamilyName;

    public async Task<IReadOnlyList<RefinedTextSegment>> RefineAsync(
        TextRefinementRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.Segments);

        if (request.Segments.Count == 0)
        {
            LastExecutionSummary = null;
            return [];
        }

        StageRuntimePlan plan = await runtimePlanner.PlanAsync(
            await StageRuntimePlanningRequestFactory.ApplyPreferredModelTierAsync(
                new StageRuntimePlanningRequest(
                    RuntimeStage.TextRefinement,
                    PreferredModelAlias: request.PreferredModelAlias,
                    PreferredExecutionProvider: ExecutionProviderRequest.ParsePreferredExecutionProvider(
                        request.PreferredExecutionProvider,
                        request.RequirePreferredExecutionProvider),
                    RequirePreferredExecutionProvider: request.RequirePreferredExecutionProvider,
                    PreferredModelVariantAlias: request.PreferredModelVariantAlias),
                runtimePlanningPreferences,
                cancellationToken),
            cancellationToken).ConfigureAwait(false);

        EnsurePlanReady(plan, RuntimeStage.TextRefinement);

        string modelRootPath = PlannedRuntimeModelResolver.ResolveModelRootPath(plan, modelPathResolver);
        EnsureGenAiModelRoot(modelRootPath);

        GenAiModelKey modelKey = await GenAiModelKey.CreateAsync(
            modelRootPath, plan.ExecutionProvider!.Value, plan.ModelId, plan.Variant, plan.DeviceIndex,
            plan.ModelRevisionHash, cancellationToken).ConfigureAwait(false);
        IReadOnlyList<RefinedTextSegment> refinedSegments = await RefineWithLeaseAsync(
            request,
            modelKey.Provider,
            token => GenAiModelPool.Shared.GetLeaseAsync(modelKey, token),
            cancellationToken).ConfigureAwait(false);

        LastExecutionSummary = CreateExecutionSummary(plan, "ONNX Runtime GenAI Qwen text refinement.");
        return refinedSegments;
    }

    // The lease factory is a seam so the native-failure handling can be tested without a native model.
    internal static async Task<IReadOnlyList<RefinedTextSegment>> RefineWithLeaseAsync(
        TextRefinementRequest request,
        ExecutionProviderKind provider,
        Func<CancellationToken, Task<GenAiModelLease>> acquireLease,
        CancellationToken cancellationToken)
    {
        var refinedSegments = new List<RefinedTextSegment>(request.Segments.Count);
        try
        {
            using GenAiModelLease modelLease = await acquireLease(cancellationToken).ConfigureAwait(false);
            using IDisposable? executionAdmission = await CpuExecutionAdmission.Shared
                .AcquireAsync(provider, cancellationToken).ConfigureAwait(false);
            Model model = modelLease.Model;
            Tokenizer tokenizer = modelLease.Tokenizer;

            foreach (TextRefinementInputSegment segment in request.Segments.OrderBy(static s => s.Index))
            {
                cancellationToken.ThrowIfCancellationRequested();
                refinedSegments.Add(RefineSegment(model, tokenizer, request, segment, cancellationToken));
            }
        }
        catch (Exception ex) when (IsNativeRuntimeFailure(ex))
        {
            throw WrapNativeFailure(ex);
        }

        return refinedSegments;
    }

    // Native model creation and generation throw runtime-specific exception types, and loading the
    // native libraries throws DllNotFoundException, EntryPointNotFoundException or BadImageFormatException
    // (possibly inside a TypeInitializationException). The optional polish stage handles none of them;
    // surface them as InvalidOperationException so the stage skips polish and keeps the raw ASR
    // transcript instead of failing the whole run. FileNotFoundException is left alone: the stage
    // already handles it.
    internal static bool IsNativeRuntimeFailure(Exception ex) =>
        ex is OnnxRuntimeGenAIException
            or Microsoft.ML.OnnxRuntime.OnnxRuntimeException
            or DllNotFoundException
            or EntryPointNotFoundException
            or BadImageFormatException
        || (ex is TypeInitializationException { InnerException: { } inner } && IsNativeRuntimeFailure(inner));

    internal static InvalidOperationException WrapNativeFailure(Exception ex) =>
        new($"Qwen text refinement failed in ONNX Runtime: {ex.Message}", ex);

    private static RefinedTextSegment RefineSegment(
        Model model,
        Tokenizer tokenizer,
        TextRefinementRequest request,
        TextRefinementInputSegment segment,
        CancellationToken cancellationToken)
    {
        string originalText = segment.Text;
        if (string.IsNullOrWhiteSpace(originalText))
        {
            return new RefinedTextSegment(
                segment.Index,
                segment.StartSeconds,
                segment.EndSeconds,
                originalText,
                originalText,
                originalText,
                Accepted: false,
                TextRefinementGuardStatus.Unchanged,
                [TextRefinementCorrectionCodes.FallbackUnchanged]);
        }

        string prompt = QwenInstructPromptBuilder.BuildPrompt(
            request.Scope,
            originalText,
            request.SourceLanguage,
            request.TargetLanguage);
        string modelOutput = GenerateSegmentText(model, tokenizer, prompt, cancellationToken);
        QwenRefinementGuardResult guardResult = QwenRefinementOutputGuard.Evaluate(originalText, modelOutput);

        return new RefinedTextSegment(
            segment.Index,
            segment.StartSeconds,
            segment.EndSeconds,
            originalText,
            guardResult.CleanedOutput,
            guardResult.DisplayedText,
            guardResult.Accepted,
            guardResult.GuardStatus,
            guardResult.AppliedCorrections);
    }

    private static string GenerateSegmentText(
        Model model,
        Tokenizer tokenizer,
        string prompt,
        CancellationToken cancellationToken)
    {
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
        return CleanDecodedOutput(decoded);
    }

    internal static string CleanDecodedOutput(string decoded)
    {
        if (string.IsNullOrWhiteSpace(decoded))
        {
            return string.Empty;
        }

        string result = decoded;
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

    private static void EnsurePlanReady(StageRuntimePlan plan, RuntimeStage stage)
    {
        if (plan.IsRunnable() && plan.ExecutionProvider is not null)
        {
            return;
        }

        throw new InvalidOperationException(
            plan.Fallback?.Detail ??
            $"Runtime planner did not produce a ready {stage} plan.");
    }

    private static void EnsureGenAiModelRoot(string modelRootPath)
    {
        string configPath = Path.Join(modelRootPath, GenAiConfigFileName);
        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException(
                "Qwen GenAI model root does not contain genai_config.json.", configPath);
        }
    }

    private static StageRuntimeExecutionSummary CreateExecutionSummary(
        StageRuntimePlan plan,
        string bootstrapDetail) =>
        new(
            "auto",
            plan.ExecutionProvider!.Value.ToString().ToLowerInvariant(),
            plan.ModelId,
            plan.ModelAlias,
            plan.Variant,
            plan.Fallback is null ? bootstrapDetail : $"{bootstrapDetail} {plan.Fallback.Detail}");
}
