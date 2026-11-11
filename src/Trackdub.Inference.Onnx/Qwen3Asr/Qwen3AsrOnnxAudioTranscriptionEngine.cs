using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Audio;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Onnx.Runtime.Routing;
using Trackdub.Inference.Runtime.Planning;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Trackdub.Inference.Onnx.Qwen3Asr;

public sealed class Qwen3AsrOnnxAudioTranscriptionEngine
    : IAudioTranscriptionEngineAdapter, IStageRuntimeExecutionReporter
{
    private readonly IRuntimePlanner runtimePlanner;
    private readonly IRuntimePlanningPreferences? runtimePlanningPreferences;
    private readonly BenchmarkModelPathResolver modelPathResolver;

    /// <summary>Production constructor: sidecar caches default to the process-wide instances.</summary>
    public Qwen3AsrOnnxAudioTranscriptionEngine(
        IRuntimePlanner runtimePlanner,
        BenchmarkModelPathResolver modelPathResolver,
        IRuntimePlanningPreferences? runtimePlanningPreferences = null)
        : this(runtimePlanner, modelPathResolver, runtimePlanningPreferences, null, null)
    {
    }

    /// <summary>Test seam: allows injecting caches so load counts can be asserted.</summary>
    internal Qwen3AsrOnnxAudioTranscriptionEngine(
        IRuntimePlanner runtimePlanner,
        BenchmarkModelPathResolver modelPathResolver,
        IRuntimePlanningPreferences? runtimePlanningPreferences,
        SidecarCache<Qwen3AsrTokenizer>? tokenizerCache,
        SidecarCache<Qwen3AsrEmbedTokens>? embedTokensCache)
    {
        this.runtimePlanner = runtimePlanner ?? throw new ArgumentNullException(nameof(runtimePlanner));
        this.modelPathResolver = modelPathResolver ?? throw new ArgumentNullException(nameof(modelPathResolver));
        this.runtimePlanningPreferences = runtimePlanningPreferences;
        this.tokenizerCache = tokenizerCache ?? Qwen3AsrSidecarCaches.Tokenizers;
        this.embedTokensCache = embedTokensCache ?? Qwen3AsrSidecarCaches.EmbedTokens;
    }

    public const string EngineFamilyName = "qwen3-asr";

    // The encoder sees a different mel length for almost every chunk, so capture would record a new
    // graph per length for little gain; it stays out of the CUDA graph default.
    internal static readonly IReadOnlyDictionary<string, string> TrtEncoderOptions = new Dictionary<string, string>
    {
        ["enable_cuda_graph"] = "0",
        ["trt_profile_min_shapes"] = "mel:1x128x1",
        ["trt_profile_max_shapes"] = "mel:1x128x3000",
        ["trt_profile_opt_shapes"] = "mel:1x128x3000",
    };

    private readonly Qwen3AsrMelFeatureExtractor featureExtractor = new();

    // Tokenizer and embedding table are audio-independent, so they are loaded once per model root
    // instead of on every transcription.
    private readonly SidecarCache<Qwen3AsrTokenizer> tokenizerCache;
    private readonly SidecarCache<Qwen3AsrEmbedTokens> embedTokensCache;

    public StageRuntimeExecutionSummary? LastExecutionSummary { get; private set; }

    public string EngineFamily => EngineFamilyName;

    public async Task<IReadOnlyList<RecognizedTranscriptSegment>> TranscribeAsync(
        string normalizedAudioPath,
        IReadOnlyList<SpeechRegion> regions,
        CancellationToken cancellationToken) =>
        await TranscribeAsync(
            new AudioTranscriptionRequest(normalizedAudioPath, regions),
            cancellationToken).ConfigureAwait(false);

    public async Task<IReadOnlyList<RecognizedTranscriptSegment>> TranscribeAsync(
        AudioTranscriptionRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        InferenceRequestOptions options = request.Options ?? InferenceRequestOptions.Default;

        StageRuntimePlan plan = await runtimePlanner.PlanAsync(
            await StageRuntimePlanningRequestFactory.ApplyPreferredModelTierAsync(
                new StageRuntimePlanningRequest(
                    RuntimeStage.Asr,
                    options.NormalizedPreferredModelAlias,
                    SourceLanguage: request.SourceLanguage,
                    RequirePreferredModelAlias: options.RequirePreferredModelAlias,
                    PreferredExecutionProvider: ExecutionProviderRequest.ParsePreferredExecutionProvider(
                        options.PreferredExecutionProvider,
                        options.RequirePreferredExecutionProvider),
                    RequirePreferredExecutionProvider: options.RequirePreferredExecutionProvider,
                    PreferredModelVariantAlias: options.NormalizedPreferredModelVariantAlias),
                runtimePlanningPreferences,
                cancellationToken),
            cancellationToken).ConfigureAwait(false);

        return await TranscribeAsync(request, plan, cancellationToken).ConfigureAwait(false);
    }

    public async Task<IReadOnlyList<RecognizedTranscriptSegment>> TranscribeAsync(
        AudioTranscriptionRequest request,
        StageRuntimePlan plan,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(request.Regions);
        EnsurePlanReady(plan, RuntimeStage.Asr);

        Qwen3AsrModelPaths modelPaths = ResolveModelPaths(ResolvePlannedModelPath(plan));
        // Cache keyed by model root: re-reading the fp16 embedding table per request costs a
        // ~300 MB file read that expands into a LOH-sized float[,] and is identical every time.
        //
        // The shared load is deliberately not cancelled by this caller: one transcription being
        // cancelled must not abort a tokenizer read that concurrent transcriptions are awaiting.
        // Cancellation applies to this caller's wait instead.
        var tokenizer = await tokenizerCache
            .GetOrAddAsync(modelPaths.ModelRootPath, static path => Qwen3AsrTokenizer.LoadAsync(path, CancellationToken.None))
            .WaitAsync(cancellationToken)
            .ConfigureAwait(false);
        Qwen3AsrEmbedTokens embedTokens = embedTokensCache
            .GetOrAdd(modelPaths.EmbedTokensPath, () => Qwen3AsrEmbedTokens.Load(modelPaths.EmbedTokensPath));

        IAudioSamples audio = await WaveAudioReader.ReadMonoPcm16Async(request.NormalizedAudioPath, cancellationToken).ConfigureAwait(false);
        using IAudioSamples targetAudio = AudioResampler.CreateResampledStream(audio, 16000);
        double durationSeconds = targetAudio.SampleFrameCount / 16000d;

        IReadOnlyList<SpeechRegion> effectiveRegions = request.Regions;
        if (request.Regions.Count == 0)
        {
            if (durationSeconds >= 30.0)
            {
                LastExecutionSummary = CreatePlannedOnlySummary(plan,
                    $"ASR skipped: VAD detected no speech regions in {durationSeconds:F1}s audio.");
                return [];
            }

            effectiveRegions = [new SpeechRegion(0, 0.0, durationSeconds)];
        }

        using OnnxExecutionSessionFactory.Qwen3AsrSessionLease sessionLease = await AcquireSessionLeaseAsync(
            plan, modelPaths, cancellationToken).ConfigureAwait(false);

        // Keep caller-provided regions (VAD or diarization plan). Whisper's nearby-region
        // merge would collapse multi-speaker turns into one clip-length segment and reindex
        // away SpeakerIdsBySegmentIndex.
        IReadOnlyList<SpeechRegion> transcriptionRegions = PreserveCallerSpeechRegions(effectiveRegions);
        string? forcedLanguageName = Qwen3AsrLanguageCodes.TryGetLanguageName(request.SourceLanguage);
        var segments = new List<RecognizedTranscriptSegment>(transcriptionRegions.Count);
        var chunkSkipReasons = new List<string>();

        foreach (SpeechRegion region in transcriptionRegions)
        {
            RegionTranscription regionResult = await TranscribeRegionAsync(
                sessionLease,
                tokenizer,
                embedTokens,
                targetAudio,
                region,
                forcedLanguageName,
                cancellationToken).ConfigureAwait(false);

            chunkSkipReasons.AddRange(regionResult.ChunkSkipReasons);

            if (string.IsNullOrWhiteSpace(regionResult.Text))
            {
                continue;
            }

            segments.Add(new RecognizedTranscriptSegment(
                region.Index,
                region.StartSeconds,
                region.EndSeconds,
                regionResult.Text,
                regionResult.DetectedLanguage,
                regionResult.Words));
        }

        LastExecutionSummary = CreateExecutionSummary(plan, sessionLease, chunkSkipReasons);
        return segments;
    }

    private async Task<RegionTranscription> TranscribeRegionAsync(
        OnnxExecutionSessionFactory.Qwen3AsrSessionLease sessionLease,
        Qwen3AsrTokenizer tokenizer,
        Qwen3AsrEmbedTokens embedTokens,
        IAudioSamples targetAudio,
        SpeechRegion region,
        string? forcedLanguageName,
        CancellationToken cancellationToken)
    {
        long startSample = Math.Max(0, (long)Math.Floor(region.StartSeconds * 16000d));
        long endSample = Math.Min(targetAudio.SampleFrameCount, (long)Math.Ceiling(region.EndSeconds * 16000d));
        if (endSample <= startSample)
        {
            return new RegionTranscription(string.Empty, DetectedLanguage: null, Words: [], ChunkSkipReasons: []);
        }

        float[] regionSamples = new float[checked((int)(endSample - startSample))];
        targetAudio.ReadMonoSamples(startSample, regionSamples);

        const int maxChunkSamples = (int)(28 * 16000);
        var chunkTexts = new List<string>();
        var detectedLanguages = new List<string>();
        var chunkSkipReasons = new List<string>();

        for (int offset = 0; offset < regionSamples.Length; offset += maxChunkSamples)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int chunkLength = Math.Min(maxChunkSamples, regionSamples.Length - offset);
            double chunkDurationSeconds = chunkLength / 16000d;
            ReadOnlySpan<float> chunkSpan = new(regionSamples, offset, chunkLength);
            DenseTensor<float> mel = featureExtractor.Extract(chunkSpan);
            int melFrames = mel.Dimensions[2];
            int audioTokenCount = Qwen3AsrFeatureLengths.GetEncoderOutputLength(melFrames);
            if (audioTokenCount <= 0)
            {
                // Explicit skip reason (mirrors the Whisper engines' dropped-chunk
                // accounting): a too-short chunk must not vanish silently.
                chunkSkipReasons.Add(FormatTooShortChunkSkipReason(melFrames, audioTokenCount));
                continue;
            }

            IReadOnlyList<int>? forcedSuffix = string.IsNullOrWhiteSpace(forcedLanguageName)
                ? null
                : Qwen3AsrPromptBuilder.BuildForcedLanguageSuffix(tokenizer, forcedLanguageName);
            IReadOnlyList<int> promptIds = Qwen3AsrPromptBuilder.BuildPromptIds(audioTokenCount, forcedSuffix);

            using var encoderInputs = new Qwen3AsrInputSet(
            [
                NamedOnnxValue.CreateFromTensor("mel", mel),
            ]);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults =
                sessionLease.EncoderSession.RunWithRetry(encoderInputs.Values, cancellationToken: cancellationToken);
            Tensor<float> audioFeatures = encoderResults.Single().AsTensor<float>();

            int maxTokens = Math.Min(512, Math.Max(48, (int)Math.Ceiling(chunkDurationSeconds * 12d) + 24));
            IReadOnlyList<int> generatedTokens = Qwen3AsrGreedyDecoder.Decode(
                sessionLease,
                embedTokens,
                audioFeatures,
                promptIds,
                maxTokens,
                cancellationToken);

            string decoded = tokenizer.Decode(generatedTokens);
            (string languageName, string text) = Qwen3AsrOutputParser.Parse(decoded, forcedLanguageName);
            if (!string.IsNullOrWhiteSpace(text))
            {
                chunkTexts.Add(text);
            }

            if (!string.IsNullOrWhiteSpace(languageName))
            {
                detectedLanguages.Add(languageName);
            }
        }

        return new RegionTranscription(
            string.Join(" ", chunkTexts).Trim(),
            ResolveDetectedLanguage(detectedLanguages),
            Words: [],
            ChunkSkipReasons: chunkSkipReasons);
    }

    private static string? ResolveDetectedLanguage(IReadOnlyList<string> detectedLanguages)
    {
        if (detectedLanguages.Count == 0)
        {
            return null;
        }

        string? pluralityLanguage = detectedLanguages
            .GroupBy(static language => language, StringComparer.OrdinalIgnoreCase)
            .OrderByDescending(static group => group.Count())
            .Select(static group => group.Key)
            .FirstOrDefault();

        return Qwen3AsrLanguageCodes.TryGetIsoCode(pluralityLanguage);
    }

    internal static IReadOnlyList<SpeechRegion> BuildTranscriptionRegionsForTesting(
        IReadOnlyList<SpeechRegion> regions) =>
        PreserveCallerSpeechRegions(regions);

    private static IReadOnlyList<SpeechRegion> PreserveCallerSpeechRegions(
        IReadOnlyList<SpeechRegion> regions) =>
        regions
            .Where(static region => region.EndSeconds > region.StartSeconds)
            .OrderBy(static region => region.Index)
            .ToArray();

    private static void EnsurePlanReady(StageRuntimePlan plan, RuntimeStage stage)
    {
        if (plan.IsRunnable() && plan.ExecutionProvider is not null && !string.IsNullOrWhiteSpace(plan.ModelAlias))
        {
            return;
        }

        throw new InvalidOperationException(
            plan.Fallback?.Detail ??
            $"Runtime planner did not produce a ready {stage} plan.");
    }

    private string ResolvePlannedModelPath(StageRuntimePlan plan) =>
        PlannedRuntimeModelResolver.ResolveModelPath(plan, modelPathResolver);

    private static Qwen3AsrModelPaths ResolveModelPaths(string encoderModelPath)
    {
        string modelRoot = Path.GetDirectoryName(encoderModelPath)
            ?? throw new InvalidOperationException("Qwen3-ASR model root path could not be resolved.");
        string decoderInitPath = Path.Join(modelRoot, "decoder_init.onnx");
        string decoderStepPath = Path.Join(modelRoot, "decoder_step.onnx");
        string embedTokensPath = Path.Join(modelRoot, "embed_tokens.bin");
        foreach (string path in new[] { decoderInitPath, decoderStepPath, embedTokensPath })
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Qwen3-ASR ONNX package is missing a required file.", path);
            }
        }

        return new Qwen3AsrModelPaths(
            modelRoot,
            encoderModelPath,
            decoderInitPath,
            decoderStepPath,
            embedTokensPath);
    }

    internal static string FormatTooShortChunkSkipReason(int melFrames, int audioTokenCount) =>
        $"chunk too short for Qwen3-ASR encoder: {melFrames} mel frame(s) produced {audioTokenCount} audio token(s), so the chunk was skipped";

    internal static string AppendChunkSkipReasons(string? bootstrapDetail, IReadOnlyList<string> chunkSkipReasons)
    {
        string detail = string.IsNullOrWhiteSpace(bootstrapDetail)
            ? "ONNX Qwen3-ASR greedy decoding."
            : bootstrapDetail;
        if (chunkSkipReasons.Count == 0)
        {
            return detail;
        }

        return $"{detail} Skipped {chunkSkipReasons.Count} chunk(s): {string.Join("; ", chunkSkipReasons.Distinct(StringComparer.Ordinal))}";
    }

    private static StageRuntimeExecutionSummary CreateExecutionSummary(
        StageRuntimePlan plan,
        OnnxExecutionSessionFactory.Qwen3AsrSessionLease sessionLease,
        IReadOnlyList<string>? chunkSkipReasons = null) =>
        new(
            sessionLease.RequestedProvider,
            sessionLease.SelectedProvider,
            plan.ModelId,
            plan.ModelAlias,
            plan.Variant,
            AppendChunkSkipReasons(sessionLease.BootstrapDetail, chunkSkipReasons ?? []));

    private static Task<OnnxExecutionSessionFactory.Qwen3AsrSessionLease> AcquireSessionLeaseAsync(
        StageRuntimePlan plan,
        Qwen3AsrModelPaths modelPaths,
        CancellationToken cancellationToken) =>
        OnnxExecutionSessionFactory.CreatePooledQwen3AsrAsync(
            new OnnxExecutionSessionFactory.PooledSessionRequest
            {
                EngineFamily = EngineFamilyName,
                Provider = plan.ExecutionProvider!.Value,
                CancellationToken = cancellationToken,
                AdditionalTrtEncoderOptions = TrtEncoderOptions,
            },
            new OnnxExecutionSessionFactory.QwenAsrModelPaths(
                modelPaths.EncoderPath,
                modelPaths.DecoderInitPath,
                modelPaths.DecoderStepPath));

    private static StageRuntimeExecutionSummary CreatePlannedOnlySummary(StageRuntimePlan plan, string bootstrapDetail) =>
        new(
            "auto",
            plan.ExecutionProvider!.Value.ToString().ToLowerInvariant(),
            plan.ModelId,
            plan.ModelAlias,
            plan.Variant,
            bootstrapDetail);

    private sealed record Qwen3AsrModelPaths(
        string ModelRootPath,
        string EncoderPath,
        string DecoderInitPath,
        string DecoderStepPath,
        string EmbedTokensPath);

    private sealed record RegionTranscription(
        string Text,
        string? DetectedLanguage,
        IReadOnlyList<RecognizedTranscriptWord> Words,
        IReadOnlyList<string> ChunkSkipReasons);
}
