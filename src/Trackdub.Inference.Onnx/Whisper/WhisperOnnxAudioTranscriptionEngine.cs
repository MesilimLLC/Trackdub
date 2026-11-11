using System.Collections.Concurrent;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Audio;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.Runtime.Routing;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Runtime.Planning;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Trackdub.Inference.Onnx.Whisper;

public sealed class WhisperOnnxAudioTranscriptionEngine : IAudioTranscriptionEngineAdapter, IStageRuntimeExecutionReporter, IDeviceDegradationReporter
{
    private readonly IRuntimePlanner runtimePlanner;
    private readonly IRuntimePlanningPreferences? runtimePlanningPreferences;
    private readonly BenchmarkModelPathResolver modelPathResolver;
    private readonly IPipelineDeviceExclusionProvider? deviceExclusionProvider;
    private readonly SidecarCache<WhisperTokenizerDecoder> tokenizerCache;

    /// <summary>Production constructor: the tokenizer cache defaults to the process-wide instance.</summary>
    public WhisperOnnxAudioTranscriptionEngine(
        IRuntimePlanner runtimePlanner,
        BenchmarkModelPathResolver modelPathResolver,
        IRuntimePlanningPreferences? runtimePlanningPreferences = null,
        IPipelineDeviceExclusionProvider? deviceExclusionProvider = null)
        : this(runtimePlanner, modelPathResolver, runtimePlanningPreferences, deviceExclusionProvider, null)
    {
    }

    /// <summary>Test seam: allows injecting a cache so load counts can be asserted.</summary>
    internal WhisperOnnxAudioTranscriptionEngine(
        IRuntimePlanner runtimePlanner,
        BenchmarkModelPathResolver modelPathResolver,
        IRuntimePlanningPreferences? runtimePlanningPreferences,
        IPipelineDeviceExclusionProvider? deviceExclusionProvider,
        SidecarCache<WhisperTokenizerDecoder>? tokenizerCache)
    {
        this.runtimePlanner = runtimePlanner ?? throw new ArgumentNullException(nameof(runtimePlanner));
        this.modelPathResolver = modelPathResolver ?? throw new ArgumentNullException(nameof(modelPathResolver));
        this.runtimePlanningPreferences = runtimePlanningPreferences;
        this.deviceExclusionProvider = deviceExclusionProvider;
        this.tokenizerCache = tokenizerCache ?? WhisperSidecarCaches.Tokenizers;
    }

    public const string EngineFamilyName = "whisper-onnx";

    private const int DefaultMelBins = 80;
    private const int MaxDecoderContextTokens = 448;
    // The 30-second mel window always subsamples to 1500 encoder positions.
    private const int EncoderSequenceLength = 1500;

    private const double MaxChunkDurationSeconds = 28;
    private const double TranscriptionMergeGapSeconds = 1.5;
    private const double TranscriptionContextPaddingSeconds = 0.5;
    private const double MinTranscriptionRegionSeconds = 3.0;
    private const string RepetitionGuardBootstrapDetail = "Decoder repetition guard flagged output for review.";
    private readonly System.Collections.Concurrent.ConcurrentDictionary<int, WhisperFeatureExtractor> featureExtractors = new();

    public StageRuntimeExecutionSummary? LastExecutionSummary { get; private set; }

    /// <summary>True when the last transcribe/language call ran through the fixed-address bound path.</summary>
    internal static bool LastBoundTrtRtxRunUsed { get; private set; }

    /// <summary>Test hook: forces the managed run path on TRT-RTX sessions for bound-path parity checks.</summary>
    internal static bool TrtRtxBoundRunDisabled;

    public DeviceDegradationReport? LastDeviceDegradation { get; private set; }

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
        StageRuntimePlan plan = await PlanStageAsync(request, cancellationToken).ConfigureAwait(false);
        return await TranscribeAsync(request, plan, cancellationToken).ConfigureAwait(false);
    }

    private async Task<StageRuntimePlan> PlanStageAsync(
        AudioTranscriptionRequest request,
        CancellationToken cancellationToken)
    {
        InferenceRequestOptions options = request.Options ?? InferenceRequestOptions.Default;
        return await runtimePlanner.PlanAsync(
            await StageRuntimePlanningRequestFactory.ApplyPreferredModelTierAsync(new StageRuntimePlanningRequest(
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

        IAudioSamples audio = await WaveAudioReader.ReadMonoPcm16Async(request.NormalizedAudioPath, cancellationToken).ConfigureAwait(false);
        // CreateResampledStream returns the source as-is when sample rates match;
        // otherwise it wraps and takes ownership of it. Dispose only the outer reader.
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

            // Short audio with no VAD regions — treat full audio as one region
            effectiveRegions = [new SpeechRegion(0, 0.0, durationSeconds)];
        }

        DeviceFallbackSessionCreator.Result<WhisperSessionBundle> acquired = await DeviceFallbackSessionCreator
            .CreateWithDeviceFallbackAsync(
                plan,
                CreateWhisperSessionBundleAsync,
                ct => PlanStageAsync(request, ct),
                deviceExclusionProvider,
                cancellationToken: cancellationToken)
            .ConfigureAwait(false);

        LastDeviceDegradation = acquired.Degradation;
        StageRuntimePlan effectivePlan = acquired.Plan;
        using OnnxExecutionSessionFactory.WhisperSessionLease sessionLease = acquired.Lease.Lease;
        // Cached per model root: the tokenizer is a ~52k-entry JSON parse that does not depend on
        // the audio, so it should not be repeated for every transcription.
        WhisperTokenizerDecoder tokenizer = await tokenizerCache
            .GetOrAddAsync(acquired.Lease.ModelRootPath, path => WhisperTokenizerDecoder.LoadAsync(path))
            .ConfigureAwait(false);
        WhisperFeatureExtractor featureExtractor = featureExtractors.GetOrAdd(
            ReadMelBins(acquired.Lease.ModelRootPath),
            static melBins => new WhisperFeatureExtractor(melBins));

        // Merged regions give the language detector more context per probe, but the
        // transcription loop must use the original effectiveRegions so that caller-provided
        // indices (e.g. from RetranscribeSegmentsAsync) are preserved in the output.
        IReadOnlyList<SpeechRegion> languageDetectionRegions = BuildTranscriptionRegions(effectiveRegions, durationSeconds);
        IReadOnlyList<SpeechRegion> transcriptionRegions = effectiveRegions
            .Where(static r => r.EndSeconds > r.StartSeconds)
            .OrderBy(static r => r.Index)
            .ToArray();
        string? requestedSourceLanguage = NormalizeLanguageCode(request.SourceLanguage);
        string? detectedTranscriptLanguage = requestedSourceLanguage is not null &&
                                             tokenizer.TryGetLanguageTokenId(requestedSourceLanguage) is not null
            ? requestedSourceLanguage
            : DetectTranscriptLanguage(
                sessionLease,
                tokenizer,
                featureExtractor,
                targetAudio,
                languageDetectionRegions,
                cancellationToken);
        var segments = new List<RecognizedTranscriptSegment>(transcriptionRegions.Count);
        bool repetitionGuarded = false;

        foreach (SpeechRegion region in transcriptionRegions)
        {
            cancellationToken.ThrowIfCancellationRequested();

            RegionTranscription transcription = await TranscribeRegionAsync(
                sessionLease,
                tokenizer,
                featureExtractor,
                targetAudio,
                region,
                detectedTranscriptLanguage,
                cancellationToken).ConfigureAwait(false);
            repetitionGuarded |= transcription.RepetitionGuarded;

            // Keep blank regions so Segment.Index stays aligned for callers that
            // map recognition by index (retranscribe falls back to original text).
            segments.Add(new RecognizedTranscriptSegment(
                region.Index,
                region.StartSeconds,
                region.EndSeconds,
                transcription.Text ?? string.Empty,
                transcription.DetectedLanguage ?? detectedTranscriptLanguage,
                transcription.Words));
        }

        LastExecutionSummary = CreateExecutionSummary(effectivePlan, sessionLease, repetitionGuarded);
        return segments;
    }

    private async Task<WhisperSessionBundle> CreateWhisperSessionBundleAsync(
        StageRuntimePlan plan,
        CancellationToken cancellationToken)
    {
        EnsurePlanReady(plan, RuntimeStage.Asr);
        string encoderModelPath = ResolvePlannedModelPath(plan);
        string decoderModelPath = ResolveWhisperDecoderPath(encoderModelPath);
        string modelRootPath = ResolveModelRootPath(encoderModelPath);
        ExecutionProviderKind provider = plan.ExecutionProvider!.Value;
        IReadOnlyDictionary<string, string>? encoderOptions = null;
        IReadOnlyDictionary<string, string>? decoderOptions = null;
        if (provider is ExecutionProviderKind.TensorRTRtx)
        {
            IReadOnlyDictionary<string, string> encoderProfile = BuildTrtEncoderOptions(ReadMelBins(modelRootPath));
            IReadOnlyDictionary<string, string> decoderProfile = BuildTrtDecoderOptions(decoderModelPath);
            bool enableCudaGraphs = ShouldEnableTrtRtxCudaGraphs(
                provider,
                GetDecoderTrtConfiguration(decoderModelPath).InputNames);
            encoderOptions = OnnxExecutionSessionFactory.WithCudaGraphs(encoderProfile, enableCudaGraphs);
            decoderOptions = OnnxExecutionSessionFactory.WithCudaGraphs(decoderProfile, enableCudaGraphs);
        }

        OnnxExecutionSessionFactory.WhisperSessionLease lease = await OnnxExecutionSessionFactory
            .CreatePooledWhisperAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "whisper-onnx", Provider = provider, CancellationToken = cancellationToken, AdditionalTrtEncoderOptions = encoderOptions, AdditionalTrtDecoderOptions = decoderOptions }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = encoderModelPath, Decoder = decoderModelPath })
            .ConfigureAwait(false);
        return new WhisperSessionBundle(lease, encoderModelPath, decoderModelPath, modelRootPath);
    }

    private sealed record WhisperSessionBundle(
        OnnxExecutionSessionFactory.WhisperSessionLease Lease,
        string EncoderModelPath,
        string DecoderModelPath,
        string ModelRootPath);

    private static string? DetectTranscriptLanguage(
        OnnxExecutionSessionFactory.WhisperSessionLease sessionLease,
        WhisperTokenizerDecoder tokenizer,
        WhisperFeatureExtractor featureExtractor,
        IAudioSamples targetAudio,
        IReadOnlyList<SpeechRegion> regions,
        CancellationToken cancellationToken)
    {
        var detectedLanguages = new List<string>(regions.Count);
        foreach (SpeechRegion region in regions)
        {
            cancellationToken.ThrowIfCancellationRequested();

            string? regionLanguage = DetectRegionLanguage(
                sessionLease,
                tokenizer,
                featureExtractor,
                targetAudio,
                region,
                cancellationToken);
            if (!string.IsNullOrWhiteSpace(regionLanguage))
            {
                detectedLanguages.Add(regionLanguage);
            }
        }

        return ResolveDetectedLanguage(detectedLanguages);
    }

    /// <summary>
    /// Votes the region's language from the same per-chunk language-token probe that
    /// <see cref="GreedyDecodeAsync"/> runs, without the full greedy decode whose text
    /// detection would discard.
    /// </summary>
    private static string? DetectRegionLanguage(
        OnnxExecutionSessionFactory.WhisperSessionLease sessionLease,
        WhisperTokenizerDecoder tokenizer,
        WhisperFeatureExtractor featureExtractor,
        IAudioSamples targetAudio,
        SpeechRegion region,
        CancellationToken cancellationToken)
    {
        long startSample = Math.Max(0, (long)Math.Floor(region.StartSeconds * 16000d));
        long endSample = Math.Min(targetAudio.SampleFrameCount, (long)Math.Ceiling(region.EndSeconds * 16000d));
        if (endSample <= startSample || tokenizer.LanguageTokenIds.Count == 0)
        {
            return null;
        }

        float[] regionSamples = new float[checked((int)(endSample - startSample))];
        targetAudio.ReadMonoSamples(startSample, regionSamples);
        List<string> chunkLanguages = [];
        int maxChunkSamples = (int)(MaxChunkDurationSeconds * 16000d);
        using var phase = BenchmarkPhaseCapture.Start("asr-language-detection");
        using WhisperTrtRtxRun? boundRun = WhisperTrtRtxRun.TryCreate(sessionLease, tokenizer);
        LastBoundTrtRtxRunUsed = boundRun is not null;

        for (int offset = 0; offset < regionSamples.Length; offset += maxChunkSamples)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int chunkLength = Math.Min(maxChunkSamples, regionSamples.Length - offset);
            DenseTensor<float> features = featureExtractor.Extract(
                new ReadOnlySpan<float>(regionSamples, offset, chunkLength));

            string? chunkLanguage;
            if (boundRun is not null)
            {
                boundRun.RunEncoder(features, cancellationToken);
                chunkLanguage = tokenizer.TryGetLanguageCode(
                    boundRun.DetectLanguageTokenBound(cancellationToken));
            }
            else
            {
                using var encoderInputs = new InputSet(
                [
                    NamedOnnxValue.CreateFromTensor("input_features", features)
                ]);
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults =
                    sessionLease.EncoderSession.RunWithRetry(encoderInputs.Values, cancellationToken: cancellationToken);
                Tensor<float> hiddenStates = encoderResults.Single(static result => result.Name == "last_hidden_state").AsTensor<float>();
                chunkLanguage = tokenizer.TryGetLanguageCode(
                    DetectLanguageToken(sessionLease.DecoderSession, tokenizer, hiddenStates, cancellationToken));
            }

            if (!string.IsNullOrWhiteSpace(chunkLanguage))
            {
                chunkLanguages.Add(chunkLanguage);
            }
        }

        return ResolveDetectedLanguage(chunkLanguages);
    }

    private static string? NormalizeLanguageCode(string? languageCode)
    {
        if (string.IsNullOrWhiteSpace(languageCode))
        {
            return null;
        }

        string normalized = languageCode.Trim().ToLowerInvariant().Replace('_', '-');
        int separatorIndex = normalized.IndexOf('-', StringComparison.Ordinal);
        if (separatorIndex > 0)
        {
            normalized = normalized[..separatorIndex];
        }

        return normalized.Length is >= 2 and <= 8 &&
               normalized.All(static character => character is >= 'a' and <= 'z')
            ? normalized
            : null;
    }

    internal static IReadOnlyList<SpeechRegion> BuildTranscriptionRegionsForTesting(
        IReadOnlyList<SpeechRegion> regions,
        double durationSeconds) =>
        BuildTranscriptionRegions(regions, durationSeconds);

    internal static IReadOnlyList<SpeechRegion> BuildTranscriptionRegions(
        IReadOnlyList<SpeechRegion> regions,
        double durationSeconds)
    {
        if (regions.Count == 0 || durationSeconds <= 0d)
        {
            return [];
        }

        SpeechRegion[] orderedRegions = regions
            .Where(static region => region.EndSeconds > region.StartSeconds)
            .OrderBy(static region => region.StartSeconds)
            .ToArray();
        if (orderedRegions.Length == 0)
        {
            return [];
        }

        var mergedRegions = new List<(double Start, double End)>();
        double currentStart = orderedRegions[0].StartSeconds;
        double currentEnd = orderedRegions[0].EndSeconds;

        foreach (SpeechRegion region in orderedRegions.Skip(1))
        {
            double gapSeconds = region.StartSeconds - currentEnd;
            double mergedDuration = region.EndSeconds - currentStart;
            if (gapSeconds <= TranscriptionMergeGapSeconds &&
                mergedDuration <= MaxChunkDurationSeconds)
            {
                currentEnd = Math.Max(currentEnd, region.EndSeconds);
                continue;
            }

            mergedRegions.Add(ExpandRegion(currentStart, currentEnd, durationSeconds));
            currentStart = region.StartSeconds;
            currentEnd = region.EndSeconds;
        }

        mergedRegions.Add(ExpandRegion(currentStart, currentEnd, durationSeconds));

        return NormalizeExpandedRegions(mergedRegions)
            .Select((region, index) => new SpeechRegion(index, region.Start, region.End))
            .ToArray();
    }

    private static IReadOnlyList<(double Start, double End)> NormalizeExpandedRegions(
        IReadOnlyList<(double Start, double End)> regions)
    {
        var normalized = new List<(double Start, double End)>(regions.Count);
        foreach ((double start, double end) in regions.OrderBy(static region => region.Start))
        {
            if (normalized.Count == 0)
            {
                normalized.Add((start, end));
                continue;
            }

            (double Start, double End) previous = normalized[^1];
            if (start <= previous.End && end - previous.Start <= MaxChunkDurationSeconds)
            {
                normalized[^1] = (previous.Start, Math.Max(previous.End, end));
                continue;
            }

            double nonOverlappingStart = Math.Max(start, previous.End);
            if (end > nonOverlappingStart)
            {
                normalized.Add((nonOverlappingStart, end));
            }
        }

        return normalized;
    }

    private static (double Start, double End) ExpandRegion(
        double startSeconds,
        double endSeconds,
        double durationSeconds)
    {
        double start = Math.Max(0d, startSeconds - TranscriptionContextPaddingSeconds);
        double end = Math.Min(durationSeconds, endSeconds + TranscriptionContextPaddingSeconds);
        double expandedDuration = end - start;
        if (expandedDuration >= MinTranscriptionRegionSeconds || durationSeconds <= expandedDuration)
        {
            return (start, end);
        }

        double extraSeconds = MinTranscriptionRegionSeconds - expandedDuration;
        double leftExtra = Math.Min(start, extraSeconds / 2d);
        start -= leftExtra;
        extraSeconds -= leftExtra;

        double rightExtra = Math.Min(durationSeconds - end, extraSeconds);
        end += rightExtra;
        extraSeconds -= rightExtra;

        start = Math.Max(0d, start - extraSeconds);
        return (start, end);
    }
    private async Task<RegionTranscription> TranscribeRegionAsync(
        OnnxExecutionSessionFactory.WhisperSessionLease sessionLease,
        WhisperTokenizerDecoder tokenizer,
        WhisperFeatureExtractor featureExtractor,
        IAudioSamples targetAudio,
        SpeechRegion region,
        string? forcedLanguage,
        CancellationToken cancellationToken)
    {
        long startSample = Math.Max(0, (long)Math.Floor(region.StartSeconds * 16000d));
        long endSample = Math.Min(targetAudio.SampleFrameCount, (long)Math.Ceiling(region.EndSeconds * 16000d));
        if (endSample <= startSample)
        {
            return new RegionTranscription(string.Empty, DetectedLanguage: null, Words: []);
        }

        float[] regionSamples = new float[checked((int)(endSample - startSample))];
        targetAudio.ReadMonoSamples(startSample, regionSamples);
        List<string> chunkTexts = [];
        List<RecognizedTranscriptWord> chunkWords = [];
        List<string> detectedLanguages = [];
        bool repetitionGuarded = false;
        int maxChunkSamples = (int)(MaxChunkDurationSeconds * 16000d);
        using WhisperTrtRtxRun? boundRun = WhisperTrtRtxRun.TryCreate(sessionLease, tokenizer);
        LastBoundTrtRtxRunUsed = boundRun is not null;

        for (int offset = 0; offset < regionSamples.Length; offset += maxChunkSamples)
        {
            int chunkLength = Math.Min(maxChunkSamples, regionSamples.Length - offset);
            double chunkDurationSeconds = chunkLength / 16000d;
            DenseTensor<float> features;
            {
                ReadOnlySpan<float> chunkSpan = new ReadOnlySpan<float>(regionSamples, offset, chunkLength);
                features = featureExtractor.Extract(chunkSpan);
            }

            var context = new WhisperChunkDecodeContext(
                tokenizer,
                features,
                boundRun,
                forcedLanguage,
                chunkDurationSeconds,
                cancellationToken);
            WhisperDecodeResult decodeResult = await DecodeChunkAsync(sessionLease, context).ConfigureAwait(false);

            repetitionGuarded |= decodeResult.RepetitionGuarded;
            string decodedText = tokenizer.DecodeText(decodeResult.OutputTokens);
            if (WhisperGenAiAudioTranscriptionEngine.IsDegenerateTranscriptText(decodedText))
            {
                // Punctuation-only hallucination (". . .") — keep the region blank so
                // index alignment holds, but do not surface garbage as transcript text.
            }
            else if (!string.IsNullOrWhiteSpace(decodedText))
            {
                chunkTexts.Add(decodedText);
                double chunkStartSeconds = region.StartSeconds + (offset / 16000d);
                double chunkEndSeconds = chunkStartSeconds + chunkDurationSeconds;
                chunkWords.AddRange(BuildRecognizedWords(
                    decodedText,
                    chunkStartSeconds,
                    chunkEndSeconds,
                    decodeResult.RepetitionGuarded ? [0d] : decodeResult.TokenConfidences));
            }

            if (!string.IsNullOrWhiteSpace(decodeResult.DetectedLanguage))
            {
                detectedLanguages.Add(decodeResult.DetectedLanguage);
            }
        }

        return CreateRegionTranscription(chunkTexts, chunkWords, detectedLanguages, repetitionGuarded);
    }

    private static Task<WhisperDecodeResult> GreedyDecodeAsync(
        InferenceSession decoderSession,
        WhisperTrtRtxRun? boundRun,
        WhisperTokenizerDecoder tokenizer,
        Tensor<float>? encoderHiddenStates,
        string? forcedLanguage,
        double chunkDurationSeconds,
        CancellationToken cancellationToken)
    {
        int? detectedLanguageToken = tokenizer.TryGetLanguageTokenId(forcedLanguage) ??
                                     (boundRun is not null
                                         ? boundRun.DetectLanguageTokenBound(cancellationToken)
                                         : DetectLanguageToken(decoderSession, tokenizer, encoderHiddenStates!, cancellationToken));
        IReadOnlyList<int> promptTokens = tokenizer.BuildTranscriptionPrompt(detectedLanguageToken);
        string? detectedLanguage = tokenizer.TryGetLanguageCode(detectedLanguageToken);
        var generated = promptTokens.Select(static token => (long)token).ToList();
        var outputTokens = new List<int>();
        var tokenConfidences = new List<double?>();
        int maxOutputTokens = Math.Min(
            CalculateMaxDecodeOutputTokens(chunkDurationSeconds),
            Math.Max(0, MaxDecoderContextTokens - generated.Count));
        bool repetitionGuarded = false;

        for (int step = 0; step < maxOutputTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int nextToken;
            double? confidence;
            if (boundRun is not null)
            {
                (nextToken, confidence) = boundRun.DecodeStep(generated, cancellationToken);
            }
            else
            {
                using var decoderInputs = CreateDecoderInputs(generated, encoderHiddenStates!);
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults = decoderSession.RunWithRetry(decoderInputs.Values, cancellationToken: cancellationToken);
                Tensor<float> logits = decoderResults.Single(static result => result.Name == "logits").AsTensor<float>();

                int sequenceLength = logits.Dimensions[1];
                int vocabularySize = logits.Dimensions[2];
                (nextToken, confidence) = SelectNextToken(logits, sequenceLength - 1, vocabularySize, tokenizer);
            }
            if (nextToken == tokenizer.EndOfTranscriptToken || nextToken < 0)
            {
                break;
            }

            generated.Add(nextToken);
            outputTokens.Add(nextToken);
            tokenConfidences.Add(confidence);
            if (TryTrimRepeatedTail(outputTokens, tokenConfidences))
            {
                repetitionGuarded = true;
                break;
            }
        }

        return Task.FromResult(new WhisperDecodeResult(
            outputTokens,
            detectedLanguage,
            tokenConfidences,
            repetitionGuarded));
    }

    private static async Task<WhisperDecodeResult> DecodeChunkAsync(
        OnnxExecutionSessionFactory.WhisperSessionLease sessionLease,
        WhisperChunkDecodeContext context)
    {
        if (context.BoundRun is not null)
        {
            context.BoundRun.RunEncoder(context.Features, context.CancellationToken);
            return await GreedyDecodeAsync(
                sessionLease.DecoderSession,
                context.BoundRun,
                context.Tokenizer,
                encoderHiddenStates: null,
                context.ForcedLanguage,
                context.ChunkDurationSeconds,
                context.CancellationToken).ConfigureAwait(false);
        }

        using var encoderInputs = new InputSet(
        [
            NamedOnnxValue.CreateFromTensor("input_features", context.Features)
        ]);
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults =
            sessionLease.EncoderSession.RunWithRetry(encoderInputs.Values, cancellationToken: context.CancellationToken);
        return await WhisperEncoderOutputLifetime.ConsumeAsync(
            encoderResults,
            hiddenStates => GreedyDecodeAsync(
                sessionLease.DecoderSession,
                context.BoundRun,
                context.Tokenizer,
                hiddenStates,
                context.ForcedLanguage,
                context.ChunkDurationSeconds,
                context.CancellationToken)).ConfigureAwait(false);
    }

    private static RegionTranscription CreateRegionTranscription(
        List<string> chunkTexts,
        List<RecognizedTranscriptWord> chunkWords,
        List<string> detectedLanguages,
        bool repetitionGuarded)
    {
        IReadOnlyList<RecognizedTranscriptWord> regionWords = repetitionGuarded
            ? ReindexWords(chunkWords
                .Select(static word => new RecognizedTranscriptWord(
                    word.WordIndex,
                    word.StartSeconds,
                    word.EndSeconds,
                    word.Text,
                    0d))
                .ToArray())
            : ReindexWords(chunkWords);

        return new RegionTranscription(
            string.Join(" ", chunkTexts).Trim(),
            ResolveDetectedLanguage(detectedLanguages),
            regionWords,
            repetitionGuarded);
    }

    private sealed record WhisperChunkDecodeContext(
        WhisperTokenizerDecoder Tokenizer,
        DenseTensor<float> Features,
        WhisperTrtRtxRun? BoundRun,
        string? ForcedLanguage,
        double ChunkDurationSeconds,
        CancellationToken CancellationToken);

    internal static int CalculateMaxDecodeOutputTokensForTesting(double chunkDurationSeconds) =>
        CalculateMaxDecodeOutputTokens(chunkDurationSeconds);

    internal static (IReadOnlyList<int> Tokens, bool RepetitionGuarded) ApplyRepetitionGuardForTesting(
        IReadOnlyList<int> outputTokens)
    {
        var guardedTokens = outputTokens.ToList();
        List<double?> tokenConfidences = Enumerable.Repeat<double?>(1d, guardedTokens.Count).ToList();
        bool repetitionGuarded = TryTrimRepeatedTail(guardedTokens, tokenConfidences);
        return (guardedTokens, repetitionGuarded);
    }

    private static int CalculateMaxDecodeOutputTokens(double chunkDurationSeconds)
    {
        double normalizedDurationSeconds = double.IsFinite(chunkDurationSeconds) && chunkDurationSeconds > 0d
            ? chunkDurationSeconds
            : 0d;
        return Math.Min(448, Math.Max(48, (int)Math.Ceiling(normalizedDurationSeconds * 12d) + 24));
    }

    // Called after each generated token to stop active runaway loops. This is
    // intentionally not a full transcript cleanup pass for earlier repeats.
    private static bool TryTrimRepeatedTail(List<int> outputTokens, List<double?> tokenConfidences)
    {
        for (int ngramLength = 3; ngramLength <= 32; ngramLength++)
        {
            int repeatLimit = ngramLength <= 3 ? 4 : ngramLength <= 8 ? 3 : 2;
            int repeatedTokenCount = ngramLength * repeatLimit;
            if (outputTokens.Count < repeatedTokenCount)
            {
                continue;
            }

            int repeatedStart = outputTokens.Count - repeatedTokenCount;
            bool repeated = true;
            for (int repeatIndex = 1; repeatIndex < repeatLimit && repeated; repeatIndex++)
            {
                int candidateStart = repeatedStart + (ngramLength * repeatIndex);
                for (int offset = 0; offset < ngramLength; offset++)
                {
                    if (outputTokens[repeatedStart + offset] != outputTokens[candidateStart + offset])
                    {
                        repeated = false;
                        break;
                    }
                }
            }

            if (!repeated)
            {
                continue;
            }

            int removeCount = ngramLength * (repeatLimit - 1);
            int removeStart = outputTokens.Count - removeCount;
            outputTokens.RemoveRange(removeStart, removeCount);
            tokenConfidences.RemoveRange(removeStart, removeCount);
            return true;
        }

        return false;
    }

    private static string? ResolveDetectedLanguage(IReadOnlyList<string> detectedLanguages) =>
        detectedLanguages
            .Where(static language => !string.IsNullOrWhiteSpace(language))
            .Select(static language => language.Trim().ToLowerInvariant())
            .GroupBy(static language => language, StringComparer.Ordinal)
            .OrderByDescending(static group => group.Count())
            .ThenBy(static group => group.Key, StringComparer.Ordinal)
            .Select(static group => group.Key)
            .FirstOrDefault();

    private static int? DetectLanguageToken(
        InferenceSession decoderSession,
        WhisperTokenizerDecoder tokenizer,
        Tensor<float> encoderHiddenStates,
        CancellationToken cancellationToken)
    {
        if (tokenizer.LanguageTokenIds.Count == 0)
        {
            return null;
        }

        using var decoderInputs = CreateDecoderInputs([tokenizer.DecoderStartToken], encoderHiddenStates);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults =
            decoderSession.RunWithRetry(decoderInputs.Values, cancellationToken: cancellationToken);
        Tensor<float> logits = decoderResults.Single(static result => result.Name == "logits").AsTensor<float>();

        int sequenceLength = logits.Dimensions[1];
        int bestToken = SelectBestToken(logits, sequenceLength - 1, tokenizer.LanguageTokenIds);
        return bestToken >= 0 ? bestToken : null;
    }

    private static (int TokenId, double? Confidence) SelectNextToken(
        Tensor<float> logits,
        int timeIndex,
        int vocabularySize,
        WhisperTokenizerDecoder tokenizer)
    {
        // Row-span read (Seq2SeqGreedyDecoder.ArgMaxAtPosition precedent) instead of the
        // slow 3-D tensor indexer in the per-token hot loops below.
        if (logits is DenseTensor<float> dense)
        {
            ReadOnlySpan<float> row = dense.Buffer.Span.Slice(timeIndex * vocabularySize, vocabularySize);
            return SelectNextToken(row, tokenizer);
        }

        float[] rowCopy = new float[vocabularySize];
        for (int tokenIndex = 0; tokenIndex < vocabularySize; tokenIndex++)
        {
            rowCopy[tokenIndex] = logits[0, timeIndex, tokenIndex];
        }

        return SelectNextToken(rowCopy.AsSpan(), tokenizer);
    }

    private static (int TokenId, double? Confidence) SelectNextToken(
        ReadOnlySpan<float> row,
        WhisperTokenizerDecoder tokenizer)
    {
        int vocabularySize = row.Length;
        int bestToken = -1;
        float bestValue = float.NegativeInfinity;
        for (int tokenIndex = 0; tokenIndex < vocabularySize; tokenIndex++)
        {
            if (tokenIndex >= tokenizer.TimestampBeginToken && tokenIndex != tokenizer.EndOfTranscriptToken)
            {
                continue;
            }

            if (tokenizer.SuppressedTokens.Contains(tokenIndex))
            {
                continue;
            }

            float value = row[tokenIndex];
            if (value > bestValue)
            {
                bestValue = value;
                bestToken = tokenIndex;
            }
        }

        if (bestToken < 0)
        {
            return (bestToken, Confidence: null);
        }

        double denominator = 0d;
        double selectedNumerator = 0d;
        for (int tokenIndex = 0; tokenIndex < vocabularySize; tokenIndex++)
        {
            if (tokenIndex >= tokenizer.TimestampBeginToken && tokenIndex != tokenizer.EndOfTranscriptToken)
            {
                continue;
            }

            if (tokenizer.SuppressedTokens.Contains(tokenIndex))
            {
                continue;
            }

            double numerator = Math.Exp(row[tokenIndex] - bestValue);
            denominator += numerator;
            if (tokenIndex == bestToken)
            {
                selectedNumerator = numerator;
            }
        }

        return (bestToken, denominator > 0d ? selectedNumerator / denominator : null);
    }

    private static int SelectBestToken(
        Tensor<float> logits,
        int timeIndex,
        IEnumerable<int> candidateTokenIds)
    {
        int bestToken = -1;
        float bestValue = float.NegativeInfinity;
        foreach (int tokenId in candidateTokenIds)
        {
            float value = logits[0, timeIndex, tokenId];
            if (value > bestValue)
            {
                bestValue = value;
                bestToken = tokenId;
            }
        }

        return bestToken;
    }

    private static InputSet CreateDecoderInputs(IReadOnlyList<long> generatedTokens, Tensor<float> encoderHiddenStates)
    {
        long[] tokenArray = generatedTokens.ToArray();
        return new InputSet(
        [
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(tokenArray, [1, tokenArray.Length])),
            NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates)
        ]);
    }

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

    private string ResolvePlannedModelPath(StageRuntimePlan plan)
    {
        return PlannedRuntimeModelResolver.ResolveModelPath(plan, modelPathResolver);
    }

    private static string ResolveWhisperDecoderPath(string encoderModelPath)
    {
        string fileName = Path.GetFileName(encoderModelPath);
        string decoderFileName = fileName.Replace("encoder_model", "decoder_model", StringComparison.OrdinalIgnoreCase);
        string decoderModelPath = Path.Join(Path.GetDirectoryName(encoderModelPath)!, decoderFileName);
        if (File.Exists(decoderModelPath))
        {
            return Path.GetFullPath(decoderModelPath);
        }

        decoderModelPath = Path.Join(Path.GetDirectoryName(encoderModelPath)!, "decoder_model.onnx");
        if (File.Exists(decoderModelPath))
        {
            return Path.GetFullPath(decoderModelPath);
        }

        throw new FileNotFoundException("Whisper decoder model was not found next to the encoder model.", decoderModelPath);
    }

    // Shared with OnnxExecutionProviderSmokeTester so smoke and stage runs hit the same pool key.
    internal static Dictionary<string, string> BuildTrtEncoderOptions(int melBins) => new()
    {
        ["trt_profile_min_shapes"] = $"input_features:1x{melBins}x1",
        ["trt_profile_max_shapes"] = $"input_features:1x{melBins}x3000",
        ["trt_profile_opt_shapes"] = $"input_features:1x{melBins}x3000"
    };

    // Decoder profile for the no-past export: input_ids is always fed padded to the full
    // 448-token decoder context (real prefix left-aligned; trailing positions are masked out by
    // causal self-attention and only their logits are ignored), and encoder_hidden_states rides
    // the encoder's 1500-frame output. Hidden width is read from the model's declared dims, so a
    // single builder covers tiny/base/small/medium/large.
    private static readonly ConcurrentDictionary<string, DecoderTrtConfiguration> DecoderOptionsCache =
        new(StringComparer.OrdinalIgnoreCase);

    internal static Dictionary<string, string> BuildTrtDecoderOptions(string decoderModelPath) =>
        new(GetDecoderTrtConfiguration(decoderModelPath).Options, StringComparer.Ordinal);

    internal static bool ShouldEnableTrtRtxCudaGraphs(
        ExecutionProviderKind provider,
        IEnumerable<string> decoderInputNames) =>
        provider is ExecutionProviderKind.TensorRTRtx
        && !TrtRtxBoundRunDisabled
        && HasFixedAddressDecoderInputs(decoderInputNames);

    internal static bool HasFixedAddressDecoderInputs(IEnumerable<string> decoderInputNames)
    {
        ArgumentNullException.ThrowIfNull(decoderInputNames);
        HashSet<string> inputNames = decoderInputNames.ToHashSet(StringComparer.Ordinal);
        return inputNames.Count == 2
            && inputNames.Contains("input_ids")
            && inputNames.Contains("encoder_hidden_states");
    }

    private static DecoderTrtConfiguration GetDecoderTrtConfiguration(string decoderModelPath) =>
        DecoderOptionsCache.GetOrAdd(decoderModelPath, static path =>
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(path);
            // Metadata only: a CPU session never touches the TRT plugin.
            using InferenceSession session = new(path);
            HashSet<string> inputNames = session.InputMetadata.Keys.ToHashSet(StringComparer.Ordinal);
            // Extra decoder inputs rule out fixed-address binding, but do not invalidate
            // the shape profiles for the two inputs this engine knows how to size.
            if (!inputNames.Contains("input_ids") || !inputNames.Contains("encoder_hidden_states"))
            {
                return new DecoderTrtConfiguration(new Dictionary<string, string>(), inputNames);
            }

            int hiddenDim = session.InputMetadata.TryGetValue("encoder_hidden_states", out NodeMetadata? hidden)
                && hidden.Dimensions.Length == 3
                && hidden.Dimensions[2] > 0
                    ? hidden.Dimensions[2]
                    : throw new InvalidOperationException(
                        "Whisper decoder export does not declare a fixed encoder_hidden_states hidden dim.");
            // min must cover the short managed probes (e.g. the smoke tester's [1,1] input_ids);
            // opt/max pin the padded context width the bound decode path always feeds.
            string minShapes = $"input_ids:1x1,encoder_hidden_states:1x1x{hiddenDim}";
            string fullShapes =
                $"input_ids:1x{MaxDecoderContextTokens},encoder_hidden_states:1x{EncoderSequenceLength}x{hiddenDim}";
            var options = new Dictionary<string, string>
            {
                ["trt_profile_min_shapes"] = minShapes,
                ["trt_profile_max_shapes"] = fullShapes,
                ["trt_profile_opt_shapes"] = fullShapes,
            };
            return new DecoderTrtConfiguration(options, inputNames);
        });

    private sealed record DecoderTrtConfiguration(
        Dictionary<string, string> Options,
        HashSet<string> InputNames);

    // large-v3 uses 128 mel bins; earlier sizes use 80.
    /// <summary>
    /// Mel-bin count declared by <c>preprocessor_config.json</c>, cached per model root: the value
    /// is fixed for a model and was previously re-parsed from disk on every transcription and on
    /// every TensorRT profile build.
    /// </summary>
    internal static int ReadMelBins(string modelRootPath) =>
        WhisperSidecarCaches.MelBins.GetOrAdd(modelRootPath, () => ParseMelBins(modelRootPath));

    private static int ParseMelBins(string modelRootPath)
    {
        string configPath = Path.Join(modelRootPath, "preprocessor_config.json");
        if (!File.Exists(configPath))
        {
            return DefaultMelBins;
        }

        using System.Text.Json.JsonDocument config = System.Text.Json.JsonDocument.Parse(File.ReadAllText(configPath));
        return config.RootElement.TryGetProperty("feature_size", out System.Text.Json.JsonElement featureSize)
            && featureSize.TryGetInt32(out int melBins) && melBins > 0
                ? melBins
                : DefaultMelBins;
    }

    private static string ResolveModelRootPath(string encoderModelPath)
    {
        string? onnxDirectory = Path.GetDirectoryName(encoderModelPath);
        string? modelRoot = Path.GetDirectoryName(onnxDirectory);
        return modelRoot ?? throw new InvalidOperationException("Whisper model root path could not be resolved.");
    }

    private static StageRuntimeExecutionSummary CreateExecutionSummary(
        StageRuntimePlan plan,
        OnnxExecutionSessionFactory.WhisperSessionLease sessionLease,
        bool repetitionGuarded)
    {
        string bootstrapDetail = sessionLease.BootstrapDetail ?? string.Empty;
        if (repetitionGuarded)
        {
            bootstrapDetail = string.IsNullOrWhiteSpace(bootstrapDetail)
                ? RepetitionGuardBootstrapDetail
                : $"{bootstrapDetail} {RepetitionGuardBootstrapDetail}";
        }

        return new StageRuntimeExecutionSummary(
            sessionLease.RequestedProvider,
            sessionLease.SelectedProvider,
            plan.ModelId,
            plan.ModelAlias,
            plan.Variant,
            bootstrapDetail);
    }

    private static StageRuntimeExecutionSummary CreatePlannedOnlySummary(StageRuntimePlan plan, string bootstrapDetail) =>
        new(
            "auto",
            plan.ExecutionProvider!.Value.ToString().ToLowerInvariant(),
            plan.ModelId,
            plan.ModelAlias,
            plan.Variant,
            bootstrapDetail);

    internal static IReadOnlyList<RecognizedTranscriptWord> BuildRecognizedWordsForTesting(
        string text,
        double startSeconds,
        double endSeconds,
        IReadOnlyList<double?> tokenConfidences) =>
        BuildRecognizedWords(text, startSeconds, endSeconds, tokenConfidences);

    internal static IReadOnlyList<RecognizedTranscriptWord> BuildRecognizedWords(
        string text,
        double startSeconds,
        double endSeconds,
        IReadOnlyList<double?> tokenConfidences)
    {
        string[] words = text
            .Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (words.Length == 0 || endSeconds <= startSeconds)
        {
            return [];
        }

        double[] confidences = tokenConfidences
            .OfType<double>()
            .ToArray();
        double? averageConfidence = confidences.Length == 0
            ? null
            : confidences.Average();

        double step = (endSeconds - startSeconds) / words.Length;
        return words
            .Select((word, index) => new RecognizedTranscriptWord(
                index,
                startSeconds + (step * index),
                index == words.Length - 1 ? endSeconds : startSeconds + (step * (index + 1)),
                word,
                averageConfidence))
            .ToArray();
    }

    internal static IReadOnlyList<RecognizedTranscriptWord> ReindexWords(
        IReadOnlyList<RecognizedTranscriptWord> words) =>
        words
            .OrderBy(static word => word.StartSeconds)
            .ThenBy(static word => word.WordIndex)
            .Select((word, index) => new RecognizedTranscriptWord(
                index,
                word.StartSeconds,
                word.EndSeconds,
                word.Text,
                word.Confidence))
            .ToArray();

    private sealed record RegionTranscription(
        string Text,
        string? DetectedLanguage,
        IReadOnlyList<RecognizedTranscriptWord> Words,
        bool RepetitionGuarded = false);

    private sealed record WhisperDecodeResult(
        IReadOnlyList<int> OutputTokens,
        string? DetectedLanguage,
        IReadOnlyList<double?> TokenConfidences,
        bool RepetitionGuarded);

    /// <summary>
    /// Persistent fixed-shape bindings for the whisper encoder + decoder under TRT-RTX, satisfying
    /// the CUDA-graph replay precondition: every input OrtValue lives in a pinned host buffer for
    /// the whole region decode, and input_ids is always fed at the padded 448-token context width
    /// with the real prefix left-aligned. Causal self-attention keeps trailing pad positions from
    /// influencing real positions (only their logits are ignored), so a single graph replays every
    /// step instead of recapturing per sequence length.
    /// </summary>
    private sealed class WhisperTrtRtxRun : IDisposable
    {
        private readonly InferenceSession encoderSession;
        private readonly InferenceSession decoderSession;
        private readonly WhisperTokenizerDecoder tokenizer;
        private readonly long padToken;

        private DeviceBoundTensor<float>? inputFeatures;
        private OrtIoBinding? encoderBinding;
        private RunOptions? encoderRunOptions;
        private string[]? encoderOutputNames;
        private DeviceBoundTensor<long>? inputIds;
        private DeviceBoundTensor<float>? encoderHidden;
        private OrtIoBinding? decoderBinding;
        private RunOptions? decoderRunOptions;
        private string[]? decoderOutputNames;

        private WhisperTrtRtxRun(
            InferenceSession encoderSession,
            InferenceSession decoderSession,
            WhisperTokenizerDecoder tokenizer)
        {
            this.encoderSession = encoderSession;
            this.decoderSession = decoderSession;
            this.tokenizer = tokenizer;
            padToken = tokenizer.EndOfTranscriptToken;
        }

        internal static WhisperTrtRtxRun? TryCreate(
            OnnxExecutionSessionFactory.WhisperSessionLease lease,
            WhisperTokenizerDecoder tokenizer)
        {
            if (TrtRtxBoundRunDisabled)
            {
                return null;
            }

            if (!string.Equals(
                    lease.SelectedProvider,
                    ProviderNames.FormatProviderLabel(ExecutionProviderKind.TensorRTRtx),
                    StringComparison.Ordinal))
            {
                return null;
            }

            // Only the no-past export pair qualifies: input_ids + encoder_hidden_states must be the
            // complete decoder input set. With-past exports grow a KV cache each step and cannot
            // satisfy the fixed-shape precondition.
            IReadOnlyDictionary<string, NodeMetadata> decoderInputs = lease.DecoderSession.InputMetadata;
            if (!lease.EncoderSession.InputMetadata.ContainsKey("input_features")
                || !HasFixedAddressDecoderInputs(decoderInputs.Keys))
            {
                return null;
            }

            return new WhisperTrtRtxRun(lease.EncoderSession, lease.DecoderSession, tokenizer);
        }

        /// <summary>Runs the encoder and stages its hidden state into the decoder's bound input.</summary>
        internal void RunEncoder(DenseTensor<float> features, CancellationToken cancellationToken)
        {
            EnsureEncoderBound(features);
            CopyTensorValues(features, inputFeatures!.Backing);
            CopyInputToDevice(inputFeatures!);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results =
                encoderSession.RunWithBindingAndNamesRetry(
                    encoderRunOptions!,
                    encoderBinding!,
                    encoderOutputNames!,
                    cancellationToken: cancellationToken,
                    provider: ExecutionProviderKind.TensorRTRtx);
            Tensor<float> hidden = results
                .Single(static result => string.Equals(result.Name, "last_hidden_state", StringComparison.Ordinal))
                .AsTensor<float>();
            EnsureDecoderBound(hidden);
            CopyTensorValues(hidden, encoderHidden!.Backing);
        }

        /// <summary>Language-token probe matching <see cref="DetectLanguageToken"/> on the bound path.</summary>
        internal int? DetectLanguageTokenBound(CancellationToken cancellationToken)
        {
            if (tokenizer.LanguageTokenIds.Count == 0)
            {
                return null;
            }

            int bestToken = RunDecoderStep(
                [tokenizer.DecoderStartToken],
                logits => SelectBestToken(logits, 0, tokenizer.LanguageTokenIds),
                cancellationToken);
            return bestToken >= 0 ? bestToken : null;
        }

        /// <summary>
        /// One greedy step on the bound decoder: the real prefix is written left-aligned into the
        /// padded input_ids buffer, the graph replays at the fixed context width, and the next
        /// token is read from the logits row at the end of the real prefix.
        /// </summary>
        internal (int Token, double? Confidence) DecodeStep(
            IReadOnlyList<long> generated,
            CancellationToken cancellationToken) =>
            RunDecoderStep(
                generated,
                logits => SelectNextToken(logits, generated.Count - 1, logits.Dimensions[2], tokenizer),
                cancellationToken);

        private TResult RunDecoderStep<TResult>(
            IReadOnlyList<long> tokens,
            Func<Tensor<float>, TResult> select,
            CancellationToken cancellationToken)
        {
            if (decoderBinding is null || inputIds is null || decoderOutputNames is null)
            {
                throw new InvalidOperationException("Whisper decoder bindings must be staged by RunEncoder first.");
            }

            for (int index = 0; index < tokens.Count; index++)
            {
                inputIds.Backing[index] = tokens[index];
            }
            if (tokens.Count < MaxDecoderContextTokens)
            {
                Array.Fill(inputIds.Backing, padToken, tokens.Count, MaxDecoderContextTokens - tokens.Count);
            }

            OrtEnv.Instance().CopyTensors(
                [inputIds.Host, encoderHidden!.Host],
                [inputIds.Device, encoderHidden.Device],
                default);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results =
                decoderSession.RunWithBindingAndNamesRetry(
                    decoderRunOptions!,
                    decoderBinding,
                    decoderOutputNames,
                    cancellationToken: cancellationToken,
                    provider: ExecutionProviderKind.TensorRTRtx);
            Tensor<float> logits = results
                .Single(static result => string.Equals(result.Name, "logits", StringComparison.Ordinal))
                .AsTensor<float>();
            return select(logits);
        }

        private void EnsureEncoderBound(DenseTensor<float> features)
        {
            if (encoderBinding is not null)
            {
                return;
            }

            long[] shape = Array.ConvertAll(features.Dimensions.ToArray(), static dim => (long)dim);
            inputFeatures = new DeviceBoundTensor<float>(RequireDeviceAllocator(encoderSession), shape);
            encoderBinding = encoderSession.CreateIoBinding();
            encoderRunOptions = new RunOptions();
            encoderBinding.BindInput("input_features", inputFeatures.Device);
            encoderOutputNames = encoderSession.OutputMetadata.Keys.ToArray();
            foreach (string name in encoderOutputNames)
            {
                encoderBinding.BindOutputToDevice(name, OrtMemoryInfo.DefaultInstance);
            }
        }

        private void EnsureDecoderBound(Tensor<float> hiddenStates)
        {
            if (decoderBinding is not null)
            {
                return;
            }

            long[] hiddenShape = Array.ConvertAll(hiddenStates.Dimensions.ToArray(), static dim => (long)dim);
            OrtAllocator allocator = RequireDeviceAllocator(decoderSession);
            encoderHidden = new DeviceBoundTensor<float>(allocator, hiddenShape);
            inputIds = new DeviceBoundTensor<long>(allocator, [1, MaxDecoderContextTokens]);
            decoderBinding = decoderSession.CreateIoBinding();
            decoderRunOptions = new RunOptions();
            decoderBinding.BindInput("input_ids", inputIds.Device);
            decoderBinding.BindInput("encoder_hidden_states", encoderHidden.Device);
            decoderOutputNames = decoderSession.OutputMetadata.Keys.ToArray();
            foreach (string name in decoderOutputNames)
            {
                decoderBinding.BindOutputToDevice(name, OrtMemoryInfo.DefaultInstance);
            }
        }

        private static OrtAllocator RequireDeviceAllocator(InferenceSession session)
        {
            OrtMemoryInfo? deviceMemInfo = session.GetMemoryInfosForInputs()
                .FirstOrDefault(mi => !mi.Equals(OrtMemoryInfo.DefaultInstance));
            if (deviceMemInfo is null)
            {
                throw new NotSupportedException(
                    "Session reports host-resident inputs only; the persistent bound path requires a device-resident provider.");
            }

            OrtAllocator? allocator = OrtEnv.Instance().GetSharedAllocator(deviceMemInfo);
            if (allocator is null || allocator.IsInvalid)
            {
                throw new NotSupportedException(
                    $"No shared allocator registered for '{deviceMemInfo.Name}'.");
            }

            return allocator;
        }

        private static void CopyInputToDevice<T>(DeviceBoundTensor<T> tensor)
            where T : unmanaged =>
            OrtEnv.Instance().CopyTensors([tensor.Host], [tensor.Device], default);

        private static void CopyTensorValues<T>(Tensor<T> source, T[] destination)
            where T : unmanaged
        {
            int index = 0;
            foreach (T value in source)
            {
                destination[index++] = value;
            }
        }

        public void Dispose()
        {
            encoderBinding?.Dispose();
            encoderRunOptions?.Dispose();
            inputFeatures?.Dispose();
            decoderBinding?.Dispose();
            decoderRunOptions?.Dispose();
            inputIds?.Dispose();
            encoderHidden?.Dispose();
        }
    }

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
