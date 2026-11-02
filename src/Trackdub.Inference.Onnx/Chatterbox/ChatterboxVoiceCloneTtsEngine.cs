using System.Buffers.Binary;
using System.Text.Json;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Audio;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.Runtime.Routing;
using Trackdub.Inference.Onnx.Translation;
using Trackdub.Inference.Runtime.Planning;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace Trackdub.Inference.Onnx.Chatterbox;

public sealed class ChatterboxVoiceCloneTtsEngine(
    IConsentService consentService,
    BenchmarkModelPathResolver modelPathResolver)
    : ITtsEngineAdapter, IStageRuntimeExecutionReporter, IDisposable
{
    public const string EngineFamilyName = "chatterbox";

    private const int SampleRate = 24_000;
    private const long ExaggerationToken = 6563;
    private const long StartSpeechToken = 6561;
    private const long StopSpeechToken = 6562;
    private const long StartTextToken = 255;
    private const long StopTextToken = 0;
    private const long EndOfTextToken = 50256;
    private const long SilenceToken = 4299;
    private const int MaxNewTokens = 1024;
    private const int MinimumDurationBudgetNewTokens = 128;
    private const double SpeechTokensPerSecond = 25.0d;
    private const double DurationBudgetMultiplier = 1.75d;
    private const double DurationBudgetSlackSeconds = 2.0d;
    private const int MinimumSpeechEncoderSamples = 1000;
    private const float RepetitionPenalty = 1.2f;
    private const int NumKvHeads = 16;
    private const int HeadDim = 64;

    private readonly IConsentService consentService = consentService ?? throw new ArgumentNullException(nameof(consentService));
    private readonly BenchmarkModelPathResolver modelPathResolver = modelPathResolver ?? throw new ArgumentNullException(nameof(modelPathResolver));
    private readonly SemaphoreSlim sessionGate = new(1, 1);
    private PinnedSessions? pinnedSessions;
    private int disposeSignaled;

    public StageRuntimeExecutionSummary? LastExecutionSummary { get; private set; }

    public string EngineFamily => EngineFamilyName;

    public Task<TtsSynthesisResult> SynthesizeAsync(
        TtsSynthesisRequest request,
        CancellationToken cancellationToken)
    {
        throw new InvalidOperationException("Chatterbox voice cloning must be invoked through the runtime-planned overload.");
    }

    public async Task<TtsSynthesisResult> SynthesizeAsync(
        TtsSynthesisRequest request,
        StageRuntimePlan plan,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(plan);

        if (request.VoiceCloneReference is null)
        {
            throw new InvalidOperationException("Chatterbox voice cloning requires a reference clip.");
        }

        if (!consentService.IsVoiceCloningConsentGranted)
        {
            throw new ConsentRequiredException();
        }

        EnsurePlanReady(plan);
        BenchmarkModelCandidate candidate = PlannedRuntimeModelResolver.ResolveCandidate(plan, modelPathResolver);
        ChatterboxModelFiles modelFiles = ChatterboxModelFiles.Resolve(candidate, plan.Variant);
        float[] referenceAudio = await Pcm16WaveReader.LoadMonoFloat32Async(
            request.VoiceCloneReference.ReferenceClipPath,
            SampleRate,
            cancellationToken);
        referenceAudio = EnsureMinimumReferenceAudioLength(referenceAudio);

        ThrowIfDisposed();
        await sessionGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed();
            PinnedSessions pinned = await GetOrCreatePinnedSessionsAsync(
                modelFiles,
                plan.ExecutionProvider!.Value,
                cancellationToken,
                allowTrtInitFallback: !plan.RequirePreferredExecutionProvider).ConfigureAwait(false);

            string conditionedText = ApplyMultilingualLanguagePrefix(
                request.Text,
                request.LanguageCode,
                modelFiles.IsMultilingual);
            long[] inputIds = BuildTextInputIds(conditionedText, pinned.Tokenizer, modelFiles.IsTurbo);

            // Residency pins keep graphs warm; execution leases last only for this synthesis.
            using LeasedSessions sessions = await pinned
                .AcquireLeasesAsync(cancellationToken)
                .ConfigureAwait(false);
            ChatterboxGenerationResult generation = GenerateSpeechTokens(
                request,
                inputIds,
                referenceAudio,
                sessions.SpeechEncoder.Session,
                sessions.EmbedTokens.Session,
                sessions.LanguageModel.Session,
                cancellationToken);
            float[] audioSamples = DecodeSpeechTokens(
                generation,
                sessions.ConditionalDecoder.Session,
                modelFiles.IsTurbo,
                cancellationToken);
            byte[] wavBytes = WaveAudioWriter.EncodeMonoPcm16(audioSamples, SampleRate);

            LastExecutionSummary = new StageRuntimeExecutionSummary(
                sessions.LanguageModel.RequestedProvider,
                sessions.LanguageModel.SelectedProvider,
                plan.ModelId,
                plan.ModelAlias,
                plan.Variant,
                BuildHybridProviderBootstrapDetail(
                    sessions.SpeechEncoder.SelectedProvider,
                    sessions.EmbedTokens.SelectedProvider,
                    sessions.LanguageModel.SelectedProvider,
                    sessions.ConditionalDecoder.SelectedProvider,
                    sessions.LanguageModel.BootstrapDetail));

            return new TtsSynthesisResult(
                wavBytes,
                DurationSamples: audioSamples.Length,
                SampleRate: SampleRate,
                ModelId: plan.ModelId ?? plan.ModelAlias ?? "chatterbox",
                VoiceId: request.Voice.VoiceId,
                Provider: sessions.LanguageModel.SelectedProvider);
        }
        finally
        {
            sessionGate.Release();
        }
    }

    private static ChatterboxGenerationResult GenerateSpeechTokens(
        TtsSynthesisRequest request,
        long[] textInputIds,
        float[] referenceAudio,
        InferenceSession speechEncoderSession,
        InferenceSession embedTokensSession,
        InferenceSession languageModelSession,
        CancellationToken cancellationToken)
    {
        bool embedNeedsPositionIds = embedTokensSession.InputMetadata.ContainsKey("position_ids");
        bool embedNeedsExaggeration = embedTokensSession.InputMetadata.ContainsKey("exaggeration");
        bool languageNeedsPositionIds = languageModelSession.InputMetadata.ContainsKey("position_ids");

        long[] currentInputIds = textInputIds;
        long[]? embedPositionIds = embedNeedsPositionIds ? BuildInitialBaseEmbedPositionIds(textInputIds) : null;
        // Growable rather than reallocated per step: the previous `Concat(...).ToArray()` copied
        // the whole token/mask sequence on every iteration, which is O(T^2) over a generation.
        var generatedTokens = new List<long> { StartSpeechToken };
        // Maintained alongside the list so the repetition-penalty lookup is O(1) per step instead
        // of rebuilding a set from every generated token on each iteration.
        var seenTokens = new HashSet<long> { StartSpeechToken };
        var attentionMask = new List<long>();
        long[]? languagePositionIds = null;
        PastTensor[] pastKeyValues = [];
        string[] pastKeyNames = languageModelSession.InputMetadata.Keys
            .Where(static name => name.Contains("past_key_values", StringComparison.Ordinal))
            .ToArray();
        long[]? promptTokenIds = null;
        float[]? speakerEmbeddings = null;
        float[]? speakerFeatures = null;
        int[]? speakerEmbeddingsDimensions = null;
        int[]? speakerFeaturesDimensions = null;

        int maxNewTokens = ResolveMaxNewTokens(request.TargetDurationSeconds);
        for (int iteration = 0; iteration < maxNewTokens; iteration++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            TensorData<float> textEmbeds = RunEmbedTokens(
                embedTokensSession,
                currentInputIds,
                embedPositionIds,
                embedNeedsExaggeration,
                cancellationToken);

            TensorData<float> inputsEmbeds = textEmbeds;
            if (iteration == 0)
            {
                using var speechEncoderInputs = new NamedOnnxValueSet();
                speechEncoderInputs.Add(CreateFloatInput(
                    speechEncoderSession,
                    "audio_values",
                    referenceAudio,
                    [1, referenceAudio.Length]));
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> speechResults =
                    speechEncoderSession.RunWithRetry(speechEncoderInputs.Values, cancellationToken: cancellationToken);
                DisposableNamedOnnxValue[] outputs = speechResults.ToArray();
                TensorData<float> condEmbeds = ReadFloatTensor(outputs[0]);
                Tensor<long> promptTensor = outputs[1].AsTensor<long>();
                promptTokenIds = promptTensor.ToArray();
                TensorData<float> speakerEmbeddingTensor = ReadFloatTensor(outputs[2]);
                TensorData<float> speakerFeatureTensor = ReadFloatTensor(outputs[3]);
                speakerEmbeddings = speakerEmbeddingTensor.Values;
                speakerFeatures = speakerFeatureTensor.Values;
                speakerEmbeddingsDimensions = speakerEmbeddingTensor.Dimensions;
                speakerFeaturesDimensions = speakerFeatureTensor.Dimensions;
                inputsEmbeds = ConcatenateEmbeddings(condEmbeds, textEmbeds);

                int batchSize = inputsEmbeds.Dimensions[0];
                int sequenceLength = inputsEmbeds.Dimensions[1];
                attentionMask.Clear();
                attentionMask.AddRange(Enumerable.Repeat(1L, checked(batchSize * sequenceLength)));
                if (languageNeedsPositionIds)
                {
                    languagePositionIds = Enumerable.Range(0, sequenceLength).Select(static value => (long)value).ToArray();
                }

                pastKeyValues = pastKeyNames
                    .Select(name => CreateEmptyPastTensor(name, languageModelSession.InputMetadata[name], batchSize))
                    .ToArray();
            }

            using var languageInputs = new NamedOnnxValueSet();
            languageInputs.Add(CreateFloatInput(
                languageModelSession,
                "inputs_embeds",
                inputsEmbeds.Values,
                inputsEmbeds.Dimensions));
            languageInputs.Add(NamedOnnxValue.CreateFromTensor(
                "attention_mask",
                new DenseTensor<long>(attentionMask.ToArray(), [1, attentionMask.Count])));
            if (languageNeedsPositionIds && languagePositionIds is not null)
            {
                languageInputs.Add(NamedOnnxValue.CreateFromTensor(
                    "position_ids",
                    new DenseTensor<long>(languagePositionIds, [1, languagePositionIds.Length])));
            }

            foreach (PastTensor past in pastKeyValues)
            {
                languageInputs.Add(past.CreateInput());
            }

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> languageResults =
                languageModelSession.RunWithRetry(languageInputs.Values, cancellationToken: cancellationToken);
            DisposableNamedOnnxValue[] languageOutputs = languageResults.ToArray();
            TensorData<float> logits = ReadFloatTensor(languageOutputs[0]);
            long nextToken = SelectNextToken(logits, seenTokens);
            generatedTokens.Add(nextToken);
            seenTokens.Add(nextToken);
            if (nextToken == StopSpeechToken)
            {
                break;
            }

            currentInputIds = [nextToken];
            if (embedNeedsPositionIds)
            {
                embedPositionIds = [iteration + 1];
            }

            attentionMask.Add(1L);
            if (languageNeedsPositionIds && languagePositionIds is not null)
            {
                languagePositionIds = [languagePositionIds[^1] + 1];
            }

            pastKeyValues = languageOutputs
                .Skip(1)
                .Select(output =>
                {
                    string pastName = MapLanguageModelPresentOutputToPastInputName(output.Name);
                    if (!languageModelSession.InputMetadata.TryGetValue(pastName, out NodeMetadata? pastMetadata))
                    {
                        throw new InvalidOperationException(
                            $"Chatterbox language model output '{output.Name}' maps to '{pastName}', but that past KV input was not found. " +
                            $"Known past inputs: {string.Join(", ", pastKeyNames.OrderBy(static n => n, StringComparer.Ordinal))}.");
                    }

                    return PastTensor.FromOutput(pastName, pastMetadata, output);
                })
                .ToArray();
        }

        if (promptTokenIds is null || speakerEmbeddings is null || speakerFeatures is null ||
            speakerEmbeddingsDimensions is null || speakerFeaturesDimensions is null)
        {
            throw new InvalidOperationException("Chatterbox speech encoder did not produce reference conditioning tensors.");
        }

        return new ChatterboxGenerationResult(
            generatedTokens.ToArray(),
            promptTokenIds,
            speakerEmbeddings,
            speakerEmbeddingsDimensions,
            speakerFeatures,
            speakerFeaturesDimensions);
    }

    /// <summary>
    /// Maps ONNX present KV output tensor names back to the matching <c>past_key_values.*</c> input names.
    /// Some exports use <c>present_key_values.N.(key|value)</c>; others shorten to <c>present.N.(key|value)</c>.
    /// A naive <c>present</c>→<c>past</c> replacement breaks the latter (<c>past.N.key</c> vs <c>past_key_values.N.key</c>).
    /// </summary>
    internal static string MapLanguageModelPresentOutputToPastInputName(string outputName)
    {
        if (outputName.Contains("present_key_values", StringComparison.Ordinal))
        {
            return outputName.Replace("present_key_values", "past_key_values", StringComparison.Ordinal);
        }

        const string shortPresentPrefix = "present.";
        if (outputName.StartsWith(shortPresentPrefix, StringComparison.Ordinal))
        {
            return string.Concat("past_key_values.", outputName.AsSpan(shortPresentPrefix.Length));
        }

        return outputName.Replace("present", "past", StringComparison.Ordinal);
    }

    private static TensorData<float> RunEmbedTokens(
        InferenceSession session,
        long[] inputIds,
        long[]? positionIds,
        bool needsExaggeration,
        CancellationToken cancellationToken)
    {
        using var inputs = new NamedOnnxValueSet();
        inputs.Add(NamedOnnxValue.CreateFromTensor(
            "input_ids",
            new DenseTensor<long>(inputIds, [1, inputIds.Length])));
        if (positionIds is not null && session.InputMetadata.ContainsKey("position_ids"))
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor(
                "position_ids",
                new DenseTensor<long>(positionIds, [1, positionIds.Length])));
        }

        if (needsExaggeration)
        {
            inputs.Add(CreateFloatInput(
                session,
                "exaggeration",
                [0.5f],
                [1]));
        }

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results =
            session.RunWithRetry(inputs.Values, cancellationToken: cancellationToken);
        return ReadFloatTensor(results.Single());
    }

    private static float[] DecodeSpeechTokens(
        ChatterboxGenerationResult generation,
        InferenceSession decoderSession,
        bool isTurbo,
        CancellationToken cancellationToken)
    {
        long[] speechTokens = generation.GeneratedTokens
            .Skip(1)
            .TakeWhile(static token => token != StopSpeechToken)
            .ToArray();
        if (isTurbo)
        {
            speechTokens = speechTokens.Concat(Enumerable.Repeat(SilenceToken, 3)).ToArray();
        }

        long[] decoderSpeechTokens = generation.PromptTokenIds.Concat(speechTokens).ToArray();
        using var inputs = new NamedOnnxValueSet();
        inputs.Add(NamedOnnxValue.CreateFromTensor(
            "speech_tokens",
            new DenseTensor<long>(decoderSpeechTokens, [1, decoderSpeechTokens.Length])));
        inputs.Add(CreateFloatInput(
            decoderSession,
            "speaker_embeddings",
            generation.SpeakerEmbeddings,
            generation.SpeakerEmbeddingsDimensions));
        inputs.Add(CreateFloatInput(
            decoderSession,
            "speaker_features",
            generation.SpeakerFeatures,
            generation.SpeakerFeaturesDimensions));

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results =
            decoderSession.RunWithRetry(inputs.Values, cancellationToken: cancellationToken);
        return ReadFloatTensor(results.Single()).Values;
    }

    private static TensorData<float> ConcatenateEmbeddings(
        TensorData<float> left,
        TensorData<float> right)
    {
        if (left.Dimensions.Length != 3 || right.Dimensions.Length != 3 ||
            left.Dimensions[0] != right.Dimensions[0] ||
            left.Dimensions[2] != right.Dimensions[2])
        {
            throw new InvalidOperationException("Chatterbox embedding tensors must have compatible [batch, sequence, hidden] shapes.");
        }

        int batch = left.Dimensions[0];
        int leftSequence = left.Dimensions[1];
        int rightSequence = right.Dimensions[1];
        int hidden = left.Dimensions[2];
        float[] values = new float[checked(batch * (leftSequence + rightSequence) * hidden)];
        for (int batchIndex = 0; batchIndex < batch; batchIndex++)
        {
            int leftOffset = batchIndex * leftSequence * hidden;
            int rightOffset = batchIndex * rightSequence * hidden;
            int destinationOffset = batchIndex * (leftSequence + rightSequence) * hidden;
            Array.Copy(left.Values, leftOffset, values, destinationOffset, leftSequence * hidden);
            Array.Copy(right.Values, rightOffset, values, destinationOffset + (leftSequence * hidden), rightSequence * hidden);
        }

        return new TensorData<float>(values, [batch, leftSequence + rightSequence, hidden]);
    }

    private static long[] BuildInitialBaseEmbedPositionIds(long[] inputIds)
    {
        var positionIds = new long[inputIds.Length];
        for (int index = 0; index < inputIds.Length; index++)
        {
            // Matches the Chatterbox ONNX reference script: text positions are arange(seq) - 1,
            // while speech token positions reset to 0 for the speech embedding space.
            positionIds[index] = inputIds[index] >= StartSpeechToken ? 0 : index - 1L;
        }

        return positionIds;
    }

    // The translation matrix is wider than the Chatterbox multilingual model card (it also lists
    // Chinese), so languages the model cannot speak are excluded explicitly.
    internal static readonly IReadOnlySet<string> UnsupportedMultilingualLanguages =
        new HashSet<string>(StringComparer.Ordinal) { "zh" };

    /// <summary>
    /// Conditions the input text for the Chatterbox multilingual model by prepending the
    /// <c>[xx]</c> language token it expects (mirrors <c>prepare_language</c> in the model
    /// card). English-only models (turbo and base) are returned unchanged. Throws when a
    /// multilingual synthesis is requested without a supported language code, rather than
    /// silently synthesizing in the wrong language.
    /// </summary>
    private static string ApplyMultilingualLanguagePrefix(
        string text,
        string languageCode,
        bool isMultilingual)
    {
        if (!isMultilingual)
        {
            return text;
        }

        string? normalized = string.IsNullOrWhiteSpace(languageCode)
            ? null
            : languageCode.Trim().ToLowerInvariant();

        if (normalized is null || UnsupportedMultilingualLanguages.Contains(normalized) || !TranslationLanguageCoverageMatrix.TryGetLanguage(normalized, out _))
        {
            throw new NotSupportedException(
                $"Chatterbox multilingual synthesis requires a supported language code; '{languageCode}' is not in the supported set.");
        }

        // Per-language text normalization (e.g. Japanese hiragana, Hebrew diacritics, Korean)
        // from the model card's prepare_language is not yet applied — tracked as a follow-up.
        // The [xx] token alone still selects the target language for synthesis.
        return $"[{normalized}]{text}";
    }

    private static long[] BuildTextInputIds(
        string text,
        ChatterboxTokenizer tokenizer,
        bool isTurbo) =>
        BuildTextInputIds(tokenizer.Encode(text), isTurbo);

    private static long[] BuildTextInputIds(
        long[] tokenIds,
        bool isTurbo) =>
        isTurbo
            ? [.. tokenIds, EndOfTextToken, EndOfTextToken]
            : [ExaggerationToken, StartTextToken, .. tokenIds, StopTextToken, StartSpeechToken, StartSpeechToken];

    private static int ResolveMaxNewTokens(double? targetDurationSeconds)
    {
        if (targetDurationSeconds is not double seconds ||
            !double.IsFinite(seconds) ||
            seconds <= 0d)
        {
            return MaxNewTokens;
        }

        double budgetSeconds = Math.Max(
            seconds * DurationBudgetMultiplier,
            seconds + DurationBudgetSlackSeconds);
        int budgetTokens = (int)Math.Ceiling(budgetSeconds * SpeechTokensPerSecond);
        return Math.Clamp(budgetTokens, MinimumDurationBudgetNewTokens, MaxNewTokens);
    }

    private static float[] EnsureMinimumReferenceAudioLength(float[] samples)
    {
        if (samples.Length >= MinimumSpeechEncoderSamples)
        {
            return samples;
        }

        var padded = new float[MinimumSpeechEncoderSamples];
        Array.Copy(samples, padded, samples.Length);
        return padded;
    }

    private static PastTensor CreateEmptyPastTensor(
        string name,
        NodeMetadata metadata,
        int batchSize)
    {
        int[] dimensions = [batchSize, NumKvHeads, 0, HeadDim];
        return metadata.ElementType == typeof(Half)
            ? new PastTensor(name, FloatTensorElementKind.SystemHalf, [], [], [], dimensions)
            : metadata.ElementType == typeof(Float16)
                ? new PastTensor(name, FloatTensorElementKind.OnnxRuntimeFloat16, [], [], [], dimensions)
                : new PastTensor(name, FloatTensorElementKind.Float32, [], [], [], dimensions);
    }

    private static long SelectNextToken(
        TensorData<float> logits,
        IReadOnlySet<long> seenTokens)
    {
        int vocabularySize = logits.Dimensions[^1];
        int offset = logits.Values.Length - vocabularySize;
        long bestToken = 0;
        float bestScore = float.NegativeInfinity;
        for (int index = 0; index < vocabularySize; index++)
        {
            float score = logits.Values[offset + index];
            if (seenTokens.Contains(index))
            {
                score = score < 0f ? score * RepetitionPenalty : score / RepetitionPenalty;
            }

            if (score > bestScore)
            {
                bestScore = score;
                bestToken = index;
            }
        }

        return bestToken;
    }

    private static TensorData<float> ReadFloatTensor(DisposableNamedOnnxValue value)
    {
        if (value.Value is Tensor<Float16> onnxRuntimeFloat16Tensor)
        {
            Float16[] halfValues = onnxRuntimeFloat16Tensor.ToArray();
            float[] values = new float[halfValues.Length];
            for (int index = 0; index < halfValues.Length; index++)
            {
                values[index] = (float)halfValues[index];
            }

            return new TensorData<float>(values, onnxRuntimeFloat16Tensor.Dimensions.ToArray());
        }

        if (value.Value is Tensor<Half> halfTensor)
        {
            Half[] halfValues = halfTensor.ToArray();
            float[] values = new float[halfValues.Length];
            for (int index = 0; index < halfValues.Length; index++)
            {
                values[index] = (float)halfValues[index];
            }

            return new TensorData<float>(values, halfTensor.Dimensions.ToArray());
        }

        Tensor<float> tensor = value.AsTensor<float>();
        return new TensorData<float>(tensor.ToArray(), tensor.Dimensions.ToArray());
    }

    internal static NamedOnnxValue CreateFloatInputForTesting(
        string name,
        Type elementType,
        float[] values,
        int[] dimensions) =>
        CreateFloatInput(name, elementType, values, dimensions);

    internal static NamedOnnxValue CreatePastInputForTesting(
        string name,
        Type elementType,
        float[] values,
        int[] dimensions) =>
        PastTensor.FromFloatValues(name, elementType, values, dimensions).CreateInput();

    private static NamedOnnxValue CreateFloatInput(
        InferenceSession session,
        string name,
        float[] values,
        int[] dimensions)
    {
        if (!session.InputMetadata.TryGetValue(name, out NodeMetadata? metadata))
        {
            throw new InvalidOperationException($"Chatterbox model input '{name}' was not found.");
        }

        return CreateFloatInput(name, metadata.ElementType, values, dimensions);
    }

    private static NamedOnnxValue CreateFloatInput(
        string name,
        Type elementType,
        float[] values,
        int[] dimensions)
    {
        if (elementType == typeof(Half))
        {
            var halfValues = new Half[values.Length];
            for (int index = 0; index < values.Length; index++)
            {
                halfValues[index] = (Half)values[index];
            }

            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<Half>(halfValues, dimensions));
        }

        if (elementType == typeof(Float16))
        {
            var halfValues = new Float16[values.Length];
            for (int index = 0; index < values.Length; index++)
            {
                halfValues[index] = (Float16)values[index];
            }

            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<Float16>(halfValues, dimensions));
        }

        if (elementType == typeof(float))
        {
            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(values, dimensions));
        }

        throw new InvalidOperationException(
            $"Chatterbox model input '{name}' must be float32 or float16, but declared '{elementType.Name}'.");
    }

    private static StageRuntimeExecutionSummary CreatePlannedOnlySummary(
        StageRuntimePlan plan,
        string bootstrapDetail) =>
        new(
            plan.ExecutionProvider is ExecutionProviderKind.Cpu ? "cpu" : "auto",
            plan.ExecutionProvider is ExecutionProviderKind.DirectMl ? "dml" : "cpu",
            plan.ModelId,
            plan.ModelAlias,
            plan.Variant,
            bootstrapDetail);

    private static ExecutionProviderKind ResolveReferenceConditioningProvider(ExecutionProviderKind _) =>
        ExecutionProviderKind.Cpu;

    private static ExecutionProviderKind ResolveLanguageModelProvider(ExecutionProviderKind _) =>
        ExecutionProviderKind.Cpu;

    private static ExecutionProviderKind ResolveConditionalDecoderProvider(ExecutionProviderKind plannedProvider) =>
        plannedProvider;

    private static string? BuildHybridProviderBootstrapDetail(
        string speechEncoderSelectedProvider,
        string embedTokensSelectedProvider,
        string languageModelSelectedProvider,
        string conditionalDecoderSelectedProvider,
        string? languageModelBootstrapDetail)
    {
        if (string.Equals(
                speechEncoderSelectedProvider,
                languageModelSelectedProvider,
                StringComparison.OrdinalIgnoreCase) &&
            string.Equals(
                embedTokensSelectedProvider,
                languageModelSelectedProvider,
                StringComparison.OrdinalIgnoreCase) &&
            string.Equals(
                conditionalDecoderSelectedProvider,
                languageModelSelectedProvider,
                StringComparison.OrdinalIgnoreCase))
        {
            return languageModelBootstrapDetail;
        }

        string speechEncoderDetail = string.Equals(
                speechEncoderSelectedProvider,
                conditionalDecoderSelectedProvider,
                StringComparison.OrdinalIgnoreCase)
            ? string.Empty
            : "Chatterbox speech encoder runs on CPU because its x-vector AveragePool graph is not compatible with DirectML GPU execution.";
        string embedTokensDetail = string.Equals(
                embedTokensSelectedProvider,
                conditionalDecoderSelectedProvider,
                StringComparison.OrdinalIgnoreCase)
            ? string.Empty
            : "Chatterbox token embedding sidecar runs on CPU because its Slice graph is not compatible with DirectML GPU execution.";
        string languageModelDetail = string.Equals(
                languageModelSelectedProvider,
                conditionalDecoderSelectedProvider,
                StringComparison.OrdinalIgnoreCase)
            ? string.Empty
            : "Chatterbox language model runs on CPU because DirectML changes the autoregressive token sequence for this graph.";
        string providerDetail = $"Language model selected provider: {languageModelSelectedProvider}. Conditional decoder selected provider: {conditionalDecoderSelectedProvider}.";
        string sidecarDetail = string.Join(' ', new[] { speechEncoderDetail, embedTokensDetail, languageModelDetail }
            .Where(static detail => detail.Length > 0));
        return string.IsNullOrWhiteSpace(languageModelBootstrapDetail)
            ? $"{sidecarDetail} {providerDetail}"
            : $"{sidecarDetail} {providerDetail} {languageModelBootstrapDetail}";
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposeSignaled, 1) == 1)
        {
            return;
        }

        if (!sessionGate.Wait(TimeSpan.FromSeconds(10)))
        {
            // Timeout — a concurrent async operation is still using the engine.
            // Proceed with disposal anyway; the concurrent operation will hit
            // ObjectDisposedException on the disposed gate/session.
            sessionGate.Dispose();
            return;
        }
        try
        {
            pinnedSessions?.Dispose();
            pinnedSessions = null;
        }
        finally
        {
            // Dispose without releasing: any concurrent WaitAsync gets ObjectDisposedException
            // rather than entering the try block on a partially-disposed engine.
            sessionGate.Dispose();
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref disposeSignaled) != 0, this);

    private async Task<PinnedSessions> GetOrCreatePinnedSessionsAsync(
        ChatterboxModelFiles modelFiles,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        bool allowTrtInitFallback = true)
    {
        if (pinnedSessions is not null &&
            string.Equals(pinnedSessions.ModelRootDirectory, modelFiles.RootDirectory, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(pinnedSessions.SpeechEncoderPath, modelFiles.SpeechEncoderPath, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(pinnedSessions.EmbedTokensPath, modelFiles.EmbedTokensPath, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(pinnedSessions.LanguageModelPath, modelFiles.LanguageModelPath, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(pinnedSessions.ConditionalDecoderPath, modelFiles.ConditionalDecoderPath, StringComparison.OrdinalIgnoreCase) &&
            pinnedSessions.Provider == provider &&
            pinnedSessions.AllowTrtInitFallback == allowTrtInitFallback)
        {
            return pinnedSessions;
        }

        pinnedSessions?.Dispose();
        pinnedSessions = null;

        OnnxExecutionSessionFactory.PooledSingleSessionPin? speechEncoder = null;
        OnnxExecutionSessionFactory.PooledSingleSessionPin? embedTokens = null;
        OnnxExecutionSessionFactory.PooledSingleSessionPin? languageModel = null;
        OnnxExecutionSessionFactory.PooledSingleSessionPin? conditionalDecoder = null;
        try
        {
            ExecutionProviderKind conditioningProvider = ResolveReferenceConditioningProvider(provider);
            ExecutionProviderKind languageModelProvider = ResolveLanguageModelProvider(provider);
            ExecutionProviderKind conditionalDecoderProvider = ResolveConditionalDecoderProvider(provider);
            speechEncoder = await OnnxExecutionSessionFactory
                .PinPooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "chatterbox", Provider = conditioningProvider, CancellationToken = cancellationToken, AllowTrtInitFallback = allowTrtInitFallback }, modelFiles.SpeechEncoderPath)
                .ConfigureAwait(false);
            embedTokens = await OnnxExecutionSessionFactory
                .PinPooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "chatterbox", Provider = conditioningProvider, CancellationToken = cancellationToken, AllowTrtInitFallback = allowTrtInitFallback }, modelFiles.EmbedTokensPath)
                .ConfigureAwait(false);
            languageModel = await OnnxExecutionSessionFactory
                .PinPooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "chatterbox", Provider = languageModelProvider, CancellationToken = cancellationToken, AllowTrtInitFallback = allowTrtInitFallback }, modelFiles.LanguageModelPath)
                .ConfigureAwait(false);
            conditionalDecoder = await OnnxExecutionSessionFactory
                .PinPooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "chatterbox", Provider = conditionalDecoderProvider, CancellationToken = cancellationToken, AllowTrtInitFallback = allowTrtInitFallback }, modelFiles.ConditionalDecoderPath)
                .ConfigureAwait(false);
        }
        catch
        {
            speechEncoder?.Dispose();
            embedTokens?.Dispose();
            languageModel?.Dispose();
            conditionalDecoder?.Dispose();
            throw;
        }

        ChatterboxTokenizer tokenizer;
        try
        {
            tokenizer = await ChatterboxTokenizer.LoadAsync(modelFiles.RootDirectory).ConfigureAwait(false);
        }
        catch
        {
            speechEncoder.Dispose();
            embedTokens.Dispose();
            languageModel.Dispose();
            conditionalDecoder.Dispose();
            throw;
        }

        pinnedSessions = new PinnedSessions(
            modelFiles.RootDirectory,
            modelFiles.SpeechEncoderPath,
            modelFiles.EmbedTokensPath,
            modelFiles.LanguageModelPath,
            modelFiles.ConditionalDecoderPath,
            provider,
            allowTrtInitFallback,
            speechEncoder,
            embedTokens,
            languageModel,
            conditionalDecoder,
            tokenizer);
        return pinnedSessions;
    }

    private static void EnsurePlanReady(StageRuntimePlan plan)
    {
        if (plan.IsRunnable() &&
            plan.ExecutionProvider is not null &&
            !string.IsNullOrWhiteSpace(plan.ModelAlias))
        {
            return;
        }

        throw new InvalidOperationException(
            plan.Fallback?.Detail ?? "Runtime planner did not produce a ready Chatterbox TTS plan.");
    }

    private sealed class PinnedSessions(
        string modelRootDirectory,
        string speechEncoderPath,
        string embedTokensPath,
        string languageModelPath,
        string conditionalDecoderPath,
        ExecutionProviderKind provider,
        bool allowTrtInitFallback,
        OnnxExecutionSessionFactory.PooledSingleSessionPin speechEncoder,
        OnnxExecutionSessionFactory.PooledSingleSessionPin embedTokens,
        OnnxExecutionSessionFactory.PooledSingleSessionPin languageModel,
        OnnxExecutionSessionFactory.PooledSingleSessionPin conditionalDecoder,
        ChatterboxTokenizer tokenizer)
        : IDisposable
    {
        public string ModelRootDirectory { get; } = modelRootDirectory;
        public string SpeechEncoderPath { get; } = speechEncoderPath;
        public string EmbedTokensPath { get; } = embedTokensPath;
        public string LanguageModelPath { get; } = languageModelPath;
        public string ConditionalDecoderPath { get; } = conditionalDecoderPath;
        public ExecutionProviderKind Provider { get; } = provider;
        public bool AllowTrtInitFallback { get; } = allowTrtInitFallback;
        private OnnxExecutionSessionFactory.PooledSingleSessionPin SpeechEncoderPin { get; } = speechEncoder;
        private OnnxExecutionSessionFactory.PooledSingleSessionPin EmbedTokensPin { get; } = embedTokens;
        private OnnxExecutionSessionFactory.PooledSingleSessionPin LanguageModelPin { get; } = languageModel;
        private OnnxExecutionSessionFactory.PooledSingleSessionPin ConditionalDecoderPin { get; } = conditionalDecoder;
        public ChatterboxTokenizer Tokenizer { get; } = tokenizer;

        /// <summary>Short execution leases for one synthesis (audit §3A: residency ≠ exclusivity).</summary>
        public async Task<LeasedSessions> AcquireLeasesAsync(CancellationToken cancellationToken)
        {
            OnnxExecutionSessionFactory.SingleSessionLease? speech = null;
            OnnxExecutionSessionFactory.SingleSessionLease? embed = null;
            OnnxExecutionSessionFactory.SingleSessionLease? lm = null;
            OnnxExecutionSessionFactory.SingleSessionLease? decoder = null;
            try
            {
                speech = await SpeechEncoderPin.AcquireAsync(cancellationToken).ConfigureAwait(false);
                embed = await EmbedTokensPin.AcquireAsync(cancellationToken).ConfigureAwait(false);
                lm = await LanguageModelPin.AcquireAsync(cancellationToken).ConfigureAwait(false);
                decoder = await ConditionalDecoderPin.AcquireAsync(cancellationToken).ConfigureAwait(false);
                return new LeasedSessions(speech, embed, lm, decoder);
            }
            catch
            {
                speech?.Dispose();
                embed?.Dispose();
                lm?.Dispose();
                decoder?.Dispose();
                throw;
            }
        }

        public void Dispose()
        {
            SpeechEncoderPin.Dispose();
            EmbedTokensPin.Dispose();
            LanguageModelPin.Dispose();
            ConditionalDecoderPin.Dispose();
        }
    }

    private sealed class LeasedSessions(
        OnnxExecutionSessionFactory.SingleSessionLease speechEncoder,
        OnnxExecutionSessionFactory.SingleSessionLease embedTokens,
        OnnxExecutionSessionFactory.SingleSessionLease languageModel,
        OnnxExecutionSessionFactory.SingleSessionLease conditionalDecoder) : IDisposable
    {
        public OnnxExecutionSessionFactory.SingleSessionLease SpeechEncoder { get; } = speechEncoder;
        public OnnxExecutionSessionFactory.SingleSessionLease EmbedTokens { get; } = embedTokens;
        public OnnxExecutionSessionFactory.SingleSessionLease LanguageModel { get; } = languageModel;
        public OnnxExecutionSessionFactory.SingleSessionLease ConditionalDecoder { get; } = conditionalDecoder;

        public void Dispose()
        {
            SpeechEncoder.Dispose();
            EmbedTokens.Dispose();
            LanguageModel.Dispose();
            ConditionalDecoder.Dispose();
        }
    }

    private sealed record ChatterboxGenerationResult(
        long[] GeneratedTokens,
        long[] PromptTokenIds,
        float[] SpeakerEmbeddings,
        int[] SpeakerEmbeddingsDimensions,
        float[] SpeakerFeatures,
        int[] SpeakerFeaturesDimensions);

    private sealed record TensorData<T>(
        T[] Values,
        int[] Dimensions);

    private enum FloatTensorElementKind
    {
        Float32,
        SystemHalf,
        OnnxRuntimeFloat16
    }

    private sealed record PastTensor(
        string Name,
        FloatTensorElementKind ElementKind,
        float[] FloatValues,
        Half[] HalfValues,
        Float16[] OnnxRuntimeFloat16Values,
        int[] Dimensions)
    {
        public NamedOnnxValue CreateInput() =>
            ElementKind switch
            {
                FloatTensorElementKind.SystemHalf =>
                    NamedOnnxValue.CreateFromTensor(Name, new DenseTensor<Half>(HalfValues, Dimensions)),
                FloatTensorElementKind.OnnxRuntimeFloat16 =>
                    NamedOnnxValue.CreateFromTensor(Name, new DenseTensor<Float16>(OnnxRuntimeFloat16Values, Dimensions)),
                _ => NamedOnnxValue.CreateFromTensor(Name, new DenseTensor<float>(FloatValues, Dimensions))
            };

        public static PastTensor FromOutput(string name, NodeMetadata metadata, DisposableNamedOnnxValue output)
        {
            if (output.Value is Tensor<Float16> onnxRuntimeFloat16Tensor)
            {
                return FromOnnxRuntimeFloat16Values(
                    name,
                    metadata.ElementType,
                    onnxRuntimeFloat16Tensor.ToArray(),
                    onnxRuntimeFloat16Tensor.Dimensions.ToArray());
            }

            if (output.Value is Tensor<Half> halfTensor)
            {
                return FromHalfValues(
                    name,
                    metadata.ElementType,
                    halfTensor.ToArray(),
                    halfTensor.Dimensions.ToArray());
            }

            Tensor<float> floatTensor = output.AsTensor<float>();
            return FromFloatValues(name, metadata.ElementType, floatTensor.ToArray(), floatTensor.Dimensions.ToArray());
        }

        public static PastTensor FromFloatValues(
            string name,
            Type elementType,
            float[] values,
            int[] dimensions)
        {
            if (elementType == typeof(Half))
            {
                var halfValues = new Half[values.Length];
                for (int index = 0; index < values.Length; index++)
                {
                    halfValues[index] = (Half)values[index];
                }

                return new PastTensor(
                    name,
                    FloatTensorElementKind.SystemHalf,
                    [],
                    halfValues,
                    [],
                    dimensions);
            }

            if (elementType == typeof(Float16))
            {
                var halfValues = new Float16[values.Length];
                for (int index = 0; index < values.Length; index++)
                {
                    halfValues[index] = (Float16)values[index];
                }

                return new PastTensor(
                    name,
                    FloatTensorElementKind.OnnxRuntimeFloat16,
                    [],
                    [],
                    halfValues,
                    dimensions);
            }

            if (elementType == typeof(float))
            {
                return new PastTensor(
                    name,
                    FloatTensorElementKind.Float32,
                    values,
                    [],
                    [],
                    dimensions);
            }

            throw new InvalidOperationException(
                $"Chatterbox past input '{name}' must be float32 or float16, but declared '{elementType.Name}'.");
        }

        private static PastTensor FromHalfValues(
            string name,
            Type elementType,
            Half[] values,
            int[] dimensions)
        {
            if (elementType == typeof(Half))
            {
                return new PastTensor(
                    name,
                    FloatTensorElementKind.SystemHalf,
                    [],
                    values,
                    [],
                    dimensions);
            }

            if (elementType == typeof(Float16))
            {
                var onnxRuntimeValues = new Float16[values.Length];
                for (int index = 0; index < values.Length; index++)
                {
                    onnxRuntimeValues[index] = (Float16)(float)values[index];
                }

                return new PastTensor(
                    name,
                    FloatTensorElementKind.OnnxRuntimeFloat16,
                    [],
                    [],
                    onnxRuntimeValues,
                    dimensions);
            }

            if (elementType == typeof(float))
            {
                var floatValues = new float[values.Length];
                for (int index = 0; index < values.Length; index++)
                {
                    floatValues[index] = (float)values[index];
                }

                return new PastTensor(
                    name,
                    FloatTensorElementKind.Float32,
                    floatValues,
                    [],
                    [],
                    dimensions);
            }

            throw new InvalidOperationException(
                $"Chatterbox past input '{name}' must be float32 or float16, but declared '{elementType.Name}'.");
        }

        private static PastTensor FromOnnxRuntimeFloat16Values(
            string name,
            Type elementType,
            Float16[] values,
            int[] dimensions)
        {
            if (elementType == typeof(Float16))
            {
                return new PastTensor(
                    name,
                    FloatTensorElementKind.OnnxRuntimeFloat16,
                    [],
                    [],
                    values,
                    dimensions);
            }

            if (elementType == typeof(Half))
            {
                var halfValues = new Half[values.Length];
                for (int index = 0; index < values.Length; index++)
                {
                    halfValues[index] = (Half)(float)values[index];
                }

                return new PastTensor(
                    name,
                    FloatTensorElementKind.SystemHalf,
                    [],
                    halfValues,
                    [],
                    dimensions);
            }

            if (elementType == typeof(float))
            {
                var floatValues = new float[values.Length];
                for (int index = 0; index < values.Length; index++)
                {
                    floatValues[index] = (float)values[index];
                }

                return new PastTensor(
                    name,
                    FloatTensorElementKind.Float32,
                    floatValues,
                    [],
                    [],
                    dimensions);
            }

            throw new InvalidOperationException(
                $"Chatterbox past input '{name}' must be float32 or float16, but declared '{elementType.Name}'.");
        }
    }

    private sealed class NamedOnnxValueSet : IDisposable
    {
        private readonly List<NamedOnnxValue> values = [];

        public IReadOnlyList<NamedOnnxValue> Values => values;

        public void Add(NamedOnnxValue value) => values.Add(value);

        public void Dispose()
        {
            foreach (IDisposable value in values.OfType<IDisposable>())
            {
                value.Dispose();
            }

            values.Clear();
        }
    }

    private sealed class ChatterboxModelFiles
    {
        private ChatterboxModelFiles(
            string rootDirectory,
            string speechEncoderPath,
            string embedTokensPath,
            string languageModelPath,
            string conditionalDecoderPath,
            bool isTurbo,
            bool isMultilingual)
        {
            RootDirectory = rootDirectory;
            SpeechEncoderPath = speechEncoderPath;
            EmbedTokensPath = embedTokensPath;
            LanguageModelPath = languageModelPath;
            ConditionalDecoderPath = conditionalDecoderPath;
            IsTurbo = isTurbo;
            IsMultilingual = isMultilingual;
        }

        public string RootDirectory { get; }

        public string SpeechEncoderPath { get; }

        public string EmbedTokensPath { get; }

        public string LanguageModelPath { get; }

        public string ConditionalDecoderPath { get; }

        public bool IsTurbo { get; }

        /// <summary>
        /// True when the resolved package is the Chatterbox multilingual model, which
        /// expects a <c>[xx]</c> language token prepended to the input text. The English-only
        /// turbo and base models must NOT receive that token.
        /// </summary>
        public bool IsMultilingual { get; }

        public static ChatterboxModelFiles Resolve(BenchmarkModelCandidate candidate, string? variant)
        {
            string rootDirectory = candidate.RootDirectory
                ?? Path.GetDirectoryName(candidate.ModelPath)
                ?? throw new InvalidOperationException("Cannot resolve Chatterbox model root path.");
            string languageModelPath = candidate.ModelPath;
            string speechEncoderPath = ResolveGraphPath(rootDirectory, "speech_encoder", variant);
            string embedTokensPath = ResolveGraphPath(rootDirectory, "embed_tokens", variant);
            string conditionalDecoderPath = ResolveGraphPath(rootDirectory, "conditional_decoder", variant);
            string tokenizerPath = Path.Join(rootDirectory, "tokenizer.json");
            foreach (string path in new[] { languageModelPath, speechEncoderPath, embedTokensPath, conditionalDecoderPath, tokenizerPath })
            {
                if (!File.Exists(path))
                {
                    throw new FileNotFoundException("Chatterbox voice cloning requires the full ONNX package.", path);
                }
            }

            bool isTurbo = rootDirectory.Contains("turbo", StringComparison.OrdinalIgnoreCase);
            bool isMultilingual = rootDirectory.Contains("multilingual", StringComparison.OrdinalIgnoreCase);
            return new ChatterboxModelFiles(
                rootDirectory,
                speechEncoderPath,
                embedTokensPath,
                languageModelPath,
                conditionalDecoderPath,
                isTurbo,
                isMultilingual);
        }

        private static string ResolveGraphPath(string modelRootPath, string graphName, string? variant)
        {
            string onnxDirectory = Path.Join(modelRootPath, "onnx");
            if (!string.IsNullOrWhiteSpace(variant) &&
                !variant.Equals("default", StringComparison.OrdinalIgnoreCase))
            {
                string variantPath = Path.Join(onnxDirectory, $"{graphName}_{variant}.onnx");
                if (File.Exists(variantPath))
                {
                    return variantPath;
                }
            }

            return Path.Join(onnxDirectory, $"{graphName}.onnx");
        }
    }

    /// <summary>
    /// Picks the BPE unknown token a Chatterbox tokenizer.json actually declares. Turbo ships a GPT-2
    /// style tokenizer whose unknown token is <c>&lt;|endoftext|&gt;</c>; the base and multilingual
    /// tokenizers declare <c>[UNK]</c> and have no <c>&lt;|endoftext|&gt;</c>, so a hard-coded value
    /// made loading them throw "Unknown Token ... was not present in 'Vocabulary'".
    /// </summary>
    internal static string? ResolveUnknownToken(JsonElement model, IReadOnlyDictionary<string, int> vocabulary)
    {
        if (model.TryGetProperty("unk_token", out JsonElement declared) &&
            declared.ValueKind is JsonValueKind.String &&
            declared.GetString() is { Length: > 0 } declaredToken &&
            vocabulary.ContainsKey(declaredToken))
        {
            return declaredToken;
        }

        return vocabulary.ContainsKey("<|endoftext|>") ? "<|endoftext|>" : null;
    }

    private sealed class ChatterboxTokenizer
    {
        private readonly BpeTokenizer tokenizer;

        private ChatterboxTokenizer(BpeTokenizer tokenizer)
        {
            this.tokenizer = tokenizer;
        }

        public static async Task<ChatterboxTokenizer> LoadAsync(string modelRootPath)
        {
            string tokenizerPath = Path.Join(modelRootPath, "tokenizer.json");
            string tokenizerText = await File.ReadAllTextAsync(tokenizerPath).ConfigureAwait(false);
            using JsonDocument document = JsonDocument.Parse(tokenizerText);
            JsonElement root = document.RootElement;
            JsonElement model = root.GetProperty("model");
            var vocabulary = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (JsonProperty entry in model.GetProperty("vocab").EnumerateObject())
            {
                vocabulary[entry.Name] = entry.Value.GetInt32();
            }

            var merges = new List<string>();
            foreach (JsonElement merge in model.GetProperty("merges").EnumerateArray())
            {
                if (merge.ValueKind is JsonValueKind.String)
                {
                    merges.Add(merge.GetString() ?? string.Empty);
                    continue;
                }

                if (merge.ValueKind is JsonValueKind.Array)
                {
                    string[] parts = merge.EnumerateArray()
                        .Select(part => part.GetString() ?? string.Empty)
                        .ToArray();
                    if (parts.Length == 2)
                    {
                        merges.Add($"{parts[0]} {parts[1]}");
                    }
                }
            }

            Dictionary<string, int> specialTokens = ReadSpecialTokens(root);
            var options = new BpeOptions(vocabulary)
            {
                Merges = merges,
                SpecialTokens = specialTokens,
                UnknownToken = ResolveUnknownToken(model, vocabulary),
                ByteLevel = true
            };

            return new ChatterboxTokenizer(BpeTokenizer.Create(options));
        }

        public long[] Encode(string text)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(text);
            return tokenizer
                .EncodeToIds(text.Trim(), false, false)
                .Select(static tokenId => (long)tokenId)
                .ToArray();
        }

        private static Dictionary<string, int> ReadSpecialTokens(JsonElement root)
        {
            var tokens = new Dictionary<string, int>(StringComparer.Ordinal);
            if (!root.TryGetProperty("added_tokens", out JsonElement addedTokens) ||
                addedTokens.ValueKind is not JsonValueKind.Array)
            {
                return tokens;
            }

            foreach (JsonElement token in addedTokens.EnumerateArray())
            {
                if (!token.TryGetProperty("content", out JsonElement contentElement) ||
                    !token.TryGetProperty("id", out JsonElement idElement) ||
                    contentElement.ValueKind is not JsonValueKind.String ||
                    idElement.ValueKind is not JsonValueKind.Number)
                {
                    continue;
                }

                string? content = contentElement.GetString();
                if (!string.IsNullOrWhiteSpace(content) && idElement.TryGetInt32(out int id))
                {
                    tokens[content] = id;
                }
            }

            return tokens;
        }
    }

    private static class Pcm16WaveReader
    {
        public static async Task<float[]> LoadMonoFloat32Async(string path, int targetSampleRate, CancellationToken cancellationToken = default)
        {
            byte[] bytes = await File.ReadAllBytesAsync(path, cancellationToken).ConfigureAwait(false);
            WaveInfo info = Parse(bytes);
            ReadOnlySpan<byte> data = bytes.AsSpan(info.DataOffset, info.DataLength);
            int frameCount = info.DataLength / info.BlockAlign;
            float[] samples = new float[frameCount];
            for (int frame = 0; frame < frameCount; frame++)
            {
                float sum = 0f;
                int frameOffset = frame * info.BlockAlign;
                for (int channel = 0; channel < info.ChannelCount; channel++)
                {
                    short sample = BinaryPrimitives.ReadInt16LittleEndian(
                        data.Slice(frameOffset + (channel * sizeof(short)), sizeof(short)));
                    sum += sample / 32768f;
                }

                samples[frame] = sum / info.ChannelCount;
            }

            return info.SampleRate == targetSampleRate
                ? samples
                : ResampleLinear(samples, info.SampleRate, targetSampleRate);
        }

        private static float[] ResampleLinear(float[] samples, int sourceSampleRate, int targetSampleRate)
        {
            if (samples.Length == 0)
            {
                return [];
            }

            int outputLength = Math.Max(1, (int)Math.Round(samples.Length * (double)targetSampleRate / sourceSampleRate));
            float[] output = new float[outputLength];
            double ratio = sourceSampleRate / (double)targetSampleRate;
            for (int index = 0; index < output.Length; index++)
            {
                double sourcePosition = index * ratio;
                int leftIndex = Math.Min(samples.Length - 1, (int)Math.Floor(sourcePosition));
                int rightIndex = Math.Min(samples.Length - 1, leftIndex + 1);
                double fraction = sourcePosition - leftIndex;
                output[index] = (float)(samples[leftIndex] + ((samples[rightIndex] - samples[leftIndex]) * fraction));
            }

            return output;
        }

        private static WaveInfo Parse(byte[] bytes)
        {
            if (bytes.Length < 44 ||
                !bytes.AsSpan(0, 4).SequenceEqual("RIFF"u8) ||
                !bytes.AsSpan(8, 4).SequenceEqual("WAVE"u8))
            {
                throw new InvalidOperationException("Chatterbox reference clips must be RIFF/WAVE audio.");
            }

            int offset = 12;
            int sampleRate = 0;
            short channelCount = 0;
            short bitsPerSample;
            short blockAlign = 0;
            int dataOffset = -1;
            int dataLength = 0;
            while (offset + 8 <= bytes.Length)
            {
                ReadOnlySpan<byte> header = bytes.AsSpan(offset, 8);
                string chunkId = System.Text.Encoding.ASCII.GetString(header[..4]);
                int chunkSize = BinaryPrimitives.ReadInt32LittleEndian(header[4..]);
                int chunkDataOffset = offset + 8;
                if (chunkSize < 0 || chunkDataOffset + chunkSize > bytes.Length)
                {
                    throw new InvalidOperationException("Wave metadata could not be parsed.");
                }

                if (chunkId.Equals("fmt ", StringComparison.Ordinal))
                {
                    ReadOnlySpan<byte> fmt = bytes.AsSpan(chunkDataOffset, chunkSize);
                    short audioFormat = BinaryPrimitives.ReadInt16LittleEndian(fmt[..2]);
                    channelCount = BinaryPrimitives.ReadInt16LittleEndian(fmt[2..4]);
                    sampleRate = BinaryPrimitives.ReadInt32LittleEndian(fmt[4..8]);
                    blockAlign = BinaryPrimitives.ReadInt16LittleEndian(fmt[12..14]);
                    bitsPerSample = BinaryPrimitives.ReadInt16LittleEndian(fmt[14..16]);
                    if (audioFormat != 1 || bitsPerSample != 16)
                    {
                        throw new InvalidOperationException("Chatterbox reference clips must be PCM16 wave files.");
                    }
                }
                else if (chunkId.Equals("data", StringComparison.Ordinal))
                {
                    dataOffset = chunkDataOffset;
                    dataLength = chunkSize;
                }

                offset = chunkDataOffset + chunkSize + (chunkSize % 2);
            }

            if (sampleRate <= 0 || channelCount <= 0 || blockAlign <= 0 || dataOffset < 0)
            {
                throw new InvalidOperationException("Wave metadata could not be parsed.");
            }

            return new WaveInfo(dataOffset, dataLength, sampleRate, channelCount, blockAlign);
        }

        private sealed record WaveInfo(
            int DataOffset,
            int DataLength,
            int SampleRate,
            short ChannelCount,
            short BlockAlign);
    }
}
