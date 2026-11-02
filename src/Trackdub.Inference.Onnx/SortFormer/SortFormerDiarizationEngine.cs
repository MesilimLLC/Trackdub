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

namespace Trackdub.Inference.Onnx.SortFormer;

public sealed class SortFormerDiarizationEngine(IRuntimePlanner runtimePlanner,
    BenchmarkModelPathResolver modelPathResolver,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null)
    : ISpeakerDiarizationEngineAdapter, IStageRuntimeExecutionReporter
{
    public const string EngineFamilyName = "sortformer";

    private const int TargetSampleRate = 16000;
    private const float SpeakerActiveThreshold = 0.5f;
    private const float OverlapThreshold = 0.5f;
    internal const double FragmentSpeakerMaxSeconds = 3.0;
    // Maximum supported speakers for the 4-speaker SortFormer diarization model.
    public const int MaxSupportedSpeakers = 4;
    // NVIDIA's recommended offline ("very high latency") streaming config for v2.1:
    // chunk 340, right context 40, FIFO 40, cache update period 300, speaker cache 188.
    private const int StreamingChunkModelFrames = 340;
    private const int StreamingRightContextModelFrames = 40;
    private const int StreamingSpeakerCacheUpdatePeriodFrames = 300;
    private const int StreamingFeatureSubsampling = 8;
    private const int StreamingFifoFrames = 40;
    private const int StreamingSpeakerCacheFrames = 188;
    internal const int StreamingEmbeddingDimension = 512;
    private const int StreamingChunkStrideFeatureFrames = StreamingChunkModelFrames * StreamingFeatureSubsampling;
    // Shared with the smoke tester so probe inputs match the streaming export's fixed
    // optimization profile instead of the (unrelated) waveform-only TRT profile.
    internal const int StreamingFeedFeatureFrames =
        (StreamingChunkModelFrames + StreamingRightContextModelFrames) * StreamingFeatureSubsampling;

    // Streaming export inputs: chunk is always fed padded to the full feed window; the speaker cache
    // and FIFO grow from empty to their caps. A profile that names no real input leaves TRT-RTX with
    // a fully dynamic range, where /pre_encode/Reshape (kMIN) and the attention broadcast (kMAX) have
    // no consistent solution and the EP-context compile emits no engine.
    private static readonly string StreamingChunkShape =
        $"chunk:1x{StreamingFeedFeatureFrames}x{SortFormerFeatureExtractor.MelBins}";

    // Shared with OnnxExecutionProviderSmokeTester and EP-context AOT so smoke, AOT and stage runs
    // build the same engine profile and hit the same pool key.
    internal static readonly IReadOnlyDictionary<string, string> TrtOptions = new Dictionary<string, string>
    {
        ["trt_profile_min_shapes"] =
            $"{StreamingChunkShape},spkcache:1x0x{StreamingEmbeddingDimension},fifo:1x0x{StreamingEmbeddingDimension}",
        ["trt_profile_opt_shapes"] =
            $"{StreamingChunkShape},spkcache:1x{StreamingSpeakerCacheFrames}x{StreamingEmbeddingDimension},fifo:1x{StreamingFifoFrames}x{StreamingEmbeddingDimension}",
        ["trt_profile_max_shapes"] =
            $"{StreamingChunkShape},spkcache:1x{StreamingSpeakerCacheFrames}x{StreamingEmbeddingDimension},fifo:1x{StreamingFifoFrames}x{StreamingEmbeddingDimension}",
    };

    internal static bool IsStreamingExportInputSet(IEnumerable<string> inputNames)
    {
        var names = inputNames as IReadOnlySet<string> ?? inputNames.ToHashSet(StringComparer.Ordinal);
        return names.Contains("chunk") && names.Contains("spkcache") && names.Contains("fifo");
    }

    private static readonly SortFormerFeatureExtractor FeatureExtractor = new();
    private readonly IRuntimePlanner runtimePlanner = runtimePlanner ?? throw new ArgumentNullException(nameof(runtimePlanner));
    private readonly BenchmarkModelPathResolver modelPathResolver = modelPathResolver ?? throw new ArgumentNullException(nameof(modelPathResolver));

    public StageRuntimeExecutionSummary? LastExecutionSummary { get; private set; }

    /// <summary>Streaming chunks served by the fixed-address bound path in the last run.</summary>
    internal int LastBoundChunkCount { get; private set; }

    public string EngineFamily => EngineFamilyName;

    public async Task<IReadOnlyList<DiarizedSpeakerTurn>> DiarizeAsync(
        string normalizedAudioPath,
        double durationSeconds,
        IReadOnlyList<SpeechRegion> speechRegions,
        CancellationToken cancellationToken) =>
        await DiarizeAsync(
            new SpeakerDiarizationRequest(
                normalizedAudioPath,
                durationSeconds,
                speechRegions,
                InferenceRequestOptions.Default),
            cancellationToken).ConfigureAwait(false);

    public async Task<IReadOnlyList<DiarizedSpeakerTurn>> DiarizeAsync(
        SpeakerDiarizationRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        InferenceRequestOptions options = request.Options ?? InferenceRequestOptions.Default;

        StageRuntimePlan plan = await runtimePlanner.PlanAsync(
            await StageRuntimePlanningRequestFactory.ApplyPreferredModelTierAsync(new StageRuntimePlanningRequest(
                RuntimeStage.Diarization,
                options.NormalizedPreferredModelAlias,
                PreferredExecutionProvider: ExecutionProviderRequest.ParsePreferredExecutionProvider(
                    options.PreferredExecutionProvider,
                    options.RequirePreferredExecutionProvider),
                RequirePreferredExecutionProvider: options.RequirePreferredExecutionProvider,
                PreferredModelVariantAlias: options.NormalizedPreferredModelVariantAlias),
            runtimePlanningPreferences,
            cancellationToken),
            cancellationToken).ConfigureAwait(false);

        return await DiarizeAsync(request, plan, cancellationToken).ConfigureAwait(false);
    }

    public async Task<IReadOnlyList<DiarizedSpeakerTurn>> DiarizeAsync(
        SpeakerDiarizationRequest request,
        StageRuntimePlan plan,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(request.SpeechRegions);
        EnsurePlanReady(plan, RuntimeStage.Diarization);

        string modelPath = ResolvePlannedModelPath(plan);
        using OnnxExecutionSessionFactory.SingleSessionLease sessionLease = await OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "sortformer", Provider = plan.ExecutionProvider!.Value, CancellationToken = cancellationToken, AdditionalTrtOptions = TrtOptions, AllowTrtInitFallback = !plan.RequirePreferredExecutionProvider }, modelPath)
            .ConfigureAwait(false);

        EnsureModelSpeakerCapacity(sessionLease.Session, plan.ModelAlias);

        IAudioSamples audio = await WaveAudioReader.ReadMonoPcm16Async(request.NormalizedAudioPath, cancellationToken).ConfigureAwait(false);
        // CreateResampledStream returns the source as-is when sample rates match;
        // otherwise it wraps and takes ownership of it. Dispose only the outer reader.
        using IAudioSamples targetAudio = AudioResampler.CreateResampledStream(audio, TargetSampleRate);
        // Do not hard-mask diarization with VAD regions; VAD misses would permanently erase speech before speaker detection.

        IReadOnlyList<DiarizedSpeakerTurn> turns;
        if (UsesStreamingFeatureInputs(sessionLease.Session))
        {
            Tensor<float> probabilityTensor = RunStreamingFeatureModel(
                sessionLease.Session,
                sessionLease.SelectedProvider,
                targetAudio,
                cancellationToken,
                out int boundChunkCount);
            LastBoundChunkCount = boundChunkCount;
            turns = DecodeTurns(probabilityTensor, request.DurationSeconds, plan.ModelAlias);
        }
        else
        {
            if (targetAudio.SampleFrameCount > int.MaxValue)
            {
                throw new InvalidOperationException(
                    $"Audio is too long for SortFormer diarization ({targetAudio.SampleFrameCount} frames at {TargetSampleRate} Hz).");
            }
            float[] samples = new float[(int)targetAudio.SampleFrameCount];
            targetAudio.ReadMonoSamples(0, samples);

            using var inputSet = CreateInputSet(sessionLease.Session, samples);
            cancellationToken.ThrowIfCancellationRequested();
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = sessionLease.Session.RunWithRetry(inputSet.Values, cancellationToken: cancellationToken);
            Tensor<float> probabilityTensor = ResolveProbabilityTensor(outputs);
            LastBoundChunkCount = 0;
            turns = DecodeTurns(probabilityTensor, request.DurationSeconds, plan.ModelAlias);
        }

        LastExecutionSummary = new StageRuntimeExecutionSummary(
            sessionLease.RequestedProvider,
            sessionLease.SelectedProvider,
            plan.ModelId,
            plan.ModelAlias,
            plan.Variant,
            sessionLease.BootstrapDetail);
        return turns;
    }

    private static bool UsesStreamingFeatureInputs(InferenceSession session) =>
        IsStreamingExportInputSet(session.InputMetadata.Keys);

    private static DenseTensor<float> RunStreamingFeatureModel(
        InferenceSession session,
        string selectedProviderLabel,
        IAudioSamples audio,
        CancellationToken cancellationToken,
        out int boundChunkCount)
    {
        boundChunkCount = 0;
        if (audio.SampleFrameCount <= 0)
        {
            return new DenseTensor<float>(Array.Empty<float>(), [0, MaxSupportedSpeakers]);
        }

        int totalFeatureFrames = SortFormerFeatureExtractor.GetFrameCount(audio.SampleFrameCount);

        var state = new SortFormerStreamingState();
        var predictionData = new List<float>();
        int speakerCount = MaxSupportedSpeakers;
        int chunkCount = CeilingDivide(totalFeatureFrames, StreamingChunkStrideFeatureFrames);

        // This session handles both managed and bound chunks, so CUDA graphs stay disabled.
        // On TRT-RTX the streaming inputs sit in pinned host buffers at their profile capacities
        // instead of fresh per-chunk tensors, but early/short chunks still use dynamic managed feeds.
        // The bound path is only entered once speaker cache and FIFO are fully saturated: the
        // export's attention mask is a contiguous prefix over [spkcache | fifo | chunk], so any
        // padded row inside that prefix would leak into attention and corrupt predictions.
        // At full capacity the feed is identical to the exact-length managed feed.
        using SortFormerStreamingRun? streamingRun =
            string.Equals(
                selectedProviderLabel,
                ProviderNames.FormatProviderLabel(ExecutionProviderKind.TensorRTRtx),
                StringComparison.Ordinal)
                ? SortFormerStreamingRun.TryCreate(session)
                : null;

        for (int chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int startFrame = chunkIndex * StreamingChunkStrideFeatureFrames;
            int currentFeatureFrameCount = Math.Min(
                StreamingFeedFeatureFrames,
                totalFeatureFrames - startFrame);

            using SortFormerFeatureInputSet features = FeatureExtractor.Extract(
                audio,
                startFrame,
                currentFeatureFrameCount,
                cancellationToken);
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs;
            if (streamingRun is not null && streamingRun.TryLoad(features, currentFeatureFrameCount, state))
            {
                boundChunkCount++;
                streamingRun.CopyInputsToDevice();
                outputs = session.RunWithBindingAndNamesRetry(
                    streamingRun.RunOptions,
                    streamingRun.Binding,
                    SortFormerStreamingRun.BoundOutputNames,
                    cancellationToken: cancellationToken,
                    provider: ExecutionProviderKind.TensorRTRtx);
            }
            else
            {
                using var inputSet = CreateStreamingFeatureInputSet(
                    session.InputMetadata,
                    features,
                    0,
                    currentFeatureFrameCount,
                    state);
                outputs = session.RunWithRetry(inputSet.Values, cancellationToken: cancellationToken);
            }

            using (outputs)
            {
                Tensor<float> rawPredictions = ResolveRequiredFloatTensor(outputs, "spkcache_fifo_chunk_preds");
                Tensor<float> rawEmbeddings = ResolveRequiredFloatTensor(outputs, "chunk_pre_encode_embs");
                int outputSpeakerCount = ResolveFeatureCount(rawPredictions);
                if (outputSpeakerCount > MaxSupportedSpeakers)
                {
                    throw new InvalidOperationException(
                        $"SortFormer ONNX export produced {outputSpeakerCount} speakers, exceeding the maximum supported count of {MaxSupportedSpeakers}.");
                }

                if (predictionData.Count == 0)
                {
                    speakerCount = outputSpeakerCount;
                }
                else if (speakerCount != outputSpeakerCount)
                {
                    throw new InvalidOperationException("SortFormer ONNX export changed speaker count across streaming chunks.");
                }

                int validModelFrameCount = CeilingDivide(currentFeatureFrameCount, StreamingFeatureSubsampling);
                int keepModelFrameCount = Math.Min(StreamingChunkModelFrames, validModelFrameCount);
                int predictionStartFrame = state.SpeakerCacheFrameCount + state.FifoFrameCount;
                float[] chunkPredictions = ExtractTensorFrameSlice(
                    rawPredictions,
                    predictionStartFrame,
                    keepModelFrameCount,
                    speakerCount);
                predictionData.AddRange(chunkPredictions);

                float[] chunkEmbeddings = ExtractTensorFrameSlice(
                    rawEmbeddings,
                    0,
                    keepModelFrameCount,
                    StreamingEmbeddingDimension);

                float[] cacheAndFifoPredictions = ExtractTensorFrameSlice(
                    rawPredictions,
                    0,
                    predictionStartFrame,
                    speakerCount);
                state.Update(
                    cacheAndFifoPredictions,
                    chunkPredictions,
                    speakerCount,
                    chunkEmbeddings,
                    keepModelFrameCount,
                    keepModelFrameCount);
            }
        }

        int frameCount = predictionData.Count / speakerCount;
        return new DenseTensor<float>(predictionData.ToArray(), [frameCount, speakerCount]);
    }

    private static InputSet CreateInputSet(InferenceSession session, float[] samples)
    {
        IReadOnlyDictionary<string, NodeMetadata> inputs = session.InputMetadata;

        KeyValuePair<string, NodeMetadata> waveformInput = default;
        if (inputs.TryGetValue("waveform", out NodeMetadata? waveformMeta))
        {
            waveformInput = new KeyValuePair<string, NodeMetadata>("waveform", waveformMeta);
        }
        else if (inputs.TryGetValue("audio_signal", out NodeMetadata? audioSignalMeta))
        {
            waveformInput = new KeyValuePair<string, NodeMetadata>("audio_signal", audioSignalMeta);
        }
        else
        {
            KeyValuePair<string, NodeMetadata>[] floatInputs = inputs
                .Where(static candidate => candidate.Value.ElementType == typeof(float))
                .ToArray();
            if (floatInputs.Length == 0)
            {
                throw new InvalidOperationException("SortFormer ONNX export does not expose any float waveform input.");
            }

            if (floatInputs.Length > 1)
            {
                throw new InvalidOperationException($"SortFormer ONNX export has {floatInputs.Length} float inputs; expected exactly one waveform input.");
            }

            waveformInput = floatInputs[0];
        }

        if (string.IsNullOrWhiteSpace(waveformInput.Key))
        {
            throw new InvalidOperationException("SortFormer ONNX export does not expose a float waveform input.");
        }

        string waveformInputName = waveformInput.Key;
        int[] waveformDimensions = waveformInput.Value.Dimensions.ToArray();
        IReadOnlyList<NamedOnnxValue> values =
        [
            NamedOnnxValue.CreateFromTensor(
                waveformInputName,
                new DenseTensor<float>(samples, ResolveWaveformShape(waveformDimensions, samples.Length)))
        ];

        KeyValuePair<string, NodeMetadata> lengthInput = default;
        if (inputs.TryGetValue("length", out NodeMetadata? lengthMeta))
        {
            lengthInput = new KeyValuePair<string, NodeMetadata>("length", lengthMeta);
        }
        else if (inputs.TryGetValue("audio_signal_length", out NodeMetadata? audioLengthMeta))
        {
            lengthInput = new KeyValuePair<string, NodeMetadata>("audio_signal_length", audioLengthMeta);
        }
        else
        {
            lengthInput = inputs.FirstOrDefault(static candidate =>
                candidate.Value.ElementType == typeof(long) &&
                candidate.Key.Contains("length", StringComparison.OrdinalIgnoreCase));
        }

        if (!string.IsNullOrWhiteSpace(lengthInput.Key))
        {
            values = values
                .Append(NamedOnnxValue.CreateFromTensor(
                    lengthInput.Key,
                    new DenseTensor<long>(new long[] { samples.Length }, [1])))
                .ToArray();
        }

        return new InputSet(values);
    }

    private static int[] ResolveWaveformShape(IReadOnlyList<int> modelDimensions, int sampleCount)
    {
        if (modelDimensions.Count == 1)
        {
            return [sampleCount];
        }

        if (modelDimensions.Count == 2)
        {
            return [1, sampleCount];
        }

        throw new InvalidOperationException("SortFormer ONNX export waveform input must be rank 1 or 2.");
    }

    private static InputSet CreateStreamingFeatureInputSet(
        IReadOnlyDictionary<string, NodeMetadata> inputs,
        SortFormerFeatureInputSet features,
        int startFrame,
        int currentFeatureFrameCount,
        SortFormerStreamingState state)
    {
        EnsureRequiredInput(inputs, "chunk", typeof(float));
        EnsureRequiredInput(inputs, "chunk_lengths", typeof(long));
        EnsureRequiredInput(inputs, "spkcache", typeof(float));
        EnsureRequiredInput(inputs, "spkcache_lengths", typeof(long));
        EnsureRequiredInput(inputs, "fifo", typeof(float));
        EnsureRequiredInput(inputs, "fifo_lengths", typeof(long));

        int chunkLength = StreamingFeedFeatureFrames * SortFormerFeatureExtractor.MelBins;
        var chunkData = System.Buffers.ArrayPool<float>.Shared.Rent(chunkLength);
        Array.Clear(chunkData, 0, chunkLength);
        features.CopyFramesTo(chunkData, 0, startFrame, currentFeatureFrameCount);

        IReadOnlyList<NamedOnnxValue> values =
        [
            NamedOnnxValue.CreateFromTensor(
                "chunk",
                new DenseTensor<float>(
                    new Memory<float>(chunkData, 0, chunkLength),
                    [1, StreamingFeedFeatureFrames, SortFormerFeatureExtractor.MelBins])),
            NamedOnnxValue.CreateFromTensor(
                "chunk_lengths",
                new DenseTensor<long>(new long[] { currentFeatureFrameCount }, [1])),
            NamedOnnxValue.CreateFromTensor(
                "spkcache",
                new DenseTensor<float>(
                    state.SpeakerCacheEmbeddings,
                    [1, state.SpeakerCacheFrameCount, StreamingEmbeddingDimension])),
            NamedOnnxValue.CreateFromTensor(
                "spkcache_lengths",
                new DenseTensor<long>(new long[] { state.SpeakerCacheFrameCount }, [1])),
            NamedOnnxValue.CreateFromTensor(
                "fifo",
                new DenseTensor<float>(
                    state.FifoEmbeddings,
                    [1, state.FifoFrameCount, StreamingEmbeddingDimension])),
            NamedOnnxValue.CreateFromTensor(
                "fifo_lengths",
                new DenseTensor<long>(new long[] { state.FifoFrameCount }, [1]))
        ];

        return new InputSet(values, chunkData);
    }

    private static void EnsureRequiredInput(
        IReadOnlyDictionary<string, NodeMetadata> inputs,
        string name,
        Type elementType)
    {
        if (!inputs.TryGetValue(name, out NodeMetadata? metadata))
        {
            throw new InvalidOperationException($"SortFormer ONNX export is missing required input '{name}'.");
        }

        if (metadata.ElementType != elementType)
        {
            throw new InvalidOperationException(
                $"SortFormer ONNX export input '{name}' has element type '{metadata.ElementType.Name}', expected '{elementType.Name}'.");
        }
    }

    private static Tensor<float> ResolveRequiredFloatTensor(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        string outputName)
    {
        foreach (DisposableNamedOnnxValue output in outputs)
        {
            if (!string.Equals(output.Name, outputName, StringComparison.Ordinal))
            {
                continue;
            }

            try
            {
                return output.AsTensor<float>();
            }
            catch (InvalidCastException exception)
            {
                throw new InvalidOperationException(
                    $"SortFormer ONNX export output '{outputName}' is not a float tensor.",
                    exception);
            }
        }

        throw new InvalidOperationException($"SortFormer ONNX export did not produce required output '{outputName}'.");
    }

    private static int ResolveFeatureCount(Tensor<float> tensor)
    {
        int[] dimensions = tensor.Dimensions.ToArray();
        if (dimensions.Length is not (2 or 3))
        {
            throw new InvalidOperationException("SortFormer ONNX export tensor must be rank 2 or batch-first rank 3.");
        }

        return dimensions[^1];
    }

    private static float[] ExtractTensorFrameSlice(
        Tensor<float> tensor,
        int startFrame,
        int frameCount,
        int featureCount)
    {
        if (frameCount <= 0)
        {
            return [];
        }

        int[] dimensions = tensor.Dimensions.ToArray();
        int availableFrames = dimensions.Length switch
        {
            2 when dimensions[1] == featureCount => dimensions[0],
            3 when dimensions[0] == 1 && dimensions[2] == featureCount => dimensions[1],
            _ => throw new InvalidOperationException(
                $"SortFormer ONNX export tensor shape [{string.Join(", ", dimensions)}] does not match expected feature count {featureCount}.")
        };

        if (startFrame < 0 || startFrame + frameCount > availableFrames)
        {
            throw new InvalidOperationException(
                $"SortFormer ONNX export tensor has {availableFrames} frame(s); requested {frameCount} frame(s) from {startFrame}.");
        }

        var data = new float[frameCount * featureCount];
        for (int frameIndex = 0; frameIndex < frameCount; frameIndex++)
        {
            for (int featureIndex = 0; featureIndex < featureCount; featureIndex++)
            {
                data[(frameIndex * featureCount) + featureIndex] = dimensions.Length == 2
                    ? tensor[startFrame + frameIndex, featureIndex]
                    : tensor[0, startFrame + frameIndex, featureIndex];
            }
        }

        return data;
    }

    private static int CeilingDivide(int value, int divisor) => (value + divisor - 1) / divisor;

    private static float[] ConcatenateFrames(
        float[] first,
        int firstFrameCount,
        float[] second,
        int secondFrameCount,
        int featureCount)
    {
        if (firstFrameCount == 0)
        {
            return second.ToArray();
        }

        if (secondFrameCount == 0)
        {
            return first.ToArray();
        }

        var combined = new float[(firstFrameCount + secondFrameCount) * featureCount];
        Array.Copy(first, 0, combined, 0, firstFrameCount * featureCount);
        Array.Copy(second, 0, combined, firstFrameCount * featureCount, secondFrameCount * featureCount);
        return combined;
    }

    private static float[] SliceFrames(float[] source, int startFrame, int frameCount, int featureCount)
    {
        if (frameCount <= 0)
        {
            return [];
        }

        var sliced = new float[frameCount * featureCount];
        Array.Copy(source, startFrame * featureCount, sliced, 0, frameCount * featureCount);
        return sliced;
    }

    private static Tensor<float> ResolveProbabilityTensor(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs)
    {
        foreach (DisposableNamedOnnxValue output in outputs)
        {
            try
            {
                Tensor<float> tensor = output.AsTensor<float>();
                int[] dimensions = tensor.Dimensions.ToArray();
                if (dimensions.Length is 2 or 3)
                {
                    return tensor;
                }
            }
            catch (InvalidCastException)
            {
            }
        }

        throw new InvalidOperationException("SortFormer ONNX export did not produce a frame probability tensor.");
    }

    private static void EnsureModelSpeakerCapacity(InferenceSession session, string? modelAlias)
    {
        int? speakerDim = ResolveOutputSpeakerDimension(session);
        if (speakerDim is > MaxSupportedSpeakers)
        {
            throw new InvalidOperationException(
                $"SortFormer model '{modelAlias ?? "unknown"}' declares {speakerDim.Value} output speakers, " +
                $"but only {MaxSupportedSpeakers} are supported. Use a 4-speaker SortFormer model export.");
        }
    }

    private static int? ResolveOutputSpeakerDimension(InferenceSession session)
    {
        if (session.OutputMetadata.TryGetValue("spkcache_fifo_chunk_preds", out NodeMetadata? streamingMeta))
        {
            int[] dims = streamingMeta.Dimensions.ToArray();
            if (dims.Length > 0 && dims[^1] > 0)
            {
                return dims[^1];
            }
        }

        foreach (KeyValuePair<string, NodeMetadata> output in session.OutputMetadata)
        {
            if (output.Value.ElementType != typeof(float))
            {
                continue;
            }

            int[] dims = output.Value.Dimensions.ToArray();
            if (dims.Length is 2 or 3 && dims[^1] > 0)
            {
                return dims[^1];
            }
        }

        return null;
    }

    private static IReadOnlyList<DiarizedSpeakerTurn> DecodeTurns(Tensor<float> probabilities, double durationSeconds, string? modelAlias)
    {
        if (durationSeconds <= 0d)
        {
            return [];
        }

        (int frameCount, int speakerCount, Func<int, int, float> accessor) = CreateTensorAccessor(probabilities);
        if (frameCount <= 0 || speakerCount <= 0)
        {
            return [];
        }

        if (speakerCount > MaxSupportedSpeakers)
        {
            throw new InvalidOperationException(
                $"SortFormer model '{modelAlias ?? "unknown"}' produced {speakerCount} speakers, " +
                $"exceeding the maximum supported count of {MaxSupportedSpeakers} (frameCount={frameCount}).");
        }

        double secondsPerFrame = durationSeconds / frameCount;
        if (!double.IsFinite(secondsPerFrame) || secondsPerFrame <= 0d)
        {
            return [];
        }

        var turns = new List<DiarizedSpeakerTurn>();
        ActiveTurn? activeTurn = null;

        for (int frameIndex = 0; frameIndex < frameCount; frameIndex++)
        {
            int primarySpeakerIndex = -1;
            float primarySpeakerProbability = float.NegativeInfinity;
            int activeSpeakerCount = 0;

            for (int speakerIndex = 0; speakerIndex < speakerCount; speakerIndex++)
            {
                float probability = accessor(frameIndex, speakerIndex);
                if (probability >= OverlapThreshold)
                {
                    activeSpeakerCount++;
                }

                if (probability > primarySpeakerProbability)
                {
                    primarySpeakerProbability = probability;
                    primarySpeakerIndex = speakerIndex;
                }
            }

            double frameStart = frameIndex * secondsPerFrame;
            double frameEnd = Math.Min(durationSeconds, (frameIndex + 1) * secondsPerFrame);
            bool isSilentFrame = primarySpeakerProbability < SpeakerActiveThreshold || primarySpeakerIndex < 0;
            if (isSilentFrame)
            {
                FlushActiveTurn(turns, ref activeTurn, durationSeconds);
                continue;
            }

            bool hasOverlap = activeSpeakerCount > 1;
            if (activeTurn is not null &&
                activeTurn.SpeakerIndex == primarySpeakerIndex &&
                Math.Abs(activeTurn.EndSeconds - frameStart) <= secondsPerFrame * 1.5d)
            {
                activeTurn = activeTurn with
                {
                    EndSeconds = frameEnd,
                    ConfidenceSum = activeTurn.ConfidenceSum + primarySpeakerProbability,
                    FrameCount = activeTurn.FrameCount + 1,
                    HasOverlap = activeTurn.HasOverlap || hasOverlap
                };
                continue;
            }

            FlushActiveTurn(turns, ref activeTurn, durationSeconds);
            activeTurn = new ActiveTurn(
                primarySpeakerIndex,
                frameStart,
                frameEnd,
                primarySpeakerProbability,
                FrameCount: 1,
                hasOverlap);
        }

        FlushActiveTurn(turns, ref activeTurn, durationSeconds);
        return MergeFragmentSpeakers(turns);
    }

    // A speaker slot with only a few seconds in the whole file is almost always a transient label at a
    // speaker change: streaming SortFormer briefly opens a new slot before the incoming speaker settles.
    // Relabel those turns to the next real speaker (the one taking over), else the previous one.
    internal static List<DiarizedSpeakerTurn> MergeFragmentSpeakers(List<DiarizedSpeakerTurn> turns)
    {
        Dictionary<string, double> totals = turns
            .GroupBy(static t => t.NormalizedSpeakerKey)
            .ToDictionary(static g => g.Key, static g => g.Sum(static t => t.EndSeconds - t.StartSeconds));
        if (totals.Count < 2 || totals.All(static kv => kv.Value < FragmentSpeakerMaxSeconds))
        {
            return turns;
        }

        bool IsFragment(DiarizedSpeakerTurn turn) => totals[turn.NormalizedSpeakerKey] < FragmentSpeakerMaxSeconds;

        var merged = new List<DiarizedSpeakerTurn>(turns.Count);
        for (int i = 0; i < turns.Count; i++)
        {
            DiarizedSpeakerTurn turn = turns[i];
            if (!IsFragment(turn))
            {
                merged.Add(turn);
                continue;
            }

            DiarizedSpeakerTurn? target = turns.Skip(i + 1).FirstOrDefault(t => !IsFragment(t))
                ?? turns.Take(i).LastOrDefault(t => !IsFragment(t));
            merged.Add(target is null ? turn : turn with { SpeakerKey = target.SpeakerKey });
        }

        return merged;
    }

    private static (int FrameCount, int SpeakerCount, Func<int, int, float> Accessor) CreateTensorAccessor(Tensor<float> tensor)
    {
        int[] dimensions = tensor.Dimensions.ToArray();
        return dimensions.Length switch
        {
            2 => (dimensions[0], dimensions[1], (frameIndex, speakerIndex) => tensor[frameIndex, speakerIndex]),
            3 when dimensions[0] == 1 => (dimensions[1], dimensions[2], (frameIndex, speakerIndex) => tensor[0, frameIndex, speakerIndex]),
            3 when dimensions[2] == 1 => (dimensions[0], dimensions[1], (frameIndex, speakerIndex) => tensor[frameIndex, speakerIndex, 0]),
            _ => throw new InvalidOperationException("SortFormer ONNX export probability tensor must be rank 2 or batch-first rank 3.")
        };
    }

    private static void FlushActiveTurn(
        ICollection<DiarizedSpeakerTurn> turns,
        ref ActiveTurn? activeTurn,
        double durationSeconds)
    {
        if (activeTurn is null)
        {
            return;
        }

        double clippedStart = Math.Clamp(activeTurn.StartSeconds, 0d, durationSeconds);
        double clippedEnd = Math.Clamp(activeTurn.EndSeconds, clippedStart, durationSeconds);
        if (clippedEnd > clippedStart)
        {
            turns.Add(new DiarizedSpeakerTurn(
                $"spk_{activeTurn.SpeakerIndex}",
                clippedStart,
                clippedEnd,
                activeTurn.ConfidenceSum / activeTurn.FrameCount,
                activeTurn.HasOverlap));
        }

        activeTurn = null;
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

    private string ResolvePlannedModelPath(StageRuntimePlan plan)
    {
        if (!string.IsNullOrWhiteSpace(plan.ModelEntryPath) && File.Exists(plan.ModelEntryPath))
        {
            return Path.GetFullPath(plan.ModelEntryPath);
        }

        BenchmarkModelCandidate candidate = modelPathResolver.ResolveSingle(plan.ModelAlias!, plan.Variant);
        return candidate.ModelPath;
    }

    private sealed class InputSet(IReadOnlyList<NamedOnnxValue> values, float[]? rentedArray = null)
        : IDisposable
    {
        public IReadOnlyList<NamedOnnxValue> Values { get; } = values;

        public void Dispose()
        {
            foreach (IDisposable value in Values.OfType<IDisposable>())
            {
                value.Dispose();
            }

            if (rentedArray is not null)
            {
                System.Buffers.ArrayPool<float>.Shared.Return(rentedArray);
            }
        }
    }

    /// <summary>
    /// Persistent TRT-RTX binding for the streaming export: chunk, caches and FIFO live in pinned
    /// host OrtValues at their profile capacities for the whole diarization pass. The bound path is
    /// only loaded once both caches are saturated, because the export's attention mask is a
    /// contiguous prefix over [spkcache | fifo | chunk] and any padded row inside that prefix would
    /// corrupt predictions; at full capacity the feed is identical to the exact-length managed feed.
    /// </summary>
    private sealed class SortFormerStreamingRun : IDisposable
    {
        internal static readonly string[] BoundOutputNames =
            ["spkcache_fifo_chunk_preds", "chunk_pre_encode_embs", "chunk_pre_encode_lengths"];

        private readonly DeviceBoundTensor<float> chunk;
        private readonly DeviceBoundTensor<long> chunkLengths;
        private readonly DeviceBoundTensor<float> spkcache;
        private readonly DeviceBoundTensor<long> spkcacheLengths;
        private readonly DeviceBoundTensor<float> fifo;
        private readonly DeviceBoundTensor<long> fifoLengths;
        private readonly OrtValue[] hostInputs;
        private readonly OrtValue[] deviceInputs;

        internal SortFormerStreamingRun(InferenceSession session)
        {
            OrtAllocator allocator = RequireDeviceAllocator(session);
            Resources resources = CreateResources(session, allocator);
            Binding = resources.Binding;
            RunOptions = resources.RunOptions;
            chunk = resources.Chunk;
            chunkLengths = resources.ChunkLengths;
            spkcache = resources.SpeakerCache;
            spkcacheLengths = resources.SpeakerCacheLengths;
            fifo = resources.Fifo;
            fifoLengths = resources.FifoLengths;
            hostInputs = resources.HostInputs;
            deviceInputs = resources.DeviceInputs;
        }

        internal static SortFormerStreamingRun? TryCreate(InferenceSession session)
        {
            try
            {
                return new SortFormerStreamingRun(session);
            }
            catch (NotSupportedException)
            {
                return null;
            }
        }

        private static Resources CreateResources(InferenceSession session, OrtAllocator allocator)
        {
            OrtIoBinding? binding = null;
            RunOptions? runOptions = null;
            DeviceBoundTensor<float>? chunk = null;
            DeviceBoundTensor<long>? chunkLengths = null;
            DeviceBoundTensor<float>? speakerCache = null;
            DeviceBoundTensor<long>? speakerCacheLengths = null;
            DeviceBoundTensor<float>? fifo = null;
            DeviceBoundTensor<long>? fifoLengths = null;
            try
            {
                binding = session.CreateIoBinding();
                runOptions = new RunOptions();
                chunk = new DeviceBoundTensor<float>(allocator, [1, StreamingFeedFeatureFrames, SortFormerFeatureExtractor.MelBins]);
                chunkLengths = new DeviceBoundTensor<long>(allocator, [1]);
                speakerCache = new DeviceBoundTensor<float>(allocator, [1, StreamingSpeakerCacheFrames, StreamingEmbeddingDimension]);
                speakerCacheLengths = new DeviceBoundTensor<long>(allocator, [1]);
                fifo = new DeviceBoundTensor<float>(allocator, [1, StreamingFifoFrames, StreamingEmbeddingDimension]);
                fifoLengths = new DeviceBoundTensor<long>(allocator, [1]);

                binding.BindInput("chunk", chunk.Device);
                binding.BindInput("chunk_lengths", chunkLengths.Device);
                binding.BindInput("spkcache", speakerCache.Device);
                binding.BindInput("spkcache_lengths", speakerCacheLengths.Device);
                binding.BindInput("fifo", fifo.Device);
                binding.BindInput("fifo_lengths", fifoLengths.Device);

                binding.BindOutputToDevice("spkcache_fifo_chunk_preds", OrtMemoryInfo.DefaultInstance);
                binding.BindOutputToDevice("chunk_pre_encode_embs", OrtMemoryInfo.DefaultInstance);
                binding.BindOutputToDevice("chunk_pre_encode_lengths", OrtMemoryInfo.DefaultInstance);

                return new Resources(
                    binding,
                    runOptions,
                    chunk,
                    chunkLengths,
                    speakerCache,
                    speakerCacheLengths,
                    fifo,
                    fifoLengths,
                    [chunk.Host, chunkLengths.Host, speakerCache.Host, speakerCacheLengths.Host, fifo.Host, fifoLengths.Host],
                    [chunk.Device, chunkLengths.Device, speakerCache.Device, speakerCacheLengths.Device, fifo.Device, fifoLengths.Device]);
            }
            catch
            {
                fifoLengths?.Dispose();
                fifo?.Dispose();
                speakerCacheLengths?.Dispose();
                speakerCache?.Dispose();
                chunkLengths?.Dispose();
                chunk?.Dispose();
                runOptions?.Dispose();
                binding?.Dispose();
                throw;
            }
        }

        private sealed record Resources(
            OrtIoBinding Binding,
            RunOptions RunOptions,
            DeviceBoundTensor<float> Chunk,
            DeviceBoundTensor<long> ChunkLengths,
            DeviceBoundTensor<float> SpeakerCache,
            DeviceBoundTensor<long> SpeakerCacheLengths,
            DeviceBoundTensor<float> Fifo,
            DeviceBoundTensor<long> FifoLengths,
            OrtValue[] HostInputs,
            OrtValue[] DeviceInputs);

        internal OrtIoBinding Binding { get; }

        internal RunOptions RunOptions { get; }

        internal bool TryLoad(
            SortFormerFeatureInputSet features,
            int featureFrameCount,
            SortFormerStreamingState state)
        {
            if (state.SpeakerCacheFrameCount != StreamingSpeakerCacheFrames
                || state.FifoFrameCount != StreamingFifoFrames)
            {
                return false;
            }

            Array.Clear(chunk.Backing);
            features.CopyFramesTo(chunk.Backing, 0, 0, featureFrameCount);
            chunkLengths.Backing[0] = featureFrameCount;
            Array.Copy(
                state.SpeakerCacheEmbeddings,
                spkcache.Backing,
                StreamingSpeakerCacheFrames * StreamingEmbeddingDimension);
            spkcacheLengths.Backing[0] = StreamingSpeakerCacheFrames;
            Array.Copy(
                state.FifoEmbeddings,
                fifo.Backing,
                StreamingFifoFrames * StreamingEmbeddingDimension);
            fifoLengths.Backing[0] = StreamingFifoFrames;
            return true;
        }

        internal void CopyInputsToDevice() =>
            OrtEnv.Instance().CopyTensors(hostInputs, deviceInputs, default);

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

        public void Dispose()
        {
            Binding.Dispose();
            RunOptions.Dispose();
            chunk.Dispose();
            chunkLengths.Dispose();
            spkcache.Dispose();
            spkcacheLengths.Dispose();
            fifo.Dispose();
            fifoLengths.Dispose();
        }
    }

    // Mirrors NeMo SortformerModules.streaming_update (synchronous path): FIFO frames pop into the
    // speaker cache, and an overfull cache is compressed by prediction scores rather than truncated.
    private sealed class SortFormerStreamingState
    {
        private readonly float[] meanSilenceEmbedding = new float[StreamingEmbeddingDimension];
        private int silenceFrameCount;
        private float[] speakerCachePredictions = [];
        private bool speakerCacheCompressed;

        public float[] SpeakerCacheEmbeddings { get; private set; } = [];

        public int SpeakerCacheFrameCount { get; private set; }

        public float[] FifoEmbeddings { get; private set; } = [];

        public int FifoFrameCount { get; private set; }

        public void Update(
            float[] cacheAndFifoPredictions,
            float[] chunkPredictions,
            int speakerCount,
            float[] chunkEmbeddings,
            int chunkEmbeddingFrameCount,
            int chunkFrameCount)
        {
            int previousFifoFrameCount = FifoFrameCount;
            float[] combinedFifoEmbeddings = ConcatenateFrames(
                FifoEmbeddings,
                FifoFrameCount,
                chunkEmbeddings,
                chunkEmbeddingFrameCount,
                StreamingEmbeddingDimension);
            int combinedFifoFrameCount = previousFifoFrameCount + chunkEmbeddingFrameCount;

            if (combinedFifoFrameCount <= StreamingFifoFrames)
            {
                FifoEmbeddings = combinedFifoEmbeddings;
                FifoFrameCount = combinedFifoFrameCount;
                return;
            }

            // FIFO predictions come from this run, so they reflect the current cache context.
            float[] combinedFifoPredictions = ConcatenateFrames(
                SliceFrames(cacheAndFifoPredictions, SpeakerCacheFrameCount, previousFifoFrameCount, speakerCount),
                previousFifoFrameCount,
                chunkPredictions,
                chunkEmbeddingFrameCount,
                speakerCount);

            int popOutFrameCount = Math.Max(
                StreamingSpeakerCacheUpdatePeriodFrames,
                chunkFrameCount - StreamingFifoFrames + previousFifoFrameCount);
            popOutFrameCount = Math.Min(popOutFrameCount, combinedFifoFrameCount);

            float[] popOutEmbeddings = SliceFrames(combinedFifoEmbeddings, 0, popOutFrameCount, StreamingEmbeddingDimension);
            float[] popOutPredictions = SliceFrames(combinedFifoPredictions, 0, popOutFrameCount, speakerCount);
            silenceFrameCount = SortFormerSpeakerCacheCompressor.UpdateSilenceProfile(
                meanSilenceEmbedding,
                silenceFrameCount,
                popOutEmbeddings,
                popOutPredictions,
                popOutFrameCount,
                speakerCount,
                StreamingEmbeddingDimension);

            int remainingFifoFrameCount = combinedFifoFrameCount - popOutFrameCount;
            FifoEmbeddings = SliceFrames(combinedFifoEmbeddings, popOutFrameCount, remainingFifoFrameCount, StreamingEmbeddingDimension);
            FifoFrameCount = remainingFifoFrameCount;

            // Before the first compression, cache predictions are refreshed from this run.
            float[] basePredictions = speakerCacheCompressed
                ? speakerCachePredictions
                : SliceFrames(cacheAndFifoPredictions, 0, SpeakerCacheFrameCount, speakerCount);
            int cacheFrameCount = SpeakerCacheFrameCount + popOutFrameCount;
            float[] cacheEmbeddings = ConcatenateFrames(
                SpeakerCacheEmbeddings,
                SpeakerCacheFrameCount,
                popOutEmbeddings,
                popOutFrameCount,
                StreamingEmbeddingDimension);
            float[] cachePredictions = ConcatenateFrames(
                basePredictions,
                SpeakerCacheFrameCount,
                popOutPredictions,
                popOutFrameCount,
                speakerCount);

            if (cacheFrameCount > StreamingSpeakerCacheFrames)
            {
                (cacheEmbeddings, cachePredictions) = SortFormerSpeakerCacheCompressor.Compress(
                    cacheEmbeddings,
                    cachePredictions,
                    cacheFrameCount,
                    speakerCount,
                    StreamingEmbeddingDimension,
                    StreamingSpeakerCacheFrames,
                    meanSilenceEmbedding);
                cacheFrameCount = StreamingSpeakerCacheFrames;
                speakerCacheCompressed = true;
            }

            SpeakerCacheEmbeddings = cacheEmbeddings;
            SpeakerCacheFrameCount = cacheFrameCount;
            speakerCachePredictions = cachePredictions;
        }
    }

    private sealed record ActiveTurn(
        int SpeakerIndex,
        double StartSeconds,
        double EndSeconds,
        double ConfidenceSum,
        int FrameCount,
        bool HasOverlap);
}
