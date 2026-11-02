using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Pool;

namespace Trackdub.Inference.Onnx.NemotronAsr;

internal sealed class NemotronAsrGreedyDecoder(
    OnnxExecutionSessionFactory.NemotronAsrSessionLease sessionLease,
    NemotronAsrSentencePieceVocab vocab,
    NemotronAsrExportConfig exportConfig)
{
    private const int MaxSymbolsPerStep = 10;

    private readonly NemotronAsrModelConfig config = NemotronAsrModelConfig.FromSessions(
        sessionLease.EncoderSession,
        sessionLease.DecoderJointSession,
        vocab.Count,
        exportConfig);

    // The session's effective execution provider, as a retry-policy hint. Resolved once from the
    // lease's already-known SelectedProvider label instead of re-detecting it per call.
    private readonly ExecutionProviderKind? providerHint = ResolveProviderHint(sessionLease.SelectedProvider);

    public string? DetectedLanguage { get; private set; }

    internal float LastEncodedAbsSum { get; private set; }

    internal int LastEncodedLength { get; private set; }

    // RunWithBindingAndNames pairs these names positionally with OrtIoBinding.GetOutputValues(),
    // which returns only bound outputs, in bind order. Must match the BindOutput* order in
    // BindEncoderInputs exactly.
    private static readonly string[] EncoderBoundOutputNames =
        ["encoded", "encoded_len", "cache_last_channel_len_next", "cache_last_channel_next", "cache_last_time_next"];

    public IReadOnlyList<int> Decode(float[,] mel, long promptIndex, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(mel);
        ResetState(out int lastToken);
        bool encoderHasDeviceAllocator = providerHint is ExecutionProviderKind.TensorRTRtx
            && HasSharedDeviceAllocator(sessionLease.EncoderSession);
        bool decoderHasDeviceAllocator = providerHint is ExecutionProviderKind.TensorRTRtx
            && HasSharedDeviceAllocator(sessionLease.DecoderJointSession);
        bool useDeviceBoundRuns = ShouldUseDeviceBoundRuns(
            providerHint, encoderHasDeviceAllocator, decoderHasDeviceAllocator);
        using var encoderRun = useDeviceBoundRuns
            ? EncoderRun.Create(sessionLease.EncoderSession, config, promptIndex)
            : null;
        using var decoderRun = useDeviceBoundRuns
            ? DecoderRun.Create(sessionLease.DecoderJointSession, config)
            : null;

        int[]? cacheChannelShape = null;
        int[]? cacheTimeShape = null;
        float[]? cacheChannel = null;
        float[]? cacheTime = null;
        float[]? decoderState1 = null;
        float[]? decoderState2 = null;
        int[]? decoderStateShape = null;
        TensorElementType targetsElementType = TensorElementType.Int64;
        TensorElementType targetLengthElementType = TensorElementType.Int64;
        long cacheChannelLength = 0;
        if (!useDeviceBoundRuns)
        {
            IReadOnlyDictionary<string, NodeMetadata> encoderMetadata = sessionLease.EncoderSession.InputMetadata;
            cacheChannelShape = ResolveCacheShape(encoderMetadata, "cache_last_channel",
                [config.NumEncoderLayers, 1, config.LeftContext, config.HiddenDim]);
            cacheTimeShape = ResolveCacheShape(encoderMetadata, "cache_last_time",
                [config.NumEncoderLayers, 1, config.HiddenDim, config.ConvContext]);
            cacheChannel = new float[GetElementCount(cacheChannelShape)];
            cacheTime = new float[GetElementCount(cacheTimeShape)];

            // Hoisted out of the per-symbol decode loop: session metadata and the
            // resolved state shape cannot change while this lease is held.
            IReadOnlyDictionary<string, NodeMetadata> decoderMetadata = sessionLease.DecoderJointSession.InputMetadata;
            decoderStateShape = ResolveCacheShape(
                decoderMetadata,
                "input_states_1",
                [config.DecoderLstmLayers, 1, config.DecoderLstmDim]);
            decoderState1 = new float[GetElementCount(decoderStateShape)];
            decoderState2 = new float[GetElementCount(decoderStateShape)];
            targetsElementType = ResolveTokenElementType(decoderMetadata, "targets");
            targetLengthElementType = ResolveTokenElementType(decoderMetadata, "target_length");
        }

        int totalFrames = mel.GetLength(1);
        var allTokens = new List<int>();
        int chunkIndex = 0;

        // BuildChunk is a pure function of `mel` and the frame offset, and `mel` is fully
        // materialized before this loop, so the next chunk's features can be assembled on a
        // worker thread while the current chunk still occupies the GPU.
        Task<float[]>? prefetchedChunk = null;

        for (int frameOffset = 0; frameOffset < totalFrames; frameOffset += NemotronAsrMelFeatureExtractor.ChunkFrames)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int mainFrameCount = Math.Min(NemotronAsrMelFeatureExtractor.ChunkFrames, totalFrames - frameOffset);
            float[] chunkData = prefetchedChunk is null
                ? NemotronAsrMelFeatureExtractor.BuildChunk(mel, frameOffset, mainFrameCount, includePreEncodeCache: chunkIndex > 0)
                : prefetchedChunk.GetAwaiter().GetResult();

            int nextFrameOffset = frameOffset + NemotronAsrMelFeatureExtractor.ChunkFrames;
            if (nextFrameOffset < totalFrames)
            {
                int nextFrameCount = Math.Min(
                    NemotronAsrMelFeatureExtractor.ChunkFrames,
                    totalFrames - nextFrameOffset);
                // A prefetched chunk is never the first one, so it always carries the pre-encode cache.
                prefetchedChunk = Task.Run(() => NemotronAsrMelFeatureExtractor.BuildChunk(
                    mel,
                    nextFrameOffset,
                    nextFrameCount,
                    includePreEncodeCache: true));
            }
            else
            {
                prefetchedChunk = null;
            }

            int chunkLength = NemotronAsrMelFeatureExtractor.PreEncodeCacheFrames + mainFrameCount;
            if (useDeviceBoundRuns)
            {
                // The device-bound path is restricted to sessions with an effective TRT-RTX
                // provider and shared device allocators. Its pinned inputs keep stable addresses
                // for CUDA graph replay; CPU and DML use managed NamedOnnxValue inputs below.
                encoderRun!.Load(chunkData, chunkLength);
                encoderRun.CopyInputsToDevice();
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults =
                    sessionLease.EncoderSession.RunWithBindingAndNamesRetry(
                        encoderRun.RunOptions,
                        encoderRun.Binding,
                        EncoderBoundOutputNames,
                        cancellationToken: cancellationToken,
                        provider: providerHint);

                Tensor<float> encoded = GetTensor<float>(encoderResults, "encoded");
                int encodedLength = (int)GetTensor<long>(encoderResults, "encoded_len").FirstOrDefault();
                Tensor<float> cacheChannelNext = GetTensor<float>(encoderResults, "cache_last_channel_next");
                Tensor<float> cacheTimeNext = GetTensor<float>(encoderResults, "cache_last_time_next");
                long cacheLengthNext = GetTensor<long>(encoderResults, "cache_last_channel_len_next").FirstOrDefault();
                LastEncodedAbsSum = SumAbs(encoded);
                LastEncodedLength = encodedLength;

                DecodeEncoderFrames(
                    encoded, encodedLength, allTokens, decoderRun!, ref lastToken, cancellationToken);
                encoderRun.CommitCaches(cacheChannelNext, cacheTimeNext, cacheLengthNext);
            }
            else
            {
                RunManagedEncoderAndDecodeFrames(
                    chunkData,
                    chunkLength,
                    promptIndex,
                    allTokens,
                    cacheChannel!,
                    cacheChannelShape!,
                    cacheTime!,
                    cacheTimeShape!,
                    ref cacheChannelLength,
                    decoderState1!,
                    decoderState2!,
                    decoderStateShape!,
                    targetsElementType,
                    targetLengthElementType,
                    ref lastToken,
                    cancellationToken);
            }

            chunkIndex++;
        }

        return allTokens;
    }

    public string DecodeText(IReadOnlyList<int> tokens)
    {
        var visibleTokens = new List<int>(tokens.Count);
        foreach (int token in tokens)
        {
            if (token < 0 || token >= vocab.Count)
            {
                continue;
            }

            if (vocab.LanguageTagIds.Contains(token))
            {
                DetectedLanguage ??= NemotronAsrLanguagePrompts.TryExtractLanguageTag(vocab.DecodeSingle(token));
                continue;
            }

            visibleTokens.Add(token);
        }

        return vocab.Decode(visibleTokens).Trim();
    }

    private void DecodeEncoderFrames(
        Tensor<float> encoded,
        int encodedLength,
        List<int> tokens,
        DecoderRun decoderRun,
        ref int lastToken,
        CancellationToken cancellationToken)
    {
        NemotronAsrEncodedTensorLayout.EncodedLayout layout = NemotronAsrEncodedTensorLayout.Resolve(
            encoded.Dimensions,
            encodedLength,
            config.HiddenDim);
        for (int frameIndex = 0; frameIndex < layout.AvailableFrames; frameIndex++)
        {
            DenseTensor<float> frame = NemotronAsrEncodedTensorLayout.SliceFrame(encoded, layout, frameIndex);
            for (int symbol = 0; symbol < MaxSymbolsPerStep; symbol++)
            {
                decoderRun.Load(frame, lastToken);
                decoderRun.CopyInputsToDevice();
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults =
                    sessionLease.DecoderJointSession.RunWithBindingAndNamesRetry(
                        decoderRun.RunOptions,
                        decoderRun.Binding,
                        DecoderBoundOutputNames,
                        cancellationToken: cancellationToken,
                        provider: providerHint);

                Tensor<float> logits = GetTensor<float>(decoderResults, "outputs");
                int nextToken = ArgMax(logits);
                if (nextToken == config.BlankId)
                {
                    // RNNT: blank consumes no token, so the predictor state must not advance.
                    break;
                }

                decoderRun.CommitStates(
                    GetTensor<float>(decoderResults, "output_states_1"),
                    GetTensor<float>(decoderResults, "output_states_2"));
                lastToken = nextToken;
                tokens.Add(nextToken);
            }
        }
    }

    private void RunManagedEncoderAndDecodeFrames(
        float[] chunkData,
        int chunkLength,
        long promptIndex,
        List<int> tokens,
        float[] cacheChannel,
        int[] cacheChannelShape,
        float[] cacheTime,
        int[] cacheTimeShape,
        ref long cacheChannelLength,
        float[] decoderState1,
        float[] decoderState2,
        int[] decoderStateShape,
        TensorElementType targetsElementType,
        TensorElementType targetLengthElementType,
        ref int lastToken,
        CancellationToken cancellationToken)
    {
        var encoderInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(
                "processed_signal",
                new DenseTensor<float>(chunkData, Array.ConvertAll(ProcessedSignalShape, static d => checked((int)d)))),
            NamedOnnxValue.CreateFromTensor("processed_signal_length", new DenseTensor<long>(new long[] { chunkLength }, [1])),
            NamedOnnxValue.CreateFromTensor("cache_last_channel", new DenseTensor<float>(cacheChannel, cacheChannelShape)),
            NamedOnnxValue.CreateFromTensor("cache_last_time", new DenseTensor<float>(cacheTime, cacheTimeShape)),
            NamedOnnxValue.CreateFromTensor("cache_last_channel_len", new DenseTensor<long>(new long[] { cacheChannelLength }, [1]))
        };
        if (config.HasPromptInput)
        {
            encoderInputs.Add(NamedOnnxValue.CreateFromTensor(
                "prompt_index", new DenseTensor<long>(new long[] { promptIndex }, [1])));
        }

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults =
            sessionLease.EncoderSession.RunWithRetry(
                encoderInputs,
                cancellationToken: cancellationToken,
                provider: providerHint);
        Tensor<float> encoded = GetTensor<float>(encoderResults, "encoded");
        int encodedLength = checked((int)GetTensor<long>(encoderResults, "encoded_len").FirstOrDefault());
        LastEncodedAbsSum = SumAbs(encoded);
        LastEncodedLength = encodedLength;

        DecodeEncoderFramesManaged(
            encoded,
            encodedLength,
            tokens,
            decoderState1,
            decoderState2,
            decoderStateShape,
            targetsElementType,
            targetLengthElementType,
            ref lastToken,
            cancellationToken);

        CopyTensor(GetTensor<float>(encoderResults, "cache_last_channel_next"), cacheChannel);
        CopyTensor(GetTensor<float>(encoderResults, "cache_last_time_next"), cacheTime);
        cacheChannelLength = GetTensor<long>(encoderResults, "cache_last_channel_len_next").FirstOrDefault();
    }

    private void DecodeEncoderFramesManaged(
        Tensor<float> encoded,
        int encodedLength,
        List<int> tokens,
        float[] decoderState1,
        float[] decoderState2,
        int[] decoderStateShape,
        TensorElementType targetsElementType,
        TensorElementType targetLengthElementType,
        ref int lastToken,
        CancellationToken cancellationToken)
    {
        NemotronAsrEncodedTensorLayout.EncodedLayout layout = NemotronAsrEncodedTensorLayout.Resolve(
            encoded.Dimensions,
            encodedLength,
            config.HiddenDim);
        for (int frameIndex = 0; frameIndex < layout.AvailableFrames; frameIndex++)
        {
            DenseTensor<float> frame = NemotronAsrEncodedTensorLayout.SliceFrame(encoded, layout, frameIndex);
            for (int symbol = 0; symbol < MaxSymbolsPerStep; symbol++)
            {
                int nextToken = RunManagedDecoder(
                    frame, lastToken, decoderState1, decoderState2, decoderStateShape,
                    targetsElementType, targetLengthElementType, cancellationToken);
                if (nextToken == config.BlankId)
                {
                    // RNNT: blank consumes no token, so the predictor state must not advance.
                    break;
                }

                lastToken = nextToken;
                tokens.Add(nextToken);
            }
        }
    }

    private int RunManagedDecoder(
        DenseTensor<float> frame,
        int lastToken,
        float[] state1,
        float[] state2,
        int[] stateShape,
        TensorElementType targetsElementType,
        TensorElementType targetLengthElementType,
        CancellationToken cancellationToken)
    {
        var decoderInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_outputs", frame),
            CreateTokenInput(targetsElementType, "targets", lastToken, [1, 1]),
            CreateTokenInput(targetLengthElementType, "target_length", 1, [1]),
            NamedOnnxValue.CreateFromTensor("input_states_1", new DenseTensor<float>(state1, stateShape)),
            NamedOnnxValue.CreateFromTensor("input_states_2", new DenseTensor<float>(state2, stateShape))
        };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults =
            sessionLease.DecoderJointSession.RunWithRetry(
                decoderInputs,
                cancellationToken: cancellationToken,
                provider: providerHint);
        int nextToken = ArgMax(GetTensor<float>(decoderResults, "outputs"));
        if (nextToken != config.BlankId)
        {
            CopyTensor(GetTensor<float>(decoderResults, "output_states_1"), state1);
            CopyTensor(GetTensor<float>(decoderResults, "output_states_2"), state2);
        }

        return nextToken;
    }

    private static TensorElementType ResolveTokenElementType(
        IReadOnlyDictionary<string, NodeMetadata> metadata,
        string name) =>
        metadata.TryGetValue(name, out NodeMetadata? nodeMetadata)
            ? nodeMetadata.ElementDataType
            : TensorElementType.Int64;

    private static NamedOnnxValue CreateTokenInput(
        TensorElementType elementType,
        string name,
        long value,
        int[] dimensions)
    {
        return elementType switch
        {
            TensorElementType.Int32 => NamedOnnxValue.CreateFromTensor(
                name, new DenseTensor<int>(new int[] { checked((int)value) }, dimensions)),
            TensorElementType.Int64 => NamedOnnxValue.CreateFromTensor(
                name, new DenseTensor<long>(new long[] { value }, dimensions)),
            _ => throw new NotSupportedException(
                $"Nemotron decoder input '{name}' uses unsupported token tensor element type '{elementType}'.")
        };
    }

    private static readonly long[] ProcessedSignalShape =
        [1, NemotronAsrMelFeatureExtractor.MelBins, NemotronAsrMelFeatureExtractor.ChunkInputFrames];
    private static readonly long[] ScalarShape = [1];

    // Same positional-pairing rule as EncoderBoundOutputNames: order matches the decoder's
    // BindOutputToDevice order in DecoderRun.
    private static readonly string[] DecoderBoundOutputNames =
        ["outputs", "prednet_lengths", "output_states_1", "output_states_2"];

    private void ResetState(out int lastToken)
    {
        // NeMo RNNT seeds the predictor with blank_id from config.json — not token 0.
        lastToken = config.BlankId;
        DetectedLanguage = null;
    }

    /// <summary>
    /// Persistent encoder binding: every input and cache output is a pinned host OrtValue whose
    /// address never changes across the chunk loop, which is the CUDA-graph replay precondition.
    /// The streaming caches use separate input/output buffers (ORT must not alias a kernel's own
    /// input and output within one run) with a post-run copy-back replacing the old ping-pong swap.
    /// </summary>
    private sealed class EncoderRun : IDisposable
    {
        private readonly DeviceBoundTensor<float> signal;
        private readonly DeviceBoundTensor<long> signalLength;
        private readonly DeviceBoundTensor<float> cacheChannelIn;
        private readonly DeviceBoundTensor<float> cacheTimeIn;
        private readonly DeviceBoundTensor<long> cacheChannelLength;
        private readonly DeviceBoundTensor<long>? promptIndex;
        private readonly OrtValue[] hostInputs;
        private readonly OrtValue[] deviceInputs;

        private EncoderRun(
            InferenceSession session,
            OrtAllocator allocator,
            NemotronAsrModelConfig config,
            long promptIndexValue,
            int[] cacheChannelShape,
            int[] cacheTimeShape)
        {
            Binding = session.CreateIoBinding();
            RunOptions = new RunOptions();
            signal = new DeviceBoundTensor<float>(allocator, ProcessedSignalShape);
            signalLength = new DeviceBoundTensor<long>(allocator, ScalarShape);
            long[] chShape = Array.ConvertAll(cacheChannelShape, static d => (long)d);
            long[] tmShape = Array.ConvertAll(cacheTimeShape, static d => (long)d);
            cacheChannelIn = new DeviceBoundTensor<float>(allocator, chShape);
            cacheTimeIn = new DeviceBoundTensor<float>(allocator, tmShape);
            cacheChannelLength = new DeviceBoundTensor<long>(allocator, ScalarShape);
            if (config.HasPromptInput)
            {
                promptIndex = new DeviceBoundTensor<long>(allocator, ScalarShape);
                promptIndex.Backing[0] = promptIndexValue;
            }

            Binding.BindInput("processed_signal", signal.Device);
            Binding.BindInput("processed_signal_length", signalLength.Device);
            Binding.BindInput("cache_last_channel", cacheChannelIn.Device);
            Binding.BindInput("cache_last_time", cacheTimeIn.Device);
            Binding.BindInput("cache_last_channel_len", cacheChannelLength.Device);
            if (promptIndex is not null)
            {
                Binding.BindInput("prompt_index", promptIndex.Device);
            }

            // Output-bind order must match EncoderBoundOutputNames. Host-bound BindOutput
            // targets stay zeroed on TRT-RTX, so every output is device-bound and fetched.
            Binding.BindOutputToDevice("encoded", OrtMemoryInfo.DefaultInstance);
            Binding.BindOutputToDevice("encoded_len", OrtMemoryInfo.DefaultInstance);
            Binding.BindOutputToDevice("cache_last_channel_len_next", OrtMemoryInfo.DefaultInstance);
            Binding.BindOutputToDevice("cache_last_channel_next", OrtMemoryInfo.DefaultInstance);
            Binding.BindOutputToDevice("cache_last_time_next", OrtMemoryInfo.DefaultInstance);

            hostInputs = promptIndex is null
                ? [signal.Host, signalLength.Host, cacheChannelIn.Host, cacheTimeIn.Host, cacheChannelLength.Host]
                : [signal.Host, signalLength.Host, cacheChannelIn.Host, cacheTimeIn.Host, cacheChannelLength.Host, promptIndex.Host];
            deviceInputs = promptIndex is null
                ? [signal.Device, signalLength.Device, cacheChannelIn.Device, cacheTimeIn.Device, cacheChannelLength.Device]
                : [signal.Device, signalLength.Device, cacheChannelIn.Device, cacheTimeIn.Device, cacheChannelLength.Device, promptIndex.Device];
        }

        internal OrtIoBinding Binding { get; }

        internal RunOptions RunOptions { get; }

        internal static EncoderRun Create(
            InferenceSession session, NemotronAsrModelConfig config, long promptIndexValue)
        {
            var meta = session.InputMetadata;
            int[] chShape = ResolveCacheShape(meta, "cache_last_channel",
                [config.NumEncoderLayers, 1, config.LeftContext, config.HiddenDim]);
            int[] tmShape = ResolveCacheShape(meta, "cache_last_time",
                [config.NumEncoderLayers, 1, config.HiddenDim, config.ConvContext]);
            return new EncoderRun(
                session, RequireDeviceAllocator(session), config, promptIndexValue, chShape, tmShape);
        }

        internal void Load(float[] chunkData, int chunkLength)
        {
            if (chunkData.Length > signal.Backing.Length)
            {
                throw new InvalidOperationException(
                    $"Nemotron encoder chunk {chunkData.Length} floats exceeds pinned buffer {signal.Backing.Length}.");
            }

            chunkData.CopyTo(signal.Backing, 0);
            if (signal.Backing.Length > chunkData.Length)
            {
                Array.Clear(signal.Backing, chunkData.Length, signal.Backing.Length - chunkData.Length);
            }

            signalLength.Backing[0] = chunkLength;
        }

        internal void CopyInputsToDevice() =>
            OrtEnv.Instance().CopyTensors(hostInputs, deviceInputs, default);

        internal void CommitCaches(Tensor<float> cacheChannelNext, Tensor<float> cacheTimeNext, long cacheLengthNext)
        {
            CopyTensor(cacheChannelNext, cacheChannelIn.Backing);
            CopyTensor(cacheTimeNext, cacheTimeIn.Backing);
            cacheChannelLength.Backing[0] = cacheLengthNext;
        }

        public void Dispose()
        {
            Binding.Dispose();
            RunOptions.Dispose();
            signal.Dispose();
            signalLength.Dispose();
            cacheChannelIn.Dispose();
            cacheTimeIn.Dispose();
            cacheChannelLength.Dispose();
            promptIndex?.Dispose();
        }
    }

    /// <summary>
    /// Persistent decoder_joint binding: encoder frame, token, and LSTM-state inputs all sit in
    /// pinned host buffers; the predictor states are copied back into their input buffers only
    /// when a non-blank token advances the RNNT predictor.
    /// </summary>
    private sealed class DecoderRun : IDisposable
    {
        private readonly DeviceBoundTensor<float> encoderOutputs;
        private readonly DeviceBoundTensor<long>? targets64;
        private readonly DeviceBoundTensor<int>? targets32;
        private readonly DeviceBoundTensor<long>? targetLength64;
        private readonly DeviceBoundTensor<int>? targetLength32;
        private readonly DeviceBoundTensor<float> state1In;
        private readonly DeviceBoundTensor<float> state2In;
        private readonly OrtValue[] hostInputs;
        private readonly OrtValue[] deviceInputs;

        private DecoderRun(
            InferenceSession session,
            OrtAllocator allocator,
            int encoderHiddenDim,
            int decoderLayers,
            int decoderDim,
            bool int32Tokens)
        {
            Binding = session.CreateIoBinding();
            RunOptions = new RunOptions();
            encoderOutputs = new DeviceBoundTensor<float>(allocator, [1, encoderHiddenDim, 1]);
            long[] stateShape = [decoderLayers, 1, decoderDim];
            state1In = new DeviceBoundTensor<float>(allocator, stateShape);
            state2In = new DeviceBoundTensor<float>(allocator, stateShape);
            if (int32Tokens)
            {
                targets32 = new DeviceBoundTensor<int>(allocator, [1, 1]);
                targetLength32 = new DeviceBoundTensor<int>(allocator, [1]);
                targetLength32.Backing[0] = 1;
            }
            else
            {
                targets64 = new DeviceBoundTensor<long>(allocator, [1, 1]);
                targetLength64 = new DeviceBoundTensor<long>(allocator, [1]);
                targetLength64.Backing[0] = 1;
            }

            Binding.BindInput("encoder_outputs", encoderOutputs.Device);
            Binding.BindInput("targets", (targets64?.Device ?? targets32!.Device));
            Binding.BindInput("target_length", (targetLength64?.Device ?? targetLength32!.Device));
            Binding.BindInput("input_states_1", state1In.Device);
            Binding.BindInput("input_states_2", state2In.Device);

            // Output-bind order must match DecoderBoundOutputNames. Host-bound BindOutput
            // targets stay zeroed on TRT-RTX, so every output is device-bound and fetched.
            Binding.BindOutputToDevice("outputs", OrtMemoryInfo.DefaultInstance);
            Binding.BindOutputToDevice("prednet_lengths", OrtMemoryInfo.DefaultInstance);
            Binding.BindOutputToDevice("output_states_1", OrtMemoryInfo.DefaultInstance);
            Binding.BindOutputToDevice("output_states_2", OrtMemoryInfo.DefaultInstance);

            OrtValue targetsDevice = targets64?.Device ?? targets32!.Device;
            OrtValue targetLengthDevice = targetLength64?.Device ?? targetLength32!.Device;
            OrtValue targetsHost = targets64?.Host ?? targets32!.Host;
            OrtValue targetLengthHost = targetLength64?.Host ?? targetLength32!.Host;
            hostInputs = [encoderOutputs.Host, targetsHost, targetLengthHost, state1In.Host, state2In.Host];
            deviceInputs = [encoderOutputs.Device, targetsDevice, targetLengthDevice, state1In.Device, state2In.Device];
        }

        internal OrtIoBinding Binding { get; }

        internal RunOptions RunOptions { get; }

        internal static DecoderRun Create(InferenceSession session, NemotronAsrModelConfig config)
        {
            var meta = session.InputMetadata;
            bool int32Tokens =
                (meta.TryGetValue("targets", out NodeMetadata? targetsMeta)
                    && targetsMeta.ElementDataType is TensorElementType.Int32)
                || (meta.TryGetValue("target_length", out NodeMetadata? lengthMeta)
                    && lengthMeta.ElementDataType is TensorElementType.Int32);
            return new DecoderRun(
                session, RequireDeviceAllocator(session),
                config.HiddenDim, config.DecoderLstmLayers, config.DecoderLstmDim, int32Tokens);
        }

        internal void Load(DenseTensor<float> frame, int lastToken)
        {
            int index = 0;
            foreach (float value in frame)
            {
                encoderOutputs.Backing[index++] = value;
            }

            if (targets64 is not null)
            {
                targets64.Backing[0] = lastToken;
            }
            else
            {
                targets32!.Backing[0] = lastToken;
            }
        }

        internal void CopyInputsToDevice() =>
            OrtEnv.Instance().CopyTensors(hostInputs, deviceInputs, default);

        internal void CommitStates(Tensor<float> state1Next, Tensor<float> state2Next)
        {
            CopyTensor(state1Next, state1In.Backing);
            CopyTensor(state2Next, state2In.Backing);
        }

        public void Dispose()
        {
            Binding.Dispose();
            RunOptions.Dispose();
            encoderOutputs.Dispose();
            targets64?.Dispose();
            targets32?.Dispose();
            targetLength64?.Dispose();
            targetLength32?.Dispose();
            state1In.Dispose();
            state2In.Dispose();
        }
    }

    // sessionLease.SelectedProvider is the formatted label of the *effective* provider the
    // session ended up on (post TRT-init-fallback), which is exactly what the retry policy needs
    // to know. Only TensorRTRtx currently changes retry behavior; every other label maps to null
    // (the retry policy's permissive default) rather than guessing at the rest of the enum.
    private static ExecutionProviderKind? ResolveProviderHint(string selectedProviderLabel) =>
        string.Equals(
            selectedProviderLabel,
            ProviderNames.FormatProviderLabel(ExecutionProviderKind.TensorRTRtx),
            StringComparison.Ordinal)
            ? ExecutionProviderKind.TensorRTRtx
            : null;

    private static int[] ResolveCacheShape(
        IReadOnlyDictionary<string, NodeMetadata> meta,
        string name,
        int[] fallback)
    {
        if (meta.TryGetValue(name, out NodeMetadata? m) && m.Dimensions.Length == fallback.Length)
        {
            var dims = new int[fallback.Length];
            for (int i = 0; i < dims.Length; i++)
            {
                dims[i] = m.Dimensions[i] > 0 ? m.Dimensions[i] : fallback[i];
            }
            return dims;
        }
        return (int[])fallback.Clone();
    }

    private static int GetElementCount(IEnumerable<int> dimensions) =>
        dimensions.Aggregate(1, static (product, dimension) => checked(product * dimension));

    internal static bool ShouldUseDeviceBoundRuns(
        ExecutionProviderKind? provider,
        bool encoderHasDeviceAllocator,
        bool decoderHasDeviceAllocator) =>
        provider is ExecutionProviderKind.TensorRTRtx
        && encoderHasDeviceAllocator
        && decoderHasDeviceAllocator;

    private static bool HasSharedDeviceAllocator(InferenceSession session)
    {
        OrtMemoryInfo? deviceMemoryInfo = session.GetMemoryInfosForInputs()
            .FirstOrDefault(static memoryInfo => !memoryInfo.Equals(OrtMemoryInfo.DefaultInstance));
        if (deviceMemoryInfo is null)
        {
            return false;
        }

        OrtAllocator? allocator = OrtEnv.Instance().GetSharedAllocator(deviceMemoryInfo);
        return allocator is not null && !allocator.IsInvalid;
    }

    private static Tensor<T> GetTensor<T>(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
        string name)
        where T : unmanaged =>
        results.Single(result => string.Equals(result.Name, name, StringComparison.Ordinal)).AsTensor<T>();

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

    private static void CopyTensor<T>(Tensor<T> source, T[] destination) where T : unmanaged
    {
        int index = 0;
        foreach (T value in source)
        {
            destination[index++] = value;
        }
    }

    /// <summary>Span loop instead of a whole-tensor LINQ Select/Sum over the encoder output.</summary>
    internal static float SumAbs(Tensor<float> encoded)
    {
        float sum = 0f;
        if (encoded is DenseTensor<float> dense)
        {
            foreach (float value in dense.Buffer.Span)
            {
                sum += MathF.Abs(value);
            }

            return sum;
        }

        foreach (float value in encoded)
        {
            sum += MathF.Abs(value);
        }

        return sum;
    }

    private static int ArgMax(Tensor<float> logits)
    {
        int bestIndex = 0;
        float bestValue = float.NegativeInfinity;
        int index = 0;
        foreach (float value in logits)
        {
            if (value > bestValue)
            {
                bestValue = value;
                bestIndex = index;
            }

            index++;
        }

        return bestIndex;
    }

    private sealed record NemotronAsrModelConfig(
        int NumEncoderLayers,
        int HiddenDim,
        int LeftContext,
        int ConvContext,
        int DecoderLstmDim,
        int DecoderLstmLayers,
        int BlankId,
        bool HasPromptInput)
    {
        public static NemotronAsrModelConfig FromSessions(
            InferenceSession encoderSession,
            InferenceSession decoderJointSession,
            int vocabSize,
            NemotronAsrExportConfig exportConfig)
        {
            int numEncoderLayers = 24;
            int hiddenDim = 1024;
            int leftContext = 56;
            int convContext = 8;
            bool hasPromptInput = encoderSession.InputMetadata.ContainsKey("prompt_index");

            if (encoderSession.InputMetadata.TryGetValue("cache_last_channel", out NodeMetadata? channelMetadata) &&
                channelMetadata.Dimensions.Length >= 4)
            {
                numEncoderLayers = PositiveOrDefault(channelMetadata.Dimensions[0], numEncoderLayers);
                leftContext = PositiveOrDefault(channelMetadata.Dimensions[2], leftContext);
                hiddenDim = PositiveOrDefault(channelMetadata.Dimensions[3], hiddenDim);
            }

            if (encoderSession.InputMetadata.TryGetValue("cache_last_time", out NodeMetadata? timeMetadata) &&
                timeMetadata.Dimensions.Length >= 4)
            {
                convContext = PositiveOrDefault(timeMetadata.Dimensions[3], convContext);
            }

            int decoderLayers = 2;
            int decoderDim = 640;
            if (decoderJointSession.InputMetadata.TryGetValue("input_states_1", out NodeMetadata? stateMetadata) &&
                stateMetadata.Dimensions.Length >= 3)
            {
                decoderLayers = PositiveOrDefault(stateMetadata.Dimensions[0], decoderLayers);
                decoderDim = PositiveOrDefault(stateMetadata.Dimensions[2], decoderDim);
            }

            // blank_id comes from the export config. This package uses blank_id == vocab_size
            // (joint logits = vocab_size + 1); vocab_size - 1 is a real SentencePiece piece.
            int blankId = exportConfig.BlankId;
            if (blankId < 0)
            {
                blankId = exportConfig.VocabSize > 0 ? exportConfig.VocabSize : checked(vocabSize);
            }

            return new NemotronAsrModelConfig(
                numEncoderLayers,
                hiddenDim,
                leftContext,
                convContext,
                decoderDim,
                decoderLayers,
                blankId,
                hasPromptInput);
        }

        private static int PositiveOrDefault(int value, int fallback) =>
            value > 0 ? value : fallback;
    }
}
