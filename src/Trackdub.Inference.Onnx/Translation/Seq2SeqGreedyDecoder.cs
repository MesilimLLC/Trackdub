using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Pool;

namespace Trackdub.Inference.Onnx.Translation;

/// <summary>
/// Greedy decoding for Optimum-exported encoder-decoder translation models.
/// </summary>
/// <remarks>
/// A merged decoder (<c>use_cache_branch</c> input plus <c>present.*</c> outputs) is decoded with its
/// key/value cache: step 0 runs the no-cache branch over the start token and yields the self-attention
/// and cross-attention caches; every later step feeds only the newest token. Decoders without a cache
/// re-run the whole generated prefix every step, which is quadratic in the output length.
/// <para>
/// Static-shape decoding serves TensorRT RTX, which JIT-compiles kernels for every new input shape: the
/// prefix is re-run padded to a length bucket and the logits are read at the last real position.
/// Causal self-attention keeps the padding from influencing earlier positions, so the result matches
/// unpadded decoding, while the engine only ever sees a handful of shapes.
/// </para>
/// <para>
/// A positive <c>noRepeatNgramSize</c> blocks any token that would repeat an n-gram already generated
/// (Hugging Face's <c>no_repeat_ngram_size</c>). Greedy decoding has no other way out of a loop: on
/// disfluent input such as "uh ... uh" MADLAD can emit one token until <c>maxSteps</c>.
/// </para>
/// </remarks>
internal static class Seq2SeqGreedyDecoder
{
    private const string UseCacheBranchInput = "use_cache_branch";
    private const string PastPrefix = "past_key_values.";
    private const string PresentPrefix = "present.";

    private static readonly int[] StaticShapeBuckets = [16, 32, 64, 128, 256, 512];
    private static readonly string[] LogitsOutput = ["logits"];

    internal static bool SupportsKeyValueCache(InferenceSession decoderSession) =>
        decoderSession.InputMetadata.ContainsKey(UseCacheBranchInput) &&
        decoderSession.InputMetadata.Keys.Any(static name => name.StartsWith(PastPrefix, StringComparison.Ordinal)) &&
        decoderSession.OutputMetadata.Keys.Any(static name => name.StartsWith(PresentPrefix, StringComparison.Ordinal));

    /// <summary>Smallest static-shape bucket holding <paramref name="length"/>; longer inputs keep their length.</summary>
    internal static int BucketLength(int length)
    {
        foreach (int bucket in StaticShapeBuckets)
        {
            if (length <= bucket)
            {
                return bucket;
            }
        }

        return length;
    }

    internal static List<long> Decode(
        InferenceSession decoderSession,
        Tensor<float> encoderHiddenStates,
        long[] encoderAttentionMask,
        long decoderStartTokenId,
        long endOfSentenceTokenId,
        int padTokenId,
        int maxSteps,
        ExecutionProviderKind? provider,
        CancellationToken cancellationToken,
        bool staticShapes = false,
        int noRepeatNgramSize = 0) =>
        !staticShapes && SupportsKeyValueCache(decoderSession)
            ? DecodeWithCache(decoderSession, encoderHiddenStates, encoderAttentionMask, decoderStartTokenId,
                endOfSentenceTokenId, padTokenId, maxSteps, provider, cancellationToken, noRepeatNgramSize)
            : DecodeWithoutCache(decoderSession, encoderHiddenStates, encoderAttentionMask, decoderStartTokenId,
                endOfSentenceTokenId, padTokenId, maxSteps, provider, cancellationToken, staticShapes, noRepeatNgramSize);

    private static List<long> DecodeWithCache(
        InferenceSession decoderSession,
        Tensor<float> encoderHiddenStates,
        long[] encoderAttentionMask,
        long decoderStartTokenId,
        long endOfSentenceTokenId,
        int padTokenId,
        int maxSteps,
        ExecutionProviderKind? provider,
        CancellationToken cancellationToken,
        int noRepeatNgramSize)
    {
        var generated = new List<long>();
        var maskTensor = new DenseTensor<long>(encoderAttentionMask, [1, encoderAttentionMask.Length]);

        // Outputs of the previous step stay alive until the next run has consumed them.
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue>? previous = null;
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue>? firstStep = null;
        try
        {
            long nextInputToken = decoderStartTokenId;
            for (int step = 0; step < maxSteps; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                bool useCache = step > 0;
                var inputs = new List<NamedOnnxValue>(decoderSession.InputMetadata.Count);
                foreach ((string name, NodeMetadata metadata) in decoderSession.InputMetadata)
                {
                    inputs.Add(name switch
                    {
                        "input_ids" => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(new[] { nextInputToken }, [1, 1])),
                        "encoder_hidden_states" => NamedOnnxValue.CreateFromTensor(name, encoderHiddenStates),
                        "encoder_attention_mask" or "attention_mask" => NamedOnnxValue.CreateFromTensor(name, maskTensor),
                        UseCacheBranchInput => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<bool>(new[] { useCache }, [1])),
                        _ when name.StartsWith(PastPrefix, StringComparison.Ordinal) =>
                            NamedOnnxValue.CreateFromTensor(name, useCache
                                ? ResolvePast(name, previous!, firstStep!)
                                : CreatePlaceholderPast(metadata)),
                        _ => throw new NotSupportedException($"Decoder input '{name}' is not supported.")
                    });
                }

                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs =
                    decoderSession.RunWithRetry(inputs, cancellationToken: cancellationToken, provider: provider);
                if (step == 0)
                {
                    firstStep = outputs;
                }
                else if (!ReferenceEquals(previous, firstStep))
                {
                    previous?.Dispose();
                }

                previous = outputs;
                Tensor<float> logits = outputs.Single(static value => value.Name == "logits").AsTensor<float>();
                int nextToken = ArgMaxAtPosition(
                    logits,
                    logits.Dimensions[1] - 1,
                    padTokenId,
                    BannedNgramContinuations(generated, noRepeatNgramSize));
                if (nextToken < 0 || nextToken == endOfSentenceTokenId)
                {
                    break;
                }

                generated.Add(nextToken);
                nextInputToken = nextToken;
            }
        }
        finally
        {
            if (!ReferenceEquals(previous, firstStep))
            {
                previous?.Dispose();
            }

            firstStep?.Dispose();
        }

        return generated;
    }

    // Self-attention cache comes from the previous step; the cross-attention cache is computed once,
    // by the no-cache branch at step 0, and reused for every later step.
    private static Tensor<float> ResolvePast(
        string pastName,
        IReadOnlyCollection<DisposableNamedOnnxValue> previous,
        IReadOnlyCollection<DisposableNamedOnnxValue> firstStep)
    {
        string presentName = string.Concat(PresentPrefix, pastName.AsSpan(PastPrefix.Length));
        IReadOnlyCollection<DisposableNamedOnnxValue> source =
            pastName.Contains(".encoder.", StringComparison.Ordinal) ? firstStep : previous;
        return source.Single(value => value.Name == presentName).AsTensor<float>();
    }

    // The no-cache branch ignores past inputs, but the graph still needs well-formed tensors.
    // One zero position avoids zero-sized tensors, which some GPU providers reject.
    private static DenseTensor<float> CreatePlaceholderPast(NodeMetadata metadata)
    {
        int[] dims = metadata.Dimensions.Select(static d => d > 0 ? d : 1).ToArray();
        dims[0] = 1;
        return new DenseTensor<float>(dims);
    }

    private static List<long> DecodeWithoutCache(
        InferenceSession decoderSession,
        Tensor<float> encoderHiddenStates,
        long[] encoderAttentionMask,
        long decoderStartTokenId,
        long endOfSentenceTokenId,
        int padTokenId,
        int maxSteps,
        ExecutionProviderKind? provider,
        CancellationToken cancellationToken,
        bool staticShapes,
        int noRepeatNgramSize)
    {
        var tokens = new List<long> { decoderStartTokenId };
        var maskTensor = new DenseTensor<long>(encoderAttentionMask, [1, encoderAttentionMask.Length]);
        for (int step = 0; step < maxSteps; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var inputs = new List<NamedOnnxValue>(decoderSession.InputMetadata.Count);
            DenseTensor<long> idsTensor = staticShapes
                ? PadToBucket(tokens, padTokenId)
                : new(tokens.ToArray(), [1, tokens.Count]);
            foreach ((string name, NodeMetadata metadata) in decoderSession.InputMetadata)
            {
                inputs.Add(name switch
                {
                    "input_ids" => NamedOnnxValue.CreateFromTensor(name, idsTensor),
                    "encoder_hidden_states" => NamedOnnxValue.CreateFromTensor(name, encoderHiddenStates),
                    "encoder_attention_mask" or "attention_mask" => NamedOnnxValue.CreateFromTensor(name, maskTensor),
                    UseCacheBranchInput => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<bool>(new[] { false }, [1])),
                    _ when name.StartsWith(PastPrefix, StringComparison.Ordinal) =>
                        NamedOnnxValue.CreateFromTensor(name, CreateEmptyPast(metadata)),
                    _ => throw new NotSupportedException($"Decoder input '{name}' is not supported.")
                });
            }

            // Only the logits are read; fetching the unused present.* caches would copy them back
            // from the device every step.
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs =
                decoderSession.RunWithRetry(inputs, cancellationToken: cancellationToken, provider: provider, outputNames: LogitsOutput);
            int nextToken = ArgMaxAtPosition(
                outputs.Single(static value => value.Name == "logits").AsTensor<float>(),
                tokens.Count - 1,
                padTokenId,
                BannedNgramContinuations(tokens.Skip(1).ToList(), noRepeatNgramSize));
            if (nextToken < 0 || nextToken == endOfSentenceTokenId)
            {
                break;
            }

            tokens.Add(nextToken);
        }

        return tokens.Skip(1).ToList();
    }

    private static DenseTensor<long> PadToBucket(List<long> tokens, int padTokenId)
    {
        long[] padded = new long[BucketLength(tokens.Count)];
        tokens.CopyTo(padded);
        Array.Fill(padded, padTokenId, tokens.Count, padded.Length - tokens.Count);
        return new DenseTensor<long>(padded, [1, padded.Length]);
    }

    private static DenseTensor<float> CreateEmptyPast(NodeMetadata metadata)
    {
        int[] dims = metadata.Dimensions.Select(static d => d > 0 ? d : 1).ToArray();
        dims[0] = 1;
        if (dims.Length > 2)
        {
            dims[2] = 0;
        }

        return new DenseTensor<float>(Array.Empty<float>(), dims);
    }

    internal static int ArgMaxLastPosition(Tensor<float> logits, int padTokenId) =>
        ArgMaxAtPosition(logits, logits.Dimensions[1] - 1, padTokenId);

    /// <summary>
    /// Tokens that would complete an n-gram already present in <paramref name="generated"/>, or null when
    /// the guard is off or nothing is banned yet.
    /// </summary>
    internal static HashSet<int>? BannedNgramContinuations(IReadOnlyList<long> generated, int ngramSize)
    {
        if (ngramSize <= 0 || generated.Count < ngramSize)
        {
            return null;
        }

        int prefixLength = ngramSize - 1;
        int prefixStart = generated.Count - prefixLength;
        HashSet<int>? banned = null;
        for (int start = 0; start + prefixLength < generated.Count; start++)
        {
            bool matches = true;
            for (int offset = 0; offset < prefixLength; offset++)
            {
                if (generated[start + offset] != generated[prefixStart + offset])
                {
                    matches = false;
                    break;
                }
            }

            if (matches)
            {
                (banned ??= []).Add((int)generated[start + prefixLength]);
            }
        }

        return banned;
    }

    internal static int ArgMaxAtPosition(Tensor<float> logits, int position, int padTokenId, HashSet<int>? bannedTokens = null)
    {
        int vocabularySize = logits.Dimensions[2];
        int bestToken = -1;
        float bestValue = float.NegativeInfinity;
        if (logits is DenseTensor<float> dense)
        {
            ReadOnlySpan<float> row = dense.Buffer.Span.Slice(position * vocabularySize, vocabularySize);
            for (int token = 0; token < row.Length; token++)
            {
                if (token != padTokenId && row[token] > bestValue && bannedTokens?.Contains(token) != true)
                {
                    bestValue = row[token];
                    bestToken = token;
                }
            }

            return bestToken;
        }

        for (int token = 0; token < vocabularySize; token++)
        {
            float value = logits[0, position, token];
            if (token != padTokenId && value > bestValue && bannedTokens?.Contains(token) != true)
            {
                bestValue = value;
                bestToken = token;
            }
        }

        return bestToken;
    }
}
