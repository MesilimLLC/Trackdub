using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Audio;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Onnx.Runtime.Routing;
using Trackdub.Inference.Onnx.Whisper;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.ParakeetTdt;

/// <summary>
/// NVIDIA Parakeet-TDT 0.6B (FastConformer encoder + TDT decoder), offline full-attention ASR.
/// Mel features come from the export's own <c>nemo128.onnx</c> preprocessor (NeMo per-feature
/// normalization baked in), so there is no hand-written feature extractor to drift from NeMo.
/// </summary>
public sealed class ParakeetTdtOnnxAudioTranscriptionEngine(
    IRuntimePlanner runtimePlanner,
    BenchmarkModelPathResolver modelPathResolver,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null)
    : IAudioTranscriptionEngineAdapter, IStageRuntimeExecutionReporter
{
    public const string EngineFamilyName = "parakeet-tdt";

    private const int SampleRate = 16_000;

    // FastConformer subsampling 8 x 10 ms hop.
    private const double SecondsPerEncoderFrame = 0.08;

    // Real audio context either side of a VAD region. Parakeet drops speech when a region is cut
    // exactly at the onset, and VAD boundaries typically trail the true onset by 0.1-0.3 s.
    private const double ContextPaddingSeconds = 0.5;

    // Longest audio window per encoder call; longer regions are split. Bounds the TRT profile.
    private const double MaxWindowSeconds = 28.0;

    internal static readonly IReadOnlyDictionary<string, string> TrtEncoderOptions = new Dictionary<string, string>
    {
        ["trt_profile_min_shapes"] = "audio_signal:1x128x1,length:1",
        ["trt_profile_opt_shapes"] = "audio_signal:1x128x500,length:1",
        ["trt_profile_max_shapes"] = "audio_signal:1x128x3000,length:1",
    };

    private readonly IRuntimePlanner runtimePlanner = runtimePlanner ?? throw new ArgumentNullException(nameof(runtimePlanner));
    private readonly BenchmarkModelPathResolver modelPathResolver = modelPathResolver ?? throw new ArgumentNullException(nameof(modelPathResolver));

    public StageRuntimeExecutionSummary? LastExecutionSummary { get; private set; }

    public string EngineFamily => EngineFamilyName;

    public async Task<IReadOnlyList<RecognizedTranscriptSegment>> TranscribeAsync(
        string normalizedAudioPath,
        IReadOnlyList<SpeechRegion> regions,
        CancellationToken cancellationToken) =>
        await TranscribeAsync(new AudioTranscriptionRequest(normalizedAudioPath, regions), cancellationToken)
            .ConfigureAwait(false);

    public async Task<IReadOnlyList<RecognizedTranscriptSegment>> TranscribeAsync(
        AudioTranscriptionRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();
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
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(request.Regions);
        if (!plan.IsRunnable() || plan.ExecutionProvider is null)
        {
            throw new InvalidOperationException(plan.Fallback?.Detail ?? "Runtime planner did not produce a ready Asr plan.");
        }

        ParakeetTdtModelPaths paths = ResolveModelPaths(PlannedRuntimeModelResolver.ResolveModelPath(plan, modelPathResolver));
        ParakeetTdtVocab vocab = await ParakeetTdtVocab.LoadAsync(paths.VocabPath, cancellationToken).ConfigureAwait(false);

        IAudioSamples audio = await WaveAudioReader.ReadMonoPcm16Async(request.NormalizedAudioPath, cancellationToken)
            .ConfigureAwait(false);
        using IAudioSamples targetAudio = AudioResampler.CreateResampledStream(audio, SampleRate);
        double durationSeconds = targetAudio.SampleFrameCount / (double)SampleRate;

        IReadOnlyList<SpeechRegion> regions = request.Regions.Count > 0
            ? request.Regions
            : durationSeconds < 30.0 ? [new SpeechRegion(0, 0.0, durationSeconds)] : [];

        // The preprocessor is a tiny STFT/mel graph; CPU keeps it off the TRT shape-profile path.
        // Pooled so repeated transcription calls do not rebuild the session every time.
        // Smoke warms this same key ("parakeet-tdt-preprocessor").
        using OnnxExecutionSessionFactory.SingleSessionLease preprocessor = await OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "parakeet-tdt-preprocessor", Provider = ExecutionProviderKind.Cpu, CancellationToken = cancellationToken }, paths.PreprocessorPath)
            .ConfigureAwait(false);
        using OnnxExecutionSessionFactory.NemotronAsrSessionLease sessions = await OnnxExecutionSessionFactory
            .CreatePooledNemotronAsrAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = EngineFamilyName, Provider = plan.ExecutionProvider.Value, CancellationToken = cancellationToken, ModelId = plan.ModelId, Variant = plan.Variant, AdditionalTrtEncoderOptions = TrtEncoderOptions }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = paths.EncoderPath, Decoder = paths.DecoderJointPath })
            .ConfigureAwait(false);
        var decoder = new ParakeetTdtGreedyDecoder(sessions.DecoderJointSession, vocab, plan.ExecutionProvider);

        SpeechRegion[] ordered = regions
            .Where(static r => r.EndSeconds > r.StartSeconds)
            .OrderBy(static r => r.StartSeconds)
            .ToArray();
        var regionWords = new List<List<RecognizedTranscriptWord>>(ordered.Length);
        for (int regionIndex = 0; regionIndex < ordered.Length; regionIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            SpeechRegion region = ordered[regionIndex];
            (double claimStart, double claimEnd) = ResolveClaimInterval(ordered, regionIndex);
            var words = new List<RecognizedTranscriptWord>();
            for (double windowStart = region.StartSeconds; windowStart < region.EndSeconds; windowStart += MaxWindowSeconds)
            {
                double windowEnd = Math.Min(region.EndSeconds, windowStart + MaxWindowSeconds);
                words.AddRange(TranscribeWindow(
                    preprocessor.Session,
                    sessions.EncoderSession,
                    decoder,
                    vocab,
                    targetAudio,
                    windowStart,
                    windowEnd,
                    windowStart == region.StartSeconds ? claimStart : windowStart,
                    windowEnd >= region.EndSeconds ? claimEnd : windowEnd,
                    durationSeconds,
                    words.Count,
                    cancellationToken));
            }

            regionWords.Add(words);
        }

        DropBoundaryDuplicates(regionWords);
        var segments = new List<RecognizedTranscriptSegment>(ordered.Length);
        for (int regionIndex = 0; regionIndex < ordered.Length; regionIndex++)
        {
            List<RecognizedTranscriptWord> words = regionWords[regionIndex]
                .Select(static (word, wordIndex) => word with { WordIndex = wordIndex })
                .ToList();
            string text = string.Join(' ', words.Select(static w => w.Text)).Trim();
            if (text.Length > 0)
            {
                SpeechRegion region = ordered[regionIndex];
                segments.Add(new RecognizedTranscriptSegment(
                    region.Index, region.StartSeconds, region.EndSeconds, text, DetectedLanguage: null, Words: words));
            }
        }

        LastExecutionSummary = new StageRuntimeExecutionSummary(
            sessions.RequestedProvider,
            sessions.SelectedProvider,
            plan.ModelId,
            plan.ModelAlias,
            plan.Variant,
            sessions.BootstrapDetail);
        return segments;
    }

    private static List<RecognizedTranscriptWord> TranscribeWindow(
        InferenceSession preprocessor,
        InferenceSession encoder,
        ParakeetTdtGreedyDecoder decoder,
        ParakeetTdtVocab vocab,
        IAudioSamples audio,
        double keepStart,
        double keepEnd,
        double claimStart,
        double claimEnd,
        double audioDuration,
        int firstWordIndex,
        CancellationToken cancellationToken)
    {
        double windowStart = Math.Max(0.0, keepStart - ContextPaddingSeconds);
        double windowEnd = Math.Min(audioDuration, keepEnd + ContextPaddingSeconds);
        long startSample = (long)Math.Floor(windowStart * SampleRate);
        long endSample = Math.Min(audio.SampleFrameCount, (long)Math.Ceiling(windowEnd * SampleRate));
        if (endSample - startSample < 400)
        {
            return [];
        }

        var samples = new float[checked((int)(endSample - startSample))];
        audio.ReadMonoSamples(startSample, samples);

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> features = preprocessor.RunWithRetry(
        [
            NamedOnnxValue.CreateFromTensor("waveforms", new DenseTensor<float>(samples, [1, samples.Length])),
            NamedOnnxValue.CreateFromTensor("waveforms_lens", new DenseTensor<long>(new long[] { samples.Length }, [1])),
        ], cancellationToken: cancellationToken);
        Tensor<float> mel = features.Single(static v => v.Name == "features").AsTensor<float>();
        long melLength = features.Single(static v => v.Name == "features_lens").AsTensor<long>().First();

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoded = encoder.RunWithRetry(
        [
            NamedOnnxValue.CreateFromTensor("audio_signal", mel),
            NamedOnnxValue.CreateFromTensor("length", new DenseTensor<long>(new[] { melLength }, [1])),
        ], cancellationToken: cancellationToken);
        Tensor<float> encoderOutput = encoded.Single(static v => v.Name == "outputs").AsTensor<float>();
        int encodedLength = (int)encoded.Single(static v => v.Name == "encoded_lengths").AsTensor<long>().First();

        IReadOnlyList<ParakeetTdtGreedyDecoder.EmittedToken> tokens = decoder.Decode(
            encoderOutput, encodedLength, cancellationToken);
        return GroupWords(tokens, vocab, windowStart, claimStart, claimEnd, firstWordIndex);
    }

    /// <summary>
    /// Adjacent windows overlap by the padding and can each timestamp the same boundary word just
    /// inside their own claim interval. Keep it in the earlier region only.
    /// </summary>
    internal static void DropBoundaryDuplicates(List<List<RecognizedTranscriptWord>> regionWords)
    {
        const double MaxDuplicateOffsetSeconds = 0.6;
        for (int index = 1; index < regionWords.Count; index++)
        {
            RecognizedTranscriptWord? previousLast = regionWords[index - 1].LastOrDefault();
            List<RecognizedTranscriptWord> current = regionWords[index];
            if (previousLast is null || current.Count == 0)
            {
                continue;
            }

            RecognizedTranscriptWord first = current[0];
            if (Math.Abs(first.StartSeconds - previousLast.StartSeconds) <= MaxDuplicateOffsetSeconds &&
                string.Equals(NormalizeWord(first.Text), NormalizeWord(previousLast.Text), StringComparison.OrdinalIgnoreCase))
            {
                current.RemoveAt(0);
            }
        }
    }

    private static string NormalizeWord(string word) =>
        new(word.Where(char.IsLetterOrDigit).ToArray());

    /// <summary>
    /// The span of word start times a region owns: its VAD bounds widened into each neighbouring
    /// gap up to the midpoint (capped at the padding). Adjacent regions partition every gap, so a
    /// word spoken between two VAD regions is kept exactly once.
    /// </summary>
    internal static (double Start, double End) ResolveClaimInterval(IReadOnlyList<SpeechRegion> ordered, int index)
    {
        SpeechRegion region = ordered[index];
        double start = region.StartSeconds - ContextPaddingSeconds;
        if (index > 0)
        {
            start = Math.Max(start, (ordered[index - 1].EndSeconds + region.StartSeconds) / 2);
        }

        double end = region.EndSeconds + ContextPaddingSeconds;
        if (index < ordered.Count - 1)
        {
            end = Math.Min(end, (region.EndSeconds + ordered[index + 1].StartSeconds) / 2);
        }

        return (Math.Max(0.0, start), end);
    }

    /// <summary>
    /// Joins SentencePiece tokens into words (a leading U+2581 starts a word) and keeps only words
    /// whose start falls inside the region's claim interval, so padding never duplicates a word.
    /// </summary>
    internal static List<RecognizedTranscriptWord> GroupWords(
        IReadOnlyList<ParakeetTdtGreedyDecoder.EmittedToken> tokens,
        ParakeetTdtVocab vocab,
        double windowStart,
        double claimStart,
        double claimEnd,
        int firstWordIndex)
    {
        var words = new List<RecognizedTranscriptWord>();
        var text = new StringBuilder();
        double wordStart = 0;
        double wordEnd = 0;

        void Flush()
        {
            string word = text.ToString().Trim();
            if (word.Length > 0 && wordStart >= claimStart && wordStart < claimEnd)
            {
                words.Add(new RecognizedTranscriptWord(firstWordIndex + words.Count, wordStart, wordEnd, word));
            }

            text.Clear();
        }

        foreach (ParakeetTdtGreedyDecoder.EmittedToken token in tokens)
        {
            if (vocab.IsControl(token.TokenId))
            {
                continue;
            }

            string piece = vocab.Piece(token.TokenId);
            double tokenStart = windowStart + (token.Frame * SecondsPerEncoderFrame);
            if (piece.StartsWith('▁') || text.Length == 0)
            {
                Flush();
                wordStart = tokenStart;
            }

            text.Append(piece.Replace('▁', ' '));
            wordEnd = tokenStart + (Math.Max(1, token.Duration) * SecondsPerEncoderFrame);
        }

        Flush();
        return words;
    }

    private static ParakeetTdtModelPaths ResolveModelPaths(string encoderModelPath)
    {
        string root = Path.GetDirectoryName(encoderModelPath)
            ?? throw new InvalidOperationException("Parakeet-TDT model root path could not be resolved.");
        var paths = new ParakeetTdtModelPaths(
            encoderModelPath,
            Path.Join(root, "decoder_joint-model.onnx"),
            Path.Join(root, "nemo128.onnx"),
            Path.Join(root, "vocab.txt"));
        foreach (string path in new[] { paths.DecoderJointPath, paths.PreprocessorPath, paths.VocabPath })
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Parakeet-TDT ONNX package is missing a required file.", path);
            }
        }

        return paths;
    }

    private sealed record ParakeetTdtModelPaths(
        string EncoderPath,
        string DecoderJointPath,
        string PreprocessorPath,
        string VocabPath);
}
