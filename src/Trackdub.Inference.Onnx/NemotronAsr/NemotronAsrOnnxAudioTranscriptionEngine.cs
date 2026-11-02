using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Audio;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Onnx.Runtime.Routing;
using Trackdub.Inference.Onnx.Whisper;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.NemotronAsr;

public sealed class NemotronAsrOnnxAudioTranscriptionEngine(
    IRuntimePlanner runtimePlanner,
    BenchmarkModelPathResolver modelPathResolver,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null)
    : IAudioTranscriptionEngineAdapter, IStageRuntimeExecutionReporter
{
    public const string EngineFamilyName = "nemotron-asr";

    private readonly IRuntimePlanner runtimePlanner = runtimePlanner ?? throw new ArgumentNullException(nameof(runtimePlanner));
    private readonly BenchmarkModelPathResolver modelPathResolver = modelPathResolver ?? throw new ArgumentNullException(nameof(modelPathResolver));

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

        NemotronAsrModelPaths modelPaths = ResolveModelPaths(ResolvePlannedModelPath(plan));
        NemotronAsrSentencePieceVocab vocab = await NemotronAsrSentencePieceVocab
            .LoadAsync(modelPaths.TokenizerPath, cancellationToken)
            .ConfigureAwait(false);
        NemotronAsrPromptDictionary promptDictionary = await NemotronAsrLanguagePrompts
            .LoadAsync(modelPaths.ConfigPath, cancellationToken)
            .ConfigureAwait(false);
        NemotronAsrExportConfig exportConfig = NemotronAsrExportConfig.Load(
            modelPaths.ConfigPath,
            hasPromptInput: true);
        var featureExtractor = new NemotronAsrMelFeatureExtractor(exportConfig.ApplyPerFeatureNormalization);

        IAudioSamples audio = await WaveAudioReader.ReadMonoPcm16Async(request.NormalizedAudioPath, cancellationToken)
            .ConfigureAwait(false);
        using IAudioSamples targetAudio = AudioResampler.CreateResampledStream(audio, NemotronAsrMelFeatureExtractor.SampleRate);
        double durationSeconds = targetAudio.SampleFrameCount / (double)NemotronAsrMelFeatureExtractor.SampleRate;

        IReadOnlyList<SpeechRegion> effectiveRegions = request.Regions;
        if (request.Regions.Count == 0)
        {
            if (durationSeconds >= 30.0)
            {
                LastExecutionSummary = CreatePlannedOnlySummary(
                    plan,
                    $"ASR skipped: VAD detected no speech regions in {durationSeconds:F1}s audio.");
                return [];
            }

            effectiveRegions = [new SpeechRegion(0, 0.0, durationSeconds)];
        }

        using OnnxExecutionSessionFactory.NemotronAsrSessionLease sessionLease = await AcquireSessionLeaseAsync(
            plan, modelPaths, cancellationToken).ConfigureAwait(false);

        // Use the original (filtered) caller-provided SpeechRegions for the transcription loop.
        // This ensures RecognizedTranscriptSegment.Index / Start / End exactly match the input
        // SpeechRegion values (no reindexing, no time expansion from any helper). This preserves
        // stable identifiers needed by retranscribe flows.
        // Matches the suggestion in the P2 "Preserve caller speech regions for Nemotron ASR" review.
        IReadOnlyList<SpeechRegion> transcriptionRegions = effectiveRegions
            .Where(static region => region.EndSeconds > region.StartSeconds)
            .OrderBy(static region => region.Index)
            .ToArray();

        long promptIndex = promptDictionary.ResolvePromptIndex(request.SourceLanguage);
        string? forcedLanguage = promptIndex == promptDictionary.AutoPromptIndex
            ? null
            : promptDictionary.TryGetIsoCode(request.SourceLanguage);

        var segments = new List<RecognizedTranscriptSegment>(transcriptionRegions.Count);
        foreach (SpeechRegion region in transcriptionRegions)
        {
            cancellationToken.ThrowIfCancellationRequested();
            RegionTranscription regionResult = TranscribeRegion(
                sessionLease,
                vocab,
                exportConfig,
                featureExtractor,
                targetAudio,
                region,
                promptIndex,
                forcedLanguage,
                cancellationToken);

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

        LastExecutionSummary = CreateExecutionSummary(plan, sessionLease);
        return segments;
    }

    private RegionTranscription TranscribeRegion(
        OnnxExecutionSessionFactory.NemotronAsrSessionLease sessionLease,
        NemotronAsrSentencePieceVocab vocab,
        NemotronAsrExportConfig exportConfig,
        NemotronAsrMelFeatureExtractor featureExtractor,
        IAudioSamples targetAudio,
        SpeechRegion region,
        long promptIndex,
        string? forcedLanguage,
        CancellationToken cancellationToken)
    {
        long startSample = Math.Max(0, (long)Math.Floor(region.StartSeconds * NemotronAsrMelFeatureExtractor.SampleRate));
        long endSample = Math.Min(
            targetAudio.SampleFrameCount,
            (long)Math.Ceiling(region.EndSeconds * NemotronAsrMelFeatureExtractor.SampleRate));
        if (endSample <= startSample)
        {
            return new RegionTranscription(string.Empty, null, []);
        }

        float[] regionSamples = new float[checked((int)(endSample - startSample))];
        targetAudio.ReadMonoSamples(startSample, regionSamples);
        float[,] mel = featureExtractor.Extract(regionSamples);
        if (mel.GetLength(1) == 0)
        {
            return new RegionTranscription(string.Empty, null, []);
        }

        var decoder = new NemotronAsrGreedyDecoder(sessionLease, vocab, exportConfig);
        IReadOnlyList<int> tokens = decoder.Decode(mel, promptIndex, cancellationToken);
        string text = decoder.DecodeText(tokens);

        return new RegionTranscription(
            text,
            forcedLanguage ?? decoder.DetectedLanguage,
            Words: []);
    }

    internal static IReadOnlyList<SpeechRegion> BuildTranscriptionRegionsForTesting(
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

    private static NemotronAsrModelPaths ResolveModelPaths(string encoderModelPath)
    {
        string modelRoot = Path.GetDirectoryName(encoderModelPath)
            ?? throw new InvalidOperationException("Nemotron ASR model root path could not be resolved.");
        string configPath = Path.Join(modelRoot, "config.json");
        string decoderJointPath = Path.Join(modelRoot, "decoder_joint.onnx");
        string tokenizerPath = Path.Join(modelRoot, "tokenizer.model");
        foreach (string path in new[] { configPath, decoderJointPath, tokenizerPath })
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Nemotron ASR ONNX package is missing a required file.", path);
            }
        }

        return new NemotronAsrModelPaths(
            modelRoot,
            encoderModelPath,
            configPath,
            decoderJointPath,
            tokenizerPath);
    }

    private static StageRuntimeExecutionSummary CreateExecutionSummary(
        StageRuntimePlan plan,
        OnnxExecutionSessionFactory.NemotronAsrSessionLease sessionLease) =>
        new(
            sessionLease.RequestedProvider,
            sessionLease.SelectedProvider,
            plan.ModelId,
            plan.ModelAlias,
            plan.Variant,
            sessionLease.BootstrapDetail);

    private static Task<OnnxExecutionSessionFactory.NemotronAsrSessionLease> AcquireSessionLeaseAsync(
        StageRuntimePlan plan,
        NemotronAsrModelPaths modelPaths,
        CancellationToken cancellationToken)
    {
        NemotronAsrEncoderTrtProfiles.SessionOptions sessionOptions =
            NemotronAsrEncoderTrtProfiles.BuildRuntimeOptions(modelPaths.EncoderPath);
        return OnnxExecutionSessionFactory.CreatePooledNemotronAsrAsync(
            new OnnxExecutionSessionFactory.PooledSessionRequest
            {
                EngineFamily = EngineFamilyName,
                Provider = plan.ExecutionProvider!.Value,
                CancellationToken = cancellationToken,
                ModelId = plan.ModelId,
                Variant = plan.Variant,
                AdditionalTrtEncoderOptions = sessionOptions.Encoder,
                AdditionalTrtDecoderOptions = sessionOptions.Decoder,
            },
            new OnnxExecutionSessionFactory.DualModelPaths
            {
                Encoder = modelPaths.EncoderPath,
                Decoder = modelPaths.DecoderJointPath,
            });
    }

    private static StageRuntimeExecutionSummary CreatePlannedOnlySummary(StageRuntimePlan plan, string bootstrapDetail) =>
        new(
            "auto",
            plan.ExecutionProvider!.Value.ToString().ToLowerInvariant(),
            plan.ModelId,
            plan.ModelAlias,
            plan.Variant,
            bootstrapDetail);

    private sealed record NemotronAsrModelPaths(
        string ModelRootPath,
        string EncoderPath,
        string ConfigPath,
        string DecoderJointPath,
        string TokenizerPath);

    private sealed record RegionTranscription(
        string Text,
        string? DetectedLanguage,
        IReadOnlyList<RecognizedTranscriptWord> Words);
}
