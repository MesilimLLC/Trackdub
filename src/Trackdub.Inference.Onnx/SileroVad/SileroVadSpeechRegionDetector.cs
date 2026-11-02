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

namespace Trackdub.Inference.Onnx.SileroVad;

public sealed class SileroVadSpeechRegionDetector(IRuntimePlanner runtimePlanner,
    BenchmarkModelPathResolver modelPathResolver,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null)
    : ISpeechRegionDetectorAdapter, IStageRuntimeExecutionReporter
{
    public const string EngineFamilyName = "silero-vad";

    private const int TargetSampleRate = 16000;
    private const int ChunkSize = 512;
    private const float SpeechThreshold = 0.5f;
    private const float SilenceThreshold = 0.35f;
    private const double MinSpeechDurationSeconds = 0.25;
    private const double MinSilenceDurationSeconds = 0.10;
    private const double SpeechPaddingSeconds = 0.03;

    private static readonly int[] InputDimensions = [1, ChunkSize];
    private static readonly int[] StateDimensions = [2, 1, 128];
    private static readonly int[] SrDimensions = [1];

    private readonly IRuntimePlanner runtimePlanner = runtimePlanner ?? throw new ArgumentNullException(nameof(runtimePlanner));
    private readonly BenchmarkModelPathResolver modelPathResolver = modelPathResolver ?? throw new ArgumentNullException(nameof(modelPathResolver));

    public StageRuntimeExecutionSummary? LastExecutionSummary { get; private set; }

    public string EngineFamily => EngineFamilyName;

    public async Task<IReadOnlyList<SpeechRegion>> DetectAsync(
        string normalizedAudioPath,
        double durationSeconds,
        CancellationToken cancellationToken) =>
        await DetectAsync(
            new SpeechRegionDetectionRequest(normalizedAudioPath, durationSeconds),
            cancellationToken).ConfigureAwait(false);

    public async Task<IReadOnlyList<SpeechRegion>> DetectAsync(
        SpeechRegionDetectionRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        InferenceRequestOptions options = request.Options ?? InferenceRequestOptions.Default;
        StageRuntimePlanningRequest planningRequest = await StageRuntimePlanningRequestFactory.ApplyPreferredModelTierAsync(
            new StageRuntimePlanningRequest(
                RuntimeStage.Vad,
                options.NormalizedPreferredModelAlias,
                PreferredExecutionProvider: ExecutionProviderRequest.ParsePreferredExecutionProvider(
                    options.PreferredExecutionProvider,
                    options.RequirePreferredExecutionProvider),
                RequirePreferredExecutionProvider: options.RequirePreferredExecutionProvider,
                PreferredModelVariantAlias: options.NormalizedPreferredModelVariantAlias),
            runtimePlanningPreferences,
            cancellationToken).ConfigureAwait(false);

        StageRuntimePlan plan = await runtimePlanner.PlanAsync(planningRequest, cancellationToken).ConfigureAwait(false);

        return await DetectAsync(request, plan, cancellationToken).ConfigureAwait(false);
    }

    public async Task<IReadOnlyList<SpeechRegion>> DetectAsync(
        SpeechRegionDetectionRequest request,
        StageRuntimePlan plan,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(plan);
        EnsurePlanReady(plan, RuntimeStage.Vad);

        string modelPath = ResolvePlannedModelPath(plan);
        using OnnxExecutionSessionFactory.SingleSessionLease sessionLease = await OnnxExecutionSessionFactory
            .CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "silero-vad", Provider = plan.ExecutionProvider!.Value, CancellationToken = cancellationToken, AllowTrtInitFallback = !plan.RequirePreferredExecutionProvider }, modelPath)
            .ConfigureAwait(false);

        IAudioSamples audio = await WaveAudioReader.ReadMonoPcm16Async(request.NormalizedAudioPath, cancellationToken).ConfigureAwait(false);
        // CreateResampledStream returns the source as-is when sample rates match;
        // otherwise it wraps and takes ownership of it. Dispose only the outer reader.
        using IAudioSamples targetAudio = AudioResampler.CreateResampledStream(audio, TargetSampleRate);

        float[] state = new float[2 * 128];
        long totalFrames = targetAudio.SampleFrameCount;
        long chunkCount = (totalFrames + ChunkSize - 1) / ChunkSize;
        var probabilities = new List<float>(Math.Max(1, (int)chunkCount));

        float[] chunkBuffer = System.Buffers.ArrayPool<float>.Shared.Rent(ChunkSize);
        try
        {
            for (long i = 0; i < chunkCount; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                long startFrame = i * ChunkSize;
                Span<float> chunkSpan = chunkBuffer.AsSpan(0, ChunkSize);

                targetAudio.ReadMonoSamples(startFrame, chunkSpan);

                using var input = CreateInputSet(chunkSpan, state);
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output = sessionLease.Session.RunWithRetry(input.Values, cancellationToken: cancellationToken);
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputCollection = output;

                Tensor<float> probabilityTensor = outputCollection.Single(static value => value.Name == "output").AsTensor<float>();
                Tensor<float> stateTensor = outputCollection.Single(static value => value.Name == "stateN").AsTensor<float>();
                probabilities.Add(probabilityTensor[0, 0]);
                CopyState(stateTensor, state);
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(chunkBuffer);
        }

        LastExecutionSummary = CreateExecutionSummary(plan, sessionLease);
        return BuildSpeechRegions(probabilities, request.DurationSeconds);
    }

    private static InputSet CreateInputSet(ReadOnlySpan<float> chunk, float[] state)
    {
        float[] inputSnapshot = System.Buffers.ArrayPool<float>.Shared.Rent(ChunkSize);
        float[] stateSnapshot = System.Buffers.ArrayPool<float>.Shared.Rent(state.Length);
        long[] sampleRateData = System.Buffers.ArrayPool<long>.Shared.Rent(1);

        try
        {
            chunk.CopyTo(inputSnapshot.AsSpan(0, ChunkSize));
            Array.Copy(state, stateSnapshot, state.Length);
            sampleRateData[0] = TargetSampleRate;

            IReadOnlyList<NamedOnnxValue> values =
            [
                NamedOnnxValue.CreateFromTensor("input", new DenseTensor<float>(new Memory<float>(inputSnapshot, 0, ChunkSize), InputDimensions)),
                NamedOnnxValue.CreateFromTensor("state", new DenseTensor<float>(new Memory<float>(stateSnapshot, 0, state.Length), StateDimensions)),
                NamedOnnxValue.CreateFromTensor("sr", new DenseTensor<long>(new Memory<long>(sampleRateData, 0, 1), SrDimensions))
            ];

            return new InputSet(values, inputSnapshot, stateSnapshot, sampleRateData);
        }
        catch
        {
            System.Buffers.ArrayPool<float>.Shared.Return(inputSnapshot);
            System.Buffers.ArrayPool<float>.Shared.Return(stateSnapshot);
            System.Buffers.ArrayPool<long>.Shared.Return(sampleRateData);
            throw;
        }
    }

    private static void CopyState(Tensor<float> tensor, float[] destination)
    {
        int index = 0;
        foreach (float value in tensor)
        {
            destination[index++] = value;
        }
    }

    private static IReadOnlyList<SpeechRegion> BuildSpeechRegions(
        IReadOnlyList<float> probabilities,
        double durationSeconds)
    {
        if (probabilities.Count == 0 || durationSeconds <= 0)
        {
            return [];
        }

        double secondsPerChunk = ChunkSize / (double)TargetSampleRate;
        double minSpeechDuration = MinSpeechDurationSeconds;
        double minSilenceDuration = MinSilenceDurationSeconds;
        double speechPadding = SpeechPaddingSeconds;
        var rawRegions = new List<(double Start, double End)>();
        bool inSpeech = false;
        int? speechStartChunk = null;
        int lastSpeechChunk = -1;
        int pendingSilenceChunks = 0;

        for (int index = 0; index < probabilities.Count; index++)
        {
            float probability = probabilities[index];
            if (probability >= SpeechThreshold)
            {
                if (!inSpeech)
                {
                    inSpeech = true;
                    speechStartChunk = index;
                }

                lastSpeechChunk = index;
                pendingSilenceChunks = 0;
                continue;
            }

            if (!inSpeech)
            {
                continue;
            }

            if (probability < SilenceThreshold)
            {
                pendingSilenceChunks++;
            }

            if (pendingSilenceChunks * secondsPerChunk < minSilenceDuration)
            {
                continue;
            }

            AppendRegion(rawRegions, speechStartChunk!.Value, lastSpeechChunk, secondsPerChunk, minSpeechDuration, durationSeconds, speechPadding);
            inSpeech = false;
            speechStartChunk = null;
            lastSpeechChunk = -1;
            pendingSilenceChunks = 0;
        }

        if (inSpeech && speechStartChunk is not null)
        {
            AppendRegion(rawRegions, speechStartChunk.Value, lastSpeechChunk, secondsPerChunk, minSpeechDuration, durationSeconds, speechPadding);
        }

        return rawRegions
            .Select((region, index) => new SpeechRegion(index, region.Start, region.End))
            .ToArray();
    }

    private static void AppendRegion(
        ICollection<(double Start, double End)> regions,
        int startChunk,
        int endChunk,
        double secondsPerChunk,
        double minSpeechDuration,
        double durationSeconds,
        double speechPadding)
    {
        double start = Math.Max(0, (startChunk * secondsPerChunk) - speechPadding);
        double end = Math.Min(durationSeconds, ((endChunk + 1) * secondsPerChunk) + speechPadding);
        if (end - start < minSpeechDuration)
        {
            return;
        }

        if (regions.Count > 0)
        {
            (double Start, double End) previous = regions.Last();
            if (start <= previous.End)
            {
                regions.Remove(previous);
                regions.Add((previous.Start, Math.Max(previous.End, end)));
                return;
            }
        }

        regions.Add((start, end));
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

    private static StageRuntimeExecutionSummary CreateExecutionSummary(
        StageRuntimePlan plan,
        OnnxExecutionSessionFactory.SingleSessionLease sessionLease) =>
        new(
            sessionLease.RequestedProvider,
            sessionLease.SelectedProvider,
            plan.ModelId,
            plan.ModelAlias,
            plan.Variant,
            sessionLease.BootstrapDetail);

    private sealed class InputSet(
        IReadOnlyList<NamedOnnxValue> values,
        float[] inputSnapshot,
        float[] stateSnapshot,
        long[] sampleRateData)
        : IDisposable
    {
        public IReadOnlyList<NamedOnnxValue> Values { get; } = values;

        public void Dispose()
        {
            foreach (IDisposable value in Values.OfType<IDisposable>())
            {
                value.Dispose();
            }

            System.Buffers.ArrayPool<float>.Shared.Return(inputSnapshot);
            System.Buffers.ArrayPool<float>.Shared.Return(stateSnapshot);
            System.Buffers.ArrayPool<long>.Shared.Return(sampleRateData);
        }
    }
}
