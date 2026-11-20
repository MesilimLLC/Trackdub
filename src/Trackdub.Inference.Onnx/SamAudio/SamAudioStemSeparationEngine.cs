using System.Globalization;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Audio;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Onnx.Runtime.Routing;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>
/// Speech/bed separation with SAM-Audio small. Speech is the model's text-prompted target; the bed is its own residual
/// output, not mix-minus-speech. Output stems are mono at the model's 48 kHz.
/// </summary>
public sealed class SamAudioStemSeparationEngine : IStemSeparationEngineAdapter, IStageRuntimeExecutionReporter
{
    public const string EngineFamilyName = SamAudioModelConstants.EngineFamilyName;

    private readonly ISamAudioSeparator separator;
    private readonly IRuntimePlanner? runtimePlanner;
    private readonly IRuntimePlanningPreferences? runtimePlanningPreferences;

    public SamAudioStemSeparationEngine(IRuntimePlanner runtimePlanner, IRuntimePlanningPreferences? runtimePlanningPreferences = null)
        : this(new SamAudioOnnxSeparator(), runtimePlanner, runtimePlanningPreferences)
    {
    }

    internal SamAudioStemSeparationEngine(ISamAudioSeparator separator)
        : this(separator, runtimePlanner: null, runtimePlanningPreferences: null)
    {
    }

    private SamAudioStemSeparationEngine(
        ISamAudioSeparator separator,
        IRuntimePlanner? runtimePlanner,
        IRuntimePlanningPreferences? runtimePlanningPreferences)
    {
        this.separator = separator ?? throw new ArgumentNullException(nameof(separator));
        this.runtimePlanner = runtimePlanner;
        this.runtimePlanningPreferences = runtimePlanningPreferences;
    }

    public string EngineFamily => EngineFamilyName;

    public StageRuntimeExecutionSummary? LastExecutionSummary { get; private set; }

    public async Task<StemSeparationResult> SeparateAsync(
        StemSeparationRequest request,
        IProgress<StemSeparationProgress>? progress,
        CancellationToken cancellationToken)
    {
        if (runtimePlanner is null)
        {
            throw new InvalidOperationException("SAM-Audio separation requires a planned runtime.");
        }

        StageRuntimePlanningRequest planningRequest = await StageRuntimePlanningRequestFactory.ApplyPreferredModelTierAsync(
            new StageRuntimePlanningRequest(
                RuntimeStage.Separation,
                request.PreferredModelAlias,
                PreferredExecutionProvider: ExecutionProviderRequest.ParsePreferredExecutionProvider(
                    request.PreferredExecutionProvider,
                    request.RequirePreferredExecutionProvider),
                RequirePreferredExecutionProvider: request.RequirePreferredExecutionProvider,
                PreferredModelVariantAlias: request.PreferredModelVariantAlias),
            runtimePlanningPreferences,
            cancellationToken).ConfigureAwait(false);

        StageRuntimePlan plan = await runtimePlanner.PlanAsync(planningRequest, cancellationToken).ConfigureAwait(false);
        return await SeparateAsync(request, plan, progress, cancellationToken).ConfigureAwait(false);
    }

    public async Task<StemSeparationResult> SeparateAsync(
        StemSeparationRequest request,
        StageRuntimePlan plan,
        IProgress<StemSeparationProgress>? progress,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(plan);
        EnsurePlanReady(plan);

        string modelRootPath = ResolveModelRootPath(plan);
        LastExecutionSummary = CreateExecutionSummary(plan, FormatProvider(plan), "SAM-Audio ONNX session creation pending.");

        try
        {
            (float[][] channels, double durationSeconds) = await ReadChannelsAsync(request.SourceAudioPath, cancellationToken).ConfigureAwait(false);
            SamAudioSeparation separated = await separator
                .SeparateAsync(new SamAudioSeparatorRequest(modelRootPath, plan, channels), progress, cancellationToken)
                .ConfigureAwait(false);

            await WriteOutputsAsync(request, separated, cancellationToken).ConfigureAwait(false);

            IReadOnlyDictionary<string, string> metadata = BuildMetadata(plan, modelRootPath, separated);
            LastExecutionSummary = CreateExecutionSummary(
                plan,
                metadata.TryGetValue("selected_provider", out string? selected) ? selected : FormatProvider(plan),
                metadata.TryGetValue("bootstrap_detail", out string? detail) ? detail : "SAM-Audio ONNX separation completed.");

            return new StemSeparationResult(durationSeconds, SamAudioModelConstants.SampleRate, ChannelCount: 1, metadata);
        }
        catch (OperationCanceledException)
        {
            SamAudioStemOutputs.DeleteAll(request);
            throw;
        }
        catch (Exception ex)
        {
            SamAudioStemOutputs.DeleteAll(request);
            throw new InvalidOperationException($"SAM-Audio separation failed: {ex.Message}", ex);
        }
    }

    private static void EnsurePlanReady(StageRuntimePlan plan)
    {
        if (!plan.IsRunnable())
        {
            throw new InvalidOperationException(plan.Fallback?.Detail ?? "SAM-Audio separation runtime is not ready.");
        }

        if (!string.Equals(plan.EngineFamily, EngineFamilyName, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException($"SAM-Audio separation cannot run engine family '{plan.EngineFamily ?? "unknown"}'.");
        }
    }

    private static string ResolveModelRootPath(StageRuntimePlan plan)
    {
        string? root = string.IsNullOrWhiteSpace(plan.ModelEntryPath) ? null : Path.GetDirectoryName(plan.ModelEntryPath);
        return !string.IsNullOrWhiteSpace(root) && Directory.Exists(root)
            ? Path.GetFullPath(root)
            : throw new FileNotFoundException("SAM-Audio model directory unavailable.");
    }

    private static async Task<(float[][] Channels, double DurationSeconds)> ReadChannelsAsync(
        string sourceAudioPath,
        CancellationToken cancellationToken)
    {
        using IAudioChannelSamples source = await WaveAudioReader.ReadPcm16Async(sourceAudioPath, cancellationToken).ConfigureAwait(false);
        int frames = checked((int)source.SampleFrameCount);
        int channelCount = source.ChannelCount <= 1 ? 1 : 2;
        float[][] channels = new float[channelCount][];
        for (int c = 0; c < channelCount; c++)
        {
            float[] samples = new float[frames];
            source.ReadChannelSamples(0, c, samples);
            channels[c] = source.SampleRate == SamAudioModelConstants.SampleRate
                ? samples
                : AudioResampler.Resample(samples, source.SampleRate, SamAudioModelConstants.SampleRate);
        }

        return (channels, frames / (double)source.SampleRate);
    }

    private static async Task WriteOutputsAsync(StemSeparationRequest request, SamAudioSeparation separated, CancellationToken cancellationToken)
    {
        float[] speech = SamAudioStemOutputs.DownmixToMono(separated.Speech);
        float[] bed = SamAudioStemOutputs.DownmixToMono(separated.Bed);
        await SamAudioStemOutputs.WriteAsync(request, speech, bed, cancellationToken).ConfigureAwait(false);
    }

    private static Dictionary<string, string> BuildMetadata(StageRuntimePlan plan, string modelRootPath, SamAudioSeparation separated)
    {
        Dictionary<string, string> metadata = new(StringComparer.Ordinal)
        {
            ["runner"] = "onnx",
            ["engine_family"] = EngineFamilyName,
            ["model"] = plan.ModelAlias ?? string.Empty,
            ["model_root"] = modelRootPath,
            ["selected_provider"] = FormatProvider(plan),
            ["variant"] = plan.Variant ?? "default",
            ["chunk_count"] = separated.WindowCount.ToString(CultureInfo.InvariantCulture),
            ["bed"] = "native-residual",
        };

        foreach ((string key, string value) in separated.Metadata ?? new Dictionary<string, string>())
        {
            metadata[key] = value;
        }

        return metadata;
    }

    private static StageRuntimeExecutionSummary CreateExecutionSummary(StageRuntimePlan plan, string selectedProvider, string detail) =>
        new(
            FormatProvider(plan),
            selectedProvider,
            plan.ModelId ?? string.Empty,
            plan.ModelAlias ?? string.Empty,
            plan.Variant,
            detail);

    private static string FormatProvider(StageRuntimePlan plan) =>
        (plan.ExecutionProvider ?? ExecutionProviderKind.Cpu) switch
        {
            ExecutionProviderKind.Cpu => "cpu",
            ExecutionProviderKind.DirectMl => "dml",
            ExecutionProviderKind.Migraphx => "migraphx",
            ExecutionProviderKind.TensorRTRtx => "tensorrt-rtx",
            ExecutionProviderKind other => other.ToString().ToLowerInvariant(),
        };
}
