using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging;
using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Dnnl;
using Trackdub.Inference.Onnx.EpContext;
using Trackdub.Inference.Onnx.ExecutionProviders;
using Trackdub.Inference.Onnx.Pool;
#if WINDOWS
using Trackdub.Inference.Onnx.WindowsMl;
#endif
using Trackdub.Inference.Onnx.Migraphx;
using Trackdub.Inference.Onnx.TensorRtRtx;
using Trackdub.Inference.Runtime.Migraphx;
using Trackdub.Inference.Runtime.TensorRtRtx;
using Trackdub.Inference.Runtime.WinMlCatalog;
using Trackdub.Inference.Runtime.Planning;
using Microsoft.ML.OnnxRuntime;

namespace Trackdub.Inference.Onnx;

internal static partial class OnnxExecutionSessionFactory
{
    /// <summary>The three Qwen3-ASR graphs, in bundle order.</summary>
    internal sealed record QwenAsrModelPaths(
        string Encoder,
        string DecoderInit,
        string DecoderStep);

    /// <summary>Per-role provider plan and resolved state for one Qwen3-ASR pooled create.</summary>
    private sealed record QwenAsrCreate
    {
        public required PooledSessionRequest Request { get; init; }
        public required QwenAsrModelPaths Paths { get; init; }
        public required ExecutionProviderKind EncoderProvider { get; init; }
        public required ExecutionProviderKind DecoderProvider { get; init; }
        public string? GraphFallbackReason { get; init; }
        public BootstrapContext? Bootstrap { get; init; }
        public DualOptionsSelections? Selections { get; init; }
        public QwenAsrPoolKeys? Keys { get; init; }
    }

    /// <summary>The three pool keys for a Qwen3-ASR bundle, in bundle order.</summary>
    private sealed record QwenAsrPoolKeys(
        SessionPoolKey Encoder,
        SessionPoolKey DecoderInit,
        SessionPoolKey DecoderStep);

    /// <summary>Per-graph effective placement for the three Qwen3-ASR sessions.</summary>
    private sealed record QwenAsrEffectives(
        ExecutionProviderKind Encoder,
        ExecutionProviderKind DecoderInit,
        ExecutionProviderKind DecoderStep)
    {
        /// <summary>Shared effective provider, or Cpu when the graphs disagree.</summary>
        public ExecutionProviderKind Shared =>
            Encoder == DecoderInit && Encoder == DecoderStep ? Encoder : ExecutionProviderKind.Cpu;
    }

    public static async Task<Qwen3AsrSessionLease> CreatePooledQwen3AsrAsync(
        PooledSessionRequest request,
        QwenAsrModelPaths paths)
    {
        ArgumentNullException.ThrowIfNull(request.EngineFamily);

        QwenAsrCreate create = ResolveQwenAsrProviders(request, paths);
        create = await PrepareQwenAsrOptionsAsync(create).ConfigureAwait(false);
        using SessionOptions encoderOptions = create.Selections!.Encoder.Options;
        using SessionOptions decoderOptions = create.Selections.Decoder.Options;
        create = create with { Keys = await BuildQwenAsrPoolKeysAsync(create).ConfigureAwait(false) };
        return await AcquireQwenAsrBundleAsync(create).ConfigureAwait(false);
    }

    /// <summary>
    /// Per-role TRT eligibility: a contrib-op decoder must not force the (often much larger)
    /// encoder off TensorRT RTX. Encoder is the cold-load hot path (qwen3-asr encoder ~711MB).
    /// </summary>
    private static QwenAsrCreate ResolveQwenAsrProviders(PooledSessionRequest request, QwenAsrModelPaths paths)
    {
        ExecutionProviderKind encoderProvider = DowngradeTrtRtxForUnsupportedGraphs(
            request.Provider, [paths.Encoder], out string? encoderGraphReason);
        ExecutionProviderKind decoderProvider = DowngradeTrtRtxForUnsupportedGraphs(
            request.Provider, [paths.DecoderInit, paths.DecoderStep], out string? decoderGraphReason);
        return new QwenAsrCreate
        {
            Request = request,
            Paths = paths,
            EncoderProvider = encoderProvider,
            DecoderProvider = decoderProvider,
            GraphFallbackReason = MergeFallbackReasons(encoderGraphReason, decoderGraphReason),
        };
    }

    /// <summary>Bootstraps under whichever role still targets TRT-RTX, then builds both selections.</summary>
    private static async Task<QwenAsrCreate> PrepareQwenAsrOptionsAsync(QwenAsrCreate create)
    {
        ExecutionProviderKind bootstrapProvider =
            create.EncoderProvider is ExecutionProviderKind.TensorRTRtx
                || create.DecoderProvider is ExecutionProviderKind.TensorRTRtx
                    ? ExecutionProviderKind.TensorRTRtx
                    : create.EncoderProvider;
        var bootstrapResult = await BootstrapTrackedAsync(
                bootstrapProvider, allowDownloads: true, create.Request.CancellationToken)
            .ConfigureAwait(false);
        WindowsMlExecutionDevicePolicy devicePolicy = await ResolveDevicePolicyAsync(create.Request.CancellationToken, create.Request.HasExplicitProviderPreference)
            .ConfigureAwait(false);
        SessionOptionsSelection encoderSelection = CreateSessionOptions(new SessionOptionsRequest(
            ResolveQwenRoleOptionsProvider(create.EncoderProvider, bootstrapProvider, bootstrapResult),
            devicePolicy,
            create.Request.AdditionalTrtEncoderOptions,
            TrtModelPath: create.Paths.Encoder));
        SessionOptionsSelection decoderSelection = CreateSessionOptions(new SessionOptionsRequest(
            ResolveQwenRoleOptionsProvider(create.DecoderProvider, bootstrapProvider, bootstrapResult),
            devicePolicy,
            create.Request.AdditionalTrtDecoderOptions,
            TrtModelPath: create.Paths.DecoderInit));
        return create with
        {
            Bootstrap = new BootstrapContext(
                bootstrapResult, devicePolicy, ProviderNames.FormatProviderLabel(create.Request.Provider)),
            Selections = new DualOptionsSelections(encoderSelection, decoderSelection),
        };
    }

    /// <summary>
    /// A role bootstrapped under TRT-RTX resolves its options provider from the bootstrap result;
    /// a role already downgraded keeps its own provider.
    /// </summary>
    private static ExecutionProviderKind ResolveQwenRoleOptionsProvider(
        ExecutionProviderKind roleProvider,
        ExecutionProviderKind bootstrapProvider,
        ExecutionProviders.ExecutionProviderBootstrapResult bootstrapResult) =>
        roleProvider == bootstrapProvider
            ? ResolveSessionOptionsProvider(roleProvider, bootstrapResult.SelectedProvider)
            : roleProvider;

    private static async Task<QwenAsrPoolKeys> BuildQwenAsrPoolKeysAsync(QwenAsrCreate create)
    {
        DualOptionsSelections selections = create.Selections!;
        WindowsMlExecutionDevicePolicy devicePolicy = create.Bootstrap!.DevicePolicy;
        string encoderFingerprint = BuildSessionOptionsFingerprint(
            selections.Encoder.SelectedProvider, devicePolicy, create.Request.AdditionalTrtEncoderOptions);
        string decoderFingerprint = BuildSessionOptionsFingerprint(
            selections.Decoder.SelectedProvider, devicePolicy, create.Request.AdditionalTrtDecoderOptions);
        SessionPoolKey[] keys = await SessionPoolKey.CreateManyAsync(
            [
                new SessionPoolKey.PooledKeyRequest(
                    create.Request.EngineFamily, create.Request.ModelId, create.Request.Variant,
                    selections.Encoder.SelectedProvider, create.Paths.Encoder, null, "encoder", encoderFingerprint),
                new SessionPoolKey.PooledKeyRequest(
                    create.Request.EngineFamily, create.Request.ModelId, create.Request.Variant,
                    selections.Decoder.SelectedProvider, create.Paths.DecoderInit, null, "decoder-init", decoderFingerprint),
                new SessionPoolKey.PooledKeyRequest(
                    create.Request.EngineFamily, create.Request.ModelId, create.Request.Variant,
                    selections.Decoder.SelectedProvider, create.Paths.DecoderStep, null, "decoder-step", decoderFingerprint),
            ],
            create.Request.CancellationToken).ConfigureAwait(false);
        return new QwenAsrPoolKeys(
            WithOpenVinoResidency(keys[0]),
            WithOpenVinoResidency(keys[1]),
            WithOpenVinoResidency(keys[2]));
    }

    /// <summary>Atomic bundle (audit §3A): never hold the encoder while waiting on decoders.</summary>
    private static async Task<Qwen3AsrSessionLease> AcquireQwenAsrBundleAsync(QwenAsrCreate create)
    {
        PooledSessionRequest request = create.Request;
        InferenceSessionPool pool = request.Pool ?? InferenceSessionPool.Shared;
        WindowsMlExecutionDevicePolicy devicePolicy = create.Bootstrap!.DevicePolicy;
        DualOptionsSelections selections = create.Selections!;
        QwenAsrPoolKeys keys = create.Keys!;

        SessionLeaseBundle bundle = await pool.GetLeaseBundleAsync(
            [
                BuildGraphLeaseRequest(
                    new GraphLeaseSpec(keys.Encoder, create.Paths.Encoder, selections.Encoder),
                    devicePolicy,
                    request.SessionFactory),
                BuildGraphLeaseRequest(
                    new GraphLeaseSpec(keys.DecoderInit, create.Paths.DecoderInit, selections.Decoder),
                    devicePolicy,
                    request.SessionFactory),
                BuildGraphLeaseRequest(
                    new GraphLeaseSpec(keys.DecoderStep, create.Paths.DecoderStep, selections.Decoder),
                    devicePolicy,
                    request.SessionFactory),
            ],
            request.CancellationToken).ConfigureAwait(false);

        try
        {
            return BuildQwenAsrLease(create, bundle);
        }
        catch
        {
            bundle.Dispose();
            throw;
        }
    }

    private static Qwen3AsrSessionLease BuildQwenAsrLease(QwenAsrCreate create, SessionLeaseBundle bundle)
    {
        SessionLease encoderPoolLease = bundle.LeaseAt(0);
        SessionLease decoderInitPoolLease = bundle.LeaseAt(1);
        SessionLease decoderStepPoolLease = bundle.LeaseAt(2);
        WindowsMlExecutionDevicePolicy devicePolicy = create.Bootstrap!.DevicePolicy;
        DualOptionsSelections selections = create.Selections!;

        QwenAsrEffectives effectives = new(
            ResolveEffectiveProvider(encoderPoolLease.Session, selections.Encoder, devicePolicy),
            ResolveEffectiveProvider(decoderInitPoolLease.Session, selections.Decoder, devicePolicy),
            ResolveEffectiveProvider(decoderStepPoolLease.Session, selections.Decoder, devicePolicy));
        string? bootstrapDetail = FormatBootstrapDetail(
            create.Bootstrap!.Bootstrap.Detail,
            MergeFallbackReasons(
                create.GraphFallbackReason,
                BuildQwenAsrFallbackReasons(create.Request.Provider, selections, effectives)));

        TrackLeasedSession(encoderPoolLease, effectives.Encoder);
        TrackLeasedSession(decoderInitPoolLease, effectives.DecoderInit);
        TrackLeasedSession(decoderStepPoolLease, effectives.DecoderStep);
        return new Qwen3AsrSessionLease(
            encoderPoolLease.Session,
            decoderInitPoolLease.Session,
            decoderStepPoolLease.Session,
            create.Bootstrap!.RequestedProviderLabel,
            ProviderNames.FormatProviderLabel(effectives.Shared),
            bootstrapDetail)
        {
            PoolBundle = bundle,
            EncoderPoolLease = encoderPoolLease,
            DecoderInitPoolLease = decoderInitPoolLease,
            DecoderStepPoolLease = decoderStepPoolLease,
        };
    }

    private static string? BuildQwenAsrFallbackReasons(
        ExecutionProviderKind requestedProvider,
        DualOptionsSelections selections,
        QwenAsrEffectives effectives) =>
        MergeFallbackReasons(
            BuildSessionOptionsFallbackReason(requestedProvider, effectives.Encoder, selections.Encoder),
            MergeFallbackReasons(
                BuildSessionOptionsFallbackReason(requestedProvider, effectives.DecoderInit, selections.Decoder),
                BuildSessionOptionsFallbackReason(requestedProvider, effectives.DecoderStep, selections.Decoder)));
}
