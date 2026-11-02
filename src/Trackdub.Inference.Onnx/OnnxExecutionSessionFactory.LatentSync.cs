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
    /// <summary>Options selections for the four LatentSync diffusion graphs, in bundle order.</summary>
    private sealed record LatentSyncSelections(
        SessionOptionsSelection Unet,
        SessionOptionsSelection VaeEncoder,
        SessionOptionsSelection VaeDecoder,
        SessionOptionsSelection WhisperEncoder);

    /// <summary>Pool keys for the four LatentSync graphs, in bundle order.</summary>
    private sealed record LatentSyncPoolKeys(
        SessionPoolKey Unet,
        SessionPoolKey VaeEncoder,
        SessionPoolKey VaeDecoder,
        SessionPoolKey WhisperEncoder);

    /// <summary>Resolved state for one LatentSync pooled create.</summary>
    private sealed record LatentSyncCreate
    {
        public required PooledSessionRequest Request { get; init; }
        public required LatentSyncModelPaths Paths { get; init; }
        public BootstrapContext? Bootstrap { get; init; }
        public LatentSyncSelections? Selections { get; init; }
        public LatentSyncPoolKeys? Keys { get; init; }
    }

    /// <summary>Per-graph effective placement for the four LatentSync sessions.</summary>
    private sealed record LatentSyncEffectives(
        ExecutionProviderKind Unet,
        ExecutionProviderKind VaeEncoder,
        ExecutionProviderKind VaeDecoder,
        ExecutionProviderKind WhisperEncoder)
    {
        /// <summary>Shared effective provider, or Cpu when the graphs disagree.</summary>
        public ExecutionProviderKind Shared =>
            Unet == VaeEncoder && Unet == VaeDecoder && Unet == WhisperEncoder
                ? Unet
                : ExecutionProviderKind.Cpu;
    }

    public static async Task<LatentSyncSessionLease> CreatePooledLatentSyncAsync(
        PooledSessionRequest request,
        LatentSyncModelPaths paths)
    {
        ArgumentNullException.ThrowIfNull(request.EngineFamily);

        LatentSyncCreate create = await PrepareLatentSyncOptionsAsync(request, paths).ConfigureAwait(false);
        using SessionOptions unetOptions = create.Selections!.Unet.Options;
        using SessionOptions vaeEncOptions = create.Selections.VaeEncoder.Options;
        using SessionOptions vaeDecOptions = create.Selections.VaeDecoder.Options;
        using SessionOptions whisperOptions = create.Selections.WhisperEncoder.Options;
        create = create with { Keys = await BuildLatentSyncPoolKeysAsync(create).ConfigureAwait(false) };
        return await AcquireLatentSyncBundleAsync(create).ConfigureAwait(false);
    }

    /// <summary>All four graphs share one options provider; only the TRT model path differs.</summary>
    private static async Task<LatentSyncCreate> PrepareLatentSyncOptionsAsync(
        PooledSessionRequest request,
        LatentSyncModelPaths paths)
    {
        var bootstrapResult = await BootstrapTrackedAsync(
                request.Provider, allowDownloads: true, request.CancellationToken)
            .ConfigureAwait(false);
        WindowsMlExecutionDevicePolicy devicePolicy = await ResolveDevicePolicyAsync(request.CancellationToken)
            .ConfigureAwait(false);

        ExecutionProviderKind sessionProvider = ResolveSessionOptionsProvider(request.Provider, bootstrapResult.SelectedProvider);
        SessionOptionsSelection ForPath(string path) =>
            CreateSessionOptions(new SessionOptionsRequest(sessionProvider, devicePolicy, TrtModelPath: path));

        return new LatentSyncCreate
        {
            Request = request,
            Paths = paths,
            Bootstrap = new BootstrapContext(
                bootstrapResult, devicePolicy, ProviderNames.FormatProviderLabel(request.Provider)),
            Selections = new LatentSyncSelections(
                ForPath(paths.UnetModelPath),
                ForPath(paths.VaeEncoderModelPath),
                ForPath(paths.VaeDecoderModelPath),
                ForPath(paths.WhisperEncoderModelPath)),
        };
    }

    /// <summary>
    /// LatentSync's graphs pool under the fixed "latentsync-diffusion" family; the caller's
    /// engineFamily argument is validated but not used for keys (preserved behavior).
    /// </summary>
    private static async Task<LatentSyncPoolKeys> BuildLatentSyncPoolKeysAsync(LatentSyncCreate create)
    {
        WindowsMlExecutionDevicePolicy devicePolicy = create.Bootstrap!.DevicePolicy;
        string Fingerprint(SessionOptionsSelection selection) =>
            BuildSessionOptionsFingerprint(selection.SelectedProvider, devicePolicy, additionalTrtOptions: null);

        const string poolEngineFamily = "latentsync-diffusion";
        LatentSyncSelections selections = create.Selections!;
        LatentSyncModelPaths paths = create.Paths;
        SessionPoolKey[] keys = await SessionPoolKey.CreateManyAsync(
            [
                new SessionPoolKey.PooledKeyRequest(
                    poolEngineFamily, create.Request.ModelId, create.Request.Variant,
                    selections.Unet.SelectedProvider, paths.UnetModelPath, null, "unet", Fingerprint(selections.Unet)),
                new SessionPoolKey.PooledKeyRequest(
                    poolEngineFamily, create.Request.ModelId, create.Request.Variant,
                    selections.VaeEncoder.SelectedProvider, paths.VaeEncoderModelPath, null, "vae-encoder", Fingerprint(selections.VaeEncoder)),
                new SessionPoolKey.PooledKeyRequest(
                    poolEngineFamily, create.Request.ModelId, create.Request.Variant,
                    selections.VaeDecoder.SelectedProvider, paths.VaeDecoderModelPath, null, "vae-decoder", Fingerprint(selections.VaeDecoder)),
                new SessionPoolKey.PooledKeyRequest(
                    poolEngineFamily, create.Request.ModelId, create.Request.Variant,
                    selections.WhisperEncoder.SelectedProvider, paths.WhisperEncoderModelPath, null, "whisper-encoder", Fingerprint(selections.WhisperEncoder)),
            ],
            create.Request.CancellationToken).ConfigureAwait(false);
        return new LatentSyncPoolKeys(
            WithOpenVinoResidency(keys[0]),
            WithOpenVinoResidency(keys[1]),
            WithOpenVinoResidency(keys[2]),
            WithOpenVinoResidency(keys[3]));
    }

    /// <summary>Atomic bundle (audit §3A): acquire all four graphs or none.</summary>
    private static async Task<LatentSyncSessionLease> AcquireLatentSyncBundleAsync(LatentSyncCreate create)
    {
        PooledSessionRequest request = create.Request;
        InferenceSessionPool pool = request.Pool ?? InferenceSessionPool.Shared;
        WindowsMlExecutionDevicePolicy devicePolicy = create.Bootstrap!.DevicePolicy;
        LatentSyncSelections selections = create.Selections!;
        LatentSyncPoolKeys keys = create.Keys!;
        LatentSyncModelPaths paths = create.Paths;

        SessionLeaseBundle bundle = await pool.GetLeaseBundleAsync(
            [
                BuildGraphLeaseRequest(
                    new GraphLeaseSpec(keys.Unet, paths.UnetModelPath, selections.Unet),
                    devicePolicy,
                    request.SessionFactory),
                BuildGraphLeaseRequest(
                    new GraphLeaseSpec(keys.VaeEncoder, paths.VaeEncoderModelPath, selections.VaeEncoder),
                    devicePolicy,
                    request.SessionFactory),
                BuildGraphLeaseRequest(
                    new GraphLeaseSpec(keys.VaeDecoder, paths.VaeDecoderModelPath, selections.VaeDecoder),
                    devicePolicy,
                    request.SessionFactory),
                BuildGraphLeaseRequest(
                    new GraphLeaseSpec(keys.WhisperEncoder, paths.WhisperEncoderModelPath, selections.WhisperEncoder),
                    devicePolicy,
                    request.SessionFactory),
            ],
            request.CancellationToken).ConfigureAwait(false);

        try
        {
            return BuildLatentSyncLease(create, bundle);
        }
        catch
        {
            bundle.Dispose();
            throw;
        }
    }

    private static LatentSyncSessionLease BuildLatentSyncLease(LatentSyncCreate create, SessionLeaseBundle bundle)
    {
        SessionLease unetPoolLease = bundle.LeaseAt(0);
        SessionLease vaeEncPoolLease = bundle.LeaseAt(1);
        SessionLease vaeDecPoolLease = bundle.LeaseAt(2);
        SessionLease whisperPoolLease = bundle.LeaseAt(3);
        WindowsMlExecutionDevicePolicy devicePolicy = create.Bootstrap!.DevicePolicy;
        LatentSyncSelections selections = create.Selections!;
        ExecutionProviderKind requestedProvider = create.Request.Provider;

        LatentSyncEffectives effectives = new(
            ResolveEffectiveProvider(unetPoolLease.Session, selections.Unet, devicePolicy),
            ResolveEffectiveProvider(vaeEncPoolLease.Session, selections.VaeEncoder, devicePolicy),
            ResolveEffectiveProvider(vaeDecPoolLease.Session, selections.VaeDecoder, devicePolicy),
            ResolveEffectiveProvider(whisperPoolLease.Session, selections.WhisperEncoder, devicePolicy));
        string? bootstrapDetail = FormatBootstrapDetail(
            create.Bootstrap!.Bootstrap.Detail,
            BuildLatentSyncFallbackReasons(requestedProvider, selections, effectives));

        TrackLeasedSession(unetPoolLease, effectives.Unet);
        TrackLeasedSession(vaeEncPoolLease, effectives.VaeEncoder);
        TrackLeasedSession(vaeDecPoolLease, effectives.VaeDecoder);
        TrackLeasedSession(whisperPoolLease, effectives.WhisperEncoder);
        return new LatentSyncSessionLease(
            unetPoolLease.Session,
            vaeEncPoolLease.Session,
            vaeDecPoolLease.Session,
            whisperPoolLease.Session,
            create.Bootstrap!.RequestedProviderLabel,
            ProviderNames.FormatProviderLabel(effectives.Shared),
            bootstrapDetail)
        {
            PoolBundle = bundle,
            UNetPoolLease = unetPoolLease,
            VaeEncoderPoolLease = vaeEncPoolLease,
            VaeDecoderPoolLease = vaeDecPoolLease,
            WhisperEncoderPoolLease = whisperPoolLease,
        };
    }

    private static string? BuildLatentSyncFallbackReasons(
        ExecutionProviderKind requestedProvider,
        LatentSyncSelections selections,
        LatentSyncEffectives effectives) =>
        MergeFallbackReasons(
            MergeFallbackReasons(
                BuildSessionOptionsFallbackReason(requestedProvider, effectives.Unet, selections.Unet),
                BuildSessionOptionsFallbackReason(requestedProvider, effectives.VaeEncoder, selections.VaeEncoder)),
            MergeFallbackReasons(
                BuildSessionOptionsFallbackReason(requestedProvider, effectives.VaeDecoder, selections.VaeDecoder),
                BuildSessionOptionsFallbackReason(requestedProvider, effectives.WhisperEncoder, selections.WhisperEncoder)));

}
