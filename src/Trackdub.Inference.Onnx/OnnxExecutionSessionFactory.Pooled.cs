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
    /// <summary>
    /// Pre-warms a pooled single session and immediately releases it, leaving the warm
    /// session in <see cref="InferenceSessionPool"/> for a later stage run. Use at startup
    /// or after plan resolution to amortise first-call latency.
    /// </summary>
    public static async Task WarmPooledSingleAsync(
        PooledSessionRequest request,
        string modelPath)
    {
        using SingleSessionLease _ = await CreatePooledSingleAsync(request, modelPath).ConfigureAwait(false);
    }

    public static async Task<SingleSessionLease> CreatePooledSingleAsync(
        PooledSessionRequest request,
        string modelPath)
    {
        ArgumentNullException.ThrowIfNull(request.EngineFamily);
        PooledSingleCreate create = await PreparePooledSingleAsync(request, modelPath).ConfigureAwait(false);
        return await AcquirePooledSingleLeaseAsync(create).ConfigureAwait(false);
    }

    /// <summary>Resolved create state for one pooled single-session acquisition.</summary>
    private sealed record PooledSingleCreate
    {
        public required PooledSessionRequest Request { get; init; }
        public required string ModelPath { get; init; }
        public required InferenceSessionPool Pool { get; init; }
        public required ExecutionProviders.ExecutionProviderBootstrapResult BootstrapResult { get; init; }
        public required WindowsMlExecutionDevicePolicy DevicePolicy { get; init; }
        public required SessionOptionsSelection OptionsSelection { get; init; }
        public required string RequestedProvider { get; init; }
        public required SessionPoolKey Key { get; init; }
    }

    private static async Task<PooledSingleCreate> PreparePooledSingleAsync(
        PooledSessionRequest request,
        string modelPath)
    {
        CancellationToken cancellationToken = request.CancellationToken;
        var bootstrapResult = await BootstrapTrackedAsync(request.Provider, allowDownloads: true, cancellationToken)
            .ConfigureAwait(false);
        WindowsMlExecutionDevicePolicy devicePolicy = await ResolveDevicePolicyAsync(cancellationToken)
            .ConfigureAwait(false);
        SessionOptionsSelection optionsSelection = CreateSessionOptions(new SessionOptionsRequest(
            ResolveSessionOptionsProvider(request.Provider, bootstrapResult.SelectedProvider),
            devicePolicy,
            request.AdditionalTrtOptions,
            TrtModelPath: modelPath));

        string optionsFingerprint =
            $"{BuildSessionOptionsFingerprint(optionsSelection.SelectedProvider, devicePolicy, request.AdditionalTrtOptions)}|trt-init-fallback:{request.AllowTrtInitFallback}";
        SessionPoolKey key;
        try
        {
            key = WithOpenVinoResidency(await SessionPoolKey.CreateAsync(
                request.EngineFamily,
                request.ModelId,
                request.Variant,
                optionsSelection.SelectedProvider,
                modelPath,
                deviceId: null,
                graphRole: "default",
                optionsFingerprint,
                cancellationToken).ConfigureAwait(false));
            if (request.AllowTrtInitFallback && optionsSelection.SelectedProvider is ExecutionProviderKind.TensorRTRtx)
            {
                // TRT RTX init can still fall back to DirectML/CPU inside the pool factory, after this
                // key is admitted; reserve for the fallback provider, not TRT RTX's measured factor.
                key = key with
                {
                    EstimatedVramMb = SessionPoolKey.EstimateVramMb(
                        modelPath, optionsSelection.SelectedProvider, providerMayFallBack: true),
                };
            }
        }
        catch
        {
            // Ownership only transfers to AcquirePooledSingleLeaseAsync's dispose holder after
            // this method returns; release the native options if key creation fails or cancels.
            optionsSelection.Options.Dispose();
            throw;
        }

        return new PooledSingleCreate
        {
            Request = request,
            ModelPath = modelPath,
            Pool = request.Pool ?? InferenceSessionPool.Shared,
            BootstrapResult = bootstrapResult,
            DevicePolicy = devicePolicy,
            OptionsSelection = optionsSelection,
            RequestedProvider = ProviderNames.FormatProviderLabel(request.Provider),
            Key = key,
        };
    }

    private static async Task<SingleSessionLease> AcquirePooledSingleLeaseAsync(PooledSingleCreate create)
    {
        PooledSessionRequest request = create.Request;
        SessionOptionsSelection optionsSelection = create.OptionsSelection;
        WindowsMlExecutionDevicePolicy devicePolicy = create.DevicePolicy;

        // Options ownership may transfer to a replacement SessionOptionsSelection when TRT init
        // falls back; keep a movable dispose target managed by a `using` scope.
        using var optionsHolder = new SessionOptionsDisposeHolder(optionsSelection.Options);
        SessionOptionsSelection leaseSelection = optionsSelection;
        bool useCatalogDevicePolicy = ProviderNames.ShouldUseCatalogDevicePolicy(devicePolicy, optionsSelection.SelectedProvider);
        ExecutionProviderKind optionsSelectedProvider = optionsSelection.SelectedProvider;

        SessionLease? poolLease = null;
        try
        {
            // The key is admitted for the pre-creation provider; reporting the provider the
            // fallback actually created the session with lets the pool move a CPU fallback's
            // accounting from the device budget to host RAM.
            poolLease = await create.Pool
                .GetLeaseReportingProviderAsync(
                    create.Key,
                    ct =>
                    {
                        (InferenceSession session, SessionOptionsSelection selection) =
                            CreateInferenceSessionWithTrtInitFallback(
                                create.ModelPath,
                                new SessionCreateRequest
                                {
                                    Selection = optionsSelection,
                                    DevicePolicy = devicePolicy,
                                    AdditionalTrtOptions = request.AdditionalTrtOptions,
                                    SessionFactory = request.SessionFactory,
                                    AllowTrtInitFallback = request.AllowTrtInitFallback,
                                },
                                ct);
                        leaseSelection = selection;
                        optionsSelectedProvider = selection.SelectedProvider;
                        useCatalogDevicePolicy = ProviderNames.ShouldUseCatalogDevicePolicy(devicePolicy, selection.SelectedProvider);
                        if (!ReferenceEquals(selection.Options, optionsHolder.Current))
                        {
                            optionsHolder.Current = selection.Options;
                        }

                        // Report where ORT actually placed the graph: a device EP that initialised
                        // but left every node on the CPU must be charged to host RAM.
                        ExecutionProviderKind placed = ResolveEffectiveProviderKindFromSession(
                            session,
                            selection.SelectedProvider,
                            useCatalogDevicePolicy);
                        TensorRtRtxTeardownGuard.ReconcileTracking(session, placed);
                        return Task.FromResult(new CreatedPoolSession(session, placed));
                    },
                    request.CancellationToken)
                .ConfigureAwait(false);

            ExecutionProviderKind effectiveProvider = ResolveEffectiveProviderKindFromSession(
                poolLease.Session,
                optionsSelectedProvider,
                useCatalogDevicePolicy);
            string selectedProvider = ProviderNames.FormatProviderLabel(effectiveProvider);
            string? epFallbackReason = BuildSessionOptionsFallbackReason(request.Provider, effectiveProvider, leaseSelection);
            string? bootstrapDetail = FormatBootstrapDetail(create.BootstrapResult.Detail, epFallbackReason);
            TrackLeasedSession(poolLease, effectiveProvider);

            return new SingleSessionLease(poolLease.Session, create.RequestedProvider, selectedProvider, bootstrapDetail)
            {
                PoolLease = poolLease,
                ResolvedPoolKey = create.Key,
            };
        }
        catch
        {
            poolLease?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Pins a pooled single session for engine residency (audit §3A) without holding the
    /// execution lease between calls. Use <see cref="PooledSingleSessionPin.AcquireAsync"/>
    /// for a short exclusive lease around each Run().
    /// </summary>
    public static async Task<PooledSingleSessionPin> PinPooledSingleAsync(
        PooledSessionRequest request,
        string modelPath)
    {
        InferenceSessionPool pool = request.Pool ?? InferenceSessionPool.Shared;
        CancellationToken cancellationToken = request.CancellationToken;

        SessionPoolKey? key = null;
        string requestedProvider = ProviderNames.FormatProviderLabel(request.Provider);
        string selectedProvider = requestedProvider;
        string? bootstrapDetail = null;

        async Task<SingleSessionLease> AcquireAsync(CancellationToken ct)
        {
            SingleSessionLease lease = await CreatePooledSingleAsync(
                request with { CancellationToken = ct, Pool = pool },
                modelPath).ConfigureAwait(false);
            if (key is null)
            {
                key = lease.ResolvedPoolKey;
                selectedProvider = lease.SelectedProvider;
                bootstrapDetail = lease.BootstrapDetail;
            }

            return lease;
        }

        // Establish the session and capture the pool key / provider metadata.
        using (await AcquireAsync(cancellationToken).ConfigureAwait(false))
        {
        }

        SessionPoolKey resolvedKey = key
            ?? throw new InvalidOperationException("Pooled single session did not resolve a pool key.");

        // Bound retries: when the pool is full of leased/pinned entries, GetLeaseAsync
        // returns an ephemeral lease that never reaches `entries`, so TryPinExisting
        // can never succeed — infinite session create/dispose churn.
        const int maxPinAttempts = 5;
        for (int attempt = 1; attempt <= maxPinAttempts; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (pool.TryPinExisting(resolvedKey, out SessionResidency? pinned) && pinned is not null)
            {
                return new PooledSingleSessionPin(pinned, AcquireAsync, requestedProvider, selectedProvider, bootstrapDetail);
            }

            if (attempt == maxPinAttempts)
            {
                break;
            }

            // Evicted before pin — recreate and retry with exponential backoff.
            // Retry delays: 50ms, 100ms, 200ms, 400ms (maxPinAttempts=5).
            int delayMs = 25 * (1 << attempt);
            await Task.Delay(delayMs, cancellationToken).ConfigureAwait(false);

            using (await AcquireAsync(cancellationToken).ConfigureAwait(false))
            {
            }
        }

        // Pool stayed at capacity (leased/pinned) across every retry: fall back to no
        // residency instead of failing engine init. Each AcquireAsync call still creates a
        // short, working (possibly ephemeral) lease — callers just lose the "stay resident
        // between calls" guarantee until pool pressure eases.
        return new PooledSingleSessionPin(
            new SessionResidency(static () => { }), AcquireAsync, requestedProvider, selectedProvider, bootstrapDetail);
    }

    /// <summary>The encoder/decoder graph paths for a pooled dual-model family.</summary>
    internal sealed record DualModelPaths
    {
        public required string Encoder { get; init; }
        public required string Decoder { get; init; }
    }

    public static async Task<WhisperSessionLease> CreatePooledWhisperAsync(
        PooledSessionRequest request,
        DualModelPaths paths)
    {
        ArgumentNullException.ThrowIfNull(request.EngineFamily);
        CancellationToken cancellationToken = request.CancellationToken;

        string requestedProvider = ProviderNames.FormatProviderLabel(request.Provider);
        ExecutionProviderKind selectedProviderKind = DowngradeTrtRtxForUnsupportedGraphs(
            request.Provider,
            [paths.Encoder, paths.Decoder],
            out string? graphFallbackReason);

        // TensorRT RTX can fail a Whisper session at engine build or load (CUDA OOM, importer
        // errors) after the pre-flight passed. Walk the same DirectML (Windows) then CPU chain the
        // single-session path uses, recording why each provider was abandoned.
        List<ExecutionProviderKind> attempts = [selectedProviderKind];
        if (request.AllowTrtInitFallback && selectedProviderKind is ExecutionProviderKind.TensorRTRtx)
        {
            attempts.AddRange(EnumerateTrtInitFallbackProviders());
        }

        string? abandonedReason = null;
        for (int attempt = 0; attempt < attempts.Count; attempt++)
        {
            ExecutionProviderKind attemptProvider = attempts[attempt];
            try
            {
                DualPooledLeasePair pair = await AcquirePooledWhisperPairAsync(
                    request, paths, attemptProvider).ConfigureAwait(false);

                return new WhisperSessionLease(
                    pair.EncoderLease.Session, pair.DecoderLease.Session,
                    requestedProvider, ProviderNames.FormatProviderLabel(pair.SelectedProvider),
                    MergeFallbackReasons(MergeFallbackReasons(graphFallbackReason, abandonedReason), pair.BootstrapDetail))
                {
                    EncoderPoolLease = pair.EncoderLease,
                    DecoderPoolLease = pair.DecoderLease
                };
            }
            catch (Exception ex)
            {
                if (attemptProvider is ExecutionProviderKind.TensorRTRtx)
                {
                    TensorRtRtxTeardownGuard.MarkPoisonedIfCudaOutOfMemory(ex);
                }

                if (!ShouldRetryWithNextProvider(attempt, attempts, cancellationToken, ex))
                {
                    throw;
                }

                await EvictAfterFailedWhisperInitAsync(request, attemptProvider)
                    .ConfigureAwait(false);
                abandonedReason = MergeFallbackReasons(
                    abandonedReason,
                    $"{ProviderNames.FormatProviderLabel(attemptProvider)} Whisper session init failed ({SummarizeExceptionMessage(ex)}); "
                    + $"fell back to {ProviderNames.FormatProviderLabel(attempts[attempt + 1])}.");
            }
        }

        throw new InvalidOperationException(
            $"Whisper session fallback chain for '{request.EngineFamily}' exhausted without a result.");
    }

    /// <summary>True while a recoverable init failure still has a next provider to try.</summary>
    private static bool ShouldRetryWithNextProvider(
        int attempt,
        List<ExecutionProviderKind> attempts,
        CancellationToken cancellationToken,
        Exception ex)
    {
        if (attempt >= attempts.Count - 1)
        {
            return false;
        }

        if (cancellationToken.IsCancellationRequested)
        {
            return false;
        }

        return IsRecoverableTrtFallbackInitFailure(ex);
    }

    /// <summary>
    /// Evicts the engine family after a failed init so the next provider's session is not
    /// pinned behind the failed entry. A failed eviction must not abort the fallback chain
    /// or replace the init failure.
    /// </summary>
    private static async Task EvictAfterFailedWhisperInitAsync(
        PooledSessionRequest request,
        ExecutionProviderKind attemptProvider)
    {
        InferenceSessionPool pool = request.Pool ?? InferenceSessionPool.Shared;
        try
        {
            await pool.EvictModelAsync(request.EngineFamily, cancellationToken: request.CancellationToken).ConfigureAwait(false);
        }
        catch (Exception evictionFailure) when (evictionFailure is not OperationCanceledException)
        {
            _logger?.LogWarning(
                evictionFailure,
                "Evicting {EngineFamily} after a failed {Provider} Whisper init failed; continuing fallback.",
                request.EngineFamily,
                ProviderNames.FormatProviderLabel(attemptProvider));
        }
    }

    /// <summary>Inputs for one pooled encoder/decoder bundle acquisition.</summary>
    private sealed record PooledDualCreate
    {
        public required PooledSessionRequest Request { get; init; }
        /// <summary>The provider this attempt/creation runs under (request or fallback).</summary>
        public required ExecutionProviderKind Provider { get; init; }
        public required string EncoderModelPath { get; init; }
        public required string DecoderModelPath { get; init; }
        /// <summary>Optional per-graph session log ids / TRT model paths for options build.</summary>
        public string? EncoderLogId { get; init; }
        public string? DecoderLogId { get; init; }
        public string? OptionsEncoderModelPath { get; init; }
        public string? OptionsDecoderModelPath { get; init; }
        public bool EnableEncoderCudaGraph { get; init; } = DefaultEnableCudaGraph;
    }

    private static async Task<DualPooledLeasePair> AcquirePooledWhisperPairAsync(
        PooledSessionRequest request,
        DualModelPaths paths,
        ExecutionProviderKind attemptProvider) =>
        await AcquirePooledDualPairAsync(new PooledDualCreate
        {
            Request = request,
            Provider = attemptProvider,
            EncoderModelPath = paths.Encoder,
            DecoderModelPath = paths.Decoder,
            EncoderLogId = ExtractLogId(paths.Encoder),
            DecoderLogId = ExtractLogId(paths.Decoder),
            OptionsEncoderModelPath = paths.Encoder,
            OptionsDecoderModelPath = paths.Decoder,
        }).ConfigureAwait(false);

    /// <summary>
    /// Bootstrap → dual options → dual pool keys → atomic bundle acquire. Shared by the
    /// Whisper, Nemotron, and Opus pooled encoder/decoder factories.
    /// </summary>
    private static async Task<DualPooledLeasePair> AcquirePooledDualPairAsync(PooledDualCreate create)
    {
        PooledSessionRequest request = create.Request;
        BootstrapContext bootstrap = await BootstrapForProviderAsync(create.Provider, request.CancellationToken)
            .ConfigureAwait(false);
        DualOptionsSelections selections = CreateDualOptionsSelections(new DualOptionsRequest
        {
            Provider = create.Provider,
            BootstrapResult = bootstrap.Bootstrap,
            DevicePolicy = bootstrap.DevicePolicy,
            AdditionalTrtEncoderOptions = request.AdditionalTrtEncoderOptions,
            AdditionalTrtDecoderOptions = request.AdditionalTrtDecoderOptions,
            EnableEncoderCudaGraph = create.EnableEncoderCudaGraph,
            EncoderLogId = create.EncoderLogId,
            DecoderLogId = create.DecoderLogId,
            EncoderModelPath = create.OptionsEncoderModelPath,
            DecoderModelPath = create.OptionsDecoderModelPath,
        });
        using SessionOptions encoderOptions = selections.Encoder.Options;
        using SessionOptions decoderOptions = selections.Decoder.Options;
        (SessionPoolKey encoderKey, SessionPoolKey decoderKey) = await BuildDualPooledKeysAsync(
            new DualPooledKeyRequest
            {
                EngineFamily = request.EngineFamily,
                EncoderModelPath = create.EncoderModelPath,
                DecoderModelPath = create.DecoderModelPath,
                Selections = selections,
                DevicePolicy = bootstrap.DevicePolicy,
                CancellationToken = request.CancellationToken,
                AdditionalTrtEncoderOptions = request.AdditionalTrtEncoderOptions,
                AdditionalTrtDecoderOptions = request.AdditionalTrtDecoderOptions,
                ModelId = request.ModelId,
                Variant = request.Variant,
                EnableEncoderCudaGraph = create.EnableEncoderCudaGraph,
            }).ConfigureAwait(false);

        return await AcquireDualPooledSessionsAsync(
            new DualPooledAcquireRequest
            {
                RequestedProvider = create.Provider,
                Bootstrap = bootstrap,
                Selections = selections,
                EncoderModelPath = create.EncoderModelPath,
                DecoderModelPath = create.DecoderModelPath,
                EncoderKey = encoderKey,
                DecoderKey = decoderKey,
                Pool = request.Pool ?? InferenceSessionPool.Shared,
                CancellationToken = request.CancellationToken,
                SessionFactory = request.SessionFactory,
            }).ConfigureAwait(false);
    }

    public static async Task<NemotronAsrSessionLease> CreatePooledNemotronAsrAsync(
        PooledSessionRequest request,
        DualModelPaths paths)
    {
        ArgumentNullException.ThrowIfNull(request.EngineFamily);

        ExecutionProviderKind provider = DowngradeTrtRtxForUnsupportedGraphs(
            request.Provider,
            [paths.Encoder, paths.Decoder],
            out _);

        // The encoder flag follows OnnxExecutionSessionFactory.DefaultEnableCudaGraph. Callers that
        // rebind fresh OrtValues (ParakeetTdtGreedyDecoder, smoke probes) rely on TensorRT-RTX
        // recapturing when buffer locations change. A caller opts out with enable_cuda_graph = 0
        // in its TRT options. Explicit per-caller graph options also flow into the pool-key
        // fingerprint so capture modes cannot collide.
        DualPooledLeasePair pair = await AcquirePooledDualPairAsync(new PooledDualCreate
        {
            Request = request,
            Provider = provider,
            EncoderModelPath = paths.Encoder,
            DecoderModelPath = paths.Decoder,
        }).ConfigureAwait(false);

        return new NemotronAsrSessionLease(
            pair.EncoderLease.Session,
            pair.DecoderLease.Session,
            pair.RequestedProviderLabel,
            ProviderNames.FormatProviderLabel(pair.SelectedProvider),
            pair.BootstrapDetail)
        {
            EncoderPoolLease = pair.EncoderLease,
            DecoderJointPoolLease = pair.DecoderLease,
        };
    }

    public static async Task<OpusSessionLease> CreatePooledOpusAsync(
        PooledSessionRequest request,
        DualModelPaths paths)
    {
        ArgumentNullException.ThrowIfNull(request.EngineFamily);

        ExecutionProviderKind provider = DowngradeTrtRtxForUnsupportedGraphs(
            request.Provider,
            [paths.Encoder, paths.Decoder],
            out _);

        DualPooledLeasePair pair = await AcquirePooledDualPairAsync(new PooledDualCreate
        {
            Request = request,
            Provider = provider,
            EncoderModelPath = paths.Encoder,
            DecoderModelPath = paths.Decoder,
            OptionsEncoderModelPath = paths.Encoder,
            OptionsDecoderModelPath = paths.Decoder,
        }).ConfigureAwait(false);

        return new OpusSessionLease(
            pair.EncoderLease.Session, pair.DecoderLease.Session,
            pair.RequestedProviderLabel, ProviderNames.FormatProviderLabel(pair.SelectedProvider), pair.BootstrapDetail)
        {
            SelectedProviderKind = pair.SelectedProvider,
            EncoderPoolLease = pair.EncoderLease,
            DecoderPoolLease = pair.DecoderLease
        };
    }
}
