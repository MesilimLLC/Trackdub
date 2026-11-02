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
    private static string ExtractLogId(string modelPath)
    {
        string fileName = Path.GetFileNameWithoutExtension(modelPath);
        string[] parts = modelPath.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return parts.Length >= 3 ? $"{parts[^3]}/{fileName}" : fileName;
    }

    private const string DataRootEnvironmentVariable = "TRACKDUB_DATA_ROOT";
    private const string CacheRootEnvironmentVariable = "TRACKDUB_CACHE_ROOT";
    private const string EngineCacheRootEnvironmentVariable = "TRACKDUB_ENGINE_CACHE_ROOT";

    private static readonly object InitializeLock = new();
    private static int initializeCompleted;

    private static IExecutionProviderBootstrapper _bootstrapper = GetPlatformBootstrapper();
    private static IOpenVinoAvailabilityProvider _openVinoAvailability = NullOpenVinoAvailabilityProvider.Instance;
    private static IWindowsMlEpDevicePolicyProvider _devicePolicyProvider = NullWindowsMlEpDevicePolicyProvider.Instance;
    private static ILogger? _logger;
    private static readonly object TensorRtIdentityLibraryLoadLock = new();
    private static IntPtr _tensorRtIdentityLibraryHandle;
    private static readonly ConcurrentDictionary<string, byte> WarnedUnmappedCatalogEpNames =
        new(StringComparer.OrdinalIgnoreCase);

    internal static bool UseOpenVinoCpuProxy => _openVinoAvailability.UseOpenVinoCpuProxy;

    /// <summary>
    /// One-time process initialization. The first successful call wins; later calls are ignored.
    /// </summary>
    internal static void Initialize(
        IExecutionProviderBootstrapper bootstrapper,
        IWindowsMlEpDevicePolicyProvider? devicePolicyProvider = null,
        ILogger? logger = null,
        IOpenVinoAvailabilityProvider? openVinoAvailabilityProvider = null)
    {
        lock (InitializeLock)
        {
            if (initializeCompleted != 0)
            {
                return;
            }

            _bootstrapper = bootstrapper ?? throw new ArgumentNullException(nameof(bootstrapper));
            _devicePolicyProvider = devicePolicyProvider ?? NullWindowsMlEpDevicePolicyProvider.Instance;
            _openVinoAvailability = openVinoAvailabilityProvider ?? NullOpenVinoAvailabilityProvider.Instance;
            _logger = logger;
            initializeCompleted = 1;
        }
    }

    /// <summary>
    /// Resets process-level static state for test isolation. Not for production use.
    /// </summary>
    internal static void ResetForTests()
    {
        lock (InitializeLock)
        {
            initializeCompleted = 0;
            _bootstrapper = GetPlatformBootstrapper();
            _devicePolicyProvider = NullWindowsMlEpDevicePolicyProvider.Instance;
            _openVinoAvailability = NullOpenVinoAvailabilityProvider.Instance;
            _logger = null;
            WarnedUnmappedCatalogEpNames.Clear();
        }
    }

    private sealed record BootstrapContext(
        ExecutionProviders.ExecutionProviderBootstrapResult Bootstrap,
        WindowsMlExecutionDevicePolicy DevicePolicy,
        string RequestedProviderLabel);

    private sealed record DualOptionsSelections(
        SessionOptionsSelection Encoder,
        SessionOptionsSelection Decoder);

    private sealed record DualSessionMetadata(
        ExecutionProviderKind SelectedProvider,
        ExecutionProviderKind EncoderProvider,
        ExecutionProviderKind DecoderProvider,
        string? BootstrapDetail);

    private sealed record DualPooledLeasePair(
        SessionLeaseBundle Bundle,
        string RequestedProviderLabel,
        ExecutionProviderKind SelectedProvider,
        string? BootstrapDetail)
    {
        /// <summary>Caller-order index 0 (encoder). See <c>AcquireDualPooledSessionsAsync</c>.</summary>
        public SessionLease EncoderLease => Bundle.LeaseAt(0);

        /// <summary>Caller-order index 1 (decoder).</summary>
        public SessionLease DecoderLease => Bundle.LeaseAt(1);
    }

    /// <summary>
    /// Inputs for one pooled session create: pool identity, provider, pooling knobs, and
    /// optional TRT/session-factory overrides. Model paths stay as method arguments so each
    /// create keeps its graph arity self-documenting.
    /// </summary>
    internal sealed record PooledSessionRequest
    {
        public required string EngineFamily { get; init; }
        public required ExecutionProviderKind Provider { get; init; }
        public required CancellationToken CancellationToken { get; init; }
        public InferenceSessionPool? Pool { get; init; }
        public string? ModelId { get; init; }
        public string? Variant { get; init; }
        public IReadOnlyDictionary<string, string>? AdditionalTrtOptions { get; init; }
        public IReadOnlyDictionary<string, string>? AdditionalTrtEncoderOptions { get; init; }
        public IReadOnlyDictionary<string, string>? AdditionalTrtDecoderOptions { get; init; }
        public Func<string, SessionOptions, InferenceSession>? SessionFactory { get; init; }
        public bool AllowTrtInitFallback { get; init; } = true;
    }

    /// <summary>The four LatentSync diffusion graphs, in bundle order.</summary>
    internal sealed record LatentSyncModelPaths(
        string UnetModelPath,
        string VaeEncoderModelPath,
        string VaeDecoderModelPath,
        string WhisperEncoderModelPath);

    /// <summary>Optional knobs for a non-pooled single-session create.</summary>
    internal sealed record SingleSessionOptions
    {
        public IReadOnlyDictionary<string, string>? AdditionalTrtOptions { get; init; }
        public bool AllowTrtInitFallback { get; init; } = true;
    }

    /// <summary>Inputs for the single-session create/fallback chain.</summary>
    internal sealed record SessionCreateRequest
    {
        public required SessionOptionsSelection Selection { get; init; }
        public required WindowsMlExecutionDevicePolicy DevicePolicy { get; init; }
        public IReadOnlyDictionary<string, string>? AdditionalTrtOptions { get; init; }
        public Func<string, SessionOptions, InferenceSession>? SessionFactory { get; init; }
        public bool AllowTrtInitFallback { get; init; } = true;
        public string? LogId { get; init; }
    }

    /// <summary>Inputs for building the encoder/decoder <see cref="SessionOptionsSelection"/> pair.</summary>
    private sealed record DualOptionsRequest
    {
        public required ExecutionProviderKind Provider { get; init; }
        public required ExecutionProviders.ExecutionProviderBootstrapResult BootstrapResult { get; init; }
        public required WindowsMlExecutionDevicePolicy DevicePolicy { get; init; }
        public IReadOnlyDictionary<string, string>? AdditionalTrtEncoderOptions { get; init; }
        public IReadOnlyDictionary<string, string>? AdditionalTrtDecoderOptions { get; init; }
        public bool EnableEncoderCudaGraph { get; init; }
        public string? EncoderLogId { get; init; }
        public string? DecoderLogId { get; init; }
        public string? EncoderModelPath { get; init; }
        public string? DecoderModelPath { get; init; }
    }

    /// <summary>Inputs for building the encoder/decoder pool keys.</summary>
    private sealed record DualPooledKeyRequest
    {
        public required string EngineFamily { get; init; }
        public required string EncoderModelPath { get; init; }
        public required string DecoderModelPath { get; init; }
        public required DualOptionsSelections Selections { get; init; }
        public required WindowsMlExecutionDevicePolicy DevicePolicy { get; init; }
        public required CancellationToken CancellationToken { get; init; }
        public IReadOnlyDictionary<string, string>? AdditionalTrtEncoderOptions { get; init; }
        public IReadOnlyDictionary<string, string>? AdditionalTrtDecoderOptions { get; init; }
        public string? ModelId { get; init; }
        public string? Variant { get; init; }
        public bool EnableEncoderCudaGraph { get; init; }
        public string? EncoderLogId { get; init; }
        public string? DecoderLogId { get; init; }
    }

    /// <summary>Inputs for one atomic dual-graph bundle acquisition.</summary>
    private sealed record DualPooledAcquireRequest
    {
        public required ExecutionProviderKind RequestedProvider { get; init; }
        public required BootstrapContext Bootstrap { get; init; }
        public required DualOptionsSelections Selections { get; init; }
        public required string EncoderModelPath { get; init; }
        public required string DecoderModelPath { get; init; }
        public required SessionPoolKey EncoderKey { get; init; }
        public required SessionPoolKey DecoderKey { get; init; }
        public required InferenceSessionPool Pool { get; init; }
        public required CancellationToken CancellationToken { get; init; }
        public Func<string, SessionOptions, InferenceSession>? SessionFactory { get; init; }
    }

    /// <summary>One graph's key/path/selection triple inside a multi-graph bundle.</summary>
    private sealed record GraphLeaseSpec(
        SessionPoolKey Key,
        string ModelPath,
        SessionOptionsSelection Selection);

    /// <summary>Builds the lease request for one graph: create via options, report effective placement.</summary>
    private static SessionLeaseRequest BuildGraphLeaseRequest(
        GraphLeaseSpec spec,
        WindowsMlExecutionDevicePolicy devicePolicy,
        Func<string, SessionOptions, InferenceSession>? sessionFactory) =>
        new(
            spec.Key,
            ct => Task.FromResult(CreateSession(
                spec.ModelPath,
                ForGraphSelection(spec.Selection, devicePolicy, sessionFactory),
                ct)),
            session => ResolveEffectiveProvider(session, spec.Selection, devicePolicy));

    /// <summary>Where ORT actually placed this graph (device EP that left all nodes on CPU reports Cpu).</summary>
    private static ExecutionProviderKind ResolveEffectiveProvider(
        InferenceSession session,
        SessionOptionsSelection selection,
        WindowsMlExecutionDevicePolicy devicePolicy) =>
        ResolveEffectiveProviderKindFromSession(
            session,
            selection.SelectedProvider,
            ProviderNames.ShouldUseCatalogDevicePolicy(devicePolicy, selection.SelectedProvider));

    /// <summary>Charges a created session to CPU admission when needed and reconciles TRT-RTX tracking.</summary>
    private static void TrackCreatedSession(InferenceSession session, ExecutionProviderKind effective)
    {
        CpuExecutionAdmission.Shared.RegisterSession(session, effective);
        TensorRtRtxTeardownGuard.ReconcileTracking(session, effective);
    }

    /// <summary>Charges a leased session to CPU admission when needed and reconciles TRT-RTX tracking.</summary>
    private static void TrackLeasedSession(SessionLease lease, ExecutionProviderKind effective) =>
        TrackCreatedSession(lease.Session, effective);

    private static async Task<BootstrapContext> BootstrapForProviderAsync(
        ExecutionProviderKind provider,
        CancellationToken cancellationToken)
    {
        var bootstrapResult = await BootstrapTrackedAsync(provider, allowDownloads: true, cancellationToken)
            .ConfigureAwait(false);
        WindowsMlExecutionDevicePolicy devicePolicy = await ResolveDevicePolicyAsync(cancellationToken)
            .ConfigureAwait(false);
        return new BootstrapContext(bootstrapResult, devicePolicy, ProviderNames.FormatProviderLabel(provider));
    }

    /// <summary>
    /// Readiness-only registration for smoke probes: never downloads, and reports the provider
    /// the bootstrapper can actually select. Callers must treat <see cref="ExecutionProviders.ExecutionProviderBootstrapResult.IsRequestFulfilled"/>
    /// of <see langword="false"/> as an unproven pair — registration alone is not readiness.
    /// </summary>
    internal static async Task<ExecutionProviders.ExecutionProviderBootstrapResult> BootstrapForSmokeAsync(
        ExecutionProviderKind provider,
        CancellationToken cancellationToken)
    {
        // Native ONNX Runtime / DirectML provider DLLs must be resolvable before the first
        // OrtEnv touch; smoke runs in headless hosts that otherwise hit OrtEnv via discovery
        // first and can no longer append DirectML.
#if WINDOWS
        WindowsMlOnnxRuntimeNativeResolver.EnsureInitialized();
#endif
        return await BootstrapTrackedAsync(provider, allowDownloads: false, cancellationToken)
            .ConfigureAwait(false);
    }

    private static readonly ConcurrentDictionary<string, byte> ReportedCaptureSkips =
        new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Surfaces a skipped EP-context capture once per model per process. A skip means this
    /// process builds TensorRT engines from source instead of deserializing them, which costs
    /// tens of seconds on multi-GB graphs and repeats on every launch until resolved.
    /// </summary>
    private static void ReportCaptureSkip(string modelPath, string reason)
    {
        string fullPath = Path.GetFullPath(modelPath);
        if (!ReportedCaptureSkips.TryAdd(fullPath, 0))
        {
            return;
        }

        DiagnosticsLogger?.LogWarning(
            $"TensorRT-RTX engine capture skipped for '{Path.GetFileName(fullPath)}': {reason} " +
            "Engines will be rebuilt from source on each launch until this is resolved.");
    }

    /// <summary>
    /// Optional process-wide warning sink installed by the composition root so engine-capture
    /// skips reach the operator log rather than only a debug trace.
    /// </summary>
    internal static IApplicationLogger? DiagnosticsLogger { get; set; }

    /// <summary>
    /// Explains why TensorRT-RTX engine capture is (or is not) available for
    /// <paramref name="modelPath"/>. Public so operator tooling can report the same decision the
    /// session factory makes, without needing a session to have been created yet.
    /// </summary>
    public static bool TryGetEngineCaptureSkipReason(string modelPath, out string reason) =>
        EpContext.EpContextCapture.TryGetSkipReason(modelPath, out reason);

    /// <summary>Minimum free bytes the capture path requires before it will write an artifact.</summary>
    public static long EngineCaptureRequiredFreeBytes(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        return EpContext.EpContextArtifact.GetSourceTotalBytes(modelPath) * 2;
    }

    /// <summary>
    /// Bootstraps an execution provider while attributing the elapsed time to the
    /// <c>provider-bootstrap</c> phase, so cold-start evidence separates provider/runtime
    /// registration cost from session creation and inference.
    /// </summary>
    private static async Task<ExecutionProviders.ExecutionProviderBootstrapResult> BootstrapTrackedAsync(
        ExecutionProviderKind provider,
        bool allowDownloads,
        CancellationToken cancellationToken)
    {
        using IDisposable phase = BenchmarkPhaseCapture.Start("provider-bootstrap");
        return await _bootstrapper.BootstrapAsync(provider, allowDownloads, cancellationToken)
            .ConfigureAwait(false);
    }

    private static DualOptionsSelections CreateDualOptionsSelections(DualOptionsRequest request)
    {
        ExecutionProviderKind sessionProvider =
            ResolveSessionOptionsProvider(request.Provider, request.BootstrapResult.SelectedProvider);
        return new DualOptionsSelections(
            CreateSessionOptions(new SessionOptionsRequest(sessionProvider, request.DevicePolicy, request.AdditionalTrtEncoderOptions, request.EnableEncoderCudaGraph, request.EncoderLogId, request.EncoderModelPath)),
            CreateSessionOptions(new SessionOptionsRequest(sessionProvider, request.DevicePolicy, request.AdditionalTrtDecoderOptions, EnableCudaGraph: false, request.DecoderLogId, request.DecoderModelPath)));
    }

    private static DualSessionMetadata ResolveDualSessionMetadata(
        DualSessionPair pair)
    {
        ExecutionProviderKind encoderEffective = ResolveEffectiveProviderKindFromSession(
            pair.EncoderSession,
            pair.Selections.Encoder.SelectedProvider,
            ProviderNames.ShouldUseCatalogDevicePolicy(pair.Bootstrap.DevicePolicy, pair.Selections.Encoder.SelectedProvider));
        ExecutionProviderKind decoderEffective = ResolveEffectiveProviderKindFromSession(
            pair.DecoderSession,
            pair.Selections.Decoder.SelectedProvider,
            ProviderNames.ShouldUseCatalogDevicePolicy(pair.Bootstrap.DevicePolicy, pair.Selections.Decoder.SelectedProvider));
        ExecutionProviderKind resolvedSelectedProvider = ResolveSharedSelectedProvider(
            pair.RequestedProvider, encoderEffective, decoderEffective);
        string? encoderFallbackReason = BuildSessionOptionsFallbackReason(
            pair.RequestedProvider, encoderEffective, pair.Selections.Encoder);
        string? decoderFallbackReason = BuildSessionOptionsFallbackReason(
            pair.RequestedProvider, decoderEffective, pair.Selections.Decoder);
        return new DualSessionMetadata(
            resolvedSelectedProvider,
            encoderEffective,
            decoderEffective,
            FormatBootstrapDetail(
                pair.Bootstrap.Bootstrap.Detail,
                MergeFallbackReasons(encoderFallbackReason, decoderFallbackReason)));
    }

    /// <summary>One created encoder/decoder pair plus the bootstrap/selections it was built with.</summary>
    private sealed record DualSessionPair
    {
        public required ExecutionProviderKind RequestedProvider { get; init; }
        public required BootstrapContext Bootstrap { get; init; }
        public required DualOptionsSelections Selections { get; init; }
        public required InferenceSession EncoderSession { get; init; }
        public required InferenceSession DecoderSession { get; init; }
    }

    private static async Task<(SessionPoolKey EncoderKey, SessionPoolKey DecoderKey)> BuildDualPooledKeysAsync(
        DualPooledKeyRequest request)
    {
        string encoderFingerprint = BuildSessionOptionsFingerprint(
            request.Selections.Encoder.SelectedProvider, request.DevicePolicy, request.AdditionalTrtEncoderOptions, request.EnableEncoderCudaGraph);
        string decoderFingerprint = BuildSessionOptionsFingerprint(
            request.Selections.Decoder.SelectedProvider, request.DevicePolicy, request.AdditionalTrtDecoderOptions);
        SessionPoolKey[] keys = await SessionPoolKey.CreateManyAsync(
            [
                new SessionPoolKey.PooledKeyRequest(
                    request.EngineFamily, request.ModelId, request.Variant, request.Selections.Encoder.SelectedProvider,
                    request.EncoderModelPath, null, "encoder", encoderFingerprint),
                new SessionPoolKey.PooledKeyRequest(
                    request.EngineFamily, request.ModelId, request.Variant, request.Selections.Decoder.SelectedProvider,
                    request.DecoderModelPath, null, "decoder", decoderFingerprint),
            ],
            request.CancellationToken).ConfigureAwait(false);
        return (WithOpenVinoResidency(keys[0]), WithOpenVinoResidency(keys[1]));
    }

    private static async Task<DualPooledLeasePair> AcquireDualPooledSessionsAsync(
        DualPooledAcquireRequest request)
    {
        // Atomic bundle (audit §3A): never hold the encoder lease while blocked on the decoder.
        // Lease order in the bundle is caller order (encoder, decoder). Each graph reports its
        // placement at creation: if the decoder then fails, a CPU-placed encoder must not stay
        // tracked as TensorRT-RTX, or a poisoned runtime would skip its disposal and leak it.
        WindowsMlExecutionDevicePolicy devicePolicy = request.Bootstrap.DevicePolicy;
        SessionLeaseBundle bundle = await request.Pool.GetLeaseBundleAsync(
            [
                BuildGraphLeaseRequest(
                    new GraphLeaseSpec(request.EncoderKey, request.EncoderModelPath, request.Selections.Encoder),
                    devicePolicy,
                    request.SessionFactory),
                BuildGraphLeaseRequest(
                    new GraphLeaseSpec(request.DecoderKey, request.DecoderModelPath, request.Selections.Decoder),
                    devicePolicy,
                    request.SessionFactory),
            ],
            request.CancellationToken).ConfigureAwait(false);

        try
        {
            DualSessionMetadata metadata = ResolveDualSessionMetadata(
                new DualSessionPair
                {
                    RequestedProvider = request.RequestedProvider,
                    Bootstrap = request.Bootstrap,
                    Selections = request.Selections,
                    EncoderSession = bundle.LeaseAt(0).Session,
                    DecoderSession = bundle.LeaseAt(1).Session,
                });
            TrackLeasedSession(bundle.LeaseAt(0), metadata.EncoderProvider);
            TrackLeasedSession(bundle.LeaseAt(1), metadata.DecoderProvider);
            return new DualPooledLeasePair(
                bundle,
                request.Bootstrap.RequestedProviderLabel,
                metadata.SelectedProvider,
                metadata.BootstrapDetail);
        }
        catch
        {
            bundle.Dispose();
            throw;
        }
    }

    /// <summary>Builds the create/fallback inputs for one graph's options selection.</summary>
    private static SessionCreateRequest ForGraphSelection(
        SessionOptionsSelection selection,
        WindowsMlExecutionDevicePolicy devicePolicy,
        Func<string, SessionOptions, InferenceSession>? sessionFactory) =>
        new()
        {
            Selection = selection,
            DevicePolicy = devicePolicy,
            SessionFactory = sessionFactory,
        };

    public static async Task<SingleSessionLease> CreateSingleAsync(
        string modelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        SingleSessionOptions? options = null)
    {
        string requestedProvider = ProviderNames.FormatProviderLabel(provider);
        var bootstrapResult = await BootstrapTrackedAsync(provider, allowDownloads: true, cancellationToken)
            .ConfigureAwait(false);
        WindowsMlExecutionDevicePolicy devicePolicy = await ResolveDevicePolicyAsync(cancellationToken)
            .ConfigureAwait(false);
        string logId = ExtractLogId(modelPath);
        SessionOptionsSelection sessionOptionsSelection = CreateSessionOptions(new SessionOptionsRequest(
            ResolveSessionOptionsProvider(provider, bootstrapResult.SelectedProvider),
            devicePolicy,
            options?.AdditionalTrtOptions,
            LogId: logId,
            TrtModelPath: modelPath));
        InferenceSession? session = null;
        try
        {
            var createRequest = new SessionCreateRequest
            {
                Selection = sessionOptionsSelection,
                DevicePolicy = devicePolicy,
                AdditionalTrtOptions = options?.AdditionalTrtOptions,
                AllowTrtInitFallback = options?.AllowTrtInitFallback ?? true,
                LogId = logId,
            };
            (session, sessionOptionsSelection) = CreateInferenceSessionWithTrtInitFallback(
                modelPath,
                createRequest,
                cancellationToken);
            bool useCatalogDevicePolicy = ProviderNames.ShouldUseCatalogDevicePolicy(devicePolicy, sessionOptionsSelection.SelectedProvider);
            ExecutionProviderKind effectiveProvider = ResolveEffectiveProviderKindFromSession(
                session,
                sessionOptionsSelection.SelectedProvider,
                useCatalogDevicePolicy);
            string selectedProvider = ProviderNames.FormatProviderLabel(effectiveProvider);
            string? bootstrapDetail = FormatBootstrapDetail(
                bootstrapResult.Detail,
                BuildSessionOptionsFallbackReason(provider, effectiveProvider, sessionOptionsSelection));
            CpuExecutionAdmission.Shared.RegisterSession(session, effectiveProvider);
            TensorRtRtxTeardownGuard.ReconcileTracking(session, effectiveProvider);
            return new SingleSessionLease(
                session,
                requestedProvider,
                selectedProvider,
                bootstrapDetail);
        }
        catch (Exception ex)
        {
            // CreateSession tracks TensorRT-RTX sessions before this catch runs. Mark a CUDA
            // OOM first, or DisposeSafely still calls native Dispose on that tracked session.
            ReleaseSessionsAfterCreateFailure(ex, session, decoder: null);
            throw;
        }
        finally
        {
            sessionOptionsSelection.Options.Dispose();
        }
    }

    /// <summary>Resolved EP-context load plan for one session create.</summary>
    private sealed record SessionLoadPlan
    {
        public required string ModelPath { get; init; }
        public required string LoadPath { get; init; }
        public required bool CaptureCandidate { get; init; }
        public required ExecutionProviderKind SelectedProvider { get; init; }
        public IReadOnlyDictionary<string, string>? TrtOptions { get; init; }
    }

    private static InferenceSession CreateSession(
        string modelPath,
        SessionCreateRequest request,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var phase = BenchmarkPhaseCapture.Start("onnx-session-create");
        SessionLoadPlan plan = ResolveSessionLoadPlan(
            modelPath, request.Selection.Options, request.SessionFactory, request.Selection.SelectedProvider);
        InferenceSession session = CreateSessionFromPlan(plan, request.Selection.Options, request.SessionFactory);

        if (request.Selection.SelectedProvider is ExecutionProviderKind.TensorRTRtx)
        {
            TensorRtRtxTeardownGuard.Track(session);
        }

#if WINDOWS
        ApplyEfficiencyPolicyToSession(session, request.Selection.Options, plan, request.DevicePolicy);
#endif

        return session;
    }

    /// <summary>
    /// Resolves the EP-context artifact load path and whether this create is a capture
    /// candidate. EP-context artifacts embed a TensorRT-RTX-serialized engine that only that
    /// EP can deserialize; only TRT-RTX sessions may load one (see EpContextWarmupService).
    /// </summary>
    private static SessionLoadPlan ResolveSessionLoadPlan(
        string modelPath,
        SessionOptions options,
        Func<string, SessionOptions, InferenceSession>? sessionFactory,
        ExecutionProviderKind selectedProvider)
    {
        IReadOnlyDictionary<string, string>? trtOptions = selectedProvider is ExecutionProviderKind.TensorRTRtx
            ? TryGetAppendedTensorRtRtxOptions(options)
            : null;
        string loadPath = trtOptions is not null
            ? EpContext.EpContextLoadPathResolver.TryResolveLoadPath(modelPath, trtOptions) ?? modelPath
            : modelPath;
        // Without a current artifact, a large graph's engine build is captured as one, so the next
        // process loads engines instead of rebuilding them (minutes for multi-GB models).
        bool captureCandidate = sessionFactory is null
            && trtOptions is not null
            && string.Equals(loadPath, modelPath, StringComparison.Ordinal);
        if (captureCandidate && EpContext.EpContextCapture.TryGetSkipReason(modelPath, out string captureSkipReason))
        {
            // A skipped capture is not harmless: every later launch rebuilds these engines.
            // Say so once per model at warning level instead of a trace-only whisper.
            ReportCaptureSkip(modelPath, captureSkipReason);
        }

        return new SessionLoadPlan
        {
            ModelPath = modelPath,
            LoadPath = loadPath,
            CaptureCandidate = captureCandidate,
            SelectedProvider = selectedProvider,
            TrtOptions = trtOptions,
        };
    }

    /// <summary>Builds the session from the load plan; poisons the TRT-RTX guard on OOM.</summary>
    private static InferenceSession CreateSessionFromPlan(
        SessionLoadPlan plan,
        SessionOptions options,
        Func<string, SessionOptions, InferenceSession>? sessionFactory)
    {
        try
        {
            return
                plan.CaptureCandidate && EpContext.EpContextCapture.ShouldCapture(plan.ModelPath)
                    ? EpContext.EpContextCapture.CreateSession(plan.ModelPath, options, plan.TrtOptions!, static (path, o) => new InferenceSession(path, o))
                    : sessionFactory is null
                        ? new InferenceSession(plan.LoadPath, options)
                        : sessionFactory(plan.LoadPath, options);
        }
        catch (Exception ex) when (plan.SelectedProvider is ExecutionProviderKind.TensorRTRtx)
        {
            TensorRtRtxTeardownGuard.MarkPoisonedIfCudaOutOfMemory(ex);
            throw;
        }
    }

#if WINDOWS
    /// <summary>
    /// MS "Efficiency tips for Windows ML inference": express efficiency intent through ORT
    /// (ep.dynamic.workload_type) for NPU catalog sessions under efficiency-flavored device
    /// policies. Only NPU EPs act on the signal; the call is a no-op elsewhere but is gated
    /// to NPU providers to keep the no-op surface minimal. Outcomes are logged, never assumed.
    /// This hook runs strictly post-registration: an NPU catalog session can only exist here
    /// if WindowsMlProviderRegistrationPolicy already readied + TryRegister()-ed the EP
    /// (MS "Register Windows ML execution providers") — no new install/registration logic
    /// is introduced, and the hook never fires when the EP was not registered.
    /// </summary>
    private static void ApplyEfficiencyPolicyToSession(
        InferenceSession session,
        SessionOptions options,
        SessionLoadPlan plan,
        WindowsMlExecutionDevicePolicy devicePolicy)
    {
        (bool efficiencyApplied, string? efficiencyError) =
            WindowsMl.WindowsMlEfficiencyPolicy.ApplyToSession(session, plan.SelectedProvider, devicePolicy);
        if (!efficiencyApplied && efficiencyError is null)
        {
            return;
        }

        string efficiencySessionLabel =
            string.IsNullOrEmpty(options.LogId) ? Path.GetFileName(plan.ModelPath) : options.LogId;
        if (efficiencyApplied)
        {
            _logger?.LogInformation(
                $"Session '{efficiencySessionLabel}' ({ProviderNames.FormatProviderLabel(plan.SelectedProvider)}): " +
                "ep.dynamic.workload_type=Efficient applied.");
            return;
        }

        _logger?.LogWarning(
            $"Session '{efficiencySessionLabel}' ({ProviderNames.FormatProviderLabel(plan.SelectedProvider)}): " +
            $"ep.dynamic.workload_type=Efficient was NOT applied: {efficiencyError}");
    }
#endif

    /// <summary>Provider and TRT knobs for one non-pooled encoder/decoder create (Whisper, Opus).</summary>
    internal sealed record DualSessionRequest
    {
        public required ExecutionProviderKind Provider { get; init; }
        public required CancellationToken CancellationToken { get; init; }
        public IReadOnlyDictionary<string, string>? AdditionalTrtEncoderOptions { get; init; }
        public IReadOnlyDictionary<string, string>? AdditionalTrtDecoderOptions { get; init; }
    }

    public static async Task<WhisperSessionLease> CreateWhisperAsync(
        string encoderModelPath,
        string decoderModelPath,
        DualSessionRequest request)
    {
        (InferenceSession encoderSession, InferenceSession decoderSession, BootstrapContext bootstrap, DualSessionMetadata metadata) =
            await CreateDualSessionsAsync(request, encoderModelPath, decoderModelPath).ConfigureAwait(false);
        return new WhisperSessionLease(
            encoderSession,
            decoderSession,
            bootstrap.RequestedProviderLabel,
            ProviderNames.FormatProviderLabel(metadata.SelectedProvider),
            metadata.BootstrapDetail);
    }

    public static async Task<OpusSessionLease> CreateOpusAsync(
        string encoderModelPath,
        string decoderModelPath,
        DualSessionRequest request)
    {
        (InferenceSession encoderSession, InferenceSession decoderSession, BootstrapContext bootstrap, DualSessionMetadata metadata) =
            await CreateDualSessionsAsync(request, encoderModelPath, decoderModelPath).ConfigureAwait(false);
        return new OpusSessionLease(
            encoderSession,
            decoderSession,
            bootstrap.RequestedProviderLabel,
            ProviderNames.FormatProviderLabel(metadata.SelectedProvider),
            metadata.BootstrapDetail)
        {
            SelectedProviderKind = metadata.SelectedProvider
        };
    }

    /// <summary>
    /// Bootstraps, builds dual session options, and creates both encoder/decoder sessions,
    /// reporting each graph's effective placement. Shared by the Whisper and Opus non-pooled
    /// creates; the caller wraps the returned sessions in its lease type.
    /// </summary>
    private static async Task<(InferenceSession Encoder, InferenceSession Decoder, BootstrapContext Bootstrap, DualSessionMetadata Metadata)>
        CreateDualSessionsAsync(DualSessionRequest request, string encoderModelPath, string decoderModelPath)
    {
        BootstrapContext bootstrap = await BootstrapForProviderAsync(request.Provider, request.CancellationToken)
            .ConfigureAwait(false);
        DualOptionsSelections selections = CreateDualOptionsSelections(new DualOptionsRequest
        {
            Provider = request.Provider,
            BootstrapResult = bootstrap.Bootstrap,
            DevicePolicy = bootstrap.DevicePolicy,
            AdditionalTrtEncoderOptions = request.AdditionalTrtEncoderOptions,
            AdditionalTrtDecoderOptions = request.AdditionalTrtDecoderOptions,
            EncoderModelPath = encoderModelPath,
            DecoderModelPath = decoderModelPath,
        });
        using SessionOptions encoderOptions = selections.Encoder.Options;
        using SessionOptions decoderOptions = selections.Decoder.Options;
        InferenceSession? encoderSession = null;
        InferenceSession? decoderSession = null;
        try
        {
            // Through CreateSession so smoke probes load (or capture) EP-context artifacts like
            // pooled sessions do, instead of rebuilding every engine from source.
            encoderSession = CreateSession(encoderModelPath,
                ForGraphSelection(selections.Encoder, bootstrap.DevicePolicy, sessionFactory: null), request.CancellationToken);
            decoderSession = CreateSession(decoderModelPath,
                ForGraphSelection(selections.Decoder, bootstrap.DevicePolicy, sessionFactory: null), request.CancellationToken);
            DualSessionMetadata metadata = ResolveDualSessionMetadata(
                new DualSessionPair
                {
                    RequestedProvider = request.Provider,
                    Bootstrap = bootstrap,
                    Selections = selections,
                    EncoderSession = encoderSession,
                    DecoderSession = decoderSession,
                });
            TrackCreatedSession(encoderSession, metadata.EncoderProvider);
            TrackCreatedSession(decoderSession, metadata.DecoderProvider);
            return (encoderSession, decoderSession, bootstrap, metadata);
        }
        catch (Exception ex)
        {
            ReleaseSessionsAfterCreateFailure(ex, encoderSession, decoderSession);
            throw;
        }
    }

    /// <summary>
    /// Marks a CUDA out-of-memory before disposing sessions from a failed create.
    /// <see cref="CreateSession"/> tracks a TensorRT-RTX session as soon as it exists, so a
    /// decoder failure after a tracked encoder must poison the guard first.
    /// </summary>
    internal static void ReleaseSessionsAfterCreateFailure(
        Exception failure,
        InferenceSession? encoder,
        InferenceSession? decoder)
    {
        TensorRtRtxTeardownGuard.MarkPoisonedIfCudaOutOfMemory(failure);
        TensorRtRtxTeardownGuard.DisposeSafely(encoder);
        TensorRtRtxTeardownGuard.DisposeSafely(decoder);
    }

    // ── Pooled factory methods ─────────────────────────────────────────────
    //
    // These methods create ONNX sessions and register them with the shared
    // InferenceSessionPool so that sessions are reused across engine calls.
    // Metadata (selected provider, bootstrap detail) is captured at creation
    // time and cached alongside the pool entry for subsequent pool hits.

    private sealed class SessionOptionsDisposeHolder : IDisposable
    {
        public SessionOptionsDisposeHolder(SessionOptions current)
        {
            Current = current;
        }

        public SessionOptions Current { get; set; }

        public void Dispose()
        {
            Current.Dispose();
        }
    }

    internal sealed record SessionOptionsFactoryBundle(
        Func<SessionOptions> CreateOptions,
        ExecutionProviderKind RequestedProvider,
        ExecutionProviderKind SelectedProvider,
        string BootstrapDetail);

    internal static async Task<SessionOptionsFactoryBundle> CreateSessionOptionsFactoryAsync(
        ExecutionProviderKind requestedProvider,
        CancellationToken cancellationToken,
        IReadOnlyDictionary<string, string>? additionalTrtOptions = null)
    {
        var bootstrapResult = await BootstrapTrackedAsync(requestedProvider, allowDownloads: true, cancellationToken)
            .ConfigureAwait(false);
        WindowsMlExecutionDevicePolicy devicePolicy = await ResolveDevicePolicyAsync(cancellationToken)
            .ConfigureAwait(false);
        ExecutionProviderKind sessionProvider = ResolveSessionOptionsProvider(
            requestedProvider,
            bootstrapResult.SelectedProvider);
        SessionOptionsSelection probeSelection = CreateSessionOptions(new SessionOptionsRequest(
            sessionProvider,
            devicePolicy,
            additionalTrtOptions));
        ExecutionProviderKind selectedProvider = probeSelection.SelectedProvider;
        probeSelection.Options.Dispose();

        return new SessionOptionsFactoryBundle(
            () => CreateSessionOptions(new SessionOptionsRequest(sessionProvider, devicePolicy, additionalTrtOptions)).Options,
            requestedProvider,
            selectedProvider,
            FormatBootstrapDetail(bootstrapResult.Detail, probeSelection.FallbackReason) ?? string.Empty);
    }

    /// <summary>
    /// Packaged DirectML is included with Windows ML. Catalog RegisterCertifiedAsync is not a
    /// prerequisite for GetEpDevices/append, so a bootstrap CPU select must not skip the DML path.
    /// Native CUDA on Windows still wins when bootstrap selected CUDA.
    /// </summary>
    internal static ExecutionProviderKind ResolveSessionOptionsProvider(
        ExecutionProviderKind requestedProvider,
        ExecutionProviderKind bootstrapSelectedProvider)
    {
        if (requestedProvider is ExecutionProviderKind.DirectMl)
        {
            return OperatingSystem.IsWindows()
                ? ExecutionProviderKind.DirectMl
                : bootstrapSelectedProvider;
        }

        if (requestedProvider is ExecutionProviderKind.Cuda)
        {
            return bootstrapSelectedProvider is ExecutionProviderKind.Cpu && OperatingSystem.IsWindows()
                ? ExecutionProviderKind.DirectMl
                : bootstrapSelectedProvider;
        }

        return bootstrapSelectedProvider;
    }

    private static SessionPoolKey WithOpenVinoResidency(SessionPoolKey key) =>
        key.Provider is ExecutionProviderKind.OpenVino
            ? key with { UseOpenVinoCpuProxy = _openVinoAvailability.UseOpenVinoCpuProxy }
            : key;

    private static ExecutionProviderKind ResolveSharedSelectedProvider(
        ExecutionProviderKind requestedProvider,
        ExecutionProviderKind firstSessionProvider,
        ExecutionProviderKind secondSessionProvider)
    {
        if (firstSessionProvider == secondSessionProvider)
        {
            return firstSessionProvider;
        }

        return requestedProvider is ExecutionProviderKind.TensorRTRtx
            ? ExecutionProviderKind.Cpu
            : requestedProvider;
    }

    internal static ExecutionProviderKind ResolveEffectiveProviderKindFromSession(
        InferenceSession session,
        ExecutionProviderKind optionsSelectedProvider,
        bool usedCatalogDevicePolicy)
    {
        // DNNL only claims a subset of ops; ORT can silently place an unsupported graph on
        // CPU even after AppendExecutionProvider_Dnnl succeeds. Verify actual placement here
        // rather than trusting the append-time selection, mirroring DnnlReadinessProbe's
        // smoke-session check.
        if (optionsSelectedProvider is ExecutionProviderKind.Dnnl)
        {
            return ResolveDnnlEffectiveProviderFromSession(session);
        }

        // Native ORT CUDA / classic TensorRT are not WinML catalog EP names. Probing
        // GetEpDeviceForInputs through the catalog mapper would treat unmapped names as CPU.
        if (optionsSelectedProvider is ExecutionProviderKind.Cuda or ExecutionProviderKind.TensorRt)
        {
            return optionsSelectedProvider;
        }

        // Linux OpenVINO is the same OpenVINOExecutionProvider plugin the Windows ML catalog
        // registers, so the catalog mapper reports it as OpenVinoCatalog.
        if (optionsSelectedProvider is ExecutionProviderKind.OpenVino)
        {
            return ResolveCatalogEffectiveProviderFromSession(session) is ExecutionProviderKind.OpenVinoCatalog
                ? ExecutionProviderKind.OpenVino
                : ExecutionProviderKind.Cpu;
        }

        // Always probe GetEpDeviceForInputs for GPU / catalog routes so TRT sessions that
        // assigned zero nodes (or fell through) report DirectML/CPU honestly.
        if (usedCatalogDevicePolicy || ShouldProbeEffectiveEpDevices(optionsSelectedProvider))
        {
            return ResolveCatalogEffectiveProviderFromSession(session);
        }

        return optionsSelectedProvider;
    }

    private static bool ShouldProbeEffectiveEpDevices(ExecutionProviderKind optionsSelectedProvider) =>
        optionsSelectedProvider is ExecutionProviderKind.TensorRTRtx
            or ExecutionProviderKind.DirectMl
            or ExecutionProviderKind.Migraphx
            or ExecutionProviderKind.Qnn
            or ExecutionProviderKind.OpenVinoCatalog
            or ExecutionProviderKind.VitisAi;

    private static ExecutionProviderKind ResolveDnnlEffectiveProviderFromSession(InferenceSession session)
    {
        try
        {
            IReadOnlyList<OrtEpDevice> devices = session.GetEpDeviceForInputs();
            return devices.Any(device => DnnlOrtProbe.IsDnnlExecutionProviderName(device.EpName))
                ? ExecutionProviderKind.Dnnl
                : ExecutionProviderKind.Cpu;
        }
        catch (OnnxRuntimeException)
        {
            return ExecutionProviderKind.Cpu;
        }
    }

    private static string? BuildEpFallbackReason(ExecutionProviderKind requestedProvider, ExecutionProviderKind effectiveProvider) =>
        effectiveProvider != requestedProvider
            ? $"Requested {ProviderNames.FormatProviderLabel(requestedProvider)} but effective {ProviderNames.FormatProviderLabel(effectiveProvider)}."
            : null;

    private static string? BuildSessionOptionsFallbackReason(
        ExecutionProviderKind requestedProvider,
        ExecutionProviderKind effectiveProvider,
        SessionOptionsSelection sessionOptionsSelection) =>
        MergeFallbackReasons(
            sessionOptionsSelection.FallbackReason,
            BuildEpFallbackReason(requestedProvider, effectiveProvider));

    // EP-context artifacts are keyed by the provider options their engines were built with, so
    // session creation needs the options appended here, not a re-derivation from the model.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<SessionOptions, IReadOnlyDictionary<string, string>>
        AppendedTensorRtRtxOptions = new();

    internal static IReadOnlyDictionary<string, string>? TryGetAppendedTensorRtRtxOptions(SessionOptions options) =>
        AppendedTensorRtRtxOptions.TryGetValue(options, out IReadOnlyDictionary<string, string>? trtOptions) ? trtOptions : null;

    internal static ExecutionProviderKind AppendTensorRtRtxOrFallbackProvider(
        SessionOptions options,
        IReadOnlyDictionary<string, string>? additionalTrtOptions = null,
        bool enableCudaGraph = false)
    {
#if WINDOWS
        WindowsMlOnnxRuntimeNativeResolver.EnsureInitialized();
#endif
        var devices = OrtEnv.Instance().GetEpDevices();
        var trtDevice = devices.FirstOrDefault(d => ProviderNames.IsTensorRtRtxDeviceCandidate(d.EpName, d.HardwareDevice.Type));
        if (trtDevice != null)
        {
            IReadOnlyDictionary<string, string> trtOptions = BuildTensorRtRtxOptions(additionalTrtOptions, enableCudaGraph);
            options.AppendExecutionProvider(OrtEnv.Instance(), new[] { trtDevice }, trtOptions);
            AppendedTensorRtRtxOptions.AddOrUpdate(options, trtOptions);
            return ExecutionProviderKind.TensorRTRtx;
        }

        if (_logger?.IsEnabled(LogLevel.Debug) == true)
        {
            foreach (OrtEpDevice device in devices.Where(d => d.HardwareDevice.Type is OrtHardwareDeviceType.GPU))
            {
                if (!TryMapCatalogEpNameToExecutionProviderKind(device.EpName, out _))
                {
                    ReportUnmappedCatalogEpName(device.EpName, CatalogEpNameSource.GetEpDevices, devices.Count);
                }
            }
        }

        // TensorRT RTX plugin device not available; fallback to DirectML if possible, else CPU is default.
        return TryAppendDirectMlProvider(options, out _)
            ? ExecutionProviderKind.DirectMl
            : ExecutionProviderKind.Cpu;
    }

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<SessionOptions, IReadOnlyDictionary<string, string>>
        AppendedTensorRtOptions = new();

    internal static IReadOnlyDictionary<string, string>? TryGetAppendedTensorRtOptions(SessionOptions options) =>
        AppendedTensorRtOptions.TryGetValue(options, out IReadOnlyDictionary<string, string>? trtOptions) ? trtOptions : null;

    /// <summary>
    /// Appends the classic TensorRT EP with the persistent engine-cache options from
    /// <see cref="BuildTensorRtOptions"/>. Prefers the generic EP-device append when the
    /// classic TRT device is enumerated by <c>OrtEnv.GetEpDevices()</c>; otherwise uses the
    /// built-in provider-options append (<see cref="OrtTensorRTProviderOptions"/>), because
    /// the built-in EP is reported by <c>GetAvailableProviders()</c> but is NOT enumerated
    /// as an EP device (verified live on ORT 1.31 in the GPU container: only the CPU device
    /// is enumerated), so the generic branch is normally unreachable for classic TRT. There is
    /// no TensorRT append without the cache options: when the provider-options append fails,
    /// this falls back to DirectML/CPU so callers can report the fallback instead of claiming
    /// a cached TensorRT session.
    /// </summary>
    internal static ExecutionProviderKind AppendTensorRtOrFallbackProvider(
        SessionOptions options,
        IReadOnlyDictionary<string, string>? additionalTrtOptions = null,
        string? trtModelPath = null)
    {
#if WINDOWS
        WindowsMlOnnxRuntimeNativeResolver.EnsureInitialized();
#endif
        var devices = OrtEnv.Instance().GetEpDevices();
        var trtDevice = devices.FirstOrDefault(d => ProviderNames.IsTensorRtDeviceCandidate(d.EpName, d.HardwareDevice.Type));
        if (trtDevice != null)
        {
            IReadOnlyDictionary<string, string> trtOptions = BuildTensorRtOptionsForModel(additionalTrtOptions, trtModelPath);
            EnsureClassicTensorRtCacheDirectory(trtOptions);
            options.AppendExecutionProvider(OrtEnv.Instance(), new[] { trtDevice }, trtOptions);
            AppendedTensorRtOptions.AddOrUpdate(options, trtOptions);
            return ExecutionProviderKind.TensorRt;
        }

        try
        {
            using OrtTensorRTProviderOptions providerOptions = new();
            IReadOnlyDictionary<string, string> trtOptions = BuildTensorRtOptionsAfterProviderLoad(
                additionalTrtOptions, trtModelPath,
                () =>
                {
                    // Creating provider options only allocates an options struct. Append to
                    // disposable options to load ORT's actual TensorRT dependency before hashing
                    // it. No inference session is created and no model engines are compiled here.
                    using SessionOptions preloadOptions = new();
                    preloadOptions.AppendExecutionProvider_Tensorrt(providerOptions);
                },
                BuildClassicTensorRtEnvironmentFingerprint);
            EnsureClassicTensorRtCacheDirectory(trtOptions);
            providerOptions.UpdateOptions(new Dictionary<string, string>(trtOptions, StringComparer.Ordinal));
            options.AppendExecutionProvider_Tensorrt(providerOptions);
            AppendedTensorRtOptions.AddOrUpdate(options, trtOptions);
            return ExecutionProviderKind.TensorRt;
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or InvalidOperationException or DllNotFoundException or EntryPointNotFoundException)
        {
            // Built-in TensorRT EP provider-options append is not usable; report a fallback.
        }

        return TryAppendDirectMlProvider(options, out _)
            ? ExecutionProviderKind.DirectMl
            : ExecutionProviderKind.Cpu;
    }

    internal static OrtTensorRTProviderOptions BuildTensorRtProviderOptions(
        IReadOnlyDictionary<string, string>? additionalTrtOptions,
        string environmentFingerprint,
        string? trtModelPath = null) =>
        BuildTensorRtProviderOptions(BuildTensorRtOptions(additionalTrtOptions, environmentFingerprint, trtModelPath));

    internal static OrtTensorRTProviderOptions BuildTensorRtProviderOptions(
        IReadOnlyDictionary<string, string> trtOptions)
    {
        EnsureClassicTensorRtCacheDirectory(trtOptions);
        var providerOptions = new OrtTensorRTProviderOptions();
        providerOptions.UpdateOptions(new Dictionary<string, string>(trtOptions, StringComparer.Ordinal));
        return providerOptions;
    }

    private static void EnsureClassicTensorRtCacheDirectory(IReadOnlyDictionary<string, string> trtOptions)
    {
        if (trtOptions.TryGetValue("trt_engine_cache_path", out string? cachePath) &&
            !string.IsNullOrWhiteSpace(cachePath))
        {
            Directory.CreateDirectory(cachePath);
        }
    }

    internal static void AppendDirectMlProvider(SessionOptions options)
    {
#if WINDOWS
        WindowsMlOnnxRuntimeNativeResolver.EnsureInitialized();
#endif
        var devices = OrtEnv.Instance().GetEpDevices();
        OrtEpDevice? directMlDevice = devices.FirstOrDefault(d => ProviderNames.IsDirectMlDeviceCandidate(d.EpName, d.HardwareDevice.Type))
            ?? devices.FirstOrDefault(d => ProviderNames.IsDirectMlExecutionProviderName(d.EpName));
        if (directMlDevice is not null)
        {
            options.AppendExecutionProvider(
                OrtEnv.Instance(),
                new[] { directMlDevice },
                new Dictionary<string, string>(StringComparer.Ordinal));
            return;
        }

        try
        {
            options.AppendExecutionProvider(ProviderNames.DirectMlExecutionProviderName);
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or InvalidOperationException or DllNotFoundException or EntryPointNotFoundException)
        {
            throw new InvalidOperationException(
                "DirectML catalog execution provider is not visible in OrtEnv.GetEpDevices().",
                ex);
        }
    }

    internal sealed record SessionOptionsSelection(
        SessionOptions Options,
        ExecutionProviderKind SelectedProvider,
        string? FallbackReason = null);

    // Classic TensorRT EP uses trt_profile_*. NvTensorRTRTX rejects those keys.
    // Same shape grammar: "input:dim1xdim2x...,input2:..."
    // https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html
    private static readonly IReadOnlyDictionary<string, string> ClassicTrtProfileKeysToNvRtx =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["trt_profile_min_shapes"] = "nv_profile_min_shapes",
            ["trt_profile_max_shapes"] = "nv_profile_max_shapes",
            ["trt_profile_opt_shapes"] = "nv_profile_opt_shapes",
        };

    private static string BuildSessionOptionsFingerprint(
        ExecutionProviderKind selectedProviderKind,
        WindowsMlExecutionDevicePolicy devicePolicy,
        IReadOnlyDictionary<string, string>? additionalTrtOptions,
        bool enableCudaGraph = false)
    {
        if (selectedProviderKind is ExecutionProviderKind.TensorRTRtx)
        {
            var combinedOptions = new Dictionary<string, string>(
                BuildTensorRtRtxOptions(additionalTrtOptions, enableCudaGraph), StringComparer.Ordinal);

            return SessionPoolKey.HashOptions(combinedOptions);
        }

        if (selectedProviderKind is ExecutionProviderKind.TensorRt)
        {
            var combinedOptions = new Dictionary<string, string>(
                BuildTensorRtOptions(additionalTrtOptions), StringComparer.Ordinal);

            return SessionPoolKey.HashOptions(combinedOptions);
        }

        if (ShouldIncludePolicyInFingerprint(devicePolicy, selectedProviderKind))
        {
            return SessionPoolKey.HashOptions(new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["windows_ml_device_policy"] = WindowsMlExecutionDevicePolicySettings.ToKey(devicePolicy)
            });
        }

        return SessionPoolKey.HashOptions(null);
    }

    private static bool ShouldIncludePolicyInFingerprint(
        WindowsMlExecutionDevicePolicy devicePolicy,
        ExecutionProviderKind selectedProviderKind) =>
        ProviderNames.ShouldUseCatalogDevicePolicy(devicePolicy, selectedProviderKind);

    private static string? MergeFallbackReasons(string? encoder, string? decoder) =>
        (encoder, decoder) switch
        {
            (null, null) => null,
            ({ } e, null) => e,
            (null, { } d) => d,
            ({ } e, { } d) when e == d => e,
            ({ } e, { } d) => $"Encoder: {e} Decoder: {d}"
        };

    private static string? FormatBootstrapDetail(
        string? bootstrapDetail,
        string? sessionOptionsFallbackReason)
    {
        if (string.IsNullOrWhiteSpace(sessionOptionsFallbackReason))
        {
            return bootstrapDetail;
        }

        return string.IsNullOrWhiteSpace(bootstrapDetail)
            ? $"Session options fallback reason: {sessionOptionsFallbackReason}"
            : $"{bootstrapDetail} Session options fallback reason: {sessionOptionsFallbackReason}";
    }

    private static SessionOptions CreateBaseSessionOptions(
        WindowsMlExecutionDevicePolicy devicePolicy,
        out bool devicePolicyApplied,
        bool tensorRtRtx = false, string? logId = null)
    {
        SessionOptions options = new()
        {
            // ORT_ENABLE_ALL re-fuses LayerNormalization+Skip / Bias+Gelu into com.microsoft
            // contrib ops (SkipLayerNormalization, BiasGelu) at graph resolve. TensorRT-RTX
            // cannot import those, so the TRT session gets a parse-error storm and falls back.
            // BASIC keeps standard Add / LayerNormalization / Gelu nodes the TRT-RTX parser accepts.
            GraphOptimizationLevel = tensorRtRtx
                ? GraphOptimizationLevel.ORT_ENABLE_BASIC
                : GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            LogId = logId ?? string.Empty,
            // MS "Efficiency tips for Windows ML inference" memory lever: opt-in arena disable
            // for memory-footprint-sensitive deployments (throughput tradeoff). Default: arena on.
            EnableCpuMemArena = !Common.SessionEfficiencyEnvironment.DisableCpuMemArena
        };

        devicePolicyApplied = false;
#if WINDOWS
        devicePolicyApplied = WindowsMlExecutionDevicePolicyMapper.ApplyIfNeeded(options, devicePolicy);
#endif
        return options;
    }

    /// <summary>
    /// Reports the truth about whether a requested extended Windows ML device policy
    /// (<see cref="WindowsMlExecutionDevicePolicy.DefaultRender"/> /
    /// <see cref="WindowsMlExecutionDevicePolicy.MinPower"/>) actually took effect on the
    /// session. <see cref="WindowsMlExecutionDevicePolicyMapper.ApplyIfNeeded"/> silently skips
    /// <c>SetEpSelectionPolicy</c> when the loaded ORT managed binding lacks the required enum
    /// members, so callers must not report the policy as active without checking
    /// <paramref name="devicePolicyApplied"/> — otherwise diagnostics/fingerprints would claim a
    /// device-selection behavior that never actually happened.
    /// </summary>
    private static string? BuildDevicePolicyFallbackReason(
        WindowsMlExecutionDevicePolicy devicePolicy,
        bool devicePolicyApplied) =>
        !devicePolicyApplied &&
        devicePolicy is WindowsMlExecutionDevicePolicy.DefaultRender or WindowsMlExecutionDevicePolicy.MinPower
            ? $"Requested Windows ML device policy '{WindowsMlExecutionDevicePolicySettings.ToKey(devicePolicy)}' " +
              "was not applied: the loaded ONNX Runtime managed binding does not expose the required " +
              "extended ExecutionProviderDevicePolicy members (DEFAULT_RENDER/MIN_POWER); the session " +
              "falls back to Explicit EP selection instead."
            : null;

    private static async Task<WindowsMlExecutionDevicePolicy> ResolveDevicePolicyAsync(CancellationToken cancellationToken)
    {
        try
        {
            return await _devicePolicyProvider.GetPolicyAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch
        {
            return WindowsMlExecutionDevicePolicy.Explicit;
        }
    }

    private sealed class NullWindowsMlEpDevicePolicyProvider : IWindowsMlEpDevicePolicyProvider
    {
        internal static readonly NullWindowsMlEpDevicePolicyProvider Instance = new();

        public Task<WindowsMlExecutionDevicePolicy> GetPolicyAsync(CancellationToken cancellationToken = default) =>
            cancellationToken.IsCancellationRequested
                ? Task.FromCanceled<WindowsMlExecutionDevicePolicy>(cancellationToken)
                : Task.FromResult(WindowsMlExecutionDevicePolicy.Explicit);

        public void InvalidateCache()
        {
        }
    }

    internal static IExecutionProviderBootstrapper GetPlatformBootstrapper()
    {
#if WINDOWS
        return new ExecutionProviders.Windows.WindowsExecutionProviderBootstrapper();
#else
        if (OperatingSystem.IsMacOSVersionAtLeast(10, 15))
        {
            return new ExecutionProviders.Mac.MacExecutionProviderBootstrapper();
        }

        if (OperatingSystem.IsLinux())
        {
            return new ExecutionProviders.Linux.LinuxExecutionProviderBootstrapper(
                NullOpenVinoAvailabilityProvider.Instance);
        }

        return new ExecutionProviders.PortableExecutionProviderBootstrapper(
            TensorRtRtx.TensorRtRtxProviderBootstrapFactory.CreateWithDefaultInstallPath(
                ResolveProductDataRoot()));
#endif
    }

    /// <summary>
    /// Product data root that owns runtime component installs (the TensorRT-RTX plugin, and
    /// anything else installed beside it). Honors the same relocation variables as the engine
    /// cache; a host that moved its data root must keep resolving installed providers, or the
    /// plugin becomes invisible and every TensorRT stage silently falls back to another EP.
    /// </summary>
    internal static string ResolveProductDataRoot()
    {
        string? dataRoot = Environment.GetEnvironmentVariable(DataRootEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(dataRoot))
        {
            dataRoot = Environment.GetEnvironmentVariable(CacheRootEnvironmentVariable);
        }

        if (!string.IsNullOrWhiteSpace(dataRoot))
        {
            return NormalizePath(dataRoot);
        }

        string localAppDataRoot = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (string.IsNullOrWhiteSpace(localAppDataRoot))
        {
            localAppDataRoot = AppContext.BaseDirectory;
        }

        return Path.Join(localAppDataRoot, "Trackdub");
    }

    private sealed class NullOpenVinoAvailabilityProvider : IOpenVinoAvailabilityProvider
    {
        public static readonly NullOpenVinoAvailabilityProvider Instance = new();
        public bool IsAvailable => false;
        public bool UseOpenVinoCpuProxy => false;
    }
}
