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

    // Canonical EP names returned by ORT device discovery. TensorRT RTX uses the standalone
    // EP ABI plugin, not the Windows ML catalog spelling. Single source of truth — must match the
    // name the plugin is registered under (see TensorRtRtxPluginService.RegistrationName); ORT reports
    // OrtEpDevice.EpName as that registration name.
    private const string TensorRtRtxExecutionProviderName = TensorRtRtxProviderConstants.PluginOrtExecutionProviderName;
    private const string DnnlExecutionProviderName = DnnlOrtProbe.OrtExecutionProviderName;
    private const string DnnlUpperExecutionProviderName = DnnlOrtProbe.OrtExecutionProviderNameUpper;
    private const string DirectMlExecutionProviderName = "DmlExecutionProvider";
    private const string ClassicTensorRtExecutionProviderName = "TensorrtExecutionProvider";
    private const string DirectMlLongExecutionProviderName = "DirectMLExecutionProvider";
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

    private static async Task<BootstrapContext> BootstrapForProviderAsync(
        ExecutionProviderKind provider,
        CancellationToken cancellationToken)
    {
        var bootstrapResult = await BootstrapTrackedAsync(provider, allowDownloads: true, cancellationToken)
            .ConfigureAwait(false);
        WindowsMlExecutionDevicePolicy devicePolicy = await ResolveDevicePolicyAsync(cancellationToken)
            .ConfigureAwait(false);
        return new BootstrapContext(bootstrapResult, devicePolicy, FormatProviderLabel(provider));
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

    private static DualOptionsSelections CreateDualOptionsSelections(
        ExecutionProviderKind provider,
        ExecutionProviders.ExecutionProviderBootstrapResult bootstrapResult,
        WindowsMlExecutionDevicePolicy devicePolicy,
        IReadOnlyDictionary<string, string>? additionalTrtEncoderOptions,
        IReadOnlyDictionary<string, string>? additionalTrtDecoderOptions,
        bool enableEncoderCudaGraph = false, string? encoderLogId = null, string? decoderLogId = null,
        string? encoderModelPath = null, string? decoderModelPath = null)
    {
        ExecutionProviderKind sessionProvider =
            ResolveSessionOptionsProvider(provider, bootstrapResult.SelectedProvider);
        return new DualOptionsSelections(
            CreateSessionOptions(new SessionOptionsRequest(sessionProvider, devicePolicy, additionalTrtEncoderOptions, enableEncoderCudaGraph, encoderLogId, encoderModelPath)),
            CreateSessionOptions(new SessionOptionsRequest(sessionProvider, devicePolicy, additionalTrtDecoderOptions, EnableCudaGraph: false, decoderLogId, decoderModelPath)));
    }

    private static DualSessionMetadata ResolveDualSessionMetadata(
        ExecutionProviderKind requestedProvider,
        BootstrapContext bootstrap,
        DualOptionsSelections selections,
        InferenceSession encoderSession,
        InferenceSession decoderSession)
    {
        ExecutionProviderKind encoderEffective = ResolveEffectiveProviderKindFromSession(
            encoderSession,
            selections.Encoder.SelectedProvider,
            ShouldUseCatalogDevicePolicy(bootstrap.DevicePolicy, selections.Encoder.SelectedProvider));
        ExecutionProviderKind decoderEffective = ResolveEffectiveProviderKindFromSession(
            decoderSession,
            selections.Decoder.SelectedProvider,
            ShouldUseCatalogDevicePolicy(bootstrap.DevicePolicy, selections.Decoder.SelectedProvider));
        ExecutionProviderKind resolvedSelectedProvider = ResolveSharedSelectedProvider(
            requestedProvider, encoderEffective, decoderEffective);
        string? encoderFallbackReason = BuildSessionOptionsFallbackReason(
            requestedProvider, encoderEffective, selections.Encoder);
        string? decoderFallbackReason = BuildSessionOptionsFallbackReason(
            requestedProvider, decoderEffective, selections.Decoder);
        return new DualSessionMetadata(
            resolvedSelectedProvider,
            encoderEffective,
            decoderEffective,
            FormatBootstrapDetail(
                bootstrap.Bootstrap.Detail,
                MergeFallbackReasons(encoderFallbackReason, decoderFallbackReason)));
    }

    private static async Task<(SessionPoolKey EncoderKey, SessionPoolKey DecoderKey)> BuildDualPooledKeysAsync(
        string engineFamily,
        string encoderModelPath,
        string decoderModelPath,
        DualOptionsSelections selections,
        WindowsMlExecutionDevicePolicy devicePolicy,
        IReadOnlyDictionary<string, string>? additionalTrtEncoderOptions,
        IReadOnlyDictionary<string, string>? additionalTrtDecoderOptions,
        string? modelId,
        string? variant,
        CancellationToken cancellationToken,
        bool enableEncoderCudaGraph = false, string? encoderLogId = null, string? decoderLogId = null)
    {
        string encoderFingerprint = BuildSessionOptionsFingerprint(
            selections.Encoder.SelectedProvider, devicePolicy, additionalTrtEncoderOptions, enableEncoderCudaGraph);
        string decoderFingerprint = BuildSessionOptionsFingerprint(
            selections.Decoder.SelectedProvider, devicePolicy, additionalTrtDecoderOptions);
        SessionPoolKey[] keys = await SessionPoolKey.CreateManyAsync(
            [
                new SessionPoolKey.PooledKeyRequest(
                    engineFamily, modelId, variant, selections.Encoder.SelectedProvider,
                    encoderModelPath, null, "encoder", encoderFingerprint),
                new SessionPoolKey.PooledKeyRequest(
                    engineFamily, modelId, variant, selections.Decoder.SelectedProvider,
                    decoderModelPath, null, "decoder", decoderFingerprint),
            ],
            cancellationToken).ConfigureAwait(false);
        return (WithOpenVinoResidency(keys[0]), WithOpenVinoResidency(keys[1]));
    }

    private static async Task<DualPooledLeasePair> AcquireDualPooledSessionsAsync(
        ExecutionProviderKind requestedProvider,
        BootstrapContext bootstrap,
        DualOptionsSelections selections,
        string encoderModelPath,
        string decoderModelPath,
        SessionPoolKey encoderKey,
        SessionPoolKey decoderKey,
        InferenceSessionPool pool,
        CancellationToken cancellationToken,
        Func<string, SessionOptions, InferenceSession>? sessionFactory)
    {
        // Atomic bundle (audit §3A): never hold the encoder lease while blocked on the decoder.
        // Lease order in the bundle is caller order (encoder, decoder). Each graph reports its
        // placement at creation: if the decoder then fails, a CPU-placed encoder must not stay
        // tracked as TensorRT-RTX, or a poisoned runtime would skip its disposal and leak it.
        SessionLeaseBundle bundle = await pool.GetLeaseBundleAsync(
            [
                new SessionLeaseRequest(
                    encoderKey,
                    ct => Task.FromResult(CreateSession(encoderModelPath, selections.Encoder.Options, sessionFactory, ct, selections.Encoder.SelectedProvider, bootstrap.DevicePolicy)),
                    session => ResolveEffectiveProviderKindFromSession(
                        session,
                        selections.Encoder.SelectedProvider,
                        ShouldUseCatalogDevicePolicy(bootstrap.DevicePolicy, selections.Encoder.SelectedProvider))),
                new SessionLeaseRequest(
                    decoderKey,
                    ct => Task.FromResult(CreateSession(decoderModelPath, selections.Decoder.Options, sessionFactory, ct, selections.Decoder.SelectedProvider, bootstrap.DevicePolicy)),
                    session => ResolveEffectiveProviderKindFromSession(
                        session,
                        selections.Decoder.SelectedProvider,
                        ShouldUseCatalogDevicePolicy(bootstrap.DevicePolicy, selections.Decoder.SelectedProvider))),
            ],
            cancellationToken).ConfigureAwait(false);

        try
        {
            DualSessionMetadata metadata = ResolveDualSessionMetadata(
                requestedProvider, bootstrap, selections,
                bundle.LeaseAt(0).Session, bundle.LeaseAt(1).Session);
            CpuExecutionAdmission.Shared.RegisterSession(bundle.LeaseAt(0).Session, metadata.EncoderProvider);
            CpuExecutionAdmission.Shared.RegisterSession(bundle.LeaseAt(1).Session, metadata.DecoderProvider);
            TensorRtRtxTeardownGuard.ReconcileTracking(bundle.LeaseAt(0).Session, metadata.EncoderProvider);
            TensorRtRtxTeardownGuard.ReconcileTracking(bundle.LeaseAt(1).Session, metadata.DecoderProvider);
            return new DualPooledLeasePair(
                bundle,
                bootstrap.RequestedProviderLabel,
                metadata.SelectedProvider,
                metadata.BootstrapDetail);
        }
        catch
        {
            bundle.Dispose();
            throw;
        }
    }

    public static async Task<SingleSessionLease> CreateSingleAsync(
        string modelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        IReadOnlyDictionary<string, string>? additionalTrtOptions = null,
        bool allowTrtInitFallback = true)
    {
        string requestedProvider = FormatProviderLabel(provider);
        var bootstrapResult = await BootstrapTrackedAsync(provider, allowDownloads: true, cancellationToken)
            .ConfigureAwait(false);
        WindowsMlExecutionDevicePolicy devicePolicy = await ResolveDevicePolicyAsync(cancellationToken)
            .ConfigureAwait(false);
        string logId = ExtractLogId(modelPath);
        SessionOptionsSelection sessionOptionsSelection = CreateSessionOptions(new SessionOptionsRequest(
            ResolveSessionOptionsProvider(provider, bootstrapResult.SelectedProvider),
            devicePolicy,
            additionalTrtOptions,
            LogId: logId,
            TrtModelPath: modelPath));
        InferenceSession? session = null;
        try
        {
            (session, sessionOptionsSelection) = CreateInferenceSessionWithTrtInitFallback(
                modelPath,
                sessionOptionsSelection,
                devicePolicy,
                additionalTrtOptions,
                sessionFactory: null,
                cancellationToken,
                allowTrtInitFallback,
                logId: logId);
            bool useCatalogDevicePolicy = ShouldUseCatalogDevicePolicy(devicePolicy, sessionOptionsSelection.SelectedProvider);
            ExecutionProviderKind effectiveProvider = ResolveEffectiveProviderKindFromSession(
                session,
                sessionOptionsSelection.SelectedProvider,
                useCatalogDevicePolicy);
            string selectedProvider = FormatProviderLabel(effectiveProvider);
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

    private static InferenceSession CreateSession(
        string modelPath,
        SessionOptions options,
        Func<string, SessionOptions, InferenceSession>? sessionFactory,
        CancellationToken cancellationToken,
        ExecutionProviderKind selectedProvider,
        WindowsMlExecutionDevicePolicy devicePolicy)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var phase = BenchmarkPhaseCapture.Start("onnx-session-create");
        // EP-context artifacts embed a TensorRT-RTX-serialized engine that only that EP can
        // deserialize; only TRT-RTX sessions may load one (see EpContextWarmupService).
        IReadOnlyDictionary<string, string>? trtOptions = selectedProvider is ExecutionProviderKind.TensorRTRtx
            ? TryGetAppendedTensorRtRtxOptions(options)
            : null;
        string loadPath = trtOptions is not null
            ? EpContext.EpContextLoadPathResolver.TryResolveLoadPath(modelPath, trtOptions) ?? modelPath
            : modelPath;
        // Without a current artifact, a large graph's engine build is captured as one, so the next
        // process loads engines instead of rebuilding them (minutes for multi-GB models).
        bool captureCandidate = sessionFactory is null &&
                               trtOptions is not null &&
                               string.Equals(loadPath, modelPath, StringComparison.Ordinal);
        if (captureCandidate && EpContext.EpContextCapture.TryGetSkipReason(modelPath, out string captureSkipReason))
        {
            // A skipped capture is not harmless: every later launch rebuilds these engines.
            // Say so once per model at warning level instead of a trace-only whisper.
            ReportCaptureSkip(modelPath, captureSkipReason);
        }

        InferenceSession session;
        try
        {
            session =
                captureCandidate && EpContext.EpContextCapture.ShouldCapture(modelPath)
                    ? EpContext.EpContextCapture.CreateSession(modelPath, options, trtOptions!, static (path, o) => new InferenceSession(path, o))
                    : sessionFactory is null
                        ? new InferenceSession(loadPath, options)
                        : sessionFactory(loadPath, options);
        }
        catch (Exception ex) when (selectedProvider is ExecutionProviderKind.TensorRTRtx)
        {
            TensorRtRtxTeardownGuard.MarkPoisonedIfCudaOutOfMemory(ex);
            throw;
        }

        if (selectedProvider is ExecutionProviderKind.TensorRTRtx)
        {
            TensorRtRtxTeardownGuard.Track(session);
        }

#if WINDOWS
        // MS "Efficiency tips for Windows ML inference": express efficiency intent through ORT
        // (ep.dynamic.workload_type) for NPU catalog sessions under efficiency-flavored device
        // policies. Only NPU EPs act on the signal; the call is a no-op elsewhere but is gated
        // to NPU providers to keep the no-op surface minimal. Outcomes are logged, never assumed.
        // This hook runs strictly post-registration: an NPU catalog session can only exist here
        // if WindowsMlProviderRegistrationPolicy already readied + TryRegister()-ed the EP
        // (MS "Register Windows ML execution providers") — no new install/registration logic
        // is introduced, and the hook never fires when the EP was not registered.
        (bool efficiencyApplied, string? efficiencyError) =
            WindowsMl.WindowsMlEfficiencyPolicy.ApplyToSession(session, selectedProvider, devicePolicy);
        if (efficiencyApplied || efficiencyError is not null)
        {
            string efficiencySessionLabel =
                string.IsNullOrEmpty(options.LogId) ? Path.GetFileName(modelPath) : options.LogId;
            if (efficiencyApplied)
            {
                _logger?.LogInformation(
                    $"Session '{efficiencySessionLabel}' ({FormatProviderLabel(selectedProvider)}): " +
                    "ep.dynamic.workload_type=Efficient applied.");
            }
            else
            {
                _logger?.LogWarning(
                    $"Session '{efficiencySessionLabel}' ({FormatProviderLabel(selectedProvider)}): " +
                    $"ep.dynamic.workload_type=Efficient was NOT applied: {efficiencyError}");
            }
        }
#endif

        return session;
    }

    /// <summary>
    /// Creates an ORT session; when TensorRT RTX was selected and init fails due to unsupported
    /// ops/kernels, optionally retries once with DirectML (Windows) then CPU and records FallbackReason.
    /// Pass <paramref name="allowTrtInitFallback"/> <see langword="false"/> for hard-pin routes
    /// (<c>RequirePreferredExecutionProvider</c> / CLI <c>--require-execution-provider</c>).
    /// </summary>
    internal static (InferenceSession Session, SessionOptionsSelection Selection) CreateInferenceSessionWithTrtInitFallback(
        string modelPath,
        SessionOptionsSelection initialSelection,
        WindowsMlExecutionDevicePolicy devicePolicy,
        IReadOnlyDictionary<string, string>? additionalTrtOptions,
        Func<string, SessionOptions, InferenceSession>? sessionFactory,
        CancellationToken cancellationToken,
        bool allowTrtInitFallback = true,
        string? logId = null)
    {
        // Honest skip: graphs with com.microsoft contrib fusions cannot be imported by
        // TensorRT-RTX (all-or-nothing ONNX catalog). Do not attempt TRT and do not emit
        // the ModelImporter error storm — fall through to DirectML/CPU immediately.
        if (initialSelection.SelectedProvider is ExecutionProviderKind.TensorRTRtx
            && TrtRtxUnsupportedOpScanner.FindUnsupportedOps(modelPath) is { Count: > 0 } unsupportedOps)
        {
            string opList = string.Join(", ", unsupportedOps);
            string skipReason =
                $"TensorRT-RTX cannot import '{Path.GetFileName(modelPath)}' (unsupported ops: {opList}).";

            if (!allowTrtInitFallback)
            {
                initialSelection.Options.Dispose();
                throw new InvalidOperationException($"{skipReason} Hard-pin route has no fallback.");
            }

            return CreateUnsupportedGraphFallback(modelPath, initialSelection, devicePolicy,
                sessionFactory, cancellationToken, skipReason, logId: logId);
        }

        try
        {
            InferenceSession session = CreateSession(
                modelPath,
                initialSelection.Options,
                sessionFactory,
                cancellationToken,
                initialSelection.SelectedProvider,
                devicePolicy);
            return (session, initialSelection);
        }
        catch (Exception ex) when (
            allowTrtInitFallback
            && initialSelection.SelectedProvider is ExecutionProviderKind.TensorRTRtx
            && LooksLikeTrtSessionInitFailure(ex))
        {
            return CreateTrtInitFailureFallback(modelPath, initialSelection, devicePolicy,
                sessionFactory, cancellationToken, ex, logId: logId);
        }
    }

    private static (InferenceSession Session, SessionOptionsSelection Selection) CreateUnsupportedGraphFallback(
        string modelPath, SessionOptionsSelection initialSelection, WindowsMlExecutionDevicePolicy devicePolicy,
        Func<string, SessionOptions, InferenceSession>? sessionFactory, CancellationToken cancellationToken,
        string skipReason, string? logId = null)
    {
        Exception? lastFailure = null;
        foreach (SessionOptionsSelection fallbackSelection in EnumerateTrtInitFallbackProviders()
                     .Select(provider => CreateSessionOptions(new SessionOptionsRequest(provider, devicePolicy, LogId: logId))))
        {
            try
            {
                InferenceSession session = CreateSession(modelPath, fallbackSelection.Options, sessionFactory,
                    cancellationToken, fallbackSelection.SelectedProvider, devicePolicy);
                initialSelection.Options.Dispose();
                string label = FormatProviderLabel(fallbackSelection.SelectedProvider);
                return (session, new SessionOptionsSelection(fallbackSelection.Options,
                    fallbackSelection.SelectedProvider,
                    MergeFallbackReasons($"{skipReason} Selected {label}.", fallbackSelection.FallbackReason)));
            }
            catch (Exception fallbackEx)
            {
                fallbackSelection.Options.Dispose();
                if (!IsRecoverableTrtFallbackInitFailure(fallbackEx)) throw;
                lastFailure = fallbackEx;
            }
        }

        initialSelection.Options.Dispose();
        throw lastFailure ?? new InvalidOperationException(skipReason);
    }

    private static (InferenceSession Session, SessionOptionsSelection Selection) CreateTrtInitFailureFallback(
        string modelPath, SessionOptionsSelection initialSelection, WindowsMlExecutionDevicePolicy devicePolicy,
        Func<string, SessionOptions, InferenceSession>? sessionFactory, CancellationToken cancellationToken,
        Exception originalFailure, string? logId = null)
    {
        string trtError = SummarizeExceptionMessage(originalFailure);
        TensorRtRtxTeardownGuard.MarkPoisonedIfCudaOutOfMemory(originalFailure);
        Exception lastFailure = originalFailure;
        foreach (SessionOptionsSelection fallbackSelection in EnumerateTrtInitFallbackProviders()
                     .Select(provider => CreateSessionOptions(new SessionOptionsRequest(provider, devicePolicy, LogId: logId))))
        {
            try
            {
                InferenceSession session = CreateSession(modelPath, fallbackSelection.Options, sessionFactory,
                    cancellationToken, fallbackSelection.SelectedProvider, devicePolicy);
                // Transfer ownership: the caller owns fallback options after success.
                initialSelection.Options.Dispose();
                string label = FormatProviderLabel(fallbackSelection.SelectedProvider);
                string reason = $"TensorRT RTX session init failed ({trtError}); fell back to {label}.";
                return (session, new SessionOptionsSelection(fallbackSelection.Options,
                    fallbackSelection.SelectedProvider, MergeFallbackReasons(reason, fallbackSelection.FallbackReason)));
            }
            catch (Exception fallbackEx)
            {
                fallbackSelection.Options.Dispose();
                if (!IsRecoverableTrtFallbackInitFailure(fallbackEx)) throw;
                lastFailure = fallbackEx;
            }
        }

        // Leave initialSelection.Options for the caller to dispose.
        throw lastFailure;
    }

    /// <summary>
    /// Rewrites a TensorRT-RTX request to DirectML/CPU when any model graph contains
    /// contrib ops the TRT-RTX ONNX parser cannot import. Multi-session factories call this
    /// before building session options so they never open a doomed TRT session.
    /// </summary>
    private static ExecutionProviderKind DowngradeTrtRtxForUnsupportedGraphs(
        ExecutionProviderKind provider,
        IReadOnlyList<string> modelPaths,
        out string? fallbackReason)
    {
        fallbackReason = null;
        if (provider is not ExecutionProviderKind.TensorRTRtx)
        {
            return provider;
        }

        foreach (string path in modelPaths)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                continue;
            }

            IReadOnlyList<string> unsupported = TensorRtRtx.TrtRtxUnsupportedOpScanner.FindUnsupportedOps(path);
            if (unsupported.Count == 0)
            {
                continue;
            }

            ExecutionProviderKind next = OperatingSystem.IsWindows()
                ? ExecutionProviderKind.DirectMl
                : ExecutionProviderKind.Cpu;
            fallbackReason =
                $"TensorRT-RTX cannot import '{Path.GetFileName(path)}' (unsupported ops: {string.Join(", ", unsupported)}); " +
                $"selected {FormatProviderLabel(next)} without attempting TensorRT-RTX.";
            return next;
        }

        return provider;
    }

    private static IEnumerable<ExecutionProviderKind> EnumerateTrtInitFallbackProviders()
    {
        if (OperatingSystem.IsWindows())
        {
            yield return ExecutionProviderKind.DirectMl;
        }

        yield return ExecutionProviderKind.Cpu;
    }

    internal static bool LooksLikeTrtSessionInitFailure(Exception exception)
    {
        for (Exception? current = exception; current is not null; current = current.InnerException)
        {
            string message = current.Message;
            if (message.Contains("Kernel not found", StringComparison.OrdinalIgnoreCase)
                || message.Contains("ModelImporter", StringComparison.OrdinalIgnoreCase)
                || message.Contains("No graph will run on TensorRT", StringComparison.OrdinalIgnoreCase)
                || message.Contains("NvTensorRTRTX", StringComparison.OrdinalIgnoreCase)
                || message.Contains("TensorRT-RTX", StringComparison.OrdinalIgnoreCase)
                || message.Contains("TensorRT RTX", StringComparison.OrdinalIgnoreCase)
                || message.Contains("transformer_memcpy", StringComparison.OrdinalIgnoreCase)
                || message.Contains("ProcessInitializers", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private static bool IsRecoverableTrtFallbackInitFailure(Exception exception) =>
        LooksLikeTrtSessionInitFailure(exception)
        || exception is OnnxRuntimeException
        || exception is InvalidOperationException
        || exception is DllNotFoundException
        || exception is EntryPointNotFoundException;

    private static string SummarizeExceptionMessage(Exception exception)
    {
        string message = exception.GetBaseException().Message.ReplaceLineEndings(" ");
        const int maxLength = 240;
        return message.Length <= maxLength ? message : message[..maxLength] + "...";
    }

    public static async Task<WhisperSessionLease> CreateWhisperAsync(
        string encoderModelPath,
        string decoderModelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        IReadOnlyDictionary<string, string>? additionalTrtEncoderOptions = null,
        IReadOnlyDictionary<string, string>? additionalTrtDecoderOptions = null)
    {
        BootstrapContext bootstrap = await BootstrapForProviderAsync(provider, cancellationToken)
            .ConfigureAwait(false);
        DualOptionsSelections selections = CreateDualOptionsSelections(
            provider, bootstrap.Bootstrap, bootstrap.DevicePolicy,
            additionalTrtEncoderOptions, additionalTrtDecoderOptions,
            encoderModelPath: encoderModelPath, decoderModelPath: decoderModelPath);
        using SessionOptions encoderOptions = selections.Encoder.Options;
        using SessionOptions decoderOptions = selections.Decoder.Options;
        InferenceSession? encoderSession = null;
        InferenceSession? decoderSession = null;
        try
        {
            // Through CreateSession so smoke probes load (or capture) EP-context artifacts like
            // pooled sessions do, instead of rebuilding every engine from source.
            encoderSession = CreateSession(encoderModelPath, encoderOptions, null, cancellationToken, selections.Encoder.SelectedProvider, bootstrap.DevicePolicy);
            decoderSession = CreateSession(decoderModelPath, decoderOptions, null, cancellationToken, selections.Decoder.SelectedProvider, bootstrap.DevicePolicy);
            DualSessionMetadata metadata = ResolveDualSessionMetadata(
                provider, bootstrap, selections, encoderSession, decoderSession);
            CpuExecutionAdmission.Shared.RegisterSession(encoderSession, metadata.EncoderProvider);
            CpuExecutionAdmission.Shared.RegisterSession(decoderSession, metadata.DecoderProvider);
            TensorRtRtxTeardownGuard.ReconcileTracking(encoderSession, metadata.EncoderProvider);
            TensorRtRtxTeardownGuard.ReconcileTracking(decoderSession, metadata.DecoderProvider);

            return new WhisperSessionLease(
                encoderSession,
                decoderSession,
                bootstrap.RequestedProviderLabel,
                FormatProviderLabel(metadata.SelectedProvider),
                metadata.BootstrapDetail);
        }
        catch (Exception ex)
        {
            ReleaseSessionsAfterCreateFailure(ex, encoderSession, decoderSession);
            throw;
        }
    }

    public static async Task<OpusSessionLease> CreateOpusAsync(
        string encoderModelPath,
        string decoderModelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        IReadOnlyDictionary<string, string>? additionalTrtEncoderOptions = null,
        IReadOnlyDictionary<string, string>? additionalTrtDecoderOptions = null)
    {
        BootstrapContext bootstrap = await BootstrapForProviderAsync(provider, cancellationToken)
            .ConfigureAwait(false);
        DualOptionsSelections selections = CreateDualOptionsSelections(
            provider, bootstrap.Bootstrap, bootstrap.DevicePolicy,
            additionalTrtEncoderOptions, additionalTrtDecoderOptions,
            encoderModelPath: encoderModelPath, decoderModelPath: decoderModelPath);
        using SessionOptions encoderOptions = selections.Encoder.Options;
        using SessionOptions decoderOptions = selections.Decoder.Options;
        InferenceSession? encoderSession = null;
        InferenceSession? decoderSession = null;
        try
        {
            // Through CreateSession so smoke probes load (or capture) EP-context artifacts like
            // pooled sessions do, instead of rebuilding every engine from source.
            encoderSession = CreateSession(encoderModelPath, encoderOptions, null, cancellationToken, selections.Encoder.SelectedProvider, bootstrap.DevicePolicy);
            decoderSession = CreateSession(decoderModelPath, decoderOptions, null, cancellationToken, selections.Decoder.SelectedProvider, bootstrap.DevicePolicy);
            DualSessionMetadata metadata = ResolveDualSessionMetadata(
                provider, bootstrap, selections, encoderSession, decoderSession);
            CpuExecutionAdmission.Shared.RegisterSession(encoderSession, metadata.EncoderProvider);
            CpuExecutionAdmission.Shared.RegisterSession(decoderSession, metadata.DecoderProvider);
            TensorRtRtxTeardownGuard.ReconcileTracking(encoderSession, metadata.EncoderProvider);
            TensorRtRtxTeardownGuard.ReconcileTracking(decoderSession, metadata.DecoderProvider);

            return new OpusSessionLease(
                encoderSession,
                decoderSession,
                bootstrap.RequestedProviderLabel,
                FormatProviderLabel(metadata.SelectedProvider),
                metadata.BootstrapDetail)
            {
                SelectedProviderKind = metadata.SelectedProvider
            };
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

        if (requestedProvider is ExecutionProviderKind.Cuda
            && bootstrapSelectedProvider is ExecutionProviderKind.Cpu
            && OperatingSystem.IsWindows())
        {
            return ExecutionProviderKind.DirectMl;
        }

        return bootstrapSelectedProvider;
    }

    internal static string FormatProviderLabel(ExecutionProviderKind provider) =>
        provider switch
        {
            ExecutionProviderKind.Cpu => "cpu",
            ExecutionProviderKind.DirectMl => "dml",
            ExecutionProviderKind.TensorRTRtx => "tensorrt-rtx",
            ExecutionProviderKind.OpenVino => "openvino",
            ExecutionProviderKind.CoreMl => "coreml",
            ExecutionProviderKind.Cuda => "cuda",
            ExecutionProviderKind.TensorRt => "tensorrt",
            ExecutionProviderKind.Migraphx => "migraphx",
            ExecutionProviderKind.Dnnl => "dnnl",
            ExecutionProviderKind.Qnn => "qnn",
            ExecutionProviderKind.OpenVinoCatalog => "openvino-catalog",
            ExecutionProviderKind.VitisAi => "vitisai",
            _ => throw new ArgumentOutOfRangeException(nameof(provider), provider, "Unsupported execution provider kind.")
        };

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
            ? $"Requested {FormatProviderLabel(requestedProvider)} but effective {FormatProviderLabel(effectiveProvider)}."
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
        var trtDevice = devices.FirstOrDefault(d => IsTensorRtRtxDeviceCandidate(d.EpName, d.HardwareDevice.Type));
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

    internal static bool IsTensorRtDeviceCandidate(string epName, OrtHardwareDeviceType hardwareDeviceType) =>
        hardwareDeviceType is OrtHardwareDeviceType.GPU &&
        string.Equals(epName, ClassicTensorRtExecutionProviderName, StringComparison.Ordinal);

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
        var trtDevice = devices.FirstOrDefault(d => IsTensorRtDeviceCandidate(d.EpName, d.HardwareDevice.Type));
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
        OrtEpDevice? directMlDevice = devices.FirstOrDefault(d => IsDirectMlDeviceCandidate(d.EpName, d.HardwareDevice.Type))
            ?? devices.FirstOrDefault(d => IsDirectMlExecutionProviderName(d.EpName));
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
            options.AppendExecutionProvider(DirectMlExecutionProviderName);
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

    internal static bool IsTensorRtRtxDeviceCandidate(string epName, OrtHardwareDeviceType hardwareDeviceType) =>
        // Only accept the single canonical standalone EP ABI plugin name.
        // "NvTensorRtExecutionProvider" and "TensorrtExecutionProvider" are the old CUDA-based TensorRT EP —
        // they must NOT be treated as TRT RTX candidates (different EP family, different options).
        hardwareDeviceType is OrtHardwareDeviceType.GPU &&
        string.Equals(epName, TensorRtRtxExecutionProviderName, StringComparison.Ordinal);

    private static bool IsDirectMlExecutionProviderName(string epName) =>
        string.Equals(epName, DirectMlExecutionProviderName, StringComparison.OrdinalIgnoreCase) ||
        string.Equals(epName, DirectMlLongExecutionProviderName, StringComparison.OrdinalIgnoreCase);

    private static bool IsDirectMlDeviceCandidate(string epName, OrtHardwareDeviceType hardwareDeviceType) =>
        hardwareDeviceType is OrtHardwareDeviceType.GPU &&
        IsDirectMlExecutionProviderName(epName);

    private static bool IsDnnlExecutionProviderName(string epName) =>
        string.Equals(epName, DnnlExecutionProviderName, StringComparison.OrdinalIgnoreCase) ||
        string.Equals(epName, DnnlUpperExecutionProviderName, StringComparison.OrdinalIgnoreCase);

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
        ShouldUseCatalogDevicePolicy(devicePolicy, selectedProviderKind);

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

    private static bool IsCatalogGpuProvider(ExecutionProviderKind provider) =>
        provider is ExecutionProviderKind.DirectMl
            or ExecutionProviderKind.Migraphx;

    private static bool IsNativeGpuProvider(ExecutionProviderKind provider) =>
        !OperatingSystem.IsMacOS() && provider is (ExecutionProviderKind.Cuda or ExecutionProviderKind.TensorRt);

    internal static bool ShouldUseCatalogDevicePolicy(
        WindowsMlExecutionDevicePolicy devicePolicy,
        ExecutionProviderKind provider) =>
        OperatingSystem.IsWindows() &&
        devicePolicy != WindowsMlExecutionDevicePolicy.Explicit &&
        IsWindowsMlCatalogProvider(provider);

    /// <summary>
    /// WinML catalog EPs that can take ORT <c>SetEpSelectionPolicy</c> (device auto-select among
    /// registered catalog devices). DirectML stays on the legacy explicit-append path; native
    /// CUDA/TensorRT/TensorRT RTX/DNNL/CoreML are not Windows ML catalog routes.
    /// </summary>
    private static bool IsWindowsMlCatalogProvider(ExecutionProviderKind provider) =>
        provider is ExecutionProviderKind.Migraphx
            or ExecutionProviderKind.Qnn
            or ExecutionProviderKind.VitisAi
            or ExecutionProviderKind.OpenVinoCatalog;

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

    internal sealed record SingleSessionLease(
        InferenceSession Session,
        string RequestedProvider,
        string SelectedProvider,
        string? BootstrapDetail) : IDisposable
    {
        internal SessionLease? PoolLease { get; init; }

        /// <summary>Set when the lease came from the shared pool — used for residency pinning.</summary>
        internal SessionPoolKey? ResolvedPoolKey { get; init; }

        public void Dispose()
        {
            if (PoolLease is not null)
            {
                PoolLease.Dispose();
            }
            else
            {
                TensorRtRtxTeardownGuard.DisposeSafely(Session);
            }
        }
    }

    internal sealed record Qwen3AsrSessionLease(
        InferenceSession EncoderSession,
        InferenceSession DecoderInitSession,
        InferenceSession DecoderStepSession,
        string RequestedProvider,
        string SelectedProvider,
        string? BootstrapDetail) : IDisposable
    {
        internal SessionLeaseBundle? PoolBundle { get; init; }
        internal SessionLease? EncoderPoolLease { get; init; }
        internal SessionLease? DecoderInitPoolLease { get; init; }
        internal SessionLease? DecoderStepPoolLease { get; init; }

        public void Dispose()
        {
            if (PoolBundle is not null)
            {
                PoolBundle.Dispose();
                return;
            }

            if (EncoderPoolLease is not null)
            {
                EncoderPoolLease.Dispose();
            }
            else
            {
                TensorRtRtxTeardownGuard.DisposeSafely(EncoderSession);
            }

            if (DecoderInitPoolLease is not null)
            {
                DecoderInitPoolLease.Dispose();
            }
            else
            {
                TensorRtRtxTeardownGuard.DisposeSafely(DecoderInitSession);
            }

            if (DecoderStepPoolLease is not null)
            {
                DecoderStepPoolLease.Dispose();
            }
            else
            {
                TensorRtRtxTeardownGuard.DisposeSafely(DecoderStepSession);
            }
        }
    }

    internal sealed record LatentSyncSessionLease(
        InferenceSession UNetSession,
        InferenceSession VaeEncoderSession,
        InferenceSession VaeDecoderSession,
        InferenceSession WhisperEncoderSession,
        string RequestedProvider,
        string SelectedProvider,
        string? BootstrapDetail) : IDisposable
    {
        internal SessionLeaseBundle? PoolBundle { get; init; }
        internal SessionLease? UNetPoolLease { get; init; }
        internal SessionLease? VaeEncoderPoolLease { get; init; }
        internal SessionLease? VaeDecoderPoolLease { get; init; }
        internal SessionLease? WhisperEncoderPoolLease { get; init; }

        public void Dispose()
        {
            if (PoolBundle is not null)
            {
                PoolBundle.Dispose();
                return;
            }

            if (UNetPoolLease is not null)
            {
                UNetPoolLease.Dispose();
            }
            else
            {
                TensorRtRtxTeardownGuard.DisposeSafely(UNetSession);
            }

            if (VaeEncoderPoolLease is not null)
            {
                VaeEncoderPoolLease.Dispose();
            }
            else
            {
                TensorRtRtxTeardownGuard.DisposeSafely(VaeEncoderSession);
            }

            if (VaeDecoderPoolLease is not null)
            {
                VaeDecoderPoolLease.Dispose();
            }
            else
            {
                TensorRtRtxTeardownGuard.DisposeSafely(VaeDecoderSession);
            }

            if (WhisperEncoderPoolLease is not null)
            {
                WhisperEncoderPoolLease.Dispose();
            }
            else
            {
                TensorRtRtxTeardownGuard.DisposeSafely(WhisperEncoderSession);
            }
        }
    }

    internal sealed record NemotronAsrSessionLease(
        InferenceSession EncoderSession,
        InferenceSession DecoderJointSession,
        string RequestedProvider,
        string SelectedProvider,
        string? BootstrapDetail) : IDisposable
    {
        internal SessionLease? EncoderPoolLease { get; init; }
        internal SessionLease? DecoderJointPoolLease { get; init; }

        public void Dispose()
        {
            if (EncoderPoolLease is not null)
            {
                EncoderPoolLease.Dispose();
            }
            else
            {
                TensorRtRtxTeardownGuard.DisposeSafely(EncoderSession);
            }

            if (DecoderJointPoolLease is not null)
            {
                DecoderJointPoolLease.Dispose();
            }
            else
            {
                TensorRtRtxTeardownGuard.DisposeSafely(DecoderJointSession);
            }
        }
    }

    internal sealed record WhisperSessionLease(
        InferenceSession EncoderSession,
        InferenceSession DecoderSession,
        string RequestedProvider,
        string SelectedProvider,
        string? BootstrapDetail) : IDisposable
    {
        internal SessionLease? EncoderPoolLease { get; init; }
        internal SessionLease? DecoderPoolLease { get; init; }

        public void Dispose()
        {
            if (EncoderPoolLease is not null)
            {
                EncoderPoolLease.Dispose();
            }
            else
            {
                TensorRtRtxTeardownGuard.DisposeSafely(EncoderSession);
            }

            if (DecoderPoolLease is not null)
            {
                DecoderPoolLease.Dispose();
            }
            else
            {
                TensorRtRtxTeardownGuard.DisposeSafely(DecoderSession);
            }
        }
    }

    internal sealed record OpusSessionLease(
        InferenceSession EncoderSession,
        InferenceSession DecoderSession,
        string RequestedProvider,
        string SelectedProvider,
        string? BootstrapDetail) : IDisposable
    {
        /// <summary>Provider the sessions actually run on, after any downgrade.</summary>
        internal ExecutionProviderKind SelectedProviderKind { get; init; } = ExecutionProviderKind.Cpu;

        internal SessionLease? EncoderPoolLease { get; init; }
        internal SessionLease? DecoderPoolLease { get; init; }

        public void Dispose()
        {
            if (EncoderPoolLease is not null)
            {
                EncoderPoolLease.Dispose();
            }
            else
            {
                TensorRtRtxTeardownGuard.DisposeSafely(EncoderSession);
            }

            if (DecoderPoolLease is not null)
            {
                DecoderPoolLease.Dispose();
            }
            else
            {
                TensorRtRtxTeardownGuard.DisposeSafely(DecoderSession);
            }
        }
    }
}

internal static partial class OnnxExecutionSessionFactory
{
    /// <summary>
    /// Pre-warms a pooled single session and immediately releases it, leaving the warm
    /// session in <see cref="InferenceSessionPool"/> for a later stage run. Use at startup
    /// or after plan resolution to amortise first-call latency.
    /// </summary>
    public static async Task WarmPooledSingleAsync(
        string engineFamily,
        string modelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        IReadOnlyDictionary<string, string>? additionalTrtOptions = null,
        bool allowTrtInitFallback = true)
    {
        using SingleSessionLease _ = await CreatePooledSingleAsync(
            engineFamily,
            modelPath,
            provider,
            cancellationToken,
            additionalTrtOptions: additionalTrtOptions,
            allowTrtInitFallback: allowTrtInitFallback).ConfigureAwait(false);
    }

    public static async Task<SingleSessionLease> CreatePooledSingleAsync(
        string engineFamily,
        string modelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        InferenceSessionPool? pool = null,
        string? modelId = null,
        string? variant = null,
        IReadOnlyDictionary<string, string>? additionalTrtOptions = null,
        Func<string, SessionOptions, InferenceSession>? sessionFactory = null,
        bool allowTrtInitFallback = true)
    {
        ArgumentNullException.ThrowIfNull(engineFamily);
        pool ??= InferenceSessionPool.Shared;

        string requestedProvider = FormatProviderLabel(provider);
        var bootstrapResult = await BootstrapTrackedAsync(provider, allowDownloads: true, cancellationToken)
            .ConfigureAwait(false);
        WindowsMlExecutionDevicePolicy devicePolicy = await ResolveDevicePolicyAsync(cancellationToken)
            .ConfigureAwait(false);
        SessionOptionsSelection optionsSelection = CreateSessionOptions(new SessionOptionsRequest(
            ResolveSessionOptionsProvider(provider, bootstrapResult.SelectedProvider),
            devicePolicy,
            additionalTrtOptions,
            TrtModelPath: modelPath));

        // Options ownership may transfer to a replacement SessionOptionsSelection when TRT init
        // falls back; keep a movable dispose target managed by a `using` scope.
        using var optionsHolder = new SessionOptionsDisposeHolder(optionsSelection.Options);
        SessionOptionsSelection leaseSelection = optionsSelection;
        bool useCatalogDevicePolicy = ShouldUseCatalogDevicePolicy(devicePolicy, optionsSelection.SelectedProvider);
        ExecutionProviderKind optionsSelectedProvider = optionsSelection.SelectedProvider;

        string optionsFingerprint =
            $"{BuildSessionOptionsFingerprint(optionsSelectedProvider, devicePolicy, additionalTrtOptions)}|trt-init-fallback:{allowTrtInitFallback}";
        SessionPoolKey key = WithOpenVinoResidency(await SessionPoolKey.CreateAsync(
            engineFamily,
            modelId,
            variant,
            optionsSelection.SelectedProvider,
            modelPath,
            deviceId: null,
            graphRole: "default",
            optionsFingerprint,
            cancellationToken).ConfigureAwait(false));
        if (allowTrtInitFallback && optionsSelection.SelectedProvider is ExecutionProviderKind.TensorRTRtx)
        {
            // TRT RTX init can still fall back to DirectML/CPU inside the pool factory, after this
            // key is admitted; reserve for the fallback provider, not TRT RTX's measured factor.
            key = key with
            {
                EstimatedVramMb = SessionPoolKey.EstimateVramMb(
                    modelPath, optionsSelection.SelectedProvider, providerMayFallBack: true),
            };
        }

        SessionLease? poolLease = null;
        try
        {
            // The key is admitted for the pre-creation provider; reporting the provider the
            // fallback actually created the session with lets the pool move a CPU fallback's
            // accounting from the device budget to host RAM.
            poolLease = await pool
                .GetLeaseReportingProviderAsync(
                    key,
                    ct =>
                    {
                        (InferenceSession session, SessionOptionsSelection selection) =
                            CreateInferenceSessionWithTrtInitFallback(
                                modelPath,
                                optionsSelection,
                                devicePolicy,
                                additionalTrtOptions,
                                sessionFactory,
                                ct,
                                allowTrtInitFallback);
                        leaseSelection = selection;
                        optionsSelectedProvider = selection.SelectedProvider;
                        useCatalogDevicePolicy = ShouldUseCatalogDevicePolicy(devicePolicy, selection.SelectedProvider);
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
                    cancellationToken)
                .ConfigureAwait(false);

            ExecutionProviderKind effectiveProvider = ResolveEffectiveProviderKindFromSession(
                poolLease.Session,
                optionsSelectedProvider,
                useCatalogDevicePolicy);
            string selectedProvider = FormatProviderLabel(effectiveProvider);
            string? epFallbackReason = BuildSessionOptionsFallbackReason(provider, effectiveProvider, leaseSelection);
            string? bootstrapDetail = FormatBootstrapDetail(bootstrapResult.Detail, epFallbackReason);
            CpuExecutionAdmission.Shared.RegisterSession(poolLease.Session, effectiveProvider);
            TensorRtRtxTeardownGuard.ReconcileTracking(poolLease.Session, effectiveProvider);

            return new SingleSessionLease(poolLease.Session, requestedProvider, selectedProvider, bootstrapDetail)
            {
                PoolLease = poolLease,
                ResolvedPoolKey = key,
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
        string engineFamily,
        string modelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        InferenceSessionPool? pool = null,
        string? modelId = null,
        string? variant = null,
        IReadOnlyDictionary<string, string>? additionalTrtOptions = null,
        Func<string, SessionOptions, InferenceSession>? sessionFactory = null,
        bool allowTrtInitFallback = true)
    {
        pool ??= InferenceSessionPool.Shared;

        SessionPoolKey? key = null;
        string requestedProvider = FormatProviderLabel(provider);
        string selectedProvider = requestedProvider;
        string? bootstrapDetail = null;

        async Task<SingleSessionLease> AcquireAsync(CancellationToken ct)
        {
            SingleSessionLease lease = await CreatePooledSingleAsync(
                engineFamily,
                modelPath,
                provider,
                ct,
                pool,
                modelId,
                variant,
                additionalTrtOptions,
                sessionFactory,
                allowTrtInitFallback).ConfigureAwait(false);
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

    public static async Task<WhisperSessionLease> CreatePooledWhisperAsync(
        string engineFamily,
        string encoderModelPath,
        string decoderModelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        InferenceSessionPool? pool = null,
        string? modelId = null,
        string? variant = null,
        IReadOnlyDictionary<string, string>? additionalTrtEncoderOptions = null,
        IReadOnlyDictionary<string, string>? additionalTrtDecoderOptions = null,
        Func<string, SessionOptions, InferenceSession>? sessionFactory = null,
        bool allowTrtInitFallback = true)
    {
        ArgumentNullException.ThrowIfNull(engineFamily);
        pool ??= InferenceSessionPool.Shared;

        string requestedProvider = FormatProviderLabel(provider);
        ExecutionProviderKind selectedProviderKind = DowngradeTrtRtxForUnsupportedGraphs(
            provider,
            [encoderModelPath, decoderModelPath],
            out string? graphFallbackReason);

        // TensorRT RTX can fail a Whisper session at engine build or load (CUDA OOM, importer
        // errors) after the pre-flight passed. Walk the same DirectML (Windows) then CPU chain the
        // single-session path uses, recording why each provider was abandoned.
        List<ExecutionProviderKind> attempts = [selectedProviderKind];
        if (allowTrtInitFallback && selectedProviderKind is ExecutionProviderKind.TensorRTRtx)
        {
            attempts.AddRange(EnumerateTrtInitFallbackProviders());
        }

        string? abandonedReason = null;
        for (int attempt = 0; ; attempt++)
        {
            ExecutionProviderKind attemptProvider = attempts[attempt];
            try
            {
                DualPooledLeasePair pair = await AcquirePooledWhisperPairAsync(
                    engineFamily, encoderModelPath, decoderModelPath, attemptProvider, cancellationToken,
                    pool, modelId, variant, additionalTrtEncoderOptions, additionalTrtDecoderOptions,
                    sessionFactory).ConfigureAwait(false);

                return new WhisperSessionLease(
                    pair.EncoderLease.Session, pair.DecoderLease.Session,
                    requestedProvider, FormatProviderLabel(pair.SelectedProvider),
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

                if (attempt < attempts.Count - 1
                    && !cancellationToken.IsCancellationRequested
                    && IsRecoverableTrtFallbackInitFailure(ex))
                {
                    try
                    {
                        await pool.EvictModelAsync(engineFamily, cancellationToken: cancellationToken).ConfigureAwait(false);
                    }
                    catch (Exception evictionFailure) when (evictionFailure is not OperationCanceledException)
                    {
                        // A failed eviction must not abort the fallback chain or replace the init failure.
                        _logger?.LogWarning(
                            evictionFailure,
                            "Evicting {EngineFamily} after a failed {Provider} Whisper init failed; continuing fallback.",
                            engineFamily,
                            FormatProviderLabel(attemptProvider));
                    }
                    abandonedReason = MergeFallbackReasons(
                        abandonedReason,
                        $"{FormatProviderLabel(attemptProvider)} Whisper session init failed ({SummarizeExceptionMessage(ex)}); "
                        + $"fell back to {FormatProviderLabel(attempts[attempt + 1])}.");
                    continue;
                }

                throw;
            }
        }
    }

    private static async Task<DualPooledLeasePair> AcquirePooledWhisperPairAsync(
        string engineFamily,
        string encoderModelPath,
        string decoderModelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        InferenceSessionPool pool,
        string? modelId,
        string? variant,
        IReadOnlyDictionary<string, string>? additionalTrtEncoderOptions,
        IReadOnlyDictionary<string, string>? additionalTrtDecoderOptions,
        Func<string, SessionOptions, InferenceSession>? sessionFactory)
    {
        BootstrapContext bootstrap = await BootstrapForProviderAsync(provider, cancellationToken)
            .ConfigureAwait(false);
        DualOptionsSelections selections = CreateDualOptionsSelections(
            provider, bootstrap.Bootstrap, bootstrap.DevicePolicy,
            additionalTrtEncoderOptions, additionalTrtDecoderOptions,
            encoderLogId: ExtractLogId(encoderModelPath),
            decoderLogId: ExtractLogId(decoderModelPath),
            encoderModelPath: encoderModelPath, decoderModelPath: decoderModelPath);
        using SessionOptions encoderOptions = selections.Encoder.Options;
        using SessionOptions decoderOptions = selections.Decoder.Options;
        (SessionPoolKey encoderKey, SessionPoolKey decoderKey) = await BuildDualPooledKeysAsync(
            engineFamily, encoderModelPath, decoderModelPath, selections,
            bootstrap.DevicePolicy, additionalTrtEncoderOptions, additionalTrtDecoderOptions,
            modelId, variant, cancellationToken).ConfigureAwait(false);

        return await AcquireDualPooledSessionsAsync(
            provider, bootstrap, selections,
            encoderModelPath, decoderModelPath,
            encoderKey, decoderKey, pool, cancellationToken, sessionFactory)
            .ConfigureAwait(false);
    }

    public static async Task<Qwen3AsrSessionLease> CreatePooledQwen3AsrAsync(
        string engineFamily,
        string encoderModelPath,
        string decoderInitModelPath,
        string decoderStepModelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        InferenceSessionPool? pool = null,
        string? modelId = null,
        string? variant = null,
        IReadOnlyDictionary<string, string>? additionalTrtEncoderOptions = null,
        IReadOnlyDictionary<string, string>? additionalTrtDecoderOptions = null,
        Func<string, SessionOptions, InferenceSession>? sessionFactory = null)
    {
        ArgumentNullException.ThrowIfNull(engineFamily);
        pool ??= InferenceSessionPool.Shared;

        // Per-role TRT eligibility: a contrib-op decoder must not force the (often much larger)
        // encoder off TensorRT RTX. Encoder is the cold-load hot path (qwen3-asr encoder ~711MB).
        ExecutionProviderKind encoderProvider = DowngradeTrtRtxForUnsupportedGraphs(
            provider,
            [encoderModelPath],
            out string? encoderGraphReason);
        ExecutionProviderKind decoderProvider = DowngradeTrtRtxForUnsupportedGraphs(
            provider,
            [decoderInitModelPath, decoderStepModelPath],
            out string? decoderGraphReason);
        string? graphFallbackReason = MergeFallbackReasons(encoderGraphReason, decoderGraphReason);

        string requestedProvider = FormatProviderLabel(provider);
        ExecutionProviderKind bootstrapProvider =
            encoderProvider is ExecutionProviderKind.TensorRTRtx ||
            decoderProvider is ExecutionProviderKind.TensorRTRtx
                ? ExecutionProviderKind.TensorRTRtx
                : encoderProvider;
        var bootstrapResult = await BootstrapTrackedAsync(bootstrapProvider, allowDownloads: true, cancellationToken)
            .ConfigureAwait(false);
        WindowsMlExecutionDevicePolicy devicePolicy = await ResolveDevicePolicyAsync(cancellationToken)
            .ConfigureAwait(false);
        SessionOptionsSelection encoderOptionsSelection = CreateSessionOptions(new SessionOptionsRequest(
            encoderProvider == bootstrapProvider
                ? ResolveSessionOptionsProvider(encoderProvider, bootstrapResult.SelectedProvider)
                : encoderProvider,
            devicePolicy,
            additionalTrtEncoderOptions,
            TrtModelPath: encoderModelPath));
        SessionOptionsSelection decoderOptionsSelection = CreateSessionOptions(new SessionOptionsRequest(
            decoderProvider == bootstrapProvider
                ? ResolveSessionOptionsProvider(decoderProvider, bootstrapResult.SelectedProvider)
                : decoderProvider,
            devicePolicy,
            additionalTrtDecoderOptions,
            TrtModelPath: decoderInitModelPath));

        using SessionOptions encoderOptions = encoderOptionsSelection.Options;
        using SessionOptions decoderInitOptions = decoderOptionsSelection.Options;
        SessionOptions decoderStepOptions = decoderOptionsSelection.Options;

        ExecutionProviderKind encoderOptionsSelectedProvider = encoderOptionsSelection.SelectedProvider;
        ExecutionProviderKind decoderOptionsSelectedProvider = decoderOptionsSelection.SelectedProvider;

        string encoderOptionsFingerprint = BuildSessionOptionsFingerprint(encoderOptionsSelectedProvider, devicePolicy, additionalTrtEncoderOptions);
        string decoderOptionsFingerprint = BuildSessionOptionsFingerprint(decoderOptionsSelectedProvider, devicePolicy, additionalTrtDecoderOptions);

        SessionPoolKey[] qwenKeys = await SessionPoolKey.CreateManyAsync(
            [
                new SessionPoolKey.PooledKeyRequest(
                    engineFamily, modelId, variant, encoderOptionsSelection.SelectedProvider,
                    encoderModelPath, null, "encoder", encoderOptionsFingerprint),
                new SessionPoolKey.PooledKeyRequest(
                    engineFamily, modelId, variant, decoderOptionsSelection.SelectedProvider,
                    decoderInitModelPath, null, "decoder-init", decoderOptionsFingerprint),
                new SessionPoolKey.PooledKeyRequest(
                    engineFamily, modelId, variant, decoderOptionsSelection.SelectedProvider,
                    decoderStepModelPath, null, "decoder-step", decoderOptionsFingerprint),
            ],
            cancellationToken).ConfigureAwait(false);
        SessionPoolKey encoderKey = WithOpenVinoResidency(qwenKeys[0]);
        SessionPoolKey decoderInitKey = WithOpenVinoResidency(qwenKeys[1]);
        SessionPoolKey decoderStepKey = WithOpenVinoResidency(qwenKeys[2]);

        // Atomic bundle (audit §3A): never hold the encoder while waiting on decoders.
        SessionLeaseBundle bundle = await pool.GetLeaseBundleAsync(
            [
                new SessionLeaseRequest(
                    encoderKey,
                    ct => Task.FromResult(CreateSession(encoderModelPath, encoderOptionsSelection.Options, sessionFactory, ct, encoderOptionsSelection.SelectedProvider, devicePolicy)),
                    session => ResolveEffectiveProviderKindFromSession(
                        session,
                        encoderOptionsSelection.SelectedProvider,
                        ShouldUseCatalogDevicePolicy(devicePolicy, encoderOptionsSelection.SelectedProvider))),
                new SessionLeaseRequest(
                    decoderInitKey,
                    ct => Task.FromResult(CreateSession(decoderInitModelPath, decoderInitOptions, sessionFactory, ct, decoderOptionsSelectedProvider, devicePolicy)),
                    session => ResolveEffectiveProviderKindFromSession(
                        session,
                        decoderOptionsSelectedProvider,
                        ShouldUseCatalogDevicePolicy(devicePolicy, decoderOptionsSelectedProvider))),
                new SessionLeaseRequest(
                    decoderStepKey,
                    ct => Task.FromResult(CreateSession(decoderStepModelPath, decoderStepOptions, sessionFactory, ct, decoderOptionsSelectedProvider, devicePolicy)),
                    session => ResolveEffectiveProviderKindFromSession(
                        session,
                        decoderOptionsSelectedProvider,
                        ShouldUseCatalogDevicePolicy(devicePolicy, decoderOptionsSelectedProvider))),
            ],
            cancellationToken).ConfigureAwait(false);

        try
        {
            SessionLease encoderPoolLease = bundle.LeaseAt(0);
            SessionLease decoderInitPoolLease = bundle.LeaseAt(1);
            SessionLease decoderStepPoolLease = bundle.LeaseAt(2);

            ExecutionProviderKind encoderEffective = ResolveEffectiveProviderKindFromSession(
                encoderPoolLease.Session,
                encoderOptionsSelectedProvider,
                ShouldUseCatalogDevicePolicy(devicePolicy, encoderOptionsSelectedProvider));
            ExecutionProviderKind decoderInitEffective = ResolveEffectiveProviderKindFromSession(
                decoderInitPoolLease.Session,
                decoderOptionsSelectedProvider,
                ShouldUseCatalogDevicePolicy(devicePolicy, decoderOptionsSelectedProvider));
            ExecutionProviderKind decoderStepEffective = ResolveEffectiveProviderKindFromSession(
                decoderStepPoolLease.Session,
                decoderOptionsSelectedProvider,
                ShouldUseCatalogDevicePolicy(devicePolicy, decoderOptionsSelectedProvider));
            ExecutionProviderKind effectiveProvider =
                encoderEffective == decoderInitEffective && encoderEffective == decoderStepEffective
                    ? encoderEffective
                    : ExecutionProviderKind.Cpu;
            string selectedProvider = FormatProviderLabel(effectiveProvider);
            string? encoderFallbackReason = BuildSessionOptionsFallbackReason(
                provider,
                encoderEffective,
                encoderOptionsSelection);
            string? decoderInitFallbackReason = BuildSessionOptionsFallbackReason(
                provider,
                decoderInitEffective,
                decoderOptionsSelection);
            string? decoderStepFallbackReason = BuildSessionOptionsFallbackReason(
                provider,
                decoderStepEffective,
                decoderOptionsSelection);
            string? bootstrapDetail = FormatBootstrapDetail(
                bootstrapResult.Detail,
                MergeFallbackReasons(
                    graphFallbackReason,
                    MergeFallbackReasons(
                        encoderFallbackReason,
                        MergeFallbackReasons(decoderInitFallbackReason, decoderStepFallbackReason))));

            CpuExecutionAdmission.Shared.RegisterSession(encoderPoolLease.Session, encoderEffective);
            CpuExecutionAdmission.Shared.RegisterSession(decoderInitPoolLease.Session, decoderInitEffective);
            CpuExecutionAdmission.Shared.RegisterSession(decoderStepPoolLease.Session, decoderStepEffective);
            TensorRtRtxTeardownGuard.ReconcileTracking(encoderPoolLease.Session, encoderEffective);
            TensorRtRtxTeardownGuard.ReconcileTracking(decoderInitPoolLease.Session, decoderInitEffective);
            TensorRtRtxTeardownGuard.ReconcileTracking(decoderStepPoolLease.Session, decoderStepEffective);
            return new Qwen3AsrSessionLease(
                encoderPoolLease.Session,
                decoderInitPoolLease.Session,
                decoderStepPoolLease.Session,
                requestedProvider,
                selectedProvider,
                bootstrapDetail)
            {
                PoolBundle = bundle,
                EncoderPoolLease = encoderPoolLease,
                DecoderInitPoolLease = decoderInitPoolLease,
                DecoderStepPoolLease = decoderStepPoolLease,
            };
        }
        catch
        {
            bundle.Dispose();
            throw;
        }
    }

    public static async Task<LatentSyncSessionLease> CreatePooledLatentSyncAsync(
        string engineFamily,
        string unetModelPath,
        string vaeEncoderModelPath,
        string vaeDecoderModelPath,
        string whisperEncoderModelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        InferenceSessionPool? pool = null,
        string? modelId = null,
        string? variant = null,
        Func<string, SessionOptions, InferenceSession>? sessionFactory = null)
    {
        ArgumentNullException.ThrowIfNull(engineFamily);
        pool ??= InferenceSessionPool.Shared;

        string requestedProvider = FormatProviderLabel(provider);
        var bootstrapResult = await BootstrapTrackedAsync(provider, allowDownloads: true, cancellationToken)
            .ConfigureAwait(false);
        WindowsMlExecutionDevicePolicy devicePolicy = await ResolveDevicePolicyAsync(cancellationToken)
            .ConfigureAwait(false);

        ExecutionProviderKind sessionProvider = ResolveSessionOptionsProvider(provider, bootstrapResult.SelectedProvider);
        SessionOptionsSelection unetOptionsSelection = CreateSessionOptions(new SessionOptionsRequest(sessionProvider, devicePolicy, TrtModelPath: unetModelPath));
        SessionOptionsSelection vaeEncOptionsSelection = CreateSessionOptions(new SessionOptionsRequest(sessionProvider, devicePolicy, TrtModelPath: vaeEncoderModelPath));
        SessionOptionsSelection vaeDecOptionsSelection = CreateSessionOptions(new SessionOptionsRequest(sessionProvider, devicePolicy, TrtModelPath: vaeDecoderModelPath));
        SessionOptionsSelection whisperOptionsSelection = CreateSessionOptions(new SessionOptionsRequest(sessionProvider, devicePolicy, TrtModelPath: whisperEncoderModelPath));
        using SessionOptions unetOptions = unetOptionsSelection.Options;
        using SessionOptions vaeEncOptions = vaeEncOptionsSelection.Options;
        using SessionOptions vaeDecOptions = vaeDecOptionsSelection.Options;
        using SessionOptions whisperOptions = whisperOptionsSelection.Options;

        ExecutionProviderKind unetSelectedProvider = unetOptionsSelection.SelectedProvider;
        ExecutionProviderKind whisperSelectedProvider = whisperOptionsSelection.SelectedProvider;

        string unetOptionsFingerprint = BuildSessionOptionsFingerprint(
            unetOptionsSelection.SelectedProvider, devicePolicy, additionalTrtOptions: null);
        string vaeEncOptionsFingerprint = BuildSessionOptionsFingerprint(
            vaeEncOptionsSelection.SelectedProvider, devicePolicy, additionalTrtOptions: null);
        string vaeDecOptionsFingerprint = BuildSessionOptionsFingerprint(
            vaeDecOptionsSelection.SelectedProvider, devicePolicy, additionalTrtOptions: null);
        string whisperOptionsFingerprint = BuildSessionOptionsFingerprint(
            whisperOptionsSelection.SelectedProvider, devicePolicy, additionalTrtOptions: null);

        const string poolEngineFamily = "latentsync-diffusion";
        SessionPoolKey[] latentKeys = await SessionPoolKey.CreateManyAsync(
            [
                new SessionPoolKey.PooledKeyRequest(
                    poolEngineFamily, modelId, variant, unetOptionsSelection.SelectedProvider,
                    unetModelPath, null, "unet", unetOptionsFingerprint),
                new SessionPoolKey.PooledKeyRequest(
                    poolEngineFamily, modelId, variant, vaeEncOptionsSelection.SelectedProvider,
                    vaeEncoderModelPath, null, "vae-encoder", vaeEncOptionsFingerprint),
                new SessionPoolKey.PooledKeyRequest(
                    poolEngineFamily, modelId, variant, vaeDecOptionsSelection.SelectedProvider,
                    vaeDecoderModelPath, null, "vae-decoder", vaeDecOptionsFingerprint),
                new SessionPoolKey.PooledKeyRequest(
                    poolEngineFamily, modelId, variant, whisperOptionsSelection.SelectedProvider,
                    whisperEncoderModelPath, null, "whisper-encoder", whisperOptionsFingerprint),
            ],
            cancellationToken).ConfigureAwait(false);
        SessionPoolKey unetKey = WithOpenVinoResidency(latentKeys[0]);
        SessionPoolKey vaeEncKey = WithOpenVinoResidency(latentKeys[1]);
        SessionPoolKey vaeDecKey = WithOpenVinoResidency(latentKeys[2]);
        SessionPoolKey whisperKey = WithOpenVinoResidency(latentKeys[3]);

        // Atomic bundle (audit §3A): acquire all four graphs or none.
        SessionLeaseBundle bundle = await pool.GetLeaseBundleAsync(
            [
                new SessionLeaseRequest(
                    unetKey,
                    ct => Task.FromResult(CreateSession(unetModelPath, unetOptions, sessionFactory, ct, unetSelectedProvider, devicePolicy)),
                    session => ResolveEffectiveProviderKindFromSession(
                        session,
                        unetOptionsSelection.SelectedProvider,
                        ShouldUseCatalogDevicePolicy(devicePolicy, unetOptionsSelection.SelectedProvider))),
                new SessionLeaseRequest(
                    vaeEncKey,
                    ct => Task.FromResult(CreateSession(vaeEncoderModelPath, vaeEncOptions, sessionFactory, ct, vaeEncOptionsSelection.SelectedProvider, devicePolicy)),
                    session => ResolveEffectiveProviderKindFromSession(
                        session,
                        vaeEncOptionsSelection.SelectedProvider,
                        ShouldUseCatalogDevicePolicy(devicePolicy, vaeEncOptionsSelection.SelectedProvider))),
                new SessionLeaseRequest(
                    vaeDecKey,
                    ct => Task.FromResult(CreateSession(vaeDecoderModelPath, vaeDecOptions, sessionFactory, ct, vaeDecOptionsSelection.SelectedProvider, devicePolicy)),
                    session => ResolveEffectiveProviderKindFromSession(
                        session,
                        vaeDecOptionsSelection.SelectedProvider,
                        ShouldUseCatalogDevicePolicy(devicePolicy, vaeDecOptionsSelection.SelectedProvider))),
                new SessionLeaseRequest(
                    whisperKey,
                    ct => Task.FromResult(CreateSession(whisperEncoderModelPath, whisperOptions, sessionFactory, ct, whisperSelectedProvider, devicePolicy)),
                    session => ResolveEffectiveProviderKindFromSession(
                        session,
                        whisperOptionsSelection.SelectedProvider,
                        ShouldUseCatalogDevicePolicy(devicePolicy, whisperOptionsSelection.SelectedProvider))),
            ],
            cancellationToken).ConfigureAwait(false);

        try
        {
            SessionLease unetPoolLease = bundle.LeaseAt(0);
            SessionLease vaeEncPoolLease = bundle.LeaseAt(1);
            SessionLease vaeDecPoolLease = bundle.LeaseAt(2);
            SessionLease whisperPoolLease = bundle.LeaseAt(3);

            ExecutionProviderKind unetEffective = ResolveEffectiveProviderKindFromSession(
                unetPoolLease.Session,
                unetOptionsSelection.SelectedProvider,
                ShouldUseCatalogDevicePolicy(devicePolicy, unetOptionsSelection.SelectedProvider));
            ExecutionProviderKind vaeEncEffective = ResolveEffectiveProviderKindFromSession(
                vaeEncPoolLease.Session,
                vaeEncOptionsSelection.SelectedProvider,
                ShouldUseCatalogDevicePolicy(devicePolicy, vaeEncOptionsSelection.SelectedProvider));
            ExecutionProviderKind vaeDecEffective = ResolveEffectiveProviderKindFromSession(
                vaeDecPoolLease.Session,
                vaeDecOptionsSelection.SelectedProvider,
                ShouldUseCatalogDevicePolicy(devicePolicy, vaeDecOptionsSelection.SelectedProvider));
            ExecutionProviderKind whisperEffective = ResolveEffectiveProviderKindFromSession(
                whisperPoolLease.Session,
                whisperOptionsSelection.SelectedProvider,
                ShouldUseCatalogDevicePolicy(devicePolicy, whisperOptionsSelection.SelectedProvider));
            ExecutionProviderKind effective =
                unetEffective == vaeEncEffective
                    && unetEffective == vaeDecEffective
                    && unetEffective == whisperEffective
                    ? unetEffective
                    : ExecutionProviderKind.Cpu;
            string selectedProvider = FormatProviderLabel(effective);
            string? unetFallbackReason = BuildSessionOptionsFallbackReason(
                provider, unetEffective, unetOptionsSelection);
            string? vaeEncFallbackReason = BuildSessionOptionsFallbackReason(
                provider, vaeEncEffective, vaeEncOptionsSelection);
            string? vaeDecFallbackReason = BuildSessionOptionsFallbackReason(
                provider, vaeDecEffective, vaeDecOptionsSelection);
            string? whisperFallbackReason = BuildSessionOptionsFallbackReason(
                provider, whisperEffective, whisperOptionsSelection);
            string? bootstrapDetail = FormatBootstrapDetail(
                bootstrapResult.Detail,
                MergeFallbackReasons(
                    MergeFallbackReasons(unetFallbackReason, vaeEncFallbackReason),
                    MergeFallbackReasons(vaeDecFallbackReason, whisperFallbackReason)));

            CpuExecutionAdmission.Shared.RegisterSession(unetPoolLease.Session, unetEffective);
            CpuExecutionAdmission.Shared.RegisterSession(vaeEncPoolLease.Session, vaeEncEffective);
            CpuExecutionAdmission.Shared.RegisterSession(vaeDecPoolLease.Session, vaeDecEffective);
            CpuExecutionAdmission.Shared.RegisterSession(whisperPoolLease.Session, whisperEffective);
            TensorRtRtxTeardownGuard.ReconcileTracking(unetPoolLease.Session, unetEffective);
            TensorRtRtxTeardownGuard.ReconcileTracking(vaeEncPoolLease.Session, vaeEncEffective);
            TensorRtRtxTeardownGuard.ReconcileTracking(vaeDecPoolLease.Session, vaeDecEffective);
            TensorRtRtxTeardownGuard.ReconcileTracking(whisperPoolLease.Session, whisperEffective);
            return new LatentSyncSessionLease(
                unetPoolLease.Session,
                vaeEncPoolLease.Session,
                vaeDecPoolLease.Session,
                whisperPoolLease.Session,
                requestedProvider,
                selectedProvider,
                bootstrapDetail)
            {
                PoolBundle = bundle,
                UNetPoolLease = unetPoolLease,
                VaeEncoderPoolLease = vaeEncPoolLease,
                VaeDecoderPoolLease = vaeDecPoolLease,
                WhisperEncoderPoolLease = whisperPoolLease,
            };
        }
        catch
        {
            bundle.Dispose();
            throw;
        }
    }

    public static async Task<NemotronAsrSessionLease> CreatePooledNemotronAsrAsync(
        string engineFamily,
        string encoderModelPath,
        string decoderJointModelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        InferenceSessionPool? pool = null,
        string? modelId = null,
        string? variant = null,
        IReadOnlyDictionary<string, string>? additionalTrtEncoderOptions = null,
        IReadOnlyDictionary<string, string>? additionalTrtDecoderOptions = null,
        Func<string, SessionOptions, InferenceSession>? sessionFactory = null)
    {
        ArgumentNullException.ThrowIfNull(engineFamily);
        pool ??= InferenceSessionPool.Shared;

        provider = DowngradeTrtRtxForUnsupportedGraphs(
            provider,
            [encoderModelPath, decoderJointModelPath],
            out _);

        BootstrapContext bootstrap = await BootstrapForProviderAsync(provider, cancellationToken)
            .ConfigureAwait(false);
        // CUDA graph replay requires every binding to stay at a fixed address across runs.
        // This flag must stay off here: the dual-session factory is shared with callers whose
        // callsites rebind fresh OrtValues (ParakeetTdtGreedyDecoder, smoke probes), and Nemotron
        // can fall back to managed feeds when shared device allocators are unavailable.
        // Explicit per-caller graph options also flow into the pool-key fingerprint so capture
        // modes cannot collide.
        const bool enableEncoderCudaGraph = false;
        DualOptionsSelections selections = CreateDualOptionsSelections(
            provider, bootstrap.Bootstrap, bootstrap.DevicePolicy,
            additionalTrtEncoderOptions, additionalTrtDecoderOptions, enableEncoderCudaGraph);
        using SessionOptions encoderOptions = selections.Encoder.Options;
        using SessionOptions decoderOptions = selections.Decoder.Options;
        (SessionPoolKey encoderKey, SessionPoolKey decoderKey) = await BuildDualPooledKeysAsync(
            engineFamily, encoderModelPath, decoderJointModelPath, selections,
            bootstrap.DevicePolicy, additionalTrtEncoderOptions, additionalTrtDecoderOptions,
            modelId, variant, cancellationToken, enableEncoderCudaGraph).ConfigureAwait(false);

        DualPooledLeasePair pair = await AcquireDualPooledSessionsAsync(
            provider, bootstrap, selections,
            encoderModelPath, decoderJointModelPath,
            encoderKey, decoderKey, pool, cancellationToken, sessionFactory)
            .ConfigureAwait(false);

        return new NemotronAsrSessionLease(
            pair.EncoderLease.Session,
            pair.DecoderLease.Session,
            pair.RequestedProviderLabel,
            FormatProviderLabel(pair.SelectedProvider),
            pair.BootstrapDetail)
        {
            EncoderPoolLease = pair.EncoderLease,
            DecoderJointPoolLease = pair.DecoderLease,
        };
    }

    public static async Task<OpusSessionLease> CreatePooledOpusAsync(
        string engineFamily,
        string encoderModelPath,
        string decoderModelPath,
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        InferenceSessionPool? pool = null,
        string? modelId = null,
        string? variant = null,
        IReadOnlyDictionary<string, string>? additionalTrtEncoderOptions = null,
        IReadOnlyDictionary<string, string>? additionalTrtDecoderOptions = null,
        Func<string, SessionOptions, InferenceSession>? sessionFactory = null)
    {
        ArgumentNullException.ThrowIfNull(engineFamily);
        pool ??= InferenceSessionPool.Shared;

        provider = DowngradeTrtRtxForUnsupportedGraphs(
            provider,
            [encoderModelPath, decoderModelPath],
            out _);

        BootstrapContext bootstrap = await BootstrapForProviderAsync(provider, cancellationToken)
            .ConfigureAwait(false);
        DualOptionsSelections selections = CreateDualOptionsSelections(
            provider, bootstrap.Bootstrap, bootstrap.DevicePolicy,
            additionalTrtEncoderOptions, additionalTrtDecoderOptions,
            encoderModelPath: encoderModelPath, decoderModelPath: decoderModelPath);
        using SessionOptions encoderOptions = selections.Encoder.Options;
        using SessionOptions decoderOptions = selections.Decoder.Options;
        (SessionPoolKey encoderKey, SessionPoolKey decoderKey) = await BuildDualPooledKeysAsync(
            engineFamily, encoderModelPath, decoderModelPath, selections,
            bootstrap.DevicePolicy, additionalTrtEncoderOptions, additionalTrtDecoderOptions,
            modelId, variant, cancellationToken).ConfigureAwait(false);

        DualPooledLeasePair pair = await AcquireDualPooledSessionsAsync(
            provider, bootstrap, selections,
            encoderModelPath, decoderModelPath,
            encoderKey, decoderKey, pool, cancellationToken, sessionFactory)
            .ConfigureAwait(false);

        return new OpusSessionLease(
            pair.EncoderLease.Session, pair.DecoderLease.Session,
            pair.RequestedProviderLabel, FormatProviderLabel(pair.SelectedProvider), pair.BootstrapDetail)
        {
            SelectedProviderKind = pair.SelectedProvider,
            EncoderPoolLease = pair.EncoderLease,
            DecoderPoolLease = pair.DecoderLease
        };
    }
}

