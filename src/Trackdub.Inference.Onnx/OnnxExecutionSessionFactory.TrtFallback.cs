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
    /// Creates an ORT session; when TensorRT RTX was selected and init fails due to unsupported
    /// ops/kernels, optionally retries once with DirectML (Windows) then CPU and records FallbackReason.
    /// Set <see cref="SessionCreateRequest.AllowTrtInitFallback"/> <see langword="false"/> for hard-pin
    /// routes (<c>RequirePreferredExecutionProvider</c> / CLI <c>--require-execution-provider</c>).
    /// </summary>
    internal static (InferenceSession Session, SessionOptionsSelection Selection) CreateInferenceSessionWithTrtInitFallback(
        string modelPath,
        SessionCreateRequest request,
        CancellationToken cancellationToken)
    {
        // Honest skip: graphs with com.microsoft contrib fusions cannot be imported by
        // TensorRT-RTX (all-or-nothing ONNX catalog). Do not attempt TRT and do not emit
        // the ModelImporter error storm — fall through to DirectML/CPU immediately.
        if (request.Selection.SelectedProvider is ExecutionProviderKind.TensorRTRtx
            && TrtRtxUnsupportedOpScanner.FindUnsupportedOps(modelPath) is { Count: > 0 } unsupportedOps)
        {
            string opList = string.Join(", ", unsupportedOps);
            string skipReason =
                $"TensorRT-RTX cannot import '{Path.GetFileName(modelPath)}' (unsupported ops: {opList}).";

            if (!request.AllowTrtInitFallback)
            {
                request.Selection.Options.Dispose();
                throw new InvalidOperationException($"{skipReason} Hard-pin route has no fallback.");
            }

            return CreateUnsupportedGraphFallback(modelPath, request, cancellationToken, skipReason);
        }

        try
        {
            InferenceSession session = CreateSession(modelPath, request, cancellationToken);
            return (session, request.Selection);
        }
        catch (Exception ex) when (ShouldFallbackAfterTrtInitFailure(request, ex))
        {
            return CreateTrtInitFailureFallback(modelPath, request, cancellationToken, ex);
        }
    }

    /// <summary>True only for a fallback-allowed TRT-RTX selection that hit an init failure.</summary>
    private static bool ShouldFallbackAfterTrtInitFailure(SessionCreateRequest request, Exception ex)
    {
        if (!request.AllowTrtInitFallback)
        {
            return false;
        }

        if (request.Selection.SelectedProvider is not ExecutionProviderKind.TensorRTRtx)
        {
            return false;
        }

        return LooksLikeTrtSessionInitFailure(ex);
    }

    private static (InferenceSession Session, SessionOptionsSelection Selection) CreateUnsupportedGraphFallback(
        string modelPath,
        SessionCreateRequest request,
        CancellationToken cancellationToken,
        string skipReason)
    {
        Exception? lastFailure = null;
        foreach (SessionOptionsSelection fallbackSelection in EnumerateTrtInitFallbackProviders()
                     .Select(provider => CreateSessionOptions(new SessionOptionsRequest(provider, request.DevicePolicy, LogId: request.LogId))))
        {
            try
            {
                InferenceSession session = CreateSession(modelPath, request with { Selection = fallbackSelection },
                    cancellationToken);
                request.Selection.Options.Dispose();
                string label = ProviderNames.FormatProviderLabel(fallbackSelection.SelectedProvider);
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

        request.Selection.Options.Dispose();
        throw lastFailure ?? new InvalidOperationException(skipReason);
    }

    private static (InferenceSession Session, SessionOptionsSelection Selection) CreateTrtInitFailureFallback(
        string modelPath,
        SessionCreateRequest request,
        CancellationToken cancellationToken,
        Exception originalFailure)
    {
        string trtError = SummarizeExceptionMessage(originalFailure);
        TensorRtRtxTeardownGuard.MarkPoisonedIfCudaOutOfMemory(originalFailure);
        Exception lastFailure = originalFailure;
        foreach (SessionOptionsSelection fallbackSelection in EnumerateTrtInitFallbackProviders()
                     .Select(provider => CreateSessionOptions(new SessionOptionsRequest(provider, request.DevicePolicy, LogId: request.LogId))))
        {
            try
            {
                InferenceSession session = CreateSession(modelPath, request with { Selection = fallbackSelection },
                    cancellationToken);
                // Transfer ownership: the caller owns fallback options after success.
                request.Selection.Options.Dispose();
                string label = ProviderNames.FormatProviderLabel(fallbackSelection.SelectedProvider);
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

        // Leave request.Selection.Options for the caller to dispose.
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
                $"selected {ProviderNames.FormatProviderLabel(next)} without attempting TensorRT-RTX.";
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

    private static readonly string[] TrtSessionInitFailureMarkers =
    [
        "Kernel not found",
        "ModelImporter",
        "No graph will run on TensorRT",
        "NvTensorRTRTX",
        "TensorRT-RTX",
        "TensorRT RTX",
        "transformer_memcpy",
        "ProcessInitializers",
    ];

    internal static bool LooksLikeTrtSessionInitFailure(Exception exception)
    {
        for (Exception? current = exception; current is not null; current = current.InnerException)
        {
            if (TrtSessionInitFailureMarkers.Any(
                    marker => current.Message.Contains(marker, StringComparison.OrdinalIgnoreCase)))
            {
                return true;
            }
        }

        return false;
    }

    private static bool IsRecoverableTrtFallbackInitFailure(Exception exception)
    {
        if (LooksLikeTrtSessionInitFailure(exception))
        {
            return true;
        }

        return exception is OnnxRuntimeException
            or InvalidOperationException
            or DllNotFoundException
            or EntryPointNotFoundException;
    }

    private static string SummarizeExceptionMessage(Exception exception)
    {
        string message = exception.GetBaseException().Message.ReplaceLineEndings(" ");
        const int maxLength = 240;
        return message.Length <= maxLength ? message : message[..maxLength] + "...";
    }

}
