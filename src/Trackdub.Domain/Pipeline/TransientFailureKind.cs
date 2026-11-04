using System.Linq;
using System.Runtime.InteropServices;

namespace Trackdub.Domain.Pipeline;

/// <summary>
/// Transient-failure kind classification consumed by the dubbing pipeline's
/// <c>PipelineTransientFaultBus</c>. See
/// <c>docs/internal/pipeline-readiness-spec.md</c> §4.1 + §4.3 for the
/// matching consumer contract. Domain-owned so the enum is referenceable from
/// SDK, Application, and Composition layers without upward dependencies.
/// </summary>
public enum TransientFailureKind
{
    /// <summary>Caller's <see cref="OperationCanceledException"/> fired; never retry, write <c>Canceled</c> row.</summary>
    UserCancellation = 0,
    /// <summary>Process or app holds the directory/file exclusively; retry with backoff.</summary>
    DirectoryLock = 1,
    /// <summary>SQLite <c>SQLITE_BUSY</c>; retry, same backoff.</summary>
    SqliteBusy = 2,
    /// <summary>ffprobe/ffmpeg crashed mid-op (often transient driver state); retry.</summary>
    FfmpegProcessExit = 3,
    /// <summary>HF mirror returned 5xx or transient download error; retry.</summary>
    ModelDownloadTransient = 4,
    /// <summary>7zr crash or tar exit non-zero + archive integrity OK; retry.</summary>
    StarterPackTransient = 5,
    /// <summary>Inference host ORT/OnnxRuntime reported memory pressure; backoff + downscale quant.</summary>
    MemoryExhausted = 6,
    /// <summary>DirectML/TensorRT-RTX/WinML catalog stalled on hot plug; retry.</summary>
    DeviceTimeoutTransient = 7,
    /// <summary>Fallback when no classifier matched; logged, retried once.</summary>
    Unknown = 8,
}

/// <summary>
/// Heuristic classifier for transient dubbing-pipeline failures. Returns
/// <see langword="true"/> when the exception's runtime type name or message
/// maps to a known transient classification. Pure reflection + message
/// matching so Domain has no upward package dependency (Microsoft.Data.Sqlite,
/// ONNX Runtime, LibVLC, SharpCompress, etc. are not referenced). See
/// <c>docs/internal/pipeline-readiness-spec.md</c> §4.1 + §7 for the
/// platform-gated heuristics.
/// </summary>
public static class TransientFailureClassifier
{
    private const string SqliteExceptionTypeName = "SqliteException";
    private const string OnnxRuntimeExceptionTypeName = "OnnxRuntimeException";
    private const string ExtractExceptionTypeName = "ExtractException";

    private static readonly int[] TransientHttpStatusCodes = [500, 502, 503, 504, 429];

    /// <summary>
    /// ONNX error codes that indicate a missing, corrupt, or incompletely
    /// downloaded model file. These are transient because re-downloading or
    /// re-trying the load may succeed; all other <c>OnnxRuntimeException</c>
    /// messages are treated as non-transient inference failures.
    /// </summary>
    private static readonly string[] OnnxModelLoadErrorCodeTokens =
    [
        "[ErrorCode:NoSuchFile]",
        "[ErrorCode:NoModel]",
        "[ErrorCode:InvalidProtobuf]",
        "[ErrorCode:ModelLoadCanceled]",
    ];

    /// <summary>
    /// True iff <paramref name="exception"/>'s runtime type or message maps to
    /// a known transient classification. Fails closed: throws
    /// <see cref="ArgumentNullException"/> on null rather than silently
    /// returning false (spec §4.1: defaults false, fail fast).
    /// </summary>
    public static bool IsTransient(Exception exception)
    {
        ArgumentNullException.ThrowIfNull(exception);
        TransientFailureKind kind = Classify(exception);
        // UserCancellation is a distinct, non-retryable classification (see its doc comment:
        // "never retry"); Classify still reports it so Snapshot()/telemetry consumers can tell a
        // cancelled run apart from Unknown, but IsTransient must not tell a caller to retry it.
        return kind != TransientFailureKind.Unknown && kind != TransientFailureKind.UserCancellation;
    }

    /// <summary>
    /// Returns the most-specific <see cref="TransientFailureKind"/> the
    /// classifier can assign, or <see cref="TransientFailureKind.Unknown"/>
    /// when no rule matches. Thread-safe; no shared mutable state.
    /// </summary>
    public static TransientFailureKind Classify(Exception exception)
    {
        ArgumentNullException.ThrowIfNull(exception);

        // Direct type checks first; cheap and unambiguous.
        if (exception is OperationCanceledException) return TransientFailureKind.UserCancellation;
        if (exception is OutOfMemoryException) return TransientFailureKind.MemoryExhausted;

        string typeName = exception.GetType().FullName ?? string.Empty;
        string message = exception.Message ?? string.Empty;

        if (ContainsTypeName(typeName, SqliteExceptionTypeName)
            && message.Contains("SQLITE_BUSY", StringComparison.OrdinalIgnoreCase))
        {
            return TransientFailureKind.SqliteBusy;
        }

        if (ContainsTypeName(typeName, OnnxRuntimeExceptionTypeName)
            && IsOnnxModelLoadFailure(message))
        {
            return TransientFailureKind.ModelDownloadTransient;
        }

        if (ContainsTypeName(typeName, ExtractExceptionTypeName)
            || (message.Contains("7zr", StringComparison.OrdinalIgnoreCase) && message.Contains("exit", StringComparison.OrdinalIgnoreCase))
            || message.Contains("tar exited", StringComparison.OrdinalIgnoreCase))
        {
            return TransientFailureKind.StarterPackTransient;
        }

        // HttpRequestException lives in System.Net.Http, which Domain can reference (BCL).
        if (exception is HttpRequestException httpException)
        {
            if (ContainsTransientHttpStatusCode(httpException.Message))
            {
                return TransientFailureKind.ModelDownloadTransient;
            }
        }

        // IOException — host-OS-aware at runtime so the same compiled binary works on net11.0,
        // net11.0-windows*, and any future TFM. Compile-time #if WINDOWS guards collapse the
        // branches on plain net11.0 builds, which silently excluded the Windows hresult path
        // from production callers.
        if (exception is IOException)
        {
            TransientFailureKind ioKind = ClassifyIOException((IOException)exception, message);
            if (ioKind != TransientFailureKind.Unknown)
            {
                return ioKind;
            }
        }

        return TransientFailureKind.Unknown;
    }

    private static TransientFailureKind ClassifyIOException(IOException exception, string message)
    {
        // Host-OS-aware at runtime so the same compiled binary works on net11.0, net11.0-windows*, and
        // any future TFM. Compile-time #if WINDOWS guards collapse the branches on plain net11.0
        // builds, which silently excluded the Windows hresult path from production callers.
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            && exception.HResult == unchecked((int)0x80070020))
        {
            // ERROR_SHARING_VIOLATION = 0x80070020 mapped through HRESULT.
            return TransientFailureKind.DirectoryLock;
        }
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            && (message.Contains("EAGAIN", StringComparison.Ordinal)
                || message.Contains("EDEADLK", StringComparison.Ordinal)
                || message.Contains("Resource deadlock", StringComparison.OrdinalIgnoreCase)))
        {
            return TransientFailureKind.DirectoryLock;
        }
        return TransientFailureKind.Unknown;
    }

    private static bool IsOnnxModelLoadFailure(string message)
    {
        // ORT embeds [ErrorCode:<Name>] at the start of every exception message.
        // Only model-load-related codes are treated as download transients;
        // runtime/execution failures (e.g. RuntimeException, InvalidArgument)
        // are not retried because re-downloading will not fix them.
        if (!message.Contains("[ErrorCode:", StringComparison.Ordinal))
        {
            return false;
        }

        return OnnxModelLoadErrorCodeTokens.Any(
            token => message.Contains(token, StringComparison.OrdinalIgnoreCase));
    }

    private static bool ContainsTypeName(string fullName, string substring) =>
        fullName.Contains(substring, StringComparison.Ordinal);

    private static bool ContainsTransientHttpStatusCode(string message)
    {
        if (string.IsNullOrEmpty(message))
        {
            return false;
        }

        foreach (int code in TransientHttpStatusCodes)
        {
            if (message.Contains(code.ToString(System.Globalization.CultureInfo.InvariantCulture), StringComparison.Ordinal))
            {
                return true;
            }
        }

        return false;
    }
}
