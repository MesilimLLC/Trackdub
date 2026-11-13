using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;
using Trackdub.Contracts;
using Trackdub.Inference.Onnx.EpContext;

namespace Trackdub.Inference.Onnx;

internal static partial class OnnxExecutionSessionFactory
{
    private static readonly ConcurrentDictionary<string, byte> ReportedCaptureSkips =
        new(StringComparer.OrdinalIgnoreCase);

    internal static IApplicationLogger? DiagnosticsLogger { get; set; }

    private static string ExtractLogId(string modelPath)
    {
        string fileName = Path.GetFileNameWithoutExtension(modelPath);
        string[] parts = modelPath.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return parts.Length >= 3 ? $"{parts[^3]}/{fileName}" : fileName;
    }

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
    /// Explains why TensorRT-RTX engine capture is (or is not) available for a model. Public so
    /// operator tooling can report the same decision without requiring a created session.
    /// </summary>
    public static bool TryGetEngineCaptureSkipReason(string modelPath, out string reason) =>
        EpContextCapture.TryGetSkipReason(modelPath, out reason);

    /// <summary>Minimum free bytes the capture path requires before it will write an artifact.</summary>
    public static long EngineCaptureRequiredFreeBytes(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        return EpContextArtifact.GetSourceTotalBytes(modelPath) * 2;
    }
}
