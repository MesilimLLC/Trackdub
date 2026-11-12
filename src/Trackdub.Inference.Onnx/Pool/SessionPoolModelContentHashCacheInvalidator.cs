using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;

namespace Trackdub.Inference.Onnx.Pool;

/// <summary>Connects successful model-file replacement to pooled-session content identity.</summary>
public sealed class SessionPoolModelContentHashCacheInvalidator(IFileFingerprintService? fingerprintService = null)
    : IModelContentHashCacheInvalidator
{
    /// <inheritdoc />
    public void Invalidate(string modelPath)
    {
        fingerprintService?.Invalidate(modelPath);
        SessionPoolKey.InvalidateModelContentHash(modelPath);
    }
}
