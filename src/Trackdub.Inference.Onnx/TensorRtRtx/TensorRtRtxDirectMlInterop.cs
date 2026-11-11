using System.Globalization;
using Trackdub.Inference.Runtime.TensorRtRtx;

namespace Trackdub.Inference.Onnx.TensorRtRtx;

/// <summary>
/// TensorRT-RTX 1.7 known issue: on Windows, TensorRT-RTX inference launched after a DirectML
/// <c>RecordDispatch</c> in the same process fails with CUDA_ERROR_INVALID_VALUE (surfacing as
/// <c>cuKernelSetAttribute returned error invalid argument</c>, then ORT's misleading
/// <c>ShapeInferenceNotRegistered</c>). NVIDIA's workaround is to limit the builder's kernel shared
/// memory to 99 KiB. Trackdub mixes DirectML and TensorRT-RTX stages in one process, so every
/// Windows engine is built under that limit.
/// NVIDIA's release notes scope the issue to "CUDA driver R615 or earlier", but it reproduces on
/// driver 621.14 (RTX 5070, EP ABI 0.5.0, TensorRT-RTX 1.7.1): a VAD stage on DirectML followed by
/// a Whisper stage on TensorRT-RTX failed without the limit and passed with it, so the limit is not
/// gated on the driver version. Engines and kernels built without it keep failing, which is why the
/// limit is part of <see cref="TensorRtRtxProviderConstants.BundledFingerprintVersion"/>.
/// </summary>
internal static class TensorRtRtxDirectMlInterop
{
    internal const string MaxSharedMemSizeOptionKey = "nv_max_shared_mem_size";

    // The EP maps nv_max_shared_mem_size to MemoryPoolType::kTACTIC_SHARED_MEMORY.
    private static readonly string LimitBytesText =
        TensorRtRtxProviderConstants.WindowsDirectMlInteropSharedMemoryLimitBytes.ToString(CultureInfo.InvariantCulture);

    internal static void Apply(IDictionary<string, string> trtOptions, bool isWindows)
    {
        if (!isWindows)
        {
            return;
        }

        if (!KeepsCallerLimit(trtOptions))
        {
            trtOptions[MaxSharedMemSizeOptionKey] = LimitBytesText;
        }
    }

    /// <summary>
    /// A smaller caller limit is kept. Zero means "device maximum" and anything above the cap leaves
    /// the engine uncapped, so both (and unparsable values) are clamped to the workaround value.
    /// </summary>
    private static bool KeepsCallerLimit(IDictionary<string, string> trtOptions)
    {
        if (!trtOptions.TryGetValue(MaxSharedMemSizeOptionKey, out string? requested))
        {
            return false;
        }

        if (!long.TryParse(requested, NumberStyles.None, CultureInfo.InvariantCulture, out long requestedBytes))
        {
            return false;
        }

        if (requestedBytes <= 0)
        {
            return false;
        }

        return requestedBytes <= TensorRtRtxProviderConstants.WindowsDirectMlInteropSharedMemoryLimitBytes;
    }
}
