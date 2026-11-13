namespace Trackdub.Inference.Runtime.NativeCudaTensorRt;

public static class NativeCudaTensorRtWindowsProviderConstants
{
    public const string CudaOrtExecutionProviderName = "CUDAExecutionProvider";

    public const string WindowsCudaInstallHint =
        "Use NVIDIA drivers and the CUDA runtime libraries required by the loaded ONNX Runtime build.";

    public const string WindowsCuDnn9InstallHint =
        "Make cuDNN 9 for the installed CUDA runtime available on PATH.";

    public const string WindowsTensorRtInstallHint =
        "Install NVIDIA drivers and TensorRT (nvinfer.dll on PATH), then verify ONNX Runtime lists TensorrtExecutionProvider.";

    public const string SettingDisabledHint =
        "The host policy disabled native CUDA and TensorRT discovery.";
}
