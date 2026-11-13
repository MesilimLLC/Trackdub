using Microsoft.ML.OnnxRuntime;
using Trackdub.Inference.Runtime.NativeCudaTensorRt;

namespace Trackdub.Inference.Onnx.Runtime.Planning;

internal static class CudaOrtProbe
{
    private const string CuDnn9LibraryWindows = "cudnn64_9.dll";
    private const string CuDnn9LibraryLinux = "libcudnn.so.9";

    public static bool IsCudaProviderListed()
    {
        try
        {
            return OrtEnv.Instance().GetAvailableProviders()
                .Any(name =>
                    string.Equals(name, NativeCudaTensorRtWindowsProviderConstants.CudaOrtExecutionProviderName, StringComparison.OrdinalIgnoreCase));
        }
        catch
        {
            return false;
        }
    }

    // GetAvailableProviders() reflects what the ORT binary compiled in, not runtime deps.
    // cuDNN is dlopen'd at InferenceSession creation — probe filesystem to avoid false-positive
    // readiness when cuDNN 9 is absent.
    public static bool IsCuDnn9Available()
    {
        string libraryName = OperatingSystem.IsLinux() ? CuDnn9LibraryLinux : CuDnn9LibraryWindows;
        try
        {
            if (!System.Runtime.InteropServices.NativeLibrary.TryLoad(libraryName, out nint handle))
            {
                return false;
            }

            System.Runtime.InteropServices.NativeLibrary.Free(handle);
            return true;
        }
        catch
        {
            return false;
        }
    }
}
