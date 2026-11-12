using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Onnx.TensorRtRtx;
using Trackdub.Inference.Runtime.NativeCudaTensorRt;
using Trackdub.Inference.Runtime.TensorRtRtx;

namespace Trackdub.Inference.Onnx.NativeCudaTensorRt;

public sealed class NativeCudaTensorRtWindowsReadinessProbe : INativeCudaTensorRtWindowsReadinessProbe
{
    public Task<NativeCudaTensorRtWindowsReadinessReport> ProbeAsync(
        bool isSettingEnabled,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (!OperatingSystem.IsWindows())
        {
            return Task.FromResult(new NativeCudaTensorRtWindowsReadinessReport(
                IsSupportedPlatform: false,
                IsSettingEnabled: isSettingEnabled,
                IsHardwareEligible: false,
                IsCudaOrtProviderListed: false,
                IsNativeTensorRtLibrariesPresent: false,
                IsNativeTensorRtOrtProviderListed: false,
                IsCudaReady: false,
                IsTensorRtReady: false,
                CudaDetail: "Native ORT CUDA/TensorRT on Windows is only applicable on Windows hosts.",
                TensorRtDetail: "Native ORT CUDA/TensorRT on Windows is only applicable on Windows hosts.",
                CudaInstallHint: null,
                TensorRtInstallHint: null));
        }

        (bool hardwareEligible, _, string hardwareDetail) = WindowsNvidiaHardwareGate.Evaluate();
        bool cudaListed = CudaOrtProbe.IsCudaProviderListed();
        bool trtLibs = NativeTensorRtLibraryProbe.IsNativeTensorRtAvailable();
        bool trtListed = TensorRtRtxOrtProbe.IsNativeTensorRtProviderListed();

        if (!isSettingEnabled)
        {
            return Task.FromResult(new NativeCudaTensorRtWindowsReadinessReport(
                IsSupportedPlatform: true,
                IsSettingEnabled: false,
                IsHardwareEligible: hardwareEligible,
                IsCudaOrtProviderListed: cudaListed,
                IsNativeTensorRtLibrariesPresent: trtLibs,
                IsNativeTensorRtOrtProviderListed: trtListed,
                IsCudaReady: false,
                IsTensorRtReady: false,
                CudaDetail: NativeCudaTensorRtWindowsProviderConstants.SettingDisabledHint,
                TensorRtDetail: NativeCudaTensorRtWindowsProviderConstants.SettingDisabledHint,
                CudaInstallHint: null,
                TensorRtInstallHint: null));
        }

        if (!hardwareEligible)
        {
            return Task.FromResult(new NativeCudaTensorRtWindowsReadinessReport(
                IsSupportedPlatform: true,
                IsSettingEnabled: true,
                IsHardwareEligible: false,
                IsCudaOrtProviderListed: cudaListed,
                IsNativeTensorRtLibrariesPresent: trtLibs,
                IsNativeTensorRtOrtProviderListed: trtListed,
                IsCudaReady: false,
                IsTensorRtReady: false,
                CudaDetail: hardwareDetail,
                TensorRtDetail: hardwareDetail,
                CudaInstallHint: NativeCudaTensorRtWindowsProviderConstants.WindowsCudaInstallHint,
                TensorRtInstallHint: NativeCudaTensorRtWindowsProviderConstants.WindowsTensorRtInstallHint));
        }

        bool cudaReady = cudaListed && CudaOrtProbe.IsCuDnn9Available();
        bool tensorRtReady = trtLibs && trtListed;

        string cudaDetail = cudaReady
            ? "CUDAExecutionProvider is listed; sessions use native ORT CUDA when explicitly requested."
            : ResolveCudaBlockedDetail(cudaListed);

        string tensorRtDetail = tensorRtReady
            ? "TensorRT libraries and TensorrtExecutionProvider are available for explicit native ORT sessions."
            : ResolveTensorRtBlockedDetail(trtLibs, trtListed);

        return Task.FromResult(new NativeCudaTensorRtWindowsReadinessReport(
            IsSupportedPlatform: true,
            IsSettingEnabled: true,
            IsHardwareEligible: true,
            IsCudaOrtProviderListed: cudaListed,
            IsNativeTensorRtLibrariesPresent: trtLibs,
            IsNativeTensorRtOrtProviderListed: trtListed,
            IsCudaReady: cudaReady,
            IsTensorRtReady: tensorRtReady,
            CudaDetail: cudaDetail,
            TensorRtDetail: tensorRtDetail,
            CudaInstallHint: cudaReady ? null : NativeCudaTensorRtWindowsProviderConstants.WindowsCudaInstallHint,
            TensorRtInstallHint: tensorRtReady ? null : NativeCudaTensorRtWindowsProviderConstants.WindowsTensorRtInstallHint));
    }

    private static string ResolveCudaBlockedDetail(bool cudaListed) =>
        cudaListed
            ? "CUDAExecutionProvider is listed, but cuDNN 9 could not be loaded."
            : "CUDAExecutionProvider is not listed by ONNX Runtime.";

    private static string ResolveTensorRtBlockedDetail(bool trtLibs, bool trtListed)
    {
        if (!trtLibs && !trtListed)
        {
            return "TensorRT libraries and TensorrtExecutionProvider are not available.";
        }

        if (!trtLibs)
        {
            return "nvinfer.dll was not found on PATH or common install locations.";
        }

        return "TensorrtExecutionProvider is not listed by ONNX Runtime.";
    }
}
