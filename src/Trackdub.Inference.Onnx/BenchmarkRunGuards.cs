using Trackdub.Domain;
using Trackdub.Domain.Benchmarking;

namespace Trackdub.Inference.Onnx;

internal static class BenchmarkRunGuards
{
    /// <summary>
    /// Validates the run settings, then refuses explicit CUDA graph modes while
    /// <c>TRACKDUB_TRT_RTX_CUDA_GRAPHS</c> selects a value. The override applies to every session, so
    /// <c>--cuda-graphs compare</c> would give both lanes the same capture mode and report a comparison
    /// that compares nothing. It needs only the settings and the environment, so it runs before provider
    /// registration (which can download the EP bundle) and model resolution.
    /// </summary>
    internal static void Validate(BenchmarkRequest request, BenchmarkExecutionSettings settings)
    {
        settings.Validate(request.RunCount);

        if (request.ProviderPreference is not BenchmarkProviderPreference.TensorRtRtx)
        {
            return;
        }

        if (settings.EffectiveTensorRtRtx.CudaGraphMode is BenchmarkCudaGraphMode.Auto)
        {
            return;
        }

        if (!OnnxExecutionSessionFactory.IsCudaGraphOverrideSet())
        {
            return;
        }

        throw new NotSupportedException(
            "--cuda-graphs on, off and compare choose the capture mode for each run, so they cannot run while "
            + $"{OnnxExecutionSessionFactory.CudaGraphsOverrideEnvironmentVariable} is set: it would give every lane "
            + "the same capture mode and report a comparison that compares nothing. Unset the variable.");
    }
}
