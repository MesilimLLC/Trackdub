using System.Runtime.CompilerServices;
using Trackdub.Inference.Onnx;

namespace Trackdub.Inference.Tests;

internal static class TestProcessEnvironment
{
    /// <summary>
    /// TRACKDUB_TRT_RTX_CUDA_GRAPHS overrides enable_cuda_graph at every call site, so a developer who
    /// leaves it set in the shell would flip every test that asserts the option value. These tests
    /// assert the call sites' own choices; the override has its own explicit-value tests.
    /// </summary>
    [ModuleInitializer]
    internal static void ClearCudaGraphOverride() =>
        Environment.SetEnvironmentVariable(OnnxExecutionSessionFactory.CudaGraphsOverrideEnvironmentVariable, null);
}
