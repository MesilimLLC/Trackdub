using Trackdub.Domain;
using Trackdub.Inference.Onnx;
using Microsoft.ML.OnnxRuntime;

namespace Trackdub.Inference.Tests;

public sealed class OnnxTensorRtFallbackSelectionTests
{
    [Theory]
    [InlineData(ExecutionProviderKind.TensorRTRtx, ExecutionProviderKind.TensorRTRtx, null)]
    [InlineData(ExecutionProviderKind.TensorRTRtx, ExecutionProviderKind.Cpu,
        "Requested tensorrt-rtx but TensorRT RTX EP device was unavailable; fell back to cpu.")]
    [InlineData(ExecutionProviderKind.TensorRTRtx, ExecutionProviderKind.DirectMl,
        "Requested tensorrt-rtx but TensorRT RTX EP device was unavailable; fell back to dml.")]
    [InlineData(ExecutionProviderKind.TensorRt, ExecutionProviderKind.TensorRt, null)]
    [InlineData(ExecutionProviderKind.TensorRt, ExecutionProviderKind.Cpu,
        "Requested tensorrt but classic TensorRT EP device was unavailable; fell back to cpu.")]
    [InlineData(ExecutionProviderKind.TensorRt, ExecutionProviderKind.DirectMl,
        "Requested tensorrt but classic TensorRT EP device was unavailable; fell back to dml.")]
    public void BuildTensorRtFallbackSelection_reports_fallback_only_when_provider_changes(
        ExecutionProviderKind requestedProvider,
        ExecutionProviderKind selectedProvider,
        string? expectedFallbackReason)
    {
        using var options = new SessionOptions();

        OnnxExecutionSessionFactory.SessionOptionsSelection selection =
            OnnxExecutionSessionFactory.BuildTensorRtFallbackSelection(options, requestedProvider, selectedProvider);

        Assert.Equal(selectedProvider, selection.SelectedProvider);
        Assert.Equal(expectedFallbackReason, selection.FallbackReason);
    }
}
