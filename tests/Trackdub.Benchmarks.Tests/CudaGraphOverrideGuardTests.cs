#if WINDOWS
using Trackdub.Domain;
using Trackdub.Domain.Benchmarking;
using Trackdub.Inference.Onnx;

namespace Trackdub.Benchmarks.Tests;

[CollectionDefinition(nameof(CudaGraphOverrideGuardCollection), DisableParallelization = true)]
public sealed class CudaGraphOverrideGuardCollection;

[Collection(nameof(CudaGraphOverrideGuardCollection))]
public sealed class CudaGraphOverrideGuardTests
{
    private const string Variable = "TRACKDUB_TRT_RTX_CUDA_GRAPHS";

    public sealed record GraphMode(BenchmarkCudaGraphMode Mode);

    public static TheoryData<GraphMode> ExplicitModes => new()
    {
        new GraphMode(BenchmarkCudaGraphMode.On),
        new GraphMode(BenchmarkCudaGraphMode.Off),
        new GraphMode(BenchmarkCudaGraphMode.Compare),
    };

    [Theory]
    [MemberData(nameof(ExplicitModes))]
    public async Task Explicit_graph_modes_are_refused_while_the_override_is_set(GraphMode graphMode)
    {
        BenchmarkReport report = await RunWithOverrideAsync("on", graphMode.Mode);

        Assert.Equal(BenchmarkStatus.Failed, report.Status);
        Assert.Contains(Variable, report.FailureReason, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Auto_mode_is_not_refused_by_the_override_guard()
    {
        BenchmarkReport report = await RunWithOverrideAsync("on", BenchmarkCudaGraphMode.Auto);

        Assert.DoesNotContain(Variable, report.FailureReason ?? string.Empty, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Explicit_graph_modes_run_past_the_guard_when_the_override_is_unset()
    {
        BenchmarkReport report = await RunWithOverrideAsync(null, BenchmarkCudaGraphMode.Compare);

        Assert.DoesNotContain(Variable, report.FailureReason ?? string.Empty, StringComparison.Ordinal);
    }

    private static async Task<BenchmarkReport> RunWithOverrideAsync(string? overrideValue, BenchmarkCudaGraphMode mode)
    {
        string? previous = Environment.GetEnvironmentVariable(Variable);
        string reportPath = Path.Join(Path.GetTempPath(), $"{Guid.NewGuid():N}.json");
        try
        {
            Environment.SetEnvironmentVariable(Variable, overrideValue);
            var request = new BenchmarkRequest(
                Path.Join(Path.GetTempPath(), $"{Guid.NewGuid():N}-missing.onnx"),
                reportPath,
                BenchmarkProviderPreference.TensorRtRtx,
                RunCount: 1,
                Execution: new BenchmarkExecutionSettings(
                    IoMode: BenchmarkIoMode.DeviceBound,
                    TensorRtRtx: new BenchmarkTensorRtRtxSettings(CudaGraphMode: mode)));

            return await new OnnxModelBenchmarkRunner().RunAsync(request, TestContext.Current.CancellationToken);
        }
        finally
        {
            Environment.SetEnvironmentVariable(Variable, previous);
            File.Delete(reportPath);
        }
    }
}
#endif
