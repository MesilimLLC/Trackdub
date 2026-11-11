using Trackdub.Inference.Onnx;
using Trackdub.Inference.Onnx.TensorRtRtx;
using Trackdub.Inference.Runtime.TensorRtRtx;

namespace Trackdub.Inference.Tests;

public sealed class TensorRtRtxDirectMlInteropTests
{
    private const string LimitText = "101376";

    [Fact]
    public void Limit_is_99_KiB()
    {
        Assert.Equal(99 * 1024, TensorRtRtxProviderConstants.WindowsDirectMlInteropSharedMemoryLimitBytes);
        Assert.Equal(99 * 1024, long.Parse(LimitText, System.Globalization.CultureInfo.InvariantCulture));
    }

    [Theory]
    [InlineData(true, true)]
    [InlineData(false, false)]
    public void Apply_caps_shared_memory_on_windows_regardless_of_driver(bool isWindows, bool expectCap)
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal);

        TensorRtRtxDirectMlInterop.Apply(options, isWindows);

        Assert.Equal(expectCap, options.ContainsKey(TensorRtRtxDirectMlInterop.MaxSharedMemSizeOptionKey));
        if (expectCap)
        {
            Assert.Equal(LimitText, options[TensorRtRtxDirectMlInterop.MaxSharedMemSizeOptionKey]);
        }
    }

    [Fact]
    public void Apply_keeps_a_smaller_caller_supplied_shared_memory_limit()
    {
        Assert.Equal("65536", ApplyAndReadCallerLimit("65536", isWindows: true));
    }

    [Theory]
    [InlineData("0")]
    [InlineData("202752")]
    [InlineData("not-a-number")]
    [InlineData("-1")]
    public void Apply_clamps_caller_limits_that_leave_the_engine_uncapped(string requested)
    {
        Assert.Equal(LimitText, ApplyAndReadCallerLimit(requested, isWindows: true));
    }

    [Fact]
    public void Apply_leaves_caller_limits_untouched_off_windows()
    {
        Assert.Equal("0", ApplyAndReadCallerLimit("0", isWindows: false));
    }

    private static string ApplyAndReadCallerLimit(string requested, bool isWindows)
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [TensorRtRtxDirectMlInterop.MaxSharedMemSizeOptionKey] = requested,
        };

        TensorRtRtxDirectMlInterop.Apply(options, isWindows);

        return options[TensorRtRtxDirectMlInterop.MaxSharedMemSizeOptionKey];
    }

    [Fact]
    public void Fingerprint_carries_the_shared_memory_limit_only_on_windows()
    {
        string fingerprint = TensorRtRtxProviderConstants.BundledFingerprintVersion;

        Assert.Equal(OperatingSystem.IsWindows(), fingerprint.EndsWith("+smem99k", StringComparison.Ordinal));
        Assert.StartsWith(
            $"{TensorRtRtxProviderConstants.BundledVersion}+{TensorRtRtxProviderConstants.BundledCudaVariant}+trt-rtx-",
            fingerprint,
            StringComparison.Ordinal);
    }

    [Fact]
    public void Built_options_carry_the_limit_and_a_cache_scoped_to_the_fingerprint_on_windows()
    {
        IReadOnlyDictionary<string, string> options = OnnxExecutionSessionFactory.BuildTensorRtRtxOptions(null);

        Assert.Equal(
            TensorRtRtxProviderConstants.BundledFingerprintVersion,
            Path.GetFileName(options["nv_runtime_cache_path"]));
        Assert.Equal(
            OperatingSystem.IsWindows() ? LimitText : null,
            options.GetValueOrDefault(TensorRtRtxDirectMlInterop.MaxSharedMemSizeOptionKey));
    }

    [Theory]
    [InlineData("on", "0", "1")]
    [InlineData("ON", "0", "1")]
    [InlineData("1", "0", "1")]
    [InlineData("true", "0", "1")]
    [InlineData("off", "1", "0")]
    [InlineData("0", "1", "0")]
    [InlineData("false", "1", "0")]
    [InlineData(null, "1", "1")]
    [InlineData("", "0", "0")]
    [InlineData("auto", "1", "1")]
    public void ApplyCudaGraphOverride_only_acts_on_on_or_off(string? overrideValue, string callSite, string expected)
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal) { ["enable_cuda_graph"] = callSite };

        OnnxExecutionSessionFactory.ApplyCudaGraphOverride(options, overrideValue);

        Assert.Equal(expected, options["enable_cuda_graph"]);
    }
}
