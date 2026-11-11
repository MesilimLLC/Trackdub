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

    [Theory]
    [InlineData("65536", true, "65536")]
    [InlineData("101376", true, "101376")]
    [InlineData("0", true, LimitText)]
    [InlineData("202752", true, LimitText)]
    [InlineData("not-a-number", true, LimitText)]
    [InlineData("-1", true, LimitText)]
    [InlineData("0", false, "0")]
    [InlineData("202752", false, "202752")]
    public void Apply_resolves_a_caller_supplied_limit(string requested, bool isWindows, string expected)
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [TensorRtRtxDirectMlInterop.MaxSharedMemSizeOptionKey] = requested,
        };

        TensorRtRtxDirectMlInterop.Apply(options, isWindows);

        Assert.Equal(expected, options[TensorRtRtxDirectMlInterop.MaxSharedMemSizeOptionKey]);
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

    [Theory]
    [InlineData("off", "0")]
    [InlineData("on", "1")]
    [InlineData(null, "1")]
    public void Override_wins_over_a_profile_built_with_WithCudaGraphsEnabled(string? overrideValue, string expected)
    {
        IReadOnlyDictionary<string, string> profile = OnnxExecutionSessionFactory.WithCudaGraphsEnabled(
            new Dictionary<string, string>(StringComparer.Ordinal) { ["trt_profile_min_shapes"] = "input_features:1x80x1" });

        IReadOnlyDictionary<string, string> options = OnnxExecutionSessionFactory.BuildTensorRtRtxOptionsWithOverride(
            profile,
            enableCudaGraph: false,
            overrideValue);

        Assert.Equal(expected, options["enable_cuda_graph"]);
        Assert.Equal("input_features:1x80x1", options["nv_profile_min_shapes"]);
    }
}
