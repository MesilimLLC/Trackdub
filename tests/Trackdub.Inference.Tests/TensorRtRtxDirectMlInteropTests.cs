using Trackdub.Inference.Onnx;
using Trackdub.Inference.Onnx.TensorRtRtx;
using Trackdub.Inference.Runtime.TensorRtRtx;

namespace Trackdub.Inference.Tests;

[Collection(nameof(OnnxExecutionSessionFactoryStateCollection))]
public sealed class TensorRtRtxDirectMlInteropTests
{
    private const string LimitText = "101376";

    public sealed record CallerLimit(string Requested, string Expected);

    public static TheoryData<CallerLimit> WindowsCallerLimits => new()
    {
        new CallerLimit("65536", "65536"),
        new CallerLimit(LimitText, LimitText),
        new CallerLimit("0", LimitText),
        new CallerLimit("202752", LimitText),
        new CallerLimit("not-a-number", LimitText),
        new CallerLimit("-1", LimitText),
    };

    public static TheoryData<CallerLimit> NonWindowsCallerLimits => new()
    {
        new CallerLimit("0", "0"),
        new CallerLimit("202752", "202752"),
    };

    [Fact]
    public void Limit_is_99_KiB()
    {
        Assert.Equal(99 * 1024, TensorRtRtxProviderConstants.WindowsDirectMlInteropSharedMemoryLimitBytes);
        Assert.Equal(99 * 1024, long.Parse(LimitText, System.Globalization.CultureInfo.InvariantCulture));
    }

    [Fact]
    public void Apply_caps_shared_memory_on_windows_regardless_of_driver()
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal);

        TensorRtRtxDirectMlInterop.Apply(options, isWindows: true);

        Assert.Equal(LimitText, options[TensorRtRtxDirectMlInterop.MaxSharedMemSizeOptionKey]);
    }

    [Fact]
    public void Apply_leaves_options_alone_off_windows()
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal);

        TensorRtRtxDirectMlInterop.Apply(options, isWindows: false);

        Assert.Empty(options);
    }

    [Theory]
    [MemberData(nameof(WindowsCallerLimits))]
    public void Apply_resolves_a_caller_supplied_limit_on_windows(CallerLimit limit)
    {
        Dictionary<string, string> options = OptionsWith(limit.Requested);

        TensorRtRtxDirectMlInterop.Apply(options, isWindows: true);

        Assert.Equal(limit.Expected, options[TensorRtRtxDirectMlInterop.MaxSharedMemSizeOptionKey]);
    }

    [Theory]
    [MemberData(nameof(NonWindowsCallerLimits))]
    public void Apply_leaves_a_caller_supplied_limit_alone_off_windows(CallerLimit limit)
    {
        Dictionary<string, string> options = OptionsWith(limit.Requested);

        TensorRtRtxDirectMlInterop.Apply(options, isWindows: false);

        Assert.Equal(limit.Expected, options[TensorRtRtxDirectMlInterop.MaxSharedMemSizeOptionKey]);
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
        // Isolate process-wide cache environment variables so the test does not depend on shared mutable state.
        string? previousEngineCacheRoot = Environment.GetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT");
        string? previousCacheRoot = Environment.GetEnvironmentVariable("TRACKDUB_CACHE_ROOT");

        Environment.SetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT", null);
        Environment.SetEnvironmentVariable("TRACKDUB_CACHE_ROOT", null);

        try
        {
            IReadOnlyDictionary<string, string> options = OnnxExecutionSessionFactory.BuildTensorRtRtxOptions(null);

            Assert.Equal(
                TensorRtRtxProviderConstants.BundledFingerprintVersion,
                Path.GetFileName(options["nv_runtime_cache_path"]));
            Assert.Equal(
                OperatingSystem.IsWindows() ? LimitText : null,
                options.GetValueOrDefault(TensorRtRtxDirectMlInterop.MaxSharedMemSizeOptionKey));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TRACKDUB_ENGINE_CACHE_ROOT", previousEngineCacheRoot);
            Environment.SetEnvironmentVariable("TRACKDUB_CACHE_ROOT", previousCacheRoot);
        }
    }

    private static Dictionary<string, string> OptionsWith(string requestedLimit) =>
        new(StringComparer.Ordinal)
        {
            [TensorRtRtxDirectMlInterop.MaxSharedMemSizeOptionKey] = requestedLimit,
        };
}
