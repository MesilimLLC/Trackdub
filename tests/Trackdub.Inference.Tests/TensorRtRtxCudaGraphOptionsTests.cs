using Trackdub.Inference.Onnx;

namespace Trackdub.Inference.Tests;

[Collection(nameof(OnnxExecutionSessionFactoryStateCollection))]
public sealed class TensorRtRtxCudaGraphOptionsTests
{
    public sealed record OverrideCase(string? Override, string CallSite, string Expected);

    public sealed record SessionDecision(bool Enabled, string Expected);

    public static TheoryData<OverrideCase> OverrideCases => new()
    {
        new OverrideCase("on", "0", "1"),
        new OverrideCase("ON", "0", "1"),
        new OverrideCase("1", "0", "1"),
        new OverrideCase("true", "0", "1"),
        new OverrideCase("off", "1", "0"),
        new OverrideCase("0", "1", "0"),
        new OverrideCase("false", "1", "0"),
        new OverrideCase(null, "1", "1"),
        new OverrideCase("", "0", "0"),
        new OverrideCase("auto", "1", "1"),
    };

    public static TheoryData<SessionDecision> SessionDecisions => new()
    {
        new SessionDecision(true, "1"),
        new SessionDecision(false, "0"),
    };

    [Fact]
    public void ParseCudaGraphOverride_selects_a_flag_only_for_recognized_values()
    {
        Assert.Equal("1", OnnxExecutionSessionFactory.ParseCudaGraphOverride(" On "));
        Assert.Equal("0", OnnxExecutionSessionFactory.ParseCudaGraphOverride("FALSE"));
        Assert.Null(OnnxExecutionSessionFactory.ParseCudaGraphOverride("auto"));
        Assert.Null(OnnxExecutionSessionFactory.ParseCudaGraphOverride(null));
        Assert.Null(OnnxExecutionSessionFactory.ParseCudaGraphOverride(string.Empty));
    }

    [Theory]
    [MemberData(nameof(OverrideCases))]
    public void ApplyCudaGraphOverride_only_acts_on_on_or_off(OverrideCase testCase)
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal) { ["enable_cuda_graph"] = testCase.CallSite };

        OnnxExecutionSessionFactory.ApplyCudaGraphOverride(options, testCase.Override);

        Assert.Equal(testCase.Expected, options["enable_cuda_graph"]);
    }

    [Theory]
    [InlineData("off", "0")]
    [InlineData("on", "1")]
    [InlineData(null, "1")]
    public void Override_wins_over_a_profile_built_with_WithCudaGraphsEnabled(string? overrideValue, string expected)
    {
        IReadOnlyDictionary<string, string> profile = OnnxExecutionSessionFactory.WithCudaGraphsEnabled(Profile());

        IReadOnlyDictionary<string, string> options = OnnxExecutionSessionFactory.BuildTensorRtRtxOptionsWithOverride(
            profile,
            enableCudaGraph: false,
            overrideValue);

        Assert.Equal(expected, options["enable_cuda_graph"]);
        Assert.Equal("input_features:1x80x1", options["nv_profile_min_shapes"]);
    }

    [Theory]
    [MemberData(nameof(SessionDecisions))]
    public void A_per_session_decision_wins_over_the_default(SessionDecision decision)
    {
        IReadOnlyDictionary<string, string> profile =
            OnnxExecutionSessionFactory.WithCudaGraphs(Profile(), decision.Enabled);

        IReadOnlyDictionary<string, string> options = OnnxExecutionSessionFactory.BuildTensorRtRtxOptionsWithOverride(
            profile,
            OnnxExecutionSessionFactory.DefaultEnableCudaGraph,
            cudaGraphOverride: null);

        Assert.Equal(decision.Expected, options["enable_cuda_graph"]);
    }

    private static Dictionary<string, string> Profile() =>
        new(StringComparer.Ordinal) { ["nv_profile_min_shapes"] = "input_features:1x80x1" };
}
