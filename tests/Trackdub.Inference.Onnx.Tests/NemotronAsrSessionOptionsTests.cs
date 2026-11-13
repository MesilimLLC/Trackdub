using Trackdub.Inference.Onnx.NemotronAsr;

namespace Trackdub.Inference.Onnx.Tests;

public sealed class NemotronAsrSessionOptionsTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Shared_runtime_and_smoke_options_follow_the_session_default_for_both_sessions(bool hasPromptInput)
    {
        NemotronAsrEncoderTrtProfiles.SessionOptions options =
            NemotronAsrEncoderTrtProfiles.BuildRuntimeOptions(hasPromptInput);

        Assert.Equal("1", OnnxExecutionSessionFactory.BuildTensorRtRtxOptions(options.Encoder)["enable_cuda_graph"]);
        Assert.Equal("1", OnnxExecutionSessionFactory.BuildTensorRtRtxOptions(options.Decoder)["enable_cuda_graph"]);
        foreach ((string name, string shape) in NemotronAsrEncoderTrtProfiles.BuildOptions(hasPromptInput))
        {
            Assert.Equal(shape, options.Encoder[name]);
        }

        Assert.Empty(options.Decoder);
        Assert.DoesNotContain("enable_cuda_graph", options.Encoder.Keys);
    }

    [Fact]
    public void Benchmark_profiles_leave_capture_mode_under_caller_control() =>
        Assert.DoesNotContain("enable_cuda_graph", NemotronAsrEncoderTrtProfiles.BuildOptions(hasPromptInput: true).Keys);

    [Fact]
    public void Qwen3_asr_encoder_opts_out_of_the_cuda_graph_default() =>
        Assert.Equal(
            "0",
            OnnxExecutionSessionFactory.BuildTensorRtRtxOptionsWithOverride(
                Qwen3Asr.Qwen3AsrTrtProfiles.Encoder,
                OnnxExecutionSessionFactory.DefaultEnableCudaGraph,
                cudaGraphOverride: null)["enable_cuda_graph"]);
}
