using Trackdub.Inference.Onnx.CosyVoice;

namespace Trackdub.Inference.Onnx.Tests;

public sealed class CosyVoiceSessionOptionsTests
{
    [Fact]
    public void Speech_tokenizer_opts_out_of_the_cuda_graph_default() =>
        Assert.Equal(
            "0",
            OnnxExecutionSessionFactory.BuildTensorRtRtxOptionsWithOverride(
                CosyVoiceSessionPins.SpeechTokenizerTrtOptions,
                OnnxExecutionSessionFactory.DefaultEnableCudaGraph,
                cudaGraphOverride: null)["enable_cuda_graph"]);
}
