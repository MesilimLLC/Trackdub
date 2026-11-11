using Trackdub.Inference.Onnx.EpContext;
using Trackdub.Inference.Onnx.Madlad;
using Trackdub.Inference.Onnx.Translation;

namespace Trackdub.Inference.Onnx.Tests;

public sealed class MadladTrtProfilesTests
{
    [Theory]
    [InlineData("encoder_model.onnx", false)]
    [InlineData("decoder_model.onnx", true)]
    public void Aot_uses_runtime_profiles_without_opening_a_cpu_session(string fileName, bool decoder)
    {
        string path = Path.Join("not-downloaded", "madlad400-3b-mt", "trt_rtx_mixed_fp16_fp32", fileName);
        var profile = decoder ? MadladTrtProfiles.Decoder : MadladTrtProfiles.Encoder;
        Assert.Same(profile, EpContextTrtProfiles.Resolve(path));
        Assert.Null(MadladTrtProfiles.GetAotExclusionReason(path));
        var options = OnnxExecutionSessionFactory.BuildTensorRtRtxOptions(profile);
        Assert.Equal("1", options["enable_cuda_graph"]);
        Assert.DoesNotContain("trt_profile_min_shapes", options.Keys);

        foreach (string bound in new[] { "min", "opt", "max" })
        {
            var shapes = Parse(options[$"nv_profile_{bound}_shapes"]);
            Assert.Equal(decoder ? 3 : 2, shapes.Count);
            foreach (int[] shape in shapes.Values)
            {
                Assert.Equal(1, shape[0]);
                Assert.InRange(shape[1], 1, 512);
            }
            if (decoder) Assert.Equal(1024, shapes["encoder_hidden_states"][2]);
        }

        var max = Parse(options["nv_profile_max_shapes"]);
        for (int length = 1; length <= MadladTrtProfiles.MaximumSequenceLength; length++)
        {
            Assert.InRange(Seq2SeqGreedyDecoder.BucketLength(length), 1, max["input_ids"][1]);
        }
    }

    [Theory]
    [InlineData("quantized", "decoder_model_quantized.onnx")]
    [InlineData("trt_rtx_mixed_fp16_fp32", "decoder_model_merged.onnx")]
    [InlineData("trt_rtx_mixed_fp16_fp32", "decoder_with_past_model.onnx")]
    public async Task Unsafe_aot_graphs_are_rejected_before_provider_registration(string directory, string fileName)
    {
        string root = Path.Join(Path.GetTempPath(), $"madlad400-test-{Guid.NewGuid():N}");
        string model = Path.Join(root, directory, fileName);
        Directory.CreateDirectory(Path.GetDirectoryName(model)!);
        File.WriteAllBytes(model, [1]);
        try
        {
            var result = await new EpContextCompiler().CompileAsync(model, Path.Join(root, "output.onnx"), TestContext.Current.CancellationToken);
            Assert.False(result.Success);
            Assert.Contains("validated trt-fp16", result.FailureReason);
            Assert.False(File.Exists(Path.Join(root, "output.onnx")));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Theory]
    [InlineData(513, 256)]
    [InlineData(32, 513)]
    public void Oversized_inputs_fail_before_native_execution(int source, int generated) =>
        Assert.Throws<InvalidOperationException>(() => MadladTrtProfiles.ValidateLengths(source, generated));

    [Fact]
    public void Unrelated_graphs_are_not_excluded_by_checkout_name() =>
        Assert.Null(MadladTrtProfiles.GetAotExclusionReason(Path.Join("madlad-trt-profiles", "silero", "model.onnx")));

    private static Dictionary<string, int[]> Parse(string profile) => profile.Split(',')
        .Select(static entry => entry.Split(':'))
        .ToDictionary(static parts => parts[0], static parts => parts[1].Split('x').Select(int.Parse).ToArray());

    [Fact]
    public void Compile_and_warmup_sessions_switch_off_capture_even_for_profiles_that_enable_it()
    {
        Assert.Equal("1", MadladTrtProfiles.Encoder["enable_cuda_graph"]);

        foreach (var profile in new[] { MadladTrtProfiles.Encoder, MadladTrtProfiles.Decoder })
        {
            var options = OnnxExecutionSessionFactory.BuildTensorRtRtxOptionsWithOverride(
                OnnxExecutionSessionFactory.ForEngineBuild(profile),
                OnnxExecutionSessionFactory.DefaultEnableCudaGraph,
                cudaGraphOverride: null);

            Assert.Equal("0", options["enable_cuda_graph"]);
            Assert.Equal(profile["trt_profile_max_shapes"], options["nv_profile_max_shapes"]);
        }
    }

    [Fact]
    public void Engine_build_options_accept_a_missing_profile() =>
        Assert.Equal("0", OnnxExecutionSessionFactory.ForEngineBuild(null)["enable_cuda_graph"]);
}
