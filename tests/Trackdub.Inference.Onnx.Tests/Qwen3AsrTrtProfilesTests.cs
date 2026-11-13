using Trackdub.Inference.Onnx.EpContext;
using Trackdub.Inference.Onnx.Qwen3Asr;

namespace Trackdub.Inference.Onnx.Tests;

public sealed class Qwen3AsrTrtProfilesTests
{
    [Fact]
    public void Resolve_maps_each_graph_to_its_profile()
    {
        string root = Path.Join("models", "qwen3-asr-0.6b-onnx");

        Assert.Same(Qwen3AsrTrtProfiles.Encoder, Qwen3AsrTrtProfiles.Resolve(Path.Join(root, "encoder.onnx")));
        Assert.Same(Qwen3AsrTrtProfiles.Decoder, Qwen3AsrTrtProfiles.Resolve(Path.Join(root, "decoder_init.onnx")));
        Assert.Same(Qwen3AsrTrtProfiles.Decoder, Qwen3AsrTrtProfiles.Resolve(Path.Join(root, "decoder_step.onnx")));
    }

    [Fact]
    public void Resolve_rejects_non_qwen3_paths()
    {
        Assert.Null(Qwen3AsrTrtProfiles.Resolve(Path.Join("models", "whisper", "encoder.onnx")));
        Assert.Null(Qwen3AsrTrtProfiles.Resolve(Path.Join("models", "qwen3-asr-0.6b-onnx", "embed_tokens.bin")));
    }

    [Fact]
    public void EpContext_resolve_returns_qwen3_profiles_without_opening_the_model()
    {
        // The path check runs before the metadata-only CPU session, so a missing file still resolves.
        string decoderStep = Path.Join("models", "qwen3-asr-0.6b-onnx", "decoder_step.onnx");
        Assert.Same(Qwen3AsrTrtProfiles.Decoder, EpContextTrtProfiles.Resolve(decoderStep));
    }

    [Fact]
    public void Encoder_keeps_bounded_mel_profile()
    {
        Assert.Equal("0", Qwen3AsrTrtProfiles.Encoder["enable_cuda_graph"]);
        Assert.Equal("mel:1x128x1", Qwen3AsrTrtProfiles.Encoder["trt_profile_min_shapes"]);
        Assert.Equal("mel:1x128x3000", Qwen3AsrTrtProfiles.Encoder["trt_profile_max_shapes"]);
        Assert.Equal("mel:1x128x3000", Qwen3AsrTrtProfiles.Encoder["trt_profile_opt_shapes"]);
    }

    [Fact]
    public void Decoder_opts_out_of_cuda_graph_without_shape_profiles()
    {
        // decoder KV seq grows every generated token; capture would recapture per token. Explicit
        // shape profiles were measured to make the cold JIT build ~3.7x slower, so the decoders
        // keep the implicit dynamic profile and only disable graph capture.
        Assert.Equal("0", Qwen3AsrTrtProfiles.Decoder["enable_cuda_graph"]);
        Assert.DoesNotContain(Qwen3AsrTrtProfiles.Decoder.Keys, static key => key.StartsWith("trt_profile_", StringComparison.Ordinal));
    }
}
