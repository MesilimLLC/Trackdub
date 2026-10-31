using Trackdub.Inference.Onnx;

namespace Trackdub.Inference.Tests;

public sealed class OnnxTensorRtCacheIdentityTests
{
    [Fact]
    public void BuildTensorRtOptions_cache_path_differs_for_punctuation_only_model_difference()
    {
        IReadOnlyDictionary<string, string> hyphenated = OnnxExecutionSessionFactory.BuildTensorRtOptions(
            null, CompleteTensorRtFingerprint(model: "RTX-5000"));
        IReadOnlyDictionary<string, string> spaced = OnnxExecutionSessionFactory.BuildTensorRtOptions(
            null, CompleteTensorRtFingerprint(model: "RTX 5000"));

        Assert.NotEqual(hyphenated["trt_engine_cache_path"], spaced["trt_engine_cache_path"]);
    }

    private static string CompleteTensorRtFingerprint(
        string architecture = "Ampere",
        string model = "RTX-4090",
        string driver = "535.104.05",
        char tensorRtHashCharacter = 'A') =>
        $"{architecture}|{model}|{driver}|trt-sha256-{new string(tensorRtHashCharacter, 64)}|ort-1.31.0";
}
