using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>Valid inputs for a warmup smoke run of the flow graph, which the generic one-input separation smoke cannot build.</summary>
internal static class SamAudioSmokeInputs
{
    private const int Frames = 4;
    private const int PromptTokens = 2;
    private const int PromptChannels = 768;

    private static readonly string[] FlowInputNames = ["noisy", "features", "text", "time"];

    /// <summary>True for the SAM-Audio flow graph, recognised by its four named inputs.</summary>
    public static bool Matches(InferenceSession session) =>
        new HashSet<string>(FlowInputNames, StringComparer.Ordinal).SetEquals(session.InputMetadata.Keys);

    public static NamedOnnxValue[] Create() =>
    [
        Zeros("noisy", 1, Frames, SamAudioModelConstants.LatentChannels),
        Zeros("features", 1, Frames, SamAudioModelConstants.FeatureChannels),
        Zeros("text", 1, PromptTokens, PromptChannels),
        Zeros("time", 1),
    ];

    private static NamedOnnxValue Zeros(string name, params int[] dimensions) =>
        NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(new float[dimensions.Aggregate(1, (a, b) => a * b)], dimensions));
}
