using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Inference.Onnx.Pool;

namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>The exported SAM-Audio graphs run through ONNX Runtime sessions leased by the caller.</summary>
internal sealed class OrtSamAudioGraphs(
    InferenceSession encoder,
    InferenceSession dit,
    InferenceSession decoder,
    SamAudioPrompt prompt) : ISamAudioGraphs
{
    public float[] Encode(float[] paddedWave, CancellationToken cancellationToken) =>
        RunSingle(encoder, cancellationToken, Input("wav", paddedWave, 1, 1, paddedWave.Length));

    public float[] Velocity(float time, float[] noisy, float[] features, CancellationToken cancellationToken)
    {
        int frames = noisy.Length / SamAudioModelConstants.LatentChannels;
        return RunSingle(
            dit,
            cancellationToken,
            Input("noisy", noisy, 1, frames, SamAudioModelConstants.LatentChannels),
            Input("features", features, 1, frames, SamAudioModelConstants.FeatureChannels),
            Input("text", prompt.Embedding, 1, prompt.Tokens, prompt.Channels),
            Input("time", [time], 1));
    }

    public (float[] Target, float[] Residual) Decode(float[] latent, int frames, float[] message, CancellationToken cancellationToken)
    {
        float[] both = RunSingle(
            decoder,
            cancellationToken,
            Input("latent", latent, 1, frames, SamAudioModelConstants.LatentChannels),
            Input("message", message, 2, SamAudioModelConstants.WatermarkBits));

        // Output is [1, 2, samples]: row 0 is the target (speech), row 1 the residual (bed).
        int samples = both.Length / 2;
        return (both.AsSpan(0, samples).ToArray(), both.AsSpan(samples, samples).ToArray());
    }

    private static NamedOnnxValue Input(string name, float[] data, params int[] dimensions) =>
        NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(data, dimensions));

    private static float[] RunSingle(InferenceSession session, CancellationToken cancellationToken, params NamedOnnxValue[] inputs)
    {
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = session.RunWithRetry(inputs, cancellationToken: cancellationToken);
        return outputs.First().AsTensor<float>().ToArray();
    }
}
