namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>The three exported graphs a window separation runs, batch size 1, with flat row-major buffers.</summary>
internal interface ISamAudioGraphs
{
    /// <summary>Codec encoder: hop-aligned mono wave to features, <c>frames x 128</c>.</summary>
    float[] Encode(float[] paddedWave, CancellationToken cancellationToken);

    /// <summary>One flow evaluation: noisy <c>frames x 256</c> and features to velocity <c>frames x 256</c>; frames come from the buffers.</summary>
    float[] Velocity(float time, float[] noisy, float[] features, CancellationToken cancellationToken);

    /// <summary>Codec decoder: latent <c>frames x 256</c> plus the watermark message to the target and residual waves.</summary>
    (float[] Target, float[] Residual) Decode(float[] latent, int frames, float[] message, CancellationToken cancellationToken);
}
