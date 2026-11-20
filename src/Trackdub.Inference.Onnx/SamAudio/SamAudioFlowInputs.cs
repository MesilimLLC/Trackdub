namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>The random inputs of one window's flow: starting noise (frames x 256) and the [2, 16] watermark message.</summary>
internal sealed record SamAudioFlowInputs(float[] Noise, float[] WatermarkMessage);
