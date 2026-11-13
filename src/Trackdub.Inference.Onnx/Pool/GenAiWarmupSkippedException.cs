namespace Trackdub.Inference.Onnx.Pool;

/// <summary>
/// An optimization-only warmup declined to load a GenAI model because doing so would evict a
/// resident model or wait for eviction headroom. Nothing was proven about the model or provider.
/// </summary>
internal sealed class GenAiWarmupSkippedException(string message) : Exception(message);
