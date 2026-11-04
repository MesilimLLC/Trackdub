using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Trackdub.Inference.Onnx.Whisper;

internal static class WhisperEncoderOutputLifetime
{
    internal static async Task<TResult> ConsumeAsync<TResult>(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults,
        Func<Tensor<float>, Task<TResult>> consumeAsync,
        string outputName = "last_hidden_state")
    {
        ArgumentNullException.ThrowIfNull(encoderResults);
        ArgumentNullException.ThrowIfNull(consumeAsync);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputName);

        using (encoderResults)
        {
            Tensor<float> hiddenStates = encoderResults
                .Single(result => result.Name == outputName)
                .AsTensor<float>();
            return await consumeAsync(hiddenStates).ConfigureAwait(false);
        }
    }
}
