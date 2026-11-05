using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Inference.Onnx.Whisper;

namespace Trackdub.Inference.Onnx.Tests.Whisper;

public sealed class WhisperEncoderOutputLifetimeTests
{
    [Fact]
    public async Task Encoder_output_stays_alive_until_async_decode_finishes()
    {
        using var session = new InferenceSession(IdentityOnnxModel);
        var input = NamedOnnxValue.CreateFromTensor(
            "x",
            new DenseTensor<float>(new float[] { 42f }, [1]));
        var outputs = new TrackingOutputs(session.Run([input]));
        var decodeStarted = new TaskCompletionSource<bool>(
            TaskCreationOptions.RunContinuationsAsynchronously);
        var releaseDecode = new TaskCompletionSource<bool>(
            TaskCreationOptions.RunContinuationsAsynchronously);

        Task<float> decode = WhisperEncoderOutputLifetime.ConsumeAsync(
            outputs,
            async hiddenStates =>
            {
                Assert.Equal(42f, hiddenStates.GetValue(0));
                decodeStarted.SetResult(true);
                await releaseDecode.Task;
                Assert.False(outputs.IsDisposed);
                return hiddenStates.GetValue(0);
            },
            outputName: "y");

        await decodeStarted.Task;
        Assert.False(outputs.IsDisposed);

        releaseDecode.SetResult(true);
        float decodedValue = await decode;

        Assert.True(outputs.IsDisposed);
        Assert.Equal(42f, decodedValue);
    }

    private sealed class TrackingOutputs : IDisposableReadOnlyCollection<DisposableNamedOnnxValue>
    {
        private readonly IDisposableReadOnlyCollection<DisposableNamedOnnxValue> inner;

        public TrackingOutputs(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> inner)
        {
            this.inner = inner;
        }

        public bool IsDisposed { get; private set; }

        public int Count => inner.Count;

        public DisposableNamedOnnxValue this[int index] => inner[index];

        public void Dispose()
        {
            IsDisposed = true;
            inner.Dispose();
        }

        public IEnumerator<DisposableNamedOnnxValue> GetEnumerator() => inner.GetEnumerator();

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
    }

    // x (float32 [1]) -> Identity -> y (float32 [1]); ir_version 7, opset 9.
    private static readonly byte[] IdentityOnnxModel =
    [
        0x08, 0x07, 0x3A, 0x3A, 0x0A, 0x10, 0x0A, 0x01, 0x78, 0x12, 0x01, 0x79, 0x22, 0x08,
        0x49, 0x64, 0x65, 0x6E, 0x74, 0x69, 0x74, 0x79, 0x12, 0x04, 0x74, 0x65, 0x73, 0x74,
        0x5A, 0x0F, 0x0A, 0x01, 0x78, 0x12, 0x0A, 0x0A, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0A,
        0x02, 0x08, 0x01, 0x62, 0x0F, 0x0A, 0x01, 0x79, 0x12, 0x0A, 0x0A, 0x08, 0x08, 0x01,
        0x12, 0x04, 0x0A, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0A, 0x00, 0x10, 0x09,
    ];
}
