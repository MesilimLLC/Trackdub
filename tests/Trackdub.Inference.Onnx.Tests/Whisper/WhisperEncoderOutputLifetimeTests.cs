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
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = session.Run([input]);

        float decodedValue = await WhisperEncoderOutputLifetime.ConsumeAsync(
            outputs,
            async hiddenStates =>
            {
                Assert.Equal(42f, hiddenStates.GetValue(0));
                await Task.Yield();
                return hiddenStates.GetValue(0);
            },
            outputName: "y");

        Assert.Equal(42f, decodedValue);
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
