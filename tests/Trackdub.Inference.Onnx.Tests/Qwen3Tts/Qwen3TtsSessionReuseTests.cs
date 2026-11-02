using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.Qwen3Tts.Models;

namespace Trackdub.Inference.Onnx.Tests.Qwen3Tts;

public sealed class Qwen3TtsSessionReuseTests
{
    // M5: two syntheses worth of prefill acquisition must share one pooled session.
    [Fact]
    public async Task Prefill_pooled_session_is_not_disposed_or_recreated_across_two_syntheses()
    {
        using var pool = new InferenceSessionPool(maxSessions: 2, enableMemoryAdmission: false);
        int createCount = 0;
        byte[] model = BuildIdentityOnnxModel();

        using var pin = await OnnxExecutionSessionFactory.PinPooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "qwen3-tts-test", Provider = ExecutionProviderKind.Cpu, CancellationToken = CancellationToken.None, Pool = pool, SessionFactory = (_, _) =>
            {
                createCount++;
                return new InferenceSession(model);
            } }, "talker_prefill.onnx");
        using var prefill = new QwenPooledSession(pin);

        InferenceSession? first = null;
        for (int synthesis = 0; synthesis < 2; synthesis++)
        {
            // Each synthesis acquires and releases its own lease; both must resolve to
            // the same pinned session without recreating it.
            using var lease = await prefill.AcquireAsync(TestContext.Current.CancellationToken);
            InferenceSession session = lease.Session;
            first ??= session;
            Assert.Same(first, session);

            // A disposed session throws ObjectDisposedException on Run; the identity
            // graph proves the reused session is still live on the second synthesis.
            var input = new DenseTensor<float>(new float[] { 42f }, [1]);
            using var results = session.Run([NamedOnnxValue.CreateFromTensor("x", input)]);
            Assert.Equal(42f, results[0].AsTensor<float>()[0]);
        }

        Assert.Equal(1, createCount);
    }

    [Fact]
    public async Task Disposed_pooled_session_throws_instead_of_acquiring_a_new_lease()
    {
        using var pool = new InferenceSessionPool(maxSessions: 2, enableMemoryAdmission: false);
        byte[] model = BuildIdentityOnnxModel();
        using var pin = await OnnxExecutionSessionFactory.PinPooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "qwen3-tts-test-disposed", Provider = ExecutionProviderKind.Cpu, CancellationToken = CancellationToken.None, Pool = pool, SessionFactory = (_, _) => new InferenceSession(model) }, "talker_prefill.onnx");
        var prefill = new QwenPooledSession(pin);
        using (await prefill.AcquireAsync(TestContext.Current.CancellationToken))
        {
        }

        prefill.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => prefill.AcquireAsync(TestContext.Current.CancellationToken));
    }

    private static byte[] BuildIdentityOnnxModel() =>
    [
        0x08, 0x07, 0x3A, 0x3A, 0x0A, 0x10, 0x0A, 0x01, 0x78, 0x12, 0x01, 0x79, 0x22, 0x08,
        0x49, 0x64, 0x65, 0x6E, 0x74, 0x69, 0x74, 0x79, 0x12, 0x04, 0x74, 0x65, 0x73, 0x74,
        0x5A, 0x0F, 0x0A, 0x01, 0x78, 0x12, 0x0A, 0x0A, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0A,
        0x02, 0x08, 0x01, 0x62, 0x0F, 0x0A, 0x01, 0x79, 0x12, 0x0A, 0x0A, 0x08, 0x08, 0x01,
        0x12, 0x04, 0x0A, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0A, 0x00, 0x10, 0x09,
    ];
}
