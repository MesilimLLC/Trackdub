using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Domain;
using Trackdub.Inference.Onnx;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.TensorRtRtx;

namespace Trackdub.Inference.Tests;

[Collection(nameof(TensorRtRtxTeardownGuardCollection))]
public sealed class PooledWhisperTrtFallbackTests : IDisposable
{
    private readonly string directory = Path.Join(Path.GetTempPath(), "trackdub-whisper-fallback-" + Guid.NewGuid().ToString("N"));

    public PooledWhisperTrtFallbackTests()
    {
        Directory.CreateDirectory(directory);
        TensorRtRtxTeardownGuard.ResetForTests();
    }

    public void Dispose()
    {
        TensorRtRtxTeardownGuard.ResetForTests();
        Directory.Delete(directory, recursive: true);
    }

    [Fact]
    public async Task CreatePooledWhisperAsync_retries_on_the_next_provider_when_trt_session_creation_fails()
    {
        string encoder = WriteModel("encoder_model.onnx");
        string decoder = WriteModel("decoder_model.onnx");
        using var pool = new InferenceSessionPool(maxSessions: 4, enableMemoryAdmission: false);
        int creations = 0;

        using OnnxExecutionSessionFactory.WhisperSessionLease lease = await OnnxExecutionSessionFactory.CreatePooledWhisperAsync(new OnnxExecutionSessionFactory.PooledSessionRequest
        {
            EngineFamily = "whisper-onnx",
            Provider = ExecutionProviderKind.TensorRTRtx,
            CancellationToken = CancellationToken.None,
            Pool = pool,
            SessionFactory = (_, _) =>
            {
                // The first creation (the TensorRT RTX attempt's first session) dies in the engine
                // build, which aborts that attempt; the next provider's attempt then succeeds.
                if (Interlocked.Increment(ref creations) == 1)
                {
                    throw new InvalidOperationException(
                        "[NvTensorRTRTX EP] CUDA failure 2: out of memory while building the engine");
                }

                return CreateMinimalSession();
            }
        }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = encoder, Decoder = decoder });

        Assert.DoesNotContain("TensorRT", lease.SelectedProvider, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("TensorRT", lease.RequestedProvider, StringComparison.OrdinalIgnoreCase);
        Assert.NotNull(lease.BootstrapDetail);
        Assert.Contains("Whisper session init failed", lease.BootstrapDetail, StringComparison.Ordinal);
        Assert.Contains("out of memory", lease.BootstrapDetail, StringComparison.Ordinal);
        Assert.True(TensorRtRtxTeardownGuard.IsPoisoned);
    }

    [Fact]
    public async Task CreatePooledWhisperAsync_when_decoder_fails_with_cuda_oom_after_encoder_succeeds_evicts_safely_and_falls_back()
    {
        string encoder = WriteModel("encoder_model.onnx");
        string decoder = WriteModel("decoder_model.onnx");
        InferenceSession? trtEncoderSession = null;
        var fallbackSessions = new List<InferenceSession>();
        try
        {
            using (var pool = new InferenceSessionPool(maxSessions: 4, enableMemoryAdmission: false))
            {
                int creations = 0;
                using OnnxExecutionSessionFactory.WhisperSessionLease lease = await OnnxExecutionSessionFactory.CreatePooledWhisperAsync(new OnnxExecutionSessionFactory.PooledSessionRequest
                {
                    EngineFamily = "whisper-onnx",
                    Provider = ExecutionProviderKind.TensorRTRtx,
                    CancellationToken = CancellationToken.None,
                    Pool = pool,
                    SessionFactory = (_, _) =>
                    {
                        int step = Interlocked.Increment(ref creations);
                        if (step == 1)
                        {
                            // Simulate encoder succeeding on TensorRT-RTX before the decoder OOMs.
                            trtEncoderSession = CreateMinimalSession();
                            TensorRtRtxTeardownGuard.Track(trtEncoderSession);
                            return trtEncoderSession;
                        }

                        if (step == 2)
                        {
                            throw new InvalidOperationException(
                                "[NvTensorRTRTX EP] CUDA failure 2: out of memory building whisper-large-v3 decoder profile");
                        }

                        InferenceSession fallback = CreateMinimalSession();
                        lock (fallbackSessions)
                        {
                            fallbackSessions.Add(fallback);
                        }

                        return fallback;
                    }
                }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = encoder, Decoder = decoder });

                Assert.True(TensorRtRtxTeardownGuard.IsPoisoned);
                Assert.NotNull(trtEncoderSession);
                Assert.Equal(2, fallbackSessions.Count);
                Assert.Equal(1f, RunIdentity(lease.EncoderSession));
                Assert.Equal(1f, RunIdentity(lease.DecoderSession));
            }

            // After pool disposal, the poisoned TRT encoder session skipped native teardown,
            // while the fallback CPU/DirectML sessions were untracked and disposed normally.
            Assert.Equal(1f, RunIdentity(trtEncoderSession!));
            foreach (InferenceSession fallback in fallbackSessions)
            {
                Assert.Throws<NullReferenceException>(() => RunIdentity(fallback));
            }
        }
        finally
        {
            TensorRtRtxTeardownGuard.ResetForTests();
            trtEncoderSession?.Dispose();
            foreach (InferenceSession fallback in fallbackSessions)
            {
                fallback.Dispose();
            }
        }
    }

    [Fact]
    public async Task CreatePooledWhisperAsync_with_allowTrtInitFallback_false_poisons_guard_and_does_not_retry()
    {
        string encoder = WriteModel("encoder_model.onnx");
        string decoder = WriteModel("decoder_model.onnx");
        using var pool = new InferenceSessionPool(maxSessions: 4, enableMemoryAdmission: false);
        int creations = 0;

        await Assert.ThrowsAsync<InvalidOperationException>(() => OnnxExecutionSessionFactory.CreatePooledWhisperAsync(new OnnxExecutionSessionFactory.PooledSessionRequest
        {
            EngineFamily = "whisper-onnx",
            Provider = ExecutionProviderKind.TensorRTRtx,
            CancellationToken = CancellationToken.None,
            Pool = pool,
            SessionFactory = (_, _) =>
            {
                Interlocked.Increment(ref creations);
                throw new InvalidOperationException("CUDA_ERROR_OUT_OF_MEMORY");
            },
            AllowTrtInitFallback = false
        }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = encoder, Decoder = decoder }));

        // Bundle graphs are created in parallel, so both may get their one TRT attempt
        // before the first failure is observed; a fallback retry would exceed 2.
        Assert.True(creations is >= 1 and <= 2,
            $"TRT requests with fallback disabled must not retry (saw {creations} creations).");
        Assert.True(TensorRtRtxTeardownGuard.IsPoisoned);
    }

    [Fact]
    public async Task CreatePooledWhisperAsync_does_not_retry_non_trt_requests()
    {
        string encoder = WriteModel("encoder_model.onnx");
        string decoder = WriteModel("decoder_model.onnx");
        using var pool = new InferenceSessionPool(maxSessions: 4, enableMemoryAdmission: false);
        int creations = 0;

        await Assert.ThrowsAsync<InvalidOperationException>(() => OnnxExecutionSessionFactory.CreatePooledWhisperAsync(new OnnxExecutionSessionFactory.PooledSessionRequest
        {
            EngineFamily = "whisper-onnx",
            Provider = ExecutionProviderKind.Cpu,
            CancellationToken = CancellationToken.None,
            Pool = pool,
            SessionFactory = (_, _) =>
            {
                Interlocked.Increment(ref creations);
                throw new InvalidOperationException("cpu session failed");
            }
        }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = encoder, Decoder = decoder }));

        Assert.True(creations <= 2, $"CPU requests must not walk a fallback chain (saw {creations} creations).");
    }

    private string WriteModel(string name)
    {
        string path = Path.Join(directory, name);
        File.WriteAllBytes(path, IdentityModel);
        return path;
    }

    private static float RunIdentity(InferenceSession session)
    {
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = session.Run(
            [NamedOnnxValue.CreateFromTensor("x", new DenseTensor<float>(new[] { 1f }, new[] { 1 }))]);
        return Assert.Single(outputs).AsTensor<float>()[0];
    }

    private static InferenceSession CreateMinimalSession() => new(IdentityModel);

    private static readonly byte[] IdentityModel =
    [
        0x08, 0x07, 0x3A, 0x3A, 0x0A, 0x10, 0x0A, 0x01, 0x78, 0x12, 0x01, 0x79, 0x22, 0x08,
        0x49, 0x64, 0x65, 0x6E, 0x74, 0x69, 0x74, 0x79, 0x12, 0x04, 0x74, 0x65, 0x73, 0x74,
        0x5A, 0x0F, 0x0A, 0x01, 0x78, 0x12, 0x0A, 0x0A, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0A,
        0x02, 0x08, 0x01, 0x62, 0x0F, 0x0A, 0x01, 0x79, 0x12, 0x0A, 0x0A, 0x08, 0x08, 0x01,
        0x12, 0x04, 0x0A, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0A, 0x00, 0x10, 0x09,
    ];
}
