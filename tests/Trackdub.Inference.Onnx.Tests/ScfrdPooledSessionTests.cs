using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx;
using Trackdub.Inference.Onnx.FaceAnalysis;
using Trackdub.Inference.Onnx.Pool;

namespace Trackdub.Inference.Onnx.Tests;

public sealed class ScfrdPooledSessionTests
{
    [Fact]
    public async Task DetectPrimaryFaceAsync_two_calls_create_only_one_pooled_session()
    {
        string modelPath = Path.Join(Path.GetTempPath(), $"scrfd-{Guid.NewGuid():N}.onnx");
        await File.WriteAllBytesAsync(modelPath, BuildIdentityOnnxModel(), TestContext.Current.CancellationToken);
        using var pool = new InferenceSessionPool(maxSessions: 2);
        int createCount = 0;

        var detector = new ScfrdOnnxFaceDetector(
            new BenchmarkModelPathResolver(),
            new FakeFrameExtractor(),
            modelPathProvider: () => modelPath,
            sessionLeaseFactory: (path, cancellationToken) =>
                OnnxExecutionSessionFactory.CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest
                {
                    EngineFamily = "scrfd",
                    Provider = ExecutionProviderKind.Cpu,
                    CancellationToken = cancellationToken,
                    Pool = pool,
                    SessionFactory = (_, _) =>
                    {
                        createCount++;
                        return new InferenceSession(BuildIdentityOnnxModel());
                    }
                }, path));
        var request = new FaceAnalysisRequest("video.mp4", TimeSpan.Zero, TimeSpan.FromSeconds(2), null);

        try
        {
            // The identity model cannot produce SCRFD outputs, so detection reports
            // no face; the assertion is on session creation, not detection quality.
            Assert.False((await detector.DetectPrimaryFaceAsync(request, TestContext.Current.CancellationToken)).FaceFound);
            Assert.False((await detector.DetectPrimaryFaceAsync(request, TestContext.Current.CancellationToken)).FaceFound);
            Assert.Equal(1, createCount);
        }
        finally
        {
            File.Delete(modelPath);
        }
    }

    [Fact]
    public async Task DetectPrimaryFaceAsync_propagates_session_lease_cancellation()
    {
        string modelPath = Path.Join(Path.GetTempPath(), $"scrfd-{Guid.NewGuid():N}.onnx");
        using var cancellationSource = new CancellationTokenSource();
        bool leaseRequested = false;
        var detector = new ScfrdOnnxFaceDetector(
            new BenchmarkModelPathResolver(),
            new FakeFrameExtractor(),
            modelPathProvider: () => modelPath,
            sessionLeaseFactory: (_, cancellationToken) =>
            {
                leaseRequested = true;
                cancellationSource.Cancel();
                return Task.FromCanceled<OnnxExecutionSessionFactory.SingleSessionLease>(cancellationToken);
            });
        var request = new FaceAnalysisRequest("video.mp4", TimeSpan.Zero, TimeSpan.FromSeconds(2), null);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => detector.DetectPrimaryFaceAsync(request, cancellationSource.Token));
        Assert.True(leaseRequested);
    }

    private sealed class FakeFrameExtractor : IVideoFrameExtractor
    {
        public Task<FrameExtractionResult> ExtractTurnFramesAsync(
            string videoPath,
            double startSeconds,
            double endSeconds,
            string outputDirectory,
            CancellationToken cancellationToken)
        {
            Directory.CreateDirectory(outputDirectory);
            File.WriteAllBytes(Path.Join(outputDirectory, "frame_000001.rgba"), new byte[2 * 2 * 4]);
            return Task.FromResult(new FrameExtractionResult(outputDirectory, 2, 2, 1, 30d));
        }
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
