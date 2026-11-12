using System.Security.Cryptography;
using Trackdub.Contracts;
using Trackdub.Contracts.Licensing;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed class DiarizationStageHandlerTests
{
    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private static string CreateTempDirectory()
    {
        string dir = Path.Join(Path.GetTempPath(), "trackdub-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static DiarizationStageHandler BuildHandler(
        string modelCacheRoot,
        IModelDownloaderContract? downloader = null,
        IModelCacheRegistrar? registrar = null,
        string? expectedSha256 = null) =>
        new DiarizationStageHandler(
            new FakeDiarizationEngine(),
            downloader ?? new NoOpModelDownloader(),
            modelCacheRegistrar: registrar,
            modelCacheRoot: modelCacheRoot,
            expectedSha256: expectedSha256);

    /// <summary>
    /// Resolves the expected model file path mirroring <c>ResolveModelRootPath</c> and
    /// <c>ResolveModelFilePath</c> in <see cref="DiarizationStageHandler"/>.
    /// </summary>
    private static string ExpectedModelFilePath(string modelCacheRoot)
    {
        // modelCacheRoot / cgus / diar_streaming_sortformer_4spk-v2.1-onnx / onnx / model.onnx
        return Path.Join(
            Path.GetFullPath(modelCacheRoot),
            "cgus",
            "diar_streaming_sortformer_4spk-v2.1-onnx",
            "onnx",
            "model.onnx");
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    [Fact]
    public void GetRequiredModelStatus_WhenModelMissing_ReturnsNotAvailable()
    {
        string modelCacheRoot = Path.Join(CreateTempDirectory(), "model-cache");
        DiarizationStageHandler handler = BuildHandler(modelCacheRoot);

        RequiredDiarizationModelStatus status = handler.GetRequiredModelStatus();

        Assert.False(status.IsAvailable);
        Assert.True(status.CanAutoDownload);
        Assert.False(status.RequiresOnnxExport);
        Assert.Equal("cgus/diar_streaming_sortformer_4spk-v2.1-onnx", status.ModelId);
    }

    [Fact]
    public void GetRequiredModelStatus_WhenCorruptFileExistsWithoutVerifiedCacheRecord_ReturnsNotAvailable()
    {
        string modelCacheRoot = Path.Join(CreateTempDirectory(), "model-cache");
        string modelPath = ExpectedModelFilePath(modelCacheRoot);
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);
        File.WriteAllBytes(modelPath, [0xDE, 0xAD, 0xBE, 0xEF]);

        DiarizationStageHandler handler = BuildHandler(modelCacheRoot);

        Assert.False(handler.GetRequiredModelStatus().IsAvailable);
    }

    [Fact]
    public async Task GetRequiredModelStatus_WhenVerifiedCacheRecordExists_ReturnsAvailable()
    {
        string modelCacheRoot = Path.Join(CreateTempDirectory(), "model-cache");
        string modelPath = ExpectedModelFilePath(modelCacheRoot);
        string modelRootPath = Path.GetFullPath(Path.Join(modelCacheRoot, "cgus", "diar_streaming_sortformer_4spk-v2.1-onnx"));
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);
        await File.WriteAllBytesAsync(modelPath, SortFormerTestFixtures.ModelBytes, TestContext.Current.CancellationToken);

        var registrar = new RecordingModelCacheRegistrar();
        await registrar.RegisterAsync(
            new LocalModelCacheRecord(
                "cgus/diar_streaming_sortformer_4spk-v2.1-onnx",
                modelRootPath,
                "2be05a08b477e8a526fd26963802845069c02c7c",
                SortFormerTestFixtures.ExpectedSha256,
                DateTimeOffset.UtcNow),
            TestContext.Current.CancellationToken);

        DiarizationStageHandler handler = BuildHandler(
            modelCacheRoot,
            registrar: registrar,
            expectedSha256: SortFormerTestFixtures.ExpectedSha256);

        Assert.True(handler.GetRequiredModelStatus().IsAvailable);
    }

    [Fact]
    public async Task DownloadRequiredModelAsync_RejectsDownloadWithInvalidChecksum()
    {
        string modelCacheRoot = Path.Join(CreateTempDirectory(), "model-cache");
        var downloader = new WritingModelDownloader();
        DiarizationStageHandler handler = BuildHandler(modelCacheRoot, downloader: downloader);

        await Assert.ThrowsAsync<RequiredModelNotAvailableException>(() =>
            handler.DownloadRequiredModelAsync(cancellationToken: TestContext.Current.CancellationToken));

        Assert.NotNull(downloader.DestinationPath);
        string expectedPath = ExpectedModelFilePath(modelCacheRoot);
        Assert.False(File.Exists(expectedPath));
        Assert.False(handler.GetRequiredModelStatus().IsAvailable);
    }

    [Fact]
    public async Task DownloadRequiredModelAsync_DoesNotRegisterWhenChecksumInvalid()
    {
        string modelCacheRoot = Path.Join(CreateTempDirectory(), "model-cache");
        var downloader = new WritingModelDownloader();
        var registrar = new RecordingModelCacheRegistrar();
        DiarizationStageHandler handler = BuildHandler(modelCacheRoot, downloader: downloader, registrar: registrar);

        await Assert.ThrowsAsync<RequiredModelNotAvailableException>(() =>
            handler.DownloadRequiredModelAsync(cancellationToken: TestContext.Current.CancellationToken));

        Assert.Null(registrar.Record);
    }

    [Fact]
    public async Task ImportModelAsync_RejectsModelWithInvalidChecksum()
    {
        string tempDir = CreateTempDirectory();
        string sourceModelPath = Path.Join(tempDir, "source.onnx");
        byte[] sourceBytes = [9, 8, 7, 6];
        await File.WriteAllBytesAsync(sourceModelPath, sourceBytes, TestContext.Current.CancellationToken);

        string modelCacheRoot = Path.Join(tempDir, "model-cache");
        var registrar = new RecordingModelCacheRegistrar();
        DiarizationStageHandler handler = BuildHandler(modelCacheRoot, registrar: registrar);

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            handler.ImportModelAsync(sourceModelPath, TestContext.Current.CancellationToken));

        string expectedPath = ExpectedModelFilePath(modelCacheRoot);
        Assert.False(File.Exists(expectedPath));
        Assert.True(File.Exists(sourceModelPath));
        Assert.Equal(sourceBytes, await File.ReadAllBytesAsync(sourceModelPath, TestContext.Current.CancellationToken));
        Assert.Null(registrar.Record);
    }

    [Fact]
    public async Task ImportModelAsync_PreservesSourceWhenImportPathIsCachePathAndChecksumInvalid()
    {
        string modelCacheRoot = Path.Join(CreateTempDirectory(), "model-cache");
        string modelPath = ExpectedModelFilePath(modelCacheRoot);
        byte[] sourceBytes = [9, 8, 7, 6];
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);
        await File.WriteAllBytesAsync(modelPath, sourceBytes, TestContext.Current.CancellationToken);

        DiarizationStageHandler handler = BuildHandler(modelCacheRoot);

        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            handler.ImportModelAsync(modelPath, TestContext.Current.CancellationToken));

        Assert.True(File.Exists(modelPath));
        Assert.Equal(sourceBytes, await File.ReadAllBytesAsync(modelPath, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task DownloadRequiredModelAsync_AcceptsDownloadWithMatchingChecksum()
    {
        string modelCacheRoot = Path.Join(CreateTempDirectory(), "model-cache");
        var downloader = new WritingModelDownloader();
        var registrar = new RecordingModelCacheRegistrar();
        DiarizationStageHandler handler = BuildHandler(
            modelCacheRoot,
            downloader: downloader,
            registrar: registrar,
            expectedSha256: SortFormerTestFixtures.ExpectedSha256);

        await handler.DownloadRequiredModelAsync(cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(handler.GetRequiredModelStatus().IsAvailable);
        Assert.NotNull(registrar.Record);
        Assert.Equal(SortFormerTestFixtures.ExpectedSha256, registrar.Record.Sha256);
    }

    [Fact]
    public async Task DiarizeAsync_ThrowsRequiredModelNotAvailableException_WhenDownloadFails()
    {
        string modelCacheRoot = Path.Join(CreateTempDirectory(), "model-cache");
        DiarizationStageHandler handler = BuildHandler(modelCacheRoot, downloader: new FailingModelDownloader());

        await Assert.ThrowsAsync<RequiredModelNotAvailableException>(() =>
            handler.DiarizeAsync(
                normalizedAudioPath: "fake.wav",
                durationSeconds: 10.0,
                speechRegions: [],
                cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task IntegrityCheck_DeletesAndRedownloads_WhenModelCorrupt()
    {
        string modelCacheRoot = Path.Join(CreateTempDirectory(), "model-cache");
        string modelPath = ExpectedModelFilePath(modelCacheRoot);

        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);
        await File.WriteAllBytesAsync(modelPath, [0xDE, 0xAD, 0xBE, 0xEF], TestContext.Current.CancellationToken);

        var downloader = new WritingModelDownloader();
        DiarizationStageHandler handler = BuildHandler(modelCacheRoot, downloader: downloader);

        await Assert.ThrowsAsync<RequiredModelNotAvailableException>(() =>
            handler.DownloadRequiredModelAsync(cancellationToken: TestContext.Current.CancellationToken));

        Assert.NotNull(downloader.DestinationPath);
        Assert.False(File.Exists(modelPath));
    }

    [Fact]
    public async Task ImportModelAsync_uses_injected_fingerprint_for_integrity_check()
    {
        string root = CreateTempDirectory();
        string sourcePath = Path.Join(root, "source.onnx");
        await File.WriteAllBytesAsync(sourcePath, [1, 2, 3, 4], TestContext.Current.CancellationToken);
        string expected = new string('b', 64);
        var fingerprints = new FakeFileFingerprintService(new FileFingerprint(expected, 4, DateTimeOffset.UnixEpoch));
        DiarizationStageHandler handler = new DiarizationStageHandler(
            new FakeDiarizationEngine(),
            new NoOpModelDownloader(),
            modelCacheRoot: Path.Join(root, "model-cache"),
            expectedSha256: expected,
            fingerprintService: fingerprints);

        await handler.ImportModelAsync(sourcePath, TestContext.Current.CancellationToken);

        Assert.Contains(ExpectedModelFilePath(Path.Join(root, "model-cache")), fingerprints.RequestedPaths);
    }

    [Fact]
    public async Task DiarizeAsync_WhenModelReady_ReturnsSpeakerTurns()
    {
        var engine = new FakeDiarizationEngine();
        DiarizationStageHandler handler = await BuildHandlerWithVerifiedModelAsync(engine);

        IReadOnlyList<DiarizedSpeakerTurn> turns = await handler.DiarizeAsync(
            normalizedAudioPath: "fake.wav",
            durationSeconds: 12.0,
            speechRegions: [new SpeechRegion(0, 0.0, 5.0)],
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(2, turns.Count);
        Assert.Equal(1, engine.CallCount);
    }

    [Fact]
    public async Task DiarizeAsync_WhenEngineFails_PropagatesException()
    {
        var engine = new FakeDiarizationEngine
        {
            ExceptionToThrow = new InvalidOperationException("Diarization engine failed.")
        };
        DiarizationStageHandler handler = await BuildHandlerWithVerifiedModelAsync(engine);

        InvalidOperationException ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            handler.DiarizeAsync(
                normalizedAudioPath: "fake.wav",
                durationSeconds: 12.0,
                speechRegions: [],
                cancellationToken: TestContext.Current.CancellationToken));

        Assert.Contains("Diarization engine failed", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task DiarizeAsync_WhenCanceled_PropagatesCancellation()
    {
        using var cts = new CancellationTokenSource();
        var engine = new FakeDiarizationEngine
        {
            ExceptionToThrow = new OperationCanceledException(cts.Token)
        };
        DiarizationStageHandler handler = await BuildHandlerWithVerifiedModelAsync(engine);

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            handler.DiarizeAsync(
                normalizedAudioPath: "fake.wav",
                durationSeconds: 12.0,
                speechRegions: [],
                cancellationToken: CancellationToken.None));
    }

    private static async Task<DiarizationStageHandler> BuildHandlerWithVerifiedModelAsync(
        FakeDiarizationEngine? engine = null)
    {
        string modelCacheRoot = Path.Join(CreateTempDirectory(), "model-cache");
        string modelPath = ExpectedModelFilePath(modelCacheRoot);
        string modelRootPath = Path.GetFullPath(Path.Join(
            modelCacheRoot,
            "cgus",
            "diar_streaming_sortformer_4spk-v2.1-onnx"));
        Directory.CreateDirectory(Path.GetDirectoryName(modelPath)!);
        await File.WriteAllBytesAsync(modelPath, SortFormerTestFixtures.ModelBytes);

        var registrar = new RecordingModelCacheRegistrar();
        await registrar.RegisterAsync(
            new LocalModelCacheRecord(
                "cgus/diar_streaming_sortformer_4spk-v2.1-onnx",
                modelRootPath,
                "2be05a08b477e8a526fd26963802845069c02c7c",
                SortFormerTestFixtures.ExpectedSha256,
                DateTimeOffset.UtcNow),
            CancellationToken.None);

        return new DiarizationStageHandler(
            engine ?? new FakeDiarizationEngine(),
            new NoOpModelDownloader(),
            modelCacheRegistrar: registrar,
            modelCacheRoot: modelCacheRoot,
            expectedSha256: SortFormerTestFixtures.ExpectedSha256);
    }

    // -----------------------------------------------------------------------
    // Private nested test doubles
    // -----------------------------------------------------------------------

    /// <summary>Records call parameters and writes <c>[1, 2, 3, 4]</c> bytes to the destination.</summary>
    private sealed class WritingModelDownloader : IModelDownloaderContract
    {
        public string? ModelId { get; private set; }
        public string? FileName { get; private set; }
        public string? DestinationPath { get; private set; }

        public Task<bool> DownloadAsync(
            string modelId,
            string fileName,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default,
            string? revision = null)
        {
            ModelId = modelId;
            FileName = fileName;
            DestinationPath = destinationPath;

            string? directory = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
            }

            File.WriteAllBytes(destinationPath, SortFormerTestFixtures.ModelBytes);
            return Task.FromResult(true);
        }

        public Task<bool> DownloadUriAsync(
            Uri sourceUri,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default)
        {
            DestinationPath = destinationPath;

            string? directory = Path.GetDirectoryName(destinationPath);
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
            }

            File.WriteAllBytes(destinationPath, [4, 3, 2, 1]);
            return Task.FromResult(true);
        }

        public Task<bool> VerifyHashAsync(
            string filePath,
            string expectedHash,
            CancellationToken cancellationToken = default) =>
            Task.FromResult(true);
    }

    /// <summary>Returns <c>false</c> without writing any file.</summary>
    private sealed class NoOpModelDownloader : IModelDownloaderContract
    {
        public Task<bool> DownloadAsync(
            string modelId,
            string fileName,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default,
            string? revision = null) =>
            Task.FromResult(false);

        public Task<bool> DownloadUriAsync(
            Uri sourceUri,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default) =>
            Task.FromResult(false);

        public Task<bool> VerifyHashAsync(
            string filePath,
            string expectedHash,
            CancellationToken cancellationToken = default) =>
            Task.FromResult(false);
    }

    /// <summary>Always fails: returns <c>false</c> and does not write the file.</summary>
    private sealed class FailingModelDownloader : IModelDownloaderContract
    {
        public Task<bool> DownloadAsync(
            string modelId,
            string fileName,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default,
            string? revision = null) =>
            Task.FromResult(false);

        public Task<bool> DownloadUriAsync(
            Uri sourceUri,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default) =>
            Task.FromResult(false);

        public Task<bool> VerifyHashAsync(
            string filePath,
            string expectedHash,
            CancellationToken cancellationToken = default) =>
            Task.FromResult(false);
    }
}
