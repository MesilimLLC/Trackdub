using Trackdub.Application.Logging;
using Trackdub.Contracts;
using Trackdub.Contracts.Licensing;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using System.Security.Cryptography;

namespace Trackdub.Application.Transcripts;

/// <summary>
/// Handles diarization stage execution with automatic model downloading.
/// </summary>
public sealed class DiarizationStageHandler(
    ISpeakerDiarizationEngine diarizationEngine,
    IModelDownloaderContract modelDownloader,
    IApplicationLogger? logger = null,
    IModelCacheRegistrar? modelCacheRegistrar = null,
    string? modelCacheRoot = null,
    string? expectedSha256 = null,
    IModelCacheRecordLookup? modelCacheLookup = null,
    IFileFingerprintService? fingerprintService = null)
    : IStageRuntimeExecutionReporter
{
    private const string SortFormerModelId = "cgus/diar_streaming_sortformer_4spk-v2.1-onnx";
    private const string SortFormerDownloadModelId = "tonythethompson/diar-streaming-sortformer-4spk-v2.1-onnx";
    private const string SortFormerSourceUrl = "https://huggingface.co/tonythethompson/diar-streaming-sortformer-4spk-v2.1-onnx";
    private const string SortFormerDownloadFileName = "onnx/model.onnx";
    private const string SortFormerRevision = "2be05a08b477e8a526fd26963802845069c02c7c";
    private const string SortFormerModelFileName = "onnx/model.onnx";
    // SHA-256 from bundled-models.manifest.json — authoritative for this model revision.
    private const string SortFormerExpectedSha256 = "82b9c735e1cfc6b36b4ff8a994d9a0573e922d0e80a58a8553b2c58f7aff0c00";
    private const string SortFormerHelpText =
        "Trackdub needs the ONNX export of NVIDIA Streaming SortFormer 4spk v2.1. " +
        "The app can download the ONNX file from Hugging Face and cache it locally.";

    private readonly ISpeakerDiarizationEngine diarizationEngine = diarizationEngine ?? throw new ArgumentNullException(nameof(diarizationEngine));
    private readonly IModelDownloaderContract modelDownloader = modelDownloader ?? throw new ArgumentNullException(nameof(modelDownloader));
    private readonly string sortFormerExpectedSha256 = string.IsNullOrWhiteSpace(expectedSha256)
        ? SortFormerExpectedSha256
        : expectedSha256;

    private readonly string modelCacheRoot = string.IsNullOrWhiteSpace(modelCacheRoot)
        ? Path.Join(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Trackdub",
            "model-cache")
        : Path.GetFullPath(modelCacheRoot);

    private readonly IModelCacheRecordLookup? modelCacheLookup =
        modelCacheLookup ?? modelCacheRegistrar as IModelCacheRecordLookup;

    private readonly IFileFingerprintService? fingerprintService = fingerprintService;

    public StageRuntimeExecutionSummary? LastExecutionSummary =>
        diarizationEngine is IStageRuntimeExecutionReporter reporter
            ? reporter.LastExecutionSummary
            : null;

    public RequiredDiarizationModelStatus GetRequiredModelStatus()
    {
        string modelRootPath = ResolveModelRootPath(modelCacheRoot);
        string modelPath = ResolveModelFilePath(modelRootPath);

        return new RequiredDiarizationModelStatus(
            SortFormerModelId,
            SortFormerModelFileName,
            modelPath,
            SortFormerSourceUrl,
            IsModelReadyForPreflight(modelPath, modelRootPath),
            CanAutoDownload: true,
            RequiresOnnxExport: false,
            SortFormerHelpText);
    }

    private bool IsModelReadyForPreflight(string modelPath, string modelRootPath)
    {
        if (!File.Exists(modelPath))
        {
            return false;
        }

        if (modelCacheLookup is null)
        {
            return false;
        }

        LocalModelCacheRecord? record = modelCacheLookup.Find(SortFormerModelId, modelRootPath);
        return record is not null &&
               string.Equals(record.Sha256, sortFormerExpectedSha256, StringComparison.OrdinalIgnoreCase);
    }

    public Task DownloadRequiredModelAsync(
        IProgress<ModelDownloadProgress>? downloadProgress = null,
        CancellationToken cancellationToken = default) =>
        EnsureModelAvailableAsync(downloadProgress, cancellationToken);

    public async Task ImportModelAsync(
        string sourceModelPath,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceModelPath);

        string sourcePath = Path.GetFullPath(sourceModelPath);
        if (!File.Exists(sourcePath))
        {
            throw new FileNotFoundException("Diarization ONNX model file was not found.", sourcePath);
        }

        if (!sourcePath.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("Diarization model import requires an ONNX file.");
        }

        string modelRootPath = ResolveModelRootPath(modelCacheRoot);
        string modelPath = ResolveModelFilePath(modelRootPath);
        string? modelDirectory = Path.GetDirectoryName(modelPath);
        if (!string.IsNullOrWhiteSpace(modelDirectory))
        {
            Directory.CreateDirectory(modelDirectory);
        }

        if (!sourcePath.Equals(modelPath, StringComparison.OrdinalIgnoreCase))
        {
            await using FileStream sourceStream = File.Open(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            await using FileStream destinationStream = File.Open(modelPath, FileMode.Create, FileAccess.Write, FileShare.None);
            await sourceStream.CopyToAsync(destinationStream, cancellationToken).ConfigureAwait(false);
        }

        bool importPathIsCachePath = sourcePath.Equals(modelPath, StringComparison.OrdinalIgnoreCase);
        if (!await TryAcceptVerifiedModelAsync(
                modelRootPath,
                modelPath,
                cancellationToken,
                deleteInvalidFile: !importPathIsCachePath)
            .ConfigureAwait(false))
        {
            throw new InvalidOperationException(
                "Imported diarization model failed integrity verification for the bundled SortFormer revision.");
        }

        logger?.LogInformation("Diarization model imported successfully: {ModelPath}", modelPath);
    }

    /// <summary>
    /// Diarizes audio with automatic model download if needed.
    /// </summary>
    /// <exception cref="RequiredModelNotAvailableException">Thrown if the model cannot be downloaded.</exception>
    public async Task<IReadOnlyList<DiarizedSpeakerTurn>> DiarizeAsync(
        string normalizedAudioPath,
        double durationSeconds,
        IReadOnlyList<SpeechRegion> speechRegions,
        string? preferredModelAlias = null,
        ExecutionProviderKind? preferredExecutionProvider = null,
        bool requirePreferredExecutionProvider = false,
        string? preferredModelVariantAlias = null,
        IProgress<ModelDownloadProgress>? downloadProgress = null,
        CancellationToken cancellationToken = default)
    {
        // Pre-flight: ensure model is available
        await EnsureModelAvailableAsync(downloadProgress, cancellationToken).ConfigureAwait(false);

        // Model is now guaranteed to be available; run diarization
        return await diarizationEngine.DiarizeAsync(
            new SpeakerDiarizationRequest(
                normalizedAudioPath,
                durationSeconds,
                speechRegions,
                new InferenceRequestOptions(
                    preferredModelAlias,
                    PreferredExecutionProvider: preferredExecutionProvider?.ToString(),
                    RequirePreferredExecutionProvider: requirePreferredExecutionProvider,
                    PreferredModelVariantAlias: preferredModelVariantAlias)),
            cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Checks if the SortFormer model is available locally; downloads if missing.
    /// </summary>
    /// <exception cref="RequiredModelNotAvailableException">Thrown if download fails.</exception>
    private async Task EnsureModelAvailableAsync(
        IProgress<ModelDownloadProgress>? downloadProgress = null,
        CancellationToken cancellationToken = default)
    {
        string modelRootPath = ResolveModelRootPath(modelCacheRoot);
        string modelPath = ResolveModelFilePath(modelRootPath);

        if (File.Exists(modelPath) &&
            await TryAcceptVerifiedModelAsync(modelRootPath, modelPath, cancellationToken).ConfigureAwait(false))
        {
            logger?.LogDebug("Diarization model integrity verified: {ModelPath}", modelPath);
            return;
        }

        // Model missing or corrupt; attempt download
        logger?.LogInformation(
            "Diarization model not found at '{ModelPath}'. Attempting download from Hugging Face.",
            modelPath);

        // Ensure directory exists
        string? modelDirectory = Path.GetDirectoryName(modelPath);
        if (!string.IsNullOrWhiteSpace(modelDirectory))
        {
            Directory.CreateDirectory(modelDirectory);
        }

        try
        {
            bool downloaded = await modelDownloader.DownloadAsync(
                SortFormerDownloadModelId,
                SortFormerDownloadFileName,
                modelPath,
                downloadProgress,
                cancellationToken,
                SortFormerRevision).ConfigureAwait(false);

            if (!downloaded || !File.Exists(modelPath))
            {
                throw new RequiredModelNotAvailableException(
                    SortFormerModelId,
                    modelPath,
                    canAutoDownload: true);
            }

            if (!await TryAcceptVerifiedModelAsync(modelRootPath, modelPath, cancellationToken).ConfigureAwait(false))
            {
                throw new RequiredModelNotAvailableException(
                    SortFormerModelId,
                    modelPath,
                    canAutoDownload: true);
            }

            logger?.LogInformation("Diarization model downloaded successfully: {ModelPath}", modelPath);
        }
        catch (OperationCanceledException)
        {
            logger?.LogWarning("Diarization model download was cancelled.");
            throw;
        }
        catch (HttpRequestException httpEx)
        {
            logger?.LogError("Network error downloading diarization model.", httpEx);
            throw new RequiredModelNotAvailableException(
                SortFormerModelId,
                modelPath,
                canAutoDownload: true);
        }
        catch (Exception ex) when (ex is not RequiredModelNotAvailableException)
        {
            logger?.LogError("Error downloading diarization model.", ex);
            throw new RequiredModelNotAvailableException(
                SortFormerModelId,
                modelPath,
                canAutoDownload: false);
        }
    }

    private async Task<bool> TryAcceptVerifiedModelAsync(
        string modelRootPath,
        string modelPath,
        CancellationToken cancellationToken,
        bool deleteInvalidFile = true)
    {
        if (!File.Exists(modelPath))
        {
            return false;
        }

        string actualSha256 = fingerprintService is null
            ? await ComputeSha256Async(modelPath, cancellationToken).ConfigureAwait(false)
            : (await fingerprintService.ComputeAsync(modelPath, cancellationToken).ConfigureAwait(false)).Sha256;
        if (string.Equals(actualSha256, sortFormerExpectedSha256, StringComparison.OrdinalIgnoreCase))
        {
            await RegisterModelCacheRecordAsync(modelRootPath, actualSha256, cancellationToken).ConfigureAwait(false);
            return true;
        }

        logger?.LogWarning(
            $"Diarization model at '{modelPath}' failed integrity check " +
            $"(expected {sortFormerExpectedSha256}, got {actualSha256}). " +
            (deleteInvalidFile ? "Deleting the invalid file." : "Leaving the file unchanged."));
        if (deleteInvalidFile)
        {
            try
            {
                File.Delete(modelPath);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                logger?.LogWarning(ex, "Could not delete invalid diarization model file");
            }
        }

        return false;
    }

    private async Task RegisterModelCacheRecordAsync(
        string modelRootPath,
        string sha256,
        CancellationToken cancellationToken)
    {
        if (modelCacheRegistrar is null)
        {
            // Production composition always supplies IModelCacheRegistrar; a null registrar means the
            // handler was constructed without one (e.g. in a test). Log a warning so the omission is
            // visible rather than silently skipped.
            logger?.LogWarning("IModelCacheRegistrar was not provided; diarization model cache record will not be registered.");
            return;
        }

        await modelCacheRegistrar.RegisterAsync(
            new LocalModelCacheRecord(
                SortFormerModelId,
                modelRootPath,
                SortFormerRevision,
                sha256,
                DateTimeOffset.UtcNow),
            cancellationToken).ConfigureAwait(false);
    }

    private static async Task<string> ComputeSha256Async(
        string filePath,
        CancellationToken cancellationToken)
    {
        await using FileStream stream = File.OpenRead(filePath);
        byte[] hash = await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static string ResolveModelRootPath(string modelCacheRoot)
    {
        string path = Path.GetFullPath(modelCacheRoot);
        foreach (string part in SortFormerModelId.Split(new[] { '/', '\\' }, StringSplitOptions.RemoveEmptyEntries))
        {
            path = Path.Join(path, part);
        }

        return path;
    }

    private static string ResolveModelFilePath(string modelRootPath)
    {
        string path = modelRootPath;
        foreach (string part in SortFormerModelFileName.Split(new[] { '/', '\\' }, StringSplitOptions.RemoveEmptyEntries))
        {
            path = Path.Join(path, part);
        }

        return path;
    }
}
