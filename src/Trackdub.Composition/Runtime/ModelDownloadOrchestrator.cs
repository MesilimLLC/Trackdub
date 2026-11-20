using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using Trackdub.Application.Pipeline;
using Trackdub.Composition.Tts;
using Trackdub.Contracts;
using Trackdub.Contracts.Licensing;
using Trackdub.Domain;
using Trackdub.Infrastructure.Persistence.Repositories;
using Trackdub.Infrastructure.Settings;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.Runtime;

public sealed class ModelDownloadOrchestrator(
    BundledModelManifestRegistry manifestRegistry,
    LocalModelCacheRecordStore cacheStore,
    IModelDownloaderContract downloader,
    TrackdubStoragePaths storagePaths,
    IModelHashVerifier? hashVerifier = null,
    IApplicationLogger? logger = null,
    IStudioSettingsService? settingsService = null,
    IEngineAutoWarmQueue? engineAutoWarmQueue = null,
    KokoroVoiceCatalogHolder? voiceCatalogHolder = null)
    : IModelDownloadOrchestrator, IModelCacheVerifier, IDisposable
{
    private readonly BundledModelManifestRegistry manifestRegistry = manifestRegistry ?? throw new ArgumentNullException(nameof(manifestRegistry));
    private readonly LocalModelCacheRecordStore cacheStore = cacheStore ?? throw new ArgumentNullException(nameof(cacheStore));
    private readonly IModelDownloaderContract downloader = downloader ?? throw new ArgumentNullException(nameof(downloader));
    private readonly IModelHashVerifier hashVerifier = hashVerifier ?? new ModelHashVerifier();
    private readonly TrackdubStoragePaths storagePaths = storagePaths ?? throw new ArgumentNullException(nameof(storagePaths));
    private readonly SimpleObservable<ModelStateChange> stateChanges = new();
    private readonly ConcurrentDictionary<string, SemaphoreSlim> downloadGates = new(StringComparer.OrdinalIgnoreCase);
    private readonly IApplicationLogger? logger = logger;
    private bool disposed;

    public IObservable<ModelStateChange> StateChanges => stateChanges;

    /// <summary>Production wiring probe: the holder passed at construction, if any.</summary>
    internal KokoroVoiceCatalogHolder? VoiceCatalogHolder => voiceCatalogHolder;

    public Task<ModelDownloadResult> DownloadAsync(
        string modelId,
        IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default) =>
        DownloadAsync(modelId, variantAlias: null, progress, cancellationToken);

    public async Task<ModelDownloadResult> DownloadAsync(
        string modelId,
        string? variantAlias,
        IProgress<ModelDownloadProgress>? progress,
        CancellationToken cancellationToken = default)
    {
        ModelDownloadResult? refusal = TryResolveDownloadableEntry(modelId, out BundledModelManifestEntry? resolved);
        if (refusal is not null)
        {
            return refusal;
        }

        BundledModelManifestEntry entry = resolved!;

        // Normalize to the canonical model id so state-change events and cache records
        // are keyed consistently even when the caller passed an alias.
        modelId = entry.ModelId;

        string modelRootDirectory = ModelDownloadPathGuard.ResolveConfiguredModelRootDirectory(
            entry.ModelId,
            storagePaths.ModelCacheDirectory);
        if (!ModelDownloadPathGuard.IsModelRootUnderConfiguredCache(modelRootDirectory, storagePaths.ModelCacheDirectory, out string? rootError))
        {
            logger?.LogError($"Model download failed for '{modelId}': {rootError}");
            return new ModelDownloadResult(modelId, false, ModelCacheState.Missing, rootError);
        }

        SemaphoreSlim gate = downloadGates.GetOrAdd(modelId, static _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            // Re-entrancy/idempotency guard: if a concurrent or prior invocation already
            // installed this model while we waited on the gate, don't re-download it.
            // Also handles manifest-delta: if the model is installed but new files were added
            // to the manifest since first install, fetch only the files absent on disk.
            LocalModelCacheRecord? selectedRecord = await GetCurrentRecordAsync(modelId, cancellationToken).ConfigureAwait(false);
            ModelCacheState gatedState = ModelInventoryService.DetermineState(entry, selectedRecord);
            // Pre-compute required files only in the Installed/Ready branch (needed for the
            // missing-file check). For Missing/Corrupt paths the computation is deferred into
            // the inner try block so that malformed manifest paths throw a caught exception
            // rather than propagating out of the gate semaphore.
            IReadOnlyList<string>? precomputedRequiredFiles = null;
            IReadOnlyList<string>? deltaFiles = null;

            if (gatedState is ModelCacheState.Installed or ModelCacheState.Ready)
            {
                try
                {
                    precomputedRequiredFiles = ModelDownloadManifestFiles.ResolveRequiredFiles(entry, variantAlias);
                }
                catch (Exception ex) when (ex is InvalidOperationException or ArgumentException)
                {
                    logger?.LogError($"Model download failed for '{modelId}': {ex.Message}", ex);
                    return new ModelDownloadResult(modelId, false, gatedState, ex.Message);
                }

                string installedRootDirectory = ResolveVerificationModelRootDirectory(entry, selectedRecord);
                IReadOnlyList<string> missingFiles = ResolveMissingRequiredFiles(installedRootDirectory, precomputedRequiredFiles);
                if (missingFiles.Count == 0)
                {
                    // Files exist on disk, but the manifest may carry per-file hashes.
                    // Re-verify so a corrupt-but-present model falls through to re-download
                    // rather than being served as Installed.
                    var (failedPath, hashResult) = await VerifyRequiredFilesAsync(
                        entry, installedRootDirectory, precomputedRequiredFiles, cancellationToken).ConfigureAwait(false);

                    if (hashResult.WasVerified && !hashResult.IsValid)
                    {
                        await SetModelIntegrityStateAsync(
                            modelId, integrityFailed: true, entry, installedRootDirectory, cancellationToken).ConfigureAwait(false);
                        logger?.LogWarning(
                            $"Model '{modelId}' file '{failedPath}' failed hash verification (expected {hashResult.ExpectedSha256}, actual {hashResult.ActualSha256}). Marking corrupt and re-downloading.");
                        // Fall through to the download path below.
                        // Reset deltaFiles to null so all required files are re-downloaded, not just the empty delta.
                        deltaFiles = null;
                    }
                    else
                    {
                        return new ModelDownloadResult(modelId, true, gatedState, null);
                    }
                }
                else
                {
                    // Manifest has grown since first install (e.g. new voice packs added).
                    // Only fetch the files that are absent on disk.
                    // Downloads only modify the configured cache; an external installation's
                    // existing files cannot satisfy a partial install in that destination.
                    if (LocalModelCacheRecordStore.RootsEqual(installedRootDirectory, modelRootDirectory))
                        deltaFiles = missingFiles;
                }
            }

            // Cached verification remains available offline, but no UI or headless
            // caller may start a network download while offline mode is enabled.
            if (settingsService is not null &&
                (await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false)).OfflineMode)
            {
                return new ModelDownloadResult(modelId, false, gatedState,
                    "Offline mode prevents downloading model files. Install the model while online first.");
            }

            EmitStateChange(modelId, gatedState, ModelCacheState.Downloading);

            try
            {
                Directory.CreateDirectory(modelRootDirectory);

                IReadOnlyList<string> allRequiredFiles = precomputedRequiredFiles ?? ModelDownloadManifestFiles.ResolveRequiredFiles(entry, variantAlias);
                IReadOnlyList<string> filesToDownload = deltaFiles ?? allRequiredFiles;
                int fileCount = filesToDownload.Count;
                int fileIndex = 0;

                foreach (string downloadFile in filesToDownload)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    if (!ModelDownloadPathGuard.TryGetSecureDownloadDestination(
                            modelRootDirectory,
                            downloadFile,
                            storagePaths.ModelCacheDirectory,
                            out string destinationPath,
                            out string? pathError))
                    {
                        return deltaFiles is not null
                            ? await FailDeltaAsync(modelId, pathError ?? "Invalid download path.", entry, modelRootDirectory, cancellationToken).ConfigureAwait(false)
                            : Fail(modelId, pathError ?? "Invalid download path.", gatedState);
                    }

                    string? destinationDir = Path.GetDirectoryName(destinationPath);
                    if (destinationDir is not null)
                    {
                        Directory.CreateDirectory(destinationDir);
                    }

                    if (!ModelDownloadManifestFiles.CanAutoDownloadFile(entry, downloadFile))
                    {
                        return deltaFiles is not null
                            ? await FailDeltaAsync(modelId, $"No downloadable source configured for '{downloadFile}'.", entry, modelRootDirectory, cancellationToken).ConfigureAwait(false)
                            : Fail(modelId, $"No downloadable source configured for '{downloadFile}'.", gatedState);
                    }

                    // Map this file's 0-100% onto the model's overall progress so the UI bar advances
                    // monotonically across all files instead of resetting per file.
                    IProgress<ModelDownloadProgress>? fileProgress = progress is null
                        ? null
                        : new AggregateDownloadProgress(progress, fileIndex, fileCount);

                    bool downloaded = await DownloadFileAsync(
                        entry,
                        downloadFile,
                        destinationPath,
                        fileProgress,
                        cancellationToken).ConfigureAwait(false);

                    if (!downloaded)
                    {
                        string downloadFailureReason = $"Download failed for '{downloadFile}' from '{ModelDownloadManifestFiles.ResolveDownloadSourceDescription(entry, downloadFile)}'.";
                        return deltaFiles is not null
                            ? await FailDeltaAsync(modelId, downloadFailureReason, entry, modelRootDirectory, cancellationToken).ConfigureAwait(false)
                            : Fail(modelId, downloadFailureReason, gatedState);
                    }

                    fileIndex++;
                }

                (string hashRelativePath, HashVerificationResult hashResult) = await VerifyRequiredFilesAsync(
                    entry,
                    modelRootDirectory,
                    allRequiredFiles,
                    cancellationToken).ConfigureAwait(false);

                if (!hashResult.IsValid)
                {
                    string hashFailureReason = $"Hash verification failed for '{hashRelativePath}': {hashResult.FailureReason}";
                    logger?.LogError($"Model download failed for '{modelId}': {hashFailureReason}");
                    EmitStateChange(modelId, ModelCacheState.Downloading, ModelCacheState.Corrupt, hashFailureReason);
                    await SetModelIntegrityStateAsync(entry.ModelId, integrityFailed: true, entry, modelRootDirectory, cancellationToken)
                        .ConfigureAwait(false);
                    return new ModelDownloadResult(modelId, false, ModelCacheState.Corrupt, hashFailureReason);
                }

                await RegisterCacheRecordAsync(entry, modelRootDirectory, hashResult, cancellationToken).ConfigureAwait(false);
                // Sidecar caches are keyed by model root, so a fresh install of the same model id
                // can differ from whatever was cached for that path (e.g. a mel-bin count read
                // from a config file that did not exist yet). Invalidate before emitting Installed, because observers run
                // synchronously and may start inference, and before warming, so every read sees the
                // installed files, not stale state.
                InvalidateSidecarCaches(modelRootDirectory);
                await RefreshVoiceCatalogIfKokoroAsync(entry, cancellationToken).ConfigureAwait(false);
                EmitStateChange(modelId, ModelCacheState.Downloading, ModelCacheState.Installed);
                // Trigger engine warming here, on the install path itself, rather than from a
                // subscriber type the DI graph may never construct: this method only runs because
                // an install actually happened, so the request cannot be silently skipped.
                engineAutoWarmQueue?.RequestWarmAfterInstall(modelId, CancellationToken.None);
                return new ModelDownloadResult(modelId, true, ModelCacheState.Installed, null);
            }
            catch (OperationCanceledException)
            {
                logger?.LogWarning($"Model download cancelled for '{modelId}'.");
                if (deltaFiles is not null)
                {
                    // Delta was in progress because required files were already absent on disk.
                    // Cancellation does not restore those files, so the model remains incomplete.
                    try
                    {
                        await SetModelIntegrityStateAsync(entry.ModelId, integrityFailed: true, entry, modelRootDirectory, CancellationToken.None)
                            .ConfigureAwait(false);
                    }
                    catch (Exception persistEx)
                    {
                        logger?.LogError($"Failed to persist integrity state for '{modelId}': {persistEx.Message}", persistEx);
                    }

                    EmitStateChange(modelId, ModelCacheState.Downloading, ModelCacheState.Corrupt, "Download cancelled with incomplete delta files.");
                }
                else
                {
                    EmitStateChange(modelId, ModelCacheState.Downloading, gatedState);
                }

                string cancelReason = deltaFiles is not null
                    ? "Download cancelled with incomplete delta files."
                    : "Download cancelled.";
                ModelCacheState cancelState = deltaFiles is not null
                    ? ModelCacheState.Corrupt
                    : gatedState;
                return new ModelDownloadResult(modelId, false, cancelState, cancelReason, Cancelled: true);
            }
            catch (Exception ex)
            {
                logger?.LogError($"Model download failed for '{modelId}': {ex.Message}", ex);
                if (deltaFiles is not null)
                {
                    try
                    {
                        await SetModelIntegrityStateAsync(entry.ModelId, integrityFailed: true, entry, modelRootDirectory, CancellationToken.None)
                            .ConfigureAwait(false);
                    }
                    catch (Exception persistEx)
                    {
                        logger?.LogError($"Failed to persist integrity state for '{modelId}': {persistEx.Message}", persistEx);
                    }

                    EmitStateChange(modelId, ModelCacheState.Downloading, ModelCacheState.Corrupt, ex.Message);
                    return new ModelDownloadResult(modelId, false, ModelCacheState.Corrupt, ex.Message);
                }

                EmitStateChange(modelId, ModelCacheState.Downloading, gatedState, ex.Message);
                return new ModelDownloadResult(modelId, false, gatedState, ex.Message);
            }
        }
        finally
        {
            gate.Release();
        }
    }

    public async Task<ModelDownloadResult> RepairAsync(
        string modelId,
        IProgress<ModelDownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        BundledModelManifestEntry? entry = FindEntry(modelId);
        if (entry is null)
        {
            return new ModelDownloadResult(modelId, false, ModelCacheState.Missing, $"Model '{modelId}' not found in manifest.");
        }

        // DownloadAsync handles all cases: delta fetch (installed + new manifest files),
        // full re-download (corrupt or missing), and the no-op (installed + complete on disk).
        return await DownloadAsync(entry.ModelId, progress, cancellationToken).ConfigureAwait(false);
    }

    public async Task<bool> UninstallAsync(string modelId, CancellationToken cancellationToken = default)
    {
        BundledModelManifestEntry? entry = FindEntry(modelId);
        if (entry is null)
        {
            return true;
        }

        ModelCacheState previousState = await GetCurrentStateAsync(modelId, cancellationToken).ConfigureAwait(false);
        await DeleteModelFilesAsync(modelId, entry, cancellationToken).ConfigureAwait(false);
        await RemoveCacheRecordAsync(modelId, cancellationToken).ConfigureAwait(false);
        await RefreshVoiceCatalogIfKokoroAsync(entry, cancellationToken).ConfigureAwait(false);

        if (previousState != ModelCacheState.Missing)
        {
            EmitStateChange(modelId, previousState, ModelCacheState.Missing);
        }

        return true;
    }

    public async Task<ModelVerificationResult> VerifyAsync(string modelId, CancellationToken cancellationToken = default)
    {
        BundledModelManifestEntry? entry = FindEntry(modelId);
        if (entry is null)
        {
            return new ModelVerificationResult(modelId, ModelCacheState.Missing, ModelCacheState.Missing, false, "Model not in manifest.");
        }

        LocalModelCacheRecord? record = await GetCurrentRecordAsync(modelId, cancellationToken).ConfigureAwait(false);
        ModelCacheState currentState = ModelInventoryService.DetermineState(entry, record);
        string modelRootDirectory = ResolveVerificationModelRootDirectory(entry, record);

        IReadOnlyList<string> requiredFiles = ModelDownloadManifestFiles.ResolveRequiredFiles(entry);
        IReadOnlyList<string> missingFiles = ResolveMissingRequiredFiles(modelRootDirectory, requiredFiles);
        if (currentState is ModelCacheState.Missing && missingFiles.Count > 0)
        {
            return new ModelVerificationResult(modelId, ModelCacheState.Missing, ModelCacheState.Missing, false, null);
        }

        if (missingFiles.Count > 0)
        {
            await SetModelIntegrityStateAsync(modelId, integrityFailed: true, entry, modelRootDirectory, cancellationToken)
                .ConfigureAwait(false);
            EmitStateChange(modelId, currentState, ModelCacheState.Corrupt);
            return new ModelVerificationResult(modelId, currentState, ModelCacheState.Corrupt, false,
                $"Model file '{missingFiles[0]}' not found on disk.");
        }

        (string hashRelativePath, HashVerificationResult hashResult) = await VerifyRequiredFilesAsync(
            entry,
            modelRootDirectory,
            requiredFiles,
            cancellationToken).ConfigureAwait(false);

        if (hashResult.IsValid)
        {
            await RegisterCacheRecordAsync(entry, modelRootDirectory, hashResult, cancellationToken).ConfigureAwait(false);
            ModelCacheState newState = record is null || currentState is ModelCacheState.Corrupt
                ? ModelCacheState.Installed
                : currentState;
            if (newState != currentState)
            {
                EmitStateChange(modelId, currentState, newState);
            }

            return new ModelVerificationResult(modelId, currentState, newState, true, null);
        }

        await SetModelIntegrityStateAsync(modelId, integrityFailed: true, entry, modelRootDirectory, cancellationToken)
            .ConfigureAwait(false);
        EmitStateChange(modelId, currentState, ModelCacheState.Corrupt);
        return new ModelVerificationResult(modelId, currentState, ModelCacheState.Corrupt, false,
            $"Hash verification failed for '{hashRelativePath}': {hashResult.FailureReason}");
    }

    public async Task<IReadOnlyList<ModelVerificationResult>> VerifyAllAsync(
        IProgress<int>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var results = new List<ModelVerificationResult>();
        int total = manifestRegistry.Entries.Count;

        for (int i = 0; i < total; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            results.Add(await VerifyAsync(manifestRegistry.Entries[i].ModelId, cancellationToken).ConfigureAwait(false));
            progress?.Report((i + 1) * 100 / Math.Max(total, 1));
        }

        return results;
    }

    private IReadOnlyList<string> ResolveMissingRequiredFiles(
        string modelRootDirectory,
        IReadOnlyList<string> requiredFiles)
    {
        string verificationAllowedRoot = ModelDownloadPathGuard.IsModelRootUnderConfiguredCache(
            modelRootDirectory, storagePaths.ModelCacheDirectory, out _)
            ? storagePaths.ModelCacheDirectory
            : modelRootDirectory;
        var missing = new List<string>();
        foreach (string relativePath in requiredFiles)
        {
            if (!ModelDownloadPathGuard.TryGetSecureDownloadDestination(
                    modelRootDirectory,
                    relativePath,
                    verificationAllowedRoot,
                    out string destinationPath,
                    out string? pathError))
            {
                missing.Add(pathError ?? relativePath);
                continue;
            }

            if (!File.Exists(destinationPath))
            {
                missing.Add(NormalizeRelativePath(relativePath));
            }
        }

        return missing;
    }

    /// <summary>
    /// Finds the manifest entry for <paramref name="modelId"/> or returns the refusal: not in the manifest, or not cleared for
    /// commercial use (MODEL_LICENSE_POLICY.md, accepted-risk exception).
    /// </summary>
    private ModelDownloadResult? TryResolveDownloadableEntry(string modelId, out BundledModelManifestEntry? entry)
    {
        entry = FindEntry(modelId);
        string? reason = entry is null
            ? $"Model '{modelId}' not found in manifest."
            : ModelDownloadManifestFiles.IsCommerciallyCleared(entry) ? null : ModelDownloadManifestFiles.DescribeGateRefusal(entry);
        if (reason is null)
        {
            return null;
        }

        logger?.LogError($"Model download failed for '{modelId}': {reason}");
        return new ModelDownloadResult(entry?.ModelId ?? modelId, false, ModelCacheState.Missing, reason);
    }

    private async Task<(string RelativePath, HashVerificationResult Result)> VerifyRequiredFilesAsync(
        BundledModelManifestEntry entry,
        string modelRootDirectory,
        IReadOnlyList<string> requiredFiles,
        CancellationToken cancellationToken)
    {
        string benchmarkRelativePath = NormalizeRelativePath(
            Path.GetRelativePath(entry.RootDirectory, entry.DefaultBenchmarkEntryPath));
        var candidates = new List<RequiredModelFileHashVerifier.Candidate>();
        string verificationAllowedRoot = ModelDownloadPathGuard.IsModelRootUnderConfiguredCache(
            modelRootDirectory, storagePaths.ModelCacheDirectory, out _)
            ? storagePaths.ModelCacheDirectory
            : modelRootDirectory;
        foreach (string relativePath in requiredFiles)
        {
            string normalizedRelativePath = NormalizeRelativePath(relativePath);
            if (!ModelDownloadPathGuard.TryGetSecureDownloadDestination(
                    modelRootDirectory,
                    normalizedRelativePath,
                    verificationAllowedRoot,
                    out string destinationPath,
                    out string? pathError))
            {
                return (normalizedRelativePath, new HashVerificationResult(false, false, null, null, pathError));
            }

            if (!File.Exists(destinationPath))
            {
                return (normalizedRelativePath, new HashVerificationResult(false, false, null, null, "Required model file is missing."));
            }

            string? expectedHash = ModelDownloadManifestFiles.ResolveExpectedSha256(
                entry,
                normalizedRelativePath,
                benchmarkRelativePath);
            if (!string.IsNullOrWhiteSpace(expectedHash))
            {
                candidates.Add(new(normalizedRelativePath, destinationPath, expectedHash));
            }
        }

        IReadOnlyList<(RequiredModelFileHashVerifier.Candidate Candidate, HashVerificationResult Result)> results =
            await RequiredModelFileHashVerifier.VerifyAsync(candidates, hashVerifier, cancellationToken).ConfigureAwait(false);
        (RequiredModelFileHashVerifier.Candidate Candidate, HashVerificationResult Result)? firstFailure =
            results.Where(item => !item.Result.IsValid)
                .Select(item => ((RequiredModelFileHashVerifier.Candidate Candidate, HashVerificationResult Result)?)item)
                .FirstOrDefault();
        if (firstFailure is { } failed)
        {
            return (failed.Candidate.RelativePath, failed.Result);
        }

        HashVerificationResult benchmarkResult = results
            .Where(item => item.Candidate.RelativePath.Equals(benchmarkRelativePath, StringComparison.OrdinalIgnoreCase))
            .Select(item => item.Result)
            .FirstOrDefault() ?? (results.Count == 0
                ? new(true, false, null, null, "No hash verification was required.")
                : results[^1].Result);
        return (benchmarkRelativePath, benchmarkResult);
    }

    /// <summary>
    /// Drops any process-cached sidecar data for a model root that just changed on disk. Caches
    /// are keyed by path only, so an install over an existing path would otherwise keep serving
    /// pre-install state (notably a mel-bin count read before the config file existed).
    /// </summary>
    /// <remarks>Internal so the flush set is testable without driving a full download.</remarks>
    internal static void InvalidateSidecarCaches(string modelRootDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelRootDirectory);
        Trackdub.Inference.Onnx.Whisper.WhisperSidecarCaches.Remove(modelRootDirectory);
        Trackdub.Inference.Onnx.Qwen3Asr.Qwen3AsrSidecarCaches.RemoveForModelRoot(modelRootDirectory);
    }

    /// <summary>
    /// Re-reads the process-wide Kokoro voice snapshot after a Kokoro install/uninstall/repair.
    /// Voice packs ship as manifest files (<c>voices/*.bin</c>) and delta installs add them to an
    /// already-installed model, so without this the singleton keeps serving pre-install state
    /// until the next process restart and the new/removed voices are rejected or omitted.
    /// A refresh failure never fails the install itself: the holder keeps its last snapshot.
    /// </summary>
    private async Task RefreshVoiceCatalogIfKokoroAsync(
        BundledModelManifestEntry entry,
        CancellationToken cancellationToken)
    {
        if (voiceCatalogHolder is null || !IsKokoroModel(entry))
        {
            return;
        }

        try
        {
            await voiceCatalogHolder.RefreshAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            logger?.LogWarning(
                $"Kokoro voice catalog refresh failed after model '{entry.ModelId}' changed; voices may be stale until restart.",
                ex);
        }
    }

    private static bool IsKokoroModel(BundledModelManifestEntry entry) =>
        string.Equals(entry.EngineFamily, "kokoro", StringComparison.OrdinalIgnoreCase) ||
        entry.ModelId.Contains("kokoro", StringComparison.OrdinalIgnoreCase) ||
        entry.Aliases.Any(alias => alias.Contains("kokoro", StringComparison.OrdinalIgnoreCase));

    private static string NormalizeRelativePath(string relativePath) =>
        relativePath.Replace('\\', '/').Trim('/');

    public void Dispose()
    {
        if (!disposed)
        {
            stateChanges.Complete();
            foreach (SemaphoreSlim gate in downloadGates.Values)
            {
                gate.Dispose();
            }
            downloadGates.Clear();
            disposed = true;
        }
    }

    // Resolve by exact model id first, then fall back to a declared alias so callers
    // (CLI `models download`, the UI, verification) can use a short alias like
    // "chatterbox-multilingual" instead of the full "onnx-community/..." id. Exact-id
    // matching takes precedence so an alias can never shadow a real model id.
    private BundledModelManifestEntry? FindEntry(string modelId) =>
        manifestRegistry.Entries.FirstOrDefault(e => e.ModelId.Equals(modelId, StringComparison.OrdinalIgnoreCase))
        ?? manifestRegistry.Entries.FirstOrDefault(
            e => e.Aliases.Any(alias => alias.Equals(modelId, StringComparison.OrdinalIgnoreCase)));

    private async Task<bool> DownloadFileAsync(
        BundledModelManifestEntry entry,
        string relativePath,
        string destinationPath,
        IProgress<ModelDownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        if (ModelDownloadManifestFiles.TryResolveExternalDownloadSource(entry, relativePath, out Uri? sourceUri))
        {
            return await downloader
                .DownloadUriAsync(sourceUri, destinationPath, progress, cancellationToken)
                .ConfigureAwait(false);
        }

        if (ModelDownloadManifestFiles.CanDownloadFromHuggingFace(entry))
        {
            string hfModelId = ModelDownloadManifestFiles.ResolveHuggingFaceModelId(entry);
            return await downloader
                .DownloadAsync(
                    hfModelId,
                    relativePath,
                    destinationPath,
                    progress,
                    cancellationToken,
                    string.IsNullOrWhiteSpace(entry.Revision) ? null : entry.Revision)
                .ConfigureAwait(false);
        }

        return false;
    }

    private string ResolveVerificationModelRootDirectory(
        BundledModelManifestEntry entry,
        LocalModelCacheRecord? record)
    {
        string configuredRoot = ModelDownloadPathGuard.ResolveConfiguredModelRootDirectory(
            entry.ModelId,
            storagePaths.ModelCacheDirectory);

        // Verification is read-only for model files and must inspect the same installation as inventory.
        if (record is not null)
        {
            return Path.GetFullPath(record.RootPath);
        }

        string candidate = configuredRoot;
        if (ModelDownloadPathGuard.IsModelRootUnderConfiguredCache(
                candidate,
                storagePaths.ModelCacheDirectory,
                out _))
        {
            return candidate;
        }

        string bundledRoot = entry.RootDirectory;
        try
        {
            string bundledBenchmarkPath = ModelDownloadPathGuard.ResolveCachedManifestPath(
                entry,
                bundledRoot,
                entry.DefaultBenchmarkEntryPath);
            if (File.Exists(bundledBenchmarkPath))
            {
                return bundledRoot;
            }
        }
        catch (InvalidOperationException)
        {
            // Fall through to configured cache root.
        }

        return configuredRoot;
    }

    private void EmitStateChange(string modelId, ModelCacheState previous, ModelCacheState next, string? failureReason = null) =>
        stateChanges.Emit(new ModelStateChange(modelId, previous, next, DateTimeOffset.UtcNow, failureReason));

    private ModelDownloadResult Fail(string modelId, string reason, ModelCacheState previousState = ModelCacheState.Missing)
    {
        logger?.LogError($"Model download failed for '{modelId}': {reason}");
        EmitStateChange(modelId, ModelCacheState.Downloading, previousState, reason);
        return new ModelDownloadResult(modelId, false, previousState, reason);
    }

    // Delta-specific failure: the model was Installed but a newly required file could not be
    // fetched. Mark the cache record IntegrityFailed=true so subsequent inventory/readiness
    // checks see Corrupt (blocking) rather than Installed (usable).
    private async Task<ModelDownloadResult> FailDeltaAsync(
        string modelId,
        string reason,
        BundledModelManifestEntry entry,
        string modelRootDirectory,
        CancellationToken cancellationToken)
    {
        logger?.LogError($"Model download failed for '{modelId}': {reason}");
        await SetModelIntegrityStateAsync(entry.ModelId, integrityFailed: true, entry, modelRootDirectory, cancellationToken)
            .ConfigureAwait(false);
        EmitStateChange(modelId, ModelCacheState.Downloading, ModelCacheState.Corrupt, reason);
        return new ModelDownloadResult(modelId, false, ModelCacheState.Corrupt, reason);
    }

    // Re-maps a single file's 0-100% progress onto the model's overall progress
    // (file fileIndex of fileCount), so concurrent model downloads each show an
    // independent, monotonically advancing bar instead of a per-file 0->100 reset.
    private sealed class AggregateDownloadProgress(IProgress<ModelDownloadProgress> inner, int fileIndex, int fileCount)
        : IProgress<ModelDownloadProgress>
    {
        public void Report(ModelDownloadProgress value)
        {
            int safeCount = fileCount <= 0 ? 1 : fileCount;
            double fileFraction = value.PercentComplete <= 0
                ? 0d
                : Math.Clamp(value.PercentComplete, 0, 100) / 100d;
            int overall = (int)Math.Clamp(((fileIndex + fileFraction) / safeCount) * 100d, 0d, 100d);
            inner.Report(value with { PercentComplete = overall });
        }
    }

    private Task RegisterCacheRecordAsync(
        BundledModelManifestEntry entry,
        string modelRootDirectory,
        HashVerificationResult hashResult,
        CancellationToken cancellationToken) =>
        cacheStore.MutateAsync(
            records =>
            {
                LocalModelCacheRecord? existing = records.FirstOrDefault(r =>
                    r.ModelId.Equals(entry.ModelId, StringComparison.OrdinalIgnoreCase) &&
                    LocalModelCacheRecordStore.RootsEqual(r.RootPath, modelRootDirectory));
                string identitySha256 = ModelDownloadManifestFiles.ResolveCacheIdentitySha256(
                    entry,
                    hashResult.ActualSha256);
                var updated = records
                    .Where(r =>
                        !(r.ModelId.Equals(entry.ModelId, StringComparison.OrdinalIgnoreCase) &&
                          LocalModelCacheRecordStore.RootsEqual(r.RootPath, modelRootDirectory)))
                    .ToList();
                updated.Add(existing is null
                    ? new LocalModelCacheRecord(
                        entry.ModelId,
                        modelRootDirectory,
                        string.IsNullOrWhiteSpace(entry.Revision) ? "main" : entry.Revision,
                        identitySha256,
                        DateTimeOffset.UtcNow,
                        IntegrityFailed: false)
                    : existing with
                    {
                        Revision = string.IsNullOrWhiteSpace(entry.Revision) ? "main" : entry.Revision,
                        Sha256 = identitySha256,
                        CachedAtUtc = DateTimeOffset.UtcNow,
                        IntegrityFailed = false,
                        Variants = LocalModelCacheRecordStore.CompatibleVariants(
                            existing with { Revision = string.IsNullOrWhiteSpace(entry.Revision) ? "main" : entry.Revision, Sha256 = identitySha256 }, existing.Variants)
                    });
                return updated;
            },
            cancellationToken);

    private Task RemoveCacheRecordAsync(string modelId, CancellationToken cancellationToken) =>
        cacheStore.MutateAsync(
            records => records.Where(r => !r.ModelId.Equals(modelId, StringComparison.OrdinalIgnoreCase)).ToList(),
            cancellationToken);

    private async Task<LocalModelCacheRecord?> GetCurrentRecordAsync(string modelId, CancellationToken cancellationToken)
    {
        IReadOnlyList<LocalModelCacheRecord> records = await cacheStore.LoadAsync(cancellationToken).ConfigureAwait(false);
        BundledModelManifestEntry? entry = FindEntry(modelId);
        return entry is null ? null : ModelInventoryService.SelectBestCacheRecord(entry,
            records.Where(r => r.ModelId.Equals(entry.ModelId, StringComparison.OrdinalIgnoreCase)).ToArray(),
            storagePaths.ModelCacheDirectory);
    }

    private async Task<ModelCacheState> GetCurrentStateAsync(string modelId, CancellationToken cancellationToken)
    {
        BundledModelManifestEntry? entry = FindEntry(modelId);
        if (entry is null)
        {
            return ModelCacheState.Missing;
        }
        LocalModelCacheRecord? record = await GetCurrentRecordAsync(modelId, cancellationToken).ConfigureAwait(false);
        return ModelInventoryService.DetermineState(entry, record);
    }

    private Task SetModelIntegrityStateAsync(
        string modelId,
        bool integrityFailed,
        BundledModelManifestEntry entry,
        string modelRootDirectory,
        CancellationToken cancellationToken) =>
        cacheStore.MutateAsync(
            records =>
            {
                LocalModelCacheRecord? existing = records.FirstOrDefault(r =>
                    r.ModelId.Equals(modelId, StringComparison.OrdinalIgnoreCase) &&
                    LocalModelCacheRecordStore.RootsEqual(r.RootPath, modelRootDirectory));

                if (existing is not null)
                {
                    return records
                        .Select(record =>
                            record.ModelId.Equals(modelId, StringComparison.OrdinalIgnoreCase) &&
                            LocalModelCacheRecordStore.RootsEqual(record.RootPath, modelRootDirectory)
                                ? record with { IntegrityFailed = integrityFailed }
                                : record)
                        .ToList();
                }

                if (!integrityFailed)
                {
                    return records.ToList();
                }

                return records
                    .Append(new LocalModelCacheRecord(
                        entry.ModelId,
                        modelRootDirectory,
                        string.IsNullOrWhiteSpace(entry.Revision) ? "main" : entry.Revision,
                        entry.Sha256,
                        DateTimeOffset.UtcNow,
                        IntegrityFailed: true))
                    .ToList();
            },
            cancellationToken);

    private async Task DeleteModelFilesAsync(
        string modelId,
        BundledModelManifestEntry entry,
        CancellationToken cancellationToken)
    {
        IReadOnlyList<LocalModelCacheRecord> records = await cacheStore.LoadAsync(cancellationToken).ConfigureAwait(false);
        string configuredRoot = ModelDownloadPathGuard.ResolveConfiguredModelRootDirectory(entry.ModelId, storagePaths.ModelCacheDirectory);
        IEnumerable<string> roots = records
            .Where(record => record.ModelId.Equals(modelId, StringComparison.OrdinalIgnoreCase))
            .Select(record => record.RootPath)
            .Append(configuredRoot)
            .Distinct(OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);

        foreach (string root in roots)
        {
            DeleteModelFiles(root);
        }
    }

    private void DeleteModelFiles(string modelRootDirectory)
    {
        if (!ModelDownloadPathGuard.IsModelRootUnderConfiguredCache(modelRootDirectory, storagePaths.ModelCacheDirectory, out _))
        {
            return;
        }

        if (ModelDownloadPathGuard.IsEffectivelyFilesystemRoot(modelRootDirectory))
        {
            return;
        }

        try
        {
            if (Directory.Exists(modelRootDirectory))
            {
                Directory.Delete(modelRootDirectory, recursive: true);
            }
        }
        catch
        {
            // Best-effort cleanup.
        }
    }
}

internal static class ModelDownloadPathGuard
{
    public static string ResolveConfiguredModelRootDirectory(
        string modelId,
        string configuredModelCacheDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        ArgumentException.ThrowIfNullOrWhiteSpace(configuredModelCacheDirectory);

        string root = Path.GetFullPath(configuredModelCacheDirectory);
        foreach (string part in modelId.Split(new[] { '/', '\\' }, StringSplitOptions.RemoveEmptyEntries))
        {
            if (part is "." or "..")
            {
                throw new InvalidOperationException($"Model id '{modelId}' contains an unsafe path segment.");
            }

            root = Path.Join(root, part);
        }

        return Path.GetFullPath(root);
    }

    public static string ResolveManifestRelativePath(
        BundledModelManifestEntry entry,
        string manifestAbsolutePath) =>
        Path.GetRelativePath(entry.RootDirectory, manifestAbsolutePath).Replace('\\', '/');

    public static string ResolveCachedManifestPath(
        BundledModelManifestEntry entry,
        string modelRootDirectory,
        string manifestAbsolutePath)
    {
        string relativePath = ResolveManifestRelativePath(entry, manifestAbsolutePath);
        if (!TryGetSecureDownloadDestination(
                modelRootDirectory,
                relativePath,
                modelRootDirectory,
                out string destinationPath,
                out string? error))
        {
            throw new InvalidOperationException(error ?? "Resolved model path is invalid.");
        }

        return destinationPath;
    }

    public static bool IsEffectivelyFilesystemRoot(string path)
    {
        string full = Path.GetFullPath(path);
        string? root = Path.GetPathRoot(full);
        if (string.IsNullOrEmpty(root))
        {
            return false;
        }

        return string.Equals(
            Path.TrimEndingDirectorySeparator(full),
            Path.TrimEndingDirectorySeparator(root),
            StringComparison.OrdinalIgnoreCase);
    }

    public static bool IsModelRootUnderConfiguredCache(
        string modelRootDirectory,
        string configuredModelCacheDirectory,
        [NotNullWhen(false)] out string? error)
    {
        error = null;
        if (!IsStrictSubpathOrEqual(Path.GetFullPath(modelRootDirectory), Path.GetFullPath(configuredModelCacheDirectory)))
        {
            error = "Model root directory is outside the configured model cache.";
            return false;
        }

        if (IsEffectivelyFilesystemRoot(modelRootDirectory))
        {
            error = "Refusing to operate on a filesystem root path.";
            return false;
        }

        return true;
    }

    public static bool TryGetSecureDownloadDestination(
        string modelRootDirectory,
        string downloadFile,
        string configuredModelCacheDirectory,
        out string destinationPath,
        [NotNullWhen(false)] out string? error)
    {
        destinationPath = string.Empty;
        error = null;

        if (!IsModelRootUnderConfiguredCache(modelRootDirectory, configuredModelCacheDirectory, out string? rootError))
        {
            error = rootError;
            return false;
        }

        if (string.IsNullOrWhiteSpace(downloadFile))
        {
            error = "Download file path is empty.";
            return false;
        }

        if (Path.IsPathRooted(downloadFile))
        {
            error = "Download file path must be relative.";
            return false;
        }

        string normalized = downloadFile.Replace('\\', '/');
        foreach (string segment in normalized.Split('/', StringSplitOptions.RemoveEmptyEntries))
        {
            if (segment is "." or "..")
            {
                error = "Download file path must not contain '.' or '..' segments.";
                return false;
            }
        }

        string rootFull = Path.GetFullPath(modelRootDirectory);
        destinationPath = Path.GetFullPath(Path.Join(rootFull, downloadFile.Replace('/', Path.DirectorySeparatorChar)));

        string rootPrefix = AppendDirectorySeparator(rootFull);
        if (!destinationPath.Equals(rootFull, StringComparison.OrdinalIgnoreCase)
            && !destinationPath.StartsWith(rootPrefix, StringComparison.OrdinalIgnoreCase))
        {
            error = "Resolved download path escapes the model root directory.";
            destinationPath = string.Empty;
            return false;
        }

        return true;
    }

    private static string AppendDirectorySeparator(string path)
    {
        string full = Path.GetFullPath(path);
        char sep = Path.DirectorySeparatorChar;
        return full.EndsWith(sep) ? full : full + sep;
    }

    public static bool IsStrictSubpathOrEqual(string path, string ancestor)
    {
        string ancestorFull = Path.GetFullPath(ancestor);
        string pathFull = Path.GetFullPath(path);
        string prefix = AppendDirectorySeparator(ancestorFull);
        return pathFull.Equals(Path.TrimEndingDirectorySeparator(ancestorFull), StringComparison.OrdinalIgnoreCase)
            || pathFull.StartsWith(prefix, StringComparison.OrdinalIgnoreCase);
    }
}

internal sealed class SimpleObservable<T> : IObservable<T>
{
    private readonly List<IObserver<T>> observers = [];
    private readonly object gate = new();

    public IDisposable Subscribe(IObserver<T> observer)
    {
        lock (gate)
        {
            observers.Add(observer);
        }

        return new Unsubscriber(() =>
        {
            lock (gate)
            {
                observers.Remove(observer);
            }
        });
    }

    public void Emit(T value)
    {
        IObserver<T>[] snapshot;
        lock (gate)
        {
            snapshot = [.. observers];
        }

        foreach (IObserver<T> observer in snapshot)
        {
            observer.OnNext(value);
        }
    }

    public void Complete()
    {
        IObserver<T>[] snapshot;
        lock (gate)
        {
            snapshot = [.. observers];
        }

        foreach (IObserver<T> observer in snapshot)
        {
            observer.OnCompleted();
        }

        lock (gate)
        {
            observers.Clear();
        }
    }

    private sealed class Unsubscriber(Action unsubscribe) : IDisposable
    {
        public void Dispose() => unsubscribe();
    }
}
