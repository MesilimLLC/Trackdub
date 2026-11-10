using Trackdub.Contracts;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Licensing;
using Trackdub.Domain;
using Trackdub.Infrastructure.Settings;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Runtime.Planning;
using System.Diagnostics.CodeAnalysis;

namespace Trackdub.Composition.Runtime;

public sealed class RuntimeModelBootstrapService(
    IRuntimePlanner runtimePlanner,
    BundledModelManifestRegistry manifestRegistry,
    IModelDownloaderContract modelDownloader,
    IModelCacheRegistrar modelCacheRegistrar,
    IFileFingerprintService fingerprintService,
    TrackdubStoragePaths storagePaths,
    IApplicationLogger? logger = null,
    IModelHashVerifier? hashVerifier = null,
    IRuntimePlanningPreferences? runtimePlanningPreferences = null)
    : IRuntimeModelBootstrapService
{
    private const string HushEngineFamily = "hush-dialogue";
    private const string HushNativeRuntimeFileName = "weya_nc.dll";
    private readonly IRuntimePlanner runtimePlanner = runtimePlanner ?? throw new ArgumentNullException(nameof(runtimePlanner));
    private readonly BundledModelManifestRegistry manifestRegistry = manifestRegistry ?? throw new ArgumentNullException(nameof(manifestRegistry));
    private readonly IModelDownloaderContract modelDownloader = modelDownloader ?? throw new ArgumentNullException(nameof(modelDownloader));
    private readonly IModelCacheRegistrar modelCacheRegistrar = modelCacheRegistrar ?? throw new ArgumentNullException(nameof(modelCacheRegistrar));
    private readonly IFileFingerprintService fingerprintService = fingerprintService ?? throw new ArgumentNullException(nameof(fingerprintService));
    private readonly TrackdubStoragePaths storagePaths = storagePaths ?? throw new ArgumentNullException(nameof(storagePaths));
    private readonly IModelHashVerifier hashVerifier = hashVerifier ?? new ModelHashVerifier(fingerprintService);

    public async Task<RequiredRuntimeModelStatus?> GetRequiredModelStatusAsync(
        RuntimeModelRequest request,
        CancellationToken cancellationToken = default)
    {
        StageRuntimePlan plan = await PlanAsync(request, cancellationToken).ConfigureAwait(false);
        if (!TryResolveEntry(plan, out BundledModelManifestEntry? entry))
        {
            return CreateUnresolvedStatus(request, plan);
        }

        string selectedEntryRelativePath = ResolveSelectedEntryRelativePath(entry, plan);
        string modelRootPath = ResolveModelRootPath(plan, entry, selectedEntryRelativePath);
        IReadOnlyList<string> requiredFiles = ResolveRequiredDownloadFiles(entry, plan, selectedEntryRelativePath);
        IReadOnlyList<string> missingFiles = ResolveMissingFiles(modelRootPath, requiredFiles);
        bool isAvailable = plan.IsRunnable() && missingFiles.Count == 0;

        if (isAvailable)
        {
            return null;
        }

        string failureReason = ResolveFailureReason(plan, entry, missingFiles);
        bool canAutoDownload = !plan.IsLocalOptimizedVariant &&
            plan.Status is not StageRuntimePlanStatus.Blocked &&
            (missingFiles.Count == 0 ||
             (entry.RedistributionAllowed &&
               missingFiles.Any(relativePath => IsAutoDownloadableFile(entry, relativePath))));
        bool canImportSingleFile = !plan.IsLocalOptimizedVariant && requiredFiles.Count == 1;

        return CreateStatus(
            request,
            plan,
            entry,
            selectedEntryRelativePath,
            modelRootPath,
            isAvailable: false,
            canAutoDownload,
            canImportSingleFile,
            failureReason,
            ResolveStatusRelativePath(entry, selectedEntryRelativePath, missingFiles));
    }

    public async Task<RequiredRuntimeModelStatus> DownloadRequiredModelAsync(
        RuntimeModelRequest request,
        IProgress<ModelDownloadProgress>? downloadProgress = null,
        CancellationToken cancellationToken = default)
    {
        StageRuntimePlan plan = await PlanAsync(request, cancellationToken).ConfigureAwait(false);
        if (!TryResolveEntry(plan, out BundledModelManifestEntry? entry))
        {
            return CreateUnresolvedStatus(request, plan);
        }

        if (plan.Status is StageRuntimePlanStatus.Blocked)
        {
            return CreateStatus(
                request,
                plan,
                entry,
                ResolveSelectedEntryRelativePath(entry, plan),
                ResolveModelRootPath(plan, entry, ResolveSelectedEntryRelativePath(entry, plan)),
                isAvailable: false,
                canAutoDownload: false,
                canImportSingleFile: false,
                plan.Fallback?.Detail ?? "The runtime planner blocked this model.");
        }

        string selectedEntryRelativePath = ResolveSelectedEntryRelativePath(entry, plan);
        string modelRootPath = ResolveModelRootPath(plan, entry, selectedEntryRelativePath);
        IReadOnlyList<string> requiredFiles = ResolveRequiredDownloadFiles(entry, plan, selectedEntryRelativePath);
        IReadOnlyList<string> missingFiles = ResolveMissingFiles(modelRootPath, requiredFiles);

        if (plan.IsLocalOptimizedVariant)
        {
            return CreateStatus(
                request,
                plan,
                entry,
                selectedEntryRelativePath,
                modelRootPath,
                isAvailable: false,
                canAutoDownload: false,
                canImportSingleFile: false,
                ResolveFailureReason(plan, entry, missingFiles),
                ResolveStatusRelativePath(entry, selectedEntryRelativePath, missingFiles));
        }

        Directory.CreateDirectory(modelRootPath);
        if (missingFiles.Count > 0 && missingFiles.All(relativePath => !IsAutoDownloadableFile(entry, relativePath)))
        {
            return CreateStatus(
                request,
                plan,
                entry,
                selectedEntryRelativePath,
                modelRootPath,
                isAvailable: false,
                canAutoDownload: false,
                canImportSingleFile: requiredFiles.Count == 1,
                ResolveFailureReason(plan, entry, missingFiles),
                ResolveStatusRelativePath(entry, selectedEntryRelativePath, missingFiles));
        }

        int maxHashRetries = 2;
        HashVerificationResult? hashResult = null;
        string? hashFailureRelativePath = null;
        string selectedEntryPath = ResolveModelFilePath(modelRootPath, selectedEntryRelativePath);

        for (int attempt = 0; attempt <= maxHashRetries; attempt++)
        {
            missingFiles = ResolveMissingFiles(modelRootPath, requiredFiles);
            IReadOnlyList<string> downloadableMissingFiles = missingFiles
                .Where(relativePath => IsAutoDownloadableFile(entry, relativePath))
                .ToArray();

            foreach (string relativePath in downloadableMissingFiles)
            {
                cancellationToken.ThrowIfCancellationRequested();
                string destinationPath = ResolveModelFilePath(modelRootPath, relativePath);
                logger?.LogInformation($"Downloading model file '{relativePath}' for '{entry.ModelId}'.");
                bool downloaded = await DownloadRequiredFileAsync(
                    entry,
                    relativePath,
                    destinationPath,
                    downloadProgress,
                    cancellationToken).ConfigureAwait(false);

                if (!downloaded || !File.Exists(destinationPath))
                {
                    return CreateStatus(
                        request,
                        plan,
                        entry,
                        selectedEntryRelativePath,
                        modelRootPath,
                        isAvailable: false,
                        canAutoDownload: true,
                        canImportSingleFile: requiredFiles.Count == 1,
                        $"Failed to download '{relativePath}' from '{ResolveDownloadSourceDescription(entry, relativePath)}'.",
                        relativePath);
                }
            }

            if (!File.Exists(selectedEntryPath))
            {
                return CreateStatus(
                    request,
                    plan,
                    entry,
                    selectedEntryRelativePath,
                    modelRootPath,
                    isAvailable: false,
                    canAutoDownload: false,
                    canImportSingleFile: requiredFiles.Count == 1,
                    $"The downloaded model package did not include '{selectedEntryRelativePath}'.");
            }

            (hashFailureRelativePath, hashResult) = await VerifyRequiredFilesAsync(
                entry,
                modelRootPath,
                requiredFiles,
                selectedEntryRelativePath,
                cancellationToken).ConfigureAwait(false);

            if (hashResult.IsValid)
            {
                break;
            }

            logger?.LogError(
                $"Hash verification failed for '{entry.ModelId}': {hashResult.FailureReason} " +
                $"(expected={hashResult.ExpectedSha256}, actual={hashResult.ActualSha256}). " +
                $"Attempt {attempt + 1} of {maxHashRetries + 1}");
            DeleteIfDownloadedFile(ResolveModelFilePath(modelRootPath, hashFailureRelativePath ?? selectedEntryRelativePath));

            if (attempt == maxHashRetries)
            {
                return CreateStatus(
                    request,
                    plan,
                    entry,
                    selectedEntryRelativePath,
                    modelRootPath,
                    isAvailable: false,
                    canAutoDownload: true,
                    canImportSingleFile: requiredFiles.Count == 1,
                    $"Hash verification failed for '{hashFailureRelativePath ?? selectedEntryRelativePath}' after {maxHashRetries + 1} attempts: {hashResult.FailureReason}",
                    hashFailureRelativePath ?? selectedEntryRelativePath);
            }
        }

        if (hashResult is { WasVerified: true } verifiedHashResult)
        {
            logger?.LogInformation($"Hash verified for '{entry.ModelId}': {verifiedHashResult.ExpectedSha256}");
        }

        FileFingerprint fingerprint = await fingerprintService
            .ComputeAsync(selectedEntryPath, cancellationToken)
            .ConfigureAwait(false);
        await modelCacheRegistrar.RegisterAsync(
            new LocalModelCacheRecord(
                entry.ModelId,
                modelRootPath,
                string.IsNullOrWhiteSpace(entry.Revision) ? "main" : entry.Revision,
                ModelDownloadManifestFiles.ResolveCacheIdentitySha256(entry, fingerprint.Sha256),
                DateTimeOffset.UtcNow),
            cancellationToken).ConfigureAwait(false);

        RequiredRuntimeModelStatus? remaining = await GetRequiredModelStatusAsync(request, cancellationToken).ConfigureAwait(false);
        return remaining ?? CreateStatus(
            request,
            plan,
            entry,
            selectedEntryRelativePath,
            modelRootPath,
            isAvailable: true,
            canAutoDownload: true,
            canImportSingleFile: requiredFiles.Count == 1,
            failureReason: null);
    }

    public async Task<RequiredRuntimeModelStatus> ImportRequiredModelAsync(
        RuntimeModelRequest request,
        string sourceModelPath,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceModelPath);

        StageRuntimePlan plan = await PlanAsync(request, cancellationToken).ConfigureAwait(false);
        if (!TryResolveEntry(plan, out BundledModelManifestEntry? entry))
        {
            return CreateUnresolvedStatus(request, plan);
        }

        string selectedEntryRelativePath = ResolveSelectedEntryRelativePath(entry, plan);
        if (plan.IsLocalOptimizedVariant)
        {
            string localVariantRootPath = ResolveModelRootPath(plan, entry, selectedEntryRelativePath);
            return CreateStatus(
                request,
                plan,
                entry,
                selectedEntryRelativePath,
                localVariantRootPath,
                isAvailable: false,
                canAutoDownload: false,
                canImportSingleFile: false,
                $"Selected optimized variant '{plan.Variant ?? "unknown"}' is local-only. Re-optimize the model or clear the variant selection.");
        }

        IReadOnlyList<string> requiredFiles = ResolveRequiredDownloadFiles(entry, plan, selectedEntryRelativePath);
        if (requiredFiles.Count != 1)
        {
            return CreateStatus(
                request,
                plan,
                entry,
                selectedEntryRelativePath,
                ResolveModelRootPath(entry),
                isAvailable: false,
                canAutoDownload: true,
                canImportSingleFile: false,
                "This model requires multiple package files and cannot be imported as a single ONNX file.");
        }

        string sourcePath = Path.GetFullPath(sourceModelPath);
        if (!File.Exists(sourcePath))
        {
            throw new FileNotFoundException("The selected model file was not found.", sourcePath);
        }

        string modelRootPath = ResolveModelRootPath(entry);
        string destinationPath = ResolveModelFilePath(modelRootPath, selectedEntryRelativePath);
        Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);
        if (!sourcePath.Equals(destinationPath, StringComparison.OrdinalIgnoreCase))
        {
            await using FileStream sourceStream = File.Open(sourcePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            await using FileStream destinationStream = File.Open(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None);
            await sourceStream.CopyToAsync(destinationStream, cancellationToken).ConfigureAwait(false);
        }

        HashVerificationResult importHashResult = await hashVerifier
            .VerifyAsync(entry.Sha256, destinationPath, cancellationToken)
            .ConfigureAwait(false);
        if (!importHashResult.IsValid)
        {
            logger?.LogError(
                $"Hash verification failed for imported '{entry.ModelId}': {importHashResult.FailureReason} " +
                $"(expected={importHashResult.ExpectedSha256}, actual={importHashResult.ActualSha256})");
            return CreateStatus(
                request,
                plan,
                entry,
                selectedEntryRelativePath,
                ResolveModelRootPath(entry),
                isAvailable: false,
                canAutoDownload: true,
                canImportSingleFile: true,
                $"Hash verification failed for the imported file: {importHashResult.FailureReason}");
        }

        if (importHashResult.WasVerified)
        {
            logger?.LogInformation($"Hash verified for imported '{entry.ModelId}': {importHashResult.ExpectedSha256}");
        }

        FileFingerprint fingerprint = await fingerprintService
            .ComputeAsync(destinationPath, cancellationToken)
            .ConfigureAwait(false);
        await modelCacheRegistrar.RegisterAsync(
            new LocalModelCacheRecord(
                entry.ModelId,
                modelRootPath,
                string.IsNullOrWhiteSpace(entry.Revision) ? "manual-import" : entry.Revision,
                ModelDownloadManifestFiles.ResolveCacheIdentitySha256(entry, fingerprint.Sha256),
                DateTimeOffset.UtcNow),
            cancellationToken).ConfigureAwait(false);

        RequiredRuntimeModelStatus? remaining = await GetRequiredModelStatusAsync(request, cancellationToken).ConfigureAwait(false);
        return remaining ?? CreateStatus(
            request,
            plan,
            entry,
            selectedEntryRelativePath,
            modelRootPath,
            isAvailable: true,
            canAutoDownload: true,
            canImportSingleFile: true,
            failureReason: null);
    }

    public async Task<RequiredRuntimeModelStatus?> GetManifestCompanionModelStatusAsync(
        string manifestAlias,
        RuntimeStage owningStage,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(manifestAlias);

        if (!manifestRegistry.TryResolve(manifestAlias, out BundledModelManifestResolution? resolution) ||
            resolution is null)
        {
            return CreateUnresolvedCompanionStatus(manifestAlias, owningStage, "Trackdub could not resolve the model manifest entry.");
        }

        BundledModelManifestEntry entry = resolution.Entry;
        string selectedEntryRelativePath = ResolveSelectedEntryRelativePath(entry, selectedVariant: null);
        string modelRootPath = ResolveModelRootPath(entry);
        IReadOnlyList<string> requiredFiles = ResolveRequiredDownloadFiles(entry, selectedVariant: null, selectedEntryRelativePath);
        IReadOnlyList<string> missingFiles = ResolveMissingFiles(modelRootPath, requiredFiles);
        bool isAvailable = missingFiles.Count == 0;

        if (isAvailable)
        {
            return null;
        }

        string failureReason = missingFiles.Count > 0 &&
            missingFiles.All(relativePath => !IsAutoDownloadableFile(entry, relativePath))
            ? "The required model files are missing, and no downloadable source is configured for this build."
            : "The model cache is missing support files required by this runtime.";

        bool canAutoDownload = entry.RedistributionAllowed &&
            (missingFiles.Count == 0 ||
             missingFiles.Any(relativePath => IsAutoDownloadableFile(entry, relativePath)));
        bool canImportSingleFile = requiredFiles.Count == 1;

        return CreateCompanionStatus(
            owningStage,
            entry,
            selectedEntryRelativePath,
            modelRootPath,
            isAvailable: false,
            canAutoDownload,
            canImportSingleFile,
            failureReason,
            ResolveStatusRelativePath(entry, selectedEntryRelativePath, missingFiles));
    }

    public async Task<RequiredRuntimeModelStatus> DownloadManifestCompanionModelAsync(
        string manifestAlias,
        RuntimeStage owningStage,
        IProgress<ModelDownloadProgress>? downloadProgress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(manifestAlias);

        if (!manifestRegistry.TryResolve(manifestAlias, out BundledModelManifestResolution? resolution) ||
            resolution is null)
        {
            return CreateUnresolvedCompanionStatus(manifestAlias, owningStage, "Trackdub could not resolve the model manifest entry.");
        }

        BundledModelManifestEntry entry = resolution.Entry;
        string selectedEntryRelativePath = ResolveSelectedEntryRelativePath(entry, selectedVariant: null);
        string modelRootPath = ResolveModelRootPath(entry);
        IReadOnlyList<string> requiredFiles = ResolveRequiredDownloadFiles(entry, selectedVariant: null, selectedEntryRelativePath);

        Directory.CreateDirectory(modelRootPath);
        IReadOnlyList<string> missingFiles = ResolveMissingFiles(modelRootPath, requiredFiles);
        if (missingFiles.Count > 0 && missingFiles.All(relativePath => !IsAutoDownloadableFile(entry, relativePath)))
        {
            return CreateCompanionStatus(
                owningStage,
                entry,
                selectedEntryRelativePath,
                modelRootPath,
                isAvailable: false,
                canAutoDownload: false,
                canImportSingleFile: requiredFiles.Count == 1,
                "The required model files are missing, and no downloadable source is configured for this build.",
                ResolveStatusRelativePath(entry, selectedEntryRelativePath, missingFiles));
        }

        int maxHashRetries = 2;
        HashVerificationResult? hashResult = null;
        string? hashFailureRelativePath = null;
        string selectedEntryPath = ResolveModelFilePath(modelRootPath, selectedEntryRelativePath);

        for (int attempt = 0; attempt <= maxHashRetries; attempt++)
        {
            missingFiles = ResolveMissingFiles(modelRootPath, requiredFiles);
            IReadOnlyList<string> downloadableMissingFiles = missingFiles
                .Where(relativePath => IsAutoDownloadableFile(entry, relativePath))
                .ToArray();

            foreach (string relativePath in downloadableMissingFiles)
            {
                cancellationToken.ThrowIfCancellationRequested();
                string destinationPath = ResolveModelFilePath(modelRootPath, relativePath);
                logger?.LogInformation($"Downloading companion model file '{relativePath}' for '{entry.ModelId}'.");
                bool downloaded = await DownloadRequiredFileAsync(
                    entry,
                    relativePath,
                    destinationPath,
                    downloadProgress,
                    cancellationToken).ConfigureAwait(false);

                if (!downloaded || !File.Exists(destinationPath))
                {
                    return CreateCompanionStatus(
                        owningStage,
                        entry,
                        selectedEntryRelativePath,
                        modelRootPath,
                        isAvailable: false,
                        canAutoDownload: true,
                        canImportSingleFile: requiredFiles.Count == 1,
                        $"Failed to download '{relativePath}' from '{ResolveDownloadSourceDescription(entry, relativePath)}'.",
                        relativePath);
                }
            }

            if (!File.Exists(selectedEntryPath))
            {
                return CreateCompanionStatus(
                    owningStage,
                    entry,
                    selectedEntryRelativePath,
                    modelRootPath,
                    isAvailable: false,
                    canAutoDownload: false,
                    canImportSingleFile: requiredFiles.Count == 1,
                    $"The downloaded model package did not include '{selectedEntryRelativePath}'.");
            }

            (hashFailureRelativePath, hashResult) = await VerifyRequiredFilesAsync(
                entry,
                modelRootPath,
                requiredFiles,
                selectedEntryRelativePath,
                cancellationToken).ConfigureAwait(false);

            if (hashResult.IsValid)
            {
                break;
            }

            logger?.LogError(
                $"Hash verification failed for companion '{entry.ModelId}': {hashResult.FailureReason} " +
                $"(expected={hashResult.ExpectedSha256}, actual={hashResult.ActualSha256}). " +
                $"Attempt {attempt + 1} of {maxHashRetries + 1}");
            DeleteIfDownloadedFile(ResolveModelFilePath(modelRootPath, hashFailureRelativePath ?? selectedEntryRelativePath));

            if (attempt == maxHashRetries)
            {
                return CreateCompanionStatus(
                    owningStage,
                    entry,
                    selectedEntryRelativePath,
                    modelRootPath,
                    isAvailable: false,
                    canAutoDownload: true,
                    canImportSingleFile: requiredFiles.Count == 1,
                    $"Hash verification failed for '{hashFailureRelativePath ?? selectedEntryRelativePath}' after {maxHashRetries + 1} attempts: {hashResult.FailureReason}",
                    hashFailureRelativePath ?? selectedEntryRelativePath);
            }
        }

        if (hashResult is { WasVerified: true } verifiedHashResult)
        {
            logger?.LogInformation($"Hash verified for companion '{entry.ModelId}': {verifiedHashResult.ExpectedSha256}");
        }

        FileFingerprint fingerprint = await fingerprintService
            .ComputeAsync(selectedEntryPath, cancellationToken)
            .ConfigureAwait(false);
        await modelCacheRegistrar.RegisterAsync(
            new LocalModelCacheRecord(
                entry.ModelId,
                modelRootPath,
                string.IsNullOrWhiteSpace(entry.Revision) ? "main" : entry.Revision,
                ModelDownloadManifestFiles.ResolveCacheIdentitySha256(entry, fingerprint.Sha256),
                DateTimeOffset.UtcNow),
            cancellationToken).ConfigureAwait(false);

        RequiredRuntimeModelStatus? remaining = await GetManifestCompanionModelStatusAsync(
            manifestAlias,
            owningStage,
            cancellationToken).ConfigureAwait(false);
        return remaining ?? CreateCompanionStatus(
            owningStage,
            entry,
            selectedEntryRelativePath,
            modelRootPath,
            isAvailable: true,
            canAutoDownload: true,
            canImportSingleFile: requiredFiles.Count == 1,
            failureReason: null);
    }

    private async Task<StageRuntimePlan> PlanAsync(
        RuntimeModelRequest request,
        CancellationToken cancellationToken)
    {
        StageRuntimePlanningRequest planningRequest =
            await StageRuntimePlanningRequestFactory.ApplyPreferredModelTierAsync(
                new StageRuntimePlanningRequest(
                    request.Stage,
                    request.PreferredModelAlias,
                    SourceLanguage: request.SourceLanguage,
                    TargetLanguage: request.TargetLanguage,
                    RequirePreferredModelAlias: request.RequirePreferredModelAlias,
                    PreferredExecutionProvider: request.PreferredExecutionProvider,
                    RequirePreferredExecutionProvider: request.RequirePreferredExecutionProvider,
                    PreferredModelVariantAlias: request.PreferredModelVariantAlias),
                runtimePlanningPreferences,
                cancellationToken)
                .ConfigureAwait(false);

        return await runtimePlanner
            .PlanAsync(planningRequest, cancellationToken)
            .ConfigureAwait(false);
    }

    private bool TryResolveEntry(
        StageRuntimePlan plan,
        [NotNullWhen(true)]
        out BundledModelManifestEntry? entry)
    {
        entry = null;
        if (!string.IsNullOrWhiteSpace(plan.ModelId))
        {
            entry = manifestRegistry.Entries.FirstOrDefault(candidate =>
                candidate.ModelId.Equals(plan.ModelId, StringComparison.OrdinalIgnoreCase));
            if (entry is not null)
            {
                return true;
            }
        }

        if (!string.IsNullOrWhiteSpace(plan.ModelAlias) &&
            manifestRegistry.TryResolve(plan.ModelAlias, out BundledModelManifestResolution? resolution) &&
            resolution is not null)
        {
            entry = resolution.Entry;
            return true;
        }

        return false;
    }

    private RequiredRuntimeModelStatus CreateCompanionStatus(
        RuntimeStage owningStage,
        BundledModelManifestEntry entry,
        string selectedEntryRelativePath,
        string modelRootPath,
        bool isAvailable,
        bool canAutoDownload,
        bool canImportSingleFile,
        string? failureReason,
        string? statusRelativePath = null)
    {
        string expectedRelativePath = NormalizeRelativePath(statusRelativePath ?? selectedEntryRelativePath);
        string stageDisplayName = string.IsNullOrWhiteSpace(entry.DisplayName) ? entry.ModelId : entry.DisplayName;
        return new RequiredRuntimeModelStatus(
            owningStage,
            stageDisplayName,
            entry.ModelId,
            entry.Aliases.FirstOrDefault(),
            Variant: null,
            expectedRelativePath,
            ResolveModelFilePath(modelRootPath, expectedRelativePath),
            entry.SourceUrl,
            entry.License,
            isAvailable,
            canAutoDownload,
            canImportSingleFile,
            entry.RequiresAttribution,
            entry.RequiresUserConsent,
            $"Trackdub needs '{entry.ModelId}' before {ResolveStageDisplayName(owningStage).ToLowerInvariant()} can run. This is cached locally after setup.",
            failureReason);
    }

    private static RequiredRuntimeModelStatus CreateUnresolvedCompanionStatus(
        string manifestAlias,
        RuntimeStage owningStage,
        string detail) =>
        new(
            owningStage,
            ResolveStageDisplayName(owningStage),
            manifestAlias,
            manifestAlias,
            Variant: null,
            string.Empty,
            string.Empty,
            string.Empty,
            "unknown",
            IsAvailable: false,
            CanAutoDownload: false,
            CanImportSingleFile: false,
            RequiresAttribution: false,
            RequiresUserConsent: false,
            "Trackdub could not resolve the model manifest entry for this runtime request.",
            detail);

    private RequiredRuntimeModelStatus CreateUnresolvedStatus(
        RuntimeModelRequest request,
        StageRuntimePlan plan)
    {
        BundledModelManifestEntry? stageCandidate = FindStageCandidateEntry(request);
        if (stageCandidate is not null)
        {
            return new RequiredRuntimeModelStatus(
                request.Stage,
                ResolveStageDisplayName(request.Stage),
                stageCandidate.ModelId,
                stageCandidate.Aliases.FirstOrDefault(),
                plan.Variant,
                string.Empty,
                string.Empty,
                stageCandidate.SourceUrl,
                stageCandidate.License,
                IsAvailable: false,
                CanAutoDownload: false,
                CanImportSingleFile: false,
                stageCandidate.RequiresAttribution,
                stageCandidate.RequiresUserConsent,
                $"Trackdub cannot use '{stageCandidate.ModelId}' for {ResolveStageDisplayName(request.Stage).ToLowerInvariant()} with the current runtime settings.",
                plan.Fallback?.Detail);
        }

        return new RequiredRuntimeModelStatus(
            request.Stage,
            ResolveStageDisplayName(request.Stage),
            plan.ModelId ?? request.PreferredModelAlias ?? request.Stage.ToString(),
            plan.ModelAlias,
            plan.Variant,
            string.Empty,
            string.Empty,
            string.Empty,
            "unknown",
            IsAvailable: false,
            CanAutoDownload: false,
            CanImportSingleFile: false,
            RequiresAttribution: false,
            RequiresUserConsent: false,
            "Trackdub could not resolve the model manifest entry for this runtime request.",
            plan.Fallback?.Detail);
    }

    private BundledModelManifestEntry? FindStageCandidateEntry(RuntimeModelRequest request)
    {
        if (!string.IsNullOrWhiteSpace(request.PreferredModelAlias) &&
            manifestRegistry.TryResolve(request.PreferredModelAlias, out BundledModelManifestResolution? resolution) &&
            resolution is not null)
        {
            return resolution.Entry;
        }

        string task = ResolveManifestTaskName(request.Stage);
        return manifestRegistry.Entries.FirstOrDefault(entry =>
            entry.Task.Equals(task, StringComparison.OrdinalIgnoreCase));
    }

    private RequiredRuntimeModelStatus CreateStatus(
        RuntimeModelRequest request,
        StageRuntimePlan plan,
        BundledModelManifestEntry entry,
        string selectedEntryRelativePath,
        string modelRootPath,
        bool isAvailable,
        bool canAutoDownload,
        bool canImportSingleFile,
        string? failureReason,
        string? statusRelativePath = null)
    {
        string expectedRelativePath = NormalizeRelativePath(statusRelativePath ?? selectedEntryRelativePath);
        return new RequiredRuntimeModelStatus(
            request.Stage,
            ResolveStageDisplayName(request.Stage),
            entry.ModelId,
            plan.ModelAlias,
            plan.Variant,
            expectedRelativePath,
            ResolveModelFilePath(modelRootPath, expectedRelativePath),
            entry.SourceUrl,
            entry.License,
            isAvailable,
            canAutoDownload,
            canImportSingleFile,
            entry.RequiresAttribution,
            entry.RequiresUserConsent,
            $"Trackdub needs '{entry.ModelId}' before {ResolveStageDisplayName(request.Stage).ToLowerInvariant()} can run. This is cached locally after setup.",
            failureReason);
    }

    private string ResolveModelRootPath(
        StageRuntimePlan plan,
        BundledModelManifestEntry entry,
        string selectedEntryRelativePath)
    {
        if (plan.IsLocalOptimizedVariant && !string.IsNullOrWhiteSpace(plan.ModelRootPath))
        {
            return Path.GetFullPath(plan.ModelRootPath);
        }

        if (plan.ModelEntryPath is not null)
        {
            string selectedEntryPath = Path.GetFullPath(plan.ModelEntryPath);
            string relativePath = NormalizeRelativePath(selectedEntryRelativePath);
            string? rootPath = selectedEntryPath;
            foreach (string _ in relativePath.Split('/'))
            {
                if (rootPath is null)
                {
                    break;
                }

                rootPath = Path.GetDirectoryName(rootPath);
            }

            if (!string.IsNullOrWhiteSpace(rootPath))
            {
                return Path.GetFullPath(rootPath);
            }
        }

        return ResolveModelRootPath(entry);
    }

    private string ResolveModelRootPath(BundledModelManifestEntry entry)
    {
        string path = Path.GetFullPath(storagePaths.ModelCacheDirectory);
        foreach (string part in entry.ModelId.Split(new[] { '/', '\\' }, StringSplitOptions.RemoveEmptyEntries))
        {
            path = Path.Join(path, part);
        }

        return path;
    }

    private static IReadOnlyList<string> ResolveRequiredDownloadFiles(
        BundledModelManifestEntry entry,
        StageRuntimePlan plan,
        string selectedEntryRelativePath)
    {
        if (plan.IsLocalOptimizedVariant)
        {
            IReadOnlyList<string> requiredFiles = plan.RequiredModelRelativePaths.Count > 0
                ? plan.RequiredModelRelativePaths
                : [selectedEntryRelativePath];
            return requiredFiles
                .Select(NormalizeRelativePath)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
        }

        return ResolveRequiredDownloadFiles(entry, plan.Variant, selectedEntryRelativePath);
    }

    private static IReadOnlyList<string> ResolveRequiredDownloadFiles(
        BundledModelManifestEntry entry,
        string? selectedVariant,
        string selectedEntryRelativePath) =>
        ModelDownloadManifestFiles.ResolveRequiredFilesForVariant(
            entry,
            selectedVariant ?? string.Empty,
            selectedEntryRelativePath);

    private static IReadOnlyList<string> ResolveMissingFiles(
        string modelRootPath,
        IReadOnlyList<string> requiredFiles) =>
        requiredFiles
            .Where(relativePath => !File.Exists(ResolveModelFilePath(modelRootPath, relativePath)))
            .ToArray();

    private async Task<(string RelativePath, HashVerificationResult Result)> VerifyRequiredFilesAsync(
        BundledModelManifestEntry entry,
        string modelRootPath,
        IReadOnlyList<string> requiredFiles,
        string selectedEntryRelativePath,
        CancellationToken cancellationToken)
    {
        // Hashing required model files streams them end to end; attribute that read pass to a
        // named phase so cold-start evidence shows how much of model provisioning is hashing.
        using IDisposable hashPhase = BenchmarkPhaseCapture.Start("model-hash-verify");
        string selectedRelativePath = NormalizeRelativePath(selectedEntryRelativePath);
        string benchmarkRelativePath = NormalizeRelativePath(
            Path.GetRelativePath(entry.RootDirectory, entry.DefaultBenchmarkEntryPath));
        var candidates = new List<RequiredModelFileHashVerifier.Candidate>();
        foreach (string relativePath in requiredFiles)
        {
            string normalizedRelativePath = NormalizeRelativePath(relativePath);
            string filePath = ResolveModelFilePath(modelRootPath, normalizedRelativePath);
            if (!File.Exists(filePath))
            {
                return (normalizedRelativePath, new HashVerificationResult(false, false, null, null, "Required model file is missing."));
            }

            string? expectedHash = ModelDownloadManifestFiles.ResolveExpectedSha256(
                entry,
                normalizedRelativePath,
                benchmarkRelativePath);
            if (!string.IsNullOrWhiteSpace(expectedHash))
            {
                candidates.Add(new(normalizedRelativePath, filePath, expectedHash));
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

        HashVerificationResult selectedEntryResult = results
            .Where(item => item.Candidate.RelativePath.Equals(selectedRelativePath, StringComparison.OrdinalIgnoreCase))
            .Select(item => item.Result)
            .FirstOrDefault() ?? new(true, false, null, null, "No hash verification was required.");
        HashVerificationResult lastResult = results.Count == 0
            ? new(true, false, null, null, "No hash verification was required.")
            : results[^1].Result;
        return (selectedRelativePath, selectedEntryResult.WasVerified ? selectedEntryResult : lastResult);
    }

    private static string ResolveStatusRelativePath(
        BundledModelManifestEntry entry,
        string selectedEntryRelativePath,
        IReadOnlyList<string> missingFiles)
    {
        if (missingFiles.FirstOrDefault(relativePath => IsHushNativeRuntimeFile(entry, relativePath)) is { } nativeRuntimePath)
        {
            return nativeRuntimePath;
        }

        return missingFiles.Count > 0 ? missingFiles[0] : selectedEntryRelativePath;
    }

    private static string ResolveFailureReason(
        StageRuntimePlan plan,
        BundledModelManifestEntry entry,
        IReadOnlyList<string> missingFiles)
    {
        if (plan.IsLocalOptimizedVariant)
        {
            if (plan.Status is StageRuntimePlanStatus.Blocked)
            {
                return plan.Fallback?.Detail ?? "The selected optimized model variant is blocked.";
            }

            if (missingFiles.Count > 0)
            {
                return $"Selected optimized variant '{plan.Variant ?? "unknown"}' is missing required local file '{missingFiles[0]}'. Re-optimize the model or clear the variant selection.";
            }

            return plan.Fallback?.Detail ??
                   $"Selected optimized variant '{plan.Variant ?? "unknown"}' is unavailable. Re-optimize the model or clear the variant selection.";
        }

        if (plan.Status is StageRuntimePlanStatus.Blocked)
        {
            return plan.Fallback?.Detail ?? "The runtime planner blocked this model.";
        }

        if (missingFiles.FirstOrDefault(relativePath => IsHushNativeRuntimeFile(entry, relativePath)) is { } nativeRuntimePath)
        {
            return $"The Hush native runtime file '{nativeRuntimePath}' is missing. Download now will install the native runtime and model bundle into the local cache.";
        }

        if (missingFiles.Count > 0 &&
            missingFiles.All(relativePath => !IsAutoDownloadableFile(entry, relativePath)))
        {
            return "The required model files are missing, and no downloadable source is configured for this build.";
        }

        return plan.IsRunnable()
            ? "The model cache is missing support files required by this runtime."
            : plan.Fallback?.Detail ?? "The model is not cached locally.";
    }

    private async Task<bool> DownloadRequiredFileAsync(
        BundledModelManifestEntry entry,
        string relativePath,
        string destinationPath,
        IProgress<ModelDownloadProgress>? downloadProgress,
        CancellationToken cancellationToken)
    {
        using IDisposable downloadPhase = BenchmarkPhaseCapture.Start("model-download");
        if (TryResolveExternalDownloadSource(entry, relativePath, out Uri? sourceUri))
        {
            return await modelDownloader
                .DownloadUriAsync(
                    sourceUri,
                    destinationPath,
                    downloadProgress,
                    cancellationToken)
                .ConfigureAwait(false);
        }

        if (CanDownloadFromHuggingFace(entry))
        {
            return await modelDownloader
                .DownloadAsync(
                    entry.ModelId,
                    relativePath,
                    destinationPath,
                    downloadProgress,
                    cancellationToken,
                    string.IsNullOrWhiteSpace(entry.Revision) ? null : entry.Revision)
                .ConfigureAwait(false);
        }

        return false;
    }

    private static void DeleteIfDownloadedFile(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
            // Best-effort cleanup; the bad file may linger but the cache record is not registered.
        }
    }

    private static string ResolveDownloadSourceDescription(
        BundledModelManifestEntry entry,
        string relativePath) =>
        ModelDownloadManifestFiles.ResolveDownloadSourceDescription(entry, relativePath);

    private static bool IsAutoDownloadableFile(
        BundledModelManifestEntry entry,
        string relativePath) =>
        entry.RedistributionAllowed &&
        ModelDownloadManifestFiles.CanAutoDownloadFile(entry, relativePath);

    private static bool CanDownloadFromHuggingFace(BundledModelManifestEntry entry) =>
        ModelDownloadManifestFiles.CanDownloadFromHuggingFace(entry);

    private static bool TryResolveExternalDownloadSource(
        BundledModelManifestEntry entry,
        string relativePath,
        [NotNullWhen(true)]
        out Uri? sourceUri) =>
        ModelDownloadManifestFiles.TryResolveExternalDownloadSource(entry, relativePath, out sourceUri);

    private static bool IsHushNativeRuntimeFile(
        BundledModelManifestEntry entry,
        string relativePath) =>
        string.Equals(entry.EngineFamily, HushEngineFamily, StringComparison.OrdinalIgnoreCase) &&
        string.Equals(Path.GetFileName(relativePath), HushNativeRuntimeFileName, StringComparison.OrdinalIgnoreCase);

    private static void AddFiles(
        ICollection<string> files,
        IEnumerable<string> paths)
    {
        foreach (string path in paths)
        {
            if (!string.IsNullOrWhiteSpace(path))
            {
                files.Add(path);
            }
        }
    }

    private static string ResolveSelectedEntryRelativePath(
        BundledModelManifestEntry entry,
        StageRuntimePlan plan)
    {
        if (plan.IsLocalOptimizedVariant && !string.IsNullOrWhiteSpace(plan.ModelEntryRelativePath))
        {
            return NormalizeRelativePath(plan.ModelEntryRelativePath);
        }

        return ResolveSelectedEntryRelativePath(entry, plan.Variant);
    }

    private static string ResolveSelectedEntryRelativePath(
        BundledModelManifestEntry entry,
        string? selectedVariant)
    {
        if (!string.IsNullOrWhiteSpace(selectedVariant))
        {
            BundledModelManifestVariant? variant = entry.Variants.FirstOrDefault(candidate =>
                candidate.Alias.Equals(selectedVariant, StringComparison.OrdinalIgnoreCase));
            if (variant is not null)
            {
                return NormalizeRelativePath(Path.GetRelativePath(entry.RootDirectory, variant.EntryPath));
            }
        }

        return NormalizeRelativePath(Path.GetRelativePath(entry.RootDirectory, entry.DefaultBenchmarkEntryPath));
    }

    private static string ResolveModelFilePath(
        string modelRootPath,
        string relativePath)
    {
        string normalizedRelativePath = NormalizeRelativePath(relativePath);
        string destinationPath = Path.GetFullPath(Path.Join(
            modelRootPath,
            Path.Join(normalizedRelativePath.Split('/'))));
        string rootWithSeparator = Path.GetFullPath(modelRootPath)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) +
            Path.DirectorySeparatorChar;
        if (!destinationPath.StartsWith(rootWithSeparator, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException($"Model path '{relativePath}' escapes the model cache root.");
        }

        return destinationPath;
    }

    private static string NormalizeRelativePath(string path)
    {
        string normalized = path.Replace('\\', '/').Trim('/');
        if (string.IsNullOrWhiteSpace(normalized) ||
            Path.IsPathRooted(normalized) ||
            normalized.Split('/').Any(part => part is "." or ".." || string.IsNullOrWhiteSpace(part)))
        {
            throw new InvalidOperationException($"Model manifest download path '{path}' must be a safe relative path.");
        }

        return normalized;
    }

    private static string ResolveStageDisplayName(RuntimeStage stage) =>
        stage switch
        {
            RuntimeStage.Asr => "Transcription",
            RuntimeStage.Translation => "Translation",
            RuntimeStage.Tts => "Text-to-speech",
            RuntimeStage.Diarization => "Speaker detection",
            RuntimeStage.Vad => "Voice activity",
            RuntimeStage.Separation => "Dialogue isolation",
            RuntimeStage.LipSync => "Lip-sync alignment",
            RuntimeStage.LipSynthesis => "Video lip synthesis",
            _ => stage.ToString()
        };

    private static string ResolveManifestTaskName(RuntimeStage stage) =>
        stage switch
        {
            RuntimeStage.Asr => "asr",
            RuntimeStage.Translation => "translation",
            RuntimeStage.Tts => "tts",
            RuntimeStage.Diarization => "diarization",
            RuntimeStage.Vad => "vad",
            RuntimeStage.Separation => "separation",
            RuntimeStage.LipSync => "forced-alignment",
            RuntimeStage.LipSynthesis => "lip-synthesis",
            _ => stage.ToString()
        };
}
