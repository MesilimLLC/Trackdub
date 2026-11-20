using System.Text.Json;

using System.Runtime.InteropServices;

using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Infrastructure.Licensing;
using Trackdub.Sdk;

namespace Trackdub.Cli.Handlers;

/// <summary>
/// Operator health checks beyond per-stage readiness.
/// </summary>
internal static class DoctorHandler
{
    public static async Task<int> ExecuteAsync(
        TrackdubSessionFactory factory,
        TextWriter output,
        CancellationToken cancellationToken)
    {
        var checks = new List<DoctorCheckRow>();

        IAppStoragePaths storagePaths = factory.GetRequiredService<IAppStoragePaths>();
        checks.Add(CheckModelCacheWritable(storagePaths));
        checks.Add(CheckHuggingFaceDownloadAcceleration(factory));

        IReadOnlyList<DoctorCheckRow> manifestChecks = await CheckManifestInventoryAsync(
            factory,
            cancellationToken).ConfigureAwait(false);
        checks.AddRange(manifestChecks);

        IFfmpegHealthCheck ffmpegHealthCheck = factory.GetRequiredService<IFfmpegHealthCheck>();
        checks.Add(CheckFfmpeg(ffmpegHealthCheck));
        checks.Add(CheckEspeakNg(factory.GetRequiredService<IEspeakNgHealthCheck>()));

        checks.Add(CheckLogPath(storagePaths));
        checks.Add(CheckAcceleratorBudget(factory));
        checks.Add(CheckEngineCache(factory));
        checks.Add(CheckEngineCaptureHeadroom(factory));
        checks.Add(CheckPlaybackNatives());
        checks.Add(CheckWindowsMlSummary());
        checks.Add(await CheckTensorRtRtxPluginAsync(factory, cancellationToken).ConfigureAwait(false));
        checks.AddRange(await CheckCatalogExecutionProvidersAsync(factory, cancellationToken).ConfigureAwait(false));

        var readinessChecker = new TrackdubPipelineReadinessChecker(factory);
        PipelineReadinessReport readinessReport = await readinessChecker
            .EvaluateDefaultPipelineAsync(cancellationToken: cancellationToken)
            .ConfigureAwait(false);
        checks.Add(BuildPipelineReadinessCheck(readinessReport));

        List<DoctorCheckRow> reportedChecks = ApplyGpuProfile(
            checks,
            Environment.GetEnvironmentVariable(GpuProfileEnvironmentVariable));
        string overallStatus = ResolveOverallStatus(reportedChecks);
        var payload = new DoctorOutput
        {
            Status = overallStatus,
            Checks = reportedChecks,
        };

        string json = JsonSerializer.Serialize(payload, CliJsonOptions.Default);
        await output.WriteLineAsync(json).ConfigureAwait(false);

        return overallStatus == "fail" ? Program.ExitPipelineFailure : Program.ExitSuccess;
    }

    private static DoctorCheckRow CheckAcceleratorBudget(TrackdubSessionFactory factory)
    {
        AcceleratorBudgetInfo info = factory
            .GetRequiredService<IAcceleratorBudgetInfoService>()
            .GetBudgetInfo();

        if (info.OverrideBudgetMb.HasValue)
        {
            return new DoctorCheckRow
            {
                Id = "accelerator-vram-budget",
                Status = "pass",
                Message =
                    $"Session VRAM admission budget is {info.EffectiveBudgetMb} MB " +
                    $"(source: {info.Source}; detected GPU VRAM {info.DetectedVramMb} MB, " +
                    $"auto-scaled default would be {info.DefaultBudgetMb} MB).",
            };
        }

        if (info.DetectedVramMb > 0)
        {
            return new DoctorCheckRow
            {
                Id = "accelerator-vram-budget",
                Status = "pass",
                Message =
                    $"Session VRAM admission budget is {info.EffectiveBudgetMb} MB " +
                    $"(source: {info.Source}; detected GPU VRAM {info.DetectedVramMb} MB).",
            };
        }

        return new DoctorCheckRow
        {
            Id = "accelerator-vram-budget",
            Status = "warn",
            Message =
                $"No GPU VRAM detected; session VRAM admission budget fell back to the " +
                $"{info.EffectiveBudgetMb} MB floor. Large GPU bundles will be refused admission.",
            Remediation =
                "Check GPU passthrough (docker run --gpus all) and driver visibility, or set " +
                "TRACKDUB_SESSION_VRAM_BUDGET_MB explicitly to override detection.",
        };
    }

    private static DoctorCheckRow CheckEngineCache(TrackdubSessionFactory factory)
    {
        IEngineCacheMaintenanceService maintenance = factory.GetRequiredService<IEngineCacheMaintenanceService>();
        EngineCacheDescription description = maintenance.Describe();

        if (!description.DirectoryExists || description.FileCount == 0)
        {
            return new DoctorCheckRow
            {
                Id = "engine-cache",
                Status = "pass",
                Message = description.DirectoryExists
                    ? $"Engine cache directory is empty: {description.CacheDirectory}"
                    : $"Engine cache directory not created yet: {description.CacheDirectory}",
            };
        }

        string sizeLabel = FormatApproximateBytes(description.ApproximateSizeBytes);
        return new DoctorCheckRow
        {
            Id = "engine-cache",
            Status = "warn",
            Message =
                $"Engine cache contains {description.FileCount} file(s) (~{sizeLabel}) at {description.CacheDirectory}.",
            Remediation =
                "After GPU/driver changes or TRT RTX EP bumps: trackdub cache clear engines, then trackdub cache warm",
        };
    }

    /// <summary>
    /// TensorRT-RTX engine capture needs free space roughly twice the size of the largest
    /// captured graph. When it cannot write the artifact, every launch rebuilds the engines
    /// from source: minutes of avoidable cold start that is otherwise invisible until someone
    /// notices slow runs. Report the real headroom so the cause is actionable.
    /// </summary>
    private static DoctorCheckRow CheckEngineCaptureHeadroom(TrackdubSessionFactory factory)
    {
        IAppStoragePaths storagePaths = factory.GetRequiredService<IAppStoragePaths>();
        string modelCacheDirectory = storagePaths.ModelCacheDirectory;

        string? largestModelPath = null;
        long largestModelBytes = 0;
        try
        {
            foreach (string path in Directory.EnumerateFiles(modelCacheDirectory, "*.onnx", SearchOption.AllDirectories))
            {
                long length = new FileInfo(path).Length;
                if (length > largestModelBytes)
                {
                    largestModelBytes = length;
                    largestModelPath = path;
                }
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or DirectoryNotFoundException)
        {
            return new DoctorCheckRow
            {
                Id = "engine-capture",
                Status = "pass",
                Message = $"Engine capture headroom not measurable (model cache unreadable): {ex.Message}",
            };
        }

        if (largestModelPath is null || largestModelBytes < 256L * 1024 * 1024)
        {
            return new DoctorCheckRow
            {
                Id = "engine-capture",
                Status = "pass",
                Message = "No model large enough to benefit from an EP-context engine artifact.",
            };
        }

        long requiredBytes = largestModelBytes * 2;
        long availableBytes;
        try
        {
            availableBytes = new DriveInfo(Path.GetPathRoot(modelCacheDirectory)!).AvailableFreeSpace;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
        {
            return new DoctorCheckRow
            {
                Id = "engine-capture",
                Status = "warn",
                Message = $"Engine capture headroom could not be read for '{modelCacheDirectory}': {ex.Message}",
            };
        }

        string modelLabel = Path.GetFileName(largestModelPath);
        if (availableBytes < requiredBytes)
        {
            return new DoctorCheckRow
            {
                Id = "engine-capture",
                Status = "fail",
                Message =
                    $"Not enough free space to capture TensorRT-RTX engines: '{modelLabel}' needs " +
                    $"{FormatApproximateBytes(requiredBytes)} free but only {FormatApproximateBytes(availableBytes)} is available. " +
                    "Engines will rebuild from source on every launch (minutes of avoidable cold start).",
                Remediation =
                    $"Free disk space on the model cache drive ({Path.GetPathRoot(modelCacheDirectory)}), " +
                    "or move the cache to a larger drive with TRACKDUB_CACHE_ROOT, then run: trackdub cache warm",
            };
        }

        return new DoctorCheckRow
        {
            Id = "engine-capture",
            Status = "pass",
            Message =
                $"Engine capture has headroom for '{modelLabel}' " +
                $"({FormatApproximateBytes(availableBytes)} free, needs {FormatApproximateBytes(requiredBytes)}).",
        };
    }

    private static string FormatApproximateBytes(long bytes)
    {
        if (bytes < 1024)
        {
            return $"{bytes} B";
        }

        if (bytes < 1024 * 1024)
        {
            return $"{bytes / 1024.0:F1} KiB";
        }

        if (bytes < 1024L * 1024 * 1024)
        {
            return $"{bytes / (1024.0 * 1024):F1} MiB";
        }

        return $"{bytes / (1024.0 * 1024 * 1024):F1} GiB";
    }

    private static DoctorCheckRow CheckHuggingFaceDownloadAcceleration(TrackdubSessionFactory factory)
    {
        HuggingFaceDownloadOptions options = factory.GetRequiredService<HuggingFaceDownloadOptions>();
        HuggingFaceDownloadDiagnostics diagnostics = options.Describe();

        string mode = diagnostics.ParallelDownloadsEnabled
            ? $"parallel HTTP ({diagnostics.MaxParallelConnections} connections)"
            : "single-stream HTTP";

        string cliStatus = diagnostics.CliPreference switch
        {
            HuggingFaceCliPreference.Required when !diagnostics.CliAvailable =>
                "required but missing",
            HuggingFaceCliPreference.Required => "required and available",
            HuggingFaceCliPreference.Never => "disabled",
            _ => diagnostics.CliAvailable ? "auto (available)" : "auto (not on PATH)",
        };

        string message =
            $"Hugging Face downloads use {mode}; disable_xet={(diagnostics.DisableXet ? "on" : "off")}; " +
            $"hf CLI={cliStatus}; cli_hf_transfer={(diagnostics.EnableCliTransfer ? "on" : "off")}.";

        if (diagnostics.CliPreference == HuggingFaceCliPreference.Required && !diagnostics.CliAvailable)
        {
            return new DoctorCheckRow
            {
                Id = "hf-download-acceleration",
                Status = "fail",
                Message = message,
                Remediation =
                    "Install the Hugging Face CLI (pip install -U huggingface_hub[cli] hf_transfer) or set TRACKDUB_HF_USE_CLI=0.",
            };
        }

        return new DoctorCheckRow
        {
            Id = "hf-download-acceleration",
            Status = "pass",
            Message = message,
            Remediation =
                "Optional speedups: Trackdub parallel HTTP is on by default (TRACKDUB_HF_PARALLEL_DOWNLOADS). " +
                "For native CLI hf_transfer only, set TRACKDUB_HF_CLI_TRANSFER=1 (requires pip install hf_transfer). " +
                "Set HF_HUB_DISABLE_XET=1 / TRACKDUB_HF_DISABLE_XET=1 to disable Xet. Install the hf CLI for TRACKDUB_HF_USE_CLI=auto.",
        };
    }

    private static DoctorCheckRow CheckModelCacheWritable(IAppStoragePaths storagePaths)
    {
        string cacheDirectory = storagePaths.ModelCacheDirectory;
        try
        {
            Directory.CreateDirectory(cacheDirectory);
            string probePath = Path.Join(cacheDirectory, $".trackdub-write-probe-{Guid.NewGuid():N}");
            File.WriteAllText(probePath, "ok");
            File.Delete(probePath);

            return new DoctorCheckRow
            {
                Id = "model-cache-writable",
                Status = "pass",
                Message = $"Model cache directory is writable: {cacheDirectory}",
            };
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return new DoctorCheckRow
            {
                Id = "model-cache-writable",
                Status = "fail",
                Message = $"Model cache directory is not writable: {cacheDirectory}. {ex.Message}",
                Remediation = "Choose a writable --model-directory or fix permissions on the default cache path.",
            };
        }
    }

    private static async Task<IReadOnlyList<DoctorCheckRow>> CheckManifestInventoryAsync(
        TrackdubSessionFactory factory,
        CancellationToken cancellationToken)
    {
        try
        {
            IModelInventoryService inventory = factory.GetRequiredService<IModelInventoryService>();
            IReadOnlyList<ModelInventoryEntry> entries = await inventory
                .GetAllAsync(cancellationToken)
                .ConfigureAwait(false);

            if (entries.Count == 0)
            {
                return
                [
                    new DoctorCheckRow
                    {
                        Id = "bundled-manifest",
                        Status = "fail",
                        Message = "Bundled model manifest loaded zero entries.",
                        Remediation = "Verify the app build includes bundled-models.manifest.json.",
                    },
                ];
            }

            return
            [
                new DoctorCheckRow
                {
                    Id = "bundled-manifest",
                    Status = "pass",
                    Message = $"Bundled manifest loaded {entries.Count} model entries.",
                },
            ];
        }
        catch (Exception ex)
        {
            return
            [
                new DoctorCheckRow
                {
                    Id = "bundled-manifest",
                    Status = "fail",
                    Message = $"Failed to load bundled model inventory: {ex.Message}",
                    Remediation = "Run trackdub check --json and inspect model cache paths.",
                },
            ];
        }
    }

    private static DoctorCheckRow CheckFfmpeg(IFfmpegHealthCheck ffmpegHealthCheck)
    {
        FfmpegHealthStatus status = ffmpegHealthCheck.CheckAvailability();
        if (status.FfmpegAvailable && status.FfprobeAvailable)
        {
            return new DoctorCheckRow
            {
                Id = "ffmpeg-tools",
                Status = "pass",
                Message = $"ffmpeg and ffprobe are available ({status.FfmpegPath}; {status.FfprobePath}).",
            };
        }

        return new DoctorCheckRow
        {
            Id = "ffmpeg-tools",
            Status = "fail",
            Message = status.ErrorMessage ?? "ffmpeg or ffprobe is not available.",
            Remediation = "Install FFmpeg on PATH or allow the app bootstrap to download tools on first media run.",
        };
    }

    private static DoctorCheckRow CheckEspeakNg(IEspeakNgHealthCheck espeakHealthCheck)
    {
        EspeakNgHealthStatus status = espeakHealthCheck.CheckAvailability();
        if (status.Available)
        {
            return new DoctorCheckRow
            {
                Id = "espeak-ng",
                Status = "pass",
                Message = $"eSpeak-NG is available ({status.ExecutablePath}).",
            };
        }

        return new DoctorCheckRow
        {
            Id = "espeak-ng",
            Status = "fail",
            Message = status.ErrorMessage ?? "eSpeak-NG is not available.",
            Remediation = OperatingSystem.IsWindows()
                ? "Kokoro TTS needs eSpeak-NG. Run tools/espeak-ng/Fetch-EspeakNg.ps1, set TRACKDUB_ESPEAK_NG_PATH to espeak-ng.exe, or install eSpeak-NG on PATH."
                : "Kokoro TTS needs eSpeak-NG. Install the espeak-ng package, set TRACKDUB_ESPEAK_NG_PATH to the espeak-ng binary, or ensure espeak-ng is on PATH.",
        };
    }

    private static DoctorCheckRow CheckLogPath(IAppStoragePaths storagePaths)
    {
        string logPath = storagePaths.LogFilePath;
        try
        {
            string? logDirectory = Path.GetDirectoryName(logPath);
            if (!string.IsNullOrWhiteSpace(logDirectory))
            {
                Directory.CreateDirectory(logDirectory);
            }

            if (File.Exists(logPath))
            {
                using FileStream stream = File.Open(logPath, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite);
            }
            else
            {
                using FileStream stream = File.Open(logPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite);
            }

            return new DoctorCheckRow
            {
                Id = "log-path",
                Status = "pass",
                Message = $"Log file path is available: {logPath}",
            };
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return new DoctorCheckRow
            {
                Id = "log-path",
                Status = "warn",
                Message = $"Log file path may not be writable: {logPath}. {ex.Message}",
                Remediation = "Check permissions under the Trackdub app data directory.",
            };
        }
    }

    private static DoctorCheckRow CheckPlaybackNatives()
    {
        string baseDirectory = AppContext.BaseDirectory;
        string runtimeIdentifier = ResolveNativeRuntimeIdentifier();
        string nativeDirectory = Path.Join(baseDirectory, "native", runtimeIdentifier);

        string[] expectedNames = OperatingSystem.IsWindows()
            ? ["libmpv-2.dll", "libmpv-1.dll", "mpv-2.dll", "mpv-1.dll"]
            : OperatingSystem.IsMacOS()
                ? ["libmpv.2.dylib", "libmpv.1.dylib", "libmpv.dylib"]
                : ["libmpv.so.2", "libmpv.so.1", "libmpv.so"];

        bool found = expectedNames.Any(name => File.Exists(Path.Join(nativeDirectory, name)));
        if (found)
        {
            return new DoctorCheckRow
            {
                Id = "playback-natives",
                Status = "pass",
                Message = $"libmpv native found under {nativeDirectory}.",
            };
        }

        return new DoctorCheckRow
        {
            Id = "playback-natives",
            Status = "warn",
            Message = $"No bundled libmpv found under {nativeDirectory}. CLI pipeline runs do not require playback natives; Avalonia preview/export playback may fall back to bootstrap or LibVLC.",
            Remediation = "For desktop preview, run tools/dev fetch scripts or publish with native/{rid}/ assets.",
        };
    }

    private static async Task<DoctorCheckRow> CheckTensorRtRtxPluginAsync(
        TrackdubSessionFactory factory,
        CancellationToken cancellationToken)
    {
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
        {
            return new DoctorCheckRow
            {
                Id = "tensorrt-rtx-plugin",
                Status = "pass",
                Message = "TensorRT RTX EP ABI plugin applies on Windows and Linux with an NVIDIA GPU only.",
            };
        }

        ITensorRtRtxRuntimeReadinessService readinessService =
            factory.GetRequiredService<ITensorRtRtxRuntimeReadinessService>();
        IStudioSettingsService settingsService = factory.GetRequiredService<IStudioSettingsService>();

        TensorRtRtxRuntimeReadinessSnapshot snapshot = await readinessService
            .ProbeAsync(allowProviderDownloads: false, cancellationToken)
            .ConfigureAwait(false);
        StudioSettings settings = await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false);

        if (snapshot.IsReady)
        {
            return new DoctorCheckRow
            {
                Id = "tensorrt-rtx-plugin",
                Status = "pass",
                Message =
                    $"TensorRT RTX EP ABI plugin ready ({snapshot.RouteDisplay}; ORT provider listed={snapshot.IsOrtProviderListed}).",
            };
        }

        if (!snapshot.IsSupportedPlatform)
        {
            return new DoctorCheckRow
            {
                Id = "tensorrt-rtx-plugin",
                Status = "pass",
                Message = snapshot.Detail,
            };
        }

        if (snapshot.Blocker is TensorRtRtxReadinessBlocker.GpuVendorMismatch)
        {
            return new DoctorCheckRow
            {
                Id = "tensorrt-rtx-plugin",
                Status = "warn",
                Message = snapshot.Detail,
                Remediation = "TensorRT RTX requires an NVIDIA GPU. DirectML or CPU routes remain available for other vendors.",
            };
        }

        string licenseNote = settings.NvidiaTensorRtRtxLicenseAccepted
            ? "License accepted."
            : "NVIDIA license not accepted.";

        return new DoctorCheckRow
        {
            Id = "tensorrt-rtx-plugin",
            Status = "warn",
            Message = $"{snapshot.StatusLabel}: {snapshot.Detail} {licenseNote}",
            Remediation =
                "Run trackdub providers trt-rtx status, then trackdub providers trt-rtx install --accept-license, then trackdub providers trt-rtx smoke, or use Model Manager Install / tools/dev/Fetch-TrtRtxEp.ps1.",
        };
    }

    private static async Task<IReadOnlyList<DoctorCheckRow>> CheckCatalogExecutionProvidersAsync(
        TrackdubSessionFactory factory,
        CancellationToken cancellationToken)
    {
        DoctorCheckRow[] rows = await Task.WhenAll(
            ProbeMigraphxAsync(factory, cancellationToken),
            ProbeWinMlCatalogAsync(
                "execution-provider-qnn",
                "qnn",
                factory.GetRequiredService<IQnnCatalogRuntimeReadinessService>(),
                cancellationToken),
            ProbeWinMlCatalogAsync(
                "execution-provider-vitisai",
                "vitisai",
                factory.GetRequiredService<IVitisAiCatalogRuntimeReadinessService>(),
                cancellationToken),
            ProbeWinMlCatalogAsync(
                "execution-provider-openvino-catalog",
                "openvino-catalog",
                factory.GetRequiredService<IOpenVinoCatalogRuntimeReadinessService>(),
                cancellationToken)).ConfigureAwait(false);

        return rows;
    }

    private static async Task<DoctorCheckRow> ProbeMigraphxAsync(
        TrackdubSessionFactory factory,
        CancellationToken cancellationToken)
    {
        IMigraphxRuntimeReadinessService readiness =
            factory.GetRequiredService<IMigraphxRuntimeReadinessService>();
        MigraphxRuntimeReadinessSnapshot snapshot = await readiness
            .ProbeAsync(allowProviderDownloads: false, cancellationToken)
            .ConfigureAwait(false);

        if (snapshot.IsReady)
        {
            return new DoctorCheckRow
            {
                Id = "execution-provider-migraphx",
                Status = "pass",
                Message = $"migraphx: ready ({snapshot.RouteDisplay}). Registered≠models ready.",
            };
        }

        if (!snapshot.IsSupportedPlatform)
        {
            return new DoctorCheckRow
            {
                Id = "execution-provider-migraphx",
                Status = "pass",
                Message = $"migraphx: {snapshot.Detail}",
            };
        }

        return new DoctorCheckRow
        {
            Id = "execution-provider-migraphx",
            Status = "warn",
            Message = $"migraphx: {snapshot.Detail}",
            Remediation = snapshot.InstallHint
                ?? "Install the Windows ML MIGraphX catalog package, or ROCm with MIGraphX 2.17+ on Linux.",
        };
    }

    private static async Task<DoctorCheckRow> ProbeWinMlCatalogAsync(
        string id,
        string tag,
        IOpenVinoCatalogRuntimeReadinessService readiness,
        CancellationToken cancellationToken) =>
        await ProbeWinMlCatalogCoreAsync(
                id,
                tag,
                () => readiness.ProbeAsync(allowProviderDownloads: false, cancellationToken))
            .ConfigureAwait(false);

    private static async Task<DoctorCheckRow> ProbeWinMlCatalogAsync(
        string id,
        string tag,
        IQnnCatalogRuntimeReadinessService readiness,
        CancellationToken cancellationToken) =>
        await ProbeWinMlCatalogCoreAsync(
                id,
                tag,
                () => readiness.ProbeAsync(allowProviderDownloads: false, cancellationToken))
            .ConfigureAwait(false);

    private static async Task<DoctorCheckRow> ProbeWinMlCatalogAsync(
        string id,
        string tag,
        IVitisAiCatalogRuntimeReadinessService readiness,
        CancellationToken cancellationToken) =>
        await ProbeWinMlCatalogCoreAsync(
                id,
                tag,
                () => readiness.ProbeAsync(allowProviderDownloads: false, cancellationToken))
            .ConfigureAwait(false);

    private static async Task<DoctorCheckRow> ProbeWinMlCatalogCoreAsync(
        string id,
        string tag,
        Func<Task<WinMlCatalogRuntimeReadinessSnapshot>> probe)
    {
        WinMlCatalogRuntimeReadinessSnapshot snapshot = await probe().ConfigureAwait(false);

        if (snapshot.IsReady)
        {
            return new DoctorCheckRow
            {
                Id = id,
                Status = "pass",
                Message = $"{tag}: ready ({snapshot.RouteDisplay}). Registered≠models ready.",
            };
        }

        if (!snapshot.IsSupportedPlatform)
        {
            return new DoctorCheckRow
            {
                Id = id,
                Status = "pass",
                Message = $"{tag}: {snapshot.Detail}",
            };
        }

        return new DoctorCheckRow
        {
            Id = id,
            Status = "warn",
            Message = $"{tag}: {snapshot.Detail}",
            Remediation = snapshot.InstallHint
                ?? $"Install the Windows ML {tag} catalog EP and accept the vendor license in settings. Pin with --execution-provider {tag}.",
        };
    }

    private static DoctorCheckRow CheckWindowsMlSummary()
    {
        if (!OperatingSystem.IsWindows())
        {
            return new DoctorCheckRow
            {
                Id = "windows-ml",
                Status = "pass",
                Message = "Windows ML execution provider policy applies on Windows only.",
            };
        }

        return new DoctorCheckRow
        {
            Id = "windows-ml",
            Status = "warn",
            Message = "Windows ML EP registration is an app-shell/inference concern. CLI doctor does not execute ONNX smoke tests here.",
            Remediation = "Run trackdub check --json for model/runtime readiness. Set the policy via --device-policy on the CLI (headless) or Avalonia Advanced settings (app).",
        };
    }

    private static DoctorCheckRow BuildPipelineReadinessCheck(PipelineReadinessReport report)
    {
        if (report.IsRunReady)
        {
            return new DoctorCheckRow
            {
                Id = "pipeline-readiness",
                Status = "pass",
                Message = "Default pipeline stages report no blocking readiness states.",
            };
        }

        IReadOnlyList<StageReadiness> blocking = report.BlockingStages.ToList();
        string detail = string.Join(
            ", ",
            blocking.Select(stage => $"{stage.StageName}:{stage.Status}"));

        return new DoctorCheckRow
        {
            Id = "pipeline-readiness",
            Status = "fail",
            Message = $"Blocking readiness states detected: {detail}",
            Remediation = "Run trackdub models bundle-needed and trackdub models download <model-id> for missing commercial models.",
        };
    }

    internal const string GpuProfileEnvironmentVariable = "TRACKDUB_GPU_PROFILE";

    // The server container image (Dockerfile --target server) runs classic TensorRT + CUDA on NVIDIA datacenter
    // GPUs only, so these rows describe components it never ships or uses and would otherwise always warn there.
    private static readonly HashSet<string> s_rowsNotUsedByServerProfile = new(StringComparer.Ordinal)
    {
        "playback-natives",
        "tensorrt-rtx-plugin",
        "execution-provider-migraphx",
        "execution-provider-openvino-catalog",
    };

    /// <summary>
    /// Under the server GPU profile, reports warnings from rows the profile does not use as passing, with the
    /// original detail kept. Failing rows and every other row are returned unchanged.
    /// </summary>
    internal static List<DoctorCheckRow> ApplyGpuProfile(IReadOnlyList<DoctorCheckRow> checks, string? gpuProfile)
    {
        if (!string.Equals(gpuProfile?.Trim(), "server", StringComparison.OrdinalIgnoreCase))
        {
            return [.. checks];
        }

        return checks
            .Select(check => check.Status == "warn" && check.Id is not null && s_rowsNotUsedByServerProfile.Contains(check.Id)
                ? new DoctorCheckRow
                {
                    Id = check.Id,
                    Status = "pass",
                    Message = $"Not used by the server GPU profile ({GpuProfileEnvironmentVariable}=server). {check.Message}",
                }
                : check)
            .ToList();
    }

    private static string ResolveOverallStatus(IReadOnlyList<DoctorCheckRow> checks)
    {
        if (checks.Any(check => check.Status == "fail"))
        {
            return "fail";
        }

        if (checks.Any(check => check.Status == "warn"))
        {
            return "warn";
        }

        return "pass";
    }

    private static string ResolveNativeRuntimeIdentifier()
    {
        if (OperatingSystem.IsWindows())
        {
            return RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.Arm64 => "win-arm64",
                _ => "win-x64",
            };
        }

        if (OperatingSystem.IsMacOS())
        {
            return RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.Arm64 => "osx-arm64",
                _ => "osx-x64",
            };
        }

        return RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.Arm64 => "linux-arm64",
            _ => "linux-x64",
        };
    }

    private sealed class DoctorOutput
    {
        public string? Status { get; init; }
        public List<DoctorCheckRow>? Checks { get; init; }
    }

    internal sealed class DoctorCheckRow
    {
        public string? Id { get; init; }
        public string? Status { get; init; }
        public string? Message { get; init; }
        public string? Remediation { get; init; }
    }
}
