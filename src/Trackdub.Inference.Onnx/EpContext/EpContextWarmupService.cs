using System.Diagnostics;
using Trackdub.Contracts;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Madlad;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Onnx.TensorRtRtx;
using Trackdub.Inference.Runtime.Planning;
using Trackdub.Inference.Runtime.TensorRtRtx;
#if WINDOWS
using Trackdub.Inference.Onnx.WindowsMl;
#endif
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Trackdub.Inference.Onnx.EpContext;

/// <summary>
/// One-shot warm pass: compile EP-context models for local ONNX graphs, then load each once so
/// <c>nv_runtime_cache_path</c> receives residual CUDA kernels. Re-run after install/update/driver change.
/// </summary>
public sealed class EpContextWarmupService : IEpContextWarmupService
{
    private static readonly string[] SkippedNameFragments =
    [
        // GenAI loads can hard-crash the host under TRT; never AOT those graphs.
        "phi-",
        "Phi-",
    ];

    private readonly IAppStoragePaths _storagePaths;
    private readonly IHardwareProfileProvider _hardwareProfileProvider;
    private readonly EpContextCompiler _compiler;

    public EpContextWarmupService(
        IAppStoragePaths storagePaths,
        IHardwareProfileProvider? hardwareProfileProvider = null,
        EpContextCompiler? compiler = null)
    {
        _storagePaths = storagePaths ?? throw new ArgumentNullException(nameof(storagePaths));
        _hardwareProfileProvider = hardwareProfileProvider ?? new MachineHardwareProfileProvider();
        _compiler = compiler ?? new EpContextCompiler();
    }

    public async Task InitializeHardwareIdentityAsync(CancellationToken cancellationToken = default) =>
        _ = await LoadHardwareIdentityAsync(cancellationToken).ConfigureAwait(false);

    private async Task<HardwareProfile> LoadHardwareIdentityAsync(CancellationToken cancellationToken)
    {
        HardwareProfile hardware = await _hardwareProfileProvider.GetCurrentAsync(cancellationToken).ConfigureAwait(false);
        EpContextLoadPathResolver.SetCurrentHardware(hardware);
        return hardware;
    }

    public async Task<EpContextWarmReport> WarmAsync(
        IReadOnlyList<string>? sourceModelPaths = null,
        IProgress<string>? progress = null,
        ExecutionProviderKind provider = ExecutionProviderKind.TensorRTRtx,
        CancellationToken cancellationToken = default)
    {
        HardwareProfile hardware = await LoadHardwareIdentityAsync(cancellationToken).ConfigureAwait(false);
        string fingerprint = provider is ExecutionProviderKind.TensorRt
            ? OnnxExecutionSessionFactory.BuildClassicTensorRtEnvironmentFingerprint()
            : EpContextLoadPathResolver.CurrentEnvironmentFingerprint;

        IReadOnlyList<string> sources = sourceModelPaths is { Count: > 0 }
            ? sourceModelPaths
            : DiscoverLocalOnnxModels(_storagePaths.ModelCacheDirectory);

        bool classicTensorRt = provider is ExecutionProviderKind.TensorRt;
        string engineCacheDirectory = classicTensorRt
            ? OnnxExecutionSessionFactory.ResolveClassicTensorRtEngineCacheDirectory()
            : _storagePaths.EngineCacheDirectory;
        if (classicTensorRt)
        {
            Directory.CreateDirectory(engineCacheDirectory);
        }

        var items = new List<EpContextWarmItem>(sources.Count);
        int compiled = 0, reused = 0, skipped = 0, failed = 0;

        foreach (string sourcePath in sources)
        {
            cancellationToken.ThrowIfCancellationRequested();
            (EpContextWarmItem item, bool didCompile, bool didReuse) = classicTensorRt
                ? WarmClassicTensorRtSource(sourcePath, engineCacheDirectory, progress, cancellationToken)
                : await WarmSourceAsync(
                    sourcePath, hardware, progress, cancellationToken).ConfigureAwait(false);
            items.Add(item);
            if (didCompile && item.Status != "failed") compiled++;
            if (didReuse && item.Status != "failed") reused++;
            if (item.Status == "skipped") skipped++;
            if (item.Status == "failed") failed++;
        }

        if (classicTensorRt)
        {
            // Provider initialization may have made the loaded TensorRT identity available.
            fingerprint = OnnxExecutionSessionFactory.BuildClassicTensorRtEnvironmentFingerprint();
            engineCacheDirectory = OnnxExecutionSessionFactory.ResolveClassicTensorRtEngineCacheDirectory(fingerprint);
        }

        return new EpContextWarmReport(
            engineCacheDirectory,
            fingerprint,
            compiled,
            reused,
            skipped,
            failed,
            items);
    }

    // Classic TensorRT has no EP-context AOT: warm by creating a real session on the
    // original graph with the persistent engine cache enabled, then classify the cache
    // graph-prefixed engine growth as compiled vs reused. Session creation alone is not enough:
    // for dynamic-shape graphs the TensorRT EP defers engine building to the first
    // inference, and only then writes the engine/timing-cache files (verified live on
    // ORT 1.31 with DeepFilterNet3 and Parakeet: no files after Create, .engine +
    // .profile + .timing after one Run). So warm runs one zero-input inference.
    private (EpContextWarmItem Item, bool Compiled, bool Reused) WarmClassicTensorRtSource(
        string sourcePath,
        string classicCacheDirectory,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        if (ShouldSkip(sourcePath))
        {
            return (new EpContextWarmItem(sourcePath, "skipped", null, 0, 0,
                "Excluded family or already an EP-context artifact."), false, false);
        }

        if (MadladTrtProfiles.GetAotExclusionReason(sourcePath) is { } exclusionReason)
        {
            return (new EpContextWarmItem(sourcePath, "skipped", null, 0, 0, exclusionReason), false, false);
        }

        progress?.Report($"warm {Path.GetFileName(sourcePath)}");
        string cachePrefix = OnnxExecutionSessionFactory.BuildClassicTensorRtModelCachePrefix(sourcePath);
        double warmMs = TryWarmLoadClassicTensorRt(
            sourcePath,
            EpContextTrtProfiles.Resolve(sourcePath),
            cancellationToken,
            ref classicCacheDirectory,
            out EngineCacheProbe.Snapshot before,
            out EngineCacheProbe.Snapshot after,
            out string? warmFailure);

        if (warmFailure is not null)
        {
            return (new EpContextWarmItem(sourcePath, "failed", null, 0, warmMs, warmFailure), false, false);
        }

        string outcome = ClassicTensorRtCacheClassifier.Classify(before, after);
        return outcome switch
        {
            "compiled" => (new EpContextWarmItem(
                sourcePath, "compiled", null, 0, warmMs,
                $"Classic TensorRT engine cache for '{cachePrefix}' grew ({before.FileCount} -> {after.FileCount} file(s), "
                + $"{before.TotalBytes} -> {after.TotalBytes} byte(s)) in '{classicCacheDirectory}'."), true, false),
            "reused" => (new EpContextWarmItem(
                sourcePath, "reused", null, 0, warmMs,
                $"Classic TensorRT engines for '{cachePrefix}' in '{classicCacheDirectory}' were already populated and did not grow."), false, true),
            _ => (new EpContextWarmItem(
                sourcePath, "failed", null, 0, warmMs,
                $"Session ran but no nonempty classic TensorRT engine for '{cachePrefix}' was found in '{classicCacheDirectory}'."), false, false),
        };
    }

    private static double TryWarmLoadClassicTensorRt(
        string modelPath,
        IReadOnlyDictionary<string, string>? modelTrtOptions,
        CancellationToken cancellationToken,
        ref string cacheDirectory,
        out EngineCacheProbe.Snapshot before,
        out EngineCacheProbe.Snapshot after,
        out string? failureReason)
    {
        before = new EngineCacheProbe.Snapshot(cacheDirectory, DirectoryExists: false, FileCount: 0, TotalBytes: 0);
        after = before;
        var stopwatch = Stopwatch.StartNew();
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
#if WINDOWS
            WindowsMlOnnxRuntimeNativeResolver.EnsureInitialized();
#endif
            using SessionOptions options = new();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
            ExecutionProviderKind selectedProvider =
                OnnxExecutionSessionFactory.AppendTensorRtOrFallbackProvider(options, modelTrtOptions, modelPath);
            if (selectedProvider is not ExecutionProviderKind.TensorRt)
            {
                stopwatch.Stop();
                failureReason =
                    $"Requested tensorrt but classic TensorRT EP was unavailable; fell back to {selectedProvider}.";
                return stopwatch.Elapsed.TotalMilliseconds;
            }

            // Snapshot the exact path passed to this provider, after native initialization.
            // The pre-initialization fingerprint may have had an unresolved TensorRT identity.
            cacheDirectory = OnnxExecutionSessionFactory.TryGetAppendedTensorRtOptions(options)!["trt_engine_cache_path"];
            string cachePrefix = OnnxExecutionSessionFactory.BuildClassicTensorRtModelCachePrefix(modelPath);
            before = ClassicTensorRtCacheClassifier.CaptureSnapshot(cacheDirectory, cachePrefix);
            using InferenceSession session = new(modelPath, options);
            using var results = session.Run(BuildClassicTensorRtWarmInputs(session.InputMetadata));
            after = ClassicTensorRtCacheClassifier.CaptureSnapshot(cacheDirectory, cachePrefix);
            stopwatch.Stop();
            failureReason = null;
            return stopwatch.Elapsed.TotalMilliseconds;
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or InvalidOperationException or DllNotFoundException or EntryPointNotFoundException or NotSupportedException)
        {
            stopwatch.Stop();
            failureReason = ex.Message;
            return stopwatch.Elapsed.TotalMilliseconds;
        }
    }

    internal static IReadOnlyList<NamedOnnxValue> BuildClassicTensorRtWarmInputs(
        IReadOnlyDictionary<string, NodeMetadata> inputMetadata)
    {
        var values = new List<NamedOnnxValue>(inputMetadata.Count);
        foreach ((string inputName, NodeMetadata metadata) in inputMetadata)
        {
            if (!metadata.IsTensor)
            {
                throw new NotSupportedException($"Warm input '{inputName}' is not a tensor input.");
            }

            values.Add(BuildClassicTensorRtWarmInput(inputName, metadata.ElementDataType, metadata.Dimensions));
        }

        return values;
    }

    internal static NamedOnnxValue BuildClassicTensorRtWarmInput(
        string inputName,
        TensorElementType elementType,
        IReadOnlyList<int> dimensions)
    {
        int[] resolvedDimensions = dimensions.Select(static dimension => dimension > 0 ? dimension : 1).ToArray();
        int elementCount = resolvedDimensions.Aggregate(1, static (product, dimension) => checked(product * dimension));

        return elementType switch
        {
            TensorElementType.Float => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<float>(new float[elementCount], resolvedDimensions)),
            TensorElementType.Float16 => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<Float16>(new Float16[elementCount], resolvedDimensions)),
            TensorElementType.Int32 => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<int>(new int[elementCount], resolvedDimensions)),
            TensorElementType.Int64 => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<long>(new long[elementCount], resolvedDimensions)),
            TensorElementType.Bool => NamedOnnxValue.CreateFromTensor(
                inputName,
                new DenseTensor<bool>(new bool[elementCount], resolvedDimensions)),
            _ => throw new NotSupportedException(
                $"Warm input '{inputName}' uses unsupported tensor element type '{elementType}'."),
        };
    }

    private async Task<(EpContextWarmItem Item, bool Compiled, bool Reused)> WarmSourceAsync(
        string sourcePath,
        HardwareProfile hardware,
        IProgress<string>? progress,
        CancellationToken cancellationToken)
    {
        if (ShouldSkip(sourcePath))
        {
            return (new EpContextWarmItem(sourcePath, "skipped", null, 0, 0,
                "Excluded family or already an EP-context artifact."), false, false);
        }

        if (MadladTrtProfiles.GetAotExclusionReason(sourcePath) is { } exclusionReason)
        {
            return (new EpContextWarmItem(sourcePath, "skipped", null, 0, 0, exclusionReason), false, false);
        }

        string epContextPath = EpContextArtifact.GetEpContextPath(sourcePath);
        bool haveValid = EpContextLoadPathResolver.TryResolveLoadPath(
            sourcePath,
            OnnxExecutionSessionFactory.BuildTensorRtRtxOptions(EpContextTrtProfiles.Resolve(sourcePath))) is not null;
        double compileMs = 0;
        if (!haveValid)
        {
            progress?.Report($"compile {Path.GetFileName(sourcePath)}");
            EpContextCompiler.CompileResult compile = await _compiler
                .CompileAsync(sourcePath, epContextPath, cancellationToken).ConfigureAwait(false);
            compileMs = compile.CompileMilliseconds;
            if (!compile.Success)
            {
                return (new EpContextWarmItem(sourcePath, "failed", null, compileMs, 0, compile.FailureReason), false, false);
            }

            // Compilation publishes the artifact and stamp under one cross-process lock.
        }

        // Residual JIT: load once so nv_runtime_cache_path stores CUDA kernels for this GPU.
        progress?.Report($"warm {Path.GetFileName(sourcePath)}");
        double warmMs = TryWarmLoad(
            epContextPath,
            EpContextTrtProfiles.Resolve(sourcePath),
            cancellationToken,
            out string? warmFailure);
        return (new EpContextWarmItem(
            sourcePath, warmFailure is null ? (haveValid ? "reused" : "compiled") : "failed",
            epContextPath, compileMs, warmMs,
            warmFailure ?? (haveValid ? "Existing EP-context artifact is current." : null)),
            !haveValid, haveValid);
    }

    private static bool ShouldSkip(string sourcePath)
    {
        if (EpContextArtifact.IsEpContextPath(sourcePath))
        {
            return true;
        }

        string fileName = Path.GetFileName(sourcePath);
        return SkippedNameFragments.Any(fragment => fileName.Contains(fragment, StringComparison.OrdinalIgnoreCase));
    }

    private static IReadOnlyList<string> DiscoverLocalOnnxModels(string modelCacheDirectory)
    {
        if (!Directory.Exists(modelCacheDirectory))
        {
            return [];
        }

        return Directory
            .EnumerateFiles(modelCacheDirectory, "*.onnx", SearchOption.AllDirectories)
            .Where(path => !EpContextArtifact.IsEpContextPath(path))
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private double TryWarmLoad(
        string modelPath,
        IReadOnlyDictionary<string, string>? modelTrtOptions,
        CancellationToken cancellationToken,
        out string? failureReason)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
#if WINDOWS
            WindowsMlOnnxRuntimeNativeResolver.EnsureInitialized();
#endif
            using SessionOptions options = new();
            ExecutionProviderKind selectedProvider =
                OnnxExecutionSessionFactory.AppendTensorRtRtxOrFallbackProvider(options, modelTrtOptions, enableCudaGraph: false);
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING;
            InferenceSession session = new(modelPath, options);
            if (selectedProvider is ExecutionProviderKind.TensorRTRtx)
            {
                TensorRtRtxTeardownGuard.Track(session);
            }

            try
            {
                stopwatch.Stop();
                failureReason = null;
                return stopwatch.Elapsed.TotalMilliseconds;
            }
            finally
            {
                TensorRtRtxTeardownGuard.DisposeSafely(session);
            }
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or InvalidOperationException or DllNotFoundException or EntryPointNotFoundException)
        {
            stopwatch.Stop();
            TensorRtRtxTeardownGuard.MarkPoisonedIfCudaOutOfMemory(ex);
            failureReason = ex.Message;
            return stopwatch.Elapsed.TotalMilliseconds;
        }
    }
}
