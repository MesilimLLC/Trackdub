using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging;
using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Dnnl;
using Trackdub.Inference.Onnx.EpContext;
using Trackdub.Inference.Onnx.ExecutionProviders;
using Trackdub.Inference.Onnx.Pool;
#if WINDOWS
using Trackdub.Inference.Onnx.WindowsMl;
#endif
using Trackdub.Inference.Onnx.Migraphx;
using Trackdub.Inference.Onnx.TensorRtRtx;
using Trackdub.Inference.Runtime.Migraphx;
using Trackdub.Inference.Runtime.TensorRtRtx;
using Trackdub.Inference.Runtime.WinMlCatalog;
using Trackdub.Inference.Runtime.Planning;
using Microsoft.ML.OnnxRuntime;

namespace Trackdub.Inference.Onnx;

internal static partial class OnnxExecutionSessionFactory
{
    /// <summary>
    /// Value object for the classic TensorRT cache identity. It keeps the fingerprint from being
    /// passed around as a bare primitive and centralises the string conversion at call sites.
    /// </summary>
    internal readonly record struct ClassicTensorRtCacheIdentity
    {
        public string Value { get; init; }

        public static implicit operator ClassicTensorRtCacheIdentity(string value) => new() { Value = value };
    }

    /// <summary>
    /// TensorRT-RTX sessions capture CUDA graphs unless a call site opts out. NVIDIA's native capture
    /// recaptures when input sizes or buffer locations change and falls back to normal execution for
    /// graphs it cannot capture (TensorRT-RTX "Working with RTX CUDA Graphs"), so a call site does not
    /// need stable buffers to be correct. Measured on an RTX 5070 with graphs forced on: identical
    /// stage output for Whisper (tiny, small, medium), Parakeet, Sortformer and Spleeter, and 57-87%
    /// lower per-graph median latency in device-bound benchmarks. To opt a model out, put
    /// <c>enable_cuda_graph = 0</c> in that call site's TRT options; to turn graphs off everywhere for a
    /// run, set <c>TRACKDUB_TRT_RTX_CUDA_GRAPHS=off</c>.
    /// </summary>
    internal const bool DefaultEnableCudaGraph = true;

    /// <summary>
    /// Builds TensorRT-RTX provider options.
    /// </summary>
    /// <param name="enableCudaGraph">
    /// Whether the session captures CUDA graphs. Defaults to <see cref="DefaultEnableCudaGraph"/>.
    /// An <c>enable_cuda_graph</c> key in <paramref name="additionalTrtOptions"/> overrides this
    /// argument, so a caller-supplied profile value takes precedence, and the
    /// <c>TRACKDUB_TRT_RTX_CUDA_GRAPHS</c> override wins over both.
    /// </param>
    internal static IReadOnlyDictionary<string, string> BuildTensorRtRtxOptions(
        IReadOnlyDictionary<string, string>? additionalTrtOptions,
        bool enableCudaGraph = DefaultEnableCudaGraph) =>
        BuildTensorRtRtxOptionsWithOverride(
            additionalTrtOptions,
            enableCudaGraph,
            Environment.GetEnvironmentVariable(CudaGraphsOverrideEnvironmentVariable));

    /// <summary>
    /// Builds the options with an explicit <c>enable_cuda_graph</c> override value. The override is
    /// applied after the call site's flag and every caller profile (including the ones from
    /// <see cref="WithCudaGraphsEnabled"/>) have been merged, so it wins over all of them.
    /// </summary>
    internal static IReadOnlyDictionary<string, string> BuildTensorRtRtxOptionsWithOverride(
        IReadOnlyDictionary<string, string>? additionalTrtOptions,
        bool enableCudaGraph,
        string? cudaGraphOverride)
    {
        var trtOptions = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            // Cache compiled TRT-RTX kernels locally to speed up subsequent session loads.
            ["nv_runtime_cache_path"] = ResolveTensorRtRtxRuntimeCachePath(),
        };

        // Always write the flag explicitly so the setting never depends on the EP's own default.
        trtOptions["enable_cuda_graph"] = enableCudaGraph ? "1" : "0";

        if (additionalTrtOptions is not null)
        {
            MergeAdditionalTensorRtRtxOptions(trtOptions, additionalTrtOptions);
        }

        TensorRtRtxDirectMlInterop.Apply(trtOptions, OperatingSystem.IsWindows());
        ApplyCudaGraphOverride(trtOptions, cudaGraphOverride);
        return trtOptions;
    }

    /// <summary>
    /// Diagnostic and A/B override for <c>enable_cuda_graph</c>: <c>on</c>, <c>1</c> or <c>true</c>, and
    /// <c>off</c>, <c>0</c> or <c>false</c> (case-insensitive), win over the call site and any caller
    /// profile; anything else leaves the call site's choice alone. The variable is read each time
    /// options are built, including for pool-key fingerprints, so set it before the process starts
    /// rather than changing it mid-run.
    /// </summary>
    internal static void ApplyCudaGraphOverride(IDictionary<string, string> trtOptions, string? overrideValue)
    {
        string? flag = overrideValue?.Trim().ToLowerInvariant() switch
        {
            "on" or "1" or "true" => "1",
            "off" or "0" or "false" => "0",
            _ => null,
        };

        if (flag is not null)
        {
            trtOptions["enable_cuda_graph"] = flag;
        }
    }

    private static void MergeAdditionalTensorRtRtxOptions(
        Dictionary<string, string> trtOptions,
        IReadOnlyDictionary<string, string> additionalTrtOptions)
    {
        CopyCallerTensorRtRtxOptions(trtOptions, additionalTrtOptions);
        CopyClassicTensorRtProfileAliases(trtOptions, additionalTrtOptions);
    }

    private static void CopyCallerTensorRtRtxOptions(
        Dictionary<string, string> trtOptions,
        IReadOnlyDictionary<string, string> additionalTrtOptions)
    {
        foreach ((string key, string value) in additionalTrtOptions)
        {
            // EP ABI 0.5.0 removed device_id (the device comes from the selected OrtEpDevice);
            // passing it through would fail session creation on an unknown option.
            if (ClassicTrtProfileKeysToNvRtx.ContainsKey(key))
            {
                continue;
            }

            if (string.Equals(key, "device_id", StringComparison.Ordinal))
            {
                continue;
            }

            trtOptions[key] = value;
        }
    }

    private static void CopyClassicTensorRtProfileAliases(
        Dictionary<string, string> trtOptions,
        IReadOnlyDictionary<string, string> additionalTrtOptions)
    {
        foreach ((string classicKey, string nvKey) in ClassicTrtProfileKeysToNvRtx)
        {
            if (trtOptions.ContainsKey(nvKey))
            {
                continue;
            }

            if (additionalTrtOptions.TryGetValue(classicKey, out string? shapes))
            {
                trtOptions[nvKey] = shapes;
            }
        }
    }

    /// <summary>
    /// Builds classic TensorRT EP provider options. Caller profile options pass through, but cache
    /// controls remain authoritative: the persistent cache is enabled only when the full hardware,
    /// driver, TensorRT binary, and ORT identity is known; otherwise caching is explicitly disabled.
    /// </summary>
    internal static IReadOnlyDictionary<string, string> BuildTensorRtOptions(
        IReadOnlyDictionary<string, string>? additionalTrtOptions) =>
        BuildTensorRtOptions(additionalTrtOptions, BuildClassicTensorRtEnvironmentFingerprint());

    internal static IReadOnlyDictionary<string, string> BuildTensorRtOptionsForModel(
        IReadOnlyDictionary<string, string>? additionalTrtOptions,
        string? trtModelPath) =>
        BuildTensorRtOptions(additionalTrtOptions, BuildClassicTensorRtEnvironmentFingerprint(), trtModelPath);

    internal static IReadOnlyDictionary<string, string> BuildTensorRtOptionsAfterProviderLoad(
        IReadOnlyDictionary<string, string>? additionalTrtOptions,
        string? trtModelPath,
        Action loadProvider,
        Func<string> resolveEnvironmentFingerprint)
    {
        loadProvider();
        return BuildTensorRtOptions(additionalTrtOptions, resolveEnvironmentFingerprint(), trtModelPath);
    }

    internal static IReadOnlyDictionary<string, string> BuildTensorRtOptions(
        IReadOnlyDictionary<string, string>? additionalTrtOptions,
        ClassicTensorRtCacheIdentity environmentFingerprint) =>
        BuildTensorRtOptions(additionalTrtOptions, environmentFingerprint, trtModelPath: null);

    private static IReadOnlyDictionary<string, string> BuildTensorRtOptions(
        IReadOnlyDictionary<string, string>? additionalTrtOptions,
        ClassicTensorRtCacheIdentity environmentFingerprint,
        string? trtModelPath)
    {
        var trtOptions = new Dictionary<string, string>(StringComparer.Ordinal);
        if (additionalTrtOptions is not null)
        {
            ApplyCallerClassicTensorRtOptions(trtOptions, additionalTrtOptions);
        }

        if (!HasCompleteClassicTensorRtCacheIdentity(environmentFingerprint))
        {
            // A process-specific TRT identity prevents cross-process reuse only when TRT itself
            // is unknown. Unknown GPU/driver/ORT identity is equally unsafe, so disable the cache
            // outright instead of creating a misleading persistent directory.
            DisableClassicTensorRtCache(trtOptions);
            return trtOptions;
        }

        trtOptions["trt_engine_cache_enable"] = "1";
        trtOptions["trt_engine_cache_path"] = ResolveClassicTensorRtEngineCacheDirectory(environmentFingerprint);
        if (!string.IsNullOrWhiteSpace(trtModelPath))
        {
            trtOptions["trt_engine_cache_prefix"] = BuildClassicTensorRtModelCachePrefix(trtModelPath);
        }
        return trtOptions;
    }

    private static void ApplyCallerClassicTensorRtOptions(
        Dictionary<string, string> trtOptions,
        IReadOnlyDictionary<string, string> additionalTrtOptions)
    {
        foreach ((string key, string value) in additionalTrtOptions)
        {
            // Caller options are shared with the TensorRT-RTX builder. The classic EP rejects
            // unknown keys, which makes UpdateOptions throw and drops the session to no-cache.
            if (string.Equals(key, "enable_cuda_graph", StringComparison.Ordinal))
            {
                trtOptions["trt_cuda_graph_enable"] = value;
            }
            else if (!key.StartsWith("nv_", StringComparison.Ordinal))
            {
                trtOptions[key] = value;
            }
        }
    }

    private static void DisableClassicTensorRtCache(Dictionary<string, string> trtOptions)
    {
        trtOptions["trt_engine_cache_enable"] = "0";
        trtOptions.Remove("trt_engine_cache_path");
        trtOptions.Remove("trt_engine_cache_prefix");
    }

    internal static string BuildClassicTensorRtModelCachePrefix(string modelPath)
    {
        string normalizedPath = Path.GetFullPath(modelPath);
        byte[] digest = System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(normalizedPath));
        return $"trackdub-{Convert.ToHexString(digest.AsSpan(0, 8)).ToLowerInvariant()}-";
    }

    internal static string ResolveClassicTensorRtEngineCacheDirectory() =>
        ResolveClassicTensorRtEngineCacheDirectory(BuildClassicTensorRtEnvironmentFingerprint());

    internal static string ResolveClassicTensorRtEngineCacheDirectory(ClassicTensorRtCacheIdentity environmentFingerprint) =>
        Path.Join(
            ResolveEngineCacheRoot(),
            "classic-trt",
            SanitizeClassicTensorRtCacheKey(environmentFingerprint));

    // Classic engine-cache key components: GPU architecture bucket | exact GPU model name |
    // GPU driver version | TensorRT binary identity | ONNX Runtime version. This prevents engines
    // from being reused across two GPUs in the same architecture bucket (e.g. a moved /data volume).
    // The cache is disabled if any component is unavailable. The TensorRT-RTX EP version in
    // EpContextLoadPathResolver's fingerprint is deliberately excluded — classic TRT engines do
    // not depend on it. TensorRT identity is sourced from the mapped TensorRT core binary; an
    // unresolved identity uses a process-specific namespace so an unverified runtime never reuses
    // engines from another process.
    internal static string BuildClassicTensorRtEnvironmentFingerprint()
    {
        // The built-in classic TRT provider is not loaded until AppendExecutionProvider_Tensorrt,
        // but its cache directory must be selected before that call. Preload the ABI library so
        // the first session and every later process can identify the same native binary.
        TryLoadTensorRtIdentityLibrary();
        (string gpuArchitecture, string? driverVersion) = EpContextLoadPathResolver.CurrentHardware;
        string driver = string.IsNullOrWhiteSpace(driverVersion) ? "unknown" : driverVersion;
        string gpuModel = AcceleratorVramProbe.QueryNvmlGpuIdentity().DeviceName ?? "unknown";
        return $"{gpuArchitecture}|{gpuModel}|{driver}|trt-{ResolveTensorRtVersionIdentity()}|ort-{EpContextArtifact.HostOrtRuntimeVersion}";
    }

    internal static bool HasCompleteClassicTensorRtCacheIdentity(ClassicTensorRtCacheIdentity environmentFingerprint)
    {
        string[] parts = environmentFingerprint.Value.Split('|');
        if (parts.Length != 5)
        {
            return false;
        }

        if (!HasKnownFingerprintParts(parts))
        {
            return false;
        }

        if (!HasValidTensorRtBinaryIdentity(parts))
        {
            return false;
        }

        return HasValidOrtIdentity(parts);
    }

    private static bool HasKnownFingerprintParts(string[] parts)
    {
        foreach (string part in parts)
        {
            if (string.IsNullOrWhiteSpace(part))
            {
                return false;
            }

            if (string.Equals(part, "unknown", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
        }

        return true;
    }

    private static bool HasValidTensorRtBinaryIdentity(string[] parts)
    {
        const string tensorRtHashPrefix = "trt-sha256-";
        string identity = parts[3];
        if (!identity.StartsWith(tensorRtHashPrefix, StringComparison.Ordinal))
        {
            return false;
        }

        if (identity.Length != tensorRtHashPrefix.Length + 64)
        {
            return false;
        }

        return identity.AsSpan(tensorRtHashPrefix.Length).ToArray().All(Uri.IsHexDigit);
    }

    private static bool HasValidOrtIdentity(string[] parts)
    {
        string identity = parts[4];
        if (!identity.StartsWith("ort-", StringComparison.Ordinal))
        {
            return false;
        }

        if (identity.Length <= "ort-".Length)
        {
            return false;
        }

        return !string.Equals(identity, "ort-unknown", StringComparison.OrdinalIgnoreCase);
    }

    private static string SanitizeClassicTensorRtCacheKey(ClassicTensorRtCacheIdentity environmentFingerprint)
    {
        // Hash the full identity so distinct fingerprints (for example GPU model names that differ
        // only in punctuation) always map to distinct directories. A character-replacement scheme
        // is not injective and could let two hardware identities share an engine directory.
        // This changes the directory for every identity, so existing classic-trt directories are
        // no longer read: engines recompile once, and `trackdub cache clear engines` removes the stale ones.
        byte[] digest = System.Security.Cryptography.SHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes(environmentFingerprint.Value));
        return Convert.ToHexString(digest).ToLowerInvariant();
    }

    private static readonly string UnresolvedTensorRtIdentity = $"unknown-process-{Guid.NewGuid():N}";
    private static readonly ConcurrentDictionary<string, string> LoadedTensorRtBinaryIdentities = new(StringComparer.Ordinal);

    private static bool TryLoadTensorRtIdentityLibrary()
    {
        // The current server image ships TensorRT 10. On an unsupported major version, keep the
        // identity unresolved and let the EP append path proceed without cross-process cache reuse.
        string libraryName = OperatingSystem.IsLinux()
            ? "libnvinfer.so.10"
            : OperatingSystem.IsWindows() ? "nvinfer.dll" : string.Empty;
        if (libraryName.Length == 0)
        {
            return false;
        }

        lock (TensorRtIdentityLibraryLoadLock)
        {
            if (_tensorRtIdentityLibraryHandle != IntPtr.Zero)
            {
                return true;
            }

            try
            {
                if (!NativeLibrary.TryLoad(libraryName, out IntPtr handle))
                {
                    return false;
                }

                // Keep a reference for process lifetime so ORT's provider append sees the same
                // module and the resolved identity cannot become stale before session creation.
                _tensorRtIdentityLibraryHandle = handle;
                return true;
            }
            catch (Exception ex) when (ex is DllNotFoundException or BadImageFormatException
                or FileLoadException or EntryPointNotFoundException)
            {
                return false;
            }
        }
    }

    internal static string ResolveTensorRtVersionIdentity()
    {
        try
        {
            using Process process = Process.GetCurrentProcess();
            return ResolveTensorRtVersionIdentity(
                process.Modules.Cast<ProcessModule>().Select(static module => module.FileName),
                static path => LoadedTensorRtBinaryIdentities.GetOrAdd(path, static loadedPath =>
                {
                    using FileStream stream = File.OpenRead(loadedPath);
                    return $"sha256-{Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream))}";
                }));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidOperationException
            or NotSupportedException or System.ComponentModel.Win32Exception)
        {
            return UnresolvedTensorRtIdentity;
        }
    }

    // Only mapped TensorRT core binaries count. Do not inspect PATH, LD_LIBRARY_PATH or
    // installed filenames: those can describe a different runtime from the one ORT loaded.
    // Hashing the actual binary also distinguishes builds with identical sonames. Missing or
    // ambiguous identities get a process-specific namespace, preventing cross-process reuse.
    internal static string ResolveTensorRtVersionIdentity(
        IEnumerable<string> loadedModulePaths,
        Func<string, string> readBinaryIdentity)
    {
        string[] candidates = loadedModulePaths
            .Where(static path => IsTensorRtCoreLibrary(Path.GetFileName(path)))
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        return candidates.Length == 1 ? readBinaryIdentity(candidates[0]) : UnresolvedTensorRtIdentity;
    }

    private static bool IsTensorRtCoreLibrary(string name) =>
        name.Equals("libnvinfer.so", StringComparison.Ordinal)
        || name.StartsWith("libnvinfer.so.", StringComparison.Ordinal)
        || name.Equals("nvinfer.dll", StringComparison.OrdinalIgnoreCase)
        || (name.StartsWith("nvinfer_", StringComparison.OrdinalIgnoreCase)
            && name.EndsWith(".dll", StringComparison.OrdinalIgnoreCase)
            && name.Length > 12 && name[8..^4].All(char.IsDigit));

    /// <summary>
    /// Composes enable_cuda_graph onto a caller-supplied TRT option set. Production callsites that
    /// keep every bound OrtValue at a fixed address opt in here; profile dictionaries themselves
    /// stay free of the flag so benchmark/AOT consumers keep explicit control of the session mode.
    /// </summary>
    internal static IReadOnlyDictionary<string, string> WithCudaGraphsEnabled(
        IReadOnlyDictionary<string, string> options)
    {
        var merged = new Dictionary<string, string>(options, StringComparer.Ordinal)
        {
            ["enable_cuda_graph"] = "1",
        };
        return merged;
    }

    // TensorRT-RTX runtime caches are not portable across runtime versions, so each pinned
    // EP ABI + runtime lineage gets its own directory and an upgrade never feeds old kernels in.
    internal static string ResolveTensorRtRtxRuntimeCachePath() =>
        Path.Join(ResolveEngineCacheRoot(), "trt-rtx", TensorRtRtxProviderConstants.BundledFingerprintVersion);

    private static string ResolveEngineCacheRoot()
    {
        string? engineCacheRoot = Environment.GetEnvironmentVariable(EngineCacheRootEnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(engineCacheRoot))
        {
            return NormalizePath(engineCacheRoot);
        }

        string? cacheRoot = Environment.GetEnvironmentVariable(CacheRootEnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(cacheRoot))
        {
            return Path.Join(NormalizePath(cacheRoot), "EngineCache");
        }

        string localAppDataRoot = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (string.IsNullOrWhiteSpace(localAppDataRoot))
        {
            localAppDataRoot = AppContext.BaseDirectory;
        }

        return Path.Join(localAppDataRoot, "Trackdub", "EngineCache");
    }

    private static string NormalizePath(string path) =>
        Path.GetFullPath(Environment.ExpandEnvironmentVariables(path));
}
