using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using Trackdub.Domain;

namespace Trackdub.Inference.Onnx.Pool;

/// <summary>
/// Composite key that uniquely identifies a pooled ONNX <see cref="Microsoft.ML.OnnxRuntime.InferenceSession"/>.
/// Every identity property participates in equality and hash code so pool lookups are exact.
/// <see cref="EstimatedVramMb"/> is admission sizing, not identity: it is excluded so a sidecar
/// re-measurement or transient IO glitch never maps the same model to a second pooled session.
/// </summary>
/// <remarks>
/// <para><strong>EngineFamily:</strong>
/// Logical engine family (e.g. "kokoro", "whisper-onnx", "opus-mt", "chatterbox", "silero-vad").
/// Stored lowercase; any uppercase input is normalised on construction.</para>
/// <para><strong>ModelId:</strong>
/// Optional manifest model ID (e.g. "onnx-community/Kokoro-82M-v1.0-ONNX").
/// Stored lowercase; any uppercase input is normalised on construction.</para>
/// <para><strong>Variant:</strong>
/// Optional model variant / quantization tag (e.g. "q4", "fp16").
/// Stored lowercase; any uppercase input is normalised on construction.</para>
/// <para><strong>Provider:</strong> Execution provider used to create the session.</para>
/// <para><strong>PathHash:</strong>
/// Stable, lowercase hex SHA-256 of the model file path (not file content).
/// On Windows, casing is normalised before hashing; on macOS and Linux the path is hashed as-is
/// to handle case-sensitive APFS and ext4 volumes correctly.
/// Separators and relative segments are not canonicalised.
/// Use <see cref="HashPath"/> to derive this value from a model file path.</para>
/// <para><strong>ModelContentHash:</strong>
/// Cached lowercase hex SHA-256 of the model file's content (see <see cref="HashModelContent"/>).
/// The key therefore carries both the path location and a content digest: replacing a model
/// file at the same path changes the helper-built key and invalidates the stale pooled
/// session. Missing or unreadable files yield <see langword="null"/> and keep path identity.
/// Directly constructed keys may leave it unset.</para>
/// <para><strong>DeviceId:</strong>
/// Device ordinal (0 for the default device). Used to distinguish sessions on different GPUs.</para>
/// <para><strong>GraphRole:</strong>
/// Role of this session within a multi-graph model. Stored lowercase; input is normalised on construction.
/// Conventional values: <c>"default"</c> (single-session models),
/// <c>"encoder"</c>, <c>"decoder"</c> (Whisper / Opus-MT / MADLAD),
/// <c>"speech-encoder"</c>, <c>"embed-tokens"</c>, <c>"lm"</c>, <c>"conditional-decoder"</c> (Chatterbox).
/// </para>
/// <para><strong>OptionsFingerprint:</strong>
/// Stable hash of session provider options that affect ONNX Runtime session construction.</para>
/// </remarks>
internal sealed record SessionPoolKey
{
    private const string DefaultOptionsFingerprint = "default";

    // Normalise string discriminators to lowercase so record equality is case-insensitive
    // and consistent with the case-insensitive matching in EvictModelAsync.
    private string _engineFamily = null!;
    private string? _modelId;
    private string? _variant;
    private string _pathHash = null!;
    private string? _modelContentHash;
    private string _graphRole = null!;
    private string _optionsFingerprint = null!;

    public SessionPoolKey(
        string engineFamily,
        string? modelId,
        string? variant,
        ExecutionProviderKind provider,
        string pathHash,
        int? deviceId,
        string graphRole,
        string? optionsFingerprint = null,
        string? modelContentHash = null)
    {
        ArgumentNullException.ThrowIfNull(engineFamily);
        ArgumentNullException.ThrowIfNull(pathHash);
        ArgumentNullException.ThrowIfNull(graphRole);

        _engineFamily = engineFamily.ToLowerInvariant();
        ModelId = modelId;
        Variant = variant;
        Provider = provider;
        _pathHash = pathHash.ToLowerInvariant();
        DeviceId = deviceId;
        _graphRole = graphRole.ToLowerInvariant();
        _optionsFingerprint = string.IsNullOrWhiteSpace(optionsFingerprint)
            ? DefaultOptionsFingerprint
            : optionsFingerprint.ToLowerInvariant();
        _modelContentHash = string.IsNullOrWhiteSpace(modelContentHash)
            ? null
            : modelContentHash.ToLowerInvariant();
    }

    public string EngineFamily
    {
        get => _engineFamily;
        init => _engineFamily = (value ?? throw new System.ArgumentNullException(nameof(EngineFamily))).ToLowerInvariant();
    }

    public string? ModelId
    {
        get => _modelId;
        init => _modelId = value?.ToLowerInvariant();
    }

    public string? Variant
    {
        get => _variant;
        init => _variant = value?.ToLowerInvariant();
    }

    public string PathHash
    {
        get => _pathHash;
        init => _pathHash = (value ?? throw new System.ArgumentNullException(nameof(PathHash))).ToLowerInvariant();
    }

    /// <summary>
    /// Cached SHA-256 digest of the model file's content, or <see langword="null"/> when the
    /// file could not be read. Helper-built keys always carry it, so replacing a model at
    /// the same path produces a different pool key instead of silently reusing the stale
    /// session; directly constructed keys may omit it and keep path-only identity.
    /// </summary>
    public string? ModelContentHash
    {
        get => _modelContentHash;
        init => _modelContentHash = string.IsNullOrWhiteSpace(value) ? null : value.ToLowerInvariant();
    }

    public string GraphRole
    {
        get => _graphRole;
        init => _graphRole = (value ?? throw new System.ArgumentNullException(nameof(GraphRole))).ToLowerInvariant();
    }

    public string OptionsFingerprint
    {
        get => _optionsFingerprint;
        init => _optionsFingerprint = string.IsNullOrWhiteSpace(value)
            ? DefaultOptionsFingerprint
            : value.ToLowerInvariant();
    }

    public ExecutionProviderKind Provider { get; init; }

    public int? DeviceId { get; init; }

    /// <summary>True when standalone OpenVINO is configured to execute on the CPU proxy.</summary>
    public bool UseOpenVinoCpuProxy { get; init; }

    /// <summary>
    /// Estimated VRAM footprint of this session in MB. Used for VRAM-budget eviction; not part
    /// of key equality.
    /// </summary>
    public long EstimatedVramMb { get; init; } = 0;

    public bool Equals(SessionPoolKey? other) =>
        other is not null &&
        string.Equals(_engineFamily, other._engineFamily, StringComparison.Ordinal) &&
        string.Equals(_modelId, other._modelId, StringComparison.Ordinal) &&
        string.Equals(_variant, other._variant, StringComparison.Ordinal) &&
        string.Equals(_pathHash, other._pathHash, StringComparison.Ordinal) &&
        string.Equals(_modelContentHash, other._modelContentHash, StringComparison.Ordinal) &&
        string.Equals(_graphRole, other._graphRole, StringComparison.Ordinal) &&
        string.Equals(_optionsFingerprint, other._optionsFingerprint, StringComparison.Ordinal) &&
        Provider == other.Provider &&
        DeviceId == other.DeviceId &&
        UseOpenVinoCpuProxy == other.UseOpenVinoCpuProxy;

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(_engineFamily, StringComparer.Ordinal);
        hash.Add(_modelId, StringComparer.Ordinal);
        hash.Add(_variant, StringComparer.Ordinal);
        hash.Add(_pathHash, StringComparer.Ordinal);
        hash.Add(_modelContentHash, StringComparer.Ordinal);
        hash.Add(_graphRole, StringComparer.Ordinal);
        hash.Add(_optionsFingerprint, StringComparer.Ordinal);
        hash.Add(Provider);
        hash.Add(DeviceId);
        hash.Add(UseOpenVinoCpuProxy);
        return hash.ToHashCode();
    }

    /// <summary>
    /// Reservation used when a key carries no <see cref="EstimatedVramMb"/> and when the
    /// model file is missing or unreadable, so admission stays pessimistic.
    /// </summary>
    public const long DefaultEstimatedVramMb = 256;

    /// <summary>
    /// Resident memory per weight byte, in percent, for <paramref name="provider"/>.
    /// </summary>
    /// <remarks>
    /// TensorRT RTX and DirectML hold each session's weights roughly once, both measured on a
    /// 12 GB RTX 5070:
    /// <list type="bullet">
    /// <item>TensorRT RTX: MADLAD-400 3B <c>trt_rtx_mixed_fp16_fp32</c> encoder + decoder
    /// (6.55 GB of external weights) raised process GPU usage by ~8 GB, about 1.2× weights.</item>
    /// <item>DirectML (Windows ML 2.4.89, ORT 1.27.1): qwen3-asr-0.6b encoder + decoder-init +
    /// decoder-step (711 MiB inline plus a 2867 MiB sidecar each decoder graph loads, 6445 MiB
    /// per-session weights) raised process GPU usage by 6526 MiB, about 1.0×.</item>
    /// </list>
    /// 125% plus the per-graph allowance covers both. Every other provider is unmeasured and
    /// keeps the conservative 2× (weights, initialization and pre-packing copies, activation
    /// slack). On Windows, process-GPU observation still charges any real usage beyond an
    /// estimate.
    /// <para>
    /// Set <paramref name="providerMayFallBack"/> when session creation can still fall back
    /// from TensorRT RTX to another provider after the key is admitted (the pooled
    /// single-session TRT init fallback, which can end on CPU). The admitted reservation cannot
    /// follow the session to its effective provider, so it uses the conservative factor.
    /// </para>
    /// </remarks>
    internal static int ResidentPercentOfWeights(ExecutionProviderKind provider, bool providerMayFallBack = false) =>
        !providerMayFallBack && provider is ExecutionProviderKind.TensorRTRtx or ExecutionProviderKind.DirectMl
            ? 125
            : 200;

    /// <summary>
    /// Resident estimate for <paramref name="weightBytes"/> of model weights on
    /// <paramref name="provider"/>: weights scaled by <see cref="ResidentPercentOfWeights"/>,
    /// plus 128 MB per graph for runtime and activation overhead.
    /// </summary>
    internal static long EstimateFromWeightBytes(
        long weightBytes,
        ExecutionProviderKind provider,
        bool providerMayFallBack = false)
    {
        long weightMb = weightBytes / (1024L * 1024L);
        return (weightMb * ResidentPercentOfWeights(provider, providerMayFallBack) / 100L) + 128L;
    }

    /// <summary>
    /// Estimates a session's resident footprint from the <c>.onnx</c> file plus every
    /// external-data sidecar its tensors reference (see <see cref="OnnxExternalDataSidecars"/>).
    /// </summary>
    internal static long EstimateVramMb(
        string? modelPath,
        ExecutionProviderKind provider,
        bool providerMayFallBack = false)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            return DefaultEstimatedVramMb;
        }

        try
        {
            var info = new FileInfo(modelPath);
            if (info.Exists)
            {
                long weightBytes = info.Length + OnnxExternalDataSidecars.GetTotalBytes(modelPath);
                return EstimateFromWeightBytes(weightBytes, provider, providerMayFallBack);
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
        {
            // Log file access failure to aid debugging VRAM estimation issues (model file inaccessible,
            // malformed path). Falling back to the pessimistic DefaultEstimatedVramMb
            // (256 MB) ensures admission accounting stays conservative but may trigger
            // unnecessary evictions if the actual model is smaller.
            // File is inaccessible or the path is malformed; fall back to the pessimistic default.
            System.Diagnostics.Trace.TraceWarning(
                $"SessionPoolKey: failed to estimate VRAM for '{modelPath}': {ex.Message}");
        }

        return DefaultEstimatedVramMb;
    }

    /// <summary>
    /// Content-digest cache keyed by normalised full path. An entry is only trusted while
    /// the file's length and <see cref="FileInfo.LastWriteTimeUtc"/> still match, so a
    /// same-path replacement rehashes once instead of reusing a stale digest — and repeated
    /// pool-key construction over an unchanged multi-GB model never re-reads it.
    /// </summary>
    private static readonly ConcurrentDictionary<string, (long Length, long WriteTicks, string Hash)> ModelContentHashCache =
        new(StringComparer.Ordinal);
    private static readonly ConcurrentDictionary<string, long> ModelContentHashGenerations = new(StringComparer.Ordinal);

    /// <summary>
    /// Streams a SHA-256 over the model file's content and returns lowercase hex, or
    /// <see langword="null"/> when the path is missing, malformed, or inaccessible (the key
    /// then falls back to path identity). Results are cached per
    /// (path, length, last-write ticks); if the file's metadata changes while it is being
    /// hashed the digest is retaken once under the fresh metadata before caching, and a
    /// file that is still mutating after that retry yields <see langword="null"/> rather
    /// than pinning pool identity to an inconsistent snapshot.
    /// </summary>
    internal static string? HashModelContent(string? modelPath)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            return null;
        }

        string cacheKey;
        try
        {
            cacheKey = Path.GetFullPath(modelPath);
            if (OperatingSystem.IsWindows() && !PreserveWindowsPathCase())
            {
                cacheKey = cacheKey.ToUpperInvariant();
            }
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"SessionPoolKey: malformed model path '{modelPath}': {ex.Message}");
            return null;
        }

        for (int attempt = 0; attempt < 2; attempt++)
        {
            long generation = ModelContentHashGenerations.GetOrAdd(cacheKey, 0);
            FileInfo info;
            long length;
            long writeTicks;
            try
            {
                info = new FileInfo(modelPath);
                if (!info.Exists)
                {
                    return null;
                }

                // Read metadata under the same guard: the file can disappear between
                // Exists and Length (FileNotFoundException), or be inaccessible.
                length = info.Length;
                writeTicks = info.LastWriteTimeUtc.Ticks;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"SessionPoolKey: failed to inspect model file '{modelPath}': {ex.Message}");
                return null;
            }

            (ContentHashLookup lookup, string? resolved) = attempt == 0
                ? TryResolveContentHash(modelPath, cacheKey, new ModelFileSnapshot(info, length, writeTicks, generation)) : default;
            if (lookup != ContentHashLookup.NotFound)
            {
                return resolved;
            }

            string hash;
            try
            {
                using FileStream stream = File.OpenRead(modelPath);
                hash = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FileNotFoundException or DirectoryNotFoundException)
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"SessionPoolKey: failed to hash model file '{modelPath}': {ex.Message}");
                return null;
            }

            // If the file changed under the stream, the digest describes an inconsistent
            // snapshot — retry once under the fresh metadata before trusting it.
            bool unchanged;
            try
            {
                info.Refresh();
                unchanged = info.Exists
                    && info.Length == length
                    && info.LastWriteTimeUtc.Ticks == writeTicks;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
            {
                return null;
            }

            if (unchanged)
            {
                if (ModelContentHashGenerations.GetOrAdd(cacheKey, 0) == generation)
                {
                    ModelContentHashCache[cacheKey] = (length, writeTicks, hash);
                    if (ModelContentHashGenerations.GetOrAdd(cacheKey, 0) == generation)
                    {
                        return hash;
                    }

                    ModelContentHashCache.TryRemove(cacheKey, out _);
                }
            }
        }

        // The file was still mutating after the second hash — conservative fallback: do
        // not pin pool identity to an inconsistent snapshot; path identity still applies.
        return null;
    }

    internal static void InvalidateModelContentHash(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        string cacheKey = Path.GetFullPath(modelPath);
        if (OperatingSystem.IsWindows() && !PreserveWindowsPathCase())
        {
            cacheKey = cacheKey.ToUpperInvariant();
        }

        ModelContentHashCache.TryRemove(cacheKey, out _);
        ModelContentHashGenerations.AddOrUpdate(cacheKey, 1, static (_, generation) => generation + 1);
    }

    /// <summary>Outcome of resolving a model content digest from the cache or a shared provider.</summary>
    private enum ContentHashLookup
    {
        /// <summary>No digest is available; the caller must hash the model file itself.</summary>
        NotFound,

        /// <summary>A digest was resolved and is safe to use and to cache.</summary>
        Found,

        /// <summary>The model file's metadata could not be re-read, so no digest can be trusted.</summary>
        Unreadable,
    }

    /// <summary>
    /// The model file metadata a content digest is resolved against: the file handle used to
    /// revalidate the file, the length and write time observed before hashing, and the cache
    /// generation the digest must belong to. Both the synchronous and asynchronous paths track
    /// generations, so an invalidation racing the lookup can never resurrect a stale digest.
    /// </summary>
    private readonly struct ModelFileSnapshot(FileInfo info, long length, long writeTicks, long generation = 0)
    {
        public FileInfo Info { get; } = info;
        public long Length { get; } = length;
        public long WriteTicks { get; } = writeTicks;
        public long Generation { get; } = generation;
    }

    /// <summary>
    /// Resolves the model content digest from the in-memory cache or — when a shared
    /// fingerprint provider is configured — from that provider, so a multi-GB model is not
    /// re-hashed from disk on every pool-key construction. A shared digest is accepted only
    /// while the file's length and write time still match the metadata the caller observed,
    /// so a same-path replacement never reuses a stale digest.
    /// </summary>
    private static (ContentHashLookup Lookup, string? Hash) TryResolveContentHash(
        string modelPath,
        string cacheKey,
        ModelFileSnapshot snapshot)
    {
        if (TryGetCachedHash(cacheKey, snapshot.Length, snapshot.WriteTicks, out string? cachedHash) &&
            ModelContentHashGenerations.GetOrAdd(cacheKey, 0) == snapshot.Generation)
        {
            return (ContentHashLookup.Found, cachedHash);
        }

        if (SessionPoolContentHashProvider.Current is not { } sharedHasher)
        {
            return (ContentHashLookup.NotFound, null);
        }

        string? sharedHash;
        try
        {
            sharedHash = sharedHasher(modelPath, CancellationToken.None).ConfigureAwait(false).GetAwaiter().GetResult();
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FileNotFoundException or DirectoryNotFoundException)
        {
            System.Diagnostics.Trace.TraceWarning($"Shared model fingerprint failed for '{modelPath}': {ex.Message}");
            return (ContentHashLookup.NotFound, null);
        }

        if (string.IsNullOrWhiteSpace(sharedHash))
        {
            return (ContentHashLookup.NotFound, null);
        }

        if (!TryReadUnchangedMetadata(snapshot.Info, snapshot.Length, snapshot.WriteTicks, out bool metadataUnreadable))
        {
            return (metadataUnreadable ? ContentHashLookup.Unreadable : ContentHashLookup.NotFound, null);
        }

        if (ModelContentHashGenerations.GetOrAdd(cacheKey, 0) != snapshot.Generation)
        {
            return (ContentHashLookup.NotFound, null);
        }

        ModelContentHashCache[cacheKey] = (snapshot.Length, snapshot.WriteTicks, sharedHash);
        return (ContentHashLookup.Found, sharedHash);
    }

    /// <summary>
    /// Asynchronous counterpart of <see cref="TryResolveContentHash"/>. The cached digest is
    /// only reused while the cache generation is unchanged, so an invalidation racing the
    /// lookup can never resurrect a stale digest.
    /// </summary>
    private static async Task<(ContentHashLookup Lookup, string? Hash)> TryResolveContentHashAsync(
        string modelPath,
        string cacheKey,
        ModelFileSnapshot snapshot,
        CancellationToken cancellationToken)
    {
        if (TryGetCachedHash(cacheKey, snapshot.Length, snapshot.WriteTicks, out string? cachedHash) &&
            ModelContentHashGenerations.GetOrAdd(cacheKey, 0) == snapshot.Generation)
        {
            return (ContentHashLookup.Found, cachedHash);
        }

        if (SessionPoolContentHashProvider.Current is not { } sharedHasher)
        {
            return (ContentHashLookup.NotFound, null);
        }

        string? sharedHash = await sharedHasher(modelPath, cancellationToken).ConfigureAwait(false);
        if (string.IsNullOrWhiteSpace(sharedHash))
        {
            return (ContentHashLookup.NotFound, null);
        }

        if (!TryReadUnchangedMetadata(snapshot.Info, snapshot.Length, snapshot.WriteTicks, out bool metadataUnreadable))
        {
            return (metadataUnreadable ? ContentHashLookup.Unreadable : ContentHashLookup.NotFound, null);
        }

        if (ModelContentHashGenerations.GetOrAdd(cacheKey, 0) != snapshot.Generation)
        {
            return (ContentHashLookup.NotFound, null);
        }

        ModelContentHashCache[cacheKey] = (snapshot.Length, snapshot.WriteTicks, sharedHash);
        return (ContentHashLookup.Found, sharedHash);
    }

    private static bool TryGetCachedHash(string cacheKey, long length, long writeTicks, out string? hash)
    {
        hash = null;
        if (!ModelContentHashCache.TryGetValue(cacheKey, out var cached))
        {
            return false;
        }

        if (cached.Length != length)
        {
            return false;
        }

        if (cached.WriteTicks != writeTicks)
        {
            return false;
        }

        hash = cached.Hash;
        return true;
    }

    /// <summary>
    /// Re-reads the model file's metadata and reports whether it still matches the metadata
    /// the caller observed. When the metadata cannot be re-read at all,
    /// <paramref name="unreadable"/> is set so the caller can refuse to trust any digest.
    /// </summary>
    private static bool TryReadUnchangedMetadata(FileInfo info, long length, long writeTicks, out bool unreadable)
    {
        unreadable = false;
        try
        {
            info.Refresh();
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
        {
            unreadable = true;
            return false;
        }

        if (!info.Exists)
        {
            return false;
        }

        if (info.Length != length)
        {
            return false;
        }

        return info.LastWriteTimeUtc.Ticks == writeTicks;
    }

    /// <summary>
    /// One graph's pooled-key inputs for <see cref="CreateManyAsync"/>.
    /// </summary>
    internal sealed record PooledKeyRequest(
        string EngineFamily,
        string? ModelId,
        string? Variant,
        ExecutionProviderKind Provider,
        string ModelPath,
        int? DeviceId,
        string GraphRole,
        string? OptionsFingerprint);

    /// <summary>
    /// Builds pooled keys for every graph in <paramref name="requests"/> concurrently
    /// (C3): each graph's content hash streams independently via
    /// <see cref="HashModelContentAsync"/>, so no graph's full model bytes are buffered
    /// and the (path, length, mtime) digest-cache semantics are unchanged. Result order
    /// matches request order. <paramref name="modelContentHasher"/> is a test seam that
    /// replaces the streaming file hash so concurrency tests can gate hash completion
    /// with synchronization instead of wall-clock timing; production passes null.
    /// </summary>
    internal static async Task<SessionPoolKey[]> CreateManyAsync(
        IReadOnlyList<PooledKeyRequest> requests,
        CancellationToken cancellationToken,
        Func<string, CancellationToken, Task<string?>>? modelContentHasher = null)
    {
        ArgumentNullException.ThrowIfNull(requests);
        var tasks = new Task<SessionPoolKey>[requests.Count];
        for (int i = 0; i < requests.Count; i++)
        {
            PooledKeyRequest request = requests[i];
            tasks[i] = CreateAsync(
                request.EngineFamily, request.ModelId, request.Variant, request.Provider,
                request.ModelPath, request.DeviceId, request.GraphRole, request.OptionsFingerprint,
                cancellationToken, modelContentHasher);
        }

        return await Task.WhenAll(tasks).ConfigureAwait(false);
    }

    internal static async Task<SessionPoolKey> CreateAsync(
        string engineFamily,
        string? modelId,
        string? variant,
        ExecutionProviderKind provider,
        string modelPath,
        int? deviceId,
        string graphRole,
        string? optionsFingerprint,
        CancellationToken cancellationToken,
        Func<string, CancellationToken, Task<string?>>? modelContentHasher = null)
    {
        cancellationToken.ThrowIfCancellationRequested();
        string? contentHash = modelContentHasher is not null
            ? await modelContentHasher(modelPath, cancellationToken).ConfigureAwait(false)
            : await HashModelContentAsync(modelPath, cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();

        return new SessionPoolKey(
            engineFamily,
            modelId,
            variant,
            provider,
            HashPath(modelPath),
            deviceId,
            graphRole,
            optionsFingerprint)
        {
            EstimatedVramMb = EstimateVramMb(modelPath, provider),
            ModelContentHash = contentHash,
        };
    }

    internal static async Task<string?> HashModelContentAsync(
        string? modelPath,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            return null;
        }

        string cacheKey;
        try
        {
            cacheKey = Path.GetFullPath(modelPath);
            if (OperatingSystem.IsWindows() && !PreserveWindowsPathCase())
            {
                cacheKey = cacheKey.ToUpperInvariant();
            }
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
        {
            return null;
        }

        for (int attempt = 0; attempt < 2; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            long generation = ModelContentHashGenerations.GetOrAdd(cacheKey, 0);
            FileInfo info;
            long length;
            long writeTicks;
            try
            {
                info = new FileInfo(modelPath);
                if (!info.Exists)
                {
                    return null;
                }

                length = info.Length;
                writeTicks = info.LastWriteTimeUtc.Ticks;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
            {
                return null;
            }

            (ContentHashLookup lookup, string? resolved) = attempt == 0
                ? await TryResolveContentHashAsync(modelPath, cacheKey, new ModelFileSnapshot(info, length, writeTicks, generation), cancellationToken).ConfigureAwait(false) : default;
            if (lookup != ContentHashLookup.NotFound)
            {
                return resolved;
            }

            string hash;
            try
            {
                await using var stream = new FileStream(
                    modelPath,
                    FileMode.Open,
                    FileAccess.Read,
                    FileShare.Read,
                    bufferSize: 1024 * 1024,
                    FileOptions.Asynchronous | FileOptions.SequentialScan);
                byte[] digest = await SHA256.HashDataAsync(stream, cancellationToken).ConfigureAwait(false);
                hash = Convert.ToHexString(digest).ToLowerInvariant();
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
            {
                return null;
            }

            bool unchanged;
            try
            {
                info.Refresh();
                unchanged = info.Exists && info.Length == length && info.LastWriteTimeUtc.Ticks == writeTicks;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
            {
                return null;
            }

            if (unchanged)
            {
                if (ModelContentHashGenerations.GetOrAdd(cacheKey, 0) == generation)
                {
                    ModelContentHashCache[cacheKey] = (length, writeTicks, hash);
                    if (ModelContentHashGenerations.GetOrAdd(cacheKey, 0) == generation)
                    {
                        return hash;
                    }

                    ModelContentHashCache.TryRemove(cacheKey, out _);
                }
            }
        }

        return null;
    }

    /// <summary>Builds a key for a single-session model (graph role = "default").</summary>
    public static SessionPoolKey ForSingle(
        string engineFamily,
        string modelPath,
        ExecutionProviderKind provider,
        string? modelId = null,
        string? variant = null,
        int? deviceId = null,
        string? optionsFingerprint = null) =>
        new(engineFamily, modelId, variant, provider, HashPath(modelPath), deviceId, "default", optionsFingerprint)
        {
            EstimatedVramMb = EstimateVramMb(modelPath, provider),
            ModelContentHash = HashModelContent(modelPath),
        };

    /// <summary>Builds an encoder key for a dual-session model.</summary>
    public static SessionPoolKey ForEncoder(
        string engineFamily,
        string encoderPath,
        ExecutionProviderKind provider,
        string? modelId = null,
        string? variant = null,
        int? deviceId = null,
        string? optionsFingerprint = null) =>
        new(engineFamily, modelId, variant, provider, HashPath(encoderPath), deviceId, "encoder", optionsFingerprint)
        {
            EstimatedVramMb = EstimateVramMb(encoderPath, provider),
            ModelContentHash = HashModelContent(encoderPath),
        };

    /// <summary>Builds a decoder key for a dual-session model.</summary>
    public static SessionPoolKey ForDecoder(
        string engineFamily,
        string decoderPath,
        ExecutionProviderKind provider,
        string? modelId = null,
        string? variant = null,
        int? deviceId = null,
        string? optionsFingerprint = null) =>
        new(engineFamily, modelId, variant, provider, HashPath(decoderPath), deviceId, "decoder", optionsFingerprint)
        {
            EstimatedVramMb = EstimateVramMb(decoderPath, provider),
            ModelContentHash = HashModelContent(decoderPath),
        };

    public static SessionPoolKey ForDecoderInit(
        string engineFamily,
        string decoderInitPath,
        ExecutionProviderKind provider,
        string? modelId = null,
        string? variant = null,
        int? deviceId = null,
        string? optionsFingerprint = null) =>
        new(engineFamily, modelId, variant, provider, HashPath(decoderInitPath), deviceId, "decoder-init", optionsFingerprint)
        {
            EstimatedVramMb = EstimateVramMb(decoderInitPath, provider),
            ModelContentHash = HashModelContent(decoderInitPath),
        };

    public static SessionPoolKey ForDecoderStep(
        string engineFamily,
        string decoderStepPath,
        ExecutionProviderKind provider,
        string? modelId = null,
        string? variant = null,
        int? deviceId = null,
        string? optionsFingerprint = null) =>
        new(engineFamily, modelId, variant, provider, HashPath(decoderStepPath), deviceId, "decoder-step", optionsFingerprint)
        {
            EstimatedVramMb = EstimateVramMb(decoderStepPath, provider),
            ModelContentHash = HashModelContent(decoderStepPath),
        };

    // ── Chatterbox four-graph helpers ─────────────────────────────────────────

    /// <summary>
    /// Builds a speech-encoder key for the Chatterbox model
    /// (graph role = <c>"speech-encoder"</c>).
    /// The speech encoder processes the reference audio clip to produce conditioning vectors.
    /// </summary>
    public static SessionPoolKey ForChatterboxSpeechEncoder(
        string modelPath,
        ExecutionProviderKind provider,
        string? modelId = null,
        string? variant = null,
        int? deviceId = null) =>
        new(ChatterboxEngineFamily, modelId, variant, provider, HashPath(modelPath), deviceId, "speech-encoder")
        {
            EstimatedVramMb = EstimateVramMb(modelPath, provider),
            ModelContentHash = HashModelContent(modelPath),
        };

    /// <summary>
    /// Builds an embed-tokens key for the Chatterbox model
    /// (graph role = <c>"embed-tokens"</c>).
    /// The embedding graph converts text token IDs into dense embedding vectors.
    /// </summary>
    public static SessionPoolKey ForChatterboxEmbedTokens(
        string modelPath,
        ExecutionProviderKind provider,
        string? modelId = null,
        string? variant = null,
        int? deviceId = null) =>
        new(ChatterboxEngineFamily, modelId, variant, provider, HashPath(modelPath), deviceId, "embed-tokens")
        {
            EstimatedVramMb = EstimateVramMb(modelPath, provider),
            ModelContentHash = HashModelContent(modelPath),
        };

    /// <summary>
    /// Builds a language-model key for the Chatterbox model
    /// (graph role = <c>"lm"</c>).
    /// The LM auto-regressively generates speech tokens from text and conditioning.
    /// </summary>
    public static SessionPoolKey ForChatterboxLanguageModel(
        string modelPath,
        ExecutionProviderKind provider,
        string? modelId = null,
        string? variant = null,
        int? deviceId = null) =>
        new(ChatterboxEngineFamily, modelId, variant, provider, HashPath(modelPath), deviceId, "lm")
        {
            EstimatedVramMb = EstimateVramMb(modelPath, provider),
            ModelContentHash = HashModelContent(modelPath),
        };

    /// <summary>
    /// Builds a conditional-decoder key for the Chatterbox model
    /// (graph role = <c>"conditional-decoder"</c>).
    /// The conditional decoder converts speech tokens into audio waveform samples.
    /// </summary>
    public static SessionPoolKey ForChatterboxConditionalDecoder(
        string modelPath,
        ExecutionProviderKind provider,
        string? modelId = null,
        string? variant = null,
        int? deviceId = null) =>
        new(ChatterboxEngineFamily, modelId, variant, provider, HashPath(modelPath), deviceId, "conditional-decoder")
        {
            EstimatedVramMb = EstimateVramMb(modelPath, provider),
            ModelContentHash = HashModelContent(modelPath),
        };

    /// <summary>Engine-family constant used by the Chatterbox factory helpers.</summary>
    private const string ChatterboxEngineFamily = "chatterbox";

    // ── LatentSync four-graph helpers ─────────────────────────────────────────

    public static SessionPoolKey ForLatentSyncUNet(
        string modelPath,
        ExecutionProviderKind provider,
        string? modelId = null,
        string? variant = null,
        int? deviceId = null) =>
        new(LatentSyncEngineFamily, modelId, variant, provider, HashPath(modelPath), deviceId, "unet")
        {
            EstimatedVramMb = EstimateVramMb(modelPath, provider),
            ModelContentHash = HashModelContent(modelPath),
        };

    public static SessionPoolKey ForLatentSyncVaeEncoder(
        string modelPath,
        ExecutionProviderKind provider,
        string? modelId = null,
        string? variant = null,
        int? deviceId = null) =>
        new(LatentSyncEngineFamily, modelId, variant, provider, HashPath(modelPath), deviceId, "vae-encoder")
        {
            EstimatedVramMb = EstimateVramMb(modelPath, provider),
            ModelContentHash = HashModelContent(modelPath),
        };

    public static SessionPoolKey ForLatentSyncVaeDecoder(
        string modelPath,
        ExecutionProviderKind provider,
        string? modelId = null,
        string? variant = null,
        int? deviceId = null) =>
        new(LatentSyncEngineFamily, modelId, variant, provider, HashPath(modelPath), deviceId, "vae-decoder")
        {
            EstimatedVramMb = EstimateVramMb(modelPath, provider),
            ModelContentHash = HashModelContent(modelPath),
        };

    public static SessionPoolKey ForLatentSyncWhisperEncoder(
        string modelPath,
        ExecutionProviderKind provider,
        string? modelId = null,
        string? variant = null,
        int? deviceId = null) =>
        new(LatentSyncEngineFamily, modelId, variant, provider, HashPath(modelPath), deviceId, "whisper-encoder")
        {
            EstimatedVramMb = EstimateVramMb(modelPath, provider),
            ModelContentHash = HashModelContent(modelPath),
        };

    private const string LatentSyncEngineFamily = "latentsync-diffusion";

    /// <summary>
    /// Total order for multi-graph bundle acquisition. Callers must take bundle leases in
    /// this order or encoder/decoder pairs can deadlock against each other.
    /// </summary>
    public static IComparer<SessionPoolKey> StableComparer { get; } = new StableKeyComparer();

    private sealed class StableKeyComparer : IComparer<SessionPoolKey>
    {
        public int Compare(SessionPoolKey? x, SessionPoolKey? y)
        {
            if (ReferenceEquals(x, y))
            {
                return 0;
            }

            if (x is null)
            {
                return -1;
            }

            if (y is null)
            {
                return 1;
            }

            int c = CompareGraphIdentity(x, y);
            if (c != 0)
            {
                return c;
            }

            c = string.CompareOrdinal(x.OptionsFingerprint, y.OptionsFingerprint);
            if (c != 0)
            {
                return c;
            }

            c = x.Provider.CompareTo(y.Provider);
            return c != 0 ? c : (x.DeviceId ?? -1).CompareTo(y.DeviceId ?? -1);
        }

        private static int CompareGraphIdentity(SessionPoolKey x, SessionPoolKey y)
        {
            int c = string.CompareOrdinal(x.EngineFamily, y.EngineFamily);
            if (c != 0)
            {
                return c;
            }

            c = string.CompareOrdinal(x.GraphRole, y.GraphRole);
            if (c != 0)
            {
                return c;
            }

            c = string.CompareOrdinal(x.PathHash, y.PathHash);
            if (c != 0)
            {
                return c;
            }

            c = string.CompareOrdinal(x.ModelContentHash ?? string.Empty, y.ModelContentHash ?? string.Empty);
            if (c != 0)
            {
                return c;
            }

            c = string.CompareOrdinal(x.ModelId ?? string.Empty, y.ModelId ?? string.Empty);
            if (c != 0)
            {
                return c;
            }

            c = string.CompareOrdinal(x.Variant ?? string.Empty, y.Variant ?? string.Empty);
            if (c != 0)
            {
                return c;
            }

            return c;
        }
    }

    /// <summary>
    /// Computes a stable, lowercase hex SHA-256 hash of the given model file path.
    /// On Windows, paths are typically case-insensitive, so casing is normalised before
    /// hashing by default; <c>C:\Models\Model.onnx</c> and <c>c:\models\model.onnx</c>
    /// resolve to the same pool slot.
    /// Windows also supports per-directory case sensitivity (via
    /// <c>fsutil file setCaseSensitiveInfo</c> or WSL2 volume mounts). Set the
    /// <c>Trackdub.Inference.Onnx.SessionPoolKey.PreserveWindowsPathCase</c>
    /// <see cref="AppContext"/> switch to <see langword="true"/> before hashing to preserve
    /// Windows path casing in deployments that rely on distinct case-sensitive paths.
    /// On macOS and Linux the path is hashed as-is to preserve case-sensitive path
    /// identity: macOS APFS volumes may be formatted as either case-sensitive or
    /// case-insensitive, so normalising on macOS could conflate distinct files on
    /// case-sensitive volumes.
    /// Note: only casing may be normalised; path separators and relative segments are not
    /// canonicalised.
    /// </summary>
    public static string HashPath(string modelPath)
    {
        ArgumentNullException.ThrowIfNull(modelPath);
        // Normalise case on Windows by default because most Windows model paths are
        // case-insensitive. Deployments using case-sensitive Windows directories can opt out.
        // On macOS and Linux, paths are hashed as provided.
        string normalised = OperatingSystem.IsWindows() && !PreserveWindowsPathCase()
            ? modelPath.ToUpperInvariant()
            : modelPath;
        byte[] bytes = Encoding.UTF8.GetBytes(normalised);
        byte[] hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    /// <summary>
    /// Computes a stable lowercase hash for provider options. Dictionary enumeration order
    /// does not affect the result.
    /// </summary>
    public static string HashOptions(IReadOnlyDictionary<string, string>? options)
    {
        if (options is null || options.Count == 0)
        {
            return DefaultOptionsFingerprint;
        }

        var builder = new StringBuilder();
        foreach ((string key, string value) in options.OrderBy(static pair => pair.Key, StringComparer.Ordinal))
        {
            builder
                .Append(key.Length)
                .Append(':')
                .Append(key)
                .Append('=')
                .Append(value.Length)
                .Append(':')
                .Append(value)
                .Append(';');
        }

        byte[] bytes = Encoding.UTF8.GetBytes(builder.ToString());
        byte[] hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static bool PreserveWindowsPathCase()
    {
        AppContext.TryGetSwitch(
            "Trackdub.Inference.Onnx.SessionPoolKey.PreserveWindowsPathCase",
            out bool preserveWindowsPathCase);
        return preserveWindowsPathCase;
    }
}
