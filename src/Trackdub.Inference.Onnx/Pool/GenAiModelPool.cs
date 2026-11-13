using System.Collections.Concurrent;
using System.Security.Cryptography;
using Microsoft.ML.OnnxRuntimeGenAI;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Domain;
using Trackdub.Inference.Onnx;
using Trackdub.Inference.Onnx.Runtime;

namespace Trackdub.Inference.Onnx.Pool;

/// <summary>
/// Residency identity for one ORT GenAI model directory. The model root participates as a
/// lowercase SHA-256 digest of its normalized full path plus content digests for the config
/// and model artifacts — the raw path is kept out of equality and diagnostics. The raw path
/// is retained on <see cref="ModelRootPath"/> for native construction only.
/// </summary>
internal sealed class GenAiModelKey : IEquatable<GenAiModelKey>
{
    internal GenAiModelKey(
        string modelRootIdentity,
        string? modelId,
        string? variant,
        ExecutionProviderKind provider,
        int? deviceId,
        long estimatedMemoryMb,
        string modelRootPath,
        string? modelRevisionHash = null,
        bool useOpenVinoCpuProxy = false)
    {
        ModelRootIdentity = modelRootIdentity;
        ModelId = modelId;
        Variant = variant;
        Provider = provider;
        DeviceId = deviceId;
        EstimatedMemoryMb = estimatedMemoryMb;
        ModelRootPath = modelRootPath;
        ModelRevisionHash = modelRevisionHash;
        UseOpenVinoCpuProxy = useOpenVinoCpuProxy;
    }

    public string ModelRootIdentity { get; }
    public string? ModelId { get; }
    public string? Variant { get; }
    public ExecutionProviderKind Provider { get; }
    public int? DeviceId { get; }
    public long EstimatedMemoryMb { get; }

    /// <summary>Planner-resolved model revision/content digest; part of identity.</summary>
    public string? ModelRevisionHash { get; }

    public bool UseOpenVinoCpuProxy { get; }

    /// <summary>Normalized full model-root path; used only for native construction.</summary>
    internal string ModelRootPath { get; }

    public static GenAiModelKey Create(
        string modelRootPath,
        ExecutionProviderKind provider,
        string? modelId = null,
        string? variant = null,
        int? deviceId = null,
        string? modelRevisionHash = null,
        string? modelArtifactsHash = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelRootPath);

        string normalized = Path.GetFullPath(modelRootPath.Trim());
        string pathHash = SessionPoolKey.HashPath(normalized);
        string? configHash = SessionPoolKey.HashModelContent(
            Path.Join(normalized, "genai_config.json"));
        // The planner's resolved revision digest joins the identity so a same-path weights
        // replacement with an unchanged genai_config.json cannot reuse stale native state.
        string? revision = NormalizeToken(modelRevisionHash);
        string identity = Convert.ToHexString(
            SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(
                $"{pathHash}|{configHash ?? "-"}|{revision ?? "-"}|{NormalizeToken(modelArtifactsHash) ?? "-"}")))
            .ToLowerInvariant();

        return new GenAiModelKey(
            identity,
            NormalizeToken(modelId),
            NormalizeToken(variant),
            provider,
            deviceId,
            EstimateMemoryMb(normalized),
            normalized,
            revision,
            provider is ExecutionProviderKind.OpenVino && OnnxExecutionSessionFactory.UseOpenVinoCpuProxy);
    }

    public static async Task<GenAiModelKey> CreateAsync(
        string modelRootPath,
        ExecutionProviderKind provider,
        string? modelId = null,
        string? variant = null,
        int? deviceId = null,
        string? modelRevisionHash = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelRootPath);
        string normalizedRoot = Path.GetFullPath(modelRootPath.Trim());
        string artifactHash = await HashModelArtifactsAsync(normalizedRoot, cancellationToken).ConfigureAwait(false);

        return Create(
            normalizedRoot,
            provider,
            modelId,
            variant,
            deviceId,
            modelRevisionHash,
            artifactHash);
    }

    private static async Task<string> HashModelArtifactsAsync(
        string modelRootPath,
        CancellationToken cancellationToken)
    {
        if (!Directory.Exists(modelRootPath))
        {
            return "missing";
        }

        var artifacts = new List<(string RelativePath, string ContentHash)>();
        foreach (string path in EnumerateModelFiles(modelRootPath).Order(StringComparer.Ordinal))
        {
            cancellationToken.ThrowIfCancellationRequested();
            string? contentHash = await SessionPoolKey.HashModelContentAsync(path, cancellationToken)
                .ConfigureAwait(false);
            artifacts.Add((Path.GetRelativePath(modelRootPath, path), contentHash ?? "unreadable"));
        }

        using var identity = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach ((string relativePath, string contentHash) in artifacts)
        {
            byte[] artifactIdentity = System.Text.Encoding.UTF8.GetBytes($"{relativePath}\0{contentHash}\n");
            identity.AppendData(artifactIdentity);
        }

        return Convert.ToHexString(identity.GetHashAndReset()).ToLowerInvariant();
    }

    /// <summary>
    /// Conservative resident-footprint estimate: file bytes under the model root doubled for
    /// runtime expansion plus a fixed overhead, floored at 1024 MB. Missing, unreadable, or
    /// empty roots take the floor. Never loads model bytes.
    /// </summary>
    internal static long EstimateMemoryMb(string? modelRootPath)
    {
        const long floorMb = 1024;
        const long overheadMb = 256;
        try
        {
            if (string.IsNullOrWhiteSpace(modelRootPath) || !Directory.Exists(modelRootPath))
            {
                return floorMb;
            }

            long totalBytes = 0;
            foreach (string path in EnumerateModelFiles(modelRootPath))
            {
                try
                {
                    long length = new FileInfo(path).Length;
                    totalBytes = length > long.MaxValue - totalBytes
                        ? long.MaxValue
                        : totalBytes + length;
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
                {
                    // File vanished or became unreadable mid-scan — skip it.
                }
            }

            long totalMb = totalBytes / (1024 * 1024);
            long estimate = totalMb > (long.MaxValue - overheadMb) / 2
                ? long.MaxValue
                : (2 * totalMb) + overheadMb;
            return Math.Max(floorMb, estimate);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
        {
            return floorMb;
        }
    }

    /// <summary>
    /// Files that belong to this model root. A subdirectory with its own <c>genai_config.json</c> is a
    /// sibling variant with a separate root (for example <c>cuda-int4</c> under the CPU variant), so it
    /// must not change this root's identity hash or inflate its memory estimate.
    /// </summary>
    private static IEnumerable<string> EnumerateModelFiles(string modelRootPath)
    {
        foreach (string file in Directory.EnumerateFiles(modelRootPath))
        {
            yield return file;
        }

        foreach (string directory in Directory.EnumerateDirectories(modelRootPath))
        {
            if (File.Exists(Path.Join(directory, "genai_config.json")))
            {
                continue;
            }

            foreach (string file in EnumerateModelFiles(directory))
            {
                yield return file;
            }
        }
    }

    private static string? NormalizeToken(string? value) =>
        string.IsNullOrWhiteSpace(value) ? null : value.Trim().ToLowerInvariant();

    public bool Equals(GenAiModelKey? other) =>
        other is not null
        && ModelRootIdentity == other.ModelRootIdentity
        && ModelId == other.ModelId
        && Variant == other.Variant
        && Provider == other.Provider
        && DeviceId == other.DeviceId
        && ModelRevisionHash == other.ModelRevisionHash
        && UseOpenVinoCpuProxy == other.UseOpenVinoCpuProxy;

    public override bool Equals(object? obj) => Equals(obj as GenAiModelKey);

    public override int GetHashCode() =>
        HashCode.Combine(
            ModelRootIdentity, ModelId, Variant, Provider, DeviceId, ModelRevisionHash, UseOpenVinoCpuProxy);

    public override string ToString() =>
        $"{Provider}/{DeviceId?.ToString() ?? "-"}/{ModelId ?? "-"}/{Variant ?? "-"}/" +
        $"{(ModelRevisionHash is null ? "-" : ModelRevisionHash[..Math.Min(8, ModelRevisionHash.Length)])}" +
        $"/{ModelRootIdentity[..8]}";
}

/// <summary>
/// Owns one loaded ORT GenAI <see cref="Model"/>. Tests fake this seam — pool lifecycle must
/// never touch <see cref="Model"/> itself.
/// </summary>
internal interface IGenAiModelResource : IDisposable
{
    Model Model { get; }
}

/// <summary>
/// Lazily-created tokenizer slot owned by one pooled GenAI model entry. The tokenizer
/// wraps the entry's native model, so it is created at most once per entry and disposed
/// with the entry (eviction, pool teardown) instead of per engine request.
/// </summary>
internal sealed class GenAiTokenizerCache : IDisposable
{
    private readonly object gate = new();
    private IDisposable? cached;
    private bool disposed;

    public T GetOrCreate<T>(Func<T> factory)
        where T : class, IDisposable
    {
        ArgumentNullException.ThrowIfNull(factory);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (cached is T existing)
            {
                return existing;
            }

            if (cached is not null)
            {
                throw new InvalidOperationException(
                    $"GenAI tokenizer cache already holds {cached.GetType().Name}; cannot also hold {typeof(T).Name}.");
            }

            T created = factory();
            cached = created;
            return created;
        }
    }

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed)
            {
                return;
            }

            disposed = true;
            cached?.Dispose();
            cached = null;
        }
    }
}

/// <summary>
/// Exclusive lease over a resident GenAI model. <see cref="Dispose"/> returns the entry's
/// execution gate and updates the admission LRU marker; the model itself stays resident.
/// </summary>
internal sealed class GenAiModelLease : IDisposable
{
    private readonly Action release;
    private readonly GenAiTokenizerCache tokenizerCache;
    private int disposed;

    internal GenAiModelLease(IGenAiModelResource resource, Action release, GenAiTokenizerCache tokenizerCache)
    {
        ArgumentNullException.ThrowIfNull(tokenizerCache);
        Resource = resource;
        this.release = release;
        this.tokenizerCache = tokenizerCache;
    }

    public Model Model => Resource.Model;

    /// <summary>Tokenizer for the pooled model, created once per pool entry (not per request).</summary>
    public Tokenizer Tokenizer => GetOrCreateCachedTokenizer(static resource => new Tokenizer(resource.Model));

    internal T GetOrCreateCachedTokenizer<T>(Func<IGenAiModelResource, T> factory)
        where T : class, IDisposable
    {
        ArgumentNullException.ThrowIfNull(factory);
        return tokenizerCache.GetOrCreate(() => factory(Resource));
    }

    internal IGenAiModelResource Resource { get; }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0)
        {
            return;
        }

        release();
    }
}

/// <summary>
/// Exclusive, single-flight, evictable residency pool for ORT GenAI <see cref="Model"/>
/// instances. Memory is admitted through <see cref="InferenceSessionPool.ReserveExternalAsync"/>
/// before native load, so GenAI models share the same host RAM / per-device accelerator
/// ceiling as pooled ONNX sessions — and become eviction candidates for ONNX admission once
/// idle. Capacity is hardware-scaled by <see cref="GenAiModelPoolCapacity"/> (1 on a 12 GB
/// card, where only the largest measured model fits with headroom).
/// </summary>
internal sealed class GenAiModelPool : IDisposable
{
    private static readonly Lazy<GenAiModelPool> SharedPool = new(
        static () => new GenAiModelPool(InferenceSessionPool.Shared));

    internal static GenAiModelPool Shared => SharedPool.Value;

    /// <summary>Observability seam for lifecycle tests; mirrors the private flag.</summary>
    internal bool IsDisposed => disposed;

    private readonly InferenceSessionPool admissionPool;
    private readonly Func<GenAiModelKey, CancellationToken, Task<IGenAiModelResource>> factory;
    private readonly int maxEntries;
    private readonly ConcurrentDictionary<GenAiModelKey, Entry> entries = new();
    private readonly ConcurrentDictionary<GenAiModelKey, SemaphoreSlim> createGates = new();
    private readonly SemaphoreSlim creationLock = new(1, 1);
    private TaskCompletionSource<bool> stateChanged = CreateStateSignal();
    private volatile bool disposed;
    private int disposeOnce;

    /// <param name="maxEntries">
    /// Resident entry cap. A positive value always wins; 0 (the default) resolves the
    /// hardware-scaled capacity from the admission pool's budgets via
    /// <see cref="GenAiModelPoolCapacity.Resolve"/>, honouring
    /// <see cref="GenAiModelPoolCapacity.MaxEntriesVariable"/>.
    /// </param>
    internal GenAiModelPool(
        InferenceSessionPool admissionPool,
        Func<GenAiModelKey, CancellationToken, Task<IGenAiModelResource>>? factory = null,
        int maxEntries = 0)
    {
        ArgumentNullException.ThrowIfNull(admissionPool);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxEntries, 0);
        this.admissionPool = admissionPool;
        this.factory = factory ?? CreateNativeResourceAsync;
        this.maxEntries = maxEntries > 0
            ? maxEntries
            : GenAiModelPoolCapacity.Resolve(
                admissionPool.MemoryBudgetMb,
                admissionPool.HostMemoryBudgetMb,
                GenAiModelPoolCapacity.ReadOverride());
    }

    /// <summary>Resolved resident entry cap (explicit or hardware-scaled).</summary>
    internal int MaxEntries => maxEntries;

    internal async Task<GenAiModelLease> GetLeaseAsync(
        GenAiModelKey key,
        CancellationToken cancellationToken) =>
        // Non-warmup acquires never skip, so the core never returns null for them.
        (await GetLeaseCoreAsync(key, cancellationToken, nonEvictingWarmup: false)
            .ConfigureAwait(false))!;

    /// <summary>
    /// Optimization-only warmup acquire: leases the model only when it is already
    /// resident or a free entry slot exists whose admission fits without evicting
    /// anything. Returns <see langword="null"/> (recording
    /// <c>genAiModelWarmupSkipped</c>) when loading would evict a resident idle model —
    /// e.g. the model the currently executing stage is about to lease, whose
    /// <c>LastReleasedTicks</c> is the pool's oldest — or would have to wait for
    /// eviction headroom. The real stage acquire (<see cref="GetLeaseAsync"/>) keeps
    /// the evicting/waiting behaviour.
    /// </summary>
    internal Task<GenAiModelLease?> TryGetWarmupLeaseAsync(
        GenAiModelKey key,
        CancellationToken cancellationToken) =>
        GetLeaseCoreAsync(key, cancellationToken, nonEvictingWarmup: true);

    private async Task<GenAiModelLease?> GetLeaseCoreAsync(
        GenAiModelKey key,
        CancellationToken cancellationToken,
        bool nonEvictingWarmup)
    {
        ArgumentNullException.ThrowIfNull(key);

        // One acquisition request records exactly one classification: a miss counted once
        // at arrival is never re-counted when a later loop iteration lands on the entry.
        bool cacheMissRecorded = false;

        while (true)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            cancellationToken.ThrowIfCancellationRequested();

            if (entries.TryGetValue(key, out Entry? existing))
            {
                using (BenchmarkPhaseCapture.Start("genai-model-wait"))
                {
                    await existing.Gate.WaitAsync(cancellationToken).ConfigureAwait(false);
                }

                if (disposed || existing.IsEvicted)
                {
                    existing.ReleaseGate();
                    continue;
                }

                if (!cacheMissRecorded)
                {
                    BenchmarkPhaseCapture.Increment("genAiModelHit");
                }

                return new GenAiModelLease(existing.Resource, () => ReleaseEntry(existing), existing.TokenizerCache);
            }

            if (!cacheMissRecorded)
            {
                BenchmarkPhaseCapture.Increment("genAiModelMiss");
                cacheMissRecorded = true;
            }

            // Single-flight per key: only one creator runs the factory; waiters queue on the
            // create gate and retry the fast path once the entry publishes.
            SemaphoreSlim createGate = createGates.GetOrAdd(key, static _ => new SemaphoreSlim(1, 1));
            await createGate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                if (entries.ContainsKey(key))
                {
                    continue;
                }

                (GenAiModelLease? lease, bool warmupSkipped) = await TryCreateAndLeaseAsync(
                    key, cancellationToken, nonEvictingWarmup).ConfigureAwait(false);
                if (warmupSkipped)
                {
                    BenchmarkPhaseCapture.Increment("genAiModelWarmupSkipped");
                    return null;
                }

                if (lease is not null)
                {
                    return lease;
                }
            }
            finally
            {
                createGate.Release();
            }
        }
    }

    /// <summary>
    /// Evicts idle entries until capacity is available, then reserves admission, creates the
    /// resource, and publishes the entry — all serialized by <see cref="creationLock"/>.
    /// Returns a null lease when the caller must re-evaluate (key appeared, or the
    /// pool stayed at capacity so it waited a bounded interval for a state change).
    /// In non-evicting warmup mode it instead reports <c>WarmupSkipped</c> the moment
    /// creating would require an eviction (entry-count or admission-budget) or a
    /// capacity wait — warmup never evicts and never parks.
    /// </summary>
    private async Task<(GenAiModelLease? Lease, bool WarmupSkipped)> TryCreateAndLeaseAsync(
        GenAiModelKey key,
        CancellationToken cancellationToken,
        bool nonEvictingWarmup)
    {
        // Captured before each attempt: any state change during the attempt fires the
        // one-shot source, so a needed capacity wait cannot miss a wakeup.
        Task signal = Volatile.Read(ref stateChanged).Task;

        // Phase 1: evict idle locals until under capacity. The evicted entries still count
        // against external admission until disposed, so their disposal must happen before
        // the reservation below — never while still holding creationLock, because
        // ReserveExternalAsync may itself try to evict those entries' external states.
        while (true)
        {
            List<Entry>? evictedToDispose = null;
            await creationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                if (entries.ContainsKey(key))
                {
                    return (null, false); // published meanwhile — caller retries the fast path
                }

                if (nonEvictingWarmup && entries.Count >= maxEntries)
                {
                    // No free slot: creating here would evict the oldest idle resident.
                    return (null, true);
                }

                while (entries.Count >= maxEntries)
                {
                    Entry? evicted = TryEvictOldestIdle();
                    if (evicted is null)
                    {
                        break;
                    }

                    (evictedToDispose ??= new List<Entry>()).Add(evicted);
                }
            }
            finally
            {
                creationLock.Release();
            }

            // Native model disposal is expensive and releases external admission accounting;
            // it runs outside the creation lock.
            if (evictedToDispose is not null)
            {
                foreach (Entry evicted in evictedToDispose)
                {
                    evicted.Dispose();
                }
            }

            if (entries.Count < maxEntries)
            {
                break;
            }

            // Every entry was busy — wait on a state transition rather than going ephemeral.
            try
            {
                await signal.WaitAsync(TimeSpan.FromMilliseconds(50), cancellationToken)
                    .ConfigureAwait(false);
            }
            catch (TimeoutException)
            {
                // Lost-wakeup safety net; the loop re-evaluates.
            }

            signal = Volatile.Read(ref stateChanged).Task;
        }

        // Phase 2: reserve admission, create, publish — serialized under the creation gate.
        await creationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (entries.ContainsKey(key))
            {
                return (null, false);
            }

            // A competing create may have filled the freed slot. Returning a null lease
            // releases creationLock in finally and lets the caller loop back through
            // phase 1 — a recursive call here would deadlock on this non-reentrant
            // semaphore. Warmup reports skipped instead of looping into an eviction.
            if (entries.Count >= maxEntries)
            {
                return (null, nonEvictingWarmup);
            }

            ExternalMemoryReservation? reservation = nonEvictingWarmup
                ? await admissionPool
                    .TryReserveExternalWithoutEvictionAsync(new(key.Provider, key.DeviceId, key.EstimatedMemoryMb, key.UseOpenVinoCpuProxy), cancellationToken)
                    .ConfigureAwait(false)
                : await admissionPool
                    .ReserveExternalAsync(new(key.Provider, key.DeviceId, key.EstimatedMemoryMb, key.UseOpenVinoCpuProxy), cancellationToken)
                    .ConfigureAwait(false);
            if (reservation is null)
            {
                // Admission would have to evict idle work to fit — warmup skips.
                return (null, true);
            }

            IGenAiModelResource? resource = null;
            try
            {
                using (BenchmarkPhaseCapture.Start("genai-model-create"))
                {
                    resource = await factory(key, cancellationToken).ConfigureAwait(false);
                }

                // Injected factories may ignore the token or outlive a pool disposal —
                // verify after completion, not only before scheduling.
                cancellationToken.ThrowIfCancellationRequested();
                ObjectDisposedException.ThrowIf(disposed, this);
                if (resource is null)
                {
                    throw new InvalidOperationException("GenAI model factory returned null.");
                }
            }
            catch
            {
                // try/finally: a resource-dispose failure cannot strand the reservation.
                try
                {
                    resource?.Dispose();
                }
                finally
                {
                    reservation.Dispose();
                }

                throw;
            }

            var entry = new Entry(resource, reservation);

            // Publish first, while the creator still owns the entry gate. If the admission
            // pool dropped the reservation meanwhile, registration fails and the entry is
            // rolled back — a resident model must never escape external accounting.
            if (!entries.TryAdd(key, entry))
            {
                entry.Dispose();
                return (null, false);
            }

            // Dispose may race after the pre-publish check above and before TryAdd. If so,
            // don't hand a lease to a caller after this entry escaped the teardown snapshot.
            if (disposed)
            {
                entries.TryRemove(new KeyValuePair<GenAiModelKey, Entry>(key, entry));
                entry.MarkEvicted();
                entry.ReleaseGate();
                entry.Dispose();
                SignalChanged();
                throw new ObjectDisposedException(nameof(GenAiModelPool));
            }

            // Admission-side eviction for this resource. It must never wait on
            // creationLock — external admission may invoke it while a creation holds
            // that lock — so it uses the entry gate and an exact dictionary remove.
            if (!await reservation.TrySetIdleEvictionCallbackAsync(
                    () => TryEvictIdleEntry(key, entry),
                    CancellationToken.None).ConfigureAwait(false))
            {
                if (entries.TryRemove(new KeyValuePair<GenAiModelKey, Entry>(key, entry)))
                {
                    entry.MarkEvicted();
                    // Release the creator-owned gate so waiters wake and observe eviction.
                    entry.ReleaseGate();
                    SignalChanged();
                }

                entry.Dispose();
                throw new ObjectDisposedException(
                    nameof(InferenceSessionPool),
                    "The admission pool was disposed while the GenAI model was loading.");
            }

            SignalChanged();
            BenchmarkPhaseCapture.Increment("genAiModelCreate");
            return (new GenAiModelLease(resource, () => ReleaseEntry(entry), entry.TokenizerCache), false);
        }
        finally
        {
            creationLock.Release();
        }
    }

    private void ReleaseEntry(Entry entry)
    {
        entry.LastReleasedTicks = Environment.TickCount64;
        entry.Reservation.MarkReleased();
        entry.ReleaseGate();
        SignalChanged();

        if (entry.IsEvicted)
        {
            // Marked evicted while leased (pool disposal) — the returning lease disposes it.
            entry.Dispose();
        }
    }

    /// <summary>Admission-side idle eviction: never takes a lock, never blocks.</summary>
    private bool TryEvictIdleEntry(GenAiModelKey key, Entry entry)
    {
        if (!entry.TryAcquireGate())
        {
            return false;
        }

        try
        {
            if (!entries.TryRemove(new KeyValuePair<GenAiModelKey, Entry>(key, entry)))
            {
                // Another path already removed this entry; we still hold its gate, so
                // dispose here rather than orphaning the native model and reservation.
                entry.Dispose();
                return false;
            }

            entry.MarkEvicted();
            SignalChanged();
            entry.Dispose();
            return true;
        }
        finally
        {
            // Releasing the (already removed) gate wakes blocked waiters so they can observe
            // the eviction and retry.
            entry.ReleaseGate();
        }
    }

    /// <summary>Oldest-idle non-evicted entry removed atomically via gate + exact pair remove.</summary>
    private Entry? TryEvictOldestIdle()
    {
        GenAiModelKey? candidateKey = null;
        Entry? candidate = null;
        long oldestTicks = long.MaxValue;
        foreach (KeyValuePair<GenAiModelKey, Entry> pair in entries)
        {
            if (pair.Value.IsEvicted || pair.Value.LastReleasedTicks >= oldestTicks)
            {
                continue;
            }

            candidateKey = pair.Key;
            candidate = pair.Value;
            oldestTicks = pair.Value.LastReleasedTicks;
        }

        if (candidate is null || !candidate.TryAcquireGate())
        {
            return null;
        }

        try
        {
            if (!entries.TryRemove(new KeyValuePair<GenAiModelKey, Entry>(candidateKey!, candidate)))
            {
                return null;
            }

            candidate.MarkEvicted();
            SignalChanged();
            return candidate; // caller disposes outside the creation lock
        }
        finally
        {
            candidate.ReleaseGate();
        }
    }

    private void SignalChanged()
    {
        Interlocked.Exchange(ref stateChanged, CreateStateSignal()).TrySetResult(true);
    }

    private static TaskCompletionSource<bool> CreateStateSignal() =>
        new(TaskCreationOptions.RunContinuationsAsynchronously);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposeOnce, 1) != 0)
        {
            return;
        }

        disposed = true;
        var toDispose = new List<Entry>();
        foreach (KeyValuePair<GenAiModelKey, Entry> pair in entries.ToArray())
        {
            if (!entries.TryRemove(pair))
            {
                continue;
            }

            pair.Value.MarkEvicted();
            if (pair.Value.TryAcquireGate())
            {
                // Idle at teardown: dispose below; release first to wake waiters, who then
                // observe the eviction and retry into the ODE path.
                pair.Value.ReleaseGate();
                toDispose.Add(pair.Value);
            }
            // Leased entries stay marked; the returning lease disposes them.
        }

        SignalChanged();
        var disposeFailures = new List<Exception>();
        foreach (Entry entry in toDispose)
        {
            try
            {
                entry.Dispose();
            }
            catch (Exception ex)
            {
                disposeFailures.Add(ex);
            }
        }

        if (disposeFailures.Count > 0)
        {
            throw new AggregateException(
                "One or more GenAI model entries failed to dispose.",
                disposeFailures);
        }
    }

    /// <summary>Runs native model construction off the caller thread.</summary>
    private static async Task<IGenAiModelResource> CreateNativeResourceAsync(
        GenAiModelKey key,
        CancellationToken cancellationToken)
    {
        IGenAiModelResource created = await Task.Run(
            () =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                return CreateNativeResource(key);
            },
            cancellationToken).ConfigureAwait(false);

        // A completed native load racing cancellation must not leak the model.
        if (cancellationToken.IsCancellationRequested)
        {
            created.Dispose();
            cancellationToken.ThrowIfCancellationRequested();
        }

        return created;
    }

    private static IGenAiModelResource CreateNativeResource(GenAiModelKey key)
    {
        GenAiNativeCompatibility.EnsureCompatible();
        if (key.Provider is ExecutionProviderKind.Cpu)
        {
            return new NativeGenAiModelResource(new Model(key.ModelRootPath));
        }

        using var config = new Config(key.ModelRootPath);
        config.ClearProviders();
        config.AppendProvider(GenAiExecutionProviderNames.Resolve(key.Provider));
        return new NativeGenAiModelResource(new Model(config));
    }

    private sealed class NativeGenAiModelResource(Model model) : IGenAiModelResource
    {
        public Model Model { get; } = model;
        public void Dispose() => Model.Dispose();
    }

    private sealed class Entry : IDisposable
    {
        private int evicted;
        private int disposeOnce;

        public Entry(IGenAiModelResource resource, ExternalMemoryReservation reservation)
        {
            Resource = resource;
            Reservation = reservation;
            Gate = new SemaphoreSlim(0, 1); // creator holds the exclusive gate
        }

        public IGenAiModelResource Resource { get; }
        public GenAiTokenizerCache TokenizerCache { get; } = new();
        public ExternalMemoryReservation Reservation { get; }
        public SemaphoreSlim Gate { get; }
        public long LastReleasedTicks { get; set; } = Environment.TickCount64;
        public bool IsEvicted => Volatile.Read(ref evicted) != 0;
        public void MarkEvicted() => Volatile.Write(ref evicted, 1);
        public bool TryAcquireGate() => Gate.Wait(0);
        public void ReleaseGate() => Gate.Release();

        public void Dispose()
        {
            if (Interlocked.Exchange(ref disposeOnce, 1) != 0)
            {
                return;
            }

            try
            {
                // The cached tokenizer wraps the native model, so it goes first.
                try
                {
                    TokenizerCache.Dispose();
                }
                finally
                {
                    Resource.Dispose();
                }
            }
            finally
            {
                Reservation.Dispose();
            }
        }
    }
}
