using System.Collections.Concurrent;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.TensorRtRtx;

namespace Trackdub.Inference.Onnx.Pool;

/// <summary>
/// Bounded, LRU-evicting pool of ONNX <see cref="InferenceSession"/> instances shared
/// across engine instances within a process.
/// </summary>
/// <remarks>
/// <para><strong>Lifecycle &amp; lease shape:</strong>
/// Callers obtain an exclusive <see cref="SessionLease"/> via <see cref="GetLeaseAsync"/>.
/// The lease holds the session until disposed; disposal releases the session back to the pool
/// without destroying it.  This allows warm sessions to survive across DI scope boundaries.</para>
///
/// <para><strong>Bounded capacity:</strong>
/// The pool holds at most <c>maxSessions</c> live sessions across all keys (default: 12).
/// This is a <em>count-based</em> cap that applies when memory admission is explicitly
/// disabled. Memory admission (on by default) adds a resident-memory budget per bucket:
/// before construct, reserve <see cref="SessionPoolKey.EstimatedVramMb"/>, evict idle
/// sessions to fit, and wait — never allocate an unbudgeted ephemeral session. CPU, DNNL,
/// and OpenVINO CPU-proxy sessions are accounted against the host RAM budget; OpenVINO NPU
/// sessions and other accelerators are accounted per device, so CPU work no longer
/// collides with GPU 0. A session whose factory falls back to a provider in another bucket
/// while it is created (see <see cref="GetLeaseReportingProviderAsync"/>) is re-admitted and
/// accounted in that bucket instead of its key's. When admission is explicitly
/// disabled and the count limit is reached with every entry leased, the new session is
/// created outside the pool (ephemeral) and disposed when its lease is released.</para>
///
/// <para><strong>Process-isolated VRAM observation:</strong>
/// When the host registers a process-GPU reader (see
/// <see cref="SharedPoolOptions.UseProcessGpuMemoryReader"/>), accelerator admission also floors
/// each device's admitted usage at this process's real dedicated GPU footprint, so GPU memory the
/// pool does not account for — driver contexts, runtime arenas, non-pooled consumers — consumes
/// the same per-device budget instead of being invisible to admission. Host-RAM buckets are
/// unaffected, an unavailable reading leaves reservation accounting exactly as it was, and
/// <c>TRACKDUB_SESSION_PROCESS_GPU_ADMISSION</c> disables the observation explicitly.</para>
///
/// <para><strong>Single-flight creation:</strong>
/// Concurrent misses for the same key share one factory invocation. Cancelling one waiter
/// does not cancel another caller’s valid acquisition.</para>
///
/// <para><strong>Model invalidation:</strong>
/// Session keys carry both the model file <em>path hash</em> and a cached content digest
/// (<see cref="SessionPoolKey.ModelContentHash"/>): helper-built keys invalidate when a
/// model file is replaced at the same path, while direct-constructed keys that omit the
/// content hash keep path-only identity.</para>
///
/// <para><strong>Memory pressure guidance:</strong>
/// Each ONNX session can consume hundreds of MB of accelerator or host memory depending on
/// model size. Admission enforces realistic budgets; call <see cref="EvictModelAsync"/> to
/// free memory for a model that is no longer needed, or <see cref="Dispose"/> to release
/// the entire pool. The default host limit can reject large graphs and graph bundles; see
/// <c>docs/reference/session-pool-memory-admission.md</c> for sizing and the explicit override.</para>
///
/// <para><strong>Thread safety:</strong>
/// All public APIs are thread-safe.  Sessions are single-threaded: only one caller may hold
/// a lease for a given key at a time, serialised by an internal per-entry gate.</para>
/// </remarks>
internal sealed partial class InferenceSessionPool : IDisposable
{
    /// <summary>Process-wide shared pool with default settings.</summary>
    /// <remarks>
    /// All production engines (Kokoro, Chatterbox, Whisper, Opus-MT, MADLAD, SepFormer,
    /// Spleeter, SortFormer, Silero-VAD) consume this pool via the
    /// <c>CreatePooled*</c> factory methods in <see cref="OnnxExecutionSessionFactory"/>.
    /// </remarks>
    /// <remarks>
    /// The shared instance is built from <see cref="SharedPoolOptions"/>, which reads
    /// <c>TRACKDUB_SESSION_ADMISSION</c>, <c>TRACKDUB_SESSION_VRAM_BUDGET_MB</c>,
    /// <c>TRACKDUB_SESSION_RAM_BUDGET_MB</c>, and <c>TRACKDUB_SESSION_MAX_SESSIONS</c>.
    /// It is resolved lazily on first use. Admission
    /// is <em>on</em> by default — the safe behaviour; an operator can opt out explicitly for
    /// diagnostics. See <see cref="SharedPoolOptions"/> for the activation contract.
    /// </remarks>
    public static InferenceSessionPool Shared => SharedPool.Value;

    private static readonly Lazy<InferenceSessionPool> SharedPool = new(
        static () => new InferenceSessionPool(
            maxSessions: SharedPoolOptions.MaxSessions,
            enableMemoryAdmission: SharedPoolOptions.EnableMemoryAdmission,
            memoryBudgetMb: SharedPoolOptions.MemoryBudgetMb,
            hostMemoryBudgetMb: SharedPoolOptions.HostMemoryBudgetMb,
            // Resolved late: a host registers its process-GPU reader through
            // SharedPoolOptions.UseProcessGpuMemoryReader, which can happen after this static
            // pool was first resolved.
            processGpuMemoryReader: static () => SharedPoolOptions.ProcessGpuMemoryReader),
        LazyThreadSafetyMode.ExecutionAndPublication);

    // Sized for a full dub working set (VAD + separation + diarization + ASR encoder/decoder
    // + translation encoder/decoder + multi-graph TTS) so LRU does not thrash mid-pipeline.
    public const int DefaultMaxSessions = 12;

    private sealed class PoolEntry(InferenceSession session, bool ephemeral, AdmissionBucket bucket) : IDisposable
    {
        private long lastReleasedTicks = Environment.TickCount64;
        private volatile bool evicted;
        private int disposeState; // 0 = live, 1 = disposed; guarded by Interlocked for idempotency
        private int pinCount;

        public InferenceSession Session { get; } = session;

        /// <summary>
        /// Admission bucket the session's memory is accounted in: the bucket of the provider it
        /// was created with, which is not always the bucket of its key's provider.
        /// </summary>
        public AdmissionBucket Bucket { get; } = bucket;

        /// <summary>Per-entry gate that serialises access (one user at a time).</summary>
        public SemaphoreSlim Gate { get; } = new(0, 1); // Starts unavailable because the creator immediately owns the first lease.

        /// <summary>
        /// Residency pins (audit §3A). Pinned entries are not idle-evicted. Independent of
        /// <see cref="Gate"/> — a session can be resident and still free for execution.
        /// </summary>
        public int PinCount => Volatile.Read(ref pinCount);

        public bool TryPin()
        {
            if (evicted || disposeState != 0)
            {
                return false;
            }

            Interlocked.Increment(ref pinCount);
            if (evicted || disposeState != 0)
            {
                Unpin();
                return false;
            }

            return true;
        }

        public void Unpin()
        {
            if (Interlocked.Decrement(ref pinCount) < 0)
            {
                Interlocked.Exchange(ref pinCount, 0);
            }
        }

        /// <summary>
        /// When <see langword="true"/> the entry was created beyond the pool limit and will not
        /// be stored; its session is disposed when the lease is released.
        /// </summary>
        public bool Ephemeral { get; private set; } = ephemeral;

        /// <summary>
        /// Heuristic snapshot: <see langword="true"/> when the gate appears idle at the moment of
        /// reading.  <em>Not</em> safe as a disposal guard — a concurrent <see cref="SemaphoreSlim.WaitAsync()"/>
        /// can race between the read and a subsequent dispose.  Eviction paths must use
        /// <c>Gate.Wait(0)</c> to atomically acquire the gate before disposing.
        /// </summary>
        public bool IsIdle => Gate.CurrentCount == 1;

        /// <summary>
        /// <see langword="true"/> once the entry has been removed from the pool by an eviction path.
        /// The lease holder is responsible for disposing the entry when it releases.
        /// </summary>
        public bool IsEvicted => evicted;

        /// <summary>Tick count at last <c>Release()</c> call — used for LRU eviction ordering.</summary>
        public long LastReleasedTicks => Volatile.Read(ref lastReleasedTicks);

        /// <summary>
        /// Marks this entry as evicted so that its lease holder disposes it on release rather than
        /// returning it to the pool.  Called under <c>creationLock</c> by eviction code.
        /// </summary>
        public void MarkEvicted() => evicted = true;

        public void MarkEphemeral() => Ephemeral = true;

        public void Release()
        {
            Volatile.Write(ref lastReleasedTicks, Environment.TickCount64);
            Gate.Release();
        }

        public void ReleaseGateWithoutTouchingLru() => Gate.Release();

        /// <summary>
        /// Idempotent dispose: safe to call from both the eviction path and the lease-release path
        /// concurrently without double-disposing the underlying session or semaphore.
        /// </summary>
        public void Dispose()
        {
            if (Interlocked.Exchange(ref disposeState, 1) == 0)
            {
                Gate.Dispose();
                // Healthy sessions always release their native memory. Only a TensorRT-RTX session
                // in a process whose runtime was poisoned by a CUDA OOM skips teardown (see
                // TensorRtRtxTeardownGuard); dispose of such a session can fault uncatchably.
                TensorRtRtxTeardownGuard.DisposeSafely(Session);
            }
        }
    }

    private readonly int maxSessions;
    private readonly ConcurrentDictionary<SessionPoolKey, PoolEntry> entries = new();
    private readonly ConcurrentDictionary<SessionPoolKey, SemaphoreSlim> createGates = new();
    /// <summary>
    /// Most recent construction failure per key. A waiter that wakes to find no entry and no
    /// in-flight creation consumes this so a failed creator's exception propagates without
    /// re-invoking the factory (single-flight: one factory call per creation wave).
    /// </summary>
    /// <summary>
    /// Monotonic creation-wave id. Incremented only when a factory fails and the failure is
    /// recorded; waiters capture the observed wave before queueing so only failures from a
    /// later wave (i.e. a creator they actually waited behind) are propagated.
    /// </summary>
    private long creationWave;
    private readonly ConcurrentDictionary<SessionPoolKey, (Exception Error, long Wave)> creationFailures = new();
    private readonly SemaphoreSlim creationLock = new(1, 1);
    private readonly bool enableMemoryAdmission;
    private readonly long memoryBudgetMb;
    private readonly long hostMemoryBudgetMb;
    private volatile bool disposed;
    private int pooledCount;
    private int disposeOnce; // 0 = not yet, 1 = disposed; Interlocked guard for single-winner teardown

    public InferenceSessionPool(
        int maxSessions = DefaultMaxSessions,
        bool enableMemoryAdmission = true,
        long? memoryBudgetMb = null,
        long? hostMemoryBudgetMb = null,
        Func<IProcessGpuMemoryReader?>? processGpuMemoryReader = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxSessions, 1);
        long resolvedMemoryBudgetMb = memoryBudgetMb ?? DefaultMemoryBudgetMb;
        ArgumentOutOfRangeException.ThrowIfLessThan(resolvedMemoryBudgetMb, 1);
        long resolvedHostBudgetMb = hostMemoryBudgetMb ?? DefaultHostMemoryBudgetMb;
        ArgumentOutOfRangeException.ThrowIfLessThan(resolvedHostBudgetMb, 1);
        this.maxSessions = maxSessions;
        this.enableMemoryAdmission = enableMemoryAdmission;
        this.memoryBudgetMb = resolvedMemoryBudgetMb;
        this.hostMemoryBudgetMb = resolvedHostBudgetMb;
        this.processGpuMemoryReader = processGpuMemoryReader;
    }

    /// <summary>Per-device accelerator admission budget backing this pool, in MiB.</summary>
    internal long MemoryBudgetMb => memoryBudgetMb;

    /// <summary>Shared host-RAM admission budget backing this pool, in MiB.</summary>
    internal long HostMemoryBudgetMb => hostMemoryBudgetMb;

    /// <summary>
    /// Returns an appropriate default max-session count for the current hardware.
    /// On Windows with a discrete NVIDIA GPU and >= 8 GB VRAM, returns 24 (covers a full
    /// multi-graph pipeline without mid-run LRU thrash). Otherwise returns 12.
    /// </summary>
    public static int RecommendedMaxSessions(IReadOnlyList<DeviceEntry>? devices)
    {
        if (devices is null)
            return DefaultMaxSessions;

        DeviceEntry? nvidiaGpu = devices.FirstOrDefault(d =>
            d.Kind == DeviceKind.DiscreteGpu &&
            string.Equals(d.VendorName, "NVIDIA", StringComparison.OrdinalIgnoreCase));

        if (nvidiaGpu is not null && nvidiaGpu.DedicatedVramMb >= 8192)
            return 24;

        return DefaultMaxSessions;
    }

    /// <summary>
    /// Pre-warms a session for <paramref name="key"/> without returning a lease.
    /// Useful at startup to amortise first-call latency.
    /// If a session for this key already exists the call is a no-op.
    /// The session is resident but <em>unpinned</em> and may be idle-evicted later.
    /// Use <see cref="GetResidencyAsync"/> when it must stay warm.
    /// </summary>
    public async Task WarmAsync(
        SessionPoolKey key,
        Func<CancellationToken, Task<InferenceSession>> factory,
        CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        using SessionLease lease = await GetLeaseAsync(key, factory, cancellationToken).ConfigureAwait(false);
        // Lease is immediately released on exit — session stays in the pool.
    }

    /// <summary>
    /// Creates or reuses <paramref name="key"/> and returns a <see cref="SessionResidency"/> pin.
    /// The pin keeps the session out of idle eviction but does <em>not</em> hold the execution
    /// gate — other callers can still <see cref="GetLeaseAsync"/> and Run() (audit §3A:
    /// residency ≠ exclusivity). Dispose the residency to allow eviction again.
    /// </summary>
    public async Task<SessionResidency> GetResidencyAsync(
        SessionPoolKey key,
        Func<CancellationToken, Task<InferenceSession>> factory,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(factory);

        return await GetResidencyReportingProviderAsync(
            key,
            async ct => new CreatedPoolSession(await factory(ct).ConfigureAwait(false), key.Provider),
            cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// <see cref="GetResidencyAsync"/> for a factory that reports the provider the session was
    /// actually created on (see <see cref="GetLeaseReportingProviderAsync"/>).
    /// </summary>
    internal async Task<SessionResidency> GetResidencyReportingProviderAsync(
        SessionPoolKey key,
        Func<CancellationToken, Task<CreatedPoolSession>> factory,
        CancellationToken cancellationToken) =>
        await GetResidencyReportingProviderAsync(key, factory, cancellationToken, bundleHold: null).ConfigureAwait(false);

    private async Task<SessionResidency> GetResidencyReportingProviderAsync(
        SessionPoolKey key,
        Func<CancellationToken, Task<CreatedPoolSession>> factory,
        CancellationToken cancellationToken,
        BundleAdmissionHold? bundleHold)
    {
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(factory);

        while (true)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            // Ensure the session exists (single-flight create). Release the lease immediately.
            using (await GetLeaseReportingProviderAsync(key, factory, cancellationToken, bundleHold).ConfigureAwait(false))
            {
            }

            if (entries.TryGetValue(key, out PoolEntry? entry) && entry.TryPin())
            {
                PoolEntry pinned = entry;
                return new SessionResidency(pinned.Unpin);
            }

            // Evicted between release and pin — retry create.
            cancellationToken.ThrowIfCancellationRequested();
        }
    }

    /// <summary>
    /// Pins an already-pooled session if present. Used after a create/warm when the caller
    /// re-acquires with the same <see cref="SessionPoolKey"/> for each execution.
    /// </summary>
    public bool TryPinExisting(SessionPoolKey key, out SessionResidency? residency)
    {
        ArgumentNullException.ThrowIfNull(key);
        if (entries.TryGetValue(key, out PoolEntry? entry) && entry.TryPin())
        {
            PoolEntry pinned = entry;
            residency = new SessionResidency(pinned.Unpin);
            return true;
        }

        residency = null;
        return false;
    }

    /// <summary>
    /// Returns an exclusive <see cref="SessionLease"/> for <paramref name="key"/>.
    /// If no session exists for the key, one is created using <paramref name="factory"/>
    /// (single-flight: concurrent misses share one factory call).
    /// When memory admission is enabled the create waits for budget after evicting idle
    /// sessions and never allocates an unbudgeted ephemeral session. When admission is off
    /// and the pool is full with no idle eviction target, the session is created outside
    /// the pool (ephemeral) and disposed when the lease is released.
    /// </summary>
    public async Task<SessionLease> GetLeaseAsync(
        SessionPoolKey key,
        Func<CancellationToken, Task<InferenceSession>> factory,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(factory);

        return await GetLeaseReportingProviderAsync(
            key,
            async ct => new CreatedPoolSession(await factory(ct).ConfigureAwait(false), key.Provider),
            cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// <see cref="GetLeaseAsync"/> for a factory whose provider fallback can create the session
    /// on a different provider than <see cref="SessionPoolKey.Provider"/>. The key is admitted
    /// against its own provider's bucket before the factory runs; when the session lands in
    /// another bucket (TensorRT RTX falling back to CPU moves it from the device budget to host
    /// RAM), the pool releases the first reservation and re-admits the session against the
    /// bucket it was created in before publishing it, under the same rules as a direct request
    /// there: an estimate over that budget fails, otherwise idle sessions in the bucket are
    /// evicted to fit or the create waits. The published entry stays accounted, and evictable,
    /// in that bucket. Single-flight creation and lease semantics are unchanged.
    /// </summary>
    public async Task<SessionLease> GetLeaseReportingProviderAsync(
        SessionPoolKey key,
        Func<CancellationToken, Task<CreatedPoolSession>> factory,
        CancellationToken cancellationToken) =>
        await GetLeaseReportingProviderAsync(key, factory, cancellationToken, bundleHold: null).ConfigureAwait(false);

    private enum LeaseFlow { Leased, Retry, Missed }

    /// <summary>
    /// Mutable state for one acquisition through the create path: the single-flight
    /// gate snapshot, admission reservation, and failure-propagation flags. Kept on an
    /// object so the phase helpers can advance the attempt without long ref-parameter lists.
    /// </summary>
    private sealed class LeaseAttempt
    {
        public required BundleAdmissionHold? BundleHold { get; init; }
        public required SessionPoolKey Key { get; init; }
        public bool CacheMissRecorded;
        public bool CreationSnapshotPending = true;
        public long ObservedCreationWave;
        public required SemaphoreSlim CreateGate { get; set; }
        public bool PropagatedRecentFailure;
        public bool Reserved;
        public long NeedMb;
        public AdmissionBucket Bucket;
        public ProcessGpuObservation? Observation;
        public InferenceSession? Session;
        public ExecutionProviderKind CreatedProvider;
        public PoolEntry? FreshEntry;
    }

    private async Task<SessionLease> GetLeaseReportingProviderAsync(
        SessionPoolKey key,
        Func<CancellationToken, Task<CreatedPoolSession>> factory,
        CancellationToken cancellationToken,
        BundleAdmissionHold? bundleHold)
    {
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(factory);

        // Snapshot the creation wave at arrival: a caller that arrives while a creator is
        // in flight — but is delayed by a gate or lock until after that creator fails —
        // must still consume the recorded failure instead of re-running the factory.
        var attempt = new LeaseAttempt
        {
            BundleHold = bundleHold,
            Key = key,
            CreateGate = createGates.GetOrAdd(key, static _ => new SemaphoreSlim(1, 1)),
            ObservedCreationWave = Volatile.Read(ref creationWave),
        };

        while (true)
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            // Fast path: entry already exists — wait on its gate (serialises inference).
            (SessionLease? fastLease, LeaseFlow fastFlow) =
                await TryLeaseExistingEntryAsync(attempt, cancellationToken).ConfigureAwait(false);
            if (fastFlow == LeaseFlow.Retry)
            {
                continue;
            }

            if (fastFlow == LeaseFlow.Leased)
            {
                return fastLease!;
            }

            if (!attempt.CacheMissRecorded)
            {
                BenchmarkPhaseCapture.Increment("poolMiss");
                attempt.CacheMissRecorded = true;
            }

            await AcquireCreateGateAsync(attempt, cancellationToken).ConfigureAwait(false);
            try
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                if (entries.ContainsKey(key))
                {
                    continue;
                }

                // If memory admission is enabled and model exceeds budget, fail fast before
                // attempting to create an unbudgeted session (audit §3B).
                ResolveAdmissionNeed(attempt);
                ConsumeRecentFailure(attempt);

                if (await ReserveForCreateAsync(attempt, cancellationToken).ConfigureAwait(false))
                {
                    continue;
                }

                await InvokeFactoryTrackedAsync(factory, attempt, cancellationToken).ConfigureAwait(false);
                await ReadmitIfBucketChangedAsync(attempt, cancellationToken).ConfigureAwait(false);

                (SessionLease? lease, bool retry) =
                    await PublishOrLeaseCompetitorAsync(attempt, cancellationToken).ConfigureAwait(false);
                if (retry)
                {
                    continue;
                }

                return lease!;
            }
            catch (Exception ex)
            {
                if (!attempt.PropagatedRecentFailure && ex is not OperationCanceledException)
                {
                    creationFailures[key] = (ex, Interlocked.Increment(ref creationWave));
                }

                throw;
            }
            finally
            {
                attempt.CreateGate.Release();
            }
        }
    }

    private async Task<(SessionLease? Lease, LeaseFlow Flow)> TryLeaseExistingEntryAsync(
        LeaseAttempt attempt,
        CancellationToken cancellationToken)
    {
        if (!entries.TryGetValue(attempt.Key, out PoolEntry? existing))
        {
            return (null, LeaseFlow.Missed);
        }

        // The graph turned out resident, so its hold share (if any) is
        // unnecessary: return it to the pending reservation before waiting
        // on the gate. Waiting first double-counts the graph (pending share
        // plus the published entry in CurrentReservedMb) for the whole lease
        // held by the concurrent publisher, spuriously evicting or stalling
        // a third acquire. If the entry is evicted while waiting, the ODE
        // retry below falls back to ordinary admission (the relinquished
        // share is simply gone).
        if (attempt.BundleHold is not null
            && attempt.BundleHold.TryRelinquish(BucketOf(attempt.Key), attempt.Key, out long relinquishedMb))
        {
            ReleaseReservation(BucketOf(attempt.Key), relinquishedMb);
        }

        try
        {
            using (BenchmarkPhaseCapture.Start("pool-wait"))
                await existing.Gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            if (disposed)
            {
                ReleaseLeaseEntry(existing);
                ObjectDisposedException.ThrowIf(disposed, this);
            }

            // One counter per acquire request: a request that already recorded
            // a miss is not reclassified as a hit if it lands on an entry later.
            if (!attempt.CacheMissRecorded)
            {
                BenchmarkPhaseCapture.Increment("poolHit");
            }

            return (BuildLease(existing), LeaseFlow.Leased);
        }
        catch (ObjectDisposedException)
        {
            // The entry was evicted and its SemaphoreSlim disposed between the dictionary
            // lookup and WaitAsync. Retry from the top so we can observe the current pool
            // state (including full-pool disposal) before deciding whether to create.
            return (null, LeaseFlow.Retry);
        }
    }

    /// <summary>
    /// Single-flight create: one factory invocation per key. Cancelling this waiter
    /// does not cancel the shared create for other callers. A caller that queues behind
    /// an in-flight creator (gate already held) consumes that creator's recorded
    /// failure so construction failures propagate without rebuilding; callers that
    /// arrive afterwards retry the factory — failed sessions are not cached.
    /// </summary>
    private async Task AcquireCreateGateAsync(
        LeaseAttempt attempt,
        CancellationToken cancellationToken)
    {
        if (!attempt.CreationSnapshotPending)
        {
            // A retry iteration re-observes the pool after the arrival snapshot was
            // consumed — refresh it so each iteration reflects current state.
            attempt.CreateGate = createGates.GetOrAdd(attempt.Key, static _ => new SemaphoreSlim(1, 1));
            attempt.ObservedCreationWave = Volatile.Read(ref creationWave);
        }
        attempt.CreationSnapshotPending = false;
        attempt.PropagatedRecentFailure = false;

        // Atomically probe the single-flight gate: Wait(0) takes it if free, so
        // there is no check-then-act window in which another creator can take
        // the gate between the probe and the wait. Only when the gate is
        // genuinely held by another creator does this warm queue behind it —
        // and only then must it first return its aggregate share: the gate
        // holder may itself be waiting for admission on this bucket, and its
        // wait can only be freed by room the share is occupying (a pinned
        // sibling is not evictable, pending credit is not an entry). Without
        // the relinquish both sides hang until cancellation. Giving the share
        // up costs the warm its all-or-nothing guarantee for this key; it
        // falls back to ordinary per-graph admission and usually lands on the
        // resident fast path once the other creator publishes.
        bool gateAcquired = attempt.CreateGate.Wait(0, cancellationToken);
        if (!gateAcquired)
        {
            if (attempt.BundleHold is not null
                && attempt.BundleHold.TryRelinquish(BucketOf(attempt.Key), attempt.Key, out long queuedShareMb))
            {
                ReleaseReservation(BucketOf(attempt.Key), queuedShareMb);
            }

            using (BenchmarkPhaseCapture.Start("pool-single-flight-wait"))
                await attempt.CreateGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
    }

    private void ResolveAdmissionNeed(LeaseAttempt attempt)
    {
        attempt.NeedMb = ResolveReservationMb(attempt.Key);
        attempt.Bucket = BucketOf(attempt.Key);
        if (enableMemoryAdmission && attempt.NeedMb > BudgetFor(attempt.Bucket))
        {
            throw new InvalidOperationException(
                $"'{attempt.Key.EngineFamily}' needs ~{attempt.NeedMb} MB, which exceeds the " +
                $"{DescribeBucket(attempt.Bucket)} admission budget of {BudgetFor(attempt.Bucket)} MB.");
        }
    }

    private void ConsumeRecentFailure(LeaseAttempt attempt)
    {
        if (!creationFailures.TryGetValue(attempt.Key, out var recentFailure))
        {
            return;
        }

        if (recentFailure.Wave > attempt.ObservedCreationWave)
        {
            // Failure from a later wave than our arrival snapshot: we arrived
            // while that creator was in flight — propagate without rebuilding.
            attempt.PropagatedRecentFailure = true;
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(recentFailure.Error).Throw();
        }

        // Stale failure from an earlier wave: retry the factory.
        creationFailures.TryRemove(new KeyValuePair<SessionPoolKey, (Exception Error, long Wave)>(attempt.Key, recentFailure));
    }

    /// <summary>
    /// Reserves the admission budget for a create. Takes it under <see cref="creationLock"/>
    /// from bundle credit or spare budget, or waits via <see cref="WaitForAdmissionBudgetAsync"/>
    /// outside the lock. Returns <see langword="true"/> when an entry appeared under the lock
    /// and the caller must retry from the fast path.
    /// </summary>
    private async Task<bool> ReserveForCreateAsync(
        LeaseAttempt attempt,
        CancellationToken cancellationToken)
    {
        // Observe outside creationLock: the first Windows performance-counter read can
        // block for around a second on a cold host, and it must not stall other creators.
        // Host-RAM buckets are never charged with the observation, so they skip the
        // probe instead of polling GPU counters on every admission.
        attempt.Observation = enableMemoryAdmission && !attempt.Bucket.IsHost
            ? ReadObservedProcessGpu()
            : null;

        if (await TryReserveUnderLockAsync(attempt, cancellationToken).ConfigureAwait(false))
        {
            return true;
        }

        if (enableMemoryAdmission && !attempt.Reserved)
        {
            await WaitForAdmissionBudgetAsync(attempt.NeedMb, attempt.Bucket, cancellationToken).ConfigureAwait(false);
            // Budget reserved by WaitForAdmissionBudgetAsync on success.
            attempt.Reserved = true;
        }

        return false;
    }

    /// <summary>
    /// Takes the create reservation under <see cref="creationLock"/> from bundle credit or
    /// spare budget. Returns <see langword="true"/> when an entry appeared under the lock
    /// and the caller must retry from the fast path.
    /// </summary>
    private async Task<bool> TryReserveUnderLockAsync(LeaseAttempt attempt, CancellationToken cancellationToken)
    {
        using (BenchmarkPhaseCapture.Start("pool-creation-lock-wait"))
            await creationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (entries.ContainsKey(attempt.Key))
            {
                // No reservation is held yet at this point (reserved is only set below).
                return true;
            }

            if (enableMemoryAdmission)
            {
                // A bundle graph whose bundle holds aggregate credit in
                // this bucket draws on it only when the credit was sized for
                // this graph: a resident graph evicted after the hold (by the
                // hold wait itself or a concurrent admission) falls back to
                // ordinary per-graph admission instead of stealing a sibling's
                // share. Re-waiting here is what let a sibling graph's freshly
                // created session — counted in the process-GPU floor —
                // admission-evict a warmed bundle graph.
                // The pending charge is already taken; the reservation releases
                // below (publish/failure) draw it down as usual.
                attempt.Reserved = attempt.BundleHold is not null
                    && attempt.BundleHold.TryConsume(attempt.Bucket, attempt.Key, attempt.NeedMb);
                if (!attempt.Reserved)
                {
                    attempt.Reserved = TryReserveAdmissionBudget(attempt.Bucket, attempt.NeedMb, attempt.Observation);
                }
                // else: do not hold a reservation while waiting — that deadlocks
                // when every waiter reserves and nobody can release.
            }
            // else (count mode): do not evict or decide ephemeral here — the entry we'd
            // evict is still usable and factory() has not succeeded yet. The lruEvicted2
            // path after a successful create handles capacity instead.
            return false;
        }
        finally
        {
            creationLock.Release();
        }
    }

    private async Task InvokeFactoryTrackedAsync(
        Func<CancellationToken, Task<CreatedPoolSession>> factory,
        LeaseAttempt attempt,
        CancellationToken cancellationToken)
    {
        try
        {
            BenchmarkPhaseCapture.Increment("sessionCreate");
            CreatedPoolSession created;
            using (BenchmarkPhaseCapture.Start("session-create"))
                created = await factory(cancellationToken).ConfigureAwait(false);
            if (created.Provider is ExecutionProviderKind.TensorRTRtx)
            {
                TensorRtRtxTeardownGuard.Track(created.Session);
            }

            attempt.Session = created.Session;
            attempt.CreatedProvider = created.Provider;
        }
        catch (Exception ex)
        {
            if (attempt.Key.Provider is ExecutionProviderKind.TensorRTRtx)
            {
                TensorRtRtxTeardownGuard.MarkPoisonedIfCudaOutOfMemory(ex);
            }

            if (attempt.Reserved)
            {
                // Always release reservation on factory failure to prevent a reservation leak
                // (audit §3A: memory accounting integrity).
                ReleaseReservation(attempt.Bucket, attempt.NeedMb);
                attempt.Reserved = false;
            }

            throw;
        }
    }

    /// <summary>
    /// Re-admits a freshly created session when the factory's provider fallback landed it
    /// in a different bucket than the key was admitted against.
    /// </summary>
    private async Task ReadmitIfBucketChangedAsync(
        LeaseAttempt attempt,
        CancellationToken cancellationToken)
    {
        AdmissionBucket createdBucket = BucketOf(attempt.Key, attempt.CreatedProvider);
        if (createdBucket == attempt.Bucket)
        {
            return;
        }

        // The factory's provider fallback created the session outside the bucket its
        // key was admitted against, so that bucket never holds this memory.
        if (attempt.Reserved)
        {
            ReleaseReservation(attempt.Bucket, attempt.NeedMb);
            attempt.Reserved = false;
        }

        attempt.Bucket = createdBucket;
        if (enableMemoryAdmission)
        {
            try
            {
                await ReadmitCreatedSessionAsync(attempt.Key, attempt.CreatedProvider, attempt.NeedMb, attempt.Bucket, cancellationToken).ConfigureAwait(false);
                attempt.Reserved = true;
            }
            catch
            {
                TensorRtRtxTeardownGuard.DisposeSafely(attempt.Session!);
                throw;
            }
        }
    }

    /// <summary>Result of the under-lock publish attempt for one created session.</summary>
    private sealed class PublishOutcome
    {
        public bool Published;
        public PoolEntry? Competitor;
        public PoolEntry? LruEvicted;
    }

    /// <summary>
    /// Publishes a created session under <see cref="creationLock"/> (evicting LRU idle in
    /// count mode when full), or discards it and leases the competitor's entry when a
    /// concurrent creator won the race.
    /// </summary>
    private async Task<(SessionLease? Lease, bool Retry)> PublishOrLeaseCompetitorAsync(
        LeaseAttempt attempt,
        CancellationToken cancellationToken)
    {
        attempt.FreshEntry = new PoolEntry(attempt.Session!, ephemeral: false, attempt.Bucket);

        if (attempt.FreshEntry.Ephemeral)
        {
            ReleaseIfReserved(attempt);
            return (BuildLease(attempt.FreshEntry), false);
        }

        PublishOutcome outcome = await TryPublishFreshEntryAsync(attempt, cancellationToken).ConfigureAwait(false);
        return await ResolvePublishOutcomeAsync(outcome, attempt, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// The serialized publish attempt: under <see cref="creationLock"/>, adds the fresh
    /// entry or observes the competitor, evicting one LRU idle in count mode when the
    /// pool is full. The outcome's evicted entry must be disposed outside the lock.
    /// </summary>
    private async Task<PublishOutcome> TryPublishFreshEntryAsync(
        LeaseAttempt attempt,
        CancellationToken cancellationToken)
    {
        var outcome = new PublishOutcome();
        PoolEntry freshEntry = attempt.FreshEntry!;
        try
        {
            await creationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                PublishUnderLock(attempt, outcome);
            }
            finally
            {
                creationLock.Release();
            }
        }
        catch
        {
            if (!outcome.Published)
            {
                freshEntry.Dispose();
            }

            ReleaseIfReserved(attempt);
            outcome.LruEvicted?.Dispose();
            throw;
        }

        return outcome;
    }

    /// <summary>
    /// Under-lock publish: a competitor already stored wins; otherwise a full count-mode
    /// pool evicts one LRU idle first, and when nothing can be evicted the fresh entry
    /// goes ephemeral instead of joining the pool. Must be called while
    /// <see cref="creationLock"/> is held.
    /// </summary>
    private void PublishUnderLock(LeaseAttempt attempt, PublishOutcome outcome)
    {
        PoolEntry freshEntry = attempt.FreshEntry!;
        if (entries.TryGetValue(attempt.Key, out outcome.Competitor))
        {
            return;
        }

        if (!enableMemoryAdmission && pooledCount >= maxSessions)
        {
            outcome.LruEvicted = TryEvictLruIdle();
            if (outcome.LruEvicted is null)
            {
                freshEntry.MarkEphemeral();
            }
        }

        if (freshEntry.Ephemeral)
        {
            return;
        }

        outcome.Published = entries.TryAdd(attempt.Key, freshEntry);
        if (!outcome.Published)
        {
            entries.TryGetValue(attempt.Key, out outcome.Competitor);
            return;
        }

        pooledCount++;
        SignalBundleStateChanged();
    }

    /// <summary>
    /// Resolves a publish outcome into a lease or retry: success and count-mode ephemeral
    /// lease the fresh entry; a lost race discards it and leases the competitor instead.
    /// </summary>
    private async Task<(SessionLease? Lease, bool Retry)> ResolvePublishOutcomeAsync(
        PublishOutcome outcome,
        LeaseAttempt attempt,
        CancellationToken cancellationToken)
    {
        PoolEntry freshEntry = attempt.FreshEntry!;
        outcome.LruEvicted?.Dispose();

        if (outcome.Published)
        {
            // Estimate now lives on the pooled entry (CurrentReservedMb sums entries);
            // drop the pending reservation so it is not double-counted.
            ReleaseIfReserved(attempt);
            return (BuildLease(freshEntry), false);
        }

        if (freshEntry.Ephemeral)
        {
            ReleaseIfReserved(attempt);
            return (BuildLease(freshEntry), false);
        }

        // Lost the race: another thread concurrently published an entry for this key.
        // Discard our duplicate session and lease the winner's entry instead.
        freshEntry.Dispose();
        ReleaseIfReserved(attempt);

        if (outcome.Competitor is null)
        {
            return (null, true);
        }

        try
        {
            await outcome.Competitor.Gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            if (disposed)
            {
                ReleaseLeaseEntry(outcome.Competitor);
                ObjectDisposedException.ThrowIf(disposed, this);
            }

            return (BuildLease(outcome.Competitor), false);
        }
        catch (ObjectDisposedException)
        {
            return (null, true);
        }
    }

    private void ReleaseIfReserved(LeaseAttempt attempt)
    {
        if (attempt.Reserved)
        {
            ReleaseReservation(attempt.Bucket, attempt.NeedMb);
            attempt.Reserved = false;
        }
    }

    /// <summary>
    /// Admits a session that already exists against <paramref name="bucket"/>, the bucket of the
    /// provider it was created with, and takes the pending reservation there. The session is not
    /// yet published, so it is never an eviction candidate for its own admission. When the bucket
    /// cannot fit it immediately, the created session is charged against the bucket before the
    /// wait: its memory is real and resident, so a wait must never hide it from the budget
    /// (concurrent CPU fallbacks for distinct keys would otherwise each park a full model
    /// unaccounted and silently exceed the host ceiling).
    /// </summary>
    private async Task ReadmitCreatedSessionAsync(
        SessionPoolKey key,
        ExecutionProviderKind createdProvider,
        long needMb,
        AdmissionBucket bucket,
        CancellationToken cancellationToken)
    {
        if (needMb > BudgetFor(bucket))
        {
            throw new InvalidOperationException(
                $"'{key.EngineFamily}' was admitted for {key.Provider} but its session was created on " +
                $"{createdProvider} and needs ~{needMb} MB, which exceeds the {DescribeBucket(bucket)} " +
                $"admission budget of {BudgetFor(bucket)} MB.");
        }

        ProcessGpuObservation? observation = bucket.IsHost ? null : ReadObservedProcessGpu();
        bool reserved;
        await creationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            reserved = TryReserveAdmissionBudget(bucket, needMb, observation);
        }
        finally
        {
            creationLock.Release();
        }

        if (!reserved)
        {
            // Charge the already-created session before waiting so the budget reflects its real
            // memory for the whole wait, then wait for the bucket total (including it) to fit.
            AddPendingReservation(bucket, needMb);
            try
            {
                await WaitForAdmissionBudgetAsync(needMb, bucket, cancellationToken, alreadyReserved: true)
                    .ConfigureAwait(false);
            }
            catch
            {
                ReleaseReservation(bucket, needMb);
                throw;
            }
        }
    }

    private long ResolveReservationMb(SessionPoolKey key) =>
        key.EstimatedVramMb > 0 ? key.EstimatedVramMb : SessionPoolKey.DefaultEstimatedVramMb;

    /// <summary>
    /// Disposes all idle pooled sessions.  Sessions currently leased out are left for their
    /// lease holders to release naturally.
    /// </summary>
    public void Dispose()
    {
        // Single-winner guard: only the first caller performs teardown.
        // Using Interlocked ensures concurrent Dispose() calls are safe and idempotent.
        if (Interlocked.Exchange(ref disposeOnce, 1) != 0)
        {
            return;
        }

        // Signal all code paths that check `disposed` so they throw ObjectDisposedException
        // instead of creating new sessions.  Must be set before acquiring the lock so that
        // slow-path waiters that win the lock after we release it observe the flag.
        disposed = true;

        List<PoolEntry> toDispose = new();

        if (creationLock.Wait(TimeSpan.FromSeconds(10)))
        {
            try
            {
                foreach (SessionPoolKey key in entries.Keys.ToArray())
                {
                    if (entries.TryRemove(key, out PoolEntry? entry))
                    {
                        pooledCount--;
                        SignalBundleStateChanged();
                        entry.MarkEvicted(); // defence-in-depth alongside `disposed` flag
                        // Atomically try to acquire the gate. If we win, schedule the entry for
                        // disposal outside the lock. If the entry is still leased, the BuildLease
                        // closure will observe disposed=true (and IsEvicted=true) on release and
                        // call entry.Dispose() idempotently.
                        if (TryAcquireGate(entry))
                        {
                            toDispose.Add(entry);
                        }
                    }
                }
            }
            finally
            {
                // Release the semaphore — do NOT dispose it.  Concurrent GetLeaseAsync slow-path
                // waiters may already be blocked on creationLock.WaitAsync(); disposing the
                // semaphore here would race with those waiters and surface as ObjectDisposedException
                // instead of the clean ObjectDisposedException.ThrowIf(disposed, this) path.
                // The semaphore is a small object and will be collected by the GC after the pool
                // itself is no longer referenced.
                pooledCount = 0;
                creationLock.Release();
            }
        }

        // Dispose sessions outside the lock — session disposal can be expensive and
        // should not block concurrent callers waiting on creationLock.
        foreach (PoolEntry entry in toDispose)
        {
            entry.Dispose();
        }

        // External reservations are dropped from accounting; each registered idle callback
        // is then attempted so a released external resource (e.g. a cached GenAI model) is
        // freed with the pool. A resource still leased may refuse eviction (false); its
        // token stays idempotent even though pool accounting is already cleared.
        foreach (KeyValuePair<Guid, ExternalReservationState> pair in externalReservations.ToArray())
        {
            if (!externalReservations.TryRemove(pair))
            {
                continue;
            }

            try
            {
                pair.Value.TryEvictIdle?.Invoke();
            }
            catch (ObjectDisposedException)
            {
                // The external owner may be disposing concurrently.
            }
            catch (InvalidOperationException)
            {
                // Teardown must tolerate an external owner that cannot evict now.
            }
            catch (Exception)
            {
                // Teardown must not throw or abandon the remaining reservations if a
                // native external resource fails while disposing.
            }
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private SessionLease BuildLease(PoolEntry entry)
    {
        return new SessionLease(entry.Session, () => ReleaseLeaseEntry(entry));
    }

    private void ReleaseLeaseEntry(PoolEntry entry)
    {
        if (entry.Ephemeral)
        {
            // Ephemeral entries are never stored in the pool; dispose immediately.
            // Admission mode never creates ephemeral sessions.
            entry.Dispose();
            return;
        }

        // Release the gate BEFORE checking eviction / disposal flags.  This closes the
        // handshake race: an eviction path that called MarkEvicted() while we held the gate
        // and then observed Gate.Wait(0)==false backed off, expecting us to clean up after
        // releasing.  After Release() the eviction path can atomically re-acquire the gate
        // and dispose, or — if it has already backed off — we detect the eviction below and
        // perform disposal ourselves.
        entry.Release();
        SignalBundleStateChanged();

        if (entry.IsEvicted || disposed)
        {
            // Re-check after releasing the gate: if an eviction path marked this entry
            // while the gate was held and left cleanup to us, attempt to re-acquire the
            // gate atomically.  If another caller (fast-path GetLeaseAsync or an eviction
            // path that beat us) already holds the gate, cleanup is theirs.
            try
            {
                if (entry.Gate.Wait(0))
                {
                    entry.Dispose();
                }
            }
            catch (ObjectDisposedException)
            {
                // Gate already disposed by a concurrent eviction / pool-dispose path.
            }
        }
    }

    private static bool TryAcquireGate(PoolEntry entry)
    {
        try
        {
            return entry.Gate.Wait(0);
        }
        catch (ObjectDisposedException)
        {
            return false;
        }
    }
}
