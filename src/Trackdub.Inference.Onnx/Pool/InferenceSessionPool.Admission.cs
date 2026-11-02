using System.Collections.Concurrent;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.TensorRtRtx;

namespace Trackdub.Inference.Onnx.Pool;

// Admission accounting: buckets, pending reservations, GPU observation, and the
// budget gate every create passes through.
internal sealed partial class InferenceSessionPool
{
    /// <summary>
    /// Late-bound source of this process's dedicated GPU reading, resolved once per accelerator
    /// admission decision. <see langword="null"/> when the host registered no reader, in which
    /// case admission accounts for the pool's own reservations only.
    /// </summary>
    private readonly Func<IProcessGpuMemoryReader?>? processGpuMemoryReader;
    /// <summary>Pending create reservations per admission bucket (host RAM vs accelerator device).</summary>
    private readonly ConcurrentDictionary<AdmissionBucket, long> pendingCreateMbByBucket = new();
    private int admissionWaiters;
    /// <summary>
    /// Admission accounting bucket: CPU/DNNL and OpenVINO CPU-proxy sessions share host RAM;
    /// standalone OpenVINO accelerator sessions use an NPU-specific bucket, separate from GPU 0.
    /// </summary>
    private readonly record struct AdmissionBucket(bool IsHost, ExecutionProviderKind? AcceleratorProvider, int DeviceId);
    private static bool IsHostProvider(ExecutionProviderKind provider) =>
        provider is ExecutionProviderKind.Cpu or ExecutionProviderKind.Dnnl;
    private static AdmissionBucket BucketOf(SessionPoolKey key) =>
        IsHostProvider(key.Provider) || (key.Provider is ExecutionProviderKind.OpenVino && key.UseOpenVinoCpuProxy)
            ? new(true, null, 0)
            : key.Provider is ExecutionProviderKind.OpenVino
                ? new(false, ExecutionProviderKind.OpenVino, key.DeviceId ?? 0)
                : new(false, null, key.DeviceId ?? 0);
    private static AdmissionBucket BucketOf(
        ExecutionProviderKind provider,
        int? deviceId,
        bool useOpenVinoCpuProxy = false) =>
        IsHostProvider(provider) || (provider is ExecutionProviderKind.OpenVino && useOpenVinoCpuProxy)
            ? new(true, null, 0)
            : provider is ExecutionProviderKind.OpenVino
                ? new(false, ExecutionProviderKind.OpenVino, deviceId ?? 0)
                : new(false, null, deviceId ?? 0);
    private static AdmissionBucket BucketOf(SessionPoolKey key, ExecutionProviderKind createdProvider) =>
        createdProvider == key.Provider
            ? BucketOf(key)
            : BucketOf(createdProvider, key.DeviceId, key.UseOpenVinoCpuProxy);
    private long BudgetFor(AdmissionBucket bucket) =>
        bucket.IsHost ? hostMemoryBudgetMb : memoryBudgetMb;
    private static string DescribeBucket(AdmissionBucket bucket) =>
        bucket.IsHost
            ? "host RAM"
            : bucket.AcceleratorProvider is ExecutionProviderKind.OpenVino
                ? $"OpenVINO NPU device {bucket.DeviceId}"
                : $"accelerator device {bucket.DeviceId}";
    private long CurrentReservedMb(AdmissionBucket bucket)
    {
        long pooled = 0;
        foreach (KeyValuePair<SessionPoolKey, PoolEntry> pair in entries.Where(pair => pair.Value.Bucket == bucket))
        {
            pooled += ResolveReservationMb(pair.Key);
        }

        long external = 0;
        foreach (KeyValuePair<Guid, ExternalReservationState> pair in externalReservations)
        {
            if (pair.Value.Bucket == bucket)
            {
                external += pair.Value.EstimatedMemoryMb;
            }
        }

        pendingCreateMbByBucket.TryGetValue(bucket, out long pending);
        return pooled + pending + external;
    }
    private const long BytesPerMegabyte = 1024L * 1024L;
    /// <summary>
    /// One observation of this process's dedicated GPU footprint for a single admission
    /// decision: the process total plus the per-adapter breakdown when the reader can
    /// attribute it. <see langword="null"/> whenever the reading cannot inform admission: no
    /// host registered a reader, the operator opted out, this platform or driver cannot report
    /// it (a GPU-idle process publishes no counter instance at all), or the probe failed. A
    /// null observation leaves reservation accounting exactly as it was.
    /// </summary>
    private sealed record ProcessGpuObservation(long TotalMb, IReadOnlyDictionary<long, long>? ByAdapterLuidMb);
    private ProcessGpuObservation? ReadObservedProcessGpu()
    {
        if (!SharedPoolOptions.EnableProcessGpuAdmission)
        {
            return null;
        }

        try
        {
            IProcessGpuMemoryReader? reader = processGpuMemoryReader?.Invoke();
            if (reader is null)
            {
                return null;
            }

            (long? bytes, IReadOnlyDictionary<long, long>? byLuid) = reader.ReadDedicatedGpuMemory();
            if (bytes is not > 0)
            {
                return null;
            }

            long totalMb = bytes.Value / BytesPerMegabyte;
            Dictionary<long, long>? byLuidMb = null;
            if (byLuid is not null)
            {
                byLuidMb = new Dictionary<long, long>(byLuid.Count);
                foreach ((long luid, long luidBytes) in byLuid)
                {
                    if (luidBytes > 0)
                    {
                        byLuidMb[luid] = luidBytes / BytesPerMegabyte;
                    }
                }
            }

            BenchmarkPhaseCapture.ObserveMaximum("observedProcessGpuMb", totalMb);
            return new ProcessGpuObservation(totalMb, byLuidMb);
        }
        catch (Exception exception) when (exception is not OperationCanceledException)
        {
            // A failing probe (missing pdh.dll, a driver without the counter set, denied access)
            // or a mis-wired source must degrade the observation, never the admission path: an
            // unobserved pool keeps its reservation-only behaviour.
            return null;
        }
    }
    /// <summary>
    /// This process's dedicated GPU usage attributed to <paramref name="bucket"/>, in MB, or 0
    /// when there is nothing to attribute.
    /// </summary>
    /// <remarks>
    /// When the reader attributes usage per adapter and the host mapped this device to its
    /// adapter LUID, the bucket is charged exactly its own adapter's footprint: usage on other
    /// adapters never blocks it, and sibling reservations are never subtracted twice. Otherwise
    /// the full process total is the conservative upper bound for the deciding GPU. Sibling
    /// reservations are estimates, not measured allocations: subtracting them could hide real
    /// usage on this adapter. Host and NPU buckets are never charged with GPU observations.
    /// </remarks>
    private long ObservedProcessGpuUsageMb(AdmissionBucket bucket, ProcessGpuObservation? observation)
    {
        if (bucket.IsHost || observation is null)
        {
            return 0;
        }

        if (bucket.AcceleratorProvider is not null)
        {
            // NPU buckets are never charged with GPU observations.
            return 0;
        }

        if (observation.ByAdapterLuidMb is null || SharedPoolOptions.AdapterLuidMap is not { } map)
        {
            // No per-adapter breakdown or no LUID mapping for this device: the full
            // process total is the conservative upper bound for the deciding GPU.
            return observation.TotalMb;
        }

        if (map.TryGetValue(bucket.DeviceId, out long adapterLuid))
        {
            // The breakdown is complete when present: no entry means nothing on this adapter.
            return observation.ByAdapterLuidMb.TryGetValue(adapterLuid, out long deviceMb) ? deviceMb : 0;
        }

        return observation.TotalMb;
    }
    /// <summary>
    /// Admission usage for <paramref name="bucket"/>: the pool's committed reservations, floored
    /// by this process's real dedicated GPU usage for accelerator buckets. The floor is what makes
    /// the process-isolated reading — and accelerator memory held outside the pool — part of the
    /// admission decision rather than telemetry only.
    /// </summary>
    private long AdmissionUsageMb(AdmissionBucket bucket, ProcessGpuObservation? observation)
    {
        // In-flight creates hold a pending reservation but have not allocated yet, so the
        // process reading cannot contain them: charge them on top of the observed floor.
        // Otherwise a pending reservation disappears into the observation whenever the
        // observation is the binding term, and concurrent admissions overshoot the budget.
        pendingCreateMbByBucket.TryGetValue(bucket, out long pendingMb);
        return Math.Max(
            CurrentReservedMb(bucket),
            ObservedProcessGpuUsageMb(bucket, observation) + pendingMb);
    }
    private void AddPendingReservation(AdmissionBucket bucket, long mb)
    {
        pendingCreateMbByBucket.AddOrUpdate(bucket, mb, (_, existing) => existing + mb);
        // Process-wide maximum: host pending plus every accelerator bucket's pending.
        BenchmarkPhaseCapture.ObserveMaximum(
            "pendingReservationMb",
            pendingCreateMbByBucket.Values.Sum());
    }
    private void ReleaseReservation(AdmissionBucket bucket, long mb) =>
        pendingCreateMbByBucket.AddOrUpdate(bucket, 0, (_, existing) => Math.Max(0, existing - mb));
    /// <summary>
    /// Whether <paramref name="bucket"/> holds any pooled session (idle or leased) or live
    /// external reservation that could still turn over. A leased session becomes evictable on
    /// release, so its presence means an observation-held wait can still make progress.
    /// </summary>
    private bool BucketHasLiveEntries(AdmissionBucket bucket)
    {
        bool pooledLive = entries.Any(pair => pair.Value.Bucket == bucket);
        bool externalLive = externalReservations.Values.Any(state => state.Bucket == bucket);
        if (pooledLive || externalLive)
        {
            return true;
        }

        // An in-flight create holds a pending reservation whose factory will turn over (publish
        // an entry or release the reservation), so it is live work too: the busy bound must
        // apply while it runs, or a slow factory in an otherwise-empty bucket trips the short
        // fail-fast before the pending drains from the observation.
        return pendingCreateMbByBucket.TryGetValue(bucket, out long pendingMb) && pendingMb > 0;
    }
    /// <summary>
    /// Evicts idle sessions in <paramref name="bucket"/> until <paramref name="needMb"/> fits,
    /// then takes the pending reservation. Eviction is driven by reservations only: evicting a
    /// pooled entry cannot lower the process-GPU observation snapshot, so an observation-blocked
    /// admission is left to <see cref="WaitForAdmissionBudgetAsync"/>, which re-observes while
    /// the released sessions' memory drains.
    /// </summary>
    /// <returns><see langword="true"/> when the reservation was taken.</returns>
    private bool TryReserveAdmissionBudget(AdmissionBucket bucket, long needMb, ProcessGpuObservation? observation)
    {
        // Other buckets have their own budgets (host RAM vs each accelerator).
        while (CurrentReservedMb(bucket) + needMb > BudgetFor(bucket))
        {
            PoolEntry? evictedForBudget = TryEvictLruIdle(onlyBucket: bucket);
            if (evictedForBudget is null)
            {
                break;
            }

            evictedForBudget.Dispose();
        }

        if (AdmissionUsageMb(bucket, observation) + needMb <= BudgetFor(bucket))
        {
            AddPendingReservation(bucket, needMb);
            return true;
        }

        return false;
    }
    /// <summary>
    /// Consecutive wait passes blocked by the process-GPU observation alone — with no evictable
    /// pooled or external work left — before admission fails fast instead of parking until the
    /// caller cancels. Each pass waits ~50 ms, so the default bounds the stall at ~5 s. Mutable
    /// (rather than const) so tests can pin the fail-fast without waiting out the production
    /// bound.
    /// </summary>
    internal static int ObservedBlockFailFastPasses { get; set; } = 100;
    /// <summary>
    /// Same bound for passes where the bucket still holds live-but-unevictable entries (held
    /// leases, pinned sessions, live external reservations): their turnover can still make
    /// progress — a released lease becomes evictable — so they get a longer stall (~60 s)
    /// before the same fail-fast. A lease held for a whole stage, a nested acquire by the
    /// waiting caller itself, or a resident external without an eviction callback never turns
    /// over while this waiter blocks, so the wait stays bounded for those too.
    /// </summary>
    internal static int ObservedBlockBusyFailFastPasses { get; set; } = 1200;
    /// <summary>
    /// How long one admission wait may run before it logs once (device, observed floor,
    /// budget, pending charge). This wait was previously invisible — no phase span and no
    /// log line — so a tight process-GPU floor could burn hundreds of seconds here with
    /// no trace in run evidence. Mutable so tests can pin the threshold.
    /// </summary>
    internal static TimeSpan AdmissionWaitLogThreshold { get; set; } = TimeSpan.FromSeconds(1);
    /// <summary>
    /// Waits until <paramref name="needMb"/> fits in <paramref name="bucket"/>'s budget
    /// (evicting idle sessions in that bucket as needed), then takes the reservation.
    /// Never holds a reservation while waiting unless <paramref name="alreadyReserved"/> is set.
    /// Host-backed providers share the host RAM budget; accelerator providers share their
    /// device's VRAM budget.
    /// </summary>
    /// <param name="alreadyReserved">
    /// When <see langword="true"/>, the caller already holds the <paramref name="needMb"/>
    /// reservation in <paramref name="bucket"/> (a session that already exists and is resident),
    /// so the wait only brings the bucket's total — including that reservation — within budget
    /// and does not take a new one on success.
    /// </param>
    /// <summary>
    /// Per-pass state for <see cref="WaitForAdmissionBudgetAsync"/>: the observed GPU
    /// reading, blocked flags, and eviction candidates collected under
    /// <see cref="creationLock"/>, plus the stall bookkeeping that carries across passes.
    /// </summary>
    private sealed class AdmissionWaitPass
    {
        public required long NeedMb { get; init; }
        public required AdmissionBucket Bucket { get; init; }
        public required bool AlreadyReserved { get; init; }
        public required System.Diagnostics.Stopwatch WaitStopwatch { get; init; }
        public bool WaitLogged;
        public int ObservedStallPasses;
        public bool HadLiveEntries;
        public ProcessGpuObservation? Observation;
        public bool Acquired;
        public List<PoolEntry>? ToDispose;
        public List<KeyValuePair<Guid, ExternalReservationState>>? IdleExternals;
        public bool ReservationBlocked;
        public bool ObservedBlocked;

        /// <summary>A reservation already held for a resident session is part of the bucket's
        /// current total; only a not-yet-reserved needMb is an additional charge.</summary>
        public long ChargeMb => AlreadyReserved ? 0 : NeedMb;

        public void Reset()
        {
            Acquired = false;
            ToDispose = null;
            IdleExternals = null;
            ReservationBlocked = false;
            ObservedBlocked = false;
        }
    }

    private async Task WaitForAdmissionBudgetAsync(
        long needMb,
        AdmissionBucket bucket,
        CancellationToken cancellationToken,
        bool alreadyReserved = false)
    {
        // Mirror the other pool waits (pool-wait, pool-single-flight-wait,
        // pool-creation-lock-wait): without this span the admission wait is the one
        // async wait in the create path that leaves no trace in phase evidence.
        using BenchmarkPhaseCapture.Timer admissionWaitSpan = BenchmarkPhaseCapture.Start("pool-admission-wait");
        var pass = new AdmissionWaitPass
        {
            NeedMb = needMb,
            Bucket = bucket,
            AlreadyReserved = alreadyReserved,
            WaitStopwatch = System.Diagnostics.Stopwatch.StartNew(),
        };
        BenchmarkPhaseCapture.ObserveMaximum("admissionWaiters", Interlocked.Increment(ref admissionWaiters));
        try
        {
            while (true)
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                cancellationToken.ThrowIfCancellationRequested();

                // Re-observe every iteration: this loop evicts idle work and then waits, so the
                // next pass can see the process's real usage fall as released sessions drain.
                // Host-RAM buckets are never charged with the observation (see
                // ObservedProcessGpuUsageMb), so they skip the probe instead of polling GPU
                // counters every 50 ms.
                pass.Observation = bucket.IsHost ? null : ReadObservedProcessGpu();
                pass.Reset();

                await EvictAndProbeBudgetAsync(pass, cancellationToken).ConfigureAwait(false);
                DisposeEvictedEntries(pass);
                if (pass.Acquired)
                {
                    return;
                }

                if (TryEvictIdleExternalReservations(pass))
                {
                    continue;
                }

                TrackObservedStall(pass);

                if (pass.ReservationBlocked || pass.ObservedBlocked)
                {
                    LogLongAdmissionWait(pass);

                    // Doesn't fit yet — poll instead of blocking a thread-pool thread on
                    // Monitor.Wait. Bounded delay; loops back to retake creationLock and
                    // recheck (an eviction, a lease release, or the process's own GPU usage
                    // draining may have freed headroom).
                    await Task.Delay(50, cancellationToken).ConfigureAwait(false);
                }
            }
        }
        finally
        {
            Interlocked.Decrement(ref admissionWaiters);
        }
    }

    /// <summary>
    /// One probe under <see cref="creationLock"/>: evicts idle entries until reservations
    /// fit, computes the blocked flags, collects eviction candidates, and takes the
    /// pending reservation when everything fits.
    /// </summary>
    private async Task EvictAndProbeBudgetAsync(AdmissionWaitPass pass, CancellationToken cancellationToken)
    {
        await creationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            EvictIdleTowardReservation(pass);

            pass.ReservationBlocked = CurrentReservedMb(pass.Bucket) + pass.ChargeMb > BudgetFor(pass.Bucket);
            pass.ObservedBlocked = !pass.ReservationBlocked
                && AdmissionUsageMb(pass.Bucket, pass.Observation) + pass.ChargeMb > BudgetFor(pass.Bucket);

            CollectBlockedCandidates(pass);
            TakeReservationIfFits(pass);
        }
        finally
        {
            creationLock.Release();
        }
    }

    /// <summary>
    /// Evicts idle pooled entries until the bucket's reservation total fits the charge.
    /// Eviction is bounded by what it can achieve: evicting a pooled entry lowers
    /// the reservation total but each eviction is re-observed before the next, so
    /// an observation-held admission converges instead of discarding the device's
    /// warm cache in one pass.
    /// </summary>
    private void EvictIdleTowardReservation(AdmissionWaitPass pass)
    {
        while (CurrentReservedMb(pass.Bucket) + pass.ChargeMb > BudgetFor(pass.Bucket))
        {
            PoolEntry? evicted = TryEvictLruIdle(onlyBucket: pass.Bucket);
            if (evicted is null)
            {
                break;
            }

            pass.ToDispose ??= new List<PoolEntry>();
            pass.ToDispose.Add(evicted);
        }
    }

    /// <summary>
    /// When the bucket stays blocked after the reservation evictions, collects the next
    /// candidates: one more pooled entry for an observation-held bucket, or the ordered
    /// list of idle external reservations when nothing pooled is evictable.
    /// </summary>
    private void CollectBlockedCandidates(AdmissionWaitPass pass)
    {
        if (pass.ObservedBlocked)
        {
            // Reservations now fit (the loop above may have evicted to get here):
            // evict at most one idle entry per pass. Its real allocation sits inside
            // the observation, so the next pass re-observes lower once that memory
            // drains; evicting more per pass would throw away warm sessions faster
            // than the observation can confirm they were the blockage.
            PoolEntry? single = TryEvictLruIdle(onlyBucket: pass.Bucket);
            if (single is not null)
            {
                pass.ToDispose ??= new List<PoolEntry>();
                pass.ToDispose.Add(single);
            }
            else
            {
                // No pooled ONNX entry in this bucket is evictable; offer every
                // idle external reservation a chance to release its resource,
                // oldest first.
                pass.IdleExternals = OrderedIdleExternals(pass.Bucket);
            }
        }
        else if (pass.ReservationBlocked)
        {
            // No pooled ONNX entry in this bucket is evictable; offer every idle
            // external reservation a chance to release its resource, oldest first.
            pass.IdleExternals = OrderedIdleExternals(pass.Bucket);
        }
    }

    /// <summary>Takes the pending reservation when neither reservations nor the observed
    /// process footprint blocks the charge.</summary>
    private void TakeReservationIfFits(AdmissionWaitPass pass)
    {
        if (pass.ReservationBlocked || pass.ObservedBlocked)
        {
            return;
        }

        if (!pass.AlreadyReserved)
        {
            AddPendingReservation(pass.Bucket, pass.NeedMb);
        }

        pass.Acquired = true;
    }

    private static void DisposeEvictedEntries(AdmissionWaitPass pass)
    {
        if (pass.ToDispose is null)
        {
            return;
        }

        foreach (PoolEntry entry in pass.ToDispose)
        {
            BenchmarkPhaseCapture.Increment("admissionWaitEvictions");
            entry.Dispose();
        }
    }

    /// <summary>
    /// Offers every idle external reservation a chance to release its resource, oldest
    /// first. Returns <see langword="true"/> when one evicted, so the caller retries the
    /// pass with the freed memory visible in the next observation.
    /// </summary>
    private bool TryEvictIdleExternalReservations(AdmissionWaitPass pass)
    {
        if (pass.IdleExternals is null)
        {
            return false;
        }

        // Invoke outside creationLock in LRU order: a refusal or throwing
        // callback must not starve newer evictable candidates. Callbacks that
        // declined stay registered — the resource may go idle later.
        foreach (KeyValuePair<Guid, ExternalReservationState> candidate in pass.IdleExternals)
        {
            bool evictedExternal;
            try
            {
                evictedExternal = candidate.Value.TryEvictIdle!();
            }
            catch
            {
                // A failing callback must not corrupt admission — treat as refusal.
                evictedExternal = false;
            }

            if (evictedExternal)
            {
                // Defensive removal: a well-behaved callback already disposed
                // the token.
                externalReservations.TryRemove(candidate);
                SignalBundleStateChanged();
                // Turnover through eviction is progress for the stall bound: the
                // freed memory needs a re-observe to drain, and a freshly-emptied
                // bucket must not inherit the busy phase's accumulated passes.
                pass.ObservedStallPasses = 0;
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Stall accounting for observation-held passes: consecutive passes that free
    /// nothing count toward the fail-fast bound, with the longer busy bound while
    /// live-but-unevictable entries can still turn over.
    /// </summary>
    private void TrackObservedStall(AdmissionWaitPass pass)
    {
        if (!pass.ObservedBlocked)
        {
            pass.ObservedStallPasses = 0;
            return;
        }

        // Any eviction this pass may still drain: the next pass re-observes, so only
        // consecutive passes that free nothing count toward the fail-fast bound.
        // Live-but-unevictable entries (held leases, pins, live externals) get the
        // longer busy bound since their turnover can still unblock the wait; an
        // empty bucket fails on the shorter bound.
        bool progressed = pass.ToDispose is { Count: > 0 };
        // Turnover that happens outside this loop — a live external released by its
        // owner, or the last live entry going idle without being evicted here — is
        // progress too: the busy bound existed to wait for it, so a live-to-empty
        // transition must not trip the empty-bucket fail-fast on the very next pass
        // before the freed memory has had a single re-observe to drain.
        bool hasLiveEntries = BucketHasLiveEntries(pass.Bucket);
        if (!hasLiveEntries && pass.HadLiveEntries)
        {
            progressed = true;
        }

        pass.HadLiveEntries = hasLiveEntries;
        if (progressed)
        {
            pass.ObservedStallPasses = 0;
        }
        else
        {
            pass.ObservedStallPasses++;
        }

        int stallBound = hasLiveEntries
            ? ObservedBlockBusyFailFastPasses
            : ObservedBlockFailFastPasses;
        if (pass.ObservedStallPasses >= stallBound)
        {
            long attributedMb = ObservedProcessGpuUsageMb(pass.Bucket, pass.Observation);
            throw new InvalidOperationException(
                $"Accelerator admission for '{DescribeBucket(pass.Bucket)}' needs ~{pass.NeedMb} MB, but ~{attributedMb} MB "
                + "of dedicated GPU memory is attributed to this bucket against its admission budget of "
                + $"{BudgetFor(pass.Bucket)} MB, with no evictable sessions or external reservations left to "
                + "free. Free GPU memory, raise the budget "
                + $"({SharedPoolOptions.BudgetMbVariable}), or opt out of process-GPU admission "
                + $"({SharedPoolOptions.ProcessGpuAdmissionVariable}=0).");
        }
    }

    private void LogLongAdmissionWait(AdmissionWaitPass pass)
    {
        if (pass.WaitLogged || pass.WaitStopwatch.Elapsed < AdmissionWaitLogThreshold)
        {
            return;
        }

        pass.WaitLogged = true;
        pendingCreateMbByBucket.TryGetValue(pass.Bucket, out long pendingMb);
        string message =
            $"Session pool admission wait on {DescribeBucket(pass.Bucket)}: needs ~{pass.NeedMb} MB against a "
            + $"{BudgetFor(pass.Bucket)} MB budget (observed process GPU floor "
            + $"~{ObservedProcessGpuUsageMb(pass.Bucket, pass.Observation)} MB, reserved "
            + $"~{CurrentReservedMb(pass.Bucket)} MB incl. ~{pendingMb} MB pending, "
            + $"waited {pass.WaitStopwatch.Elapsed.TotalSeconds:F1}s); waiting for memory "
            + "to drain or sessions to be released.";
        if (ColdStartCapture.Current?.Logger is { } logger)
        {
            logger.LogInformation(message);
        }
        else
        {
            System.Diagnostics.Trace.TraceInformation(message);
        }
    }
}
