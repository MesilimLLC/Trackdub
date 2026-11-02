using System.Collections.Concurrent;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.TensorRtRtx;

namespace Trackdub.Inference.Onnx.Pool;

// Multi-graph bundle acquisition: all-or-nothing admission holds, pin/publish,
// and validation for SessionLeaseRequest batches.
internal sealed partial class InferenceSessionPool
{
    /// <summary>
    /// Serializes only the pin/publish + gate-acquire phase of bundle acquisition
    /// (C1 shape b), in <see cref="SessionPoolKey.StableComparer"/> order. Session
    /// creation itself runs in parallel before this lock is taken.
    /// </summary>
    private readonly SemaphoreSlim bundleAcquireLock = new(1, 1);
    private TaskCompletionSource<bool> bundleStateChanged = CreateBundleStateChangedSignal();
    /// <summary>
    /// One bundle's aggregate admission credit, per bucket and per graph. The credit
    /// already sits in <see cref="pendingCreateMbByBucket"/> (taken all-or-nothing by
    /// <see cref="AcquireBundleAdmissionHoldAsync"/> before any graph in the bundle is
    /// created); a graph create that consumes credit here skips its own per-graph
    /// admission wait, and the create path's existing reservation release on publish or
    /// failure draws the pending charge down exactly as for a per-graph reservation.
    /// Unconsumed credit (graphs that turned out resident, or a failed bundle) is
    /// returned by <see cref="ReleaseBundleHoldCredit"/>.
    /// Credit is scoped to the keys it was sized for: only a graph that was missing at
    /// hold time may consume, and only its own share. A resident graph evicted by the
    /// hold wait itself (or by a concurrent admission) falls back to ordinary per-graph
    /// admission instead of stealing a sibling's share — stealing leaves an
    /// all-or-nothing sliver no sibling can consume, and the leftover stays pending
    /// until every warm finishes, so the fallback wait sees a phantom over-count with
    /// the bundle's own early pins unevictable and no fail-fast on reservation-blocked
    /// waits. A graph that turns out resident after all relinquishes its share on the
    /// fast path so concurrent fallback waits never see it as phantom pending.
    /// </summary>
    private sealed class BundleAdmissionHold
    {
        private readonly object gate = new();
        private readonly Dictionary<AdmissionBucket, Dictionary<SessionPoolKey, long>> creditByKey = new();

        public void AddCredit(AdmissionBucket bucket, SessionPoolKey key, long mb)
        {
            lock (gate)
            {
                if (!creditByKey.TryGetValue(bucket, out Dictionary<SessionPoolKey, long>? byKey))
                {
                    byKey = new Dictionary<SessionPoolKey, long>();
                    creditByKey[bucket] = byKey;
                }

                byKey[key] = byKey.GetValueOrDefault(key) + mb;
            }
        }

        public bool TryConsume(AdmissionBucket bucket, SessionPoolKey key, long mb)
        {
            lock (gate)
            {
                if (!creditByKey.TryGetValue(bucket, out Dictionary<SessionPoolKey, long>? byKey))
                {
                    return false;
                }

                if (!byKey.TryGetValue(key, out long credit) || credit < mb)
                {
                    return false;
                }

                long remaining = credit - mb;
                if (remaining <= 0)
                {
                    byKey.Remove(key);
                }
                else
                {
                    byKey[key] = remaining;
                }

                return true;
            }
        }

        /// <summary>
        /// Returns the still-unconsumed share for <paramref name="key"/> (a graph that
        /// turned out resident needs no create) so the caller can release it from the
        /// pending reservation immediately instead of leaving phantom pending until the
        /// whole bundle finishes warming.
        /// </summary>
        public bool TryRelinquish(AdmissionBucket bucket, SessionPoolKey key, out long mb)
        {
            lock (gate)
            {
                mb = 0;
                if (!creditByKey.TryGetValue(bucket, out Dictionary<SessionPoolKey, long>? byKey)
                    || !byKey.TryGetValue(key, out long credit))
                {
                    return false;
                }

                byKey.Remove(key);
                mb = credit;
                return mb > 0;
            }
        }

        public List<KeyValuePair<AdmissionBucket, long>> DrainRemaining()
        {
            lock (gate)
            {
                var remaining = new List<KeyValuePair<AdmissionBucket, long>>(creditByKey.Count);
                foreach (KeyValuePair<AdmissionBucket, Dictionary<SessionPoolKey, long>> pair in creditByKey)
                {
                    long total = 0;
                    foreach (long mb in pair.Value.Values)
                    {
                        total += mb;
                    }

                    if (total > 0)
                    {
                        remaining.Add(new KeyValuePair<AdmissionBucket, long>(pair.Key, total));
                    }
                }

                creditByKey.Clear();
                return remaining;
            }
        }
    }
    private void ReleaseBundleHoldCredit(BundleAdmissionHold hold)
    {
        foreach (KeyValuePair<AdmissionBucket, long> remaining in hold.DrainRemaining())
        {
            if (remaining.Value > 0)
            {
                ReleaseReservation(remaining.Key, remaining.Value);
            }
        }
    }
    /// <summary>
    /// Takes one aggregate admission reservation per bucket for every graph in the
    /// bundle that is not already resident, before any graph is created. The wait is
    /// all-or-nothing per bucket — <see cref="WaitForAdmissionBudgetAsync"/> only takes
    /// its pending reservation on success — so a bundle still waiting here holds no
    /// pins and (within a bucket) no reservation another bundle could deadlock against.
    /// Buckets are taken in one global order (host, then device id, then accelerator
    /// provider) for every bundle, so multi-bucket bundles cannot take them in
    /// opposite orders either. Graphs already resident are excluded: their estimate
    /// already counts in the bucket's pooled total, and charging them again could make
    /// this wait evict the very sessions the bundle is about to reuse. A resident graph
    /// evicted before its warm runs simply falls back to ordinary per-graph admission.
    /// </summary>
    private async Task<BundleAdmissionHold> AcquireBundleAdmissionHoldAsync(
        SessionLeaseRequest[] ordered,
        CancellationToken cancellationToken)
    {
        var hold = new BundleAdmissionHold();
        try
        {
            IEnumerable<IGrouping<AdmissionBucket, SessionLeaseRequest>> groups = ordered
                .GroupBy(request => BucketOf(request.Key))
                .OrderBy(group => group.Key.IsHost ? 0 : 1)
                .ThenBy(group => group.Key.DeviceId)
                .ThenBy(group => group.Key.AcceleratorProvider?.ToString(), StringComparer.Ordinal);
            foreach (IGrouping<AdmissionBucket, SessionLeaseRequest> group in groups)
            {
                SessionLeaseRequest[] missing = group
                    .Where(request => !entries.ContainsKey(request.Key))
                    .ToArray();
                if (missing.Length == 0)
                {
                    continue;
                }

                long missingMb = missing.Sum(request => ResolveReservationMb(request.Key));
                await WaitForAdmissionBudgetAsync(missingMb, group.Key, cancellationToken).ConfigureAwait(false);
                foreach (SessionLeaseRequest request in missing)
                {
                    hold.AddCredit(group.Key, request.Key, ResolveReservationMb(request.Key));
                }
            }

            return hold;
        }
        catch
        {
            ReleaseBundleHoldCredit(hold);
            throw;
        }
    }
    private static TaskCompletionSource<bool> CreateBundleStateChangedSignal() =>
        new(TaskCreationOptions.RunContinuationsAsynchronously);
    /// <summary>
    /// Wakes bundle acquisition after a gate or pool membership changes. The exchange makes the
    /// signal one-shot: a waiter captures the current source before its attempt, while a later
    /// transition installs a fresh source for the next waiter.
    /// </summary>
    private void SignalBundleStateChanged()
    {
        TaskCompletionSource<bool> previous =
            Interlocked.Exchange(ref bundleStateChanged, CreateBundleStateChangedSignal());
        previous.TrySetResult(true);
    }
    /// <summary>
    /// Validates that a bundle's aggregate estimate per admission bucket fits that bucket's
    /// budget. Buckets follow the provider each session was actually created with: a published
    /// entry carries its creation bucket (<see cref="PoolEntry.Bucket"/>), so a GPU-keyed CPU
    /// fallback is charged against host RAM rather than the accelerator bucket implied by its
    /// key. Missing entries are checked against their key's bucket. Fail-fast at this point
    /// (before any factory runs, and again after each residency pin) prevents a bundle from
    /// pinning a fallback session and then waiting forever on memory that its own pin holds.
    /// </summary>
    private void ValidateBundleAdmission(SessionLeaseRequest[] ordered)
    {
        foreach (IGrouping<AdmissionBucket, SessionLeaseRequest> group in
                 ordered.GroupBy(request =>
                     entries.TryGetValue(request.Key, out PoolEntry? entry) && entry is not null
                         ? entry.Bucket
                         : BucketOf(request.Key)))
        {
            long totalMb = group.Sum(request => ResolveReservationMb(request.Key));
            long budgetMb = BudgetFor(group.Key);
            if (totalMb > budgetMb)
            {
                throw new InvalidOperationException(
                    $"Session bundle needs ~{totalMb} MB across {group.Count()} graph(s) on " +
                    $"{DescribeBucket(group.Key)}, which exceeds its admission budget of {budgetMb} MB.");
            }
        }
    }
    /// <summary>
    /// Creates (or reuses) one bundle graph and, under memory admission, pins it
    /// resident the moment its own create completes, so it is never idle-eviction-
    /// eligible while sibling graphs are still being created. The caller owns the
    /// returned pin (<see langword="null"/> in count mode, where a warm may land
    /// ephemeral and never reach <c>entries</c>, so a residency pin could spin — the
    /// warm lease is released instead, as in the pin phase). Drawing on the bundle's
    /// aggregate <paramref name="bundleHold"/> means the create does not re-run
    /// per-graph admission against the floor its own siblings raised.
    /// </summary>
    private async Task<SessionResidency?> WarmBundleGraphAsync(
        SessionLeaseRequest request,
        BundleAdmissionHold? bundleHold,
        CancellationToken cancellationToken)
    {
        // Production factories are ct => Task.FromResult(CreateSession(...)): the whole
        // create runs synchronously on the calling thread, so an uncontended create path
        // never yields and the bundle loop would only start graph i+1 after graph i
        // completed — serial creation despite the warm-task list. Offload each graph so
        // the loop actually overlaps them. Task.Run flows the ambient async context
        // (phase capture), the admission hold is lock-gated, the token stops not-yet-
        // started work and is still passed through to the create itself, and the pin-on-
        // complete inside the residency path stays glued to its create.
        if (enableMemoryAdmission)
        {
            return await Task.Run(
                () => GetResidencyReportingProviderAsync(
                    request.Key, request.CreateReportingProviderAsync, cancellationToken, bundleHold),
                cancellationToken).ConfigureAwait(false);
        }

        using SessionLease warm = await Task.Run(
            () => GetLeaseReportingProviderAsync(
                request.Key, request.CreateReportingProviderAsync, cancellationToken),
            cancellationToken).ConfigureAwait(false);
        return null;
    }
    /// <summary>
    /// Acquires every graph in <paramref name="requests"/> as one all-or-nothing bundle
    /// (audit §3A: never hold an encoder while waiting indefinitely for a decoder).
    /// C1 shape (b), admission-safe: under memory admission one aggregate reservation
    /// per bucket is taken up front (<see cref="AcquireBundleAdmissionHoldAsync"/>),
    /// so graph sessions can then be created in parallel with no process-wide lock
    /// held and no per-graph admission re-wait — per-key single-flight creation in
    /// <see cref="GetLeaseReportingProviderAsync"/> draws on the aggregate credit and
    /// serializes only the brief publish step per key. Each graph is pinned resident
    /// the moment its own create completes (see <see cref="WarmBundleGraphAsync"/>),
    /// so a warmed graph is never idle-eviction-eligible while its siblings are still
    /// creating. Holding those early pins through phase 1 is deadlock-safe precisely
    /// because of the aggregate order: a bundle only pins graphs after its whole
    /// aggregate was granted and its remaining creates are pre-reserved, while a
    /// bundle still waiting for its aggregate holds neither pins nor (per bucket) a
    /// reservation for another bundle to deadlock against. The early pins are
    /// released before waiting for <c>bundleAcquireLock</c> (phase 2 re-pins every
    /// graph), so a bundle parked at the lock never holds unevictable memory a lock
    /// holder's phase-2 admission re-create could wait on. Finally only pin/publish
    /// is serialized,
    /// under <c>bundleAcquireLock</c> and in <see cref="SessionPoolKey.StableComparer"/>
    /// order: residency pins are (re)taken in that order and exclusive gates are tried
    /// non-blockingly in that order, rolling back on any busy gate. A bundle whose
    /// graphs were removed between the parallel create and the pin phase (explicit
    /// eviction ignores pins) re-creates them inside the pin phase, where the
    /// aggregate admission check (repeated after each pin) still fails fast instead
    /// of waiting on memory its own pins hold. Duplicate keys are rejected. When
    /// memory admission is enabled, an aggregate per-bucket budget check fails the
    /// bundle before any factory runs.
    /// </summary>
    public async Task<SessionLeaseBundle> GetLeaseBundleAsync(
        IReadOnlyList<SessionLeaseRequest> requests,
        CancellationToken cancellationToken)
    {
        SessionLeaseRequest[] ordered = ValidateBundleRequests(requests);

        if (enableMemoryAdmission)
        {
            // Aggregate preflight: a bundle whose graphs jointly exceed a bucket's budget
            // can never fit, so fail before invoking any factory. Buckets follow the provider
            // each session was actually created with (a published CPU fallback is charged to
            // host RAM, not its GPU key's bucket), so the check is repeated after each
            // residency pin below — pinning a fallback session can change a bucket's total.
            ValidateBundleAdmission(ordered);
        }

        List<SessionResidency> earlyPins = await WarmAllBundleGraphsAsync(ordered, cancellationToken).ConfigureAwait(false);

        // C1 shape (b), phase 2: serialize only pin/publish in StableComparer order.
        // While one bundle pins (re-creating any graph removed since phase 1) and
        // takes its gates, no other bundle is incrementally pinning in the opposite
        // order. Phase-1 early pins are NOT carried into this phase: phase 2 re-pins
        // every graph below, and holding early pins across the lock wait deadlocks
        // against a lock holder blocked in phase-2 admission re-creating an
        // explicitly-evicted graph — the waiter's pins are unevictable and cannot be
        // released until it owns the lock, while the holder waits on memory only those
        // pins could free, with no fail-fast on reservation-blocked waits.
        foreach (SessionResidency earlyPin in earlyPins)
        {
            earlyPin.Dispose();
        }

        var pins = new List<SessionResidency>();
        try
        {
            await bundleAcquireLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            foreach (SessionResidency pin in pins)
            {
                pin.Dispose();
            }

            throw;
        }
        try
        {
            try
            {
                return await PinAndAcquireBundleAsync(ordered, requests, pins, cancellationToken).ConfigureAwait(false);
            }
            finally
            {
                foreach (SessionResidency pin in pins)
                {
                    pin.Dispose();
                }
            }
        }
        finally
        {
            bundleAcquireLock.Release();
        }
    }

    /// <summary>
    /// Re-warm throttle state for the acquire loop: bounded retries (~5s at the 200ms
    /// cadence) so a pool stuck at capacity cannot spin until the caller cancels while
    /// holding <c>bundleAcquireLock</c>.
    /// </summary>
    private sealed class BundleRewarmState
    {
        public const int MaxAttempts = 25;
        public int Attempts;
        public readonly System.Diagnostics.Stopwatch Stopwatch = System.Diagnostics.Stopwatch.StartNew();
    }

    /// <summary>
    /// Serialized pin + acquire under <c>bundleAcquireLock</c>: re-pins every graph, then
    /// loops on acquire-or-rewarm until the bundle forms, the re-warm bound trips, or the
    /// caller cancels. Pins are released by the caller's finally.
    /// </summary>
    private async Task<SessionLeaseBundle> PinAndAcquireBundleAsync(
        SessionLeaseRequest[] ordered,
        IReadOnlyList<SessionLeaseRequest> requests,
        List<SessionResidency> pins,
        CancellationToken cancellationToken)
    {
        await PinAllBundleGraphsAsync(ordered, pins, cancellationToken).ConfigureAwait(false);

        var rewarm = new BundleRewarmState();
        while (true)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            cancellationToken.ThrowIfCancellationRequested();

            // Capture the signal before attempting the bundle. A lease release or pool
            // publication that races this attempt either wakes this waiter or is observed
            // by the next iteration, so no state change can be lost between check and wait.
            Task bundleStateSignal = Volatile.Read(ref bundleStateChanged).Task;
            SessionLeaseBundle? bundle = TryAcquireBundle(ordered, requests);
            if (bundle is not null)
            {
                // Execution gates now protect the entries; the residency pins have
                // done their job.
                foreach (SessionResidency pin in pins)
                {
                    pin.Dispose();
                }

                pins.Clear();
                return bundle;
            }

            await RewarmBundleIfDueAsync(rewarm, ordered, pins, cancellationToken).ConfigureAwait(false);
            await AwaitBundleStateChangeAsync(bundleStateSignal, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Re-warms missing graphs on a throttled ~200ms cadence and bounds retries: a pool
    /// staying at capacity (every entry leased or pinned) across
    /// <see cref="BundleRewarmState.MaxAttempts"/> attempts throws instead of spinning
    /// until the caller's cancellation token fires while holding bundleAcquireLock.
    /// </summary>
    private async Task RewarmBundleIfDueAsync(
        BundleRewarmState rewarm,
        SessionLeaseRequest[] ordered,
        List<SessionResidency> pins,
        CancellationToken cancellationToken)
    {
        if (rewarm.Stopwatch.Elapsed < TimeSpan.FromMilliseconds(200))
        {
            return;
        }

        rewarm.Stopwatch.Restart();
        SessionLeaseRequest[] missing =
            await RewarmMissingBundleGraphsAsync(ordered, pins, cancellationToken).ConfigureAwait(false);
        if (missing.Length == 0 || ordered.All(request => entries.ContainsKey(request.Key)))
        {
            return;
        }

        rewarm.Attempts++;
        if (rewarm.Attempts >= BundleRewarmState.MaxAttempts)
        {
            throw new InvalidOperationException(
                "Unable to acquire session bundle: the pool stayed at capacity " +
                $"(every entry leased or pinned) across {BundleRewarmState.MaxAttempts} re-warm attempts.");
        }
    }

    /// <summary>
    /// Waits for a state transition rather than polling every 10ms. The timeout is
    /// only a lost-wakeup safety net; normal progress is signaled by release,
    /// publication, and eviction paths below. Kept at the same ~200ms cadence as the
    /// re-warm gate above so the stuck-pool give-up bound stays near the documented
    /// ~5s (maxRewarmAttempts * 200ms) even when no signal ever fires.
    /// </summary>
    private static async Task AwaitBundleStateChangeAsync(Task bundleStateSignal, CancellationToken cancellationToken)
    {
        try
        {
            await bundleStateSignal.WaitAsync(TimeSpan.FromMilliseconds(200), cancellationToken).ConfigureAwait(false);
        }
        catch (TimeoutException)
        {
            // Lost-wakeup safety net; retry the bundle attempt.
        }
    }

    /// <summary>
    /// C1 shape (b), phase 1: takes one aggregate admission reservation for the
    /// whole bundle, then creates every graph in parallel, holding no lock. Each
    /// graph draws on the aggregate credit instead of re-running admission against
    /// the process-GPU floor its already-created siblings raised, and is pinned
    /// resident the moment its own create completes — so later graphs (and any
    /// concurrent admission) cannot evict it mid-acquire. A failure or
    /// cancellation here releases every pin taken so far and any unconsumed
    /// aggregate credit: no pinned half-bundle survives.
    /// Returns the residency pins of graphs whose create completed; the caller
    /// disposes them before phase 2.
    /// </summary>
    private async Task<List<SessionResidency>> WarmAllBundleGraphsAsync(
        SessionLeaseRequest[] ordered,
        CancellationToken cancellationToken)
    {
        BundleAdmissionHold? bundleHold = enableMemoryAdmission
            ? await AcquireBundleAdmissionHoldAsync(ordered, cancellationToken).ConfigureAwait(false)
            : null;
        var warmTasks = new List<Task<SessionResidency?>>(ordered.Length);
        List<SessionResidency> earlyPins;
        try
        {
            await StartBundleWarmTasksAsync(ordered, bundleHold, warmTasks, cancellationToken).ConfigureAwait(false);
            earlyPins = CollectEarlyPins(await Task.WhenAll(warmTasks).ConfigureAwait(false));
        }
        catch
        {
            // Drain any creates already in flight so no factory is still running when
            // the failure propagates, then release the pins of the graphs that did
            // complete: a failed or cancelled graph releases the whole bundle.
            await DrainWarmTasksAsync(warmTasks).ConfigureAwait(false);
            throw;
        }
        finally
        {
            // Every warm has finished (or been drained): consumed credit was drawn
            // down by the create path, so return whatever remains. Phase 2 re-creates
            // use ordinary per-graph admission under the bundle lock, as before.
            if (bundleHold is not null)
            {
                ReleaseBundleHoldCredit(bundleHold);
            }
        }

        return earlyPins;
    }

    /// <summary>
    /// Starts every bundle warm create and fills <paramref name="warmTasks"/> in request
    /// order. A create already known faulted before the next graph starts fails fast rather
    /// than starting more graphs for a failed bundle. (Best-effort now that warms run
    /// offloaded — a graph already in flight when a sibling faults still counts as its one
    /// attempt.)
    /// </summary>
    private async Task StartBundleWarmTasksAsync(
        SessionLeaseRequest[] ordered,
        BundleAdmissionHold? bundleHold,
        List<Task<SessionResidency?>> warmTasks,
        CancellationToken cancellationToken)
    {
        for (int i = 0; i < ordered.Length; i++)
        {
            Task<SessionResidency?> warm = WarmBundleGraphAsync(ordered[i], bundleHold, cancellationToken);
            warmTasks.Add(warm);
            if (warm.IsFaulted)
            {
                await warm.ConfigureAwait(false);
            }
        }
    }

    /// <summary>Residency pins of graphs whose warm create completed, in request order.</summary>
    private static List<SessionResidency> CollectEarlyPins(SessionResidency?[] pins)
    {
        var earlyPins = new List<SessionResidency>(pins.Length);
        foreach (SessionResidency? pin in pins)
        {
            if (pin is not null)
            {
                earlyPins.Add(pin);
            }
        }

        return earlyPins;
    }

    /// <summary>
    /// Drains in-flight warm creates and disposes the residency pins of graphs that
    /// completed, so a failed or cancelled bundle leaves no factory running and no
    /// pinned half-bundle behind.
    /// </summary>
    private static async Task DrainWarmTasksAsync(List<Task<SessionResidency?>> warmTasks)
    {
        try
        {
            await Task.WhenAll(warmTasks).ConfigureAwait(false);
        }
        catch
        {
            // The original failure (rethrown by the caller) is the meaningful one.
        }

        foreach (Task<SessionResidency?> warm in warmTasks)
        {
            if (warm.IsCompletedSuccessfully && warm.Result is { } pin)
            {
                pin.Dispose();
            }
        }
    }

    /// <summary>
    /// Pin phase (serialized, key order): sessions normally already exist
    /// from the parallel create phase — no execution gates held. Under hard
    /// admission, pin each graph resident so idle eviction cannot remove it
    /// before its gate is taken; a graph evicted since its create is
    /// re-created here. In count mode a warm may land ephemeral and
    /// never reach <c>entries</c>, so a residency pin could spin forever — just
    /// warm and release instead; the bounded re-warm retries.
    /// </summary>
    private async Task PinAllBundleGraphsAsync(
        SessionLeaseRequest[] ordered,
        List<SessionResidency> pins,
        CancellationToken cancellationToken)
    {
        foreach (SessionLeaseRequest request in ordered)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await PinOrWarmBundleGraphAsync(ordered, request, pins, cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Re-warms bundle graphs still missing from <c>entries</c>: a pinned entry survives
    /// idle eviction, but explicit EvictAllIdleAsync/EvictModelAsync calls remove it.
    /// Returns the requests that were missing when the pass started.
    /// </summary>
    private async Task<SessionLeaseRequest[]> RewarmMissingBundleGraphsAsync(
        SessionLeaseRequest[] ordered,
        List<SessionResidency> pins,
        CancellationToken cancellationToken)
    {
        SessionLeaseRequest[] missing =
            ordered.Where(request => !entries.ContainsKey(request.Key)).ToArray();
        foreach (SessionLeaseRequest request in missing)
        {
            await PinOrWarmBundleGraphAsync(ordered, request, pins, cancellationToken).ConfigureAwait(false);
        }

        return missing;
    }

    /// <summary>
    /// Under admission, pins one bundle graph resident and re-validates the aggregate
    /// fit; in count mode, warms and immediately releases the lease.
    /// </summary>
    private async Task PinOrWarmBundleGraphAsync(
        SessionLeaseRequest[] ordered,
        SessionLeaseRequest request,
        List<SessionResidency> pins,
        CancellationToken cancellationToken)
    {
        if (enableMemoryAdmission)
        {
            pins.Add(await GetResidencyReportingProviderAsync(request.Key, request.CreateReportingProviderAsync, cancellationToken)
                .ConfigureAwait(false));
            // Recheck after each pin: a session pinned now is counted in the bucket
            // it was created in, and a later admission must not wait on memory this
            // bundle's own pin holds.
            ValidateBundleAdmission(ordered);
        }
        else
        {
            using SessionLease warm = await GetLeaseReportingProviderAsync(request.Key, request.CreateReportingProviderAsync, cancellationToken)
                .ConfigureAwait(false);
        }
    }
    private SessionLeaseBundle? TryAcquireBundle(
        SessionLeaseRequest[] ordered,
        IReadOnlyList<SessionLeaseRequest> requests)
    {
        var held = new List<(SessionPoolKey Key, PoolEntry Entry)>(ordered.Length);
        bool allAcquired = true;
        foreach (SessionLeaseRequest request in ordered)
        {
            if (!entries.TryGetValue(request.Key, out PoolEntry? entry) || !TryAcquireGate(entry))
            {
                allAcquired = false;
                break;
            }

            if (entry.IsEvicted)
            {
                entry.ReleaseGateWithoutTouchingLru();
                allAcquired = false;
                break;
            }

            held.Add((request.Key, entry));
        }

        if (allAcquired && held.Count == ordered.Length)
        {
            return BuildBundleInRequestOrder(ordered, requests, held);
        }

        foreach ((SessionPoolKey _, PoolEntry entry) in held)
        {
            entry.ReleaseGateWithoutTouchingLru();
        }

        return null;
    }
    private SessionLeaseBundle BuildBundleInRequestOrder(
        SessionLeaseRequest[] ordered,
        IReadOnlyList<SessionLeaseRequest> requests,
        List<(SessionPoolKey Key, PoolEntry Entry)> held)
    {
        var leases = new SessionLease[ordered.Length];
        for (int i = 0; i < ordered.Length; i++)
        {
            SessionPoolKey key = ordered[i].Key;
            int requestIndex = -1;
            for (int r = 0; r < requests.Count; r++)
            {
                if (Equals(requests[r].Key, key))
                {
                    requestIndex = r;
                    break;
                }
            }

            PoolEntry entry = held.First(h => Equals(h.Key, key)).Entry;
            leases[requestIndex] = BuildLease(entry);
        }

        return new SessionLeaseBundle(leases);
    }
    private static SessionLeaseRequest[] ValidateBundleRequests(IReadOnlyList<SessionLeaseRequest> requests)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0)
        {
            throw new ArgumentException("Bundle requires at least one session.", nameof(requests));
        }

        SessionLeaseRequest[] ordered = requests.OrderBy(r => r.Key, SessionPoolKey.StableComparer).ToArray();
        for (int i = 1; i < ordered.Length; i++)
        {
            if (Equals(ordered[i - 1].Key, ordered[i].Key))
            {
                throw new ArgumentException(
                    "Bundle contains duplicate session keys; multi-graph bundles require distinct graphs.",
                    nameof(requests));
            }
        }

        return ordered;
    }
}
