using System.Collections.Concurrent;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.TensorRtRtx;

namespace Trackdub.Inference.Onnx.Pool;

// Eviction: LRU idle eviction, per-model eviction, and trimming to the VRAM budget.
internal sealed partial class InferenceSessionPool
{
    /// <summary>
    /// Idle sessions released within this window are treated as part of the active pipeline
    /// working set and are only evicted when no older idle entry exists.
    /// </summary>
    private const long RecentReleaseWindowMs = 120_000;
    /// <summary>
    /// Evicts every idle pooled session. Sessions currently leased out remain until released.
    /// </summary>
    /// <returns>The number of idle entries that were evicted and disposed.</returns>
    public async Task<int> EvictAllIdleAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        List<PoolEntry> toDispose = new();
        int count = 0;

        await creationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            foreach (SessionPoolKey key in entries.Keys.ToArray())
            {
                cancellationToken.ThrowIfCancellationRequested();

                if (TryRemovePooledEntry(key, toDispose))
                {
                    count++;
                }
            }
        }
        finally
        {
            creationLock.Release();
        }

        DisposeEvicted(toDispose);

        return count;
    }
    /// <summary>
    /// Evicts all idle pool entries whose <see cref="SessionPoolKey.EngineFamily"/> matches
    /// <paramref name="engineFamily"/> and, when specified, whose
    /// <see cref="SessionPoolKey.ModelId"/> matches <paramref name="modelId"/>.
    /// In-use sessions (leased out) are not evicted.
    /// </summary>
    /// <returns>The number of entries that were evicted and disposed.</returns>
    public async Task<int> EvictModelAsync(
        string engineFamily,
        string? modelId = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(engineFamily);
        ObjectDisposedException.ThrowIf(disposed, this);

        List<PoolEntry> toDispose = new();
        int count = 0;

        await creationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            foreach (SessionPoolKey key in entries.Keys.ToArray())
            {
                if (!MatchesModel(key, engineFamily, modelId))
                {
                    continue;
                }

                if (TryRemovePooledEntry(key, toDispose))
                {
                    count++;
                }
            }
        }
        finally
        {
            creationLock.Release();
        }

        // Dispose sessions outside the lock — session disposal can be expensive and
        // should not block other callers waiting on creationLock.
        DisposeEvicted(toDispose);

        return count;
    }

    /// <summary>
    /// Removes one entry by key after acquiring its gate, marks it evicted, and queues it
    /// for disposal outside <see cref="creationLock"/>. Returns <see langword="false"/> when
    /// the entry is gone, leased, or lost a concurrent-removal race (its gate is released
    /// in that case).
    /// </summary>
    private bool TryRemovePooledEntry(SessionPoolKey key, List<PoolEntry> toDispose)
    {
        if (!entries.TryGetValue(key, out PoolEntry? entry))
        {
            return false;
        }

        return TryRemoveGatedEntry(key, entry, toDispose);
    }

    /// <summary>
    /// Gate-acquire + remove + mark for a known entry. The gate is released when removal
    /// fails because a concurrent eviction won the race.
    /// </summary>
    private bool TryRemoveGatedEntry(SessionPoolKey key, PoolEntry entry, List<PoolEntry> toDispose)
    {
        if (!TryAcquireGate(entry))
        {
            return false;
        }

        if (!entries.TryRemove(new KeyValuePair<SessionPoolKey, PoolEntry>(key, entry)))
        {
            entry.Gate.Release(); // entry removed by a concurrent eviction; release gate
            return false;
        }

        pooledCount--;
        SignalBundleStateChanged();
        entry.MarkEvicted();
        toDispose.Add(entry); // dispose outside the lock below
        return true;
    }

    private static bool MatchesModel(SessionPoolKey key, string engineFamily, string? modelId)
    {
        if (!string.Equals(key.EngineFamily, engineFamily, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        return modelId is null || string.Equals(key.ModelId, modelId, StringComparison.OrdinalIgnoreCase);
    }
    /// <summary>
    /// Evicts idle pool entries until the estimated total VRAM footprint of pooled sessions
    /// is at or below <paramref name="targetVramMb"/>. Sessions currently leased out are not
    /// evicted. Returns the number of sessions evicted.
    /// </summary>
    /// <remarks>
    /// Eviction order: least-recently-released idle sessions first (same LRU order as the
    /// count-based eviction in <see cref="GetLeaseAsync"/>).
    /// </remarks>
    public async Task<int> TrimToVramBudgetAsync(long targetVramMb, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentOutOfRangeException.ThrowIfNegative(targetVramMb);

        List<PoolEntry> toDispose = new();
        int count = 0;

        await creationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            // Compute current total estimated VRAM of idle sessions
            long currentVram = entries.Sum(kvp => kvp.Value.IsIdle ? kvp.Key.EstimatedVramMb : 0);
            if (currentVram <= targetVramMb)
                return 0;

            foreach (var (key, entry) in IdleEntriesOldestFirst())
            {
                if (currentVram <= targetVramMb)
                {
                    break;
                }

                cancellationToken.ThrowIfCancellationRequested();

                if (!TryRemoveGatedEntry(key, entry, toDispose))
                {
                    continue;
                }

                currentVram -= key.EstimatedVramMb;
                count++;
            }
        }
        finally
        {
            creationLock.Release();
        }

        DisposeEvicted(toDispose);

        return count;
    }

    /// <summary>Idle entries sorted least-recently-released first (LRU eviction order).</summary>
    private List<KeyValuePair<SessionPoolKey, PoolEntry>> IdleEntriesOldestFirst() =>
        entries
            .Where(kvp => kvp.Value.IsIdle)
            .OrderBy(kvp => kvp.Value.LastReleasedTicks)
            .ToList();

    /// <summary>
    /// Disposes evicted entries outside <see cref="creationLock"/>: session teardown can be
    /// expensive and must not block other callers waiting on the lock.
    /// </summary>
    private static void DisposeEvicted(List<PoolEntry> toDispose)
    {
        foreach (PoolEntry entry in toDispose)
        {
            entry.Dispose();
        }
    }
    /// <summary>
    /// Evicts the least-recently-released idle entry.
    /// Must be called while <see cref="creationLock"/> is held.
    /// </summary>
    /// <returns>
    /// The evicted <see cref="PoolEntry"/> (gate already acquired, removed from the pool, marked evicted)
    /// when an idle entry was found and successfully evicted; <see langword="null"/> when all entries are
    /// currently leased.  The caller is responsible for disposing the returned entry <em>outside</em>
    /// <see cref="creationLock"/> to avoid blocking unrelated pool operations during potentially expensive
    /// <see cref="InferenceSession"/> teardown.
    /// </returns>
    /// <summary>
    /// Evicts the least-recently-released idle entry, optionally restricted to one admission
    /// bucket (so a host RAM shortfall is not “fixed” by dropping GPU sessions, and a
    /// shortfall on accelerator device 0 is not “fixed” by dropping device 1 sessions).
    /// Must be called while <see cref="creationLock"/> is held.
    /// </summary>
    private PoolEntry? TryEvictLruIdle(AdmissionBucket? onlyBucket = null)
    {
        // Prefer evicting entries that have been idle for a while. Sessions released within
        // the recent window are treated as the active pipeline working set (e.g. the next
        // stage's graphs) and are only chosen when nothing older is idle.
        long now = Environment.TickCount64;
        long recentCutoff = now - RecentReleaseWindowMs;

        (SessionPoolKey? candidateKey, PoolEntry? candidateEntry) = FindOldestIdle(onlyBucket, recentCutoff);
        if (candidateEntry is null)
        {
            (candidateKey, candidateEntry) = FindOldestIdle(onlyBucket, recentCutoff: null);
        }

        if (candidateEntry is null)
        {
            return null;
        }

        if (!entries.TryGetValue(candidateKey!, out PoolEntry? evicted) || !TryAcquireGate(evicted))
        {
            return null;
        }

        bool removed = false;
        try
        {
            removed = entries.TryRemove(new KeyValuePair<SessionPoolKey, PoolEntry>(candidateKey!, evicted));
            if (removed)
            {
                pooledCount--;
                SignalBundleStateChanged();
                evicted.MarkEvicted();
                return evicted; // Caller must dispose outside creationLock.
            }
        }
        finally
        {
            if (!removed)
            {
                evicted.ReleaseGateWithoutTouchingLru();
            }
        }

        return null;
    }
    private (SessionPoolKey? Key, PoolEntry? Entry) FindOldestIdle(AdmissionBucket? onlyBucket, long? recentCutoff)
    {
        SessionPoolKey? candidateKey = null;
        PoolEntry? candidateEntry = null;
        long candidateLastReleasedTicks = long.MaxValue;
        foreach (KeyValuePair<SessionPoolKey, PoolEntry> pair in entries)
        {
            if (!IsIdleCandidate(pair.Value, onlyBucket, recentCutoff))
            {
                continue;
            }

            long lastReleasedTicks = pair.Value.LastReleasedTicks;
            // candidateLastReleasedTicks starts at MaxValue while no candidate exists, so
            // the first idle entry always wins without a separate null check.
            if (lastReleasedTicks < candidateLastReleasedTicks)
            {
                candidateKey = pair.Key;
                candidateEntry = pair.Value;
                candidateLastReleasedTicks = lastReleasedTicks;
            }
        }

        return (candidateKey, candidateEntry);
    }

    private static bool IsIdleCandidate(PoolEntry entry, AdmissionBucket? onlyBucket, long? recentCutoff)
    {
        if (onlyBucket is not null && entry.Bucket != onlyBucket.Value)
        {
            return false;
        }

        if (!entry.IsIdle || entry.PinCount > 0)
        {
            return false;
        }

        return recentCutoff is null || entry.LastReleasedTicks < recentCutoff.Value;
    }
    /// <summary>
    /// Snapshot of live external reservations in <paramref name="bucket"/> that registered an
    /// idle-eviction callback, ordered least-recently-released first (Guid tie-break for
    /// determinism). Reservations still under construction have no callback and are never
    /// candidates — pending accounting already covers them.
    /// </summary>
    private List<KeyValuePair<Guid, ExternalReservationState>> OrderedIdleExternals(AdmissionBucket bucket)
    {
        List<KeyValuePair<Guid, ExternalReservationState>> candidates = externalReservations
            .Where(pair => pair.Value.Bucket == bucket && pair.Value.TryEvictIdle is not null)
            .ToList();

        candidates.Sort(static (x, y) =>
        {
            int c = x.Value.LastReleasedTicks.CompareTo(y.Value.LastReleasedTicks);
            return c != 0 ? c : x.Key.CompareTo(y.Key);
        });
        return candidates;
    }
}
