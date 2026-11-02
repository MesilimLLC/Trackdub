using System.Collections.Concurrent;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.TensorRtRtx;

namespace Trackdub.Inference.Onnx.Pool;

/// <summary>
/// Parameters for one external memory reservation: the provider's admission bucket,
/// device, and estimated footprint. CPU/DNNL/OpenVINO share the host RAM budget when
/// <paramref name="UseOpenVinoCpuProxy"/> is set; otherwise OpenVINO and every other
/// accelerator provider use their device budget.
/// </summary>
internal readonly record struct ExternalReservationRequest(
    ExecutionProviderKind Provider,
    int? DeviceId,
    long EstimatedMemoryMb,
    bool UseOpenVinoCpuProxy = false);

// External memory reservations: accounting for memory owned outside the pool
// (e.g. GenAI model loads) so admission sees real residency.
internal sealed partial class InferenceSessionPool
{
    /// <summary>Live external-residency reservations (e.g. GenAI model loads) per bucket.</summary>
    private readonly ConcurrentDictionary<Guid, ExternalReservationState> externalReservations = new();
    /// <summary>
    /// Live state for one external memory reservation. The pool owns accounting; the
    /// <see cref="ExternalMemoryReservation"/> token is only a handle into this entry.
    /// <see cref="TryEvictIdle"/> is registered by the owner of the external resource and
    /// returns <see langword="true"/> only when it actually released the idle resource.
    /// </summary>
    private sealed class ExternalReservationState(
        AdmissionBucket bucket,
        long estimatedMemoryMb)
    {
        public AdmissionBucket Bucket { get; } = bucket;
        public long EstimatedMemoryMb { get; } = estimatedMemoryMb;
        // volatile: the callback is registered after the state is already visible in the
        // dictionary, so readers must see the write without a lock.
        public volatile Func<bool>? TryEvictIdle;
        public long LastReleasedTicks { get; set; } = Environment.TickCount64;
    }
    /// <summary>
    /// Reserves <paramref name="request"/>'s estimated memory against its admission bucket
    /// and returns a token whose disposal releases the accounting. Used by external
    /// residency owners (e.g. ORT GenAI model caches) so model resources share the same
    /// host RAM / per-device accelerator ceiling as pooled ONNX sessions instead of an
    /// independent cache.
    /// </summary>
    /// <remarks>
    /// External reservations always use hard admission — they cannot opt into ephemeral
    /// overflow — and one token occupies exactly one bucket: CPU/DNNL/OpenVINO share the
    /// host RAM budget when OpenVINO CPU-proxy mode is active; otherwise OpenVINO and every
    /// other accelerator provider use their device budget.
    /// </remarks>
    internal async Task<ExternalMemoryReservation> ReserveExternalAsync(
        ExternalReservationRequest request,
        CancellationToken cancellationToken) =>
        (await ReserveExternalCoreAsync(request, cancellationToken, allowEviction: true).ConfigureAwait(false))!;
    /// <summary>
    /// Non-evicting variant of <see cref="ReserveExternalAsync"/> for optimization-only
    /// warmup: returns <see langword="null"/> immediately when the reservation does not
    /// already fit the bucket budget, instead of evicting idle pooled/external entries
    /// or waiting for headroom. A request larger than the whole budget also returns
    /// <see langword="null"/> (the real acquire keeps the authoritative failure).
    /// </summary>
    internal Task<ExternalMemoryReservation?> TryReserveExternalWithoutEvictionAsync(
        ExternalReservationRequest request,
        CancellationToken cancellationToken) =>
        ReserveExternalCoreAsync(request, cancellationToken, allowEviction: false);
    private async Task<ExternalMemoryReservation?> ReserveExternalCoreAsync(
        ExternalReservationRequest request,
        CancellationToken cancellationToken,
        bool allowEviction)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(request.EstimatedMemoryMb, 1);
        ObjectDisposedException.ThrowIf(disposed, this);

        AdmissionBucket bucket = BucketOf(request.Provider, request.DeviceId, request.UseOpenVinoCpuProxy);
        long budgetMb = BudgetFor(bucket);
        if (request.EstimatedMemoryMb > budgetMb)
        {
            if (!allowEviction)
            {
                return null;
            }

            throw new InvalidOperationException(
                $"External resource for '{request.Provider}' needs ~{request.EstimatedMemoryMb} MB, which exceeds the " +
                $"{DescribeBucket(bucket)} admission budget of {budgetMb} MB.");
        }

        if (allowEviction)
        {
            // Takes the pending reservation (evicting idle pooled/external entries to fit) so
            // benchmark maxima see external construction honestly; publishing the live state
            // below then transfers the accounting so nothing is double-counted.
            await WaitForAdmissionBudgetAsync(request.EstimatedMemoryMb, bucket, cancellationToken).ConfigureAwait(false);
        }
        else if (!await TryTakePendingReservationWithoutEvictionAsync(
                     request, bucket, cancellationToken).ConfigureAwait(false))
        {
            return null;
        }

        // Publish under creationLock so pool disposal cannot interleave: either the reserve
        // wins the lock and publishes first (Dispose then sees and clears the state), or
        // Dispose wins and the publish throws ObjectDisposedException before accounting
        // for a resource the pool can no longer track.
        Guid id = Guid.NewGuid();
        bool published = false;
        bool pendingReleased = false;
        try
        {
            await creationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                externalReservations[id] = new ExternalReservationState(bucket, request.EstimatedMemoryMb);
                // Transfer pending -> live while holding creationLock so admission readers
                // cannot observe the same reservation in both accounting buckets.
                ReleaseReservation(bucket, request.EstimatedMemoryMb);
                pendingReleased = true;
                published = true;
            }
            finally
            {
                creationLock.Release();
            }

            SignalBundleStateChanged();
            return new ExternalMemoryReservation(this, id);
        }
        finally
        {
            // Release pending accounting on cancellation, disposal, or publish failure.
            if (!pendingReleased)
            {
                ReleaseReservation(bucket, request.EstimatedMemoryMb);
            }
            if (!published)
            {
                externalReservations.TryRemove(id, out _);
            }
        }
    }
    /// <summary>
    /// Takes the pending reservation only when it already fits <paramref name="bucket"/>'s
    /// budget under the same usage accounting as <see cref="WaitForAdmissionBudgetAsync"/>
    /// (reservations floored by the observed process GPU footprint) — never evicting and
    /// never waiting. Returns <see langword="false"/> when it would not fit as-is.
    /// </summary>
    private async Task<bool> TryTakePendingReservationWithoutEvictionAsync(
        ExternalReservationRequest request,
        AdmissionBucket bucket,
        CancellationToken cancellationToken)
    {
        ProcessGpuObservation? observation = bucket.IsHost ? null : ReadObservedProcessGpu();
        await creationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (AdmissionUsageMb(bucket, observation) + request.EstimatedMemoryMb > BudgetFor(bucket))
            {
                return false;
            }

            AddPendingReservation(bucket, request.EstimatedMemoryMb);
            return true;
        }
        finally
        {
            creationLock.Release();
        }
    }
    /// <summary>
    /// Attaches the idle-eviction callback to a still-live reservation. Returns
    /// <see langword="false"/> when the pool is disposed or the id is already gone — callers
    /// must treat false as "the reservation no longer exists" and roll back rather than
    /// publish an unaccounted resource.
    /// </summary>
    internal bool TrySetExternalEvictionCallback(Guid id, Func<bool> callback)
    {
        creationLock.Wait();
        try
        {
            if (disposed
                || !externalReservations.TryGetValue(id, out ExternalReservationState? state))
            {
                return false;
            }

            state.TryEvictIdle = callback;
            return true;
        }
        finally
        {
            creationLock.Release();
        }
    }
    internal void NotifyExternalReservationReleased(Guid id)
    {
        if (externalReservations.TryGetValue(id, out ExternalReservationState? state))
        {
            state.LastReleasedTicks = Environment.TickCount64;
            SignalBundleStateChanged();
        }
    }
    internal void ReleaseExternalReservation(Guid id)
    {
        if (externalReservations.TryRemove(id, out _))
        {
            SignalBundleStateChanged();
        }
    }
}
