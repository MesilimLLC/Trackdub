using Microsoft.ML.OnnxRuntime;
using Trackdub.Domain;

namespace Trackdub.Inference.Onnx.Pool;

/// <summary>One graph in a multi-graph bundle acquire.</summary>
/// <param name="ResolvePlacedProvider">
/// Where ORT actually placed the created session, when it can differ from
/// <see cref="SessionPoolKey.Provider"/>. Reported per graph at creation so admission and
/// TensorRT-RTX teardown tracking are right even if a later graph in the bundle fails.
/// </param>
internal sealed record SessionLeaseRequest(
    SessionPoolKey Key,
    Func<CancellationToken, Task<InferenceSession>> Factory,
    Func<InferenceSession, ExecutionProviderKind>? ResolvePlacedProvider = null)
{
    internal async Task<CreatedPoolSession> CreateReportingProviderAsync(CancellationToken cancellationToken)
    {
        InferenceSession session = await Factory(cancellationToken).ConfigureAwait(false);
        return new CreatedPoolSession(session, ResolvePlacedProvider?.Invoke(session) ?? Key.Provider);
    }
}

/// <summary>
/// A session created for a pool miss and the provider it was created with. The provider differs
/// from <see cref="SessionPoolKey.Provider"/> when the factory's provider fallback ran after the
/// key was admitted (for example TensorRT RTX init failing over to CPU).
/// </summary>
internal sealed record CreatedPoolSession(InferenceSession Session, ExecutionProviderKind Provider);

/// <summary>
/// Scoped, exclusive handle to a pooled <see cref="InferenceSession"/>.
/// Disposing this lease releases the session back to the <see cref="InferenceSessionPool"/>
/// so another caller can acquire it. Never dispose the underlying <see cref="SessionLease.Session"/>
/// directly — always dispose the lease instead.
/// </summary>
internal sealed class SessionLease : IDisposable
{
    private readonly Action release;
    private int disposed;

    internal SessionLease(InferenceSession session, Action release)
    {
        Session = session;
        this.release = release;
    }

    /// <summary>The pooled session.  Valid only while the lease is held (before <see cref="Dispose"/>).</summary>
    public InferenceSession Session { get; }

    /// <summary>Releases the session back to the pool.  Safe to call more than once.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) == 0)
        {
            release();
        }
    }
}

/// <summary>
/// Keeps a pooled session <em>resident</em> (protected from idle eviction) without holding
/// the exclusive execution gate. Audit §3A: residency ≠ execution exclusivity — an idle
/// reusable model must not require a permanently held <see cref="SessionLease"/>.
/// Take a <see cref="SessionLease"/> only for the duration of a Run() call.
/// </summary>
internal sealed class SessionResidency : IDisposable
{
    private readonly Action unpin;
    private int disposed;

    internal SessionResidency(Action unpin)
    {
        this.unpin = unpin;
    }

    /// <summary>Drops the residency pin. The session stays in the pool until evicted.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) == 0)
        {
            unpin();
        }
    }
}

/// <summary>
/// All-or-nothing set of exclusive session leases (multi-graph models). Disposing the
/// bundle releases every lease. Never dispose individual leases from a bundle.
/// </summary>
internal sealed class SessionLeaseBundle : IDisposable
{
    private readonly SessionLease[] leases;
    private int disposed;

    internal SessionLeaseBundle(IReadOnlyList<SessionLease> leases)
    {
        this.leases = leases.ToArray();
    }

    public int Count => leases.Length;

    public InferenceSession this[int index] => leases[index].Session;

    public SessionLease LeaseAt(int index) => leases[index];

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0)
        {
            return;
        }

        for (int i = leases.Length - 1; i >= 0; i--)
        {
            leases[i].Dispose();
        }
    }
}

/// <summary>
/// Token returned by <see cref="InferenceSessionPool.ReserveExternalAsync"/> for memory
/// held outside an ONNX <see cref="InferenceSession"/> (e.g. a cached GenAI model object).
/// The pool owns the admission accounting; the token is an idempotent handle whose
/// <see cref="Dispose"/> releases the reservation and wakes admission waiters.
/// </summary>
internal sealed class ExternalMemoryReservation : IDisposable
{
    private readonly InferenceSessionPool owner;
    private readonly Guid id;
    private int disposed;

    internal ExternalMemoryReservation(InferenceSessionPool owner, Guid id)
    {
        this.owner = owner;
        this.id = id;
    }

    /// <summary>
    /// Registers the callback the pool may invoke to evict this reservation when its host
    /// RAM / accelerator bucket is under pressure. The callback must return
    /// <see langword="true"/> only when it actually released the idle resource (and should
    /// dispose the token). A busy resource returns <see langword="false"/> and stays
    /// resident. Returns <see langword="false"/> once the token is disposed or the owning
    /// pool no longer tracks the reservation — callers must treat false as "registration
    /// failed; the reservation is gone".
    /// </summary>
    internal async Task<bool> TrySetIdleEvictionCallbackAsync(
        Func<bool> callback,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(callback);
        if (Volatile.Read(ref disposed) != 0)
        {
            return false;
        }

        return await owner.TrySetExternalEvictionCallbackAsync(id, callback, cancellationToken)
            .ConfigureAwait(false);
    }

    /// <summary>
    /// Records that the external resource went idle (for LRU eviction ordering) and wakes
    /// admission waiters. No-op once the token is disposed.
    /// </summary>
    internal void MarkReleased()
    {
        if (Volatile.Read(ref disposed) != 0)
        {
            return;
        }

        owner.NotifyExternalReservationReleased(id);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) != 0)
        {
            return;
        }

        owner.ReleaseExternalReservation(id);
    }
}
