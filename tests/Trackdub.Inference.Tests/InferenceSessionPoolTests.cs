using Trackdub.Contracts;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.TestDoubles;
using Microsoft.ML.OnnxRuntime;

namespace Trackdub.Inference.Tests;

/// <summary>
/// Unit tests for <see cref="InferenceSessionPool"/>.
/// Tests that require creating real <see cref="InferenceSession"/> objects are guarded by
/// <see cref="RequiresBundledModelFactAttribute"/> or use a fake session factory.
/// </summary>
public sealed class InferenceSessionPoolTests
{
    private static readonly SessionPoolKey TestKey = new(
        "test-engine", "test-model", null, ExecutionProviderKind.Cpu, "abc123", 0, "default");

    // ── Shared pool ───────────────────────────────────────────────────────────

    [Fact]
    public void Shared_IsSingleton()
    {
        Assert.Same(InferenceSessionPool.Shared, InferenceSessionPool.Shared);
    }

    // ── Constructor guard ─────────────────────────────────────────────────────

    [Fact]
    public void Constructor_ZeroMaxSessions_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new InferenceSessionPool(0));
    }

    [Fact]
    public void Constructor_NegativeMaxSessions_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new InferenceSessionPool(-1));
    }

    [Fact]
    public void Constructor_OneMaxSession_DoesNotThrow()
    {
        using var pool = new InferenceSessionPool(1);
        Assert.NotNull(pool);
    }

    // ── Dispose guard ─────────────────────────────────────────────────────────

    [Fact]
    public async Task GetLeaseAsync_AfterDispose_Throws()
    {
        var pool = new InferenceSessionPool(4);
        pool.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(
            () => pool.GetLeaseAsync(TestKey, _ => Task.FromException<InferenceSession>(new Exception("should not reach")), CancellationToken.None));
    }

    [Fact]
    public async Task WarmAsync_AfterDispose_Throws()
    {
        var pool = new InferenceSessionPool(4);
        pool.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(
            () => pool.WarmAsync(TestKey, _ => Task.FromException<InferenceSession>(new Exception("should not reach")), CancellationToken.None));
    }

    [Fact]
    public async Task EvictModelAsync_AfterDispose_Throws()
    {
        var pool = new InferenceSessionPool(4);
        pool.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(
            () => pool.EvictModelAsync("test-engine"));
    }

    // ── Dispose idempotency ───────────────────────────────────────────────────

    [Fact]
    public void Dispose_CalledTwice_DoesNotThrow()
    {
        var pool = new InferenceSessionPool(4);
        pool.Dispose();
        pool.Dispose(); // must not throw
    }

    // ── SessionLease ──────────────────────────────────────────────────────────

    [Fact]
    public void SessionLease_Dispose_CalledTwice_DoesNotThrow()
    {
        // Ensure the double-dispose guard in SessionLease works.
        int releaseCount = 0;

        var session = CreateMinimalSession();
        try
        {
            var lease = new SessionLease(session, () => releaseCount++);
            lease.Dispose();
            lease.Dispose(); // must not throw or call release twice

            Assert.Equal(1, releaseCount);
        }
        finally
        {
            session.Dispose();
        }
    }

    [Fact]
    public void SessionLease_Dispose_InvokesRelease()
    {
        bool released = false;
        var session = CreateMinimalSession();
        try
        {
            using (new SessionLease(session, () => released = true))
            {
                Assert.False(released);
            }

            Assert.True(released);
        }
        finally
        {
            session.Dispose();
        }
    }

    // ── EvictModelAsync ───────────────────────────────────────────────────────

    [Fact]
    public async Task EvictModelAsync_NoMatchingEntries_ReturnsZero()
    {
        using var pool = new InferenceSessionPool(4);

        int evicted = await pool.EvictModelAsync("nonexistent-engine");

        Assert.Equal(0, evicted);
    }

    [Fact]
    public async Task EvictModelAsync_DoesNotEvictLeasedEntry()
    {
        using var pool = new InferenceSessionPool(2);
        var key = new SessionPoolKey("eng", "model", null, ExecutionProviderKind.Cpu, "hash1", 0, "default");

        using (await pool.GetLeaseAsync(key, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None))
        {
            int evicted = await pool.EvictModelAsync("eng", "model");

            Assert.Equal(0, evicted);
        }

        int factoryCalls = 0;
        using (await pool.GetLeaseAsync(
            key,
            _ => { factoryCalls++; return Task.FromResult(CreateMinimalSession()); },
            CancellationToken.None))
        {
        }

        Assert.Equal(0, factoryCalls);
    }

    [Fact]
    public async Task GetLeaseAsync_WaitingFastPathThrows_WhenPoolDisposed()
    {
        var pool = new InferenceSessionPool(1);
        var key = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "hash1", 0, "default");

        using SessionLease activeLease = await pool.GetLeaseAsync(
            key,
            _ => Task.FromResult(CreateMinimalSession()),
            CancellationToken.None);

        Task<SessionLease> waitingLease = pool.GetLeaseAsync(
            key,
            _ => Task.FromResult(CreateMinimalSession()),
            CancellationToken.None);

        pool.Dispose();
        activeLease.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(() => waitingLease);
    }

    // ── Single-flight creation ───────────────────────────────────────────────

    [Fact]
    public async Task ConcurrentColdMisses_ConstructOneSession()
    {
        using var pool = new InferenceSessionPool(4);
        var releaseFactory = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        int factoryCalls = 0;
        Func<CancellationToken, Task<InferenceSession>> factory = async _ =>
        {
            Interlocked.Increment(ref factoryCalls);
            await releaseFactory.Task;
            return CreateMinimalSession();
        };

        Task<SessionLease> creator = pool.GetLeaseAsync(TestKey, factory, CancellationToken.None);
        Task<SessionLease> waiter1 = pool.GetLeaseAsync(TestKey, factory, CancellationToken.None);
        Task<SessionLease> waiter2 = pool.GetLeaseAsync(TestKey, factory, CancellationToken.None);
        releaseFactory.SetResult();

        // Leases on one key are exclusive, so take them in whatever order they are granted.
        var remaining = new List<Task<SessionLease>> { creator, waiter1, waiter2 };
        InferenceSession? session = null;
        while (remaining.Count > 0)
        {
            Task<SessionLease> granted = await Task.WhenAny(remaining);
            remaining.Remove(granted);
            using SessionLease lease = await granted;
            session ??= lease.Session;
            Assert.Same(session, lease.Session);
        }

        Assert.Equal(1, factoryCalls);
    }

    [Fact]
    public async Task CancellingTheCreator_LetsWaiterCreateInstead()
    {
        using var pool = new InferenceSessionPool(4);
        var creatorStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        int factoryCalls = 0;
        using var creatorCts = new CancellationTokenSource();

        Task<SessionLease> creator = pool.GetLeaseAsync(
            TestKey,
            async ct =>
            {
                Interlocked.Increment(ref factoryCalls);
                creatorStarted.SetResult();
                await Task.Delay(Timeout.Infinite, ct);
                return CreateMinimalSession();
            },
            creatorCts.Token);
        await creatorStarted.Task;

        Task<SessionLease> waiter = pool.GetLeaseAsync(
            TestKey,
            _ => { Interlocked.Increment(ref factoryCalls); return Task.FromResult(CreateMinimalSession()); },
            CancellationToken.None);
        creatorCts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => creator);
        using SessionLease lease = await waiter;
        Assert.NotNull(lease.Session);
        Assert.Equal(2, factoryCalls);
    }

    [Fact]
    public async Task CancellingAWaiter_DoesNotCancelTheCreator()
    {
        using var pool = new InferenceSessionPool(4);
        var releaseFactory = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var waiterCts = new CancellationTokenSource();

        Task<SessionLease> creator = pool.GetLeaseAsync(
            TestKey,
            async _ => { await releaseFactory.Task; return CreateMinimalSession(); },
            CancellationToken.None);
        Task<SessionLease> waiter = pool.GetLeaseAsync(
            TestKey,
            _ => Task.FromException<InferenceSession>(new InvalidOperationException("waiter must not build")),
            waiterCts.Token);

        waiterCts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waiter);
        releaseFactory.SetResult();
        using SessionLease lease = await creator;
        Assert.NotNull(lease.Session);
    }

    [Fact]
    public async Task CreatorFailure_PropagatesToWaiters()
    {
        using var pool = new InferenceSessionPool(4);
        var releaseFactory = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        int factoryCalls = 0;
        Func<CancellationToken, Task<InferenceSession>> factory = async _ =>
        {
            Interlocked.Increment(ref factoryCalls);
            await releaseFactory.Task;
            throw new InvalidOperationException("model load failed");
        };

        Task<SessionLease> creator = pool.GetLeaseAsync(TestKey, factory, CancellationToken.None);
        Task<SessionLease> waiter = pool.GetLeaseAsync(TestKey, factory, CancellationToken.None);
        releaseFactory.SetResult();

        await Assert.ThrowsAsync<InvalidOperationException>(() => creator);
        await Assert.ThrowsAsync<InvalidOperationException>(() => waiter);
        Assert.Equal(1, factoryCalls);
    }

    // ── LRU eviction & ephemeral fallback ────────────────────────────────────

    [Fact]
    public async Task LruEviction_EjectsOldestIdleEntry_WhenCapacityExceeded()
    {
        // Count-mode overflow behaviour only exists with admission explicitly disabled.
        using var pool = new InferenceSessionPool(maxSessions: 2, enableMemoryAdmission: false);

        var key1 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "hash1", 0, "default");
        var key2 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "hash2", 0, "default");
        var key3 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "hash3", 0, "default");

        // Warm key1, then key2; key1 will have an older LastReleasedTicks and be the LRU candidate.
        // Spin until TickCount64 advances to guarantee key2's release stamp is strictly greater
        // than key1's, regardless of the platform's timer resolution.
        using (await pool.GetLeaseAsync(key1, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None)) { }
        // Spin until TickCount64 advances to guarantee key2's release stamp is strictly greater
        // than key1's, regardless of the platform's timer resolution.
        // Bounded to 5 s to prevent indefinite hangs in constrained/paused environments.
        var sw = System.Diagnostics.Stopwatch.StartNew();
        long t0 = Environment.TickCount64;
        while (Environment.TickCount64 == t0)
        {
            if (sw.Elapsed > TimeSpan.FromSeconds(5))
            {
                throw new TimeoutException("Timer did not advance within 5 seconds.");
            }
            await Task.Delay(1);
        }
        using (await pool.GetLeaseAsync(key2, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None)) { }

        // Pool at capacity (2: key1, key2). Request key3 → should evict key1 (LRU).
        using (await pool.GetLeaseAsync(key3, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None)) { }

        // key1 was evicted; requesting it must invoke the factory again.
        int key1FactoryCalls = 0;
        using (await pool.GetLeaseAsync(
            key1,
            _ => { key1FactoryCalls++; return Task.FromResult(CreateMinimalSession()); },
            CancellationToken.None)) { }

        Assert.Equal(1, key1FactoryCalls);
    }

    [Fact]
    public async Task ColdMiss_WithSpareCapacity_DoesNotEvictExistingIdleEntry()
    {
        using var pool = new InferenceSessionPool(maxSessions: 8);

        var key1 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "hash1", 0, "default");
        var key2 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "hash2", 0, "default");

        using (await pool.GetLeaseAsync(key1, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None)) { }
        using (await pool.GetLeaseAsync(key2, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None)) { }

        int key1FactoryCalls = 0;
        using (await pool.GetLeaseAsync(
            key1,
            _ => { key1FactoryCalls++; return Task.FromResult(CreateMinimalSession()); },
            CancellationToken.None))
        {
        }

        Assert.Equal(0, key1FactoryCalls);
    }

    [Fact]
    public async Task ReacquireAtCapacity_DoesNotCallFactory_EntryReused()
    {
        // Pool at capacity (maxSessions == 2, two idle entries). Re-acquiring an
        // existing key must be a fast-path hit: factory is never called and no
        // eviction occurs, even though pooledCount == maxSessions.
        using var pool = new InferenceSessionPool(maxSessions: 2);

        var key1 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "hash1", 0, "default");
        var key2 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "hash2", 0, "default");

        using (await pool.GetLeaseAsync(key1, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None)) { }
        using (await pool.GetLeaseAsync(key2, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None)) { }

        int key1FactoryCalls = 0;
        using (await pool.GetLeaseAsync(
            key1,
            _ => { key1FactoryCalls++; return Task.FromResult(CreateMinimalSession()); },
            CancellationToken.None))
        {
        }

        Assert.Equal(0, key1FactoryCalls);
    }

    [Fact]
    public async Task LruEviction_DoesNotEvictLeasedEntry_FallsBackToEphemeral()
    {
        // Count-mode ephemeral overflow exists only with admission explicitly disabled.
        using var pool = new InferenceSessionPool(maxSessions: 1, enableMemoryAdmission: false);

        var key1 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "hash1", 0, "default");
        var key2 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "hash2", 0, "default");

        // Acquire key1 and hold the lease (not released) so it cannot be evicted.
        using var lease1 = await pool.GetLeaseAsync(key1, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        // Pool full (1) and the only entry is leased → key2 must be created as ephemeral.
        int key2Calls = 0;
        using (await pool.GetLeaseAsync(
            key2,
            _ => { key2Calls++; return Task.FromResult(CreateMinimalSession()); },
            CancellationToken.None)) { }

        Assert.Equal(1, key2Calls); // created once (ephemeral — not cached)

        // Because key2 was ephemeral, a second request must call the factory again.
        using (await pool.GetLeaseAsync(
            key2,
            _ => { key2Calls++; return Task.FromResult(CreateMinimalSession()); },
            CancellationToken.None)) { }

        Assert.Equal(2, key2Calls); // factory called again; key2 was never stored in the pool
    }

    [Fact]
    public async Task LruEviction_SkipsLeasedEntry_AndEvictsIdleCandidate()
    {
        // Count-mode eviction, so admission is explicitly disabled.
        using var pool = new InferenceSessionPool(maxSessions: 2, enableMemoryAdmission: false);

        var key1 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "hash1", 0, "default");
        var key2 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "hash2", 0, "default");
        var key3 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "hash3", 0, "default");

        using SessionLease key1Lease = await pool.GetLeaseAsync(key1, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        using (await pool.GetLeaseAsync(key2, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None)) { }

        using (await pool.GetLeaseAsync(key3, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None)) { }

        int key2FactoryCalls = 0;
        using (await pool.GetLeaseAsync(
            key2,
            _ => { key2FactoryCalls++; return Task.FromResult(CreateMinimalSession()); },
            CancellationToken.None))
        {
        }

        Assert.Equal(1, key2FactoryCalls);
    }

    [Fact]
    public async Task EphemeralLease_SessionIsValid_AndReleasesWithoutThrowing()
    {
        using var pool = new InferenceSessionPool(maxSessions: 1, enableMemoryAdmission: false);

        var key1 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "hash1", 0, "default");
        var key2 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "hash2", 0, "default");

        // key1 held so pool is full and no eviction is possible.
        using var lease1 = await pool.GetLeaseAsync(key1, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        // key2 must be created as ephemeral.
        SessionLease ephemeralLease = await pool.GetLeaseAsync(
            key2,
            _ => Task.FromResult(CreateMinimalSession()),
            CancellationToken.None);

        Assert.NotNull(ephemeralLease.Session);

        // Disposing an ephemeral lease must not throw.
        ephemeralLease.Dispose();
    }

    // ── Single-flight creation ────────────────────────────────────────────────

    [Fact]
    public async Task GetLeaseAsync_ConcurrentMisses_SameKey_InvokesFactoryOnce()
    {
        using var pool = new InferenceSessionPool(maxSessions: 4);
        int factoryCalls = 0;

        Task<InferenceSession> Factory(CancellationToken _)
        {
            Interlocked.Increment(ref factoryCalls);
            return Task.Delay(50).ContinueWith(
                _ => CreateMinimalSession(),
                CancellationToken.None,
                TaskContinuationOptions.ExecuteSynchronously,
                TaskScheduler.Default);
        }

        // Same key is exclusive (one lease at a time), so collect leases sequentially:
        // the point is that the second acquire must not rebuild the session.
        Task<SessionLease> first = pool.GetLeaseAsync(TestKey, Factory, CancellationToken.None);
        Task<SessionLease> second = pool.GetLeaseAsync(TestKey, Factory, CancellationToken.None);

        SessionLease lease1 = await first;
        Assert.Equal(1, factoryCalls);
        lease1.Dispose();

        SessionLease lease2 = await second;
        lease2.Dispose();
        Assert.Equal(1, factoryCalls);
    }

    [Fact]
    public async Task GetLeaseAsync_CancelledWaiter_DoesNotCancelSharedCreate()
    {
        using var pool = new InferenceSessionPool(maxSessions: 4);
        int factoryCalls = 0;
        using var creatorCts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        using var waiterCts = new CancellationTokenSource(TimeSpan.FromMilliseconds(100));

        Task<SessionLease> creator = pool.GetLeaseAsync(
            TestKey,
            async _ =>
            {
                Interlocked.Increment(ref factoryCalls);
                await Task.Delay(300);
                return CreateMinimalSession();
            },
            creatorCts.Token);

        await Task.Delay(20);
        Task<SessionLease> waiter = pool.GetLeaseAsync(
            TestKey,
            _ =>
            {
                Interlocked.Increment(ref factoryCalls);
                return Task.FromResult(CreateMinimalSession());
            },
            waiterCts.Token);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waiter);

        SessionLease lease = await creator;
        lease.Dispose();

        Assert.Equal(1, factoryCalls);
    }

    // ── Memory admission (default-on hard admission) ─────────────────────────────────────────────

    [Fact]
    public async Task MemoryAdmission_DisabledExplicitly_StillAllowsEphemeral()
    {
        // Explicit opt-out preserves the historical count-mode ephemeral overflow.
        using var pool = new InferenceSessionPool(maxSessions: 1, enableMemoryAdmission: false);
        var key1 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "h1", 0, "default") { EstimatedVramMb = 64 };
        var key2 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "h2", 0, "default") { EstimatedVramMb = 64 };

        using SessionLease lease1 = await pool.GetLeaseAsync(key1, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        using SessionLease lease2 = await pool.GetLeaseAsync(key2, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        Assert.NotNull(lease2.Session);
    }

    [Fact]
    public async Task MemoryAdmission_WaitsInsteadOfEphemeral_WhenBudgetExhausted()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 4,
            enableMemoryAdmission: true,
            memoryBudgetMb: 100,
            hostMemoryBudgetMb: 100);
        var key1 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "h1", 0, "default") { EstimatedVramMb = 80 };
        var key2 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "h2", 0, "default") { EstimatedVramMb = 80 };

        SessionLease lease1 = await pool.GetLeaseAsync(key1, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
        // key2 needs 80MB while key1 holds 80MB against a 100MB budget — must wait, not go ephemeral.
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.GetLeaseAsync(key2, _ => Task.FromResult(CreateMinimalSession()), cts.Token));

        lease1.Dispose();

        using SessionLease lease2 = await pool.GetLeaseAsync(
            key2,
            _ => Task.FromResult(CreateMinimalSession()),
            CancellationToken.None);
        Assert.NotNull(lease2.Session);
    }

    [Fact]
    public async Task MemoryAdmission_EvictsIdleToFit_InsteadOfWaiting()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            enableMemoryAdmission: true,
            memoryBudgetMb: 120,
            hostMemoryBudgetMb: 120);
        var key1 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "h1", 0, "default") { EstimatedVramMb = 80 };
        var key2 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "h2", 0, "default") { EstimatedVramMb = 80 };

        using (await pool.GetLeaseAsync(key1, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None))
        {
        }

        // key1 is idle and must be evicted so key2 (80MB) fits in the 120MB budget.
        using SessionLease lease2 = await pool.GetLeaseAsync(
            key2,
            _ => Task.FromResult(CreateMinimalSession()),
            CancellationToken.None);
        Assert.NotNull(lease2.Session);
    }

    [Fact]
    public async Task MemoryAdmission_BudgetIsPerDevice_AndSharedAcrossEps()
    {
        // Same budget number applies per device. DML vs TRT is irrelevant — DeviceId is the bucket.
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            enableMemoryAdmission: true,
            memoryBudgetMb: 100);
        var gpu0Dml = new SessionPoolKey("eng", null, null, ExecutionProviderKind.DirectMl, "h0a", 0, "default") { EstimatedVramMb = 80 };
        var gpu0Trt = new SessionPoolKey("eng", null, null, ExecutionProviderKind.TensorRTRtx, "h0b", 0, "default") { EstimatedVramMb = 80 };
        var gpu1 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.DirectMl, "h1", 1, "default") { EstimatedVramMb = 80 };

        using SessionLease holdGpu0 = await pool.GetLeaseAsync(
            gpu0Dml, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        // GPU 1 has its own 100MB budget — must fit even though GPU 0 holds 80MB.
        using SessionLease takeGpu1 = await pool.GetLeaseAsync(
            gpu1, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        Assert.NotNull(takeGpu1.Session);

        // GPU 0 TRT shares the GPU 0 budget with DML — must wait.
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.GetLeaseAsync(gpu0Trt, _ => Task.FromResult(CreateMinimalSession()), cts.Token));
    }

    [Fact]
    public async Task GetLeaseAsync_PhaseCapture_RecordsTruthfulPoolCountersAndMaxima()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 4,
            enableMemoryAdmission: true,
            memoryBudgetMb: 100,
            hostMemoryBudgetMb: 100);
        var key1 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "pc1", 0, "default") { EstimatedVramMb = 80 };
        var key2 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "pc2", 0, "default") { EstimatedVramMb = 80 };

        var capture = new BenchmarkPhaseCapture();
        using (BenchmarkPhaseCapture.Activate(capture))
        {
            // Cold acquire: miss + create, reservation observed while pending.
            SessionLease lease1 = await pool.GetLeaseAsync(
                key1, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

            // 80MB held of a 100MB device budget — key2 becomes an admission waiter.
            using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));
            await Assert.ThrowsAnyAsync<OperationCanceledException>(
                () => pool.GetLeaseAsync(key2, _ => Task.FromResult(CreateMinimalSession()), cts.Token));

            // Releasing key1 frees budget; key2 misses and creates (evicting idle key1).
            lease1.Dispose();
            using (await pool.GetLeaseAsync(key2, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None))
            {
            }

            // Reacquire of the published key2 entry is a true hit: no factory call.
            using SessionLease reacquired = await pool.GetLeaseAsync(
                key2,
                _ => Task.FromException<InferenceSession>(new Exception("factory must not run on a hit")),
                CancellationToken.None);
        }

        IReadOnlyDictionary<string, long> counters = capture.SnapshotCounters();
        // The canceled admission waiter is still a miss — one miss or hit per acquire
        // request, while sessionCreate counts actual factory invocations.
        Assert.Equal(3, counters["poolMiss"]);
        Assert.Equal(2, counters["sessionCreate"]);
        Assert.Equal(1, counters["poolHit"]);

        IReadOnlyDictionary<string, long> maxima = capture.SnapshotMaxima();
        Assert.Equal(1, maxima["admissionWaiters"]);
        Assert.Equal(80, maxima["pendingReservationMb"]);
    }

    [Fact]
    public async Task GetLeaseAsync_AdmissionWait_RecordsPoolAdmissionWaitSpan()
    {
        // The admission wait was the one un-spanned async wait in the create path: a
        // tight-budget wait could burn hundreds of seconds invisibly (no phase total,
        // no log line). It must record under 'pool-admission-wait' like the other
        // pool waits (pool-wait, pool-single-flight-wait, pool-creation-lock-wait).
        using var pool = new InferenceSessionPool(
            maxSessions: 4,
            enableMemoryAdmission: true,
            memoryBudgetMb: 100,
            hostMemoryBudgetMb: 100);
        var key1 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "aw1", 0, "default") { EstimatedVramMb = 80 };
        var key2 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "aw2", 0, "default") { EstimatedVramMb = 80 };

        var capture = new BenchmarkPhaseCapture();
        using (BenchmarkPhaseCapture.Activate(capture))
        {
            using SessionLease lease1 = await pool.GetLeaseAsync(
                key1, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

            // 80 MB leased of a 100 MB budget: key2 parks in the admission wait until
            // the caller cancels — that wait is the span under test.
            using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));
            await Assert.ThrowsAnyAsync<OperationCanceledException>(
                () => pool.GetLeaseAsync(key2, _ => Task.FromResult(CreateMinimalSession()), cts.Token));
        }

        IReadOnlyDictionary<string, double?> phases = capture.SnapshotMilliseconds();
        Assert.True(
            phases.TryGetValue("phase:pool-admission-wait", out double? waitMs),
            "admission wait recorded no 'pool-admission-wait' phase span");
        Assert.True(waitMs > 0, "admission wait span recorded no elapsed time");
    }

    // ── Residency ≠ execution exclusivity (audit §3A) ─────────────────────────

    [Fact]
    public async Task GetResidencyAsync_PinsWithoutHoldingExecutionGate()
    {
        using var pool = new InferenceSessionPool(maxSessions: 4);
        SessionResidency residency = await pool.GetResidencyAsync(
            TestKey, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        // Pin held, but a lease must still be acquirable — residency is not exclusivity.
        using SessionLease lease = await pool.GetLeaseAsync(
            TestKey, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        Assert.NotNull(lease.Session);
        residency.Dispose();
    }

    [Fact]
    public async Task GetResidencyAsync_PinnedSession_IsNotIdleEvicted()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 4,
            enableMemoryAdmission: true,
            memoryBudgetMb: 100,
            hostMemoryBudgetMb: 100);
        var key1 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "p1", 0, "default") { EstimatedVramMb = 80 };
        var key2 = new SessionPoolKey("eng", null, null, ExecutionProviderKind.Cpu, "p2", 0, "default") { EstimatedVramMb = 80 };

        SessionResidency pin = await pool.GetResidencyAsync(
            key1, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        // key1 is idle (no lease) but pinned — must not be evicted to fit key2.
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.GetLeaseAsync(key2, _ => Task.FromResult(CreateMinimalSession()), cts.Token));

        // Unpin: key1 becomes eligible and is evicted so key2 fits.
        pin.Dispose();
        using SessionLease lease2 = await pool.GetLeaseAsync(
            key2, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        Assert.NotNull(lease2.Session);
    }

    [Fact]
    public void SessionPoolKey_EstimateVramMb_FromFileSize()
    {
        string path = Path.Join(Path.GetTempPath(), $"vram-est-{Guid.NewGuid():N}.bin");
        try
        {
            File.WriteAllBytes(path, new byte[3 * 1024 * 1024]); // 3 MB → 3*2+128 = 134
            long estimate = SessionPoolKey.EstimateVramMb(path, ExecutionProviderKind.Cpu);
            Assert.Equal(134, estimate);
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ── Multi-graph lease bundles (audit §3A) ────────────────────────────────

    private static SessionPoolKey KeyA() =>
        new("dual", null, null, ExecutionProviderKind.Cpu, "hashA", 0, "encoder") { EstimatedVramMb = 32 };

    private static SessionPoolKey KeyB() =>
        new("dual", null, null, ExecutionProviderKind.Cpu, "hashB", 0, "decoder") { EstimatedVramMb = 32 };

    [Fact]
    public async Task GetLeaseBundleAsync_AcquiresAllGraphs()
    {
        using var pool = new InferenceSessionPool(maxSessions: 4);
        using SessionLeaseBundle bundle = await pool.GetLeaseBundleAsync(
            [
                new SessionLeaseRequest(KeyA(), _ => Task.FromResult(CreateMinimalSession())),
                new SessionLeaseRequest(KeyB(), _ => Task.FromResult(CreateMinimalSession())),
            ],
            CancellationToken.None);

        Assert.Equal(2, bundle.Count);
        Assert.NotNull(bundle[0]);
        Assert.NotNull(bundle[1]);
    }

    [Fact]
    public async Task GetLeaseBundleAsync_DuplicateKeys_Throws()
    {
        using var pool = new InferenceSessionPool(maxSessions: 4);
        await Assert.ThrowsAsync<ArgumentException>(() => pool.GetLeaseBundleAsync(
            [
                new SessionLeaseRequest(KeyA(), _ => Task.FromResult(CreateMinimalSession())),
                new SessionLeaseRequest(KeyA(), _ => Task.FromResult(CreateMinimalSession())),
            ],
            CancellationToken.None));
    }

    [Fact]
    public async Task GetLeaseBundleAsync_OpposingCallerOrders_DoNotDeadlock()
    {
        using var pool = new InferenceSessionPool(maxSessions: 8);
        SessionLeaseRequest a = new(KeyA(), _ => Task.FromResult(CreateMinimalSession()));
        SessionLeaseRequest b = new(KeyB(), _ => Task.FromResult(CreateMinimalSession()));
        // Same graphs are exclusive, so bundles serialize — the point is that reversed
        // caller order still completes (stable sort) instead of deadlocking half-held. The
        // acquisitions carry no token and the awaits below carry the hang guard: what is under
        // test is completion, and a five-second token reported a slow runner as a deadlock.
        Task<SessionLeaseBundle> t1 = Task.Run(() => pool.GetLeaseBundleAsync([a, b], CancellationToken.None));
        Task<SessionLeaseBundle> t2 = Task.Run(() => pool.GetLeaseBundleAsync([b, a], CancellationToken.None));

        Task<SessionLeaseBundle> first = await Task.WhenAny(t1, t2)
            .AwaitWithHangGuard("neither bundle acquisition completed");
        SessionLeaseBundle firstBundle = await first.AwaitWithHangGuard("the first bundle never resolved");
        firstBundle.Dispose();

        Task<SessionLeaseBundle> second = ReferenceEquals(first, t1) ? t2 : t1;
        SessionLeaseBundle secondBundle = await second.AwaitWithHangGuard("the second bundle never resolved");
        secondBundle.Dispose();
    }

    [Fact]
    public async Task GetLeaseBundleAsync_DisjointBundles_CreateFactoriesConcurrently()
    {
        // C1: disjoint bundles must not serialize behind a process-wide bundle lock.
        // Every factory signals entry and then blocks on a gate; all four factories
        // (two per bundle) can only be in flight at once if bundle creation overlaps.
        // Synchronization proves overlap — no wall-clock timing is asserted.
        using var pool = new InferenceSessionPool(maxSessions: 8);
        int enteredCount = 0;
        var allEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        Func<CancellationToken, Task<InferenceSession>> GatedFactory() => async ct =>
        {
            if (Interlocked.Increment(ref enteredCount) == 4)
            {
                allEntered.TrySetResult();
            }

            await release.Task.WaitAsync(ct);
            return CreateMinimalSession();
        };

        SessionLeaseRequest Req(string hash) => new(
            new SessionPoolKey("disjoint", null, null, ExecutionProviderKind.Cpu, hash, 0, "default") { EstimatedVramMb = 16 },
            GatedFactory());

        Task<SessionLeaseBundle> t1 = Task.Run(() => pool.GetLeaseBundleAsync(
            [Req("dx1"), Req("dx2")], CancellationToken.None));
        Task<SessionLeaseBundle> t2 = Task.Run(() => pool.GetLeaseBundleAsync(
            [Req("dy1"), Req("dy2")], CancellationToken.None));

        try
        {
            await allEntered.Task.AwaitWithHangGuard(
                "factories from disjoint bundles never ran concurrently (bundle creation is serialized)");
            Assert.Equal(4, enteredCount);
        }
        finally
        {
            release.TrySetResult();
        }

        using SessionLeaseBundle bundle1 = await t1.AwaitWithHangGuard("first disjoint bundle never completed");
        using SessionLeaseBundle bundle2 = await t2.AwaitWithHangGuard("second disjoint bundle never completed");
        Assert.Equal(2, bundle1.Count);
        Assert.Equal(2, bundle2.Count);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task GetLeaseBundleAsync_SynchronousFactories_CreateBundleGraphsConcurrently(bool enableMemoryAdmission)
    {
        // Production factories are ct => Task.FromResult(CreateSession(...)): the session
        // is created synchronously on the calling thread before the task exists, so no
        // await inside the create path ever yields. Bundle warming must still overlap the
        // graphs of one bundle (the yielding factories in the disjoint-bundle test above
        // cannot catch a serial loop). Every factory signals entry, then blocks its thread
        // on a gate; all three can only be entered before release if creation overlaps.
        using var pool = new InferenceSessionPool(maxSessions: 8, enableMemoryAdmission: enableMemoryAdmission);
        int enteredCount = 0;
        using var allEntered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();

        Func<CancellationToken, Task<InferenceSession>> SyncGatedFactory() => _ =>
        {
            if (Interlocked.Increment(ref enteredCount) == 3)
            {
                allEntered.Set();
            }

            release.Wait();
            return Task.FromResult(CreateMinimalSession());
        };

        SessionLeaseRequest Req(string hash) => new(
            new SessionPoolKey("syncbundle", null, null, ExecutionProviderKind.Cpu, hash, 0, "default") { EstimatedVramMb = 16 },
            SyncGatedFactory());

        Task<SessionLeaseBundle> acquire = Task.Run(() => pool.GetLeaseBundleAsync(
            [Req("sx1"), Req("sx2"), Req("sx3")], CancellationToken.None));

        // Release (and drain the acquire) before asserting so a serial bundle fails
        // cleanly instead of leaving a factory blocked on the gate.
        bool overlapped = allEntered.Wait(TimeSpan.FromSeconds(10));
        release.Set();
        using SessionLeaseBundle bundle = await acquire.AwaitWithHangGuard("the synchronous-factory bundle never completed");

        Assert.True(overlapped,
            "synchronous factories never overlapped within one bundle (bundle graph creation is serial)");
        Assert.Equal(3, Volatile.Read(ref enteredCount));
        Assert.Equal(3, bundle.Count);
    }

    [Fact]
    public async Task GetLeaseBundleAsync_OverlappingOppositeOrders_UnderEvictionPressure_BothComplete()
    {
        // C1 deadlock regression: two bundles request the same graphs in opposite caller
        // orders while the host budget forces eviction of an idle filler. Both must
        // complete (one gate-set at a time); neither may pin itself into a deadlock.
        using var pool = new InferenceSessionPool(maxSessions: 8, hostMemoryBudgetMb: 100);
        var shared1 = HostKey("sh1", 40);
        var shared2 = HostKey("sh2", 40);
        var filler = HostKey("fill", 40);

        // Idle filler occupies budget that bundle creation must evict to fit.
        using (await pool.GetLeaseAsync(filler, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None))
        {
        }

        SessionLeaseRequest a = new(shared1, _ => Task.FromResult(CreateMinimalSession()));
        SessionLeaseRequest b = new(shared2, _ => Task.FromResult(CreateMinimalSession()));

        Task<SessionLeaseBundle> t1 = Task.Run(() => pool.GetLeaseBundleAsync([a, b], CancellationToken.None));
        Task<SessionLeaseBundle> t2 = Task.Run(() => pool.GetLeaseBundleAsync([b, a], CancellationToken.None));

        Task<SessionLeaseBundle> first = await Task.WhenAny(t1, t2)
            .AwaitWithHangGuard("neither overlapping bundle completed under eviction pressure");
        SessionLeaseBundle firstBundle = await first.AwaitWithHangGuard("the first overlapping bundle never resolved");
        firstBundle.Dispose();

        Task<SessionLeaseBundle> second = ReferenceEquals(first, t1) ? t2 : t1;
        SessionLeaseBundle secondBundle = await second.AwaitWithHangGuard("the second overlapping bundle never resolved");
        secondBundle.Dispose();
    }

    [Fact]
    public async Task GetLeaseBundleAsync_CancellationDuringCreate_LeavesNoPinnedHalfBundle()
    {
        // A bundle cancelled while a later graph is still creating must not leave the
        // earlier graph pinned: a probe that only fits by evicting that graph succeeds
        // only if the pin was released (same probe shape as the preparation-failure test).
        using var pool = new InferenceSessionPool(maxSessions: 8, hostMemoryBudgetMb: 100);
        var first = HostKey("cx1", 40);
        var second = HostKey("cx2", 40);
        var probe = HostKey("cxp", 80);
        var secondEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var never = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        using var cts = new CancellationTokenSource();
        Task<SessionLeaseBundle> acquire = pool.GetLeaseBundleAsync(
            [
                new SessionLeaseRequest(first, _ => Task.FromResult(CreateMinimalSession())),
                new SessionLeaseRequest(second, async ct =>
                {
                    secondEntered.TrySetResult();
                    await never.Task.WaitAsync(ct);
                    return CreateMinimalSession();
                }),
            ],
            cts.Token);

        await secondEntered.Task.AwaitWithHangGuard("the second graph factory never started");
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => acquire);

        int probeCalls = 0;
        using SessionLease lease = await pool.GetLeaseAsync(
            probe,
            _ => { probeCalls++; return Task.FromResult(CreateMinimalSession()); },
            CancellationToken.None).AwaitWithHangGuard("probe never acquired — a cancelled bundle left a pinned half-bundle");
        Assert.Equal(1, probeCalls);
        Assert.NotNull(lease.Session);
    }

    [Fact]
    public async Task GetLeaseBundleAsync_SiblingCountsAgainstGpuFloor_DoesNotEvictWarmedGraphMidAcquire()
    {
        // Regression for the C1 bundle-admission stall: graph A's freshly created
        // session raises the process-GPU observation floor, so on the old shape
        // graph B's per-graph admission wait found the bucket observation-blocked
        // and evicted the only idle entry — A itself, still unpinned until phase 2 —
        // then phase 2 re-created A (in production each such cycle also resets the
        // other wait's stall progress). The scripted reader models exactly that:
        // no floor until A is created, a hot 90 MB floor for the next few reads
        // (A's session counts against the 100 MB budget), then drained.
        var reader = new FloorAfterFirstCreateReader(hotMb: 90, drainedMb: 10, hotReads: 3);
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 100,
            hostMemoryBudgetMb: 100,
            processGpuMemoryReader: () => reader);
        var graphA = new SessionPoolKey("bundlefloor", null, null, ExecutionProviderKind.DirectMl, "bf-a", 0, "a-graph") { EstimatedVramMb = 40 };
        var graphB = new SessionPoolKey("bundlefloor", null, null, ExecutionProviderKind.DirectMl, "bf-b", 0, "b-graph") { EstimatedVramMb = 40 };
        int aCalls = 0;
        int bCalls = 0;

        using SessionLeaseBundle bundle = await pool.GetLeaseBundleAsync(
            [
                new SessionLeaseRequest(graphA, _ =>
                {
                    aCalls++;
                    reader.MarkCreated();
                    return Task.FromResult(CreateMinimalSession());
                }),
                new SessionLeaseRequest(graphB, _ =>
                {
                    bCalls++;
                    return Task.FromResult(CreateMinimalSession());
                }),
            ],
            CancellationToken.None).AwaitWithHangGuard("bundle under a sibling-raised GPU floor never completed");

        Assert.Equal(2, bundle.Count);
        // A created exactly once: it was never evicted mid-acquire and re-created in
        // the pin phase. B likewise.
        Assert.Equal(1, aCalls);
        Assert.Equal(1, bCalls);
    }

    [Fact]
    public async Task GetLeaseBundleAsync_WarmQueuedBehindAdmissionWaitingCreator_DoesNotDeadlock()
    {
        // Regression for the createGate hold deadlock: a plain acquire takes K1's
        // single-flight gate and then waits for admission (the bundle's aggregate
        // hold plus a pinned sibling leave no room), while the bundle's warm for K1
        // queues behind that gate still holding its K1 share of the aggregate. The
        // share is the only thing the plain wait could ever be freed by, so both
        // hang until cancellation unless the warm returns its share before queueing.
        // The scripted reader parks the plain acquire (gate held) until the bundle
        // hold exists, making the interleave deterministic.
        var reader = new FirstReadBlockingReader();
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 100,
            hostMemoryBudgetMb: 100,
            processGpuMemoryReader: () => reader);
        var graphK1 = new SessionPoolKey("gatehold", null, null, ExecutionProviderKind.DirectMl, "gh-1", 0, "g1") { EstimatedVramMb = 40 };
        var graphK2 = new SessionPoolKey("gatehold", null, null, ExecutionProviderKind.DirectMl, "gh-2", 0, "g2") { EstimatedVramMb = 40 };
        int plainCalls = 0;
        int bundleK1Calls = 0;
        int bundleK2Calls = 0;
        var k2Created = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);

        Task<SessionLease> plainTask = Task.Run(() => pool.GetLeaseAsync(
            graphK1,
            _ =>
            {
                Interlocked.Increment(ref plainCalls);
                return Task.FromResult(CreateMinimalSession());
            },
            CancellationToken.None));

        await reader.FirstReadEntered.AwaitWithHangGuard("the plain acquire never reached its admission observation");

        Task<SessionLeaseBundle> bundleTask = Task.Run(() => pool.GetLeaseBundleAsync(
            [
                new SessionLeaseRequest(graphK1, _ =>
                {
                    Interlocked.Increment(ref bundleK1Calls);
                    return Task.FromResult(CreateMinimalSession());
                }),
                new SessionLeaseRequest(graphK2, _ =>
                {
                    Interlocked.Increment(ref bundleK2Calls);
                    k2Created.TrySetResult();
                    return Task.FromResult(CreateMinimalSession());
                }),
            ],
            CancellationToken.None));

        await k2Created.Task.AwaitWithHangGuard("the bundle's second graph never created");
        reader.ReleaseFirstRead();

        using SessionLease plainLease = await plainTask.AwaitWithHangGuard(
            "the plain acquire deadlocked against the bundle warm queued behind its gate");
        // Release the plain lease before awaiting the bundle: the warm's residency
        // acquire serializes on the same entry gate the lease holds.
        plainLease.Dispose();
        using SessionLeaseBundle bundle = await bundleTask.AwaitWithHangGuard(
            "the bundle warm deadlocked behind the admission-waiting plain acquire");

        Assert.Equal(2, bundle.Count);
        Assert.Equal(1, plainCalls);
        // The bundle's K1 warm landed on the session the plain acquire published.
        Assert.Equal(0, bundleK1Calls);
        Assert.Equal(1, bundleK2Calls);
    }

    /// <summary>
    /// Reader whose first read parks until released: holds a plain acquire (which
    /// has already taken its key's single-flight gate) at the admission observation
    /// while a bundle takes its aggregate hold. Later reads return immediately.
    /// </summary>
    private sealed class FirstReadBlockingReader : IProcessGpuMemoryReader
    {
        private int reads;
        private readonly TaskCompletionSource firstReadEntered = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private readonly TaskCompletionSource releaseFirstRead = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public string UnavailableReason => "Test double: first read blocks until released.";

        public Task FirstReadEntered => firstReadEntered.Task;

        public void ReleaseFirstRead() => releaseFirstRead.TrySetResult();

        public long? ReadDedicatedGpuMemoryBytes()
        {
            if (Interlocked.Increment(ref reads) == 1)
            {
                firstReadEntered.TrySetResult();
                releaseFirstRead.Task.Wait();
            }

            return null;
        }
    }

    [Fact]
    public async Task GetLeaseBundleAsync_DisjointBundles_BudgetFitsOnlyOne_BothCompleteWithoutDeadlock()
    {
        // Pinning each graph as its create completes is only deadlock-safe because the
        // aggregate reservation is taken all-or-nothing first: with a budget that fits
        // just one bundle, the losing bundle must wait at the aggregate step holding
        // no pins, and complete once the winner's leases are released. (Pin-first
        // without the aggregate would let each bundle pin one graph and then wait
        // forever on the other's memory.)
        using var pool = new InferenceSessionPool(maxSessions: 8, hostMemoryBudgetMb: 100);

        SessionLeaseRequest Req(string hash) => new(
            HostKey(hash, 40), _ => Task.FromResult(CreateMinimalSession()));

        Task<SessionLeaseBundle> t1 = Task.Run(() => pool.GetLeaseBundleAsync(
            [Req("db1a"), Req("db1b")], CancellationToken.None));
        Task<SessionLeaseBundle> t2 = Task.Run(() => pool.GetLeaseBundleAsync(
            [Req("db2a"), Req("db2b")], CancellationToken.None));

        Task<SessionLeaseBundle> first = await Task.WhenAny(t1, t2)
            .AwaitWithHangGuard("neither disjoint bundle completed under a one-bundle budget");
        SessionLeaseBundle firstBundle = await first.AwaitWithHangGuard("the first disjoint bundle never resolved");
        firstBundle.Dispose();

        Task<SessionLeaseBundle> second = ReferenceEquals(first, t1) ? t2 : t1;
        SessionLeaseBundle secondBundle = await second.AwaitWithHangGuard(
            "the second disjoint bundle never resolved — early pins deadlocked the aggregate wait");
        secondBundle.Dispose();
    }

    [Fact]
    public async Task GetLeaseBundleAsync_DoesNotHoldFirstGraphWhileBlockedOnSecond()
    {
        using var pool = new InferenceSessionPool(maxSessions: 8);
        SessionLeaseRequest a = new(KeyA(), _ => Task.FromResult(CreateMinimalSession()));
        SessionLeaseRequest b = new(KeyB(), _ => Task.FromResult(CreateMinimalSession()));

        // Warm both graphs so phase-2 is pure gate acquire.
        using (await pool.GetLeaseBundleAsync([a, b], CancellationToken.None))
        {
        }

        // Hold graph A exclusively.
        SessionLease holdA = await pool.GetLeaseAsync(KeyA(), _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(250));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.GetLeaseBundleAsync([a, b], cts.Token));

        // Graph B must still be acquirable: the failed bundle did not keep B while stuck on A.
        using SessionLease takeB = await pool.GetLeaseAsync(
            KeyB(),
            _ => Task.FromResult(CreateMinimalSession()),
            CancellationToken.None);
        Assert.NotNull(takeB.Session);
        holdA.Dispose();
    }

    [Fact]
    public async Task GetLeaseBundleAsync_EvictedResidentRecreate_DoesNotStealSiblingCredit()
    {
        // T1 admission-credit correctness: the aggregate hold is sized only for graphs
        // missing at hold time. The hold wait itself can evict an idle resident graph of
        // the same bundle; its re-create must not draw on the siblings' shared credit.
        // With per-bucket shared credit and all-or-nothing TryConsume, a steal leaves a
        // sliver the last sibling cannot consume, so it falls back to a full reservation
        // while the sliver stays pending (phantom over-count) with early pins unevictable
        // and no fail-fast on reservation-blocked waits — the bundle hangs though it fits.
        // Sizes are chosen so a steal strands the large sibling: B=50 cannot consume a
        // sliver left by A=30/C=10 sharing the 60 MB hold, while the scoped fix lets B/C
        // consume exactly their shares and A fall back to ordinary admission (evicting
        // only the idle filler). Loop iterations to cover the racy consume order; the
        // delay between the pre-populations keeps the LRU ticks distinct so the hold
        // wait deterministically evicts the resident A rather than the filler.
        for (int iteration = 0; iteration < 10; iteration++)
        {
            using var pool = new InferenceSessionPool(maxSessions: 8, hostMemoryBudgetMb: 100);
            var residentA = new SessionPoolKey("t1", "ma", null, ExecutionProviderKind.Cpu, "t1a-" + iteration, null, "default") { EstimatedVramMb = 30 };
            var missingB = new SessionPoolKey("t1", "mb", null, ExecutionProviderKind.Cpu, "t1b-" + iteration, null, "default") { EstimatedVramMb = 50 };
            var missingC = new SessionPoolKey("t1", "mc", null, ExecutionProviderKind.Cpu, "t1c-" + iteration, null, "default") { EstimatedVramMb = 10 };
            var filler = new SessionPoolKey("t1", "mf", null, ExecutionProviderKind.Cpu, "t1f-" + iteration, null, "default") { EstimatedVramMb = 40 };

            // Resident A is oldest (evicted first); filler is recent.
            using (await pool.GetLeaseAsync(residentA, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None))
            {
            }
            await Task.Delay(20);
            using (await pool.GetLeaseAsync(filler, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None))
            {
            }

            int aCalls = 0, bCalls = 0, cCalls = 0;
            using SessionLeaseBundle bundle = await pool.GetLeaseBundleAsync(
                [
                    new SessionLeaseRequest(residentA, _ => { Interlocked.Increment(ref aCalls); return Task.FromResult(CreateMinimalSession()); }),
                    new SessionLeaseRequest(missingB, _ => { Interlocked.Increment(ref bCalls); return Task.FromResult(CreateMinimalSession()); }),
                    new SessionLeaseRequest(missingC, _ => { Interlocked.Increment(ref cCalls); return Task.FromResult(CreateMinimalSession()); }),
                ],
                CancellationToken.None).AwaitWithHangGuard(
                    $"bundle with an evicted resident hung on phantom credit (iteration {iteration})");

            Assert.Equal(3, bundle.Count);
            // A was evicted by the hold wait itself, so it is re-created once; B and C
            // each consume exactly their scoped share — never a stolen fallback.
            Assert.Equal(1, Volatile.Read(ref aCalls));
            Assert.Equal(1, Volatile.Read(ref bCalls));
            Assert.Equal(1, Volatile.Read(ref cCalls));
        }
    }

    [Fact]
    public async Task GetLeaseBundleAsync_ExplicitEvictWhileBundleWaiting_CompletesWithoutDeadlock()
    {
        // T2 lock-ordering liveness: phase-1 early pins must not be carried into the
        // bundleAcquireLock wait. A bundle stuck holding that lock while re-creating an
        // explicitly-evicted graph in phase-2 admission deadlocks against a waiter whose
        // early pins are unevictable (and unreleasable until it owns the lock), with no
        // fail-fast on reservation-blocked waits. Dropping early pins before the lock
        // wait is safe because phase 2 re-pins every graph; the probe below fits only
        // once the waiter's (and holder's) early pins are gone.
        using var pool = new InferenceSessionPool(maxSessions: 8, hostMemoryBudgetMb: 200);
        SessionPoolKey Key(string engine, string model, string hash, long mb) =>
            new SessionPoolKey(engine, model, null, ExecutionProviderKind.Cpu, hash, null, "default") { EstimatedVramMb = mb };

        var keyHA = Key("h", "a", "t2ha", 40);
        var keyHB = Key("h", "b", "t2hb", 40);
        var keyWC = Key("w", "c", "t2wc", 40);
        var keyWD = Key("w", "d", "t2wd", 40);
        var keyProbe = Key("p", "q", "t2pq", 100);

        int haCalls = 0, hbEntered = 0;
        var releaseHB = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var releaseHA2 = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        Func<CancellationToken, Task<InferenceSession>> Hafactory() => async ct =>
        {
            // First call (phase 1) runs immediately; the phase-2 re-create after the
            // explicit eviction below stays blocked, holding the bundle lock.
            if (Interlocked.Increment(ref haCalls) == 1)
            {
                return CreateMinimalSession();
            }

            await releaseHA2.Task.WaitAsync(ct);
            return CreateMinimalSession();
        };
        Func<CancellationToken, Task<InferenceSession>> HbFactory() => async ct =>
        {
            Interlocked.Increment(ref hbEntered);
            await releaseHB.Task.WaitAsync(ct);
            return CreateMinimalSession();
        };

        Task<SessionLeaseBundle> holder = Task.Run(() => pool.GetLeaseBundleAsync(
            [new SessionLeaseRequest(keyHA, Hafactory()), new SessionLeaseRequest(keyHB, HbFactory())],
            CancellationToken.None));

        // Wait until HA is published and HB's factory is in flight (phase 1 spans the
        // wide window below because HB stays blocked). Counter polling only.
        await SpinUntilAsync(() => Volatile.Read(ref haCalls) >= 1, "HA was never published");
        await SpinUntilAsync(() => Volatile.Read(ref hbEntered) >= 1, "HB factory never entered");

        // Explicitly evict HA while the holder is still in phase 1; its phase-2 pin
        // loop must re-create it via admission while holding the bundle lock. Spin
        // until the eviction lands: HA#1 may not have published yet when we arrive.
        await SpinUntilTaskAsync(
            async () => await pool.EvictModelAsync("h", "a", CancellationToken.None) == 1,
            "HA never became evictable");

        // Wait until the holder is stuck in its phase-2 pin loop: HA was re-requested
        // (call #2) after the explicit eviction, so the holder owns the bundle lock
        // while its HA factory stays blocked. Only then start the waiter, which
        // parks at the lock after finishing phase 1.
        releaseHB.TrySetResult();
        await SpinUntilAsync(() => Volatile.Read(ref haCalls) >= 2, "holder never stuck re-creating HA");

        int wcCalls = 0, wdCalls = 0;
        Task<SessionLeaseBundle> waiter = Task.Run(() => pool.GetLeaseBundleAsync(
            [
                new SessionLeaseRequest(keyWC, _ => { Interlocked.Increment(ref wcCalls); return Task.FromResult(CreateMinimalSession()); }),
                new SessionLeaseRequest(keyWD, _ => { Interlocked.Increment(ref wdCalls); return Task.FromResult(CreateMinimalSession()); }),
            ],
            CancellationToken.None));

        // Wait until the waiter finished phase 1 (both graphs published) so it is
        // parked at the bundle lock while the holder is stuck owning it.
        await SpinUntilAsync(
            () => Volatile.Read(ref wcCalls) >= 1 && Volatile.Read(ref wdCalls) >= 1,
            "waiter never finished phase 1");

        // The holder owns the lock and is stuck re-creating evicted HA; the waiter
        // parks at the lock. A 100 MB probe fits the 200 MB budget only by evicting a
        // 40 MB graph, which is possible only when no early pins are held across the
        // lock wait.
        try
        {
            using SessionLease probe = await pool.GetLeaseAsync(
                    keyProbe, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None)
                .WaitAsync(TimeSpan.FromSeconds(15));
            Assert.NotNull(probe.Session);
        }
        finally
        {
            releaseHB.TrySetResult();
            releaseHA2.TrySetResult();
        }

        // Let every bundle finish (re-created graphs fit once the probe is gone).
        releaseHA2.TrySetResult();
        using SessionLeaseBundle holderBundle = await holder.WaitAsync(TimeSpan.FromSeconds(15));
        Assert.Equal(2, holderBundle.Count);
        holderBundle.Dispose();
        using SessionLeaseBundle waiterBundle = await waiter.WaitAsync(TimeSpan.FromSeconds(15));
        Assert.Equal(2, waiterBundle.Count);

        static async Task SpinUntilAsync(Func<bool> condition, string expectation)
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
            while (!condition())
            {
                await Task.Delay(5, cts.Token);
            }

            if (!condition())
            {
                throw new TimeoutException(expectation);
            }
        }

        static async Task SpinUntilTaskAsync(Func<Task<bool>> condition, string expectation)
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
            while (!await condition().WaitAsync(cts.Token))
            {
                await Task.Delay(5, cts.Token);
            }
        }
    }

    // ── TrimToVramBudgetAsync ─────────────────────────────────────────────────

    [Fact]
    public async Task TrimToVramBudgetAsync_AfterDispose_Throws()
    {
        var pool = new InferenceSessionPool(4);
        pool.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(
            () => pool.TrimToVramBudgetAsync(1024));
    }

    [Fact]
    public async Task TrimToVramBudgetAsync_NegativeTarget_Throws()
    {
        using var pool = new InferenceSessionPool(4);

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            () => pool.TrimToVramBudgetAsync(-1));
    }

    [Fact]
    public async Task TrimToVramBudgetAsync_EmptyPool_ReturnsZero()
    {
        using var pool = new InferenceSessionPool(4);

        int evicted = await pool.TrimToVramBudgetAsync(0);

        Assert.Equal(0, evicted);
    }

    [Fact]
    public async Task TrimToVramBudgetAsync_BudgetNotExceeded_EvictsNothing()
    {
        using var pool = new InferenceSessionPool(4);
        var key = new SessionPoolKey("eng", "m", null, ExecutionProviderKind.Cpu, "h1", 0, "default")
        {
            EstimatedVramMb = 500
        };

        // Add and release the session so it's idle
        using (await pool.GetLeaseAsync(key, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None)) { }

        // Budget is well above the session's footprint
        int evicted = await pool.TrimToVramBudgetAsync(1000);

        Assert.Equal(0, evicted);
    }

    [Fact]
    public async Task TrimToVramBudgetAsync_BudgetExceeded_EvictsLruIdleSession()
    {
        using var pool = new InferenceSessionPool(4);
        var key1 = new SessionPoolKey("eng", "m", null, ExecutionProviderKind.Cpu, "h1", 0, "default")
        {
            EstimatedVramMb = 600
        };
        var key2 = new SessionPoolKey("eng", "m2", null, ExecutionProviderKind.Cpu, "h2", 0, "default")
        {
            EstimatedVramMb = 600
        };

        using (await pool.GetLeaseAsync(key1, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None)) { }
        using (await pool.GetLeaseAsync(key2, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None)) { }

        // Total 1200 MB; trim to 700 MB — should evict exactly 1 session (the LRU one)
        int evicted = await pool.TrimToVramBudgetAsync(700);

        Assert.Equal(1, evicted);
    }

    [Fact]
    public async Task TrimToVramBudgetAsync_DoesNotEvictLeasedSession()
    {
        using var pool = new InferenceSessionPool(4);
        var key = new SessionPoolKey("eng", "m", null, ExecutionProviderKind.Cpu, "h1", 0, "default")
        {
            EstimatedVramMb = 2000
        };

        // Hold the lease while we try to trim
        using (await pool.GetLeaseAsync(key, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None))
        {
            int evicted = await pool.TrimToVramBudgetAsync(0);

            // Session is leased; cannot be evicted
            Assert.Equal(0, evicted);
        }
    }

    // ── Process-isolated GPU admission (the reading drives admission, not only telemetry) ──

    [Fact]
    public async Task ProcessGpuAdmission_ProcessAtDeviceCeiling_FailsFastWithNothingIdleLeftToFree()
    {
        // This process already holds the device budget's worth of dedicated GPU memory outside
        // the pool's reservations (driver context, runtime arenas, non-pooled consumers). The
        // reservation fits the configured budget on its own, the observation is what blocks, and
        // there is no idle pooled session or external reservation to free — so admission fails
        // fast with a diagnosis instead of polling a wait that can never succeed. It must never
        // construct the session anyway.
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => new StubProcessGpuMemoryReader(4096L * 1024 * 1024));
        int factoryCalls = 0;

        InvalidOperationException failure = await Assert.ThrowsAsync<InvalidOperationException>(
            () => pool.GetLeaseAsync(
                AcceleratorKey("pg1", 256),
                _ => { factoryCalls++; return Task.FromResult(CreateMinimalSession()); },
                CancellationToken.None));

        Assert.Contains("dedicated GPU memory is attributed to this bucket", failure.Message, StringComparison.Ordinal);
        Assert.Contains("TRACKDUB_SESSION_PROCESS_GPU_ADMISSION", failure.Message, StringComparison.Ordinal);
        // MadladTranslationEngine falls back to CPU on messages naming the admission budget.
        Assert.Contains("admission budget", failure.Message, StringComparison.Ordinal);
        Assert.Equal(0, factoryCalls);
    }

    [Fact]
    public async Task ProcessGpuAdmission_ObservedUsageUnderBudget_KeepsAdmittingReservations()
    {
        // The floor only ever tightens: an observation that still leaves headroom behaves exactly
        // like reservation-only accounting.
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => new StubProcessGpuMemoryReader(1024L * 1024 * 1024));

        using SessionLease first = await pool.GetLeaseAsync(
            AcceleratorKey("pg2a", 2000), _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        using SessionLease second = await pool.GetLeaseAsync(
            AcceleratorKey("pg2b", 2000), _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        Assert.NotNull(second.Session);
    }

    [Fact]
    public async Task ProcessGpuAdmission_HostRamBucketsIgnoreTheAcceleratorObservation()
    {
        // CPU/DNNL sessions live in RAM, so the process's GPU footprint must not gate them.
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => new StubProcessGpuMemoryReader(8192L * 1024 * 1024));

        using SessionLease lease = await pool.GetLeaseAsync(
            HostKey("pg3", 3000), _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        Assert.NotNull(lease.Session);
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(0, true)]
    [InlineData(1, false)]
    [InlineData(1, true)]
    [InlineData(2, false)]
    [InlineData(2, true)]
    public async Task ProcessGpuAdmission_UnattributedUsageCannotBeReducedBySiblingEstimates(
        int attributionGap, bool externalSibling)
    {
        IProcessGpuMemoryReader reader = new StubProcessGpuMemoryReader(0);
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => reader);

        using SessionLease? sibling = externalSibling ? null : await pool.GetLeaseAsync(
            AcceleratorKey("pg4a", 3000, deviceId: 1),
            _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        using ExternalMemoryReservation? external = externalSibling
            ? await pool.ReserveExternalAsync(new(ExecutionProviderKind.DirectMl, 1, 3000), CancellationToken.None)
            : null;

        IReadOnlyDictionary<int, long>? previousMap = SharedPoolOptions.AdapterLuidMap;
        int previousBound = InferenceSessionPool.ObservedBlockFailFastPasses;
        SharedPoolOptions.UseAdapterLuidMap(attributionGap == 0 ? null
            : attributionGap == 2 ? new Dictionary<int, long> { [1] = 200 }
            : new Dictionary<int, long> { [0] = 100 });
        InferenceSessionPool.ObservedBlockFailFastPasses = 1;
        try
        {
            // The sibling estimates 3000 MB but really holds 1024 MB; GPU 0 holds 3072 MB
            // outside the pool. Subtracting the estimate from the 4096 MB process total
            // would admit another 2048 MB on GPU 0 and hide the budget breach.
            reader = new PerAdapterProcessGpuMemoryReader(4096L * 1024 * 1024,
                attributionGap == 1 ? null : new Dictionary<long, long>
                {
                    [100] = 3072L * 1024 * 1024,
                    [200] = 1024L * 1024 * 1024,
                });
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            int factoryCalls = 0;
            await Assert.ThrowsAsync<InvalidOperationException>(() => pool.GetLeaseAsync(
                AcceleratorKey("pg4b", 2048, deviceId: 0),
                _ => { factoryCalls++; return Task.FromResult(CreateMinimalSession()); }, cts.Token));
            Assert.Equal(0, factoryCalls);
        }
        finally
        {
            SharedPoolOptions.UseAdapterLuidMap(previousMap);
            InferenceSessionPool.ObservedBlockFailFastPasses = previousBound;
        }
    }

    [Fact]
    public async Task ProcessGpuAdmission_ProcessTotalDoesNotBlockNpuAdmission()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, memoryBudgetMb: 4096, hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => new StubProcessGpuMemoryReader(8192L * 1024 * 1024));
        var key = new SessionPoolKey("eng", null, null, ExecutionProviderKind.OpenVino, "npu-gpu-floor", null, "default")
        {
            EstimatedVramMb = 256,
        };
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        using SessionLease lease = await pool.GetLeaseAsync(
            key, _ => Task.FromResult(CreateMinimalSession()), cts.Token);
        Assert.NotNull(lease.Session);
    }

    [Fact]
    public async Task ProcessGpuAdmission_UsageDrainingWhileWaiting_LetsAdmissionProceed()
    {
        // The wait path re-observes on every pass: once the process's real usage drains below the
        // budget, the parked admission proceeds instead of timing out. The reader instance is
        // shared so its read count, not the delegate, carries the draining state.
        var reader = new DrainingProcessGpuMemoryReader(4096L * 1024 * 1024);
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => reader);

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        using SessionLease lease = await pool.GetLeaseAsync(
            AcceleratorKey("pg5", 256), _ => Task.FromResult(CreateMinimalSession()), cts.Token);

        Assert.NotNull(lease.Session);
    }

    [Fact]
    public async Task ProcessGpuAdmission_PendingCreatesAreChargedOnTopOfTheObservation()
    {
        // A pending create holds a reservation but has not allocated yet, so the process reading
        // cannot contain it. Charging it on top of the observed floor keeps a second concurrent
        // admission from disappearing into the observation and overshooting the device budget.
        var reader = new MutableProcessGpuMemoryReader(5120L * 1024 * 1024);
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 8192,
            hostMemoryBudgetMb: 8192,
            processGpuMemoryReader: () => reader);

        // One idle pooled session: the reservation total is 3072 MB against the 5120 MB observed.
        SessionLease pooled = await pool.GetLeaseAsync(
            AcceleratorKey("pend0", 3072), _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        pooled.Dispose();

        // Creator A reserves 2048 MB but its factory has not allocated yet.
        var factoryGate = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
        Task<SessionLease> creatorA = pool.GetLeaseAsync(
            AcceleratorKey("pendA", 2048),
            async _ => { await factoryGate.Task; return CreateMinimalSession(); },
            CancellationToken.None);

        // Let A take its pending reservation before B decides, so the overlap is deterministic.
        await Task.Delay(TimeSpan.FromMilliseconds(150));

        // Creator B needs 2048 MB more: 5120 observed + 2048 pending + 2048 new = 9216 > 8192,
        // so it must wait rather than slip into the observation's shadow.
        int factoryCallsB = 0;
        Task<SessionLease> creatorB = pool.GetLeaseAsync(
            AcceleratorKey("pendB", 2048),
            _ =>
            {
                Interlocked.Increment(ref factoryCallsB);
                return Task.FromResult(CreateMinimalSession());
            },
            CancellationToken.None);

        await Task.Delay(TimeSpan.FromMilliseconds(300));
        Assert.Equal(0, Volatile.Read(ref factoryCallsB));

        // Once A's allocation lands and is published, B fits and proceeds. B re-polls on a 50 ms
        // cadence, so the generous bound only absorbs a starved thread pool on a loaded CI runner;
        // a genuine stall would still fail here.
        factoryGate.TrySetResult(true);
        using SessionLease leaseA = await creatorA.WaitAsync(TimeSpan.FromSeconds(30));
        using SessionLease leaseB = await creatorB.WaitAsync(TimeSpan.FromSeconds(30));

        Assert.NotNull(leaseA.Session);
        Assert.NotNull(leaseB.Session);
    }

    [Fact]
    public async Task ProcessGpuAdmission_EvictedMemoryDrainingWhileWaiting_LetsAdmissionProceed()
    {
        // One idle session plus unexplained observed usage: the wait evicts at most one idle
        // entry per pass and re-observes, so once the evicted memory drains the parked
        // admission proceeds instead of failing fast.
        var reader = new CountdownDrainingReader(8500L * 1024 * 1024, hotReads: 10);
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 8192,
            hostMemoryBudgetMb: 8192,
            processGpuMemoryReader: () => reader);

        SessionLease warm = await pool.GetLeaseAsync(
            AcceleratorKey("drain0", 3000), _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        warm.Dispose();

        int factoryCalls = 0;
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        using SessionLease lease = await pool.GetLeaseAsync(
            AcceleratorKey("drain1", 3000),
            _ => { Interlocked.Increment(ref factoryCalls); return Task.FromResult(CreateMinimalSession()); },
            cts.Token);

        Assert.NotNull(lease.Session);
        Assert.Equal(1, Volatile.Read(ref factoryCalls));
    }

    [Fact]
    public async Task ProcessGpuAdmission_LiveButUnevictableEntriesGetALongerBound()
    {
        // A held lease can still turn over (release, then evictable), so live-but-unevictable
        // entries wait on a longer bound than an empty bucket — but the wait stays bounded:
        // a lease held for a whole stage must not park the waiter until caller cancel.
        var reader = new MutableProcessGpuMemoryReader(0);
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => reader);

        SessionLease held = await pool.GetLeaseAsync(
            AcceleratorKey("leased0", 512), _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        // Non-pooled GPU memory arrives while the lease is held: the reservation total fits,
        // but the observation alone blocks the next admission.
        reader.Set(4096L * 1024 * 1024);

        int previousBound = InferenceSessionPool.ObservedBlockFailFastPasses;
        int previousBusyBound = InferenceSessionPool.ObservedBlockBusyFailFastPasses;
        InferenceSessionPool.ObservedBlockFailFastPasses = 100;
        InferenceSessionPool.ObservedBlockBusyFailFastPasses = 4;
        try
        {
            int factoryCalls = 0;
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            InvalidOperationException failure = await Assert.ThrowsAsync<InvalidOperationException>(() => pool.GetLeaseAsync(
                AcceleratorKey("leased1", 256),
                _ => { Interlocked.Increment(ref factoryCalls); return Task.FromResult(CreateMinimalSession()); },
                cts.Token));

            Assert.Contains("4096", failure.Message, StringComparison.Ordinal);
            Assert.Equal(0, Volatile.Read(ref factoryCalls));
        }
        finally
        {
            InferenceSessionPool.ObservedBlockFailFastPasses = previousBound;
            InferenceSessionPool.ObservedBlockBusyFailFastPasses = previousBusyBound;
        }

        held.Dispose();
    }

    [Fact]
    public async Task ProcessGpuAdmission_InFlightCreateAlone_GetsTheLongerBound()
    {
        // A slow factory in an otherwise-empty bucket is live work: its pending reservation
        // drains from the observation once it publishes. A concurrent waiter must not be told
        // the bucket is empty and fail on the short bound before that factory completes.
        var reader = new MutableProcessGpuMemoryReader(0);
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => reader);

        var factoryStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var releaseFactory = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        Task<SessionLease> slowCreate = pool.GetLeaseAsync(
            AcceleratorKey("inflight0", 256),
            async _ =>
            {
                factoryStarted.SetResult();
                await releaseFactory.Task;
                return CreateMinimalSession();
            },
            CancellationToken.None);
        await factoryStarted.Task.WaitAsync(TimeSpan.FromSeconds(5));

        reader.Set(4096L * 1024 * 1024);

        int previousBound = InferenceSessionPool.ObservedBlockFailFastPasses;
        int previousBusyBound = InferenceSessionPool.ObservedBlockBusyFailFastPasses;
        InferenceSessionPool.ObservedBlockFailFastPasses = 3;
        InferenceSessionPool.ObservedBlockBusyFailFastPasses = 100_000;
        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(600));
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pool.GetLeaseAsync(
                AcceleratorKey("inflight1", 256),
                _ => Task.FromResult(CreateMinimalSession()),
                cts.Token));
        }
        finally
        {
            InferenceSessionPool.ObservedBlockFailFastPasses = previousBound;
            InferenceSessionPool.ObservedBlockBusyFailFastPasses = previousBusyBound;
            releaseFactory.SetResult();
        }

        (await slowCreate).Dispose();
    }

    [Fact]
    public async Task ProcessGpuAdmission_UnfreeableObservation_FailsFastAfterEvictingIdleWork()
    {
        // When the observation alone blocks, each pass evicts at most one idle entry and
        // re-observes: evicted sessions get a chance to drain before more cache is discarded.
        // Once no idle work remains and the reading still blocks, the wait fails with a
        // diagnostic instead of parking until the caller cancels.
        var reader = new MutableProcessGpuMemoryReader(0);
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => reader);

        SessionLease warm1 = await pool.GetLeaseAsync(
            AcceleratorKey("stall1", 512), _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        warm1.Dispose();
        SessionLease warm2 = await pool.GetLeaseAsync(
            AcceleratorKey("stall2", 512), _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        warm2.Dispose();

        // Non-pooled GPU memory arrives after the warm cache is built: the reservation total
        // (1024 MB) fits, but the 4096 MB observation alone blocks the next 256 MB admission.
        reader.Set(4096L * 1024 * 1024);

        int previousBound = InferenceSessionPool.ObservedBlockFailFastPasses;
        InferenceSessionPool.ObservedBlockFailFastPasses = 3;
        try
        {
            InvalidOperationException failure = await Assert.ThrowsAsync<InvalidOperationException>(() => pool.GetLeaseAsync(
                AcceleratorKey("stall3", 256),
                _ => Task.FromResult(CreateMinimalSession()),
                CancellationToken.None));
            Assert.Contains("4096", failure.Message, StringComparison.Ordinal);
        }
        finally
        {
            InferenceSessionPool.ObservedBlockFailFastPasses = previousBound;
        }

        // Both idle entries were offered one per pass before the fail-fast: the first passes
        // evicted while idle work remained, and only the passes with nothing left to free
        // counted toward the bound.
        Assert.False(pool.TryPinExisting(AcceleratorKey("stall1", 512), out _));
        Assert.False(pool.TryPinExisting(AcceleratorKey("stall2", 512), out _));
    }

    [Fact]
    public async Task ProcessGpuAdmission_PerAdapterAttributionIsolatesDevices()
    {
        // With a per-adapter breakdown and a host LUID map, device 1's real usage is never
        // charged to device 0: no sibling subtraction, no cross-adapter blocking.
        var reader = new PerAdapterProcessGpuMemoryReader(
            totalBytes: 5000L * 1024 * 1024,
            byLuidBytes: new Dictionary<long, long>
            {
                [100] = 1500L * 1024 * 1024,
                [200] = 3500L * 1024 * 1024,
            });
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => reader);

        IReadOnlyDictionary<int, long>? previous = SharedPoolOptions.AdapterLuidMap;
        SharedPoolOptions.UseAdapterLuidMap(new Dictionary<int, long> { [0] = 100, [1] = 200 });
        try
        {
            // 1500 MB on this adapter + 256 MB need fits the 4096 MB budget, even though the
            // process total (5000 MB) would block under the fallback accounting.
            using SessionLease lease = await pool.GetLeaseAsync(
                AcceleratorKey("pa1", 256, deviceId: 0),
                _ => Task.FromResult(CreateMinimalSession()),
                CancellationToken.None);

            Assert.NotNull(lease.Session);
        }
        finally
        {
            SharedPoolOptions.UseAdapterLuidMap(previous);
        }
    }

    [Fact]
    public async Task ProcessGpuAdmission_PerAdapter_MappedAdapterWithNoUsage_IsChargedNothing()
    {
        // The breakdown is complete when present, so a mapped adapter with no entry holds nothing:
        // the first admission on a fresh second GPU must not be charged the other adapter's usage.
        var reader = new PerAdapterProcessGpuMemoryReader(
            totalBytes: 3900L * 1024 * 1024,
            byLuidBytes: new Dictionary<long, long> { [100] = 3900L * 1024 * 1024 });
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => reader);

        IReadOnlyDictionary<int, long>? previous = SharedPoolOptions.AdapterLuidMap;
        SharedPoolOptions.UseAdapterLuidMap(new Dictionary<int, long> { [0] = 100, [1] = 200 });
        try
        {
            using SessionLease lease = await pool.GetLeaseAsync(
                AcceleratorKey("pa0", 256, deviceId: 1),
                _ => Task.FromResult(CreateMinimalSession()),
                CancellationToken.None);

            Assert.NotNull(lease.Session);
        }
        finally
        {
            SharedPoolOptions.UseAdapterLuidMap(previous);
        }
    }

    [Fact]
    public async Task ProcessGpuAdmission_ExternalTurnoverAfterBusyPasses_DoesNotTripEmptyBucketFailFast()
    {
        // The busy bound waits for live-but-unevictable entries to turn over. When that
        // turnover happens (an idle external evicted through its callback), the stall counter
        // must reset: otherwise the bucket empties, the bound drops to the short fail-fast,
        // and a waiter parked through the busy phase throws before the freed memory has had a
        // single re-observe to drain.
        var reader = new MutableDrainGpuReader();
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => reader);

        using ExternalMemoryReservation external = await pool.ReserveExternalAsync(new(ExecutionProviderKind.DirectMl, null, 500), CancellationToken.None);
        reader.Set(4096L * 1024 * 1024);

        int previousBound = InferenceSessionPool.ObservedBlockFailFastPasses;
        int previousBusyBound = InferenceSessionPool.ObservedBlockBusyFailFastPasses;
        InferenceSessionPool.ObservedBlockFailFastPasses = 3;
        InferenceSessionPool.ObservedBlockBusyFailFastPasses = 20;
        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
            Task<ExternalMemoryReservation> waiter = pool.ReserveExternalAsync(new(ExecutionProviderKind.DirectMl, null, 256), cts.Token);

            // Busy passes accumulate past the short bound while the external stays live (no
            // eviction callback yet): the reservations fit, the observation alone blocks.
            await Task.Delay(TimeSpan.FromMilliseconds(400));

            // Register the idle-eviction callback. The first pass that offers the external
            // evicts it; the reading stays hot for two more reads, so the freshly-emptied
            // bucket is still observation-blocked — the reset stall counter is what lets the
            // waiter survive that pass and admit once the memory drains.
            var evicted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            Assert.True(external.TrySetIdleEvictionCallback(() =>
            {
                reader.ArmDrain(2);
                evicted.TrySetResult();
                return true;
            }));

            await evicted.Task.WaitAsync(TimeSpan.FromSeconds(5));
            using ExternalMemoryReservation admitted = await waiter.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.NotNull(admitted);
        }
        finally
        {
            InferenceSessionPool.ObservedBlockFailFastPasses = previousBound;
            InferenceSessionPool.ObservedBlockBusyFailFastPasses = previousBusyBound;
        }
    }

    [Fact]
    public async Task ProcessGpuAdmission_MissingMapFallsBackToProcessTotal()
    {
        // Without a host LUID map the pool keeps the conservative process-total accounting:
        // the 5000 MB total blocks a 256 MB admission on a 4096 MB budget.
        var reader = new PerAdapterProcessGpuMemoryReader(
            totalBytes: 5000L * 1024 * 1024,
            byLuidBytes: new Dictionary<long, long> { [100] = 1500L * 1024 * 1024 });
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => reader);

        IReadOnlyDictionary<int, long>? previous = SharedPoolOptions.AdapterLuidMap;
        SharedPoolOptions.UseAdapterLuidMap(null);
        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(250));
            int factoryCalls = 0;
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pool.GetLeaseAsync(
                AcceleratorKey("pa2", 256, deviceId: 0),
                _ => { factoryCalls++; return Task.FromResult(CreateMinimalSession()); },
                cts.Token));

            Assert.Equal(0, factoryCalls);
        }
        finally
        {
            SharedPoolOptions.UseAdapterLuidMap(previous);
        }
    }

    [Fact]
    public async Task ProcessGpuAdmission_UnattributedBreakdownFallsBackToProcessTotal()
    {
        // A reader that cannot attribute per adapter (null breakdown) degrades to the same
        // fallback even when the host registered a map.
        var reader = new PerAdapterProcessGpuMemoryReader(totalBytes: 5000L * 1024 * 1024, byLuidBytes: null);
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => reader);

        IReadOnlyDictionary<int, long>? previous = SharedPoolOptions.AdapterLuidMap;
        SharedPoolOptions.UseAdapterLuidMap(new Dictionary<int, long> { [0] = 100 });
        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(250));
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pool.GetLeaseAsync(
                AcceleratorKey("pa3", 256, deviceId: 0),
                _ => Task.FromResult(CreateMinimalSession()),
                cts.Token));
        }
        finally
        {
            SharedPoolOptions.UseAdapterLuidMap(previous);
        }
    }

    private sealed class PerAdapterProcessGpuMemoryReader(long? totalBytes, IReadOnlyDictionary<long, long>? byLuidBytes)
        : IProcessGpuMemoryReader
    {
        public string UnavailableReason => "Test double: scripted per-adapter reading.";

        public long? ReadDedicatedGpuMemoryBytes() => totalBytes;

        public IReadOnlyDictionary<long, long>? ReadDedicatedGpuMemoryBytesByAdapterLuid() => byLuidBytes;
    }

    [Fact]
    public async Task ProcessGpuAdmission_UnavailableReading_LeavesAdmissionUnchanged()
    {
        // A reader that cannot report (no GPU, a driver without the counter set, a GPU-idle
        // process) must neither tighten nor loosen admission: the reservation model stands alone.
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => new StubProcessGpuMemoryReader(null));

        using SessionLease lease = await pool.GetLeaseAsync(
            AcceleratorKey("pg6", 4000), _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        Assert.NotNull(lease.Session);
    }

    [Theory]
    [InlineData(-1L)]
    [InlineData(0L)]
    public async Task ProcessGpuAdmission_NonPositiveReading_LeavesAdmissionUnchanged(long bytes)
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => new StubProcessGpuMemoryReader(bytes));

        using SessionLease lease = await pool.GetLeaseAsync(
            AcceleratorKey("pg7", 4000), _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        Assert.NotNull(lease.Session);
    }

    [Fact]
    public async Task ProcessGpuAdmission_FailingProbe_DoesNotBreakAdmission()
    {
        // A throwing probe must degrade the observation, never the admission path.
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: () => new ThrowingProcessGpuMemoryReader());

        using SessionLease lease = await pool.GetLeaseAsync(
            AcceleratorKey("pg8", 4000), _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        Assert.NotNull(lease.Session);
    }

    [Fact]
    public async Task ProcessGpuAdmission_NoRegisteredReader_LeavesAdmissionUnchanged()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8,
            memoryBudgetMb: 4096,
            hostMemoryBudgetMb: 4096,
            processGpuMemoryReader: static () => null);

        using SessionLease lease = await pool.GetLeaseAsync(
            AcceleratorKey("pg9", 4000), _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        Assert.NotNull(lease.Session);
    }

    /// <summary>Deterministic stand-in for the host's process-isolated GPU reading.</summary>
    private sealed class StubProcessGpuMemoryReader(long? bytes) : IProcessGpuMemoryReader
    {
        public string UnavailableReason => "Test double: no reading configured.";

        public long? ReadDedicatedGpuMemoryBytes() => bytes;
    }

    /// <summary>Stand-in whose reading follows the host's real usage over time.</summary>
    private sealed class MutableProcessGpuMemoryReader(long? bytes) : IProcessGpuMemoryReader
    {
        // -1 stands for "no reading", so the field can stay a Volatile-friendly primitive.
        private long current = bytes ?? -1;

        public string UnavailableReason => "Test double: no reading configured.";

        public void Set(long? value) => Volatile.Write(ref current, value ?? -1);

        public long? ReadDedicatedGpuMemoryBytes()
        {
            long value = Volatile.Read(ref current);
            return value < 0 ? null : value;
        }
    }

    /// <summary>Reports one hot reading, then drains — stands in for freed GPU memory.</summary>
    private sealed class DrainingProcessGpuMemoryReader(long firstBytes) : IProcessGpuMemoryReader
    {
        private int reads;

        public string UnavailableReason => "Test double: one hot reading, then drained.";

        public long? ReadDedicatedGpuMemoryBytes() =>
            Interlocked.Increment(ref reads) == 1 ? firstBytes : 0;
    }

    /// <summary>Reports a hot reading for a fixed number of reads, then drains.</summary>
    private sealed class CountdownDrainingReader(long hotBytes, int hotReads) : IProcessGpuMemoryReader
    {
        private int reads;

        public string UnavailableReason => "Test double: hot readings, then drained.";

        public long? ReadDedicatedGpuMemoryBytes() =>
            Interlocked.Increment(ref reads) <= hotReads ? hotBytes : 0;
    }

    /// <summary>
    /// No GPU floor until <see cref="MarkCreated"/> (the first bundle graph's factory),
    /// then a hot floor for a fixed number of reads — the created sibling counting
    /// against admission — then a drained floor.
    /// </summary>
    private sealed class FloorAfterFirstCreateReader(long hotMb, long drainedMb, int hotReads) : IProcessGpuMemoryReader
    {
        private int created;
        private int hotReadsRemaining = hotReads;

        public string UnavailableReason => "Test double: floor appears after first create.";

        public void MarkCreated() => Volatile.Write(ref created, 1);

        public long? ReadDedicatedGpuMemoryBytes()
        {
            if (Volatile.Read(ref created) == 0)
            {
                return null;
            }

            long mb = Interlocked.Decrement(ref hotReadsRemaining) >= 0 ? hotMb : drainedMb;
            return mb * 1024 * 1024;
        }
    }

    /// <summary>
    /// Stand-in whose reading stays hot until <see cref="ArmDrain"/> counts the remaining hot
    /// reads, then reports 0 — models freed GPU memory that takes a couple of re-observes to
    /// drain after an external is evicted.
    /// </summary>
    private sealed class MutableDrainGpuReader : IProcessGpuMemoryReader
    {
        private long current = -1;
        private int hotReads = int.MaxValue;

        public string UnavailableReason => "Test double: draining after armed reads.";

        public void Set(long? value) => Volatile.Write(ref current, value ?? -1);

        public void ArmDrain(int reads) => Volatile.Write(ref hotReads, reads);

        public long? ReadDedicatedGpuMemoryBytes()
        {
            if (Interlocked.Decrement(ref hotReads) < 0)
            {
                return 0;
            }

            long value = Volatile.Read(ref current);
            return value < 0 ? null : value;
        }
    }

    private sealed class ThrowingProcessGpuMemoryReader : IProcessGpuMemoryReader
    {
        public string UnavailableReason => "Test double: probe always fails.";

        public long? ReadDedicatedGpuMemoryBytes() =>
            throw new InvalidOperationException("probe exploded.");
    }

    // ── SharedPoolOptions (production activation contract) ────────────────────

    [Theory]
    [InlineData("1", true)]
    [InlineData("true", true)]
    [InlineData("TRUE", true)]
    [InlineData("on", true)]
    [InlineData("enabled", true)]
    [InlineData(" 1 ", true)]
    [InlineData("0", false)]
    [InlineData("false", false)]
    [InlineData("FALSE", false)]
    [InlineData("off", false)]
    [InlineData("disabled", false)]
    [InlineData("", true)]
    [InlineData("yes", true)]
    [InlineData("2", true)]
    [InlineData("typpo", true)]
    public void SharedPoolOptions_AdmissionFlag_OnlyExplicitNegativeDisables(string raw, bool expected)
    {
        // Admission is the safe default: unset, unknown, and typo values keep it enabled;
        // only an explicit negative token turns it off.
        Assert.Equal(expected, SharedPoolOptions.ParseAdmissionFlag(raw));
    }

    [Fact]
    public void SharedPoolOptions_AdmissionFlag_UnsetStaysOn()
    {
        using var env = new ScopedEnvironment();
        env.Clear(SharedPoolOptions.AdmissionVariable);

        Assert.True(SharedPoolOptions.ReadAdmissionFlag(SharedPoolOptions.AdmissionVariable));
    }

    [Theory]
    [InlineData(null, null)]
    [InlineData("", null)]
    [InlineData("   ", null)]
    [InlineData("4", 4)]
    [InlineData(" 20 ", 20)]
    [InlineData("0", null)]
    [InlineData("-1", null)]
    [InlineData("abc", null)]
    [InlineData("12.5", null)]
    [InlineData("3000000000", null)]            // exceeds int.MaxValue — falls back
    [InlineData("9223372036854775807", null)]   // long.MaxValue — falls back, never throws
    public void SharedPoolOptions_PositiveInt32_RejectsUnsetInvalidAndOverflow(string? raw, int? expected)
    {
        Assert.Equal(expected, SharedPoolOptions.ParsePositiveInt32(raw));
    }

    [Theory]
    [InlineData(null, null)]
    [InlineData("", null)]
    [InlineData("   ", null)]
    [InlineData("4096", 4096L)]
    [InlineData(" 8192 ", 8192L)]
    [InlineData("0", null)]
    [InlineData("-1", null)]
    [InlineData("abc", null)]
    [InlineData("12.5", null)]
    public void SharedPoolOptions_PositiveInt64_RejectsUnsetAndInvalid(string? raw, long? expected)
    {
        Assert.Equal(expected, SharedPoolOptions.ParsePositiveInt64(raw));
    }

    [Fact]
    public void SharedPoolOptions_BudgetAndCapacity_AreReadFromEnvironment()
    {
        using var env = new ScopedEnvironment();
        env.Set(SharedPoolOptions.BudgetMbVariable, "8192");
        env.Set(SharedPoolOptions.HostBudgetMbVariable, "2048");
        env.Set(SharedPoolOptions.MaxSessionsVariable, "20");
        env.Set(SharedPoolOptions.AdmissionVariable, "0");

        Assert.False(SharedPoolOptions.ReadAdmissionFlag(SharedPoolOptions.AdmissionVariable));
        Assert.Equal(8192L, SharedPoolOptions.ReadPositiveInt64(SharedPoolOptions.BudgetMbVariable));
        Assert.Equal(2048L, SharedPoolOptions.ReadPositiveInt64(SharedPoolOptions.HostBudgetMbVariable));
        Assert.Equal(20, SharedPoolOptions.ReadPositiveInt32(SharedPoolOptions.MaxSessionsVariable));
    }

    [Fact]
    public void SharedPoolOptions_ProcessGpuAdmission_IsOptOutOnly()
    {
        // Process-isolated GPU admission is on by default and only an explicit negative token
        // turns it off. This test exercises the pure readers against an isolated environment;
        // it deliberately does not assert SharedPoolOptions.EnableProcessGpuAdmission, which is
        // captured during type initialization and cannot be re-read — the admission tests above
        // are what depend on that cached default staying on, so they fail if anything flips it.
        using var env = new ScopedEnvironment();
        env.Clear(SharedPoolOptions.ProcessGpuAdmissionVariable);

        Assert.Equal("TRACKDUB_SESSION_PROCESS_GPU_ADMISSION", SharedPoolOptions.ProcessGpuAdmissionVariable);
        // Startup-cached configuration cannot be reset by changing this process environment.
        Assert.True(SharedPoolOptions.ReadAdmissionFlag(SharedPoolOptions.ProcessGpuAdmissionVariable));
        Assert.False(SharedPoolOptions.ParseAdmissionFlag("0"));
        Assert.False(SharedPoolOptions.ParseAdmissionFlag("false"));
        Assert.False(SharedPoolOptions.ParseAdmissionFlag("off"));
        Assert.False(SharedPoolOptions.ParseAdmissionFlag("disabled"));
        Assert.True(SharedPoolOptions.ParseAdmissionFlag("typpo"));
    }

    [Fact]
    public void SharedPoolOptions_UseProcessGpuMemoryReader_RoundTripsAndClears()
    {
        var reader = new StubProcessGpuMemoryReader(1024);
        try
        {
            SharedPoolOptions.UseProcessGpuMemoryReader(reader);
            Assert.Same(reader, SharedPoolOptions.ProcessGpuMemoryReader);
        }
        finally
        {
            SharedPoolOptions.UseProcessGpuMemoryReader(null);
        }

        Assert.Null(SharedPoolOptions.ProcessGpuMemoryReader);
    }

    [Fact]
    public void SharedPoolOptions_Defaults_AreSafeOnAndMatchPoolDefaults()
    {
        // With no environment set, admission is hard-on with the pool's documented
        // budgets and capacity.
        using var env = new ScopedEnvironment();
        env.Clear(SharedPoolOptions.AdmissionVariable);
        env.Clear(SharedPoolOptions.BudgetMbVariable);
        env.Clear(SharedPoolOptions.HostBudgetMbVariable);
        env.Clear(SharedPoolOptions.MaxSessionsVariable);

        Assert.True(SharedPoolOptions.ReadAdmissionFlag(SharedPoolOptions.AdmissionVariable));
        Assert.Null(SharedPoolOptions.ReadPositiveInt64(SharedPoolOptions.BudgetMbVariable));
        Assert.Null(SharedPoolOptions.ReadPositiveInt64(SharedPoolOptions.HostBudgetMbVariable));
        Assert.Null(SharedPoolOptions.ReadPositiveInt32(SharedPoolOptions.MaxSessionsVariable));
        Assert.Equal(12, InferenceSessionPool.DefaultMaxSessions);
        // Accelerator default scales with detected GPU VRAM (floor 4GB, 75% up to
        // 16GB cards, 85% above) so it is machine-dependent; it must stay within
        // the documented band.
        Assert.InRange(
            InferenceSessionPool.DefaultMemoryBudgetMb,
            InferenceSessionPool.AcceleratorBudgetFloorMb,
            InferenceSessionPool.AcceleratorBudgetCapMb);
        // Host default scales with physical RAM (floor 4GB, cap 16GB) so it is
        // machine-dependent; it must stay within the documented band.
        Assert.InRange(
            InferenceSessionPool.DefaultHostMemoryBudgetMb,
            InferenceSessionPool.HostMemoryBudgetFloorMb,
            InferenceSessionPool.HostMemoryBudgetCapMb);
    }

    [Theory]
    [InlineData(8L * 1024, 4096L)]
    [InlineData(16L * 1024, 4096L)]
    [InlineData(24L * 1024, 6144L)]
    [InlineData(32L * 1024, 8192L)]
    [InlineData(64L * 1024, 16384L)]
    [InlineData(128L * 1024, 16384L)]
    public void ScaleHostMemoryBudgetMb_QuarterRam_Floor4Gb_Cap16Gb(long totalRamMb, long expectedMb)
    {
        Assert.Equal(expectedMb, InferenceSessionPool.ScaleHostMemoryBudgetMb(totalRamMb));
    }

    [Theory]
    [InlineData(0L, 4096L)]
    [InlineData(2048L, 4096L)]
    [InlineData(6144L, 4608L)]
    [InlineData(8192L, 6144L)]
    [InlineData(10240L, 7680L)]
    [InlineData(12288L, 9216L)]
    [InlineData(16384L, 12288L)]
    // Above 16 GB, large-memory cards (consumer RTX 4090/5090 as well as
    // L4 24 GB, L40S 48 GB) have the headroom to keep 85% instead of 75% so
    // the admission budget does not strand a third of the device. The 16 GB
    // hard cap is gone for these.
    [InlineData(24576L, 20889L)]
    [InlineData(49152L, 41779L)]
    public void ScaleAcceleratorBudgetMb_ScalesWithVram_LargeMemoryCardsKeep85Percent(long maxVramMb, long expectedMb)
    {
        Assert.Equal(expectedMb, InferenceSessionPool.ScaleAcceleratorBudgetMb(maxVramMb));
    }

    [Fact]
    public void DefaultMemoryBudgetMb_MatchesScaledDetectedVram()
    {
        // The process default must equal the scale function applied to this
        // machine's largest GPU: MADLAD-class bundles (~6.4GB) are admissible by
        // default on 10GB+ GPUs and stay guarded on small ones.
        Assert.Equal(
            InferenceSessionPool.ScaleAcceleratorBudgetMb(InferenceSessionPool.DetectMaxAcceleratorVramMb()),
            InferenceSessionPool.DefaultMemoryBudgetMb);
    }

    [Theory]
    [InlineData("Model: NVIDIA GeForce RTX 5070\nVideo Memory:      12288 MB\n", 12288L)]
    [InlineData("Video Memory: 8192MB\n", 8192L)]
    [InlineData("Video Memory: 16384 MiB\n", 16384L)]
    [InlineData("Model: Something\nNoMemoryHere\n", 0L)]
    [InlineData("", 0L)]
    public void ParseNvidiaInformationMb_ReadsProcDriverFormat(string text, long expectedMb)
    {
        Assert.Equal(expectedMb, AcceleratorVramProbe.ParseNvidiaInformationMb(text));
    }

    [Fact]
    public void AcceleratorBudgetInfo_NoOverride_ReportsDetectedSource()
    {
        using var env = new ScopedEnvironment();
        env.Clear(SharedPoolOptions.BudgetMbVariable);

        AcceleratorBudgetInfo info = new AcceleratorBudgetInfoService().GetBudgetInfo();

        Assert.Null(info.OverrideBudgetMb);
        Assert.Equal(InferenceSessionPool.DetectMaxAcceleratorVramMb(), info.DetectedVramMb);
        Assert.Equal(InferenceSessionPool.DefaultMemoryBudgetMb, info.DefaultBudgetMb);
        Assert.Equal(info.DefaultBudgetMb, info.EffectiveBudgetMb);
        Assert.Equal(
            info.DetectedVramMb > 0 ? "detected VRAM" : "floor fallback (no GPU VRAM detected)",
            info.Source);
    }

    [Fact]
    public void AcceleratorBudgetInfo_WithOverride_OverrideWins()
    {
        using var env = new ScopedEnvironment();
        env.Set(SharedPoolOptions.BudgetMbVariable, "12345");

        AcceleratorBudgetInfo info = new AcceleratorBudgetInfoService().GetBudgetInfo();

        Assert.Equal(12345, info.OverrideBudgetMb);
        Assert.Equal(12345, info.EffectiveBudgetMb);
        Assert.Equal("override (TRACKDUB_SESSION_VRAM_BUDGET_MB)", info.Source);
    }

    [Theory]
    [InlineData(12UL * 1024 * 1024 * 1024, 12288L)]
    [InlineData(48UL * 1024 * 1024 * 1024, 49152L)]
    [InlineData(0UL, 0L)]
    public void MbFromNvmlMemoryInfo_ConvertsTotalBytesToMb(ulong totalBytes, long expectedMb)
    {
        Assert.Equal(expectedMb, AcceleratorVramProbe.MbFromNvmlMemoryInfo(totalBytes));
    }

    [Fact]
    public void QueryNvmlMaxDedicatedVramMb_ReturnsZeroOffLinux()
    {
        // The NVML fallback only applies on Linux (containers/EC2). Off-Linux it
        // must report unknown (0) without touching native libraries.
        if (OperatingSystem.IsLinux())
        {
            return;
        }

        Assert.Equal(0, AcceleratorVramProbe.QueryNvmlMaxDedicatedVramMb());
    }

    [Fact]
    public void DecodeNvmlString_StripsAtFirstNulAndTrims()
    {
        byte[] buffer = new byte[64];
        byte[] name = System.Text.Encoding.UTF8.GetBytes("  NVIDIA GeForce RTX 5070  ");
        name.CopyTo(buffer, 0);
        buffer[name.Length] = 0;
        System.Text.Encoding.UTF8.GetBytes("trailing-garbage").CopyTo(buffer, name.Length + 1);

        Assert.Equal("NVIDIA GeForce RTX 5070", AcceleratorVramProbe.DecodeNvmlString(buffer));
    }

    [Fact]
    public void DecodeNvmlString_EmptyOrAllNul_ReturnsNull()
    {
        Assert.Null(AcceleratorVramProbe.DecodeNvmlString([]));
        Assert.Null(AcceleratorVramProbe.DecodeNvmlString(new byte[16]));
        Assert.Null(AcceleratorVramProbe.DecodeNvmlString(null));
    }

    [Fact]
    public void DecodeNvmlString_WhitespaceOnly_ReturnsNull()
    {
        byte[] buffer = System.Text.Encoding.UTF8.GetBytes("   \0");
        Assert.Null(AcceleratorVramProbe.DecodeNvmlString(buffer));
    }

    [Fact]
    public void QueryNvmlGpuIdentity_ReturnsNullsOffLinux()
    {
        if (OperatingSystem.IsLinux())
        {
            return;
        }

        (string? deviceName, string? driverVersion) = AcceleratorVramProbe.QueryNvmlGpuIdentity();
        Assert.Null(deviceName);
        Assert.Null(driverVersion);
    }

    [Theory]
    [InlineData("8589934592", 8192L)]
    [InlineData("  12884901888\n", 12288L)]
    [InlineData(null, 0L)]
    [InlineData("", 0L)]
    [InlineData("not-a-number", 0L)]
    [InlineData("0", 0L)]
    public void ParseAmdVramBytes_ConvertsSysfsBytesToMb(string? bytesText, long expectedMb)
    {
        Assert.Equal(expectedMb, AcceleratorVramProbe.ParseAmdVramBytes(bytesText));
    }

    [Theory]
    [InlineData("0x10de", 0x10de)]
    [InlineData("0X1002", 0x1002)]
    [InlineData("030200", 0x030200)]
    public void TryParseHex_ParsesSysfsHexFields(string text, uint expected)
    {
        Assert.True(AcceleratorVramProbe.TryParseHex(text, out uint value));
        Assert.Equal(expected, value);
    }

    [Fact]
    public void TryParseHex_RejectsNonHex()
    {
        Assert.False(AcceleratorVramProbe.TryParseHex("xyz", out _));
    }

    [Fact]
    public void QueryLinuxMaxDedicatedVramMb_NeverReturnsNegative()
    {
        // OS-dependent live probe: asserts only the 0-as-unknown contract.
        Assert.True(AcceleratorVramProbe.QueryLinuxMaxDedicatedVramMb() >= 0);
    }

    [Fact]
    public void DefaultHostMemoryBudgetMb_MatchesScaledPhysicalRam()
    {
        // The process default must equal the scale function applied to this
        // machine's RAM: MADLAD-class bundles (~6.4GB) are admissible by default
        // on 32GB+ machines and stay guarded on small ones.
        long totalRamMb = (long)(GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / (1024 * 1024));
        Assert.Equal(
            InferenceSessionPool.ScaleHostMemoryBudgetMb(totalRamMb),
            InferenceSessionPool.DefaultHostMemoryBudgetMb);
    }

    [Fact]
    public void Shared_IsStable_AcrossAccesses()
    {
        // Shared is now lazily built; every consumer must observe the same instance.
        Assert.Same(InferenceSessionPool.Shared, InferenceSessionPool.Shared);
    }

    /// <summary>
    /// Sets an environment variable for the duration of a test and restores it afterwards,
    /// so option parsing can be exercised without leaking state across the test run.
    /// </summary>
    private sealed class ScopedEnvironment : IDisposable
    {
        private readonly Dictionary<string, string?> original = new();

        public void Set(string variable, string? value) => SetCore(variable, value);

        public void Clear(string variable) => SetCore(variable, null);

        public void Dispose()
        {
            foreach ((string variable, string? value) in original)
            {
                Environment.SetEnvironmentVariable(variable, value);
            }
        }

        private void SetCore(string variable, string? value)
        {
            if (!original.ContainsKey(variable))
            {
                original[variable] = Environment.GetEnvironmentVariable(variable);
            }

            Environment.SetEnvironmentVariable(variable, value);
        }
    }

    // ── Hard admission defaults & host/accelerator buckets ──────────────────

    private static SessionPoolKey AcceleratorKey(string hash, long mb, int deviceId = 0, ExecutionProviderKind provider = ExecutionProviderKind.DirectMl) =>
        new("eng", null, null, provider, hash, deviceId, "default") { EstimatedVramMb = mb };

    private static SessionPoolKey HostKey(string hash, long mb, ExecutionProviderKind provider = ExecutionProviderKind.Cpu) =>
        new("eng", null, null, provider, hash, null, "default") { EstimatedVramMb = mb };

    [Fact]
    public async Task DefaultConstructor_HardAdmission_BlocksAcceleratorBeyondDefaultBudget()
    {
        // Defaults: admission on, a hardware-scaled accelerator budget per device. A second DML
        // graph that cannot fit must wait — never fall back to an ephemeral session.
        using var pool = new InferenceSessionPool();
        var key1 = AcceleratorKey("ha1", InferenceSessionPool.DefaultMemoryBudgetMb);
        var key2 = AcceleratorKey("ha2", 1);
        int key2FactoryCalls = 0;

        using SessionLease lease1 = await pool.GetLeaseAsync(
            key1, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(250));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pool.GetLeaseAsync(
            key2,
            _ => { key2FactoryCalls++; return Task.FromResult(CreateMinimalSession()); },
            cts.Token));

        Assert.Equal(0, key2FactoryCalls);
    }

    [Fact]
    public async Task AdmissionBuckets_HostRamAndAcceleratorDevice0_AreSeparate()
    {
        // CPU sessions no longer collide with accelerator device 0: each has its own budget.
        using var pool = new InferenceSessionPool(
            maxSessions: 8, memoryBudgetMb: 4096, hostMemoryBudgetMb: 4096);
        var cpuKey = HostKey("hs1", 3000);
        var dmlKey = AcceleratorKey("hs2", 3000);

        // Host RAM held does not block accelerator admission.
        using (SessionLease cpuLease = await pool.GetLeaseAsync(
                   cpuKey, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None))
        using (SessionLease dmlLease = await pool.GetLeaseAsync(
                   dmlKey, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None))
        {
            Assert.NotNull(dmlLease.Session);
        }

        // Accelerator device 0 held does not block host admission either.
        using SessionLease heldDml = await pool.GetLeaseAsync(
            AcceleratorKey("hs3", 3000), _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        using SessionLease cpu2 = await pool.GetLeaseAsync(
            HostKey("hs4", 3000), _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        Assert.NotNull(cpu2.Session);
    }

    [Fact]
    public async Task AdmissionBuckets_CpuAndDnnl_ShareHostRamBudget()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, hostMemoryBudgetMb: 100);
        var cpuKey = HostKey("hc1", 80, ExecutionProviderKind.Cpu);
        var dnnlKey = HostKey("hc2", 80, ExecutionProviderKind.Dnnl);

        using SessionLease lease1 = await pool.GetLeaseAsync(
            cpuKey, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        // CPU and DNNL draw from the same host RAM budget: 80 + 80 > 100 waits.
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(250));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.GetLeaseAsync(dnnlKey, _ => Task.FromResult(CreateMinimalSession()), cts.Token));

        // After release the DNNL session fits in the freed host budget.
        lease1.Dispose();
        using SessionLease lease2 = await pool.GetLeaseAsync(
            dnnlKey, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        Assert.NotNull(lease2.Session);
    }

    [Fact]
    public async Task AdmissionBuckets_OpenVinoNpuDoesNotCollideWithGpuDeviceZero()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, memoryBudgetMb: 100, hostMemoryBudgetMb: 100);
        var npuKey = new SessionPoolKey("eng", null, null, ExecutionProviderKind.OpenVino, "npu", null, "default")
        {
            EstimatedVramMb = 80,
        };
        var gpuKey = AcceleratorKey("gpu", 80, deviceId: 0);

        using SessionLease npuLease = await pool.GetLeaseAsync(
            npuKey, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        using SessionLease gpuLease = await pool.GetLeaseAsync(
            gpuKey, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);

        Assert.NotNull(gpuLease.Session);
    }

    [Fact]
    public async Task AdmissionBuckets_OpenVinoCpuProxySharesHostBudget()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, memoryBudgetMb: 100, hostMemoryBudgetMb: 100);
        var cpuKey = HostKey("proxy-cpu", 80);
        var openVinoProxyKey = new SessionPoolKey(
            "eng", null, null, ExecutionProviderKind.OpenVino, "proxy-openvino", null, "default")
        {
            EstimatedVramMb = 80,
            UseOpenVinoCpuProxy = true,
        };

        using SessionLease cpuLease = await pool.GetLeaseAsync(
            cpuKey, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None);
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(250));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pool.GetLeaseAsync(
            openVinoProxyKey,
            _ => Task.FromResult(CreateMinimalSession()),
            cts.Token));
    }

    // ── Bundle hard admission ───────────────────────────────────────────────

    [Fact]
    public async Task GetLeaseBundleAsync_AggregateOversize_FailsBeforeAnyFactory()
    {
        using var pool = new InferenceSessionPool();
        var big1 = AcceleratorKey("bo1", InferenceSessionPool.DefaultMemoryBudgetMb);
        var big2 = AcceleratorKey("bo2", 1);
        int factoryCalls = 0;

        Task<InferenceSession> Factory(CancellationToken _)
        {
            factoryCalls++;
            return Task.FromResult(CreateMinimalSession());
        }

        // The combined estimates exceed the hardware-scaled device-0 budget by 1 MB.
        InvalidOperationException ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => pool.GetLeaseBundleAsync(
                [new SessionLeaseRequest(big1, Factory), new SessionLeaseRequest(big2, Factory)],
                CancellationToken.None));

        Assert.Equal(0, factoryCalls);
        Assert.Contains("accelerator device 0", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task GetLeaseBundleAsync_PreparationFailure_ReleasesPins()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, hostMemoryBudgetMb: 100);
        // Sorted order puts "aaaa" first: its residency pin is held when the second
        // member's factory fails. Cleanup must release the pin so idle eviction can
        // reclaim the host budget it held.
        var pinned = HostKey("aaaa", 80);
        var failing = HostKey("bbbb", 10);
        var probe = HostKey("cccc", 80);
        int probeCalls = 0;

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => pool.GetLeaseBundleAsync(
                [new SessionLeaseRequest(pinned, _ => Task.FromResult(CreateMinimalSession())),
                 new SessionLeaseRequest(failing,
                     _ => Task.FromException<InferenceSession>(new InvalidOperationException("synthetic create failure")))],
                CancellationToken.None));

        // The probe fits the 100MB host budget only if the failed preparation released
        // the pinned 80MB entry, letting idle eviction make room for it.
        using SessionLease lease = await pool.GetLeaseAsync(
            probe,
            _ => { probeCalls++; return Task.FromResult(CreateMinimalSession()); },
            CancellationToken.None)
            .WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(1, probeCalls);
        Assert.NotNull(lease.Session);
    }

    [Fact]
    public async Task GetLeaseBundleAsync_CountMode_CancellationDuringPreparation_DoesNotHang()
    {
        // Count mode never publishes an ephemeral warm, so residency-style preparation
        // would spin forever; the bundle must stay on the bounded warm path and honour
        // the caller's cancellation token.
        using var pool = new InferenceSessionPool(
            maxSessions: 1, enableMemoryAdmission: false);
        SessionLeaseRequest a = new(AcceleratorKey("cm1", 64), _ => Task.FromResult(CreateMinimalSession()));
        SessionLeaseRequest b = new(AcceleratorKey("cm2", 64), _ => Task.FromResult(CreateMinimalSession()));

        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(500));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.GetLeaseBundleAsync([a, b], cts.Token));
    }

    // ── External memory reservations ─────────────────────────────────────────

    [Fact]
    public async Task ReserveExternalAsync_HostAndAcceleratorUseSeparateBudgets()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, memoryBudgetMb: 100, hostMemoryBudgetMb: 100);

        using ExternalMemoryReservation host = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), CancellationToken.None);
        using ExternalMemoryReservation accel = await pool.ReserveExternalAsync(new(ExecutionProviderKind.DirectMl, 0, 80), CancellationToken.None);

        Assert.NotNull(host);
        Assert.NotNull(accel);
    }

    [Fact]
    public async Task ReserveExternalAsync_OpenVinoCpuProxySharesHostBudget()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, memoryBudgetMb: 100, hostMemoryBudgetMb: 100);

        using ExternalMemoryReservation cpu = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), CancellationToken.None);

        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.ReserveExternalAsync(new(ExecutionProviderKind.OpenVino, null, 80, UseOpenVinoCpuProxy: true), cts.Token));
    }

    [Fact]
    public async Task ReserveExternalAsync_AcceleratorProvidersShareDeviceBudget()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, memoryBudgetMb: 100, hostMemoryBudgetMb: 100);

        using ExternalMemoryReservation dml0 = await pool.ReserveExternalAsync(new(ExecutionProviderKind.DirectMl, 0, 80), CancellationToken.None);

        // TRT-RTX on device 0 shares the same accelerator budget: 80 + 80 > 100 waits.
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(250));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.ReserveExternalAsync(new(ExecutionProviderKind.TensorRTRtx, 0, 80), cts.Token));

        // A different device ordinal has its own budget.
        using ExternalMemoryReservation dml1 = await pool.ReserveExternalAsync(new(ExecutionProviderKind.DirectMl, 1, 80), CancellationToken.None);
        Assert.NotNull(dml1);
    }

    [Fact]
    public async Task ReserveExternalAsync_ReleasedReservationFreesBudget()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, hostMemoryBudgetMb: 100);

        ExternalMemoryReservation first = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), CancellationToken.None);

        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.ReserveExternalAsync(new(ExecutionProviderKind.Dnnl, null, 80), cts.Token));

        first.Dispose();
        using ExternalMemoryReservation second = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Dnnl, null, 80), CancellationToken.None)
            .WaitAsync(TimeSpan.FromSeconds(5));
        Assert.NotNull(second);
    }

    [Fact]
    public async Task ReserveExternalAsync_OversizeFailsBeforeExternalConstructionSeam()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, hostMemoryBudgetMb: 100);

        InvalidOperationException ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 101), CancellationToken.None));
        Assert.Contains("host RAM", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task SessionAdmission_EvictsIdleExternalReservationToFit()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, hostMemoryBudgetMb: 100);
        var cpuKey = HostKey("ex1", 80);
        int callbackCalls = 0;
        int factoryCalls = 0;

        ExternalMemoryReservation reservation = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), CancellationToken.None);
        reservation.TrySetIdleEvictionCallback(() =>
        {
            callbackCalls++;
            reservation.Dispose();
            return true;
        });

        // The pooled session needs the same host budget: the idle external reservation is
        // evicted through its callback so the ONNX session fits.
        using SessionLease lease = await pool.GetLeaseAsync(
            cpuKey,
            _ => { factoryCalls++; return Task.FromResult(CreateMinimalSession()); },
            CancellationToken.None)
            .WaitAsync(TimeSpan.FromSeconds(5));

        Assert.Equal(1, callbackCalls);
        Assert.Equal(1, factoryCalls);
        Assert.NotNull(lease.Session);
    }

    [Fact]
    public async Task ReserveExternalAsync_DoesNotEvictBusyExternalReservation()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, hostMemoryBudgetMb: 100);
        int callbackCalls = 0;

        ExternalMemoryReservation busy = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), CancellationToken.None);
        busy.TrySetIdleEvictionCallback(() => { callbackCalls++; return false; });

        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), cts.Token));

        // Refusal leaves accounting intact: the reservation still occupies the budget.
        busy.Dispose();
        using ExternalMemoryReservation next = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), CancellationToken.None)
            .WaitAsync(TimeSpan.FromSeconds(5));
        Assert.NotNull(next);
        Assert.True(callbackCalls >= 1);
    }

    [Fact]
    public async Task ReserveExternalAsync_ThrowingEvictionCallback_KeepsAccountingIntact()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, hostMemoryBudgetMb: 100);

        ExternalMemoryReservation reservation = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), CancellationToken.None);
        reservation.TrySetIdleEvictionCallback(
            () => throw new InvalidOperationException("synthetic eviction failure"));

        // The throwing callback must not corrupt the admission loop or escape the wait.
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), cts.Token));

        reservation.Dispose();
        using ExternalMemoryReservation next = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), CancellationToken.None)
            .WaitAsync(TimeSpan.FromSeconds(5));
        Assert.NotNull(next);
    }

    [Fact]
    public async Task Dispose_ThrowingExternalEvictionCallback_DoesNotAbortRemainingTeardown()
    {
        var pool = new InferenceSessionPool(maxSessions: 8, hostMemoryBudgetMb: 100);
        ExternalMemoryReservation first = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 10), CancellationToken.None);
        ExternalMemoryReservation second = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 10), CancellationToken.None);
        int callbackCount = 0;
        first.TrySetIdleEvictionCallback(() =>
        {
            callbackCount++;
            throw new NotSupportedException("synthetic native disposal failure");
        });
        second.TrySetIdleEvictionCallback(() =>
        {
            callbackCount++;
            return true;
        });

        pool.Dispose();

        Assert.Equal(2, callbackCount);
    }

    [Fact]
    public async Task ExternalMemoryReservation_Dispose_IsIdempotent_AndTokensGoInert()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, hostMemoryBudgetMb: 100);

        ExternalMemoryReservation reservation = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), CancellationToken.None);

        reservation.Dispose();
        reservation.Dispose(); // second dispose is a no-op

        // Post-disposal: callback registration reports failure and release markers no-op.
        Assert.False(reservation.TrySetIdleEvictionCallback(() => true));
        reservation.MarkReleased();

        using ExternalMemoryReservation next = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), CancellationToken.None)
            .WaitAsync(TimeSpan.FromSeconds(5));
        Assert.NotNull(next);
    }

    [Fact]
    public async Task PoolDispose_AttemptsExternalEviction_AndSwallowsCallbackFailure()
    {
        int callbackCalls = 0;
        ExternalMemoryReservation reservation;
        var pool = new InferenceSessionPool(maxSessions: 8, hostMemoryBudgetMb: 100);

        reservation = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), CancellationToken.None);
        reservation.TrySetIdleEvictionCallback(() =>
        {
            callbackCalls++;
            throw new InvalidOperationException("synthetic teardown failure");
        });

        // Teardown must attempt the callback and must not propagate its failure.
        pool.Dispose();
        Assert.Equal(1, callbackCalls);

        reservation.Dispose(); // token remains idempotent even though accounting is cleared
    }

    [Fact]
    public async Task SessionAdmission_ExternalEviction_TriesNewerCandidateWhenOldestRefuses()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, hostMemoryBudgetMb: 100);
        var cpuKey = HostKey("exs1", 80);
        int refusedCalls = 0;
        int evictedCalls = 0;

        // Oldest reservation refuses eviction; the newer one releases itself. After the
        // newer 60MB is freed, the refusing 20MB plus the 80MB session just fit 100MB.
        ExternalMemoryReservation oldest = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 20), CancellationToken.None);
        oldest.TrySetIdleEvictionCallback(() => { refusedCalls++; return false; });

        ExternalMemoryReservation newer = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 60), CancellationToken.None);
        newer.TrySetIdleEvictionCallback(() =>
        {
            evictedCalls++;
            newer.Dispose();
            return true;
        });

        using SessionLease lease = await pool.GetLeaseAsync(
            cpuKey, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None)
            .WaitAsync(TimeSpan.FromSeconds(5));

        Assert.NotNull(lease.Session);
        Assert.Equal(1, evictedCalls);

        // The refusing reservation still counts against the budget until disposed.
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), cts.Token));

        oldest.Dispose();
        using ExternalMemoryReservation fits = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 20), CancellationToken.None)
            .WaitAsync(TimeSpan.FromSeconds(5));
        Assert.NotNull(fits);
    }

    [Fact]
    public async Task SessionAdmission_ExternalEviction_ThrowingCallbackDoesNotStarveOthers()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, hostMemoryBudgetMb: 100);
        var cpuKey = HostKey("ext1", 80);

        ExternalMemoryReservation throwing = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 20), CancellationToken.None);
        throwing.TrySetIdleEvictionCallback(
            () => throw new InvalidOperationException("synthetic eviction failure"));

        ExternalMemoryReservation newer = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 60), CancellationToken.None);
        int evictedCalls = 0;
        newer.TrySetIdleEvictionCallback(() =>
        {
            evictedCalls++;
            newer.Dispose();
            return true;
        });

        // The exception must not escape admission and must not starve the next candidate.
        using SessionLease lease = await pool.GetLeaseAsync(
            cpuKey, _ => Task.FromResult(CreateMinimalSession()), CancellationToken.None)
            .WaitAsync(TimeSpan.FromSeconds(5));

        Assert.NotNull(lease.Session);
        Assert.Equal(1, evictedCalls);
        throwing.Dispose();
    }

    [Fact]
    public async Task ReserveExternalAsync_CancelledWaiter_LeavesNoHiddenAccounting()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, hostMemoryBudgetMb: 100);

        ExternalMemoryReservation blocker = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), CancellationToken.None);

        // The cancelled waiter must not leave a pending reservation behind.
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 80), cts.Token));

        blocker.Dispose();
        using ExternalMemoryReservation full = await pool.ReserveExternalAsync(new(ExecutionProviderKind.Cpu, null, 100), CancellationToken.None)
            .WaitAsync(TimeSpan.FromSeconds(5));
        Assert.NotNull(full);
    }

    // ── RecommendedMaxSessions ────────────────────────────────────────────────

    [Fact]
    public void RecommendedMaxSessions_NullDevices_ReturnsDefault()
    {
        int result = InferenceSessionPool.RecommendedMaxSessions(null);
        Assert.Equal(12, result);
    }

    [Fact]
    public void RecommendedMaxSessions_NoGpu_ReturnsDefault()
    {
        var devices = new List<DeviceEntry>
        {
            new(DeviceKind.Cpu, 0, "CPU", "System", 0, 0, [ExecutionProviderKind.Cpu])
        };

        int result = InferenceSessionPool.RecommendedMaxSessions(devices);
        Assert.Equal(12, result);
    }

    [Fact]
    public void RecommendedMaxSessions_NvidiaGpuBelow8Gb_ReturnsDefault()
    {
        var devices = new List<DeviceEntry>
        {
            new(DeviceKind.DiscreteGpu, 0, "NVIDIA RTX 3060", "NVIDIA", 6144, 0, [ExecutionProviderKind.DirectMl])
        };

        int result = InferenceSessionPool.RecommendedMaxSessions(devices);
        Assert.Equal(12, result);
    }

    [Fact]
    public void RecommendedMaxSessions_NvidiaGpuAtLeast8Gb_Returns24()
    {
        var devices = new List<DeviceEntry>
        {
            new(DeviceKind.DiscreteGpu, 0, "NVIDIA RTX 4080", "NVIDIA", 16384, 0, [ExecutionProviderKind.DirectMl])
        };

        int result = InferenceSessionPool.RecommendedMaxSessions(devices);
        Assert.Equal(24, result);
    }

    [Fact]
    public void RecommendedMaxSessions_NonNvidiaGpuWith16Gb_ReturnsDefault()
    {
        var devices = new List<DeviceEntry>
        {
            new(DeviceKind.DiscreteGpu, 0, "AMD Radeon RX 7900 XTX", "AMD", 24576, 0, [ExecutionProviderKind.DirectMl])
        };

        int result = InferenceSessionPool.RecommendedMaxSessions(devices);
        Assert.Equal(12, result);
    }

    // ── Smoke ↔ stage pool-key identity ──────────────────────────────────────

    [Fact]
    public void SessionPoolKey_ModelIdAndVariant_participate_in_identity()
    {
        // Smoke and stage must build identical keys or smoke cannot warm the stage session.
        var smokeStyle = SessionPoolKey.ForEncoder(
            "nemotron-asr",
            @"C:\models\encoder.onnx",
            ExecutionProviderKind.TensorRTRtx,
            modelId: "vendor/nemotron-asr",
            variant: "default");
        var stageStyle = SessionPoolKey.ForEncoder(
            "nemotron-asr",
            @"C:\models\encoder.onnx",
            ExecutionProviderKind.TensorRTRtx,
            modelId: "vendor/nemotron-asr",
            variant: "default");
        var omittedIdentity = SessionPoolKey.ForEncoder(
            "nemotron-asr",
            @"C:\models\encoder.onnx",
            ExecutionProviderKind.TensorRTRtx);

        Assert.Equal(smokeStyle, stageStyle);
        Assert.NotEqual(smokeStyle, omittedIdentity);
    }

    [Fact]
    public void SessionPoolKey_Qwen3OmittingIdentity_matches_across_smoke_and_stage()
    {
        // Qwen3 production and smoke both omit modelId/variant — that pair must stay aligned.
        var smoke = SessionPoolKey.ForEncoder(
            "qwen3-asr",
            @"C:\models\encoder.onnx",
            ExecutionProviderKind.Cpu);
        var stage = SessionPoolKey.ForEncoder(
            "qwen3-asr",
            @"C:\models\encoder.onnx",
            ExecutionProviderKind.Cpu,
            modelId: null,
            variant: null);

        Assert.Equal(smoke, stage);
    }

    // ── Helper: build a minimal real InferenceSession from the ONNX operator ──

    /// <summary>
    /// Creates a minimal ONNX identity model in-memory so tests can obtain a real
    /// <see cref="InferenceSession"/> without loading any file from disk.
    /// The session accepts a single float input named "x" and returns it as "y".
    /// </summary>
    internal static InferenceSession CreateMinimalSession()
    {
        // Build the smallest valid ONNX protobuf by hand (bytes match the official wire format).
        // Graph: x (float32 [1]) → Identity → y (float32 [1])
        byte[] model = BuildIdentityOnnxModel();
        return new InferenceSession(model);
    }

    private static byte[] BuildIdentityOnnxModel()
    {
        // Minimal ONNX model (ir_version=7, opset 9, Identity op).
        // ModelProto: ir_version=7, opset_imports=[{domain:"", version:9}],
        // graph={node=[{input:x, output:y, op_type:Identity}], input=[x:float[1]], output=[y:float[1]]}
        // Generated via: onnx.helper.make_model / SerializeToString
        return
        [
            0x08, 0x07, 0x3A, 0x3A, 0x0A, 0x10, 0x0A, 0x01, 0x78, 0x12, 0x01, 0x79, 0x22, 0x08,
            0x49, 0x64, 0x65, 0x6E, 0x74, 0x69, 0x74, 0x79, 0x12, 0x04, 0x74, 0x65, 0x73, 0x74,
            0x5A, 0x0F, 0x0A, 0x01, 0x78, 0x12, 0x0A, 0x0A, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0A,
            0x02, 0x08, 0x01, 0x62, 0x0F, 0x0A, 0x01, 0x79, 0x12, 0x0A, 0x0A, 0x08, 0x08, 0x01,
            0x12, 0x04, 0x0A, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0A, 0x00, 0x10, 0x09,
        ];
    }
}
