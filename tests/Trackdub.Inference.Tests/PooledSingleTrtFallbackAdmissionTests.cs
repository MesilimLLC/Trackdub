using Microsoft.ML.OnnxRuntime;
using Trackdub.Domain;
using Trackdub.Inference.Onnx;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.TensorRtRtx;

namespace Trackdub.Inference.Tests;

/// <summary>
/// A pooled session whose provider fallback lands it in another admission bucket than its key
/// (TensorRT RTX init falling over to CPU) is accounted where its memory lives: host RAM, not
/// the device budget its key was admitted against.
/// </summary>
[Collection(nameof(TensorRtRtxTeardownGuardCollection))]
public sealed class PooledSingleTrtFallbackAdmissionTests : IDisposable
{
    private readonly string directory = Path.Join(Path.GetTempPath(), "trackdub-single-fallback-" + Guid.NewGuid().ToString("N"));

    public PooledSingleTrtFallbackAdmissionTests()
    {
        Directory.CreateDirectory(directory);
        TensorRtRtxTeardownGuard.ResetForTests();
    }

    public void Dispose()
    {
        TensorRtRtxTeardownGuard.ResetForTests();
        Directory.Delete(directory, recursive: true);
    }

    [Fact]
    public async Task Cpu_fallback_is_charged_to_host_ram_and_not_to_the_device()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, enableMemoryAdmission: true, memoryBudgetMb: 100, hostMemoryBudgetMb: 100);

        using SessionLease fallback = await pool.GetLeaseReportingProviderAsync(
            Key(ExecutionProviderKind.TensorRTRtx, "trt", 80),
            CreatedOn(ExecutionProviderKind.Cpu),
            CancellationToken.None);

        // The device budget is untouched: a graph needing all of it is admitted while the
        // fallback session is still leased.
        using (var guard = new CancellationTokenSource(ConcurrencyTestTimeouts.HangGuard))
        using (await pool.GetLeaseAsync(Key(ExecutionProviderKind.DirectMl, "gpu", 100), CreateSession, guard.Token))
        {
        }

        // Host RAM holds it: a CPU graph that fits an empty host budget has to wait.
        using var blocked = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.GetLeaseAsync(Key(ExecutionProviderKind.Cpu, "cpu", 80), CreateSession, blocked.Token));
    }

    [Fact]
    public async Task DirectMl_fallback_stays_on_the_device_budget()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, enableMemoryAdmission: true, memoryBudgetMb: 100, hostMemoryBudgetMb: 100);

        using SessionLease fallback = await pool.GetLeaseReportingProviderAsync(
            Key(ExecutionProviderKind.TensorRTRtx, "trt", 80),
            CreatedOn(ExecutionProviderKind.DirectMl),
            CancellationToken.None);

        using (var guard = new CancellationTokenSource(ConcurrencyTestTimeouts.HangGuard))
        using (await pool.GetLeaseAsync(Key(ExecutionProviderKind.Cpu, "cpu", 100), CreateSession, guard.Token))
        {
        }

        using var blocked = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.GetLeaseAsync(Key(ExecutionProviderKind.DirectMl, "gpu", 80), CreateSession, blocked.Token));
    }

    [Fact]
    public async Task Idle_cpu_fallback_is_evicted_for_host_pressure_not_device_pressure()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, enableMemoryAdmission: true, memoryBudgetMb: 100, hostMemoryBudgetMb: 100);
        SessionPoolKey trtKey = Key(ExecutionProviderKind.TensorRTRtx, "trt", 80);
        int creations = 0;

        using (await pool.GetLeaseReportingProviderAsync(
                   trtKey, CreatedOn(ExecutionProviderKind.Cpu, () => creations++), CancellationToken.None))
        {
        }

        using (var guard = new CancellationTokenSource(ConcurrencyTestTimeouts.HangGuard))
        using (await pool.GetLeaseAsync(Key(ExecutionProviderKind.DirectMl, "gpu", 100), CreateSession, guard.Token))
        {
        }

        Assert.True(pool.TryPinExisting(trtKey, out SessionResidency? pin), "Device admission must not evict a host-RAM session.");
        pin!.Dispose();

        using (var guard = new CancellationTokenSource(ConcurrencyTestTimeouts.HangGuard))
        using (await pool.GetLeaseAsync(Key(ExecutionProviderKind.Cpu, "cpu", 80), CreateSession, guard.Token))
        {
        }

        Assert.False(pool.TryPinExisting(trtKey, out _), "Host admission must evict the idle host-RAM fallback session.");
        using (await pool.GetLeaseReportingProviderAsync(
                   trtKey, CreatedOn(ExecutionProviderKind.Cpu, () => creations++), CancellationToken.None))
        {
        }

        Assert.Equal(2, creations);
    }

    [Fact]
    public async Task Readmission_evicts_idle_host_sessions_to_fit()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, enableMemoryAdmission: true, memoryBudgetMb: 1000, hostMemoryBudgetMb: 120);
        SessionPoolKey cpuKey = Key(ExecutionProviderKind.Cpu, "cpu", 80);

        using (await pool.GetLeaseAsync(cpuKey, CreateSession, CancellationToken.None))
        {
        }

        using var guard = new CancellationTokenSource(ConcurrencyTestTimeouts.HangGuard);
        using SessionLease fallback = await pool.GetLeaseReportingProviderAsync(
            Key(ExecutionProviderKind.TensorRTRtx, "trt", 80),
            CreatedOn(ExecutionProviderKind.Cpu),
            guard.Token);

        Assert.False(pool.TryPinExisting(cpuKey, out _));
    }

    [Fact]
    public async Task Readmission_waits_for_host_budget_without_holding_the_device_reservation()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, enableMemoryAdmission: true, memoryBudgetMb: 100, hostMemoryBudgetMb: 100);
        using SessionLease hostHolder = await pool.GetLeaseAsync(
            Key(ExecutionProviderKind.Cpu, "cpu", 80), CreateSession, CancellationToken.None);
        var created = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var guard = new CancellationTokenSource(ConcurrencyTestTimeouts.HangGuard);

        Task<SessionLease> fallbackTask = pool.GetLeaseReportingProviderAsync(
            Key(ExecutionProviderKind.TensorRTRtx, "trt", 80),
            CreatedOn(ExecutionProviderKind.Cpu, created.SetResult),
            guard.Token);
        await created.Task.WaitAsync(guard.Token);

        // The session exists on CPU and is waiting for host RAM; the device budget it was first
        // admitted against is free for a graph that needs all of it.
        using (await pool.GetLeaseAsync(Key(ExecutionProviderKind.DirectMl, "gpu", 100), CreateSession, guard.Token))
        {
        }

        Assert.False(fallbackTask.IsCompleted, "The fallback must wait while host RAM is leased out.");

        hostHolder.Dispose();
        using SessionLease fallback = await fallbackTask.WaitAsync(guard.Token);
        Assert.NotNull(fallback.Session);
    }

    [Fact]
    public async Task A_waiting_cpu_fallback_is_accounted_against_host_ram()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, enableMemoryAdmission: true, memoryBudgetMb: 100, hostMemoryBudgetMb: 100);
        using SessionLease hostHolder = await pool.GetLeaseAsync(
            Key(ExecutionProviderKind.Cpu, "cpu", 50), CreateSession, CancellationToken.None);
        var created = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var guard = new CancellationTokenSource(ConcurrencyTestTimeouts.HangGuard);

        Task<SessionLease> fallbackTask = pool.GetLeaseReportingProviderAsync(
            Key(ExecutionProviderKind.TensorRTRtx, "trt", 80),
            CreatedOn(ExecutionProviderKind.Cpu, created.SetResult),
            guard.Token);
        await created.Task.WaitAsync(guard.Token);

        // The created session is charged against host RAM while it waits: a direct CPU request
        // that would fit the leftover budget (100 - 50) is blocked because the waiting fallback's
        // 80 MB already consumes the host budget. Without the immediate charge it would be
        // admitted against invisible memory.
        using var blocked = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.GetLeaseAsync(Key(ExecutionProviderKind.Cpu, "cpu2", 30), CreateSession, blocked.Token));

        hostHolder.Dispose();
        using SessionLease fallback = await fallbackTask.WaitAsync(guard.Token);
        Assert.NotNull(fallback.Session);
    }

    [Fact]
    public async Task Bundle_with_a_published_cpu_fallback_fails_fast_instead_of_hanging()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, enableMemoryAdmission: true, memoryBudgetMb: 1000, hostMemoryBudgetMb: 100);
        SessionPoolKey gpuFallbackKey = Key(ExecutionProviderKind.TensorRTRtx, "trt", 80);

        // Publish a GPU-keyed session that fell back to CPU: it lives in the host-RAM bucket.
        using (await pool.GetLeaseReportingProviderAsync(
                   gpuFallbackKey, CreatedOn(ExecutionProviderKind.Cpu), CancellationToken.None))
        {
        }

        // A bundle pairing that fallback session with a direct CPU session needs 80 + 80 = 160 MB
        // of host RAM, more than the 100 MB budget. Preflight must charge the fallback to host RAM
        // and fail fast — never pin it and then wait on memory the bundle's own pin holds.
        SessionPoolKey cpuKey = Key(ExecutionProviderKind.Cpu, "cpu", 80);
        int factoryCalls = 0;
        using var guard = new CancellationTokenSource(ConcurrencyTestTimeouts.HangGuard);
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => pool.GetLeaseBundleAsync(
            [
                new SessionLeaseRequest(gpuFallbackKey, _ =>
                {
                    factoryCalls++;
                    return Task.FromResult(CreateMinimalSession());
                }),
                new SessionLeaseRequest(cpuKey, _ =>
                {
                    factoryCalls++;
                    return Task.FromResult(CreateMinimalSession());
                }),
            ],
            guard.Token));

        Assert.Equal(0, factoryCalls);
    }

    [Fact]
    public async Task Readmission_over_the_host_budget_fails_and_frees_the_device_reservation()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, enableMemoryAdmission: true, memoryBudgetMb: 100, hostMemoryBudgetMb: 50);
        SessionPoolKey trtKey = Key(ExecutionProviderKind.TensorRTRtx, "trt", 80);

        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(
            () => pool.GetLeaseReportingProviderAsync(trtKey, CreatedOn(ExecutionProviderKind.Cpu), CancellationToken.None));

        Assert.Contains("host RAM", error.Message, StringComparison.Ordinal);
        Assert.False(pool.TryPinExisting(trtKey, out _));
        using var guard = new CancellationTokenSource(ConcurrencyTestTimeouts.HangGuard);
        using (await pool.GetLeaseAsync(Key(ExecutionProviderKind.DirectMl, "gpu", 100), CreateSession, guard.Token))
        {
        }
    }

    [Fact]
    public async Task Concurrent_acquires_of_a_falling_back_key_share_one_creation()
    {
        using var pool = new InferenceSessionPool(
            maxSessions: 8, enableMemoryAdmission: true, memoryBudgetMb: 1000, hostMemoryBudgetMb: 1000);
        SessionPoolKey trtKey = Key(ExecutionProviderKind.TensorRTRtx, "trt", 80);
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        int creations = 0;
        using var guard = new CancellationTokenSource(ConcurrencyTestTimeouts.HangGuard);

        Task<SessionLease> first = pool.GetLeaseReportingProviderAsync(trtKey, Create, guard.Token);
        await entered.Task.WaitAsync(guard.Token);
        Task<SessionLease> second = pool.GetLeaseReportingProviderAsync(trtKey, Create, guard.Token);
        release.SetResult();

        SessionLease firstLease = await first.WaitAsync(guard.Token);
        InferenceSession session = firstLease.Session;
        Assert.False(second.IsCompleted, "Leases stay exclusive across the re-admission.");
        firstLease.Dispose();
        using SessionLease secondLease = await second.WaitAsync(guard.Token);

        Assert.Same(session, secondLease.Session);
        Assert.Equal(1, creations);

        async Task<CreatedPoolSession> Create(CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref creations);
            entered.TrySetResult();
            await release.Task.WaitAsync(cancellationToken);
            return new CreatedPoolSession(CreateMinimalSession(), ExecutionProviderKind.Cpu);
        }
    }

    [Fact]
    public async Task CreatePooledSingleAsync_accounts_a_trt_rtx_cpu_fallback_in_host_ram()
    {
        string model = WriteModel("fallback-single.onnx");

        // Learn the key this host builds for a TensorRT RTX request on this model.
        SessionPoolKey probeKey;
        using (var probePool = new InferenceSessionPool(maxSessions: 2, enableMemoryAdmission: false))
        using (OnnxExecutionSessionFactory.SingleSessionLease probe = await OnnxExecutionSessionFactory.CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = "fallback-engine", Provider = ExecutionProviderKind.TensorRTRtx, CancellationToken = CancellationToken.None, Pool = probePool, SessionFactory = (_, _) => CreateMinimalSession() }, model))
        {
            probeKey = probe.ResolvedPoolKey!;
        }

        if (probeKey.Provider is not ExecutionProviderKind.TensorRTRtx)
        {
            // This host cannot build TensorRT RTX session options, so there is no TensorRT RTX
            // attempt to fall back from; the pool-level tests above cover the re-admission.
            return;
        }

        long estimateMb = probeKey.EstimatedVramMb;
        using var pool = new InferenceSessionPool(
            maxSessions: 8, enableMemoryAdmission: true, memoryBudgetMb: estimateMb, hostMemoryBudgetMb: estimateMb + 10);
        // TensorRT RTX fails, then on Windows the DirectML attempt (or a CPU selection standing in for
        // it when DirectML options are unavailable) fails too, so the chain ends on CPU.
        int failingAttempts = OperatingSystem.IsWindows() ? 2 : 1;
        int attempts = 0;

        using OnnxExecutionSessionFactory.SingleSessionLease lease = await OnnxExecutionSessionFactory.CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest
        {
            EngineFamily = "fallback-engine",
            Provider = ExecutionProviderKind.TensorRTRtx,
            CancellationToken = CancellationToken.None,
            Pool = pool,
            SessionFactory = (_, _) =>
            {
                int attempt = Interlocked.Increment(ref attempts);
                if (attempt == 1)
                {
                    throw new InvalidOperationException(
                        "[NvTensorRTRTX EP] Failed to create serialized engine for fused node");
                }

                if (attempt <= failingAttempts)
                {
                    throw new InvalidOperationException("DirectML device was removed.");
                }

                return CreateMinimalSession();
            }
        }, model);

        Assert.Equal(ExecutionProviderKind.TensorRTRtx, lease.ResolvedPoolKey!.Provider);
        Assert.Equal("cpu", lease.SelectedProvider);
        Assert.Contains("TensorRT RTX session init failed", lease.BootstrapDetail, StringComparison.Ordinal);

        using (var guard = new CancellationTokenSource(ConcurrencyTestTimeouts.HangGuard))
        using (await pool.GetLeaseAsync(Key(ExecutionProviderKind.DirectMl, "gpu", estimateMb), CreateSession, guard.Token))
        {
        }

        using var blocked = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            () => pool.GetLeaseAsync(Key(ExecutionProviderKind.Cpu, "cpu", 20), CreateSession, blocked.Token));
    }

    private static SessionPoolKey Key(ExecutionProviderKind provider, string modelPathHash, long estimatedMb) =>
        new("eng", null, null, provider, modelPathHash, 0, "default") { EstimatedVramMb = estimatedMb };

    private static Func<CancellationToken, Task<CreatedPoolSession>> CreatedOn(
        ExecutionProviderKind provider,
        Action? onCreated = null) =>
        _ =>
        {
            var created = new CreatedPoolSession(CreateMinimalSession(), provider);
            onCreated?.Invoke();
            return Task.FromResult(created);
        };

    private static Task<InferenceSession> CreateSession(CancellationToken cancellationToken) =>
        Task.FromResult(CreateMinimalSession());

    private string WriteModel(string name)
    {
        string path = Path.Join(directory, name);
        File.WriteAllBytes(path, IdentityModel);
        return path;
    }

    private static InferenceSession CreateMinimalSession() => new(IdentityModel);

    private static readonly byte[] IdentityModel =
    [
        0x08, 0x07, 0x3A, 0x3A, 0x0A, 0x10, 0x0A, 0x01, 0x78, 0x12, 0x01, 0x79, 0x22, 0x08,
        0x49, 0x64, 0x65, 0x6E, 0x74, 0x69, 0x74, 0x79, 0x12, 0x04, 0x74, 0x65, 0x73, 0x74,
        0x5A, 0x0F, 0x0A, 0x01, 0x78, 0x12, 0x0A, 0x0A, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0A,
        0x02, 0x08, 0x01, 0x62, 0x0F, 0x0A, 0x01, 0x79, 0x12, 0x0A, 0x0A, 0x08, 0x08, 0x01,
        0x12, 0x04, 0x0A, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0A, 0x00, 0x10, 0x09,
    ];
}
