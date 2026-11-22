using System.Runtime.Versioning;
using Trackdub.Contracts.Benchmarking;

namespace Trackdub.Composition.Runtime;

/// <summary>
/// Reads this process's dedicated GPU memory on Linux from NVML's per-process compute list.
/// </summary>
/// <remarks>
/// <para>
/// <c>nvmlDeviceGetComputeRunningProcesses_v3</c> lists each process with a CUDA context on a
/// device together with the memory it holds ("all of the memory used by the application", NVML API
/// reference), so summing the entries for this process id across devices gives the process-isolated
/// footprint <see cref="IAvailableVramReader"/> cannot attribute. The reading is the NVML
/// counterpart of the Windows per-process performance counters and feeds the same admission floor
/// in the shared session pool.
/// </para>
/// <para>
/// Which process id NVML reports depends on the driver and the PID namespace. Measured on an NVIDIA
/// L4 with driver 595.112.07, a default Docker container (own PID namespace) gets the container's
/// own process id back with accurate memory, so matching <see cref="Environment.ProcessId"/> works
/// there. NVIDIA's issue tracker and forum describe older drivers (before 555) returning host
/// process ids to containers, and one report of 0 bytes for container processes on 555; neither can
/// be matched or trusted, so those cases report an explicit unavailable result instead of a guess
/// and the pool keeps accounting for its own reservations only.
/// </para>
/// <para>
/// NVML has no adapter LUID, so only the process total is reported and every device is charged the
/// full total, the conservative bound the pool already uses when a reader gives no breakdown.
/// </para>
/// </remarks>
[SupportedOSPlatform("linux")]
internal sealed class LinuxProcessGpuMemoryReader : IProcessGpuMemoryReader
{
    /// <summary>NVML_VALUE_NOT_AVAILABLE for the 64-bit <c>usedGpuMemory</c> field.</summary>
    internal const ulong NvmlValueNotAvailable = ulong.MaxValue;

    private readonly INvmlComputeProcessSource source;
    private readonly uint processId;
    private volatile string? lastFailure;

    public LinuxProcessGpuMemoryReader()
        : this(NvmlNativeComputeProcessSource.Shared, checked((uint)Environment.ProcessId))
    {
    }

    internal LinuxProcessGpuMemoryReader(INvmlComputeProcessSource source, uint processId)
    {
        this.source = source;
        this.processId = processId;
    }

    public string UnavailableReason => lastFailure ?? "No current Linux process GPU measurement is available.";

    public long? ReadDedicatedGpuMemoryBytes() => Read();

    public (long? TotalBytes, IReadOnlyDictionary<long, long>? ByAdapterLuid) ReadDedicatedGpuMemory() =>
        (Read(), null);

    private long? Read()
    {
        IReadOnlyList<IReadOnlyList<NvmlComputeProcess>>? devices = source.QueryComputeProcesses(out string? failure);
        if (devices is null)
        {
            lastFailure = failure ?? "NVML could not list compute processes.";
            return null;
        }

        long? total = SelectOwnUsage(devices, processId, out failure);
        lastFailure = failure;
        return total;
    }

    /// <summary>
    /// This process's footprint from the per-device compute process lists, or null with
    /// <paramref name="failure"/> set when the lists cannot be trusted to describe it.
    /// </summary>
    /// <remarks>
    /// A process id that appears on several devices is summed. An entry the driver reports as
    /// unavailable or as 0 bytes fails the reading: a process holding a CUDA context always holds
    /// some memory, so 0 means the driver is not attributing it, and a partial footprint would
    /// under-report real usage. When processes are listed but none carries this id, the lists are
    /// probably in a different PID namespace, so the reading is unavailable rather than a zero;
    /// only empty lists on every device mean this process holds nothing.
    /// </remarks>
    internal static long? SelectOwnUsage(
        IReadOnlyList<IReadOnlyList<NvmlComputeProcess>> devices,
        uint processId,
        out string? failure)
    {
        failure = null;
        NvmlComputeProcess[] listed = devices.SelectMany(processes => processes).ToArray();
        NvmlComputeProcess[] own = listed.Where(process => process.Pid == processId).ToArray();
        if (own.Length > 0)
        {
            return SumOwnEntries(own, out failure);
        }

        if (listed.Length == 0)
        {
            return 0;
        }

        failure = "NVML lists compute processes but none with this process id; in a container on a driver "
            + "older than 555 NVML reports host process ids, which cannot be matched without --pid=host.";
        return null;
    }

    private static long? SumOwnEntries(IEnumerable<NvmlComputeProcess> own, out string? failure)
    {
        failure = null;
        ulong total = 0;
        foreach (NvmlComputeProcess process in own)
        {
            failure = UntrustedReason(process);
            if (failure is not null)
            {
                return null;
            }

            if (process.UsedGpuMemoryBytes > long.MaxValue - total)
            {
                failure = "NVML reported GPU memory beyond the supported range.";
                return null;
            }

            total += process.UsedGpuMemoryBytes;
        }

        return (long)total;
    }

    /// <summary>Why this process's entry cannot be trusted as a footprint, or null when it can.</summary>
    private static string? UntrustedReason(NvmlComputeProcess process) => process.UsedGpuMemoryBytes switch
    {
        NvmlValueNotAvailable => "NVML lists this process but reports its GPU memory as not available.",
        0 => "NVML lists this process but reports 0 bytes for it, so the driver is not "
            + "attributing its GPU memory in this PID namespace.",
        _ => null,
    };
}
