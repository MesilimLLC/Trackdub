using System.Runtime.Versioning;
using Trackdub.Contracts;
using Trackdub.Contracts.Benchmarking;

namespace Trackdub.Composition.Runtime;

/// <summary>
/// Reads this process's dedicated GPU memory on Linux from NVML's per-process compute and graphics
/// lists.
/// </summary>
/// <remarks>
/// <para>
/// <c>nvmlDeviceGetComputeRunningProcesses_v3</c> and <c>nvmlDeviceGetGraphicsRunningProcesses_v3</c>
/// list each process with a CUDA or graphics context on a device together with the memory it holds
/// ("all of the memory used by the application", NVML API reference), so summing the entries for
/// this process id across devices gives the process-isolated footprint
/// <see cref="IAvailableVramReader"/> cannot attribute. The reading is the NVML counterpart of the
/// Windows per-process performance counters and feeds the same admission floor in the shared
/// session pool.
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
/// The two lists are merged per device by <see cref="NvmlProcessListMerge"/>, never added. Measured
/// on the same L4, a process with both context types appears in both lists with the same
/// process-wide total (graphics memory included), and a graphics-only process appears in the
/// graphics list alone, so reading only the compute list would report it as holding nothing.
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

    private const int StateActive = 1;
    private const int StateUnavailable = 2;

    private readonly INvmlComputeProcessSource source;
    private readonly uint processId;
    private readonly IApplicationLogger? logger;
    private readonly TextWriter? statusOutput;
    private volatile string? lastFailure;
    private int reportedState;

    public LinuxProcessGpuMemoryReader(IApplicationLogger? logger = null)
        : this(NvmlNativeComputeProcessSource.Shared, checked((uint)Environment.ProcessId), logger, Console.Error)
    {
    }

    internal LinuxProcessGpuMemoryReader(
        INvmlComputeProcessSource source,
        uint processId,
        IApplicationLogger? logger = null,
        TextWriter? statusOutput = null)
    {
        this.source = source;
        this.processId = processId;
        this.logger = logger;
        this.statusOutput = statusOutput;
    }

    public string UnavailableReason => lastFailure ?? "No current Linux process GPU measurement is available.";

    public long? ReadDedicatedGpuMemoryBytes() => Read();

    public (long? TotalBytes, IReadOnlyDictionary<long, long>? ByAdapterLuid) ReadDedicatedGpuMemory() =>
        (Read(), null);

    private long? Read()
    {
        long? total = ReadTotal(out string? failure);
        lastFailure = failure;
        ReportStateChange(total, failure);
        return total;
    }

    private long? ReadTotal(out string? failure)
    {
        IReadOnlyList<IReadOnlyList<NvmlComputeProcess>>? devices = source.QueryComputeProcesses(out failure);
        if (devices is null)
        {
            failure ??= "NVML could not list compute processes.";
            return null;
        }

        return SelectOwnUsage(devices, processId, out failure);
    }

    /// <summary>
    /// Reports one line the first time the reader is read and again whenever it flips between
    /// active and unavailable, so a run's log shows which admission model the pool used without
    /// logging on every admission decision. The line goes to the application logger and to the
    /// status writer (stderr in production): the file logger is not collected from containers, and
    /// stdout carries machine-readable output such as the batch report.
    /// </summary>
    private void ReportStateChange(long? total, string? failure)
    {
        if (logger is null && statusOutput is null)
        {
            return;
        }

        int state = total is null ? StateUnavailable : StateActive;
        if (Interlocked.Exchange(ref reportedState, state) == state)
        {
            return;
        }

        string message = state == StateActive
            ? $"Process GPU memory reader (Linux/NVML) is active: this process holds "
                + $"{total.GetValueOrDefault() / (1024 * 1024)} MiB on the GPU, which accelerator admission "
                + "counts against the per-device budget."
            : $"Process GPU memory reader (Linux/NVML) is unavailable: {failure} Accelerator admission "
                + "uses the pool's own reservations only.";

        logger?.LogInformation(message);
        TryWriteStatus(message);
    }

    private void TryWriteStatus(string message)
    {
        try
        {
            statusOutput?.WriteLine(message);
        }
        catch (Exception ex) when (ex is IOException or ObjectDisposedException)
        {
            // A closed or broken stderr must not fail the measurement the line only describes.
        }
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
