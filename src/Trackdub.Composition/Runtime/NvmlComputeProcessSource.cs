using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace Trackdub.Composition.Runtime;

/// <summary>One process NVML lists on a device, with the memory it reports for it.</summary>
internal readonly record struct NvmlComputeProcess(uint Pid, ulong UsedGpuMemoryBytes);

/// <summary>
/// The NVML query <see cref="LinuxProcessGpuMemoryReader"/> needs, so its selection logic can be
/// tested without a GPU or driver.
/// </summary>
internal interface INvmlComputeProcessSource
{
    /// <summary>
    /// The compute processes NVML lists on each device, indexed by device, or null with
    /// <paramref name="failure"/> set when NVML cannot answer for every device.
    /// </summary>
    IReadOnlyList<IReadOnlyList<NvmlComputeProcess>>? QueryComputeProcesses(out string? failure);
}

/// <summary>
/// Queries compute processes through <c>libnvidia-ml.so.1</c>, which the NVIDIA container toolkit
/// injects whenever a GPU is passed through.
/// </summary>
/// <remarks>
/// <para>
/// NVML is bound at run time (<see cref="NativeLibrary"/>) so hosts without an NVIDIA driver
/// report an explicit unavailable result instead of failing to load. Resolution of the library
/// and its exports is cached for the process; <c>nvmlInit_v2</c> is retried on each query until
/// it succeeds and is then kept for the process lifetime, because init blocks on driver device
/// initialization when persistence mode is off and the reader is queried on every admission
/// decision.
/// </para>
/// <para>
/// Only the <c>_v3</c> entry points are used (<c>nvmlDeviceGetComputeRunningProcesses_v3</c> and
/// <c>nvmlDeviceGetGraphicsRunningProcesses_v3</c>): their <c>nvmlProcessInfo_t</c> is
/// <c>{pid, usedGpuMemory, gpuInstanceId, computeInstanceId}</c> (NVML API reference), while the
/// older entry points use a different struct layout. A driver without the v3 exports reports an
/// explicit unavailable result. Both lists are read for every device and merged per process by
/// <see cref="NvmlProcessListMerge"/>.
/// </para>
/// </remarks>
[SupportedOSPlatform("linux")]
internal sealed class NvmlNativeComputeProcessSource : INvmlComputeProcessSource
{
    private const int Success = 0;
    private const int NotSupported = 3;
    private const int InsufficientSize = 7;
    private const int MaxQueryAttempts = 4;
    private const int ProcessCountSlack = 8;

    private readonly object gate = new();
    private Exports? exports;
    private string? loadFailure;
    private bool initialized;

    internal static NvmlNativeComputeProcessSource Shared { get; } = new();

    public IReadOnlyList<IReadOnlyList<NvmlComputeProcess>>? QueryComputeProcesses(out string? failure)
    {
        lock (gate)
        {
            try
            {
                return QueryLocked(out failure);
            }
            catch (Exception exception) when (IsNativeFailure(exception))
            {
                failure = $"NVML query failed ({exception.GetType().Name}).";
                return null;
            }
        }
    }

    private static bool IsNativeFailure(Exception exception) =>
        exception is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException
            or InvalidOperationException or ArgumentException or NotSupportedException
            or MarshalDirectiveException or SEHException;

    private IReadOnlyList<IReadOnlyList<NvmlComputeProcess>>? QueryLocked(out string? failure)
    {
        Exports? ready = Prepare(out failure);
        return ready is null ? null : QueryAllDevices(ready, out failure);
    }

    /// <summary>Loads NVML once and initializes it, or reports why it cannot be used.</summary>
    private Exports? Prepare(out string? failure)
    {
        if (exports is null && loadFailure is null)
        {
            exports = Resolve(out loadFailure);
        }

        failure = loadFailure;
        if (exports is null || failure is not null)
        {
            return null;
        }

        failure = EnsureInitialized(exports);
        return failure is null ? exports : null;
    }

    private string? EnsureInitialized(Exports ready)
    {
        if (initialized)
        {
            return null;
        }

        int status = ready.Init();
        if (status != Success)
        {
            return $"NVML initialization failed (status {status}).";
        }

        initialized = true;
        return null;
    }

    private static IReadOnlyList<IReadOnlyList<NvmlComputeProcess>>? QueryAllDevices(Exports ready, out string? failure)
    {
        failure = null;
        if (ready.GetCount(out uint deviceCount) != Success)
        {
            failure = "NVML could not count devices.";
            return null;
        }

        var devices = new List<IReadOnlyList<NvmlComputeProcess>>((int)deviceCount);
        for (uint index = 0; index < deviceCount; index++)
        {
            // A device that cannot be queried leaves the footprint partial, which would
            // under-report real usage, so the whole reading is rejected instead.
            IReadOnlyList<NvmlComputeProcess>? processes = QueryDevice(ready, index, out failure);
            if (processes is null)
            {
                return null;
            }

            devices.Add(processes);
        }

        return devices;
    }

    private static IReadOnlyList<NvmlComputeProcess>? QueryDevice(Exports ready, uint index, out string? failure)
    {
        failure = null;
        if (ready.GetHandle(index, out nint handle) != Success || handle == nint.Zero)
        {
            failure = $"NVML could not open device {index}.";
            return null;
        }

        var device = new NvmlDevice(index, handle);
        IReadOnlyList<NvmlComputeProcess>? compute = QueryList(ready.Lists.Compute, device, out failure);
        if (compute is null)
        {
            return null;
        }

        IReadOnlyList<NvmlComputeProcess>? graphics = QueryList(ready.Lists.Graphics, device, out failure);
        return graphics is null ? null : NvmlProcessListMerge.ByProcess(compute, graphics);
    }

    private static IReadOnlyList<NvmlComputeProcess>? QueryList(
        ProcessLister lister, NvmlDevice device, out string? failure)
    {
        failure = null;
        for (int attempt = 0; attempt < MaxQueryAttempts; attempt++)
        {
            ListResult result = new ListQuery(lister, device).Run();
            if (result.Outcome == ListOutcome.Listed)
            {
                return result.Processes;
            }

            if (result.Outcome == ListOutcome.Failed)
            {
                failure = result.Failure;
                return null;
            }
        }

        failure = $"NVML {lister.Name} process list on device {device.Index} kept changing size.";
        return null;
    }

    private static IReadOnlyList<NvmlComputeProcess> ToProcesses(ReadOnlySpan<NvmlProcessInfo> infos)
    {
        var processes = new NvmlComputeProcess[infos.Length];
        for (int i = 0; i < infos.Length; i++)
        {
            processes[i] = new NvmlComputeProcess(infos[i].Pid, infos[i].UsedGpuMemory);
        }

        return processes;
    }

    private static Exports? Resolve(out string? failure)
    {
        failure = null;
        if (!NvmlLibrary.TryLoad(out NvmlLibrary library))
        {
            failure = "NVML (libnvidia-ml.so.1) is not available on this host.";
            return null;
        }

        Exports? bound = Bind(library);
        if (bound is null)
        {
            // Nothing from the library will be used, so release the handle instead of keeping
            // it mapped for the process lifetime.
            library.Free();
            failure = "The installed NVML does not export the functions the reader needs "
                + "(including the _v3 compute and graphics process queries).";
        }

        return bound;
    }

    private static Exports? Bind(NvmlLibrary library)
    {
        if (!library.TryGetExport<InitDelegate>("nvmlInit_v2", out var init))
        {
            return null;
        }

        if (!library.TryGetExport<GetCountDelegate>("nvmlDeviceGetCount_v2", out var getCount))
        {
            return null;
        }

        if (!library.TryGetExport<GetHandleDelegate>("nvmlDeviceGetHandleByIndex_v2", out var getHandle))
        {
            return null;
        }

        ProcessLists? lists = BindProcessLists(library);
        return lists is null ? null : new Exports(init, getCount, getHandle, lists);
    }

    private static ProcessLists? BindProcessLists(NvmlLibrary library)
    {
        if (!library.TryGetExport<GetProcessesDelegate>(
                "nvmlDeviceGetComputeRunningProcesses_v3", out var getCompute))
        {
            return null;
        }

        if (!library.TryGetExport<GetProcessesDelegate>(
                "nvmlDeviceGetGraphicsRunningProcesses_v3", out var getGraphics))
        {
            return null;
        }

        // A GPU without graphics support has no graphics contexts, so "not supported" reads as an
        // empty list there; the compute list has no such reading and fails the whole query.
        return new ProcessLists(
            new ProcessLister(ListKind.Compute, getCompute),
            new ProcessLister(ListKind.Graphics, getGraphics));
    }

    private enum ListOutcome
    {
        Listed,
        Retry,
        Failed,
    }

    private readonly record struct NvmlDevice(uint Index, nint Handle);

    private readonly record struct NvmlLibrary(nint Handle)
    {
        internal static bool TryLoad(out NvmlLibrary library)
        {
            library = default;
            if (!NativeLibrary.TryLoad("libnvidia-ml.so.1", out nint handle)
                && !NativeLibrary.TryLoad("libnvidia-ml.so", out handle))
            {
                return false;
            }

            library = new NvmlLibrary(handle);
            return true;
        }

        internal void Free() => NativeLibrary.Free(Handle);

        internal bool TryGetExport<TDelegate>(string name, out TDelegate function)
            where TDelegate : Delegate
        {
            function = null!;
            if (!NativeLibrary.TryGetExport(Handle, name, out nint address) || address == nint.Zero)
            {
                return false;
            }

            function = Marshal.GetDelegateForFunctionPointer<TDelegate>(address);
            return true;
        }
    }

    /// <summary>One sizing call and one read of a device's process list; retries are the caller's loop.</summary>
    private sealed class ListQuery(ProcessLister lister, NvmlDevice device)
    {
        private int status;
        private uint count;

        internal ListResult Run()
        {
            status = lister.Get(device.Handle, ref count, null);
            return DecidedBySizing() ?? Read();
        }

        private ListResult? DecidedBySizing()
        {
            if (status == Success && count == 0)
            {
                return ListResult.Listed([]);
            }

            if (status == NotSupported && lister.UnsupportedMeansEmpty)
            {
                return ListResult.Listed([]);
            }

            return status is Success or InsufficientSize ? null : Failed();
        }

        private ListResult Read()
        {
            // Processes can start between the sizing call and the read, so leave slack; the list
            // can still outgrow it, which the caller retries.
            var buffer = new NvmlProcessInfo[count + ProcessCountSlack];
            uint filled = (uint)buffer.Length;
            status = lister.Get(device.Handle, ref filled, buffer);
            if (status == InsufficientSize)
            {
                return ListResult.Retry;
            }

            return status == Success
                ? ListResult.Listed(ToProcesses(buffer.AsSpan(0, (int)Math.Min(filled, (uint)buffer.Length))))
                : Failed();
        }

        private ListResult Failed() => ListResult.Failed(
            $"NVML could not list {lister.Name} processes on device {device.Index} (status {status}).");
    }

    private sealed class ListResult
    {
        private ListResult()
        {
        }

        internal static ListResult Retry { get; } = new() { Outcome = ListOutcome.Retry };

        internal ListOutcome Outcome { get; private init; }

        internal IReadOnlyList<NvmlComputeProcess> Processes { get; private init; } = [];

        internal string? Failure { get; private init; }

        internal static ListResult Listed(IReadOnlyList<NvmlComputeProcess> processes) =>
            new() { Outcome = ListOutcome.Listed, Processes = processes };

        internal static ListResult Failed(string failure) =>
            new() { Outcome = ListOutcome.Failed, Failure = failure };
    }

    private enum ListKind
    {
        Compute,
        Graphics,
    }

    private sealed record ProcessLister(ListKind Kind, GetProcessesDelegate Get)
    {
        internal string Name => Kind == ListKind.Compute ? "compute" : "graphics";

        internal bool UnsupportedMeansEmpty => Kind == ListKind.Graphics;
    }

    private sealed record ProcessLists(ProcessLister Compute, ProcessLister Graphics);

    private sealed record Exports(
        InitDelegate Init,
        GetCountDelegate GetCount,
        GetHandleDelegate GetHandle,
        ProcessLists Lists);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int InitDelegate();

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GetCountDelegate(out uint count);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GetHandleDelegate(uint index, out nint device);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GetProcessesDelegate(
        nint device, ref uint infoCount, [In, Out] NvmlProcessInfo[]? infos);

    /// <summary>
    /// Mirrors <c>nvmlProcessInfo_t</c> as the <c>_v3</c> compute and graphics process queries fill
    /// it; field order must match the native layout.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct NvmlProcessInfo
    {
        public uint Pid;
        public ulong UsedGpuMemory;
        public uint GpuInstanceId;
        public uint ComputeInstanceId;
    }
}

/// <summary>Combines a device's compute and graphics process lists into one entry per process.</summary>
internal static class NvmlProcessListMerge
{
    /// <summary>
    /// One entry per process id, taking the larger of its compute and graphics totals.
    /// </summary>
    /// <remarks>
    /// Measured on an NVIDIA L4 (driver 595.112.07): a process holding both a CUDA and a graphics
    /// context appears in both lists with the same process-wide total, graphics memory included, so
    /// adding the lists would double-count it. A process with only a graphics context appears in the
    /// graphics list alone, which is why the compute list by itself under-reports it. Entries for the
    /// same id within one list are added first, so a process listed several times on a device keeps
    /// its full footprint.
    /// </remarks>
    internal static IReadOnlyList<NvmlComputeProcess> ByProcess(
        IReadOnlyList<NvmlComputeProcess> compute,
        IReadOnlyList<NvmlComputeProcess> graphics)
    {
        Dictionary<uint, ulong> merged = TotalsByProcess(compute);
        foreach ((uint pid, ulong bytes) in TotalsByProcess(graphics))
        {
            merged[pid] = merged.TryGetValue(pid, out ulong seen) ? Math.Max(seen, bytes) : bytes;
        }

        return merged.Select(entry => new NvmlComputeProcess(entry.Key, entry.Value)).ToArray();
    }

    private static Dictionary<uint, ulong> TotalsByProcess(IReadOnlyList<NvmlComputeProcess> processes)
    {
        var totals = new Dictionary<uint, ulong>();
        foreach (NvmlComputeProcess process in processes)
        {
            ulong seen = totals.GetValueOrDefault(process.Pid);
            totals[process.Pid] = ulong.MaxValue - seen < process.UsedGpuMemoryBytes
                ? ulong.MaxValue
                : seen + process.UsedGpuMemoryBytes;
        }

        return totals;
    }
}
