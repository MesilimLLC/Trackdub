using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace Trackdub.Composition.Runtime;

/// <summary>One compute process NVML lists on a device, with the memory it reports for it.</summary>
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
/// Only <c>nvmlDeviceGetComputeRunningProcesses_v3</c> is used: its <c>nvmlProcessInfo_t</c> is
/// <c>{pid, usedGpuMemory, gpuInstanceId, computeInstanceId}</c> (NVML API reference), while the
/// older entry points use a different struct layout. A driver without the v3 export reports an
/// explicit unavailable result.
/// </para>
/// </remarks>
[SupportedOSPlatform("linux")]
internal sealed class NvmlNativeComputeProcessSource : INvmlComputeProcessSource
{
    private const int Success = 0;
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
        if (ready.GetHandle(index, out nint device) != Success || device == nint.Zero)
        {
            failure = $"NVML could not open device {index}.";
            return null;
        }

        for (int attempt = 0; attempt < MaxQueryAttempts; attempt++)
        {
            ListResult result = ListOnce(ready, device, index);
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

        failure = $"NVML compute process list on device {index} kept changing size.";
        return null;
    }

    /// <summary>One sizing call and one read; retries are the caller's loop.</summary>
    private static ListResult ListOnce(Exports ready, nint device, uint index)
    {
        uint count = 0;
        int status = ready.GetComputeProcesses(device, ref count, null);
        if (status == Success && count == 0)
        {
            return ListResult.Listed([]);
        }

        if (status != Success && status != InsufficientSize)
        {
            return ListResult.Failed(ListFailure(index, status));
        }

        // Processes can start between the sizing call and the read, so leave slack; the list
        // can still outgrow it, which the caller retries.
        var buffer = new NvmlProcessInfo[count + ProcessCountSlack];
        uint filled = (uint)buffer.Length;
        status = ready.GetComputeProcesses(device, ref filled, buffer);
        if (status == InsufficientSize)
        {
            return ListResult.Retry;
        }

        return status == Success
            ? ListResult.Listed(ToProcesses(buffer, filled))
            : ListResult.Failed(ListFailure(index, status));
    }

    private static string ListFailure(uint index, int status) =>
        $"NVML could not list compute processes on device {index} (status {status}).";

    private static IReadOnlyList<NvmlComputeProcess> ToProcesses(NvmlProcessInfo[] buffer, uint filled)
    {
        int count = (int)Math.Min(filled, (uint)buffer.Length);
        var processes = new NvmlComputeProcess[count];
        for (int i = 0; i < count; i++)
        {
            processes[i] = new NvmlComputeProcess(buffer[i].Pid, buffer[i].UsedGpuMemory);
        }

        return processes;
    }

    private static Exports? Resolve(out string? failure)
    {
        failure = null;
        if (!TryLoadLibrary(out nint library))
        {
            failure = "NVML (libnvidia-ml.so.1) is not available on this host.";
            return null;
        }

        Exports? bound = Bind(library);
        if (bound is null)
        {
            // Nothing from the library will be used, so release the handle instead of keeping
            // it mapped for the process lifetime.
            NativeLibrary.Free(library);
            failure = "The installed NVML does not export the functions the reader needs "
                + "(including nvmlDeviceGetComputeRunningProcesses_v3).";
        }

        return bound;
    }

    private static bool TryLoadLibrary(out nint library) =>
        NativeLibrary.TryLoad("libnvidia-ml.so.1", out library)
        || NativeLibrary.TryLoad("libnvidia-ml.so", out library);

    private static Exports? Bind(nint library)
    {
        if (!TryGetExport<InitDelegate>(library, "nvmlInit_v2", out var init))
        {
            return null;
        }

        if (!TryGetExport<GetCountDelegate>(library, "nvmlDeviceGetCount_v2", out var getCount))
        {
            return null;
        }

        if (!TryGetExport<GetHandleDelegate>(library, "nvmlDeviceGetHandleByIndex_v2", out var getHandle))
        {
            return null;
        }

        if (!TryGetExport<GetComputeProcessesDelegate>(
                library, "nvmlDeviceGetComputeRunningProcesses_v3", out var getProcesses))
        {
            return null;
        }

        return new Exports(init, getCount, getHandle, getProcesses);
    }

    private static bool TryGetExport<TDelegate>(nint library, string name, out TDelegate function)
        where TDelegate : Delegate
    {
        function = null!;
        if (!NativeLibrary.TryGetExport(library, name, out nint address) || address == nint.Zero)
        {
            return false;
        }

        function = Marshal.GetDelegateForFunctionPointer<TDelegate>(address);
        return true;
    }

    private enum ListOutcome
    {
        Listed,
        Retry,
        Failed,
    }

    private readonly record struct ListResult(
        ListOutcome Outcome,
        IReadOnlyList<NvmlComputeProcess> Processes,
        string? Failure)
    {
        internal static ListResult Retry { get; } = new(ListOutcome.Retry, [], null);

        internal static ListResult Listed(IReadOnlyList<NvmlComputeProcess> processes) =>
            new(ListOutcome.Listed, processes, null);

        internal static ListResult Failed(string failure) => new(ListOutcome.Failed, [], failure);
    }

    private sealed record Exports(
        InitDelegate Init,
        GetCountDelegate GetCount,
        GetHandleDelegate GetHandle,
        GetComputeProcessesDelegate GetComputeProcesses);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int InitDelegate();

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GetCountDelegate(out uint count);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GetHandleDelegate(uint index, out nint device);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int GetComputeProcessesDelegate(
        nint device, ref uint infoCount, [In, Out] NvmlProcessInfo[]? infos);

    /// <summary>
    /// Mirrors <c>nvmlProcessInfo_t</c> as <c>nvmlDeviceGetComputeRunningProcesses_v3</c> fills it;
    /// field order must match the native layout.
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
