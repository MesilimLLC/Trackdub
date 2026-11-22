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
            catch (Exception exception) when (exception is DllNotFoundException or EntryPointNotFoundException
                or BadImageFormatException or InvalidOperationException or ArgumentException
                or NotSupportedException or MarshalDirectiveException or SEHException)
            {
                failure = $"NVML query failed ({exception.GetType().Name}).";
                return null;
            }
        }
    }

    private IReadOnlyList<IReadOnlyList<NvmlComputeProcess>>? QueryLocked(out string? failure)
    {
        failure = null;
        if (loadFailure is not null)
        {
            failure = loadFailure;
            return null;
        }

        exports ??= Resolve(out loadFailure);
        if (exports is null)
        {
            failure = loadFailure;
            return null;
        }

        if (!initialized)
        {
            int initStatus = exports.Init();
            if (initStatus != Success)
            {
                failure = $"NVML initialization failed (status {initStatus}).";
                return null;
            }

            initialized = true;
        }

        if (exports.GetCount(out uint deviceCount) != Success)
        {
            failure = "NVML could not count devices.";
            return null;
        }

        var devices = new List<IReadOnlyList<NvmlComputeProcess>>((int)deviceCount);
        for (uint index = 0; index < deviceCount; index++)
        {
            if (exports.GetHandle(index, out nint device) != Success || device == nint.Zero)
            {
                failure = $"NVML could not open device {index}.";
                return null;
            }

            IReadOnlyList<NvmlComputeProcess>? processes = QueryDevice(exports, device, index, out failure);
            if (processes is null)
            {
                // A device that cannot be queried leaves the footprint partial, which would
                // under-report real usage, so the whole reading is rejected instead.
                return null;
            }

            devices.Add(processes);
        }

        return devices;
    }

    private static IReadOnlyList<NvmlComputeProcess>? QueryDevice(Exports exports, nint device, uint index, out string? failure)
    {
        failure = null;
        for (int attempt = 0; attempt < MaxQueryAttempts; attempt++)
        {
            uint count = 0;
            int status = exports.GetComputeProcesses(device, ref count, null);
            if (status == Success && count == 0)
            {
                return [];
            }

            if (status != InsufficientSize && status != Success)
            {
                failure = $"NVML could not list compute processes on device {index} (status {status}).";
                return null;
            }

            // Processes can start between the sizing call and the read, so leave slack and retry
            // when the list still outgrows the buffer.
            uint capacity = count + ProcessCountSlack;
            var buffer = new NvmlProcessInfo[capacity];
            uint filled = capacity;
            status = exports.GetComputeProcesses(device, ref filled, buffer);
            if (status == InsufficientSize)
            {
                continue;
            }

            if (status != Success)
            {
                failure = $"NVML could not list compute processes on device {index} (status {status}).";
                return null;
            }

            var processes = new List<NvmlComputeProcess>((int)filled);
            for (int i = 0; i < filled && i < buffer.Length; i++)
            {
                processes.Add(new NvmlComputeProcess(buffer[i].Pid, buffer[i].UsedGpuMemory));
            }

            return processes;
        }

        failure = $"NVML compute process list on device {index} kept changing size.";
        return null;
    }

    private static Exports? Resolve(out string? failure)
    {
        failure = null;
        if (!NativeLibrary.TryLoad("libnvidia-ml.so.1", out nint library)
            && !NativeLibrary.TryLoad("libnvidia-ml.so", out library))
        {
            failure = "NVML (libnvidia-ml.so.1) is not available on this host.";
            return null;
        }

        if (TryGetExport<InitDelegate>(library, "nvmlInit_v2", out var init)
            && TryGetExport<GetCountDelegate>(library, "nvmlDeviceGetCount_v2", out var getCount)
            && TryGetExport<GetHandleDelegate>(library, "nvmlDeviceGetHandleByIndex_v2", out var getHandle)
            && TryGetExport<GetComputeProcessesDelegate>(library, "nvmlDeviceGetComputeRunningProcesses_v3", out var getProcesses))
        {
            return new Exports(init, getCount, getHandle, getProcesses);
        }

        failure = "The installed NVML does not export nvmlDeviceGetComputeRunningProcesses_v3.";
        return null;
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
