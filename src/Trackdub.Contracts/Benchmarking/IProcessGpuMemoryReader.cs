namespace Trackdub.Contracts.Benchmarking;

/// <summary>
/// Reports dedicated GPU memory allocated by this process.
/// </summary>
/// <remarks>
/// <para>
/// This is process-isolated usage of the adapter's own memory, which is the attribution
/// <see cref="IAvailableVramReader"/> cannot provide: DXGI's <c>QueryVideoMemoryInfo</c> reports
/// adapter-wide budget minus usage, so its reading also moves with every other process on the
/// same GPU. On Windows this reader instead uses the per-process "GPU Process Memory"
/// performance counter set, whose instance names are keyed by process id; on Linux it sums this
/// process's entries in NVML's per-process compute list.
/// </para>
/// <para>
/// The two ports answer different questions and are not interchangeable: use this one to
/// attribute bytes to a stage or to bound one process's footprint, and
/// <see cref="IAvailableVramReader"/> to detect adapter-wide pressure. Returns null where the
/// platform or driver cannot report it.
/// </para>
/// <para>
/// A host can also make this reading drive the shared ONNX session pool's accelerator admission
/// instead of only reporting it, so GPU memory this process holds outside the pool's own
/// reservations consumes the same per-device budget; see <c>SharedPoolOptions</c> in
/// <c>Trackdub.Inference.Onnx</c>.
/// </para>
/// </remarks>
public interface IProcessGpuMemoryReader
{
    /// <summary>
    /// Dedicated GPU memory currently allocated by this process, in bytes, or null when this
    /// platform or driver cannot report it.
    /// </summary>
    long? ReadDedicatedGpuMemoryBytes();

    /// <summary>
    /// Dedicated GPU memory currently allocated by this process, broken down by adapter LUID,
    /// or null when this platform, driver, or reader cannot attribute usage per adapter. The
    /// values sum to <see cref="ReadDedicatedGpuMemoryBytes"/> when both are available. The
    /// default implementation reports no per-adapter attribution, in which case consumers fall
    /// back to the process-total reading.
    /// </summary>
    IReadOnlyDictionary<long, long>? ReadDedicatedGpuMemoryBytesByAdapterLuid() => null;

    /// <summary>
    /// Reads the process total and the per-adapter breakdown together. A reader that can derive
    /// both from one underlying collection overrides this so the two numbers describe the same
    /// instant and cost one collection; the default composes the two separate reads.
    /// </summary>
    (long? TotalBytes, IReadOnlyDictionary<long, long>? ByAdapterLuid) ReadDedicatedGpuMemory() =>
        (ReadDedicatedGpuMemoryBytes(), ReadDedicatedGpuMemoryBytesByAdapterLuid());

    /// <summary>Why the reading is unavailable, surfaced verbatim in benchmark evidence.</summary>
    string UnavailableReason { get; }
}
