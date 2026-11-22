using Trackdub.Composition.Runtime;

namespace Trackdub.Composition.Tests;

/// <summary>
/// How a device's NVML compute and graphics process lists combine into one entry per process.
/// The L4 figures come from a probe holding a CUDA context, an EGL context, or both.
/// </summary>
public sealed class NvmlProcessListMergeTests
{
    private const uint Pid = 7;
    private const ulong MiB = 1024UL * 1024UL;

    private static IReadOnlyList<NvmlComputeProcess> List(params (uint Pid, ulong Bytes)[] processes) =>
        processes.Select(p => new NvmlComputeProcess(p.Pid, p.Bytes)).ToArray();

    [Fact]
    public void A_process_in_both_lists_with_the_same_total_is_not_counted_twice()
    {
        // L4: 957 MiB after a CUDA allocation, an EGL context and a render target, in both lists.
        IReadOnlyList<NvmlComputeProcess> merged = NvmlProcessListMerge.ByProcess(
            List((Pid, 957 * MiB)), List((Pid, 957 * MiB)));

        NvmlComputeProcess only = Assert.Single(merged);
        Assert.Equal(957 * MiB, only.UsedGpuMemoryBytes);
    }

    [Fact]
    public void A_graphics_only_process_is_kept_instead_of_reading_as_empty()
    {
        // L4: an EGL render target alone read 259 MiB in the graphics list and nothing in compute.
        IReadOnlyList<NvmlComputeProcess> merged = NvmlProcessListMerge.ByProcess(
            List(), List((Pid, 259 * MiB)));

        Assert.Equal(new NvmlComputeProcess(Pid, 259 * MiB), Assert.Single(merged));
    }

    [Fact]
    public void A_compute_only_process_is_kept()
    {
        IReadOnlyList<NvmlComputeProcess> merged = NvmlProcessListMerge.ByProcess(
            List((Pid, 698 * MiB)), List());

        Assert.Equal(new NvmlComputeProcess(Pid, 698 * MiB), Assert.Single(merged));
    }

    [Fact]
    public void When_the_lists_disagree_the_larger_value_wins()
    {
        IReadOnlyList<NvmlComputeProcess> merged = NvmlProcessListMerge.ByProcess(
            List((Pid, 100 * MiB)), List((Pid, 300 * MiB)));

        Assert.Equal(300 * MiB, Assert.Single(merged).UsedGpuMemoryBytes);
    }

    [Fact]
    public void Different_processes_stay_separate()
    {
        IReadOnlyList<NvmlComputeProcess> merged = NvmlProcessListMerge.ByProcess(
            List((1, 10 * MiB), (2, 20 * MiB)), List((2, 20 * MiB), (3, 30 * MiB)));

        Assert.Equal(
            new[] { (1u, 10 * MiB), (2u, 20 * MiB), (3u, 30 * MiB) },
            merged.OrderBy(p => p.Pid).Select(p => (p.Pid, p.UsedGpuMemoryBytes)).ToArray());
    }

    [Fact]
    public void Entries_for_one_process_within_a_list_are_added_before_the_lists_are_compared()
    {
        IReadOnlyList<NvmlComputeProcess> merged = NvmlProcessListMerge.ByProcess(
            List((Pid, 100 * MiB), (Pid, 50 * MiB)), List((Pid, 120 * MiB)));

        Assert.Equal(150 * MiB, Assert.Single(merged).UsedGpuMemoryBytes);
    }

    [Fact]
    public void A_not_available_marker_survives_the_merge_so_the_reader_can_reject_it()
    {
        IReadOnlyList<NvmlComputeProcess> merged = NvmlProcessListMerge.ByProcess(
            List((Pid, 100 * MiB), (Pid, ulong.MaxValue)), List((Pid, 100 * MiB)));

        Assert.Equal(ulong.MaxValue, Assert.Single(merged).UsedGpuMemoryBytes);
    }
}
