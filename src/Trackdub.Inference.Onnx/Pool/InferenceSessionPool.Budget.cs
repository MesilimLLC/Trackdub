namespace Trackdub.Inference.Onnx.Pool;

/// <summary>
/// Admission budget policy: default accelerator/host budgets, hardware scaling,
/// and VRAM detection. Hardware-dependent, resolved once per process.
/// </summary>
internal sealed partial class InferenceSessionPool
{
    /// <summary>
    /// Default accelerator (VRAM) admission budget per device when <c>enableMemoryAdmission</c> is on.
    /// Scaled with detected GPU VRAM (three-quarters of the largest adapter up to 16GB,
    /// 85% above that for large-memory cards above 16 GiB — a band that includes
    /// consumer cards such as the RTX 4090 (24GB) and RTX 5090 (32GB) as well as
    /// L4/L40S — floor 4GB, cap 64GB) so large bundles such as MADLAD-400 (~6.4GB quantized reservation at 2x, ~8.1GB
    /// for the external-data fp16 export on TensorRT RTX) fit on 12GB GPUs out-of-the-box,
    /// while small GPUs keep the conservative 4GB guard.
    /// Deliberately a larger fraction than the host-RAM default (quarter): VRAM is the
    /// working set's home, and reservations already include weights plus slack (2x on
    /// unmeasured providers, a measured 1.25x on TensorRT RTX and DirectML — see
    /// <see cref="SessionPoolKey.ResidentPercentOfWeights"/>), while process-GPU observation
    /// charges any real usage beyond them.
    /// An explicit <c>TRACKDUB_SESSION_VRAM_BUDGET_MB</c> always wins over this default.
    /// Unknown/unavailable VRAM (non-Windows, no GPU, detection failure) falls back to
    /// the 4GB floor — never lower than the historical default.
    /// </summary>
    public static long DefaultMemoryBudgetMb { get; } = ResolveDefaultMemoryBudgetMb();

    /// <summary>Floor for the scaled accelerator budget: the historical conservative guard.</summary>
    public const long AcceleratorBudgetFloorMb = 4096;

    /// <summary>Cap for the scaled accelerator budget. Only binds on very large-memory
    /// cards; the binding constraint there is the 85% fraction.</summary>
    public const long AcceleratorBudgetCapMb = 65536;

    internal static long ScaleAcceleratorBudgetMb(long maxVramMb)
    {
        // Large-memory cards above 16 GiB (consumer RTX 4090/5090 as well as
        // L4/L40S/A100) have the headroom to keep 85% of VRAM for admission
        // instead of 75%: the old 16GB hard cap stranded two-thirds of an L40S.
        // Cards at or below 16 GiB keep the more conservative 75%.
        long scaled = maxVramMb > 16384 ? maxVramMb * 85 / 100 : maxVramMb * 3 / 4;
        return Math.Clamp(scaled, AcceleratorBudgetFloorMb, AcceleratorBudgetCapMb);
    }

    internal static long DetectMaxAcceleratorVramMb()
    {
#if WINDOWS
        if (!OperatingSystem.IsWindows())
            return 0;

        try
        {
            return AcceleratorVramProbe.QueryMaxDedicatedVramMb();
        }
        catch
        {
            return 0;
        }
#else
        if (!OperatingSystem.IsLinux())
            return 0;

        try
        {
            return AcceleratorVramProbe.QueryLinuxMaxDedicatedVramMb();
        }
        catch
        {
            return 0;
        }
#endif
    }

    private static long ResolveDefaultMemoryBudgetMb()
    {
        try
        {
            return ScaleAcceleratorBudgetMb(DetectMaxAcceleratorVramMb());
        }
        catch
        {
            return AcceleratorBudgetFloorMb;
        }
    }

    /// <summary>
    /// Default host RAM admission budget shared by CPU/DNNL and OpenVINO CPU-proxy sessions.
    /// Scaled with physical RAM (quarter of total, floor 4GB, cap 16GB) so large CPU
    /// bundles such as MADLAD-400 (~6.4GB reservation) are admissible out-of-the-box on
    /// ample machines, while small machines keep the conservative 4GB guard.
    /// An explicit <c>TRACKDUB_SESSION_RAM_BUDGET_MB</c> always wins over this default.
    /// </summary>
    public static long DefaultHostMemoryBudgetMb { get; } = ResolveDefaultHostMemoryBudgetMb();

    /// <summary>Floor for the scaled host RAM budget: the historical conservative guard.</summary>
    public const long HostMemoryBudgetFloorMb = 4096;

    /// <summary>Cap for the scaled host RAM budget.</summary>
    public const long HostMemoryBudgetCapMb = 16384;

    internal static long ScaleHostMemoryBudgetMb(long totalRamMb) =>
        Math.Clamp(totalRamMb / 4, HostMemoryBudgetFloorMb, HostMemoryBudgetCapMb);

    private static long ResolveDefaultHostMemoryBudgetMb()
    {
        try
        {
            long totalRamMb = (long)(GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / (1024 * 1024));
            return ScaleHostMemoryBudgetMb(totalRamMb);
        }
        catch
        {
            return HostMemoryBudgetFloorMb;
        }
    }
}
