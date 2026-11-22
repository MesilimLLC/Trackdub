using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text;
using Trackdub.Composition.Runtime;
using Trackdub.Contracts;

namespace Trackdub.Composition.Tests;

/// <summary>
/// How the Linux reader turns NVML's per-device process lists into this process's
/// footprint, and when it refuses to report one.
/// </summary>
[SupportedOSPlatform("linux")]
public sealed class LinuxProcessGpuMemoryReaderTests
{
    private const uint Own = 1;
    private const long MiB = 1024 * 1024;

    private static IReadOnlyList<NvmlComputeProcess> Device(params (uint Pid, ulong Bytes)[] processes) =>
        processes.Select(p => new NvmlComputeProcess(p.Pid, p.Bytes)).ToArray();

    private static LinuxProcessGpuMemoryReader ReaderFor(params IReadOnlyList<NvmlComputeProcess>[] devices) =>
        new(new FakeSource(devices), Own);

    [Fact]
    public void Reports_this_processes_memory_and_ignores_other_processes()
    {
        // Measured on an L4: a 512 MiB allocation plus the CUDA context read 698 MiB.
        LinuxProcessGpuMemoryReader reader = ReaderFor(Device((Own, 698UL * MiB), (4242, 5000UL * MiB)));

        Assert.Equal(698 * MiB, reader.ReadDedicatedGpuMemoryBytes());
    }

    [Fact]
    public void Sums_this_process_across_devices()
    {
        LinuxProcessGpuMemoryReader reader = ReaderFor(
            Device((Own, 100UL * MiB)),
            Device((7, 1UL * MiB), (Own, 250UL * MiB)));

        Assert.Equal(350 * MiB, reader.ReadDedicatedGpuMemoryBytes());
    }

    [Fact]
    public void Reports_no_per_adapter_breakdown_so_the_pool_charges_every_device_the_total()
    {
        LinuxProcessGpuMemoryReader reader = ReaderFor(Device((Own, 10UL * MiB)));

        (long? total, IReadOnlyDictionary<long, long>? byAdapter) = reader.ReadDedicatedGpuMemory();

        Assert.Equal(10 * MiB, total);
        Assert.Null(byAdapter);
    }

    [Fact]
    public void Empty_lists_on_every_device_mean_this_process_holds_nothing()
    {
        LinuxProcessGpuMemoryReader reader = ReaderFor(Device(), Device());

        Assert.Equal(0, reader.ReadDedicatedGpuMemoryBytes());
    }

    [Fact]
    public void No_devices_means_this_process_holds_nothing()
    {
        Assert.Equal(0, new LinuxProcessGpuMemoryReader(new FakeSource([]), Own).ReadDedicatedGpuMemoryBytes());
    }

    [Fact]
    public void Processes_listed_without_this_id_are_unavailable_not_zero()
    {
        // Older drivers hand containers host process ids, which cannot be matched to ours.
        LinuxProcessGpuMemoryReader reader = ReaderFor(Device((2566, 1210UL * MiB)));

        Assert.Null(reader.ReadDedicatedGpuMemoryBytes());
        Assert.Contains("--pid=host", reader.UnavailableReason);
    }

    [Fact]
    public void A_listed_process_reporting_zero_bytes_is_unavailable()
    {
        // A process with a CUDA context always holds some memory, so 0 means the driver is not
        // attributing it; reporting 0 would let admission ignore real usage.
        LinuxProcessGpuMemoryReader reader = ReaderFor(Device((Own, 0)));

        Assert.Null(reader.ReadDedicatedGpuMemoryBytes());
        Assert.Contains("0 bytes", reader.UnavailableReason);
    }

    [Fact]
    public void A_listed_process_with_memory_not_available_is_unavailable()
    {
        LinuxProcessGpuMemoryReader reader = ReaderFor(Device((Own, LinuxProcessGpuMemoryReader.NvmlValueNotAvailable)));

        Assert.Null(reader.ReadDedicatedGpuMemoryBytes());
        Assert.Contains("not available", reader.UnavailableReason);
    }

    [Fact]
    public void One_unreadable_entry_rejects_the_whole_reading_instead_of_a_partial_footprint()
    {
        LinuxProcessGpuMemoryReader reader = ReaderFor(
            Device((Own, 100UL * MiB)),
            Device((Own, LinuxProcessGpuMemoryReader.NvmlValueNotAvailable)));

        Assert.Null(reader.ReadDedicatedGpuMemoryBytes());
    }

    [Fact]
    public void Values_beyond_the_signed_range_are_rejected()
    {
        LinuxProcessGpuMemoryReader reader = ReaderFor(
            Device((Own, (ulong)long.MaxValue)),
            Device((Own, 1)));

        Assert.Null(reader.ReadDedicatedGpuMemoryBytes());
    }

    [Fact]
    public void A_source_failure_is_reported_verbatim()
    {
        var reader = new LinuxProcessGpuMemoryReader(new FakeSource(null, "NVML initialization failed (status 9)."), Own);

        Assert.Null(reader.ReadDedicatedGpuMemoryBytes());
        Assert.Equal("NVML initialization failed (status 9).", reader.UnavailableReason);
    }

    [Fact]
    public void A_later_good_reading_clears_the_failure_reason()
    {
        var source = new FakeSource(null, "NVML initialization failed (status 9).");
        var reader = new LinuxProcessGpuMemoryReader(source, Own);
        Assert.Null(reader.ReadDedicatedGpuMemoryBytes());

        source.Devices = [Device((Own, 5UL * MiB))];

        Assert.Equal(5 * MiB, reader.ReadDedicatedGpuMemoryBytes());
        Assert.DoesNotContain("failed", reader.UnavailableReason);
    }

    [Fact]
    public void Native_source_reports_a_reason_when_the_nvidia_driver_is_absent()
    {
        Assert.SkipWhen(!OperatingSystem.IsLinux(), "NVML is bound through libnvidia-ml.so on Linux only.");
        if (NativeLibrary.TryLoad("libnvidia-ml.so.1", out nint library))
        {
            NativeLibrary.Free(library);
            Assert.Skip("This host has an NVIDIA driver, so the unavailable path cannot be observed.");
        }

        var source = new NvmlNativeComputeProcessSource();

        Assert.Null(source.QueryComputeProcesses(out string? failure));
        Assert.Contains("libnvidia-ml.so.1", failure);
        // The reason is cached for the process rather than probing the loader on every read.
        Assert.Null(source.QueryComputeProcesses(out string? second));
        Assert.Equal(failure, second);
    }

    [Fact]
    public void Logs_once_when_the_reader_first_reads_this_processes_memory()
    {
        var logger = new RecordingLogger();
        var reader = new LinuxProcessGpuMemoryReader(new FakeSource([Device((Own, 698UL * MiB))]), Own, logger);

        reader.ReadDedicatedGpuMemoryBytes();
        reader.ReadDedicatedGpuMemoryBytes();
        reader.ReadDedicatedGpuMemory();

        string line = Assert.Single(logger.Lines);
        Assert.Contains("is active", line);
        Assert.Contains("698 MiB", line);
    }

    [Fact]
    public void Logs_the_reason_once_when_the_reader_is_unavailable()
    {
        var logger = new RecordingLogger();
        var reader = new LinuxProcessGpuMemoryReader(
            new FakeSource(null, "NVML (libnvidia-ml.so.1) is not available on this host."), Own, logger);

        reader.ReadDedicatedGpuMemoryBytes();
        reader.ReadDedicatedGpuMemoryBytes();

        string line = Assert.Single(logger.Lines);
        Assert.Contains("is unavailable", line);
        Assert.Contains("NVML (libnvidia-ml.so.1) is not available on this host.", line);
        Assert.Contains("reservations only", line);
    }

    [Fact]
    public void Logs_again_when_the_reader_changes_between_active_and_unavailable()
    {
        var logger = new RecordingLogger();
        var source = new FakeSource([Device((Own, 100UL * MiB))]);
        var reader = new LinuxProcessGpuMemoryReader(source, Own, logger);

        reader.ReadDedicatedGpuMemoryBytes();
        source.Devices = [Device((4242, 5UL * MiB))];
        reader.ReadDedicatedGpuMemoryBytes();
        reader.ReadDedicatedGpuMemoryBytes();
        source.Devices = [Device((Own, 120UL * MiB))];
        reader.ReadDedicatedGpuMemoryBytes();

        Assert.Equal(3, logger.Lines.Count);
        Assert.Contains("is active", logger.Lines[0]);
        Assert.Contains("is unavailable", logger.Lines[1]);
        Assert.Contains("is active", logger.Lines[2]);
    }

    [Fact]
    public void Writes_the_state_line_to_the_status_output_even_without_a_logger()
    {
        using var output = new StringWriter();
        var reader = new LinuxProcessGpuMemoryReader(
            new FakeSource([Device((Own, 698UL * MiB))]), Own, logger: null, statusOutput: output);

        reader.ReadDedicatedGpuMemoryBytes();
        reader.ReadDedicatedGpuMemoryBytes();

        string line = Assert.Single(output.ToString().Split(output.NewLine, StringSplitOptions.RemoveEmptyEntries));
        Assert.Contains("is active", line);
        Assert.Contains("698 MiB", line);
    }

    [Fact]
    public void Writes_the_unavailable_reason_to_the_status_output_once_per_flip()
    {
        using var output = new StringWriter();
        var source = new FakeSource([Device((Own, 100UL * MiB))]);
        var reader = new LinuxProcessGpuMemoryReader(source, Own, logger: null, statusOutput: output);

        reader.ReadDedicatedGpuMemoryBytes();
        source.Devices = [Device((4242, 5UL * MiB))];
        reader.ReadDedicatedGpuMemoryBytes();
        reader.ReadDedicatedGpuMemoryBytes();

        string[] lines = output.ToString().Split(output.NewLine, StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(2, lines.Length);
        Assert.Contains("is active", lines[0]);
        Assert.Contains("is unavailable", lines[1]);
        Assert.Contains("reservations only", lines[1]);
    }

    [Fact]
    public void A_failing_status_output_does_not_fail_the_reading()
    {
        var reader = new LinuxProcessGpuMemoryReader(
            new FakeSource([Device((Own, 64UL * MiB))]), Own, logger: null, statusOutput: new ThrowingWriter());

        Assert.Equal(64 * MiB, reader.ReadDedicatedGpuMemoryBytes());
        Assert.Equal(64 * MiB, reader.ReadDedicatedGpuMemoryBytes());
    }

    [Fact]
    public void Reads_without_a_logger_still_work()
    {
        LinuxProcessGpuMemoryReader reader = ReaderFor(Device((Own, 64UL * MiB)));

        Assert.Equal(64 * MiB, reader.ReadDedicatedGpuMemoryBytes());
    }

    private sealed class ThrowingWriter : TextWriter
    {
        public override Encoding Encoding => Encoding.UTF8;

        public override void WriteLine(string? value) => throw new IOException("stderr is closed.");
    }

    private sealed class RecordingLogger : IApplicationLogger
    {
        public List<string> Lines { get; } = [];

        public void LogDebug(string message)
        {
        }

        public void LogInformation(string message) => Lines.Add(message);

        public void LogWarning(string message, Exception? exception = null)
        {
        }

        public void LogError(string message, Exception? exception = null)
        {
        }
    }

    private sealed class FakeSource(IReadOnlyList<IReadOnlyList<NvmlComputeProcess>>? devices, string? failure = null)
        : INvmlComputeProcessSource
    {
        public IReadOnlyList<IReadOnlyList<NvmlComputeProcess>>? Devices { get; set; } = devices;

        public IReadOnlyList<IReadOnlyList<NvmlComputeProcess>>? QueryComputeProcesses(out string? reason)
        {
            reason = Devices is null ? failure : null;
            return Devices;
        }
    }
}
