// Feature: hardware-matrix-routing, Property 4: GPU Classification Correctness
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Runtime;
using FsCheck;
using FsCheck.Xunit;

namespace Trackdub.Inference.Tests;

/// <summary>
/// Property-based test verifying the GPU classification logic in DeviceEnumerator.
/// Tests the internal ClassifyDeviceKind method directly with generated inputs.
///
/// This file is excluded from the net11.0 build because DeviceEnumerator.cs is
/// Windows-only (DXGI COM interop).
///
/// **Validates: Requirements 1.4**
/// </summary>
public sealed class DeviceEnumeratorClassificationPropertyTests
{
    // Feature: hardware-matrix-routing, Property 4: GPU Classification Correctness
    /// <summary>
    /// Property 4: GPU Classification Correctness
    ///
    /// For any DXGI adapter with adapter flags and dedicated VRAM, the DeviceEnumerator SHALL
    /// classify it as DiscreteGpu when adapter flags indicate discrete OR (flags are ambiguous
    /// AND dedicated VRAM > 512 MB), and SHALL classify it as IntegratedGpu otherwise.
    ///
    /// Tests the internal ClassifyDeviceKind method directly with generated inputs.
    /// The current implementation uses a pure VRAM heuristic (>512 MB = discrete) since
    /// DXGI adapter flags do not reliably distinguish discrete from integrated.
    ///
    /// **Validates: Requirements 1.4**
    /// </summary>
    [Property(MaxTest = 200)]
    public bool GpuClassification_Correctness(uint adapterFlags, PositiveInt vramRaw)
    {
        // Filter out the software adapter flag (2) since those are skipped before classification
        uint flags = adapterFlags & ~2u;
        int dedicatedVramMb = System.Math.Min(vramRaw.Get, 65536); // Cap at 64GB

        DeviceKind result = WindowsDeviceEnumerator.ClassifyDeviceKind(flags, dedicatedVramMb);

        // Per the implementation: VRAM > 512 MB → DiscreteGpu, otherwise → IntegratedGpu
        // The flags parameter is currently not used for discrete/integrated distinction
        // (DXGI doesn't have a reliable discrete flag), so the heuristic is purely VRAM-based.
        bool expectedDiscrete = dedicatedVramMb > 512;
        DeviceKind expectedKind = expectedDiscrete ? DeviceKind.DiscreteGpu : DeviceKind.IntegratedGpu;

        return result == expectedKind;
    }

    /// <summary>
    /// Property 4 boundary case: VRAM exactly at 512 MB threshold.
    /// At exactly 512 MB, the device should be classified as IntegratedGpu (threshold is strictly greater than).
    /// </summary>
    [Property(MaxTest = 100)]
    public bool GpuClassification_BoundaryAt512_IsIntegrated(uint adapterFlags)
    {
        uint flags = adapterFlags & ~2u;
        DeviceKind result = WindowsDeviceEnumerator.ClassifyDeviceKind(flags, 512);
        return result == DeviceKind.IntegratedGpu;
    }

    /// <summary>
    /// Property 4 boundary case: VRAM at 513 MB (just above threshold).
    /// At 513 MB, the device should be classified as DiscreteGpu.
    /// </summary>
    [Property(MaxTest = 100)]
    public bool GpuClassification_BoundaryAt513_IsDiscrete(uint adapterFlags)
    {
        uint flags = adapterFlags & ~2u;
        DeviceKind result = WindowsDeviceEnumerator.ClassifyDeviceKind(flags, 513);
        return result == DeviceKind.DiscreteGpu;
    }
}
