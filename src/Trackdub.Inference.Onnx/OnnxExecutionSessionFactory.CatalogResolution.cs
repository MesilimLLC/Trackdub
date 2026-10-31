using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging;
using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Dnnl;
using Trackdub.Inference.Onnx.EpContext;
using Trackdub.Inference.Onnx.ExecutionProviders;
using Trackdub.Inference.Onnx.Pool;
#if WINDOWS
using Trackdub.Inference.Onnx.WindowsMl;
#endif
using Trackdub.Inference.Onnx.Migraphx;
using Trackdub.Inference.Onnx.TensorRtRtx;
using Trackdub.Inference.Runtime.Migraphx;
using Trackdub.Inference.Runtime.TensorRtRtx;
using Trackdub.Inference.Runtime.WinMlCatalog;
using Trackdub.Inference.Runtime.Planning;
using Microsoft.ML.OnnxRuntime;

namespace Trackdub.Inference.Onnx;

internal static partial class OnnxExecutionSessionFactory
{
    private static readonly IReadOnlyDictionary<ExecutionProviderKind, int> WildcardCatalogEpRank =
        new Dictionary<ExecutionProviderKind, int>
        {
            [ExecutionProviderKind.TensorRTRtx] = 0,
            [ExecutionProviderKind.Migraphx] = 1,
            [ExecutionProviderKind.VitisAi] = 4,
            [ExecutionProviderKind.DirectMl] = 5,
            [ExecutionProviderKind.Cpu] = 10,
        };

    private static readonly IReadOnlyDictionary<(ExecutionProviderKind Provider, OrtHardwareDeviceType DeviceType), int> DeviceCatalogEpRank =
        new Dictionary<(ExecutionProviderKind Provider, OrtHardwareDeviceType DeviceType), int>
        {
            [(ExecutionProviderKind.Qnn, OrtHardwareDeviceType.NPU)] = 2,
            [(ExecutionProviderKind.OpenVinoCatalog, OrtHardwareDeviceType.NPU)] = 3,
            [(ExecutionProviderKind.Qnn, OrtHardwareDeviceType.GPU)] = 6,
            [(ExecutionProviderKind.OpenVinoCatalog, OrtHardwareDeviceType.GPU)] = 7,
            [(ExecutionProviderKind.OpenVinoCatalog, OrtHardwareDeviceType.CPU)] = 8,
            [(ExecutionProviderKind.Dnnl, OrtHardwareDeviceType.CPU)] = 9,
        };

    // Names that map to a provider without inspecting the device type. TensorRT RTX, DNNL, and
    // DirectML are intentionally absent: TryMapDeviceCandidateCatalogEpName maps them with
    // hardware-type awareness (and the DNNL spelling aliases), so adding them here would let a
    // direct reader bypass that check and claim a GPU placement for a non-GPU device.
    private static readonly IReadOnlyDictionary<string, ExecutionProviderKind> ExactCatalogEpNames =
        new Dictionary<string, ExecutionProviderKind>(StringComparer.OrdinalIgnoreCase)
        {
            [MigraphxProviderConstants.OrtExecutionProviderName] = ExecutionProviderKind.Migraphx,
            [OpenVinoCatalogProviderConstants.OrtExecutionProviderName] = ExecutionProviderKind.OpenVinoCatalog,
            [QnnProviderConstants.OrtExecutionProviderName] = ExecutionProviderKind.Qnn,
            [VitisAiProviderConstants.OrtExecutionProviderName] = ExecutionProviderKind.VitisAi,
            ["CPUExecutionProvider"] = ExecutionProviderKind.Cpu,
        };

    internal static ExecutionProviderKind ResolveCatalogEffectiveProviderFromSession(InferenceSession session) =>
        ResolveBestCatalogProvider(GetEpDevicesForInputs(session));

    private static IReadOnlyList<OrtEpDevice> GetEpDevicesForInputs(InferenceSession session)
    {
        try
        {
            return session.GetEpDeviceForInputs();
        }
        catch (OnnxRuntimeException)
        {
            return [];
        }
    }

    private static ExecutionProviderKind ResolveBestCatalogProvider(IReadOnlyList<OrtEpDevice> devices)
    {
        ExecutionProviderKind? bestProvider = null;
        int bestRank = int.MaxValue;
        foreach (OrtEpDevice device in devices)
        {
            (ExecutionProviderKind provider, int rank) = RankCatalogDevice(device, devices.Count);
            if (rank >= bestRank)
            {
                continue;
            }

            bestRank = rank;
            bestProvider = provider;
        }

        return bestProvider ?? ExecutionProviderKind.Cpu;
    }

    private static (ExecutionProviderKind Provider, int Rank) RankCatalogDevice(OrtEpDevice device, int deviceCount)
    {
        if (!TryMapCatalogEpNameToExecutionProviderKind(device.EpName, out ExecutionProviderKind provider))
        {
            ReportUnmappedCatalogEpName(device.EpName, CatalogEpNameSource.GetEpDeviceForInputs, deviceCount);
            return (ExecutionProviderKind.Cpu, int.MaxValue);
        }

        return (provider, GetCatalogEpHonestyRank(provider, device.HardwareDevice.Type));
    }

    internal static bool TryMapCatalogEpNameToExecutionProviderKind(
        string epName,
        out ExecutionProviderKind executionProviderKind)
    {
        if (TryMapDeviceCandidateCatalogEpName(epName, out executionProviderKind))
        {
            return true;
        }

        return ExactCatalogEpNames.TryGetValue(epName, out executionProviderKind);
    }

    private static bool TryMapDeviceCandidateCatalogEpName(
        string epName,
        out ExecutionProviderKind executionProviderKind)
    {
        if (IsTensorRtRtxDeviceCandidate(epName, OrtHardwareDeviceType.GPU))
        {
            executionProviderKind = ExecutionProviderKind.TensorRTRtx;
            return true;
        }

        if (IsDnnlExecutionProviderName(epName))
        {
            executionProviderKind = ExecutionProviderKind.Dnnl;
            return true;
        }

        if (IsDirectMlDeviceCandidate(epName, OrtHardwareDeviceType.GPU))
        {
            executionProviderKind = ExecutionProviderKind.DirectMl;
            return true;
        }

        executionProviderKind = default;
        return false;
    }

    internal enum CatalogEpNameSource
    {
        GetEpDeviceForInputs,
        GetEpDevices
    }

    internal static void ReportUnmappedCatalogEpName(
        string epName,
        CatalogEpNameSource source,
        int deviceCount)
    {
        if (string.IsNullOrWhiteSpace(epName))
        {
            return;
        }

        if (!WarnedUnmappedCatalogEpNames.TryAdd(epName, 0))
        {
            return;
        }

        _logger?.LogWarning(
            "Unmapped Windows ML catalog EP name '{EpName}' encountered ({Source}, deviceCount={DeviceCount}); treating as CPU only for honesty resolution. Add a mapping in TryMapCatalogEpNameToExecutionProviderKind if this EP is supported. See docs/internal/windows-ml-phase-5-catalog-eps.md.",
            epName,
            source,
            deviceCount);
    }

    private static int GetCatalogEpHonestyRank(ExecutionProviderKind provider, OrtHardwareDeviceType deviceType)
    {
        if (DeviceCatalogEpRank.TryGetValue((provider, deviceType), out int rank))
        {
            return rank;
        }

        return WildcardCatalogEpRank.GetValueOrDefault(provider, 11);
    }
}
