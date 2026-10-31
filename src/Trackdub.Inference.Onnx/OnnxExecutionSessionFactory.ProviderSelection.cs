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
    internal sealed record SessionOptionsRequest(
        ExecutionProviderKind Provider,
        WindowsMlExecutionDevicePolicy DevicePolicy,
        IReadOnlyDictionary<string, string>? AdditionalTrtOptions = null,
        bool EnableCudaGraph = false,
        string? LogId = null,
        string? TrtModelPath = null);

    private static SessionOptionsSelection CreateSessionOptions(SessionOptionsRequest request)
    {
        ExecutionProviderKind provider = request.Provider;
        bool useCatalogDevicePolicy = ShouldUseCatalogDevicePolicy(request.DevicePolicy, provider);
        SessionOptions options = CreateBaseSessionOptions(
            useCatalogDevicePolicy ? request.DevicePolicy : WindowsMlExecutionDevicePolicy.Explicit,
            out bool devicePolicyApplied,
            tensorRtRtx: provider is ExecutionProviderKind.TensorRTRtx, logId: request.LogId);

        if (provider is ExecutionProviderKind.Cpu)
        {
            return new SessionOptionsSelection(options, ExecutionProviderKind.Cpu);
        }

        if (useCatalogDevicePolicy && IsCatalogGpuProvider(provider))
        {
            return new SessionOptionsSelection(
                options,
                provider,
                BuildDevicePolicyFallbackReason(request.DevicePolicy, devicePolicyApplied));
        }

        return AppendProviderSpecificSelection(options, request)
            ?? throw new ArgumentOutOfRangeException(nameof(request), provider, "Unsupported execution provider kind.");
    }

    private static SessionOptionsSelection? AppendProviderSpecificSelection(
        SessionOptions options,
        SessionOptionsRequest request) =>
        AppendComputeProviderSelection(options, request)
        ?? AppendCatalogProviderSelection(options, request)
        ?? AppendPlatformProviderSelection(options, request);

    private static SessionOptionsSelection? AppendComputeProviderSelection(
        SessionOptions options,
        SessionOptionsRequest request) =>
        request.Provider switch
        {
            ExecutionProviderKind.DirectMl => CreateDirectMlSelection(options),
            ExecutionProviderKind.Dnnl => CreateDnnlSelection(options),
            ExecutionProviderKind.TensorRTRtx => CreateTensorRtRtxSelection(options, request.AdditionalTrtOptions, request.EnableCudaGraph),
            ExecutionProviderKind.Migraphx => CreateMigraphxSelection(options),
            _ => null,
        };

    private static SessionOptionsSelection? AppendCatalogProviderSelection(
        SessionOptions options,
        SessionOptionsRequest request) =>
        request.Provider switch
        {
            ExecutionProviderKind.CoreMl => CreateCoreMlSelection(options),
            ExecutionProviderKind.OpenVinoCatalog => CreateOpenVinoCatalogSelection(options),
            ExecutionProviderKind.Qnn => CreateQnnSelection(options),
            ExecutionProviderKind.VitisAi => CreateVitisAiSelection(options),
            _ => null,
        };

    private static SessionOptionsSelection? AppendPlatformProviderSelection(
        SessionOptions options,
        SessionOptionsRequest request) =>
        request.Provider switch
        {
            ExecutionProviderKind.Cuda when !OperatingSystem.IsMacOS() => CreateCudaSelection(options),
            ExecutionProviderKind.TensorRt when !OperatingSystem.IsMacOS() => CreateTensorRtSelection(options, request.AdditionalTrtOptions, request.TrtModelPath),
#if LINML
            ExecutionProviderKind.OpenVino when OperatingSystem.IsLinux() => CreateLinuxOpenVinoSelection(options),
#endif
            _ => null,
        };

    private static SessionOptionsSelection CreateDirectMlSelection(SessionOptions options)
    {
        if (!TryAppendDirectMlProvider(options, out string? catalogFailure) &&
            !TryAppendDirectMlProviderDirect(options, out string? directFailure))
        {
            return new SessionOptionsSelection(
                options,
                ExecutionProviderKind.Cpu,
                "Requested dml but DirectML append failed; CPU fallback activated. "
                + $"catalog: {catalogFailure ?? "n/a"}; direct: {directFailure ?? "n/a"}");
        }

        return new SessionOptionsSelection(options, ExecutionProviderKind.DirectMl);
    }

    private static SessionOptionsSelection CreateDnnlSelection(SessionOptions options)
    {
        ExecutionProviderKind selectedProvider = DnnlSessionOptionsExtensions.AppendDnnlOrFallback(
            options,
            out string? failureReason);
        return selectedProvider is ExecutionProviderKind.Dnnl
            ? new SessionOptionsSelection(options, selectedProvider)
            : new SessionOptionsSelection(
                options,
                ExecutionProviderKind.Cpu,
                $"Requested dnnl but AppendExecutionProvider_Dnnl failed: {failureReason ?? "unknown failure"}");
    }

    private static SessionOptionsSelection CreateTensorRtRtxSelection(
        SessionOptions options,
        IReadOnlyDictionary<string, string>? additionalTrtOptions,
        bool enableCudaGraph = false)
    {
        ExecutionProviderKind selectedProvider =
            AppendTensorRtRtxOrFallbackProvider(options, additionalTrtOptions, enableCudaGraph);
        return BuildTensorRtFallbackSelection(options, ExecutionProviderKind.TensorRTRtx, selectedProvider);
    }

    internal static SessionOptionsSelection BuildTensorRtFallbackSelection(
        SessionOptions options,
        ExecutionProviderKind requestedProvider,
        ExecutionProviderKind selectedProvider)
    {
        if (selectedProvider == requestedProvider)
        {
            return new SessionOptionsSelection(options, selectedProvider);
        }

        (string requestedLabel, string deviceLabel) = requestedProvider switch
        {
            ExecutionProviderKind.TensorRTRtx => ("tensorrt-rtx", "TensorRT RTX EP device"),
            ExecutionProviderKind.TensorRt => ("tensorrt", "classic TensorRT EP device"),
            _ => throw new ArgumentOutOfRangeException(nameof(requestedProvider), requestedProvider, "Not a TensorRT provider."),
        };

        return new SessionOptionsSelection(
            options,
            selectedProvider,
            $"Requested {requestedLabel} but {deviceLabel} was unavailable; fell back to {FormatProviderLabel(selectedProvider)}.");
    }

    private static SessionOptionsSelection CreateMigraphxSelection(SessionOptions options)
    {
        ExecutionProviderKind selectedProvider = MigraphxSessionOptionsExtensions.AppendMigraphxOrFallback(options);
        return new SessionOptionsSelection(options, selectedProvider);
    }

    private static SessionOptionsSelection CreateCoreMlSelection(SessionOptions options)
    {
        if (!OperatingSystem.IsMacOS())
            throw new InvalidOperationException("CoreML EP is only available on macOS.");
        options.AppendExecutionProvider_CoreML(GetCoreMlFlags());
        return new SessionOptionsSelection(options, ExecutionProviderKind.CoreMl);
    }

    private static SessionOptionsSelection CreateCudaSelection(SessionOptions options)
    {
        options.AppendExecutionProvider_CUDA(deviceId: 0);
        return new SessionOptionsSelection(options, ExecutionProviderKind.Cuda);
    }

    private static SessionOptionsSelection CreateTensorRtSelection(
        SessionOptions options,
        IReadOnlyDictionary<string, string>? additionalTrtOptions,
        string? trtModelPath = null)
    {
        ExecutionProviderKind selectedProvider = AppendTensorRtOrFallbackProvider(options, additionalTrtOptions, trtModelPath);
        return BuildTensorRtFallbackSelection(options, ExecutionProviderKind.TensorRt, selectedProvider);
    }

#if LINML
    [System.Runtime.Versioning.SupportedOSPlatform("linux")]
    private static SessionOptionsSelection CreateLinuxOpenVinoSelection(SessionOptions options)
    {
        if (!ExecutionProviders.LinMl.LinMlOpenVinoCatalog.Default.TryAppend(
                options, cpuOnly: _openVinoAvailability.UseOpenVinoCpuProxy, out string? failureReason))
        {
            return new SessionOptionsSelection(
                options,
                ExecutionProviderKind.Cpu,
                $"Requested openvino but the OpenVINO plugin EP append failed; CPU fallback activated. {failureReason}");
        }

        return new SessionOptionsSelection(options, ExecutionProviderKind.OpenVino);
    }
#endif

    private static SessionOptionsSelection CreateOpenVinoCatalogSelection(SessionOptions options)
    {
        ExecutionProviderKind selectedProvider =
            WinMlCatalog.WinMlCatalogSessionOptionsExtensions.AppendOpenVinoCatalogOrFallback(options);
        return new SessionOptionsSelection(options, selectedProvider);
    }

    private static SessionOptionsSelection CreateQnnSelection(SessionOptions options)
    {
        ExecutionProviderKind selectedProvider =
            WinMlCatalog.WinMlCatalogSessionOptionsExtensions.AppendQnnOrFallback(options);
        return new SessionOptionsSelection(options, selectedProvider);
    }

    private static SessionOptionsSelection CreateVitisAiSelection(SessionOptions options)
    {
        ExecutionProviderKind selectedProvider =
            WinMlCatalog.WinMlCatalogSessionOptionsExtensions.AppendVitisAiOrFallback(options);
        return new SessionOptionsSelection(options, selectedProvider);
    }

    private static CoreMLFlags GetCoreMlFlags()
    {
        const CoreMLFlags appleSiliconFlags =
            CoreMLFlags.COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE |
            CoreMLFlags.COREML_FLAG_CREATE_MLPROGRAM;

        bool isAppleSilicon = RuntimeInformation.ProcessArchitecture == Architecture.Arm64
            && OperatingSystem.IsMacOS();

        return isAppleSilicon
            ? appleSiliconFlags
            : CoreMLFlags.COREML_FLAG_USE_NONE;
    }

    internal static bool TryAppendDirectMlProvider(SessionOptions options, out string? failureReason) =>
        TryAppendProvider(options, AppendDirectMlProvider, out failureReason);

    // Direct DML path — no WinML catalog/WinAppSDK bootstrap required.
    // Used when the catalog path is unavailable (e.g., test processes, headless runners).
    internal static bool TryAppendDirectMlProviderDirect(SessionOptions options, out string? failureReason) =>
        TryAppendProvider(options, static target => target.AppendExecutionProvider_DML(deviceId: 0), out failureReason);

    private static bool TryAppendProvider(
        SessionOptions options,
        Action<SessionOptions> appendProvider,
        out string? failureReason)
    {
        try
        {
            appendProvider(options);
            failureReason = null;
            return true;
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or InvalidOperationException or DllNotFoundException or EntryPointNotFoundException)
        {
            failureReason = ex.Message;
            return false;
        }
    }
}
