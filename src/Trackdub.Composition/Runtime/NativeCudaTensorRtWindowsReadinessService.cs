using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Inference.Runtime.NativeCudaTensorRt;

namespace Trackdub.Composition.Runtime;

public sealed class NativeCudaTensorRtWindowsReadinessService(
    INativeCudaTensorRtWindowsPolicy policy,
    INativeCudaTensorRtWindowsReadinessProbe readinessProbe)
    : INativeCudaTensorRtWindowsReadinessService
{
    private readonly INativeCudaTensorRtWindowsPolicy policy =
        policy ?? throw new ArgumentNullException(nameof(policy));
    private readonly INativeCudaTensorRtWindowsReadinessProbe readinessProbe =
        readinessProbe ?? throw new ArgumentNullException(nameof(readinessProbe));

    public async Task<NativeCudaTensorRtWindowsReadinessSnapshot> ProbeAsync(
        CancellationToken cancellationToken = default)
    {
        bool allowNative = await policy.IsNativeProvidersAllowedOnWindowsAsync(cancellationToken).ConfigureAwait(false);
        NativeCudaTensorRtWindowsReadinessReport report = await readinessProbe
            .ProbeAsync(allowNative, cancellationToken)
            .ConfigureAwait(false);

        if (!report.IsSupportedPlatform)
        {
            return new NativeCudaTensorRtWindowsReadinessSnapshot(
                PanelVisible: false,
                IsSettingEnabled: false,
                CudaStatusLabel: "Unavailable",
                CudaDetail: report.CudaDetail,
                TensorRtStatusLabel: "Unavailable",
                TensorRtDetail: report.TensorRtDetail,
                SettingHint: null,
                CudaInstallHint: null,
                TensorRtInstallHint: null);
        }

        return new NativeCudaTensorRtWindowsReadinessSnapshot(
            PanelVisible: true,
            IsSettingEnabled: report.IsSettingEnabled,
            CudaStatusLabel: report.IsCudaReady ? "Ready" : "Not ready",
            CudaDetail: report.CudaDetail,
            TensorRtStatusLabel: report.IsTensorRtReady ? "Ready" : "Not ready",
            TensorRtDetail: report.TensorRtDetail,
            SettingHint: report.IsSettingEnabled
                ? null
                : NativeCudaTensorRtWindowsProviderConstants.SettingDisabledHint,
            CudaInstallHint: report.CudaInstallHint,
            TensorRtInstallHint: report.TensorRtInstallHint);
    }
}
