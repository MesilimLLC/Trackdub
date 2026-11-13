using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;

namespace Trackdub.Composition.Runtime;

public sealed class StudioSettingsNativeCudaTensorRtWindowsPolicy(IStudioSettingsService settingsService)
    : INativeCudaTensorRtWindowsPolicy
{
    private readonly IStudioSettingsService settingsService =
        settingsService ?? throw new ArgumentNullException(nameof(settingsService));

    public async Task<bool> IsNativeProvidersAllowedOnWindowsAsync(CancellationToken cancellationToken = default)
    {
        if (!OperatingSystem.IsWindows())
        {
            return false;
        }

        StudioSettings settings = await settingsService.LoadAsync(cancellationToken).ConfigureAwait(false);
        return settings.AllowNativeCudaTensorRtOnWindows;
    }
}
