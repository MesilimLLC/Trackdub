using Trackdub.Contracts.ApplicationContracts;

namespace Trackdub.Composition.Runtime;

internal sealed class AutomaticNativeCudaTensorRtWindowsPolicy
    : INativeCudaTensorRtWindowsPolicy
{
    public Task<bool> IsNativeProvidersAllowedOnWindowsAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(OperatingSystem.IsWindows());
    }
}
