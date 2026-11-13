using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Benchmarking;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.ExecutionProviders;
using Trackdub.Inference.Onnx.TensorRtRtx;
using Trackdub.Inference.Runtime.Planning;
using Trackdub.Inference.Runtime.TensorRtRtx;
#if WINDOWS
using Trackdub.Inference.Onnx.WindowsMl;
#endif

namespace Trackdub.Inference.Onnx;

internal static partial class OnnxExecutionSessionFactory
{
    private const string DataRootEnvironmentVariable = "TRACKDUB_DATA_ROOT";
    private const string CacheRootEnvironmentVariable = "TRACKDUB_CACHE_ROOT";

    private static readonly object InitializeLock = new();
    private static int initializeCompleted;
    private static IExecutionProviderBootstrapper _bootstrapper = GetPlatformBootstrapper();
    private static IOpenVinoAvailabilityProvider _openVinoAvailability = NullOpenVinoAvailabilityProvider.Instance;
    private static IWindowsMlEpDevicePolicyProvider _devicePolicyProvider = NullWindowsMlEpDevicePolicyProvider.Instance;

    internal static bool UseOpenVinoCpuProxy => _openVinoAvailability.UseOpenVinoCpuProxy;

    /// <summary>One-time process initialization. The first successful call wins; later calls are ignored.</summary>
    internal static void Initialize(
        IExecutionProviderBootstrapper bootstrapper,
        IWindowsMlEpDevicePolicyProvider? devicePolicyProvider = null,
        ILogger? logger = null,
        IOpenVinoAvailabilityProvider? openVinoAvailabilityProvider = null)
    {
        lock (InitializeLock)
        {
            if (initializeCompleted != 0)
            {
                return;
            }

            _bootstrapper = bootstrapper ?? throw new ArgumentNullException(nameof(bootstrapper));
            _devicePolicyProvider = devicePolicyProvider ?? NullWindowsMlEpDevicePolicyProvider.Instance;
            _openVinoAvailability = openVinoAvailabilityProvider ?? NullOpenVinoAvailabilityProvider.Instance;
            _logger = logger;
            initializeCompleted = 1;
        }
    }

    /// <summary>Resets process-level static state for test isolation. Not for production use.</summary>
    internal static void ResetForTests()
    {
        lock (InitializeLock)
        {
            initializeCompleted = 0;
            _bootstrapper = GetPlatformBootstrapper();
            _devicePolicyProvider = NullWindowsMlEpDevicePolicyProvider.Instance;
            _openVinoAvailability = NullOpenVinoAvailabilityProvider.Instance;
            _logger = null;
            WarnedUnmappedCatalogEpNames.Clear();
        }
    }

    private sealed record BootstrapContext(
        ExecutionProviderBootstrapResult Bootstrap,
        WindowsMlExecutionDevicePolicy DevicePolicy,
        string RequestedProviderLabel);

    private static async Task<BootstrapContext> BootstrapForProviderAsync(
        ExecutionProviderKind provider,
        CancellationToken cancellationToken,
        bool hasExplicitProviderPreference = false)
    {
        ExecutionProviderBootstrapResult bootstrapResult = await BootstrapTrackedAsync(
                provider,
                allowDownloads: true,
                cancellationToken)
            .ConfigureAwait(false);
        WindowsMlExecutionDevicePolicy devicePolicy = await ResolveDevicePolicyAsync(
                cancellationToken,
                hasExplicitProviderPreference)
            .ConfigureAwait(false);
        return new BootstrapContext(bootstrapResult, devicePolicy, ProviderNames.FormatProviderLabel(provider));
    }

    /// <summary>
    /// Readiness-only registration for smoke probes: never downloads, and reports the provider
    /// the bootstrapper can actually select. Bootstrap registration alone does not prove readiness.
    /// </summary>
    internal static async Task<ExecutionProviderBootstrapResult> BootstrapForSmokeAsync(
        ExecutionProviderKind provider,
        CancellationToken cancellationToken)
    {
#if WINDOWS
        WindowsMlOnnxRuntimeNativeResolver.EnsureInitialized();
#endif
        return await BootstrapTrackedAsync(provider, allowDownloads: false, cancellationToken)
            .ConfigureAwait(false);
    }

    private static async Task<ExecutionProviderBootstrapResult> BootstrapTrackedAsync(
        ExecutionProviderKind provider,
        bool allowDownloads,
        CancellationToken cancellationToken)
    {
        using IDisposable phase = BenchmarkPhaseCapture.Start("provider-bootstrap");
        return await _bootstrapper.BootstrapAsync(provider, allowDownloads, cancellationToken)
            .ConfigureAwait(false);
    }

    internal static async Task<WindowsMlExecutionDevicePolicy> ResolveDevicePolicyAsync(
        CancellationToken cancellationToken,
        bool hasExplicitProviderPreference = false)
    {
        if (hasExplicitProviderPreference)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return WindowsMlExecutionDevicePolicy.Explicit;
        }

        try
        {
            return await _devicePolicyProvider.GetPolicyAsync(cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch
        {
            return WindowsMlExecutionDevicePolicy.Explicit;
        }
    }

    internal static IExecutionProviderBootstrapper GetPlatformBootstrapper()
    {
#if WINDOWS
        return new ExecutionProviders.Windows.WindowsExecutionProviderBootstrapper();
#else
        if (OperatingSystem.IsMacOSVersionAtLeast(10, 15))
        {
            return new ExecutionProviders.Mac.MacExecutionProviderBootstrapper();
        }

        if (OperatingSystem.IsLinux())
        {
            return new ExecutionProviders.Linux.LinuxExecutionProviderBootstrapper(
                NullOpenVinoAvailabilityProvider.Instance);
        }

        return new ExecutionProviders.PortableExecutionProviderBootstrapper(
            TensorRtRtxProviderBootstrapFactory.CreateWithDefaultInstallPath(ResolveProductDataRoot()));
#endif
    }

    /// <summary>
    /// Product data root that owns runtime component installs. It honors the same relocation
    /// variables as the engine cache so installed provider components remain discoverable.
    /// </summary>
    internal static string ResolveProductDataRoot()
    {
        string? dataRoot = Environment.GetEnvironmentVariable(DataRootEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(dataRoot))
        {
            dataRoot = Environment.GetEnvironmentVariable(CacheRootEnvironmentVariable);
        }

        if (!string.IsNullOrWhiteSpace(dataRoot))
        {
            return NormalizePath(dataRoot);
        }

        string localAppDataRoot = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (string.IsNullOrWhiteSpace(localAppDataRoot))
        {
            localAppDataRoot = AppContext.BaseDirectory;
        }

        return Path.Join(localAppDataRoot, "Trackdub");
    }

    internal static async Task<bool> CanPolicySelectProviderAsync(
        ExecutionProviderKind provider,
        bool hasExplicitProviderPreference,
        CancellationToken cancellationToken)
    {
        WindowsMlExecutionDevicePolicy policy = await ResolveDevicePolicyAsync(
                cancellationToken,
                hasExplicitProviderPreference)
            .ConfigureAwait(false);
        if (!ProviderNames.ShouldUseCatalogDevicePolicy(policy, provider))
        {
            return false;
        }

        using SessionOptions options = CreateBaseSessionOptions(policy, out bool applied);
        return applied;
    }

    private sealed class NullWindowsMlEpDevicePolicyProvider : IWindowsMlEpDevicePolicyProvider
    {
        internal static readonly NullWindowsMlEpDevicePolicyProvider Instance = new();

        public Task<WindowsMlExecutionDevicePolicy> GetPolicyAsync(CancellationToken cancellationToken = default) =>
            cancellationToken.IsCancellationRequested
                ? Task.FromCanceled<WindowsMlExecutionDevicePolicy>(cancellationToken)
                : Task.FromResult(WindowsMlExecutionDevicePolicy.Explicit);

        public void InvalidateCache()
        {
        }
    }

    private sealed class NullOpenVinoAvailabilityProvider : IOpenVinoAvailabilityProvider
    {
        public static readonly NullOpenVinoAvailabilityProvider Instance = new();
        public bool IsAvailable => false;
        public bool UseOpenVinoCpuProxy => false;
    }
}
