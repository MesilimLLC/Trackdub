using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Inference.Runtime.TensorRtRtx;

namespace Trackdub.Inference.Onnx.TensorRtRtx;

public sealed record TensorRtRtxBundleEnsureResult(
    bool Succeeded,
    string? InstallDirectory,
    string? Detail);

internal sealed class TensorRtRtxPluginService : ITensorRtRtxProviderBootstrap
{
    // Must equal the canonical EP name: ORT reports OrtEpDevice.EpName as the registration name passed to
    // RegisterExecutionProviderLibrary (per ORT plugin-EP contract). Registering under a custom handle made
    // the device invisible to IsPluginProviderListed / ProviderNames.IsTensorRtRtxDeviceCandidate, which match this name.
    private const string RegistrationName = TensorRtRtxProviderConstants.PluginOrtExecutionProviderName;
    private static readonly SemaphoreSlim RegistrationGate = new(1, 1);
    private static string? registeredProviderLibraryPath;

    private readonly Func<CancellationToken, ValueTask<string?>> explicitPluginDirectoryProvider;
    private readonly Func<CancellationToken, ValueTask<string?>> defaultInstallDirectoryProvider;
    private readonly Func<bool, CancellationToken, ValueTask<TensorRtRtxBundleEnsureResult>>? bundleEnsureAsync;

    public static TensorRtRtxPluginService Shared { get; } = new();

    public TensorRtRtxPluginService()
        : this(
            static _ => ValueTask.FromResult<string?>(null),
            static _ => ValueTask.FromResult<string?>(null),
            bundleEnsureAsync: null)
    {
    }

    public TensorRtRtxPluginService(
        Func<CancellationToken, ValueTask<string?>> explicitPluginDirectoryProvider,
        Func<CancellationToken, ValueTask<string?>> defaultInstallDirectoryProvider,
        Func<bool, CancellationToken, ValueTask<TensorRtRtxBundleEnsureResult>>? bundleEnsureAsync = null)
    {
        this.explicitPluginDirectoryProvider = explicitPluginDirectoryProvider
            ?? throw new ArgumentNullException(nameof(explicitPluginDirectoryProvider));
        this.defaultInstallDirectoryProvider = defaultInstallDirectoryProvider
            ?? throw new ArgumentNullException(nameof(defaultInstallDirectoryProvider));
        this.bundleEnsureAsync = bundleEnsureAsync;
    }

    public async Task<TensorRtRtxBootstrapResult> EnsureRegisteredAsync(
        bool allowProviderDownloads,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        (bool eligible, TensorRtRtxReadinessBlocker hardwareBlocker, string hardwareDetail) = EvaluateHardwareEligibility();
        if (!eligible)
        {
            return new TensorRtRtxBootstrapResult(
                false,
                TensorRtRtxProviderIds.PluginEpAbi,
                hardwareBlocker,
                hardwareDetail);
        }

        // If another instance (e.g. the DI-wired bootstrap) has already registered the plugin and
        // a matching GPU OrtEpDevice is visible, this instance can report success without needing to
        // resolve a plugin directory itself. This keeps mixed Shared/DI instances consistent.
        if (registeredProviderLibraryPath is not null && TensorRtRtxOrtProbe.IsPluginProviderListed())
        {
            return new TensorRtRtxBootstrapResult(
                true,
                TensorRtRtxProviderIds.PluginEpAbi,
                null,
                $"TensorRT RTX EP ABI plugin already registered from '{registeredProviderLibraryPath}'.");
        }

        string? explicitPluginDirectory = await explicitPluginDirectoryProvider(cancellationToken)
            .ConfigureAwait(false);
        string? defaultInstallDirectory = await defaultInstallDirectoryProvider(cancellationToken)
            .ConfigureAwait(false);
        TensorRtRtxPluginResolution resolution = TensorRtRtxPluginLocator.Resolve(
            explicitPluginDirectory,
            defaultInstallDirectory);

        if (!resolution.Succeeded &&
            allowProviderDownloads &&
            bundleEnsureAsync is not null &&
            resolution.Blocker is TensorRtRtxReadinessBlocker.EpNotPresent or TensorRtRtxReadinessBlocker.EpNotReady)
        {
            TensorRtRtxBundleEnsureResult ensureResult = await bundleEnsureAsync(true, cancellationToken)
                .ConfigureAwait(false);
            if (!ensureResult.Succeeded)
            {
                return new TensorRtRtxBootstrapResult(
                    false,
                    TensorRtRtxProviderIds.PluginEpAbi,
                    TensorRtRtxReadinessBlocker.EpDownloadFailed,
                    string.IsNullOrWhiteSpace(ensureResult.Detail)
                        ? "TensorRT RTX EP bundle download did not complete."
                        : ensureResult.Detail);
            }

            explicitPluginDirectory = await explicitPluginDirectoryProvider(cancellationToken).ConfigureAwait(false);
            defaultInstallDirectory = await defaultInstallDirectoryProvider(cancellationToken).ConfigureAwait(false);
            resolution = TensorRtRtxPluginLocator.Resolve(explicitPluginDirectory, defaultInstallDirectory);
        }

        if (!resolution.Succeeded || string.IsNullOrWhiteSpace(resolution.ProviderLibraryPath))
        {
            return new TensorRtRtxBootstrapResult(
                false,
                TensorRtRtxProviderIds.PluginEpAbi,
                resolution.Blocker,
                resolution.Detail);
        }

        TensorRtRtxCudaRuntimeEnsureResult cudaRuntime =
            TensorRtRtxCudaRuntimeBootstrap.TryEnsureLoadedResult(resolution.DirectoryPath);
        if (!cudaRuntime.Succeeded)
        {
            return new TensorRtRtxBootstrapResult(
                false,
                TensorRtRtxProviderIds.PluginEpAbi,
                TensorRtRtxReadinessBlocker.CudaRuntimeMissing,
                cudaRuntime.Detail);
        }

        // ORT loads the plugin library with an altered search path, so process PATH
        // prepends are ignored for the plugin's dependent libraries. Co-locate a CUDA runtime
        // resolved from outside the bundle next to the plugin so registration can resolve it.
        if (TensorRtRtxCudaRuntimeBootstrap.RequiredCudaRuntimeFileName is { } cudaRuntimeFileName)
        {
            EnsureCudartBesidePlugin(resolution.DirectoryPath!, cudaRuntime.LoadedPath, cudaRuntimeFileName);
        }

        try
        {
            await RegistrationGate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                if (!string.Equals(registeredProviderLibraryPath, resolution.ProviderLibraryPath, StringComparison.OrdinalIgnoreCase))
                {
                    OrtEnv.Instance().RegisterExecutionProviderLibrary(
                        RegistrationName,
                        resolution.ProviderLibraryPath);
                    registeredProviderLibraryPath = resolution.ProviderLibraryPath;
                }
            }
            finally
            {
                RegistrationGate.Release();
            }

            bool providerListed = TensorRtRtxOrtProbe.IsPluginProviderListed();
            if (!providerListed)
            {
                return new TensorRtRtxBootstrapResult(
                    false,
                    TensorRtRtxProviderIds.PluginEpAbi,
                    TensorRtRtxReadinessBlocker.OrtProviderUnavailable,
                    $"{TensorRtRtxProviderConstants.PluginOrtExecutionProviderName} plugin library registered, but no matching GPU OrtEpDevice is visible.");
            }

            return new TensorRtRtxBootstrapResult(
                true,
                TensorRtRtxProviderIds.PluginEpAbi,
                null,
                $"TensorRT RTX EP ABI plugin registered from '{resolution.ProviderLibraryPath}'.");
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or DllNotFoundException or BadImageFormatException or InvalidOperationException)
        {
            // A missing dependent module (Win32 126 / dlopen failure) must not collapse into a generic
            // registration failure. Only blame the CUDA runtime where the bundle actually links it
            // dynamically; the Windows cu13 plugin imports just the bundle DLLs, the driver's nvml.dll
            // and the VC++ runtime.
            bool looksLikeMissingModule =
                ex is DllNotFoundException
                || (ex is OnnxRuntimeException &&
                    (ex.Message.Contains("126", StringComparison.Ordinal) ||
                     ex.Message.Contains("cudart", StringComparison.OrdinalIgnoreCase) ||
                     ex.Message.Contains("The specified module could not be found", StringComparison.OrdinalIgnoreCase)));
            bool blameCudaRuntime = looksLikeMissingModule &&
                (TensorRtRtxCudaRuntimeBootstrap.RequiredCudaRuntimeFileName is not null ||
                 ex.Message.Contains("cudart", StringComparison.OrdinalIgnoreCase));

            TensorRtRtxReadinessBlocker blocker = blameCudaRuntime
                ? TensorRtRtxReadinessBlocker.CudaRuntimeMissing
                : TensorRtRtxReadinessBlocker.EpRegisterFailed;

            string detailPrefix = (looksLikeMissingModule, blameCudaRuntime) switch
            {
                (true, true) => "TensorRT RTX EP ABI plugin registration failed: a required native module was not found "
                                + "(likely the bundled CUDA runtime). ",
                (true, false) => "TensorRT RTX EP ABI plugin registration failed: a required native module was not found "
                                 + "(check the bundle's tensorrt_rtx_* DLLs, the NVIDIA driver's nvml.dll, and the VC++ runtime). ",
                _ => "TensorRT RTX EP ABI plugin registration failed: ",
            };

            return new TensorRtRtxBootstrapResult(
                false,
                TensorRtRtxProviderIds.PluginEpAbi,
                blocker,
                detailPrefix + ex.Message);
        }
    }

    private static (bool Eligible, TensorRtRtxReadinessBlocker Blocker, string Detail) EvaluateHardwareEligibility()
    {
        if (OperatingSystem.IsWindows())
        {
            return WindowsNvidiaHardwareGate.Evaluate();
        }

        if (OperatingSystem.IsLinux())
        {
            return LinuxNvidiaHardwareGate.Evaluate();
        }

        return (false, TensorRtRtxReadinessBlocker.PlatformUnsupported,
            "TensorRT RTX EP ABI plugin registration is supported on Windows and Linux only.");
    }

    /// <summary>
    /// Copies a CUDA runtime discovered outside the bundle next to the plugin when missing.
    /// Best-effort: failure is non-fatal because bootstrap already loaded the runtime.
    /// </summary>
    private static void EnsureCudartBesidePlugin(
        string pluginDirectory,
        string? loadedCudartPath,
        string runtimeFileName)
    {
        if (string.IsNullOrWhiteSpace(pluginDirectory) ||
            string.IsNullOrWhiteSpace(loadedCudartPath))
        {
            return;
        }

        try
        {
            string destination = Path.Join(pluginDirectory, runtimeFileName);
            if (File.Exists(destination))
            {
                return;
            }

            string source = Path.GetFullPath(loadedCudartPath);
            if (!File.Exists(source) ||
                string.Equals(source, Path.GetFullPath(destination), StringComparison.OrdinalIgnoreCase))
            {
                return;
            }

            File.Copy(source, destination, overwrite: false);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            // Best-effort co-location only.
        }
    }
}
