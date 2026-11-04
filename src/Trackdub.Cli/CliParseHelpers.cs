using System.CommandLine;
using System.CommandLine.Parsing;

using Trackdub.Contracts;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Domain;
using Trackdub.Sdk;

namespace Trackdub.Cli;

internal static class CliParseHelpers
{
    internal static IReadOnlyList<string> SupportedExecutionProviderKeys => ExecutionProviderTokens.CliTags;

    internal static string FormatSupportedExecutionProviders() =>
        ExecutionProviderTokens.FormatSupportedCliTags();

    internal static T? GetGlobalOptionValue<T>(ParseResult parseResult, string optionName)
    {
        if (FindGlobalOption(parseResult, optionName) is not Option<T> option)
        {
            return default;
        }

        return parseResult.GetValue(option);
    }

    internal static bool TryGetExplicitGlobalOptionValue<T>(
        ParseResult parseResult,
        string optionName,
        out T? value)
    {
        if (FindGlobalOption(parseResult, optionName) is not Option<T> option
            || !HasExplicitOptionToken(parseResult, optionName))
        {
            value = default;
            return false;
        }

        value = parseResult.GetValue(option);
        return true;
    }

    /// <summary>
    /// Resolves the execution provider and Windows ML device policy using the precedence
    /// explicit CLI flags &gt; preset values &gt; application defaults.
    /// When the user explicitly passed <c>--execution-provider</c> / <c>--device-policy</c> those
    /// win; otherwise values stored in the loaded <paramref name="preset"/> are used.
    /// </summary>
    internal static void ResolvePresetExecutionPreferences(
        ParseResult parseResult,
        PipelinePreset? preset,
        out string? executionProvider,
        out string? devicePolicy)
    {
        bool executionProviderExplicit = TryGetExplicitGlobalOptionValue<string>(
                parseResult, "execution-provider", out string? cliExecutionProvider)
            && !string.IsNullOrWhiteSpace(cliExecutionProvider);
        bool devicePolicyExplicit = TryGetExplicitGlobalOptionValue<string>(
                parseResult, "device-policy", out string? cliDevicePolicy)
            && !string.IsNullOrWhiteSpace(cliDevicePolicy);

        executionProvider = executionProviderExplicit ? cliExecutionProvider : preset?.ExecutionProvider;
        devicePolicy = devicePolicyExplicit
            ? cliDevicePolicy
            : preset?.DevicePolicy ?? TryReadDiskDevicePolicyKey();

        // --prefer-gpu / --require-gpu fill auto/empty EP when CLI/preset did not pin one.
        bool requireGpu = false;
        try
        {
            requireGpu = GetGlobalOptionValue<bool>(parseResult, "require-gpu");
        }
        catch
        {
            // Option not registered (some unit-test parse graphs); leave false.
        }

        bool preferGpu = false;
        try
        {
            preferGpu = GetGlobalOptionValue<bool>(parseResult, "prefer-gpu");
        }
        catch
        {
            // ignore
        }

        (executionProvider, _, _) = CliGpuPreference.Apply(
            executionProvider,
            requireExecutionProvider: false,
            preferGpu,
            requireGpu);
    }

    /// <summary>
    /// Reads <c>windowsMlExecutionDevicePolicy</c> from studio <c>settings.json</c> so headless
    /// runs honor desktop WinML policy prefs when no explicit <c>--device-policy</c> / preset is set.
    /// </summary>
    internal static string? TryReadDiskDevicePolicyKey()
    {
        try
        {
            string settingsPath = Path.Join(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "Trackdub",
                "settings.json");
            if (!File.Exists(settingsPath))
            {
                return null;
            }

            using FileStream stream = File.OpenRead(settingsPath);
            using System.Text.Json.JsonDocument doc = System.Text.Json.JsonDocument.Parse(stream);
            if (doc.RootElement.TryGetProperty("windowsMlExecutionDevicePolicy", out System.Text.Json.JsonElement value)
                && value.ValueKind == System.Text.Json.JsonValueKind.String)
            {
                string? key = value.GetString();
                return string.IsNullOrWhiteSpace(key) ? null : key.Trim();
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or System.Text.Json.JsonException)
        {
            // Disk prefs are best-effort; factory defaults remain Explicit.
        }

        return null;
    }

    private static Option? FindGlobalOption(ParseResult parseResult, string optionName)
    {
        Command rootCommand = GetRootCommand(parseResult);
        string dashedName = optionName.StartsWith("--", StringComparison.Ordinal)
            ? optionName
            : "--" + optionName;

        return rootCommand.Options.FirstOrDefault(o =>
            o.Name.Equals(dashedName, StringComparison.OrdinalIgnoreCase)
            || o.Name.Equals(optionName, StringComparison.OrdinalIgnoreCase));
    }

    private static bool HasExplicitOptionToken(ParseResult parseResult, string optionName)
    {
        string dashedName = optionName.StartsWith("--", StringComparison.Ordinal)
            ? optionName
            : "--" + optionName;

        return parseResult.Tokens.Any(token => MatchesExplicitOptionToken(token.Value, dashedName, optionName));
    }

    private static bool MatchesExplicitOptionToken(string tokenValue, string dashedName, string optionName)
    {
        if (string.Equals(tokenValue, dashedName, StringComparison.OrdinalIgnoreCase)
            || string.Equals(tokenValue, optionName, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        // System.CommandLine accepts --option=value and --option:value as single tokens.
        return tokenValue.StartsWith(dashedName + "=", StringComparison.OrdinalIgnoreCase)
               || tokenValue.StartsWith(dashedName + ":", StringComparison.OrdinalIgnoreCase);
    }

    internal static TrackdubBuilder ApplyModelDirectory(TrackdubBuilder builder, string? modelDirectory)
    {
        if (modelDirectory is null)
        {
            return builder;
        }

        string resolvedDirectory = Path.GetFullPath(UserPathText.Normalize(modelDirectory));
        try
        {
            Directory.CreateDirectory(resolvedDirectory);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            throw new DirectoryNotFoundException(
                $"Model directory could not be created: {resolvedDirectory}",
                ex);
        }

        return builder
            .WithModelDirectory(resolvedDirectory)
            .WithModelCacheDirectory(resolvedDirectory);
    }

    internal static TrackdubSessionFactory? TryBuildFactory(ParseResult parseResult, out int exitCode)
    {
        string? modelDirectory = GetGlobalOptionValue<string?>(parseResult, "model-directory");
        string? executionProvider = GetGlobalOptionValue<string?>(parseResult, "execution-provider");
        string? devicePolicy = GetGlobalOptionValue<string?>(parseResult, "device-policy");
        bool requireExecutionProvider = GetGlobalOptionValue<bool>(parseResult, "require-execution-provider");

        (executionProvider, requireExecutionProvider, int? gpuError) = CliGpuPreference.Apply(
            parseResult,
            executionProvider,
            requireExecutionProvider);
        if (gpuError is int code)
        {
            ReportGpuPreferenceConflict();
            exitCode = code;
            return null;
        }

        return TryBuildFactory(
            modelDirectory,
            executionProvider,
            devicePolicy,
            out exitCode,
            requireExecutionProvider);
    }

    /// <summary>
    /// The execution-provider pin the global options resolve to (after --prefer-gpu/--require-gpu),
    /// for commands that pick a provider per model instead of through the planner. Call after
    /// <see cref="TryBuildFactory(ParseResult, out int)"/>, which reports invalid values.
    /// </summary>
    internal static bool TryResolveExecutionProviderPin(
        ParseResult parseResult,
        out ExecutionProviderKind? provider,
        out bool requireExecutionProvider)
    {
        string? executionProvider = GetGlobalOptionValue<string?>(parseResult, "execution-provider");
        requireExecutionProvider = GetGlobalOptionValue<bool>(parseResult, "require-execution-provider");
        (executionProvider, requireExecutionProvider, int? gpuError) = CliGpuPreference.Apply(
            parseResult,
            executionProvider,
            requireExecutionProvider);
        provider = null;
        return gpuError is null && TryParseExecutionProvider(executionProvider, out provider, out _);
    }

    private static void ReportGpuPreferenceConflict()
    {
        CliErrorReporter.ReportValidationError(
            ErrorCode.InvalidArgument,
            "Options '--require-gpu' and '--execution-provider cpu' cannot both be set.",
            "--require-gpu");
    }

    internal static TrackdubSessionFactory? TryBuildFactoryForPresetLoad(ParseResult parseResult, out int exitCode)
    {
        string? modelDirectory = GetGlobalOptionValue<string?>(parseResult, "model-directory");
        string? executionProvider = GetGlobalOptionValue<string?>(parseResult, "execution-provider");
        string? devicePolicy = GetGlobalOptionValue<string?>(parseResult, "device-policy");
        // Preset-load factories do not need GPU hard-require; still honor --prefer-gpu soft pin.
        (executionProvider, _, int? gpuError) = CliGpuPreference.Apply(
            parseResult,
            executionProvider,
            requireExecutionProvider: false);
        if (gpuError is int code)
        {
            exitCode = code;
            return null;
        }

        return TryBuildFactory(
            modelDirectory,
            executionProvider,
            devicePolicy,
            out exitCode,
            requireExecutionProvider: false);
    }

    internal static TrackdubSessionFactory? TryBuildFactory(string? modelDirectory, out int exitCode) =>
        TryBuildFactory(
            modelDirectory,
            executionProvider: null,
            devicePolicy: null,
            out exitCode,
            requireExecutionProvider: false);

    internal static TrackdubSessionFactory? TryBuildFactory(
        string? modelDirectory,
        string? executionProvider,
        string? devicePolicy,
        out int exitCode) =>
        TryBuildFactory(
            modelDirectory,
            executionProvider,
            devicePolicy,
            out exitCode,
            requireExecutionProvider: false);

    internal static TrackdubSessionFactory? TryBuildFactory(
        ParseResult parseResult,
        string? modelDirectory,
        string? executionProvider,
        string? devicePolicy,
        out int exitCode)
    {
        bool requireExecutionProvider = GetGlobalOptionValue<bool>(parseResult, "require-execution-provider");
        (executionProvider, requireExecutionProvider, int? gpuError) = CliGpuPreference.Apply(
            parseResult,
            executionProvider,
            requireExecutionProvider);
        if (gpuError is int code)
        {
            ReportGpuPreferenceConflict();
            exitCode = code;
            return null;
        }

        return TryBuildFactory(
            modelDirectory,
            executionProvider,
            devicePolicy,
            out exitCode,
            requireExecutionProvider);
    }

    internal static TrackdubSessionFactory? TryBuildFactory(
        string? modelDirectory,
        string? executionProvider,
        string? devicePolicy,
        out int exitCode,
        bool requireExecutionProvider)
    {
        exitCode = Program.ExitSuccess;

        if (!TryParseExecutionProvider(executionProvider, out ExecutionProviderKind? providerKind, out string? parseWarning))
        {
            // parseWarning carries targeted remediation (e.g. Windows cuda/tensorrt rejection);
            // fall back to the generic unknown-provider message only when it is empty.
            CliErrorReporter.ReportValidationError(
                ErrorCode.InvalidArgument,
                !string.IsNullOrWhiteSpace(parseWarning)
                    ? parseWarning
                    : $"Unknown execution provider: '{executionProvider}'. Expected one of: {FormatSupportedExecutionProviders()}.",
                "--execution-provider");
            exitCode = Program.ExitArgumentError;
            return null;
        }

        if (!string.IsNullOrWhiteSpace(parseWarning))
        {
            Console.Error.WriteLine(parseWarning);
        }

        if (providerKind is ExecutionProviderKind requestedKind
            && !IsProviderSupportedInThisBuild(requestedKind))
        {
            Console.Error.WriteLine(
                $"Warning: execution provider '{ExecutionProviderTokens.ToCanonicalTag(requestedKind)}' "
                + "cannot run in this build. DirectML and Windows ML catalog providers require the "
                + "net11.0-windows10.0.19041.0 target; this build supports trt-rtx and cpu.");
        }

        // CLI/preset already merged by ResolvePresetExecutionPreferences; when still empty,
        // fall through to studio settings.json WinML device policy.
        string? effectiveDevicePolicy = string.IsNullOrWhiteSpace(devicePolicy)
            ? TryReadDiskDevicePolicyKey()
            : devicePolicy;

        if (!TryParseDevicePolicy(effectiveDevicePolicy, out WindowsMlExecutionDevicePolicy resolvedDevicePolicy))
        {
            CliErrorReporter.ReportValidationError(
                ErrorCode.InvalidArgument,
                $"Unknown device policy: '{effectiveDevicePolicy}'. Expected one of: {WindowsMlExecutionDevicePolicySettings.FormatSupportedKeys()}.",
                "--device-policy");
            exitCode = Program.ExitArgumentError;
            return null;
        }

        if (requireExecutionProvider && providerKind is null)
        {
            CliErrorReporter.ReportValidationError(
                ErrorCode.InvalidArgument,
                "--require-execution-provider needs a non-auto --execution-provider Kind pin.",
                "--require-execution-provider");
            exitCode = Program.ExitArgumentError;
            return null;
        }

        try
        {
            TrackdubBuilder builder = ApplyModelDirectory(new TrackdubBuilder(), modelDirectory)
                .WithWindowsMlExecutionDevicePolicy(resolvedDevicePolicy);

            if (providerKind is ExecutionProviderKind kind)
            {
                builder = builder.WithExecutionProvider(kind, requireExecutionProvider);
            }

            TrackdubSessionFactory factory = builder.Build();
            CliLoggingBootstrap.EnsureReady(factory);
            return factory;
        }
        catch (DirectoryNotFoundException ex)
        {
            CliErrorReporter.ReportValidationError(
                ErrorCode.InvalidArgument,
                ex.Message,
                "--model-directory");
            exitCode = Program.ExitArgumentError;
            return null;
        }
    }

    // Mirrors OnnxRuntimeBuildCapabilities in Trackdub.Inference.Onnx; kept local because the
    // layering rules keep Inference.Onnx compile-private to Composition. WINDOWS is defined by
    // this project for the net11.0-windows10.0.19041.0 target only.
    private static bool IsProviderSupportedInThisBuild(ExecutionProviderKind kind) =>
        kind switch
        {
            ExecutionProviderKind.DirectMl or
            ExecutionProviderKind.OpenVinoCatalog or
            ExecutionProviderKind.Qnn or
            ExecutionProviderKind.VitisAi => BuildHasWindowsMlRoutes,
            ExecutionProviderKind.Migraphx => BuildHasWindowsMlRoutes || OperatingSystem.IsLinux(),
            ExecutionProviderKind.CoreMl => OperatingSystem.IsMacOS(),
            _ => true,
        };

#if WINDOWS
    private static bool BuildHasWindowsMlRoutes { get; } = true;
#else
    private static bool BuildHasWindowsMlRoutes { get; } = false;
#endif

    internal static bool TryParseDevicePolicy(string? value, out WindowsMlExecutionDevicePolicy policy)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            policy = WindowsMlExecutionDevicePolicy.Explicit;
            return true;
        }

        return WindowsMlExecutionDevicePolicySettings.TryParseKey(value, out policy);
    }

    /// <summary>
    /// Parses a CLI/preset execution-provider token. Empty or auto yields <c>null</c> kind.
    /// On Windows, <c>cuda</c> and <c>tensorrt</c> are rejected with remediation (native
    /// CUDA/TensorRT are not wired on Windows); use <c>trt-rtx</c> instead.
    /// </summary>
    internal static bool TryParseExecutionProvider(
        string? value,
        out ExecutionProviderKind? kind,
        out string? warning)
    {
        warning = null;
        kind = null;

        if (!ExecutionProviderTokens.TryParseCli(value, out kind))
        {
            return false;
        }

        if (kind is ExecutionProviderKind parsedKind)
        {
            if (OperatingSystem.IsWindows()
                && parsedKind is ExecutionProviderKind.Cuda or ExecutionProviderKind.TensorRt)
            {
                string tag = ExecutionProviderTokens.ToCanonicalTag(parsedKind);
                warning =
                    $"Execution provider '{tag}' is not available on Windows: native CUDA/TensorRT "
                    + "are wired for Linux only. Use --execution-provider trt-rtx for NVIDIA "
                    + "acceleration on Windows, or run on Linux for native CUDA.";
                kind = null;
                return false;
            }

            kind = ExecutionProviderTokens.ResolvePlatformPin(parsedKind, out _);
        }

        return true;
    }

    /// <summary>
    /// Legacy overload used by preset validation that only needs accept/reject.
    /// </summary>
    internal static bool TryParseExecutionProvider(string? value, out ExecutionProviderPreference preference)
    {
        if (!TryParseExecutionProvider(value, out ExecutionProviderKind? kind, out _))
        {
            preference = ExecutionProviderPreference.Auto;
            return false;
        }

        preference = ExecutionProviderPreferenceMapping.ToLegacyPreference(kind);

        // Non-legacy kinds still count as valid tokens for presets.
        return true;
    }

    private static Command GetRootCommand(ParseResult parseResult)
    {
        SymbolResult current = parseResult.RootCommandResult;
        while (current.Parent is not null)
        {
            current = current.Parent;
        }

        return ((CommandResult)current).Command;
    }
}
