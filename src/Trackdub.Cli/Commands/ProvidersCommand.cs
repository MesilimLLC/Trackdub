using System.CommandLine;
using System.CommandLine.Parsing;

using Trackdub.Cli.Handlers;
using Trackdub.Domain;
using Trackdub.Sdk;

namespace Trackdub.Cli.Commands;

/// <summary>
/// <c>providers list</c> plus <c>providers trt-rtx status|install|smoke</c> for EP discoverability,
/// the TensorRT RTX EP ABI plugin installer, and starter-pack smoke tests, and
/// <c>providers qnn status|install</c> for the Qualcomm QNN Windows ML catalog EP.
/// </summary>
internal static class ProvidersCommand
{
    public static Command Create()
    {
        var providersCommand = new Command("providers", """
            Inspect inference execution providers and install downloadable EP bundles.

            Examples:
              trackdub providers list
              trackdub providers trt-rtx status
              trackdub providers trt-rtx install --accept-license
              trackdub providers trt-rtx smoke
              trackdub providers cuda smoke
              trackdub providers tensorrt smoke
              trackdub providers qnn status
              trackdub providers qnn install --accept-license
            """);

        providersCommand.Add(CreateListCommand());

        var trtRtxCommand = new Command("trt-rtx", "TensorRT RTX EP ABI plugin (Windows/Linux NVIDIA GPU).");
        trtRtxCommand.Add(CreateStatusCommand());
        trtRtxCommand.Add(CreateInstallCommand());
        trtRtxCommand.Add(CreateSmokeCommand());
        trtRtxCommand.Add(CreateVerifyCommand());
        providersCommand.Add(trtRtxCommand);
        providersCommand.Add(CreateProviderCommand(
            "cuda",
            "Native CUDA execution provider (NVIDIA GPU).",
            ExecutionProviderKind.Cuda));
        providersCommand.Add(CreateProviderCommand(
            "tensorrt",
            "Native TensorRT execution provider (NVIDIA GPU).",
            ExecutionProviderKind.TensorRt));
        providersCommand.Add(CreateQnnCommand());

        return providersCommand;
    }

    private static Command CreateProviderCommand(
        string name,
        string description,
        ExecutionProviderKind provider)
    {
        var providerCommand = new Command(name, description);
        providerCommand.Add(CreateProviderSmokeCommand(name, provider));
        return providerCommand;
    }

    private static Command CreateProviderSmokeCommand(string providerName, ExecutionProviderKind provider)
    {
        var command = new Command("smoke", $"""
            Run planner-style ONNX smoke tests for bundled models on the {providerName} execution provider.
            Skips models that are not cached locally.

            Examples:
              trackdub providers {providerName} smoke
              trackdub providers {providerName} smoke --model nemotron
            """);

        var modelFilterOption = new Option<string[]>("--model")
        {
            Description = "Limit smoke targets to labels or model references containing this value (repeatable, case-insensitive).",
            AllowMultipleArgumentsPerToken = true,
        };
        command.Options.Add(modelFilterOption);

        command.SetAction(async (ParseResult parseResult, CancellationToken cancellationToken) =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            TrackdubSessionFactory? factory = CliParseHelpers.TryBuildFactory(parseResult, out int buildExitCode);
            if (factory is null)
            {
                return buildExitCode;
            }

            using (factory)
            {
                return await ProviderSmokeHandler
                    .SmokeAsync(
                        factory,
                        provider,
                        parseResult.GetValue(modelFilterOption),
                        Console.Out,
                        Console.Error,
                        cancellationToken)
                    .ConfigureAwait(false);
            }
        });

        return command;
    }

    private static Command CreateQnnCommand()
    {
        var qnnCommand = new Command("qnn", "Qualcomm QNN Windows ML catalog EP (Snapdragon NPU).");

        var statusCommand = new Command("status", """
            Probe QNN catalog EP readiness without downloading.

            Examples:
              trackdub providers qnn status
            """);
        statusCommand.SetAction(async (ParseResult parseResult, CancellationToken cancellationToken) =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            TrackdubSessionFactory? factory = CliParseHelpers.TryBuildFactory(parseResult, out int buildExitCode);
            if (factory is null)
            {
                return buildExitCode;
            }

            using (factory)
            {
                return await QnnProvidersHandler
                    .StatusAsync(factory, Console.Out, cancellationToken)
                    .ConfigureAwait(false);
            }
        });

        var acceptLicenseOption = new Option<bool>("--accept-license")
        {
            Description =
                "Persist Qualcomm QNN license acceptance in studio settings before install (required unless already accepted in Model Manager)",
            DefaultValueFactory = _ => false,
        };

        var installCommand = new Command("install", """
            Download (when allowed), register, and probe the QNN Windows ML catalog EP.

            Examples:
              trackdub providers qnn install --accept-license
            """)
        {
            acceptLicenseOption,
        };
        installCommand.SetAction(async (ParseResult parseResult, CancellationToken cancellationToken) =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            bool acceptLicense = parseResult.GetValue(acceptLicenseOption);

            TrackdubSessionFactory? factory = CliParseHelpers.TryBuildFactory(parseResult, out int buildExitCode);
            if (factory is null)
            {
                return buildExitCode;
            }

            using (factory)
            {
                return await QnnProvidersHandler
                    .InstallAsync(factory, acceptLicense, Console.Out, Console.Error, cancellationToken)
                    .ConfigureAwait(false);
            }
        });

        qnnCommand.Add(statusCommand);
        qnnCommand.Add(installCommand);
        return qnnCommand;
    }

    private static Command CreateListCommand()
    {
        var command = new Command("list", """
            List every execution-provider kind with canonical tag, aliases, availability, and remediation (JSON).

            Examples:
              trackdub providers list
            """);

        command.SetAction(async (ParseResult parseResult, CancellationToken cancellationToken) =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            TrackdubSessionFactory? factory = CliParseHelpers.TryBuildFactory(parseResult, out int buildExitCode);
            if (factory is null)
            {
                return buildExitCode;
            }

            using (factory)
            {
                return await ProvidersListHandler
                    .ListAsync(factory, Console.Out, cancellationToken)
                    .ConfigureAwait(false);
            }
        });

        return command;
    }

    private static Command CreateStatusCommand()
    {
        var command = new Command("status", """
            Probe TensorRT RTX plugin readiness without downloading.

            Examples:
              trackdub providers trt-rtx status
            """);

        command.SetAction(async (ParseResult parseResult, CancellationToken cancellationToken) =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            TrackdubSessionFactory? factory = CliParseHelpers.TryBuildFactory(parseResult, out int buildExitCode);
            if (factory is null)
            {
                return buildExitCode;
            }

            using (factory)
            {
                return await TrtRtxProvidersHandler
                    .StatusAsync(factory, Console.Out, cancellationToken)
                    .ConfigureAwait(false);
            }
        });

        return command;
    }

    private static Command CreateInstallCommand()
    {
        var acceptLicenseOption = new Option<bool>("--accept-license")
        {
            Description =
                "Persist NVIDIA TensorRT RTX license acceptance in studio settings before install (required unless already accepted in Model Manager)",
            DefaultValueFactory = _ => false,
        };

        var command = new Command("install", """
            Download (when allowed), register, and probe the TensorRT RTX EP ABI plugin.

            Examples:
              trackdub providers trt-rtx install --accept-license
            """)
        {
            acceptLicenseOption,
        };

        command.SetAction(async (ParseResult parseResult, CancellationToken cancellationToken) =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            bool acceptLicense = parseResult.GetValue(acceptLicenseOption);

            TrackdubSessionFactory? factory = CliParseHelpers.TryBuildFactory(parseResult, out int buildExitCode);
            if (factory is null)
            {
                return buildExitCode;
            }

            using (factory)
            {
                return await TrtRtxProvidersHandler
                    .InstallAsync(factory, acceptLicense, Console.Out, Console.Error, cancellationToken)
                    .ConfigureAwait(false);
            }
        });

        return command;
    }

    private static Command CreateSmokeCommand()
    {
        var command = new Command("smoke", """
            Run planner-style ONNX smoke tests for bundled models not in the turbo starter-pack catalog.
            Skips Kokoro and models that are not cached locally.

            Examples:
              trackdub providers trt-rtx smoke
              trackdub providers trt-rtx smoke --model nemotron
            """);

        var modelFilterOption = new Option<string[]>("--model")
        {
            Description = "Limit smoke targets to labels or model references containing this value (repeatable, case-insensitive).",
            AllowMultipleArgumentsPerToken = true,
        };
        command.Options.Add(modelFilterOption);

        command.SetAction(async (ParseResult parseResult, CancellationToken cancellationToken) =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            TrackdubSessionFactory? factory = CliParseHelpers.TryBuildFactory(parseResult, out int buildExitCode);
            if (factory is null)
            {
                return buildExitCode;
            }

            using (factory)
            {
                return await TrtRtxProvidersHandler
                    .SmokeAsync(
                        factory,
                        parseResult.GetValue(modelFilterOption),
                        Console.Out,
                        Console.Error,
                        cancellationToken)
                    .ConfigureAwait(false);
            }
        });

        return command;
    }

    private static Command CreateVerifyCommand()
    {
        var command = new Command("verify", """
            Run the real TRT RTX smoke path against an explicit ONNX entry path (e.g. an
            Olive-recipe-staged output directory) instead of the model cache. Confirms the
            model loads and the effective provider is TensorRT RTX, not a silent CPU/DirectML
            fallback. Used by Olive validator scripts before trusting staged output.

            Examples:
              trackdub providers trt-rtx verify --model cgus/diar_streaming_sortformer_4spk-v2.1-onnx --entry build/sortformer-4spk-onnx-trtrtx-validated-fp16/onnx/model.onnx
            """);

        var modelOption = new Option<string>("--model")
        {
            Description = "Bundled manifest model id whose engine family/stage govern the smoke test.",
            Required = true,
        };
        var entryOption = new Option<string>("--entry")
        {
            Description = "Path to the ONNX entry file to verify (typically an Olive-staged output).",
            Required = true,
        };
        var variantOption = new Option<string?>("--variant")
        {
            Description = "Variant alias to record in the smoke request. Defaults to 'default'.",
        };
        command.Options.Add(modelOption);
        command.Options.Add(entryOption);
        command.Options.Add(variantOption);

        command.SetAction(async (ParseResult parseResult, CancellationToken cancellationToken) =>
        {
            cancellationToken.ThrowIfCancellationRequested();

            TrackdubSessionFactory? factory = CliParseHelpers.TryBuildFactory(parseResult, out int buildExitCode);
            if (factory is null)
            {
                return buildExitCode;
            }

            using (factory)
            {
                return await TrtRtxProvidersHandler
                    .VerifyAsync(
                        factory,
                        parseResult.GetValue(modelOption)!,
                        parseResult.GetValue(entryOption)!,
                        parseResult.GetValue(variantOption),
                        Console.Out,
                        cancellationToken)
                    .ConfigureAwait(false);
            }
        });

        return command;
    }
}
