using System.Text.Json;
using Trackdub.Tools;

namespace Trackdub.Benchmarks.Tests;

public sealed class ModelLabCommandOptionsTests
{
    [Fact]
    public void TryParse_ParsesExplicitCandidateAndPaths()
    {
        bool success = ModelLabCommandOptions.TryParse(
            [
                "--model", "openai/whisper-tiny",
                "--model-root", "whisper-tiny-genai",
                "--models-root", "models",
                "--manifest-fragment", "models/manifest-fragments/trackdub-model-lab.manifest.json",
                "--python", ".venv/Scripts/python.exe",
                "--builder", "onnxruntime-genai/src/python/py/models/builder.py",
                "--olive", "olive",
                "--cache", "models/.cache",
                "--benchmark-project", "src/Trackdub.Benchmarks/Trackdub.Benchmarks.csproj",
                "--benchmark-runs", "2",
                "--candidate", "directml-fp16:dml:fp16:DmlExecutionProvider:gpu:dml"
            ],
            TextWriter.Null,
            out ModelLabCommandOptions options);

        Assert.True(success);
        Assert.False(options.ShowHelp);
        Assert.Equal("openai/whisper-tiny", options.HuggingFaceModelId);
        Assert.Equal("whisper-tiny-genai", options.ModelRootName);
        Assert.EndsWith("models", options.ModelsRootPath, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith(Path.Join("manifest-fragments", "trackdub-model-lab.manifest.json"), options.ManifestFragmentPath, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith(Path.Join(".venv", "Scripts", "python.exe"), options.PythonPath, StringComparison.OrdinalIgnoreCase);
        Assert.EndsWith(Path.Join("onnxruntime-genai", "src", "python", "py", "models", "builder.py"), options.OrtGenAiBuilderPath, StringComparison.OrdinalIgnoreCase);
        Assert.False(options.UseOrtGenAiBuilderModule);
        Assert.Equal("olive", options.OliveExecutablePath);
        Assert.EndsWith(Path.Join("models", ".cache"), options.CacheDirectoryPath, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(2, options.BenchmarkRuns);

        ModelLabCandidateOptions candidate = Assert.Single(options.Candidates);
        Assert.Equal("directml-fp16", candidate.Alias);
        Assert.Equal("dml", candidate.BuilderProvider);
        Assert.Equal("fp16", candidate.Precision);
        Assert.Equal("DmlExecutionProvider", candidate.OliveProvider);
        Assert.Equal("gpu", candidate.OliveDevice);
        Assert.Equal("dml", candidate.BenchmarkProvider);
    }

    [Fact]
    public void TryParse_NoBenchmarkFlagSetsSkipBenchmark()
    {
        bool success = ModelLabCommandOptions.TryParse(
            ["--model", "openai/whisper-tiny", "--no-benchmark"],
            TextWriter.Null,
            out ModelLabCommandOptions options);

        Assert.True(success);
        Assert.True(options.SkipBenchmark);
    }

    [Fact]
    public void TryParse_RejectsCandidateWithMissingFields()
    {
        bool success = ModelLabCommandOptions.TryParse(
            ["--candidate", "directml-fp16:dml:fp16"],
            TextWriter.Null,
            out _);

        Assert.False(success);
    }

    [Fact]
    public void TryParse_DefaultTensorRtRtxCandidateUsesWinMlBenchmarkRoute()
    {
        bool success = ModelLabCommandOptions.TryParse(
            ["--model", "openai/whisper-tiny"],
            TextWriter.Null,
            out ModelLabCommandOptions options);

        Assert.True(success);
        ModelLabCandidateOptions candidate = Assert.Single(
            options.Candidates,
            candidate => candidate.Alias.Equals("trt-rtx-fp16", StringComparison.Ordinal));
        Assert.Equal("NvTensorRtRtx", candidate.BuilderProvider);
        Assert.Equal("fp16", candidate.Precision);
        Assert.Equal("NvTensorRTRTXExecutionProvider", candidate.OliveProvider);
        Assert.Equal("gpu", candidate.OliveDevice);
        Assert.Equal("trt-rtx", candidate.BenchmarkProvider);
    }
}

public sealed class ModelLabCommandTests
{
    [Fact]
    public async Task RunAsync_BuildsOliveBenchmarksAndWritesManifestFragment()
    {
        string tempRoot = Path.Join(Path.GetTempPath(), "trackdub-model-lab-tests", Guid.NewGuid().ToString("N"));
        string modelsRoot = Path.Join(tempRoot, "models");
        string fragmentPath = Path.Join(modelsRoot, "manifest-fragments", "trackdub-model-lab.manifest.json");
        string cachePath = Path.Join(tempRoot, "cache");
        var processRunner = new FakeModelLabProcessRunner();
        using var output = new StringWriter();
        using var error = new StringWriter();

        try
        {
            int exitCode = await ModelLabCommand.RunAsync(
                [
                    "--model", "openai/whisper-tiny",
                    "--model-root", "whisper-tiny-genai",
                    "--models-root", modelsRoot,
                    "--manifest-fragment", fragmentPath,
                    "--python", "python.exe",
                    "--builder", Path.Join(tempRoot, "builder.py"),
                    "--olive", "olive.exe",
                    "--cache", cachePath,
                    "--benchmark-project", Path.Join(tempRoot, "Trackdub.Benchmarks.csproj"),
                    "--benchmark-framework", "net11.0-windows10.0.19041.0",
                    "--benchmark-runs", "1",
                    "--candidate", "directml-fp16:dml:fp16:DmlExecutionProvider:gpu:dml"
                ],
                output,
                error,
                processRunner,
                TestContext.Current.CancellationToken);

            Assert.Equal(0, exitCode);
            Assert.True(File.Exists(fragmentPath));
            Assert.Contains("ModelLab complete", output.ToString(), StringComparison.Ordinal);

            Assert.Contains(processRunner.Calls, call =>
                call.Executable.Equals("python.exe", StringComparison.OrdinalIgnoreCase) &&
                call.Arguments.Contains(Path.Join(tempRoot, "builder.py"), StringComparer.OrdinalIgnoreCase) &&
                call.Arguments.Contains("dml", StringComparer.OrdinalIgnoreCase) &&
                call.Arguments.Contains("hf_token=false", StringComparer.OrdinalIgnoreCase));
            Assert.Contains(processRunner.Calls, call =>
                call.Executable.Equals("olive.exe", StringComparison.OrdinalIgnoreCase) &&
                call.Arguments.Contains("optimize", StringComparer.OrdinalIgnoreCase) &&
                call.Arguments.Contains(Path.Join(modelsRoot, "whisper-tiny-genai", "directml-fp16", "encoder.onnx"), StringComparer.OrdinalIgnoreCase) &&
                call.Arguments.Contains(Path.Join(cachePath, "olive", "directml-fp16", "encoder"), StringComparer.OrdinalIgnoreCase) &&
                call.Arguments.Contains("DmlExecutionProvider", StringComparer.OrdinalIgnoreCase) &&
                call.WorkingDirectory.Equals(cachePath, StringComparison.OrdinalIgnoreCase));
            Assert.Contains(processRunner.Calls, call =>
                call.Executable.Equals("python.exe", StringComparison.OrdinalIgnoreCase) &&
                call.Arguments.Any(argument => argument.EndsWith("decompose-whisper-cross-attention.py", StringComparison.OrdinalIgnoreCase)) &&
                call.Arguments.Contains(Path.Join(modelsRoot, "whisper-tiny-genai", "directml-fp16", "decoder.onnx"), StringComparer.OrdinalIgnoreCase));
            Assert.Contains(processRunner.Calls, call =>
                call.Executable.Equals("dotnet", StringComparison.OrdinalIgnoreCase) &&
                call.Arguments.Contains("-p:Platform=x64", StringComparer.OrdinalIgnoreCase) &&
                call.Arguments.Contains("-p:WindowsAppSDKSelfContained=true", StringComparer.OrdinalIgnoreCase) &&
                call.Arguments.Contains("--provider", StringComparer.OrdinalIgnoreCase) &&
                call.Arguments.Contains("dml", StringComparer.OrdinalIgnoreCase));

            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(fragmentPath));
            JsonElement model = Assert.Single(document.RootElement.GetProperty("models").EnumerateArray());
            Assert.Equal("openai/whisper-tiny", model.GetProperty("model_id").GetString());
            Assert.Equal("whisper-genai", model.GetProperty("engine_family").GetString());
            Assert.Equal("../whisper-tiny-genai", model.GetProperty("root_path").GetString());
            Assert.Equal("directml-fp16/encoder.onnx", model.GetProperty("benchmark_entry").GetString());

            JsonElement variant = Assert.Single(model.GetProperty("variants").EnumerateArray());
            Assert.Equal("directml-fp16", variant.GetProperty("alias").GetString());
            Assert.Equal("directml-fp16/encoder.onnx", variant.GetProperty("entry_path").GetString());
            Assert.NotEqual(string.Empty, variant.GetProperty("sha256").GetString());
            Assert.Contains(variant.GetProperty("download_files").EnumerateArray(), file =>
                file.GetString() == "directml-fp16/genai_config.json");
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public async Task RunAsync_UsesInstalledOrtGenAiBuilderModuleByDefault()
    {
        string tempRoot = Path.Join(Path.GetTempPath(), "trackdub-model-lab-tests", Guid.NewGuid().ToString("N"));
        string modelsRoot = Path.Join(tempRoot, "models");
        string fragmentPath = Path.Join(modelsRoot, "manifest-fragments", "trackdub-model-lab.manifest.json");
        var processRunner = new FakeModelLabProcessRunner();

        try
        {
            int exitCode = await ModelLabCommand.RunAsync(
                [
                    "--model", "openai/whisper-tiny",
                    "--model-root", "whisper-tiny-genai",
                    "--models-root", modelsRoot,
                    "--manifest-fragment", fragmentPath,
                    "--python", "python.exe",
                    "--olive", "olive.exe",
                    "--cache", Path.Join(tempRoot, "cache"),
                    "--benchmark-project", Path.Join(tempRoot, "Trackdub.Benchmarks.csproj"),
                    "--benchmark-runs", "1",
                    "--candidate", "cpu-fp32:cpu:fp32:CPUExecutionProvider:cpu:cpu"
                ],
                TextWriter.Null,
                TextWriter.Null,
                processRunner,
                TestContext.Current.CancellationToken);

            Assert.Equal(0, exitCode);
            ModelLabProcessStartInfo pythonCall = Assert.Single(
                processRunner.Calls,
                call => call.Executable.Equals("python.exe", StringComparison.OrdinalIgnoreCase)
                    && call.Arguments.Contains("onnxruntime_genai.models.builder", StringComparer.Ordinal));
            Assert.Collection(
                pythonCall.Arguments.Take(2),
                first => Assert.Equal("-m", first),
                second => Assert.Equal("onnxruntime_genai.models.builder", second));
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public async Task RunAsync_RejectsCandidateWhenBenchmarkReportFailed()
    {
        string tempRoot = Path.Join(Path.GetTempPath(), "trackdub-model-lab-tests", Guid.NewGuid().ToString("N"));
        string modelsRoot = Path.Join(tempRoot, "models");
        string fragmentPath = Path.Join(modelsRoot, "manifest-fragments", "trackdub-model-lab.manifest.json");
        var processRunner = new FakeModelLabProcessRunner
        {
            BenchmarkReportJson = """
                {
                  "Status": "Failed",
                  "RequestedProvider": "dml",
                  "SelectedProvider": "dml",
                  "SupportsExecution": false,
                  "FailureReason": "DirectML catalog execution provider is not visible."
                }
                """
        };
        var error = new StringWriter();

        try
        {
            int exitCode = await ModelLabCommand.RunAsync(
                [
                    "--model", "openai/whisper-tiny",
                    "--model-root", "whisper-tiny-genai",
                    "--models-root", modelsRoot,
                    "--manifest-fragment", fragmentPath,
                    "--python", "python.exe",
                    "--olive", "olive.exe",
                    "--cache", Path.Join(tempRoot, "cache"),
                    "--benchmark-project", Path.Join(tempRoot, "Trackdub.Benchmarks.csproj"),
                    "--benchmark-runs", "1",
                    "--candidate", "directml-fp16:dml:fp16:DmlExecutionProvider:gpu:dml"
                ],
                TextWriter.Null,
                error,
                processRunner,
                TestContext.Current.CancellationToken);

            Assert.Equal(1, exitCode);
            Assert.False(File.Exists(fragmentPath));
            Assert.Contains("benchmark report status was Failed", error.ToString(), StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public async Task RunAsync_RejectsCandidateWhenBenchmarkSelectedProviderDiffers()
    {
        string tempRoot = Path.Join(Path.GetTempPath(), "trackdub-model-lab-tests", Guid.NewGuid().ToString("N"));
        string modelsRoot = Path.Join(tempRoot, "models");
        string fragmentPath = Path.Join(modelsRoot, "manifest-fragments", "trackdub-model-lab.manifest.json");
        var processRunner = new FakeModelLabProcessRunner
        {
            BenchmarkReportJson = """
                {
                  "Status": "Completed",
                  "RequestedProvider": "trt-rtx",
                  "SelectedProvider": "cpu",
                  "SupportsExecution": true
                }
                """
        };
        using var error = new StringWriter();

        try
        {
            int exitCode = await ModelLabCommand.RunAsync(
                [
                    "--model", "openai/whisper-tiny",
                    "--model-root", "whisper-tiny-genai",
                    "--models-root", modelsRoot,
                    "--manifest-fragment", fragmentPath,
                    "--python", "python.exe",
                    "--olive", "olive.exe",
                    "--cache", Path.Join(tempRoot, "cache"),
                    "--benchmark-project", Path.Join(tempRoot, "Trackdub.Benchmarks.csproj"),
                    "--benchmark-runs", "1",
                    "--candidate", "trt-rtx-fp16:NvTensorRtRtx:fp16:NvTensorRTRTXExecutionProvider:gpu:trt-rtx"
                ],
                TextWriter.Null,
                error,
                processRunner,
                TestContext.Current.CancellationToken);

            Assert.Equal(1, exitCode);
            Assert.False(File.Exists(fragmentPath));
            Assert.Contains("did not match requested provider 'trt-rtx'", error.ToString(), StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public async Task RunAsync_RejectsCandidateWhenBenchmarkReportRecordsCpuFallback()
    {
        string tempRoot = Path.Join(Path.GetTempPath(), "trackdub-model-lab-tests", Guid.NewGuid().ToString("N"));
        string modelsRoot = Path.Join(tempRoot, "models");
        string fragmentPath = Path.Join(modelsRoot, "manifest-fragments", "trackdub-model-lab.manifest.json");
        string cachePath = Path.Join(tempRoot, "cache");
        var processRunner = new FakeModelLabProcessRunner
        {
            BenchmarkReportJson =
                """
                {
                  "Status": "Completed",
                  "SelectedProvider": "trt-rtx",
                  "SupportsExecution": true,
                  "Notes": [
                    "TensorRT RTX route fell back to cpu."
                  ]
                }
                """
        };
        using var error = new StringWriter();

        try
        {
            int exitCode = await ModelLabCommand.RunAsync(
                [
                    "--models-root", modelsRoot,
                    "--manifest-fragment", fragmentPath,
                    "--python", "python.exe",
                    "--builder", Path.Join(tempRoot, "builder.py"),
                    "--olive", "olive.exe",
                    "--cache", cachePath,
                    "--benchmark-project", Path.Join(tempRoot, "Trackdub.Benchmarks.csproj"),
                    "--benchmark-runs", "1",
                    "--candidate", "trt-rtx-fp16:NvTensorRtRtx:fp16:NvTensorRTRTXExecutionProvider:gpu:trt-rtx"
                ],
                TextWriter.Null,
                error,
                processRunner,
                TestContext.Current.CancellationToken);

            Assert.Equal(1, exitCode);
            Assert.Contains("benchmark report recorded an execution-provider fallback", error.ToString(), StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public async Task ProgramRunAsync_ModelLabHelpDispatchesToModelLabUsage()
    {
        using var output = new StringWriter();
        using var error = new StringWriter();

        int exitCode = await Trackdub.Tools.Program.RunAsync(["model-lab", "--help"], output, error, TestContext.Current.CancellationToken);

        Assert.Equal(0, exitCode);
        Assert.Contains("Trackdub.Tools model-lab", output.ToString(), StringComparison.Ordinal);
        Assert.DoesNotContain("Trackdub.Tools ingest", output.ToString(), StringComparison.Ordinal);
    }

    private sealed class FakeModelLabProcessRunner : IModelLabProcessRunner
    {
        public List<ModelLabProcessStartInfo> Calls { get; } = [];

        public string? BenchmarkReportJson { get; set; }

        public async Task<int> RunAsync(
            ModelLabProcessStartInfo startInfo,
            TextWriter output,
            TextWriter error,
            CancellationToken cancellationToken)
        {
            Calls.Add(startInfo);

            if (startInfo.Executable.EndsWith("olive.exe", StringComparison.OrdinalIgnoreCase))
            {
                string outputPath = ReadArgumentValue(startInfo.Arguments, "--output_path");
                Directory.CreateDirectory(outputPath);
                await File.WriteAllTextAsync(Path.Join(outputPath, "model.onnx"), "optimized", cancellationToken);
                return 0;
            }

            if (startInfo.Executable.EndsWith("python.exe", StringComparison.OrdinalIgnoreCase))
            {
                if (startInfo.Arguments.Any(argument => argument.EndsWith("decompose-whisper-cross-attention.py", StringComparison.OrdinalIgnoreCase))
                    || startInfo.Arguments.Any(argument => argument.EndsWith("decompose-microsoft-contrib-ops.py", StringComparison.OrdinalIgnoreCase)))
                {
                    return 0;
                }

                string outputDirectory = ReadArgumentValue(startInfo.Arguments, "-o");
                Directory.CreateDirectory(outputDirectory);
                await File.WriteAllTextAsync(Path.Join(outputDirectory, "encoder.onnx"), "encoder", cancellationToken);
                await File.WriteAllTextAsync(Path.Join(outputDirectory, "decoder.onnx"), "decoder", cancellationToken);
                await File.WriteAllTextAsync(Path.Join(outputDirectory, "genai_config.json"), "{}", cancellationToken);
                return 0;
            }

            if (startInfo.Executable.Equals("dotnet", StringComparison.OrdinalIgnoreCase))
            {
                string reportPath = ReadArgumentValue(startInfo.Arguments, "--output");
                string provider = ReadArgumentValue(startInfo.Arguments, "--provider");
                Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
                string reportJson = BenchmarkReportJson ??
                    $$"""
                    {
                      "Status": "Completed",
                      "RequestedProvider": "{{provider}}",
                      "SelectedProvider": "{{provider}}",
                      "SupportsExecution": true
                    }
                    """;
                await File.WriteAllTextAsync(reportPath, reportJson, cancellationToken);
                return 0;
            }

            return 1;
        }

        private static string ReadArgumentValue(IReadOnlyList<string> arguments, string name)
        {
            for (var index = 0; index < arguments.Count - 1; index++)
            {
                if (arguments[index].Equals(name, StringComparison.Ordinal))
                {
                    return arguments[index + 1];
                }
            }

            Assert.Fail($"Expected argument '{name}'.");
            return string.Empty;
        }
    }

    [Fact]
    public async Task RunAsync_SkipsBenchmarkWhenNoBenchmarkFlagSet()
    {
        string tempRoot = Path.Join(Path.GetTempPath(), "trackdub-model-lab-tests", Guid.NewGuid().ToString("N"));
        string modelsRoot = Path.Join(tempRoot, "models");
        string fragmentPath = Path.Join(modelsRoot, "manifest-fragments", "trackdub-model-lab.manifest.json");
        string cachePath = Path.Join(tempRoot, "cache");
        var processRunner = new FakeModelLabProcessRunner();

        try
        {
            int exitCode = await ModelLabCommand.RunAsync(
                [
                    "--model", "openai/whisper-tiny",
                    "--model-root", "whisper-tiny-genai",
                    "--models-root", modelsRoot,
                    "--manifest-fragment", fragmentPath,
                    "--python", "python.exe",
                    "--builder", Path.Join(tempRoot, "builder.py"),
                    "--olive", "olive.exe",
                    "--cache", cachePath,
                    "--benchmark-project", Path.Join(tempRoot, "Trackdub.Benchmarks.csproj"),
                    "--benchmark-runs", "1",
                    "--no-benchmark",
                    "--candidate", "cpu-fp32:cpu:fp32:CPUExecutionProvider:cpu:cpu"
                ],
                TextWriter.Null,
                TextWriter.Null,
                processRunner,
                TestContext.Current.CancellationToken);

            Assert.Equal(0, exitCode);
            Assert.True(File.Exists(fragmentPath));

            Assert.Contains(processRunner.Calls, call =>
                call.Executable.Equals("python.exe", StringComparison.OrdinalIgnoreCase));
            Assert.Contains(processRunner.Calls, call =>
                call.Executable.Equals("olive.exe", StringComparison.OrdinalIgnoreCase));
            Assert.DoesNotContain(processRunner.Calls, call =>
                call.Executable.Equals("dotnet", StringComparison.OrdinalIgnoreCase));
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public async Task RunAsync_WhenNoBenchmarkAndNoVariants_WritesBenchmarkSkippedFailure()
    {
        string tempRoot = Path.Join(Path.GetTempPath(), "trackdub-model-lab-tests", Guid.NewGuid().ToString("N"));
        string modelsRoot = Path.Join(tempRoot, "models");
        string fragmentPath = Path.Join(modelsRoot, "manifest-fragments", "trackdub-model-lab.manifest.json");
        var processRunner = new FakeModelLabProcessRunner();
        using var error = new StringWriter();

        try
        {
            int exitCode = await ModelLabCommand.RunAsync(
                [
                    "--model", "openai/whisper-tiny",
                    "--model-root", "whisper-tiny-genai",
                    "--models-root", modelsRoot,
                    "--manifest-fragment", fragmentPath,
                    "--python", "missing-builder.exe",
                    "--builder", Path.Join(tempRoot, "builder.py"),
                    "--olive", "olive.exe",
                    "--cache", Path.Join(tempRoot, "cache"),
                    "--benchmark-project", Path.Join(tempRoot, "Trackdub.Benchmarks.csproj"),
                    "--benchmark-runs", "1",
                    "--no-benchmark",
                    "--candidate", "cpu-fp32:cpu:fp32:CPUExecutionProvider:cpu:cpu"
                ],
                TextWriter.Null,
                error,
                processRunner,
                TestContext.Current.CancellationToken);

            Assert.Equal(1, exitCode);
            Assert.Contains("No ModelLab candidates produced a manifest variant (benchmarks were skipped).", error.ToString(), StringComparison.Ordinal);
            Assert.DoesNotContain("benchmarked manifest variant", error.ToString(), StringComparison.Ordinal);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }
}
