using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;

namespace Trackdub.Tools;

public static class ModelLabCommand
{
    public static async Task<int> RunAsync(
        string[] args,
        TextWriter output,
        TextWriter error,
        CancellationToken cancellationToken) =>
        await RunAsync(args, output, error, new ModelLabProcessRunner(), cancellationToken).ConfigureAwait(false);

    public static async Task<int> RunAsync(
        string[] args,
        TextWriter output,
        TextWriter error,
        IModelLabProcessRunner processRunner,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(error);
        ArgumentNullException.ThrowIfNull(processRunner);

        if (!ModelLabCommandOptions.TryParse(args, error, out ModelLabCommandOptions options))
        {
            WriteUsage(error);
            return 1;
        }

        if (options.ShowHelp)
        {
            WriteUsage(output);
            return 0;
        }

        var variants = new List<ModelLabVariantResult>();
        var failedCandidates = new List<string>();

        foreach (ModelLabCandidateOptions candidate in options.Candidates)
        {
            output.WriteLine($"ModelLab candidate: {candidate.Alias}");
            ModelLabCandidateResult result = await RunCandidateAsync(
                options,
                candidate,
                processRunner,
                output,
                error,
                cancellationToken).ConfigureAwait(false);

            if (result.Variant is not null)
            {
                variants.Add(result.Variant);
                output.WriteLine($"  ready: {result.Variant.EntryPath}");
            }
            else
            {
                failedCandidates.Add(candidate.Alias);
                error.WriteLine($"  failed: {candidate.Alias} ({result.Error})");
            }
        }

        if (variants.Count == 0)
        {
            error.WriteLine(options.SkipBenchmark
                ? "No ModelLab candidates produced a manifest variant (benchmarks were skipped)."
                : "No ModelLab candidates produced a benchmarked manifest variant.");
            return 1;
        }

        await ModelLabManifestFragmentWriter.WriteAsync(
            options,
            variants,
            cancellationToken).ConfigureAwait(false);

        output.WriteLine($"ModelLab complete: {variants.Count} manifest variant(s) written to {options.ManifestFragmentPath}");
        return failedCandidates.Count == 0 ? 0 : 1;
    }

    private static async Task<ModelLabCandidateResult> RunCandidateAsync(
        ModelLabCommandOptions options,
        ModelLabCandidateOptions candidate,
        IModelLabProcessRunner processRunner,
        TextWriter output,
        TextWriter error,
        CancellationToken cancellationToken)
    {
        string variantDirectory = Path.Join(options.ModelsRootPath, options.ModelRootName, candidate.Alias);
        string entryPath = Path.Join(variantDirectory, "encoder.onnx");
        string benchmarkReportPath = Path.Join(variantDirectory, "benchmark-report.json");

        Directory.CreateDirectory(variantDirectory);
        Directory.CreateDirectory(options.CacheDirectoryPath);

        var builderArguments = new List<string>();
        if (options.UseOrtGenAiBuilderModule)
        {
            builderArguments.Add("-m");
        }

        builderArguments.Add(options.OrtGenAiBuilderPath);
        builderArguments.AddRange(
        [
            "-m",
            options.HuggingFaceModelId,
            "-o",
            variantDirectory,
            "-p",
            candidate.Precision,
            "-e",
            candidate.BuilderProvider,
            "-c",
            options.CacheDirectoryPath,
            "--extra_options",
            "hf_token=false"
        ]);

        int builderExitCode = await processRunner.RunAsync(
            new ModelLabProcessStartInfo(
                options.PythonPath,
                builderArguments,
                options.RepositoryRootPath),
            output,
            error,
            cancellationToken).ConfigureAwait(false);

        if (builderExitCode != 0)
        {
            return new ModelLabCandidateResult(null, $"ORT GenAI builder exited with code {builderExitCode}");
        }

        if (!File.Exists(entryPath))
        {
            return new ModelLabCandidateResult(null, $"expected entry file '{entryPath}' was not generated");
        }

        ModelLabCandidateResult? oliveResult = await OptimizeGeneratedOnnxComponentsAsync(
            options,
            candidate,
            variantDirectory,
            processRunner,
            output,
            error,
            cancellationToken).ConfigureAwait(false);
        if (oliveResult is not null)
        {
            return oliveResult;
        }

        ModelLabCandidateResult? directMlGraphResult = await DecomposeWhisperCrossAttentionForDirectMlAsync(
            options,
            candidate,
            variantDirectory,
            processRunner,
            output,
            error,
            cancellationToken).ConfigureAwait(false);
        if (directMlGraphResult is not null)
        {
            return directMlGraphResult;
        }

        if (!File.Exists(entryPath))
        {
            return new ModelLabCandidateResult(null, $"expected entry file '{entryPath}' was not preserved after Olive optimization");
        }

        if (!options.SkipBenchmark)
        {
            int benchmarkExitCode = await processRunner.RunAsync(
                new ModelLabProcessStartInfo(
                    "dotnet",
                    [
                        "run",
                        "--project",
                        options.BenchmarkProjectPath,
                        "--framework",
                        options.BenchmarkFramework,
                        "-p:Platform=x64",
                        "-p:WindowsAppSDKSelfContained=true",
                        "--",
                        "--model",
                        entryPath,
                        "--provider",
                        candidate.BenchmarkProvider,
                        "--runs",
                        options.BenchmarkRuns.ToString(System.Globalization.CultureInfo.InvariantCulture),
                        "--output",
                        benchmarkReportPath,
                        "--format",
                        "json"
                    ],
                    options.RepositoryRootPath),
                output,
                error,
                cancellationToken).ConfigureAwait(false);

            if (benchmarkExitCode != 0)
            {
                return new ModelLabCandidateResult(null, $"benchmark exited with code {benchmarkExitCode}");
            }

            ModelLabCandidateResult? benchmarkReportResult = ValidateBenchmarkReport(benchmarkReportPath, candidate);
            if (benchmarkReportResult is not null)
            {
                return benchmarkReportResult;
            }
        }

        string relativeEntryPath = ToManifestRelativePath(options.ModelsRootPath, options.ModelRootName, entryPath);
        string sha256 = ComputeSha256(entryPath);
        string? benchmarkReportPathToExclude = options.SkipBenchmark ? null : benchmarkReportPath;
        IReadOnlyList<string> downloadFiles = EnumerateDownloadFiles(
            options.ModelsRootPath,
            options.ModelRootName,
            variantDirectory,
            entryPath,
            benchmarkReportPathToExclude);

        var variant = new ModelLabVariantResult(candidate.Alias, relativeEntryPath, sha256, downloadFiles);
        return new ModelLabCandidateResult(variant, null);
    }

    private static ModelLabCandidateResult? ValidateBenchmarkReport(
        string benchmarkReportPath,
        ModelLabCandidateOptions candidate)
    {
        if (!File.Exists(benchmarkReportPath))
        {
            return new ModelLabCandidateResult(null, $"benchmark report '{benchmarkReportPath}' was not generated");
        }

        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(benchmarkReportPath));
        JsonElement root = document.RootElement;
        string? status = ReadStringProperty(root, "Status");
        if (!string.Equals(status, "Completed", StringComparison.OrdinalIgnoreCase))
        {
            string failureReason = ReadStringProperty(root, "FailureReason") ?? "no failure reason was recorded";
            return new ModelLabCandidateResult(
                null,
                $"benchmark report status was {status ?? "missing"}: {failureReason}");
        }

        bool? supportsExecution = ReadBooleanProperty(root, "SupportsExecution");
        if (supportsExecution is not true)
        {
            return new ModelLabCandidateResult(null, "benchmark report did not confirm execution support");
        }

        string? selectedProvider = ReadStringProperty(root, "SelectedProvider");
        if (string.IsNullOrWhiteSpace(selectedProvider))
        {
            return new ModelLabCandidateResult(null, "benchmark report did not record a selected provider");
        }

        if (!IsAutoProvider(candidate.BenchmarkProvider) &&
            !selectedProvider.Equals(candidate.BenchmarkProvider, StringComparison.OrdinalIgnoreCase))
        {
            return new ModelLabCandidateResult(
                null,
                $"benchmark selected provider '{selectedProvider}' did not match requested provider '{candidate.BenchmarkProvider}'");
        }

        if (!IsAutoProvider(candidate.BenchmarkProvider) && BenchmarkReportRecordsProviderFallback(root))
        {
            return new ModelLabCandidateResult(
                null,
                "benchmark report recorded an execution-provider fallback for an explicit provider candidate");
        }

        return null;
    }

    private static bool BenchmarkReportRecordsProviderFallback(JsonElement root)
    {
        if (!TryGetProperty(root, "Notes", out JsonElement notes) ||
            notes.ValueKind is not JsonValueKind.Array)
        {
            return false;
        }

        foreach (JsonElement note in notes.EnumerateArray())
        {
            if (note.ValueKind is JsonValueKind.String &&
                note.GetString() is { } value &&
                value.Contains("fell back", StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private static string? ReadStringProperty(JsonElement root, string propertyName)
    {
        if (!TryGetProperty(root, propertyName, out JsonElement value))
        {
            return null;
        }

        return value.ValueKind switch
        {
            JsonValueKind.String => value.GetString(),
            JsonValueKind.Number => value.GetRawText(),
            _ => null
        };
    }

    private static bool? ReadBooleanProperty(JsonElement root, string propertyName)
    {
        if (!TryGetProperty(root, propertyName, out JsonElement value))
        {
            return null;
        }

        return value.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => null
        };
    }

    private static bool TryGetProperty(JsonElement root, string propertyName, out JsonElement value)
    {
        if (root.TryGetProperty(propertyName, out value))
        {
            return true;
        }

        foreach (JsonProperty property in root.EnumerateObject())
        {
            if (property.Name.Equals(propertyName, StringComparison.OrdinalIgnoreCase))
            {
                value = property.Value;
                return true;
            }
        }

        value = default;
        return false;
    }

    private static bool IsAutoProvider(string provider) =>
        provider.Equals("auto", StringComparison.OrdinalIgnoreCase);

    private static async Task<ModelLabCandidateResult?> DecomposeWhisperCrossAttentionForDirectMlAsync(
        ModelLabCommandOptions options,
        ModelLabCandidateOptions candidate,
        string variantDirectory,
        IModelLabProcessRunner processRunner,
        TextWriter output,
        TextWriter error,
        CancellationToken cancellationToken)
    {
        // SkipLayerNormalization / BiasGelu exist on every provider route and block
        // TensorRT-RTX import; decompose them first so residual MHA is the only remaining
        // contrib surface.
        await DecomposeMicrosoftContribOpsAsync(
            options, variantDirectory, processRunner, output, error, cancellationToken).ConfigureAwait(false);

        if (!candidate.BenchmarkProvider.Equals("dml", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        string decoderPath = Path.Join(variantDirectory, "decoder.onnx");
        if (!File.Exists(decoderPath))
        {
            return null;
        }

        string scriptPath = Path.Join(
            options.RepositoryRootPath,
            "tools",
            "model-lab",
            "decompose-whisper-cross-attention.py");
        if (!File.Exists(scriptPath))
        {
            return new ModelLabCandidateResult(
                null,
                $"DirectML Whisper graph transform script was not found at '{scriptPath}'");
        }

        int transformExitCode = await processRunner.RunAsync(
            new ModelLabProcessStartInfo(
                options.PythonPath,
                [
                    scriptPath,
                    decoderPath
                ],
                options.RepositoryRootPath),
            output,
            error,
            cancellationToken).ConfigureAwait(false);

        if (transformExitCode != 0)
        {
            return new ModelLabCandidateResult(null, $"DirectML Whisper graph transform exited with code {transformExitCode}");
        }

        output.WriteLine("  graph: decomposed Whisper cross-attention for DirectML");
        return null;
    }

    private static async Task DecomposeMicrosoftContribOpsAsync(
        ModelLabCommandOptions options,
        string variantDirectory,
        IModelLabProcessRunner processRunner,
        TextWriter output,
        TextWriter error,
        CancellationToken cancellationToken)
    {
        string scriptPath = Path.Join(
            options.RepositoryRootPath,
            "tools".TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
            "model-lab".TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar),
            "decompose-microsoft-contrib-ops.py".TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        if (!File.Exists(scriptPath))
        {
            return;
        }

        foreach (string onnxPath in Directory
                     .EnumerateFiles(variantDirectory, "*.onnx", SearchOption.TopDirectoryOnly)
                     .Order(StringComparer.OrdinalIgnoreCase))
        {
            int exitCode = await processRunner.RunAsync(
                new ModelLabProcessStartInfo(
                    options.PythonPath,
                    [scriptPath, onnxPath],
                    options.RepositoryRootPath),
                output,
                error,
                cancellationToken).ConfigureAwait(false);
            if (exitCode == 0)
            {
                output.WriteLine($"  graph: decomposed com.microsoft contrib ops in {Path.GetFileName(onnxPath)}");
            }
        }
    }

    private static async Task<ModelLabCandidateResult?> OptimizeGeneratedOnnxComponentsAsync(
        ModelLabCommandOptions options,
        ModelLabCandidateOptions candidate,
        string variantDirectory,
        IModelLabProcessRunner processRunner,
        TextWriter output,
        TextWriter error,
        CancellationToken cancellationToken)
    {
        string[] componentPaths = Directory
            .EnumerateFiles(variantDirectory, "*.onnx", SearchOption.TopDirectoryOnly)
            .Order(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (componentPaths.Length == 0)
        {
            return new ModelLabCandidateResult(null, $"no ONNX component files were generated in '{variantDirectory}'");
        }

        if (!string.IsNullOrWhiteSpace(options.OliveRecipeConfigPath))
        {
            string recipeConfigPath = Path.GetFullPath(options.OliveRecipeConfigPath);
            if (!File.Exists(recipeConfigPath))
            {
                return new ModelLabCandidateResult(null, $"Olive recipe override not found: '{recipeConfigPath}'");
            }

            string oliveOutputDirectory = Path.Join(options.CacheDirectoryPath, "olive", candidate.Alias, "recipe");
            ResetDirectoryWithin(oliveOutputDirectory, options.CacheDirectoryPath);

            int oliveExitCode = await processRunner.RunAsync(
                new ModelLabProcessStartInfo(
                    options.OliveExecutablePath,
                    [
                        "run",
                        "--config",
                        recipeConfigPath
                    ],
                    options.CacheDirectoryPath),
                output,
                error,
                cancellationToken).ConfigureAwait(false);

            if (oliveExitCode != 0)
            {
                return new ModelLabCandidateResult(null, $"Olive recipe run exited with code {oliveExitCode}");
            }

            output.WriteLine($"  olive: recipe run completed ({Path.GetFileName(recipeConfigPath)})");
            return null;
        }

        foreach (string componentPath in componentPaths)
        {
            string componentName = Path.GetFileNameWithoutExtension(componentPath);
            string oliveOutputDirectory = Path.Join(options.CacheDirectoryPath, "olive", candidate.Alias, componentName);
            ResetDirectoryWithin(oliveOutputDirectory, options.CacheDirectoryPath);

            int oliveExitCode = await processRunner.RunAsync(
                new ModelLabProcessStartInfo(
                    options.OliveExecutablePath,
                    [
                        "optimize",
                        "--model_name_or_path",
                        componentPath,
                        "--output_path",
                        oliveOutputDirectory,
                        "--device",
                        candidate.OliveDevice,
                        "--provider",
                        candidate.OliveProvider,
                        "--precision",
                        candidate.Precision,
                        "--log_level",
                        "1"
                    ],
                    options.CacheDirectoryPath),
                output,
                error,
                cancellationToken).ConfigureAwait(false);

            if (oliveExitCode != 0)
            {
                return new ModelLabCandidateResult(null, $"Olive optimize exited with code {oliveExitCode} for {Path.GetFileName(componentPath)}");
            }

            string optimizedModelPath = Path.Join(oliveOutputDirectory, "model.onnx");
            if (!File.Exists(optimizedModelPath))
            {
                return new ModelLabCandidateResult(null, $"Olive did not produce '{optimizedModelPath}' for {Path.GetFileName(componentPath)}");
            }

            File.Copy(optimizedModelPath, componentPath, overwrite: true);
            ReplaceExternalDataFileIfPresent(oliveOutputDirectory, componentPath);
            output.WriteLine($"  olive: optimized {Path.GetFileName(componentPath)}");
        }

        return null;
    }

    public static void WriteUsage(TextWriter writer)
    {
        writer.WriteLine("Trackdub.Tools model-lab");
        writer.WriteLine();
        writer.WriteLine("Build, Olive-optimize, and manifest hardware-specific ORT GenAI model variants.");
        writer.WriteLine("Benchmark step validates provider availability on the current machine (Windows only).");
        writer.WriteLine();
        writer.WriteLine("Required inputs are inferred for the Trackdub repo, but can be overridden:");
        writer.WriteLine("  --model <hf-model-id>");
        writer.WriteLine("  --model-root <models-subdirectory>");
        writer.WriteLine("  --models-root <path>");
        writer.WriteLine("  --manifest-fragment <path>");
        writer.WriteLine("  --python <python-exe>");
        writer.WriteLine("  --builder <onnxruntime-genai builder.py>");
        writer.WriteLine("  --builder-module <python-module>");
        writer.WriteLine("  --olive <olive-exe>");
        writer.WriteLine("  --cache <path>");
        writer.WriteLine("  --benchmark-project <Trackdub.Benchmarks.csproj>");
        writer.WriteLine("  --benchmark-framework <tfm>");
        writer.WriteLine("  --benchmark-runs <count>");
        writer.WriteLine("  --no-benchmark        skip benchmark validation (enables cross-platform Olive optimization)");
        writer.WriteLine("  --candidate <alias>:<builder-provider>:<precision>:<olive-provider>:<olive-device>:<benchmark-provider>");
        writer.WriteLine("  --olive-recipe-config <path>  developer override: run `olive run --config` instead of optimize");
    }

    private static string ComputeSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        byte[] hash = SHA256.HashData(stream);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    private static IReadOnlyList<string> EnumerateDownloadFiles(
        string modelsRootPath,
        string modelRootName,
        string variantDirectory,
        string entryPath,
        string? benchmarkReportPath)
    {
        string fullEntryPath = Path.GetFullPath(entryPath);
        string? fullBenchmarkReportPath = benchmarkReportPath is null
            ? null
            : Path.GetFullPath(benchmarkReportPath);

        return Directory.EnumerateFiles(variantDirectory, "*", SearchOption.AllDirectories)
            .Select(Path.GetFullPath)
            .Where(path => !path.Equals(fullEntryPath, StringComparison.OrdinalIgnoreCase))
            .Where(path => fullBenchmarkReportPath is null ||
                           !path.Equals(fullBenchmarkReportPath, StringComparison.OrdinalIgnoreCase))
            .Select(path => ToManifestRelativePath(modelsRootPath, modelRootName, path))
            .Order(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static bool IsWithinDirectory(string path, string directory)
    {
        string relativePath = Path.GetRelativePath(directory, path);
        return relativePath.Length > 0 &&
               relativePath[0] != '.' &&
               !Path.IsPathRooted(relativePath);
    }

    private static void ResetDirectoryWithin(string directoryPath, string requiredRootPath)
    {
        string fullDirectoryPath = Path.GetFullPath(directoryPath);
        string fullRequiredRootPath = Path.GetFullPath(requiredRootPath);
        if (!IsWithinDirectory(fullDirectoryPath, fullRequiredRootPath))
        {
            throw new InvalidOperationException($"Refusing to reset directory outside cache root: {fullDirectoryPath}");
        }

        if (Directory.Exists(fullDirectoryPath))
        {
            Directory.Delete(fullDirectoryPath, recursive: true);
        }

        Directory.CreateDirectory(fullDirectoryPath);
    }

    private static void ReplaceExternalDataFileIfPresent(string oliveOutputDirectory, string componentPath)
    {
        string sourceDataPath = Path.Join(oliveOutputDirectory, "model.onnx.data");
        string targetDataPath = componentPath + ".data";
        if (File.Exists(sourceDataPath))
        {
            File.Copy(sourceDataPath, targetDataPath, overwrite: true);
        }
        else if (File.Exists(targetDataPath))
        {
            File.Delete(targetDataPath);
        }
    }

    private static string ToManifestRelativePath(string modelsRootPath, string modelRootName, string path)
    {
        string modelRootPath = Path.Join(modelsRootPath, modelRootName);
        return Path.GetRelativePath(modelRootPath, path).Replace('\\', '/');
    }
}

public sealed record ModelLabCommandOptions(
    string RepositoryRootPath,
    string HuggingFaceModelId,
    string ModelRootName,
    string ModelsRootPath,
    string ManifestFragmentPath,
    string PythonPath,
    string OrtGenAiBuilderPath,
    bool UseOrtGenAiBuilderModule,
    string OliveExecutablePath,
    string CacheDirectoryPath,
    string BenchmarkProjectPath,
    string BenchmarkFramework,
    int BenchmarkRuns,
    IReadOnlyList<ModelLabCandidateOptions> Candidates,
    bool SkipBenchmark,
    string? OliveRecipeConfigPath,
    bool ShowHelp)
{
    private const string DefaultOrtGenAiBuilderModule = "onnxruntime_genai.models.builder";

    private static readonly IReadOnlyList<ModelLabCandidateOptions> DefaultCandidates =
    [
        new("cpu-fp32", "cpu", "fp32", "CPUExecutionProvider", "cpu", "cpu"),
        new("directml-fp16", "dml", "fp16", "DmlExecutionProvider", "gpu", "dml"),
        new("trt-rtx-fp16", "NvTensorRtRtx", "fp16", "NvTensorRTRTXExecutionProvider", "gpu", "trt-rtx")
    ];

    public static bool TryParse(
        IReadOnlyList<string> args,
        TextWriter errorWriter,
        out ModelLabCommandOptions options)
    {
        string repositoryRootPath = FindRepositoryRoot(Environment.CurrentDirectory);
        string huggingFaceModelId = "openai/whisper-tiny";
        string modelRootName = "whisper-tiny-genai";
        string modelsRootPath = Path.Join(repositoryRootPath, "models");
        string manifestFragmentPath = Path.Join(modelsRootPath, "manifest-fragments", "trackdub-model-lab.manifest.json");
        string pythonPath = "python";
        string ortGenAiBuilderPath = DefaultOrtGenAiBuilderModule;
        bool useOrtGenAiBuilderModule = true;
        string oliveExecutablePath = "olive";
        string cacheDirectoryPath = Path.Join(modelsRootPath, ".model-lab-cache");
        string benchmarkProjectPath = Path.Join(repositoryRootPath, "src", "Trackdub.Benchmarks", "Trackdub.Benchmarks.csproj");
        string benchmarkFramework = "net11.0-windows10.0.19041.0";
        int benchmarkRuns = 3;
        var candidates = new List<ModelLabCandidateOptions>();
        bool skipBenchmark = false;
        string? oliveRecipeConfigPath = null;
        bool showHelp = false;

        for (var index = 0; index < args.Count; index++)
        {
            string arg = args[index];

            switch (arg)
            {
                case "--help":
                case "-h":
                case "/?":
                    showHelp = true;
                    break;

                case "--model":
                    if (!TryReadValue(args, ref index, arg, errorWriter, out huggingFaceModelId))
                    {
                        options = DefaultWithHelp();
                        return false;
                    }

                    break;

                case "--model-root":
                    if (!TryReadValue(args, ref index, arg, errorWriter, out modelRootName))
                    {
                        options = DefaultWithHelp();
                        return false;
                    }

                    break;

                case "--models-root":
                    if (!TryReadValue(args, ref index, arg, errorWriter, out modelsRootPath))
                    {
                        options = DefaultWithHelp();
                        return false;
                    }

                    break;

                case "--manifest-fragment":
                    if (!TryReadValue(args, ref index, arg, errorWriter, out manifestFragmentPath))
                    {
                        options = DefaultWithHelp();
                        return false;
                    }

                    break;

                case "--python":
                    if (!TryReadValue(args, ref index, arg, errorWriter, out pythonPath))
                    {
                        options = DefaultWithHelp();
                        return false;
                    }

                    break;

                case "--builder":
                    if (!TryReadValue(args, ref index, arg, errorWriter, out ortGenAiBuilderPath))
                    {
                        options = DefaultWithHelp();
                        return false;
                    }

                    useOrtGenAiBuilderModule = false;
                    break;

                case "--builder-module":
                    if (!TryReadValue(args, ref index, arg, errorWriter, out ortGenAiBuilderPath))
                    {
                        options = DefaultWithHelp();
                        return false;
                    }

                    useOrtGenAiBuilderModule = true;
                    break;

                case "--olive":
                    if (!TryReadValue(args, ref index, arg, errorWriter, out oliveExecutablePath))
                    {
                        options = DefaultWithHelp();
                        return false;
                    }

                    break;

                case "--cache":
                    if (!TryReadValue(args, ref index, arg, errorWriter, out cacheDirectoryPath))
                    {
                        options = DefaultWithHelp();
                        return false;
                    }

                    break;

                case "--benchmark-project":
                    if (!TryReadValue(args, ref index, arg, errorWriter, out benchmarkProjectPath))
                    {
                        options = DefaultWithHelp();
                        return false;
                    }

                    break;

                case "--benchmark-framework":
                    if (!TryReadValue(args, ref index, arg, errorWriter, out benchmarkFramework))
                    {
                        options = DefaultWithHelp();
                        return false;
                    }

                    break;

                case "--benchmark-runs":
                    if (!TryReadValue(args, ref index, arg, errorWriter, out string benchmarkRunsText))
                    {
                        options = DefaultWithHelp();
                        return false;
                    }

                    if (!int.TryParse(benchmarkRunsText, out benchmarkRuns) || benchmarkRuns <= 0)
                    {
                        errorWriter.WriteLine($"Invalid benchmark run count '{benchmarkRunsText}'.");
                        options = DefaultWithHelp();
                        return false;
                    }

                    break;

                case "--no-benchmark":
                    skipBenchmark = true;
                    break;

                case "--candidate":
                    if (!TryReadValue(args, ref index, arg, errorWriter, out string candidateText))
                    {
                        options = DefaultWithHelp();
                        return false;
                    }

                    if (!TryParseCandidate(candidateText, errorWriter, out ModelLabCandidateOptions candidate))
                    {
                        options = DefaultWithHelp();
                        return false;
                    }

                    candidates.Add(candidate);
                    break;

                case "--olive-recipe-config":
                    if (!TryReadValue(args, ref index, arg, errorWriter, out oliveRecipeConfigPath))
                    {
                        options = DefaultWithHelp();
                        return false;
                    }

                    break;

                default:
                    errorWriter.WriteLine($"Unknown argument '{arg}'.");
                    options = DefaultWithHelp();
                    return false;
            }
        }

        if (showHelp)
        {
            options = DefaultWithHelp() with { ShowHelp = true };
            return true;
        }

        if (string.IsNullOrWhiteSpace(huggingFaceModelId))
        {
            errorWriter.WriteLine("Missing required argument --model <hf-model-id>.");
            options = DefaultWithHelp();
            return false;
        }

        if (string.IsNullOrWhiteSpace(modelRootName) ||
            modelRootName.IndexOfAny([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar, ':']) >= 0)
        {
            errorWriter.WriteLine("--model-root must be a single models/ directory name.");
            options = DefaultWithHelp();
            return false;
        }

        options = new ModelLabCommandOptions(
            Path.GetFullPath(repositoryRootPath),
            huggingFaceModelId.Trim(),
            modelRootName.Trim(),
            Path.GetFullPath(modelsRootPath),
            Path.GetFullPath(manifestFragmentPath),
            NormalizeExecutablePath(pythonPath),
            NormalizeBuilderInvocation(ortGenAiBuilderPath, useOrtGenAiBuilderModule),
            useOrtGenAiBuilderModule,
            NormalizeExecutablePath(oliveExecutablePath),
            Path.GetFullPath(cacheDirectoryPath),
            Path.GetFullPath(benchmarkProjectPath),
            benchmarkFramework.Trim(),
            benchmarkRuns,
            candidates.Count == 0 ? DefaultCandidates : candidates.ToArray(),
            SkipBenchmark: skipBenchmark,
            OliveRecipeConfigPath: string.IsNullOrWhiteSpace(oliveRecipeConfigPath)
                ? null
                : Path.GetFullPath(oliveRecipeConfigPath.Trim()),
            ShowHelp: false);

        return true;
    }

    private static bool TryParseCandidate(
        string value,
        TextWriter errorWriter,
        out ModelLabCandidateOptions candidate)
    {
        string[] parts = value.Split(':', StringSplitOptions.TrimEntries);
        if (parts.Length != 6 || parts.Any(string.IsNullOrWhiteSpace))
        {
            errorWriter.WriteLine(
                "Candidate must use '<alias>:<builder-provider>:<precision>:<olive-provider>:<olive-device>:<benchmark-provider>'.");
            candidate = new ModelLabCandidateOptions(string.Empty, string.Empty, string.Empty, string.Empty, string.Empty, string.Empty);
            return false;
        }

        candidate = new ModelLabCandidateOptions(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]);
        return true;
    }

    private static ModelLabCommandOptions DefaultWithHelp() =>
        new(
            Environment.CurrentDirectory,
            string.Empty,
            string.Empty,
            Environment.CurrentDirectory,
            Path.Join(Environment.CurrentDirectory, "trackdub-model-lab.manifest.json"),
            "python",
            DefaultOrtGenAiBuilderModule,
            true,
            "olive",
            Path.Join(Environment.CurrentDirectory, ".model-lab-cache"),
            Path.Join(Environment.CurrentDirectory, "Trackdub.Benchmarks.csproj"),
            "net11.0-windows10.0.19041.0",
            3,
            [],
            SkipBenchmark: false,
            OliveRecipeConfigPath: null,
            ShowHelp: true);

    private static bool TryReadValue(
        IReadOnlyList<string> args,
        ref int index,
        string optionName,
        TextWriter errorWriter,
        out string value)
    {
        if (index + 1 >= args.Count)
        {
            errorWriter.WriteLine($"Missing value for {optionName}.");
            value = string.Empty;
            return false;
        }

        value = args[++index];
        return true;
    }

    private static string FindRepositoryRoot(string seed)
    {
        DirectoryInfo? current = new(Path.GetFullPath(seed));
        while (current is not null)
        {
            if (Directory.Exists(Path.Join(current.FullName, ".git")) ||
                File.Exists(Path.Join(current.FullName, "Trackdub.slnx")))
            {
                return current.FullName;
            }

            current = current.Parent;
        }

        return Path.GetFullPath(seed);
    }

    private static string NormalizeBuilderInvocation(string builderInvocation, bool useBuilderModule)
    {
        string trimmed = builderInvocation.Trim();
        return useBuilderModule ? trimmed : Path.GetFullPath(trimmed);
    }

    private static string NormalizeExecutablePath(string executablePath)
    {
        string trimmed = executablePath.Trim();
        return trimmed.IndexOfAny([Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar]) >= 0 || Path.IsPathRooted(trimmed)
            ? Path.GetFullPath(trimmed)
            : trimmed;
    }
}

public sealed record ModelLabCandidateOptions(
    string Alias,
    string BuilderProvider,
    string Precision,
    string OliveProvider,
    string OliveDevice,
    string BenchmarkProvider);

public sealed record ModelLabProcessStartInfo(
    string Executable,
    IReadOnlyList<string> Arguments,
    string WorkingDirectory);

public interface IModelLabProcessRunner
{
    Task<int> RunAsync(
        ModelLabProcessStartInfo startInfo,
        TextWriter output,
        TextWriter error,
        CancellationToken cancellationToken);
}

public sealed class ModelLabProcessRunner : IModelLabProcessRunner
{
    public async Task<int> RunAsync(
        ModelLabProcessStartInfo startInfo,
        TextWriter output,
        TextWriter error,
        CancellationToken cancellationToken)
    {
        var processStartInfo = new ProcessStartInfo(startInfo.Executable)
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            WorkingDirectory = startInfo.WorkingDirectory
        };

        foreach (string argument in startInfo.Arguments)
        {
            processStartInfo.ArgumentList.Add(argument);
        }

        using var process = new Process { StartInfo = processStartInfo };
        process.Start();

        Task<string> outputTask = process.StandardOutput.ReadToEndAsync(cancellationToken);
        Task<string> errorTask = process.StandardError.ReadToEndAsync(cancellationToken);

        await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);

        string standardOutput = await outputTask.ConfigureAwait(false);
        string standardError = await errorTask.ConfigureAwait(false);
        await output.WriteAsync(standardOutput).ConfigureAwait(false);
        await error.WriteAsync(standardError).ConfigureAwait(false);

        return process.ExitCode;
    }
}

internal sealed record ModelLabCandidateResult(
    ModelLabVariantResult? Variant,
    string? Error);

internal sealed record ModelLabVariantResult(
    string Alias,
    string EntryPath,
    string Sha256,
    IReadOnlyList<string> DownloadFiles);

internal static class ModelLabManifestFragmentWriter
{
    public static async Task WriteAsync(
        ModelLabCommandOptions options,
        IReadOnlyList<ModelLabVariantResult> variants,
        CancellationToken cancellationToken)
    {
        string? directory = Path.GetDirectoryName(options.ManifestFragmentPath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        await using var stream = new FileStream(
            options.ManifestFragmentPath,
            FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            bufferSize: 4096,
            options: FileOptions.Asynchronous);

        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });
        ModelLabVariantResult firstVariant = variants[0];

        writer.WriteStartObject();
        writer.WritePropertyName("models");
        writer.WriteStartArray();
        writer.WriteStartObject();
        writer.WriteString("model_id", options.HuggingFaceModelId);
        writer.WriteString("task", "asr");
        writer.WriteString("engine_family", "whisper-genai");
        WriteStringArray(writer, "capabilities", ["asr", "language-detection"]);
        writer.WritePropertyName("language_coverage");
        writer.WriteStartObject();
        WriteStringArray(writer, "source_languages", ["auto"]);
        writer.WriteEndObject();
        writer.WriteString("tier", "fast");
        writer.WriteString("license", "Apache-2.0");
        writer.WriteBoolean("commercial_allowed", true);
        writer.WriteBoolean("redistribution_allowed", true);
        writer.WriteBoolean("requires_attribution", false);
        writer.WriteBoolean("requires_user_consent", false);
        writer.WriteBoolean("voice_cloning", false);
        writer.WriteBoolean("commercial_use_verified", true);
        writer.WriteString("source_url", $"https://huggingface.co/{options.HuggingFaceModelId}");
        writer.WriteString("revision", "model-lab");
        writer.WriteString("sha256", firstVariant.Sha256);
        WriteStringArray(writer, "aliases", [options.ModelRootName, $"{options.ModelRootName}-model-lab"]);
        writer.WriteString("root_path", $"../{options.ModelRootName}");
        writer.WriteString("benchmark_entry", firstVariant.EntryPath);

        writer.WritePropertyName("variants");
        writer.WriteStartArray();
        foreach (ModelLabVariantResult variant in variants.OrderBy(variant => variant.Alias, StringComparer.OrdinalIgnoreCase))
        {
            writer.WriteStartObject();
            writer.WriteString("alias", variant.Alias);
            writer.WriteString("entry_path", variant.EntryPath);
            writer.WriteString("sha256", variant.Sha256);
            WriteStringArray(writer, "download_files", variant.DownloadFiles);
            writer.WriteEndObject();
        }

        writer.WriteEndArray();
        writer.WriteEndObject();
        writer.WriteEndArray();
        writer.WriteEndObject();
        await writer.FlushAsync(cancellationToken).ConfigureAwait(false);
    }

    private static void WriteStringArray(
        Utf8JsonWriter writer,
        string propertyName,
        IReadOnlyList<string> values)
    {
        writer.WritePropertyName(propertyName);
        writer.WriteStartArray();
        foreach (string value in values)
        {
            writer.WriteStringValue(value);
        }

        writer.WriteEndArray();
    }
}
