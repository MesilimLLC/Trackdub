using System.Text.Json;
using System.Text.RegularExpressions;
using System.Xml.Linq;

namespace Trackdub.Architecture.Tests;

/// <summary>
/// Validates the public-core dependency graph: strict layer direction,
/// acyclicity, AGENTS.md diagram consistency, and structural invariants
/// for inference/ONNX/DNNL projects.
/// </summary>
public sealed class DependencyGraphTests
{
    [Theory]
    [InlineData("Api")]
    [InlineData("CloudJobAdapter")]
    public void CloudHostsExcludeInferenceAndSdkTransitively(string host)
    {
        var graph = ReadAllSrcCsprojDependencies(Path.Join(FindRepoRoot(), "src"));
        var allowed = new[] { "Api", "CloudJobAdapter", "Cloud.Aws", "Cloud", "Contracts", "Domain" }.Select(Normalize).ToHashSet();
        var pending = new Stack<string>(); var visited = new HashSet<string>(); pending.Push(Normalize(host));
        while (pending.TryPop(out string? project))
        {
            if (!visited.Add(project)) continue;
            Assert.Contains(project, allowed);
            foreach (string dependency in graph[project]) pending.Push(dependency);
        }
    }
    [Fact]
    public void AgentsMdDiagramMatchesEveryCsprojProjectReference()
    {
        var repoRoot = FindRepoRoot();
        var diagram = ParseAgentsMdDiagram(Path.Join(repoRoot, "AGENTS.md"));
        var csprojs = ReadAllSrcCsprojDependencies(Path.Join(repoRoot, "src"));

        var diagramProjects = diagram.Keys.ToHashSet();
        var csprojProjects = csprojs.Keys.ToHashSet();

        var missingFromDiagram = csprojProjects.Except(diagramProjects).OrderBy(x => x).ToList();
        var missingFromCsprojs = diagramProjects.Except(csprojProjects).OrderBy(x => x).ToList();

        Assert.True(
            missingFromDiagram.Count == 0,
            $"src/ contains csproj(s) not listed in AGENTS.md diagram: {string.Join(", ", missingFromDiagram)}");
        Assert.True(
            missingFromCsprojs.Count == 0,
            $"AGENTS.md diagram lists project(s) that do not exist under src/: {string.Join(", ", missingFromCsprojs)}");

        var mismatches = new List<string>();
        foreach (var (project, diagramDeps) in diagram.OrderBy(kv => kv.Key))
        {
            var actual = csprojs[project];
            var expected = diagramDeps;
            var extraInCsproj = actual.Except(expected).OrderBy(x => x).ToList();
            var extraInDiagram = expected.Except(actual).OrderBy(x => x).ToList();

            if (extraInCsproj.Count > 0 || extraInDiagram.Count > 0)
            {
                mismatches.Add(
                    $"  {project}: " +
                    (extraInCsproj.Count > 0 ? $"csproj has extra [{string.Join(", ", extraInCsproj)}] " : "") +
                    (extraInDiagram.Count > 0 ? $"diagram has extra [{string.Join(", ", extraInDiagram)}]" : ""));
            }
        }

        Assert.True(
            mismatches.Count == 0,
            "AGENTS.md dependency diagram does not match csproj ProjectReferences:\n" +
            string.Join("\n", mismatches));
    }

    [Fact]
    public void DomainHasNoProjectReferences()
    {
        var repoRoot = FindRepoRoot();
        var csprojs = ReadAllSrcCsprojDependencies(Path.Join(repoRoot, "src"));
        Assert.Empty(csprojs["Trackdub.Domain"]);
    }

    /// <summary>
    /// Enforces ADR-0011: <c>Trackdub.Contracts</c> may reference
    /// <c>Trackdub.Domain</c> and nothing else.
    /// </summary>
    [Fact]
    public void ContractsReferencesOnlyDomain()
    {
        var repoRoot = FindRepoRoot();
        var csprojs = ReadAllSrcCsprojDependencies(Path.Join(repoRoot, "src"));
        Assert.Equal(["Trackdub.Domain"], csprojs["Trackdub.Contracts"].OrderBy(x => x).ToArray());
    }

    [Fact]
    public void WindowsOnnxRuntimePackagesUseWinMlCatalogProvider()
    {
        var repoRoot = FindRepoRoot();
        string packagesProps = Path.Join(repoRoot, "Directory.Packages.props");
        string inferenceProject = Path.Join(repoRoot, "src", "Trackdub.Inference.Onnx", "Trackdub.Inference.Onnx.csproj");
        string compositionProject = Path.Join(repoRoot, "src", "Trackdub.Composition", "Trackdub.Composition.csproj");

        XDocument packageVersions = XDocument.Load(packagesProps);
        XDocument inference = XDocument.Load(inferenceProject);
        XDocument composition = XDocument.Load(compositionProject);

        Assert.DoesNotContain(
            packageVersions.Descendants("PackageVersion"),
            element => element.Attribute("Include")?.Value == "Microsoft.ML.OnnxRuntime.DirectML");

        Assert.DoesNotContain(
            inference.Descendants("PackageReference"),
            element => element.Attribute("Include")?.Value == "Microsoft.ML.OnnxRuntime.DirectML");

        Assert.DoesNotContain(
            composition.Descendants("PackageReference"),
            element => element.Attribute("Include")?.Value == "Microsoft.ML.OnnxRuntime.DirectML");

        Assert.Contains(
            inference.Descendants("PackageReference"),
            element => element.Attribute("Include")?.Value == "Microsoft.Windows.AI.MachineLearning");

        Assert.Contains(
            inference.Descendants("PackageReference"),
            element => element.Attribute("Include")?.Value == "Microsoft.ML.OnnxRuntimeGenAI.WinML");

        Assert.Contains(
            composition.Descendants("PackageReference"),
            element => element.Attribute("Include")?.Value == "Microsoft.Windows.AI.MachineLearning");

        Assert.Single(
            composition.Descendants("Target"),
            element => element.Attribute("Name")?.Value == "CopyWinMlAssetsToOutput");

        Assert.Contains(
            composition.Descendants("Target"),
            element => element.Attribute("Name")?.Value == "AddWinMlAssetsToOutputItems");

        string[] winMlAssetIncludes = composition
            .Descendants("_WinMlAssets")
            .Select(element => element.Attribute("Include")?.Value)
            .Where(include => !string.IsNullOrWhiteSpace(include))
            .Cast<string>()
            .ToArray();

        Assert.Contains(
            winMlAssetIncludes,
            include => include.Contains("$(PkgMicrosoft_Windows_AI_MachineLearning)", StringComparison.Ordinal));

        static string NormalizeSeparators(string value) => value.Replace('\\', '/');

        string[] nativeWinMlAssetIncludes = winMlAssetIncludes
            .Where(include => NormalizeSeparators(include).Contains("/runtimes/", StringComparison.OrdinalIgnoreCase))
            .ToArray();

        string[] managedWinMlAssetIncludes = winMlAssetIncludes
            .Where(include => NormalizeSeparators(include).Contains("/lib/", StringComparison.OrdinalIgnoreCase))
            .ToArray();

        Assert.NotEmpty(nativeWinMlAssetIncludes);
        Assert.All(
            nativeWinMlAssetIncludes,
            include => Assert.Contains("$(_WinMlRuntimeIdentifier)", include, StringComparison.Ordinal));

        Assert.Contains(
            managedWinMlAssetIncludes,
            include => include.Contains("Microsoft.Windows.AI.MachineLearning.Projection.dll", StringComparison.OrdinalIgnoreCase));

        Assert.Contains(
            managedWinMlAssetIncludes,
            include => include.Contains("$(_WinMlProjectionLibTfm)", StringComparison.Ordinal));

        Assert.DoesNotContain(
            nativeWinMlAssetIncludes,
            include => NormalizeSeparators(include).Contains("/runtimes/win-x64/native/", StringComparison.OrdinalIgnoreCase));

        string[] versionGlobs = winMlAssetIncludes
            .Where(include => NormalizeSeparators(include).Contains("/*/", StringComparison.Ordinal))
            .OrderBy(include => include)
            .ToArray();

        Assert.True(
            versionGlobs.Length == 0,
            "WinML native asset copies must use NuGet-resolved package paths instead of version globs: " +
            string.Join(", ", versionGlobs));

        string[] forbiddenTargets =
        [
            "CopyDirectMLAssetsToOutput",
            "AddDirectMLAssetsToOutputItems",
            "MirrorDirectMLAssetsForProjectReferenceCopy"
        ];

        string[] offenders = composition
            .Descendants("Target")
            .Select(target => target.Attribute("Name")?.Value)
            .Where(name => name is not null && forbiddenTargets.Contains(name, StringComparer.Ordinal))
            .Cast<string>()
            .OrderBy(name => name)
            .ToArray();

        Assert.True(
            offenders.Length == 0,
            "Windows DirectML assets must be supplied by WinML package assets, not legacy DirectML copy targets: " +
            string.Join(", ", offenders));
    }

    [Fact]
    public void DnnlNativePackageDeclaresInitialRidAssetsAndChecksumProvenance()
    {
        var repoRoot = FindRepoRoot();
        string packageRoot = Path.Join(repoRoot, "src", "Trackdub.OnnxRuntime.Dnnl.Native");
        string packageProject = Path.Join(packageRoot, "Trackdub.OnnxRuntime.Dnnl.Native.csproj");
        XDocument package = XDocument.Load(packageProject);

        string[] requiredRidNativeDirs =
        [
            Path.Join(packageRoot, "runtimes", "win-x64", "native"),
            Path.Join(packageRoot, "runtimes", "linux-x64", "native"),
            Path.Join(packageRoot, "runtimes", "osx-x64", "native")
        ];

        Assert.All(requiredRidNativeDirs, path => Assert.True(Directory.Exists(path), $"Missing DNNL native RID directory: {path}"));

        Assert.Contains(
            package.Descendants("None"),
            element => string.Equals(element.Attribute("Include")?.Value, @"runtimes\**\*.*", StringComparison.Ordinal));

        Assert.Contains(
            package.Descendants("None"),
            element => string.Equals(element.Attribute("Include")?.Value, @"provenance\**\*.*", StringComparison.Ordinal));

        string provenanceTemplate = Path.Join(packageRoot, "provenance", "dnnl-native-assets.template.json");
        Assert.True(File.Exists(provenanceTemplate), "DNNL native package must carry a checksum provenance template.");

        string provenanceJson = File.ReadAllText(provenanceTemplate);
        Assert.Contains("\"sha256\"", provenanceJson, StringComparison.Ordinal);
        Assert.Contains("\"onnxruntime_version\"", provenanceJson, StringComparison.Ordinal);
    }

    [Fact]
    public void CompositionOnlyCopiesDnnlNativeAssetsForDnnlRuntimeFlavor()
    {
        var repoRoot = FindRepoRoot();
        string compositionProject = Path.Join(repoRoot, "src", "Trackdub.Composition", "Trackdub.Composition.csproj");
        XDocument composition = XDocument.Load(compositionProject);

        Assert.Contains(
            composition.Descendants("Target"),
            target => target.Attribute("Name")?.Value == "ValidateDnnlOrtAssets" &&
                      (target.Attribute("Condition")?.Value.Contains("TrackdubOrtRuntimeFlavor", StringComparison.Ordinal) ?? false) &&
                      (target.Attribute("Condition")?.Value.Contains("Dnnl", StringComparison.Ordinal) ?? false));

        Assert.Contains(
            composition.Descendants("Target"),
            target => target.Attribute("Name")?.Value == "CopyDnnlOrtAssetsToOutput" &&
                      (target.Attribute("Condition")?.Value.Contains("TrackdubOrtRuntimeFlavor", StringComparison.Ordinal) ?? false) &&
                      (target.Attribute("Condition")?.Value.Contains("Dnnl", StringComparison.Ordinal) ?? false));

        string[] targetsThatMustStayOutOfDnnlFlavor =
        [
            "CopyWinMlAssetsToOutput",
            "AddWinMlAssetsToOutputItems",
            "CopyOrtGpuAssetsToOutput"
        ];

        foreach (string targetName in targetsThatMustStayOutOfDnnlFlavor)
        {
            Assert.Contains(
                composition.Descendants("Target"),
                target => target.Attribute("Name")?.Value == targetName &&
                          (target.Attribute("Condition")?.Value.Contains("TrackdubOrtRuntimeFlavor", StringComparison.Ordinal) ?? false) &&
                          (target.Attribute("Condition")?.Value.Contains("!= 'Dnnl'", StringComparison.Ordinal) ?? false));
        }
    }

    [Fact]
    public void DnnlFlavorStripsStockOrtRuntimeAssetsFromPackageReferences()
    {
        var repoRoot = FindRepoRoot();
        AssertDnnlRuntimeAssetExclusion(
            Path.Join(repoRoot, "src", "Trackdub.Composition", "Trackdub.Composition.csproj"),
            "Microsoft.ML.OnnxRuntime.Gpu");

        AssertDnnlRuntimeAssetExclusion(
            Path.Join(repoRoot, "src", "Trackdub.Inference.Onnx", "Trackdub.Inference.Onnx.csproj"),
            "Microsoft.Windows.AI.MachineLearning",
            "Microsoft.ML.OnnxRuntimeGenAI",
            "Microsoft.ML.OnnxRuntimeGenAI.WinML",
            "Microsoft.ML.OnnxRuntime.Gpu",
            "Microsoft.ML.OnnxRuntimeGenAI.Cuda",
            "Microsoft.ML.OnnxRuntime");
    }

    [Fact]
    public void DnnlNativePackageScriptRejectsArm64Hosts()
    {
        var repoRoot = FindRepoRoot();
        string scriptPath = Path.Join(repoRoot, "tools", "onnxruntime-dnnl", "Build-OnnxRuntimeDnnlNativePackage.ps1");
        string script = File.ReadAllText(scriptPath);

        Assert.Contains("OSArchitecture", script, StringComparison.Ordinal);
        Assert.Contains("Architecture]::X64", script, StringComparison.Ordinal);
        Assert.Contains("supports x64 hosts only", script, StringComparison.Ordinal);
    }

    [Fact]
    public void InferenceOnnxDoesNotImportApplicationContractsNamespace()
    {
        var repoRoot = FindRepoRoot();
        string inferenceOnnxRoot = Path.Join(repoRoot, "src", "Trackdub.Inference.Onnx");

        string[] offenders = Directory
            .EnumerateFiles(inferenceOnnxRoot, "*.cs", SearchOption.AllDirectories)
            .Where(path => !IsBuildOutput(path))
            .Where(path =>
            {
                var content = File.ReadAllText(path);
                bool hasRootImport = content.Contains("using Trackdub.Contracts;", StringComparison.Ordinal);
                bool hasProperImport = content.Contains("using Trackdub.Contracts.ApplicationContracts;", StringComparison.Ordinal);
                bool usesAppContractsTypes = content.Contains("ApplicationContracts", StringComparison.Ordinal) ||
                                               content.Contains("IRuntimePlanningPreferences", StringComparison.Ordinal) ||
                                               content.Contains("IInferenceSessionPoolEvictor", StringComparison.Ordinal);
                return hasRootImport && !hasProperImport && usesAppContractsTypes;
            })
            .Select(path => Path.GetRelativePath(repoRoot, path).Replace('\\', '/'))
            .OrderBy(path => path, StringComparer.Ordinal)
            .ToArray();

        Assert.True(
            offenders.Length == 0,
            "Trackdub.Inference.Onnx must use Trackdub.Contracts.ApplicationContracts for shared execution-provider contracts: " +
            string.Join(", ", offenders));
    }

    [Fact]
    public void OnnxLockFilePreservesPortableRuntimeIdentifierGraphs()
    {
        // Windows TFM lock entries only exist when restored on Windows (non-Windows strips the TFM)
        if (!OperatingSystem.IsWindows())
            return;

        var repoRoot = FindRepoRoot();
        string[] portableRuntimeIdentifiers =
        [
            "win-x64",
            "win-arm64",
            "linux-x64",
            "linux-arm64",
            "osx-x64",
            "osx-arm64"
        ];

        var requiredGraphsByLockFile = new Dictionary<string, string[]>
        {
            ["src/Trackdub.Contracts/packages.lock.json"] = portableRuntimeIdentifiers.Select(rid => $"net11.0/{rid}").ToArray(),
            ["src/Trackdub.Domain/packages.lock.json"] = portableRuntimeIdentifiers.Select(rid => $"net11.0/{rid}").ToArray(),
            ["src/Trackdub.Inference/packages.lock.json"] = portableRuntimeIdentifiers.Select(rid => $"net11.0/{rid}").ToArray(),
            ["src/Trackdub.Inference.Onnx/packages.lock.json"] = portableRuntimeIdentifiers
                .SelectMany(rid => new[] { $"net11.0/{rid}", $"net11.0-windows10.0.19041/{rid}" })
                .ToArray()
        };

        var missingGraphs = new List<string>();
        foreach (var (relativeLockFile, requiredGraphs) in requiredGraphsByLockFile)
        {
            string lockFile = Path.Join(repoRoot, relativeLockFile.Replace('/', Path.DirectorySeparatorChar));
            Assert.True(File.Exists(lockFile), $"{relativeLockFile} is required for deterministic portable RID locked restores.");

            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(lockFile));
            JsonElement dependencies = document.RootElement.GetProperty("dependencies");

            missingGraphs.AddRange(requiredGraphs
                .Where(graph => !dependencies.TryGetProperty(graph, out _))
                .Select(graph => $"{relativeLockFile}: {graph}"));
        }

        Assert.True(
            missingGraphs.Count == 0,
            "The Inference.Onnx restore graph must include RID restore entries for deterministic portable RID locked restores: " +
            string.Join(", ", missingGraphs));
    }

    [Fact]
    public void DependencyGraphIsAcyclic()
    {
        var repoRoot = FindRepoRoot();
        var csprojs = ReadAllSrcCsprojDependencies(Path.Join(repoRoot, "src"));

        var color = new Dictionary<string, int>();
        foreach (var node in csprojs.Keys)
        {
            color[node] = 0;
        }

        foreach (var start in csprojs.Keys)
        {
            var path = new Stack<string>();
            if (HasCycle(start, csprojs, color, path, out var cyclePath))
            {
                Assert.Fail($"Cycle detected in project dependencies: {cyclePath}");
            }
        }
    }

    private static bool HasCycle(
        string node,
        IReadOnlyDictionary<string, HashSet<string>> graph,
        Dictionary<string, int> color,
        Stack<string> path,
        out string cyclePath)
    {
        cyclePath = string.Empty;
        if (color[node] == 1)
        {
            cyclePath = string.Join(" -> ", path.Reverse().Append(node));
            return true;
        }
        if (color[node] == 2)
        {
            return false;
        }
        color[node] = 1;
        path.Push(node);
        foreach (var next in graph[node])
        {
            if (!graph.ContainsKey(next))
            {
                continue;
            }
            if (HasCycle(next, graph, color, path, out cyclePath))
            {
                return true;
            }
        }
        path.Pop();
        color[node] = 2;
        return false;
    }

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Join(dir.FullName, "Trackdub.slnx")))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }
        throw new InvalidOperationException("Could not locate Trackdub.slnx from the test runner base directory.");
    }

    private static bool IsBuildOutput(string path)
    {
        string normalized = path.Replace('\\', '/');
        return normalized.Contains("/bin/", StringComparison.OrdinalIgnoreCase) ||
               normalized.Contains("/obj/", StringComparison.OrdinalIgnoreCase);
    }

    private static void AssertDnnlRuntimeAssetExclusion(string projectPath, params string[] packageNames)
    {
        XDocument doc = XDocument.Load(projectPath);
        foreach (string packageName in packageNames)
        {
            Assert.Contains(
                doc.Descendants("PackageReference"),
                reference => string.Equals(reference.Attribute("Update")?.Value, packageName, StringComparison.Ordinal) &&
                              (reference.Attribute("Condition")?.Value.Contains("TrackdubOrtRuntimeFlavor", StringComparison.Ordinal) ?? false) &&
                              (reference.Attribute("Condition")?.Value.Contains("Dnnl", StringComparison.Ordinal) ?? false) &&
                              (reference.Attribute("ExcludeAssets")?.Value?.Contains("runtime", StringComparison.OrdinalIgnoreCase) ?? false));
        }
    }

    private static IReadOnlyDictionary<string, HashSet<string>> ParseAgentsMdDiagram(string agentsMdPath)
    {
        var text = File.ReadAllText(agentsMdPath);
        var blockMatch = Regex.Match(
            text,
            @"Strict dependency direction.*?```\s*(?<body>.*?)```",
            RegexOptions.Singleline);
        Assert.True(blockMatch.Success, "AGENTS.md is missing the expected 'Strict dependency direction' fenced code block.");

        var body = blockMatch.Groups["body"].Value;
        var result = new Dictionary<string, HashSet<string>>();

        foreach (var rawLine in body.Split('\n'))
        {
            var line = rawLine.Trim();
            if (line.Length == 0)
            {
                continue;
            }
            var parts = line.Split('\u2192'); // →
            Assert.Equal(2, parts.Length);
            var project = Normalize(parts[0]);
            var rhs = parts[1].Trim();
            var deps = new HashSet<string>();
            if (!rhs.StartsWith('('))
            {
                foreach (var raw in rhs.Split(','))
                {
                    deps.Add(Normalize(raw));
                }
            }
            result[project] = deps;
        }
        return result;
    }

    private static IReadOnlyDictionary<string, HashSet<string>> ReadAllSrcCsprojDependencies(string srcRoot)
    {
        var result = new Dictionary<string, HashSet<string>>();
        foreach (var csproj in Directory.EnumerateFiles(srcRoot, "*.csproj", SearchOption.AllDirectories))
        {
            var name = Path.GetFileNameWithoutExtension(csproj);
            var doc = XDocument.Load(csproj);
            var refs = new HashSet<string>();
            foreach (var pr in doc.Descendants("ProjectReference"))
            {
                var include = pr.Attribute("Include")?.Value;
                if (string.IsNullOrEmpty(include))
                {
                    continue;
                }
                var normalized = include.Replace('\\', '/');
                var refName = Path.GetFileNameWithoutExtension(normalized);
                refs.Add(refName);
            }
            result[name] = refs;
        }
        return result;
    }

    private static string Normalize(string rawShortName)
    {
        var trimmed = rawShortName.Trim();
        return trimmed.StartsWith("Trackdub.", StringComparison.Ordinal)
            ? trimmed
            : "Trackdub." + trimmed;
    }
}
