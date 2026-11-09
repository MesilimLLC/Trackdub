using System.Runtime.InteropServices;
using System.Xml.Linq;
using Trackdub.Inference.Onnx.Runtime;

namespace Trackdub.Inference.Tests;

public sealed class GenAiNativeCompatibilityTests
{
    [Fact]
    public void StaleWindowsMlMarkerDoesNotVerifyADeployment()
    {
        WithFiles((directory, ort, genAi) =>
        {
            File.WriteAllBytes(Path.Join(directory, "Microsoft.Windows.AI.MachineLearning.dll"), [0]);
            Assert.Equal(NativePairStatus.NotVerified,
                GenAiNativeRuntimeSelection.Verify(ort, genAi, "win-x64", []).Status);
        });
    }

    [Fact]
    public void PreRegisteredResolverIsAcceptedOnlyForTheVerifiedLoadedModule()
    {
        if (!OperatingSystem.IsWindows()) return;

        // SetDllImportResolver is once-per-assembly, so this test owns the test assembly's registration.
        var assembly = typeof(GenAiNativeCompatibilityTests).Assembly;
        DllImportResolver resolver = static (_, _, _) => nint.Zero;
        NativeLibrary.SetDllImportResolver(assembly, resolver);

        string loadedModule = Path.Join(Environment.SystemDirectory, "kernel32.dll");
        GenAiNativeRuntimeSelection.RegisterResolver(assembly, resolver, "kernel32.dll", loadedModule);

        string foreignCopy = Path.Join(Path.GetTempPath(), $"trackdub-resolver-{Guid.NewGuid():N}", "kernel32.dll");
        Assert.Throws<InvalidOperationException>(() =>
            GenAiNativeRuntimeSelection.RegisterResolver(assembly, resolver, "kernel32.dll", foreignCopy));
        Assert.Throws<InvalidOperationException>(() =>
            GenAiNativeRuntimeSelection.RegisterResolver(assembly, resolver, "not-a-loaded-module.dll", loadedModule));
    }

    [Fact]
    public void MissingRuntimeIdentityIsNotCompatible()
    {
        Assert.Equal(NativePairStatus.NotVerified,
            GenAiNativeRuntimeSelection.Verify(null, null, "win-x64", []).Status);
    }

    [Fact]
    public void IntendedWindowsMlPairIsAcceptedWithoutManagedVersionEquality()
    {
        WithFiles((directory, ort, genAi) =>
        {
            var pair = Pair(ort, genAi);
            var result = GenAiNativeRuntimeSelection.Verify(ort, genAi, "win-x64", [pair]);
            Assert.Equal(NativePairStatus.Compatible, result.Status);
            Assert.Equal("1.27.1", result.Pair!.OrtVersion);
            Assert.Equal(Path.GetFullPath(ort), result.OrtPath);
        });
    }

    [Fact]
    public void UnsupportedNativePairIsRejected()
    {
        WithFiles((directory, ort, genAi) =>
        {
            var pair = Pair(ort, genAi) with { OrtVersion = "1.30.0", GenAiFlavor = "Cpu" };
            Assert.Equal(NativePairStatus.Incompatible,
                GenAiNativeRuntimeSelection.Verify(ort, genAi, "win-x64", [pair]).Status);
        });
    }

    [Fact]
    public void ReplacedNativeFileInvalidatesProvenance()
    {
        WithFiles((directory, ort, genAi) =>
        {
            var pair = Pair(ort, genAi);
            File.WriteAllBytes(ort, [9, 8, 7]);
            Assert.Equal(NativePairStatus.NotVerified,
                GenAiNativeRuntimeSelection.Verify(ort, genAi, "win-x64", [pair]).Status);
        });
    }

    [Fact]
    public void AlternateResolverDirectoryIsUsedAndOnlyCurrentArchitectureIsConsidered()
    {
        WithFiles((directory, ort, genAi) =>
        {
            string managed = Path.Join(directory, "managed");
            string native = Path.Join(managed, "runtimes", "win-arm64", "native");
            Directory.CreateDirectory(native);
            string alternate = Path.Join(native, "onnxruntime.dll");
            File.WriteAllBytes(alternate, [3]);
            Assert.Equal(Path.GetFullPath(alternate),
                GenAiNativeRuntimeSelection.FindCandidate(managed, directory, "win-arm64", "onnxruntime.dll"));
            Assert.Equal(Path.GetFullPath(ort),
                GenAiNativeRuntimeSelection.FindCandidate(managed, directory, "win-x64", "onnxruntime.dll"));
        });
    }

    [Theory]
    [InlineData("Cpu", "Microsoft.ML.OnnxRuntimeGenAI")]
    [InlineData("Cuda", "Microsoft.ML.OnnxRuntimeGenAI.Cuda")]
    [InlineData("WinML", "Microsoft.ML.OnnxRuntimeGenAI.WinML")]
    public void CentralPackagePinsAreSupportedWithGenAi(string genAiFlavor, string genAiPackageId)
    {
        string? directory = AppContext.BaseDirectory;
        while (directory is not null && !File.Exists(Path.Join(directory, "Directory.Packages.props")))
        {
            directory = Path.GetDirectoryName(directory);
        }

        Assert.NotNull(directory);
        XElement packageVersions = XDocument.Load(Path.Join(directory, "Directory.Packages.props")).Root!;
        bool winMl = genAiFlavor == "WinML";
        string ortVersion = packageVersions.Descendants(winMl ? "WindowsMlOnnxRuntimeVersion" : "OnnxRuntimeVersion")
            .Single().Value.Trim();
        string ortPackageVersion = winMl
            ? packageVersions.Descendants("PackageVersion")
                .Single(element => (string?)element.Attribute("Include") == "Microsoft.Windows.AI.MachineLearning")
                .Attribute("Version")!.Value
            : ortVersion;
        string genAiVersion = packageVersions.Descendants("PackageVersion")
            .Single(element => (string?)element.Attribute("Include") == genAiPackageId)
            .Attribute("Version")!.Value;
        Assert.All(packageVersions.Descendants("PackageVersion")
                .Where(element => ((string?)element.Attribute("Include"))?.StartsWith("Microsoft.ML.OnnxRuntimeGenAI", StringComparison.Ordinal) == true),
            element => Assert.Equal(genAiVersion, element.Attribute("Version")!.Value));
        var pair = new NativePairManifest(winMl ? "WindowsML" : "Stock", ortPackageVersion, ortVersion,
            genAiFlavor, genAiVersion, "win-x64", "ort", "genai");

        Assert.True(GenAiNativeRuntimeSelection.IsSupported(pair),
            $"Update the stock ORT and GenAI allow-list in GenAiNativeRuntimeSelection for {genAiPackageId} {genAiVersion} with ORT {ortVersion}.");
        Assert.True(GenAiNativeRuntimeSelection.IsKnownOrt(pair),
            $"Update the stock ORT allow-list in GenAiNativeRuntimeSelection to {ortVersion}.");
    }

    private static NativePairManifest Pair(string ort, string genAi) =>
        new("WindowsML", "2.4.89", "1.27.1", "WinML", "0.17.1", "win-x64",
            GenAiNativeRuntimeSelection.HashFile(ort), GenAiNativeRuntimeSelection.HashFile(genAi));

    private static void WithFiles(Action<string, string, string> action)
    {
        string directory = Path.Join(Path.GetTempPath(), $"trackdub-native-pair-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            string ort = Path.Join(directory, "onnxruntime.dll");
            string genAi = Path.Join(directory, "onnxruntime-genai.dll");
            File.WriteAllBytes(ort, [1, 2, 3]);
            File.WriteAllBytes(genAi, [4, 5, 6]);
            action(directory, ort, genAi);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }
}
