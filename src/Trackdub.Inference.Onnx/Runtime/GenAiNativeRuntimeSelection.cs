using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace Trackdub.Inference.Onnx.Runtime;

internal enum NativePairStatus { NotVerified, Incompatible, Compatible }
internal sealed record NativePairManifest(string OrtFlavor, string OrtPackageVersion,
    string OrtVersion, string GenAiFlavor, string GenAiVersion, string Rid,
    string OrtSha256, string GenAiSha256);
internal sealed record NativePairVerification(NativePairStatus Status, string Detail,
    string? OrtPath = null, string? GenAiPath = null, NativePairManifest? Pair = null);

internal static class GenAiNativeRuntimeSelection
{
    private static readonly object Gate = new();
    private static nint ortHandle;
    private static nint genAiHandle;
    private static NativePairManifest? selectedPair;
    private static string? selectedOrtPath;
    private static string? selectedGenAiPath;

    internal static bool IsSupported(NativePairManifest pair) =>
        (pair.GenAiVersion is "0.17.1" or "0.18.0" &&
         ((pair.OrtFlavor == "WindowsML" && pair.OrtPackageVersion == "2.4.89" &&
           pair.OrtVersion == "1.27.1" && pair.GenAiFlavor == "WinML") ||
          (pair.OrtFlavor == "Stock" && pair.OrtPackageVersion == "1.31.0" &&
           pair.OrtVersion == "1.31.0" && pair.GenAiFlavor is "Cpu" or "Cuda")));

    /// <summary>
    /// ORT file identity for a Windows ML host or the stock CUDA worker.
    /// A Stock row recorded next to GenAI.WinML is the worker's onnxruntime.dll, not a GenAI pair.
    /// GenAI loading still requires <see cref="IsSupported"/>.
    /// </summary>
    internal static bool IsKnownOrt(NativePairManifest pair) =>
        (pair.OrtFlavor == "WindowsML" && pair.OrtPackageVersion == "2.4.89" && pair.OrtVersion == "1.27.1") ||
        (pair.OrtFlavor == "Stock" && pair.OrtPackageVersion == "1.31.0" && pair.OrtVersion == "1.31.0");

    internal static string HashFile(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream));
    }

    internal static string? FindCandidate(string? assemblyDirectory, string baseDirectory,
        string rid, string fileName)
    {
        string[] roots = assemblyDirectory is null ? [baseDirectory] : [assemblyDirectory, baseDirectory];
        foreach (string root in roots.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            string rootFile = Path.Join(root, fileName);
            if (File.Exists(rootFile)) return Path.GetFullPath(rootFile);
            string ridFile = Path.Join(root, "runtimes", rid, "native", fileName);
            if (File.Exists(ridFile)) return Path.GetFullPath(ridFile);
        }
        return null;
    }

    internal static NativePairVerification Verify(string? ortPath, string? genAiPath,
        string rid, IReadOnlyList<NativePairManifest> manifests)
    {
        if (ortPath is null || genAiPath is null || !File.Exists(ortPath) || !File.Exists(genAiPath))
            return new(NativePairStatus.NotVerified, "ORT or GenAI native path is missing.", ortPath, genAiPath);
        try
        {
            string ortHash = HashFile(ortPath);
            string genAiHash = HashFile(genAiPath);
            NativePairManifest? pair = manifests.FirstOrDefault(p => p.Rid == rid &&
                string.Equals(p.OrtSha256, ortHash, StringComparison.OrdinalIgnoreCase) &&
                string.Equals(p.GenAiSha256, genAiHash, StringComparison.OrdinalIgnoreCase));
            if (pair is null)
                return new(NativePairStatus.NotVerified,
                    "Selected native files do not match build-time package provenance.", ortPath, genAiPath);
            return new(IsSupported(pair) ? NativePairStatus.Compatible : NativePairStatus.Incompatible,
                $"ORT {pair.OrtFlavor}/{pair.OrtPackageVersion} (native {pair.OrtVersion}); GenAI {pair.GenAiFlavor}/{pair.GenAiVersion}.",
                Path.GetFullPath(ortPath), Path.GetFullPath(genAiPath), pair);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return new(NativePairStatus.NotVerified, ex.Message, ortPath, genAiPath);
        }
    }

    private static string Rid => RuntimeInformation.ProcessArchitecture switch
    {
        Architecture.X64 => "win-x64",
        Architecture.Arm64 => "win-arm64",
        _ => throw new InvalidOperationException("Native runtime identity not verified: unsupported process architecture.")
    };

    private static IReadOnlyList<NativePairManifest> ReadManifests()
    {
        return typeof(GenAiNativeRuntimeSelection).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
            .Where(attribute => attribute.Key == "Trackdub.NativePair" && attribute.Value is not null)
            .Select(attribute => attribute.Value!.Split('|'))
            .Where(parts => parts.Length == 8)
            .Select(parts => new NativePairManifest(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7]))
            .ToList();
    }

    private static string? LoadedPath(string fileName)
    {
        string[] matches;
        try
        {
            using var process = Process.GetCurrentProcess();
            matches = process.Modules.Cast<ProcessModule>()
                .Where(m => string.Equals(m.ModuleName, fileName, StringComparison.OrdinalIgnoreCase))
                .Select(m => Path.GetFullPath(m.FileName)).Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
        }
        catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or NotSupportedException or InvalidOperationException)
        {
            // Restricted processes cannot list their modules. Callers then fall back to the on-disk
            // candidate, and the post-load identity check fails closed on a null path.
            return null;
        }

        if (matches.Length > 1)
            throw new InvalidOperationException($"Native runtime identity not verified: multiple loaded {fileName} modules.");
        return matches.SingleOrDefault();
    }

    private static string? SelectPath(Type managedType, string fileName) => LoadedPath(fileName) ??
        FindCandidate(Path.GetDirectoryName(managedType.Assembly.Location), AppContext.BaseDirectory, Rid, fileName);

    internal static nint EnsureOrtLoaded(bool requireVerified = false)
    {
        lock (Gate)
        {
            if (ortHandle != nint.Zero)
            {
                if (requireVerified && selectedPair is null)
                    throw new InvalidOperationException("Native runtime identity not verified: loaded ORT has unknown provenance.");
                return ortHandle;
            }
            string? path = SelectPath(typeof(OrtEnv), "onnxruntime.dll");
            if (path is null) throw new InvalidOperationException("Native runtime identity not verified: ORT path is missing.");
            string hash = HashFile(path);
            selectedPair = ReadManifests().FirstOrDefault(p => p.Rid == Rid && IsKnownOrt(p) &&
                string.Equals(p.OrtSha256, hash, StringComparison.OrdinalIgnoreCase));
            if (selectedPair is null && requireVerified)
                throw new InvalidOperationException($"Native runtime identity not verified: '{path}' has no supported build-time provenance.");
            nint handle = NativeLibrary.Load(path);
            AssertLoadedPath(handle, path);
            var getBase = Marshal.GetDelegateForFunctionPointer<GetApiBase>(NativeLibrary.GetExport(handle, "OrtGetApiBase"));
            nint apiBase = getBase();
            if (apiBase == nint.Zero) throw new InvalidOperationException("Native runtime identity not verified: OrtGetApiBase returned null.");
            var getVersion = Marshal.GetDelegateForFunctionPointer<GetVersionString>(Marshal.ReadIntPtr(apiBase, IntPtr.Size));
            string? version = Marshal.PtrToStringAnsi(getVersion());
            if (selectedPair is not null && version != selectedPair.OrtVersion)
                throw new InvalidOperationException($"Native ORT version '{version}' does not match supported version '{selectedPair.OrtVersion}'.");
            RegisterResolver(typeof(OrtEnv).Assembly, ResolveOrt, "onnxruntime.dll", path);
            selectedOrtPath = Path.GetFullPath(path);
            ortHandle = handle;
            return handle;
        }
    }

    internal static void EnsureGenAiLoaded()
    {
        lock (Gate)
        {
            if (genAiHandle != nint.Zero) return;
            string? path = SelectPath(typeof(Model), "onnxruntime-genai.dll");
            string? ortPath = selectedOrtPath ?? SelectPath(typeof(OrtEnv), "onnxruntime.dll");
            NativePairVerification verification = Verify(ortPath, path, Rid, ReadManifests());
            if (verification.Status != NativePairStatus.Compatible)
                throw new InvalidOperationException($"GenAI native pair {verification.Status}: {verification.Detail} ORT='{verification.OrtPath}', GenAI='{verification.GenAiPath}'.");
            EnsureOrtLoaded(requireVerified: true);
            nint handle = NativeLibrary.Load(verification.GenAiPath!);
            AssertLoadedPath(handle, verification.GenAiPath!);
            string? loadedOrt = LoadedPath("onnxruntime.dll");
            if (!string.Equals(loadedOrt, selectedOrtPath, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("GenAI native pair not verified: ORT module identity changed during GenAI loading.");
            RegisterResolver(typeof(Model).Assembly, ResolveGenAi, "onnxruntime-genai.dll", verification.GenAiPath!);
            selectedPair = verification.Pair;
            selectedGenAiPath = verification.GenAiPath;
            genAiHandle = handle;
        }
    }

    internal static string Describe() =>
        $"ORT='{selectedOrtPath}', flavor={selectedPair?.OrtFlavor}, nativeVersion={selectedPair?.OrtVersion}; " +
        $"GenAI='{selectedGenAiPath}', flavor={selectedPair?.GenAiFlavor}, version={selectedPair?.GenAiVersion}";

    private static nint ResolveOrt(string name, Assembly assembly, DllImportSearchPath? searchPath)
    {
        lock (Gate)
            return name.Equals("onnxruntime", StringComparison.OrdinalIgnoreCase) || name.Equals("onnxruntime.dll", StringComparison.OrdinalIgnoreCase)
                ? ortHandle : nint.Zero;
    }
    private static nint ResolveGenAi(string name, Assembly assembly, DllImportSearchPath? searchPath)
    {
        lock (Gate)
            return name.Equals("onnxruntime-genai", StringComparison.OrdinalIgnoreCase) || name.Equals("onnxruntime-genai.dll", StringComparison.OrdinalIgnoreCase)
                ? genAiHandle : nint.Zero;
    }

    // Register before publishing initialized state. A competing resolver is not proof of identity, but the
    // managed ORT/GenAI bindings register their own resolver on first use (for example, when an earlier stage
    // already ran ORT in this process). That is acceptable only when the module they resolved is the file whose
    // hash was just verified; any other state, including an unreadable module list, still fails closed.
    internal static void RegisterResolver(Assembly assembly, DllImportResolver resolver, string moduleFileName, string verifiedPath)
    {
        try
        {
            NativeLibrary.SetDllImportResolver(assembly, resolver);
        }
        catch (InvalidOperationException) when (string.Equals(
            LoadedPath(moduleFileName), Path.GetFullPath(verifiedPath), StringComparison.OrdinalIgnoreCase))
        {
        }
    }

    private static void AssertLoadedPath(nint handle, string expected)
    {
        var buffer = new System.Text.StringBuilder(32768);
        uint length = GetModuleFileName(handle, buffer, buffer.Capacity);
        if (length == 0 || length >= buffer.Capacity ||
            !string.Equals(Path.GetFullPath(buffer.ToString()), Path.GetFullPath(expected), StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException($"Native runtime identity not verified: loaded module differs from '{expected}'.");
    }

    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate nint GetApiBase();
    [UnmanagedFunctionPointer(CallingConvention.Winapi)]
    private delegate nint GetVersionString();
    [DllImport("kernel32.dll", EntryPoint = "GetModuleFileNameW", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern uint GetModuleFileName(nint module, System.Text.StringBuilder buffer, int size);
}
