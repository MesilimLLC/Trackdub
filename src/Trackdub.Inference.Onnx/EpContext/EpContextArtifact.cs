using System.Text.Json;
using Trackdub.Domain;
using Trackdub.Inference.Runtime.TensorRtRtx;

namespace Trackdub.Inference.Onnx.EpContext;

/// <summary>
/// EP-context artifact paths and machine stamp. A stamp is valid only for the model bytes,
/// GPU architecture, driver, and TRT RTX EP version it was compiled against (the triggers it
/// shares with <c>SmokeVerdictKey</c> and the residual engine cache), and additionally for the
/// host ORT runtime version and, when supplied, the compile-option identity, which smoke
/// verdicts do not track.
/// Load-path checks require schema 2 content hashes for the source graph and its external
/// weights. Length and modification time reject obvious replacements before that hash.
/// Unknown hardware or driver values do not match. Compile-option identity is compared when
/// the caller supplies the options used for the session.
/// </summary>
public static class EpContextArtifact
{
    public const string EpContextSuffix = ".epc";
    public const int CurrentSchemaVersion = 2;
    private const int LargeModelEmbedThresholdBytes = 2_000_000_000;

    public sealed record Stamp(
        int SchemaVersion,
        string SourceFileName,
        long SourceLengthBytes,
        long SourceLastWriteUtcTicks,
        string? SourceSha256,
        string GpuArchitecture,
        string? DriverVersion,
        string? TrtRtxEpVersion,
        DateTimeOffset CreatedAtUtc,
        long? ExternalDataLengthBytes = null,
        long? ExternalDataLastWriteUtcTicks = null,
        long? ArtifactExternalInitializersLengthBytes = null,
        long? ArtifactExternalInitializersLastWriteUtcTicks = null,
        IReadOnlyList<ArtifactFile>? ArtifactFiles = null,
        string? ExternalDataSha256 = null,
        string? OrtRuntimeVersion = null,
        string? CompileOptionsIdentity = null)
    {
        public string EnvironmentFingerprint =>
            $"{GpuArchitecture}|{Normalize(DriverVersion)}|{Normalize(TrtRtxEpVersion)}|{Normalize(OrtRuntimeVersion)}";

        public bool HasIdentifiedHardware =>
            !IsUnknown(GpuArchitecture) && !IsUnknown(DriverVersion) && !IsUnknown(OrtRuntimeVersion);

        /// <summary>
        /// Validates the compiled artifact's optional external-initializers sidecar identity.
        /// A missing sidecar or a replaced sidecar invalidates the artifact even when the
        /// main ONNX file is unchanged.
        /// </summary>
        public bool MatchesArtifactExternalInitializers(FileInfo? sidecar, bool sidecarRequired = false) =>
            (!sidecarRequired || (sidecar?.Exists ?? false)) &&
            (sidecar?.Exists ?? false) == ArtifactExternalInitializersLengthBytes.HasValue &&
            (!ArtifactExternalInitializersLengthBytes.HasValue ||
                (sidecar is not null &&
                 sidecar.Length == ArtifactExternalInitializersLengthBytes.Value &&
                 sidecar.LastWriteTimeUtc.Ticks == ArtifactExternalInitializersLastWriteUtcTicks));

        /// <summary>
        /// Compares the source model file and, when present, its external-weights sibling
        /// (audit: replacing external data must invalidate the artifact even though the
        /// .onnx container's length and mtime do not change).
        /// </summary>
        public bool MatchesSource(FileInfo source, FileInfo? externalData = null)
        {
            if (SchemaVersion < CurrentSchemaVersion || string.IsNullOrWhiteSpace(SourceSha256))
            {
                return false;
            }

            if (source.Length != SourceLengthBytes ||
                source.LastWriteTimeUtc.Ticks != SourceLastWriteUtcTicks ||
                !string.Equals(ComputeSha256(source.FullName), SourceSha256, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            bool externalExists = externalData?.Exists ?? false;
            if (externalExists != ExternalDataLengthBytes.HasValue)
            {
                return false;
            }

            if (!externalExists)
            {
                return ExternalDataSha256 is null;
            }

            return externalData is not null &&
                externalData.Length == ExternalDataLengthBytes &&
                externalData.LastWriteTimeUtc.Ticks == ExternalDataLastWriteUtcTicks &&
                !string.IsNullOrWhiteSpace(ExternalDataSha256) &&
                string.Equals(ComputeSha256(externalData.FullName), ExternalDataSha256, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Checks the engine files a non-embedded artifact loads from its own directory. Stamps
        /// written before this list existed carry <see langword="null"/> and check nothing here.
        /// </summary>
        public bool MatchesArtifactFiles(string artifactDirectory) =>
            ArtifactFiles is null ||
            ArtifactFiles.All(file =>
            {
                var info = new FileInfo(Path.Join(artifactDirectory, file.Name));
                return info.Exists && info.Length == file.LengthBytes &&
                    info.LastWriteTimeUtc.Ticks == file.LastWriteUtcTicks;
            });

        private static string Normalize(string? value) =>
            string.IsNullOrWhiteSpace(value) ? "unknown" : value.Trim();

        private static bool IsUnknown(string? value) =>
            string.IsNullOrWhiteSpace(value) ||
            value.Trim().Equals("unknown", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>A file published beside the EP-context model, such as a serialized engine.</summary>
    public sealed record ArtifactFile(string Name, long LengthBytes, long LastWriteUtcTicks = 0);

    public static bool IsEpContextPath(string path) =>
        path.EndsWith(EpContextSuffix + ".onnx", StringComparison.OrdinalIgnoreCase);

    public static string GetEpContextPath(string sourceModelPath)
    {
        string? directory = Path.GetDirectoryName(sourceModelPath);
        string name = Path.GetFileNameWithoutExtension(sourceModelPath);
        return directory is null
            ? name + EpContextSuffix + ".onnx"
            : Path.Join(directory, name + EpContextSuffix + ".onnx");
    }

    public static string GetStampPath(string sourceModelPath)
    {
        string epContextPath = GetEpContextPath(sourceModelPath);
        return epContextPath[..^".onnx".Length] + ".stamp.json";
    }

    /// <summary>
    /// Sibling external-weights file for <paramref name="sourceModelPath"/>: the
    /// <c>&lt;model&gt;.onnx.data</c> convention this codebase writes (see
    /// <c>OliveModelOptimizationService</c>), or the <c>&lt;model&gt;.onnx_data</c> convention of
    /// Optimum exports when only that exists. Not every model has one.
    /// </summary>
    public static string GetSourceExternalDataPath(string sourceModelPath)
    {
        string dotData = sourceModelPath + ".data";
        string underscoreData = sourceModelPath + "_data";
        return !File.Exists(dotData) && File.Exists(underscoreData) ? underscoreData : dotData;
    }

    /// <summary>Bytes of the model file plus its external weights, which the engine roughly mirrors.</summary>
    public static long GetSourceTotalBytes(string sourceModelPath)
    {
        var source = new FileInfo(sourceModelPath);
        var externalData = new FileInfo(GetSourceExternalDataPath(sourceModelPath));
        return (source.Exists ? source.Length : 0) + (externalData.Exists ? externalData.Length : 0);
    }

    /// <summary>
    /// External-initializers sidecar ORT writes next to a compiled EP-context artifact when
    /// the artifact is too large to embed (see <see cref="EpContextCompiler"/>). Not every
    /// artifact has one — only models compiled with embedding disabled.
    /// </summary>
    public static string GetArtifactExternalInitializersPath(string epContextPath)
    {
        string? directory = Path.GetDirectoryName(epContextPath);
        string sidecarName = Path.GetFileNameWithoutExtension(epContextPath) + ".ext_init";
        return string.IsNullOrEmpty(directory) ? sidecarName : Path.Join(directory, sidecarName);
    }

    /// <summary>
    /// Embeds the engine in the EP-context graph only while it fits protobuf's 2 GB limit. The
    /// engine roughly mirrors the weights, so models whose weights live in external data count
    /// those bytes too.
    /// </summary>
    public static bool ShouldEmbedEpContext(string sourceModelPath)
    {
        try
        {
            return GetSourceTotalBytes(sourceModelPath) < LargeModelEmbedThresholdBytes;
        }
        catch (IOException)
        {
            return true;
        }
    }

    /// <summary>
    /// Returns a loadable EP-context path when an artifact exists and its stamp matches the
    /// current machine fingerprint and source identity; otherwise <see langword="null"/>.
    /// </summary>
    public static string? TryResolveValidLoadPath(
        string sourceModelPath,
        string currentEnvironmentFingerprint,
        string? currentCompileOptionsIdentity = null)
    {
        if (IsEpContextPath(sourceModelPath) || !File.Exists(sourceModelPath))
        {
            return null;
        }

        string epContextPath = GetEpContextPath(sourceModelPath);
        string stampPath = GetStampPath(sourceModelPath);
        if (!File.Exists(epContextPath) || !File.Exists(stampPath))
        {
            return null;
        }

        return HasMatchingSourceStamp(
            sourceModelPath,
            epContextPath,
            stampPath,
            currentEnvironmentFingerprint,
            currentCompileOptionsIdentity);
    }

    private static string? HasMatchingSourceStamp(
        string sourceModelPath,
        string epContextPath,
        string stampPath,
        string currentEnvironmentFingerprint,
        string? currentCompileOptionsIdentity = null)
    {
        Stamp? stamp = TryReadStamp(stampPath);
        if (stamp is null ||
            stamp.SchemaVersion < CurrentSchemaVersion ||
            !stamp.HasIdentifiedHardware ||
            !stamp.EnvironmentFingerprint.Equals(currentEnvironmentFingerprint, StringComparison.Ordinal))
        {
            return null;
        }

        if (currentCompileOptionsIdentity is not null &&
            !string.Equals(stamp.CompileOptionsIdentity, currentCompileOptionsIdentity, StringComparison.Ordinal))
        {
            return null;
        }

        try
        {
            string externalDataPath = GetSourceExternalDataPath(sourceModelPath);
            FileInfo? externalData = File.Exists(externalDataPath) ? new FileInfo(externalDataPath) : null;
            string sidecarPath = GetArtifactExternalInitializersPath(epContextPath);
            FileInfo? sidecar = File.Exists(sidecarPath) ? new FileInfo(sidecarPath) : null;
            // Stamps that predate ArtifactFiles cannot say what a non-embedded artifact needs, so
            // they keep requiring the external-initializers sidecar.
            return stamp.MatchesSource(new FileInfo(sourceModelPath), externalData) &&
                stamp.MatchesArtifactExternalInitializers(
                    sidecar,
                    sidecarRequired: stamp.ArtifactFiles is null && !ShouldEmbedEpContext(sourceModelPath)) &&
                stamp.MatchesArtifactFiles(Path.GetDirectoryName(epContextPath) ?? string.Empty)
                ? epContextPath
                : null;
        }
        catch (IOException)
        {
            return null;
        }
    }

    /// <summary>
    /// Native ORT version this build compiles EP-context artifacts against, stamped into the assembly
    /// from the central package pins (<c>WindowsMlOnnxRuntimeVersion</c> / <c>OnnxRuntimeVersion</c>).
    /// </summary>
    public static string HostOrtRuntimeVersion { get; } =
        typeof(EpContextArtifact).Assembly
            .GetCustomAttributes(typeof(System.Reflection.AssemblyMetadataAttribute), inherit: false)
            .Cast<System.Reflection.AssemblyMetadataAttribute>()
            .FirstOrDefault(attribute => attribute.Key == "TrackdubHostOrtRuntimeVersion")?.Value
        ?? "unknown";

    /// <summary>
    /// Stable identity of the compile/provider options that affect the engine.
    /// Path-valued options such as the runtime cache directory are omitted, and so is
    /// <c>enable_cuda_graph</c>: capture is a runtime session setting, not an engine-build option, so an
    /// artifact is valid whichever capture mode the session that loads it uses.
    /// </summary>
    public static string BuildCompileOptionsIdentity(
        IReadOnlyDictionary<string, string>? providerOptions,
        bool embedEpContext)
    {
        var pairs = new SortedDictionary<string, string>(StringComparer.Ordinal)
        {
            ["ep_context_embed_mode"] = embedEpContext ? "1" : "0",
        };
        if (providerOptions is not null)
        {
            foreach ((string key, string value) in providerOptions)
            {
                if (key.EndsWith("_path", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(key, "enable_cuda_graph", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                pairs[key] = value;
            }
        }

        return string.Join('\n', pairs.Select(pair => pair.Key + "=" + pair.Value));
    }

    public static Stamp CreateStamp(
        string sourceModelPath,
        FileInfo source,
        string? sourceSha256,
        string gpuArchitecture,
        string? driverVersion,
        IReadOnlyList<ArtifactFile>? artifactFiles = null,
        string? outputPath = null,
        string? compileOptionsIdentity = null)
    {
        string externalDataPath = GetSourceExternalDataPath(sourceModelPath);
        var externalData = new FileInfo(externalDataPath);
        string epContextPath = outputPath ?? GetEpContextPath(sourceModelPath);
        var artifactSidecar = new FileInfo(GetArtifactExternalInitializersPath(epContextPath));
        return new(
            SchemaVersion: CurrentSchemaVersion,
            SourceFileName: Path.GetFileName(sourceModelPath),
            SourceLengthBytes: source.Length,
            SourceLastWriteUtcTicks: source.LastWriteTimeUtc.Ticks,
            SourceSha256: sourceSha256 ?? ComputeSha256(source.FullName),
            GpuArchitecture: gpuArchitecture,
            DriverVersion: driverVersion,
            TrtRtxEpVersion: TensorRtRtxProviderConstants.BundledFingerprintVersion,
            ExternalDataLengthBytes: externalData.Exists ? externalData.Length : null,
            ExternalDataLastWriteUtcTicks: externalData.Exists ? externalData.LastWriteTimeUtc.Ticks : null,
            CreatedAtUtc: DateTimeOffset.UtcNow,
            ArtifactExternalInitializersLengthBytes: artifactSidecar.Exists ? artifactSidecar.Length : null,
            ArtifactExternalInitializersLastWriteUtcTicks: artifactSidecar.Exists ? artifactSidecar.LastWriteTimeUtc.Ticks : null,
            ArtifactFiles: artifactFiles,
            ExternalDataSha256: externalData.Exists ? ComputeSha256(externalData.FullName) : null,
            OrtRuntimeVersion: HostOrtRuntimeVersion,
            CompileOptionsIdentity: compileOptionsIdentity);
    }

    /// <summary>
    /// Moves a freshly compiled artifact from <paramref name="stagingDirectory"/> (where it was
    /// written under its final file names) next to <paramref name="sourceModelPath"/>, replacing
    /// any previous artifact. The old stamp goes first so no reader validates a half-replaced
    /// set, and the EP-context model goes last, after the engine files it references. Returns the
    /// published engine files (everything except the EP-context model and its
    /// external-initializers sidecar) for the new stamp.
    /// </summary>
    public static IReadOnlyList<ArtifactFile> PublishStagedArtifact(
        string stagingDirectory,
        string sourceModelPath,
        string? outputPath = null,
        Func<IReadOnlyList<ArtifactFile>, Stamp>? createStamp = null,
        Func<string, bool>? existingArtifactUsable = null)
    {
        string epContextPath = Path.GetFullPath(outputPath ?? GetEpContextPath(sourceModelPath));
        string targetDirectory = Path.GetDirectoryName(Path.GetFullPath(epContextPath))!;
        string epContextName = Path.GetFileName(epContextPath);
        string sidecarName = Path.GetFileName(GetArtifactExternalInitializersPath(epContextPath));
        string stampPath = GetOutputStampPath(epContextPath);
        // Validate before touching a previous artifact, including for noncanonical output paths.
        if (!File.Exists(Path.Join(stagingDirectory, epContextName)))
        {
            throw new FileNotFoundException("Staged EP-context model is missing.", epContextPath);
        }

        // FileShare.None serializes publishers across processes, not just sessions in this host.
        // Keep the lock file: deleting it after releasing would let waiters lock different inodes.
        using FileStream publicationLock = AcquirePublicationLock(epContextPath + ".publish.lock");
        Stamp? probeStamp = null;
        if (createStamp is not null)
        {
            // Refuse before touching any existing artifact: a stamp built from the
            // placeholder "unknown" hardware identity would replace a usable artifact
            // with one no lookup can ever validate.
            probeStamp = createStamp([]);
            if (!probeStamp.HasIdentifiedHardware)
            {
                throw new InvalidOperationException(
                    "Refusing to publish an EP-context artifact stamped with unidentified hardware " +
                    $"(architecture '{probeStamp.GpuArchitecture}', driver '{probeStamp.DriverVersion ?? "unknown"}'). " +
                    "Load the hardware identity (host startup / EP-context warmup) before compiling or capturing.");
            }
        }

        if (createStamp is not null && File.Exists(epContextPath) && File.Exists(stampPath))
        {
            Stamp probe = probeStamp!;
            if (HasMatchingSourceStamp(
                    sourceModelPath,
                    epContextPath,
                    stampPath,
                    probe.EnvironmentFingerprint,
                    probe.CompileOptionsIdentity) is not null &&
                (existingArtifactUsable?.Invoke(epContextPath) ?? true))
            {
                return TryReadStamp(stampPath)!.ArtifactFiles ?? [];
            }
        }

        string rollbackDirectory = Path.Join(targetDirectory, ".epc-rollback-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(rollbackDirectory);
        var movedPaths = new List<string>();
        var backups = new List<(string Original, string Backup)>();
        var published = new List<ArtifactFile>();
        bool mayDeleteRollback = true;
        try
        {
            Stamp? previous = File.Exists(stampPath) ? TryReadStamp(stampPath) : null;
            Backup(stampPath);
            foreach (ArtifactFile file in (previous?.ArtifactFiles ?? []).Where(file => Path.GetFileName(file.Name) == file.Name))
            {
                Backup(Path.Join(targetDirectory, file.Name));
            }

            Backup(GetArtifactExternalInitializersPath(epContextPath));
            Backup(epContextPath);
            foreach (string stagedPath in Directory.EnumerateFiles(stagingDirectory)
                .OrderBy(path => Path.GetFileName(path).Equals(epContextName, StringComparison.OrdinalIgnoreCase)))
            {
                string name = Path.GetFileName(stagedPath);
                string target = Path.Join(targetDirectory, name);
                Backup(target);
                File.Move(stagedPath, target);
                movedPaths.Add(target);
                if (!name.Equals(epContextName, StringComparison.OrdinalIgnoreCase) &&
                    !name.Equals(sidecarName, StringComparison.OrdinalIgnoreCase))
                {
                    var info = new FileInfo(target);
                    published.Add(new ArtifactFile(name, info.Length, info.LastWriteTimeUtc.Ticks));
                }
            }

            if (createStamp is not null)
            {
                WriteStamp(sourceModelPath, createStamp(published), epContextPath);
                movedPaths.Add(stampPath);
            }

            return published;
        }
        catch (Exception publishError)
        {
            // This list also covers engines moved before a stamp could record them.
            mayDeleteRollback = false;
            try
            {
                foreach (string path in movedPaths) File.Delete(path);
                // Restore the old stamp last so readers cannot validate a partially restored set.
                foreach ((string original, string backup) in backups.AsEnumerable().Reverse())
                {
                    File.Move(backup, original, overwrite: true);
                }
                mayDeleteRollback = true;
            }
            catch (Exception rollbackError) when (rollbackError is IOException or UnauthorizedAccessException)
            {
                // Keep backups for recovery, and retain the original publication failure for callers.
                System.Diagnostics.Trace.TraceWarning(
                    $"EpContextArtifact: rollback failed after publish error '{publishError.Message}': {rollbackError.Message}");
            }
            throw;
        }
        finally
        {
            // If restoration itself fails, preserve the backup files for recovery.
            if (mayDeleteRollback) EpContextCompiler.TryDeleteDirectory(rollbackDirectory);
        }

        void Backup(string path)
        {
            if (!File.Exists(path)) return;
            string backup = Path.Join(rollbackDirectory, backups.Count.ToString(System.Globalization.CultureInfo.InvariantCulture));
            File.Move(path, backup);
            backups.Add((path, backup));
        }
    }

    private static FileStream AcquirePublicationLock(string path)
    {
        var timer = System.Diagnostics.Stopwatch.StartNew();
        while (true)
        {
            try
            {
                return new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
            }
            catch (IOException) when (timer.Elapsed < TimeSpan.FromSeconds(30))
            {
                // Another process may be publishing; wait briefly, with a bounded failure path.
                Thread.Sleep(25);
            }
        }
    }

    private static string GetOutputStampPath(string epContextPath) =>
        Path.ChangeExtension(epContextPath, ".stamp.json");

    /// <summary>Removes the artifact for <paramref name="sourceModelPath"/>: stamp first, then its files.</summary>
    public static void DeleteArtifact(string sourceModelPath)
    {
        string epContextPath = GetEpContextPath(sourceModelPath);
        string stampPath = GetStampPath(sourceModelPath);
        Stamp? stamp = File.Exists(stampPath) ? TryReadStamp(stampPath) : null;
        TryDeleteFile(stampPath);
        string directory = Path.GetDirectoryName(Path.GetFullPath(epContextPath))!;
        // Recorded names come from our own staging directory; never follow a path out of it.
        foreach (ArtifactFile file in (stamp?.ArtifactFiles ?? []).Where(file => Path.GetFileName(file.Name) == file.Name))
        {
            TryDeleteFile(Path.Join(directory, file.Name));
        }

        TryDeleteFile(GetArtifactExternalInitializersPath(epContextPath));
        TryDeleteFile(epContextPath);
    }

    public static void WriteStamp(string sourceModelPath, Stamp stamp, string? outputPath = null)
    {
        ArgumentNullException.ThrowIfNull(stamp);
        // A stamp is the artifact's machine identity. Persisting one built from the
        // placeholder "unknown" hardware identity would create a dead artifact and,
        // on publish, displace a valid one. Refuse instead of faking readiness.
        if (!stamp.HasIdentifiedHardware)
        {
            throw new InvalidOperationException(
                "Refusing to write an EP-context stamp with unidentified hardware " +
                $"(architecture '{stamp.GpuArchitecture}', driver '{stamp.DriverVersion ?? "unknown"}', " +
                $"ORT '{stamp.OrtRuntimeVersion ?? "unknown"}'). " +
                "Load the hardware identity (host startup / EP-context warmup) before stamping.");
        }

        string stampPath = outputPath is null ? GetStampPath(sourceModelPath) : GetOutputStampPath(outputPath);
        string? directory = Path.GetDirectoryName(stampPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        string json = JsonSerializer.Serialize(stamp, new JsonSerializerOptions { WriteIndented = true });
        string tempPath = stampPath + ".tmp-" + Guid.NewGuid().ToString("N");
        try
        {
            File.WriteAllText(tempPath, json);
            try
            {
                File.Move(tempPath, stampPath, overwrite: true);
            }
            catch (IOException)
            {
                if (File.Exists(stampPath))
                {
                    File.Delete(stampPath);
                }

                File.Move(tempPath, stampPath);
            }
        }
        catch
        {
            TryDeleteFile(tempPath);
            throw;
        }
    }

    private static void TryDeleteFile(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"EpContextArtifact: failed to delete temp file '{path}': {ex.Message}");
        }
    }

    public static Stamp? TryReadStamp(string stampPath)
    {
        try
        {
            return JsonSerializer.Deserialize<Stamp>(File.ReadAllText(stampPath));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException)
        {
            return null;
        }
    }

    // Hashes are paid once per file version per process; length and modification time gate
    // replacements. A same-size, same-mtime overwrite is caught by the next process.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<(string Path, long Length, long LastWriteUtcTicks), string>
        Sha256ByFileVersion = new();

    public static string ComputeSha256(string filePath)
    {
        var info = new FileInfo(filePath);
        var key = (info.FullName, info.Length, info.LastWriteTimeUtc.Ticks);
        if (Sha256ByFileVersion.TryGetValue(key, out string? cached))
        {
            return cached;
        }

        using FileStream stream = File.OpenRead(filePath);
        string hash = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream)).ToLowerInvariant();
        Sha256ByFileVersion[key] = hash;
        return hash;
    }

    internal static void ClearHashCacheForTesting() => Sha256ByFileVersion.Clear();
}
