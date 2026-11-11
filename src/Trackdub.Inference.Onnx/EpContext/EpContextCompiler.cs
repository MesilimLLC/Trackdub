using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Madlad;
using Trackdub.Inference.Onnx.TensorRtRtx;
using Trackdub.Inference.Runtime.TensorRtRtx;
#if WINDOWS
using Trackdub.Inference.Onnx.WindowsMl;
#endif

namespace Trackdub.Inference.Onnx.EpContext;

/// <summary>
/// AOT-compiles a source ONNX graph into an EP-context model via ORT <c>OrtModelCompilationOptions</c>
/// so the next session load deserializes engines instead of JIT-compiling them at create time.
/// Residual CUDA kernels still land in <c>nv_runtime_cache_path</c> on first load of the EP-context model.
/// </summary>
public sealed class EpContextCompiler
{
    private readonly ITensorRtRtxProviderBootstrap _tensorRtRtxBootstrap;

    public EpContextCompiler(ITensorRtRtxProviderBootstrap? tensorRtRtxBootstrap = null)
    {
        _tensorRtRtxBootstrap = tensorRtRtxBootstrap ?? CreateDefaultBootstrap();
    }

    /// <summary>
    /// Same wiring shape as <c>WindowsExecutionProviderBootstrapper</c>'s default: resolve the
    /// installed bundle under the user-data root. A bare <c>TensorRtRtxPluginService.Shared</c>
    /// has no install path and would report the plugin missing on a normal install.
    /// </summary>
    private static ITensorRtRtxProviderBootstrap CreateDefaultBootstrap()
    {
        string userDataRoot = Path.Join(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Trackdub");
        return TensorRtRtxProviderBootstrapFactory.CreateWithDefaultInstallPath(userDataRoot);
    }
    public sealed record CompileResult(
        bool Success,
        string? EpContextPath,
        double CompileMilliseconds,
        string? FailureReason,
        string SelectedProvider = "unknown",
        IReadOnlyList<EpContextArtifact.ArtifactFile>? ArtifactFiles = null);

    public async Task<CompileResult> CompileAsync(string sourceModelPath, string epContextPath, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (!File.Exists(sourceModelPath))
        {
            return new CompileResult(false, null, 0, $"Source model not found: {sourceModelPath}");
        }

        if (MadladTrtProfiles.GetAotExclusionReason(sourceModelPath) is { } exclusionReason)
        {
            return new CompileResult(false, null, 0, exclusionReason);
        }

        // Compiling now would stamp the artifact with the placeholder "unknown" hardware
        // identity, producing an artifact no lookup can validate (and displacing a valid
        // one on publish). Defer honestly until startup/warmup loads the identity.
        if (!EpContextLoadPathResolver.HasIdentifiedHardware)
        {
            return new CompileResult(false, null, 0,
                "Hardware identity is not available (GPU architecture, driver version, or " +
                "host ORT version is unknown), so an EP-context compile would be stamped " +
                "'unknown'. Run host startup or EP-context warmup on a machine with an identifiable GPU.");
        }

        // TensorRT RTX is a standalone ORT EP ABI plugin. Unless registered before the first OrtEnv
        // touch, GetEpDevices() lists no TRT device and AppendTensorRtRtxOrFallbackProvider silently
        // compiles under DirectML/CPU — producing a reserialized graph with zero EP-context nodes
        // (measured: +146KB, and a ~6s cold-load regression versus the source). Register here so
        // the compile cannot quietly AOT the wrong provider.
        TensorRtRtxBootstrapResult registration = await _tensorRtRtxBootstrap
            .EnsureRegisteredAsync(allowProviderDownloads: false, cancellationToken)
            .ConfigureAwait(false);
        if (!registration.Succeeded)
        {
            return new CompileResult(
                false,
                null,
                0,
                $"TensorRT RTX plugin registration failed, so EP-context AOT compile is unavailable: {registration.Detail}");
        }

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(epContextPath))!);
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        // Compile into a guid-named temp subdirectory rather than a temp-suffixed final path: the
        // model, its sidecar, and any engine files keep their FINAL filenames while isolated, so the
        // names ORT embeds in the compiled model (derived from the output path) still resolve once
        // EpContextArtifact.PublishStagedArtifact moves the set next to the source model.
        string? tempDir = null;
        try
        {
#if WINDOWS
            WindowsMlOnnxRuntimeNativeResolver.EnsureInitialized();
#endif
            IReadOnlyDictionary<string, string>? modelTrtOptions = EpContextTrtProfiles.Resolve(sourceModelPath);
            using SessionOptions sessionOptions = CreateCompileSessionOptions(
                modelTrtOptions,
                out ExecutionProviderKind selectedProvider);
            string selectedLabel = selectedProvider.ToString();
            // EP context engines are provider-specific. Compiling on anything but TensorRT RTX
            // only reserializes (and graph-optimizes) the source — it cannot skip engine rebuild.
            if (selectedProvider is not ExecutionProviderKind.TensorRTRtx)
            {
                stopwatch.Stop();
                return new CompileResult(
                    false,
                    null,
                    stopwatch.Elapsed.TotalMilliseconds,
                    "EP-context AOT compile requires TensorRT RTX (the engine blob is TRT-RTX specific); "
                    + $"selected provider was '{selectedLabel}'.",
                    selectedLabel);
            }

            bool embed = EpContextArtifact.ShouldEmbedEpContext(sourceModelPath);
            string tempDirName = Path.GetFileName(".epc-tmp-" + Guid.NewGuid().ToString("N"));
            if (string.IsNullOrEmpty(tempDirName) || Path.IsPathRooted(tempDirName))
            {
                throw new InvalidOperationException("Generated EP-context temp directory name must be a relative, non-empty path segment.");
            }

            tempDir = Path.Join(Path.GetDirectoryName(Path.GetFullPath(epContextPath))!, tempDirName);
            Directory.CreateDirectory(tempDir);
            string outputFileName = Path.GetFileName(epContextPath);
            if (string.IsNullOrEmpty(outputFileName) || Path.IsPathRooted(outputFileName))
            {
                throw new InvalidOperationException($"EP-context output path '{epContextPath}' does not resolve to a valid relative file name.");
            }

            string tempPath = Path.Join(tempDir, outputFileName);
            using (var compileOptions = new OrtModelCompilationOptions(sessionOptions))
            {
                compileOptions.SetInputModelPath(sourceModelPath);
                compileOptions.SetOutputModelPath(tempPath);
                // Embed the compiled engine in the EP-context graph under 2GB (protobuf limit).
                // Otherwise ORT writes the engine as a separate file beside the output model and
                // externalizes any remaining initializers.
                compileOptions.SetEpContextEmbedMode(embed);
                if (!embed)
                {
                    compileOptions.SetOutputModelExternalInitializersFile(
                        Path.GetFileNameWithoutExtension(epContextPath) + ".ext_init",
                        64);
                }

                compileOptions.CompileModel();
            }

            stopwatch.Stop();

            // CompileModel() can succeed yet emit a plain reserialized graph (no com.microsoft EPContext
            // nodes) when the EP performs no AOT capture. Loading such an artifact rebuilds engines just
            // like the source and adds parse overhead, so refuse to keep it — measured: +146KB of extra
            // Conv/Squeeze nodes and a ~6s cold-load regression versus the source graph.
            if (!TryContainsEpContextNodes(tempPath))
            {
                return new CompileResult(
                    false,
                    null,
                    stopwatch.Elapsed.TotalMilliseconds,
                    "CompileModel() produced no EP-context nodes (EPContext / com.microsoft "
                    + $"markers absent) under provider '{selectedLabel}'. The output would not skip engine "
                    + "rebuild on load, so it was discarded.",
                    selectedLabel);
            }

            if (!EpContextCompatibility.AllowsTensorRtRtxArtifact(tempPath))
            {
                return new CompileResult(
                    false,
                    null,
                    stopwatch.Elapsed.TotalMilliseconds,
                    "Compiled artifact is not EP_SUPPORTED_OPTIMAL for the selected TensorRT-RTX devices "
                    + $"(missing compatibility metadata, a non-optimal status, or an empty device group) under provider '{selectedLabel}'. "
                    + "It was not published.",
                    selectedLabel);
            }

            string compileOptionsIdentity = EpContextLoadPathResolver.BuildCompileOptionsIdentity(
                sourceModelPath,
                OnnxExecutionSessionFactory.TryGetAppendedTensorRtRtxOptions(sessionOptions)
                    ?? OnnxExecutionSessionFactory.BuildTensorRtRtxOptions(modelTrtOptions));
            IReadOnlyList<EpContextArtifact.ArtifactFile> artifactFiles =
                EpContextArtifact.PublishStagedArtifact(tempDir, sourceModelPath, epContextPath,
                    existingArtifactUsable: EpContextCompatibility.AllowsTensorRtRtxArtifact,
                    createStamp: files => EpContextArtifact.CreateStamp(sourceModelPath, new FileInfo(sourceModelPath), null,
                        EpContextLoadPathResolver.CurrentHardware.GpuArchitecture,
                        EpContextLoadPathResolver.CurrentHardware.DriverVersion, files, epContextPath,
                        compileOptionsIdentity));
            return new CompileResult(true, epContextPath, stopwatch.Elapsed.TotalMilliseconds, null, selectedLabel, artifactFiles);
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or InvalidOperationException or IOException or UnauthorizedAccessException or DllNotFoundException or EntryPointNotFoundException)
        {
            stopwatch.Stop();
            TensorRtRtxTeardownGuard.MarkPoisonedIfCudaOutOfMemory(ex);
            // Publication rolls back its own files; failures before it must preserve old artifacts.
            return new CompileResult(false, null, stopwatch.Elapsed.TotalMilliseconds, ex.Message);
        }
        finally
        {
            if (tempDir is not null)
            {
                TryDeleteDirectory(tempDir);
            }
        }
    }

    private static SessionOptions CreateCompileSessionOptions(
        IReadOnlyDictionary<string, string>? modelTrtOptions,
        out ExecutionProviderKind selectedProvider)
    {
        var options = new SessionOptions
        {
            // ORT_ENABLE_ALL re-fuses contrib ops (SkipLayerNormalization, BiasGelu) the TRT-RTX
            // parser cannot import — same reasoning as CreateBaseSessionOptions(tensorRtRtx: true).
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC,
        };
        // Match production TRT RTX options (including the model's optimization profile) so the
        // compiled engines match the inference path.
        selectedProvider = OnnxExecutionSessionFactory.AppendTensorRtRtxOrFallbackProvider(
            options, OnnxExecutionSessionFactory.ForEngineBuild(modelTrtOptions));
        return options;
    }

    /// <summary>
    /// Inspect the top-level ONNX graph's nodes without loading model weights into memory.
    /// </summary>
    internal static bool TryContainsEpContextNodes(string modelPath)
    {
        try
        {
            using FileStream stream = File.OpenRead(modelPath);
            return ContainsNode(stream, stream.Length, 7, inspectNodes: false);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or InvalidDataException or OverflowException)
        {
            // Cannot verify; treat as missing so we never keep an unproven artifact.
            return false;
        }
    }

    private static bool ContainsNode(Stream stream, long end, int childField, bool inspectNodes)
    {
        while (stream.Position < end)
        {
            ulong tag = ReadVarint(stream, end);
            int field = (int)(tag >> 3);
            int wireType = (int)(tag & 7);
            if (field == 0) throw new InvalidDataException("Invalid ONNX field tag.");
            if (wireType == 2)
            {
                long length = checked((long)ReadVarint(stream, end));
                long fieldEnd = checked(stream.Position + length);
                if (fieldEnd > end) throw new InvalidDataException("Truncated ONNX field.");
                if (field == childField && (inspectNodes ? IsEpContextNode(stream, fieldEnd) : ContainsNode(stream, fieldEnd, 1, inspectNodes: true)))
                    return true;
                stream.Position = fieldEnd;
            }
            else if (wireType == 0) ReadVarint(stream, end);
            else if (wireType is 1 or 5)
            {
                stream.Position = checked(stream.Position + (wireType == 1 ? 8 : 4));
                if (stream.Position > end) throw new InvalidDataException("Truncated ONNX field.");
            }
            else throw new InvalidDataException("Unsupported ONNX wire type.");
        }
        return false;
    }

    private static bool IsEpContextNode(Stream stream, long end)
    {
        bool opType = false;
        bool domain = false;
        while (stream.Position < end)
        {
            ulong tag = ReadVarint(stream, end);
            int field = (int)(tag >> 3);
            int wireType = (int)(tag & 7);
            if (field == 0) throw new InvalidDataException("Invalid ONNX node tag.");
            if (wireType == 2)
            {
                long length = checked((long)ReadVarint(stream, end));
                long fieldEnd = checked(stream.Position + length);
                if (fieldEnd > end) throw new InvalidDataException("Truncated ONNX node.");
                if (field == 4) opType = Matches(stream, length, "EPContext"u8);
                if (field == 7) domain = Matches(stream, length, "com.microsoft"u8);
                stream.Position = fieldEnd;
            }
            else if (wireType == 0) ReadVarint(stream, end);
            else if (wireType is 1 or 5)
            {
                stream.Position = checked(stream.Position + (wireType == 1 ? 8 : 4));
                if (stream.Position > end) throw new InvalidDataException("Truncated ONNX node.");
            }
            else throw new InvalidDataException("Unsupported ONNX wire type.");
        }
        return opType && domain;
    }

    private static bool Matches(Stream stream, long length, ReadOnlySpan<byte> expected)
    {
        if (length != expected.Length) return false;
        Span<byte> actual = stackalloc byte[expected.Length];
        stream.ReadExactly(actual);
        return actual.SequenceEqual(expected);
    }

    private static ulong ReadVarint(Stream stream, long end)
    {
        ulong value = 0;
        for (int shift = 0; shift < 70 && stream.Position < end; shift += 7)
        {
            int next = stream.ReadByte();
            if (next < 0) break;
            if (shift == 63 && next > 1) break;
            value |= (ulong)(next & 0x7f) << shift;
            if ((next & 0x80) == 0) return value;
        }
        throw new InvalidDataException("Invalid ONNX varint.");
    }

    internal static void TryDeleteDirectory(string path)
    {
        try
        {
            if (Directory.Exists(path))
            {
                Directory.Delete(path, recursive: true);
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"EpContextCompiler: failed to delete temp directory '{path}': {ex.Message}");
        }
    }
}
