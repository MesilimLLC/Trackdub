using Trackdub.Inference.Onnx.EpContext;
using Trackdub.Inference.Runtime.TensorRtRtx;

namespace Trackdub.Inference.Onnx.Tests;

public sealed class EpContextArtifactStampTests
{
    [Fact]
    public void Stamp_records_runtime_lineage_so_a_trt_rtx_runtime_bump_invalidates_the_artifact()
    {
        string directory = Path.Join(Path.GetTempPath(), $"trackdub-epc-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        string sourcePath = Path.Join(directory, "model.onnx");

        try
        {
            File.WriteAllBytes(sourcePath, [1, 2, 3, 4]);
            File.WriteAllBytes(EpContextArtifact.GetEpContextPath(sourcePath), [5, 6, 7, 8]);
            EpContextArtifact.Stamp stamp = EpContextArtifact.CreateStamp(
                sourcePath,
                new FileInfo(sourcePath),
                sourceSha256: null,
                gpuArchitecture: "Blackwell",
                driverVersion: "32.0.16.1714");
            EpContextArtifact.WriteStamp(sourcePath, stamp);

            Assert.Equal(TensorRtRtxProviderConstants.BundledFingerprintVersion, stamp.TrtRtxEpVersion);
            Assert.Equal(
                EpContextArtifact.GetEpContextPath(sourcePath),
                EpContextArtifact.TryResolveValidLoadPath(sourcePath, stamp.EnvironmentFingerprint));

            // Same EP ABI plugin version, different vendored TensorRT-RTX runtime → stale artifact.
            string otherRuntime =
                $"Blackwell|32.0.16.1714|{TensorRtRtxProviderConstants.BundledVersion}+trt-rtx-1.5.0|{EpContextArtifact.HostOrtRuntimeVersion}";
            Assert.Null(EpContextArtifact.TryResolveValidLoadPath(sourcePath, otherRuntime));

            // Pre-upgrade stamps carried only the EP ABI version and must not match either.
            string legacy = $"Blackwell|32.0.16.1714|{TensorRtRtxProviderConstants.BundledVersion}|{EpContextArtifact.HostOrtRuntimeVersion}";
            Assert.Null(EpContextArtifact.TryResolveValidLoadPath(sourcePath, legacy));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Same_size_and_timestamp_replacement_invalidates_the_artifact()
    {
        string directory = Path.Join(Path.GetTempPath(), $"trackdub-epc-hash-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        string sourcePath = Path.Join(directory, "model.onnx");
        try
        {
            File.WriteAllBytes(sourcePath, [1, 2, 3, 4]);
            File.WriteAllBytes(EpContextArtifact.GetEpContextPath(sourcePath), [5, 6, 7, 8]);
            EpContextArtifact.Stamp stamp = EpContextArtifact.CreateStamp(
                sourcePath,
                new FileInfo(sourcePath),
                sourceSha256: null,
                gpuArchitecture: "Ada",
                driverVersion: "560.35.03");
            EpContextArtifact.WriteStamp(sourcePath, stamp);
            Assert.False(string.IsNullOrWhiteSpace(stamp.SourceSha256));
            Assert.Equal(
                EpContextArtifact.GetEpContextPath(sourcePath),
                EpContextArtifact.TryResolveValidLoadPath(sourcePath, stamp.EnvironmentFingerprint));

            DateTime written = File.GetLastWriteTimeUtc(sourcePath);
            File.WriteAllBytes(sourcePath, [9, 8, 7, 6]);
            File.SetLastWriteTimeUtc(sourcePath, written);
            // Within one process the memoized hash still matches; a fresh process re-hashes.
            EpContextArtifact.ClearHashCacheForTesting();

            Assert.Equal(4, new FileInfo(sourcePath).Length);
            Assert.Null(EpContextArtifact.TryResolveValidLoadPath(sourcePath, stamp.EnvironmentFingerprint));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Unknown_hardware_does_not_match_another_unknown_machine()
    {
        string directory = Path.Join(Path.GetTempPath(), $"trackdub-epc-unknown-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        string sourcePath = Path.Join(directory, "model.onnx");
        try
        {
            File.WriteAllBytes(sourcePath, [1, 2, 3, 4]);
            File.WriteAllBytes(EpContextArtifact.GetEpContextPath(sourcePath), [5]);
            EpContextArtifact.Stamp stamp = EpContextArtifact.CreateStamp(
                sourcePath,
                new FileInfo(sourcePath),
                sourceSha256: null,
                gpuArchitecture: "unknown",
                driverVersion: null);
            Assert.False(stamp.HasIdentifiedHardware);
            // Writing such a stamp is refused: it would persist a bogus machine identity.
            Assert.Throws<InvalidOperationException>(() => EpContextArtifact.WriteStamp(sourcePath, stamp));
            Assert.False(File.Exists(EpContextArtifact.GetStampPath(sourcePath)));

            // A pre-existing unknown stamp already on disk (written before the refusal)
            // still never resolves, even against its own fingerprint.
            string json = System.Text.Json.JsonSerializer.Serialize(stamp);
            File.WriteAllText(EpContextArtifact.GetStampPath(sourcePath), json);
            Assert.Null(EpContextArtifact.TryResolveValidLoadPath(sourcePath, stamp.EnvironmentFingerprint));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Theory]
    [InlineData("0")]
    [InlineData("1")]
    public void Compile_option_identity_ignores_the_cuda_graph_capture_mode(string captureMode)
    {
        var baseOptions = new Dictionary<string, string> { ["nv_profile_min_shapes"] = "x:1" };
        var withCapture = new Dictionary<string, string>(baseOptions) { ["enable_cuda_graph"] = captureMode };

        Assert.Equal(
            EpContextArtifact.BuildCompileOptionsIdentity(baseOptions, embedEpContext: true),
            EpContextArtifact.BuildCompileOptionsIdentity(withCapture, embedEpContext: true));
    }

    [Fact]
    public void Compile_option_identity_mismatch_is_a_cache_miss()
    {
        string directory = Path.Join(Path.GetTempPath(), $"trackdub-epc-options-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        string sourcePath = Path.Join(directory, "model.onnx");
        try
        {
            File.WriteAllBytes(sourcePath, [1, 2, 3, 4]);
            File.WriteAllBytes(EpContextArtifact.GetEpContextPath(sourcePath), [5]);
            string identity = EpContextArtifact.BuildCompileOptionsIdentity(
                new Dictionary<string, string> { ["nv_profile_min_shapes"] = "x:1" },
                embedEpContext: true);
            EpContextArtifact.Stamp stamp = EpContextArtifact.CreateStamp(
                sourcePath,
                new FileInfo(sourcePath),
                sourceSha256: null,
                gpuArchitecture: "Ada",
                driverVersion: "560.35.03",
                compileOptionsIdentity: identity);
            EpContextArtifact.WriteStamp(sourcePath, stamp);

            Assert.Equal(
                EpContextArtifact.GetEpContextPath(sourcePath),
                EpContextArtifact.TryResolveValidLoadPath(sourcePath, stamp.EnvironmentFingerprint, identity));
            Assert.Null(EpContextArtifact.TryResolveValidLoadPath(
                sourcePath,
                stamp.EnvironmentFingerprint,
                identity + "\nnv_profile_max_shapes=x:2"));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Schema1_stamp_is_not_reused()
    {
        string directory = Path.Join(Path.GetTempPath(), $"trackdub-epc-schema-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        string sourcePath = Path.Join(directory, "model.onnx");
        try
        {
            File.WriteAllBytes(sourcePath, [1, 2, 3, 4]);
            File.WriteAllBytes(EpContextArtifact.GetEpContextPath(sourcePath), [5]);
            var source = new FileInfo(sourcePath);
            var legacy = new EpContextArtifact.Stamp(
                SchemaVersion: 1,
                SourceFileName: "model.onnx",
                SourceLengthBytes: source.Length,
                SourceLastWriteUtcTicks: source.LastWriteTimeUtc.Ticks,
                SourceSha256: EpContextArtifact.ComputeSha256(sourcePath),
                GpuArchitecture: "Ada",
                DriverVersion: "560.35.03",
                TrtRtxEpVersion: TensorRtRtxProviderConstants.BundledFingerprintVersion,
                CreatedAtUtc: DateTimeOffset.UtcNow,
                OrtRuntimeVersion: EpContextArtifact.HostOrtRuntimeVersion);
            EpContextArtifact.WriteStamp(sourcePath, legacy);

            Assert.Null(EpContextArtifact.TryResolveValidLoadPath(sourcePath, legacy.EnvironmentFingerprint));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Legacy_stamp_rejects_missing_required_artifact_external_initializers()
    {
        var legacyStamp = new EpContextArtifact.Stamp(
            SchemaVersion: 1,
            SourceFileName: "model.onnx",
            SourceLengthBytes: 4,
            SourceLastWriteUtcTicks: DateTimeOffset.UtcNow.Ticks,
            SourceSha256: null,
            GpuArchitecture: "Ada",
            DriverVersion: "560.35.03",
            TrtRtxEpVersion: null,
            CreatedAtUtc: DateTimeOffset.UtcNow);

        Assert.False(legacyStamp.MatchesArtifactExternalInitializers(null, sidecarRequired: true));
    }

    [Fact]
    public void Stamp_invalidates_when_external_model_data_is_replaced()
    {
        string directory = Path.Join(Path.GetTempPath(), $"trackdub-epc-external-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        string sourcePath = Path.Join(directory, "model.onnx");
        string externalDataPath = EpContextArtifact.GetSourceExternalDataPath(sourcePath);
        try
        {
            File.WriteAllBytes(sourcePath, [1, 2, 3, 4]);
            File.WriteAllBytes(externalDataPath, [5, 6, 7, 8]);
            File.WriteAllBytes(EpContextArtifact.GetEpContextPath(sourcePath), [9, 10, 11, 12]);

            EpContextArtifact.Stamp stamp = EpContextArtifact.CreateStamp(
                sourcePath, new FileInfo(sourcePath), sourceSha256: null,
                gpuArchitecture: "Ada", driverVersion: "560.35.03");
            EpContextArtifact.WriteStamp(sourcePath, stamp);
            Assert.Equal(
                EpContextArtifact.GetEpContextPath(sourcePath),
                EpContextArtifact.TryResolveValidLoadPath(sourcePath, stamp.EnvironmentFingerprint));

            File.WriteAllBytes(externalDataPath, [5, 6, 7, 9]);
            File.SetLastWriteTimeUtc(
                externalDataPath,
                new DateTime(stamp.ExternalDataLastWriteUtcTicks!.Value, DateTimeKind.Utc).AddDays(1));
            Assert.Null(EpContextArtifact.TryResolveValidLoadPath(sourcePath, stamp.EnvironmentFingerprint));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Stamp_invalidates_when_artifact_external_initializers_change()
    {
        string directory = Path.Join(Path.GetTempPath(), $"trackdub-epc-sidecar-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        string sourcePath = Path.Join(directory, "model.onnx");
        string epContextPath = EpContextArtifact.GetEpContextPath(sourcePath);
        string sidecarPath = EpContextArtifact.GetArtifactExternalInitializersPath(epContextPath);
        try
        {
            File.WriteAllBytes(sourcePath, [1, 2, 3, 4]);
            File.WriteAllBytes(epContextPath, [5, 6, 7, 8]);
            File.WriteAllBytes(sidecarPath, [9, 10, 11, 12]);

            EpContextArtifact.Stamp stamp = EpContextArtifact.CreateStamp(
                sourcePath, new FileInfo(sourcePath), sourceSha256: null,
                gpuArchitecture: "Ada", driverVersion: "560.35.03");
            EpContextArtifact.WriteStamp(sourcePath, stamp);
            Assert.Equal(
                epContextPath,
                EpContextArtifact.TryResolveValidLoadPath(sourcePath, stamp.EnvironmentFingerprint));

            File.WriteAllBytes(sidecarPath, [9, 10, 11, 13]);
            File.SetLastWriteTimeUtc(
                sidecarPath,
                new DateTime(stamp.ArtifactExternalInitializersLastWriteUtcTicks!.Value, DateTimeKind.Utc).AddDays(1));
            Assert.Null(EpContextArtifact.TryResolveValidLoadPath(sourcePath, stamp.EnvironmentFingerprint));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Optimum_onnx_data_sidecar_counts_as_external_data()
    {
        // Optimum exports name the weights sidecar <model>.onnx_data; replacing it must invalidate the
        // artifact, and its bytes decide whether the engine can be embedded under protobuf's 2 GB limit.
        string directory = Path.Join(Path.GetTempPath(), $"trackdub-epc-onnxdata-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        string sourcePath = Path.Join(directory, "decoder_model.onnx");
        try
        {
            File.WriteAllBytes(sourcePath, [1, 2, 3, 4]);
            File.WriteAllBytes(sourcePath + "_data", [5, 6, 7, 8, 9, 10]);

            Assert.Equal(sourcePath + "_data", EpContextArtifact.GetSourceExternalDataPath(sourcePath));
            Assert.Equal(10, EpContextArtifact.GetSourceTotalBytes(sourcePath));

            File.WriteAllBytes(EpContextArtifact.GetEpContextPath(sourcePath), [11]);
            EpContextArtifact.Stamp stamp = EpContextArtifact.CreateStamp(
                sourcePath, new FileInfo(sourcePath), sourceSha256: null,
                gpuArchitecture: "Blackwell", driverVersion: "32.0.16.1714");
            Assert.Equal(6, stamp.ExternalDataLengthBytes);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Published_artifact_requires_its_engine_files_and_replaces_the_previous_set()
    {
        string directory = Path.Join(Path.GetTempPath(), $"trackdub-epc-publish-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        string sourcePath = Path.Join(directory, "encoder_model.onnx");
        string epContextName = Path.GetFileName(EpContextArtifact.GetEpContextPath(sourcePath));
        try
        {
            File.WriteAllBytes(sourcePath, [1, 2, 3, 4]);

            IReadOnlyList<EpContextArtifact.ArtifactFile> first = Publish(directory, sourcePath, epContextName, "old.engine", [7, 7]);
            EpContextArtifact.ArtifactFile engine = Assert.Single(first);
            Assert.Equal("old.engine", engine.Name);
            Assert.Equal(2, engine.LengthBytes);
            Assert.Equal(File.GetLastWriteTimeUtc(Path.Join(directory, engine.Name)).Ticks, engine.LastWriteUtcTicks);
            string fingerprint = WriteStamp(sourcePath, first);
            Assert.Equal(
                EpContextArtifact.GetEpContextPath(sourcePath),
                EpContextArtifact.TryResolveValidLoadPath(sourcePath, fingerprint));

            // A truncated or missing engine invalidates the artifact even though the stamp and model remain.
            File.WriteAllBytes(Path.Join(directory, "old.engine"), [7]);
            Assert.Null(EpContextArtifact.TryResolveValidLoadPath(sourcePath, fingerprint));

            // Republishing drops the previous stamp and the engine files it recorded.
            IReadOnlyList<EpContextArtifact.ArtifactFile> second = Publish(directory, sourcePath, epContextName, "new.engine", [8, 8, 8]);
            Assert.False(File.Exists(Path.Join(directory, "old.engine")));
            Assert.False(File.Exists(EpContextArtifact.GetStampPath(sourcePath)));
            fingerprint = WriteStamp(sourcePath, second);
            Assert.Equal(
                EpContextArtifact.GetEpContextPath(sourcePath),
                EpContextArtifact.TryResolveValidLoadPath(sourcePath, fingerprint));

            EpContextArtifact.DeleteArtifact(sourcePath);
            Assert.False(File.Exists(Path.Join(directory, "new.engine")));
            Assert.False(File.Exists(EpContextArtifact.GetEpContextPath(sourcePath)));
            Assert.True(File.Exists(sourcePath));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }

        static IReadOnlyList<EpContextArtifact.ArtifactFile> Publish(
            string directory, string sourcePath, string epContextName, string engineName, byte[] engineBytes)
        {
            string staging = Path.Join(directory, ".epc-staging-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(staging);
            File.WriteAllBytes(Path.Join(staging, epContextName), [5, 6]);
            File.WriteAllBytes(Path.Join(staging, engineName), engineBytes);
            IReadOnlyList<EpContextArtifact.ArtifactFile> files = EpContextArtifact.PublishStagedArtifact(staging, sourcePath);
            Directory.Delete(staging, recursive: true);
            return files;
        }

        static string WriteStamp(string sourcePath, IReadOnlyList<EpContextArtifact.ArtifactFile> files)
        {
            EpContextArtifact.Stamp stamp = EpContextArtifact.CreateStamp(
                sourcePath, new FileInfo(sourcePath), sourceSha256: null,
                gpuArchitecture: "Blackwell", driverVersion: "32.0.16.1714", files);
            EpContextArtifact.WriteStamp(sourcePath, stamp);
            return stamp.EnvironmentFingerprint;
        }
    }
}
