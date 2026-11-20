using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using Trackdub.Contracts;
using Trackdub.Contracts.Licensing;
using Trackdub.Composition.Runtime;
using Trackdub.Domain;
using Trackdub.Infrastructure.Persistence.Repositories;
using Trackdub.Infrastructure.Settings;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.TestDoubles;

namespace Trackdub.Composition.Tests;

public sealed class ModelDownloadOrchestratorTests : IDisposable
{
    private readonly string tempRoot = Path.Join(Path.GetTempPath(), "Trackdub.ModelDownloadOrchestrator.Tests", Guid.NewGuid().ToString("N"));

    [Fact]
    public async Task OfflineModePreventsNetworkDownloads()
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache();
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();
        var settings = new FakeStudioSettingsService();
        await settings.SaveAsync(StudioSettings.Default with { OfflineMode = true }, TestContext.Current.CancellationToken);
        using var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths,
            settingsService: settings);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.False(result.Success);
        Assert.Contains("Offline", result.FailureReason, StringComparison.Ordinal);
        Assert.Empty(downloader.Destinations);
        Assert.Empty(await store.LoadAsync(TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task Successful_install_invalidates_sidecar_caches_for_that_model_root()
    {
        // End-to-end proof for the mel-bin case specifically: ReadMelBins caches the 80-bin
        // fallback when preprocessor_config.json is absent, so an install that adds that file must
        // flush the stale entry or the model keeps running the wrong number of mel bins. The full
        // flush set (both tokenizers and the Qwen3-ASR embed table) is pinned directly by
        // InvalidateSidecarCaches_flushes_every_cache_for_the_root.
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache(
                sha256: Sha256Hex("correct-content"));

        // Seed the cache for the root the orchestrator actually installs into: it derives
        // "<cache>/<owner>/<name>" from the model id, not the manifest's root_path.
        string installRoot = ModelDownloadPathGuard.ResolveConfiguredModelRootDirectory(
            "example/model",
            storagePaths.ModelCacheDirectory);

        var whisperMelBins = Trackdub.Inference.Onnx.Whisper.WhisperSidecarCaches.MelBins;
        whisperMelBins.GetOrAdd(installRoot, () => 80);
        Assert.True(whisperMelBins.Count > 0, "seed must populate the mel-bin cache");

        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new HashMatchingDownloader();
        using var orchestrator = new ModelDownloadOrchestrator(
            registry, store, downloader, storagePaths, engineAutoWarmQueue: new RecordingWarmQueue());

        ModelDownloadResult result = await orchestrator.DownloadAsync(
            "example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);

        // Every seeded entry for this root is gone, so the next read re-parses installed files
        // rather than serving pre-install state.
        Assert.Equal(0, whisperMelBins.GetOrAdd(installRoot, () => 0));
    }

    [Fact]
    public void InvalidateSidecarCaches_flushes_every_cache_for_the_root()
    {
        // Pins the full flush set, not just mel bins: dropping any one of these calls would leave
        // a stale entry serving pre-install state. Sentinels stand in for the tokenizer types,
        // which cannot be constructed without loading real model files.
        string root = Path.Join(Path.GetTempPath(), "trackdub-invalidate", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            // These are process-wide static caches, so assert on the delta from this test's own
            // seeds rather than on absolute counts (other tests may have populated them).
            int melBinsBefore = Trackdub.Inference.Onnx.Whisper.WhisperSidecarCaches.MelBins.Count;
            int whisperTokBefore = Trackdub.Inference.Onnx.Whisper.WhisperSidecarCaches.Tokenizers.Count;
            int asrTokBefore = Trackdub.Inference.Onnx.Qwen3Asr.Qwen3AsrSidecarCaches.Tokenizers.Count;
            int asrEmbBefore = Trackdub.Inference.Onnx.Qwen3Asr.Qwen3AsrSidecarCaches.EmbedTokens.Count;

            Trackdub.Inference.Onnx.Whisper.WhisperSidecarCaches.MelBins.GetOrAdd(root, () => 80);
            Trackdub.Inference.Onnx.Whisper.WhisperSidecarCaches.Tokenizers.GetOrAdd(root, () => null!);
            Trackdub.Inference.Onnx.Qwen3Asr.Qwen3AsrSidecarCaches.Tokenizers.GetOrAdd(root, () => null!);
            // Embed table is keyed by file path inside the root, so it needs the prefix flush.
            Trackdub.Inference.Onnx.Qwen3Asr.Qwen3AsrSidecarCaches.EmbedTokens.GetOrAdd(
                Path.Join(root, "embeddings", "embed_tokens.bin"),
                () => null!);

            Assert.Equal(melBinsBefore + 1, Trackdub.Inference.Onnx.Whisper.WhisperSidecarCaches.MelBins.Count);
            Assert.Equal(whisperTokBefore + 1, Trackdub.Inference.Onnx.Whisper.WhisperSidecarCaches.Tokenizers.Count);
            Assert.Equal(asrTokBefore + 1, Trackdub.Inference.Onnx.Qwen3Asr.Qwen3AsrSidecarCaches.Tokenizers.Count);
            Assert.Equal(asrEmbBefore + 1, Trackdub.Inference.Onnx.Qwen3Asr.Qwen3AsrSidecarCaches.EmbedTokens.Count);

            ModelDownloadOrchestrator.InvalidateSidecarCaches(root);

            Assert.Equal(melBinsBefore, Trackdub.Inference.Onnx.Whisper.WhisperSidecarCaches.MelBins.Count);
            Assert.Equal(whisperTokBefore, Trackdub.Inference.Onnx.Whisper.WhisperSidecarCaches.Tokenizers.Count);
            Assert.Equal(asrTokBefore, Trackdub.Inference.Onnx.Qwen3Asr.Qwen3AsrSidecarCaches.Tokenizers.Count);
            Assert.Equal(asrEmbBefore, Trackdub.Inference.Onnx.Qwen3Asr.Qwen3AsrSidecarCaches.EmbedTokens.Count);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task Successful_install_requests_engine_warming_for_the_installed_model()
    {
        // End-to-end proof of the wiring: a real download must reach the warm queue. An earlier
        // revision registered the queue but nothing on the install path ever called it, so
        // installing a model warmed nothing and TensorRT engines rebuilt on every launch.
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache(
                sha256: Sha256Hex("correct-content"));
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new HashMatchingDownloader();
        var queue = new RecordingWarmQueue();
        using var orchestrator = new ModelDownloadOrchestrator(
            registry, store, downloader, storagePaths, engineAutoWarmQueue: queue);

        ModelDownloadResult result = await orchestrator.DownloadAsync(
            "example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);
        Assert.Equal(["example/model"], queue.Requests);
    }

    [Fact]
    public async Task Failed_install_does_not_request_engine_warming()
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache(
                sha256: Sha256Hex("expected-different-content"));
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new HashMatchingDownloader();
        var queue = new RecordingWarmQueue();
        using var orchestrator = new ModelDownloadOrchestrator(
            registry, store, downloader, storagePaths, engineAutoWarmQueue: queue);

        ModelDownloadResult result = await orchestrator.DownloadAsync(
            "example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.False(result.Success);
        Assert.Empty(queue.Requests);
    }

    [Fact]
    public async Task DownloadAsync_rejects_path_traversal_in_download_file()
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths) = CreateRegistryWithMaliciousDownloadPath();
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();
        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.False(result.Success);
        Assert.Contains("segment", result.FailureReason, StringComparison.OrdinalIgnoreCase);
        Assert.Empty(downloader.Destinations);
    }

    [Fact]
    public async Task DownloadAsync_downloads_to_configured_cache_when_manifest_root_is_outside_cache()
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, string manifestRoot) =
            CreateRegistryWithManifestRootOutsideConfiguredCache();
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();
        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);
        string destination = Assert.Single(downloader.Destinations);
        Assert.StartsWith(
            Path.GetFullPath(storagePaths.ModelCacheDirectory),
            Path.GetFullPath(destination),
            StringComparison.OrdinalIgnoreCase);
        Assert.False(
            Path.GetFullPath(destination).StartsWith(Path.GetFullPath(manifestRoot), StringComparison.OrdinalIgnoreCase),
            $"Destination should not be under manifest root: {destination}");

        LocalModelCacheRecord record = Assert.Single(await store.LoadAsync(TestContext.Current.CancellationToken));
        Assert.Equal(Path.Join(storagePaths.ModelCacheDirectory, "example", "model"), record.RootPath);
    }

    [Fact]
    public async Task DownloadAsync_resolves_model_by_alias_and_keys_records_by_canonical_id()
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache();
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();
        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        // "example" is the declared alias of "example/model"; the download must resolve
        // via the alias and key all state/records by the canonical model id.
        ModelDownloadResult result = await orchestrator.DownloadAsync(
            "example",
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);
        Assert.Equal("example/model", result.ModelId);
        LocalModelCacheRecord record = Assert.Single(await store.LoadAsync(TestContext.Current.CancellationToken));
        Assert.Equal("example/model", record.ModelId);
    }

    [Fact]
    public async Task DownloadAsync_current_qwen3_asr_manifest_downloads_component_files()
    {
        BundledModelManifestRegistry registry = LoadBundledRegistryWithFakeDownloadHash(
            "tonythethompson/qwen3-asr-0.6b-onnx",
            "encoder.onnx");
        TrackdubStoragePaths storagePaths = new(tempRoot);
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();
        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths,
            hashVerifier: new NullModelHashVerifier());

        // Run the download; the result may be Corrupt if the bundled manifest now carries
        // real SHA256 hashes (synthetic downloader content won't match). The purpose of this
        // test is to verify that the orchestrator resolves the correct set of component files
        // from the real manifest — not to verify hash integrity with fake content.
        await orchestrator.DownloadAsync(
            "tonythethompson/qwen3-asr-0.6b-onnx",
            cancellationToken: TestContext.Current.CancellationToken);

        string modelRoot = Path.Join(
            storagePaths.ModelCacheDirectory,
            "tonythethompson",
            "qwen3-asr-0.6b-onnx");
        string[] downloadedFiles = downloader.Destinations
            .Select(destination => Path.GetRelativePath(modelRoot, destination).Replace('\\', '/'))
            .ToArray();

        Assert.DoesNotContain("model.onnx", downloadedFiles, StringComparer.OrdinalIgnoreCase);
        Assert.Equal(
            [
                "added_tokens.json",
                "config.json",
                "decoder_init.onnx",
                "decoder_step.onnx",
                "decoder_weights.data",
                "embed_tokens.bin",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "encoder.onnx"
            ],
            downloadedFiles);
        Assert.Equal(downloadedFiles.Length, downloader.HubDownloads.Count + downloader.UriDownloads.Count);
        Assert.All(
            downloader.Destinations,
            destination => Assert.StartsWith(
                Path.GetFullPath(storagePaths.ModelCacheDirectory),
                Path.GetFullPath(destination),
                StringComparison.OrdinalIgnoreCase));
    }

    [Fact]
    public async Task DownloadAsync_includes_default_entry_when_manifest_lists_sidecar_files()
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache("[ \"model.onnx.data\" ]");
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();
        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);
        Assert.Equal(2, downloader.Destinations.Count);
        Assert.Contains(
            Path.Join(storagePaths.ModelCacheDirectory, "example", "model", "model.onnx.data"),
            downloader.Destinations);
        Assert.Contains(
            Path.Join(storagePaths.ModelCacheDirectory, "example", "model", "onnx", "model.onnx"),
            downloader.Destinations);
    }

    [Fact]
    public async Task DownloadAsync_verifies_each_required_download_file_hash()
    {
        string sidecarHash = Sha256Hex("downloaded:example/model:sidecar.txt");
        string entryHash = Sha256Hex("downloaded:example/model:onnx/model.onnx");
        string hashesJson =
            $$"""
              "download_file_hashes": {
                "sidecar.txt": "{{sidecarHash}}",
                "onnx/model.onnx": "{{entryHash}}"
              },
              "hash_verification": {
                "mode": "required",
                "algorithm": "SHA-256"
              },
            """;
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache(
                downloadFilesJson: "[ \"sidecar.txt\" ]",
                downloadFileHashesJson: hashesJson,
                sha256: entryHash);
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();
        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);
        Assert.Equal(ModelCacheState.Installed, result.NewState);
        LocalModelCacheRecord record = Assert.Single(await store.LoadAsync(TestContext.Current.CancellationToken));
        Assert.Equal(entryHash, record.Sha256);
        Assert.NotEqual(sidecarHash, record.Sha256);
    }

    [Fact]
    public async Task DownloadAsync_variant_keeps_manifest_identity_sha_when_sidecar_is_last_hashed_file()
    {
        const string identitySha = "c12e31df78c74f9589b165c8d51e65171f5028b77b7fedb41900f55f7f410dc8";
        string tokenizerHash = Sha256Hex("downloaded:example/model:tokenizer.json");
        string hashesJson =
            $$"""
              "download_file_hashes": {
                "tokenizer.json": "{{tokenizerHash}}"
              },
            """;
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithKnownSha256(identitySha, hashesJson);
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();
        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync(
            "example/model",
            "q4",
            progress: null,
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);
        LocalModelCacheRecord record = Assert.Single(await store.LoadAsync(TestContext.Current.CancellationToken));
        Assert.Equal(identitySha, record.Sha256);
        Assert.NotEqual(tokenizerHash, record.Sha256);
    }

    [Fact]
    public async Task VerifyAsync_repairs_cache_record_when_sidecar_digest_was_stored_as_identity()
    {
        const string expectedSha = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        string tokenizerHash = Sha256Hex("{}");
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithKnownSha256(expectedSha);
        var store = new LocalModelCacheRecordStore(storagePaths);
        string cacheRoot = Path.Join(storagePaths.ModelCacheDirectory, "example", "model");
        string cachedBenchmarkPath = Path.Join(cacheRoot, "onnx", "model.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(cachedBenchmarkPath)!);
        await File.WriteAllTextAsync(cachedBenchmarkPath, "hello", TestContext.Current.CancellationToken);
        await File.WriteAllTextAsync(Path.Join(cacheRoot, "tokenizer.json"), "{}", TestContext.Current.CancellationToken);
        await store.SaveAsync(
            [
                new LocalModelCacheRecord(
                    "example/model",
                    cacheRoot,
                    "main",
                    tokenizerHash,
                    DateTimeOffset.UtcNow)
            ],
            TestContext.Current.CancellationToken);

        var orchestrator = new ModelDownloadOrchestrator(registry, store, new StubDownloader(), storagePaths);
        ModelVerificationResult verification = await orchestrator.VerifyAsync("example/model", TestContext.Current.CancellationToken);

        Assert.True(verification.HashMatch, verification.FailureReason);
        Assert.Equal(ModelCacheState.Corrupt, verification.PreviousState);
        Assert.Equal(ModelCacheState.Installed, verification.NewState);
        LocalModelCacheRecord record = Assert.Single(await store.LoadAsync(TestContext.Current.CancellationToken));
        Assert.Equal(expectedSha, record.Sha256);
        Assert.NotEqual(tokenizerHash, record.Sha256);
    }

    [Fact]
    public async Task VerifyAsync_refreshes_CachedAtUtc_when_identity_sha_already_matches()
    {
        const string expectedSha = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        DateTimeOffset cachedAt = new(2024, 1, 2, 3, 4, 5, TimeSpan.Zero);
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithKnownSha256(expectedSha);
        var store = new LocalModelCacheRecordStore(storagePaths);
        string cacheRoot = Path.Join(storagePaths.ModelCacheDirectory, "example", "model");
        string cachedBenchmarkPath = Path.Join(cacheRoot, "onnx", "model.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(cachedBenchmarkPath)!);
        await File.WriteAllTextAsync(cachedBenchmarkPath, "hello", TestContext.Current.CancellationToken);
        await File.WriteAllTextAsync(Path.Join(cacheRoot, "tokenizer.json"), "{}", TestContext.Current.CancellationToken);
        await store.SaveAsync(
            [
                new LocalModelCacheRecord(
                    "example/model",
                    cacheRoot,
                    "main",
                    expectedSha,
                    cachedAt)
            ],
            TestContext.Current.CancellationToken);

        var orchestrator = new ModelDownloadOrchestrator(registry, store, new StubDownloader(), storagePaths);
        ModelVerificationResult verification = await orchestrator.VerifyAsync("example/model", TestContext.Current.CancellationToken);

        Assert.True(verification.HashMatch, verification.FailureReason);
        LocalModelCacheRecord record = Assert.Single(await store.LoadAsync(TestContext.Current.CancellationToken));
        Assert.Equal(expectedSha, record.Sha256);
        Assert.True(record.CachedAtUtc > cachedAt);
        Assert.False(record.IntegrityFailed);
    }

    [Fact]
    public async Task DownloadAsync_marks_corrupt_when_sidecar_hash_fails()
    {
        string entryHash = Sha256Hex("downloaded:example/model:onnx/model.onnx");
        string badSidecarHash = new('0', 64);
        string hashesJson =
            $$"""
              "download_file_hashes": {
                "sidecar.txt": "{{badSidecarHash}}",
                "onnx/model.onnx": "{{entryHash}}"
              },
              "hash_verification": {
                "mode": "required",
                "algorithm": "SHA-256"
              },
            """;
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache(
                downloadFilesJson: "[ \"sidecar.txt\" ]",
                downloadFileHashesJson: hashesJson);
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();
        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.False(result.Success);
        Assert.Equal(ModelCacheState.Corrupt, result.NewState);
        Assert.Contains("sidecar.txt", result.FailureReason, StringComparison.Ordinal);
    }

    [Fact]
    public async Task DownloadAsync_uses_explicit_sources_for_entry_file_and_sidecars()
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache(
                downloadFilesJson: "[ \"model.onnx.data\" ]",
                downloadFileSourcesJson:
                """
                  "download_file_sources": {
                    "onnx/model.onnx": "https://example.test/models/model.onnx",
                    "model.onnx.data": "https://example.test/models/model.onnx.data"
                  },
                """);
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();
        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);
        Assert.Empty(downloader.HubDownloads);
        Assert.Equal(
            [
                "https://example.test/models/model.onnx.data",
                "https://example.test/models/model.onnx"
            ],
            downloader.UriDownloads);
    }

    [Fact]
    public async Task DownloadAsync_does_not_generate_huggingface_download_for_cache_installed_entry()
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache(revision: "cache-installed");
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();
        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.False(result.Success);
        Assert.Equal(ModelCacheState.Missing, result.NewState);
        Assert.Contains("No downloadable source", result.FailureReason, StringComparison.OrdinalIgnoreCase);
        Assert.Empty(downloader.HubDownloads);
        Assert.Empty(downloader.UriDownloads);
    }

    [Fact]
    public async Task DownloadAsync_includes_default_variant_entry_and_support_files()
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache(
                downloadFilesJson: "[ \"tokenizer.json\" ]",
                variantsJson:
                """
                  "variants": [
                    {
                      "alias": "default",
                      "entry_path": "onnx/decoder_model_merged.onnx",
                      "download_files": [ "onnx/decoder_model_merged.onnx.data" ]
                    }
                  ]
                """);
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();
        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);
        Assert.Equal(
            [
                "tokenizer.json",
                "onnx/decoder_model_merged.onnx.data",
                "onnx/decoder_model_merged.onnx",
                "onnx/model.onnx"
            ],
            downloader.HubDownloads);
    }

    [Fact]
    public async Task DownloadAsync_does_not_pull_model_lab_variant_files_for_base_download()
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache(
                downloadFilesJson: "[ \"genai_config.json\" ]",
                variantsJson:
                """
                  "variants": [
                    {
                      "alias": "trt-rtx-fp16",
                      "entry_path": "trt-rtx-fp16/encoder.onnx",
                      "download_files": [
                        "trt-rtx-fp16/audio_processor_config.json",
                        "trt-rtx-fp16/genai_config.json"
                      ]
                    }
                  ]
                """);
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();
        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);
        Assert.Equal(
            ["genai_config.json", "onnx/model.onnx"],
            downloader.HubDownloads);
    }

    [Fact]
    public async Task DownloadAsync_when_installed_fetches_only_delta_files_added_to_manifest()
    {
        // P2-12: manifest grew (new voice pack) after first install; only absent files are fetched.
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache(
                downloadFilesJson: "[ \"tokenizer.json\", \"voices/af_bella.bin\" ]");
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();

        string cacheRoot = Path.Join(storagePaths.ModelCacheDirectory, "example", "model");
        string benchmarkPath = Path.Join(cacheRoot, "onnx", "model.onnx");
        string tokenizerPath = Path.Join(cacheRoot, "tokenizer.json");
        Directory.CreateDirectory(Path.GetDirectoryName(benchmarkPath)!);
        await File.WriteAllTextAsync(benchmarkPath, "existing-benchmark", TestContext.Current.CancellationToken);
        await File.WriteAllTextAsync(tokenizerPath, "existing-tokenizer", TestContext.Current.CancellationToken);

        await store.SaveAsync(
            [new LocalModelCacheRecord("example/model", cacheRoot, "main", "", DateTimeOffset.UtcNow)],
            TestContext.Current.CancellationToken);

        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);
        Assert.Equal(ModelCacheState.Installed, result.NewState);
        string deltaFile = Assert.Single(downloader.HubDownloads);
        Assert.Equal("voices/af_bella.bin", deltaFile);
    }

    [Fact]
    public async Task DownloadAsync_when_installed_and_delta_file_has_no_source_marks_corrupt_not_installed()
    {
        // P2-12 / Codex review: delta failure must produce Corrupt so readiness checks block.
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache(
                downloadFilesJson: "[ \"tokenizer.json\", \"voices/af_bella.bin\" ]",
                revision: "cache-installed"); // cache-installed → no HF source → no downloadable source
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();

        string cacheRoot = Path.Join(storagePaths.ModelCacheDirectory, "example", "model");
        string benchmarkPath = Path.Join(cacheRoot, "onnx", "model.onnx");
        string tokenizerPath = Path.Join(cacheRoot, "tokenizer.json");
        Directory.CreateDirectory(Path.GetDirectoryName(benchmarkPath)!);
        await File.WriteAllTextAsync(benchmarkPath, "existing-benchmark", TestContext.Current.CancellationToken);
        await File.WriteAllTextAsync(tokenizerPath, "existing-tokenizer", TestContext.Current.CancellationToken);

        await store.SaveAsync(
            [new LocalModelCacheRecord("example/model", cacheRoot, "main", "", DateTimeOffset.UtcNow)],
            TestContext.Current.CancellationToken);

        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.False(result.Success);
        Assert.Equal(ModelCacheState.Corrupt, result.NewState);
        Assert.Empty(downloader.HubDownloads);

        // Cache record must have IntegrityFailed=true so inventory/readiness sees Corrupt, not Installed.
        IReadOnlyList<LocalModelCacheRecord> records = await store.LoadAsync(TestContext.Current.CancellationToken);
        Assert.True(Assert.Single(records).IntegrityFailed);
    }

    [Fact]
    public async Task DownloadAsync_when_installed_and_delta_download_cancelled_marks_corrupt_not_installed()
    {
        // Codex P2 review: cancelling a delta download must not leave the model as Installed,
        // because the missing required files remain absent on disk.
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache(
                downloadFilesJson: "[ \"tokenizer.json\", \"voices/af_bella.bin\" ]");
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CancellingDownloader();

        string cacheRoot = Path.Join(storagePaths.ModelCacheDirectory, "example", "model");
        string benchmarkPath = Path.Join(cacheRoot, "onnx", "model.onnx");
        string tokenizerPath = Path.Join(cacheRoot, "tokenizer.json");
        Directory.CreateDirectory(Path.GetDirectoryName(benchmarkPath)!);
        await File.WriteAllTextAsync(benchmarkPath, "existing-benchmark", TestContext.Current.CancellationToken);
        await File.WriteAllTextAsync(tokenizerPath, "existing-tokenizer", TestContext.Current.CancellationToken);

        await store.SaveAsync(
            [new LocalModelCacheRecord("example/model", cacheRoot, "main", "", DateTimeOffset.UtcNow)],
            TestContext.Current.CancellationToken);

        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.False(result.Success);
        Assert.True(result.Cancelled);
        Assert.Equal(ModelCacheState.Corrupt, result.NewState);

        IReadOnlyList<LocalModelCacheRecord> records = await store.LoadAsync(TestContext.Current.CancellationToken);
        Assert.True(Assert.Single(records).IntegrityFailed,
            "Cache record must be marked IntegrityFailed so inventory sees Corrupt, not Installed, after a cancelled delta.");
    }

    [Fact]
    public async Task DownloadAsync_when_missing_and_cancelled_returns_cancelled_result_without_throwing()
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache();
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CancellingDownloader();
        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync(
            "example/model",
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.False(result.Success);
        Assert.True(result.Cancelled);
        Assert.Equal(ModelCacheState.Missing, result.NewState);
        Assert.Equal("Download cancelled.", result.FailureReason);
    }

    [Fact]
    public async Task DownloadAsync_when_installed_with_all_files_present_skips_download()
    {
        // P2-12: all manifest files already on disk → immediate Installed, no network calls.
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache();
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();

        string cacheRoot = Path.Join(storagePaths.ModelCacheDirectory, "example", "model");
        string benchmarkPath = Path.Join(cacheRoot, "onnx", "model.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(benchmarkPath)!);
        await File.WriteAllTextAsync(benchmarkPath, "existing", TestContext.Current.CancellationToken);

        await store.SaveAsync(
            [new LocalModelCacheRecord("example/model", cacheRoot, "main", "", DateTimeOffset.UtcNow)],
            TestContext.Current.CancellationToken);

        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);
        Assert.Equal(ModelCacheState.Installed, result.NewState);
        Assert.Empty(downloader.HubDownloads);
        Assert.Empty(downloader.UriDownloads);
    }

    [Fact]
    public async Task DownloadAsync_when_installed_but_hash_mismatch_marks_corrupt_and_redownloads()
    {
        // Package C joint test: files exist on disk but manifest carries hashes.
        // When hash verification fails, the model must be marked corrupt and fall through
        // to re-download, not return Installed.
        string correctHash = Sha256Hex("correct-content");
        string hashesJson =
            $$"""
              "download_file_hashes": {
                "onnx/model.onnx": "{{correctHash}}"
              },
            """;
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache(
                downloadFilesJson: "[ \"onnx/model.onnx\" ]",
                downloadFileHashesJson: hashesJson,
                sha256: correctHash);
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new HashMatchingDownloader();

        string cacheRoot = Path.Join(storagePaths.ModelCacheDirectory, "example", "model");
        string benchmarkPath = Path.Join(cacheRoot, "onnx", "model.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(benchmarkPath)!);
        // File exists but with wrong content (simulating corruption)
        await File.WriteAllTextAsync(benchmarkPath, "wrong-content", TestContext.Current.CancellationToken);

        await store.SaveAsync(
            [new LocalModelCacheRecord("example/model", cacheRoot, "main", correctHash, DateTimeOffset.UtcNow)],
            TestContext.Current.CancellationToken);

        // Use real hash verifier - it will compute actual SHA-256 and detect the mismatch
        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        // Must not return Installed - should fall through to download path
        Assert.True(result.Success, result.FailureReason);
        Assert.Equal(ModelCacheState.Installed, result.NewState);
        // The file should have been re-downloaded
        Assert.Single(downloader.HubDownloads);
        Assert.Equal("onnx/model.onnx", downloader.HubDownloads[0]);
        // After successful re-download, IntegrityFailed should be false again
        IReadOnlyList<LocalModelCacheRecord> recordsAfterDownload = await store.LoadAsync(TestContext.Current.CancellationToken);
        Assert.False(Assert.Single(recordsAfterDownload).IntegrityFailed);
    }

    [Fact]
    public async Task RepairAsync_preserves_optimized_variants_in_cache_index()
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache(
                downloadFilesJson: "[ \"tokenizer.json\", \"voices/af_bella.bin\" ]");
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();

        string cacheRoot = Path.Join(storagePaths.ModelCacheDirectory, "example", "model");
        string benchmarkPath = Path.Join(cacheRoot, "onnx", "model.onnx");
        string tokenizerPath = Path.Join(cacheRoot, "tokenizer.json");
        string variantRoot = Path.Join(cacheRoot, "optimized", "dml");
        Directory.CreateDirectory(Path.GetDirectoryName(benchmarkPath)!);
        Directory.CreateDirectory(variantRoot);
        await File.WriteAllTextAsync(benchmarkPath, "existing", TestContext.Current.CancellationToken);
        await File.WriteAllTextAsync(tokenizerPath, "existing", TestContext.Current.CancellationToken);
        await File.WriteAllTextAsync(Path.Join(variantRoot, "model.onnx"), "optimized", TestContext.Current.CancellationToken);

        var existingVariant = new LocalModelVariantRecord(
            "dml-fp16",
            variantRoot,
            "model.onnx",
            ["model.onnx"],
            "olive",
            ExecutionProviderKind.DirectMl,
            "fp16",
            DateTimeOffset.UtcNow,
            "main");

        await store.SaveAsync(
            [
                new LocalModelCacheRecord(
                    "example/model",
                    cacheRoot,
                    "main",
                    "",
                    DateTimeOffset.UtcNow,
                    Variants: [existingVariant])
            ],
            TestContext.Current.CancellationToken);

        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.RepairAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);
        LocalModelCacheRecord record = Assert.Single(await store.LoadAsync(TestContext.Current.CancellationToken));
        LocalModelVariantRecord variant = Assert.Single(record.Variants);
        Assert.Equal("dml-fp16", variant.Alias);
    }

    [Fact]
    public async Task RepairAsync_when_installed_with_missing_files_downloads_only_missing_without_deleting_existing()
    {
        // P2-12: RepairAsync must not nuke existing files when only delta files are missing.
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache(
                downloadFilesJson: "[ \"tokenizer.json\", \"voices/af_bella.bin\" ]");
        var store = new LocalModelCacheRecordStore(storagePaths);
        var downloader = new CaptureDestinationDownloader();

        string cacheRoot = Path.Join(storagePaths.ModelCacheDirectory, "example", "model");
        string benchmarkPath = Path.Join(cacheRoot, "onnx", "model.onnx");
        string tokenizerPath = Path.Join(cacheRoot, "tokenizer.json");
        Directory.CreateDirectory(Path.GetDirectoryName(benchmarkPath)!);
        const string existingContent = "should-not-be-deleted";
        await File.WriteAllTextAsync(benchmarkPath, existingContent, TestContext.Current.CancellationToken);
        await File.WriteAllTextAsync(tokenizerPath, existingContent, TestContext.Current.CancellationToken);

        await store.SaveAsync(
            [new LocalModelCacheRecord("example/model", cacheRoot, "main", "", DateTimeOffset.UtcNow)],
            TestContext.Current.CancellationToken);

        var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, storagePaths);

        ModelDownloadResult result = await orchestrator.RepairAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);
        Assert.Equal(["voices/af_bella.bin"], downloader.HubDownloads);
        Assert.Equal(existingContent, await File.ReadAllTextAsync(benchmarkPath, TestContext.Current.CancellationToken));
        Assert.Equal(existingContent, await File.ReadAllTextAsync(tokenizerPath, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task Inventory_uses_registered_cache_root_when_manifest_root_is_outside_cache()
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithManifestRootOutsideConfiguredCache();
        var store = new LocalModelCacheRecordStore(storagePaths);
        string cacheRoot = Path.Join(storagePaths.ModelCacheDirectory, "example", "model");
        string cachedModelPath = Path.Join(cacheRoot, "onnx", "model.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(cachedModelPath)!);
        await File.WriteAllTextAsync(cachedModelPath, "cached", TestContext.Current.CancellationToken);
        await store.SaveAsync(
            [
                new LocalModelCacheRecord(
                    "example/model",
                    cacheRoot,
                    "main",
                    "",
                    DateTimeOffset.UtcNow)
            ],
            TestContext.Current.CancellationToken);
        var inventory = new ModelInventoryService(registry, store, storagePaths);

        ModelInventoryEntry? entry = await inventory.GetByModelIdAsync("example/model", TestContext.Current.CancellationToken);

        Assert.NotNull(entry);
        Assert.Equal(ModelCacheState.Installed, entry!.State);
        Assert.Null(entry.FailureReason);
        Assert.Equal(new FileInfo(cachedModelPath).Length, entry.FileSizeBytes);
    }

    [Fact]
    public async Task VerifyAsync_marks_integrity_failed_so_inventory_stays_corrupt_not_missing()
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithKnownSha256("0000000000000000000000000000000000000000000000000000000000000000");
        var store = new LocalModelCacheRecordStore(storagePaths);
        string cacheRoot = Path.Join(storagePaths.ModelCacheDirectory, "example", "model");
        string cachedBenchmarkPath = Path.Join(cacheRoot, "onnx", "model.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(cachedBenchmarkPath)!);
        await File.WriteAllTextAsync(cachedBenchmarkPath, "wrong-content", TestContext.Current.CancellationToken);

        await store.SaveAsync(
            [
                new LocalModelCacheRecord(
                    "example/model",
                    cacheRoot,
                    "main",
                    "0000000000000000000000000000000000000000000000000000000000000000",
                    DateTimeOffset.UtcNow,
                    IntegrityFailed: false)
            ],
            TestContext.Current.CancellationToken);

        var orchestrator = new ModelDownloadOrchestrator(registry, store, new StubDownloader(), storagePaths);
        ModelVerificationResult verification = await orchestrator.VerifyAsync("example/model", TestContext.Current.CancellationToken);

        Assert.False(verification.HashMatch);
        Assert.Equal(ModelCacheState.Corrupt, verification.NewState);

        IReadOnlyList<LocalModelCacheRecord> records = await store.LoadAsync(TestContext.Current.CancellationToken);
        LocalModelCacheRecord record = Assert.Single(records);
        Assert.True(record.IntegrityFailed);

        var inventory = new ModelInventoryService(registry, store, storagePaths);
        ModelInventoryEntry? entry = await inventory.GetByModelIdAsync("example/model", TestContext.Current.CancellationToken);
        Assert.NotNull(entry);
        Assert.Equal(ModelCacheState.Corrupt, entry!.State);
    }

    [Fact]
    public async Task VerifyAsync_accepts_bundled_manifest_root_outside_configured_cache()
    {
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, string manifestRoot) =
            CreateRegistryWithManifestRootOutsideConfiguredCache();
        string benchmarkPath = Path.Join(manifestRoot, "onnx", "model.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(benchmarkPath)!);
        await File.WriteAllTextAsync(benchmarkPath, "bundled-model", TestContext.Current.CancellationToken);

        var store = new LocalModelCacheRecordStore(storagePaths);
        await store.SaveAsync(
            [
                new LocalModelCacheRecord(
                    "example/model",
                    manifestRoot,
                    "main",
                    "",
                    DateTimeOffset.UtcNow)
            ],
            TestContext.Current.CancellationToken);

        var orchestrator = new ModelDownloadOrchestrator(registry, store, new StubDownloader(), storagePaths);
        ModelVerificationResult verification = await orchestrator.VerifyAsync("example/model", TestContext.Current.CancellationToken);

        Assert.True(verification.HashMatch);
        Assert.Equal(ModelCacheState.Installed, verification.NewState);
    }

    [Fact]
    public async Task DownloadAsync_emits_installed_then_verify_can_clear_integrity_flag()
    {
        const string expectedSha = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths, _) =
            CreateRegistryWithKnownSha256(expectedSha);
        var store = new LocalModelCacheRecordStore(storagePaths);
        string cacheRoot = Path.Join(storagePaths.ModelCacheDirectory, "example", "model");
        string cachedBenchmarkPath = Path.Join(cacheRoot, "onnx", "model.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(cachedBenchmarkPath)!);
        await File.WriteAllTextAsync(cachedBenchmarkPath, "hello", TestContext.Current.CancellationToken);
        // tokenizer.json is also a required download file in the manifest; must exist for verification.
        await File.WriteAllTextAsync(Path.Join(cacheRoot, "tokenizer.json"), "{}", TestContext.Current.CancellationToken);

        await store.SaveAsync(
            [
                new LocalModelCacheRecord(
                    "example/model",
                    cacheRoot,
                    "main",
                    expectedSha,
                    DateTimeOffset.UtcNow,
                    IntegrityFailed: true)
            ],
            TestContext.Current.CancellationToken);

        var orchestrator = new ModelDownloadOrchestrator(registry, store, new StubDownloader(), storagePaths);
        ModelVerificationResult verification = await orchestrator.VerifyAsync("example/model", TestContext.Current.CancellationToken);

        Assert.True(verification.HashMatch);
        IReadOnlyList<LocalModelCacheRecord> records = await store.LoadAsync(TestContext.Current.CancellationToken);
        Assert.False(Assert.Single(records).IntegrityFailed);
    }

    [Fact]
    public async Task VerifyAsync_does_not_mark_corrupt_when_only_optional_variant_hashes_are_pinned()
    {
        const string expectedSha = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        string fp16Hash = Sha256Hex("optional-fp16");
        (BundledModelManifestRegistry registry, TrackdubStoragePaths storagePaths) =
            CreateRegistryWithOptionalVariantHashes(expectedSha, fp16Hash);
        var store = new LocalModelCacheRecordStore(storagePaths);
        string cacheRoot = Path.Join(storagePaths.ModelCacheDirectory, "example", "model");
        string cachedBenchmarkPath = Path.Join(cacheRoot, "onnx", "model.onnx");
        Directory.CreateDirectory(Path.GetDirectoryName(cachedBenchmarkPath)!);
        await File.WriteAllTextAsync(cachedBenchmarkPath, "hello", TestContext.Current.CancellationToken);
        await File.WriteAllTextAsync(Path.Join(cacheRoot, "tokenizer.json"), "{}", TestContext.Current.CancellationToken);

        await store.SaveAsync(
            [
                new LocalModelCacheRecord(
                    "example/model",
                    cacheRoot,
                    "main",
                    expectedSha,
                    DateTimeOffset.UtcNow,
                    IntegrityFailed: true)
            ],
            TestContext.Current.CancellationToken);

        var orchestrator = new ModelDownloadOrchestrator(registry, store, new StubDownloader(), storagePaths);
        ModelVerificationResult verification = await orchestrator.VerifyAsync("example/model", TestContext.Current.CancellationToken);

        Assert.True(verification.HashMatch, verification.FailureReason);
        Assert.Equal(ModelCacheState.Installed, verification.NewState);
        IReadOnlyList<LocalModelCacheRecord> records = await store.LoadAsync(TestContext.Current.CancellationToken);
        Assert.False(Assert.Single(records).IntegrityFailed);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Download_and_repair_reuse_complete_external_installation(bool repair)
    {
        var (registry, paths, externalRoot) = CreateRegistryWithManifestRootOutsideConfiguredCache();
        Directory.CreateDirectory(Path.Join(externalRoot, "onnx"));
        await File.WriteAllTextAsync(Path.Join(externalRoot, "onnx", "model.onnx"), "hello", TestContext.Current.CancellationToken);
        var store = new LocalModelCacheRecordStore(paths);
        await store.SaveAsync([new LocalModelCacheRecord("example/model", externalRoot, "main", "", DateTimeOffset.UtcNow)],
            TestContext.Current.CancellationToken);
        var downloader = new CaptureDestinationDownloader();
        using var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, paths);

        ModelDownloadResult result = repair
            ? await orchestrator.RepairAsync("example/model", cancellationToken: TestContext.Current.CancellationToken)
            : await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);
        Assert.Empty(downloader.Destinations);
        Assert.Equal(externalRoot, Assert.Single(await store.LoadAsync(TestContext.Current.CancellationToken)).RootPath);
    }

    [Fact]
    public async Task Download_external_installation_with_missing_sidecar_creates_complete_configured_installation()
    {
        var (registry, paths, externalRoot) = CreateRegistryWithManifestRootOutsideConfiguredCache(
            "[ \"onnx/model.onnx\", \"tokenizer.json\" ]");
        Directory.CreateDirectory(Path.Join(externalRoot, "onnx"));
        await File.WriteAllTextAsync(Path.Join(externalRoot, "onnx", "model.onnx"), "external", TestContext.Current.CancellationToken);
        var store = new LocalModelCacheRecordStore(paths);
        await store.SaveAsync([new LocalModelCacheRecord("example/model", externalRoot, "main", "", DateTimeOffset.UtcNow)],
            TestContext.Current.CancellationToken);
        var downloader = new CaptureDestinationDownloader();
        using var orchestrator = new ModelDownloadOrchestrator(registry, store, downloader, paths);

        ModelDownloadResult result = await orchestrator.DownloadAsync("example/model", cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(result.Success, result.FailureReason);
        Assert.Equal(["onnx/model.onnx", "tokenizer.json"], downloader.HubDownloads);
        Assert.Equal("external", await File.ReadAllTextAsync(Path.Join(externalRoot, "onnx", "model.onnx"), TestContext.Current.CancellationToken));
        Assert.False(File.Exists(Path.Join(externalRoot, "tokenizer.json")));
        Assert.Equal(2, (await store.LoadAsync(TestContext.Current.CancellationToken)).Count);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task VerifyAsync_selects_same_usable_root_as_inventory_and_preserves_other_installations(bool externalRoot)
    {
        const string hash = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
        var (registry, paths, _) = CreateRegistryWithKnownSha256(hash);
        var store = new LocalModelCacheRecordStore(paths);
        string goodRoot = Path.Join(externalRoot ? tempRoot : paths.ModelCacheDirectory, "alternate");
        Directory.CreateDirectory(Path.Join(goodRoot, "onnx"));
        await File.WriteAllTextAsync(Path.Join(goodRoot, "onnx", "model.onnx"), "hello", TestContext.Current.CancellationToken);
        await File.WriteAllTextAsync(Path.Join(goodRoot, "tokenizer.json"), "{}", TestContext.Current.CancellationToken);
        var bad = new LocalModelCacheRecord("example/model", Path.Join(paths.ModelCacheDirectory, "old"),
            "main", hash, DateTimeOffset.UtcNow, IntegrityFailed: true);
        var good = bad with { RootPath = goodRoot, IntegrityFailed = false, CachedAtUtc = bad.CachedAtUtc.AddDays(-1) };
        await store.SaveAsync([bad, good], TestContext.Current.CancellationToken);
        var inventory = new ModelInventoryService(registry, store, paths);
        Assert.Equal(goodRoot, (await inventory.GetByModelIdAsync("example/model", TestContext.Current.CancellationToken))!.ModelRootPath);
        using var orchestrator = new ModelDownloadOrchestrator(registry, store, new StubDownloader(), paths);
        ModelVerificationResult result = await orchestrator.VerifyAsync("example/model", TestContext.Current.CancellationToken);
        Assert.True(result.HashMatch, result.FailureReason);
        IReadOnlyList<LocalModelCacheRecord> records = await store.LoadAsync(TestContext.Current.CancellationToken);
        Assert.Equal(2, records.Count);
        Assert.True(records.Single(record => record.RootPath == bad.RootPath).IntegrityFailed);
        Assert.False(records.Single(record => record.RootPath == goodRoot).IntegrityFailed);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
        catch
        {
            // best-effort cleanup
        }
    }

    private (BundledModelManifestRegistry Registry, TrackdubStoragePaths StoragePaths) CreateRegistryWithMaliciousDownloadPath()
    {
        TrackdubStoragePaths storagePaths = new(tempRoot);
        string manifestPath = Path.Join(storagePaths.ModelCacheDirectory, "_orch", "manifest.json");
        Directory.CreateDirectory(Path.GetDirectoryName(manifestPath)!);
        Directory.CreateDirectory(Path.Join(storagePaths.ModelCacheDirectory, "example-model"));
        File.WriteAllText(
            manifestPath,
            """
            {
              "models": [
                {
                  "model_id": "example/model",
                  "task": "tts",
                  "engine_family": "example-tts",
                  "capabilities": [ "tts" ],
                  "language_coverage": {
                    "target_languages": [ "en" ]
                  },
                  "tier": "balanced",
                  "license": "MIT",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_use_verified": false,
                  "source_url": "https://huggingface.co/example/model",
                  "revision": "main",
                  "sha256": "",
                  "aliases": [ "example" ],
                  "root_path": "../example-model",
                  "benchmark_entry": "onnx/model.onnx",
                  "download_files": [ "../outside/token.json" ],
                  "variants": [
                    {
                      "alias": "q4",
                      "entry_path": "onnx/model_q4.onnx",
                      "download_files": [ "onnx/model_q4.onnx_data" ]
                    }
                  ]
                }
              ]
            }
            """);

        return (ClearedManifestRegistry.Load(manifestPath), storagePaths);
    }

    private (BundledModelManifestRegistry Registry, TrackdubStoragePaths StoragePaths, string ManifestRoot)
        CreateRegistryWithManifestRootOutsideConfiguredCache(
            string downloadFilesJson = "[ \"onnx/model.onnx\" ]",
            string downloadFileSourcesJson = "",
            string downloadFileHashesJson = "",
            string revision = "main",
            string variantsJson = "\"variants\": []",
            string sha256 = "")
    {
        TrackdubStoragePaths storagePaths = new(tempRoot);
        string manifestDirectory = Path.Join(tempRoot, "manifest-outside-cache");
        string manifestPath = Path.Join(manifestDirectory, "manifest.json");
        string manifestRoot = Path.Join(tempRoot, "repo-models", "example-model");
        Directory.CreateDirectory(manifestDirectory);
        Directory.CreateDirectory(manifestRoot);
        File.WriteAllText(
            manifestPath,
            $$"""
            {
              "models": [
                {
                  "model_id": "example/model",
                  "task": "tts",
                  "engine_family": "example-tts",
                  "capabilities": [ "tts" ],
                  "language_coverage": {
                    "target_languages": [ "en" ]
                  },
                  "tier": "balanced",
                  "license": "MIT",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_use_verified": false,
                  "source_url": "https://huggingface.co/example/model",
                  "revision": "{{revision}}",
                  "sha256": "{{sha256}}",
                  "aliases": [ "example" ],
                  "root_path": "{{manifestRoot.Replace("\\", "\\\\")}}",
                  "benchmark_entry": "onnx/model.onnx",
                  "download_files": {{downloadFilesJson}},
            {{downloadFileSourcesJson}}{{downloadFileHashesJson}}      {{variantsJson}}
                }
              ]
            }
            """);

        return (ClearedManifestRegistry.Load(manifestPath), storagePaths, manifestRoot);
    }

    private static string Sha256Hex(string value) =>
        Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(value)))
            .ToLowerInvariant();

    private (BundledModelManifestRegistry Registry, TrackdubStoragePaths StoragePaths, string BenchmarkPath) CreateRegistryWithKnownSha256(
        string sha256,
        string downloadFileHashesJson = "")
    {
        TrackdubStoragePaths storagePaths = new(tempRoot);
        string manifestPath = Path.Join(storagePaths.ModelCacheDirectory, "_orch", "manifest-sha.json");
        Directory.CreateDirectory(Path.GetDirectoryName(manifestPath)!);
        Directory.CreateDirectory(Path.Join(storagePaths.ModelCacheDirectory, "example-model"));
        File.WriteAllText(
            manifestPath,
            $$"""
            {
              "models": [
                {
                  "model_id": "example/model",
                  "task": "tts",
                  "engine_family": "example-tts",
                  "capabilities": [ "tts" ],
                  "language_coverage": {
                    "target_languages": [ "en" ]
                  },
                  "tier": "balanced",
                  "license": "MIT",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_use_verified": true,
                  "source_url": "https://huggingface.co/example/model",
                  "revision": "main",
                  "sha256": "{{sha256}}",
                  "aliases": [ "example" ],
                  "root_path": "../example-model",
                  "benchmark_entry": "onnx/model.onnx",
                  "download_files": [ "tokenizer.json" ],
            {{downloadFileHashesJson}}      "variants": [
                    {
                      "alias": "q4",
                      "entry_path": "onnx/model_q4.onnx",
                      "download_files": [ "onnx/model_q4.onnx_data" ]
                    }
                  ]
                }
              ]
            }
            """);

        BundledModelManifestRegistry registry = ClearedManifestRegistry.Load(manifestPath);
        BundledModelManifestEntry entry = registry.Entries.Single(e => e.ModelId == "example/model");
        return (registry, storagePaths, entry.DefaultBenchmarkEntryPath);
    }

    private (BundledModelManifestRegistry Registry, TrackdubStoragePaths StoragePaths) CreateRegistryWithOptionalVariantHashes(
        string sha256,
        string optionalVariantHash)
    {
        TrackdubStoragePaths storagePaths = new(tempRoot);
        string manifestPath = Path.Join(storagePaths.ModelCacheDirectory, "_orch", "manifest-optional-hashes.json");
        Directory.CreateDirectory(Path.GetDirectoryName(manifestPath)!);
        Directory.CreateDirectory(Path.Join(storagePaths.ModelCacheDirectory, "example-model"));
        File.WriteAllText(
            manifestPath,
            $$"""
            {
              "models": [
                {
                  "model_id": "example/model",
                  "task": "tts",
                  "engine_family": "example-tts",
                  "capabilities": [ "tts" ],
                  "language_coverage": {
                    "target_languages": [ "en" ]
                  },
                  "tier": "balanced",
                  "license": "MIT",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_use_verified": true,
                  "source_url": "https://huggingface.co/example/model",
                  "revision": "main",
                  "sha256": "{{sha256}}",
                  "aliases": [ "example" ],
                  "root_path": "../example-model",
                  "benchmark_entry": "onnx/model.onnx",
                  "download_files": [ "tokenizer.json" ],
                  "download_file_hashes": {
                    "onnx/model_fp16.onnx": "{{optionalVariantHash}}"
                  },
                  "variants": [
                    {
                      "alias": "default",
                      "entry_path": "onnx/model.onnx",
                      "is_default": true
                    },
                    {
                      "alias": "fp16",
                      "entry_path": "onnx/model_fp16.onnx"
                    }
                  ]
                }
              ]
            }
            """);

        return (ClearedManifestRegistry.Load(manifestPath), storagePaths);
    }

    private static BundledModelManifestRegistry LoadBundledRegistry()
    {
        string repoRoot = FindRepoRoot();
        string manifestPath = Path.Join(
            repoRoot,
            "src",
            "Trackdub.Inference",
            "Runtime",
            "ModelManifest",
            "bundled-models.manifest.json");
        return ClearedManifestRegistry.Load(manifestPath);
    }

    private BundledModelManifestRegistry LoadBundledRegistryWithFakeDownloadHash(string modelId, string benchmarkEntry)
    {
        string repoRoot = FindRepoRoot();
        string sourceManifestPath = Path.Join(
            repoRoot,
            "src",
            "Trackdub.Inference",
            "Runtime",
            "ModelManifest",
            "bundled-models.manifest.json");
        string testManifestPath = Path.Join(tempRoot, "_bundled", "bundled-models.manifest.json");
        Directory.CreateDirectory(Path.GetDirectoryName(testManifestPath)!);

        JsonObject root = JsonNode.Parse(File.ReadAllText(sourceManifestPath))!.AsObject();
        JsonArray models = root["models"]!.AsArray();
        JsonObject model = models
            .Select(node => node!.AsObject())
            .Single(candidate => string.Equals(
                candidate["model_id"]!.GetValue<string>(),
                modelId,
                StringComparison.OrdinalIgnoreCase));

        // The production manifest pins the real benchmark-entry hash; this test's downloader writes fake bytes.
        model["sha256"] = Sha256Hex($"downloaded:{modelId}:{benchmarkEntry}");
        File.WriteAllText(testManifestPath, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));

        return ClearedManifestRegistry.Load(testManifestPath);
    }

    private static string FindRepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            string candidate = Path.Join(
                directory.FullName,
                "src",
                "Trackdub.Inference",
                "Runtime",
                "ModelManifest",
                "bundled-models.manifest.json");
            if (File.Exists(candidate))
                return directory.FullName;

            directory = directory.Parent;
        }

        throw new FileNotFoundException("Could not locate bundled-models.manifest.json from the test output directory.");
    }

    private sealed class CaptureDestinationDownloader : IModelDownloaderContract
    {
        public List<string> Destinations { get; } = [];
        public List<string> HubDownloads { get; } = [];
        public List<string> UriDownloads { get; } = [];

        public Task<bool> DownloadAsync(
            string modelId,
            string fileName,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default,
            string? revision = null)
        {
            HubDownloads.Add(fileName.Replace('\\', '/'));
            Destinations.Add(destinationPath);
            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);
            File.WriteAllText(destinationPath, $"downloaded:{modelId}:{fileName}");
            progress?.Report(new ModelDownloadProgress(1, 1, 100, null));
            return Task.FromResult(true);
        }

        public Task<bool> DownloadUriAsync(
            Uri sourceUri,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default)
        {
            UriDownloads.Add(sourceUri.AbsoluteUri);
            Destinations.Add(destinationPath);
            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);
            File.WriteAllText(destinationPath, $"downloaded:{sourceUri}");
            progress?.Report(new ModelDownloadProgress(1, 1, 100, null));
            return Task.FromResult(true);
        }

        public Task<bool> VerifyHashAsync(
            string filePath,
            string expectedHash,
            CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(filePath) || string.IsNullOrWhiteSpace(expectedHash) || !File.Exists(filePath))
                return Task.FromResult(false);

            cancellationToken.ThrowIfCancellationRequested();

            byte[] bytes = File.ReadAllBytes(filePath);
            byte[] hashBytes = SHA256.HashData(bytes);
            string actualHex = Convert.ToHexString(hashBytes).ToLowerInvariant();

            string normalizedExpected = expectedHash.Trim().ToLowerInvariant();
            if (normalizedExpected.StartsWith("sha256:"))
                normalizedExpected = normalizedExpected["sha256:".Length..];

            return Task.FromResult(actualHex == normalizedExpected);
        }
    }

    private sealed class CancellingDownloader : IModelDownloaderContract
    {
        public Task<bool> DownloadAsync(
            string modelId,
            string fileName,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default,
            string? revision = null) => throw new OperationCanceledException("cancelled by test");

        public Task<bool> DownloadUriAsync(
            Uri sourceUri,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default) => throw new OperationCanceledException("cancelled by test");

        public Task<bool> VerifyHashAsync(
            string filePath,
            string expectedHash,
            CancellationToken cancellationToken = default) => Task.FromResult(true);
    }

    private sealed class StubDownloader : IModelDownloaderContract
    {
        public Task<bool> DownloadAsync(
            string modelId,
            string fileName,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default,
            string? revision = null) => Task.FromResult(true);

        public Task<bool> DownloadUriAsync(
            Uri sourceUri,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default) => Task.FromResult(true);

        public Task<bool> VerifyHashAsync(
            string filePath,
            string expectedHash,
            CancellationToken cancellationToken = default) => Task.FromResult(true);
    }

    private sealed class RecordingWarmQueue : Trackdub.Application.Pipeline.IEngineAutoWarmQueue
    {
        public List<string> Requests { get; } = [];

        public bool IsActive { get; private set; }

        public bool IsHostStarted { get; private set; }

        public bool Activate()
        {
            bool changed = !IsActive;
            IsActive = true;
            return changed;
        }

        public bool MarkHostStarted()
        {
            Activate();
            bool changed = !IsHostStarted;
            IsHostStarted = true;
            return changed;
        }

        public void RequestWarmAfterInstall(string modelId, CancellationToken cancellationToken = default) =>
            Requests.Add(modelId);

        public Task DrainAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
    }

    private sealed class HashMatchingDownloader : IModelDownloaderContract
    {
        public List<string> HubDownloads { get; } = [];

        public Task<bool> DownloadAsync(
            string modelId,
            string fileName,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default,
            string? revision = null)
        {
            HubDownloads.Add(fileName.Replace('\\', '/'));
            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);
            // Write content that matches the expected hash in the test
            File.WriteAllText(destinationPath, "correct-content");
            progress?.Report(new ModelDownloadProgress(1, 1, 100, null));
            return Task.FromResult(true);
        }

        public Task<bool> DownloadUriAsync(
            Uri sourceUri,
            string destinationPath,
            IProgress<ModelDownloadProgress>? progress = null,
            CancellationToken cancellationToken = default)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);
            File.WriteAllText(destinationPath, "correct-content");
            progress?.Report(new ModelDownloadProgress(1, 1, 100, null));
            return Task.FromResult(true);
        }

        public Task<bool> VerifyHashAsync(
            string filePath,
            string expectedHash,
            CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(filePath) || string.IsNullOrWhiteSpace(expectedHash) || !File.Exists(filePath))
                return Task.FromResult(false);

            cancellationToken.ThrowIfCancellationRequested();

            byte[] bytes = File.ReadAllBytes(filePath);
            byte[] hashBytes = SHA256.HashData(bytes);
            string actualHex = Convert.ToHexString(hashBytes).ToLowerInvariant();

            string normalizedExpected = expectedHash.Trim().ToLowerInvariant();
            if (normalizedExpected.StartsWith("sha256:"))
                normalizedExpected = normalizedExpected["sha256:".Length..];

            return Task.FromResult(actualHex == normalizedExpected);
        }
    }
}
