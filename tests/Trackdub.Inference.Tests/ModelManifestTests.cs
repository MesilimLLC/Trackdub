using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Onnx.Kokoro;
using Trackdub.Infrastructure.FileSystem;
using Trackdub.Infrastructure.Persistence.Repositories;
using Trackdub.Infrastructure.Settings;
using Trackdub.Domain;

namespace Trackdub.Inference.Tests;

public sealed class ModelManifestLoaderTests
{
    private const string ValidSha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    [Fact]
    public void LoadCatalog_LoadsValidManifest()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "models": [
                {
                  "model_id": "example/model",
                  "task": "asr",
                  "engine_family": "whisper-onnx",
                  "capabilities": [ "asr", "language-detection" ],
                  "language_coverage": {
                    "source_languages": [ "auto" ]
                  },
                  "tier": "fast",
                  "license": "MIT",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_safe_mode": true,
                  "source_url": "https://example.invalid/model",
                  "revision": "main",
                  "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                  "aliases": [ "example-model" ],
                  "root_path": "../../../../models/example-model",
                  "benchmark_entry": "onnx/model.onnx",
                  "download_files": [ "tokenizer.json" ],
                  "variants": [
                    {
                      "alias": "fp16",
                      "entry_path": "onnx/model_fp16.onnx",
                      "sha256": "def456",
                      "download_files": [ "onnx/model_fp16.onnx_data" ]
                    }
                  ],
                  "hash_verification": {
                    "mode": "required",
                    "algorithm": "SHA-256"
                  }
                }
              ]
            }
            """);

        try
        {
            ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);

            ModelManifest manifest = Assert.Single(catalog.Models);
            Assert.Equal("example/model", manifest.ModelId);
            Assert.Equal(ModelTask.Asr, manifest.Task);
            Assert.Equal("whisper-onnx", manifest.EngineFamily);
            Assert.Contains("language-detection", manifest.Capabilities);
            Assert.Contains("auto", manifest.LanguageCoverage.SourceLanguages);
            Assert.Equal("fast", manifest.Tier);
            Assert.Equal(ModelLicenseKind.Mit, manifest.License);
            Assert.True(manifest.CommercialAllowed);
            Assert.Single(manifest.Aliases);
            Assert.Equal(["tokenizer.json"], manifest.DownloadFiles);
            ModelVariantManifest variant = Assert.Single(manifest.Variants);
            Assert.Equal(["onnx/model_fp16.onnx_data"], variant.DownloadFiles);
            Assert.False(variant.IsDefault);
            Assert.Equal(HashVerificationMode.Required, manifest.HashVerificationPolicy.Mode);
            Assert.Equal("SHA-256", manifest.HashVerificationPolicy.Algorithm);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_LoadsDownloadFileHashes()
    {
        const string sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        string manifestPath = WriteTempManifest(
            $$"""
            {
              "model_id": "example/model",
              "task": "tts",
              "engine_family": "cosyvoice",
              "license": "Apache-2.0",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": true,
              "requires_user_consent": true,
              "voice_cloning": true,
              "commercial_safe_mode": false,
              "source_url": "https://huggingface.co/example/model",
              "revision": "main",
              "sha256": "",
              "root_path": "../../../../models/example-model",
              "benchmark_entry": "trackdub/model/v1/flow/decoder_estimator.onnx",
              "download_files": [
                "trackdub/model/v1/manifest.json",
                "trackdub/model/v1/flow/decoder_estimator.onnx"
              ],
              "download_file_hashes": {
                "trackdub/model/v1/manifest.json": "{{sha}}",
                "trackdub/model/v1/flow/decoder_estimator.onnx": "{{sha}}"
              },
              "variants": [
                {
                  "alias": "default",
                  "entry_path": "trackdub/model/v1/flow/decoder_estimator.onnx",
                  "is_default": true
                }
              ],
              "hash_verification": {
                "mode": "required",
                "algorithm": "SHA-256"
              }
            }
            """);

        try
        {
            ModelManifest manifest = Assert.Single(ModelManifestLoader.LoadCatalog(manifestPath).Models);

            Assert.Equal(sha, manifest.DownloadFileHashes["trackdub/model/v1/manifest.json"]);
            Assert.Equal(sha, manifest.DownloadFileHashes["trackdub/model/v1/flow/decoder_estimator.onnx"]);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsRequiredDownloadFileHashMissing()
    {
        const string sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        string manifestPath = WriteTempManifest(
            $$"""
            {
              "model_id": "example/model",
              "task": "tts",
              "engine_family": "cosyvoice",
              "license": "Apache-2.0",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": true,
              "requires_user_consent": true,
              "voice_cloning": true,
              "commercial_safe_mode": false,
              "source_url": "https://huggingface.co/example/model",
              "revision": "main",
              "sha256": "",
              "root_path": "../../../../models/example-model",
              "benchmark_entry": "trackdub/model/v1/flow/decoder_estimator.onnx",
              "download_files": [
                "trackdub/model/v1/manifest.json",
                "trackdub/model/v1/flow/decoder_estimator.onnx"
              ],
              "download_file_hashes": {
                "trackdub/model/v1/manifest.json": "{{sha}}"
              },
              "variants": [
                {
                  "alias": "default",
                  "entry_path": "trackdub/model/v1/flow/decoder_estimator.onnx",
                  "is_default": true
                }
              ],
              "hash_verification": {
                "mode": "required",
                "algorithm": "SHA-256"
              }
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("download_file_hashes", exception.Message, StringComparison.Ordinal);
            Assert.Contains("flow/decoder_estimator.onnx", exception.Message, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_Loads_variant_metadata_and_default_flag()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "asr",
              "engine_family": "whisper-onnx",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": true,
              "source_url": "",
              "revision": "",
              "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "variants": [
                {
                  "alias": "default",
                  "entry_path": "onnx/model.onnx",
                  "is_default": true,
                  "display_name": "Default",
                  "description": "General-purpose profile",
                  "supported_providers": ["cpu", "directml"]
                }
              ]
            }
            """);

        try
        {
            ModelManifest manifest = Assert.Single(ModelManifestLoader.LoadCatalog(manifestPath).Models);
            ModelVariantManifest variant = Assert.Single(manifest.Variants);
            Assert.True(variant.IsDefault);
            Assert.Equal("Default", variant.DisplayName);
            Assert.Equal("General-purpose profile", variant.Description);
            Assert.Equal(["cpu", "directml"], variant.SupportedProviders);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_Rejects_multiple_default_variants()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "asr",
              "engine_family": "whisper-onnx",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": true,
              "source_url": "",
              "revision": "",
              "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "variants": [
                { "alias": "v1", "entry_path": "onnx/model.onnx", "is_default": true },
                { "alias": "v2", "entry_path": "onnx/model_v2.onnx", "is_default": true }
              ]
            }
            """);

        try
        {
            ModelManifestValidationException ex = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));
            Assert.Contains("multiple defaults", ex.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_LoadsOliveOptimizationProfile()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "separation",
              "engine_family": "example-onnx",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": false,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "aliases": [ "example-model" ],
              "root_path": "../../../../models/example-model",
              "benchmark_entry": "onnx/model.onnx",
              "optimization": {
                "olive": {
                  "mode": "existing-onnx-components",
                  "components": [ "onnx/model.onnx" ],
                  "supported_providers": [ "cpu", "dml" ]
                }
              },
              "variants": []
            }
            """);

        try
        {
            ModelManifest manifest = Assert.Single(ModelManifestLoader.LoadCatalog(manifestPath).Models);

            Assert.NotNull(manifest.Optimization?.Olive);
            ModelOliveOptimizationProfile olive = manifest.Optimization!.Olive!;
            Assert.Equal("existing-onnx-components", olive.Mode);
            Assert.Equal(["onnx/model.onnx"], olive.Components);
            Assert.Equal([OliveOptimizationProvider.Cpu, OliveOptimizationProvider.Dml], olive.SupportedProviders);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_LoadsOliveRecipeBindings()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "openai/whisper-tiny",
              "task": "asr",
              "engine_family": "whisper-genai",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": false,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "aliases": [ "whisper-tiny-genai" ],
              "root_path": "../../../../models/whisper-tiny-genai",
              "benchmark_entry": "encoder.onnx",
              "optimization": {
                "olive": {
                  "mode": "ort-genai-builder",
                  "components": [ "encoder.onnx", "decoder.onnx" ],
                  "supported_providers": [ "cpu", "dml" ],
                  "recipe_bindings": [
                    {
                      "provider": "dml",
                      "precision": "int8",
                      "config_relative_path": "openai-whisper-tiny/cpu/whisper-tiny_cpu_int8.json",
                      "operations": [ "provider_optimization" ],
                      "expected_output": "onnx_components"
                    }
                  ]
                }
              },
              "variants": []
            }
            """);

        try
        {
            ModelManifest manifest = Assert.Single(ModelManifestLoader.LoadCatalog(manifestPath).Models);
            OliveRecipeBinding binding = Assert.Single(manifest.Optimization!.Olive!.RecipeBindings);
            Assert.Equal("dml", binding.Provider);
            Assert.Equal("int8", binding.Precision);
            Assert.Equal(
                "openai-whisper-tiny/cpu/whisper-tiny_cpu_int8.json",
                binding.ConfigRelativePath);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_LoadsTensorRtRtxOliveProviderAliases()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "separation",
              "engine_family": "example-onnx",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": false,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "aliases": [ "example-model" ],
              "root_path": "../../../../models/example-model",
              "benchmark_entry": "model.onnx",
              "optimization": {
                "olive": {
                  "mode": "existing-onnx-components",
                  "components": [ "model.onnx" ],
                  "supported_providers": [ "trt-rtx", "tensorrt" ]
                }
              },
              "variants": []
            }
            """);

        try
        {
            ModelManifest manifest = Assert.Single(ModelManifestLoader.LoadCatalog(manifestPath).Models);

            Assert.Equal(
                [OliveOptimizationProvider.TensorRtRtx, OliveOptimizationProvider.TensorRt],
                manifest.Optimization!.Olive!.SupportedProviders);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsDuplicateTensorRtRtxOliveProviderAliases()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "separation",
              "engine_family": "example-onnx",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": false,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "aliases": [ "example-model" ],
              "root_path": "../../../../models/example-model",
              "benchmark_entry": "model.onnx",
              "optimization": {
                "olive": {
                  "mode": "existing-onnx-components",
                  "components": [ "model.onnx" ],
                  "supported_providers": [ "trt-rtx", "tensorrt-rtx" ]
                }
              },
              "variants": []
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("duplicate", exception.Message, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("tensorrt-rtx", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsInvalidOliveProvider()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "separation",
              "engine_family": "example-onnx",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": false,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "aliases": [ "example-model" ],
              "root_path": "../../../../models/example-model",
              "benchmark_entry": "model.onnx",
              "optimization": {
                "olive": {
                  "mode": "existing-onnx-components",
                  "components": [ "model.onnx" ],
                  "supported_providers": [ "bad-gpu" ]
                }
              },
              "variants": []
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("supported_providers", exception.Message, StringComparison.Ordinal);
            Assert.Contains("bad-gpu", exception.Message, StringComparison.Ordinal);
            Assert.Contains("trt-rtx", exception.Message, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsUnsafeOliveComponentPath()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "separation",
              "engine_family": "example-onnx",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": false,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "aliases": [ "example-model" ],
              "root_path": "../../../../models/example-model",
              "benchmark_entry": "model.onnx",
              "optimization": {
                "olive": {
                  "mode": "existing-onnx-components",
                  "components": [ "../model.onnx" ],
                  "supported_providers": [ "cpu" ]
                }
              },
              "variants": []
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("safe relative path", exception.Message, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_ParsesLegacyOliveOptimizableFlag()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "separation",
              "engine_family": "example-onnx",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": false,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "aliases": [ "example-model" ],
              "root_path": "../../../../models/example-model",
              "benchmark_entry": "model.onnx",
              "olive_optimizable": true,
              "variants": []
            }
            """);

        try
        {
            ModelManifest manifest = Assert.Single(ModelManifestLoader.LoadCatalog(manifestPath).Models);

            Assert.True(manifest.OliveOptimizable);
            Assert.Null(manifest.Optimization);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_LoadsNvidiaOpenModelLicense()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "models": [
                {
                  "model_id": "example/nvidia-open-model",
                  "task": "diarization",
                  "engine_family": "sortformer",
                  "license": "NVIDIA-Open-Model-License",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": true,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_safe_mode": true,
                  "source_url": "https://example.invalid/model",
                  "revision": "main",
                  "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                  "aliases": [ "example-sortformer" ],
                  "root_path": "../../../../models/example-sortformer",
                  "benchmark_entry": "onnx/model.onnx",
                  "variants": []
                }
              ]
            }
            """);

        try
        {
            ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);

            ModelManifest manifest = Assert.Single(catalog.Models);
            Assert.Equal(ModelLicenseKind.NvidiaOpenModelLicense, manifest.License);
            Assert.True(manifest.CommercialAllowed);
            Assert.True(manifest.CommercialSafeMode);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_LoadsOpenMdwLicense()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "models": [
                {
                  "model_id": "example/openmdw",
                  "task": "asr",
                  "engine_family": "nemotron-asr",
                  "license": "OpenMDW-1.1",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": true,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_safe_mode": true,
                  "source_url": "https://example.invalid/model",
                  "revision": "main",
                  "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                  "aliases": [ "example-nemotron" ],
                  "root_path": "../../../../models/example-nemotron",
                  "benchmark_entry": "encoder.onnx",
                  "variants": []
                }
              ]
            }
            """);

        try
        {
            ModelManifest manifest = Assert.Single(ModelManifestLoader.LoadCatalog(manifestPath).Models);

            Assert.Equal(ModelLicenseKind.OpenMdw11, manifest.License);
            Assert.True(manifest.CommercialAllowed);
            Assert.True(manifest.RedistributionAllowed);
            Assert.True(manifest.RequiresAttribution);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_LoadsOpenRailPlusPlusLicense()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "models": [
                {
                  "model_id": "example/latentsync",
                  "task": "lip-synthesis",
                  "engine_family": "latentsync-diffusion",
                  "license": "openrail++",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": true,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_safe_mode": true,
                  "source_url": "https://example.invalid/model",
                  "revision": "main",
                  "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                  "aliases": [ "example-latentsync" ],
                  "root_path": "../../../../models/example-latentsync",
                  "benchmark_entry": "unet.onnx",
                  "variants": []
                }
              ]
            }
            """);

        try
        {
            ModelManifest manifest = Assert.Single(ModelManifestLoader.LoadCatalog(manifestPath).Models);

            Assert.Equal(ModelLicenseKind.OpenRailPlusPlus, manifest.License);
            Assert.True(manifest.CommercialAllowed);
            Assert.True(manifest.RequiresAttribution);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsInvalidLicense()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "asr",
              "engine_family": "whisper-onnx",
              "license": "Proprietary",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": false,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "variants": []
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains(".license", exception.Message, StringComparison.Ordinal);
            Assert.Contains("Proprietary", exception.Message, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsVoiceCloningWithoutConsent()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "tts",
              "engine_family": "kokoro",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": true,
              "commercial_safe_mode": false,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "variants": []
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("requires_user_consent", exception.Message, StringComparison.Ordinal);
            Assert.Contains("voice_cloning", exception.Message, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_AssignsNonCommercialLane_WhenCcByNc40License()
    {
        // Non-commercial licenses infer NonCommercial lane regardless of commercial_use_verified.
        string manifestPath = WriteTempManifest(
            """
            {
              "models": [
                {
                  "model_id": "example/noncommercial-model",
                  "task": "diarization",
                  "engine_family": "sortformer",
                  "license": "CC-BY-NC-4.0",
                  "commercial_allowed": false,
                  "redistribution_allowed": true,
                  "requires_attribution": true,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_safe_mode": false,
                  "source_url": "",
                  "revision": "",
                  "sha256": "",
                  "variants": []
                }
              ]
            }
            """);

        try
        {
            ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);

            Assert.Single(catalog.Models);
            Assert.Equal(ModelLane.NonCommercial, catalog.Models[0].Lane);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsCcByNc40LicenseWithCommercialAllowed()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "models": [
                {
                  "model_id": "example/noncommercial-model",
                  "task": "diarization",
                  "engine_family": "sortformer",
                  "license": "CC-BY-NC-4.0",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": true,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_safe_mode": false,
                  "source_url": "",
                  "revision": "",
                  "sha256": "",
                  "variants": []
                }
              ]
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("commercial_allowed", exception.Message, StringComparison.Ordinal);
            Assert.Contains("non-commercial", exception.Message, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsDuplicateAliases()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "asr",
              "engine_family": "whisper-onnx",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": true,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "aliases": [ "example-model", "example-model" ],
              "variants": []
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("duplicate alias", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsDuplicateCapabilities()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "asr",
              "engine_family": "whisper-onnx",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": true,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "capabilities": [ "asr", "asr" ],
              "variants": []
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("duplicate", exception.Message, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("capabilities", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsDuplicateSourceLanguages()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "translation",
              "engine_family": "opus-mt",
              "license": "Apache-2.0",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": true,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "language_coverage": {
                "source_languages": [ "en", "en" ],
                "target_languages": [ "es" ]
              },
              "variants": []
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("duplicate", exception.Message, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("source_languages", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsDuplicateTargetLanguages()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "translation",
              "engine_family": "opus-mt",
              "license": "Apache-2.0",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": true,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "language_coverage": {
                "source_languages": [ "en" ],
                "target_languages": [ "es", "es" ]
              },
              "variants": []
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("duplicate", exception.Message, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("target_languages", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsInvalidTask()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "embedding",
              "engine_family": "embedding-engine",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": true,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "variants": []
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains(".task", exception.Message, StringComparison.Ordinal);
            Assert.Contains("embedding", exception.Message, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsMissingEngineFamily()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "asr",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": true,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "variants": []
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("engine_family", exception.Message, StringComparison.Ordinal);
            Assert.Contains("missing required field", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsInvalidHashVerificationMode()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "asr",
              "engine_family": "whisper-onnx",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": true,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "variants": [],
              "hash_verification": {
                "mode": "always",
                "algorithm": "SHA-256"
              }
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("hash_verification.mode", exception.Message, StringComparison.Ordinal);
            Assert.Contains("always", exception.Message, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsRequiredHashVerificationWithoutSha256()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "asr",
              "engine_family": "whisper-onnx",
              "license": "MIT",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": false,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "variants": [],
              "hash_verification": {
                "mode": "required",
                "algorithm": "SHA-256"
              }
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("required", exception.Message, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("sha256", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsCommercialUseVerified_WhenHashEvidenceMissing()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "models": [
                {
                  "model_id": "example/model",
                  "task": "asr",
                  "engine_family": "whisper-onnx",
                  "license": "MIT",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_use_verified": true,
                  "source_url": "",
                  "revision": "",
                  "sha256": "",
                  "variants": []
                }
              ]
            }
            """);

        try
        {
            ModelManifestValidationException exception =
                Assert.Throws<ModelManifestValidationException>(() => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("commercial_use_verified", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_AcceptsCommercialUseVerified_WhenDownloadFileHashesCoverBenchmarkEntry()
    {
        const string sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        string manifestPath = WriteTempManifest(
            $$"""
            {
              "model_id": "example/verified-bundle",
              "task": "tts",
              "engine_family": "cosyvoice",
              "license": "Apache-2.0",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": true,
              "requires_user_consent": true,
              "voice_cloning": true,
              "commercial_use_verified": true,
              "source_url": "https://huggingface.co/example/verified-bundle",
              "revision": "main",
              "sha256": "",
              "root_path": "../../../../models/example-verified-bundle",
              "benchmark_entry": "trackdub/model/v1/flow/decoder_estimator.onnx",
              "download_files": [
                "trackdub/model/v1/manifest.json",
                "trackdub/model/v1/flow/decoder_estimator.onnx"
              ],
              "download_file_hashes": {
                "trackdub/model/v1/manifest.json": "{{sha}}",
                "trackdub/model/v1/flow/decoder_estimator.onnx": "{{sha}}"
              },
              "variants": [
                {
                  "alias": "default",
                  "entry_path": "trackdub/model/v1/flow/decoder_estimator.onnx",
                  "is_default": true
                }
              ]
            }
            """);

        try
        {
            ModelManifest manifest = Assert.Single(ModelManifestLoader.LoadCatalog(manifestPath).Models);

            Assert.True(manifest.CommercialUseVerified);
            Assert.Equal(ModelLane.Commercial, manifest.Lane);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsCommercialUseVerified_WhenDownloadFileHashesMissBenchmarkEntry()
    {
        const string sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        string manifestPath = WriteTempManifest(
            $$"""
            {
              "model_id": "example/wrong-hash-path",
              "task": "tts",
              "engine_family": "cosyvoice",
              "license": "Apache-2.0",
              "commercial_allowed": true,
              "redistribution_allowed": true,
              "requires_attribution": true,
              "requires_user_consent": true,
              "voice_cloning": true,
              "commercial_use_verified": true,
              "source_url": "https://huggingface.co/example/wrong-hash-path",
              "revision": "main",
              "sha256": "",
              "root_path": "../../../../models/example-wrong-hash-path",
              "benchmark_entry": "trackdub/model/v1/flow/decoder_estimator.onnx",
              "download_files": [
                "trackdub/model/v1/manifest.json"
              ],
              "download_file_hashes": {
                "trackdub/model/v1/manifest.json": "{{sha}}"
              },
              "variants": [
                {
                  "alias": "default",
                  "entry_path": "trackdub/model/v1/flow/decoder_estimator.onnx",
                  "is_default": true
                }
              ]
            }
            """);

        try
        {
            ModelManifestValidationException exception =
                Assert.Throws<ModelManifestValidationException>(() => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("commercial_use_verified", exception.Message, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("benchmark_entry", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsDownloadFileHashes_WhenDigestIsNotSha256Hex()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "models": [
                {
                  "model_id": "example/invalid-hash",
                  "task": "asr",
                  "engine_family": "whisper-onnx",
                  "license": "MIT",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_use_verified": false,
                  "source_url": "",
                  "revision": "",
                  "sha256": "",
                  "download_file_hashes": {
                    "weights.onnx": "not-a-sha256-digest"
                  },
                  "variants": []
                }
              ]
            }
            """);

        try
        {
            ModelManifestValidationException exception =
                Assert.Throws<ModelManifestValidationException>(() => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("download_file_hashes", exception.Message, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("SHA-256", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }
    [Fact]
    public void LoadCatalog_AssignsExperimentalLane_WhenCommercialUseNotVerified()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "models": [
                {
                  "model_id": "example/unverified-model",
                  "task": "asr",
                  "engine_family": "whisper-onnx",
                  "license": "MIT",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_use_verified": false,
                  "source_url": "",
                  "revision": "",
                  "sha256": "",
                  "variants": []
                }
              ]
            }
            """);

        try
        {
            ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);

            Assert.Single(catalog.Models);
            Assert.Equal(ModelLane.Experimental, catalog.Models[0].Lane);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_AssignsNonCommercialLane_WhenCommercialAllowedIsFalse()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "models": [
                {
                  "model_id": "example/restricted-model",
                  "task": "asr",
                  "engine_family": "whisper-onnx",
                  "license": "MIT",
                  "commercial_allowed": false,
                  "redistribution_allowed": true,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "source_url": "",
                  "revision": "",
                  "sha256": "",
                  "variants": []
                }
              ]
            }
            """);

        try
        {
            ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);

            Assert.Single(catalog.Models);
            Assert.Equal(ModelLane.NonCommercial, catalog.Models[0].Lane);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_AssignsNonCommercialLane_WhenUnknownLicense()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "models": [
                {
                  "model_id": "example/unknown-license-model",
                  "task": "tts",
                  "engine_family": "kokoro",
                  "license": "Unknown",
                  "commercial_allowed": false,
                  "redistribution_allowed": false,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "source_url": "",
                  "revision": "",
                  "sha256": "",
                  "variants": []
                }
              ]
            }
            """);

        try
        {
            ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);

            Assert.Single(catalog.Models);
            Assert.Equal(ModelLane.NonCommercial, catalog.Models[0].Lane);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_AssignsCommercialLane_WhenNvidiaOpenModelLicenseIsVerified()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "models": [
                {
                  "model_id": "example/nvidia-model",
                  "task": "asr",
                  "engine_family": "whisper-onnx",
                  "license": "NVIDIA-Open-Model-License",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": true,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_use_verified": true,
                  "source_url": "",
                  "revision": "",
                  "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                  "variants": []
                }
              ]
            }
            """);

        try
        {
            ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);

            Assert.Single(catalog.Models);
            Assert.Equal(ModelLane.Commercial, catalog.Models[0].Lane);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_HonorsExplicitLaneOverride()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "models": [
                {
                  "model_id": "example/explicit-lane-model",
                  "task": "tts",
                  "engine_family": "kokoro",
                  "license": "MIT",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_use_verified": false,
                  "lane": "experimental",
                  "source_url": "",
                  "revision": "",
                  "sha256": "",
                  "variants": []
                }
              ]
            }
            """);

        try
        {
            ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);

            Assert.Single(catalog.Models);
            Assert.Equal(ModelLane.Experimental, catalog.Models[0].Lane);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsMissingRequiredField()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "model_id": "example/model",
              "task": "asr",
              "engine_family": "whisper-onnx",
              "license": "MIT",
              "redistribution_allowed": true,
              "requires_attribution": false,
              "requires_user_consent": false,
              "voice_cloning": false,
              "commercial_safe_mode": true,
              "source_url": "",
              "revision": "",
              "sha256": "",
              "variants": []
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("commercial_allowed", exception.Message, StringComparison.Ordinal);
            Assert.Contains("missing required field", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_LoadsBundledManifest()
    {
        string repoRoot = FindRepoRoot();
        string manifestPath = Path.Join(
            repoRoot,
            "src", "Trackdub.Inference", "Runtime", "ModelManifest", "bundled-models.manifest.json");

        ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);

        Assert.NotEmpty(catalog.Models);
        Assert.All(
            catalog.Models.Where(manifest => manifest.CommercialSafeMode),
            manifest =>
            {
                Assert.True(manifest.CommercialAllowed);
                Assert.True(
                    !string.IsNullOrWhiteSpace(manifest.Sha256) ||
                    manifest.EngineFamily.Equals("whisper-onnx", StringComparison.Ordinal),
                    "Commercial-safe entries must provide a sha256 unless they are legacy whisper-onnx manifests.");
            });
        Assert.Contains(catalog.Models, manifest => manifest.ModelId.Equals("onnx-community/silero-vad", StringComparison.Ordinal));
        Assert.All(
            catalog.Models.Where(manifest => manifest.ModelId.Equals("onnx-community/whisper-tiny", StringComparison.Ordinal)),
            manifest =>
            {
                Assert.Equal(ModelLicenseKind.Apache20, manifest.License);
                Assert.True(manifest.CommercialAllowed);
                Assert.True(manifest.RedistributionAllowed);
                Assert.True(manifest.CommercialSafeMode);
            });
        Assert.Contains(catalog.Models, manifest =>
            manifest.ModelId.Equals("onnx-community/whisper-tiny", StringComparison.Ordinal) &&
            manifest.Aliases.Contains("whisper-tiny", StringComparer.OrdinalIgnoreCase));
        Assert.Contains(catalog.Models, manifest =>
            manifest.ModelId.Equals("onnx-community/whisper-tiny", StringComparison.Ordinal) &&
            manifest.Aliases.Contains("whisper-tiny-onnx", StringComparer.OrdinalIgnoreCase));
        Assert.Contains(catalog.Models, manifest =>
            manifest.Task is ModelTask.Tts &&
            manifest.EngineFamily.Equals("kokoro", StringComparison.Ordinal) &&
            manifest.ModelId.Equals("onnx-community/Kokoro-82M-v1.0-ONNX", StringComparison.Ordinal) &&
            string.Equals(manifest.SourceUrl, "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX", StringComparison.Ordinal) &&
            string.Equals(manifest.Revision, "1939ad2a8e416c0acfeecc08a694d14ef25f2231", StringComparison.Ordinal) &&
            manifest.Aliases.Contains("kokoro-onnx", StringComparer.OrdinalIgnoreCase));
        Assert.Contains(catalog.Models, manifest =>
            manifest.Task is ModelTask.Separation &&
            manifest.ModelId.Equals("csukuangfj/sherpa-onnx-spleeter-2stems", StringComparison.Ordinal) &&
            manifest.EngineFamily.Equals("spleeter", StringComparison.Ordinal) &&
            manifest.Capabilities.Contains("speech-music-sfx-separation", StringComparer.OrdinalIgnoreCase) &&
            string.Equals(manifest.SourceUrl, "https://huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems", StringComparison.Ordinal) &&
            string.Equals(manifest.Revision, "3e5a4dd68259675ab0ddc38269fdae1d0c41f20c", StringComparison.Ordinal) &&
            string.Equals(manifest.BenchmarkEntry, "vocals.onnx", StringComparison.Ordinal) &&
            manifest.DownloadFiles.Contains("vocals.onnx", StringComparer.OrdinalIgnoreCase) &&
            manifest.DownloadFiles.Contains("accompaniment.onnx", StringComparer.OrdinalIgnoreCase) &&
            manifest.Tier.Equals("fast", StringComparison.Ordinal) &&
            manifest.License is ModelLicenseKind.Mit &&
            manifest.CommercialAllowed &&
            manifest.CommercialSafeMode &&
            string.Equals(manifest.Sha256, "932fff35af1ded8eedc501031bfd6dc12cb0024a3dfa8e2c4537fd0aee9eb149", StringComparison.OrdinalIgnoreCase) &&
            manifest.Aliases.Contains("spleeter", StringComparer.OrdinalIgnoreCase) &&
            !manifest.Aliases.Contains("spleeter-2stems", StringComparer.OrdinalIgnoreCase));
        Assert.DoesNotContain(catalog.Models, manifest =>
            manifest.ModelId.Equals("kcsongor/sherpa-onnx-spleeter", StringComparison.Ordinal));
        Assert.DoesNotContain(catalog.Models, manifest =>
            manifest.ModelId.Equals("tonythethompson/mrx-cocktail-fork-onnx", StringComparison.Ordinal));
        Assert.Contains(catalog.Models, manifest =>
            manifest.ModelId.Equals("ByteDance/LatentSync-1.6", StringComparison.Ordinal) &&
            manifest.Task is ModelTask.LipSynthesis &&
            manifest.License is ModelLicenseKind.OpenRailPlusPlus &&
            manifest.CommercialAllowed &&
            manifest.CommercialUseVerified &&
            manifest.Lane is ModelLane.Commercial &&
            manifest.EngineFamily.Equals("latentsync-diffusion", StringComparison.Ordinal));
        Assert.Contains(catalog.Models, manifest =>
            manifest.ModelId.Equals("InsightFace/scrfd-500m", StringComparison.Ordinal) &&
            manifest.Task is ModelTask.FaceDetection &&
            manifest.CommercialAllowed &&
            manifest.EngineFamily.Equals("scrfd", StringComparison.Ordinal) &&
            manifest.Aliases.Contains("scrfd-500m", StringComparer.OrdinalIgnoreCase));
        Assert.Contains(catalog.Models, manifest =>
            manifest.ModelId.Equals("InsightFace/2d106det", StringComparison.Ordinal) &&
            manifest.Task is ModelTask.FaceLandmarks &&
            manifest.CommercialAllowed &&
            manifest.EngineFamily.Equals("insightface-2d106", StringComparison.Ordinal) &&
            manifest.Aliases.Contains("2d106det", StringComparer.OrdinalIgnoreCase));
    }

    [Fact]
    public void LoadCatalog_WhisperAsrEntriesHaveOliveOptimizationProfile()
    {
        string repoRoot = FindRepoRoot();
        string manifestPath = Path.Join(
            repoRoot,
            "src", "Trackdub.Inference", "Runtime", "ModelManifest", "bundled-models.manifest.json");

        ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);

        Assert.All(
            catalog.Models.Where(manifest => manifest.Task is ModelTask.Asr),
            manifest =>
            {
                Assert.NotNull(manifest.Optimization);
                Assert.NotNull(manifest.Optimization!.Olive);
                ModelOliveOptimizationProfile olive = manifest.Optimization.Olive!;
                Assert.NotEmpty(olive.Mode);
                Assert.NotEmpty(olive.Components);
                Assert.Contains(OliveOptimizationProvider.Cpu, olive.SupportedProviders);
            });

        // GenAI-family whisper entries must use ort-genai-builder mode and may include TRT/TRT-RTX providers.
        Assert.All(
            catalog.Models.Where(m => m.Task is ModelTask.Asr && m.EngineFamily == "whisper-genai"),
            manifest =>
            {
                Assert.Equal("ort-genai-builder", manifest.Optimization!.Olive!.Mode);
                Assert.Contains(OliveOptimizationProvider.TensorRtRtx, manifest.Optimization.Olive.SupportedProviders);
                Assert.Contains(OliveOptimizationProvider.TensorRt, manifest.Optimization.Olive.SupportedProviders);
            });

        // Standard ONNX whisper entries use existing-onnx-components mode; TRT-RTX is hardware validated,
        // classic TensorRT is not.
        Assert.All(
            catalog.Models.Where(m => m.Task is ModelTask.Asr && m.EngineFamily == "whisper-onnx"),
            manifest =>
            {
                Assert.Equal("existing-onnx-components", manifest.Optimization!.Olive!.Mode);
                Assert.DoesNotContain(OliveOptimizationProvider.TensorRt, manifest.Optimization.Olive.SupportedProviders);
                Assert.Contains(OliveOptimizationProvider.TensorRtRtx, manifest.Optimization.Olive.SupportedProviders);
            });
    }

    [Fact]
    public void LoadCatalog_BindingsSharingProviderAndPrecisionNameDistinctComponents()
    {
        string manifestPath = Path.Join(
            FindRepoRoot(),
            "src", "Trackdub.Inference", "Runtime", "ModelManifest", "bundled-models.manifest.json");

        ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);

        // OliveRecipeResolver can only tell same provider+precision bindings apart by component;
        // without one, the first binding (usually the encoder recipe) wins for every component.
        Assert.All(
            catalog.Models.Where(manifest => manifest.Optimization?.Olive is not null),
            manifest =>
            {
                foreach (IGrouping<(string? Provider, string? Precision), OliveRecipeBinding> group in manifest.Optimization!.Olive!.RecipeBindings
                             .GroupBy(binding => (binding.Provider, binding.Precision))
                             .Where(group => group.Count() > 1))
                {
                    Assert.All(group, binding => Assert.False(
                        string.IsNullOrWhiteSpace(binding.Component),
                        $"{manifest.ModelId}: {binding.ConfigRelativePath} shares {group.Key} with another binding but names no component."));
                    Assert.Equal(group.Count(), group.Select(binding => binding.Component).Distinct(StringComparer.OrdinalIgnoreCase).Count());
                }
            });
    }

    [Fact]
    public void LoadCatalog_BundledManifestSeparationModelsUseSupportedEngineFamilies()
    {
        string repoRoot = FindRepoRoot();
        string manifestPath = Path.Join(
            repoRoot,
            "src", "Trackdub.Inference", "Runtime", "ModelManifest", "bundled-models.manifest.json");

        ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);

        string[] supportedSeparationEngineFamilies = ["spleeter"];

        Assert.All(
            catalog.Models.Where(manifest => manifest.Task is ModelTask.Separation),
            manifest => Assert.Contains(
                manifest.EngineFamily,
                supportedSeparationEngineFamilies,
                StringComparer.OrdinalIgnoreCase));

        ModelManifest overlapRescue = Assert.Single(
            catalog.Models,
            manifest => manifest.Aliases.Contains("sepformer", StringComparer.OrdinalIgnoreCase));
        Assert.Equal(ModelTask.OverlapRescue, overlapRescue.Task);
        Assert.Equal("sepformer", overlapRescue.EngineFamily, StringComparer.OrdinalIgnoreCase);
    }

    [Fact]
    public void LoadCatalog_KokoroDownloadVoiceFilesMatchKnownAvailableCatalog()
    {
        string repoRoot = FindRepoRoot();
        string manifestPath = Path.Join(
            repoRoot,
            "src", "Trackdub.Inference", "Runtime", "ModelManifest", "bundled-models.manifest.json");

        ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);
        ModelManifest manifest = Assert.Single(catalog.Models, model =>
            model.Task is ModelTask.Tts &&
            model.EngineFamily.Equals("kokoro", StringComparison.OrdinalIgnoreCase) &&
            model.Aliases.Contains("kokoro-onnx", StringComparer.OrdinalIgnoreCase));
        string[] downloadedVoiceIds = manifest.DownloadFiles
            .Where(path => path.StartsWith("voices/", StringComparison.OrdinalIgnoreCase) &&
                           path.EndsWith(".bin", StringComparison.OrdinalIgnoreCase))
            .Select(path => Path.GetFileNameWithoutExtension(path.Replace('/', Path.DirectorySeparatorChar)))
            .OrderBy(voiceId => voiceId, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        string[] knownAvailableVoiceIds = KokoroVoiceCatalog.KnownAvailable()
            .GetVoices()
            .Select(voice => voice.VoiceId)
            .OrderBy(voiceId => voiceId, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        Assert.Equal(downloadedVoiceIds, knownAvailableVoiceIds);
    }

    [Fact]
    public void LoadCatalog_MadladDownloadFilesMatchKnownOnnxExportLayout()
    {
        string repoRoot = FindRepoRoot();
        string manifestPath = Path.Join(
            repoRoot,
            "src", "Trackdub.Inference", "Runtime", "ModelManifest", "bundled-models.manifest.json");

        ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);
        ModelManifest manifest = Assert.Single(catalog.Models, model =>
            model.ModelId.Equals("google/madlad400-3b-mt", StringComparison.OrdinalIgnoreCase));

        Assert.DoesNotContain(manifest.Variants, variant =>
            variant.Alias.Equals("fp16", StringComparison.OrdinalIgnoreCase));
        ModelVariantManifest variant = Assert.Single(manifest.Variants, variant =>
            variant.Alias.Equals("quantized", StringComparison.OrdinalIgnoreCase));
        Assert.Equal("encoder_model_quantized.onnx", variant.EntryPath);
        Assert.Equal(["cpu", "dml", "cuda"], variant.SupportedProviders);
        Assert.Equal(["decoder_model_quantized.onnx"], variant.DownloadFiles);
        Assert.True(variant.IsDefault);
        // Quantized exports belong to their variant only, so int4-kv neither requires nor downloads them.
        Assert.DoesNotContain("encoder_model_quantized.onnx", manifest.DownloadFiles);
        Assert.DoesNotContain("decoder_model_quantized.onnx", manifest.DownloadFiles);
        Assert.Contains("spiece.model", manifest.DownloadFiles);
        Assert.Contains("config.json", manifest.DownloadFiles);
        Assert.Contains("encoder_model_quantized.onnx", manifest.DownloadFileSources.Keys);
        Assert.Contains("decoder_model_quantized.onnx", manifest.DownloadFileSources.Keys);
        Assert.All(
            manifest.DownloadFileSources.Where(source =>
                !source.Key.StartsWith("mixed_fp16_int4/", StringComparison.Ordinal) &&
                !source.Key.StartsWith("trt_rtx_mixed_fp16_fp32/", StringComparison.Ordinal)),
            source => Assert.StartsWith(
                "https://huggingface.co/tonythethompson/madlad400-3b-mt-onnx/resolve/67037ad42f58d6c0fc3dafaa45f3ec97a46e7eb9/",
                source.Value,
                StringComparison.Ordinal));

        ModelVariantManifest kvVariant = Assert.Single(manifest.Variants, variant =>
            variant.Alias.Equals("int4-kv", StringComparison.OrdinalIgnoreCase));
        Assert.Equal("mixed_fp16_int4/encoder_model.onnx", kvVariant.EntryPath);
        Assert.Contains("mixed_fp16_int4/decoder_model_merged.onnx", kvVariant.DownloadFiles);
        Assert.Contains("mixed_fp16_int4/decoder_model_merged.onnx.data", kvVariant.DownloadFiles);
        Assert.Contains("mixed_fp16_int4/encoder_model.onnx.data", kvVariant.DownloadFiles);
        Assert.All(kvVariant.DownloadFiles, file =>
        {
            Assert.StartsWith(
                "https://huggingface.co/tonythethompson/madlad400-3b-mt-onnx/resolve/c165d741425c82365761aae98d0891bd9e7995ed/",
                manifest.DownloadFileSources[file],
                StringComparison.Ordinal);
            Assert.True(manifest.DownloadFileHashes.ContainsKey(file), $"Missing pinned hash for {file}.");
        });

        // TensorRT RTX decodes with static shapes over the plain decoder, so the variant ships the
        // encoder and decoder_model only (no decoder_with_past). It is limited to the two TensorRT
        // providers; classic TensorRT must be listed or the planner never offers the variant there.
        ModelVariantManifest trtVariant = Assert.Single(manifest.Variants, variant =>
            variant.Alias.Equals("trt-fp16", StringComparison.OrdinalIgnoreCase));
        Assert.Equal("trt_rtx_mixed_fp16_fp32/encoder_model.onnx", trtVariant.EntryPath);
        Assert.Equal(["trt-rtx", "tensorrt"], trtVariant.SupportedProviders);
        Assert.Contains("trt_rtx_mixed_fp16_fp32/encoder_model.onnx_data", trtVariant.DownloadFiles);
        Assert.Contains("trt_rtx_mixed_fp16_fp32/decoder_model.onnx", trtVariant.DownloadFiles);
        Assert.Contains("trt_rtx_mixed_fp16_fp32/decoder_model.onnx_data", trtVariant.DownloadFiles);
        Assert.DoesNotContain(trtVariant.DownloadFiles, file => file.Contains("with_past", StringComparison.Ordinal));
        Assert.All(trtVariant.DownloadFiles, file =>
        {
            Assert.StartsWith(
                "https://huggingface.co/tonythethompson/madlad400-3b-mt-onnx/resolve/c165d741425c82365761aae98d0891bd9e7995ed/",
                manifest.DownloadFileSources[file],
                StringComparison.Ordinal);
            Assert.True(manifest.DownloadFileHashes.ContainsKey(file), $"Missing pinned hash for {file}.");
        });
    }

    [Theory]
    [InlineData("tonythethompson/qwen3-asr-0.6b-onnx")]
    [InlineData("tonythethompson/qwen3-asr-1.7b-onnx")]
    public void LoadCatalog_QwenAsrDownloadFilesMatchKnownOnnxExportLayout(string modelId)
    {
        string repoRoot = FindRepoRoot();
        string manifestPath = Path.Join(
            repoRoot,
            "src", "Trackdub.Inference", "Runtime", "ModelManifest", "bundled-models.manifest.json");

        ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);
        ModelManifest manifest = Assert.Single(catalog.Models, model =>
            model.ModelId.Equals(modelId, StringComparison.OrdinalIgnoreCase));

        Assert.Equal("encoder.onnx", manifest.BenchmarkEntry);
        Assert.DoesNotContain("model.onnx", manifest.DownloadFiles, StringComparer.OrdinalIgnoreCase);
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
                "vocab.json"
            ],
            manifest.DownloadFiles);
        Assert.NotNull(manifest.Optimization?.Olive);
        Assert.Equal(
            ["encoder.onnx", "decoder_init.onnx", "decoder_step.onnx"],
            manifest.Optimization!.Olive!.Components);
        Assert.Contains(OliveOptimizationProvider.TensorRtRtx, manifest.Optimization.Olive.SupportedProviders);
        Assert.Equal(
            ModelExpectedRuntime.TrtRtxOrWindowsMlCatalogOrMigraphxOrDirectMlOrCpu,
            manifest.ExpectedRuntime);
    }

    [Fact]
    public void LoadCatalog_QwenTextRefinementRequiresGenAiBundle()
    {
        string repoRoot = FindRepoRoot();
        string manifestPath = Path.Join(
            repoRoot,
            "src", "Trackdub.Inference", "Runtime", "ModelManifest", "bundled-models.manifest.json");

        ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);
        ModelManifest manifest = Assert.Single(catalog.Models, model =>
            model.ModelId.Equals("tonythethompson/Qwen2.5-1.5B-Instruct", StringComparison.OrdinalIgnoreCase));

        Assert.Equal(ModelTask.TextRefinement, manifest.Task);
        Assert.Equal("qwen-instruct", manifest.EngineFamily);
        Assert.True(manifest.CommercialUseVerified);
        Assert.Equal("genai_config.json", manifest.BenchmarkEntry);
        Assert.Contains("genai_config.json", manifest.DownloadFiles);
        Assert.Equal(HashVerificationMode.Required, manifest.HashVerificationPolicy.Mode);
        Assert.Equal(
            "b1fabffd833cfdd244a06ea3db3ddfc5eaaaafa360ec6d8c5704f1a97d0b8a0f",
            manifest.DownloadFileHashes["genai_config.json"]);
        Assert.Contains(
            "f56ee6525bf4377fd6c6dcf6b17de010c5f51d26",
            manifest.DownloadFileSources["model.onnx.data"],
            StringComparison.Ordinal);
        Assert.NotNull(manifest.Optimization?.Olive);
        Assert.Equal("ort-genai-builder", manifest.Optimization!.Olive!.Mode);
        Assert.Equal(["genai_config.json"], manifest.Optimization.Olive.Components);
        Assert.Contains(manifest.Optimization.Olive.RecipeBindings, binding =>
            binding.Provider == "dml" &&
            binding.ConfigRelativePath.Contains("qwen2_5_dml_config.json", StringComparison.Ordinal));
        Assert.Equal(2, manifest.Variants.Count);
        ModelVariantManifest defaultVariant = manifest.Variants.Single(
            variant => variant.Alias.Equals("default", StringComparison.OrdinalIgnoreCase));
        Assert.True(defaultVariant.IsDefault);
        Assert.Equal("genai_config.json", defaultVariant.EntryPath);
        Assert.Contains("genai_config.json", defaultVariant.DownloadFiles);

        ModelVariantManifest mxfp8Variant = manifest.Variants.Single(
            variant => variant.Alias.Equals("mxfp8", StringComparison.OrdinalIgnoreCase));
        Assert.Equal("mxfp8/genai_config.json", mxfp8Variant.EntryPath);
        Assert.Equal(["trt-rtx"], mxfp8Variant.SupportedProviders);
    }

    [Fact]
    public void LoadCatalog_Phi35MiniDefaultVariantIncludesCpuInt4Package()
    {
        string repoRoot = FindRepoRoot();
        string manifestPath = Path.Join(
            repoRoot,
            "src", "Trackdub.Inference", "Runtime", "ModelManifest", "bundled-models.manifest.json");

        ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);
        ModelManifest manifest = Assert.Single(catalog.Models, model =>
            model.ModelId.Equals("microsoft/Phi-3.5-mini-instruct-onnx", StringComparison.OrdinalIgnoreCase));

        ModelVariantManifest variant = Assert.Single(manifest.Variants);
        Assert.True(variant.IsDefault);
        Assert.Equal(
            "cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/genai_config.json",
            variant.EntryPath);
        Assert.Contains(
            "cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/phi-3.5-mini-instruct-cpu-int4-awq-block-128-acc-level-4.onnx",
            variant.DownloadFiles);
    }

    [Fact]
    public void LoadCatalog_ParakeetTdtEntryMatchesPinnedOnnxBundle()
    {
        string repoRoot = FindRepoRoot();
        string manifestPath = Path.Join(
            repoRoot,
            "src", "Trackdub.Inference", "Runtime", "ModelManifest", "bundled-models.manifest.json");

        ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);
        ModelManifest manifest = Assert.Single(catalog.Models, model =>
            model.ModelId.Equals("tonythethompson/parakeet-tdt-0.6b-v3-onnx", StringComparison.OrdinalIgnoreCase));

        Assert.Equal(ModelTask.Asr, manifest.Task);
        Assert.Equal("parakeet-tdt", manifest.EngineFamily);
        Assert.Equal(ModelLicenseKind.CcBy40, manifest.License);
        Assert.True(manifest.CommercialAllowed);
        Assert.True(manifest.CommercialUseVerified);
        Assert.True(manifest.RequiresAttribution);
        Assert.Equal("107353b64181b47e250dfa846cf3de6a0b084c73", manifest.Revision);
        Assert.Equal(HashVerificationMode.Required, manifest.HashVerificationPolicy.Mode);
        Assert.Equal("encoder-model.onnx", manifest.BenchmarkEntry);
        Assert.Contains("parakeet-tdt-0.6b-v3", manifest.Aliases, StringComparer.OrdinalIgnoreCase);
        foreach (string required in new[] { "nemo128.onnx", "encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx", "vocab.txt" })
        {
            Assert.Contains(required, manifest.DownloadFiles);
        }

        Assert.All(manifest.DownloadFiles, file =>
        {
            Assert.True(manifest.DownloadFileHashes.ContainsKey(file), $"Missing hash for '{file}'.");
            Assert.StartsWith(
                "https://huggingface.co/tonythethompson/parakeet-tdt-0.6b-v3-onnx/resolve/107353b64181b47e250dfa846cf3de6a0b084c73/",
                manifest.DownloadFileSources[file],
                StringComparison.Ordinal);
        });
    }

    [Fact]
    public void LoadCatalog_NemotronAsrEntryMatchesPinnedOnnxBundle()
    {
        string repoRoot = FindRepoRoot();
        string manifestPath = Path.Join(
            repoRoot,
            "src", "Trackdub.Inference", "Runtime", "ModelManifest", "bundled-models.manifest.json");

        ModelManifestCatalog catalog = ModelManifestLoader.LoadCatalog(manifestPath);
        ModelManifest manifest = Assert.Single(catalog.Models, model =>
            model.ModelId.Equals("tonythethompson/nemotron-3.5-asr-streaming-0.6b-onnx", StringComparison.OrdinalIgnoreCase));

        Assert.Equal(ModelTask.Asr, manifest.Task);
        Assert.Equal("nemotron-asr", manifest.EngineFamily);
        Assert.Equal(ModelLicenseKind.OpenMdw11, manifest.License);
        Assert.True(manifest.CommercialAllowed);
        Assert.True(manifest.CommercialUseVerified);
        Assert.True(manifest.RedistributionAllowed);
        Assert.True(manifest.RequiresAttribution);
        Assert.False(manifest.RequiresUserConsent);
        Assert.Equal("b3ea33d792e4edd1ea9ffe222d250bc3239ee4ae", manifest.Revision);
        Assert.Equal(HashVerificationMode.Required, manifest.HashVerificationPolicy.Mode);
        Assert.Equal("encoder.onnx", manifest.BenchmarkEntry);
        Assert.Contains("nemotron-3.5-asr", manifest.Aliases, StringComparer.OrdinalIgnoreCase);
        Assert.Equal(
            [
                "README.md",
                "NOTICE.md",
                "LICENSE.OpenMDW-1.1",
                "config.json",
                "encoder.onnx",
                "encoder.onnx.data",
                "decoder_joint.onnx",
                "tokenizer.model"
            ],
            manifest.DownloadFiles);
        Assert.All(manifest.DownloadFiles, file =>
        {
            Assert.True(manifest.DownloadFileHashes.ContainsKey(file), $"Missing hash for '{file}'.");
            Assert.StartsWith(
                "https://huggingface.co/tonythethompson/nemotron-3.5-asr-streaming-0.6b-onnx/resolve/b3ea33d792e4edd1ea9ffe222d250bc3239ee4ae/",
                manifest.DownloadFileSources[file],
                StringComparison.Ordinal);
        });
        Assert.Equal(
            ["encoder.onnx", "decoder_joint.onnx"],
            manifest.Optimization!.Olive!.Components);
        Assert.Contains(OliveOptimizationProvider.TensorRtRtx, manifest.Optimization.Olive.SupportedProviders);
        Assert.Contains(
            manifest.Optimization.Olive.RecipeBindings,
            binding => binding.Provider == "trt-rtx" &&
                       binding.ConfigRelativePath.Contains(
                           "nemotron-3.5-asr-streaming-0.6b-onnx/NvTensorRtRtx/encoder_trtrtx_fp16.json",
                           StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("encoder.onnx", true)]
    [InlineData("encoder_typo.onnx", false)]
    public void LoadCatalog_RecipeBindingComponentMustBeADeclaredComponent(string component, bool valid)
    {
        string manifestPath = WriteTempManifest(
            $$"""
            {
              "models": [
                {
                  "model_id": "example/two-component",
                  "task": "asr",
                  "engine_family": "whisper-onnx",
                  "capabilities": [ "asr" ],
                  "license": "MIT",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_safe_mode": true,
                  "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                  "root_path": "../../../../models/example-two-component",
                  "download_files": [ "encoder.onnx", "decoder.onnx" ],
                  "variants": [],
                  "hash_verification": { "mode": "verify-if-sha-present", "algorithm": "SHA-256" },
                  "optimization": {
                    "olive": {
                      "mode": "existing-onnx-components",
                      "components": [ "encoder.onnx", "decoder.onnx" ],
                      "supported_providers": [ "trt-rtx" ],
                      "recipe_bindings": [
                        {
                          "provider": "trt-rtx",
                          "precision": "fp16",
                          "component": "{{component}}",
                          "config_relative_path": "example/encoder_fp16.json",
                          "operations": [ "provider_optimization" ]
                        }
                      ]
                    }
                  }
                }
              ]
            }
            """);

        if (valid)
        {
            ModelManifest manifest = Assert.Single(ModelManifestLoader.LoadCatalog(manifestPath).Models);
            Assert.Equal(component, Assert.Single(manifest.Optimization!.Olive!.RecipeBindings).Component);
        }
        else
        {
            ModelManifestValidationException ex = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));
            Assert.Contains("encoder_typo.onnx", ex.Message, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void LoadCatalog_LoadsNewestOliveProvidersAndRecipeMetadata()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "models": [
                {
                  "model_id": "example/olive-modern",
                  "task": "translation",
                  "engine_family": "phi-genai",
                  "capabilities": [ "translation" ],
                  "tier": "quality",
                  "license": "MIT",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_safe_mode": true,
                  "source_url": "https://example.invalid/model",
                  "revision": "main",
                  "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                  "root_path": "../../../../models/example-olive-modern",
                  "benchmark_entry": "genai_config.json",
                  "download_files": [ "genai_config.json" ],
                  "variants": [
                    {
                      "alias": "default",
                      "entry_path": "genai_config.json",
                      "is_default": true,
                      "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                      "download_files": [ "genai_config.json" ],
                      "opset": 17
                    }
                  ],
                  "hash_verification": {
                    "mode": "verify-if-sha-present",
                    "algorithm": "SHA-256"
                  },
                  "optimization": {
                    "olive": {
                      "mode": "ort-genai-builder",
                      "components": [ "genai_config.json" ],
                      "supported_providers": [ "migraphx", "qnn", "openvino", "rocm", "vitisai", "trt-rtx" ],
                      "supported_precisions": [ "fp16", "int8", "int4" ],
                      "fallback_policy": "none",
                      "recipe_bindings": [
                        {
                          "provider": "qnn",
                          "precision": "int8",
                          "config_relative_path": "example-olive-modern/qnn-int8.json",
                          "operations": [ "qnn_conversion", "compression", "evaluation" ],
                          "expected_output": "qnn_model_library",
                          "fallback_policy": "none",
                          "quantization_method": "qnn-int8",
                          "requires_calibration_data": true,
                          "script_relative_path": "example-olive-modern/scripts/qnn_eval.py",
                          "script_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                          "evaluator": "translation-smoke",
                          "output_manifest_relative_path": "example-olive-modern/qnn-int8.outputs.json"
                        },
                        {
                          "provider": "openvino",
                          "precision": "int4",
                          "config_relative_path": "example-olive-modern/openvino-int4.json",
                          "operations": [ "openvino_conversion", "compression" ],
                          "expected_output": "openvino_model",
                          "fallback_policy": "base_variant_allowed",
                          "quantization_method": "openvino-int4"
                        }
                      ]
                    }
                  }
                }
              ]
            }
            """);

        try
        {
            ModelManifest manifest = Assert.Single(ModelManifestLoader.LoadCatalog(manifestPath).Models);
            ModelOliveOptimizationProfile profile = manifest.Optimization!.Olive!;

            Assert.Contains(OliveOptimizationProvider.Migraphx, profile.SupportedProviders);
            Assert.Contains(OliveOptimizationProvider.Qnn, profile.SupportedProviders);
            Assert.Contains(OliveOptimizationProvider.OpenVino, profile.SupportedProviders);
            Assert.Contains(OliveOptimizationProvider.Rocm, profile.SupportedProviders);
            Assert.Contains(OliveOptimizationProvider.VitisAi, profile.SupportedProviders);
            Assert.Equal(OliveRecipeFallbackPolicy.None, profile.FallbackPolicy);

            OliveRecipeBinding qnn = Assert.Single(profile.RecipeBindings, binding => binding.Provider == "qnn");
            Assert.Contains(OliveOptimizationOperation.QnnConversion, qnn.Operations);
            Assert.Contains(OliveOptimizationOperation.Compression, qnn.Operations);
            Assert.Equal(OliveRecipeExpectedOutput.QnnModelLibrary, qnn.ExpectedOutput);
            Assert.Equal("qnn-int8", qnn.QuantizationMethod);
            Assert.True(qnn.RequiresCalibrationData);
            Assert.Equal("translation-smoke", qnn.Evaluator);
            Assert.Equal("example-olive-modern/qnn-int8.outputs.json", qnn.OutputManifestRelativePath);

            OliveRecipeBinding openVino = Assert.Single(profile.RecipeBindings, binding => binding.Provider == "openvino");
            Assert.Contains(OliveOptimizationOperation.OpenVinoConversion, openVino.Operations);
            Assert.Equal(OliveRecipeExpectedOutput.OpenVinoModel, openVino.ExpectedOutput);
            Assert.Equal(OliveRecipeFallbackPolicy.BaseVariantAllowed, openVino.FallbackPolicy);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    [Fact]
    public void LoadCatalog_RejectsRecipeBindingWithoutOperations()
    {
        string manifestPath = WriteTempManifest(
            """
            {
              "models": [
                {
                  "model_id": "example/bad-olive",
                  "task": "translation",
                  "engine_family": "phi-genai",
                  "license": "MIT",
                  "commercial_allowed": true,
                  "redistribution_allowed": true,
                  "requires_attribution": false,
                  "requires_user_consent": false,
                  "voice_cloning": false,
                  "commercial_safe_mode": true,
                  "source_url": "https://example.invalid/model",
                  "revision": "main",
                  "sha256": "",
                  "root_path": "../../../../models/example-bad-olive",
                  "benchmark_entry": "genai_config.json",
                  "download_files": [ "genai_config.json" ],
                  "variants": [
                    {
                      "alias": "default",
                      "entry_path": "genai_config.json",
                      "is_default": true,
                      "sha256": "",
                      "download_files": [ "genai_config.json" ]
                    }
                  ],
                  "optimization": {
                    "olive": {
                      "mode": "ort-genai-builder",
                      "components": [ "genai_config.json" ],
                      "supported_providers": [ "qnn" ],
                      "supported_precisions": [ "int8" ],
                      "recipe_bindings": [
                        {
                          "provider": "qnn",
                          "precision": "int8",
                          "config_relative_path": "bad/qnn.json",
                          "expected_output": "qnn_model_library"
                        }
                      ]
                    }
                  }
                }
              ]
            }
            """);

        try
        {
            ModelManifestValidationException exception = Assert.Throws<ModelManifestValidationException>(
                () => ModelManifestLoader.LoadCatalog(manifestPath));

            Assert.Contains("operations", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(manifestPath);
        }
    }

    private static string WriteTempManifest(string json)
    {
        string manifestPath = Path.Join(Path.GetTempPath(), $"{Guid.NewGuid():N}.json");
        File.WriteAllText(manifestPath, json);
        return manifestPath;
    }

    private static string FindRepoRoot()
    {
        DirectoryInfo? current = new DirectoryInfo(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Join(current.FullName, "Trackdub.slnx")))
            {
                return current.FullName;
            }

            current = current.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate the repository root.");
    }
}

public sealed partial class ModelHashVerifierTests
{
    [Fact]
    public async Task Verify_RequiredPolicyFailsWhenShaMissing()
    {
        var manifest = CreateManifest(HashVerificationMode.Required, sha256: "");
        var verifier = new ModelHashVerifier();
        string filePath = WriteTempFile([1, 2, 3]);

        try
        {
            HashVerificationResult result = await verifier.VerifyAsync(manifest, filePath);

            Assert.False(result.IsValid);
            Assert.False(result.WasVerified);
            Assert.Contains("required", result.FailureReason!, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(filePath);
        }
    }

    [Fact]
    public async Task Verify_VerifiesMatchingSha()
    {
        string filePath = WriteTempFile([1, 2, 3, 4]);
        string sha = new Sha256FileHasher().Compute(filePath);
        var manifest = CreateManifest(HashVerificationMode.VerifyIfShaPresent, sha);
        var verifier = new ModelHashVerifier();

        try
        {
            HashVerificationResult result = await verifier.VerifyAsync(manifest, filePath);

            Assert.True(result.IsValid);
            Assert.True(result.WasVerified);
            Assert.Equal(sha, result.ActualSha256);
        }
        finally
        {
            File.Delete(filePath);
        }
    }

    [Fact]
    public async Task Verify_ReportsHashMismatch()
    {
        string filePath = WriteTempFile([5, 6, 7, 8]);
        var manifest = CreateManifest(HashVerificationMode.VerifyIfShaPresent, sha256: "deadbeef");
        var verifier = new ModelHashVerifier();

        try
        {
            HashVerificationResult result = await verifier.VerifyAsync(manifest, filePath);

            Assert.False(result.IsValid);
            Assert.True(result.WasVerified);
            Assert.Contains("did not match", result.FailureReason!, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(filePath);
        }
    }

    [Fact]
    public async Task VerifyRaw_SkipsWhenSha256IsEmpty()
    {
        string filePath = WriteTempFile([9, 10, 11]);
        var verifier = new ModelHashVerifier();

        try
        {
            HashVerificationResult result = await verifier.VerifyAsync(
                expectedSha256: "",
                filePath: filePath);

            Assert.True(result.IsValid);
            Assert.False(result.WasVerified);
        }
        finally
        {
            File.Delete(filePath);
        }
    }

    [Fact]
    public async Task VerifyRaw_SkipsWhenSha256IsNull()
    {
        string filePath = WriteTempFile([12, 13]);
        var verifier = new ModelHashVerifier();

        try
        {
            HashVerificationResult result = await verifier.VerifyAsync(
                expectedSha256: null,
                filePath: filePath);

            Assert.True(result.IsValid);
            Assert.False(result.WasVerified);
        }
        finally
        {
            File.Delete(filePath);
        }
    }

    [Fact]
    public async Task VerifyRaw_VerifiesMatchingSha256()
    {
        string filePath = WriteTempFile([1, 2, 3, 4]);
        string sha = new Sha256FileHasher().Compute(filePath);
        var verifier = new ModelHashVerifier();

        try
        {
            HashVerificationResult result = await verifier.VerifyAsync(sha, filePath);

            Assert.True(result.IsValid);
            Assert.True(result.WasVerified);
            Assert.Equal(sha, result.ActualSha256);
        }
        finally
        {
            File.Delete(filePath);
        }
    }

    [Fact]
    public async Task VerifyRaw_ReportsMismatch()
    {
        string filePath = WriteTempFile([5, 6, 7, 8]);
        var verifier = new ModelHashVerifier();

        try
        {
            HashVerificationResult result = await verifier.VerifyAsync(
                expectedSha256: "deadbeef",
                filePath: filePath);

            Assert.False(result.IsValid);
            Assert.True(result.WasVerified);
            Assert.Contains("did not match", result.FailureReason!, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            File.Delete(filePath);
        }
    }

    private static ModelManifest CreateManifest(HashVerificationMode mode, string sha256) =>
        new(
            ModelId: "example/model",
            Task: ModelTask.Asr,
            EngineFamily: "test-engine",
            Capabilities: [],
            LanguageCoverage: ModelLanguageCoverage.Empty,
            Tier: "balanced",
            Lane: ModelLane.Commercial,
            License: ModelLicenseKind.Mit,
            CommercialAllowed: true,
            RedistributionAllowed: true,
            RequiresAttribution: false,
            RequiresUserConsent: false,
            VoiceCloning: false,
            CommercialUseVerified: true,
            SourceUrl: "",
            Revision: "",
            Sha256: sha256,
            DownloadFiles: [],
            DownloadFileSources: new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase),
            Variants: [],
            Aliases: [],
            RootPath: null,
            BenchmarkEntry: null,
            HashVerificationPolicy: new HashVerificationPolicy(mode, "SHA-256"));

    private static string WriteTempFile(byte[] bytes)
    {
        string filePath = Path.Join(Path.GetTempPath(), $"{Guid.NewGuid():N}.bin");
        File.WriteAllBytes(filePath, bytes);
        return filePath;
    }
}

public sealed class LocalModelCacheRecordStoreTests
{
    [Fact]
    public async Task SaveAndLoad_RoundTripsRecords()
    {
        string rootPath = Path.Join(Path.GetTempPath(), $"trackdub-cache-{Guid.NewGuid():N}");
        var storagePaths = new TrackdubStoragePaths(rootPath);
        var store = new LocalModelCacheRecordStore(storagePaths);
        LocalModelCacheRecord[] records =
        [
            new("example/model", @"D:\models\example", "main", "abc123", DateTimeOffset.UtcNow)
        ];

        try
        {
            await store.SaveAsync(records);
            IReadOnlyList<LocalModelCacheRecord> loaded = await store.LoadAsync();

            LocalModelCacheRecord record = Assert.Single(loaded);
            Assert.Equal(records[0].ModelId, record.ModelId);
            Assert.Equal(records[0].RootPath, record.RootPath);
            Assert.Equal(records[0].Sha256, record.Sha256);
            Assert.False(Directory.EnumerateFiles(storagePaths.ModelCacheDirectory, "*.tmp").Any());
        }
        finally
        {
            if (Directory.Exists(rootPath))
            {
                Directory.Delete(rootPath, recursive: true);
            }
        }
    }

    [Fact]
    public async Task MutateAsync_concurrent_updates_preserve_both_records()
    {
        string rootPath = Path.Join(Path.GetTempPath(), $"trackdub-cache-{Guid.NewGuid():N}");
        var storagePaths = new TrackdubStoragePaths(rootPath);
        var store = new LocalModelCacheRecordStore(storagePaths);
        LocalModelCacheRecord modelA = new(
            "example/model-a",
            Path.Join(rootPath, "model-a"),
            "main",
            "aaa",
            DateTimeOffset.UtcNow);

        try
        {
            await store.SaveAsync([modelA], default);

            Task addModelB = store.MutateAsync(
                records => records
                    .Append(new LocalModelCacheRecord(
                        "example/model-b",
                        Path.Join(rootPath, "model-b"),
                        "main",
                        "bbb",
                        DateTimeOffset.UtcNow))
                    .ToArray(),
                default);

            Task markModelACorrupt = store.MutateAsync(
                records => records
                    .Select(record => record.ModelId.Equals("example/model-a", StringComparison.OrdinalIgnoreCase)
                        ? record with { IntegrityFailed = true }
                        : record)
                    .ToArray(),
                default);

            await Task.WhenAll(addModelB, markModelACorrupt);

            IReadOnlyList<LocalModelCacheRecord> loaded = await store.LoadAsync(default);
            Assert.Equal(2, loaded.Count);
            LocalModelCacheRecord loadedA = Assert.Single(loaded, record => record.ModelId == "example/model-a");
            LocalModelCacheRecord loadedB = Assert.Single(loaded, record => record.ModelId == "example/model-b");
            Assert.True(loadedA.IntegrityFailed);
            Assert.False(loadedB.IntegrityFailed);
        }
        finally
        {
            if (Directory.Exists(rootPath))
            {
                Directory.Delete(rootPath, recursive: true);
            }
        }
    }

    [Fact]
    public async Task LoadAsync_deserializes_legacy_records_without_variants()
    {
        string rootPath = Path.Join(Path.GetTempPath(), $"trackdub-cache-{Guid.NewGuid():N}");
        var storagePaths = new TrackdubStoragePaths(rootPath);

        try
        {
            Directory.CreateDirectory(storagePaths.ModelCacheDirectory);
            await File.WriteAllTextAsync(
                storagePaths.ModelCacheIndexPath,
                """
                [
                  {
                    "ModelId": "example/model",
                    "RootPath": "D:\\models\\example",
                    "Revision": "main",
                    "Sha256": "abc123",
                    "CachedAtUtc": "2026-01-01T00:00:00+00:00",
                    "IntegrityFailed": false
                  }
                ]
                """);

            var store = new LocalModelCacheRecordStore(storagePaths);

            LocalModelCacheRecord record = Assert.Single(await store.LoadAsync());

            Assert.Equal("example/model", record.ModelId);
            Assert.Empty(record.Variants);
        }
        finally
        {
            if (Directory.Exists(rootPath))
            {
                Directory.Delete(rootPath, recursive: true);
            }
        }
    }

    [Fact]
    public async Task SaveAndLoad_RoundTripsVariantRecords()
    {
        string rootPath = Path.Join(Path.GetTempPath(), $"trackdub-cache-{Guid.NewGuid():N}");
        var storagePaths = new TrackdubStoragePaths(rootPath);
        var store = new LocalModelCacheRecordStore(storagePaths);
        var createdAt = new DateTimeOffset(2026, 1, 2, 3, 4, 5, TimeSpan.Zero);
        LocalModelCacheRecord[] records =
        [
            new(
                "example/model",
                @"D:\models\example",
                "main",
                "abc123",
                DateTimeOffset.UtcNow,
                Variants:
                [
                    new LocalModelVariantRecord(
                        "olive-cpu-fp32",
                        @"D:\models\example\optimized\olive-cpu-fp32",
                        "nested/model.onnx",
                        [ "nested/model.onnx" ],
                        "olive",
                        ExecutionProviderKind.Cpu,
                        "fp32",
                        createdAt,
                        "main",
                        "abc123")
                ])
        ];

        try
        {
            await store.SaveAsync(records);
            IReadOnlyList<LocalModelCacheRecord> loaded = await store.LoadAsync();

            LocalModelVariantRecord variant = Assert.Single(Assert.Single(loaded).Variants);
            Assert.Equal("olive-cpu-fp32", variant.Alias);
            Assert.Equal("nested/model.onnx", variant.EntryRelativePath);
            Assert.Equal(["nested/model.onnx"], variant.ComponentRelativePaths);
            Assert.Equal(ExecutionProviderKind.Cpu, variant.ExecutionProvider);
            Assert.Equal("main", variant.SourceModelRevision);
            Assert.Equal("abc123", variant.SourceModelSha256);
        }
        finally
        {
            if (Directory.Exists(rootPath))
            {
                Directory.Delete(rootPath, recursive: true);
            }
        }
    }
}

public sealed class LocalModelCacheRecordLookupTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Find_uses_authoritative_integrity_state_without_falling_back_to_stale_duplicates(bool newestFailed)
    {
        string rootPath = Path.Join(Path.GetTempPath(), "trackdub-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(rootPath);
        var storagePaths = new TrackdubStoragePaths(rootPath);
        var store = new LocalModelCacheRecordStore(storagePaths);
        var lookup = new LocalModelCacheRecordLookup(store);
        string modelRoot = Path.Join(rootPath, "models", "example-model");

        await store.MutateAsync(_ =>
        [
            new LocalModelCacheRecord("example/model", modelRoot, "rev-a", ValidSha256, DateTimeOffset.UnixEpoch, IntegrityFailed: !newestFailed),
            new LocalModelCacheRecord("example/model", modelRoot, "rev-b", ValidSha256, DateTimeOffset.UnixEpoch.AddDays(1), IntegrityFailed: newestFailed),
        ]);

        LocalModelCacheRecord? found = lookup.Find("example/model", modelRoot);

        if (newestFailed)
        {
            Assert.Null(found);
        }
        else
        {
            Assert.NotNull(found);
            Assert.Equal("rev-b", found.Revision);
        }
    }

    private const string ValidSha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
}
