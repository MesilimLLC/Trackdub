using System.Text.Json;
using Trackdub.Inference.Runtime.ModelManifest;
using static Trackdub.Inference.Tests.TranslationRouterManifestSpecs;

namespace Trackdub.Inference.Tests;

internal sealed class TranslationRouterTestWorkspace : IDisposable
{
    public TranslationRouterTestWorkspace()
    {
        RootPath = Path.Join(Path.GetTempPath(), $"trackdub-translation-router-{Guid.NewGuid():N}");
        Directory.CreateDirectory(RootPath);
    }

    public string RootPath { get; }

    public BundledModelManifestRegistry WriteManifest(params ManifestSpec[] models)
    {
        string manifestPath = Path.Join(RootPath, "bundled-models.manifest.json");
        string json = JsonSerializer.Serialize(
            new
            {
                models = models.Select(model => new
                {
                    model_id = model.ModelId,
                    task = model.Task,
                    engine_family = model.EngineFamily,
                    capabilities = model.Capabilities,
                    language_coverage = new
                    {
                        source_languages = model.SourceLanguages,
                        target_languages = model.TargetLanguages,
                        language_pairs = model.LanguagePairs.Select(pair => new
                        {
                            source = pair.SourceLanguage,
                            target = pair.TargetLanguage
                        })
                    },
                    tier = model.Tier,
                    license = model.License,
                    commercial_allowed = model.CommercialAllowed,
                    redistribution_allowed = true,
                    requires_attribution = model.RequiresAttribution,
                    requires_user_consent = false,
                    voice_cloning = false,
                    commercial_use_verified = model.CommercialAllowed && IsValidSha256(model.Sha256),
                    source_url = $"https://example.invalid/{model.ModelId.Replace('/', '-')}",
                    revision = "main",
                    sha256 = model.Sha256,
                    aliases = model.Aliases,
                    root_path = $"./{model.RootFolder}",
                    benchmark_entry = model.BenchmarkEntry,
                    variants = model.Variants.Select(variant => new
                    {
                        alias = variant.Alias,
                        entry_path = variant.EntryPath
                    })
                })
            },
            new JsonSerializerOptions
            {
                WriteIndented = true
            });

        File.WriteAllText(manifestPath, json);
        return BundledModelManifestRegistry.Load(manifestPath);
    }

    public string CreateCacheRoot(string name)
    {
        string cacheRoot = Path.Join(RootPath, "machine-cache", name.Replace('/', Path.DirectorySeparatorChar));
        Directory.CreateDirectory(cacheRoot);
        return cacheRoot;
    }

    public void WriteOpusCacheFiles(string cacheRoot)
    {
        WriteCacheFile(cacheRoot, "onnx/encoder_model.onnx");
        WriteCacheFile(cacheRoot, "onnx/decoder_model_merged.onnx");
        WriteCacheFile(cacheRoot, "vocab.json", "{}");
        WriteCacheFile(cacheRoot, "source.model");
        WriteCacheFile(cacheRoot, "target.model");
    }

    public void WriteMadladCacheFiles(string cacheRoot, bool includeDefaultEncoder = true)
    {
        if (includeDefaultEncoder)
        {
            WriteCacheFile(cacheRoot, "encoder_model.onnx");
        }

        WriteCacheFile(cacheRoot, "encoder_model_int8.onnx");
        WriteCacheFile(cacheRoot, "decoder_model_int8.onnx");
        WriteCacheFile(cacheRoot, "spiece.model");
    }

    public void WriteGenAiCacheFiles(string cacheRoot)
    {
        WriteCacheFile(cacheRoot, "genai_config.json", "{}");
    }

    public void WriteCacheFile(string cacheRoot, string relativePath, string contents = "placeholder")
    {
        string filePath = Path.Join(cacheRoot, relativePath);
        string? directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(filePath, contents);
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(RootPath))
            {
                Directory.Delete(RootPath, recursive: true);
            }
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }
    }
}
