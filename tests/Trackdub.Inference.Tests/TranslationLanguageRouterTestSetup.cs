using Trackdub.Contracts;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Translation;
using Trackdub.Inference.Runtime.ModelManifest;
using static Trackdub.Inference.Tests.TranslationRouterManifestSpecs;

namespace Trackdub.Inference.Tests;

internal static class TranslationLanguageRouterTestSetup
{
    internal static TranslationRouterTestWorkspace CreateWorkspace() => new();

    internal static TranslationLanguageRouter CreateRouter(
        BundledModelManifestRegistry registry,
        IStudioSettingsService? settings,
        params (string ModelId, string RootPath)[] cachedModels) =>
        new(
            registry,
            new InMemoryModelCacheInventory(
                cachedModels
                    .Select(model => new LocalModelCacheRecord(model.ModelId, model.RootPath, "main", ValidSha256, DateTimeOffset.UtcNow))
                    .ToArray()),
            settingsService: settings);

    internal static TranslationLanguageRouter CreateDirectAndPivotRouter(
        TranslationRouterTestWorkspace workspace,
        BundledModelManifestRegistry registry)
    {
        string opusCacheRoot = workspace.CreateCacheRoot("Helsinki-NLP/opus-mt-en-es");
        workspace.WriteOpusCacheFiles(opusCacheRoot);
        string madladCacheRoot = workspace.CreateCacheRoot("google/madlad400-3b-mt");
        workspace.WriteMadladCacheFiles(madladCacheRoot);
        return CreateRouter(
            registry,
            settings: null,
            ("Helsinki-NLP/opus-mt-en-es", opusCacheRoot),
            ("google/madlad400-3b-mt", madladCacheRoot));
    }

    internal static TranslationLanguageRouter CreateMadladRouter(
        TranslationRouterTestWorkspace workspace,
        BundledModelManifestRegistry registry,
        bool includeDefaultEncoder = true)
    {
        string madladCacheRoot = workspace.CreateCacheRoot("google/madlad400-3b-mt");
        workspace.WriteMadladCacheFiles(madladCacheRoot, includeDefaultEncoder);
        return CreateRouter(registry, settings: null, ("google/madlad400-3b-mt", madladCacheRoot));
    }

    internal static TranslationLanguageRouter CreateCachedRouter(
        TranslationRouterTestWorkspace workspace,
        BundledModelManifestRegistry registry,
        string modelId,
        bool writeOpusFiles)
    {
        string cacheRoot = workspace.CreateCacheRoot(modelId);
        if (writeOpusFiles)
        {
            workspace.WriteOpusCacheFiles(cacheRoot);
        }
        else
        {
            workspace.WriteMadladCacheFiles(cacheRoot);
        }

        return CreateRouter(registry, settings: null, (modelId, cacheRoot));
    }

    internal static ManifestSpec CreatePivotSpec(
        string modelId,
        IReadOnlyList<string> aliases,
        IReadOnlyList<string> capabilities,
        IReadOnlyList<ManifestLanguagePairSpec> languagePairs) =>
        new(
            ModelId: modelId,
            Task: "translation",
            EngineFamily: "madlad",
            Capabilities: capabilities,
            LanguagePairs: languagePairs,
            SourceLanguages: ["multi"],
            TargetLanguages: ["multi"],
            Tier: "quality",
            License: "Apache-2.0",
            CommercialAllowed: true,
            RequiresAttribution: true,
            Aliases: aliases,
            RootFolder: $"manifest-models/{aliases[0]}",
            BenchmarkEntry: "encoder_model.onnx",
            Variants: [new ManifestVariantSpec("int8", "encoder_model_int8.onnx")]);
}
