namespace Trackdub.Inference.Tests;

internal static class TranslationRouterManifestSpecs
{
    internal const string ValidSha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    internal static ManifestSpec CreateTranslationSpec(
        string modelId,
        IReadOnlyList<string> aliases,
        string rootFolder,
        string sourceLanguage = "en",
        string targetLanguage = "es") =>
        new(
            ModelId: modelId,
            Task: "translation",
            EngineFamily: "opus-mt",
            Capabilities: ["translation", "direct-translation"],
            LanguagePairs: [new ManifestLanguagePairSpec(sourceLanguage, targetLanguage)],
            SourceLanguages: [],
            TargetLanguages: [],
            Tier: "fast",
            License: "Apache-2.0",
            CommercialAllowed: true,
            RequiresAttribution: true,
            Aliases: aliases,
            RootFolder: rootFolder,
            BenchmarkEntry: "onnx/encoder_model.onnx",
            Variants:
            [
                new ManifestVariantSpec("merged-decoder", "onnx/decoder_model_merged.onnx")
            ]);

    internal static ManifestSpec CreateMadladSpec() =>
        new(
            ModelId: "google/madlad400-3b-mt",
            Task: "translation",
            EngineFamily: "madlad",
            Capabilities: ["translation", "pivot-translation"],
            LanguagePairs: [],
            SourceLanguages: ["multi"],
            TargetLanguages: ["multi"],
            Tier: "quality",
            License: "Apache-2.0",
            CommercialAllowed: true,
            RequiresAttribution: true,
            Aliases: ["broad-pivot-mt"],
            RootFolder: "manifest-models/madlad400",
            BenchmarkEntry: "encoder_model.onnx",
            Variants:
            [
                new ManifestVariantSpec("int8", "encoder_model_int8.onnx"),
                new ManifestVariantSpec("fp16", "encoder_model_fp16.onnx")
            ]);

    internal static ManifestSpec CreatePhiGenAiSpec() =>
        new(
            ModelId: "microsoft/Phi-3.5-mini-instruct-onnx",
            Task: "translation",
            EngineFamily: "phi-genai",
            Capabilities: ["translation", "pivot-translation"],
            LanguagePairs: [],
            SourceLanguages: ["multi"],
            TargetLanguages: ["multi"],
            Tier: "quality",
            License: "MIT",
            CommercialAllowed: true,
            RequiresAttribution: true,
            Aliases: ["phi-genai-pivot"],
            RootFolder: "manifest-models/phi-genai",
            BenchmarkEntry: "genai_config.json",
            Variants: []);

    internal static bool IsValidSha256(string hash) =>
        hash.Length == 64 && hash.All(static c =>
            c is >= '0' and <= '9' or >= 'a' and <= 'f' or >= 'A' and <= 'F');
}

internal sealed record ManifestSpec(
    string ModelId,
    string Task,
    string EngineFamily,
    IReadOnlyList<string> Capabilities,
    IReadOnlyList<ManifestLanguagePairSpec> LanguagePairs,
    IReadOnlyList<string> SourceLanguages,
    IReadOnlyList<string> TargetLanguages,
    string Tier,
    string License,
    bool CommercialAllowed,
    bool RequiresAttribution,
    IReadOnlyList<string> Aliases,
    string RootFolder,
    string BenchmarkEntry,
    IReadOnlyList<ManifestVariantSpec> Variants,
    string Sha256 = TranslationRouterManifestSpecs.ValidSha256);

internal sealed record ManifestVariantSpec(
    string Alias,
    string EntryPath);

internal sealed record ManifestLanguagePairSpec(
    string SourceLanguage,
    string TargetLanguage);
