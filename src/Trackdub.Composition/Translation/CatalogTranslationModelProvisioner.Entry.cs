using Trackdub.Contracts.Translation;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.Translation;

public sealed partial class CatalogTranslationModelProvisioner
{
    private BundledModelManifestEntry? CreateEntry(IGrouping<string, TranslationModelDescriptor> group, string policyVersion)
    {
        var model = group.First();
        if (!IsEligibleModel(model)) return null;
        string[] required = ["onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx", "config.json", "tokenizer_config.json", "vocab.json", "source.spm", "target.spm"];
        if (!HasRequiredFiles(model, required)) return null;
        if (model.Files.Select(file => file.Path).Distinct().Count() != model.Files.Length) return null;
        string root = Path.Join(storagePaths.ModelCacheDirectory, model.Repository);
        string? encoder = model.Files.FirstOrDefault(file => file.Path.EndsWith("encoder_model.onnx", StringComparison.Ordinal))?.Path;
        if (encoder is null) return null;
        string[] files = model.Files.Select(file => file.Path).ToArray();
        return new BundledModelManifestEntry(model.Repository, "translation", "opus-mt", ["translation", "direct-translation"],
            new ModelLanguageCoverage([], [], group.Select(pair => new ModelLanguagePair(pair.SourceLanguage, pair.TargetLanguage)).ToArray()),
            "fast", ModelLane.Commercial, model.License, true, true, model.License == "cc-by-4.0", false, false, true,
            $"https://huggingface.co/{model.Repository}", model.Revision, model.Files.First(file => file.Path == encoder).Sha256,
            files, model.Files.ToDictionary(file => file.Path, file => $"https://huggingface.co/{model.Repository}/resolve/{model.Revision}/{file.Path}"),
            model.Files.ToDictionary(file => file.Path, file => file.Sha256), [model.ModelId, model.Repository], root, Path.Join(root, encoder), [],
            ExpectedDownloadBytes: model.Files.Sum(file => file.SizeBytes),
            CatalogPolicyVersion: policyVersion,
            TargetPrefixes: group.GroupBy(pair => pair.TargetLanguage).ToDictionary(pair => pair.Key, pair => pair.First().TargetPrefix));
    }
}
