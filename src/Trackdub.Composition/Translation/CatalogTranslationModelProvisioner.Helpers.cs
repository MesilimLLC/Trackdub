using Trackdub.Contracts.Licensing;
using Trackdub.Contracts.Translation;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Composition.Translation;

public sealed partial class CatalogTranslationModelProvisioner
{
    private static string? ResolveEffectiveAlias(string? preferredModelAlias) =>
        OpusMtAutoModel.IsAutoAlias(preferredModelAlias) ? null : preferredModelAlias;

    private static bool CanProvisionSelectedModel(BundledModelManifestEntry selected, bool explicitSelection) =>
        selected.EngineFamily == "opus-mt" ||
        (explicitSelection && selected.EngineFamily == "qwen-genai");

    private sealed class PreparationDownloadProgress(IProgress<string> progress, string modelId) : IProgress<ModelDownloadProgress>
    {
        public void Report(ModelDownloadProgress value) => progress.Report(
            value.PercentComplete < 100
                ? $"Downloading {modelId}: {value.PercentComplete}%"
                : $"Verifying {modelId}");
    }
}
