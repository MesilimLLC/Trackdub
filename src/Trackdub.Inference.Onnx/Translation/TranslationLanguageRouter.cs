using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Contracts.Translation;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter(
    BundledModelManifestRegistry manifestRegistry,
    IModelCacheInventory modelCacheInventory,
    IRuntimePlanner? runtimePlanner = null,
    ITranslationModelProvisioner? modelProvisioner = null,
    IStudioSettingsService? settingsService = null)
    : ITranslationLanguageRouter
{
    private readonly BundledModelManifestRegistry manifestRegistry = manifestRegistry ?? throw new ArgumentNullException(nameof(manifestRegistry));
    private readonly IModelCacheInventory modelCacheInventory = modelCacheInventory ?? throw new ArgumentNullException(nameof(modelCacheInventory));

    // Routing through IRuntimePlanner lets the router share the same planning rules
    // (model integrity, smoke tests, EP availability) as the rest of the pipeline.
    private readonly IRuntimePlanner? runtimePlanner = runtimePlanner;
    private readonly ITranslationModelProvisioner? modelProvisioner = modelProvisioner;
    private readonly IStudioSettingsService? settingsService = settingsService;

    public async Task<IReadOnlyList<TranslationTargetLanguageOption>> GetSupportedTargetLanguagesAsync(
        string sourceLanguage,
        CancellationToken cancellationToken)
    {
        TranslationRoutingContext context = await BuildContextAsync(cancellationToken).ConfigureAwait(false);
        return BuildTargetOptions(sourceLanguage, context);
    }

    public async Task<TranslationRouteSelection> ResolveRouteAsync(
        string sourceLanguage,
        string targetLanguage,
        CancellationToken cancellationToken,
        string? preferredModelAlias = null)
    {
        TranslationRoutingContext context = await BuildContextAsync(cancellationToken).ConfigureAwait(false);
        RouteQuery query = CreateRouteQuery(sourceLanguage, targetLanguage, displayName: null, preferredModelAlias);
        return ResolveRoute(context, query);
    }
}
