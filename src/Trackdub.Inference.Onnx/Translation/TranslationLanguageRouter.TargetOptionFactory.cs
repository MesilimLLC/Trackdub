using Trackdub.Contracts.Pipeline;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private TranslationTargetLanguageOption CreateResolvedOption(
        TranslationRoutingContext context,
        string sourceLanguage,
        TranslationLanguageDefinition language)
    {
        RouteQuery query = CreateRouteQuery(sourceLanguage, language.Code, language.DisplayName, preferredModelAlias: null);
        TranslationRouteSelection route = ResolveRoute(context, query);
        return new TranslationTargetLanguageOption(
            language.Code,
            language.DisplayName,
            route.RoutingKind,
            route.IsAvailable,
            route.IsAvailable ? route.RouteDetail : route.UnavailableReason ?? route.RouteDetail,
            route.Status,
            route.ExpectedDownloadBytes);
    }

    private static TranslationTargetLanguageOption CreateUnresolvedOption(TranslationLanguageDefinition language) =>
        new(
            language.Code,
            language.DisplayName,
            TranslationRoutingKind.Unavailable,
            IsAvailable: false,
            Detail: "Translation model selection waits for source language detection.",
            PreparationStatus: TranslationPreparationStatus.SourceUnresolved);
}
