using Trackdub.Contracts.Pipeline;

namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private static TranslationRouteSelection CreateSourceUnresolvedRoute(RouteQuery query) =>
        new(
            query.SourceLanguage,
            query.TargetLanguage,
            TranslationRoutingKind.Unavailable,
            IsAvailable: false,
            ProviderName: "none",
            RouteDetail: query.IsOpusMtAuto ? "Unavailable" : "Waiting for source language detection",
            UnavailableReason: query.IsOpusMtAuto ? "Transcribe first so the source language is known." : null,
            PreparationStatus: TranslationPreparationStatus.SourceUnresolved);

    private static TranslationRouteSelection CreateUnknownPairRoute(TranslationRoutingContext context, RouteQuery query)
    {
        if (query.IsOpusMtAuto)
        {
            return CreateUnavailablePreferredRoute(context, query);
        }

        return new TranslationRouteSelection(
            query.SourceLanguage,
            query.TargetLanguage,
            TranslationRoutingKind.Unavailable,
            IsAvailable: false,
            ProviderName: "none",
            RouteDetail: "Unavailable",
            UnavailableReason: $"No approved translation route is known for {query.SourceLanguage} -> {query.TargetLanguage}.");
    }

    private static TranslationRouteSelection CreateMissingRoute(
        TranslationRoutingContext context,
        RouteQuery query,
        string displayName)
    {
        string pivotLabel = ResolveUnavailablePivotLabel(context.PivotEntries);
        bool directPairKnown = context.DirectEntriesByPair.ContainsKey((query.SourceLanguage, query.TargetLanguage));
        string missingReason = directPairKnown
            ? $"Direct translation is not installed for {displayName}, and {pivotLabel} is unavailable."
            : $"{pivotLabel} is unavailable for {displayName}.";
        return new TranslationRouteSelection(
            query.SourceLanguage,
            query.TargetLanguage,
            TranslationRoutingKind.Unavailable,
            IsAvailable: false,
            ProviderName: "none",
            RouteDetail: "Unavailable",
            UnavailableReason: missingReason);
    }
}
