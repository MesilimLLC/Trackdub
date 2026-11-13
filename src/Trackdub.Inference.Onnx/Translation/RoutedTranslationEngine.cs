using System.Runtime.CompilerServices;
using Trackdub.Contracts.Translation;
using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Onnx.Runtime.Routing;

namespace Trackdub.Inference.Onnx.Translation;

public sealed class RoutedTranslationEngine(
    ITranslationLanguageRouter translationLanguageRouter,
    IEnumerable<ITranslationEngineAdapter> adapters,
    ITranslationModelProvisioner? modelProvisioner = null)
    : IStreamingTranslationEngine, IStageRuntimeExecutionReporter, ITranslationExecutionMetadataReporter
{
    private readonly ITranslationLanguageRouter translationLanguageRouter = translationLanguageRouter ?? throw new ArgumentNullException(nameof(translationLanguageRouter));
    private readonly IReadOnlyList<ITranslationEngineAdapter> adapters = (adapters ?? throw new ArgumentNullException(nameof(adapters))).ToArray();

    public StageRuntimeExecutionSummary? LastExecutionSummary { get; private set; }

    public TranslationExecutionMetadata? LastExecutionMetadata { get; private set; }

    public async Task<IReadOnlyList<TranslatedTextSegment>> TranslateAsync(
        TranslationRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        (TranslationRouteSelection route, TranslationRequest routedRequest, ITranslationEngineAdapter selectedEngine) =
            await ResolveRouteAdapterAsync(request, cancellationToken).ConfigureAwait(false);

        IReadOnlyList<TranslatedTextSegment> translatedSegments = await selectedEngine.TranslateAsync(
            routedRequest,
            cancellationToken).ConfigureAwait(false);

        LastExecutionSummary = selectedEngine is IStageRuntimeExecutionReporter reporter
            ? reporter.LastExecutionSummary
            : null;
        LastExecutionMetadata = new TranslationExecutionMetadata(
            route.ProviderName,
            LastExecutionSummary?.ModelId ?? route.ModelId,
            LastExecutionSummary?.ModelAlias ?? route.PreferredModelAlias,
            LastExecutionSummary?.SelectedProvider,
            route.RoutingKind, (selectedEngine as ITranslationExecutionMetadataReporter)?.LastExecutionMetadata?.ModelProvenance ?? route.ModelProvenance);
        return translatedSegments;
    }

    /// <summary>
    /// Streams through a streaming-capable adapter unchanged; batch-only adapters execute once
    /// and their completed segments are wrapped in input order — compatibility, not progressive
    /// inference. Execution metadata/summary are captured after enumeration, including on
    /// partial consumption or failure.
    /// </summary>
    public async IAsyncEnumerable<PipelineStreamItem<TranslatedTextSegment>> TranslateStreamAsync(
        TranslationRequest request,
        Guid runId,
        string snapshotId,
        Guid sourceRevisionId,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        PipelineStreamItemFactory.ValidateTranslationStreamContext(runId, snapshotId, sourceRevisionId);

        (TranslationRouteSelection route, TranslationRequest routedRequest, ITranslationEngineAdapter adapter) =
            await ResolveRouteAdapterAsync(request, cancellationToken).ConfigureAwait(false);

        try
        {
            if (adapter is IStreamingTranslationEngineAdapter streamingAdapter)
            {
                await foreach (PipelineStreamItem<TranslatedTextSegment> item in streamingAdapter
                    .TranslateStreamAsync(routedRequest, runId, snapshotId, sourceRevisionId, cancellationToken)
                    .ConfigureAwait(false))
                {
                    yield return item;
                }
            }
            else
            {
                IReadOnlyList<TranslatedTextSegment> batch = await adapter
                    .TranslateAsync(routedRequest, cancellationToken)
                    .ConfigureAwait(false);
                long sequence = 0;
                foreach (TranslatedTextSegment segment in batch.OrderBy(static s => s.Index))
                {
                    yield return PipelineStreamItemFactory.CreateTranslation(
                        segment, runId, snapshotId, sourceRevisionId, sequence++);
                }
            }
        }
        finally
        {
            LastExecutionSummary = adapter is IStageRuntimeExecutionReporter reporter
                ? reporter.LastExecutionSummary
                : null;
            LastExecutionMetadata = new TranslationExecutionMetadata(
                route.ProviderName,
                LastExecutionSummary?.ModelId ?? route.ModelId,
                LastExecutionSummary?.ModelAlias ?? route.PreferredModelAlias,
                LastExecutionSummary?.SelectedProvider,
                route.RoutingKind, (adapter as ITranslationExecutionMetadataReporter)?.LastExecutionMetadata?.ModelProvenance ?? route.ModelProvenance);
        }
    }

    private async Task<(TranslationRouteSelection Route, TranslationRequest RoutedRequest, ITranslationEngineAdapter Adapter)>
        ResolveRouteAdapterAsync(TranslationRequest request, CancellationToken cancellationToken)
    {
        string? provisionedAlias = modelProvisioner is null ? request.PreferredModelAlias :
            await modelProvisioner.EnsureAsync(request.SourceLanguage, request.TargetLanguage, request.PreferredModelAlias, cancellationToken,
                preferredModelVariantAlias: request.PreferredModelVariantAlias,
                preferredExecutionProvider: request.PreferredExecutionProvider).ConfigureAwait(false);
        TranslationRouteSelection route = await translationLanguageRouter.ResolveRouteAsync(
            request.SourceLanguage,
            request.TargetLanguage,
            cancellationToken,
            provisionedAlias).ConfigureAwait(false);
        if (!route.IsAvailable)
        {
            throw new InvalidOperationException(
                route.UnavailableReason ??
                $"Translation route {request.SourceLanguage} -> {request.TargetLanguage} is not available.");
        }

        TranslationRequest routedRequest = request with
        {
            PreferredModelAlias = route.PreferredModelAlias,
            ResolvedModelEntryPath = route.ResolvedModelEntryPath,
            ModelProvenance = route.ModelProvenance
        };

        return (route, routedRequest, SelectAdapter(route));
    }

    private ITranslationEngineAdapter SelectAdapter(TranslationRouteSelection route)
    {
        if (adapters.Count == 0)
        {
            throw new InvalidOperationException("No translation inference adapters are registered.");
        }

        string? engineFamily = InferenceEngineAdapterSelector.NormalizeEngineFamily(route.EngineFamily);
        if (engineFamily is null)
        {
            throw new InvalidOperationException(
                $"Translation route {route.SourceLanguage} -> {route.TargetLanguage} did not specify an engine family.");
        }

        ITranslationEngineAdapter? adapter = adapters.FirstOrDefault(candidate =>
            string.Equals(candidate.EngineFamily, engineFamily, StringComparison.OrdinalIgnoreCase));
        if (adapter is null)
        {
            throw new InvalidOperationException(
                $"No translation inference adapter is registered for engine family '{engineFamily}'.");
        }

        return adapter;
    }
}
