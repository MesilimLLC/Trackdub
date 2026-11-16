using Trackdub.Contracts.Translation;

namespace Trackdub.Composition.Tests;

internal sealed class Catalog : ITranslationModelCatalog
{
    public bool FailRefresh { get; init; }
    public int RefreshCalls { get; private set; }
    public int RefreshDelayMs { get; init; }
    public TranslationModelCatalogSnapshot? RefreshResult { get; init; }
    public TranslationModelCatalogSnapshot Snapshot { get; set; } = new(DateTimeOffset.UtcNow, "opus-mt-publishers-v1",
        [CreateOpusDescriptor("xx", "yy")], []);

    public async Task<TranslationModelCatalogSnapshot> RefreshAsync(bool offline, CancellationToken cancellationToken)
    {
        RefreshCalls++;
        if (FailRefresh) throw new HttpRequestException("Unrelated catalog is unavailable.");
        if (RefreshDelayMs > 0)
            await Task.Delay(RefreshDelayMs, cancellationToken).ConfigureAwait(false);
        return RefreshResult ?? Snapshot;
    }

    internal static TranslationModelDescriptor CreateOpusDescriptor(string source, string target) =>
        new($"hf:onnx-community/opus-mt-{source}-{target}", $"onnx-community/opus-mt-{source}-{target}", new string('a', 40),
            source, target, null, false, "apache-2.0", "pinned repository card",
            [.. new[] { "onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx", "config.json", "tokenizer_config.json", "vocab.json", "source.spm", "target.spm" }
                .Select(path => new TranslationModelFile(path, 10, new string('c', 64)))]);
}
