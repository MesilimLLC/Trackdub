using Trackdub.Contracts.Translation;

namespace Trackdub.Composition.Tests;

internal sealed class Catalog : ITranslationModelCatalog
{
    public bool FailRefresh { get; init; }
    public int RefreshCalls { get; private set; }
    public TranslationModelCatalogSnapshot Snapshot { get; set; } = new(DateTimeOffset.UtcNow, "opus-mt-publishers-v1",
        [new("hf:onnx-community/opus-mt-xx-yy", "onnx-community/opus-mt-xx-yy", new string('a', 40), "xx", "yy", null, false,
            "apache-2.0", "pinned repository card", [.. new[] { "onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx", "config.json", "tokenizer_config.json", "vocab.json", "source.spm", "target.spm" }.Select(path => new TranslationModelFile(path, 10, new string('c', 64)))])], []);

    public Task<TranslationModelCatalogSnapshot> RefreshAsync(bool offline, CancellationToken cancellationToken)
    {
        RefreshCalls++;
        if (FailRefresh) throw new HttpRequestException("Unrelated catalog is unavailable.");
        return Task.FromResult(Snapshot);
    }
}
