namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private static bool HasOpusSupportingFiles(string modelRootPath)
    {
        string onnxDirectory = Path.Join(modelRootPath, "onnx");
        return HasOpusDecoderFiles(modelRootPath, onnxDirectory) && HasOpusTokenizerFiles(modelRootPath);
    }

    private static bool HasOpusDecoderFiles(string modelRootPath, string onnxDirectory) =>
        File.Exists(Path.Join(modelRootPath, "decoder_model.onnx")) ||
        File.Exists(Path.Join(modelRootPath, "decoder_model_merged.onnx")) ||
        File.Exists(Path.Join(onnxDirectory, "decoder_model.onnx")) ||
        File.Exists(Path.Join(onnxDirectory, "decoder_model_merged.onnx"));

    private static bool HasOpusTokenizerFiles(string modelRootPath) =>
        File.Exists(Path.Join(modelRootPath, "vocab.json")) &&
        HasSourceTokenizer(modelRootPath) &&
        HasTargetTokenizer(modelRootPath);

    private static bool HasSourceTokenizer(string modelRootPath) =>
        File.Exists(Path.Join(modelRootPath, "source.spm")) || File.Exists(Path.Join(modelRootPath, "source.model"));

    private static bool HasTargetTokenizer(string modelRootPath) =>
        File.Exists(Path.Join(modelRootPath, "target.spm")) || File.Exists(Path.Join(modelRootPath, "target.model"));

    private static string ResolveOpusModelRootPath(string entryPath)
    {
        string entryDirectory = Path.GetDirectoryName(entryPath)
            ?? throw new InvalidOperationException("Direct Opus model root path could not be resolved.");
        if (HasOpusTokenizerFiles(entryDirectory))
        {
            return entryDirectory;
        }

        string? parentDirectory = Path.GetDirectoryName(entryDirectory);
        return parentDirectory is not null && HasOpusTokenizerFiles(parentDirectory)
            ? parentDirectory
            : entryDirectory;
    }
}
