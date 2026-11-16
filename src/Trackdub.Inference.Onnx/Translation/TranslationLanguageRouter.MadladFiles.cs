namespace Trackdub.Inference.Onnx.Translation;

public sealed partial class TranslationLanguageRouter
{
    private static bool HasMadladSupportingFiles(string modelRootPath) =>
        HasMadladDecoder(modelRootPath) && HasMadladTokenizer(modelRootPath);

    private static bool HasMadladDecoder(string modelRootPath) =>
        File.Exists(Path.Join(modelRootPath, "decoder_model.onnx")) ||
        File.Exists(Path.Join(modelRootPath, "decoder_model_merged.onnx")) ||
        File.Exists(Path.Join(modelRootPath, "decoder_model_quantized.onnx")) ||
        File.Exists(Path.Join(modelRootPath, "decoder_model_int8.onnx"));

    private static bool HasMadladTokenizer(string modelRootPath) =>
        File.Exists(Path.Join(modelRootPath, "spiece.model")) ||
        File.Exists(Path.Join(modelRootPath, "tokenizer.model")) ||
        File.Exists(Path.Join(modelRootPath, "sentencepiece.model"));
}
