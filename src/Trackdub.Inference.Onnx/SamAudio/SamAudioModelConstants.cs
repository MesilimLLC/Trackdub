namespace Trackdub.Inference.Onnx.SamAudio;

/// <summary>
/// Fixed properties of the ONNX export of SAM-Audio small (speech prompt, audio only) published as
/// <c>tonythethompson/sam-audio-small-onnx</c>. The export's own <c>sam_audio_onnx.json</c> repeats these.
/// </summary>
internal static class SamAudioModelConstants
{
    public const string EngineFamilyName = "sam-audio";

    public const int SampleRate = 48000;

    /// <summary>Codec hop: one latent frame per this many samples. Encoder input is padded to a multiple of it.</summary>
    public const int HopLength = 1920;

    public const int LatentChannels = 256;

    public const int FeatureChannels = 128;

    /// <summary>Explicit midpoint steps over t in [0, 1]; the model's default of 2/32 step size.</summary>
    public const int OdeSteps = 16;

    /// <summary>Length of the random output-watermark message DACVAE embeds in every decode.</summary>
    public const int WatermarkBits = 16;

    /// <summary>Training clip length; longer input is separated in overlapping windows of this size.</summary>
    public const double WindowSeconds = 30.0;

    /// <summary>Codec decode chunk, in latent frames (6 s). A whole 30 s window does not decode on a 12 GB GPU (cuDNN fails).</summary>
    public const int DecodeChunkFrames = 150;

    /// <summary>Overlap between decode chunks, in latent frames (0.5 s), crossfaded; 65 dB agreement with a full decode.</summary>
    public const int DecodeOverlapFrames = 25;

    public const double OverlapSeconds = 5.0;

    public const string EncoderFileName = "codec_encoder.onnx";

    public const string DitFileName = "dit_step.onnx";

    public const string DecoderFileName = "codec_decoder.onnx";

    public const string PromptEmbeddingFileName = "prompt_speech.bin";

    public const string MetadataFileName = "sam_audio_onnx.json";

    public static int WindowSamples => (int)(WindowSeconds * SampleRate);

    public static int OverlapSamples => (int)(OverlapSeconds * SampleRate);

    public static string ResolveModelPath(string modelRootPath, string fileName)
    {
        string path = Path.Join(modelRootPath, fileName);
        return File.Exists(path)
            ? path
            : throw new FileNotFoundException($"SAM-Audio model file '{fileName}' not found under '{modelRootPath}'.", path);
    }
}
