using Trackdub.Domain;

namespace Trackdub.Composition.StarterPacks;

/// <summary>
/// Bundled starter-pack models to exercise with <c>Trackdub.Benchmarks --scope trt-rtx-smoke</c>
/// and optional TRT RTX integration smoke tests.
/// Skips cleanly when a model is not cached locally.
/// </summary>
public static class TrtRtxSmokeCatalog
{
    public const string ScopeName = "trt-rtx-smoke";

    public sealed record Target(string ModelReference, string? Variant, string Label);

    /// <summary>
    /// Starter-pack GPU models where TRT RTX smoke is worth attempting.
    /// Silero VAD is omitted (planner excludes TensorRT families).
    /// Qwen ASR is included; planner keeps TensorRT RTX only if encoder, decoder_init, and decoder_step smoke all pass.
    /// </summary>
    public static IReadOnlyList<Target> StarterPackTurboGpu { get; } =
    [
        new("cgus/diar_streaming_sortformer_4spk-v2.1-onnx", null, "diarization"),
        new("openai/whisper-small", null, "asr-whisper-small"),
        new("openai/whisper-medium", null, "asr-whisper-medium"),
        new("tonythethompson/qwen3-asr-0.6b-onnx", null, "asr-qwen-0.6b"),
        new("tonythethompson/qwen3-asr-1.7b-onnx", null, "asr-qwen-1.7b"),
        new("microsoft/Phi-4-mini-instruct-onnx", "gpu-int4", "translation-phi"),
        new("google/madlad400-3b-mt", "quantized", "translation-madlad"),
        new("ResembleAI/chatterbox-turbo-ONNX", "fp16", "tts-chatterbox"),
    ];

    /// <summary>
    /// Bundled ONNX GPU models not in <see cref="StarterPackTurboGpu"/>.
    /// Includes Silero VAD. Omits Kokoro (no TRT-RTX route; CUDA or CPU only) and
    /// Phi-3.5 (CPU-only bundle: no GPU variant in the manifest).
    /// </summary>
    public static IReadOnlyList<Target> RemainingOnnxGpu { get; } =
    [
        new("openai/whisper-tiny", null, "asr-whisper-tiny"),
        new("openai/whisper-base", null, "asr-whisper-base"),
        new("openai/whisper-large-v3", null, "asr-whisper-large-v3"),
        new("onnx-community/whisper-tiny", null, "asr-whisper-tiny-onnx"),
        new("onnx-community/whisper-base", null, "asr-whisper-base-onnx"),
        new("onnx-community/whisper-small", null, "asr-whisper-small-onnx"),
        new("Xenova/whisper-medium", null, "asr-whisper-medium-onnx"),
        new("Xenova/whisper-large-v3", null, "asr-whisper-large-v3-onnx"),
        new("tonythethompson/nemotron-3.5-asr-streaming-0.6b-onnx", null, "asr-nemotron-0.6b"),
        new("tonythethompson/parakeet-tdt-0.6b-v3-onnx", null, "asr-parakeet-tdt-0.6b-v3"),
        new("tonythethompson/Qwen2.5-1.5B-Instruct", null, "text-refinement-qwen"),
        new("google/madlad400-3b-mt", "trt-fp16", "translation-madlad-trt-fp16"),
        new("onnx-community/opus-mt-en-es", "merged-decoder", "translation-opus-en-es"),
        new("onnx-community/opus-mt-es-en", "merged-decoder", "translation-opus-es-en"),
        new("onnx-community/opus-mt-en-fr", "merged-decoder", "translation-opus-en-fr"),
        new("onnx-community/opus-mt-en-de", "merged-decoder", "translation-opus-en-de"),
        new("onnx-community/opus-mt-en-it", "merged-decoder", "translation-opus-en-it"),
        new("onnx-community/opus-mt-en-ROMANCE", "merged-decoder", "translation-opus-en-romance"),
        new("Xenova/opus-mt-es-fr", "merged-decoder", "translation-opus-es-fr"),
        new("Xenova/opus-mt-es-de", "merged-decoder", "translation-opus-es-de"),
        new("Xenova/opus-mt-es-it", "merged-decoder", "translation-opus-es-it"),
        new("onnx-community/chatterbox-ONNX", null, "tts-chatterbox-onnx"),
        new("onnx-community/chatterbox-multilingual-ONNX", null, "tts-chatterbox-multilingual"),
        new("tonythethompson/CosyVoice-300M-ONNX", null, "tts-cosyvoice"),
        new("tonythethompson/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX", null, "tts-qwen-0.6b-custom"),
        new("tonythethompson/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX", null, "tts-qwen-1.7b-custom"),
        new("tonythethompson/Qwen3-TTS-12Hz-0.6B-Base-ONNX", null, "tts-qwen-0.6b-base"),
        new("tonythethompson/Qwen3-TTS-12Hz-1.7B-Base-ONNX", null, "tts-qwen-1.7b-base"),
        new("csukuangfj/sherpa-onnx-spleeter-2stems", null, "separation-spleeter"),
        new("Rikorose/DeepFilterNet3", null, "enhancement-deepfilternet3"),
        new("onnx-community/silero-vad", null, "vad-silero"),
        new("tonythethompson/sepformer-whamr16k-onnx", null, "overlap-sepformer"),
        new("wav2vec2-lv60-espeak-cv-ft-onnx", null, "align-wav2vec2"),
        new("qwen3-forced-aligner-0.6b-q4-onnx", null, "align-qwen"),
        new("ByteDance/LatentSync-1.6", "fp16", "lip-latentsync"),
        new("InsightFace/scrfd-500m", null, "face-scrfd"),
        new("InsightFace/2d106det", null, "face-2d106"),
    ];

    /// <summary>
    /// Smoke targets for a given execution provider. Kokoro is CUDA-only here (no TRT-RTX or
    /// native TensorRT route per planner policy), so it appears only in the CUDA list.
    /// </summary>
    public static IReadOnlyList<Target> TargetsForProvider(ExecutionProviderKind provider) => provider switch
    {
        ExecutionProviderKind.TensorRTRtx => RemainingOnnxGpu,
        ExecutionProviderKind.Cuda => DeduplicateByLabel(
        [
            .. RemainingOnnxGpu,
            .. StarterPackTurboGpu,
            new Target("onnx-community/Kokoro-82M-v1.0-ONNX", null, "tts-kokoro"),
            new Target("tonythethompson/CosyVoice-300M-ONNX", "int8", "tts-cosyvoice-int8"),
            new Target("google/madlad400-3b-mt", "int4-kv", "translation-madlad-int4-kv"),
        ]),
        ExecutionProviderKind.TensorRt => DeduplicateByLabel(
        [
            .. RemainingOnnxGpu,
            .. StarterPackTurboGpu,
            new Target("tonythethompson/CosyVoice-300M-ONNX", "int8", "tts-cosyvoice-int8"),
        ]),
        _ => [],
    };

    private static IReadOnlyList<Target> DeduplicateByLabel(IEnumerable<Target> targets) =>
        targets.DistinctBy(target => target.Label, StringComparer.OrdinalIgnoreCase).ToArray();
}
