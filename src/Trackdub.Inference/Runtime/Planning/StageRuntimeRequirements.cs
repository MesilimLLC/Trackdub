using Trackdub.Domain;
using Trackdub.Inference.Runtime.ModelManifest;

namespace Trackdub.Inference.Runtime.Planning;

public sealed record StageRuntimeRequirements(
    RuntimeStage Stage,
    ModelTask RequiredTask,
    IReadOnlyList<string> PreferredModelAliases,
    IReadOnlyList<ExecutionProviderKind> AllowedProvidersThisMilestone,
    IReadOnlyList<string> PreferredGpuVariants,
    IReadOnlyList<string> PreferredCpuVariants,
    IReadOnlyDictionary<string, IReadOnlyList<ExecutionProviderKind>>? AllowedProvidersByEngineFamily = null,
    IReadOnlyList<string>? AllowedEngineFamilies = null,
    IReadOnlyList<string>? RequiredCapabilities = null,
    bool PreferTopRankedModelUntilReady = false);

internal static class Milestone5PlanningPolicy
{
    public static IReadOnlyList<ExecutionProviderKind> SupportedProvidersThisMilestone { get; } =
    [
        ExecutionProviderKind.TensorRTRtx,
        ExecutionProviderKind.Migraphx,
        ExecutionProviderKind.OpenVinoCatalog,
        ExecutionProviderKind.Qnn,
        ExecutionProviderKind.VitisAi,
        ExecutionProviderKind.TensorRt,
        ExecutionProviderKind.Cuda,
        ExecutionProviderKind.OpenVino,
        ExecutionProviderKind.CoreMl,
        ExecutionProviderKind.DirectMl,
        ExecutionProviderKind.Dnnl,
        ExecutionProviderKind.Cpu
    ];
}

internal static class StageRuntimeRequirementsCatalog
{
    private static IReadOnlyList<ExecutionProviderKind> DefaultOnnxStageAllowedProviders =>
        Milestone5PlanningPolicy.SupportedProvidersThisMilestone;

    private static IReadOnlyList<ExecutionProviderKind> WithoutTensorRtFamilies(
        IReadOnlyList<ExecutionProviderKind> providers) =>
        [.. providers.Where(static p => p is not ExecutionProviderKind.TensorRTRtx and not ExecutionProviderKind.TensorRt)];

    // Approved-export policy for the bundled GenAI graphs, not a platform ban on DirectML or
    // TensorRT. Dated observation (GenAI 0.17.1 on Windows ML 2.4, CPU/CUDA-targeted exports):
    // the bundled Whisper export segfaults in the encoder pass on DirectML, and the bundled
    // Qwen2.5 export fails its first DirectML kernel (0x80070057). NvTensorRtRtx has terminated
    // the host on these same exports. Keep the guard until an export built for that provider is
    // retested in an isolated process. Do not treat the observation as a provider-wide restriction.
    private static IReadOnlyList<ExecutionProviderKind> GenAiProviders(
        IReadOnlyList<ExecutionProviderKind> providers) =>
        [.. WithoutTensorRtFamilies(providers).Where(static p => p is not ExecutionProviderKind.DirectMl)];

    private static IReadOnlyList<ExecutionProviderKind> CpuFirst(
        IReadOnlyList<ExecutionProviderKind> providers) =>
        [ExecutionProviderKind.Cpu, .. providers.Where(static p => p is not ExecutionProviderKind.Cpu)];

    private static IReadOnlyList<ExecutionProviderKind> WithoutTensorRtRtx(
        IReadOnlyList<ExecutionProviderKind> providers) =>
        [.. providers.Where(static p => p is not ExecutionProviderKind.TensorRTRtx)];

    private static IReadOnlyList<ExecutionProviderKind> WithoutClassicTensorRt(
        IReadOnlyList<ExecutionProviderKind> providers) =>
        [.. providers.Where(static p => p is not ExecutionProviderKind.TensorRt)];

    private static IReadOnlyList<ExecutionProviderKind> PreferDirectMl(
        IReadOnlyList<ExecutionProviderKind> providers) =>
        [ExecutionProviderKind.DirectMl, .. providers.Where(static p => p != ExecutionProviderKind.DirectMl)];

    public static IReadOnlyDictionary<RuntimeStage, StageRuntimeRequirements> All { get; } =
        new Dictionary<RuntimeStage, StageRuntimeRequirements>
        {
            [RuntimeStage.Vad] = new(
                RuntimeStage.Vad,
                ModelTask.Vad,
                ["silero-vad", "silero"],
                // Both TensorRT providers are excluded for VAD. Silero-vad's If/else-branch
                // subgraph fails TensorRT engine building on both: TensorRT-RTX rejects it
                // (EP ABI 0.4.2/cu13, TensorRT-RTX 1.6.1: "squeeze index (1) must be less than
                // length (0)", then IConditionalOutputLayer rank mismatch, #329), and the
                // 2026-10-18 isolated retest showed classic TensorRT rejecting the same export
                // with a catchable shape-inference failure ("TensorRT input ... has no shape
                // specified"). Session-init fallback recovers it to CPU/DirectML, but each run
                // paid a failed engine build. The model is tiny, so CPU/DirectML cost nothing
                // meaningful. Revisit if a shape-inferred Silero export ships.
                WithoutTensorRtFamilies(DefaultOnnxStageAllowedProviders),
                ["fp16", "q4f16"],
                ["int8", "quantized", "uint8", "q4"]),
            [RuntimeStage.Asr] = new(
                RuntimeStage.Asr,
                ModelTask.Asr,
                [
                    "parakeet-tdt-0.6b-v3",
                    "qwen3-asr-0.6b",
                    "qwen3-asr-balanced",
                    "qwen3-asr-1.7b",
                    "qwen3-asr-quality",
                    "whisper-tiny-onnx",
                    "whisper-tiny",
                    "whisper-tiny-local",
                    "whisper-tiny-genai",
                ],
                DefaultOnnxStageAllowedProviders,
                ["default", "fp16"],
                ["default", "int8", "quantized", "uint8", "q4"],
                // Parakeet TDT 0.6B v3 is the default ASR POC model: the bundled ONNX encoder and
                // decoder passed the live Linux TRT-RTX smoke and cache-warm tests on an RTX 5070.
                // Qwen3 remains available but is not TensorRT-qualified: its bundled export uses
                // plugin ops the TensorRT importer cannot resolve, and the 2026-10-18 isolated
                // retest on classic TensorRT terminated the host process during import, so prefer
                // DML (or CUDA where usable) ahead of TRT. Whisper ONNX was previously denied
                // TensorRT over fused contrib ops; the 2026-10-18 retest passed Xenova
                // whisper-medium on both TensorRT providers, so it uses the default list and
                // the planner smoke-gates each Whisper variant individually.
                new Dictionary<string, IReadOnlyList<ExecutionProviderKind>>(StringComparer.OrdinalIgnoreCase)
                {
                    ["qwen3-asr"] = PreferDirectMl(WithoutClassicTensorRt(DefaultOnnxStageAllowedProviders)),
                    ["whisper-onnx"] = DefaultOnnxStageAllowedProviders,
                    ["whisper-genai"] = GenAiProviders(DefaultOnnxStageAllowedProviders),
                },
                // Nemotron ASR is not in the shipping auto-planning lane: quality-tier ranking
                // previously selected it ahead of working ONNX models and produced empty
                // transcripts. Explicit Nemotron override still resolves by alias if needed.
                AllowedEngineFamilies: ["parakeet-tdt", "qwen3-asr", "whisper-onnx", "whisper-genai"]),
            [RuntimeStage.Translation] = new(
                RuntimeStage.Translation,
                ModelTask.Translation,
                ["opus-en-es", "helsinki-opus-en-es", "opus-en-fr", "opus-en-de", "opus-en-it", "opus-en-pt", "opus-es-en", "helsinki-opus-es-en", "madlad400-mt", "madlad400"],
                DefaultOnnxStageAllowedProviders,
                ["trt-fp16", "int4-kv", "merged-decoder", "quantized", "fp16"],
                ["int4-kv", "merged-decoder", "quantized", "int8", "fp16"],
                // OPUS-MT split by provider after the 2026-10-18 isolated retest: the
                // merged-decoder export passed classic TensorRT smoke, but TensorRT-RTX
                // session construction aborted the host process (exit 134), so only RTX is
                // excluded here. (The older note blamed encoder-decoder construction under
                // TensorRT generally, ORT 1.24.5 era; classic TensorRT on ORT 1.31 is fine.)
                new Dictionary<string, IReadOnlyList<ExecutionProviderKind>>(StringComparer.OrdinalIgnoreCase)
                {
                    ["opus-mt"] = WithoutTensorRtRtx(DefaultOnnxStageAllowedProviders),
                    // MADLAD-400 3B: TensorRT support is an export property, and trt-fp16 is
                    // currently the only TensorRT-qualified MADLAD export; it runs on
                    // both TensorRT providers with bucketed static shapes, ~19x faster than the
                    // int4 KV-cache decoder on CPU at fp16 quality. TensorRT RTX stays first
                    // (per-worker EP-context warm, ~8 GB admission estimate vs ~12.7 GB for
                    // classic TensorRT); classic TensorRT follows as a GPU fallback after the
                    // 2026-10-18 isolated-process retest passed it. The smoke guard still
                    // refuses every non-trt-fp16 MADLAD pairing on either TensorRT provider.
                    // Otherwise CPU-first: the int4 decoder translates a warm clip in ~8 s on
                    // CPU, while DirectML pages on a 12 GB card (int4 81-89 s, fp16 160+ s)
                    // because the model and DirectML's working set do not fit beside the
                    // desktop. DirectML stays reachable through an explicit provider pin.
                    ["madlad"] = [ExecutionProviderKind.TensorRTRtx, ExecutionProviderKind.TensorRt, .. CpuFirst(WithoutTensorRtFamilies(DefaultOnnxStageAllowedProviders))],
                    // phi-genai loads through ORT GenAI, whose NvTensorRtRtx device terminates
                    // the host process (confirmed in the 2026-10-18 isolated retest, exit 134).
                    ["phi-genai"] = GenAiProviders(DefaultOnnxStageAllowedProviders),
                    ["qwen-genai"] = GenAiProviders(DefaultOnnxStageAllowedProviders),
                    ["gemma-genai"] = GenAiProviders(DefaultOnnxStageAllowedProviders),
                }),
            [RuntimeStage.Diarization] = new(
                RuntimeStage.Diarization,
                ModelTask.Diarization,
                ["diar-streaming-sortformer-4spk-v2.1", "sortformer-diarizer-4spk-v2.1", "sortformer-4spk", "nvidia-streaming-sortformer-4spk-v2.1"],
                DefaultOnnxStageAllowedProviders,
                ["default"],
                ["default"]),
            [RuntimeStage.Separation] = new(
                RuntimeStage.Separation,
                ModelTask.Separation,
                ["spleeter"],
                DefaultOnnxStageAllowedProviders,
                ["default"],
                ["default"],
                AllowedEngineFamilies: ["spleeter"],
                PreferTopRankedModelUntilReady: true),
            [RuntimeStage.OverlapRescue] = new(
                RuntimeStage.OverlapRescue,
                ModelTask.OverlapRescue,
                ["sepformer"],
                DefaultOnnxStageAllowedProviders,
                ["default"],
                ["default"],
                AllowedEngineFamilies: ["sepformer"]),
            [RuntimeStage.SpeechEnhancement] = new(
                RuntimeStage.SpeechEnhancement,
                ModelTask.SpeechEnhancement,
                ["deepfilternet3", "deepfilter"],
                DefaultOnnxStageAllowedProviders,
                ["default"],
                ["default"],
                AllowedEngineFamilies: ["deepfilternet3"]),
            [RuntimeStage.Tts] = new(
                RuntimeStage.Tts,
                ModelTask.Tts,
                ["kokoro-onnx", "kokoro", "chatterbox-turbo-onnx", "chatterbox-turbo", "chatterbox-onnx", "chatterbox",
                    "cosyvoice-300m", "cosyvoice",
                    "qwen3-tts-0.6b-customvoice", "qwen3-tts-0.6b", "qwen3-tts", "qwen-tts",
                    "qwen3-tts-1.7b-customvoice", "qwen3-tts-1.7b",
                    "qwen3-tts-0.6b-base", "qwen3-tts-1.7b-base"],
                DefaultOnnxStageAllowedProviders,
                ["int8", "q4f16", "fp16", "q4", "default", "int4"],
                ["int8", "q4", "quantized", "default"],
                // Per-graph TrtRtxUnsupportedOpScanner + session-init fallback isolate failures.
                // CosyVoice multi-graph packages compile under TRT RTX in micro-benchmarks
                // (token_generator 788MB, text_encoder, speech_tokenizer); the family-level
                // deny was over-broad. kokoro runs on CUDA or CPU only: DirectML fails its decoder
                // ConvTranspose (0x80070057) and TensorRT RTX cannot infer its duration-dependent
                // shapes; on Windows CUDA comes from the ORT 1.30 inference worker (ADR-0017).
                // chatterbox/qwen3-tts keep the deny until their multi-graph init is smoke-clean.
                new Dictionary<string, IReadOnlyList<ExecutionProviderKind>>(StringComparer.OrdinalIgnoreCase)
                {
                    ["kokoro"] = [ExecutionProviderKind.Cuda, ExecutionProviderKind.Cpu],
                    ["chatterbox"] = WithoutTensorRtFamilies(DefaultOnnxStageAllowedProviders),
                    ["qwen3-tts"] = WithoutTensorRtFamilies(DefaultOnnxStageAllowedProviders),
                },
                AllowedEngineFamilies: ["kokoro", "chatterbox", "cosyvoice", "qwen3-tts"]),
            [RuntimeStage.LipSync] = new(
                RuntimeStage.LipSync,
                ModelTask.ForcedAlignment,
                ["wav2vec2-lv60-espeak", "wav2vec2-lv60-espeak-cv-ft-onnx"],
                DefaultOnnxStageAllowedProviders,
                ["default"],
                ["performance", "default"],
                AllowedEngineFamilies: ["onnx-ctc-phoneme-aligner"],
                PreferTopRankedModelUntilReady: true),
            [RuntimeStage.TextRefinement] = new(
                RuntimeStage.TextRefinement,
                ModelTask.TextRefinement,
                ["qwen2.5-1.5b-instruct", "qwen-polisher", "text-refiner"],
                DefaultOnnxStageAllowedProviders,
                ["default", "fp16"],
                ["default", "int8", "quantized"],
                // ORT GenAI NvTensorRtRtx terminates the process (native stack overflow) when
                // loading/running bundled GenAI models, e.g. qwen-instruct (Qwen2.5-1.5B). A
                // smoke failure cannot gate a fatal crash, so GenAI families never see TRT RTX.
                new Dictionary<string, IReadOnlyList<ExecutionProviderKind>>(StringComparer.OrdinalIgnoreCase)
                {
                    ["qwen-instruct"] = GenAiProviders(DefaultOnnxStageAllowedProviders),
                    ["phi-genai"] = GenAiProviders(DefaultOnnxStageAllowedProviders),
                }),
            [RuntimeStage.LipSynthesis] = new(
                RuntimeStage.LipSynthesis,
                ModelTask.LipSynthesis,
                ["latentsync-1.6", "latentsync"],
                DefaultOnnxStageAllowedProviders,
                ["fp16", "default"],
                ["int8", "default"],
                // LatentSync UNet exports MultiHeadAttention, which TensorRT RTX cannot import.
                AllowedProvidersByEngineFamily: new Dictionary<string, IReadOnlyList<ExecutionProviderKind>>(StringComparer.OrdinalIgnoreCase)
                {
                    ["latentsync-diffusion"] = WithoutTensorRtFamilies(DefaultOnnxStageAllowedProviders),
                },
                AllowedEngineFamilies: ["latentsync-diffusion"]),
        };
}
