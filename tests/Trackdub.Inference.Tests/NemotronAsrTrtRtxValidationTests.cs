using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx;
using Trackdub.Inference.Onnx.Audio;
using Trackdub.Inference.Onnx.NemotronAsr;
using Trackdub.Inference.Onnx.TensorRtRtx;
using Trackdub.Inference.Runtime.Planning;
using Trackdub.TestDoubles;

namespace Trackdub.Inference.Tests;

/// <summary>
/// Hardware validation for the Nemotron ASR streaming decoder on TRT-RTX. The decoder uses
/// persistent OrtIoBinding tensors (fixed addresses) so the compiled CUDA graphs replay across
/// encoder windows and decoder steps; this test proves the bound path end to end on real speech.
/// Skips at discovery unless build/nemotron-3.5-asr-onnx-trtrtx-validated-fp16 exists.
/// </summary>
public sealed class NemotronAsrTrtRtxValidationTests
{
    private const string StagingDirectoryName = "nemotron-3.5-asr-onnx-trtrtx-validated-fp16";
    private const string FixtureRelativePath = "fixtures/whisper-real-speech-16k-mono.wav";

    [RequiresTrtRtxStagingFact(
        StagingDirectoryName,
        "run the nemotron TRT-RTX staging pipeline (tools/olive)")]
    [Trait("Category", "Integration")]
    public async Task NemotronAsrTrtRtx_TranscribesRealSpeech_DeterministicAcrossRuns()
    {
        TensorRtRtxBootstrapResult bootstrap = await TensorRtRtxPluginService.Shared
            .EnsureRegisteredAsync(allowProviderDownloads: false, CancellationToken.None);
        Assert.True(bootstrap.Succeeded, $"TRT-RTX plugin registration failed: {bootstrap.Detail}");

        string stagingDir = Path.Join(FindRepoRoot(), "build", StagingDirectoryName);
        string encoderPath = Path.Join(stagingDir, "encoder.onnx");
        Assert.True(File.Exists(encoderPath), $"Staged encoder not found: {encoderPath}");

        string wavePath = Path.GetFullPath(Path.Join(
            FindRepoRoot(), "tests", "Trackdub.Inference.Tests", FixtureRelativePath));
        Assert.True(File.Exists(wavePath), $"Speech fixture not found: {wavePath}");

        var regions = new[] { new SpeechRegion(0, 0.0, 6.0) };

        var firstEngine = CreateEngine(encoderPath, stagingDir);
        IReadOnlyList<RecognizedTranscriptSegment> first = await firstEngine.TranscribeAsync(
            wavePath, regions, CancellationToken.None);

        Assert.NotEmpty(first);
        string firstTranscript = string.Join(' ', first.Select(static segment => segment.Text)).Trim();
        Assert.True(
            firstTranscript.Any(char.IsLetter),
            $"Expected speech transcript from real-speech fixture, got: '{firstTranscript}'");

        // A second engine instance re-leases the pooled sessions; identical tokens prove the
        // persistent buffers do not leak state across Decode() calls.
        var secondEngine = CreateEngine(encoderPath, stagingDir);
        IReadOnlyList<RecognizedTranscriptSegment> second = await secondEngine.TranscribeAsync(
            wavePath, regions, CancellationToken.None);

        Assert.Equal(
            first.Select(static s => s.Text).ToArray(),
            second.Select(static s => s.Text).ToArray());
        Assert.False(
            string.IsNullOrWhiteSpace(firstEngine.LastExecutionSummary?.SelectedProvider),
            "SelectedProvider must not be empty — session provider resolution failed.");
    }

    /// <summary>
    /// Isolates CUDA-graph behavior from bound-path mechanics: the same mel drives the persistent-
    /// binding decoder on two pooled sessions that differ only in enable_cuda_graph. Token
    /// equality proves replay reads the synced input staging, not stale captured buffers.
    /// </summary>
    [RequiresTrtRtxStagingFact(
        StagingDirectoryName,
        "run the nemotron TRT-RTX staging pipeline (tools/olive)")]
    [Trait("Category", "Integration")]
    public async Task NemotronAsrTrtRtx_BoundDecoderTokens_MatchBetweenGraphModes()
    {
        TensorRtRtxBootstrapResult bootstrap = await TensorRtRtxPluginService.Shared
            .EnsureRegisteredAsync(allowProviderDownloads: false, CancellationToken.None);
        Assert.True(bootstrap.Succeeded, $"TRT-RTX plugin registration failed: {bootstrap.Detail}");

        string stagingDir = Path.Join(FindRepoRoot(), "build", StagingDirectoryName);
        string encoderPath = Path.Join(stagingDir, "encoder.onnx");
        string decoderPath = Path.Join(stagingDir, "decoder_joint.onnx");
        string tokenizerPath = Path.Join(stagingDir, "tokenizer.model");
        string configPath = Path.Join(stagingDir, "config.json");

        NemotronAsrSentencePieceVocab vocab = await NemotronAsrSentencePieceVocab
            .LoadAsync(tokenizerPath, CancellationToken.None);
        NemotronAsrExportConfig exportConfig = NemotronAsrExportConfig.Load(configPath, hasPromptInput: true);

        string wavePath = Path.GetFullPath(Path.Join(
            FindRepoRoot(), "tests", "Trackdub.Inference.Tests", FixtureRelativePath));
        Assert.True(File.Exists(wavePath), $"Speech fixture not found: {wavePath}");
        IAudioSamples audio = await WaveAudioReader.ReadMonoPcm16Async(wavePath, CancellationToken.None);
        var samples = new float[(int)audio.SampleFrameCount];
        audio.ReadMonoSamples(0, samples);
        audio.Dispose();

        var extractor = new NemotronAsrMelFeatureExtractor(exportConfig.ApplyPerFeatureNormalization);
        float[,] mel = extractor.Extract(samples);
        Assert.True(mel.GetLength(1) > 0);

        IReadOnlyDictionary<string, string> encoderProfiles =
            NemotronAsrEncoderTrtProfiles.BuildOptions(encoderPath);

        IReadOnlyList<int> tokensGraphsOff = await DecodeWithModeAsync(
            encoderPath, decoderPath, vocab, exportConfig, mel, encoderProfiles, enableGraphs: false);
        IReadOnlyList<int> tokensGraphsOn = await DecodeWithModeAsync(
            encoderPath, decoderPath, vocab, exportConfig, mel, encoderProfiles, enableGraphs: true);

        Assert.NotEmpty(tokensGraphsOff);
        Assert.NotEmpty(tokensGraphsOn);
        Assert.Equal(tokensGraphsOff, tokensGraphsOn);
    }

    private async Task<IReadOnlyList<int>> DecodeWithModeAsync(
        string encoderPath,
        string decoderPath,
        NemotronAsrSentencePieceVocab vocab,
        NemotronAsrExportConfig exportConfig,
        float[,] mel,
        IReadOnlyDictionary<string, string> encoderProfiles,
        bool enableGraphs)
    {
        using OnnxExecutionSessionFactory.NemotronAsrSessionLease lease = await OnnxExecutionSessionFactory
            .CreatePooledNemotronAsrAsync(new OnnxExecutionSessionFactory.PooledSessionRequest
            {
                EngineFamily = "nemotron-asr",
                Provider = ExecutionProviderKind.TensorRTRtx,
                CancellationToken = CancellationToken.None,
                ModelId = "diag/nemotron-graph-mode",
                Variant = enableGraphs ? "graphs-on" : "graphs-off",
                AdditionalTrtEncoderOptions = enableGraphs
                    ? OnnxExecutionSessionFactory.WithCudaGraphsEnabled(encoderProfiles)
                    : encoderProfiles,
                AdditionalTrtDecoderOptions = enableGraphs
                    ? NemotronAsrEncoderTrtProfiles.DecoderJointCudaGraphOptions
                    : null
            }, new OnnxExecutionSessionFactory.DualModelPaths { Encoder = encoderPath, Decoder = decoderPath });

        Assert.Equal(
            ProviderNames.FormatProviderLabel(ExecutionProviderKind.TensorRTRtx),
            lease.SelectedProvider);

        var decoder = new NemotronAsrGreedyDecoder(lease, vocab, exportConfig);
        IReadOnlyList<int> tokens = decoder.Decode(mel, promptIndex: 0);
        Assert.True(
            decoder.LastEncodedLength > 0,
            $"graphs={enableGraphs}: bound encoder produced encodedLen=0 — device input staging failed.");
        return tokens;
    }

    private static NemotronAsrOnnxAudioTranscriptionEngine CreateEngine(string encoderPath, string stagingDir) =>
        new(
            new StubRuntimePlanner(new StageRuntimePlan
            {
                Stage = RuntimeStage.Asr,
                Status = StageRuntimePlanStatus.Ready,
                ModelId = "tonythethompson/nemotron-3.5-asr-streaming-0.6b-onnx",
                ModelAlias = stagingDir,
                Variant = "default",
                ExecutionProvider = ExecutionProviderKind.TensorRTRtx,
                ModelEntryPath = encoderPath
            }),
            new BenchmarkModelPathResolver());

    private static string FindRepoRoot()
    {
        DirectoryInfo? current = new DirectoryInfo(AppContext.BaseDirectory);
        while (current is not null)
        {
            if (File.Exists(Path.Join(current.FullName, "Trackdub.slnx")))
                return current.FullName;
            current = current.Parent;
        }
        throw new DirectoryNotFoundException("Could not locate the repository root.");
    }

    private sealed class StubRuntimePlanner(StageRuntimePlan plan) : IRuntimePlanner
    {
        public Task<StageRuntimePlan> PlanAsync(
            StageRuntimePlanningRequest request,
            CancellationToken cancellationToken = default) =>
            Task.FromResult(plan);
    }
}
