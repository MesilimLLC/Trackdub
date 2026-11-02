using Trackdub.Domain;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.Runtime;
using Trackdub.Inference.Onnx.Runtime.Planning;
using Trackdub.Inference.Runtime.Planning;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Reflection;

namespace Trackdub.Inference.Tests;

public sealed class OnnxExecutionProviderSmokeTesterTests
{
    [Fact]
    public void ResolveWhisperEncoderSmokeDimensions_uses_valid_mel_window_for_symbolic_inputs()
    {
        // The onnx-community whisper encoder declares input_features as
        // (batch_size, feature_size, encoder_sequence_length) — all symbolic.
        // The generic all-ones mapping would yield [1,1,1] and fail the first
        // Conv, which expects feature_size (80) mel channels.
        int[] dimensions = OnnxExecutionProviderSmokeTester.ResolveWhisperEncoderSmokeDimensionsForTesting(
            new OnnxExecutionProviderSmokeTester.WhisperSmokeInputSpec
            {
                InputName = "input_features",
                ModelDimensions = [-1, -1, -1],
                MelBins = 80,
            });

        Assert.Equal([1, 80, 3000], dimensions);
    }

    [Fact]
    public void ResolveWhisperEncoderSmokeDimensions_honours_large_model_mel_bins()
    {
        int[] dimensions = OnnxExecutionProviderSmokeTester.ResolveWhisperEncoderSmokeDimensionsForTesting(
            new OnnxExecutionProviderSmokeTester.WhisperSmokeInputSpec
            {
                InputName = "input_features",
                ModelDimensions = [-1, -1, -1],
                MelBins = 128,
            });

        Assert.Equal([1, 128, 3000], dimensions);
    }

    [Fact]
    public void ResolveWhisperEncoderSmokeDimensions_falls_back_for_non_feature_inputs()
    {
        int[] other = OnnxExecutionProviderSmokeTester.ResolveWhisperEncoderSmokeDimensionsForTesting(
            new OnnxExecutionProviderSmokeTester.WhisperSmokeInputSpec
            {
                InputName = "attention_mask",
                ModelDimensions = [-1, -1],
                MelBins = 80,
            });
        int[] unsetMelBins = OnnxExecutionProviderSmokeTester.ResolveWhisperEncoderSmokeDimensionsForTesting(
            new OnnxExecutionProviderSmokeTester.WhisperSmokeInputSpec
            {
                InputName = "input_features",
                ModelDimensions = [-1, -1, -1],
                MelBins = 0,
            });

        Assert.Equal([1, 1], other);
        Assert.Equal([1, 1, 1], unsetMelBins);
    }

    [Fact]
    public void ResolveSeparationSmokeInputDimensions_SpleeterShape()
    {
        int[] dimensions = OnnxExecutionProviderSmokeTester.ResolveSeparationSmokeInputDimensionsForTesting(
            [1, 2, 44100]);

        Assert.Equal([1, 2, 44100], dimensions);
    }

    [Fact]
    public void ResolveSeparationSmokeInputDimensions_SplitsFirstSpleeterExport_PutsSingleSplitBeforeTwoChannels()
    {
        // Pinned csukuangfj/sherpa-onnx-spleeter-2stems @ 3e5a4dd (bundled manifest):
        // x = [num_splits, 2, 512, 1024]. Feeding the old channels-first guess
        // [2, 1, 512, 1024] fails with "index: 1 Got: 1 Expected: 2".
        int[] dimensions = OnnxExecutionProviderSmokeTester.ResolveSeparationSmokeInputDimensionsForTesting(
            [-1, 2, 512, 1024]);

        Assert.Equal([1, 2, 512, 1024], dimensions);
    }

    [Fact]
    public void ResolveSeparationSmokeInputDimensions_ChannelsFirstSpleeterExport_PutsTwoChannelsBeforeSingleSplit()
    {
        // Older sherpa-onnx export: x = [2, num_splits, 512, 1024].
        int[] dimensions = OnnxExecutionProviderSmokeTester.ResolveSeparationSmokeInputDimensionsForTesting(
            [2, -1, 512, 1024]);

        Assert.Equal([2, 1, 512, 1024], dimensions);
    }

    [Fact]
    public void IsSortFormerStreamingInputSet_requires_the_complete_streaming_contract()
    {
        MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
            .GetMethod("IsSortFormerStreamingInputSet", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate SortFormer streaming-input detector.");
        string[] inputNames = [
            "chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo", "fifo_lengths"];

        Assert.True(Assert.IsType<bool>(method.Invoke(null, [inputNames])));
        Assert.False(Assert.IsType<bool>(method.Invoke(null, [inputNames[..^1]])));
        Assert.False(Assert.IsType<bool>(method.Invoke(null, [new[] { "waveform", "waveform_lengths" }])));
    }

    [Fact]
    public void ResolveDiarizationInputNames_prefers_audio_length_and_generic_single_float_waveform_inputs()
    {
        MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
            .GetMethod("ResolveDiarizationInputNames", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate diarization input-name resolver.");

        object? rawResult = method.Invoke(
            null,
            [
                new Dictionary<string, Type>
                {
                    ["audio_signal"] = typeof(float),
                    ["audio_length"] = typeof(long)
                }
            ]);
        (string waveformName, string? lengthName) inputNames = Assert.IsType<(string, string?)>(rawResult);

        Assert.Equal("audio_signal", inputNames.waveformName);
        Assert.Equal("audio_length", inputNames.lengthName);
    }

    [Fact]
    public void ResolveDiarizationInputNames_falls_back_to_single_float_input_and_any_length_named_long_input()
    {
        MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
            .GetMethod("ResolveDiarizationInputNames", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate diarization input-name resolver.");

        object? rawResult = method.Invoke(
            null,
            [
                new Dictionary<string, Type>
                {
                    ["samples"] = typeof(float),
                    ["chunk_lengths"] = typeof(long)
                }
            ]);
        (string waveformName, string? lengthName) inputNames = Assert.IsType<(string, string?)>(rawResult);

        Assert.Equal("samples", inputNames.waveformName);
        Assert.Equal("chunk_lengths", inputNames.lengthName);
    }

    [Fact]
    public void CreateTtsInputValue_supports_onnxruntime_float16_metadata()
    {
        MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
            .GetMethod("CreateTtsInputValue", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate TTS smoke-input helper.");

        object? rawResult = method.Invoke(null, ["conditioning", typeof(Float16), new[] { 1, 2 }]);
        NamedOnnxValue value = Assert.IsType<NamedOnnxValue>(rawResult);
        Tensor<Float16> tensor = Assert.IsAssignableFrom<Tensor<Float16>>(value.Value);

        Assert.Equal([1, 2], tensor.Dimensions.ToArray());
        Assert.Equal(0f, (float)tensor[0, 0]);
        Assert.Equal(0f, (float)tensor[0, 1]);
    }

    /// <summary>Verifies that Half conditioning inputs retain their declared shape and zero-filled values.</summary>
    [Fact]
    public void CreateTtsInputValue_supports_half_metadata()
    {
        MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
            .GetMethod("CreateTtsInputValue", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate TTS smoke-input helper.");

        object? rawResult = method.Invoke(null, ["conditioning", typeof(Half), new[] { 1, 2 }]);
        NamedOnnxValue value = Assert.IsType<NamedOnnxValue>(rawResult);
        Tensor<Half> tensor = Assert.IsAssignableFrom<Tensor<Half>>(value.Value);

        Assert.Equal([1, 2], tensor.Dimensions.ToArray());
        Assert.Equal((Half)0f, tensor[0, 0]);
        Assert.Equal((Half)0f, tensor[0, 1]);
    }

    /// <summary>Verifies that the Kokoro probe receives unit speed to prevent a zero-duration Reshape failure.</summary>
    [Fact]
    public void CreateTtsInputValue_uses_unit_speed_for_kokoro_speed_input()
    {
        MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
            .GetMethod("CreateTtsInputValue", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate TTS smoke-input helper.");

        object? rawResult = method.Invoke(null, ["speed", typeof(float), new[] { 1 }]);
        NamedOnnxValue value = Assert.IsType<NamedOnnxValue>(rawResult);
        Tensor<float> tensor = Assert.IsAssignableFrom<Tensor<float>>(value.Value);

        Assert.Equal([1], tensor.Dimensions.ToArray());
        Assert.Equal(1f, tensor[0]);
    }

    [Fact]
    public void ResolveTtsProbeModelPath_uses_chatterbox_conditional_decoder_for_planned_provider_probe()
    {
        string root = Path.Join(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Join(root, "onnx"));
        try
        {
            string languageModelPath = Path.Join(root, "onnx", "language_model_q4f16.onnx");
            string conditionalDecoderPath = Path.Join(root, "onnx", "conditional_decoder_q4f16.onnx");
            File.WriteAllText(languageModelPath, string.Empty);
            File.WriteAllText(conditionalDecoderPath, string.Empty);
            MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
                .GetMethod("ResolveTtsProbeModelPath", BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException("Could not locate TTS smoke probe resolver.");

            object? rawResult = method.Invoke(
                null,
                [
                    new ExecutionProviderSmokeTestRequest(
                        RuntimeStage.Tts,
                        "ResembleAI/chatterbox-turbo-ONNX",
                        "chatterbox-turbo-onnx",
                        EngineFamily: null,
                        "q4f16",
                        ExecutionProviderKind.Cpu,
                        root,
                        languageModelPath)
                ]);
            string probePath = Assert.IsType<string>(rawResult);

            Assert.Equal(conditionalDecoderPath, probePath);
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    [Fact]
    public void ResolveTtsProbeModelPath_falls_back_to_default_chatterbox_conditional_decoder()
    {
        string root = Path.Join(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Join(root, "onnx"));
        try
        {
            string languageModelPath = Path.Join(root, "onnx", "language_model_fp16.onnx");
            string conditionalDecoderPath = Path.Join(root, "onnx", "conditional_decoder.onnx");
            File.WriteAllText(languageModelPath, string.Empty);
            File.WriteAllText(conditionalDecoderPath, string.Empty);
            MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
                .GetMethod("ResolveTtsProbeModelPath", BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException("Could not locate TTS smoke probe resolver.");

            object? rawResult = method.Invoke(
                null,
                [
                    new ExecutionProviderSmokeTestRequest(
                        RuntimeStage.Tts,
                        "onnx-community/chatterbox-ONNX",
                        "chatterbox-onnx",
                        EngineFamily: null,
                        "fp16",
                        ExecutionProviderKind.Cpu,
                        root,
                        languageModelPath)
                ]);
            string probePath = Assert.IsType<string>(rawResult);

            Assert.Equal(conditionalDecoderPath, probePath);
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    [Fact]
    public void ResolveTtsProbeModelPath_uses_entry_path_directory_when_model_root_is_blank()
    {
        string root = Path.Join(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        string onnxRoot = Path.Join(root, "onnx");
        Directory.CreateDirectory(onnxRoot);
        try
        {
            string languageModelPath = Path.Join(onnxRoot, "language_model_q4f16.onnx");
            string conditionalDecoderPath = Path.Join(onnxRoot, "conditional_decoder_q4f16.onnx");
            File.WriteAllText(languageModelPath, string.Empty);
            File.WriteAllText(conditionalDecoderPath, string.Empty);
            MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
                .GetMethod("ResolveTtsProbeModelPath", BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException("Could not locate TTS smoke probe resolver.");

            object? rawResult = method.Invoke(
                null,
                [
                    new ExecutionProviderSmokeTestRequest(
                        RuntimeStage.Tts,
                        "ResembleAI/chatterbox-turbo-ONNX",
                        "chatterbox-turbo-onnx",
                        EngineFamily: null,
                        "q4f16",
                        ExecutionProviderKind.Cpu,
                        string.Empty,
                        languageModelPath)
                ]);
            string probePath = Assert.IsType<string>(rawResult);

            Assert.Equal(conditionalDecoderPath, probePath);
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    [Fact]
    public void ResolveTtsProbeModelPath_keeps_entry_path_for_non_chatterbox_tts_models()
    {
        MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
            .GetMethod("ResolveTtsProbeModelPath", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate TTS smoke probe resolver.");
        string entryPath = Path.Join(Path.GetTempPath(), "kokoro.onnx");

        object? rawResult = method.Invoke(
            null,
            [
                new ExecutionProviderSmokeTestRequest(
                    RuntimeStage.Tts,
                    "onnx-community/Kokoro-82M-v1.0-ONNX",
                    "kokoro",
                    EngineFamily: null,
                    "default",
                    ExecutionProviderKind.Cpu,
                    Path.GetTempPath(),
                    entryPath)
            ]);
        string probePath = Assert.IsType<string>(rawResult);

        Assert.Equal(entryPath, probePath);
    }

    [Fact]
    public void ResolveNemotronDecoderJointPath_uses_decoder_joint_next_to_encoder()
    {
        string root = Path.Join(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string encoderPath = Path.Join(root, "encoder.onnx");
            string decoderJointPath = Path.Join(root, "decoder_joint.onnx");
            File.WriteAllText(encoderPath, string.Empty);
            File.WriteAllText(decoderJointPath, string.Empty);
            MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
                .GetMethod("ResolveNemotronDecoderJointPath", BindingFlags.NonPublic | BindingFlags.Static)
                ?? throw new InvalidOperationException("Could not locate Nemotron smoke decoder resolver.");

            object? rawResult = method.Invoke(null, [encoderPath]);
            string resolved = Assert.IsType<string>(rawResult);

            Assert.Equal(decoderJointPath, resolved);
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    [Fact]
    public void CreateNemotronDecoderInputs_defaults_to_int64_token_tensors()
    {
        MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
            .GetMethod("CreateNemotronDecoderInputs", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate Nemotron smoke decoder input builder.");
        var encoded = new DenseTensor<float>(new float[1024], [1, 1024, 1]);

        using var inputSet = Assert.IsAssignableFrom<IDisposable>(method.Invoke(
            null,
            [
                new Dictionary<string, NodeMetadata>(),
                encoded
            ]));
        object values = inputSet.GetType().GetProperty("Values")!.GetValue(inputSet)!;
        IReadOnlyList<NamedOnnxValue> namedValues = Assert.IsAssignableFrom<IReadOnlyList<NamedOnnxValue>>(values);
        NamedOnnxValue targets = namedValues.Single(static value => value.Name == "targets");
        NamedOnnxValue targetLength = namedValues.Single(static value => value.Name == "target_length");

        // Per review (P2 on PR #335): smoke path must default to int64 to match the Nemotron
        // decoder export. Empty metadata dict triggers the (now-correct) Int64 fallback.
        Assert.Equal(0L, targets.AsTensor<long>().Single());
        Assert.Equal(1L, targetLength.AsTensor<long>().Single());
    }

    [Fact]
    public void CreateNemotronDecoderInputs_resolves_hidden_major_encoded_layout()
    {
        // TRT RTX returned the bundled encoder's output as [1, 1024, 7] (hidden-major);
        // the decoder's encoder_outputs input expects [batch, hidden, time].
        MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
            .GetMethod("CreateNemotronDecoderInputs", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate Nemotron smoke decoder input builder.");
        var encoded = new DenseTensor<float>(new float[1024 * 7], [1, 1024, 7]);
        encoded[0, 5, 0] = 42f;

        using var inputSet = Assert.IsAssignableFrom<IDisposable>(method.Invoke(
            null,
            [
                new Dictionary<string, NodeMetadata>(),
                encoded
            ]));
        object values = inputSet.GetType().GetProperty("Values")!.GetValue(inputSet)!;
        IReadOnlyList<NamedOnnxValue> namedValues = Assert.IsAssignableFrom<IReadOnlyList<NamedOnnxValue>>(values);
        NamedOnnxValue encoderOutputs = namedValues.Single(static value => value.Name == "encoder_outputs");
        Tensor<float> frame = encoderOutputs.AsTensor<float>();

        Assert.Equal([1, 1024, 1], frame.Dimensions.ToArray());
        Assert.Equal(42f, frame[0, 5, 0]);
    }

    [Fact]
    public void CreateNemotronDecoderInputs_resolves_time_major_encoded_layout()
    {
        MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
            .GetMethod("CreateNemotronDecoderInputs", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate Nemotron smoke decoder input builder.");
        var encoded = new DenseTensor<float>(new float[7 * 1024], [1, 7, 1024]);
        encoded[0, 0, 5] = 42f;

        using var inputSet = Assert.IsAssignableFrom<IDisposable>(method.Invoke(
            null,
            [
                new Dictionary<string, NodeMetadata>(),
                encoded
            ]));
        object values = inputSet.GetType().GetProperty("Values")!.GetValue(inputSet)!;
        IReadOnlyList<NamedOnnxValue> namedValues = Assert.IsAssignableFrom<IReadOnlyList<NamedOnnxValue>>(values);
        NamedOnnxValue encoderOutputs = namedValues.Single(static value => value.Name == "encoder_outputs");
        Tensor<float> frame = encoderOutputs.AsTensor<float>();

        Assert.Equal([1, 1024, 1], frame.Dimensions.ToArray());
        Assert.Equal(42f, frame[0, 5, 0]);
    }

    [Theory]
    [InlineData(ExecutionProviderKind.Cpu, "cpu")]
    [InlineData(ExecutionProviderKind.DirectMl, "dml")]
    [InlineData(ExecutionProviderKind.TensorRTRtx, "tensorrt-rtx")]
    public void EnsureSelectedProviderMatchesRequested_accepts_matching_provider(
        ExecutionProviderKind requestedProvider,
        string selectedProvider)
    {
        MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
            .GetMethod("EnsureSelectedProviderMatchesRequested", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate selected-provider assertion helper.");

        method.Invoke(null, [requestedProvider, selectedProvider]);
    }

    [Fact]
    public void EnsureSelectedProviderMatchesRequested_rejects_tensorrt_fallback_to_directml()
    {
        MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
            .GetMethod("EnsureSelectedProviderMatchesRequested", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate selected-provider assertion helper.");

        TargetInvocationException exception = Assert.Throws<TargetInvocationException>(() =>
            method.Invoke(null, [ExecutionProviderKind.TensorRTRtx, "dml"]));
        InvalidOperationException inner = Assert.IsType<InvalidOperationException>(exception.InnerException);

        Assert.Contains("requested provider 'tensorrt-rtx'", inner.Message, StringComparison.Ordinal);
        Assert.Contains("effective provider 'dml'", inner.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ResolveChatterboxDecoderSmokeDimensions_uses_istft_safe_speech_token_length()
    {
        int[] speechTokens = OnnxExecutionProviderSmokeTester.ResolveChatterboxDecoderSmokeDimensionsForTesting(
            new OnnxExecutionProviderSmokeTester.ChatterboxSmokeInputSpec
            {
                InputName = "speech_tokens",
                ModelDimensions = [-1, -1],
            });
        int[] speakerFeatures = OnnxExecutionProviderSmokeTester.ResolveChatterboxDecoderSmokeDimensionsForTesting(
            new OnnxExecutionProviderSmokeTester.ChatterboxSmokeInputSpec
            {
                InputName = "speaker_features",
                ModelDimensions = [-1, -1, 80],
            });
        int[] speakerEmbeddings = OnnxExecutionProviderSmokeTester.ResolveChatterboxDecoderSmokeDimensionsForTesting(
            new OnnxExecutionProviderSmokeTester.ChatterboxSmokeInputSpec
            {
                InputName = "speaker_embeddings",
                ModelDimensions = [-1, 192],
            });

        Assert.Equal([1, 8], speechTokens);
        Assert.Equal([1, 8, 80], speakerFeatures);
        Assert.Equal([1, 192], speakerEmbeddings);
    }

    [Theory]
    [InlineData(ExecutionProviderKind.TensorRTRtx, "NvTensorRtRtx")]
    [InlineData(ExecutionProviderKind.DirectMl, "dml")]
    [InlineData(ExecutionProviderKind.Cpu, "cpu")]
    [InlineData(ExecutionProviderKind.Cuda, "cuda")]
    [InlineData(ExecutionProviderKind.CoreMl, "coreml")]
    [InlineData(ExecutionProviderKind.TensorRt, "NvTensorRtRtx")]
    [InlineData(ExecutionProviderKind.Migraphx, "AMDGPU")]
    [InlineData(ExecutionProviderKind.OpenVino, "OpenVINO")]
    [InlineData(ExecutionProviderKind.OpenVinoCatalog, "OpenVINO")]
    [InlineData(ExecutionProviderKind.Qnn, "QNN")]
    [InlineData(ExecutionProviderKind.VitisAi, "VitisAI")]
    [InlineData(ExecutionProviderKind.Dnnl, "cpu")]
    public void GenAiExecutionProviderNames_maps_trackdub_pins_to_ort_genai_names(
        ExecutionProviderKind provider,
        string expectedName)
    {
        Assert.Equal(expectedName, GenAiExecutionProviderNames.Resolve(provider));
    }

    [Theory]
    [InlineData("whisper-genai", true)]
    [InlineData("whisper-onnx", false)]
    [InlineData("qwen3-asr", false)]
    [InlineData(null, false)]
    public void UsesOrtGenAiModelLoad_routes_whisper_genai_away_from_inference_session(
        string? engineFamily,
        bool expected)
    {
        Assert.Equal(expected, OnnxExecutionProviderSmokeTester.UsesOrtGenAiModelLoad(engineFamily));
    }

    [Theory]
    [InlineData("phi-genai", true)]
    [InlineData("qwen-instruct", true)]
    [InlineData("opus-mt", false)]
    [InlineData("madlad", false)]
    [InlineData(null, false)]
    public void UsesOrtGenAiTranslationSmoke_routes_genai_families_away_from_opus_sessions(
        string? engineFamily,
        bool expected)
    {
        Assert.Equal(expected, OnnxExecutionProviderSmokeTester.UsesOrtGenAiTranslationSmoke(engineFamily));
    }

    [Fact]
    public async Task SmokeTestAsync_whisper_genai_looks_for_nested_genai_config()
    {
        using TempDirectoryFixture fixture = new();
        string nestedRoot = Path.Join(fixture.RootPath, "cpu-int4");
        Directory.CreateDirectory(nestedRoot);
        string nestedConfig = Path.Join(nestedRoot, "genai_config.json");
        var tester = new OnnxExecutionProviderSmokeTester();

        ExecutionProviderSmokeTestResult result = await tester.SmokeTestAsync(
            new ExecutionProviderSmokeTestRequest(
                RuntimeStage.Asr,
                "whisper-tiny-genai",
                "whisper-tiny-genai",
                "whisper-genai",
                "default",
                ExecutionProviderKind.Cpu,
                fixture.RootPath,
                nestedConfig));

        Assert.False(result.Passed);
        Assert.Contains("cpu-int4", result.Detail, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task SmokeTestAsync_whisper_genai_uses_genai_model_load_path_not_inference_session()
    {
        using TempDirectoryFixture fixture = new();

        // Seed the model root with an intentionally INVALID ONNX decoy (the literal bytes
        // "not-a-real-onnx-graph", which are not a parseable ONNX graph) at the model.onnx path
        // that the generic InferenceSession fallback (SmokeTestGenericSessionAsync) would try to
        // load, and leave out genai_config.json. If routing regressed to that fallback path the
        // failure Detail would reference the .onnx parse/load; because the correct GenAI
        // model-load path is taken instead, the failure Detail reports the missing
        // genai_config.json, which is what we assert here.
        string entryPath = Path.Join(fixture.RootPath, "model.onnx");
        File.WriteAllText(entryPath, "not-a-real-onnx-graph");
        var tester = new OnnxExecutionProviderSmokeTester();

        ExecutionProviderSmokeTestResult result = await tester.SmokeTestAsync(
            new ExecutionProviderSmokeTestRequest(
                RuntimeStage.Asr,
                "whisper-tiny-genai",
                "whisper-tiny-genai",
                "whisper-genai",
                "default",
                ExecutionProviderKind.Cpu,
                fixture.RootPath,
                entryPath));

        Assert.False(result.Passed);
        Assert.Contains("genai_config.json", result.Detail, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task SmokeTestAsync_phi_genai_looks_for_nested_genai_config()
    {
        using TempDirectoryFixture fixture = new();
        string nestedRoot = Path.Join(fixture.RootPath, "cpu_and_mobile", "cpu-int4");
        Directory.CreateDirectory(nestedRoot);
        string nestedConfig = Path.Join(nestedRoot, "genai_config.json");
        var tester = new OnnxExecutionProviderSmokeTester();

        ExecutionProviderSmokeTestResult result = await tester.SmokeTestAsync(
            new ExecutionProviderSmokeTestRequest(
                RuntimeStage.Translation,
                "phi-3.5-mini-instruct-genai",
                "phi-genai-pivot",
                "phi-genai",
                "cpu-int4",
                ExecutionProviderKind.Cpu,
                fixture.RootPath,
                nestedConfig));

        Assert.False(result.Passed);
        Assert.Contains("cpu-int4", result.Detail, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task SmokeTestAsync_opus_translation_failure_evicts_pooled_family()
    {
        using TempDirectoryFixture fixture = new();
        // Decoder must exist for path resolution; both files are invalid ONNX so the
        // pooled session create fails — the failure path under test.
        string encoderPath = Path.Join(fixture.RootPath, "encoder_model.onnx");
        File.WriteAllText(encoderPath, "not-a-real-onnx-graph");
        File.WriteAllText(Path.Join(fixture.RootPath, "decoder_model.onnx"), "not-a-real-onnx-graph");

        // Unique family so the shared pool is isolated from parallel tests; seed it
        // with an idle entry the failed smoke must evict rather than leave leasable.
        string family = "opus-evict-" + Guid.NewGuid().ToString("N");
        var seedKey = new SessionPoolKey(
            family, "seed-model", null, ExecutionProviderKind.Cpu, "seed-hash", 0, "default");
        int factoryCalls = 0;
        try
        {
            using (await InferenceSessionPool.Shared.GetLeaseAsync(
                seedKey,
                _ => { factoryCalls++; return Task.FromResult(InferenceSessionPoolTests.CreateMinimalSession()); },
                CancellationToken.None))
            {
            }

            Assert.Equal(1, factoryCalls);

            ExecutionProviderSmokeTestResult result = await new OnnxExecutionProviderSmokeTester()
                .SmokeTestAsync(
                    new ExecutionProviderSmokeTestRequest(
                        RuntimeStage.Translation,
                        "opus-evict-seed",
                        "opus-evict-seed",
                        family,
                        "default",
                        ExecutionProviderKind.Cpu,
                        fixture.RootPath,
                        encoderPath),
                    CancellationToken.None);

            Assert.False(result.Passed);

            // Seeded entry evicted: re-leasing the same key must call the factory again.
            using (await InferenceSessionPool.Shared.GetLeaseAsync(
                seedKey,
                _ => { factoryCalls++; return Task.FromResult(InferenceSessionPoolTests.CreateMinimalSession()); },
                CancellationToken.None))
            {
            }

            Assert.Equal(2, factoryCalls);
        }
        finally
        {
            await InferenceSessionPool.Shared.EvictModelAsync(family);
        }
    }

    private sealed class TempDirectoryFixture : IDisposable
    {
        public string RootPath { get; } = Path.Join(
            Path.GetTempPath(),
            "trackdub-tests",
            Guid.NewGuid().ToString("N"));

        public TempDirectoryFixture() => Directory.CreateDirectory(RootPath);

        public void Dispose()
        {
            try
            {
                if (Directory.Exists(RootPath))
                {
                    Directory.Delete(RootPath, recursive: true);
                }
            }
            catch (IOException)
            {
            }
            catch (UnauthorizedAccessException)
            {
            }
        }
    }
}
