using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Domain;
using Trackdub.Inference.Onnx;
using Trackdub.Inference.Onnx.NemotronAsr;

namespace Trackdub.Inference.Onnx.Tests;

public sealed class NemotronAsrEncodedTensorLayoutTests
{
    private const int HiddenDim = 1024;

    [Theory]
    [InlineData(ExecutionProviderKind.Cpu, true, true, false)]
    [InlineData(ExecutionProviderKind.DirectMl, true, true, false)]
    [InlineData(ExecutionProviderKind.TensorRTRtx, false, true, false)]
    [InlineData(ExecutionProviderKind.TensorRTRtx, true, false, false)]
    [InlineData(ExecutionProviderKind.TensorRTRtx, true, true, true)]
    public void Device_bound_runs_require_trt_rtx_and_both_shared_allocators(
        ExecutionProviderKind provider,
        bool encoderHasDeviceAllocator,
        bool decoderHasDeviceAllocator,
        bool expected) =>
        Assert.Equal(expected, NemotronAsrGreedyDecoder.ShouldUseDeviceBoundRuns(
            provider, encoderHasDeviceAllocator, decoderHasDeviceAllocator));

    [Fact]
    public void Decode_uses_managed_sessions_on_cpu()
    {
        string fixtureDirectory = Path.Join(AppContext.BaseDirectory, "Fixtures", "NemotronAsr");
        var lease = new OnnxExecutionSessionFactory.NemotronAsrSessionLease(
            new InferenceSession(Path.Join(fixtureDirectory, "managed-encoder.onnx")),
            new InferenceSession(Path.Join(fixtureDirectory, "managed-decoder.onnx")),
            "cpu",
            ProviderNames.FormatProviderLabel(ExecutionProviderKind.Cpu),
            null);
        using (lease)
        {
            var vocab = new NemotronAsrSentencePieceVocab(["piece"]);
            var exportConfig = new NemotronAsrExportConfig(1, 1, "NA", HasPromptInput: false);
            var decoder = new NemotronAsrGreedyDecoder(lease, vocab, exportConfig);

            IReadOnlyList<int> tokens = decoder.Decode(
                new float[128, 1], promptIndex: 0, TestContext.Current.CancellationToken);

            Assert.Empty(tokens);
            Assert.Equal(1, decoder.LastEncodedLength);
        }
    }

    [Theory]
    [InlineData(1, 7, 1024, true)]
    [InlineData(1, 1024, 7, false)]
    public void ResolveTimeMajor_matches_hidden_dim(int batch, int dim1, int dim2, bool expectedTimeMajor)
    {
        bool timeMajor = NemotronAsrEncodedTensorLayout.ResolveTimeMajor([batch, dim1, dim2], HiddenDim);
        Assert.Equal(expectedTimeMajor, timeMajor);
    }

    [Fact]
    public void Resolve_uses_encoded_length_and_time_axis()
    {
        NemotronAsrEncodedTensorLayout.EncodedLayout layout = NemotronAsrEncodedTensorLayout.Resolve(
            [1, 1024, 12],
            encodedLength: 7,
            HiddenDim);

        Assert.False(layout.TimeMajor);
        Assert.Equal(7, layout.AvailableFrames);
        Assert.Equal(HiddenDim, layout.HiddenSize);
    }

    [Fact]
    public void SliceFrame_channel_major_produces_decoder_input_shape()
    {
        var encoded = new DenseTensor<float>(new float[HiddenDim * 3], [1, HiddenDim, 3]);
        encoded[0, 42, 2] = 3.5f;

        NemotronAsrEncodedTensorLayout.EncodedLayout layout = NemotronAsrEncodedTensorLayout.Resolve(
            encoded.Dimensions,
            encodedLength: 3,
            HiddenDim);
        DenseTensor<float> frame = NemotronAsrEncodedTensorLayout.SliceFrame(encoded, layout, frameIndex: 2);

        Assert.Equal(new[] { 1, HiddenDim, 1 }, frame.Dimensions.ToArray());
        Assert.Equal(3.5f, frame[0, 42, 0]);
    }

    [Fact]
    public void SliceFrame_time_major_produces_decoder_input_shape()
    {
        var encoded = new DenseTensor<float>(new float[HiddenDim * 3], [1, 3, HiddenDim]);
        encoded[0, 2, 42] = 2.25f;

        NemotronAsrEncodedTensorLayout.EncodedLayout layout = NemotronAsrEncodedTensorLayout.Resolve(
            encoded.Dimensions,
            encodedLength: 3,
            HiddenDim);
        DenseTensor<float> frame = NemotronAsrEncodedTensorLayout.SliceFrame(encoded, layout, frameIndex: 2);

        Assert.Equal(new[] { 1, HiddenDim, 1 }, frame.Dimensions.ToArray());
        Assert.Equal(2.25f, frame[0, 42, 0]);
    }
}
