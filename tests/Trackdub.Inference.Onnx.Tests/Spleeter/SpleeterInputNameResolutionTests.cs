using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Trackdub.Inference.Onnx.Tests.Spleeter;

/// <summary>
/// Verifies that InferenceSession.InputMetadata exposes a model's declared input name,
/// which is the API contract that SpleeterOnnxSeparator relies on after the fix for
/// the "Input name: 'input' is not in the metadata" runtime crash.
///
/// SpleeterOnnxSeparator creates sessions via OnnxExecutionSessionFactory (no injection seam),
/// so these tests operate on InferenceSession directly from raw ONNX bytes to prove the
/// contract without requiring real model files on disk.
/// </summary>
public sealed class SpleeterInputNameResolutionTests
{
    /// <summary>
    /// Minimal ONNX identity model with one input named "mix_wave" (not "input").
    ///
    /// Hand-encoded ModelProto protobuf:
    ///   ir_version: 8
    ///   opset_import: { domain: "", version: 17 }
    ///   graph: {
    ///     node:   Identity(mix_wave → output_0)
    ///     input:  ValueInfo{ name: "mix_wave",  type: tensor(float32) }
    ///     output: ValueInfo{ name: "output_0",  type: tensor(float32) }
    ///   }
    /// </summary>
    internal static readonly byte[] MinimalOnnxIdentityModel =
    [
        0x08, 0x08,                                                           // ir_version: 8
        0x42, 0x04, 0x0A, 0x00, 0x10, 0x11,                                 // opset: domain="", version=17
        0x3A, 0x44,                                                           // graph (68 bytes)
        0x0A, 0x1E,                                                           //   node (30 bytes)
        0x0A, 0x08, 0x6D, 0x69, 0x78, 0x5F, 0x77, 0x61, 0x76, 0x65,       //     input:   "mix_wave"
        0x12, 0x08, 0x6F, 0x75, 0x74, 0x70, 0x75, 0x74, 0x5F, 0x30,       //     output:  "output_0"
        0x22, 0x08, 0x49, 0x64, 0x65, 0x6E, 0x74, 0x69, 0x74, 0x79,       //     op_type: "Identity"
        0x5A, 0x10,                                                           //   graph input[0] (16 bytes)
        0x0A, 0x08, 0x6D, 0x69, 0x78, 0x5F, 0x77, 0x61, 0x76, 0x65,       //     name: "mix_wave"
        0x12, 0x04, 0x0A, 0x02, 0x08, 0x01,                                 //     type: tensor(float32)
        0x62, 0x10,                                                           //   graph output[0] (16 bytes)
        0x0A, 0x08, 0x6F, 0x75, 0x74, 0x70, 0x75, 0x74, 0x5F, 0x30,       //     name: "output_0"
        0x12, 0x04, 0x0A, 0x02, 0x08, 0x01,                                 //     type: tensor(float32)
    ];

    [Fact]
    public void InputMetadata_reflects_model_declared_input_name_not_hardcoded_input()
    {
        // This is the contract SpleeterOnnxSeparator relies on after the fix.
        // session.InputMetadata.Keys.First() must return "mix_wave", not "input".
        using var session = new InferenceSession(MinimalOnnxIdentityModel);

        Assert.True(session.InputMetadata.ContainsKey("mix_wave"),
            "InputMetadata must contain the model's declared input name 'mix_wave'.");
        Assert.False(session.InputMetadata.ContainsKey("input"),
            "InputMetadata must not contain a hardcoded name 'input' that the model doesn't declare.");
    }

    [Fact]
    public void Running_with_declared_input_name_from_InputMetadata_succeeds()
    {
        using var session = new InferenceSession(MinimalOnnxIdentityModel);

        // Fix: use the model's actual input name
        string inputName = session.InputMetadata.Keys.First();
        var tensor = new DenseTensor<float>(new float[] { 1.0f, 2.0f, 3.0f }, new int[] { 3 });

        using var outputs = session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, tensor) });

        Assert.NotEmpty(outputs);
    }

    [Fact]
    public void Running_with_hardcoded_input_name_throws_when_model_uses_different_name()
    {
        // Regression: before the fix, SpleeterOnnxSeparator called
        //   NamedOnnxValue.CreateFromTensor("input", inputTensor)
        // which crashes when the bundled model's input node is not named "input".
        using var session = new InferenceSession(MinimalOnnxIdentityModel);

        var tensor = new DenseTensor<float>(new float[] { 1.0f }, new int[] { 1 });

        Assert.Throws<OnnxRuntimeException>(() =>
            session.Run(new[] { NamedOnnxValue.CreateFromTensor("input", tensor) }));
    }
}
