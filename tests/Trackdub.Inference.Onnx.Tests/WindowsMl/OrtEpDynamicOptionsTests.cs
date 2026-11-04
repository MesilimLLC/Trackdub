using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Trackdub.Inference.Onnx.Tests.WindowsMl;

// Windows TFM only: the WindowsMl folder is compiled out of the net11.0 leg, and the
// interop targets the Windows ML ORT binding.
#if WINDOWS
public sealed class OrtEpDynamicOptionsTests
{
    // Minimal identity ONNX model (ir_version=7, opset 9, Identity op) — the same bytes as
    // InferenceSessionPoolTests.BuildIdentityOnnxModel (tests/Trackdub.Inference.Tests).
    private const string IdentityModelBase64 =
        "CAc6OgoQCgF4EgF5IghJZGVudGl0eRIEdGVzdFoPCgF4EgoKCAgBEgQKAggBYg8KAXkSCgoICAESBAoCCAFCBAoAEAk=";

    private static InferenceSession CreateIdentitySession() =>
        new(Convert.FromBase64String(IdentityModelBase64));

    [Fact]
    public void Reflection_surface_is_available_on_the_pinned_binding()
    {
        // Guards against silent breakage when the WinML package pin moves: a bump that drops
        // or renames NativeMethods.api_ / OrtApi.SetEpDynamicOptions must fail loudly here.
        Assert.True(Trackdub.Inference.Onnx.WindowsMl.OrtEpDynamicOptions.IsAvailable);
    }

    [Fact]
    public void TrySetWorkloadType_Efficient_succeeds_and_session_stays_healthy()
    {
        using InferenceSession session = CreateIdentitySession();

        bool applied = Trackdub.Inference.Onnx.WindowsMl.OrtEpDynamicOptions.TrySetWorkloadType(
            session, Trackdub.Inference.Onnx.WindowsMl.OrtEpDynamicOptions.EfficientWorkload,
            out string? error);

        Assert.True(applied);
        Assert.Null(error);

        // The session must still run inference after the dynamic-option call.
        string inputName = session.InputMetadata.Keys.Single();
        var tensor = new DenseTensor<float>(new[] { 1 }, false);
        using var results = session.Run(
            [NamedOnnxValue.CreateFromTensor(inputName, tensor)]);
        Assert.Single(results);
    }

    [Fact]
    public void TrySetWorkloadType_missing_binding_returns_false_without_throwing()
    {
        using InferenceSession session = CreateIdentitySession();

        // Disposed session cannot be a false positive source here: the handle read is guarded.
        session.Dispose();
        bool applied = Trackdub.Inference.Onnx.WindowsMl.OrtEpDynamicOptions.TrySetWorkloadType(
            session, "Efficient", out string? error);

        Assert.False(applied);
        Assert.NotNull(error);
    }
}
#endif
