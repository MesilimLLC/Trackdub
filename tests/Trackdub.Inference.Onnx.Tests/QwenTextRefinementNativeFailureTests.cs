using System.Runtime.CompilerServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntimeGenAI;
using Trackdub.Inference.Onnx.QwenTextRefinement;

namespace Trackdub.Inference.Onnx.Tests;

public sealed class QwenTextRefinementNativeFailureTests
{
    // The native exception types have no public constructors, so create instances without running one.
    public static TheoryData<Exception> NativeFailures => new()
    {
        Uninitialized<OnnxRuntimeGenAIException>(),
        Uninitialized<OnnxRuntimeException>(),
    };

    public static TheoryData<Exception> NonNativeFailures => new()
    {
        new InvalidOperationException("already handled by the stage"),
        new FileNotFoundException("missing genai_config.json"),
        new OperationCanceledException(),
        new IOException("disk error"),
    };

    [Theory]
    [MemberData(nameof(NativeFailures))]
    public void Native_runtime_exceptions_are_classified_as_native_failures(Exception exception) =>
        Assert.True(QwenTextRefinementEngine.IsNativeRuntimeFailure(exception));

    [Theory]
    [MemberData(nameof(NonNativeFailures))]
    public void Other_exceptions_pass_through_unclassified(Exception exception) =>
        Assert.False(QwenTextRefinementEngine.IsNativeRuntimeFailure(exception));

    [Fact]
    public void Wrapped_native_failure_is_an_InvalidOperationException_that_keeps_the_cause()
    {
        OnnxRuntimeGenAIException native = Uninitialized<OnnxRuntimeGenAIException>();

        InvalidOperationException wrapped = QwenTextRefinementEngine.WrapNativeFailure(native);

        Assert.Same(native, wrapped.InnerException);
        Assert.StartsWith("Qwen text refinement failed in ONNX Runtime", wrapped.Message, StringComparison.Ordinal);
    }

    private static T Uninitialized<T>() where T : Exception =>
        (T)RuntimeHelpers.GetUninitializedObject(typeof(T));
}
