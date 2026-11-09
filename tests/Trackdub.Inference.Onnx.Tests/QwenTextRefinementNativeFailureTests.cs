using System.Runtime.CompilerServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntimeGenAI;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Onnx.QwenTextRefinement;

namespace Trackdub.Inference.Onnx.Tests;

public sealed class QwenTextRefinementNativeFailureTests
{
    // The native exception types have no public constructors, so create instances without running one.
    public static TheoryData<Exception> NativeFailures => new()
    {
        Uninitialized<OnnxRuntimeGenAIException>(),
        Uninitialized<OnnxRuntimeException>(),
        new DllNotFoundException("onnxruntime-genai.dll"),
        new EntryPointNotFoundException("OgaCreateModel"),
        new BadImageFormatException("wrong architecture"),
        new TypeInitializationException("Microsoft.ML.OnnxRuntimeGenAI.NativeMethods", new DllNotFoundException("onnxruntime-genai.dll")),
    };

    public static TheoryData<Exception> NonNativeFailures => new()
    {
        new InvalidOperationException("already handled by the stage"),
        new FileNotFoundException("missing genai_config.json"),
        new OperationCanceledException(),
        new IOException("disk error"),
        new TypeInitializationException("Some.Other.Type", new InvalidOperationException("not native")),
        new TypeInitializationException("Some.Other.Type", null),
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

    private static readonly TextRefinementRequest Request =
        new([new TextRefinementInputSegment(0, 0, 1, "hello")]);

    [Theory]
    [MemberData(nameof(NativeFailures))]
    public async Task Native_failure_acquiring_the_model_lease_surfaces_as_InvalidOperationException(Exception native)
    {
        InvalidOperationException thrown = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            QwenTextRefinementEngine.RefineWithLeaseAsync(
                Request, ExecutionProviderKind.Cpu, _ => throw native, CancellationToken.None));

        Assert.Same(native, thrown.InnerException);
    }

    [Fact]
    public async Task Cancellation_acquiring_the_model_lease_is_not_wrapped() =>
        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            QwenTextRefinementEngine.RefineWithLeaseAsync(
                Request,
                ExecutionProviderKind.Cpu,
                _ => throw new OperationCanceledException(),
                CancellationToken.None));

    [Fact]
    public async Task Non_native_failure_acquiring_the_model_lease_is_not_wrapped()
    {
        var io = new IOException("disk error");

        IOException thrown = await Assert.ThrowsAsync<IOException>(() =>
            QwenTextRefinementEngine.RefineWithLeaseAsync(
                Request, ExecutionProviderKind.Cpu, _ => throw io, CancellationToken.None));

        Assert.Same(io, thrown);
    }

    private static T Uninitialized<T>() where T : Exception =>
        (T)RuntimeHelpers.GetUninitializedObject(typeof(T));
}
