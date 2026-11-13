using System.Reflection;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Runtime.Planning;

namespace Trackdub.Inference.Tests;

public sealed partial class OnnxExecutionProviderSmokeTesterTests
{
    [Fact]
    public void EnsureSelectedProviderMatchesRequested_rejects_cpu_for_policy_selected_provider()
    {
        MethodInfo method = typeof(OnnxExecutionProviderSmokeTester)
            .GetMethod("EnsureSelectedProviderMatchesRequested", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Could not locate selected-provider assertion helper.");

        TargetInvocationException exception = Assert.Throws<TargetInvocationException>(() =>
            method.Invoke(null, [ExecutionProviderKind.Qnn, "cpu", true]));
        InvalidOperationException inner = Assert.IsType<InvalidOperationException>(exception.InnerException);

        Assert.Contains("requested provider 'qnn'", inner.Message, StringComparison.Ordinal);
        Assert.Contains("effective provider 'cpu'", inner.Message, StringComparison.Ordinal);
    }
}
