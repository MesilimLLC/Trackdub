using Microsoft.ML.OnnxRuntime;
using Trackdub.Contracts.ApplicationContracts;
using Trackdub.Domain;

namespace Trackdub.Inference.Onnx.Tests.WindowsMl;

#if WINDOWS
public sealed class WindowsMlEfficiencyPolicyTests
{
    private const string IdentityModelBase64 =
        "CAc6OgoQCgF4EgF5IghJZGVudGl0eRIEdGVzdFoPCgF4EgoKCAgBEgQKAggBYg8KAXkSCgoICAESBAoCCAFCBAoAEAk=";

    [Theory]
    [InlineData(WindowsMlExecutionDevicePolicy.MaxEfficiency, true)]
    [InlineData(WindowsMlExecutionDevicePolicy.MinOverallPower, true)]
    [InlineData(WindowsMlExecutionDevicePolicy.MinPower, true)]
    [InlineData(WindowsMlExecutionDevicePolicy.Explicit, false)]
    [InlineData(WindowsMlExecutionDevicePolicy.MaxPerformance, false)]
    [InlineData(WindowsMlExecutionDevicePolicy.PreferNpu, false)]
    [InlineData(WindowsMlExecutionDevicePolicy.DefaultRender, false)]
    public void RequestsEfficientWorkload_matches_efficiency_flavored_policies(
        WindowsMlExecutionDevicePolicy policy, bool expected)
    {
        Assert.Equal(expected, Trackdub.Inference.Onnx.WindowsMl.WindowsMlEfficiencyPolicy
            .RequestsEfficientWorkload(policy));
    }

    [Theory]
    [InlineData(ExecutionProviderKind.Qnn, true)]
    [InlineData(ExecutionProviderKind.VitisAi, true)]
    [InlineData(ExecutionProviderKind.OpenVinoCatalog, true)]
    [InlineData(ExecutionProviderKind.Migraphx, false)]   // WinML catalog EP, but a GPU EP
    [InlineData(ExecutionProviderKind.DirectMl, false)]
    [InlineData(ExecutionProviderKind.Cpu, false)]
    [InlineData(ExecutionProviderKind.TensorRTRtx, false)]
    public void IsNpuCatalogProvider_matches_only_npu_catalog_providers(
        ExecutionProviderKind provider, bool expected)
    {
        Assert.Equal(expected, Trackdub.Inference.Onnx.WindowsMl.WindowsMlEfficiencyPolicy
            .IsNpuCatalogProvider(provider));
    }

    [Fact]
    public void ApplyToSession_applies_for_npu_provider_under_efficiency_policy()
    {
        using var session = new InferenceSession(Convert.FromBase64String(IdentityModelBase64));

        (bool applied, string? error) = Trackdub.Inference.Onnx.WindowsMl.WindowsMlEfficiencyPolicy
            .ApplyToSession(session, ExecutionProviderKind.Qnn, WindowsMlExecutionDevicePolicy.MinPower);

        // ORT accepts the option on any session (only NPU EPs act on it), so the interop
        // result is the assertion target — this proves the gating called through.
        Assert.True(applied);
        Assert.Null(error);
    }

    [Fact]
    public void ApplyToSession_skips_for_cpu_provider_even_under_efficiency_policy()
    {
        using var session = new InferenceSession(Convert.FromBase64String(IdentityModelBase64));

        (bool applied, string? error) = Trackdub.Inference.Onnx.WindowsMl.WindowsMlEfficiencyPolicy
            .ApplyToSession(session, ExecutionProviderKind.Cpu, WindowsMlExecutionDevicePolicy.MinPower);

        Assert.False(applied);
        Assert.Null(error); // skip is not a failure — nothing to report
    }

    [Fact]
    public void ApplyToSession_skips_for_npu_provider_under_explicit_policy()
    {
        using var session = new InferenceSession(Convert.FromBase64String(IdentityModelBase64));

        (bool applied, string? error) = Trackdub.Inference.Onnx.WindowsMl.WindowsMlEfficiencyPolicy
            .ApplyToSession(session, ExecutionProviderKind.Qnn, WindowsMlExecutionDevicePolicy.Explicit);

        Assert.False(applied);
        Assert.Null(error);
    }

    [Fact]
    public void CreateSession_device_policy_parameter_is_required_not_defaulted()
    {
        // The efficiency hook at the session-creation chokepoint must not be silently
        // droppable: CreateSession takes SessionCreateRequest, whose DevicePolicy property
        // is `required`, so every call site must thread a policy explicitly and a new site
        // cannot silently fall back to Explicit. (Review finding: this test's predecessor
        // asserted nothing about the factory — 13 pooled/dual call sites had silently
        // defaulted — hence this pin.)
        System.Reflection.MethodInfo? createSession = typeof(OnnxExecutionSessionFactory).GetMethod(
            "CreateSession",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        Assert.NotNull(createSession);

        System.Reflection.ParameterInfo? requestParameter = createSession.GetParameters()
            .SingleOrDefault(p => p.ParameterType == typeof(OnnxExecutionSessionFactory.SessionCreateRequest));
        Assert.NotNull(requestParameter);

        System.Reflection.PropertyInfo? policyProperty =
            typeof(OnnxExecutionSessionFactory.SessionCreateRequest).GetProperty("DevicePolicy");
        Assert.NotNull(policyProperty);
        Assert.Equal(typeof(WindowsMlExecutionDevicePolicy), policyProperty.PropertyType);
        Assert.NotNull(System.Attribute.GetCustomAttribute(
            policyProperty, typeof(System.Runtime.CompilerServices.RequiredMemberAttribute)));
    }
}
#endif
