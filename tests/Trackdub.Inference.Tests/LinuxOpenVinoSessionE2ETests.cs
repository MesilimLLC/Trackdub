using System.Runtime.Versioning;
using Trackdub.Domain;
using Trackdub.Inference.Onnx;

namespace Trackdub.Inference.Tests;

/// <summary>
/// Opt-in end-to-end check: a requested OpenVINO session on Linux runs on the LinML catalog's
/// plugin EP. Set TRACKDUB_LINML_OPENVINO_MODEL to a small ONNX model; the first run may download
/// the pinned plugin into LinML's default install root.
/// </summary>
[SupportedOSPlatform("linux")]
public sealed class LinuxOpenVinoSessionE2ETests
{
    private const string ModelVariable = "TRACKDUB_LINML_OPENVINO_MODEL";

    [Fact]
    public async Task OpenVino_session_runs_on_plugin_ep()
    {
        string? model = Environment.GetEnvironmentVariable(ModelVariable);
        if (string.IsNullOrEmpty(model))
        {
            return;
        }

        Assert.True(File.Exists(model), $"{ModelVariable} points at a missing file: {model}");

        using var lease = await OnnxExecutionSessionFactory.CreateSingleAsync(model, ExecutionProviderKind.OpenVino, CancellationToken.None);

        Assert.Equal("openvino", lease.SelectedProvider);
    }
}
