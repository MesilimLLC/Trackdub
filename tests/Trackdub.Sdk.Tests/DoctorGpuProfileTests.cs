using Trackdub.Cli.Handlers;

using DoctorCheckRow = Trackdub.Cli.Handlers.DoctorHandler.DoctorCheckRow;

namespace Trackdub.Sdk.Tests;

public sealed class DoctorGpuProfileTests
{
    private static readonly DoctorCheckRow[] s_rows =
    [
        Row("playback-natives", "warn"),
        Row("tensorrt-rtx-plugin", "warn"),
        Row("execution-provider-migraphx", "warn"),
        Row("execution-provider-openvino-catalog", "warn"),
        Row("accelerator-vram-budget", "warn"),
        Row("ffmpeg-tools", "pass"),
    ];

    [Theory]
    [InlineData("server")]
    [InlineData(" Server ")]
    public void ServerProfile_ReportsUnusedComponentWarningsAsPassing(string profile)
    {
        List<DoctorCheckRow> result = DoctorHandler.ApplyGpuProfile(s_rows, profile);

        Assert.Equal(s_rows.Select(row => row.Id), result.Select(row => row.Id));
        foreach (string id in new[]
                 {
                     "playback-natives", "tensorrt-rtx-plugin", "execution-provider-migraphx",
                     "execution-provider-openvino-catalog",
                 })
        {
            DoctorCheckRow row = result.Single(check => check.Id == id);
            Assert.Equal("pass", row.Status);
            Assert.StartsWith("Not used by the server GPU profile", row.Message, StringComparison.Ordinal);
            Assert.EndsWith($"{id} detail", row.Message, StringComparison.Ordinal);
            Assert.Null(row.Remediation);
        }

        Assert.Same(s_rows[4], result.Single(check => check.Id == "accelerator-vram-budget"));
        Assert.Same(s_rows[5], result.Single(check => check.Id == "ffmpeg-tools"));
    }

    [Fact]
    public void ServerProfile_NeverHidesAFailingRow()
    {
        DoctorCheckRow failing = Row("tensorrt-rtx-plugin", "fail");

        List<DoctorCheckRow> result = DoctorHandler.ApplyGpuProfile([failing], "server");

        Assert.Same(failing, Assert.Single(result));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("rtx")]
    public void OtherProfiles_ReturnRowsUnchanged(string? profile)
    {
        List<DoctorCheckRow> result = DoctorHandler.ApplyGpuProfile(s_rows, profile);

        Assert.Equal(s_rows, result);
    }

    private static DoctorCheckRow Row(string id, string status) => new()
    {
        Id = id,
        Status = status,
        Message = $"{id} detail",
        Remediation = $"{id} remediation",
    };
}
