using Trackdub.Application.Editing;
using Trackdub.Composition;
using Trackdub.Contracts.Editing;
using Trackdub.Media.Muxing;
using Microsoft.Extensions.DependencyInjection;

namespace Trackdub.Composition.Tests;

public sealed class SequenceExportCompositionTests
{
    [Fact]
    public void AddTrackdub_registers_sequence_export_delivery_services()
    {
        var services = new ServiceCollection();
        services.AddTrackdub();
        using ServiceProvider provider = services.BuildServiceProvider(new ServiceProviderOptions
        {
            ValidateOnBuild = true,
            ValidateScopes = true
        });

        Assert.IsType<FfmpegSequenceRenderer>(provider.GetRequiredService<ISequenceExportRenderer>());
        using IServiceScope scope = provider.CreateScope();
        Assert.IsType<SequenceExportDeliveryService>(
            scope.ServiceProvider.GetRequiredService<SequenceExportDeliveryService>());
    }
}
