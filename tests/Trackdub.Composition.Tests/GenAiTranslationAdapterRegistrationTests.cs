using Microsoft.Extensions.DependencyInjection;
using Trackdub.Composition;
using Trackdub.Contracts;
using Trackdub.Inference.Onnx.Runtime.Routing;

namespace Trackdub.Composition.Tests;

public sealed class GenAiTranslationAdapterRegistrationTests
{
    [Fact]
    public void AddTrackdub_called_twice_does_not_duplicate_genai_translation_adapters()
    {
        var services = new ServiceCollection();
        services.AddTrackdub();
        services.AddTrackdub();

        using ServiceProvider provider = services.BuildServiceProvider();
        using IServiceScope scope = provider.CreateScope();
        ITranslationEngineAdapter[] adapters = scope.ServiceProvider.GetServices<ITranslationEngineAdapter>().ToArray();

        Assert.Single(adapters, adapter => adapter.EngineFamily == "qwen-genai");
        Assert.Single(adapters, adapter => adapter.EngineFamily == "gemma-genai");
    }
}
