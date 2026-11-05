using Trackdub.Contracts;
using Trackdub.Composition.NvidiaAfx;

namespace Trackdub.Composition.Tests;

public sealed class NvidiaAfxStudioVoiceProfileTests
{
    [Fact]
    public void StudioVoice_UsesOfflineHighQualityEffectWithSixSecondFrames()
    {
        NvidiaAfxProfileDefinition definition = NvidiaAfxProfileCatalog.GetDefinition(NvidiaAfxProfile.StudioVoice);

        Assert.Equal("studio_voice_high_quality", definition.Selector);
        Assert.Equal([16000, 48000], definition.SupportedSampleRates);
        Assert.Equal(6000, definition.InputFrameDurationMilliseconds);
        Assert.Equal(1, definition.ReadinessProbeFrameCount);
        Assert.False(definition.SupportsIntensityRatio);
        Assert.Contains(
            NvidiaAfxProfile.StudioVoice,
            NvidiaAfxProfileCatalog.SelectableDefinitions.Select(profile => profile.Profile));
        Assert.Equal(
            ["studio_voice_high_quality_16k", "studio_voice_high_quality_48k"],
            definition.AllRequiredModels.Select(model => model.ModelStem).OrderBy(stem => stem));
    }
}
