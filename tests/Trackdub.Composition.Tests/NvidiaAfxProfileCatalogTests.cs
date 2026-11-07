using Trackdub.Contracts;
using Trackdub.Composition.NvidiaAfx;

namespace Trackdub.Composition.Tests;

public sealed class NvidiaAfxProfileCatalogTests
{
    [Fact]
    public void Definitions_ContainAllShippingProfiles()
    {
        var expected = new[]
        {
            NvidiaAfxProfile.NoiseOnly,
            NvidiaAfxProfile.ReverbOnly,
            NvidiaAfxProfile.NoiseAndReverb,
            NvidiaAfxProfile.TelephonyUpscale,
            NvidiaAfxProfile.AcousticEchoCancellation,
            NvidiaAfxProfile.SpeakerFocus,
            NvidiaAfxProfile.StudioVoice
        };

        foreach (NvidiaAfxProfile profile in expected)
        {
            NvidiaAfxProfileDefinition definition = NvidiaAfxProfileCatalog.GetDefinition(profile);
            Assert.Equal(profile, definition.Profile);
            Assert.False(string.IsNullOrWhiteSpace(definition.Selector));
            Assert.NotEmpty(definition.SupportedSampleRates);
            Assert.NotEmpty(definition.ModelsBySampleRate);
            Assert.NotEmpty(definition.AllRequiredModels);
            Assert.NotEmpty(definition.PreferredModelRelativePaths);
            Assert.Equal(
                definition.SupportedSampleRates.OrderBy(rate => rate),
                definition.ModelsBySampleRate.Select(entry => entry.InputSampleRate).OrderBy(rate => rate));
        }
    }

    [Fact]
    public void NoiseAndReverb_DefaultsToDereverbDenoiserSelector()
    {
        NvidiaAfxProfileDefinition definition = NvidiaAfxProfileCatalog.GetDefinition(NvidiaAfxProfile.NoiseAndReverb);
        Assert.Equal("dereverb_denoiser", definition.Selector);
        Assert.False(definition.RequiresFarEndReference);
    }

    [Fact]
    public void NoiseOnly_ResolvesRateSpecificMaxineModels()
    {
        NvidiaAfxProfileDefinition definition = NvidiaAfxProfileCatalog.GetDefinition(NvidiaAfxProfile.NoiseOnly);
        Assert.Equal(48000, definition.PreferredProbeSampleRate);

        NvidiaAfxRequiredModel[] models16k = definition.ResolveRequiredModels(16000);
        Assert.Single(models16k);
        Assert.Equal("nvafxdenoiser", models16k[0].FeatureFolder);
        Assert.Equal("denoiser_16k", models16k[0].ModelStem);

        NvidiaAfxRequiredModel[] models48k = definition.ResolveRequiredModels(48000);
        Assert.Single(models48k);
        Assert.Equal("denoiser_48k", models48k[0].ModelStem);

        Assert.Equal(2, definition.AllRequiredModels.Count);
    }

    [Fact]
    public void AcousticEchoCancellation_RequiresFarEndReference()
    {
        NvidiaAfxProfileDefinition definition =
            NvidiaAfxProfileCatalog.GetDefinition(NvidiaAfxProfile.AcousticEchoCancellation);
        Assert.Equal("aec", definition.Selector);
        Assert.True(definition.RequiresFarEndReference);
        Assert.False(definition.SupportsIntensityRatio);
    }

    [Fact]
    public void TelephonyUpscale_UsesMaxineChainedSelector()
    {
        NvidiaAfxProfileDefinition definition =
            NvidiaAfxProfileCatalog.GetDefinition(NvidiaAfxProfile.TelephonyUpscale);
        Assert.Equal("superres8kto16k_denoiser16k", definition.Selector);
        Assert.True(definition.IsChainedEffect);
        Assert.Contains(8000, definition.SupportedSampleRates);
        Assert.Equal(16000, definition.OutputSampleRate);
        Assert.Equal(16000, definition.ResolveOutputSampleRate(8000));
    }

    [Fact]
    public void TelephonyUpscale_DoesNotAdvertiseIntensity_BecauseChainedIntensityHasNoEffect()
    {
        NvidiaAfxProfileDefinition definition =
            NvidiaAfxProfileCatalog.GetDefinition(NvidiaAfxProfile.TelephonyUpscale);

        Assert.False(definition.SupportsIntensityRatio);
    }

    [Fact]
    public void SpeakerFocus_IsASingleNonChainedEffectWithBothRateModels()
    {
        NvidiaAfxProfileDefinition definition =
            NvidiaAfxProfileCatalog.GetDefinition(NvidiaAfxProfile.SpeakerFocus);

        Assert.Equal(NvidiaAfxProfile.SpeakerFocus, definition.Profile);
        Assert.Equal("speaker_focus", definition.Selector);
        Assert.False(definition.IsChainedEffect);
        Assert.False(definition.RequiresFarEndReference);
        Assert.Equal([16000, 48000], definition.SupportedSampleRates);
        Assert.Equal("speaker_focus_16k", Assert.Single(definition.ResolveRequiredModels(16000)).ModelStem);
        Assert.Equal("speaker_focus_48k", Assert.Single(definition.ResolveRequiredModels(48000)).ModelStem);
        Assert.Equal(["nvafxspeakerfocus"], definition.RequiredFeatureFolders);
    }

    [Fact]
    public void SpeakerFocus_IsEarlyAccess_AndHiddenFromSelectionUnlessAllowed()
    {
        Assert.True(NvidiaAfxProfileCatalog.GetDefinition(NvidiaAfxProfile.SpeakerFocus).IsEarlyAccess);
        Assert.DoesNotContain(
            NvidiaAfxProfileCatalog.Definitions.Where(definition => definition.Profile != NvidiaAfxProfile.SpeakerFocus),
            definition => definition.IsEarlyAccess);

        if (!NvidiaAfxIntegration.AllowEarlyAccessEffects())
        {
            Assert.DoesNotContain(
                NvidiaAfxProfileCatalog.SelectableDefinitions,
                definition => definition.Profile == NvidiaAfxProfile.SpeakerFocus);
        }
    }

    [Fact]
    public void SpeakerFocus_DoesNotAdvertiseIntensity_BecauseTheSdkIgnoresIt()
    {
        Assert.False(NvidiaAfxProfileCatalog.GetDefinition(NvidiaAfxProfile.SpeakerFocus).SupportsIntensityRatio);
    }

    [Fact]
    public void ProfileEnumValues_AreStable_BecauseSettingsPersistThemAsNumbers()
    {
        Assert.Equal(0, (int)NvidiaAfxProfile.NoiseOnly);
        Assert.Equal(1, (int)NvidiaAfxProfile.ReverbOnly);
        Assert.Equal(2, (int)NvidiaAfxProfile.NoiseAndReverb);
        Assert.Equal(3, (int)NvidiaAfxProfile.TelephonyUpscale);
        Assert.Equal(4, (int)NvidiaAfxProfile.AcousticEchoCancellation);
        Assert.Equal(5, (int)NvidiaAfxProfile.SpeakerFocus);
        Assert.Equal(6, (int)NvidiaAfxProfile.StudioVoice);
    }

    [Fact]
    public void TelephonyUpscale_UsesTheShippedSuperresModelStem()
    {
        NvidiaAfxProfileDefinition definition =
            NvidiaAfxProfileCatalog.GetDefinition(NvidiaAfxProfile.TelephonyUpscale);

        Assert.Contains(
            definition.ResolveRequiredModels(8000),
            model => model.FeatureFolder == "nvafxsuperres" && model.ModelStem == "superres_8kto16k");
    }
}
