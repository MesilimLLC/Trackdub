using Trackdub.Application.Transcripts;

namespace Trackdub.Application.Tests;

public sealed class SpokenFillerFilterTests
{
    [Fact]
    public void Removes_fillers_from_the_poc_clip_text()
    {
        const string text = "It's hard to put one particular thing uh out there. I I would just say that, you know, " +
                            "we have done simulations, uh actual simulations, launch simulations in simulators, uh in mock-ups. " +
                            "Uh we've been in both sets of space suits multiple times. Um and";

        string stripped = SpokenFillerFilter.StripForTranslation(text, "en");

        Assert.Equal(
            "It's hard to put one particular thing out there. I I would just say that, you know, " +
            "we have done simulations, actual simulations, launch simulations in simulators, in mock-ups. " +
            "we've been in both sets of space suits multiple times. and",
            stripped);
    }

    [Theory]
    [InlineData("Um, so we left.", "So we left.")]
    [InlineData("We left, uh, early.", "We left, early.")]
    [InlineData("Hmm. Okay then.", "Okay then.")]
    [InlineData("Errm we waited, ummm, and then left.", "We waited, and then left.")]
    public void Handles_commas_and_sentence_starts(string text, string expected)
    {
        Assert.Equal(expected, SpokenFillerFilter.StripForTranslation(text, "en-US"));
    }

    [Theory]
    [InlineData("The umbrella is here.")]
    [InlineData("Ahead of the hum, Erma ran.")]
    [InlineData("Ummah and Ahmed met at the summit.")]
    public void Keeps_words_that_only_contain_filler_letters(string text)
    {
        Assert.Equal(text, SpokenFillerFilter.StripForTranslation(text, "en"));
    }

    [Fact]
    public void Leaves_non_english_text_alone()
    {
        Assert.Equal("eh, um, bueno", SpokenFillerFilter.StripForTranslation("eh, um, bueno", "es"));
        Assert.Equal("uh huh", SpokenFillerFilter.StripForTranslation("uh huh", null));
    }

    [Fact]
    public void Keeps_the_original_when_only_fillers_remain()
    {
        Assert.Equal("Uh, um.", SpokenFillerFilter.StripForTranslation("Uh, um.", "en"));
    }
}
