using Trackdub.Application.Transcripts;

namespace Trackdub.Application.Tests;

public sealed class TtsTimingResumeIdentityTests
{
    [Fact]
    public void Normalize_falls_back_to_default_when_value_is_null()
    {
        Assert.Equal(TtsTimingResumeIdentity.Normalize(0.20d), TtsTimingResumeIdentity.Normalize(null));
    }

    [Fact]
    public void Normalize_clamps_out_of_range_values_to_default()
    {
        Assert.Equal(TtsTimingResumeIdentity.Normalize(0.20d), TtsTimingResumeIdentity.Normalize(2.0d));
        Assert.Equal(TtsTimingResumeIdentity.Normalize(0.20d), TtsTimingResumeIdentity.Normalize(-0.1d));
    }

    [Theory]
    [InlineData(0.20d, 0.20d, true)]
    [InlineData(0.20d, 0.10d, false)]
    public void Append_and_Matches_round_trip_the_timing_identity(double storedOverrun, double expectedOverrun, bool matches)
    {
        string fingerprint = TtsTimingResumeIdentity.Append("abcdef", storedOverrun);

        Assert.True(TtsTimingResumeIdentity.Matches(fingerprint, TtsTimingResumeIdentity.Normalize(expectedOverrun)) == matches);
    }

    [Fact]
    public void WithoutSuffix_returns_the_bare_hash_for_suffixed_fingerprints()
    {
        const string baseHash = "0123456789abcdef";
        string suffixed = TtsTimingResumeIdentity.Append(baseHash, 0.20d);

        Assert.Equal(baseHash, TtsTimingResumeIdentity.WithoutSuffix(suffixed));
        Assert.NotEqual(suffixed, TtsTimingResumeIdentity.WithoutSuffix(suffixed));
    }

    [Fact]
    public void WithoutSuffix_is_identity_for_legacy_fingerprints()
    {
        Assert.Equal("abcdef", TtsTimingResumeIdentity.WithoutSuffix("abcdef"));
    }

    [Fact]
    public void Matches_returns_false_for_legacy_fingerprints()
    {
        Assert.False(TtsTimingResumeIdentity.Matches("abcdef", TtsTimingResumeIdentity.Normalize(0.20d)));
        Assert.False(TtsTimingResumeIdentity.Matches(null, TtsTimingResumeIdentity.Normalize(0.20d)));
    }

    [Fact]
    public void Snapshot_key_is_stable()
    {
        Assert.Equal("TtsTiming.AutoStretchMaxOverrun", TtsTimingResumeIdentity.SnapshotKey);
    }

    [Fact]
    public void Auto_stretch_to_limit_changes_the_identity_only_when_on()
    {
        Assert.Equal(TtsTimingResumeIdentity.Normalize(0.20d), TtsTimingResumeIdentity.Normalize(0.20d, autoStretchToLimit: false));
        Assert.NotEqual(TtsTimingResumeIdentity.Normalize(0.20d), TtsTimingResumeIdentity.Normalize(0.20d, autoStretchToLimit: true));
    }

    [Theory]
    [InlineData(false, true)]
    [InlineData(true, false)]
    public void Resume_rejects_a_take_made_under_the_other_auto_stretch_policy(bool takeToLimit, bool runToLimit)
    {
        string fingerprint = TtsTimingResumeIdentity.Append("abcdef", 0.20d, takeToLimit);

        Assert.False(TtsTimingResumeIdentity.Matches(fingerprint, TtsTimingResumeIdentity.Normalize(0.20d, runToLimit)));
        Assert.True(TtsTimingResumeIdentity.Matches(fingerprint, TtsTimingResumeIdentity.Normalize(0.20d, takeToLimit)));
        Assert.Equal("abcdef", TtsTimingResumeIdentity.WithoutSuffix(fingerprint));
    }
}
