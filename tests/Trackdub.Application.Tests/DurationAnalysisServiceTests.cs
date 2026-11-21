using Trackdub.Application.Transcripts;
using Trackdub.Domain.Transcript;

namespace Trackdub.Application.Tests;

public sealed class DurationAnalysisServiceTests
{
    [Fact]
    public void Analyze_calculates_overrun_ratio()
    {
        var service = new DurationAnalysisService();
        TranscriptSegment segment = TranscriptSegment.Create(
            Guid.NewGuid(),
            segmentIndex: 0,
            startSeconds: 10.0d,
            endSeconds: 12.0d,
            text: "Hello");

        DurationAnalysisResult result = service.Analyze(segment, 2.3d);

        Assert.Equal(2.0d, result.OriginalDurationSeconds);
        Assert.Equal(0.15d, result.OverrunRatio!.Value, precision: 6);
        Assert.Equal(TtsDurationSeverity.Yellow, result.Severity);
        Assert.True(result.AutoStretchEligible);
    }

    [Fact]
    public void Analyze_flags_large_overrun_without_auto_stretch()
    {
        var service = new DurationAnalysisService();
        TranscriptSegment segment = TranscriptSegment.Create(
            Guid.NewGuid(),
            segmentIndex: 0,
            startSeconds: 0.0d,
            endSeconds: 2.0d,
            text: "Hello");

        DurationAnalysisResult result = service.Analyze(segment, 2.5d);

        Assert.Equal(0.25d, result.OverrunRatio!.Value, precision: 6);
        Assert.False(result.AutoStretchEligible);
        Assert.Equal(TtsDurationSeverity.Yellow, result.Severity);
    }

    [Fact]
    public void Analyze_flags_speed_limit_warning()
    {
        var service = new DurationAnalysisService();
        TranscriptSegment segment = TranscriptSegment.Create(
            Guid.NewGuid(),
            segmentIndex: 0,
            startSeconds: 0.0d,
            endSeconds: 2.0d,
            text: "Hello");

        DurationAnalysisResult result = service.Analyze(segment, 3.2d);

        Assert.True(result.HasSpeedLimitWarning);
        Assert.Equal(TtsDurationSeverity.Red, result.Severity);
    }

    [Fact]
    public void Within_the_limit_the_take_is_fitted_exactly()
    {
        DurationAnalysisResult result = new DurationAnalysisService().Analyze(Segment(0d, 2d), 2.3d);

        Assert.Equal(1.15d, result.AutoStretchTempoRatio!.Value, precision: 6);
    }

    [Fact]
    public void Over_the_limit_the_take_is_left_alone_by_default()
    {
        DurationAnalysisResult result = new DurationAnalysisService().Analyze(Segment(26.75d, 40.15d), 19.47d);

        Assert.Null(result.AutoStretchTempoRatio);
    }

    [Fact]
    public void Over_the_limit_the_take_is_compressed_by_the_limit_when_enabled()
    {
        // The POC Chinese take: 19.47 s for a 13.40 s segment, 45% over the default 20% limit.
        DurationAnalysisResult result = new DurationAnalysisService().Analyze(Segment(26.75d, 40.15d), 19.47d, ToLimit);

        Assert.False(result.AutoStretchEligible);
        Assert.Equal(TtsDurationSeverity.Red, result.Severity);
        Assert.Equal(1.20d, result.AutoStretchTempoRatio!.Value, precision: 6);
    }

    [Fact]
    public void Over_the_limit_uses_the_configured_limit()
    {
        TtsTimingOptions options = ToLimit with { AutoStretchMaxOverrun = 0.35d };

        DurationAnalysisResult result = new DurationAnalysisService().Analyze(Segment(0d, 10d), 15d, options);

        Assert.Equal(1.35d, result.AutoStretchTempoRatio!.Value, precision: 6);
    }

    [Theory]
    [InlineData(1.8d)] // Shorter than the segment: nothing to compress.
    [InlineData(2.0d)] // Exact fit.
    public void Takes_that_fit_are_not_stretched(double ttsSeconds)
    {
        Assert.Null(new DurationAnalysisService().Analyze(Segment(0d, 2d), ttsSeconds).AutoStretchTempoRatio);
    }

    [Fact]
    public void Zero_limit_disables_automatic_stretching()
    {
        TtsTimingOptions options = ToLimit with { AutoStretchMaxOverrun = 0d };

        Assert.Null(new DurationAnalysisService().Analyze(Segment(0d, 2d), 3d, options).AutoStretchTempoRatio);
    }

    [Fact]
    public void Unstretchable_takes_are_not_stretched()
    {
        // Beyond the 4x tempo bound the take is not stretchable at all.
        Assert.Null(new DurationAnalysisService().Analyze(Segment(0d, 1d), 5d, ToLimit).AutoStretchTempoRatio);
    }

    private static readonly TtsTimingOptions ToLimit = TtsTimingOptions.Default with { AutoStretchToLimit = true };

    private static TranscriptSegment Segment(double start, double end) =>
        TranscriptSegment.Create(Guid.NewGuid(), segmentIndex: 0, startSeconds: start, endSeconds: end, text: "Hello");
}
