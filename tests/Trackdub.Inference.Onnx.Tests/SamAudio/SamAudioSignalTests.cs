using Trackdub.Inference.Onnx.SamAudio;

namespace Trackdub.Inference.Onnx.Tests.SamAudio;

public sealed class SamAudioSignalTests
{
    [Theory]
    [InlineData(100, 1000, 200, new[] { 0 })]
    [InlineData(1000, 1000, 200, new[] { 0 })]
    [InlineData(2000, 1000, 200, new[] { 0, 800, 1000 })]
    [InlineData(1800, 1000, 200, new[] { 0, 800 })]
    public void WindowStarts_cover_the_input_and_end_exactly_at_the_end(int length, int window, int overlap, int[] expected)
    {
        int[] starts = new WindowLayout(length, window, overlap).Starts();

        Assert.Equal(expected, starts);
        Assert.True(starts[^1] + Math.Min(window, length) == length || starts.Length == 1);
    }

    [Fact]
    public void WindowStarts_rejects_an_overlap_that_does_not_advance()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new WindowLayout(5000, 1000, 1000).Starts());
    }

    [Fact]
    public void CrossfadeJoin_of_identical_overlapping_pieces_reproduces_the_signal()
    {
        int length = 2000;
        float[] signal = [.. Enumerable.Range(0, length).Select(i => (float)Math.Sin(i * 0.01))];
        WindowLayout layout = new(length, 1000, 200);
        List<float[]> pieces = [.. layout.Starts().Select(s => signal.AsSpan(s, 1000).ToArray())];

        float[] joined = SamAudioCrossfade.Join(pieces, layout);

        for (int i = 0; i < length; i++)
        {
            Assert.Equal(signal[i], joined[i], 1e-5f);
        }
    }

    [Fact]
    public void CrossfadeJoin_ramps_between_windows_that_disagree()
    {
        float[] first = [.. Enumerable.Repeat(0f, 1000)];
        float[] second = [.. Enumerable.Repeat(1f, 1000)];

        float[] joined = SamAudioCrossfade.Join([first, second], new WindowLayout(1800, 1000, 200));

        Assert.Equal(0f, joined[799], 1e-6f);
        Assert.Equal(1f, joined[1000], 1e-6f);
        Assert.InRange(joined[900], 0.4f, 0.6f);
    }

    [Fact]
    public void Midpoint_solver_is_exact_for_constant_velocity()
    {
        float[] result = SamAudioOdeSolver.Midpoint([1f, 2f], 16, (_, x) => [3f, -1f]);

        Assert.Equal(4f, result[0], 1e-5f);
        Assert.Equal(1f, result[1], 1e-5f);
    }

    [Fact]
    public void Midpoint_solver_integrates_exponential_growth_to_second_order()
    {
        float[] result = SamAudioOdeSolver.Midpoint([1f], 16, (_, x) => [x[0]]);

        Assert.Equal(Math.E, result[0], 2e-3);
    }

    [Fact]
    public void Midpoint_solver_evaluates_the_velocity_twice_per_step_at_the_right_times()
    {
        List<float> times = [];

        SamAudioOdeSolver.Midpoint([0f], 4, (t, _) =>
        {
            times.Add(t);
            return [0f];
        });

        Assert.Equal([0f, 0.125f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f], times);
    }

    [Theory]
    [InlineData(1920, 1920)]
    [InlineData(1921, 3840)]
    [InlineData(5000, 5760)]
    public void PadToHop_pads_to_a_multiple_of_the_codec_hop(int length, int expected)
    {
        float[] padded = SamAudioPadding.PadToHop(new float[length]);

        Assert.Equal(expected, padded.Length);
        Assert.Equal(expected / 1920, SamAudioPadding.FrameCount(padded.Length));
    }

    [Fact]
    public void PadToHop_reflects_the_end_of_the_signal()
    {
        float[] samples = [.. Enumerable.Range(0, 1915).Select(i => (float)i)];

        float[] padded = SamAudioPadding.PadToHop(samples);

        Assert.Equal(1913f, padded[1915]);
        Assert.Equal(1909f, padded[1919]);
    }

    [Fact]
    public void PadToHop_zero_pads_input_too_short_to_reflect()
    {
        float[] padded = SamAudioPadding.PadToHop([1f, 2f, 3f]);

        Assert.Equal(1920, padded.Length);
        Assert.All(padded.Skip(3), sample => Assert.Equal(0f, sample));
    }

    [Fact]
    public void Random_is_reproducible_for_a_seed_and_standard_normal()
    {
        float[] a = new SamAudioRandom(7).Gaussian(20000);
        float[] b = new SamAudioRandom(7).Gaussian(20000);
        double mean = a.Average();
        double variance = a.Select(v => (v - mean) * (v - mean)).Average();

        Assert.Equal(a, b);
        Assert.InRange(mean, -0.05, 0.05);
        Assert.InRange(variance, 0.95, 1.05);
    }

    [Fact]
    public void Watermark_message_is_two_rows_of_sixteen_bits()
    {
        float[] message = new SamAudioRandom(1).WatermarkMessage();

        Assert.Equal(32, message.Length);
        Assert.All(message, bit => Assert.True(bit is 0f or 1f));
    }
}
