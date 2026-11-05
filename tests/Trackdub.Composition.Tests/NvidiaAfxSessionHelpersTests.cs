using Trackdub.Composition.NvidiaAfx;

namespace Trackdub.Composition.Tests;

public sealed class NvidiaAfxSessionHelpersTests
{
    [Fact]
    public void AlignFarEndToNearEnd_ZeroPadsShorterFarEnd()
    {
        float[] near = [1f, 2f, 3f, 4f, 5f];
        float[] far = [9f, 8f];

        float[] aligned = NvidiaAfxSession.AlignFarEndToNearEnd(near, far);

        Assert.Equal(near.Length, aligned.Length);
        Assert.Equal(9f, aligned[0]);
        Assert.Equal(8f, aligned[1]);
        Assert.Equal(0f, aligned[2]);
        Assert.Equal(0f, aligned[3]);
        Assert.Equal(0f, aligned[4]);
    }

    [Fact]
    public void AlignFarEndToNearEnd_TruncatesLongerFarEnd()
    {
        float[] near = [1f, 2f];
        float[] far = [9f, 8f, 7f, 6f];

        float[] aligned = NvidiaAfxSession.AlignFarEndToNearEnd(near, far);

        Assert.Equal([9f, 8f], aligned);
    }

    [Fact]
    public void ValidateOutputFrameRatio_AcceptsExpectedTelephonyRatio()
    {
        NvidiaAfxSession.ValidateOutputFrameRatio(
            "superres8kto16k_denoiser16k",
            inputSampleRate: 8000,
            outputSampleRate: 16000,
            numInputSamples: 480,
            numOutputSamples: 960);
    }

    [Fact]
    public void ValidateOutputFrameRatio_RejectsMismatchedRateChangingFrames()
    {
        InvalidOperationException ex = Assert.Throws<InvalidOperationException>(() =>
            NvidiaAfxSession.ValidateOutputFrameRatio(
                "superres8kto16k_denoiser16k",
                inputSampleRate: 8000,
                outputSampleRate: 16000,
                numInputSamples: 480,
                numOutputSamples: 480));

        Assert.Contains("does not match", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ComputeTrimmedOutputLength_SameRate_MatchesNearSampleCount()
    {
        Assert.Equal(100, NvidiaAfxSession.ComputeTrimmedOutputLength(100, inputFrame: 480, outputFrame: 480));
    }

    [Fact]
    public void ComputeTrimmedOutputLength_RateChanging_AppliesFrameRatio()
    {
        // 8 kHz → 16 kHz telephony: 100 input samples → 200 output samples (ratio 960/480).
        Assert.Equal(200, NvidiaAfxSession.ComputeTrimmedOutputLength(100, inputFrame: 480, outputFrame: 960));
    }
}
