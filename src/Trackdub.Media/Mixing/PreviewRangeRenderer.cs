using System.Buffers;
using Trackdub.Contracts;
using Trackdub.Domain.Mixing;
using Trackdub.Media.Waveforms;

namespace Trackdub.Media.Mixing;

public sealed class PreviewRangeRenderer(IArtifactStore artifactStore) : IPreviewRangeRenderer
{
    private const double PanAnalysisWindowSeconds = 0.2d;
    private const double PanSilenceRmsThreshold = 0.00001d;
    private const float DownmixCenterGain = 0.70710677f;
    private const float DownmixSurroundGain = 0.70710677f;
    private const uint SpeakerFrontLeft = 0x1u;
    private const uint SpeakerFrontRight = 0x2u;
    private const uint SpeakerFrontCenter = 0x4u;
    private const uint SpeakerLowFrequency = 0x8u;
    private const uint SpeakerBackLeft = 0x10u;
    private const uint SpeakerBackRight = 0x20u;
    private const uint SpeakerFrontLeftOfCenter = 0x40u;
    private const uint SpeakerFrontRightOfCenter = 0x80u;
    private const uint SpeakerBackCenter = 0x100u;
    private const uint SpeakerSideLeft = 0x200u;
    private const uint SpeakerSideRight = 0x400u;
    private const uint SpeakerTopCenter = 0x800u;
    private const uint SpeakerTopFrontLeft = 0x1000u;
    private const uint SpeakerTopFrontCenter = 0x2000u;
    private const uint SpeakerTopFrontRight = 0x4000u;
    private const uint SpeakerTopBackLeft = 0x8000u;
    private const uint SpeakerTopBackCenter = 0x10000u;
    private const uint SpeakerTopBackRight = 0x20000u;
    private static readonly PanGains MonoPanGains = new(1f, 1f);
    private static readonly uint[] ChannelMaskSpeakerOrder =
    [
        SpeakerFrontLeft,
        SpeakerFrontRight,
        SpeakerFrontCenter,
        SpeakerLowFrequency,
        SpeakerBackLeft,
        SpeakerBackRight,
        SpeakerFrontLeftOfCenter,
        SpeakerFrontRightOfCenter,
        SpeakerBackCenter,
        SpeakerSideLeft,
        SpeakerSideRight,
        SpeakerTopCenter,
        SpeakerTopFrontLeft,
        SpeakerTopFrontCenter,
        SpeakerTopFrontRight,
        SpeakerTopBackLeft,
        SpeakerTopBackCenter,
        SpeakerTopBackRight
    ];

    private readonly IArtifactStore artifactStore = artifactStore ?? throw new ArgumentNullException(nameof(artifactStore));

    public async Task<PreviewRangeRenderResult> RenderAsync(
        PreviewRangeRenderRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(request.MixPlan);
        ArgumentException.ThrowIfNullOrWhiteSpace(request.OutputPath);
        if (!double.IsFinite(request.StartSeconds) || request.StartSeconds < 0d)
        {
            throw new ArgumentOutOfRangeException(nameof(request.StartSeconds), "Preview range start must be finite and non-negative.");
        }

        if (!double.IsFinite(request.EndSeconds) || request.EndSeconds <= request.StartSeconds)
        {
            throw new ArgumentOutOfRangeException(nameof(request.EndSeconds), "Preview range end must be greater than the start.");
        }

        string sourcePath = artifactStore.GetPath(request.MixPlan.SourceAudioRelativePath);
        if (!File.Exists(sourcePath))
        {
            throw new FileNotFoundException("Mix source audio file was not found.", sourcePath);
        }

        WavePcm16Info sourceInfo = await WavePcm16.ReadInfoAsync(sourcePath, cancellationToken).ConfigureAwait(false);
        EffectivePreviewRange range = ResolveEffectiveRange(request, sourceInfo);
        WavePcm16Samples source = await WavePcm16
            .ReadSamplesAsync(sourcePath, sourceInfo, request.StartSeconds, range.DurationSeconds, cancellationToken)
            .ConfigureAwait(false);
        int sampleRate = range.SampleRate;
        int outputFrameCount = range.SampleCount;
        int outputChannelCount = ResolveOutputChannelCount(request.MixPlan, sourceInfo);
        int outputSampleCount = checked(outputFrameCount * outputChannelCount);
        float[] output = ArrayPool<float>.Shared.Rent(outputSampleCount);
        float[] duckingGains = ArrayPool<float>.Shared.Rent(outputFrameCount);

        try
        {
            FillDuckingGains(duckingGains.AsSpan(0, outputFrameCount), request.MixPlan, request.StartSeconds, sampleRate);
            float sourceGain = DecibelsToLinear(request.MixPlan.SourceGainDb);

            for (int frame = 0; frame < outputFrameCount; frame++)
            {
                float frameGain = sourceGain * duckingGains[frame];
                int outputOffset = frame * outputChannelCount;
                if (outputChannelCount == 1)
                {
                    output[outputOffset] = MapSourceFrameToMono(source, frame) * frameGain;
                    continue;
                }

                (float left, float right) = MapSourceFrameToStereo(source, frame);
                output[outputOffset] = left * frameGain;
                output[outputOffset + 1] = right * frameGain;
            }

            float dubGain = DecibelsToLinear(request.MixPlan.DubbedSpeechGainDb);
            var panGainCache = new Dictionary<PanCacheKey, PanGains>();
            var panAnalysisContext = new PanAnalysisContext(
                NormalizePanCachePath(request.MixPlan.OriginalMixAudioRelativePath));
            foreach (MixSpeechClip clip in request.MixPlan.SpeechClips.Where(static clip => !clip.IsSilentGap))
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!ClipOverlapsRange(clip, request.StartSeconds, range.EndSeconds))
                {
                    continue;
                }

                if (string.IsNullOrWhiteSpace(clip.TakeRelativePath))
                {
                    throw new InvalidOperationException(
                        $"Mix plan contains an audible dubbed clip for segment {clip.SegmentIndex} without a take path.");
                }

                string takePath = artifactStore.GetPath(clip.TakeRelativePath);
                if (!File.Exists(takePath))
                {
                    throw new FileNotFoundException(
                        $"Dubbed speech take file was not found for segment {clip.SegmentIndex}.",
                        takePath);
                }

                WaveMonoSamples take = await WavePcm16
                    .ReadAllMonoSamplesAsync(takePath, cancellationToken)
                    .ConfigureAwait(false);
                float[] takeSamples = take.SampleRate == sampleRate
                    ? take.Samples
                    : ResampleLinear(take.Samples, take.SampleRate, sampleRate);
                float[] envelopeMatched = request.MixPlan.ApplyTimbrePolish
                    ? await TryApplySpectralEnvelopeAsync(takeSamples, clip, request.MixPlan.VocalStemRelativePath, sampleRate, cancellationToken)
                    : takeSamples;
                float[] reverbedSamples = request.MixPlan.ApplyTimbrePolish
                    ? await TryApplyRoomToneReverbAsync(envelopeMatched, clip, sourcePath, cancellationToken)
                    : envelopeMatched;
                PanGains panGains = await ResolveClipPanGainsAsync(
                    request.MixPlan,
                    clip,
                    outputChannelCount,
                    panGainCache,
                    panAnalysisContext,
                    cancellationToken).ConfigureAwait(false);
                MixTakeIntoOutput(
                    output,
                    outputFrameCount,
                    outputChannelCount,
                    reverbedSamples,
                    clip.StartSeconds,
                    request.StartSeconds,
                    sampleRate,
                    dubGain,
                    panGains);
            }

            foreach (MixMediaClip clip in request.MixPlan.MediaClips)
            {
                await MixPlacedMediaAsync(clip, request.StartSeconds, range.EndSeconds,
                    output, outputFrameCount, outputChannelCount, sampleRate, cancellationToken).ConfigureAwait(false);
            }

            await WavePcm16.WriteSamplesAsync(
                    request.OutputPath,
                    new ArraySegment<float>(output, 0, outputSampleCount),
                    sampleRate,
                    outputChannelCount,
                    normalizePeak: true,
                    cancellationToken)
                .ConfigureAwait(false);
            return new PreviewRangeRenderResult(
                request.OutputPath,
                outputFrameCount / (double)sampleRate,
                sampleRate,
                outputChannelCount);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(duckingGains, clearArray: false);
            output.AsSpan(0, outputSampleCount).Clear();
            ArrayPool<float>.Shared.Return(output, clearArray: false);
        }
    }

    private async Task MixPlacedMediaAsync(
        MixMediaClip clip, double rangeStart, double rangeEnd,
        float[] output, int frameCount, int channelCount, int sampleRate, CancellationToken cancellationToken)
    {
        float gain = DecibelsToLinear(clip.GainDb);
        if (!double.IsFinite(clip.StartSeconds) || clip.StartSeconds < 0d ||
            !double.IsFinite(clip.DurationSeconds) || clip.DurationSeconds <= 0d)
            throw new InvalidOperationException("Timeline mix clip has invalid timing.");
        double start = Math.Max(rangeStart, clip.StartSeconds);
        double end = Math.Min(rangeEnd, clip.StartSeconds + clip.DurationSeconds);
        if (end <= start) return;
        string path = artifactStore.GetPath(clip.AudioRelativePath);
        if (!File.Exists(path)) throw new FileNotFoundException("Prepared timeline audio is missing.", path);
        WavePcm16Info info = await WavePcm16.ReadInfoAsync(path, cancellationToken).ConfigureAwait(false);
        if (end > clip.StartSeconds + info.DurationSeconds + 0.01d)
        {
            throw new InvalidOperationException(
                "Timeline clip audio is shorter than its placed slot and would be cut off.");
        }

        double sourceOriginSeconds = start - clip.StartSeconds;
        double exactSourceFrames = sourceOriginSeconds * info.SampleRate;
        double fractionalSourceFrame = exactSourceFrames - Math.Floor(exactSourceFrames);
        double readDurationSeconds = (end - start) + (1d / info.SampleRate);
        WavePcm16Samples samples = await WavePcm16.ReadSamplesAsync(
            path, info, sourceOriginSeconds, readDurationSeconds, cancellationToken).ConfigureAwait(false);
        int firstOutputFrame = (int)Math.Round((start - rangeStart) * sampleRate);
        int framesToMix = Math.Min(frameCount - firstOutputFrame, (int)Math.Ceiling((end - start) * sampleRate));
        for (int frame = 0; frame < framesToMix; frame++)
        {
            if ((frame & 4095) == 0) cancellationToken.ThrowIfCancellationRequested();
            double sourceFrame = fractionalSourceFrame + (frame * (double)info.SampleRate / sampleRate);
            int lastSourceFrame = Math.Max(0, samples.FrameCount - 1);
            int index = Math.Clamp((int)Math.Floor(sourceFrame), 0, lastSourceFrame);
            int nextIndex = Math.Min(lastSourceFrame, index + 1);
            float fraction = (float)(sourceFrame - index);
            int outputIndex = (firstOutputFrame + frame) * channelCount;
            if (channelCount == 1)
            {
                float first = MapSourceFrameToMono(samples, index);
                float next = MapSourceFrameToMono(samples, nextIndex);
                output[outputIndex] += (first + ((next - first) * fraction)) * gain;
            }
            else
            {
                (float Left, float Right) first = MapSourceFrameToStereo(samples, index);
                (float Left, float Right) next = MapSourceFrameToStereo(samples, nextIndex);
                output[outputIndex] += (first.Left + ((next.Left - first.Left) * fraction)) * gain;
                output[outputIndex + 1] += (first.Right + ((next.Right - first.Right) * fraction)) * gain;
            }
        }
    }

    private async Task<float[]> TryApplySpectralEnvelopeAsync(
        float[] dryTake,
        MixSpeechClip clip,
        string? vocalStemRelativePath,
        int sampleRate,
        CancellationToken cancellationToken)
    {
        if (vocalStemRelativePath is null)
            return dryTake;
        try
        {
            string vocalPath = artifactStore.GetPath(vocalStemRelativePath);
            if (!File.Exists(vocalPath))
                return dryTake;
            double segmentDuration = clip.EndSeconds - clip.StartSeconds;
            if (segmentDuration <= 0d)
                return dryTake;
            WaveMonoSamples refSamples = await WavePcm16
                .ReadMonoSamplesAsync(vocalPath, clip.StartSeconds, segmentDuration, cancellationToken)
                .ConfigureAwait(false);
            float[] refResampled = refSamples.SampleRate == sampleRate
                ? refSamples.Samples
                : ResampleLinear(refSamples.Samples, refSamples.SampleRate, sampleRate);
            return SpectralEnvelopeMatcher.TryApply(dryTake, refResampled, new StftProcessor()) ?? dryTake;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            return dryTake;
        }
    }

    private async Task<float[]> TryApplyRoomToneReverbAsync(
        float[] samples,
        MixSpeechClip clip,
        string sourcePath,
        CancellationToken cancellationToken)
    {
        try
        {
            const double PreRollWindowSeconds = 0.3;
            double preRollEnd = clip.StartSeconds;
            double preRollStart = Math.Max(0d, preRollEnd - PreRollWindowSeconds);
            double preRollDuration = preRollEnd - preRollStart;

            WaveMonoSamples preRoll = await WavePcm16
                .ReadMonoSamplesAsync(sourcePath, preRollStart, preRollDuration, cancellationToken)
                .ConfigureAwait(false);

            return RoomToneConvolver.TryApply(samples, preRoll.Samples) ?? samples;
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch
        {
            return samples;
        }
    }

    private static EffectivePreviewRange ResolveEffectiveRange(
        PreviewRangeRenderRequest request,
        WavePcm16Info sourceInfo)
    {
        double sourceDurationSeconds = sourceInfo.DurationSeconds;
        if (!double.IsFinite(sourceDurationSeconds) || sourceDurationSeconds <= 0d)
        {
            throw new InvalidOperationException("Mix source audio file has no usable duration.");
        }

        if (request.StartSeconds >= sourceDurationSeconds)
        {
            throw new ArgumentOutOfRangeException(
                nameof(request.StartSeconds),
                "Preview range start must be within the source audio duration.");
        }

        double effectiveEndSeconds = Math.Min(request.EndSeconds, sourceDurationSeconds);
        double effectiveDurationSeconds = effectiveEndSeconds - request.StartSeconds;
        if (!double.IsFinite(effectiveDurationSeconds) || effectiveDurationSeconds <= 0d)
        {
            throw new ArgumentOutOfRangeException(
                nameof(request.EndSeconds),
                "Preview range end must be within the source audio duration and greater than the start.");
        }

        double sampleCount = Math.Ceiling(effectiveDurationSeconds * sourceInfo.SampleRate);
        if (!double.IsFinite(sampleCount) || sampleCount <= 0d || sampleCount > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(request.EndSeconds),
                "Preview range is too long to render in one preview.");
        }

        int outputSampleCount = Math.Max(1, (int)sampleCount);
        double outputDurationSeconds = outputSampleCount / (double)sourceInfo.SampleRate;
        return new EffectivePreviewRange(
            sourceInfo.SampleRate,
            outputSampleCount,
            outputDurationSeconds,
            request.StartSeconds + outputDurationSeconds);
    }

    private static void FillDuckingGains(
        Span<float> gains,
        MixPlan mixPlan,
        double rangeStartSeconds,
        int sampleRate)
    {
        gains.Fill(1f);
        foreach (MixDuckRegion region in mixPlan.DuckingRegions)
        {
            float regionGain = DecibelsToLinear(region.GainDb);
            int startSample = Math.Clamp(
                (int)Math.Floor((region.StartSeconds - rangeStartSeconds) * sampleRate),
                0,
                gains.Length);
            int endSample = Math.Clamp(
                (int)Math.Ceiling((region.EndSeconds - rangeStartSeconds) * sampleRate),
                0,
                gains.Length);
            for (int index = startSample; index < endSample; index++)
            {
                gains[index] = Math.Min(gains[index], regionGain);
            }
        }
    }

    private static bool ClipOverlapsRange(MixSpeechClip clip, double rangeStartSeconds, double rangeEndSeconds)
    {
        double clipEndSeconds = ResolveClipEndSeconds(clip);
        return clip.StartSeconds < rangeEndSeconds && clipEndSeconds > rangeStartSeconds;
    }

    private static double ResolveClipEndSeconds(MixSpeechClip clip)
    {
        if (clip.TakeDurationSeconds is double durationSeconds &&
            double.IsFinite(durationSeconds) &&
            durationSeconds > 0d)
        {
            return clip.StartSeconds + durationSeconds;
        }

        return clip.EndSeconds;
    }

    private static void MixTakeIntoOutput(
        float[] output,
        int outputFrameCount,
        int outputChannelCount,
        float[] takeSamples,
        double clipStartSeconds,
        double rangeStartSeconds,
        int sampleRate,
        float gain,
        PanGains panGains)
    {
        int outputStart = (int)Math.Round((clipStartSeconds - rangeStartSeconds) * sampleRate);
        int takeStart = outputStart < 0 ? -outputStart : 0;
        int mixStart = Math.Max(0, outputStart);
        int availableOutputFrames = outputFrameCount - mixStart;
        int availableTakeFrames = takeSamples.Length - takeStart;
        int framesToMix = Math.Min(availableOutputFrames, availableTakeFrames);
        if (framesToMix <= 0)
        {
            return;
        }

        for (int i = 0; i < framesToMix; i++)
        {
            float sample = takeSamples[takeStart + i] * gain;
            int outputOffset = (mixStart + i) * outputChannelCount;
            if (outputChannelCount == 1)
            {
                output[outputOffset] += sample;
                continue;
            }

            output[outputOffset] += sample * panGains.Left;
            output[outputOffset + 1] += sample * panGains.Right;
        }
    }

    private async Task<PanGains> ResolveClipPanGainsAsync(
        MixPlan mixPlan,
        MixSpeechClip clip,
        int outputChannelCount,
        IDictionary<PanCacheKey, PanGains> panGainCache,
        PanAnalysisContext panAnalysisContext,
        CancellationToken cancellationToken)
    {
        if (outputChannelCount == 1)
        {
            return MonoPanGains;
        }

        if (!mixPlan.RestoreOriginalPan)
        {
            return BuildConstantPowerPan(balance: 0d);
        }

        var cacheKey = new PanCacheKey(
            panAnalysisContext.NormalizedOriginalMixAudioRelativePath,
            clip.SegmentId,
            outputChannelCount);
        if (panGainCache.TryGetValue(cacheKey, out PanGains cachedGains))
        {
            return cachedGains;
        }

        PanGains resolvedGains;
        try
        {
            string originalPath = panAnalysisContext.OriginalMixPath ??= artifactStore.GetPath(mixPlan.OriginalMixAudioRelativePath);
            if (!File.Exists(originalPath))
            {
                resolvedGains = BuildConstantPowerPan(balance: 0d);
                panGainCache[cacheKey] = resolvedGains;
                return resolvedGains;
            }

            double durationSeconds = ResolveClipEndSeconds(clip) - clip.StartSeconds;
            if (!double.IsFinite(durationSeconds) || durationSeconds <= 0d)
            {
                resolvedGains = BuildConstantPowerPan(balance: 0d);
                panGainCache[cacheKey] = resolvedGains;
                return resolvedGains;
            }

            WavePcm16Info originalInfo = panAnalysisContext.OriginalMixInfo ??=
                await WavePcm16.ReadInfoAsync(originalPath, cancellationToken).ConfigureAwait(false);
            double analysisDurationSeconds = Math.Min(durationSeconds, PanAnalysisWindowSeconds);
            double analysisStartSeconds = clip.StartSeconds + ((durationSeconds - analysisDurationSeconds) / 2d);
            WavePcm16Samples original = await WavePcm16
                .ReadSamplesAsync(originalPath, originalInfo, analysisStartSeconds, analysisDurationSeconds, cancellationToken)
                .ConfigureAwait(false);
            if (original.FrameCount == 0)
            {
                resolvedGains = BuildConstantPowerPan(balance: 0d);
                panGainCache[cacheKey] = resolvedGains;
                return resolvedGains;
            }

            double leftSquareSum = 0d;
            double rightSquareSum = 0d;
            for (int frame = 0; frame < original.FrameCount; frame++)
            {
                (float left, float right) = MapSourceFrameToStereo(original, frame);
                leftSquareSum += left * left;
                rightSquareSum += right * right;
            }

            double leftRms = Math.Sqrt(leftSquareSum / original.FrameCount);
            double rightRms = Math.Sqrt(rightSquareSum / original.FrameCount);
            double levelSum = leftRms + rightRms;
            if (!double.IsFinite(levelSum) || Math.Max(leftRms, rightRms) < PanSilenceRmsThreshold)
            {
                resolvedGains = BuildConstantPowerPan(balance: 0d);
                panGainCache[cacheKey] = resolvedGains;
                return resolvedGains;
            }

            resolvedGains = BuildConstantPowerPan((rightRms - leftRms) / levelSum);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception)
        {
            resolvedGains = BuildConstantPowerPan(balance: 0d);
        }

        panGainCache[cacheKey] = resolvedGains;
        return resolvedGains;
    }

    private static float MapSourceFrameToMono(WavePcm16Samples source, int frame)
    {
        if (frame >= source.FrameCount)
        {
            return 0f;
        }

        int sourceOffset = frame * source.ChannelCount;
        if (source.ChannelCount == 1)
        {
            return source.Samples[sourceOffset];
        }

        float sum = 0f;
        for (int channel = 0; channel < source.ChannelCount; channel++)
        {
            sum += source.Samples[sourceOffset + channel];
        }

        return sum / source.ChannelCount;
    }

    private static (float Left, float Right) MapSourceFrameToStereo(WavePcm16Samples source, int frame)
    {
        if (frame >= source.FrameCount)
        {
            return (0f, 0f);
        }

        int sourceOffset = frame * source.ChannelCount;
        if (source.ChannelCount == 1)
        {
            float sample = source.Samples[sourceOffset];
            return (sample, sample);
        }

        if (source.ChannelCount == 2)
        {
            return (source.Samples[sourceOffset], source.Samples[sourceOffset + 1]);
        }

        return MapMultichannelToStereo(source, sourceOffset);
    }

    private static (float Left, float Right) MapMultichannelToStereo(WavePcm16Samples source, int sourceOffset)
    {
        if (source.ChannelMask is { } channelMask &&
            channelMask != 0u &&
            TryMapChannelMaskToStereo(source, sourceOffset, channelMask, out (float Left, float Right) maskedStereo))
        {
            return maskedStereo;
        }

        float left = source.Samples[sourceOffset];
        float right = source.Samples[sourceOffset + 1];
        if (source.ChannelCount == 4)
        {
            // Common 4-channel WAV layouts are quad FL,FR,BL,BR; preserve rear placement.
            left += source.Samples[sourceOffset + 2] * DownmixSurroundGain;
            right += source.Samples[sourceOffset + 3] * DownmixSurroundGain;
            return (left, right);
        }

        if (source.ChannelCount >= 3)
        {
            float center = source.Samples[sourceOffset + 2] * DownmixCenterGain;
            left += center;
            right += center;
        }

        if (source.ChannelCount == 5)
        {
            left += source.Samples[sourceOffset + 3] * DownmixSurroundGain;
            right += source.Samples[sourceOffset + 4] * DownmixSurroundGain;
        }
        else if (source.ChannelCount == 6)
        {
            // Common unmasked 6-channel PCM WAVs are 5.1: L, R, C, LFE, LS, RS.
            left += source.Samples[sourceOffset + 4] * DownmixSurroundGain;
            right += source.Samples[sourceOffset + 5] * DownmixSurroundGain;
        }
        else if (source.ChannelCount == 7)
        {
            // Common 6.1 order is L, R, C, LFE, back center, LS, RS.
            float backCenter = source.Samples[sourceOffset + 4] * DownmixSurroundGain;
            left += backCenter;
            right += backCenter;
            left += source.Samples[sourceOffset + 5] * DownmixSurroundGain;
            right += source.Samples[sourceOffset + 6] * DownmixSurroundGain;
        }
        else if (source.ChannelCount >= 8)
        {
            // Common 7.1 order is L, R, C, LFE, BL, BR, SL, SR.
            left += source.Samples[sourceOffset + 4] * DownmixSurroundGain;
            right += source.Samples[sourceOffset + 5] * DownmixSurroundGain;
            left += source.Samples[sourceOffset + 6] * DownmixSurroundGain;
            right += source.Samples[sourceOffset + 7] * DownmixSurroundGain;

            if (source.ChannelCount >= 9)
            {
                float backCenter = source.Samples[sourceOffset + 8] * DownmixSurroundGain;
                left += backCenter;
                right += backCenter;
            }

            if (source.ChannelCount >= 10)
            {
                left += source.Samples[sourceOffset + 9] * DownmixSurroundGain;
            }

            if (source.ChannelCount >= 11)
            {
                right += source.Samples[sourceOffset + 10] * DownmixSurroundGain;
            }

            for (int channel = 11; channel < source.ChannelCount; channel++)
            {
                float contribution = source.Samples[sourceOffset + channel] * DownmixSurroundGain;
                left += contribution;
                right += contribution;
            }
        }

        return (left, right);
    }

    private static bool TryMapChannelMaskToStereo(
        WavePcm16Samples source,
        int sourceOffset,
        uint channelMask,
        out (float Left, float Right) stereo)
    {
        stereo = (0f, 0f);
        float left = 0f;
        float right = 0f;
        int sourceChannel = 0;
        foreach (uint speaker in ChannelMaskSpeakerOrder)
        {
            if ((channelMask & speaker) == 0u)
            {
                continue;
            }

            if (sourceChannel >= source.ChannelCount)
            {
                stereo = (0f, 0f);
                return false;
            }

            AddMaskedChannelToStereo(speaker, source.Samples[sourceOffset + sourceChannel], ref left, ref right);
            sourceChannel++;
        }

        if (sourceChannel != source.ChannelCount)
        {
            return false;
        }

        stereo = (left, right);
        return true;
    }

    private static void AddMaskedChannelToStereo(uint speaker, float sample, ref float left, ref float right)
    {
        switch (speaker)
        {
            case SpeakerFrontLeft:
                left += sample;
                break;

            case SpeakerFrontRight:
                right += sample;
                break;

            case SpeakerFrontCenter:
            case SpeakerTopCenter:
            case SpeakerTopFrontCenter:
                left += sample * DownmixCenterGain;
                right += sample * DownmixCenterGain;
                break;

            case SpeakerLowFrequency:
                break;

            case SpeakerBackLeft:
            case SpeakerSideLeft:
            case SpeakerTopFrontLeft:
            case SpeakerTopBackLeft:
            case SpeakerFrontLeftOfCenter:
                left += sample * DownmixSurroundGain;
                break;

            case SpeakerBackRight:
            case SpeakerSideRight:
            case SpeakerTopFrontRight:
            case SpeakerTopBackRight:
            case SpeakerFrontRightOfCenter:
                right += sample * DownmixSurroundGain;
                break;

            case SpeakerBackCenter:
            case SpeakerTopBackCenter:
                left += sample * DownmixSurroundGain;
                right += sample * DownmixSurroundGain;
                break;
        }
    }

    private static int ResolveOutputChannelCount(MixPlan mixPlan, WavePcm16Info sourceInfo) =>
        mixPlan.OutputChannelCount >= 2 || sourceInfo.ChannelCount >= 2 ? 2 : 1;

    private static string NormalizePanCachePath(string path) =>
        path.Replace('\\', '/').ToUpperInvariant();

    private static PanGains BuildConstantPowerPan(double balance)
    {
        double clampedBalance = Math.Clamp(balance, -1d, 1d);
        double angle = (clampedBalance + 1d) * Math.PI / 4d;
        return new PanGains((float)Math.Cos(angle), (float)Math.Sin(angle));
    }

    private static float[] ResampleLinear(float[] samples, int sourceSampleRate, int targetSampleRate)
    {
        if (samples.Length == 0 || sourceSampleRate <= 0 || targetSampleRate <= 0)
        {
            return [];
        }

        double targetSampleCountValue = samples.Length / (double)sourceSampleRate * targetSampleRate;
        if (!double.IsFinite(targetSampleCountValue) || targetSampleCountValue > int.MaxValue)
        {
            throw new InvalidOperationException("Resampled dubbed speech take is too large to render in one mix.");
        }

        int targetSampleCount = Math.Max(1, (int)Math.Round(targetSampleCountValue));
        var result = new float[targetSampleCount];
        double scale = sourceSampleRate / (double)targetSampleRate;
        for (int index = 0; index < result.Length; index++)
        {
            double sourcePosition = index * scale;
            int lowerIndex = Math.Clamp((int)Math.Floor(sourcePosition), 0, samples.Length - 1);
            int upperIndex = Math.Min(samples.Length - 1, lowerIndex + 1);
            double blend = sourcePosition - lowerIndex;
            result[index] = (float)((samples[lowerIndex] * (1d - blend)) + (samples[upperIndex] * blend));
        }

        return result;
    }

    private static float DecibelsToLinear(double decibels)
    {
        if (!double.IsFinite(decibels))
        {
            return 1f;
        }

        if (decibels <= -96d)
        {
            return 0f;
        }

        return (float)Math.Pow(10d, decibels / 20d);
    }

    private sealed record EffectivePreviewRange(
        int SampleRate,
        int SampleCount,
        double DurationSeconds,
        double EndSeconds);

    private readonly record struct PanGains(float Left, float Right);

    private sealed class PanAnalysisContext(string normalizedOriginalMixAudioRelativePath)
    {
        public string NormalizedOriginalMixAudioRelativePath { get; } = normalizedOriginalMixAudioRelativePath;

        public string? OriginalMixPath { get; set; }

        public WavePcm16Info? OriginalMixInfo { get; set; }
    }

    private readonly record struct PanCacheKey(
        string OriginalMixAudioRelativePath,
        Guid SegmentId,
        int OutputChannelCount);
}
