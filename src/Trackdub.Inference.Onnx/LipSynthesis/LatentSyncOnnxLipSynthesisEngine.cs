using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Inference.Onnx.Pool;
using Trackdub.Inference.Runtime.ModelManifest;
using Trackdub.Inference.Runtime.Planning;

namespace Trackdub.Inference.Onnx.LipSynthesis;

/// <summary>
/// LatentSync 1.6 ONNX lip-synthesis engine (ByteDance, openrail++ license).
/// Requires four ONNX subgraphs: UNet, VAE encoder, VAE decoder, Whisper encoder.
/// All quality gating (face detection, pose, landmarks) is handled by
/// <see cref="Trackdub.Application.LipSynthesis.LipSynthesisStageHandler"/> upstream — this
/// engine only runs when all guards have passed.
/// </summary>
public sealed class LatentSyncOnnxLipSynthesisEngine(
    IRuntimePlanner runtimePlanner,
    BenchmarkModelPathResolver modelPathResolver,
    IVideoFrameExtractor frameExtractor,
    IVideoFrameAssembler frameAssembler,
    IAudioSegmentExtractor audioExtractor,
    BundledModelManifestRegistry manifestRegistry)
    : ILipSynthesisEngine, IStageRuntimeExecutionReporter
{
    private const string EngineFamilyName = LatentSyncModelPaths.EngineFamily;
    private const int WhisperFeatureRateHz = 50;
    private const int AudioContextFramesBefore = 2;
    private const int AudioContextFramesAfter = 2;

    public StageRuntimeExecutionSummary? LastExecutionSummary { get; private set; }

    public bool IsAvailable
    {
        get
        {
            if (!HasValidatedRepairPipeline())
                return false;
            try
            {
                BenchmarkModelResolutionResult discovery = modelPathResolver.Discover(LatentSyncModelPaths.ManifestAlias);
                if (!string.IsNullOrWhiteSpace(discovery.Error) || discovery.Candidates.Count == 0)
                    return false;
                BenchmarkModelCandidate candidate = discovery.Candidates[0];
                string modelRoot = candidate.RootDirectory
                    ?? Path.GetDirectoryName(candidate.ModelPath)
                    ?? string.Empty;
                return LatentSyncModelPaths.AreLatentSyncFilesPresent(modelRoot);
            }
            catch
            {
                return false;
            }
        }
    }

    public bool IsExperimental => IsExperimentalFromManifest(manifestRegistry);

    public string ProviderId => "latentsync-onnx";

    public string ModelId => LatentSyncModelPaths.ModelId;

    public async Task<LipSynthesisResult> SynthesizeTurnAsync(
        LipSynthesisRequest request,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();

        StageRuntimePlan plan = await runtimePlanner.PlanAsync(
            new StageRuntimePlanningRequest(
                RuntimeStage.LipSynthesis,
                PreferredModelAlias: request.PreferredModelAlias),
            cancellationToken)
            .ConfigureAwait(false);

        LastExecutionSummary = new StageRuntimeExecutionSummary(
            RequestedProvider: plan.ExecutionProvider?.ToString() ?? ExecutionProviderKind.Cpu.ToString(),
            SelectedProvider: plan.ExecutionProvider?.ToString() ?? ExecutionProviderKind.Cpu.ToString(),
            ModelId: plan.ModelId ?? LatentSyncModelPaths.ModelId,
            ModelAlias: plan.ModelAlias ?? LatentSyncModelPaths.ManifestAlias,
            ModelVariant: plan.Variant,
            BootstrapDetail: "LatentSync session creation pending.");

        if (!plan.IsRunnable())
        {
            return Skipped(request,
                plan.Fallback?.Detail ?? "LatentSync runtime plan is not ready.");
        }

        if (!string.Equals(plan.EngineFamily, EngineFamilyName, StringComparison.OrdinalIgnoreCase))
        {
            return Failed(request,
                $"LatentSync engine cannot run engine family '{plan.EngineFamily ?? "unknown"}'.");
        }

        string modelRoot = PlannedRuntimeModelResolver.ResolveModelRootPath(plan, modelPathResolver);
        if (!LatentSyncModelPaths.AreLatentSyncFilesPresent(modelRoot))
            return Skipped(request, "LatentSync model files are not present.");

        // The existing renderer is a single-frame, full-frame scaffold. It does
        // not provide the temporal conditioning or tracked face compositor used
        // by the released model. Preserve the source video until those pieces
        // pass the clip-level acceptance tests.
        if (!HasValidatedRepairPipeline())
            return Skipped(request, "LatentSync repair is unavailable: the ONNX temporal pipeline and face compositor have not passed validation.");

        ExecutionProviderKind provider = plan.ExecutionProvider ?? ExecutionProviderKind.Cpu;

        using OnnxExecutionSessionFactory.LatentSyncSessionLease lease = await OnnxExecutionSessionFactory.CreatePooledLatentSyncAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = EngineFamilyName, Provider = provider, CancellationToken = cancellationToken }, new OnnxExecutionSessionFactory.LatentSyncModelPaths(LatentSyncModelPaths.UNetPath(modelRoot), LatentSyncModelPaths.VaeEncoderPath(modelRoot), LatentSyncModelPaths.VaeDecoderPath(modelRoot), LatentSyncModelPaths.WhisperEncoderPath(modelRoot)))
            .ConfigureAwait(false);

        string tempDir = Path.Join(Path.GetTempPath(), $"lipsync_{request.SegmentId:N}");
        Directory.CreateDirectory(tempDir);
        string framesDir = Path.Join(tempDir, "frames");
        // Clear stale frames from any previous interrupted run at this deterministic path.
        if (Directory.Exists(framesDir))
            Directory.Delete(framesDir, recursive: true);
        string patchedPath = Path.Join(tempDir, "patched.mp4");
        string? standalonePatched = null;

        try
        {
            // Extract frames from the turn window.
            FrameExtractionResult frames = await frameExtractor.ExtractTurnFramesAsync(
                request.OriginalVideoPath,
                request.TurnStart.TotalSeconds,
                request.TurnEnd.TotalSeconds,
                framesDir,
                cancellationToken)
                .ConfigureAwait(false);

            if (frames.FrameCount == 0)
                return Skipped(request, "No frames extracted for the turn.");

            // Extract the dubbed audio segment and compute mel spectrogram → Whisper embeddings.
            string segmentWav = Path.Join(tempDir, "segment.wav");
            await audioExtractor.ExtractSegmentAsync(
                request.DubbedAudioPath,
                request.TurnStart,
                request.TurnEnd,
                segmentWav,
                cancellationToken)
                .ConfigureAwait(false);

            byte[] waveBytes = await File.ReadAllBytesAsync(segmentWav, cancellationToken)
                .ConfigureAwait(false);
            float[] segmentPcm = LatentSyncWavAudioReader.ReadMono16Khz(waveBytes);

            var scheduler = new DdimScheduler();

            // Whisper features are computed once for the turn. The U-Net conditions each
            // output frame on a centered 10-step window from the 50 Hz feature timeline.
            (float[] turnWhisperEmbeds, int whisperSeqLen, int whisperHiddenDim) = RunWhisperEncoder(
                lease.WhisperEncoderSession, segmentPcm, cancellationToken);

            // Process each frame through the diffusion pipeline.
            string[] frameFiles = Directory.GetFiles(framesDir, "*.rgba")
                .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            for (int frameIndex = 0; frameIndex < frameFiles.Length; frameIndex++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                string framePath = frameFiles[frameIndex];
                float[] frameWhisperEmbeds = SliceWhisperContext(
                    turnWhisperEmbeds,
                    whisperSeqLen,
                    whisperHiddenDim,
                    frameIndex,
                    frames.FrameRate,
                    AudioContextFramesBefore,
                    AudioContextFramesAfter);

                byte[] rgbaBytes = await File.ReadAllBytesAsync(framePath, cancellationToken)
                    .ConfigureAwait(false);
                int w = frames.FrameWidth;
                int h = frames.FrameHeight;

                // Encode reference frame to latent space.
                float[] normalizedFrame = LatentSyncTensorPreprocessor.RgbaToNormalizedTensor(rgbaBytes, w, h);
                float[] refLatent = RunVaeEncoder(
                    lease.VaeEncoderSession, normalizedFrame, cancellationToken);

                // Initialize noisy latent.
                float[] noisyLatent = CreateGaussianNoise(refLatent.Length);
                noisyLatent = scheduler.AddNoise(refLatent, noisyLatent, scheduler.Timesteps[0]);

                // DDIM denoising loop.
                foreach (int t in scheduler.Timesteps)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    float[] noise = RunUNet(
                        lease.UNetSession,
                        noisyLatent,
                        t,
                        frameWhisperEmbeds,
                        AudioContextFramesBefore + AudioContextFramesAfter + 1,
                        whisperHiddenDim,
                        cancellationToken);
                    noisyLatent = scheduler.Step(noise, t, noisyLatent);
                }

                // Decode latent to pixel space and write back.
                float[] decoded = RunVaeDecoder(
                    lease.VaeDecoderSession, noisyLatent, cancellationToken);

                Span<byte> outRgba = rgbaBytes;
                LatentSyncTensorPreprocessor.PasteFloatTensorIntoRgba(
                    decoded, outRgba, w, h,
                    faceX: 0, faceY: 0, faceW: w, faceH: h);

                await File.WriteAllBytesAsync(framePath, rgbaBytes, cancellationToken)
                    .ConfigureAwait(false);
            }

            // Assemble processed frames back to video.
            await frameAssembler.AssembleFramesAsync(
                framesDir,
                patchedPath,
                frames.FrameWidth,
                frames.FrameHeight,
                frames.FrameRate,
                cancellationToken)
                .ConfigureAwait(false);

            // Move patched clip out of tempDir so the whole working dir can be deleted below.
            standalonePatched = Path.Join(Path.GetTempPath(), $"lipsync_patched_{request.SegmentId:N}.mp4");
            File.Move(patchedPath, standalonePatched, overwrite: true);

            LastExecutionSummary = LastExecutionSummary with
            {
                SelectedProvider = lease.SelectedProvider,
                BootstrapDetail = $"LatentSync synthesized {frames.FrameCount} frames."
            };

            return new LipSynthesisResult(
                SegmentId: request.SegmentId,
                Status: LipSynthesisEngineStatus.Synthesized,
                PatchedClipPath: standalonePatched,
                SkipReason: null,
                FailureReason: null,
                ProviderId: ProviderId,
                ModelId: ModelId);
        }
        catch (Exception)
        {
            if (standalonePatched is not null)
                try { File.Delete(standalonePatched); } catch { }
            throw;
        }
        finally
        {
            try { Directory.Delete(tempDir, recursive: true); } catch { }
        }
    }

    private static (float[] Embeddings, int SeqLen, int HiddenDim) RunWhisperEncoder(
        InferenceSession session,
        float[] pcm16000Hz,
        CancellationToken cancellationToken)
    {
        if (pcm16000Hz.Length == 0)
            throw new InvalidDataException("LatentSync audio turn is empty.");

        (int melBins, int melFrames) = LatentSyncTensorPreprocessor.MelShape;
        int totalFeatureFrames = LatentSyncTensorPreprocessor.GetWhisperFeatureFrameCount(pcm16000Hz.Length);
        float[]? embeddings = null;
        int hiddenDimension = 0;
        int destinationFeatureFrame = 0;

        for (int sampleOffset = 0; sampleOffset < pcm16000Hz.Length; sampleOffset += LatentSyncTensorPreprocessor.WhisperWindowSamples)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int windowSampleCount = Math.Min(
                LatentSyncTensorPreprocessor.WhisperWindowSamples,
                pcm16000Hz.Length - sampleOffset);
            float[] windowPcm = pcm16000Hz.AsSpan(sampleOffset, windowSampleCount).ToArray();
            float[] mel = LatentSyncTensorPreprocessor.ComputeWhisperMelSpectrogram(windowPcm);

            var melTensor = new DenseTensor<float>(mel, [1, melBins, melFrames]);
            var inputs = new[] { NamedOnnxValue.CreateFromTensor("input_features", melTensor) };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = session.RunWithRetry(
                inputs,
                cancellationToken: cancellationToken);
            var hidden = outputs.Single(o => o.Name == "last_hidden_state").AsTensor<float>();
            int sequenceLength = hidden.Dimensions[1];
            int currentHiddenDimension = hidden.Dimensions[2];
            int validFeatureFrames = LatentSyncTensorPreprocessor.GetWhisperFeatureFrameCount(windowSampleCount);
            if (hidden.Dimensions[0] != 1 || sequenceLength < validFeatureFrames || currentHiddenDimension <= 0)
            {
                throw new InvalidDataException(
                    $"LatentSync Whisper encoder returned an invalid feature shape [{string.Join(',', hidden.Dimensions.ToArray())}].");
            }

            embeddings ??= new float[checked(totalFeatureFrames * currentHiddenDimension)];
            if (currentHiddenDimension != hiddenDimension && hiddenDimension != 0)
                throw new InvalidDataException("LatentSync Whisper encoder changed hidden size between audio windows.");
            hiddenDimension = currentHiddenDimension;

            float[] windowEmbeddings = hidden.ToArray();
            int valuesToCopy = checked(validFeatureFrames * hiddenDimension);
            Array.Copy(
                windowEmbeddings,
                sourceIndex: 0,
                embeddings,
                destinationFeatureFrame * hiddenDimension,
                valuesToCopy);
            destinationFeatureFrame += validFeatureFrames;
        }

        if (embeddings is null || destinationFeatureFrame != totalFeatureFrames)
            throw new InvalidDataException("LatentSync Whisper features do not cover the complete audio turn.");

        return (embeddings, totalFeatureFrames, hiddenDimension);
    }

    internal static float[] SliceWhisperContextForTest(
        float[] embeddings,
        int sequenceLength,
        int hiddenDimension,
        int frameIndex,
        double frameRate,
        int framesBefore,
        int framesAfter) =>
        SliceWhisperContext(embeddings, sequenceLength, hiddenDimension, frameIndex, frameRate, framesBefore, framesAfter);

    private static float[] SliceWhisperContext(
        float[] embeddings,
        int sequenceLength,
        int hiddenDimension,
        int frameIndex,
        double frameRate,
        int framesBefore,
        int framesAfter)
    {
        if (sequenceLength <= 0 || hiddenDimension <= 0 || frameIndex < 0 ||
            !double.IsFinite(frameRate) || frameRate <= 0d || framesBefore < 0 || framesAfter < 0 ||
            embeddings.Length != checked(sequenceLength * hiddenDimension))
        {
            throw new ArgumentOutOfRangeException(nameof(frameIndex), "Whisper context dimensions or frame timing are invalid.");
        }

        int center = checked((int)(frameIndex * (WhisperFeatureRateHz / frameRate)));
        int before = checked(framesBefore * 2);
        int afterExclusive = checked((framesAfter + 1) * 2);
        int contextLength = before + afterExclusive;
        var context = new float[checked(contextLength * hiddenDimension)];
        for (int contextFrame = 0; contextFrame < contextLength; contextFrame++)
        {
            int featureIndex = Math.Clamp(center - before + contextFrame, 0, sequenceLength - 1);
            Array.Copy(
                embeddings,
                featureIndex * hiddenDimension,
                context,
                contextFrame * hiddenDimension,
                hiddenDimension);
        }

        return context;
    }

    private static float[] RunVaeEncoder(
        InferenceSession session,
        float[] normalizedFrame,
        CancellationToken cancellationToken)
    {
        int h = LatentSyncTensorPreprocessor.TargetHeight;
        int w = LatentSyncTensorPreprocessor.TargetWidth;
        var frameTensor = new DenseTensor<float>(normalizedFrame, [1, 3, h, w]);
        var inputs = new[] { NamedOnnxValue.CreateFromTensor("sample", frameTensor) };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = session.RunWithRetry(
            inputs,
            cancellationToken: cancellationToken);
        return outputs.Single(o => o.Name == "latent_sample")
            .AsTensor<float>()
            .ToArray();
    }

    private static float[] RunVaeDecoder(
        InferenceSession session,
        float[] latent,
        CancellationToken cancellationToken)
    {
        int lh = LatentSyncTensorPreprocessor.LatentHeight;
        int lw = LatentSyncTensorPreprocessor.LatentWidth;
        int lc = LatentSyncTensorPreprocessor.LatentChannels;
        var latentTensor = new DenseTensor<float>(latent, [1, lc, lh, lw]);
        var inputs = new[] { NamedOnnxValue.CreateFromTensor("latent_sample", latentTensor) };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = session.RunWithRetry(
            inputs,
            cancellationToken: cancellationToken);
        return outputs.Single(o => o.Name == "sample")
            .AsTensor<float>()
            .ToArray();
    }

    private static float[] RunUNet(
        InferenceSession session,
        float[] sample,
        int timestep,
        float[] encoderHiddenStates,
        int seqLen,
        int hiddenDim,
        CancellationToken cancellationToken)
    {
        int lh = LatentSyncTensorPreprocessor.LatentHeight;
        int lw = LatentSyncTensorPreprocessor.LatentWidth;
        int lc = LatentSyncTensorPreprocessor.LatentChannels;

        var sampleTensor = new DenseTensor<float>(sample, [1, lc, lh, lw]);
        var timestepTensor = new DenseTensor<long>(new[] { (long)timestep }, [1]);
        var hiddenTensor = new DenseTensor<float>(encoderHiddenStates, [1, seqLen, hiddenDim]);

        var inputs = new[]
        {
            NamedOnnxValue.CreateFromTensor("sample", sampleTensor),
            NamedOnnxValue.CreateFromTensor("timestep", timestepTensor),
            NamedOnnxValue.CreateFromTensor("encoder_hidden_states", hiddenTensor),
        };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = session.RunWithRetry(
            inputs,
            cancellationToken: cancellationToken);
        return outputs.Single(o => o.Name == "out_sample")
            .AsTensor<float>()
            .ToArray();
    }

    private static float[] CreateGaussianNoise(int length)
    {
        // Box-Muller transform for Gaussian noise.
        var rng = new Random();
        float[] noise = new float[length];
        for (int i = 0; i < length - 1; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            float mag = (float)Math.Sqrt(-2.0 * Math.Log(u1));
            noise[i] = mag * (float)Math.Cos(2.0 * Math.PI * u2);
            noise[i + 1] = mag * (float)Math.Sin(2.0 * Math.PI * u2);
        }
        if (length % 2 == 1)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            noise[length - 1] = (float)Math.Sqrt(-2.0 * Math.Log(u1)) * (float)Math.Cos(2.0 * Math.PI * u2);
        }
        return noise;
    }

    internal static bool HasValidatedRepairPipeline()
    {
        // Explicit rollout gate: no version has a validated video repair path.
        return false;
    }

    private static LipSynthesisResult Skipped(LipSynthesisRequest request, string reason) =>
        new(request.SegmentId, LipSynthesisEngineStatus.Skipped,
            PatchedClipPath: null, SkipReason: reason,
            FailureReason: null, ProviderId: "latentsync-onnx", ModelId: LatentSyncModelPaths.ModelId);

    private static LipSynthesisResult Failed(LipSynthesisRequest request, string reason) =>
        new(request.SegmentId, LipSynthesisEngineStatus.Failed,
            PatchedClipPath: null, SkipReason: null,
            FailureReason: reason, ProviderId: "latentsync-onnx", ModelId: LatentSyncModelPaths.ModelId);

    internal static bool IsExperimentalFromManifest(BundledModelManifestRegistry registry)
    {
        if (!registry.TryResolve(LatentSyncModelPaths.ManifestAlias, out BundledModelManifestResolution? resolution) ||
            resolution is null)
        {
            return true;
        }

        return IsExperimentalFromEntry(resolution.Entry);
    }

    internal static bool IsExperimentalFromEntry(BundledModelManifestEntry entry) =>
        entry.Lane != ModelLane.Commercial
        || !entry.CommercialUseVerified
        || !entry.CommercialAllowed;
}
