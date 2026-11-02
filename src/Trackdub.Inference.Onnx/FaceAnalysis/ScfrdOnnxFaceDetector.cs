using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Contracts.Diagnostics;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.LipSynthesis;
using Trackdub.Inference.Onnx.LipSynthesis;

namespace Trackdub.Inference.Onnx.FaceAnalysis;

/// <summary>
/// SCRFD-500M ONNX face detector (InsightFace, MIT license).
/// Reports IsAvailable=false until scrfd_500m.onnx is present at the configured model root.
/// Sessions come from the shared pool (as in sibling pooled engines) instead of a new
/// <see cref="InferenceSession"/> per detection call.
/// </summary>
public sealed class ScfrdOnnxFaceDetector : IFaceDetector
{
    private readonly BenchmarkModelPathResolver modelPathResolver;
    private readonly IVideoFrameExtractor frameExtractor;
    private readonly Func<string?> modelPathProvider;
    private readonly Func<string, CancellationToken, Task<OnnxExecutionSessionFactory.SingleSessionLease>> sessionLeaseFactory;

    public ScfrdOnnxFaceDetector(
        BenchmarkModelPathResolver modelPathResolver,
        IVideoFrameExtractor frameExtractor)
        : this(modelPathResolver, frameExtractor, modelPathProvider: null, sessionLeaseFactory: null)
    {
    }

    internal ScfrdOnnxFaceDetector(
        BenchmarkModelPathResolver modelPathResolver,
        IVideoFrameExtractor frameExtractor,
        Func<string?>? modelPathProvider,
        Func<string, CancellationToken, Task<OnnxExecutionSessionFactory.SingleSessionLease>>? sessionLeaseFactory)
    {
        this.modelPathResolver = modelPathResolver ?? throw new ArgumentNullException(nameof(modelPathResolver));
        this.frameExtractor = frameExtractor ?? throw new ArgumentNullException(nameof(frameExtractor));
        this.modelPathProvider = modelPathProvider ?? (() => ResolveModelPath());
        this.sessionLeaseFactory = sessionLeaseFactory ?? CreatePooledSessionLeaseAsync;
    }

    private static Task<OnnxExecutionSessionFactory.SingleSessionLease> CreatePooledSessionLeaseAsync(
        string modelPath,
        CancellationToken cancellationToken) =>
        OnnxExecutionSessionFactory.CreatePooledSingleAsync(new OnnxExecutionSessionFactory.PooledSessionRequest { EngineFamily = LatentSyncModelPaths.ScfrdEngineFamily, Provider = ExecutionProviderKind.Cpu, CancellationToken = cancellationToken }, modelPath);
    private const string ModelAlias = "InsightFace/scrfd-500m";
    private const int InferSize = 640;
    private const float ScoreThreshold = 0.3f;
    private const float NmsIouThreshold = 0.4f;

    private static readonly int[] Strides = [8, 16, 32];
    private static readonly int[] AnchorCounts = [12800, 3200, 800]; // H*W*2 per stride

    public bool IsAvailable
    {
        get
        {
            try
            {
                return modelPathProvider() is not null;
            }
            catch
            {
                return false;
            }
        }
    }

    public async Task<FaceDetectionResult> DetectPrimaryFaceAsync(
        FaceAnalysisRequest request,
        CancellationToken cancellationToken)
    {
        string? modelPath = modelPathProvider();
        if (modelPath is null)
            return NoFace;

        // Extract one frame near the midpoint of the turn for turn-level face analysis.
        // Single frame is sufficient because face position is stable within a turn; NMS handles
        // multi-face scenarios by selecting the highest-confidence detection.
        TimeSpan mid = request.Start + ((request.End - request.Start) / 2);
        TimeSpan sampleEnd = (request.End - mid) < TimeSpan.FromSeconds(0.5)
            ? request.End
            : mid + TimeSpan.FromSeconds(0.5);
        string tempDir = Path.Join(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(tempDir);

        try
        {
            FrameExtractionResult extracted = await frameExtractor.ExtractTurnFramesAsync(
                request.VideoPath, mid.TotalSeconds, sampleEnd.TotalSeconds, tempDir, cancellationToken).ConfigureAwait(false);

            if (extracted.FrameCount == 0)
                return NoFace;

            string framePath = Path.Join(extracted.FramesDirectory, "frame_000001.rgba");
            if (!File.Exists(framePath))
                return NoFace;

            byte[] rgba = await File.ReadAllBytesAsync(framePath, cancellationToken).ConfigureAwait(false);
            float letterboxScale = LetterboxScale(extracted.FrameWidth, extracted.FrameHeight);
            float[] inputData = NormalizeFrameBgr(rgba, extracted.FrameWidth, extracted.FrameHeight);

            using OnnxExecutionSessionFactory.SingleSessionLease sessionLease =
                await sessionLeaseFactory(modelPath, cancellationToken).ConfigureAwait(false);
            InferenceSession session = sessionLease.Session;
            string inputName = session.InputMetadata.Keys.First();
            var inputTensor = new DenseTensor<float>(inputData, [1, 3, InferSize, InferSize]);

            using var outputs = session.Run(
                [NamedOnnxValue.CreateFromTensor(inputName, inputTensor)]);
            ColdStartCapture.MarkInferenceCompleted("scrfd:detect");

            var detections = DecodeAndFilter(outputs);
            if (detections.Count == 0)
                return NoFace;

            var nmsResults = ApplyNms(detections);
            var best = nmsResults.MaxBy(d => d.Score);

            var region = new FaceRegion(
                X: (int)(best.X1 / letterboxScale),
                Y: (int)(best.Y1 / letterboxScale),
                Width: (int)((best.X2 - best.X1) / letterboxScale),
                Height: (int)((best.Y2 - best.Y1) / letterboxScale));

            return new FaceDetectionResult(FaceFound: true, Confidence: best.Score, PrimaryFace: region);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            return NoFace;
        }
        finally
        {
            try { Directory.Delete(tempDir, recursive: true); } catch { }
        }
    }

    private string? ResolveModelPath()
    {
        BenchmarkModelResolutionResult discovery = modelPathResolver.Discover(ModelAlias);
        if (!string.IsNullOrWhiteSpace(discovery.Error) || discovery.Candidates.Count == 0)
            return null;
        BenchmarkModelCandidate candidate = discovery.Candidates[0];
        string modelRoot = candidate.RootDirectory
            ?? Path.GetDirectoryName(candidate.ModelPath)
            ?? string.Empty;
        if (!LatentSyncModelPaths.AreScfrdFilesPresent(modelRoot))
            return null;
        return LatentSyncModelPaths.ScfrdModelPath(modelRoot);
    }

    /// <summary>
    /// Uniform scale that fits the frame inside the square detector input. Stretching each axis
    /// separately squashes faces in portrait or ultra-wide video until SCRFD no longer finds them.
    /// </summary>
    internal static float LetterboxScale(int srcW, int srcH) =>
        Math.Min((float)InferSize / srcW, (float)InferSize / srcH);

    // SCRFD expects CHW, normalized (pixel - 127.5) / 128.0
    internal static float[] NormalizeFrameBgr(byte[] rgba, int srcW, int srcH)
    {
        // Letterbox: scale uniformly into the top-left corner and leave the rest as black padding,
        // matching InsightFace's SCRFD preprocessing.
        const float padValue = (0f - 127.5f) / 128.0f;
        float[] tensor = new float[3 * InferSize * InferSize];
        Array.Fill(tensor, padValue);
        float scale = LetterboxScale(srcW, srcH);
        int scaledW = Math.Min(InferSize, (int)Math.Round(srcW * scale));
        int scaledH = Math.Min(InferSize, (int)Math.Round(srcH * scale));

        for (int y = 0; y < scaledH; y++)
        {
            int sy = Math.Min((int)(y / scale), srcH - 1);
            for (int x = 0; x < scaledW; x++)
            {
                int sx = Math.Min((int)(x / scale), srcW - 1);
                int srcIdx = ((sy * srcW) + sx) * 4;

                // CHW layout, RGB channel order (SCRFD trained with swapRB=True → expects RGB)
                tensor[(0 * InferSize * InferSize) + (y * InferSize) + x] = (rgba[srcIdx] - 127.5f) / 128.0f;
                tensor[(1 * InferSize * InferSize) + (y * InferSize) + x] = (rgba[srcIdx + 1] - 127.5f) / 128.0f;
                tensor[(2 * InferSize * InferSize) + (y * InferSize) + x] = (rgba[srcIdx + 2] - 127.5f) / 128.0f;
            }
        }

        return tensor;
    }

    private static List<Detection> DecodeAndFilter(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs) =>
        DecodeAndFilter(outputs.Select(static output => output.Value).OfType<Tensor<float>>());

    internal static List<Detection> DecodeAndFilter(IEnumerable<Tensor<float>> outputs)
    {
        // Group tensors by anchor-count N: scores=[N,1] or [1,N,1], bboxes=[N,4] or [1,N,4]
        var scoreTensors = new Dictionary<int, Tensor<float>>();
        var bboxTensors = new Dictionary<int, Tensor<float>>();

        foreach (Tensor<float> tensor in outputs)
        {
            ReadOnlySpan<int> dims = tensor.Dimensions;

            int n, c;
            if (dims.Length == 3 && dims[0] == 1)
            {
                // Shape [1, N, C]
                n = dims[1];
                c = dims[2];
            }
            else if (dims.Length == 2)
            {
                // Shape [N, C] (no batch dimension)
                n = dims[0];
                c = dims[1];
            }
            else
            {
                continue;
            }

            if (c == 1) scoreTensors[n] = tensor;
            else if (c == 4) bboxTensors[n] = tensor;
            // c == 10 = keypoints, ignored for face detection
        }

        var detections = new List<Detection>();

        for (int si = 0; si < Strides.Length; si++)
        {
            int stride = Strides[si];
            int n = AnchorCounts[si];
            int fSize = InferSize / stride; // 80, 40, 20

            if (!scoreTensors.TryGetValue(n, out Tensor<float>? scores) ||
                !bboxTensors.TryGetValue(n, out Tensor<float>? bboxes))
                continue;

            int anchorIdx = 0;
            bool scoresIs3D = scores.Dimensions.Length == 3;
            bool bboxesIs3D = bboxes.Dimensions.Length == 3;
            if (scoresIs3D != bboxesIs3D)
            {
                throw new InvalidOperationException(
                    $"SCRFD output dimension mismatch: scores shape [{string.Join("x", scores.Dimensions.ToArray())}] " +
                    $"vs bboxes shape [{string.Join("x", bboxes.Dimensions.ToArray())}].");
            }

            for (int row = 0; row < fSize; row++)
            {
                float cy = (row + 0.5f) * stride;
                for (int col = 0; col < fSize; col++)
                {
                    float cx = (col + 0.5f) * stride;
                    for (int a = 0; a < 2; a++) // 2 anchors per cell
                    {
                        // The InsightFace SCRFD export already applies the sigmoid; applying it again
                        // maps every score to at least 0.5, so every anchor would pass the threshold.
                        float score = scoresIs3D ? scores[0, anchorIdx, 0] : scores[anchorIdx, 0];

                        if (score >= ScoreThreshold)
                        {
                            float d0 = bboxesIs3D ? bboxes[0, anchorIdx, 0] : bboxes[anchorIdx, 0];
                            float d1 = bboxesIs3D ? bboxes[0, anchorIdx, 1] : bboxes[anchorIdx, 1];
                            float d2 = bboxesIs3D ? bboxes[0, anchorIdx, 2] : bboxes[anchorIdx, 2];
                            float d3 = bboxesIs3D ? bboxes[0, anchorIdx, 3] : bboxes[anchorIdx, 3];
                            float x1 = cx - (d0 * stride);
                            float y1 = cy - (d1 * stride);
                            float x2 = cx + (d2 * stride);
                            float y2 = cy + (d3 * stride);
                            detections.Add(new Detection(x1, y1, x2, y2, score));
                        }

                        anchorIdx++;
                    }
                }
            }
        }

        return detections;
    }

    // Non-maximum suppression: handles multi-face scenarios by suppressing low-confidence
    // detections that overlap with higher-confidence ones. Caller (DetectPrimaryFaceAsync)
    // selects the highest-confidence face as the primary subject.
    private static List<Detection> ApplyNms(List<Detection> detections)
    {
        detections.Sort((a, b) => b.Score.CompareTo(a.Score));
        bool[] suppressed = new bool[detections.Count];
        var keep = new List<Detection>();

        for (int i = 0; i < detections.Count; i++)
        {
            if (suppressed[i]) continue;
            keep.Add(detections[i]);
            for (int j = i + 1; j < detections.Count; j++)
            {
                if (!suppressed[j] && Iou(detections[i], detections[j]) > NmsIouThreshold)
                    suppressed[j] = true;
            }
        }

        return keep;
    }

    private static float Iou(Detection a, Detection b)
    {
        float ix1 = Math.Max(a.X1, b.X1);
        float iy1 = Math.Max(a.Y1, b.Y1);
        float ix2 = Math.Min(a.X2, b.X2);
        float iy2 = Math.Min(a.Y2, b.Y2);

        float iw = Math.Max(0f, ix2 - ix1);
        float ih = Math.Max(0f, iy2 - iy1);
        float intersection = iw * ih;
        float aArea = (a.X2 - a.X1) * (a.Y2 - a.Y1);
        float bArea = (b.X2 - b.X1) * (b.Y2 - b.Y1);
        float union = aArea + bArea - intersection;
        return union <= 0f ? 0f : intersection / union;
    }

    private static FaceDetectionResult NoFace =>
        new(FaceFound: false, Confidence: 0d, PrimaryFace: null);

    internal readonly record struct Detection(float X1, float Y1, float X2, float Y2, float Score);
}
