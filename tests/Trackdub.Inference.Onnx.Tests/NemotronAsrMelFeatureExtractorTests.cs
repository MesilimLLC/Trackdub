using Trackdub.Inference.Onnx.NemotronAsr;

namespace Trackdub.Inference.Onnx.Tests;

// Reads the process-wide filterbank build counter, so it must not overlap any other
// class that constructs NemotronAsrMelFeatureExtractor (see MelGoldenTests).
[Collection("NemotronMelFilterBankBuildCount")]
public sealed class NemotronAsrMelFeatureExtractorTests
{
    [Fact]
    public void BuildChunk_reuses_the_single_filterbank_built_for_the_region()
    {
        // The engine builds one extractor (and therefore one mel filterbank) for a
        // transcription, and the decoder only assembles chunks from the finished mel.
        // Building a chunk must not construct another extractor/filterbank.
        int before = NemotronAsrMelFeatureExtractor.MelFilterBankBuildCountForTesting;
        var extractor = new NemotronAsrMelFeatureExtractor();
        _ = extractor.Extract(new float[1600]);

        var mel = new float[NemotronAsrMelFeatureExtractor.MelBins, 20];
        for (int bin = 0; bin < mel.GetLength(0); bin++)
        {
            for (int frame = 0; frame < mel.GetLength(1); frame++)
            {
                mel[bin, frame] = frame;
            }
        }

        float[] chunk = NemotronAsrMelFeatureExtractor.BuildChunk(
            mel, frameOffset: 10, mainFrameCount: 5, includePreEncodeCache: true);

        Assert.Equal(before + 1, NemotronAsrMelFeatureExtractor.MelFilterBankBuildCountForTesting);
        // First cache frame is source frame 1 (10 - 9 pre-encode frames), in
        // [bin, frame] layout: bin 0, frame 0 of the chunk.
        Assert.Equal(1f, chunk[0]);
        // First main frame follows the 9 cache frames and is source frame 10.
        Assert.Equal(10f, chunk[NemotronAsrMelFeatureExtractor.PreEncodeCacheFrames]);
    }
}
