using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Onnx.Audio;
using Trackdub.Inference.Onnx.SamAudio;

namespace Trackdub.Inference.Onnx.Tests.SamAudio;

public sealed class SamAudioStemOutputsTests
{
    /// <summary>WAV header field 'channels' sits at byte 22.</summary>
    private static int ChannelCount(string path) => BitConverter.ToUInt16(File.ReadAllBytes(path), 22);

    [Fact]
    public async Task A_stereo_bed_is_written_as_stereo_to_the_ambiance_music_and_raw_other_paths_while_speech_stays_mono()
    {
        string directory = Directory.CreateDirectory(Path.Join(Path.GetTempPath(), Guid.NewGuid().ToString("N"))).FullName;
        try
        {
            string Path_(string name) => Path.Join(directory, name);
            StemSeparationRequest request = new(
                "source.wav",
                Path_("vocals.wav"),
                Path_("ambiance.wav"),
                MusicOutputPath: Path_("music.wav"),
                RawStemOutputPaths: new Dictionary<string, string> { ["vocals"] = Path_("raw-vocals.wav"), ["other"] = Path_("raw-other.wav") });

            await SamAudioStemOutputs.WriteAsync(request, new float[100], [new float[100], new float[100]], CancellationToken.None);

            Assert.Equal(
                [1, 2, 2, 1, 2],
                new[] { "vocals.wav", "ambiance.wav", "music.wav", "raw-vocals.wav", "raw-other.wav" }.Select(name => ChannelCount(Path_(name))));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Channels_are_interleaved_frame_by_frame_as_16_bit_pcm()
    {
        byte[] wav = WaveAudioWriter.EncodePcm16([[0.5f, 0f], [-0.5f, 1f]], SamAudioModelConstants.SampleRate);

        short[] pcm = [.. Enumerable.Range(0, 4).Select(i => BitConverter.ToInt16(wav, 44 + (i * sizeof(short))))];

        Assert.Equal([16383, -16383, 0, 32767], pcm.Select(static v => (int)v));
    }

    [Fact]
    public void Ragged_or_empty_channels_are_rejected()
    {
        Assert.Throws<ArgumentException>(() => WaveAudioWriter.EncodePcm16([new float[3], new float[4]], SamAudioModelConstants.SampleRate));
        Assert.Throws<ArgumentException>(() => WaveAudioWriter.EncodePcm16([], SamAudioModelConstants.SampleRate));
    }
}
