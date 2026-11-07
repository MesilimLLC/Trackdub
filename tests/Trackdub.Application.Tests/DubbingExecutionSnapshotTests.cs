using Trackdub.Application.Dubbing;
using Trackdub.Contracts.Dubbing;
using Trackdub.Domain.Translation;

namespace Trackdub.Application.Tests;

public sealed class DubbingExecutionSnapshotTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CaptureExecutionSnapshot_records_speech_audio_preparation_choice(bool enabled)
    {
        var options = new DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "fr",
            EnableSpeechAudioPreparation = enabled,
        };

        Dictionary<string, string> snapshot = DubbingExecutionSnapshot.CaptureExecutionSnapshot(options);

        Assert.Equal(enabled.ToString(), snapshot["EnableSpeechAudioPreparation"]);
    }

    [Fact]
    public void CaptureExecutionSnapshot_records_translation_scope()
    {
        var missingOnly = new DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "fr",
            TranslationScope = TranslationRunScope.MissingOrOutdated,
        };
        var all = missingOnly with { TranslationScope = TranslationRunScope.All };

        Dictionary<string, string> missingSnapshot = DubbingExecutionSnapshot.CaptureExecutionSnapshot(missingOnly);
        Dictionary<string, string> allSnapshot = DubbingExecutionSnapshot.CaptureExecutionSnapshot(all);

        Assert.Equal(nameof(TranslationRunScope.MissingOrOutdated), missingSnapshot["TranslationScope"]);
        Assert.False(allSnapshot.ContainsKey("TranslationScope"));
    }

    [Fact]
    public void CaptureExecutionSnapshot_records_background_media_gain_for_export_resume()
    {
        var options = new DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "fr",
            ExportBackgroundMediaGainDb = -8d,
        };

        Dictionary<string, string> snapshot = DubbingExecutionSnapshot.CaptureExecutionSnapshot(options);

        Assert.Equal("-8", snapshot["ExportBackgroundMediaGainDb"]);
    }
}
