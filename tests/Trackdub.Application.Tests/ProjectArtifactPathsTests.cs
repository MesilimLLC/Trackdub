using Trackdub.Application.Projects;

namespace Trackdub.Application.Tests;

public sealed class ProjectArtifactPathsTests
{
    // ── GetTtsTakeRelativePath ────────────────────────────────────────────────

    [Fact]
    public void GetTtsTakeRelativePath_ValidInputs_ReturnsExpectedPath()
    {
        Guid speakerId = Guid.Parse("11111111-1111-1111-1111-111111111111");
        Guid segmentId = Guid.Parse("22222222-2222-2222-2222-222222222222");

        string path = ProjectArtifactPaths.GetTtsTakeRelativePath(speakerId, segmentId, takeNumber: 1);

        Assert.Equal(
            "artifacts/tts/11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222-take-0001.wav",
            path);
    }

    [Fact]
    public void GetTtsTakeRelativePath_EmptySpeakerId_Throws()
    {
        ArgumentException ex = Assert.Throws<ArgumentException>(
            () => ProjectArtifactPaths.GetTtsTakeRelativePath(Guid.Empty, Guid.NewGuid(), takeNumber: 1));

        Assert.Contains("speakerId", ex.ParamName, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void GetTtsTakeRelativePath_EmptySegmentId_Throws()
    {
        ArgumentException ex = Assert.Throws<ArgumentException>(
            () => ProjectArtifactPaths.GetTtsTakeRelativePath(Guid.NewGuid(), Guid.Empty, takeNumber: 1));

        Assert.Contains("segmentId", ex.ParamName, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void GetTtsTakeRelativePath_ZeroTakeNumber_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => ProjectArtifactPaths.GetTtsTakeRelativePath(Guid.NewGuid(), Guid.NewGuid(), takeNumber: 0));
    }

    [Fact]
    public void GetTtsTakeRelativePath_NegativeTakeNumber_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => ProjectArtifactPaths.GetTtsTakeRelativePath(Guid.NewGuid(), Guid.NewGuid(), takeNumber: -5));
    }

    [Theory]
    [InlineData(1, "0001")]
    [InlineData(9, "0009")]
    [InlineData(10, "0010")]
    [InlineData(999, "0999")]
    [InlineData(9999, "9999")]
    [InlineData(10000, "10000")]
    public void GetTtsTakeRelativePath_FormatsTheTakeNumberWithLeadingZeros(int takeNumber, string expectedPad)
    {
        Guid speakerId = Guid.NewGuid();
        Guid segmentId = Guid.NewGuid();

        string path = ProjectArtifactPaths.GetTtsTakeRelativePath(speakerId, segmentId, takeNumber);

        Assert.EndsWith($"-take-{expectedPad}.wav", path, StringComparison.Ordinal);
    }

    [Fact]
    public void GetTtsTakeRelativePath_PathStartsWithTtsDirectory()
    {
        string path = ProjectArtifactPaths.GetTtsTakeRelativePath(Guid.NewGuid(), Guid.NewGuid(), takeNumber: 1);

        Assert.StartsWith("artifacts/tts/", path, StringComparison.Ordinal);
    }

    [Fact]
    public void GetTtsTakeRelativePath_PathEndsWithWavExtension()
    {
        string path = ProjectArtifactPaths.GetTtsTakeRelativePath(Guid.NewGuid(), Guid.NewGuid(), takeNumber: 3);

        Assert.EndsWith(".wav", path, StringComparison.Ordinal);
    }

    [Fact]
    public void GetTtsTakeRelativePath_ContainsSpeakerIdInPath()
    {
        Guid speakerId = Guid.NewGuid();

        string path = ProjectArtifactPaths.GetTtsTakeRelativePath(speakerId, Guid.NewGuid(), takeNumber: 1);

        Assert.Contains(speakerId.ToString("D"), path, StringComparison.Ordinal);
    }

    [Fact]
    public void GetTtsTakeRelativePath_ContainsSegmentIdInFilename()
    {
        Guid segmentId = Guid.NewGuid();

        string path = ProjectArtifactPaths.GetTtsTakeRelativePath(Guid.NewGuid(), segmentId, takeNumber: 1);

        string filename = Path.GetFileName(path);
        Assert.StartsWith(segmentId.ToString("D"), filename, StringComparison.Ordinal);
    }

    // ── Constants ─────────────────────────────────────────────────────────────

    [Fact]
    public void TtsDirectoryRelativePath_HasExpectedValue()
    {
        Assert.Equal("artifacts/tts", ProjectArtifactPaths.TtsDirectoryRelativePath);
    }

    [Fact]
    public void RequiredDirectories_IncludesTtsDirectory()
    {
        Assert.Contains("artifacts/tts", ProjectArtifactPaths.RequiredDirectories);
    }

    [Fact]
    public void GetStemVocalsRelativePath_ValidStageRunId_ReturnsExpectedPath()
    {
        Guid stageRunId = Guid.Parse("33333333-3333-3333-3333-333333333333");

        string path = ProjectArtifactPaths.GetStemVocalsRelativePath(stageRunId);

        Assert.Equal("artifacts/stems/33333333-3333-3333-3333-333333333333/vocals.wav", path);
    }

    [Fact]
    public void GetEngineScopedStemRelativePaths_ReturnExpectedPaths()
    {
        Guid stageRunId = Guid.Parse("33333333-3333-3333-3333-333333333333");

        Assert.Equal(
            "artifacts/stems/33333333-3333-3333-3333-333333333333/spleeter/vocals.wav",
            ProjectArtifactPaths.GetStemVocalsRelativePath(stageRunId, "spleeter"));
        Assert.Equal(
            "artifacts/stems/33333333-3333-3333-3333-333333333333/spleeter/ambiance.wav",
            ProjectArtifactPaths.GetStemAmbianceRelativePath(stageRunId, "spleeter"));
        Assert.Equal(
            "artifacts/stems/33333333-3333-3333-3333-333333333333/spleeter/music.wav",
            ProjectArtifactPaths.GetStemMusicRelativePath(stageRunId, "spleeter"));
        Assert.Equal(
            "artifacts/stems/33333333-3333-3333-3333-333333333333/spleeter/sfx.wav",
            ProjectArtifactPaths.GetStemSoundEffectsRelativePath(stageRunId, "spleeter"));
    }

    [Fact]
    public void GetRawStemRelativePath_ReturnsEngineScopedSidecarPath()
    {
        Guid stageRunId = Guid.Parse("33333333-3333-3333-3333-333333333333");

        string path = ProjectArtifactPaths.GetRawStemRelativePath(stageRunId, "spleeter", "drums");

        Assert.Equal("artifacts/stems/33333333-3333-3333-3333-333333333333/spleeter/drums.wav", path);
    }

    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData("../spleeter")]
    [InlineData("spleeter/v2")]
    [InlineData("SpleeterV2")]
    public void EngineScopedStemPaths_InvalidEngineFamily_Throws(string engineFamily)
    {
        Guid stageRunId = Guid.NewGuid();

        Assert.Throws<ArgumentException>(() => ProjectArtifactPaths.GetStemVocalsRelativePath(stageRunId, engineFamily));
        Assert.Throws<ArgumentException>(() => ProjectArtifactPaths.GetRawStemRelativePath(stageRunId, engineFamily, "drums"));
    }

    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData("../drums")]
    [InlineData("drums/raw")]
    [InlineData("Drums")]
    public void GetRawStemRelativePath_InvalidStemName_Throws(string stemName)
    {
        Assert.Throws<ArgumentException>(() => ProjectArtifactPaths.GetRawStemRelativePath(Guid.NewGuid(), "spleeter", stemName));
    }

    [Fact]
    public void GetStemAmbianceRelativePath_ValidStageRunId_ReturnsExpectedPath()
    {
        Guid stageRunId = Guid.Parse("33333333-3333-3333-3333-333333333333");

        string path = ProjectArtifactPaths.GetStemAmbianceRelativePath(stageRunId);

        Assert.Equal("artifacts/stems/33333333-3333-3333-3333-333333333333/ambiance.wav", path);
    }

    [Fact]
    public void GetStemMusicRelativePath_ValidStageRunId_ReturnsExpectedPath()
    {
        Guid stageRunId = Guid.Parse("33333333-3333-3333-3333-333333333333");

        string path = ProjectArtifactPaths.GetStemMusicRelativePath(stageRunId);

        Assert.Equal("artifacts/stems/33333333-3333-3333-3333-333333333333/music.wav", path);
    }

    [Fact]
    public void GetStemSoundEffectsRelativePath_ValidStageRunId_ReturnsExpectedPath()
    {
        Guid stageRunId = Guid.Parse("33333333-3333-3333-3333-333333333333");

        string path = ProjectArtifactPaths.GetStemSoundEffectsRelativePath(stageRunId);

        Assert.Equal("artifacts/stems/33333333-3333-3333-3333-333333333333/sfx.wav", path);
    }

    [Fact]
    public void GetStemPaths_EmptyStageRunId_Throws()
    {
        Assert.Throws<ArgumentException>(() => ProjectArtifactPaths.GetStemVocalsRelativePath(Guid.Empty));
        Assert.Throws<ArgumentException>(() => ProjectArtifactPaths.GetStemAmbianceRelativePath(Guid.Empty));
        Assert.Throws<ArgumentException>(() => ProjectArtifactPaths.GetStemMusicRelativePath(Guid.Empty));
        Assert.Throws<ArgumentException>(() => ProjectArtifactPaths.GetStemSoundEffectsRelativePath(Guid.Empty));
    }

    [Fact]
    public void RequiredDirectories_IncludesStemsDirectory()
    {
        Assert.Equal("artifacts/stems", ProjectArtifactPaths.StemsDirectoryRelativePath);
        Assert.Contains("artifacts/stems", ProjectArtifactPaths.RequiredDirectories);
    }

    [Fact]
    public void GetSpeechEnhancedAudioRelativePath_ValidStageRunId_ReturnsExpectedPath()
    {
        Guid stageRunId = Guid.Parse("44444444-4444-4444-4444-444444444444");

        string path = ProjectArtifactPaths.GetSpeechEnhancedAudioRelativePath(stageRunId);

        Assert.Equal("artifacts/audio/speech-enhancement/44444444-4444-4444-4444-444444444444/speech.wav", path);
    }

    [Fact]
    public void GetSpeechEnhancedAudioRelativePath_EmptyStageRunId_Throws()
    {
        Assert.Throws<ArgumentException>(() => ProjectArtifactPaths.GetSpeechEnhancedAudioRelativePath(Guid.Empty));
    }

    [Fact]
    public void RequiredDirectories_IncludesSpeechEnhancedAudioDirectory()
    {
        Assert.Equal("artifacts/audio/speech-enhancement", ProjectArtifactPaths.SpeechEnhancedAudioDirectoryRelativePath);
        Assert.Contains("artifacts/audio/speech-enhancement", ProjectArtifactPaths.RequiredDirectories);
    }

    [Fact]
    public void GetAudioQualityAnalysisRelativePath_ValidStageRunId_ReturnsExpectedPath()
    {
        Guid stageRunId = Guid.Parse("55555555-5555-5555-5555-555555555555");

        string path = ProjectArtifactPaths.GetAudioQualityAnalysisRelativePath(stageRunId);

        Assert.Equal("artifacts/audio/quality/55555555-5555-5555-5555-555555555555/analysis.json", path);
    }

    [Fact]
    public void GetSpeechProcessedAudioRelativePath_ValidStageRunId_ReturnsExpectedPath()
    {
        Guid stageRunId = Guid.Parse("66666666-6666-6666-6666-666666666666");

        string path = ProjectArtifactPaths.GetSpeechProcessedAudioRelativePath(stageRunId, "asr");

        Assert.Equal("artifacts/audio/speech-processing/66666666-6666-6666-6666-666666666666/asr.wav", path);
    }

    [Fact]
    public void RequiredDirectories_IncludesAudioQualityAndProcessingDirectories()
    {
        Assert.Contains("artifacts/audio/quality", ProjectArtifactPaths.RequiredDirectories);
        Assert.Contains("artifacts/audio/speech-processing", ProjectArtifactPaths.RequiredDirectories);
    }

    [Fact]
    public void ResolveAbsolutePath_combines_project_root_with_normalized_relative_path()
    {
        string root = Path.Join(Path.GetTempPath(), "project.trackdub");

        string? path = ProjectArtifactPaths.ResolveAbsolutePath(root, "artifacts/tts/take.wav");

        Assert.Equal(Path.GetFullPath(Path.Join(root, "artifacts", "tts", "take.wav")), path);
    }

    [Theory]
    [InlineData(null, "artifacts/tts/take.wav")]
    [InlineData("", "artifacts/tts/take.wav")]
    [InlineData("D:\\Projects\\clip.trackdub", null)]
    [InlineData("D:\\Projects\\clip.trackdub", "")]
    public void ResolveAbsolutePath_returns_null_when_project_root_or_relative_path_is_missing(
        string? projectRootPath,
        string? relativePath)
    {
        Assert.Null(ProjectArtifactPaths.ResolveAbsolutePath(projectRootPath, relativePath));
    }
}
