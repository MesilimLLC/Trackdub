using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain.Media;
using Trackdub.Domain.Projects;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Tts;

namespace Trackdub.Sdk.Tests;

/// <summary>
/// Covers <see cref="TrackdubDubbingEngine.BuildUnattendedFallbackVoiceIds"/> — the headless
/// equivalent of the shell's fallback-voice flow for speakers without a voice assignment.
/// </summary>
public sealed class UnattendedFallbackVoiceTests
{
    private static readonly Guid ProjectId = Guid.NewGuid();

    [Fact]
    public void Picks_language_matching_voice_for_all_unassigned_speakers()
    {
        ProjectSpeaker speakerA = CreateSpeaker("Speaker 1");
        ProjectSpeaker speakerB = CreateSpeaker("Speaker 2");
        TranscriptProjectState state = BuildState(
            speakers: [speakerA, speakerB],
            availableVoices:
            [
                new VoiceCatalogEntry("am_adam", "en-us", "male", "Adam"),
                new VoiceCatalogEntry("ef_dora", "es", "female", "Dora"),
            ],
            voiceAssignments: []);

        Dictionary<Guid, string>? fallbackVoiceIds =
            TrackdubDubbingEngine.BuildUnattendedFallbackVoiceIds(state, "es");

        Assert.NotNull(fallbackVoiceIds);
        Assert.Equal(2, fallbackVoiceIds.Count);
        Assert.Equal("ef_dora", fallbackVoiceIds[speakerA.Id]);
        Assert.Equal("ef_dora", fallbackVoiceIds[speakerB.Id]);
    }

    [Fact]
    public void Skips_speakers_with_deliberate_voice_assignments()
    {
        ProjectSpeaker assigned = CreateSpeaker("Assigned");
        ProjectSpeaker unassigned = CreateSpeaker("Unassigned");
        TranscriptProjectState state = BuildState(
            speakers: [assigned, unassigned],
            availableVoices: [new VoiceCatalogEntry("ef_dora", "es", "female", "Dora")],
            voiceAssignments: [VoiceAssignment.Create(ProjectId, assigned.Id, "kokoro-onnx", "af_bella")]);

        Dictionary<Guid, string>? fallbackVoiceIds =
            TrackdubDubbingEngine.BuildUnattendedFallbackVoiceIds(state, "es");

        Assert.NotNull(fallbackVoiceIds);
        Guid speakerId = Assert.Single(fallbackVoiceIds).Key;
        Assert.Equal(unassigned.Id, speakerId);
    }

    [Fact]
    public void Prior_fallback_assignments_do_not_count_as_deliberate()
    {
        ProjectSpeaker speaker = CreateSpeaker("Speaker 1");
        TranscriptProjectState state = BuildState(
            speakers: [speaker],
            availableVoices: [new VoiceCatalogEntry("ef_dora", "es", "female", "Dora")],
            voiceAssignments: [VoiceAssignment.CreateFallback(ProjectId, speaker.Id, "kokoro-onnx", "af_bella")]);

        Dictionary<Guid, string>? fallbackVoiceIds =
            TrackdubDubbingEngine.BuildUnattendedFallbackVoiceIds(state, "es");

        Assert.NotNull(fallbackVoiceIds);
        Assert.Equal(speaker.Id, Assert.Single(fallbackVoiceIds).Key);
    }

    [Fact]
    public void Returns_null_when_no_stock_voice_matches_a_kokoro_language()
    {
        ProjectSpeaker speaker = CreateSpeaker("Speaker 1");
        TranscriptProjectState state = BuildState(
            speakers: [speaker],
            availableVoices: [new VoiceCatalogEntry("am_adam", "en-us", "male", "Adam")],
            voiceAssignments: []);

        Assert.Null(TrackdubDubbingEngine.BuildUnattendedFallbackVoiceIds(state, "es"));
    }

    [Fact]
    public void CustomVoice_spanish_assigns_qwen3_ryan_without_kokoro_catalog()
    {
        // Quality seed volumes ship CustomVoice only; Spanish catalog listing omits Qwen presets.
        ProjectSpeaker speaker = CreateSpeaker("Speaker 1");
        TranscriptProjectState state = BuildState(
            speakers: [speaker],
            availableVoices: [],
            voiceAssignments: []);

        Dictionary<Guid, string>? fallbackVoiceIds =
            TrackdubDubbingEngine.BuildUnattendedFallbackVoiceIds(
                state,
                "es",
                Qwen3TtsDefaults.CustomVoice17Alias);

        Assert.NotNull(fallbackVoiceIds);
        Assert.Equal("qwen3:ryan", fallbackVoiceIds[speaker.Id]);
    }

    [Fact]
    public void Chinese_default_stays_vivian_when_the_list_already_contains_every_qwen3_preset()
    {
        ProjectSpeaker speaker = CreateSpeaker("Speaker 1");
        TranscriptProjectState state = BuildState(
            speakers: [speaker],
            availableVoices:
            [
                new VoiceCatalogEntry("qwen3:aiden", "mul", "male", "Aiden (English)"),
                new VoiceCatalogEntry("qwen3:dylan", "mul", "male", "Dylan (Chinese, Beijing dialect)"),
                new VoiceCatalogEntry("qwen3:vivian", "mul", "female", "Vivian (Chinese)"),
            ],
            voiceAssignments: []);

        Dictionary<Guid, string>? fallbackVoiceIds =
            TrackdubDubbingEngine.BuildUnattendedFallbackVoiceIds(state, "zh");

        Assert.NotNull(fallbackVoiceIds);
        Assert.Equal("qwen3:vivian", fallbackVoiceIds[speaker.Id]);
    }

    [Theory]
    [InlineData("zh", "qwen3:vivian")]
    [InlineData("ja", "qwen3:ono_anna")]
    [InlineData("fr", "qwen3:ryan")]
    public void Uses_native_qwen3_preset_when_kokoro_has_no_voice_for_the_language(string targetLanguage, string expectedVoiceId)
    {
        ProjectSpeaker speaker = CreateSpeaker("Speaker 1");
        TranscriptProjectState state = BuildState(
            speakers: [speaker],
            availableVoices: [new VoiceCatalogEntry("am_adam", "en-us", "male", "Adam")],
            voiceAssignments: []);

        Dictionary<Guid, string>? fallbackVoiceIds =
            TrackdubDubbingEngine.BuildUnattendedFallbackVoiceIds(state, targetLanguage);

        Assert.NotNull(fallbackVoiceIds);
        Assert.Equal(expectedVoiceId, fallbackVoiceIds[speaker.Id]);
    }

    [Theory]
    [InlineData("nl")]
    [InlineData("tr")]
    public void Leaves_speakers_unassigned_when_neither_kokoro_nor_qwen3_speaks_the_language(string targetLanguage)
    {
        ProjectSpeaker speaker = CreateSpeaker("Speaker 1");
        TranscriptProjectState state = BuildState(
            speakers: [speaker],
            availableVoices: [new VoiceCatalogEntry("am_adam", "en-us", "male", "Adam")],
            voiceAssignments: []);

        Assert.Null(TrackdubDubbingEngine.BuildUnattendedFallbackVoiceIds(state, targetLanguage));
    }

    [Fact]
    public void Returns_null_when_every_speaker_already_has_a_deliberate_assignment()
    {
        ProjectSpeaker speaker = CreateSpeaker("Speaker 1");
        TranscriptProjectState state = BuildState(
            speakers: [speaker],
            availableVoices: [new VoiceCatalogEntry("ef_dora", "es", "female", "Dora")],
            voiceAssignments: [VoiceAssignment.Create(ProjectId, speaker.Id, "kokoro-onnx", "ef_dora")]);

        Assert.Null(TrackdubDubbingEngine.BuildUnattendedFallbackVoiceIds(state, "es"));
    }

    [Fact]
    public void Orders_matching_voices_by_display_name_like_the_shell_voice_picker()
    {
        ProjectSpeaker speaker = CreateSpeaker("Speaker 1");
        TranscriptProjectState state = BuildState(
            speakers: [speaker],
            availableVoices:
            [
                new VoiceCatalogEntry("ef_zoe", "es", "female", "Zoe"),
                new VoiceCatalogEntry("ef_ana", "es", "female", "Ana"),
            ],
            voiceAssignments: []);

        Dictionary<Guid, string>? fallbackVoiceIds =
            TrackdubDubbingEngine.BuildUnattendedFallbackVoiceIds(state, "es");

        Assert.NotNull(fallbackVoiceIds);
        Assert.Equal("ef_ana", fallbackVoiceIds[speaker.Id]);
    }

    [Fact]
    public void Region_variant_voice_matches_base_target_language()
    {
        ProjectSpeaker speaker = CreateSpeaker("Speaker 1");
        TranscriptProjectState state = BuildState(
            speakers: [speaker],
            availableVoices: [new VoiceCatalogEntry("ef_lat", "es-419", "female", "Latina")],
            voiceAssignments: []);

        Dictionary<Guid, string>? fallbackVoiceIds =
            TrackdubDubbingEngine.BuildUnattendedFallbackVoiceIds(state, "es");

        Assert.NotNull(fallbackVoiceIds);
        Assert.Equal("ef_lat", fallbackVoiceIds[speaker.Id]);
    }

    private static ProjectSpeaker CreateSpeaker(string displayName) =>
        new(Guid.NewGuid(), ProjectId, displayName, DateTimeOffset.UtcNow);

    private static TranscriptProjectState BuildState(
        IReadOnlyList<ProjectSpeaker> speakers,
        IReadOnlyList<VoiceCatalogEntry> availableVoices,
        IReadOnlyList<VoiceAssignment> voiceAssignments)
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        var project = new TrackdubProject(ProjectId, "Test Project", now, now);
        var mediaAsset = new MediaAsset(
            Guid.NewGuid(),
            ProjectId,
            "source.mp4",
            "source.mp4",
            "source-hash",
            100,
            now,
            "mp4",
            4.0d,
            HasAudio: true,
            HasVideo: true,
            now);
        var projectState = new OpenProjectResult(
            project,
            mediaAsset,
            null,
            SourceMediaStatus.Available,
            null,
            [],
            "en");
        TranscriptRevision transcriptRevision = TranscriptRevision.Create(
            ProjectId,
            stageRunId: null,
            revisionNumber: 1,
            now);

        return new TranscriptProjectState(
            projectState,
            transcriptRevision,
            [],
            speakers,
            [],
            null,
            [],
            IsTranslationStale: false,
            TranscriptLanguage: "en",
            StageRuns: [],
            SupportedTargetLanguages: [],
            SelectedTranslationTargetLanguage: "es",
            StaleTranslatedSegmentIndices: new HashSet<int>(),
            WaveformSummary: null,
            AvailableVoices: availableVoices,
            VoiceAssignments: voiceAssignments,
            TtsTakes: [],
            TtsSegmentStates: [],
            VoiceAssignmentWarnings: []);
    }
}
