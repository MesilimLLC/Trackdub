using Trackdub.Contracts;
using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Projects;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Trackdub.Domain.Tts;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed partial class OrchestrationServiceTests
{
    [Fact]
    public async Task TranslationOrchestrationService_GenerateTranslationAsync_persists_fake_engine_output()
    {
        FakeTranslationEngine translationEngine = new(
            static (request, segment) => $"{request.TargetLanguage}:{segment.Index}:{segment.Text}",
            static _ => new TranslationExecutionMetadata(
                "fake-runtime-provider",
                "fake-runtime-model",
                "fake-alias",
                "cpu",
                TranslationRoutingKind.Direct));
        TranslationServiceContext context = CreateTranslationServiceContext(translationEngine);
        TranscriptProjectState state = CreateTranscriptProjectState();

        await context.Service.GenerateTranslationAsync(
            state,
            new GenerateTranslationRequest("en", "es"),
            TestContext.Current.CancellationToken);

        TranslationRevision revision = Assert.Single(context.TranslationRepository.Revisions);
        Assert.Equal(1, revision.RevisionNumber);
        Assert.Equal("es", revision.TargetLanguage);
        Assert.Equal("fake-runtime-provider", revision.TranslationProvider);
        Assert.Equal("fake-runtime-model", revision.ModelId);
        Assert.Equal("cpu", revision.ExecutionProvider);

        IReadOnlyList<TranslatedSegment> segments = await context.TranslationRepository
            .GetSegmentsAsync(revision.Id, TestContext.Current.CancellationToken);
        Assert.Collection(
            segments,
            segment =>
            {
                Assert.Equal(0, segment.SegmentIndex);
                Assert.Equal("es:0:Hello there.", segment.Text);
                Assert.False(string.IsNullOrWhiteSpace(segment.SourceSegmentHash));
            },
            segment =>
            {
                Assert.Equal(1, segment.SegmentIndex);
                Assert.Equal("es:1:Second line.", segment.Text);
                Assert.False(string.IsNullOrWhiteSpace(segment.SourceSegmentHash));
            });

        StageRunRecord stageRun = Assert.Single(context.StageRunStore.All);
        Assert.Equal(StageNames.Translation, stageRun.StageName);
        Assert.Equal(StageRunStatus.Completed, stageRun.Status);
        Assert.Equal(stageRun.Id, revision.StageRunId);

        ProjectArtifact artifact = Assert.Single(
            context.MediaAssetRepository.Artifacts,
            artifact => artifact.Kind == ArtifactKind.TranslationRevision);
        Assert.True(context.ArtifactStore.Exists(artifact.RelativePath));
    }

    [Fact]
    public async Task TranslationOrchestrationService_GenerateTranslationAsync_uses_translated_word_alignment_when_available()
    {
        var translationEngine = new FakeTranslationEngine((_, segment) =>
            segment.Index == 0 ? "hola mundo" : "segunda linea");
        var aligner = new FakeTranslatedWordAlignmentService(request =>
            TranslatedWordAlignmentResult.Succeeded(
            [
                TranslatedWord.Create(0, request.TranslatedSegment.StartSeconds, request.TranslatedSegment.StartSeconds + 0.8d, request.TranslatedSegment.Text.Split(' ')[0]),
                TranslatedWord.Create(1, request.TranslatedSegment.StartSeconds + 0.8d, request.TranslatedSegment.EndSeconds, request.TranslatedSegment.Text.Split(' ')[1])
            ]));
        TranslationServiceContext context = CreateTranslationServiceContext(
            translationEngine,
            translatedWordAlignmentService: aligner);
        TranscriptProjectState state = CreateTranscriptProjectState();

        await context.Service.GenerateTranslationAsync(
            state,
            new GenerateTranslationRequest("en", "es"),
            TestContext.Current.CancellationToken);

        TranslationRevision revision = Assert.Single(context.TranslationRepository.Revisions);
        IReadOnlyList<TranslatedSegment> segments = await context.TranslationRepository
            .GetSegmentsAsync(revision.Id, TestContext.Current.CancellationToken);

        Assert.All(segments, segment => Assert.Equal(2, segment.Words.Count));
        Assert.Equal("hola", segments[0].Words[0].Text);
        Assert.Equal("mundo", segments[0].Words[1].Text);
    }

    [Fact]
    public async Task TranslationOrchestrationService_GenerateTranslationAsync_leaves_words_empty_when_alignment_is_unavailable()
    {
        var translationEngine = new FakeTranslationEngine((_, segment) =>
            segment.Index == 0 ? "hola mundo" : "segunda linea");
        var aligner = new FakeTranslatedWordAlignmentService(_ =>
            TranslatedWordAlignmentResult.Unavailable("aligner missing"));
        TranslationServiceContext context = CreateTranslationServiceContext(
            translationEngine,
            translatedWordAlignmentService: aligner);
        TranscriptProjectState state = CreateTranscriptProjectState();

        await context.Service.GenerateTranslationAsync(
            state,
            new GenerateTranslationRequest("en", "es"),
            TestContext.Current.CancellationToken);

        TranslationRevision revision = Assert.Single(context.TranslationRepository.Revisions);
        IReadOnlyList<TranslatedSegment> segments = await context.TranslationRepository
            .GetSegmentsAsync(revision.Id, TestContext.Current.CancellationToken);

        Assert.All(segments, segment => Assert.Empty(segment.Words));
    }

    [Fact]
    public async Task TranslationOrchestrationService_GenerateTranslationAsync_rejects_invalid_aligned_words()
    {
        var translationEngine = new FakeTranslationEngine((_, segment) =>
            segment.Index == 0 ? "hola mundo" : "segunda linea");
        var aligner = new FakeTranslatedWordAlignmentService(request =>
            TranslatedWordAlignmentResult.Succeeded(
            [
                TranslatedWord.Create(0, request.TranslatedSegment.StartSeconds, request.TranslatedSegment.EndSeconds, "adios")
            ]));
        TranslationServiceContext context = CreateTranslationServiceContext(
            translationEngine,
            translatedWordAlignmentService: aligner);
        TranscriptProjectState state = CreateTranscriptProjectState();

        await context.Service.GenerateTranslationAsync(
            state,
            new GenerateTranslationRequest("en", "es"),
            TestContext.Current.CancellationToken);

        TranslationRevision revision = Assert.Single(context.TranslationRepository.Revisions);
        IReadOnlyList<TranslatedSegment> segments = await context.TranslationRepository
            .GetSegmentsAsync(revision.Id, TestContext.Current.CancellationToken);

        Assert.All(segments, segment => Assert.Empty(segment.Words));
    }

    [Fact]
    public async Task TranslationOrchestrationService_GenerateTranslationAsync_passes_project_glossary_hints()
    {
        var translationEngine = new FakeTranslationEngine(static (_, segment) => segment.Text);
        var glossaryRepository = new FakeGlossaryRepository();
        TranslationServiceContext context = CreateTranslationServiceContext(translationEngine, glossaryRepository);
        TranscriptProjectState state = CreateTranscriptProjectState();
        await glossaryRepository.SaveAsync(
            GlossaryEntry.Create(
                state.ProjectState.Project.Id,
                "en",
                "es",
                "Hello",
                "Hola",
                isCaseSensitive: false,
                DateTimeOffset.UtcNow),
            TestContext.Current.CancellationToken);

        await context.Service.GenerateTranslationAsync(
            state,
            new GenerateTranslationRequest("en", "es"),
            TestContext.Current.CancellationToken);

        TranslationRevision revision = Assert.Single(context.TranslationRepository.Revisions);
        IReadOnlyList<TranslatedSegment> segments = await context.TranslationRepository
            .GetSegmentsAsync(revision.Id, TestContext.Current.CancellationToken);
        Assert.Equal("Hola there.", segments[0].Text);
    }

    [Fact]
    public async Task TranslationOrchestrationService_GenerateTranslationAsync_passes_only_matched_cjk_glossary_hints_with_spans()
    {
        IReadOnlyList<TranslationGlossaryHint>? observedHints = null;
        var translationEngine = new FakeTranslationEngine((request, segment) =>
        {
            observedHints = request.GlossaryHints;
            return segment.Text;
        });
        var glossaryRepository = new FakeGlossaryRepository();
        var translationRouter = new FakeTranslationLanguageRouter();
        translationRouter.SetSupportedTargetLanguages(
            "ja",
            new TranslationTargetLanguageOption("en", "English", TranslationRoutingKind.Direct, IsAvailable: true, "Direct fake route"));
        translationRouter.SetRoute(new TranslationRouteSelection(
            "ja",
            "en",
            TranslationRoutingKind.Direct,
            IsAvailable: true,
            "fake",
            "Direct fake route",
            "fake-ja-en",
            "fake-ja-en",
            EngineFamily: "fake"));
        TranslationServiceContext context = CreateTranslationServiceContext(
            translationEngine,
            glossaryRepository,
            translationRouter);
        TranscriptProjectState state = CreateTranscriptProjectState("ja");
        state = state with
        {
            TranscriptSegments =
            [
                TranscriptSegment.Create(
                    state.CurrentTranscriptRevision!.Id,
                    0,
                    0.0d,
                    2.0d,
                    "星宮先輩",
                    state.Speakers[0].Id,
                    "ja")
            ]
        };
        await glossaryRepository.SaveAsync(
            GlossaryEntry.Create(
                state.ProjectState.Project.Id,
                "ja",
                "en",
                "星宮",
                "Hoshimiya",
                isCaseSensitive: false,
                DateTimeOffset.UtcNow),
            TestContext.Current.CancellationToken);
        await glossaryRepository.SaveAsync(
            GlossaryEntry.Create(
                state.ProjectState.Project.Id,
                "ja",
                "en",
                "未登場",
                "unseen",
                isCaseSensitive: false,
                DateTimeOffset.UtcNow),
            TestContext.Current.CancellationToken);

        await context.Service.GenerateTranslationAsync(
            state,
            new GenerateTranslationRequest("ja", "en"),
            TestContext.Current.CancellationToken);

        TranslationRevision revision = Assert.Single(context.TranslationRepository.Revisions);
        IReadOnlyList<TranslatedSegment> segments = await context.TranslationRepository
            .GetSegmentsAsync(revision.Id, TestContext.Current.CancellationToken);
        Assert.Equal("Hoshimiya先輩", Assert.Single(segments).Text);

        TranslationGlossaryHint hint = Assert.Single(observedHints ?? []);
        Assert.Equal("星宮", hint.SourceTerm);
        TranslationGlossarySourceMatch match = Assert.Single(hint.SourceMatches ?? []);
        Assert.Equal(0, match.SegmentIndex);
        Assert.Equal(0, match.StartTextElementIndex);
        Assert.Equal(2, match.TextElementLength);
        Assert.Equal("星宮", match.MatchedSourceTerm);
    }

    [Fact]
    public async Task TranslationOrchestrationService_GenerateTranslationAsync_passes_only_normalized_latin_matches_with_spans()
    {
        IReadOnlyList<TranslationGlossaryHint>? observedHints = null;
        var translationEngine = new FakeTranslationEngine((request, segment) =>
        {
            observedHints = request.GlossaryHints;
            return segment.Text;
        });
        var glossaryRepository = new FakeGlossaryRepository();
        var translationRouter = new FakeTranslationLanguageRouter();
        translationRouter.SetSupportedTargetLanguages(
            "fr",
            new TranslationTargetLanguageOption("en", "English", TranslationRoutingKind.Direct, IsAvailable: true, "Direct fake route"));
        translationRouter.SetRoute(new TranslationRouteSelection(
            "fr",
            "en",
            TranslationRoutingKind.Direct,
            IsAvailable: true,
            "fake",
            "Direct fake route",
            "fake-fr-en",
            "fake-fr-en",
            EngineFamily: "fake"));
        TranslationServiceContext context = CreateTranslationServiceContext(
            translationEngine,
            glossaryRepository,
            translationRouter);
        TranscriptProjectState state = CreateTranscriptProjectState("fr");
        state = state with
        {
            TranscriptSegments =
            [
                TranscriptSegment.Create(
                    state.CurrentTranscriptRevision!.Id,
                    0,
                    0.0d,
                    2.0d,
                    "Le Café ouvre.",
                    state.Speakers[0].Id,
                    "fr")
            ]
        };
        await glossaryRepository.SaveAsync(
            GlossaryEntry.Create(
                state.ProjectState.Project.Id,
                "fr",
                "en",
                "cafe",
                "cafe glossary",
                isCaseSensitive: false,
                DateTimeOffset.UtcNow),
            TestContext.Current.CancellationToken);
        await glossaryRepository.SaveAsync(
            GlossaryEntry.Create(
                state.ProjectState.Project.Id,
                "fr",
                "en",
                "absent",
                "absent glossary",
                isCaseSensitive: false,
                DateTimeOffset.UtcNow),
            TestContext.Current.CancellationToken);

        await context.Service.GenerateTranslationAsync(
            state,
            new GenerateTranslationRequest("fr", "en"),
            TestContext.Current.CancellationToken);

        TranslationRevision revision = Assert.Single(context.TranslationRepository.Revisions);
        IReadOnlyList<TranslatedSegment> segments = await context.TranslationRepository
            .GetSegmentsAsync(revision.Id, TestContext.Current.CancellationToken);
        Assert.Equal("Le cafe glossary ouvre.", Assert.Single(segments).Text);

        TranslationGlossaryHint hint = Assert.Single(observedHints ?? []);
        Assert.Equal("cafe", hint.SourceTerm);
        TranslationGlossarySourceMatch match = Assert.Single(hint.SourceMatches ?? []);
        Assert.Equal(0, match.SegmentIndex);
        Assert.Equal(3, match.StartTextElementIndex);
        Assert.Equal(4, match.TextElementLength);
        Assert.Equal("Café", match.MatchedSourceTerm);
    }

    [Fact]
    public async Task TranslationOrchestrationService_RetranslateSegmentAsync_preserves_other_segment_words_and_clears_replaced_segment_words()
    {
        var translationEngine = new FakeTranslationEngine((_, segment) =>
            segment.Index == 0
                ? "Linea re-generada."
                : segment.Text);
        TranslationServiceContext context = CreateTranslationServiceContext(translationEngine);
        TranscriptProjectState currentState = CreateTranslatedProjectStateWithWords();

        context.TranslationRepository.Seed(currentState.CurrentTranslationRevision!, currentState.TranslatedSegments);

        await context.Service.RetranslateSegmentAsync(
            currentState,
            new RetranslateSegmentRequest(
                currentState.CurrentTranslationRevision!.Id,
                currentState.TranscriptSegments[0].Id,
                "en",
                "es"),
            TestContext.Current.CancellationToken);

        TranslationRevision savedRevision = Assert.Single(context.TranslationRepository.Revisions, revision => revision.RevisionNumber == 2);
        IReadOnlyList<TranslatedSegment> savedSegments = await context.TranslationRepository
            .GetSegmentsAsync(savedRevision.Id, TestContext.Current.CancellationToken);

        Assert.Collection(
            savedSegments.OrderBy(static segment => segment.SegmentIndex),
            changedSegment =>
            {
                Assert.Equal("Linea re-generada.", changedSegment.Text);
                Assert.Empty(changedSegment.Words);
            },
            unchangedSegment =>
            {
                Assert.Equal("Segunda linea.", unchangedSegment.Text);
                Assert.Collection(
                    unchangedSegment.Words,
                    first => Assert.Equal("Segunda", first.Text),
                    second => Assert.Equal("linea", second.Text));
            });
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_persists_fake_engine_take()
    {
        var ttsEngine = new FakeTtsEngine { SampleRate = 1000, DurationSamples = 1000 };
        TtsServiceContext context = CreateTtsServiceContext(ttsEngine);
        TranscriptProjectState state = CreateTranslatedProjectState();
        Guid speakerId = state.Speakers[0].Id;

        await context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId),
            TestContext.Current.CancellationToken);

        Assert.Equal(1, ttsEngine.SynthesizeCallCount);
        Assert.Equal("Linea traducida.", ttsEngine.LastInputText);
        Assert.Equal("af_heart", ttsEngine.LastVoicepack?.VoiceId);
        Assert.Equal("kokoro-onnx", ttsEngine.LastOptions?.PreferredModelAlias);
        Assert.True(ttsEngine.LastOptions?.RequirePreferredModelAlias);

        TtsTake take = Assert.Single(context.TtsTakeRepository.All);
        Assert.Equal(TtsTakeStatus.Completed, take.Status);
        Assert.Equal(0, take.SegmentIndex);
        Assert.Equal("fake", take.Provider);
        Assert.Equal("fake", take.ModelId);
        Assert.Equal("af_heart", take.VoiceId);
        Assert.Equal(TtsTakeKind.Stock, take.Kind);
        Assert.Null(take.ReferenceClipArtifactId);
        Assert.Equal(1000, take.DurationSamples);
        Assert.Equal(1000, take.SampleRate);

        ProjectArtifact artifact = Assert.Single(
            context.MediaAssetRepository.Artifacts,
            artifact => artifact.Kind == ArtifactKind.TtsTake);
        Assert.Equal(artifact.Id, take.ArtifactId);
        Assert.True(context.ArtifactStore.Exists(artifact.RelativePath));

        StageRunRecord stageRun = Assert.Single(context.StageRunStore.All);
        Assert.Equal(StageNames.Tts, stageRun.StageName);
        Assert.Equal(StageRunStatus.Completed, stageRun.Status);
        Assert.Equal(stageRun.Id, take.StageRunId);
    }

    [Fact]
    public void ApplyPresetVoiceModelSelection_points_preflight_at_qwen3_for_persisted_preset_assignments_without_overrides()
    {
        TranscriptProjectState state = CreateTranslatedProjectState();
        VoiceAssignment presetAssignment = state.VoiceAssignments[0] with
        {
            VoiceModelId = Qwen3TtsDefaults.CustomVoice06Alias,
            VoiceVariant = Qwen3TtsDefaults.PresetVoicePrefix + "ryan",
        };
        state = state with { VoiceAssignments = [presetAssignment] };
        var options = new Trackdub.Contracts.Dubbing.DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "es",
            AutoAssignFallbackVoices = true,
        };
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            new Dictionary<string, ExecutionProviderKind>(),
            TtsModelAlias: "kokoro-onnx");

        RuntimeModelSelections result = Trackdub.Application.Dubbing.UnattendedVoiceSelection
            .ApplyPresetVoiceModelSelection(selections, options, state);

        Assert.Equal(Qwen3TtsDefaults.ResolveCustomVoiceAlias(tier: null), result.TtsModelAlias);
    }

    [Fact]
    public void ApplyPresetVoiceModelSelection_rewrites_an_explicit_base_alias_when_voices_are_presets()
    {
        TranscriptProjectState state = CreateTranslatedProjectState();
        VoiceAssignment presetAssignment = state.VoiceAssignments[0] with
        {
            VoiceModelId = Qwen3TtsDefaults.Base06Alias,
            VoiceVariant = Qwen3TtsDefaults.PresetVoicePrefix + "ryan",
        };
        state = state with { VoiceAssignments = [presetAssignment] };
        var options = new Trackdub.Contracts.Dubbing.DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "es",
            AutoAssignFallbackVoices = true,
            ModelPreferences = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                [StageNames.Tts] = Qwen3TtsDefaults.Base06Alias,
            },
        };
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            new Dictionary<string, ExecutionProviderKind>(),
            TtsModelAlias: Qwen3TtsDefaults.Base06Alias);

        RuntimeModelSelections result = Trackdub.Application.Dubbing.UnattendedVoiceSelection
            .ApplyPresetVoiceModelSelection(selections, options, state);

        Assert.Equal(Qwen3TtsDefaults.ResolveCustomVoiceAlias(tier: null), result.TtsModelAlias);
    }

    [Fact]
    public void ApplyPresetVoiceModelSelection_rewrites_a_leftover_clone_alias_when_cloning_is_off()
    {
        TranscriptProjectState state = CreateTranslatedProjectState();
        VoiceAssignment presetAssignment = state.VoiceAssignments[0] with
        {
            VoiceModelId = Qwen3TtsDefaults.CustomVoice06Alias,
            VoiceVariant = Qwen3TtsDefaults.PresetVoicePrefix + "vivian",
        };
        state = state with { VoiceAssignments = [presetAssignment] };
        var options = new Trackdub.Contracts.Dubbing.DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "zh",
            UseVoiceCloning = false,
        };
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            new Dictionary<string, ExecutionProviderKind>(),
            TtsModelAlias: VoiceCloningDefaults.CosyVoicePrimaryAlias);

        RuntimeModelSelections result = Trackdub.Application.Dubbing.UnattendedVoiceSelection
            .ApplyPresetVoiceModelSelection(selections, options, state);

        Assert.Equal(Qwen3TtsDefaults.ResolveCustomVoiceAlias(tier: null), result.TtsModelAlias);
    }

    [Fact]
    public void ApplyPresetVoiceModelSelection_keeps_an_explicit_custom_voice_alias()
    {
        TranscriptProjectState state = CreateTranslatedProjectState();
        VoiceAssignment presetAssignment = state.VoiceAssignments[0] with
        {
            VoiceModelId = Qwen3TtsDefaults.CustomVoice17Alias,
            VoiceVariant = Qwen3TtsDefaults.PresetVoicePrefix + "ryan",
        };
        state = state with { VoiceAssignments = [presetAssignment] };
        var options = new Trackdub.Contracts.Dubbing.DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "es",
            ModelPreferences = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                [StageNames.Tts] = Qwen3TtsDefaults.CustomVoice17Alias,
            },
        };
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            new Dictionary<string, ExecutionProviderKind>(),
            TtsModelAlias: Qwen3TtsDefaults.CustomVoice17Alias);

        RuntimeModelSelections result = Trackdub.Application.Dubbing.UnattendedVoiceSelection
            .ApplyPresetVoiceModelSelection(selections, options, state);

        Assert.Equal(Qwen3TtsDefaults.CustomVoice17Alias, result.TtsModelAlias);
    }

    [Fact]
    public void ApplyPresetVoiceModelSelection_ignores_assignments_for_speakers_synthesis_skips()
    {
        TranscriptProjectState state = CreateTranslatedProjectState();
        Guid synthesizingSpeakerId = state.Speakers[0].Id;
        VoiceAssignment presetAssignment = state.VoiceAssignments[0] with
        {
            SpeakerId = synthesizingSpeakerId,
            VoiceModelId = Qwen3TtsDefaults.CustomVoice06Alias,
            VoiceVariant = Qwen3TtsDefaults.PresetVoicePrefix + "ryan",
        };
        VoiceAssignment unusedKokoro = VoiceAssignment.Create(
            state.ProjectState.Project.Id,
            Guid.NewGuid(),
            "kokoro-onnx",
            "af_heart");
        state = state with { VoiceAssignments = [presetAssignment, unusedKokoro] };
        var options = new Trackdub.Contracts.Dubbing.DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "es",
        };
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            new Dictionary<string, ExecutionProviderKind>(),
            TtsModelAlias: "kokoro-onnx");

        RuntimeModelSelections result = Trackdub.Application.Dubbing.UnattendedVoiceSelection
            .ApplyPresetVoiceModelSelection(selections, options, state);

        Assert.Equal(Qwen3TtsDefaults.ResolveCustomVoiceAlias(tier: null), result.TtsModelAlias);
    }

    [Fact]
    public void ApplyPresetVoiceModelSelection_keeps_kokoro_when_one_synthesized_speaker_is_not_a_preset()
    {
        TranscriptProjectState state = CreateTwoSpeakerTranslatedProjectState();
        Guid projectId = state.ProjectState.Project.Id;
        state = state with
        {
            VoiceAssignments =
            [
                VoiceAssignment.Create(
                    projectId,
                    state.Speakers[0].Id,
                    Qwen3TtsDefaults.CustomVoice06Alias,
                    Qwen3TtsDefaults.PresetVoicePrefix + "ryan"),
                VoiceAssignment.Create(projectId, state.Speakers[1].Id, "kokoro-onnx", "af_heart"),
            ],
        };
        var options = new Trackdub.Contracts.Dubbing.DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "es",
            VoiceAssignmentOverrides = new Dictionary<string, string>
            {
                [state.Speakers[0].DisplayName] = Qwen3TtsDefaults.PresetVoicePrefix + "ryan",
            },
        };
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            new Dictionary<string, ExecutionProviderKind>(),
            TtsModelAlias: "kokoro-onnx");

        RuntimeModelSelections result = Trackdub.Application.Dubbing.UnattendedVoiceSelection
            .ApplyPresetVoiceModelSelection(selections, options, state);

        Assert.Equal("kokoro-onnx", result.TtsModelAlias);
        Assert.True(Trackdub.Application.Dubbing.UnattendedVoiceSelection.RequiresCompanionCustomVoiceModel(
            selections,
            options,
            state));
    }

    [Fact]
    public void ResolveVoiceAssignmentOverrides_defers_whisper_style_keys_on_fresh_projects_without_speakers()
    {
        // Pre-flight resolves the voice plan against the fresh project state, before
        // diarization has created any speakers. Overrides must defer matching rather
        // than fail the run; the TTS stage re-resolves once speakers are known.
        TranscriptProjectState freshState = CreateTranscriptProjectState() with
        {
            Speakers = [],
            TranscriptSegments = [],
            VoiceAssignments = [],
        };
        var overrides = new Dictionary<string, string>
        {
            ["SPEAKER_00"] = "af_bella",
        };

        IReadOnlyDictionary<Guid, string> resolved = Trackdub.Application.Dubbing.UnattendedVoiceSelection
            .ResolveVoiceAssignmentOverrides(freshState, overrides);

        Assert.Empty(resolved);
    }

    [Fact]
    public void ApplyPresetVoiceModelSelection_tolerates_overrides_on_fresh_projects_without_speakers()
    {
        TranscriptProjectState freshState = CreateTranscriptProjectState() with
        {
            Speakers = [],
            TranscriptSegments = [],
            VoiceAssignments = [],
        };
        var options = new Trackdub.Contracts.Dubbing.DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "fr",
            VoiceAssignmentOverrides = new Dictionary<string, string>
            {
                ["SPEAKER_00"] = "af_bella",
            },
        };
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            new Dictionary<string, ExecutionProviderKind>(),
            TtsModelAlias: "kokoro-onnx");

        RuntimeModelSelections result = Trackdub.Application.Dubbing.UnattendedVoiceSelection
            .ApplyPresetVoiceModelSelection(selections, options, freshState);

        Assert.Equal("kokoro-onnx", result.TtsModelAlias);
        Assert.True(Trackdub.Application.Dubbing.UnattendedVoiceSelection.RequiresCompanionCustomVoiceModel(
            selections,
            options,
            freshState));
    }

    [Fact]
    public void ResolveVoiceAssignmentOverrides_still_rejects_unknown_keys_once_speakers_exist()
    {
        TranscriptProjectState state = CreateTranscriptProjectState();
        var overrides = new Dictionary<string, string>
        {
            ["SPEAKER_99"] = "af_bella",
        };

        InvalidOperationException exception = Assert.Throws<InvalidOperationException>(() =>
            Trackdub.Application.Dubbing.UnattendedVoiceSelection.ResolveVoiceAssignmentOverrides(state, overrides));
        Assert.Contains("SPEAKER_99", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ApplyPresetVoiceModelSelection_does_not_treat_a_leftover_clone_assignment_as_a_preset()
    {
        TranscriptProjectState state = CreateTwoSpeakerTranslatedProjectState();
        Guid projectId = state.ProjectState.Project.Id;
        state = state with
        {
            VoiceAssignments =
            [
                VoiceAssignment.Create(
                    projectId,
                    state.Speakers[0].Id,
                    Qwen3TtsDefaults.CustomVoice06Alias,
                    Qwen3TtsDefaults.PresetVoicePrefix + "ryan"),
                VoiceAssignment.Create(
                    projectId,
                    state.Speakers[1].Id,
                    VoiceCloningDefaults.CosyVoicePrimaryAlias,
                    voiceVariant: null),
            ],
        };
        var options = new Trackdub.Contracts.Dubbing.DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "es",
            UseVoiceCloning = false,
        };
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            new Dictionary<string, ExecutionProviderKind>(),
            TtsModelAlias: "kokoro-onnx");

        RuntimeModelSelections result = Trackdub.Application.Dubbing.UnattendedVoiceSelection
            .ApplyPresetVoiceModelSelection(selections, options, state);

        Assert.Equal("kokoro-onnx", result.TtsModelAlias);
        Assert.True(Trackdub.Application.Dubbing.UnattendedVoiceSelection.RequiresCompanionCustomVoiceModel(
            selections,
            options,
            state));
    }

    [Fact]
    public void ApplyPresetVoiceModelSelection_leaves_stock_alias_when_leftover_clone_language_is_unsupported()
    {
        TranscriptProjectState state = CreateTranslatedProjectState();
        state = state with
        {
            VoiceAssignments =
            [
                VoiceAssignment.Create(
                    state.ProjectState.Project.Id,
                    state.Speakers[0].Id,
                    VoiceCloningDefaults.CosyVoicePrimaryAlias,
                    voiceVariant: null),
            ],
        };
        var options = new Trackdub.Contracts.Dubbing.DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = "nl",
            UseVoiceCloning = false,
        };
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            new Dictionary<string, ExecutionProviderKind>(),
            TtsModelAlias: "kokoro-onnx");

        RuntimeModelSelections result = Trackdub.Application.Dubbing.UnattendedVoiceSelection
            .ApplyPresetVoiceModelSelection(selections, options, state);

        Assert.Equal("kokoro-onnx", result.TtsModelAlias);
        Assert.False(Trackdub.Application.Dubbing.UnattendedVoiceSelection.RequiresCompanionCustomVoiceModel(
            result,
            options,
            state));
    }

    [Theory]
    [InlineData("zh")]
    [InlineData("ja")]
    public void ApplyPresetVoiceModelSelection_retargets_leftover_clone_when_qwen3_speaks_the_language(string targetLanguage)
    {
        TranscriptProjectState state = CreateTranslatedProjectState();
        state = state with
        {
            VoiceAssignments =
            [
                VoiceAssignment.Create(
                    state.ProjectState.Project.Id,
                    state.Speakers[0].Id,
                    VoiceCloningDefaults.CosyVoicePrimaryAlias,
                    voiceVariant: null),
            ],
        };
        var options = new Trackdub.Contracts.Dubbing.DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = targetLanguage,
            UseVoiceCloning = false,
            AutoAssignFallbackVoices = true,
        };
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            new Dictionary<string, ExecutionProviderKind>(),
            TtsModelAlias: VoiceCloningDefaults.CosyVoicePrimaryAlias);

        RuntimeModelSelections result = Trackdub.Application.Dubbing.UnattendedVoiceSelection
            .ApplyPresetVoiceModelSelection(selections, options, state);

        Assert.Equal(Qwen3TtsDefaults.ResolveCustomVoiceAlias(tier: null), result.TtsModelAlias);
        Assert.False(Trackdub.Application.Dubbing.UnattendedVoiceSelection.RequiresCompanionCustomVoiceModel(
            result,
            options,
            state));
    }

    [Theory]
    [InlineData("zh", true)]
    [InlineData("es", false)]
    public void ApplyPresetVoiceModelSelection_points_preflight_at_qwen3_custom_voice_only_for_preset_runs(
        string targetLanguage,
        bool expectCustomVoice)
    {
        TranscriptProjectState state = CreateTranslatedProjectState() with { VoiceAssignments = [] };
        var options = new Trackdub.Contracts.Dubbing.DubbingSessionOptions
        {
            SourceMediaPath = "source.mp4",
            TargetLanguageCode = targetLanguage,
            AutoAssignFallbackVoices = true,
        };
        var selections = new RuntimeModelSelections(
            AsrModelOverride.Auto,
            IsDevBuild: false,
            new Dictionary<string, ExecutionProviderKind>(),
            TtsModelAlias: "kokoro-onnx");

        RuntimeModelSelections result = Trackdub.Application.Dubbing.UnattendedVoiceSelection
            .ApplyPresetVoiceModelSelection(selections, options, state);

        Assert.Equal(
            expectCustomVoice ? Qwen3TtsDefaults.ResolveCustomVoiceAlias(tier: null) : "kokoro-onnx",
            result.TtsModelAlias);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_refuses_clone_only_substitution_when_qwen3_cannot_speak_the_target()
    {
        TtsServiceContext context = CreateTtsServiceContext(
            new FakeTtsEngine { SampleRate = 1000, DurationSamples = 1000 },
            voiceCatalog: new FakeVoiceCatalog([new("af_heart", "mul", "female", "Heart")]));
        TranscriptProjectState spanishState = CreateTranslatedProjectState();
        TranslationRevision dutchRevision = spanishState.CurrentTranslationRevision! with { TargetLanguage = "nl" };
        Guid speakerId = spanishState.Speakers[0].Id;
        VoiceAssignment persistedClone = VoiceAssignment.Create(
            spanishState.ProjectState.Project.Id,
            speakerId,
            Qwen3TtsDefaults.Base06Alias,
            voiceVariant: null,
            requiresConsent: true,
            isFallback: false,
            referenceClipArtifactId: Guid.NewGuid());
        TranscriptProjectState state = spanishState with
        {
            CurrentTranslationRevision = dutchRevision,
            SelectedTranslationTargetLanguage = "nl",
            VoiceAssignments = [persistedClone]
        };

        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(() => context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId, UseReferenceClipForVoiceCloning: false),
            TestContext.Current.CancellationToken));
        Assert.Contains("Qwen3 does not speak target language 'nl'", error.Message, StringComparison.Ordinal);
        Assert.Contains("nl", error.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("no Kokoro voice matched", error.Message, StringComparison.Ordinal);
        Assert.Empty(context.VoiceAssignmentRepository.All);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_substitutes_qwen3_stock_for_persisted_qwen3_base_clone_on_non_kokoro_target()
    {
        // Regression for the clone-alias coverage gap: a persisted non-fallback Qwen3-base
        // (clone-only) assignment on a NON-clone run must be substituted with a stock voice
        // instead of throwing "Voicepack 'qwen3-tts-0.6b-base' is not available.". The target
        // language is Japanese (SupportsKokoro == false), which exercises the untested non-Kokoro
        // substitution branch resolving the Qwen3 fallback (custom-voice) alias.
        var ttsEngine = new FakeTtsEngine { SampleRate = 1000, DurationSamples = 1000 };
        // Production's IVoiceCatalog is Kokoro-only; it does NOT contain the Qwen3 custom-voice
        // alias. On this non-Kokoro target the substitution routes through the synthetic
        // native Qwen3 preset voice (no catalog lookup), so the catalog deliberately omits the alias.
        // This fake catalog also omits the Qwen3 presets, exercising the synthetic-entry fallback;
        // production's composite catalog resolves those IDs.
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            voiceCatalog: new FakeVoiceCatalog(
            [
                new("af_heart", "mul", "female", "Heart"),
                new("am_adam", "mul", "male", "Adam"),
                new("bf_alice", "en-gb", "female", "Alice"),
            ]));
        TranscriptProjectState spanishState = CreateTranslatedProjectState();
        TranslationRevision japaneseRevision = spanishState.CurrentTranslationRevision! with { TargetLanguage = "ja" };
        Guid projectId = spanishState.ProjectState.Project.Id;
        Guid speakerId = spanishState.Speakers[0].Id;
        VoiceAssignment persistedClone = VoiceAssignment.Create(
            projectId,
            speakerId,
            Qwen3TtsDefaults.Base06Alias,
            voiceVariant: null,
            requiresConsent: true,
            isFallback: false,
            referenceClipArtifactId: Guid.NewGuid());
        TranscriptProjectState state = spanishState with
        {
            CurrentTranslationRevision = japaneseRevision,
            SelectedTranslationTargetLanguage = "ja",
            VoiceAssignments = [persistedClone]
        };

        await context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId, UseReferenceClipForVoiceCloning: false),
            TestContext.Current.CancellationToken);

        VoiceAssignment substituted = Assert.Single(context.VoiceAssignmentRepository.All);
        Assert.True(substituted.IsFallback);
        Assert.Equal(Qwen3TtsDefaults.ResolveCustomVoiceAlias(tier: null), substituted.VoiceModelId);
        Assert.Null(substituted.VoiceVariant);
        Assert.Null(substituted.ReferenceClipArtifactId);
        Assert.False(VoiceCloningDefaults.IsCloneOnlyModelAlias(substituted.VoiceModelId));
        TtsTake take = Assert.Single(context.TtsTakeRepository.All);
        Assert.Equal(TtsTakeStatus.Completed, take.Status);
        Assert.Equal(TtsTakeKind.Stock, take.Kind);
        // The non-Kokoro clone-only substitution leaves PreferredModelAlias blank so
        // StartTtsStageHandler routes to the Qwen3 preset whose native language matches the
        // Japanese target (Ono Anna) rather than a Kokoro-only catalog lookup that would throw.
        Assert.Equal("qwen3:ono_anna", take.VoiceId);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_substitutes_stock_for_persisted_f5_clone_on_non_clone_run()
    {
        // Regression for the F5 half of the clone-alias coverage gap: a persisted non-fallback
        // F5 (clone-only) assignment on a NON-clone run must be substituted with a stock voice
        // rather than throwing "Voicepack 'f5tts-onnx' is not available.". The target language is
        // Spanish (SupportsKokoro == true), so a Kokoro voice is chosen via PickClosest.
        var ttsEngine = new FakeTtsEngine { SampleRate = 1000, DurationSamples = 1000 };
        TtsServiceContext context = CreateTtsServiceContext(ttsEngine);
        TranscriptProjectState baseState = CreateTranslatedProjectState();
        Guid projectId = baseState.ProjectState.Project.Id;
        Guid speakerId = baseState.Speakers[0].Id;
        VoiceAssignment persistedClone = VoiceAssignment.Create(
            projectId,
            speakerId,
            "f5tts-onnx",
            voiceVariant: null,
            requiresConsent: true,
            isFallback: false,
            referenceClipArtifactId: Guid.NewGuid());
        TranscriptProjectState state = baseState with { VoiceAssignments = [persistedClone] };

        await context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId, UseReferenceClipForVoiceCloning: false),
            TestContext.Current.CancellationToken);

        VoiceAssignment substituted = Assert.Single(context.VoiceAssignmentRepository.All);
        Assert.True(substituted.IsFallback);
        Assert.Equal(StockTtsDefaults.KokoroPrimaryAlias, substituted.VoiceModelId);
        Assert.False(VoiceCloningDefaults.IsCloneOnlyModelAlias(substituted.VoiceModelId));
        Assert.Null(substituted.ReferenceClipArtifactId);
        Assert.NotNull(substituted.VoiceVariant);
        TtsTake take = Assert.Single(context.TtsTakeRepository.All);
        Assert.Equal(TtsTakeStatus.Completed, take.Status);
        Assert.Equal(TtsTakeKind.Stock, take.Kind);
        Assert.Contains(take.VoiceId, new[] { "af_heart", "am_adam", "bf_alice" });
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_skips_orphaned_artifact_path()
    {
        var ttsEngine = new FakeTtsEngine { SampleRate = 1000, DurationSamples = 1000 };
        TtsServiceContext context = CreateTtsServiceContext(ttsEngine);
        TranscriptProjectState state = CreateTranslatedProjectState();
        Guid speakerId = state.Speakers[0].Id;
        Guid translatedSegmentId = state.TranslatedSegments[0].Id;
        string orphanedPath = ProjectArtifactPaths.GetTtsTakeRelativePath(
            speakerId,
            translatedSegmentId,
            takeNumber: 1);
        await context.MediaAssetRepository.SaveArtifactAsync(
            new ProjectArtifact(
                Guid.NewGuid(),
                state.ProjectState.Project.Id,
                state.ProjectState.MediaAsset!.Id,
                ArtifactKind.TtsTake,
                orphanedPath,
                "orphaned-sha",
                4,
                1.0d,
                1000,
                1,
                DateTimeOffset.UtcNow),
            TestContext.Current.CancellationToken);

        await context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId),
            TestContext.Current.CancellationToken);

        TtsTake take = Assert.Single(context.TtsTakeRepository.All);
        ProjectArtifact artifact = Assert.Single(
            context.MediaAssetRepository.Artifacts,
            artifact => artifact.Id == take.ArtifactId);
        Assert.Equal(
            ProjectArtifactPaths.GetTtsTakeRelativePath(speakerId, translatedSegmentId, takeNumber: 2)[..^4] + $"-{take.Id:N}.wav",
            artifact.RelativePath);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_requires_voice_clone_consent()
    {
        var consent = new FakeConsentService();
        var ttsEngine = new FakeVoiceCloneTtsEngine(consent);
        var auditLog = new FakeAuditLog();
        using var workspace = new TemporaryTestWorkspace();
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            consent,
            auditLog,
            new FakeReferenceClipAnalyzer(),
            workspace.Root);
        (TranscriptProjectState state, Guid speakerId, _) = CreateVoiceClonedTranslatedProjectState(context.ArtifactStore);
        state = AddNormalizedAudioArtifact(state, context.ArtifactStore);

        await Assert.ThrowsAsync<ConsentRequiredException>(() =>
            context.Service.GenerateTtsForSpeakerAsync(
                state,
                new GenerateTtsForSpeakerRequest(speakerId, UseReferenceClipForVoiceCloning: true),
                TestContext.Current.CancellationToken));

        Assert.Equal(0, ttsEngine.SynthesizeCallCount);
        Assert.Empty(auditLog.Entries);
        Assert.Empty(context.TtsTakeRepository.All);
        Assert.DoesNotContain(context.MediaAssetRepository.Artifacts, artifact => artifact.Kind == ArtifactKind.TtsTake);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_records_cloned_take_and_audit_entry()
    {
        var consent = new FakeConsentService();
        consent.GrantVoiceCloningConsent();
        var ttsEngine = new FakeVoiceCloneTtsEngine(consent) { SampleRate = 1000, DurationSamples = 1000 };
        var auditLog = new FakeAuditLog();
        using var workspace = new TemporaryTestWorkspace();
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            consent,
            auditLog,
            new FakeReferenceClipAnalyzer(),
            workspace.Root);
        (TranscriptProjectState state, Guid speakerId, Guid referenceClipArtifactId) =
            CreateVoiceClonedTranslatedProjectState(context.ArtifactStore);

        await context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId, UseReferenceClipForVoiceCloning: true),
            TestContext.Current.CancellationToken);

        TtsTake take = Assert.Single(context.TtsTakeRepository.All);
        Assert.Equal(TtsTakeKind.VoiceCloned, take.Kind);
        Assert.Equal(referenceClipArtifactId, take.ReferenceClipArtifactId);
        Assert.Equal(referenceClipArtifactId, ttsEngine.LastReferenceClipArtifactId);
        Assert.Equal(2.0d, ttsEngine.LastRequest?.TargetDurationSeconds);

        VoiceCloneAuditEntry entry = Assert.Single(auditLog.Entries);
        Assert.Equal(speakerId, entry.SpeakerId);
        Assert.Equal(referenceClipArtifactId, entry.ReferenceClipArtifactId);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_auto_captures_reference_for_clone_without_reference_clip()
    {
        var consent = new FakeConsentService();
        consent.GrantVoiceCloningConsent();
        var ttsEngine = new FakeVoiceCloneTtsEngine(consent) { SampleRate = 1000, DurationSamples = 1000 };
        var auditLog = new FakeAuditLog();
        var analyzer = new FakeReferenceClipAnalyzer
        {
            Analysis = new ReferenceClipAnalysis(
                TotalDurationSeconds: 4d,
                ActiveSpeechSeconds: 4d,
                SampleRate: 24000,
                ChannelCount: 1)
        };
        using var workspace = new TemporaryTestWorkspace();
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            consent,
            auditLog,
            analyzer,
            workspace.Root);
        TranscriptProjectState state = AddNormalizedAudioArtifact(
            CreateTranslatedProjectState(),
            context.ArtifactStore);
        Guid speakerId = state.Speakers[0].Id;

        await context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId, UseReferenceClipForVoiceCloning: true),
            TestContext.Current.CancellationToken);

        TtsTake take = Assert.Single(context.TtsTakeRepository.All);
        Assert.Equal(TtsTakeKind.VoiceCloned, take.Kind);
        Assert.NotNull(take.ReferenceClipArtifactId);
        Assert.Equal(take.ReferenceClipArtifactId, ttsEngine.LastReferenceClipArtifactId);
        ProjectArtifact referenceArtifact = Assert.Single(
            context.MediaAssetRepository.Artifacts,
            artifact => artifact.Kind == ArtifactKind.ReferenceClip);
        Assert.Equal(take.ReferenceClipArtifactId, referenceArtifact.Id);
        Assert.Contains("auto-speaker-reference:v2", referenceArtifact.Provenance);
        Assert.Contains("mode:single", referenceArtifact.Provenance);
        Assert.Equal(1, context.ReferenceClipTrimmer.TrimCallCount);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForAllSpeakersAsync_request_voice_override_wins_over_existing_assignment()
    {
        var ttsEngine = new FakeTtsEngine { SampleRate = 1000, DurationSamples = 1000 };
        TtsServiceContext context = CreateTtsServiceContext(ttsEngine);
        // CreateTranslatedProjectState seeds a pre-existing non-fallback assignment of "af_heart".
        TranscriptProjectState state = CreateTranslatedProjectState();
        Guid speakerId = state.Speakers[0].Id;
        VoiceAssignment seededAssignment = state.VoiceAssignments.Single();
        context.VoiceAssignmentRepository.Seed(seededAssignment);

        await context.Service.GenerateTtsForAllSpeakersAsync(
            state,
            new GenerateTtsForAllSpeakersRequest(
                VoiceIdsBySpeakerId: new Dictionary<Guid, string>
                {
                    [speakerId] = "am_adam"
                }),
            TestContext.Current.CancellationToken);

        // The explicit request override must win over the pre-existing "af_heart" assignment.
        Assert.Equal("am_adam", ttsEngine.LastVoicepack?.VoiceId);
        TtsTake take = Assert.Single(context.TtsTakeRepository.All);
        Assert.Equal("am_adam", take.VoiceId);
        Assert.Equal(TtsTakeKind.Stock, take.Kind);
        VoiceAssignment assignment = Assert.Single(
            context.VoiceAssignmentRepository.All,
            candidate => candidate.SpeakerId == speakerId);
        Assert.Equal(seededAssignment.Id, assignment.Id);
        Assert.Equal("am_adam", assignment.VoiceVariant);
        Assert.False(assignment.IsFallback);
        Assert.Single(context.VoiceAssignmentRepository.All);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForAllSpeakersAsync_keeps_existing_assignment_when_no_override_supplied()
    {
        var ttsEngine = new FakeTtsEngine { SampleRate = 1000, DurationSamples = 1000 };
        TtsServiceContext context = CreateTtsServiceContext(ttsEngine);
        // CreateTranslatedProjectState seeds a pre-existing non-fallback assignment of "af_heart".
        TranscriptProjectState state = CreateTranslatedProjectState();
        Guid speakerId = state.Speakers[0].Id;

        await context.Service.GenerateTtsForAllSpeakersAsync(
            state,
            new GenerateTtsForAllSpeakersRequest(),
            TestContext.Current.CancellationToken);

        // With no request override, the pre-existing non-fallback assignment is honored.
        Assert.Equal("af_heart", ttsEngine.LastVoicepack?.VoiceId);
        TtsTake take = Assert.Single(context.TtsTakeRepository.All);
        Assert.Equal("af_heart", take.VoiceId);
        Assert.Equal(TtsTakeKind.Stock, take.Kind);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForAllSpeakersAsync_applies_request_override_per_speaker_independently()
    {
        var ttsEngine = new FakeTtsEngine { SampleRate = 1000, DurationSamples = 1000 };
        TtsServiceContext context = CreateTtsServiceContext(ttsEngine);
        TranscriptProjectState state = CreateTwoSpeakerTranslatedProjectState();
        Guid projectId = state.ProjectState.Project.Id;
        Guid speakerA = state.Speakers[0].Id;
        Guid speakerB = state.Speakers[1].Id;
        VoiceAssignment seededAssignmentB = VoiceAssignment.Create(projectId, speakerB, "kokoro-onnx", "af_heart");
        context.VoiceAssignmentRepository.Seed(seededAssignmentB);
        // Speaker B relies on a pre-existing non-fallback assignment; speaker A gets an explicit
        // request override. The per-speaker precedence must be independent: A synthesizes/persists
        // its override while B keeps its untouched pre-existing assignment.
        state = state with
        {
            VoiceAssignments =
            [
                seededAssignmentB
            ]
        };

        await context.Service.GenerateTtsForAllSpeakersAsync(
            state,
            new GenerateTtsForAllSpeakersRequest(
                VoiceIdsBySpeakerId: new Dictionary<Guid, string>
                {
                    [speakerA] = "am_adam"
                }),
            TestContext.Current.CancellationToken);

        Assert.Equal(2, context.TtsTakeRepository.All.Count);
        TtsTake takeA = Assert.Single(
            context.TtsTakeRepository.All,
            take => state.TranscriptSegments.Single(segment => segment.SegmentIndex == take.SegmentIndex).SpeakerId == speakerA);
        TtsTake takeB = Assert.Single(
            context.TtsTakeRepository.All,
            take => state.TranscriptSegments.Single(segment => segment.SegmentIndex == take.SegmentIndex).SpeakerId == speakerB);
        // Speaker A synthesized the override voice; speaker B synthesized its pre-existing assignment.
        Assert.Equal("am_adam", takeA.VoiceId);
        Assert.Equal(TtsTakeKind.Stock, takeA.Kind);
        Assert.Equal("af_heart", takeB.VoiceId);
        Assert.Equal(TtsTakeKind.Stock, takeB.Kind);

        // Speaker A persisted the override as a non-fallback assignment.
        VoiceAssignment assignmentA = Assert.Single(
            context.VoiceAssignmentRepository.All,
            assignment => assignment.SpeakerId == speakerA);
        Assert.Equal("am_adam", assignmentA.VoiceVariant);
        Assert.False(assignmentA.IsFallback);

        // Speaker B's pre-existing assignment is left untouched: only speaker A's override is
        // (re)persisted, so no new assignment row is written for speaker B and the af_heart take
        // above confirms B still synthesized with its original assignment.
        VoiceAssignment assignmentB = Assert.Single(
            context.VoiceAssignmentRepository.All,
            assignment => assignment.SpeakerId == speakerB);
        Assert.Equal(seededAssignmentB.Id, assignmentB.Id);
        Assert.Equal("af_heart", assignmentB.VoiceVariant);
        Assert.Equal(2, context.VoiceAssignmentRepository.All.Count);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForAllSpeakersAsync_fallback_request_mints_new_assignment_when_non_fallback_row_exists()
    {
        var ttsEngine = new FakeTtsEngine { SampleRate = 1000, DurationSamples = 1000 };
        TtsServiceContext context = CreateTtsServiceContext(ttsEngine);
        // Resume scenario: a non-fallback assignment was persisted on a prior run and lives in the
        // repository, but it is not surfaced through the in-memory state map (VoiceAssignments is
        // empty). With only a fallback voice supplied, TryPersistRequestedVoiceAssignmentAsync's
        // fallback branch runs; GetAsync returns the seeded non-fallback row. Previously, the
        // branch reused that row's primary-key Id with IsFallback=true, which collides with the
        // existing PRIMARY KEY on a real SQLite-backed save (the fake now mirrors that conflict).
        TranscriptProjectState state = CreateTranslatedProjectState() with { VoiceAssignments = [] };
        Guid speakerId = state.Speakers[0].Id;
        VoiceAssignment seededAssignment = VoiceAssignment.Create(
            state.ProjectState.Project.Id,
            speakerId,
            "kokoro-onnx",
            "af_heart");
        context.VoiceAssignmentRepository.Seed(seededAssignment);

        await context.Service.GenerateTtsForAllSpeakersAsync(
            state,
            new GenerateTtsForAllSpeakersRequest(
                FallbackVoiceIdsBySpeakerId: new Dictionary<Guid, string>
                {
                    [speakerId] = "am_adam"
                }),
            TestContext.Current.CancellationToken);

        // The pre-existing non-fallback row survives untouched.
        VoiceAssignment survivingNonFallback = Assert.Single(
            context.VoiceAssignmentRepository.All,
            candidate => candidate.SpeakerId == speakerId && !candidate.IsFallback);
        Assert.Equal(seededAssignment.Id, survivingNonFallback.Id);
        Assert.Equal("af_heart", survivingNonFallback.VoiceVariant);

        // The fallback assignment was minted with a brand-new Id (never the seeded row's Id).
        VoiceAssignment fallback = Assert.Single(
            context.VoiceAssignmentRepository.All,
            candidate => candidate.SpeakerId == speakerId && candidate.IsFallback);
        Assert.True(fallback.IsFallback);
        Assert.NotEqual(seededAssignment.Id, fallback.Id);
        Assert.Equal("am_adam", fallback.VoiceVariant);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForAllSpeakersAsync_auto_captures_clone_without_voice_assignment()
    {
        var consent = new FakeConsentService();
        consent.GrantVoiceCloningConsent();
        var ttsEngine = new FakeVoiceCloneTtsEngine(consent) { SampleRate = 1000, DurationSamples = 1000 };
        var analyzer = new FakeReferenceClipAnalyzer
        {
            Analysis = new ReferenceClipAnalysis(
                TotalDurationSeconds: 4d,
                ActiveSpeechSeconds: 4d,
                SampleRate: 24000,
                ChannelCount: 1)
        };
        using var workspace = new TemporaryTestWorkspace();
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            consent,
            new FakeAuditLog(),
            analyzer,
            workspace.Root);
        TranscriptProjectState state = AddNormalizedAudioArtifact(
            CreateTranslatedProjectState() with { VoiceAssignments = [] },
            context.ArtifactStore);
        Guid speakerId = state.Speakers[0].Id;

        await context.Service.GenerateTtsForAllSpeakersAsync(
            state,
            new GenerateTtsForAllSpeakersRequest(
                UseReferenceClipForVoiceCloningBySpeakerId: new Dictionary<Guid, bool>
                {
                    [speakerId] = true
                }),
            TestContext.Current.CancellationToken);

        TtsTake take = Assert.Single(context.TtsTakeRepository.All);
        Assert.Equal(TtsTakeKind.VoiceCloned, take.Kind);
        Assert.NotNull(take.ReferenceClipArtifactId);
        VoiceAssignment assignment = Assert.Single(context.VoiceAssignmentRepository.All);
        Assert.Equal(VoiceCloningDefaults.ChatterboxPrimaryAlias, assignment.VoiceModelId);
        Assert.True(assignment.RequiresConsent);
        Assert.False(assignment.IsFallback);
        Assert.Equal(take.ReferenceClipArtifactId, assignment.ReferenceClipArtifactId);
        Assert.Equal(take.ReferenceClipArtifactId, ttsEngine.LastReferenceClipArtifactId);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForAllSpeakersAsync_falls_back_to_gender_matched_kokoro_when_clone_reference_is_too_short()
    {
        var consent = new FakeConsentService();
        consent.GrantVoiceCloningConsent();
        var ttsEngine = new FakeVoiceCloneTtsEngine(consent) { SampleRate = 1000, DurationSamples = 1000 };
        var analyzer = new FakeReferenceClipAnalyzer();
        analyzer.QueueAnalysis(new ReferenceClipAnalysis(2.66d, 2.66d, 24000, 1, "male"));
        analyzer.QueueAnalysis(new ReferenceClipAnalysis(2.66d, 2.66d, 24000, 1, "male"));
        using var workspace = new TemporaryTestWorkspace();
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            consent,
            new FakeAuditLog(),
            analyzer,
            workspace.Root);
        TranscriptProjectState state = AddNormalizedAudioArtifact(
            CreateTranslatedProjectState() with { VoiceAssignments = [] },
            context.ArtifactStore);
        Guid speakerId = state.Speakers[0].Id;

        await context.Service.GenerateTtsForAllSpeakersAsync(
            state,
            new GenerateTtsForAllSpeakersRequest(
                UseReferenceClipForVoiceCloningBySpeakerId: new Dictionary<Guid, bool>
                {
                    [speakerId] = true
                }),
            TestContext.Current.CancellationToken);

        TtsTake take = Assert.Single(context.TtsTakeRepository.All);
        Assert.Equal(TtsTakeKind.Stock, take.Kind);
        Assert.Null(take.ReferenceClipArtifactId);
        Assert.Equal("am_adam", take.VoiceId);
        VoiceAssignment assignment = Assert.Single(context.VoiceAssignmentRepository.All);
        Assert.True(assignment.IsFallback);
        Assert.Equal(StockTtsDefaults.KokoroPrimaryAlias, assignment.VoiceModelId);
        Assert.Equal("am_adam", assignment.VoiceVariant);
        Assert.Equal("am_adam", ttsEngine.LastVoicepack?.VoiceId);
        Assert.Null(ttsEngine.LastReferenceClipArtifactId);
        Assert.DoesNotContain(context.MediaAssetRepository.Artifacts, artifact => artifact.Kind == ArtifactKind.ReferenceClip);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForAllSpeakersAsync_uses_qwen3_stock_when_short_clone_target_is_not_kokoro()
    {
        var consent = new FakeConsentService();
        consent.GrantVoiceCloningConsent();
        var ttsEngine = new FakeVoiceCloneTtsEngine(consent) { SampleRate = 1000, DurationSamples = 1000 };
        var analyzer = new FakeReferenceClipAnalyzer();
        analyzer.QueueAnalysis(new ReferenceClipAnalysis(2.66d, 2.66d, 24000, 1, "male"));
        analyzer.QueueAnalysis(new ReferenceClipAnalysis(2.66d, 2.66d, 24000, 1, "male"));
        using var workspace = new TemporaryTestWorkspace();
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            consent,
            new FakeAuditLog(),
            analyzer,
            workspace.Root,
            voiceCatalog: new FakeVoiceCatalog(
            [
                new("af_heart", "mul", "female", "Heart"),
                new("am_adam", "mul", "male", "Adam"),
                new("bf_alice", "en-gb", "female", "Alice"),
                new(Qwen3TtsDefaults.CustomVoice06Alias, "ja", "synthetic", "Qwen3 custom"),
            ]));
        TranscriptProjectState spanishState = CreateTranslatedProjectState();
        TranslationRevision japaneseRevision = spanishState.CurrentTranslationRevision! with { TargetLanguage = "ja" };
        TranscriptProjectState state = AddNormalizedAudioArtifact(
            spanishState with
            {
                CurrentTranslationRevision = japaneseRevision,
                SelectedTranslationTargetLanguage = "ja",
            },
            context.ArtifactStore);
        Guid speakerId = state.Speakers[0].Id;

        await context.Service.GenerateTtsForAllSpeakersAsync(
            state,
            new GenerateTtsForAllSpeakersRequest(
                UseReferenceClipForVoiceCloningBySpeakerId: new Dictionary<Guid, bool>
                {
                    [speakerId] = true
                }),
            TestContext.Current.CancellationToken);

        VoiceAssignment assignment = Assert.Single(context.VoiceAssignmentRepository.All);
        Assert.True(assignment.IsFallback);
        Assert.Equal(Qwen3TtsDefaults.CustomVoice06Alias, assignment.VoiceModelId);
        Assert.Null(assignment.VoiceVariant);
        TtsTake take = Assert.Single(context.TtsTakeRepository.All);
        Assert.Equal(TtsTakeKind.Stock, take.Kind);
        // The take is voiced by a real Qwen3 preset (the native Japanese speaker), not the model
        // alias string, which the Qwen3 engine would reject as an unknown speaker.
        Assert.Equal("qwen3:ono_anna", take.VoiceId);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForAllSpeakersAsync_keeps_cloning_speakers_that_meet_the_reference_floor()
    {
        var consent = new FakeConsentService();
        consent.GrantVoiceCloningConsent();
        var ttsEngine = new FakeVoiceCloneTtsEngine(consent) { SampleRate = 1000, DurationSamples = 1000 };
        var analyzer = new FakeReferenceClipAnalyzer();
        analyzer.QueueAnalysis(new ReferenceClipAnalysis(4d, 4d, 24000, 1, "female"));
        analyzer.QueueAnalysis(new ReferenceClipAnalysis(4d, 4d, 24000, 1, "female"));
        analyzer.QueueAnalysis(new ReferenceClipAnalysis(2d, 2d, 24000, 1, "male"));
        analyzer.QueueAnalysis(new ReferenceClipAnalysis(2d, 2d, 24000, 1, "male"));
        using var workspace = new TemporaryTestWorkspace();
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            consent,
            new FakeAuditLog(),
            analyzer,
            workspace.Root);
        TranscriptProjectState state = AddNormalizedAudioArtifact(
            CreateTwoSpeakerTranslatedProjectState(),
            context.ArtifactStore);
        Guid speakerA = state.Speakers[0].Id;
        Guid speakerB = state.Speakers[1].Id;

        await context.Service.GenerateTtsForAllSpeakersAsync(
            state,
            new GenerateTtsForAllSpeakersRequest(
                UseReferenceClipForVoiceCloningBySpeakerId: new Dictionary<Guid, bool>
                {
                    [speakerA] = true,
                    [speakerB] = true
                }),
            TestContext.Current.CancellationToken);

        Assert.Equal(2, context.TtsTakeRepository.All.Count);
        TtsTake cloned = Assert.Single(context.TtsTakeRepository.All, take => take.Kind == TtsTakeKind.VoiceCloned);
        TtsTake stock = Assert.Single(context.TtsTakeRepository.All, take => take.Kind == TtsTakeKind.Stock);
        Assert.Equal(speakerA, state.TranscriptSegments.Single(segment => segment.SegmentIndex == cloned.SegmentIndex).SpeakerId);
        Assert.Equal(speakerB, state.TranscriptSegments.Single(segment => segment.SegmentIndex == stock.SegmentIndex).SpeakerId);
        Assert.Equal("am_adam", stock.VoiceId);
        VoiceAssignment fallback = Assert.Single(context.VoiceAssignmentRepository.All, assignment => assignment.IsFallback);
        Assert.Equal(speakerB, fallback.SpeakerId);
        Assert.Equal("am_adam", fallback.VoiceVariant);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_packs_segments_when_single_auto_reference_is_too_short()
    {
        var consent = new FakeConsentService();
        consent.GrantVoiceCloningConsent();
        var ttsEngine = new FakeVoiceCloneTtsEngine(consent) { SampleRate = 1000, DurationSamples = 1000 };
        var analyzer = new FakeReferenceClipAnalyzer();
        analyzer.QueueAnalysis(new ReferenceClipAnalysis(2d, 2d, 24000, 1));
        analyzer.QueueAnalysis(new ReferenceClipAnalysis(4d, 4d, 24000, 1));
        var clipExtractor = new FakeAudioClipExtractor();
        using var workspace = new TemporaryTestWorkspace();
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            consent,
            new FakeAuditLog(),
            analyzer,
            workspace.Root,
            clipExtractor);
        TranscriptProjectState state = AddNormalizedAudioArtifact(
            CreateTranslatedProjectState(),
            context.ArtifactStore);
        Guid speakerId = state.Speakers[0].Id;

        await context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId, UseReferenceClipForVoiceCloning: true),
            TestContext.Current.CancellationToken);

        Assert.Collection(
            clipExtractor.LastRanges,
            range =>
            {
                Assert.Equal(0d, range.StartSeconds);
                Assert.Equal(2d, range.EndSeconds);
            },
            range =>
            {
                Assert.Equal(2d, range.StartSeconds);
                Assert.Equal(4d, range.EndSeconds);
            });
        ProjectArtifact referenceArtifact = Assert.Single(
            context.MediaAssetRepository.Artifacts,
            artifact => artifact.Kind == ArtifactKind.ReferenceClip);
        Assert.Contains("mode:packed", referenceArtifact.Provenance);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_uses_transcript_assignment_for_auto_reference_over_diarization_turn()
    {
        var consent = new FakeConsentService();
        consent.GrantVoiceCloningConsent();
        var ttsEngine = new FakeVoiceCloneTtsEngine(consent) { SampleRate = 1000, DurationSamples = 1000 };
        var clipExtractor = new FakeAudioClipExtractor();
        using var workspace = new TemporaryTestWorkspace();
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            consent,
            new FakeAuditLog(),
            new FakeReferenceClipAnalyzer(),
            workspace.Root,
            clipExtractor);
        TranscriptProjectState state = AddNormalizedAudioArtifact(
            CreateTranslatedProjectState(),
            context.ArtifactStore);
        Guid speakerId = state.Speakers[0].Id;
        TranscriptRevision transcriptRevision = state.CurrentTranscriptRevision!;
        TranscriptSegment correctedSegment = TranscriptSegment.Create(
            transcriptRevision.Id,
            0,
            1d,
            3d,
            "Corrected speaker line.",
            speakerId,
            "en");
        state = state with
        {
            TranscriptSegments = [correctedSegment],
            SpeakerTurns =
            [
                SpeakerTurn.Create(
                    state.ProjectState.Project.Id,
                    speakerId,
                    0d,
                    4d,
                    confidence: 0.99d)
            ]
        };

        await context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId, UseReferenceClipForVoiceCloning: true),
            TestContext.Current.CancellationToken);

        AudioClipRange range = Assert.Single(clipExtractor.LastRanges);
        Assert.Equal(1d, range.StartSeconds);
        Assert.Equal(3d, range.EndSeconds);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_preserves_uploaded_reference_clip()
    {
        var consent = new FakeConsentService();
        consent.GrantVoiceCloningConsent();
        var ttsEngine = new FakeVoiceCloneTtsEngine(consent) { SampleRate = 1000, DurationSamples = 1000 };
        using var workspace = new TemporaryTestWorkspace();
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            consent,
            new FakeAuditLog(),
            new FakeReferenceClipAnalyzer(),
            workspace.Root);
        TranscriptProjectState state = AddNormalizedAudioArtifact(
            CreateTranslatedProjectState(),
            context.ArtifactStore);
        Guid speakerId = state.Speakers[0].Id;
        Guid manualReferenceArtifactId = Guid.NewGuid();
        string relativePath = $"artifacts/reference-clips/{speakerId:D}/manual.wav";
        context.ArtifactStore.Seed(relativePath, [1, 2, 3, 4]);
        ProjectArtifact manualReference = new(
            manualReferenceArtifactId,
            state.ProjectState.Project.Id,
            state.ProjectState.MediaAsset!.Id,
            ArtifactKind.ReferenceClip,
            relativePath,
            "manual-sha",
            4,
            4d,
            24000,
            1,
            DateTimeOffset.UtcNow,
            Provenance: $"manual-speaker-reference:{speakerId:D};active-speech:4.000");
        state = state with
        {
            ProjectState = state.ProjectState with
            {
                Artifacts = state.ProjectState.Artifacts.Concat([manualReference]).ToArray()
            },
            VoiceAssignments =
            [
                state.VoiceAssignments[0] with
                {
                    ReferenceClipArtifactId = manualReferenceArtifactId,
                    RequiresConsent = true
                }
            ]
        };

        await context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId, UseReferenceClipForVoiceCloning: true),
            TestContext.Current.CancellationToken);

        TtsTake take = Assert.Single(context.TtsTakeRepository.All);
        Assert.Equal(TtsTakeKind.VoiceCloned, take.Kind);
        Assert.Equal(manualReferenceArtifactId, take.ReferenceClipArtifactId);
        Assert.DoesNotContain(context.MediaAssetRepository.Artifacts, artifact => artifact.Kind == ArtifactKind.ReferenceClip);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_refreshes_stale_auto_reference_clip()
    {
        var consent = new FakeConsentService();
        consent.GrantVoiceCloningConsent();
        var ttsEngine = new FakeVoiceCloneTtsEngine(consent) { SampleRate = 1000, DurationSamples = 1000 };
        using var workspace = new TemporaryTestWorkspace();
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            consent,
            new FakeAuditLog(),
            new FakeReferenceClipAnalyzer(),
            workspace.Root);
        TranscriptProjectState state = AddNormalizedAudioArtifact(
            CreateTranslatedProjectState(),
            context.ArtifactStore);
        Guid speakerId = state.Speakers[0].Id;
        Guid staleReferenceArtifactId = Guid.NewGuid();
        string relativePath = $"artifacts/reference-clips/{speakerId:D}/stale.wav";
        context.ArtifactStore.Seed(relativePath, [1, 2, 3, 4]);
        ProjectArtifact staleReference = new(
            staleReferenceArtifactId,
            state.ProjectState.Project.Id,
            state.ProjectState.MediaAsset!.Id,
            ArtifactKind.ReferenceClip,
            relativePath,
            "stale-sha",
            4,
            4d,
            24000,
            1,
            DateTimeOffset.UtcNow,
            Provenance: $"auto-speaker-reference:v1;speaker:{speakerId:D};fingerprint:old");
        state = state with
        {
            ProjectState = state.ProjectState with
            {
                Artifacts = state.ProjectState.Artifacts.Concat([staleReference]).ToArray()
            },
            VoiceAssignments =
            [
                state.VoiceAssignments[0] with
                {
                    ReferenceClipArtifactId = staleReferenceArtifactId,
                    RequiresConsent = true
                }
            ]
        };

        await context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId, UseReferenceClipForVoiceCloning: true),
            TestContext.Current.CancellationToken);

        TtsTake take = Assert.Single(context.TtsTakeRepository.All);
        Assert.NotEqual(staleReferenceArtifactId, take.ReferenceClipArtifactId);
        ProjectArtifact refreshedReference = Assert.Single(
            context.MediaAssetRepository.Artifacts,
            artifact => artifact.Kind == ArtifactKind.ReferenceClip);
        Assert.Equal(take.ReferenceClipArtifactId, refreshedReference.Id);
        Assert.Contains("auto-speaker-reference:v2", refreshedReference.Provenance);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_defaults_to_stock_for_synthetic_voice_with_reference_clip()
    {
        var consent = new FakeConsentService();
        var ttsEngine = new FakeVoiceCloneTtsEngine(consent) { SampleRate = 1000, DurationSamples = 1000 };
        var auditLog = new FakeAuditLog();
        using var workspace = new TemporaryTestWorkspace();
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            consent,
            auditLog,
            new FakeReferenceClipAnalyzer(),
            workspace.Root);
        (TranscriptProjectState state, Guid speakerId, Guid referenceClipArtifactId) =
            CreateVoiceClonedTranslatedProjectState(context.ArtifactStore);
        state = state with
        {
            VoiceAssignments =
            [
                state.VoiceAssignments[0].AssignVoice("kokoro-onnx", "af_heart", referenceClipArtifactId: referenceClipArtifactId)
            ]
        };

        await context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId),
            TestContext.Current.CancellationToken);

        TtsTake take = Assert.Single(context.TtsTakeRepository.All);
        Assert.Equal(TtsTakeKind.Stock, take.Kind);
        Assert.Null(take.ReferenceClipArtifactId);
        Assert.Null(ttsEngine.LastReferenceClipArtifactId);
        Assert.Empty(auditLog.Entries);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_analyzes_reference_clip_only_once_for_multiple_segments()
    {
        var consent = new FakeConsentService();
        consent.GrantVoiceCloningConsent();
        var ttsEngine = new FakeVoiceCloneTtsEngine(consent) { SampleRate = 1000, DurationSamples = 1000 };
        var auditLog = new FakeAuditLog();
        var analyzer = new FakeReferenceClipAnalyzer();
        using var workspace = new TemporaryTestWorkspace();
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            consent,
            auditLog,
            analyzer,
            workspace.Root);
        (TranscriptProjectState state, Guid speakerId, _) = CreateVoiceClonedTranslatedProjectStateWithMultipleSegments(context.ArtifactStore);

        await context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId, UseReferenceClipForVoiceCloning: true),
            TestContext.Current.CancellationToken);

        Assert.Equal(2, context.TtsTakeRepository.All.Count);
        Assert.Equal(1, analyzer.AnalyzeCallCount);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_without_reference_clip_saves_stock_take()
    {
        var consent = new FakeConsentService();
        var ttsEngine = new FakeVoiceCloneTtsEngine(consent) { SampleRate = 1000, DurationSamples = 1000 };
        var auditLog = new FakeAuditLog();
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            consent,
            auditLog,
            new FakeReferenceClipAnalyzer());
        TranscriptProjectState state = CreateTranslatedProjectState();
        Guid speakerId = state.Speakers[0].Id;

        await context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId),
            TestContext.Current.CancellationToken);

        TtsTake take = Assert.Single(context.TtsTakeRepository.All);
        Assert.Equal(TtsTakeKind.Stock, take.Kind);
        Assert.Null(take.ReferenceClipArtifactId);
        Assert.Null(ttsEngine.LastReferenceClipArtifactId);
        Assert.Empty(auditLog.Entries);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_preserves_non_cloning_preferred_alias_containing_f5()
    {
        var ttsEngine = new FakeTtsEngine();
        TtsServiceContext context = CreateTtsServiceContext(ttsEngine);
        TranscriptProjectState state = CreateTranslatedProjectState();
        Guid speakerId = state.Speakers[0].Id;

        await context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId, PreferredModelAlias: "bf5-asr"),
            TestContext.Current.CancellationToken);

        Assert.Equal("bf5-asr", ttsEngine.LastOptions?.PreferredModelAlias);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_redirects_explicit_voice_clone_alias_without_reference_clip()
    {
        var ttsEngine = new FakeTtsEngine();
        TtsServiceContext context = CreateTtsServiceContext(ttsEngine);
        TranscriptProjectState state = CreateTranslatedProjectState();
        Guid speakerId = state.Speakers[0].Id;

        await context.Service.GenerateTtsForSpeakerAsync(
            state,
            new GenerateTtsForSpeakerRequest(speakerId, PreferredModelAlias: "f5-tts"),
            TestContext.Current.CancellationToken);

        Assert.Equal("kokoro-onnx", ttsEngine.LastOptions?.PreferredModelAlias);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_throws_when_assigned_stock_voice_is_missing()
    {
        var ttsEngine = new FakeTtsEngine();
        TtsServiceContext context = CreateTtsServiceContext(ttsEngine);
        TranscriptProjectState state = CreateTranslatedProjectState();
        Guid speakerId = state.Speakers[0].Id;
        state = state with
        {
            VoiceAssignments =
            [
                VoiceAssignment.Create(
                    state.ProjectState.Project.Id,
                    speakerId,
                    "kokoro-onnx",
                    "missing_voice")
            ]
        };

        InvalidOperationException exception = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            context.Service.GenerateTtsForSpeakerAsync(
                state,
                new GenerateTtsForSpeakerRequest(speakerId),
                TestContext.Current.CancellationToken));

        Assert.Equal("Voicepack 'missing_voice' is not available.", exception.Message);
        Assert.Equal(0, ttsEngine.SynthesizeCallCount);
        Assert.Empty(context.TtsTakeRepository.All);
        Assert.DoesNotContain(context.MediaAssetRepository.Artifacts, artifact => artifact.Kind == ArtifactKind.TtsTake);
    }

    [Fact]
    public async Task TtsOrchestrationService_GenerateTtsForSpeakerAsync_throws_when_translation_revision_has_no_segments()
    {
        var ttsEngine = new FakeTtsEngine();
        TtsServiceContext context = CreateTtsServiceContext(ttsEngine);
        TranscriptProjectState state = CreateTranslatedProjectState() with { TranslatedSegments = [] };
        Guid speakerId = state.Speakers[0].Id;

        InvalidOperationException exception = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            context.Service.GenerateTtsForSpeakerAsync(
                state,
                new GenerateTtsForSpeakerRequest(speakerId),
                TestContext.Current.CancellationToken));

        Assert.Equal("The current translation revision has no translated segments.", exception.Message);
        Assert.Equal(0, ttsEngine.SynthesizeCallCount);
    }

    [Fact]
    public async Task TtsOrchestrationService_PreviewVoiceAsync_uses_fake_engine_without_persistence()
    {
        var ttsEngine = new FakeTtsEngine();
        TtsServiceContext context = CreateTtsServiceContext(ttsEngine);

        PreviewVoiceResult result = await context.Service.PreviewVoiceAsync(
            new PreviewVoiceRequest("af_heart", "en-us", "Preview text."),
            TestContext.Current.CancellationToken);

        Assert.NotEmpty(result.WavBytes);
        Assert.Equal("fake", result.Provider);
        Assert.Equal("fake", result.ModelId);
        Assert.Equal("af_heart", result.VoiceId);
        Assert.Equal("Preview text.", ttsEngine.LastInputText);
        Assert.Equal("af_heart", ttsEngine.LastVoicepack?.VoiceId);
        Assert.Equal("kokoro-onnx", ttsEngine.LastOptions?.PreferredModelAlias);
        Assert.True(ttsEngine.LastOptions?.RequirePreferredModelAlias);
        Assert.Empty(context.TtsTakeRepository.All);
        Assert.Empty(context.MediaAssetRepository.Artifacts);
        Assert.Empty(context.StageRunStore.All);
    }

    [Fact]
    public async Task TtsOrchestrationService_PreviewVoiceAsync_routes_qwen3_presets_to_custom_voice_without_an_alias()
    {
        var ttsEngine = new FakeTtsEngine();
        var catalog = new FakeVoiceCatalog(
        [
            new("af_heart", "mul", "female", "Heart"),
            new("qwen3:vivian", "mul", "female", "Vivian (Chinese)"),
        ]);
        TtsServiceContext context = CreateTtsServiceContext(ttsEngine, voiceCatalog: catalog);

        await context.Service.PreviewVoiceAsync(
            new PreviewVoiceRequest("qwen3:vivian", "zh", "Preview text."),
            TestContext.Current.CancellationToken);

        Assert.Equal("qwen3:vivian", ttsEngine.LastVoicepack?.VoiceId);
        Assert.Equal(Qwen3TtsDefaults.CustomVoice06Alias, ttsEngine.LastOptions?.PreferredModelAlias);
        Assert.True(ttsEngine.LastOptions?.RequirePreferredModelAlias);
    }

    [Fact]
    public async Task TtsOrchestrationService_PreviewVoiceAsync_defaults_qwen3_custom_voice_alias_for_preset_voices()
    {
        var ttsEngine = new FakeTtsEngine();
        TtsServiceContext context = CreateTtsServiceContext(
            ttsEngine,
            voiceCatalog: new FakeVoiceCatalog(
            [
                new("qwen3:ono_anna", "ja", "female", "Ono Anna"),
            ]));

        await context.Service.PreviewVoiceAsync(
            new PreviewVoiceRequest("qwen3:ono_anna", "ja", "Preview text."),
            TestContext.Current.CancellationToken);

        Assert.Equal(Qwen3TtsDefaults.ResolveCustomVoiceAlias(tier: null), ttsEngine.LastOptions!.PreferredModelAlias);
        Assert.True(ttsEngine.LastOptions.RequirePreferredModelAlias);
    }

    private static TranslationServiceContext CreateTranslationServiceContext(
        FakeTranslationEngine translationEngine,
        FakeGlossaryRepository? glossaryRepository = null,
        FakeTranslationLanguageRouter? translationLanguageRouter = null,
        ITranslatedWordAlignmentService? translatedWordAlignmentService = null)
    {
        var translationRepository = new FakeTranslationRepository();
        glossaryRepository ??= new FakeGlossaryRepository();
        var ttsTakeRepository = new FakeTtsTakeRepository();
        var stageRunStore = new FakeProjectStageRunStore();
        var artifactStore = new FakeArtifactStore();
        var fileFingerprintService = new FakeFileFingerprintService();
        var mediaAssetRepository = new FakeMediaAssetRepository();
        var artifactWriter = new TranscriptArtifactWriter(
            artifactStore,
            fileFingerprintService,
            mediaAssetRepository);
        var service = new TranslationOrchestrationService(
            translationRepository,
            new GlossaryService(glossaryRepository),
            new GlossaryTermMatcher(),
            translationLanguageRouter ?? new FakeTranslationLanguageRouter(),
            translationEngine,
            ttsTakeRepository,
            stageRunStore,
            artifactStore,
            artifactWriter,
            translatedWordAlignmentService: translatedWordAlignmentService,
            commitBoundary: TestAtomicCommitBoundary.Create(translationRepository, ttsTakeRepository, artifactStore, mediaAssetRepository));

        return new TranslationServiceContext(
            service,
            translationRepository,
            stageRunStore,
            artifactStore,
            mediaAssetRepository);
    }

    private static TtsServiceContext CreateTtsServiceContext(
        FakeTtsEngine ttsEngine,
        IConsentService? consentService = null,
        IVoiceCloneAuditLog? auditLog = null,
        IReferenceClipAnalyzer? referenceClipAnalyzer = null,
        string? artifactRoot = null,
        IAudioClipExtractor? audioClipExtractor = null,
        FakeReferenceClipTrimmer? referenceClipTrimmer = null,
        IVoiceCatalog? voiceCatalog = null)
    {
        var voiceAssignmentRepository = new FakeVoiceAssignmentRepository();
        var ttsTakeRepository = new FakeTtsTakeRepository();
        voiceCatalog ??= new FakeVoiceCatalog();
        var artifactStore = new FakeArtifactStore(artifactRoot);
        var fileFingerprintService = new FakeFileFingerprintService();
        var mediaAssetRepository = new FakeMediaAssetRepository();
        var stageRunStore = new FakeProjectStageRunStore();
        var durationAnalysisService = new DurationAnalysisService();
        audioClipExtractor ??= new FakeAudioClipExtractor();
        referenceClipTrimmer ??= new FakeReferenceClipTrimmer();
        var startTtsStageHandler = new StartTtsStageHandler(
            ttsEngine,
            voiceCatalog,
            artifactStore,
            fileFingerprintService,
            mediaAssetRepository,
            ttsTakeRepository,
            stageRunStore,
            durationAnalysisService,
            consentService: consentService,
            voiceCloneAuditLog: auditLog,
            referenceClipAnalyzer: referenceClipAnalyzer,
            commitBoundary: TestAtomicCommitBoundary.Create(null, ttsTakeRepository, artifactStore, mediaAssetRepository));
        var service = new TtsOrchestrationService(
            startTtsStageHandler,
            voiceAssignmentRepository,
            ttsTakeRepository,
            ttsEngine,
            voiceCatalog,
            artifactStore,
            fileFingerprintService,
            mediaAssetRepository,
            referenceClipTrimmer,
            durationAnalysisService,
            audioClipExtractor: audioClipExtractor,
            referenceClipAnalyzer: referenceClipAnalyzer);

        return new TtsServiceContext(
            service,
            ttsTakeRepository,
            stageRunStore,
            artifactStore,
            mediaAssetRepository,
            voiceAssignmentRepository,
            referenceClipTrimmer);
    }

    private static TranscriptProjectState CreateTranscriptProjectState(
        string transcriptLanguage = "en",
        TranslationRevision? translationRevision = null,
        IReadOnlyList<TranslatedSegment>? translatedSegments = null,
        IReadOnlyList<VoiceAssignment>? voiceAssignments = null)
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        Guid projectId = Guid.NewGuid();
        var project = new TrackdubProject(projectId, "Test Project", now, now);
        var mediaAsset = new MediaAsset(
            Guid.NewGuid(),
            projectId,
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
            transcriptLanguage);
        TranscriptRevision transcriptRevision = TranscriptRevision.Create(
            projectId,
            stageRunId: null,
            revisionNumber: 1,
            now);
        var speaker = new ProjectSpeaker(Guid.NewGuid(), projectId, "Speaker 1", now);
        TranscriptSegment[] transcriptSegments =
        [
            TranscriptSegment.Create(
                transcriptRevision.Id,
                0,
                0.0d,
                2.0d,
                "Hello there.",
                speaker.Id,
                transcriptLanguage),
            TranscriptSegment.Create(
                transcriptRevision.Id,
                1,
                2.0d,
                4.0d,
                "Second line.",
                speaker.Id,
                transcriptLanguage)
        ];

        return new TranscriptProjectState(
            projectState,
            transcriptRevision,
            transcriptSegments,
            [speaker],
            [],
            translationRevision,
            translatedSegments ?? [],
            false,
            transcriptLanguage,
            [],
            [],
            translationRevision?.TargetLanguage,
            new HashSet<int>(),
            null,
            new FakeVoiceCatalog().GetVoices(),
            voiceAssignments ?? [],
            [],
            [],
            []);
    }

    private static TranscriptProjectState CreateTranslatedProjectState()
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        TranscriptProjectState state = CreateTranscriptProjectState();
        Guid projectId = state.ProjectState.Project.Id;
        Guid speakerId = state.Speakers[0].Id;
        TranslationRevision translationRevision = TranslationRevision.Create(
            projectId,
            stageRunId: null,
            state.CurrentTranscriptRevision!.Id,
            "es",
            revisionNumber: 1,
            now,
            translationProvider: "fake",
            modelId: "fake-translation-model");
        TranslatedSegment translatedSegment = TranslatedSegment.Create(
            translationRevision.Id,
            0,
            0.0d,
            2.0d,
            "Linea traducida.");
        VoiceAssignment voiceAssignment = VoiceAssignment.Create(
            projectId,
            speakerId,
            "kokoro-onnx",
            "af_heart");

        return state with
        {
            CurrentTranslationRevision = translationRevision,
            TranslatedSegments = [translatedSegment],
            SelectedTranslationTargetLanguage = "es",
            VoiceAssignments = [voiceAssignment]
        };
    }

    private static TranscriptProjectState CreateTwoSpeakerTranslatedProjectState()
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        TranscriptProjectState state = CreateTranscriptProjectState();
        Guid projectId = state.ProjectState.Project.Id;
        var speakerA = new ProjectSpeaker(Guid.NewGuid(), projectId, "Speaker 1", now);
        var speakerB = new ProjectSpeaker(Guid.NewGuid(), projectId, "Speaker 2", now.AddSeconds(1));
        TranscriptRevision transcriptRevision = state.CurrentTranscriptRevision!;
        TranscriptSegment[] transcriptSegments =
        [
            TranscriptSegment.Create(
                transcriptRevision.Id,
                0,
                0.0d,
                2.0d,
                "Hello there.",
                speakerA.Id,
                "en"),
            TranscriptSegment.Create(
                transcriptRevision.Id,
                1,
                2.0d,
                4.0d,
                "Second line.",
                speakerB.Id,
                "en"),
        ];
        TranslationRevision translationRevision = TranslationRevision.Create(
            projectId,
            stageRunId: null,
            transcriptRevision.Id,
            "es",
            revisionNumber: 1,
            now,
            translationProvider: "fake",
            modelId: "fake-translation-model");

        return state with
        {
            Speakers = [speakerA, speakerB],
            TranscriptSegments = transcriptSegments,
            CurrentTranslationRevision = translationRevision,
            TranslatedSegments =
            [
                TranslatedSegment.Create(translationRevision.Id, 0, 0.0d, 2.0d, "Linea traducida."),
                TranslatedSegment.Create(translationRevision.Id, 1, 2.0d, 4.0d, "Segunda linea."),
            ],
            SelectedTranslationTargetLanguage = "es",
            VoiceAssignments = []
        };
    }

    private static TranscriptProjectState CreateTranslatedProjectStateWithWords()
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        TranscriptProjectState state = CreateTranscriptProjectState();
        Guid projectId = state.ProjectState.Project.Id;
        Guid speakerId = state.Speakers[0].Id;
        TranslationRevision translationRevision = TranslationRevision.Create(
            projectId,
            stageRunId: null,
            state.CurrentTranscriptRevision!.Id,
            "es",
            revisionNumber: 1,
            now,
            translationProvider: "fake",
            modelId: "fake-translation-model");
        VoiceAssignment voiceAssignment = VoiceAssignment.Create(
            projectId,
            speakerId,
            "kokoro-onnx",
            "af_heart");

        return state with
        {
            CurrentTranslationRevision = translationRevision,
            TranslatedSegments =
            [
                TranslatedSegment.Create(
                    translationRevision.Id,
                    0,
                    0.0d,
                    2.0d,
                    "Linea traducida.",
                    "hash-0",
                    [
                        TranslatedWord.Create(0, 0.0d, 1.0d, "Linea"),
                        TranslatedWord.Create(1, 1.0d, 2.0d, "traducida")
                    ]),
                TranslatedSegment.Create(
                    translationRevision.Id,
                    1,
                    2.0d,
                    4.0d,
                    "Segunda linea.",
                    "hash-1",
                    [
                        TranslatedWord.Create(0, 2.0d, 3.0d, "Segunda"),
                        TranslatedWord.Create(1, 3.0d, 4.0d, "linea")
                    ])
            ],
            SelectedTranslationTargetLanguage = "es",
            VoiceAssignments = [voiceAssignment]
        };
    }

    private static (TranscriptProjectState State, Guid SpeakerId, Guid ReferenceClipArtifactId) CreateVoiceClonedTranslatedProjectState(
        FakeArtifactStore artifactStore)
    {
        TranscriptProjectState state = CreateTranslatedProjectState();
        Guid projectId = state.ProjectState.Project.Id;
        Guid mediaAssetId = state.ProjectState.MediaAsset!.Id;
        Guid speakerId = state.Speakers[0].Id;
        Guid referenceClipArtifactId = Guid.NewGuid();
        string relativePath = $"artifacts/reference-clips/{speakerId:D}/reference.wav";
        artifactStore.Seed(relativePath, [0, 1, 2, 3]);

        ProjectArtifact referenceArtifact = new(
            referenceClipArtifactId,
            projectId,
            mediaAssetId,
            ArtifactKind.ReferenceClip,
            relativePath,
            "reference-sha",
            4,
            5d,
            24000,
            1,
            DateTimeOffset.UtcNow,
            Provenance: $"speaker-reference:{speakerId:D}");
        VoiceAssignment voiceAssignment = VoiceAssignment.Create(
            projectId,
            speakerId,
            VoiceCloningDefaults.ChatterboxPrimaryAlias,
            requiresConsent: true,
            referenceClipArtifactId: referenceClipArtifactId);

        return (state with
        {
            ProjectState = state.ProjectState with
            {
                Artifacts = state.ProjectState.Artifacts.Concat([referenceArtifact]).ToArray()
            },
            VoiceAssignments = [voiceAssignment]
        }, speakerId, referenceClipArtifactId);
    }

    private static (TranscriptProjectState State, Guid SpeakerId, Guid ReferenceClipArtifactId) CreateVoiceClonedTranslatedProjectStateWithMultipleSegments(
        FakeArtifactStore artifactStore)
    {
        (TranscriptProjectState state, Guid speakerId, Guid referenceClipArtifactId) = CreateVoiceClonedTranslatedProjectState(artifactStore);

        // Add a second translated segment so that HandleAsync iterates over two segments.
        TranslatedSegment secondSegment = TranslatedSegment.Create(
            state.CurrentTranslationRevision!.Id,
            1,
            2.0d,
            4.0d,
            "Segunda linea.");

        return (state with
        {
            TranslatedSegments = [.. state.TranslatedSegments, secondSegment]
        }, speakerId, referenceClipArtifactId);
    }

    private static TranscriptProjectState AddNormalizedAudioArtifact(
        TranscriptProjectState state,
        FakeArtifactStore artifactStore)
    {
        Guid artifactId = Guid.NewGuid();
        string relativePath = "artifacts/audio/normalized.wav";
        artifactStore.Seed(relativePath, [0, 1, 2, 3]);
        ProjectArtifact artifact = new(
            artifactId,
            state.ProjectState.Project.Id,
            state.ProjectState.MediaAsset!.Id,
            ArtifactKind.NormalizedAudio,
            relativePath,
            "normalized-sha",
            4,
            4d,
            24000,
            1,
            DateTimeOffset.UtcNow);

        return state with
        {
            ProjectState = state.ProjectState with
            {
                Artifacts = state.ProjectState.Artifacts.Concat([artifact]).ToArray()
            }
        };
    }

    private sealed record TranslationServiceContext(
        TranslationOrchestrationService Service,
        FakeTranslationRepository TranslationRepository,
        FakeProjectStageRunStore StageRunStore,
        FakeArtifactStore ArtifactStore,
        FakeMediaAssetRepository MediaAssetRepository);

    private sealed record TtsServiceContext(
        TtsOrchestrationService Service,
        FakeTtsTakeRepository TtsTakeRepository,
        FakeProjectStageRunStore StageRunStore,
        FakeArtifactStore ArtifactStore,
        FakeMediaAssetRepository MediaAssetRepository,
        FakeVoiceAssignmentRepository VoiceAssignmentRepository,
        FakeReferenceClipTrimmer ReferenceClipTrimmer);

    private sealed class FakeTranslatedWordAlignmentService(
        Func<TranslatedWordAlignmentRequest, TranslatedWordAlignmentResult> handler) : ITranslatedWordAlignmentService
    {
        public Task<TranslatedWordAlignmentResult> AlignAsync(
            TranslatedWordAlignmentRequest request,
            CancellationToken cancellationToken) =>
            Task.FromResult(handler(request));
    }

    private sealed class TemporaryTestWorkspace : IDisposable
    {
        public TemporaryTestWorkspace()
        {
            Root = Path.Join(Path.GetTempPath(), "Trackdub.Tests", Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Root);
        }

        public string Root { get; }

        public void Dispose()
        {
            if (Directory.Exists(Root))
            {
                Directory.Delete(Root, recursive: true);
            }
        }
    }

    private sealed class FakeAudioClipExtractor : IAudioClipExtractor
    {
        public IReadOnlyList<AudioClipRange> LastRanges { get; private set; } = [];

        public async Task<AudioClipExtractionResult> ExtractAsync(
            string sourceWavePath,
            double startSeconds,
            double endSeconds,
            string destinationPath,
            CancellationToken cancellationToken) =>
            await ExtractAsync(
                sourceWavePath,
                [new AudioClipRange(startSeconds, endSeconds)],
                destinationPath,
                cancellationToken).ConfigureAwait(false);

        public async Task<AudioClipExtractionResult> ExtractAsync(
            string sourceWavePath,
            IReadOnlyList<AudioClipRange> ranges,
            string destinationPath,
            CancellationToken cancellationToken)
        {
            LastRanges = ranges.ToArray();
            Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);
            await File.WriteAllBytesAsync(destinationPath, [1, 2, 3, 4], cancellationToken);
            return new AudioClipExtractionResult(
                destinationPath,
                ranges.Sum(range => range.EndSeconds - range.StartSeconds),
                24000,
                1);
        }
    }
}
