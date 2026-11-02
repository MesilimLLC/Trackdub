using System.Globalization;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts.Dubbing;
using Trackdub.Contracts.Transcripts;
using Trackdub.Domain;
using Trackdub.Domain.Speakers;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Tts;

namespace Trackdub.Application.Dubbing;

/// <summary>
/// Voice decisions for unattended (CLI/SDK) runs, which have no interactive voice-assignment
/// step: explicit <c>--voice</c> overrides, stock fallbacks, per-speaker cloning, and which TTS
/// model preflight must provision for the voices that will actually speak.
/// </summary>
internal static class UnattendedVoiceSelection
{
    internal static bool RequestsVoiceCloning(DubbingSessionOptions options) =>
        options.UseVoiceCloning ||
        (options.VoiceCloneBySpeakerId?.Values.Any(static clone => clone) ?? false);

    /// <summary>
    /// Unattended runs have no voice-assignment step, so speakers without a deliberate
    /// (non-fallback) voice assignment would fail TTS outright. Mirrors the shell's fallback
    /// behavior: picks the first catalog voice whose language matches the dub target language,
    /// ordered by display name like the shell's voice picker. Languages Kokoro does not cover use
    /// the Qwen3 CustomVoice default preset when Qwen3 speaks the language. When the selected TTS
    /// model is already CustomVoice, use that language's Qwen default even for Kokoro languages
    /// (Spanish/English catalogs omit Qwen presets, and quality seed volumes may lack Kokoro).
    /// Speakers stay unassigned otherwise, so TTS fails with the explicit assignment error instead
    /// of dubbing with a wrong-language voice.
    /// </summary>
    internal static Dictionary<Guid, string>? BuildUnattendedFallbackVoiceIds(
        TranscriptProjectState state,
        string? targetLanguageCode,
        string? preferredTtsModelAlias = null)
    {
        if (state.Speakers.Count == 0)
        {
            return null;
        }

        HashSet<Guid> deliberatelyAssignedSpeakerIds = state.VoiceAssignments
            .Where(static assignment => !assignment.IsFallback)
            .Select(static assignment => assignment.SpeakerId)
            .ToHashSet();

        string? defaultVoiceId = ResolveUnattendedFallbackVoiceId(state, targetLanguageCode, preferredTtsModelAlias);
        if (defaultVoiceId is null)
        {
            return null;
        }

        Dictionary<Guid, string> fallbackVoiceIds = state.Speakers
            .Where(speaker => !deliberatelyAssignedSpeakerIds.Contains(speaker.Id))
            .ToDictionary(static speaker => speaker.Id, _ => defaultVoiceId);
        return fallbackVoiceIds.Count > 0 ? fallbackVoiceIds : null;
    }

    private static string? ResolveUnattendedFallbackVoiceId(
        TranscriptProjectState state,
        string? targetLanguageCode,
        string? preferredTtsModelAlias)
    {
        // CustomVoice runs (e.g. poc-server-quality-v1) must not depend on Kokoro catalog entries.
        if (Qwen3TtsDefaults.IsCustomVoiceAlias(preferredTtsModelAlias) &&
            Qwen3TtsDefaults.SupportsLanguage(targetLanguageCode))
        {
            return Qwen3TtsDefaults.ResolveDefaultPresetVoiceId(targetLanguageCode);
        }

        // Languages Kokoro does not cover use the Qwen3 CustomVoice preset whose native language
        // matches. The picker list can contain every preset for these languages, so take the
        // language default instead of the alphabetically first preset (Aiden is English).
        if (!string.IsNullOrWhiteSpace(targetLanguageCode) &&
            !StockTtsVoiceMatcher.SupportsKokoro(targetLanguageCode))
        {
            return Qwen3TtsDefaults.SupportsLanguage(targetLanguageCode)
                ? Qwen3TtsDefaults.ResolveDefaultPresetVoiceId(targetLanguageCode)
                : null;
        }

        return state.AvailableVoices
            .Where(voice => IsVoiceLanguageMatch(voice.LanguageCode, targetLanguageCode))
            .OrderBy(static voice => voice.DisplayName, StringComparer.OrdinalIgnoreCase)
            .Select(static voice => voice.VoiceId)
            .FirstOrDefault();
    }

    /// <summary>
    /// Pins Chatterbox (or an explicit TTS override) when headless voice cloning is requested.
    /// </summary>
    internal static DubbingSessionOptions ApplyVoiceCloningDefaults(DubbingSessionOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (!RequestsVoiceCloning(options))
        {
            return options;
        }

        Dictionary<string, string> preferences = CopyCaseInsensitive(options.ModelPreferences);

        // Preserve the explicit TTS override only when it names a clone-capable model.
        // A stock alias (e.g. kokoro-onnx) paired with UseVoiceCloning=true would otherwise
        // reach an incompatible voicepack lookup that throws "Voicepack '...' is not available.";
        // pin the default Chatterbox clone model instead so the clone run stays runnable.
        // Return the case-insensitive copy so BuildModelPreferences' GetValueOrDefault
        // lookup resolves regardless of the casing the caller used for the original TTS key.
        bool keepsExplicitCloneModel =
            preferences.TryGetValue(StageNames.Tts, out string? explicitTtsAlias) &&
            VoiceCloningDefaults.IsCloneOnlyModelAlias(explicitTtsAlias);
        if (!keepsExplicitCloneModel)
        {
            preferences[StageNames.Tts] = VoiceCloningDefaults.ResolveDefaultCloneModelAlias(options.TargetLanguageCode);
        }

        return options with { ModelPreferences = preferences };
    }

    /// <summary>
    /// Callers may hand us an arbitrary IReadOnlyDictionary whose comparer is case-sensitive
    /// (e.g. StringComparer.Ordinal), so it can legitimately contain keys that differ only by
    /// case such as both "TTS" and "tts". Copying via the collection constructor with an
    /// OrdinalIgnoreCase comparer throws ArgumentException on those duplicates, so normalize
    /// by explicit enumeration. Duplicate-precedence: last write wins in the source's
    /// enumeration order (matches Dictionary enumeration, but that order is not guaranteed).
    /// </summary>
    private static Dictionary<string, string> CopyCaseInsensitive(IReadOnlyDictionary<string, string>? source)
    {
        var copy = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (KeyValuePair<string, string> pair in source ?? new Dictionary<string, string>())
        {
            copy[pair.Key] = pair.Value;
        }

        return copy;
    }

    /// <summary>
    /// Builds the unattended TTS request: explicit <c>--voice</c> assignments, stock Kokoro
    /// fallbacks for remaining speakers, and per-speaker source-audio cloning when
    /// <see cref="DubbingSessionOptions.UseVoiceCloning"/> is set. Cloning is skipped for
    /// speakers that have a voice override.
    /// </summary>
    internal static GenerateTtsForAllSpeakersRequest BuildUnattendedTtsRequest(
        TranscriptProjectState state,
        DubbingSessionOptions options,
        string? ttsModelAlias)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(options);

        string? preferredModelAlias = ttsModelAlias;
        if (RequestsVoiceCloning(options) && string.IsNullOrWhiteSpace(preferredModelAlias))
        {
            preferredModelAlias = VoiceCloningDefaults.ResolveDefaultCloneModelAlias(options.TargetLanguageCode);
        }

        UnattendedVoicePlan voicePlan = ResolveUnattendedVoicePlan(state, options, preferredModelAlias);
        IReadOnlyDictionary<Guid, string> explicitVoiceIds = voicePlan.ExplicitVoiceIds;
        Dictionary<Guid, bool>? cloneBySpeaker = voicePlan.CloneBySpeaker;
        Dictionary<Guid, string>? fallbackVoiceIds = voicePlan.FallbackVoiceIds;

        return new GenerateTtsForAllSpeakersRequest(
            FallbackVoiceIdsBySpeakerId: fallbackVoiceIds,
            PreferredModelAlias: preferredModelAlias,
            UseReferenceClipForVoiceCloningBySpeakerId: cloneBySpeaker,
            VoiceIdsBySpeakerId: explicitVoiceIds.Count > 0 ? explicitVoiceIds : null);
    }

    /// <summary>
    /// Maps CLI <c>--voice SPEAKER_ID:voice_id</c> keys onto project speakers.
    /// Accepts a speaker Guid, display name (Speaker 1), or Whisper-style <c>SPEAKER_00</c> index.
    /// </summary>
    internal static IReadOnlyDictionary<Guid, string> ResolveVoiceAssignmentOverrides(
        TranscriptProjectState state,
        IReadOnlyDictionary<string, string>? overrides)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (overrides is null || overrides.Count == 0)
        {
            return new Dictionary<Guid, string>();
        }

        ProjectSpeaker[] speakers =
        [
            .. state.Speakers
                .OrderBy(static speaker => speaker.CreatedAtUtc)
                .ThenBy(static speaker => speaker.Id)
        ];

        // Fresh projects have no speakers yet (diarization hasn't run), and pre-flight
        // resolves the voice plan against that empty state. Index- and name-based keys
        // cannot be validated until speakers exist, so defer matching: pre-flight
        // proceeds on fallback voices and the TTS stage re-resolves — rejecting
        // genuinely bad keys — once speakers are known.
        if (speakers.Length == 0)
        {
            return new Dictionary<Guid, string>();
        }

        var resolved = new Dictionary<Guid, string>();
        foreach ((string key, string voiceId) in overrides)
        {
            if (!TryMatchSpeaker(speakers, key, out ProjectSpeaker? speaker) || speaker is null)
            {
                string known = speakers.Length == 0
                    ? "(none)"
                    : string.Join(", ", speakers.Select(static candidate => candidate.DisplayName));
                throw new InvalidOperationException(
                    $"Voice override '{key}' did not match a speaker. Known speakers: {known}. Use a display name (Speaker 1) or Whisper-style id (SPEAKER_00).");
            }

            resolved[speaker.Id] = voiceId;
        }

        return resolved;
    }

    internal static bool TryMatchSpeaker(
        IReadOnlyList<ProjectSpeaker> speakers,
        string key,
        out ProjectSpeaker? speaker)
    {
        speaker = null;
        if (string.IsNullOrWhiteSpace(key) || speakers.Count == 0)
        {
            return false;
        }

        string trimmed = key.Trim();
        speaker = Guid.TryParse(trimmed, out Guid speakerId)
            ? speakers.FirstOrDefault(candidate => candidate.Id == speakerId)
            : speakers.FirstOrDefault(candidate => candidate.DisplayName.Equals(trimmed, StringComparison.OrdinalIgnoreCase))
              ?? FindByWhisperIndex(speakers, trimmed);
        return speaker is not null;
    }

    /// <summary>Resolves a Whisper-style <c>SPEAKER_00</c> key to the speaker at that index.</summary>
    private static ProjectSpeaker? FindByWhisperIndex(IReadOnlyList<ProjectSpeaker> speakers, string key)
    {
        const string whisperPrefix = "speaker_";
        if (!key.StartsWith(whisperPrefix, StringComparison.OrdinalIgnoreCase) ||
            !int.TryParse(key[whisperPrefix.Length..], NumberStyles.Integer, CultureInfo.InvariantCulture, out int index))
        {
            return null;
        }

        return index >= 0 && index < speakers.Count ? speakers[index] : null;
    }

    private static bool IsVoiceLanguageMatch(string voiceLanguageCode, string? targetLanguageCode)
    {
        string? normalizedVoiceLanguage = NormalizeVoiceLanguageCode(voiceLanguageCode);
        string? normalizedTargetLanguage = NormalizeVoiceLanguageCode(targetLanguageCode);
        if (normalizedVoiceLanguage is null || normalizedTargetLanguage is null)
        {
            return true;
        }

        return string.Equals(normalizedVoiceLanguage, normalizedTargetLanguage, StringComparison.Ordinal) ||
               normalizedVoiceLanguage.StartsWith($"{normalizedTargetLanguage}-", StringComparison.Ordinal) ||
               normalizedTargetLanguage.StartsWith($"{normalizedVoiceLanguage}-", StringComparison.Ordinal);
    }

    private static string? NormalizeVoiceLanguageCode(string? languageCode) =>
        string.IsNullOrWhiteSpace(languageCode)
            ? null
            : languageCode.Trim().Replace('_', '-').ToLowerInvariant();

    /// <summary>
    /// When every voice this run will synthesize is a Qwen3 preset, point preflight at CustomVoice.
    /// Clone runs are left unchanged, because that single alias is the clone model. An alias that
    /// is already CustomVoice is left unchanged, including a larger tier. A mix of preset and stock
    /// voices keeps the stock alias and is provisioned alongside CustomVoice by
    /// <see cref="RequiresCompanionCustomVoiceModel"/>.
    /// </summary>
    internal static RuntimeModelSelections ApplyPresetVoiceModelSelection(
        RuntimeModelSelections selections,
        DubbingSessionOptions options,
        TranscriptProjectState? state)
    {
        if (state is null || !CanRetargetToCustomVoice(selections, options))
        {
            return selections;
        }

        List<string> allVoiceIds = EnumerateUnattendedStockVoiceIds(
            state,
            ResolveUnattendedVoicePlan(state, options, selections.TtsModelAlias),
            options.TargetLanguageCode).ToList();
        return allVoiceIds.Count > 0 && allVoiceIds.All(Qwen3TtsDefaults.IsPresetVoiceId)
            ? selections with
            {
                TtsModelAlias = Qwen3TtsDefaults.ResolveCustomVoiceAlias(tier: null),
                SelectionIntents = Enum.GetValues<RuntimeStage>().ToDictionary(stage => stage, selections.GetSelectionIntent)
            }
            : selections;
    }

    /// <summary>
    /// True when synthesis will use a CustomVoice preset while the selected TTS alias is a
    /// different model. Preflight must provision both.
    /// </summary>
    internal static bool RequiresCompanionCustomVoiceModel(
        RuntimeModelSelections selections,
        DubbingSessionOptions options,
        TranscriptProjectState? state)
    {
        if (state is null || IsCustomVoiceSelection(selections))
        {
            return false;
        }

        // Diarization may not have produced speakers yet. The later voice plan can
        // still choose a preset through an override or the language fallback.
        if (state.Speakers.Count == 0 && state.TranscriptSegments.Count == 0)
        {
            return FreshProjectMayUsePreset(options);
        }

        List<string> voiceIds = EnumerateUnattendedStockVoiceIds(
            state,
            ResolveUnattendedVoicePlan(state, options, selections.TtsModelAlias),
            options.TargetLanguageCode).ToList();
        return voiceIds.Any(Qwen3TtsDefaults.IsPresetVoiceId);
    }

    private static bool IsCustomVoiceSelection(RuntimeModelSelections selections) =>
        Qwen3TtsDefaults.IsCustomVoiceAlias(selections.TtsModelAlias);

    /// <summary>
    /// Clone runs and CustomVoice selections keep their own model, so neither retargets to CustomVoice.
    /// </summary>
    private static bool CanRetargetToCustomVoice(RuntimeModelSelections selections, DubbingSessionOptions options) =>
        !RequestsVoiceCloning(options) && !IsCustomVoiceSelection(selections);

    private static bool FreshProjectMayUsePreset(DubbingSessionOptions options) =>
        (options.VoiceAssignmentOverrides?.Values.Any(Qwen3TtsDefaults.IsPresetVoiceId) ?? false) ||
        (options.AutoAssignFallbackVoices && !options.UseVoiceCloning &&
         Qwen3TtsDefaults.SupportsLanguage(options.TargetLanguageCode) &&
         (Qwen3TtsDefaults.IsCustomVoiceAlias(TryGetPreferredTtsAlias(options)) ||
          !StockTtsVoiceMatcher.SupportsKokoro(options.TargetLanguageCode)));

    private static string? TryGetPreferredTtsAlias(DubbingSessionOptions options)
    {
        if (options.ModelPreferences is null)
        {
            return null;
        }

        foreach (KeyValuePair<string, string> pair in options.ModelPreferences)
        {
            if (pair.Key.Equals(StageNames.Tts, StringComparison.OrdinalIgnoreCase))
            {
                return pair.Value;
            }
        }

        return null;
    }

    private readonly record struct UnattendedVoicePlan(
        IReadOnlyDictionary<Guid, string> ExplicitVoiceIds,
        Dictionary<Guid, string>? FallbackVoiceIds,
        Dictionary<Guid, bool>? CloneBySpeaker);

    /// <summary>
    /// Speakers this unattended run will synthesize without cloning, and the voice id each one uses.
    /// Shared by the TTS request and preset preflight so fallback and clone filters cannot drift.
    /// </summary>
    private static UnattendedVoicePlan ResolveUnattendedVoicePlan(
        TranscriptProjectState state,
        DubbingSessionOptions options,
        string? preferredTtsModelAlias = null)
    {
        IReadOnlyDictionary<Guid, string> explicitVoiceIds = ResolveVoiceAssignmentOverrides(
            state,
            options.VoiceAssignmentOverrides);

        Dictionary<Guid, bool>? cloneBySpeaker = null;
        if (options.VoiceCloneBySpeakerId is not null)
        {
            cloneBySpeaker = new Dictionary<Guid, bool>(options.VoiceCloneBySpeakerId);
        }
        else if (options.UseVoiceCloning && state.Speakers.Count > 0)
        {
            cloneBySpeaker = state.Speakers.ToDictionary(
                static speaker => speaker.Id,
                speaker => !explicitVoiceIds.ContainsKey(speaker.Id));
        }

        string? ttsAlias = preferredTtsModelAlias ?? TryGetPreferredTtsAlias(options);
        Dictionary<Guid, string>? fallbackVoiceIds = null;
        if (options.AutoAssignFallbackVoices)
        {
            fallbackVoiceIds = BuildUnattendedFallbackVoiceIds(state, options.TargetLanguageCode, ttsAlias);
            if (fallbackVoiceIds is not null)
            {
                fallbackVoiceIds = fallbackVoiceIds
                    .Where(pair => !explicitVoiceIds.ContainsKey(pair.Key))
                    .Where(pair => cloneBySpeaker?.GetValueOrDefault(pair.Key) != true)
                    .ToDictionary(static pair => pair.Key, static pair => pair.Value);
                if (fallbackVoiceIds.Count == 0)
                {
                    fallbackVoiceIds = null;
                }
            }
        }

        return new UnattendedVoicePlan(explicitVoiceIds, fallbackVoiceIds, cloneBySpeaker);
    }

    private static IEnumerable<string> EnumerateUnattendedStockVoiceIds(
        TranscriptProjectState state,
        UnattendedVoicePlan plan,
        string? targetLanguage)
    {
        // GenerateTtsForAllSpeakersAsync skips speakers with no transcript segments, then
        // prefers a request override, then a persisted non-fallback assignment, then the fallback.
        HashSet<Guid> synthesizingSpeakerIds = state.TranscriptSegments
            .Select(static segment => segment.SpeakerId)
            .Where(static speakerId => speakerId is not null)
            .Select(static speakerId => speakerId!.Value)
            .ToHashSet();

        return EnumerateOverrides(plan, synthesizingSpeakerIds)
            .Concat(EnumerateFallbacks(plan, synthesizingSpeakerIds))
            .Concat(EnumerateAssignments(state, plan, synthesizingSpeakerIds, targetLanguage));
    }

    private static IEnumerable<string> EnumerateOverrides(UnattendedVoicePlan plan, HashSet<Guid> speakers) =>
        plan.ExplicitVoiceIds.Where(pair => speakers.Contains(pair.Key))
            .Where(pair => plan.CloneBySpeaker?.GetValueOrDefault(pair.Key) != true)
            .Select(pair => pair.Value).Where(voice => !string.IsNullOrWhiteSpace(voice))
            .Select(voice => voice.Trim());

    private static IEnumerable<string> EnumerateFallbacks(UnattendedVoicePlan plan, HashSet<Guid> speakers) =>
        (plan.FallbackVoiceIds ?? []).Where(pair => speakers.Contains(pair.Key))
            .Where(pair => !plan.ExplicitVoiceIds.ContainsKey(pair.Key))
            .Select(pair => pair.Value).Where(voice => !string.IsNullOrWhiteSpace(voice))
            .Select(voice => voice.Trim());

    private static IEnumerable<string> EnumerateAssignments(
        TranscriptProjectState state, UnattendedVoicePlan plan, HashSet<Guid> speakers, string? language) =>
        state.VoiceAssignments.Where(assignment => IsEligibleAssignment(assignment, plan, speakers))
            .Select(assignment => ResolveAssignmentVoice(assignment, language))
            .Where(voice => voice is not null).Select(voice => voice!);

    private static bool IsEligibleAssignment(VoiceAssignment assignment, UnattendedVoicePlan plan, HashSet<Guid> speakers) =>
        !assignment.IsFallback && speakers.Contains(assignment.SpeakerId) &&
        !plan.ExplicitVoiceIds.ContainsKey(assignment.SpeakerId) &&
        plan.CloneBySpeaker?.GetValueOrDefault(assignment.SpeakerId) != true &&
        plan.FallbackVoiceIds?.ContainsKey(assignment.SpeakerId) != true;

    private static string? ResolveAssignmentVoice(VoiceAssignment assignment, string? language)
    {
        if (!string.IsNullOrWhiteSpace(assignment.VoiceVariant))
        {
            return assignment.VoiceVariant.Trim();
        }

        // A prior clone run persists a clone-only model with no voice id. Synthesis
        // substitutes Kokoro where Kokoro speaks, and a Qwen preset where Qwen speaks.
        // Languages neither speaks stay unassigned, matching BuildUnattendedFallbackVoiceIds.
        // KokoroPrimaryAlias is a model alias, not a catalog voice id; it only has to fail
        // IsPresetVoiceId so this speaker is not counted as a Qwen preset.
        if (!VoiceCloningDefaults.IsCloneOnlyModelAlias(assignment.VoiceModelId))
        {
            return null;
        }

        if (StockTtsVoiceMatcher.SupportsKokoro(language))
        {
            return StockTtsDefaults.KokoroPrimaryAlias;
        }

        return Qwen3TtsDefaults.SupportsLanguage(language)
            ? Qwen3TtsDefaults.ResolveDefaultPresetVoiceId(language)
            : null;
    }
}
