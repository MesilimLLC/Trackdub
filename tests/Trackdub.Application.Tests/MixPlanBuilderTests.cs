using Trackdub.Contracts;
using Trackdub.Application.Mixing;
using Trackdub.Application.Projects;
using Trackdub.Application.Transcripts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain;
using Trackdub.Domain.Artifacts;
using Trackdub.Domain.Media;
using Trackdub.Domain.Mixing;
using Trackdub.Domain.Projects;
using Trackdub.Domain.StageRuns;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Trackdub.Domain.Tts;
using Trackdub.TestDoubles;

namespace Trackdub.Application.Tests;

public sealed class MixPlanBuilderTests
{
    [Theory]
    [InlineData(StageRunStatus.Failed)]
    [InlineData(StageRunStatus.Canceled)]
    [InlineData(StageRunStatus.Running)]
    public void Build_keeps_both_mix_stems_on_the_prior_completed_producer(StageRunStatus status)
    {
        TestProjectContext context = CreateContext();
        ProjectArtifact ambiance = CreateArtifact(context, ArtifactKind.Ambiance, "accepted/ambiance.wav", 0);
        (ProjectArtifact vocals, StageRunRecord run) = CreateCompletedStemPair(ref ambiance);
        StageRunRecord interrupted = StageRunRecord.Start(context.ProjectId, StageNames.Separation, run.StartedAtUtc.AddMinutes(1)) with { Status = status };
        ProjectArtifact orphanVocals = vocals with { Id = Guid.NewGuid(), StageRunId = interrupted.Id, RelativePath = "orphan/vocals.wav", CreatedAtUtc = vocals.CreatedAtUtc.AddMinutes(1) };
        ProjectArtifact orphanAmbiance = ambiance with { Id = Guid.NewGuid(), StageRunId = interrupted.Id, RelativePath = "orphan/ambiance.wav", CreatedAtUtc = ambiance.CreatedAtUtc.AddMinutes(1) };

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(context.ProjectId, context.MediaAssetId,
            [vocals, ambiance, orphanVocals, orphanAmbiance], [], [], [], StageRuns: [run, interrupted]));

        Assert.Equal(vocals.RelativePath, plan.VocalStemRelativePath);
        Assert.Equal(ambiance.RelativePath, plan.SourceAudioRelativePath);
    }

    [Fact]
    public async Task Timeline_media_is_prepared_once_by_content_and_default_source_is_not_duplicated()
    {
        string root = Path.Join(Path.GetTempPath(), "Trackdub.Timeline.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string input = Path.Join(root, "media.mp4");
            await File.WriteAllBytesAsync(input, [1, 2, 3], TestContext.Current.CancellationToken);
            TestProjectContext context = CreateContext();
            var now = DateTimeOffset.UtcNow;
            var project = new TrackdubProject(context.ProjectId, "Timeline", now, now);
            var media = new MediaAsset(context.MediaAssetId, context.ProjectId, "source.mp4", "source.mp4", "source",
                100, now, "mp4", 8d, true, true, now);
            var state = CreateState(project, media, [], [], [], []) with
            {
                ProjectUiSettings = new ProjectUiSettings(TimelinePlacements:
                [
                    new ProjectTimelineMediaPlacement("missing-old-source.mp4", "Source", 0, 0d, 8d),
                    new ProjectTimelineMediaPlacement(input, "Added media", 1, 2d, 1d),
                ])
            };
            var store = new FakeArtifactStore(root);
            var extraction = new TimelineTestExtractionService();
            var fingerprints = new FakeFileFingerprintService(new FileFingerprint("abc123", 3, now));
            var preparer = new TimelineMediaMixPreparer(store, extraction, fingerprints, new FakeMediaProbe());
            var clips = await TimelineMediaMixPreparer.PrepareAsync(preparer, state, TestContext.Current.CancellationToken);
            var clip = Assert.Single(clips);
            Assert.Equal(2d, clip.StartSeconds);
            Assert.Equal(1d, clip.DurationSeconds);
            Assert.True(File.Exists(store.GetPath(clip.AudioRelativePath)));
            await TimelineMediaMixPreparer.PrepareAsync(preparer, state, TestContext.Current.CancellationToken);
            Assert.Equal(1, extraction.Calls);
            await Assert.ThrowsAsync<InvalidOperationException>(() => TimelineMediaMixPreparer.PrepareAsync(null, state,
                TestContext.Current.CancellationToken));
            var invalid = state with
            {
                ProjectUiSettings = new ProjectUiSettings(TimelinePlacements:
                [new ProjectTimelineMediaPlacement(input, "Too long", 1, 7d, 2d)])
            };
            await Assert.ThrowsAsync<InvalidOperationException>(() => TimelineMediaMixPreparer.PrepareAsync(preparer, invalid,
                TestContext.Current.CancellationToken));
            File.Delete(input);
            await Assert.ThrowsAsync<FileNotFoundException>(() => TimelineMediaMixPreparer.PrepareAsync(preparer, state,
                TestContext.Current.CancellationToken));
        }
        finally { Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public async Task Timeline_media_at_origin_is_mixed_when_it_is_not_the_source_file()
    {
        string root = Path.Join(Path.GetTempPath(), "Trackdub.Timeline.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string source = Path.Join(root, "source.mp4");
            string added = Path.Join(root, "added.wav");
            await File.WriteAllBytesAsync(source, [1], TestContext.Current.CancellationToken);
            await File.WriteAllBytesAsync(added, [2], TestContext.Current.CancellationToken);
            TestProjectContext context = CreateContext();
            DateTimeOffset now = DateTimeOffset.UtcNow;
            var project = new TrackdubProject(context.ProjectId, "Timeline", now, now);
            var media = new MediaAsset(context.MediaAssetId, context.ProjectId, source, "source.mp4", "source",
                100, now, "mp4", 8d, true, true, now);
            var state = CreateState(project, media, [], [], [], []) with
            {
                ProjectUiSettings = new ProjectUiSettings(TimelinePlacements:
                [
                    new ProjectTimelineMediaPlacement(source, "Source", 1, 3d, 1d),
                    new ProjectTimelineMediaPlacement(added, "Added at origin", 0, 0d, 0.5d),
                ])
            };
            var preparer = new TimelineMediaMixPreparer(
                new FakeArtifactStore(root),
                new TimelineTestExtractionService(),
                new FakeFileFingerprintService(new FileFingerprint("origin", 1, now)),
                new FakeMediaProbe());

            MixMediaClip clip = Assert.Single(await TimelineMediaMixPreparer.PrepareAsync(
                preparer, state, TestContext.Current.CancellationToken));

            Assert.Equal(0d, clip.StartSeconds);
            Assert.Equal(0.5d, clip.DurationSeconds);
        }
        finally { Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public async Task Timeline_media_marked_primary_is_excluded_even_after_the_path_changes()
    {
        string root = Path.Join(Path.GetTempPath(), "Trackdub.Timeline.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string added = Path.Join(root, "added.wav");
            await File.WriteAllBytesAsync(added, [2], TestContext.Current.CancellationToken);
            TestProjectContext context = CreateContext();
            DateTimeOffset now = DateTimeOffset.UtcNow;
            var project = new TrackdubProject(context.ProjectId, "Timeline", now, now);
            var media = new MediaAsset(context.MediaAssetId, context.ProjectId, Path.Join(root, "relinked.mp4"), "relinked.mp4", "source",
                100, now, "mp4", 8d, true, true, now);
            var state = CreateState(project, media, [], [], [], []) with
            {
                ProjectUiSettings = new ProjectUiSettings(TimelinePlacements:
                [
                    new ProjectTimelineMediaPlacement("old-source.mp4", "Old source", 0, 0d, 8d, IsPrimarySource: true),
                    new ProjectTimelineMediaPlacement(added, "Added at origin", 0, 0d, 0.5d),
                ])
            };
            var preparer = new TimelineMediaMixPreparer(
                new FakeArtifactStore(root),
                new TimelineTestExtractionService(),
                new FakeFileFingerprintService(new FileFingerprint("primary", 1, now)),
                new FakeMediaProbe());

            MixMediaClip clip = Assert.Single(await TimelineMediaMixPreparer.PrepareAsync(
                preparer, state, TestContext.Current.CancellationToken));

            Assert.Equal(0d, clip.StartSeconds);
        }
        finally { Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public async Task Timeline_media_shorter_than_its_slot_fails()
    {
        string root = Path.Join(Path.GetTempPath(), "Trackdub.Timeline.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string added = Path.Join(root, "short.wav");
            await File.WriteAllBytesAsync(added, [1], TestContext.Current.CancellationToken);
            TestProjectContext context = CreateContext();
            DateTimeOffset now = DateTimeOffset.UtcNow;
            var media = new MediaAsset(context.MediaAssetId, context.ProjectId, Path.Join(root, "source.mp4"), "source.mp4", "source",
                100, now, "mp4", 8d, true, true, now);
            var state = CreateState(new TrackdubProject(context.ProjectId, "Timeline", now, now), media, [], [], [], []) with
            {
                ProjectUiSettings = new ProjectUiSettings(TimelinePlacements:
                [
                    new ProjectTimelineMediaPlacement(added, "Short", 1, 1d, 1d),
                ])
            };
            var probe = new FakeMediaProbe
            {
                Snapshot = new MediaProbeSnapshot(
                    "wav",
                    "WAV",
                    DurationSeconds: 0.2d,
                    BitRate: null,
                    AudioStreams: [new MediaAudioStream(0, "pcm", 2, 48000, 0.2d)],
                    VideoStreams: [],
                    SubtitleStreams: [])
            };
            var preparer = new TimelineMediaMixPreparer(
                new FakeArtifactStore(root),
                new TimelineTestExtractionService(),
                new FakeFileFingerprintService(new FileFingerprint("short", 1, now)),
                probe);

            InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(() =>
                TimelineMediaMixPreparer.PrepareAsync(preparer, state, TestContext.Current.CancellationToken));
            Assert.Contains("longer than its audio", error.Message, StringComparison.Ordinal);
        }
        finally { Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public async Task Timeline_media_uses_the_first_audio_stream_duration()
    {
        string root = Path.Join(Path.GetTempPath(), "Trackdub.Timeline.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string added = Path.Join(root, "multi.mkv");
            await File.WriteAllBytesAsync(added, [1], TestContext.Current.CancellationToken);
            TestProjectContext context = CreateContext();
            DateTimeOffset now = DateTimeOffset.UtcNow;
            var media = new MediaAsset(context.MediaAssetId, context.ProjectId, Path.Join(root, "source.mp4"), "source.mp4", "source",
                100, now, "mp4", 8d, true, true, now);
            var state = CreateState(new TrackdubProject(context.ProjectId, "Timeline", now, now), media, [], [], [], []) with
            {
                ProjectUiSettings = new ProjectUiSettings(TimelinePlacements:
                [
                    new ProjectTimelineMediaPlacement(added, "Commentary", 1, 1d, 1d),
                ])
            };
            var extraction = new TimelineTestExtractionService();
            var probe = new FakeMediaProbe
            {
                Snapshot = new MediaProbeSnapshot(
                    "matroska",
                    "Matroska",
                    DurationSeconds: 5d,
                    BitRate: null,
                    AudioStreams:
                    [
                        new MediaAudioStream(1, "aac", 2, 48000, 0.2d),
                        new MediaAudioStream(2, "aac", 2, 48000, 5d),
                    ],
                    VideoStreams: [],
                    SubtitleStreams: [])
            };
            var preparer = new TimelineMediaMixPreparer(
                new FakeArtifactStore(root),
                extraction,
                new FakeFileFingerprintService(new FileFingerprint("multi", 1, now)),
                probe);

            InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(() =>
                TimelineMediaMixPreparer.PrepareAsync(preparer, state, TestContext.Current.CancellationToken));

            Assert.Contains("longer than its audio", error.Message, StringComparison.Ordinal);
            Assert.Equal(0, extraction.Calls);
        }
        finally { Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public async Task Timeline_media_past_the_source_audio_end_fails()
    {
        string root = Path.Join(Path.GetTempPath(), "Trackdub.Timeline.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string added = Path.Join(root, "late.wav");
            await File.WriteAllBytesAsync(added, [1], TestContext.Current.CancellationToken);
            TestProjectContext context = CreateContext();
            DateTimeOffset now = DateTimeOffset.UtcNow;
            ProjectArtifact normalized = CreateArtifact(
                context,
                ArtifactKind.NormalizedAudio,
                ProjectArtifactPaths.NormalizedAudioRelativePath,
                createdOffsetSeconds: 0,
                durationSeconds: 4d);
            var media = new MediaAsset(context.MediaAssetId, context.ProjectId, Path.Join(root, "source.mp4"), "source.mp4", "source",
                100, now, "mp4", 8d, true, true, now);
            var state = CreateState(new TrackdubProject(context.ProjectId, "Timeline", now, now), media, [normalized], [], [], []) with
            {
                ProjectUiSettings = new ProjectUiSettings(TimelinePlacements:
                [
                    new ProjectTimelineMediaPlacement(added, "Late", 1, 3.5d, 1d),
                ])
            };
            var preparer = new TimelineMediaMixPreparer(
                new FakeArtifactStore(root),
                new TimelineTestExtractionService(),
                new FakeFileFingerprintService(new FileFingerprint("late", 1, now)),
                new FakeMediaProbe());

            await Assert.ThrowsAsync<InvalidOperationException>(() =>
                TimelineMediaMixPreparer.PrepareAsync(preparer, state, TestContext.Current.CancellationToken));
        }
        finally { Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public async Task Timeline_video_without_audio_is_skipped()
    {
        string root = Path.Join(Path.GetTempPath(), "Trackdub.Timeline.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string added = Path.Join(root, "silent.mp4");
            await File.WriteAllBytesAsync(added, [1], TestContext.Current.CancellationToken);
            TestProjectContext context = CreateContext();
            DateTimeOffset now = DateTimeOffset.UtcNow;
            var media = new MediaAsset(context.MediaAssetId, context.ProjectId, Path.Join(root, "source.mp4"), "source.mp4", "source",
                100, now, "mp4", 8d, true, true, now);
            var state = CreateState(new TrackdubProject(context.ProjectId, "Timeline", now, now), media, [], [], [], []) with
            {
                ProjectUiSettings = new ProjectUiSettings(TimelinePlacements:
                [
                    new ProjectTimelineMediaPlacement(added, "Silent", 1, 1d, 1d),
                ])
            };
            var extraction = new TimelineTestExtractionService();
            var probe = new FakeMediaProbe
            {
                Snapshot = new MediaProbeSnapshot(
                    "mp4",
                    "MP4",
                    DurationSeconds: 1d,
                    BitRate: null,
                    AudioStreams: [],
                    VideoStreams: [new MediaVideoStream(0, "h264", 64, 64, 24d, 1d)],
                    SubtitleStreams: [])
            };
            var preparer = new TimelineMediaMixPreparer(
                new FakeArtifactStore(root),
                extraction,
                new FakeFileFingerprintService(new FileFingerprint("silent", 1, now)),
                probe);

            IReadOnlyList<MixMediaClip> clips = await TimelineMediaMixPreparer.PrepareAsync(
                preparer, state, TestContext.Current.CancellationToken);

            Assert.Empty(clips);
            Assert.Equal(0, extraction.Calls);
        }
        finally { Directory.Delete(root, recursive: true); }
    }

    private sealed class TimelineTestExtractionService : IAudioExtractionService
    {
        public int Calls { get; private set; }
        public async Task<AudioExtractionResult> ExtractNormalizedAudioAsync(string sourcePath, string destinationPath,
            CancellationToken cancellationToken, int? maxEncoderThreads = null)
        {
            Calls++;
            await File.WriteAllBytesAsync(destinationPath, [1, 2, 3], cancellationToken);
            return new AudioExtractionResult(destinationPath, 1d, 48000, 2, 48000);
        }
        public Task<AudioExtractionResult> ExtractStemSeparationAudioAsync(string sourcePath, string destinationPath,
            CancellationToken cancellationToken) => throw new NotSupportedException();
    }

    [Fact]
    public void Build_prefers_ambiance_and_places_fresh_takes_at_segment_start()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment first = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 1.0d, 2.5d, "Hello", context.SpeakerId);
        TranscriptSegment second = TranscriptSegment.Create(context.TranscriptRevisionId, 1, 3.0d, 4.0d, "Again", context.SpeakerId);
        ProjectArtifact normalized = CreateArtifact(context, ArtifactKind.NormalizedAudio, ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0);
        ProjectArtifact ambiance = CreateArtifact(context, ArtifactKind.Ambiance, "artifacts/stems/run/ambiance.wav", createdOffsetSeconds: 1);
        (ProjectArtifact vocals, StageRunRecord run) = CreateCompletedStemPair(ref ambiance);
        ProjectArtifact takeArtifact = CreateArtifact(context, ArtifactKind.TtsTake, "artifacts/tts/take-0001.wav", createdOffsetSeconds: 2, durationSeconds: 2.25d);
        TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: first.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 60000, sampleRate: 48000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, ambiance, vocals, takeArtifact],
            [first, second],
            [],
            [take],
            StageRuns: [run]));

        Assert.Equal(ArtifactKind.Ambiance, plan.SourceAudioKind);
        Assert.Equal(ambiance.RelativePath, plan.SourceAudioRelativePath);
        Assert.Equal(2, plan.SpeechClips.Count);
        Assert.Equal(first.StartSeconds, plan.SpeechClips[0].StartSeconds);
        Assert.Equal(takeArtifact.RelativePath, plan.SpeechClips[0].TakeRelativePath);
        Assert.False(plan.SpeechClips[0].IsSilentGap);
        Assert.True(plan.SpeechClips[1].IsSilentGap);
        Assert.Single(plan.Warnings);
        Assert.Contains("Missing", plan.Warnings[0].Message);
        MixDuckRegion duckRegion = Assert.Single(plan.DuckingRegions);
        Assert.Equal(first.SegmentIndex, duckRegion.SegmentIndex);
        Assert.True(duckRegion.StartSeconds < first.StartSeconds);
        Assert.True(duckRegion.EndSeconds > first.EndSeconds);
        Assert.Equal(first.StartSeconds + takeArtifact.DurationSeconds!.Value + 0.18d, duckRegion.EndSeconds, precision: 3);
        Assert.Equal(0d, plan.DuckingGainDb);
        Assert.Equal(0d, duckRegion.GainDb);
    }

    [Fact]
    public void Build_warns_when_take_extends_past_source_audio_end()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 8.0d, 9.5d, "Hello", context.SpeakerId);
        ProjectArtifact normalized = CreateArtifact(
            context,
            ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath,
            createdOffsetSeconds: 0,
            durationSeconds: 10.0d);
        ProjectArtifact takeArtifact = CreateArtifact(
            context,
            ArtifactKind.TtsTake,
            "artifacts/tts/take.wav",
            createdOffsetSeconds: 2,
            durationSeconds: 7.0d);
        TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: segment.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 168000, sampleRate: 24000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, takeArtifact],
            [segment],
            [],
            [take], StageRuns: null));

        // The 7.0s take at 8.0s may move back at most 4.0s, to 4.0s, and still ends at 11.0s —
        // 1.0s past the 10.0s source audio.
        MixPlanWarning warning = Assert.Single(plan.Warnings);
        Assert.Equal(MixPlanWarningCode.TakeExceedsSourceAudio, warning.Code);
        Assert.Equal(segment.SegmentIndex, warning.SegmentIndex);
        Assert.Contains("1", warning.Message);
    }

    [Theory]
    [InlineData(10.0d)]
    [InlineData(null)]
    public void Build_warns_when_take_extends_past_shorter_media_duration(double? sourceDurationSeconds)
    {
        // Export renders only through the media duration, so a take that fits the longer (or
        // unknown) source audio but passes the media end must still warn.
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 5.0d, 6.0d, "Hello", context.SpeakerId);
        ProjectArtifact normalized = CreateArtifact(
            context,
            ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath,
            createdOffsetSeconds: 0,
            durationSeconds: sourceDurationSeconds);
        ProjectArtifact takeArtifact = CreateArtifact(
            context,
            ArtifactKind.TtsTake,
            "artifacts/tts/take.wav",
            createdOffsetSeconds: 2,
            durationSeconds: 6.0d);
        TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: segment.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 144000, sampleRate: 24000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, takeArtifact],
            [segment],
            [],
            [take],
            MediaDurationSeconds: 6.0d, StageRuns: null));

        // The 6.0s take moves back from 5.0s to 1.0s (the 4.0s limit) and runs to 7.0s: inside the
        // 10.0s source audio but 1.0s past the 6.0s media end.
        MixPlanWarning warning = Assert.Single(plan.Warnings);
        Assert.Equal(MixPlanWarningCode.TakeExceedsSourceAudio, warning.Code);
        Assert.Contains("(6 s)", warning.Message);
    }

    [Fact]
    public void Build_does_not_warn_when_take_fits_within_source_audio()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 1.0d, 9.5d, "Hello", context.SpeakerId);
        ProjectArtifact normalized = CreateArtifact(
            context,
            ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath,
            createdOffsetSeconds: 0,
            durationSeconds: 10.0d);
        ProjectArtifact takeArtifact = CreateArtifact(
            context,
            ArtifactKind.TtsTake,
            "artifacts/tts/take.wav",
            createdOffsetSeconds: 2,
            durationSeconds: 8.0d);
        TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: segment.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 192000, sampleRate: 24000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, takeArtifact],
            [segment],
            [],
            [take], StageRuns: null));

        Assert.Empty(plan.Warnings);
    }

    [Fact]
    public void Build_does_not_warn_on_take_overrun_without_source_duration()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 8.0d, 9.5d, "Hello", context.SpeakerId);
        ProjectArtifact normalized = CreateArtifact(
            context,
            ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath,
            createdOffsetSeconds: 0,
            durationSeconds: null);
        ProjectArtifact takeArtifact = CreateArtifact(
            context,
            ArtifactKind.TtsTake,
            "artifacts/tts/take.wav",
            createdOffsetSeconds: 2,
            durationSeconds: 3.0d);
        TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: segment.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 72000, sampleRate: 24000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, takeArtifact],
            [segment],
            [],
            [take], StageRuns: null));

        Assert.Empty(plan.Warnings);
    }

    [Fact]
    public void Build_does_not_warn_when_take_fits_exactly_to_source_audio_end()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 7.0d, 9.5d, "Hello", context.SpeakerId);
        ProjectArtifact normalized = CreateArtifact(
            context,
            ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath,
            createdOffsetSeconds: 0,
            durationSeconds: 10.0d);
        ProjectArtifact takeArtifact = CreateArtifact(
            context,
            ArtifactKind.TtsTake,
            "artifacts/tts/take.wav",
            createdOffsetSeconds: 2,
            durationSeconds: 3.0d);
        TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: segment.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 72000, sampleRate: 24000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, takeArtifact],
            [segment],
            [],
            [take], StageRuns: null));

        Assert.Empty(plan.Warnings);
    }

    [Theory]
    [InlineData(10.009d, 10d, false)]
    [InlineData(10.011d, 10d, true)]
    [InlineData(10d, 9d, true)]
    public void Build_applies_overrun_tolerance_and_media_boundary(double takeDuration, double mediaDuration, bool warns)
    {
        // The segment starts at 0, so the tail reflow has nowhere to move the take.
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 0d, 9.5d, "Hello", context.SpeakerId);
        ProjectArtifact source = CreateArtifact(context, ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0, durationSeconds: 10d);
        ProjectArtifact artifact = CreateArtifact(context, ArtifactKind.TtsTake,
            "artifacts/tts/take.wav", createdOffsetSeconds: 2, durationSeconds: takeDuration);
        TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: 0)
            .Complete(artifact.Id, durationSamples: 72000, sampleRate: 24000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId, context.MediaAssetId, [source, artifact], [segment], [], [take],
            MediaDurationSeconds: mediaDuration, StageRuns: null));

        Assert.Equal(warns, plan.Warnings.Any(warning => warning.Code == MixPlanWarningCode.TakeExceedsSourceAudio));
    }

    [Fact]
    public void Build_uses_normalized_audio_for_original_pan_reference_when_ambiance_is_source_lane()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 1.0d, 2.5d, "Hello", context.SpeakerId);
        ProjectArtifact normalized = CreateArtifact(
            context,
            ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath,
            createdOffsetSeconds: 0,
            channelCount: 2);
        ProjectArtifact ambiance = CreateArtifact(
            context,
            ArtifactKind.Ambiance,
            "artifacts/stems/run/ambiance.wav",
            createdOffsetSeconds: 1,
            channelCount: 1);
        (ProjectArtifact vocals, StageRunRecord run) = CreateCompletedStemPair(ref ambiance);
        ProjectArtifact takeArtifact = CreateArtifact(context, ArtifactKind.TtsTake, "artifacts/tts/take.wav", createdOffsetSeconds: 2, durationSeconds: 1d);
        TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: segment.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 48000, sampleRate: 48000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, ambiance, vocals, takeArtifact],
            [segment],
            [],
            [take],
            RestoreOriginalPan: true,
            StageRuns: [run]));

        Assert.Equal(ArtifactKind.Ambiance, plan.SourceAudioKind);
        Assert.Equal(ambiance.RelativePath, plan.SourceAudioRelativePath);
        Assert.Equal(ProjectArtifactPaths.NormalizedAudioRelativePath, plan.OriginalMixAudioRelativePath);
        Assert.Equal(2, plan.OutputChannelCount);
        Assert.True(plan.RestoreOriginalPan);
    }

    [Fact]
    public void Build_filters_original_pan_reference_to_current_media_asset()
    {
        TestProjectContext context = CreateContext();
        Guid otherMediaAssetId = Guid.NewGuid();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 1.0d, 2.5d, "Hello", context.SpeakerId);
        ProjectArtifact normalized = CreateArtifact(
            context,
            ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath,
            createdOffsetSeconds: 0,
            channelCount: 2);
        ProjectArtifact newerOtherMediaNormalized = normalized with
        {
            Id = Guid.NewGuid(),
            MediaAssetId = otherMediaAssetId,
            RelativePath = "media/other-normalized.wav",
            CreatedAtUtc = normalized.CreatedAtUtc.AddMinutes(5)
        };
        ProjectArtifact ambiance = CreateArtifact(
            context,
            ArtifactKind.Ambiance,
            "artifacts/stems/run/ambiance.wav",
            createdOffsetSeconds: 1,
            channelCount: 1);
        (ProjectArtifact vocals, StageRunRecord run) = CreateCompletedStemPair(ref ambiance);
        ProjectArtifact takeArtifact = CreateArtifact(context, ArtifactKind.TtsTake, "artifacts/tts/take.wav", createdOffsetSeconds: 2, durationSeconds: 1d);
        TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: segment.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 48000, sampleRate: 48000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [newerOtherMediaNormalized, normalized, ambiance, vocals, takeArtifact],
            [segment],
            [],
            [take],
            RestoreOriginalPan: true,
            StageRuns: [run]));

        Assert.Equal(ProjectArtifactPaths.NormalizedAudioRelativePath, plan.OriginalMixAudioRelativePath);
    }

    [Fact]
    public void Build_defaults_to_original_mix_ducking_when_ambiance_is_unavailable()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 1.0d, 2.5d, "Hello", context.SpeakerId);
        ProjectArtifact normalized = CreateArtifact(context, ArtifactKind.NormalizedAudio, ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0);
        ProjectArtifact takeArtifact = CreateArtifact(context, ArtifactKind.TtsTake, "artifacts/tts/take-0001.wav", createdOffsetSeconds: 1, durationSeconds: 1.2d);
        TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: segment.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 57600, sampleRate: 48000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, takeArtifact],
            [segment],
            [],
            [take], StageRuns: null));

        Assert.Equal(ArtifactKind.NormalizedAudio, plan.SourceAudioKind);
        Assert.Equal(-13d, plan.DuckingGainDb);
        Assert.Equal(-13d, Assert.Single(plan.DuckingRegions).GainDb);
    }

    [Fact]
    public void Build_ignores_legacy_ambiance_and_uses_original_mix_ducking()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 1.0d, 2.5d, "Hello", context.SpeakerId);
        ProjectArtifact normalized = CreateArtifact(context, ArtifactKind.NormalizedAudio, ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0);
        ProjectArtifact legacyAmbiance = CreateArtifact(
            context,
            ArtifactKind.Ambiance,
            "artifacts/stems/hush/ambiance.wav",
            createdOffsetSeconds: 1,
            provenance: "generated-hush-dialogue-ambiance;model=hush-dialogue");
        ProjectArtifact takeArtifact = CreateArtifact(context, ArtifactKind.TtsTake, "artifacts/tts/take-0001.wav", createdOffsetSeconds: 2, durationSeconds: 1.2d);
        TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: segment.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 57600, sampleRate: 48000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, legacyAmbiance, takeArtifact],
            [segment],
            [],
            [take], StageRuns: null));

        Assert.Equal(ArtifactKind.NormalizedAudio, plan.SourceAudioKind);
        Assert.Equal(ProjectArtifactPaths.NormalizedAudioRelativePath, plan.SourceAudioRelativePath);
        Assert.Equal(-13d, plan.DuckingGainDb);
    }

    [Fact]
    public void Build_preserves_explicit_ducking_gain_for_ambiance_source()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 0.0d, 1.0d, "Hello", context.SpeakerId);
        ProjectArtifact ambiance = CreateArtifact(context, ArtifactKind.Ambiance, "artifacts/stems/run/ambiance.wav", createdOffsetSeconds: 0);
        (ProjectArtifact vocals, StageRunRecord run) = CreateCompletedStemPair(ref ambiance);
        ProjectArtifact takeArtifact = CreateArtifact(context, ArtifactKind.TtsTake, "artifacts/tts/take.wav", createdOffsetSeconds: 1, durationSeconds: 1d);
        TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: segment.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 48000, sampleRate: 48000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [ambiance, vocals, takeArtifact],
            [segment],
            [],
            [take],
            DuckingGainDb: -6d,
            StageRuns: [run]));

        Assert.Equal(-6d, plan.DuckingGainDb);
        Assert.Equal(-6d, Assert.Single(plan.DuckingRegions).GainDb);
    }

    [Fact]
    public void Build_replaces_stale_take_with_silent_gap_warning()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 0.0d, 1.0d, "Hello", context.SpeakerId);
        ProjectArtifact normalized = CreateArtifact(context, ArtifactKind.NormalizedAudio, ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0);
        ProjectArtifact takeArtifact = CreateArtifact(context, ArtifactKind.TtsTake, "artifacts/tts/take.wav", createdOffsetSeconds: 1, durationSeconds: 1d);
        TtsTake staleTake = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: segment.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 48000, sampleRate: 48000, provider: "fake")
            .MarkStale();

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, takeArtifact],
            [segment],
            [],
            [staleTake], StageRuns: null));

        MixSpeechClip clip = Assert.Single(plan.SpeechClips);
        Assert.True(clip.IsSilentGap);
        Assert.Null(clip.TakeRelativePath);
        Assert.Empty(plan.DuckingRegions);
        MixPlanWarning warning = Assert.Single(plan.Warnings);
        Assert.Contains("stale", warning.Message);
        Assert.Equal("Segment 1", warning.SegmentReference);
    }

    [Fact]
    public void Build_uses_newest_fresh_take_when_absolute_latest_take_is_stale()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 0.0d, 1.0d, "Hello", context.SpeakerId);
        TranslatedSegment translatedSegment = TranslatedSegment.Create(context.TranslationRevisionId, 0, 0.0d, 1.0d, "Hola");
        ProjectArtifact normalized = CreateArtifact(context, ArtifactKind.NormalizedAudio, ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0);
        ProjectArtifact freshArtifact = CreateArtifact(context, ArtifactKind.TtsTake, "artifacts/tts/fresh.wav", createdOffsetSeconds: 1, durationSeconds: 1d);
        ProjectArtifact staleArtifact = CreateArtifact(context, ArtifactKind.TtsTake, "artifacts/tts/stale.wav", createdOffsetSeconds: 2, durationSeconds: 1d);
        TtsTake freshTake = TtsTake.Create(
                context.ProjectId,
                context.VoiceAssignmentId,
                translatedSegment.Id,
                segment.SegmentIndex,
                ComputeTtsTextHash(segment.SegmentIndex, translatedSegment.Text))
            .Complete(freshArtifact.Id, durationSamples: 48000, sampleRate: 48000, provider: "fake");
        TtsTake staleTake = TtsTake.Create(
                context.ProjectId,
                context.VoiceAssignmentId,
                translatedSegment.Id,
                segment.SegmentIndex,
                ComputeTtsTextHash(segment.SegmentIndex, "Old text"))
            .Complete(staleArtifact.Id, durationSamples: 48000, sampleRate: 48000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, freshArtifact, staleArtifact],
            [segment],
            [translatedSegment],
            [freshTake, staleTake], StageRuns: null));

        MixSpeechClip clip = Assert.Single(plan.SpeechClips);
        Assert.False(clip.IsSilentGap);
        Assert.Equal(freshTake.Id, clip.TakeId);
        Assert.Equal(freshArtifact.RelativePath, clip.TakeRelativePath);
        Assert.Empty(plan.Warnings);
    }

    [Fact]
    public void Build_replaces_hashed_take_without_current_translation_with_silent_gap_warning()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 0.0d, 1.0d, "Hello", context.SpeakerId);
        ProjectArtifact normalized = CreateArtifact(context, ArtifactKind.NormalizedAudio, ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0);
        ProjectArtifact takeArtifact = CreateArtifact(context, ArtifactKind.TtsTake, "artifacts/tts/take.wav", createdOffsetSeconds: 1, durationSeconds: 1d);
        TtsTake orphanedTake = TtsTake.Create(
                context.ProjectId,
                context.VoiceAssignmentId,
                translatedSegmentId: Guid.NewGuid(),
                segmentIndex: segment.SegmentIndex,
                translatedTextHash: ComputeTtsTextHash(segment.SegmentIndex, "Hola"))
            .Complete(takeArtifact.Id, durationSamples: 48000, sampleRate: 48000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, takeArtifact],
            [segment],
            TranslatedSegments: [],
            [orphanedTake], StageRuns: null));

        MixSpeechClip clip = Assert.Single(plan.SpeechClips);
        Assert.True(clip.IsSilentGap);
        MixPlanWarning warning = Assert.Single(plan.Warnings);
        Assert.Contains("stale", warning.Message);
    }

    [Fact]
    public void Build_replaces_hashless_take_without_current_translation_with_silent_gap_warning()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 0.0d, 1.0d, "Hello", context.SpeakerId);
        ProjectArtifact normalized = CreateArtifact(context, ArtifactKind.NormalizedAudio, ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0);
        ProjectArtifact takeArtifact = CreateArtifact(context, ArtifactKind.TtsTake, "artifacts/tts/take.wav", createdOffsetSeconds: 1, durationSeconds: 1d);
        TtsTake orphanedTake = TtsTake.Create(
                context.ProjectId,
                context.VoiceAssignmentId,
                translatedSegmentId: Guid.NewGuid(),
                segmentIndex: segment.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 48000, sampleRate: 48000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, takeArtifact],
            [segment],
            TranslatedSegments: [],
            [orphanedTake], StageRuns: null));

        MixSpeechClip clip = Assert.Single(plan.SpeechClips);
        Assert.True(clip.IsSilentGap);
        MixPlanWarning warning = Assert.Single(plan.Warnings);
        Assert.Contains("stale", warning.Message);
    }

    [Fact]
    public void Build_keeps_hashed_take_when_translated_segment_id_changed_but_text_matches()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 0.0d, 1.0d, "Hello", context.SpeakerId);
        TranslatedSegment translatedSegment = TranslatedSegment.Create(context.TranslationRevisionId, 0, 0.0d, 1.0d, "Hola");
        ProjectArtifact normalized = CreateArtifact(context, ArtifactKind.NormalizedAudio, ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0);
        ProjectArtifact takeArtifact = CreateArtifact(context, ArtifactKind.TtsTake, "artifacts/tts/take.wav", createdOffsetSeconds: 1, durationSeconds: 1d);
        TtsTake take = TtsTake.Create(
                context.ProjectId,
                context.VoiceAssignmentId,
                translatedSegmentId: Guid.NewGuid(),
                segmentIndex: segment.SegmentIndex,
                translatedTextHash: ComputeTtsTextHash(segment.SegmentIndex, translatedSegment.Text))
            .Complete(takeArtifact.Id, durationSamples: 48000, sampleRate: 48000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, takeArtifact],
            [segment],
            [translatedSegment],
            [take], StageRuns: null));

        MixSpeechClip clip = Assert.Single(plan.SpeechClips);
        Assert.False(clip.IsSilentGap);
        Assert.Empty(plan.Warnings);
    }

    [Fact]
    public void Build_keeps_hashless_take_when_translated_segment_id_changed()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 0.0d, 1.0d, "Hello", context.SpeakerId);
        TranslatedSegment translatedSegment = TranslatedSegment.Create(context.TranslationRevisionId, 0, 0.0d, 1.0d, "Hola");
        ProjectArtifact normalized = CreateArtifact(context, ArtifactKind.NormalizedAudio, ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0);
        ProjectArtifact takeArtifact = CreateArtifact(context, ArtifactKind.TtsTake, "artifacts/tts/take.wav", createdOffsetSeconds: 1, durationSeconds: 1d);
        TtsTake take = TtsTake.Create(
                context.ProjectId,
                context.VoiceAssignmentId,
                translatedSegmentId: Guid.NewGuid(),
                segmentIndex: segment.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 48000, sampleRate: 48000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, takeArtifact],
            [segment],
            [translatedSegment],
            [take], StageRuns: null));

        MixSpeechClip clip = Assert.Single(plan.SpeechClips);
        Assert.False(clip.IsSilentGap);
        Assert.Empty(plan.Warnings);
    }

    [Fact]
    public void Build_keeps_segment_index_only_legacy_take_when_translation_exists()
    {
        TestProjectContext context = CreateContext();
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 0.0d, 1.0d, "Hello", context.SpeakerId);
        TranslatedSegment translatedSegment = TranslatedSegment.Create(context.TranslationRevisionId, 0, 0.0d, 1.0d, "Hola");
        ProjectArtifact normalized = CreateArtifact(context, ArtifactKind.NormalizedAudio, ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0);
        ProjectArtifact takeArtifact = CreateArtifact(context, ArtifactKind.TtsTake, "artifacts/tts/take.wav", createdOffsetSeconds: 1, durationSeconds: 1d);
        TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: segment.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 48000, sampleRate: 48000, provider: "fake");

        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, takeArtifact],
            [segment],
            [translatedSegment],
            [take], StageRuns: null));

        MixSpeechClip clip = Assert.Single(plan.SpeechClips);
        Assert.False(clip.IsSilentGap);
        Assert.Empty(plan.Warnings);
    }

    [Fact]
    public async Task MixPlanStore_round_trips_plan_as_project_json()
    {
        TestProjectContext context = CreateContext();
        var store = new FakeArtifactStore();
        var mixPlanStore = new MixPlanStore(store);
        var plan = new MixPlan(
            context.ProjectId,
            context.MediaAssetId,
            ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath,
            SourceGainDb: -3d,
            DubbedSpeechGainDb: 2d,
            DuckingGainDb: -12d,
            DuckingLeadSeconds: 0.05d,
            DuckingTailSeconds: 0.18d,
            DateTimeOffset.UtcNow,
            SpeechClips:
            [
                new MixSpeechClip(0, Guid.NewGuid(), 1d, 2d, Guid.NewGuid(), Guid.NewGuid(), "artifacts/tts/take.wav", 1d, IsSilentGap: false, WarningMessage: null)
            ],
            DuckingRegions:
            [
                new MixDuckRegion(0, Guid.NewGuid(), 0.95d, 2.18d, -12d)
            ],
            Warnings: [],
            OriginalMixAudioRelativePath: ProjectArtifactPaths.NormalizedAudioRelativePath,
            OutputChannelCount: 2,
            RestoreOriginalPan: true);

        await mixPlanStore.SaveAsync(plan, TestContext.Current.CancellationToken);
        MixPlan? reloaded = await mixPlanStore.LoadAsync(TestContext.Current.CancellationToken);

        Assert.NotNull(reloaded);
        Assert.Equal(plan.ProjectId, reloaded!.ProjectId);
        Assert.Equal(plan.SourceAudioRelativePath, reloaded.SourceAudioRelativePath);
        Assert.Equal(plan.SourceGainDb, reloaded.SourceGainDb);
        Assert.Equal(plan.SpeechClips[0].TakeRelativePath, reloaded.SpeechClips[0].TakeRelativePath);
        Assert.Equal(plan.DuckingRegions[0].EndSeconds, reloaded.DuckingRegions[0].EndSeconds);
        Assert.Equal(plan.OriginalMixAudioRelativePath, reloaded.OriginalMixAudioRelativePath);
        Assert.Equal(2, reloaded.OutputChannelCount);
        Assert.True(reloaded.RestoreOriginalPan);
    }

    [Fact]
    public async Task MixPlanStore_loads_legacy_plan_without_channel_metadata()
    {
        var store = new FakeArtifactStore();
        var mixPlanStore = new MixPlanStore(store);
        Guid projectId = Guid.NewGuid();
        string legacyJson = $$"""
            {
              "ProjectId": "{{projectId}}",
              "MediaAssetId": null,
              "SourceAudioKind": 1,
              "SourceAudioRelativePath": "media/normalized_audio.wav",
              "SourceGainDb": 0,
              "DubbedSpeechGainDb": 0,
              "DuckingGainDb": -13,
              "DuckingLeadSeconds": 0.05,
              "DuckingTailSeconds": 0.18,
              "CreatedAtUtc": "2026-05-07T00:00:00+00:00",
              "SpeechClips": [],
              "DuckingRegions": [],
              "Warnings": []
            }
            """;
        store.Seed(ProjectArtifactPaths.MixPlanRelativePath, System.Text.Encoding.UTF8.GetBytes(legacyJson));

        MixPlan? reloaded = await mixPlanStore.LoadAsync(TestContext.Current.CancellationToken);

        Assert.NotNull(reloaded);
        Assert.Equal(projectId, reloaded!.ProjectId);
        Assert.Equal("media/normalized_audio.wav", reloaded.OriginalMixAudioRelativePath);
        Assert.Equal(1, reloaded.OutputChannelCount);
        Assert.False(reloaded.RestoreOriginalPan);
    }

    [Fact]
    public async Task PreviewMixWorkflow_persists_plan_and_records_renderer_output()
    {
        TestProjectContext context = CreateContext();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        var project = new TrackdubProject(context.ProjectId, "Preview", now, now);
        var mediaAsset = new MediaAsset(
            context.MediaAssetId,
            context.ProjectId,
            "source.mp4",
            "source.mp4",
            "source-hash",
            100,
            now,
            "mp4",
            8.0d,
            HasAudio: true,
            HasVideo: true,
            now);
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 1.0d, 2.0d, "Hello", context.SpeakerId);
        TranslatedSegment translatedSegment = TranslatedSegment.Create(context.TranslationRevisionId, 0, 1.0d, 2.0d, "Hola");
        ProjectArtifact sourceArtifact = CreateArtifact(context, ArtifactKind.NormalizedAudio, ProjectArtifactPaths.NormalizedAudioRelativePath, 0, durationSeconds: 8d);
        ProjectArtifact takeArtifact = CreateArtifact(context, ArtifactKind.TtsTake, "artifacts/tts/take.wav", 1, durationSeconds: 1d);
        TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, translatedSegment.Id, segment.SegmentIndex)
            .Complete(takeArtifact.Id, durationSamples: 48000, sampleRate: 48000, provider: "fake");
        var artifactStore = new FakeArtifactStore();
        var renderer = new FakeMixRenderer();
        var mediaAssetRepository = new FakeMediaAssetRepository();
        TranscriptProjectState state = CreateState(project, mediaAsset, [sourceArtifact, takeArtifact], [segment], [translatedSegment], [take]);
        var workflow = new PreviewMixWorkflow(
            new MixPlanBuilder(),
            new MixPlanStore(artifactStore),
            renderer,
            artifactStore,
            new FakeFileFingerprintService(new FileFingerprint("preview-hash", 44, now)),
            mediaAssetRepository,
            new FakeProjectStageRunStore());

        PreviewMixStageResult result = await workflow.GeneratePreviewAsync(
            state,
            new PreviewMixStageRequest(context.ProjectId, 1.0d, 3.0d, SourceGainDb: -4d, DubbedSpeechGainDb: 1d, DuckingGainDb: -10d),
            TestContext.Current.CancellationToken);

        Assert.Equal(2.0d, result.DurationSeconds);
        Assert.Equal(StageRunStatus.Completed, result.StageRun.Status);
        Assert.Single(renderer.Calls);
        Assert.Equal(-4d, renderer.LastMixPlan!.SourceGainDb);
        Assert.True(artifactStore.Exists(ProjectArtifactPaths.MixPlanRelativePath));
        ProjectArtifact previewArtifact = Assert.Single(mediaAssetRepository.Artifacts, artifact => artifact.Kind == ArtifactKind.PreviewMix);
        Assert.Equal(result.PreviewAudioRelativePath, previewArtifact.RelativePath);
    }

    [Fact]
    public async Task PreviewMixWorkflow_marks_stage_canceled_when_render_is_canceled()
    {
        TestProjectContext context = CreateContext();
        DateTimeOffset now = DateTimeOffset.UtcNow;
        var project = new TrackdubProject(context.ProjectId, "Preview", now, now);
        var mediaAsset = new MediaAsset(
            context.MediaAssetId,
            context.ProjectId,
            "source.mp4",
            "source.mp4",
            "source-hash",
            100,
            now,
            "mp4",
            8.0d,
            HasAudio: true,
            HasVideo: true,
            now);
        TranscriptSegment segment = TranscriptSegment.Create(context.TranscriptRevisionId, 0, 1.0d, 2.0d, "Hello", context.SpeakerId);
        ProjectArtifact sourceArtifact = CreateArtifact(context, ArtifactKind.NormalizedAudio, ProjectArtifactPaths.NormalizedAudioRelativePath, 0, durationSeconds: 8d);
        var artifactStore = new FakeArtifactStore();
        var stageRunStore = new FakeProjectStageRunStore();
        TranscriptProjectState state = CreateState(project, mediaAsset, [sourceArtifact], [segment], [], []);
        var workflow = new PreviewMixWorkflow(
            new MixPlanBuilder(),
            new MixPlanStore(artifactStore),
            new CancelingMixRenderer(),
            artifactStore,
            new FakeFileFingerprintService(new FileFingerprint("preview-hash", 44, now)),
            new FakeMediaAssetRepository(),
            stageRunStore);

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            workflow.GeneratePreviewAsync(
                state,
                new PreviewMixStageRequest(context.ProjectId, 1.0d, 3.0d),
                TestContext.Current.CancellationToken));

        StageRunRecord stageRun = Assert.Single(stageRunStore.All);
        Assert.Equal(StageRunStatus.Canceled, stageRun.Status);
        Assert.Contains("canceled", stageRun.FailureReason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task FakeMixRenderer_rejects_invalid_ranges_like_real_renderer()
    {
        var renderer = new FakeMixRenderer();
        var plan = new MixPlan(
            Guid.NewGuid(),
            MediaAssetId: null,
            ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath,
            SourceGainDb: 0d,
            DubbedSpeechGainDb: 0d,
            DuckingGainDb: -12d,
            DuckingLeadSeconds: 0.05d,
            DuckingTailSeconds: 0.18d,
            DateTimeOffset.UtcNow,
            SpeechClips: [],
            DuckingRegions: [],
            Warnings: []);

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            renderer.RenderAsync(
                new PreviewRangeRenderRequest(plan, StartSeconds: 3d, EndSeconds: 2d, Path.Join(Path.GetTempPath(), $"{Guid.NewGuid():N}.wav")),
                TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task FakeMixRenderer_can_promote_output_to_stereo_from_source_channel_metadata()
    {
        var renderer = new FakeMixRenderer();
        renderer.SeedSourceChannelCount(ProjectArtifactPaths.NormalizedAudioRelativePath, 6);
        var plan = new MixPlan(
            Guid.NewGuid(),
            MediaAssetId: null,
            ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath,
            SourceGainDb: 0d,
            DubbedSpeechGainDb: 0d,
            DuckingGainDb: -12d,
            DuckingLeadSeconds: 0.05d,
            DuckingTailSeconds: 0.18d,
            DateTimeOffset.UtcNow,
            SpeechClips: [],
            DuckingRegions: [],
            Warnings: [],
            OutputChannelCount: 1);
        string outputPath = Path.Join(Path.GetTempPath(), $"{Guid.NewGuid():N}.wav");

        try
        {
            PreviewRangeRenderResult result = await renderer.RenderAsync(
                new PreviewRangeRenderRequest(plan, StartSeconds: 0d, EndSeconds: 1d, outputPath),
                TestContext.Current.CancellationToken);

            Assert.Equal(2, result.ChannelCount);
        }
        finally
        {
            if (File.Exists(outputPath))
            {
                File.Delete(outputPath);
            }
        }
    }

    [Fact]
    public void FakeMixRenderer_rejects_non_positive_source_channel_metadata()
    {
        var renderer = new FakeMixRenderer();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            renderer.SeedSourceChannelCount(ProjectArtifactPaths.NormalizedAudioRelativePath, 0));
    }

    private static TranscriptProjectState CreateState(
        TrackdubProject project,
        MediaAsset mediaAsset,
        IReadOnlyList<ProjectArtifact> artifacts,
        IReadOnlyList<TranscriptSegment> transcriptSegments,
        IReadOnlyList<TranslatedSegment> translatedSegments,
        IReadOnlyList<TtsTake> ttsTakes)
    {
        var sourceReference = new SourceMediaReference(
            mediaAsset.SourceFilePath,
            mediaAsset.SourceFileName,
            new FileFingerprint(mediaAsset.FingerprintSha256, mediaAsset.SourceSizeBytes, mediaAsset.SourceLastWriteTimeUtc),
            new MediaProbeSnapshot(
                "mp4",
                "MP4",
                mediaAsset.DurationSeconds,
                mediaAsset.SourceSizeBytes,
                [new MediaAudioStream(0, "aac", 2, 48000, mediaAsset.DurationSeconds)],
                [new MediaVideoStream(1, "h264", 1920, 1080, 24.0d, mediaAsset.DurationSeconds)]),
            DateTimeOffset.UtcNow);
        var openResult = new OpenProjectResult(
            project,
            mediaAsset,
            sourceReference,
            SourceMediaStatus.Available,
            SourceStatusMessage: null,
            artifacts,
            TranscriptLanguage: "en");
        TranscriptRevision transcriptRevision = TranscriptRevision.Create(
            project.Id,
            stageRunId: null,
            revisionNumber: 1,
            DateTimeOffset.UtcNow);
        TranslationRevision translationRevision = TranslationRevision.Create(
            project.Id,
            stageRunId: null,
            transcriptRevision.Id,
            "es",
            revisionNumber: 1,
            DateTimeOffset.UtcNow);
        return new TranscriptProjectState(
            openResult,
            transcriptRevision,
            transcriptSegments,
            Speakers: [],
            SpeakerTurns: [],
            translationRevision,
            translatedSegments,
            IsTranslationStale: false,
            TranscriptLanguage: "en",
            StageRuns: [],
            SupportedTargetLanguages: [],
            SelectedTranslationTargetLanguage: "es",
            StaleTranslatedSegmentIndices: new HashSet<int>(),
            WaveformSummary: null,
            AvailableVoices: [],
            VoiceAssignments: [],
            ttsTakes,
            TtsSegmentStates: [],
            VoiceAssignmentWarnings: []);
    }

    private static (ProjectArtifact Vocals, StageRunRecord Run) CreateCompletedStemPair(ref ProjectArtifact ambiance)
    {
        StageRunRecord run = StageRunRecord.Start(ambiance.ProjectId, StageNames.Separation, ambiance.CreatedAtUtc.AddSeconds(-1))
            .Complete(ambiance.CreatedAtUtc);
        ambiance = ambiance with { StageRunId = run.Id };
        return (ambiance with
        {
            Id = Guid.NewGuid(),
            Kind = ArtifactKind.Vocals,
            RelativePath = "artifacts/stems/run/vocals.wav",
            Provenance = "generated-spleeter-vocals;engine_family=spleeter;model=spleeter"
        }, run);
    }

    private static ProjectArtifact CreateArtifact(
        TestProjectContext context,
        ArtifactKind kind,
        string relativePath,
        int createdOffsetSeconds,
        double? durationSeconds = null,
        string? provenance = null,
        int channelCount = 1)
    {
        DateTimeOffset createdAtUtc = DateTimeOffset.UtcNow.AddSeconds(createdOffsetSeconds);
        provenance ??= kind switch
        {
            ArtifactKind.Vocals => "generated-spleeter-vocals;engine_family=spleeter;model=spleeter",
            ArtifactKind.Ambiance => "generated-spleeter-ambiance;engine_family=spleeter;model=spleeter",
            _ => null
        };

        return new ProjectArtifact(
            Guid.NewGuid(),
            context.ProjectId,
            context.MediaAssetId,
            kind,
            relativePath,
            $"{kind.ToString().ToLowerInvariant()}-hash",
            100,
            durationSeconds,
            48000,
            channelCount,
            createdAtUtc,
            Provenance: provenance);
    }

    private static TestProjectContext CreateContext() =>
        new(
            Guid.NewGuid(),
            Guid.NewGuid(),
            Guid.NewGuid(),
            Guid.NewGuid(),
            Guid.NewGuid(),
            Guid.NewGuid());

    private static string ComputeTtsTextHash(int segmentIndex, string text)
    {
        string payload = string.Create(
            System.Globalization.CultureInfo.InvariantCulture,
            $"{segmentIndex}|{text.Trim()}");
        byte[] hash = System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(payload));
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    // ---------------------------------------------------------------------------
    // Tail reflow: takes pulled back from the end of the source audio
    // ---------------------------------------------------------------------------

    [Fact]
    public void Build_pulls_the_last_take_into_slack_left_by_the_previous_one()
    {
        // The POC Chinese clip: segment 1 runs 26.75–40.15s, its take (after the 1.2x partial
        // stretch) is 16.23s, and the previous take ends at 20.86s, well before segment 1 starts.
        MixPlan plan = BuildTwoSegmentPlan(
            (0.14d, 26.75d, 20.72d),
            (26.75d, 40.15d, 16.23d),
            sourceSeconds: 40.171d);

        Assert.Empty(plan.Warnings);
        Assert.Equal(0.14d, plan.SpeechClips[0].StartSeconds, precision: 6);
        Assert.Equal(40.171d - 16.23d, plan.SpeechClips[1].StartSeconds, precision: 6);
        MixDuckRegion duck = plan.DuckingRegions[1];
        Assert.True(duck.StartSeconds < plan.SpeechClips[1].StartSeconds);
    }

    [Fact]
    public void Build_moves_the_previous_take_too_when_the_last_one_would_overlap_it()
    {
        // Last take needs to start at 7.0s; the previous take ends at 7.5s, so it moves back 0.5s.
        MixPlan plan = BuildTwoSegmentPlan(
            (2d, 8d, 5.5d),
            (8d, 10d, 3d),
            sourceSeconds: 10d);

        Assert.Empty(plan.Warnings);
        Assert.Equal(1.5d, plan.SpeechClips[0].StartSeconds, precision: 6);
        Assert.Equal(7d, plan.SpeechClips[1].StartSeconds, precision: 6);
    }

    [Fact]
    public void Build_does_not_overlap_a_previous_take_that_cannot_move_far_enough()
    {
        // The last take would need to start at 4.0s, but the previous take fills 0–5s and starts at 0,
        // so it cannot move. The last take stays at 5.0s rather than overlapping it, and the 1.0s
        // overhang is still reported.
        MixPlan plan = BuildTwoSegmentPlan(
            (0d, 5d, 5d),
            (5d, 10d, 6d),
            sourceSeconds: 10d);

        Assert.Equal(0d, plan.SpeechClips[0].StartSeconds, precision: 6);
        Assert.Equal(5d, plan.SpeechClips[1].StartSeconds, precision: 6);
        MixPlanWarning warning = Assert.Single(plan.Warnings);
        Assert.Equal(MixPlanWarningCode.TakeExceedsSourceAudio, warning.Code);
        Assert.Equal(1, warning.SegmentIndex);
    }

    [Fact]
    public void Build_moves_a_take_at_most_the_reflow_lead_into_silence()
    {
        // A long silence before the last segment: the take moves back only 4.0s, not into the gap.
        MixPlan plan = BuildTwoSegmentPlan(
            (0d, 2d, 2d),
            (20d, 25d, 10d),
            sourceSeconds: 25d);

        Assert.Equal(16d, plan.SpeechClips[1].StartSeconds, precision: 6);
        MixPlanWarning warning = Assert.Single(plan.Warnings);
        Assert.Contains("1 s past", warning.Message);
    }

    [Fact]
    public void Build_leaves_takes_that_fit_where_their_segments_start()
    {
        MixPlan plan = BuildTwoSegmentPlan(
            (1d, 5d, 3d),
            (6d, 9d, 3.5d),
            sourceSeconds: 10d);

        Assert.Empty(plan.Warnings);
        Assert.Equal(1d, plan.SpeechClips[0].StartSeconds, precision: 6);
        Assert.Equal(6d, plan.SpeechClips[1].StartSeconds, precision: 6);
    }

    private static MixPlan BuildTwoSegmentPlan(
        (double Start, double End, double Take) first,
        (double Start, double End, double Take) second,
        double sourceSeconds)
    {
        TestProjectContext context = CreateContext();
        ProjectArtifact source = CreateArtifact(context, ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0, durationSeconds: sourceSeconds);
        var segments = new List<TranscriptSegment>();
        var artifacts = new List<ProjectArtifact> { source };
        var takes = new List<TtsTake>();
        foreach ((int index, (double start, double end, double takeSeconds)) in new[] { (0, first), (1, second) })
        {
            segments.Add(TranscriptSegment.Create(context.TranscriptRevisionId, index, start, end, $"Segment {index}", context.SpeakerId));
            ProjectArtifact artifact = CreateArtifact(context, ArtifactKind.TtsTake,
                $"artifacts/tts/take-{index}.wav", createdOffsetSeconds: 2 + index, durationSeconds: takeSeconds);
            artifacts.Add(artifact);
            takes.Add(TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: index)
                .Complete(artifact.Id, durationSamples: (int)Math.Round(takeSeconds * 24000), sampleRate: 24000, provider: "fake"));
        }

        return new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId, context.MediaAssetId, artifacts, segments, [], takes, StageRuns: null));
    }

    // ---------------------------------------------------------------------------
    // LipSync artifact preference
    // ---------------------------------------------------------------------------

    [Fact]
    public void Build_prefers_lipsync_artifact_over_tts_take_when_file_exists_on_disk()
    {
        // Arrange
        string directory = Path.Join(Path.GetTempPath(), $"mix-lipsync-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            TestProjectContext context = CreateContext();
            var artifactStore = new FakeArtifactStore(directory);

            TranscriptSegment segment = TranscriptSegment.Create(
                context.TranscriptRevisionId, 0, 1.0d, 2.5d, "Hello", context.SpeakerId);
            ProjectArtifact normalized = CreateArtifact(
                context, ArtifactKind.NormalizedAudio,
                ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0);

            var ttsRelPath = $"artifacts/tts/{Guid.NewGuid():N}.wav";
            artifactStore.Seed(ttsRelPath);
            ProjectArtifact ttsArtifact = CreateArtifact(
                context, ArtifactKind.TtsTake, ttsRelPath,
                createdOffsetSeconds: 1, durationSeconds: 2.0d);

            TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: segment.SegmentIndex)
                .Complete(ttsArtifact.Id, durationSamples: 96000, sampleRate: 48000, provider: "fake");

            // Register a LipSync artifact pointing to the same take.
            var lipSyncRelPath = $"artifacts/lip-sync/{Guid.NewGuid():N}/{take.TranslatedSegmentId ?? Guid.NewGuid():N}.wav";
            artifactStore.Seed(lipSyncRelPath);
            ProjectArtifact lipSyncArtifact = CreateArtifact(
                context, ArtifactKind.LipSyncTake, lipSyncRelPath,
                createdOffsetSeconds: 2, durationSeconds: 1.85d,
                provenance: $"lipsync:take:{take.Id:N}");

            // Act — pass artifactStore so file-existence check runs.
            MixPlan plan = new MixPlanBuilder(artifactStore).Build(new MixPlanBuildRequest(
                context.ProjectId,
                context.MediaAssetId,
                [normalized, ttsArtifact, lipSyncArtifact],
                [segment],
                [],
                [take], StageRuns: null));

            // Assert — speech clip should use the LipSync path.
            Assert.Single(plan.SpeechClips);
            MixSpeechClip clip = plan.SpeechClips[0];
            Assert.False(clip.IsSilentGap);
            Assert.Equal(lipSyncRelPath, clip.TakeRelativePath);
            Assert.Equal(lipSyncArtifact.Id, clip.ArtifactId);
            Assert.Equal(1.85d, clip.TakeDurationSeconds);
            Assert.Empty(plan.Warnings);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public void Build_falls_back_to_tts_take_when_lipsync_file_missing_on_disk()
    {
        // Arrange
        string directory = Path.Join(Path.GetTempPath(), $"mix-lipsync-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            TestProjectContext context = CreateContext();
            var artifactStore = new FakeArtifactStore(directory);

            TranscriptSegment segment = TranscriptSegment.Create(
                context.TranscriptRevisionId, 0, 1.0d, 2.5d, "Hello", context.SpeakerId);
            ProjectArtifact normalized = CreateArtifact(
                context, ArtifactKind.NormalizedAudio,
                ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0);

            var ttsRelPath = $"artifacts/tts/{Guid.NewGuid():N}.wav";
            artifactStore.Seed(ttsRelPath);
            ProjectArtifact ttsArtifact = CreateArtifact(
                context, ArtifactKind.TtsTake, ttsRelPath,
                createdOffsetSeconds: 1, durationSeconds: 2.0d);

            TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: segment.SegmentIndex)
                .Complete(ttsArtifact.Id, durationSamples: 96000, sampleRate: 48000, provider: "fake");

            // LipSync artifact registered in DB but NOT seeded on disk.
            var lipSyncRelPath = $"artifacts/lip-sync/{Guid.NewGuid():N}/{Guid.NewGuid():N}.wav";
            ProjectArtifact lipSyncArtifact = CreateArtifact(
                context, ArtifactKind.LipSyncTake, lipSyncRelPath,
                createdOffsetSeconds: 2, durationSeconds: 1.85d,
                provenance: $"lipsync:take:{take.Id:N}");

            // Act
            MixPlan plan = new MixPlanBuilder(artifactStore).Build(new MixPlanBuildRequest(
                context.ProjectId,
                context.MediaAssetId,
                [normalized, ttsArtifact, lipSyncArtifact],
                [segment],
                [],
                [take], StageRuns: null));

            // Assert — should fall back to TTS take and emit a warning.
            Assert.Single(plan.SpeechClips);
            MixSpeechClip clip = plan.SpeechClips[0];
            Assert.False(clip.IsSilentGap);
            Assert.Equal(ttsRelPath, clip.TakeRelativePath);
            Assert.Equal(ttsArtifact.Id, clip.ArtifactId);

            MixPlanWarning warning = Assert.Single(plan.Warnings);
            Assert.Equal(MixPlanWarningCode.LipSyncArtifactMissing, warning.Code);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public void Build_falls_back_to_tts_take_when_no_artifact_store_provided()
    {
        // Arrange — no artifactStore, so file-existence check is skipped.
        TestProjectContext context = CreateContext();

        TranscriptSegment segment = TranscriptSegment.Create(
            context.TranscriptRevisionId, 0, 1.0d, 2.5d, "Hello", context.SpeakerId);
        ProjectArtifact normalized = CreateArtifact(
            context, ArtifactKind.NormalizedAudio,
            ProjectArtifactPaths.NormalizedAudioRelativePath, createdOffsetSeconds: 0);

        var ttsRelPath = $"artifacts/tts/{Guid.NewGuid():N}.wav";
        ProjectArtifact ttsArtifact = CreateArtifact(
            context, ArtifactKind.TtsTake, ttsRelPath,
            createdOffsetSeconds: 1, durationSeconds: 2.0d);

        TtsTake take = TtsTake.Create(context.ProjectId, context.VoiceAssignmentId, segmentIndex: segment.SegmentIndex)
            .Complete(ttsArtifact.Id, durationSamples: 96000, sampleRate: 48000, provider: "fake");

        var lipSyncRelPath = $"artifacts/lip-sync/{Guid.NewGuid():N}/{Guid.NewGuid():N}.wav";
        ProjectArtifact lipSyncArtifact = CreateArtifact(
            context, ArtifactKind.LipSyncTake, lipSyncRelPath,
            createdOffsetSeconds: 2, durationSeconds: 1.85d,
            provenance: $"lipsync:take:{take.Id:N}");

        // Act — no artifactStore passed to MixPlanBuilder.
        MixPlan plan = new MixPlanBuilder().Build(new MixPlanBuildRequest(
            context.ProjectId,
            context.MediaAssetId,
            [normalized, ttsArtifact, lipSyncArtifact],
            [segment],
            [],
            [take], StageRuns: null));

        // Assert — falls through to TTS take because store is null (no file check, no warning).
        Assert.Single(plan.SpeechClips);
        MixSpeechClip clip = plan.SpeechClips[0];
        Assert.False(clip.IsSilentGap);
        Assert.Equal(ttsRelPath, clip.TakeRelativePath);
        Assert.Equal(ttsArtifact.Id, clip.ArtifactId);
        Assert.Empty(plan.Warnings);
    }

    private sealed record TestProjectContext(
        Guid ProjectId,
        Guid MediaAssetId,
        Guid TranscriptRevisionId,
        Guid TranslationRevisionId,
        Guid SpeakerId,
        Guid VoiceAssignmentId);

    private sealed class CancelingMixRenderer : IPreviewRangeRenderer
    {
        public Task<PreviewRangeRenderResult> RenderAsync(
            PreviewRangeRenderRequest request,
            CancellationToken cancellationToken) =>
            throw new OperationCanceledException(cancellationToken);
    }
}
