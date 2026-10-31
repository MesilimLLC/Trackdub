using Trackdub.Domain;
using Trackdub.Domain.Projects;
using Trackdub.Domain.Transcript;
using Trackdub.Domain.Translation;
using Trackdub.Infrastructure.Persistence.Sqlite;

namespace Trackdub.Infrastructure.Tests;

public sealed class SqliteTranslationRepositoryTests
{
    [Fact]
    public async Task Repository_round_trips_current_translation_revision_segments_and_revision_numbers_per_language()
    {
        string projectRoot = Path.Join(Path.GetTempPath(), "Trackdub.Infrastructure.Tests", Guid.NewGuid().ToString("N"), "Translation.trackdub");
        try
        {
            var database = new SqliteProjectDatabase(projectRoot);
            var projectRepository = new SqliteProjectRepository(database);
            var transcriptRepository = new SqliteTranscriptRepository(database);
            var translationRepository = new SqliteTranslationRepository(database);
            var stageRunStore = new SqliteProjectStageRunStore(database);
            DateTimeOffset now = DateTimeOffset.UtcNow;
            var project = new TrackdubProject(Guid.NewGuid(), "Translation", now, now);

            await projectRepository.InitializeAsync(project, TestContext.Current.CancellationToken);

            TranscriptRevision transcriptRevision = TranscriptRevision.Create(project.Id, stageRunId: null, revisionNumber: 1, now.AddSeconds(1));
            TranscriptSegment[] transcriptSegments =
            [
                TranscriptSegment.Create(transcriptRevision.Id, 0, 0.0, 1.5, "Hello"),
                TranscriptSegment.Create(transcriptRevision.Id, 1, 1.5, 3.0, "World")
            ];
            await transcriptRepository.SaveRevisionAsync(transcriptRevision, transcriptSegments, TestContext.Current.CancellationToken);

            StageRunRecord translationStageRun = StageRunRecord.Start(project.Id, "translation", now.AddSeconds(2))
                .WithRuntimeInfo("auto", "cpu", "Helsinki-NLP/opus-mt-en-es", "opus-en-es", "merged-decoder", "bootstrap skipped")
                .Complete(now.AddSeconds(3));
            await stageRunStore.CreateAsync(translationStageRun, TestContext.Current.CancellationToken);
            await stageRunStore.UpdateAsync(translationStageRun, TestContext.Current.CancellationToken);

            TranslationRevision translationRevision = TranslationRevision.Create(
                project.Id,
                translationStageRun.Id,
                transcriptRevision.Id,
                "es",
                revisionNumber: 1,
                now.AddSeconds(4),
                translationProvider: "opus-mt",
                modelId: "Helsinki-NLP/opus-mt-en-es",
                executionProvider: "cpu",
                modelProvenance: new TranslationModelProvenance("onnx-community/opus-mt-en-es", new string('a', 40), new string('b', 64),
                    new Dictionary<string, string> { ["onnx/encoder_model.onnx"] = new string('b', 64) }, ">>spa<<", "opus-policy-v1"));
            TranslatedSegment[] translatedSegments =
            [
                TranslatedSegment.Create(
                    translationRevision.Id,
                    0,
                    0.0,
                    1.5,
                    "Hola",
                    "hash-0",
                    [
                        TranslatedWord.Create(0, 0.0, 0.7, "Ho"),
                        TranslatedWord.Create(1, 0.7, 1.5, "la")
                    ]),
                TranslatedSegment.Create(translationRevision.Id, 1, 1.5, 3.0, "Mundo", "hash-1") with { Origin = TranslationSegmentOrigin.Manual }
            ];

            await translationRepository.SaveRevisionAsync(translationRevision, translatedSegments, TestContext.Current.CancellationToken);

            TranslationRevision? current = await translationRepository.GetCurrentRevisionAsync(project.Id, "es", TestContext.Current.CancellationToken);
            IReadOnlyList<TranslatedSegment> reloadedSegments = await translationRepository.GetSegmentsAsync(translationRevision.Id, TestContext.Current.CancellationToken);
            int nextSpanishRevision = await translationRepository.GetNextRevisionNumberAsync(project.Id, "es", TestContext.Current.CancellationToken);
            int nextGermanRevision = await translationRepository.GetNextRevisionNumberAsync(project.Id, "de", TestContext.Current.CancellationToken);

            Assert.NotNull(current);
            Assert.Equal(transcriptRevision.Id, current!.SourceTranscriptRevisionId);
            Assert.Equal("opus-mt", current.TranslationProvider);
            Assert.Equal("Helsinki-NLP/opus-mt-en-es", current.ModelId);
            Assert.Equal("cpu", current.ExecutionProvider);
            Assert.NotNull(current.ModelProvenance);
            Assert.Equal(translationRevision.ModelProvenance!.Repository, current.ModelProvenance.Repository);
            Assert.Equal(new string('a', 40), current.ModelProvenance.Revision);
            Assert.Equal(new string('b', 64), current.ModelProvenance.Sha256);
            Assert.Equal(new string('b', 64), current.ModelProvenance.FileHashes["onnx/encoder_model.onnx"]);
            Assert.Equal(">>spa<<", current.ModelProvenance.TargetPrefix);
            Assert.Equal("opus-policy-v1", current.ModelProvenance.CatalogPolicyVersion);
            Assert.Equal((2, "Hola", TranslationSegmentOrigin.Unknown), (reloadedSegments.Count, reloadedSegments[0].Text, reloadedSegments[0].Origin));
            Assert.Equal("hash-0", reloadedSegments[0].SourceSegmentHash);
            Assert.Equal(2, reloadedSegments[0].Words.Count);
            Assert.Equal("Ho", reloadedSegments[0].Words[0].Text);
            Assert.Equal("la", reloadedSegments[0].Words[1].Text);
            Assert.Equal((0, TranslationSegmentOrigin.Manual), (reloadedSegments[1].Words.Count, reloadedSegments[1].Origin));
            Assert.Equal(2, nextSpanishRevision);
            Assert.Equal(1, nextGermanRevision);
        }
        finally
        {
            if (Directory.Exists(projectRoot))
            {
                Directory.Delete(projectRoot, recursive: true);
            }
        }
    }
}
