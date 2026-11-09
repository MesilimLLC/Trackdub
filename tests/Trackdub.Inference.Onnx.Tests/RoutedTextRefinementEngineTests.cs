using Trackdub.Contracts.Pipeline;
using Trackdub.Inference.Onnx.QwenTextRefinement;

namespace Trackdub.Inference.Onnx.Tests;

public sealed class RoutedTextRefinementEngineTests
{
    private const string GeminiFamily = "gemini-refinement-cloud";

    private static readonly TextRefinementInputSegment[] Segments = [new(0, 0, 1, "hello")];

    [Fact]
    public async Task Null_alias_uses_the_local_qwen_engine()
    {
        var qwen = new RecordingEngine(QwenTextRefinementEngine.EngineFamilyName);
        var gemini = new RecordingEngine(GeminiFamily);
        var router = new RoutedTextRefinementEngine([gemini, qwen]);

        IReadOnlyList<RefinedTextSegment> result =
            await router.RefineAsync(new TextRefinementRequest(Segments), CancellationToken.None);

        Assert.Equal(QwenTextRefinementEngine.EngineFamilyName, Assert.Single(result).RefinedText);
        Assert.Equal(1, qwen.Calls);
        Assert.Equal(0, gemini.Calls);
        Assert.Equal(QwenTextRefinementEngine.EngineFamilyName, router.EngineFamily);
    }

    [Theory]
    [InlineData("gemini")]
    [InlineData("gemini-refinement-cloud")]
    [InlineData("Gemini-2.5-Flash")]
    public async Task Gemini_alias_uses_the_cloud_engine_and_reports_its_summary(string alias)
    {
        var qwen = new RecordingEngine(QwenTextRefinementEngine.EngineFamilyName);
        var gemini = new RecordingEngine(GeminiFamily);
        var router = new RoutedTextRefinementEngine([qwen, gemini]);

        IReadOnlyList<RefinedTextSegment> result = await router.RefineAsync(
            new TextRefinementRequest(Segments, PreferredModelAlias: alias),
            CancellationToken.None);

        Assert.Equal(GeminiFamily, Assert.Single(result).RefinedText);
        Assert.Equal(alias, gemini.LastAlias);
        Assert.Equal(0, qwen.Calls);
        Assert.Equal(1, gemini.Calls);
        Assert.Equal(gemini.LastExecutionSummary, router.LastExecutionSummary);
    }

    [Fact]
    public async Task Unknown_alias_falls_back_to_qwen()
    {
        var qwen = new RecordingEngine(QwenTextRefinementEngine.EngineFamilyName);
        var gemini = new RecordingEngine(GeminiFamily);
        var router = new RoutedTextRefinementEngine([qwen, gemini]);

        IReadOnlyList<RefinedTextSegment> result = await router.RefineAsync(
            new TextRefinementRequest(Segments, PreferredModelAlias: "something-else"),
            CancellationToken.None);

        Assert.Equal(QwenTextRefinementEngine.EngineFamilyName, Assert.Single(result).RefinedText);
        Assert.Equal(1, qwen.Calls);
        Assert.Equal(0, gemini.Calls);
    }

    private sealed class RecordingEngine(string family) : ITextRefinementEngine, IStageRuntimeExecutionReporter
    {
        public int Calls { get; private set; }

        public string? LastAlias { get; private set; }

        public string EngineFamily => family;

        public StageRuntimeExecutionSummary? LastExecutionSummary { get; } =
            new("requested-" + family, "selected-" + family);

        public Task<IReadOnlyList<RefinedTextSegment>> RefineAsync(
            TextRefinementRequest request,
            CancellationToken cancellationToken)
        {
            Calls++;
            LastAlias = request.PreferredModelAlias;
            return Task.FromResult<IReadOnlyList<RefinedTextSegment>>(
            [
                new RefinedTextSegment(
                    0, 0, 1, "hello", family, family, true, TextRefinementGuardStatus.Accepted, [])
            ]);
        }
    }
}
