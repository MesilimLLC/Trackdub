using Microsoft.ML.OnnxRuntime.Tensors;
using Trackdub.Inference.Onnx.Translation;

namespace Trackdub.Inference.Onnx.Tests;

public sealed class Seq2SeqGreedyDecoderNgramTests
{
    [Fact]
    public void Repeated_ngram_bans_the_token_that_would_complete_it_again()
    {
        // "x ah ah ah" with n = 3: the prefix "ah ah" already occurred followed by "ah".
        long[] generated = [7, 42, 42, 42];

        HashSet<int>? banned = Seq2SeqGreedyDecoder.BannedNgramContinuations(generated, 3);

        Assert.NotNull(banned);
        Assert.Contains(42, banned);
    }

    [Fact]
    public void Prefix_seen_with_different_continuations_bans_each_of_them()
    {
        long[] generated = [1, 2, 3, 1, 2, 4, 1, 2];

        HashSet<int>? banned = Seq2SeqGreedyDecoder.BannedNgramContinuations(generated, 3);

        Assert.Equal(new HashSet<int> { 3, 4 }, banned);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Guard_off_bans_nothing(int ngramSize)
    {
        Assert.Null(Seq2SeqGreedyDecoder.BannedNgramContinuations([42, 42, 42, 42], ngramSize));
    }

    [Fact]
    public void Short_or_unrepeated_output_bans_nothing()
    {
        Assert.Null(Seq2SeqGreedyDecoder.BannedNgramContinuations([1, 2, 3], 4));
        Assert.Null(Seq2SeqGreedyDecoder.BannedNgramContinuations([1, 2, 3, 4, 5], 4));
    }

    [Fact]
    public void ArgMax_skips_a_banned_token_and_takes_the_next_best()
    {
        // Vocabulary of 4; token 2 has the highest logit but is banned, token 0 is padding.
        var logits = new DenseTensor<float>(new float[] { 9f, 1f, 5f, 3f }, [1, 1, 4]);

        int unbanned = Seq2SeqGreedyDecoder.ArgMaxAtPosition(logits, 0, padTokenId: 0);
        int withBan = Seq2SeqGreedyDecoder.ArgMaxAtPosition(logits, 0, padTokenId: 0, bannedTokens: [2]);

        Assert.Equal(2, unbanned);
        Assert.Equal(3, withBan);
    }

    [Fact]
    public void Greedy_loop_cannot_repeat_a_four_token_ngram()
    {
        // Simulates a decoder whose favourite token never changes: with the guard it must switch
        // tokens instead of emitting the same one until maxSteps.
        const int vocabulary = 8;
        var generated = new List<long>();
        for (int step = 0; step < 40; step++)
        {
            float[] row = new float[vocabulary];
            for (int token = 0; token < vocabulary; token++)
            {
                row[token] = token == 5 ? 10f : token;
            }

            var logits = new DenseTensor<float>(row, [1, 1, vocabulary]);
            int next = Seq2SeqGreedyDecoder.ArgMaxAtPosition(
                logits,
                0,
                padTokenId: 0,
                Seq2SeqGreedyDecoder.BannedNgramContinuations(generated, MadladNgramSize));
            if (next < 0)
            {
                break; // The decoder stops when every continuation is banned.
            }

            generated.Add(next);
        }

        Assert.True(generated.Count(token => token == 5) < generated.Count);

        for (int start = 0; start + MadladNgramSize <= generated.Count; start++)
        {
            long[] ngram = generated.Skip(start).Take(MadladNgramSize).ToArray();
            int occurrences = Enumerable.Range(0, generated.Count - MadladNgramSize + 1)
                .Count(other => generated.Skip(other).Take(MadladNgramSize).SequenceEqual(ngram));
            Assert.Equal(1, occurrences);
        }
    }

    private const int MadladNgramSize = Madlad.MadladTranslationEngine.NoRepeatNgramSize;
}
