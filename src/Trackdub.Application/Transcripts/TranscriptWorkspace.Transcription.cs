using Trackdub.Contracts;
using Trackdub.Contracts.Pipeline;
using Trackdub.Domain.StageRuns;

namespace Trackdub.Application.Transcripts;

public sealed partial class TranscriptWorkspace
{
    public Task<TranscriptProjectState> RunInitialTranscriptionAsync(
        TranscriptRunOptions options,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(options);
        return RunPipelineAsync(
            nameof(RunInitialTranscriptionAsync),
            async ct =>
            {
                await EnsureImportModelsForTranscriptionAsync(
                        options.ModelPreferences,
                        ct,
                        options.SourceLanguage)
                    .ConfigureAwait(false);
                return await Project.RunInitialTranscriptionAsync(options, ct)
                    .ConfigureAwait(false);
            },
            cancellationToken);
    }

    public Task<TranscriptProjectState> RunTranscriptStageAsync(
        string stageName,
        TranscriptRunOptions options,
        CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(stageName);
        ArgumentNullException.ThrowIfNull(options);
        return RunPipelineAsync(
            nameof(RunTranscriptStageAsync),
            async ct =>
            {
                if (string.Equals(stageName, StageNames.Diarization, StringComparison.OrdinalIgnoreCase))
                {
                    await EnsureDiarizationModelsForTranscriptionAsync(options.ModelPreferences, ct).ConfigureAwait(false);
                }
                else if (string.Equals(stageName, StageNames.Vad, StringComparison.OrdinalIgnoreCase))
                {
                    // VAD uses rule-based detection — no ML model provisioning required.
                    // Skipping import-model provisioning here avoids failing a VAD-only
                    // run when the ASR model is missing or not yet downloaded.
                }
                else if (string.Equals(stageName, StageNames.TextRefinementAsr, StringComparison.OrdinalIgnoreCase))
                {
                    // Text-refinement model provisioning is handled by the shell before isolated runs.
                }
                else
                {
                    await EnsureImportModelsForTranscriptionAsync(
                            options.ModelPreferences,
                            ct,
                            options.SourceLanguage)
                        .ConfigureAwait(false);
                }
                return await Project.RunTranscriptStageAsync(stageName, options, ct).ConfigureAwait(false);
            },
            cancellationToken);
    }
}
