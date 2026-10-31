using Trackdub.Application.Editing;
using Trackdub.Contracts;
using Trackdub.Contracts.Editing;
using Trackdub.Domain.Editing;
using Trackdub.Media.Process;
using Trackdub.Media.Probe;

namespace Trackdub.Media.Muxing;

/// <summary>Renders ordered source placements while keeping their linked source-time ranges intact.</summary>
public sealed class FfmpegSequenceRenderer : ISequenceExportRenderer
{
    private static readonly TimeSpan RenderTimeout = TimeSpan.FromHours(6);
    private readonly IProcessRunner processRunner;
    private readonly FfmpegToolResolver toolResolver;
    private readonly IFfmpegVideoEncoderCapabilities capabilities;
    private readonly IMediaGpuHintProvider? gpuHintProvider;
    private readonly FfmpegSequenceMediaRangeValidator mediaRangeValidator;

    public FfmpegSequenceRenderer(IMediaProbe mediaProbe, string? ffmpegPath = null)
        : this(mediaProbe, gpuHintProvider: null, ffmpegPath)
    {
    }

    public FfmpegSequenceRenderer(
        IMediaProbe mediaProbe,
        IMediaGpuHintProvider? gpuHintProvider,
        string? ffmpegPath = null)
        : this(
            new ProcessRunner(),
            mediaProbe,
            new FfmpegVideoEncoderCapabilityService(ffmpegPath),
            gpuHintProvider,
            ffmpegPath)
    {
    }

    internal FfmpegSequenceRenderer(
        IProcessRunner processRunner,
        IMediaProbe mediaProbe,
        IFfmpegVideoEncoderCapabilities capabilities,
        IMediaGpuHintProvider? gpuHintProvider,
        string? ffmpegPath = null)
    {
        this.processRunner = processRunner ?? throw new ArgumentNullException(nameof(processRunner));
        ArgumentNullException.ThrowIfNull(mediaProbe);
        this.capabilities = capabilities ?? throw new ArgumentNullException(nameof(capabilities));
        this.gpuHintProvider = gpuHintProvider;
        mediaRangeValidator = new FfmpegSequenceMediaRangeValidator(mediaProbe);
        toolResolver = new FfmpegToolResolver(ffmpegPath);
    }

    public async Task<SequenceExportRenderResult> RenderAsync(
        SequenceExportPlan plan,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(plan);
        FfmpegSequencePlanValidator.Validate(plan);
        await mediaRangeValidator.ValidateAsync(plan, cancellationToken).ConfigureAwait(false);

        string outputPath = Path.GetFullPath(plan.OutputPath);
        string outputDirectory = Path.GetDirectoryName(outputPath)!;
        Directory.CreateDirectory(outputDirectory);
        string temporaryOutputPath = CreateTemporaryOutputPath(outputPath);
        var warnings = new List<string>();

        try
        {
            await RenderToTemporaryFileAsync(plan, temporaryOutputPath, warnings, cancellationToken)
                .ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();
            File.Move(temporaryOutputPath, outputPath, overwrite: true);
            return new SequenceExportRenderResult(outputPath, warnings);
        }
        finally
        {
            if (File.Exists(temporaryOutputPath))
            {
                File.Delete(temporaryOutputPath);
            }
        }
    }

    private async Task RenderToTemporaryFileAsync(
        SequenceExportPlan plan,
        string temporaryOutputPath,
        ICollection<string> warnings,
        CancellationToken cancellationToken)
    {
        string ffmpegPath = toolResolver.ResolveFfmpegPath();
        SequenceExportPlan renderPlan = PrepareWatermark(plan, warnings);
        VideoEncodeProfile? encodeProfile = await ResolveEncoderAsync(renderPlan, cancellationToken).ConfigureAwait(false);
        AddEncoderWarning(encodeProfile, warnings);

        ProcessResult result = await RunRenderAsync(
            new RenderInvocation(ffmpegPath, renderPlan, encodeProfile, temporaryOutputPath),
            cancellationToken).ConfigureAwait(false);
        if (ShouldRetryWithSoftware(result, encodeProfile))
        {
            warnings.Add($"Hardware video encoder '{encodeProfile!.EncoderName}' failed; retrying with software libx264.");
            VideoEncodeProfile software = FfmpegVideoEncoderSelector.Resolve(
                VideoEncoderPreference.Software,
                capabilities.GetSnapshot());
            result = await RunRenderAsync(
                new RenderInvocation(ffmpegPath, renderPlan, software, temporaryOutputPath),
                cancellationToken).ConfigureAwait(false);
        }

        EnsureSuccessfulRender(result);
        EnsureTemporaryOutputExists(temporaryOutputPath);
    }

    private async Task<VideoEncodeProfile> ResolveEncoderAsync(
        SequenceExportPlan plan,
        CancellationToken cancellationToken)
    {
        if (plan.VideoEncoder == VideoEncoderPreference.Software)
        {
            return FfmpegVideoEncoderSelector.Resolve(
                VideoEncoderPreference.Software,
                FfmpegVideoEncoderSnapshot.Empty);
        }

        FfmpegVideoEncoderSnapshot snapshot = await capabilities
            .RefreshAsync(cancellationToken)
            .ConfigureAwait(false);
        MediaGpuHint gpuHint = new(HasGpu: false);
        if (gpuHintProvider is not null)
        {
            try
            {
                gpuHint = await gpuHintProvider.GetCurrentAsync(cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                cancellationToken.ThrowIfCancellationRequested();
            }
        }
        return FfmpegVideoEncoderSelector.Resolve(plan.VideoEncoder, snapshot, gpuHint);
    }

    private Task<ProcessResult> RunRenderAsync(RenderInvocation invocation, CancellationToken cancellationToken) =>
        processRunner.RunAsync(
            invocation.FfmpegPath,
            FfmpegSequenceRenderCommandBuilder.BuildArguments(
                invocation.Plan,
                invocation.EncodeProfile,
                invocation.OutputPath),
            cancellationToken,
            new ProcessRunOptions(Timeout: RenderTimeout));

    private static SequenceExportPlan PrepareWatermark(SequenceExportPlan plan, ICollection<string> warnings)
    {
        if (!plan.RequiresWatermark || !string.IsNullOrWhiteSpace(plan.WatermarkFontPath))
        {
            return plan;
        }

        string? systemFont = WatermarkFontResolver.ResolveFontPath();
        if (systemFont is null)
        {
            warnings.Add("No system font was found for the export watermark; ffmpeg may fail to render the watermark.");
            return plan;
        }

        return plan with { WatermarkFontPath = systemFont };
    }

    private static void AddEncoderWarning(VideoEncodeProfile? profile, ICollection<string> warnings)
    {
        if (!string.IsNullOrWhiteSpace(profile?.Message))
        {
            warnings.Add(profile.Message);
        }
    }

    private static bool ShouldRetryWithSoftware(ProcessResult result, VideoEncodeProfile? profile) =>
        result.ExitCode != 0 && profile is not null &&
        !string.Equals(profile.EncoderName, "libx264", StringComparison.OrdinalIgnoreCase);

    private static void EnsureSuccessfulRender(ProcessResult result)
    {
        if (result.ExitCode != 0)
        {
            throw new InvalidOperationException(FfmpegErrorFormatter.BuildFailureMessage(
                "ffmpeg sequence export",
                result.ExitCode,
                result.StandardError));
        }
    }

    private static void EnsureTemporaryOutputExists(string path)
    {
        if (!File.Exists(path) || new FileInfo(path).Length == 0)
        {
            throw new InvalidOperationException("ffmpeg completed without producing a usable sequence video.");
        }
    }

    private static string CreateTemporaryOutputPath(string outputPath)
    {
        string directory = Path.GetDirectoryName(outputPath)!;
        string fileName = Path.GetFileNameWithoutExtension(outputPath);
        string extension = Path.GetExtension(outputPath);
        return Path.Join(directory, $"{fileName}.{Guid.NewGuid():N}.partial{extension}");
    }

    private sealed record RenderInvocation(
        string FfmpegPath,
        SequenceExportPlan Plan,
        VideoEncodeProfile? EncodeProfile,
        string OutputPath);
}
