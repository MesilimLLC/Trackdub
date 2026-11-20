namespace Trackdub.Inference.Runtime.ModelManifest;

public sealed record ModelManifestCatalog(
    IReadOnlyList<ModelManifest> Models);

public sealed record ModelManifest(
    string ModelId,
    ModelTask Task,
    string EngineFamily,
    IReadOnlyList<string> Capabilities,
    ModelLanguageCoverage LanguageCoverage,
    string Tier,
    ModelLane Lane,
    ModelLicenseKind License,
    bool CommercialAllowed,
    bool RedistributionAllowed,
    bool RequiresAttribution,
    bool RequiresUserConsent,
    bool VoiceCloning,
    bool CommercialUseVerified,
    string SourceUrl,
    string Revision,
    string Sha256,
    IReadOnlyList<string> DownloadFiles,
    IReadOnlyDictionary<string, string> DownloadFileSources,
    IReadOnlyList<ModelVariantManifest> Variants,
    IReadOnlyList<string> Aliases,
    string? RootPath,
    string? BenchmarkEntry,
    HashVerificationPolicy HashVerificationPolicy,
    string? DisplayName = null,
    bool OliveOptimizable = false,
    ModelOptimizationManifest? Optimization = null,
    string? ProviderId = null,
    string? ExpectedRuntime = null,
    IReadOnlyDictionary<string, string>? DownloadFileHashes = null,
    int EstimatedVramMb = 0,
    int MinVramMb = 0,
    bool SupportsPartialOffload = false,
    bool? Deprecated = null,
    string? DeprecatedReason = null)
{
    public bool CommercialSafeMode => ModelRiskAcceptanceReader.IsCommercialSafe(CommercialUseVerified, RiskAcceptance);

    public ModelRiskAcceptance? RiskAcceptance { get; init; }

    public IReadOnlyDictionary<string, string> DownloadFileHashes { get; init; } =
        DownloadFileHashes ?? new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
}

/// <summary>
/// Owner-recorded exception for a model whose licence grants commercial use but whose
/// training-data provenance cannot be verified. It never substitutes for <c>commercial_use_verified</c>;
/// see docs/legal/MODEL_LICENSE_POLICY.md.
/// </summary>
public sealed record ModelRiskAcceptance(
    string DecisionRecord,
    IReadOnlyList<string> AcceptedRisks,
    string Owner,
    string Date);

public sealed record ModelOptimizationManifest(
    ModelOliveOptimizationProfile? Olive);

public sealed record ModelOliveOptimizationProfile(
    string Mode,
    IReadOnlyList<string> Components,
    IReadOnlyList<OliveOptimizationProvider> SupportedProviders,
    IReadOnlyList<string>? SupportedPrecisions = null,
    IReadOnlyList<OliveOpsetPolicy>? OpsetPolicies = null,
    bool RequireOpsetMetadata = false,
    IReadOnlyList<OliveRecipeBinding>? RecipeBindings = null,
    OliveRecipeFallbackPolicy FallbackPolicy = OliveRecipeFallbackPolicy.None)
{
    public IReadOnlyList<string> SupportedPrecisions { get; init; } = SupportedPrecisions ?? [];

    public IReadOnlyList<OliveOpsetPolicy> OpsetPolicies { get; init; } = OpsetPolicies ?? [];

    public IReadOnlyList<OliveRecipeBinding> RecipeBindings { get; init; } = RecipeBindings ?? [];
}

public sealed record OliveRecipeBinding(
    string? Provider,
    string? Precision,
    string ConfigRelativePath,
    IReadOnlyList<OliveOptimizationOperation>? Operations = null,
    OliveRecipeExpectedOutput ExpectedOutput = OliveRecipeExpectedOutput.OnnxComponents,
    OliveRecipeFallbackPolicy? FallbackPolicy = null,
    string? QuantizationMethod = null,
    bool RequiresCalibrationData = false,
    string? ScriptRelativePath = null,
    string? ScriptSha256 = null,
    string? Evaluator = null,
    int? SplitCount = null,
    string? CostModelRelativePath = null,
    string? AdapterRelativePath = null,
    string? AdapterMode = null,
    string? OutputManifestRelativePath = null,
    // Disambiguates bindings when a model has multiple ONNX components (e.g. Nemotron's
    // encoder + decoder_joint) that would otherwise both match the same provider/precision
    // and collide under a first-match resolver. Null matches any component.
    string? Component = null)
{
    public IReadOnlyList<OliveOptimizationOperation> Operations { get; init; } = Operations ?? [];
}

public sealed record OliveOpsetPolicy(
    OliveOptimizationProvider? Provider,
    string? Precision,
    int MinimumOpset);

public enum OliveOptimizationProvider
{
    Cpu,
    Dml,
    Cuda,
    TensorRt,
    TensorRtRtx,
    Migraphx,
    Rocm,
    VitisAi,
    Qnn,
    OpenVino
}

public enum OliveOptimizationOperation
{
    OnnxExport,
    QnnConversion,
    OpenVinoConversion,
    Compression,
    ProviderOptimization,
    GenAiPackaging,
    ModelSplitting,
    Evaluation,
    AdapterHandling,
    Registration
}

public enum OliveRecipeExpectedOutput
{
    OnnxComponents,
    OrtGenAi,
    QnnModelLibrary,
    OpenVinoModel,
    SplitOnnxComponents,
    AdapterPackage
}

public enum OliveRecipeFallbackPolicy
{
    None,
    AutoOptAllowed,
    BaseVariantAllowed,
    CpuRuntimeAllowed
}

public sealed record ModelLanguageCoverage(
    IReadOnlyList<string> SourceLanguages,
    IReadOnlyList<string> TargetLanguages,
    IReadOnlyList<ModelLanguagePair> LanguagePairs)
{
    public static ModelLanguageCoverage Empty { get; } = new([], [], []);
}

public sealed record ModelLanguagePair(
    string SourceLanguage,
    string TargetLanguage);

public sealed record ModelVariantManifest(
    string Alias,
    string EntryPath,
    string Sha256,
    IReadOnlyList<string> DownloadFiles,
    string? DisplayName = null,
    string? Description = null,
    bool IsDefault = false,
    IReadOnlyList<string>? SupportedProviders = null,
    int? Opset = null,
    int? EstimatedVramMb = null,
    int? MinVramMb = null)
{
    public string Key => Alias;
}

public sealed record HashVerificationPolicy(
    HashVerificationMode Mode,
    string Algorithm);

public enum HashVerificationMode
{
    None,
    VerifyIfShaPresent,
    Required
}

public enum ModelTask
{
    Asr,
    Translation,
    Tts,
    Diarization,
    Vad,
    Separation,
    SpeechEnhancement,
    ForcedAlignment,
    TextRefinement,
    OverlapRescue,
    LipSynthesis,
    FaceDetection,
    FaceLandmarks
}

public enum ModelLane
{
    Commercial,
    NonCommercial,
    Experimental
}

public enum ModelLicenseKind
{
    Mit,
    Apache20,
    CcBy40,
    CcByNc40,
    NvidiaOpenModelLicense,
    OpenMdw11,
    OpenRailPlusPlus,
    Custom,
    Unknown,
    NonCommercial
}

internal static class ModelManifestText
{
    public static string ToManifestValue(this ModelTask task) =>
        task switch
        {
            ModelTask.Asr => "asr",
            ModelTask.Translation => "translation",
            ModelTask.Tts => "tts",
            ModelTask.Diarization => "diarization",
            ModelTask.Vad => "vad",
            ModelTask.Separation => "separation",
            ModelTask.SpeechEnhancement => "speech-enhancement",
            ModelTask.ForcedAlignment => "forced-alignment",
            ModelTask.TextRefinement => "text-refinement",
            ModelTask.OverlapRescue => "overlap-rescue",
            ModelTask.LipSynthesis => "lip-synthesis",
            ModelTask.FaceDetection => "face-detection",
            ModelTask.FaceLandmarks => "face-landmarks",
            _ => throw new ArgumentOutOfRangeException(nameof(task), task, "Unknown model task.")
        };

    public static string ToManifestValue(this ModelLicenseKind license) =>
        license switch
        {
            ModelLicenseKind.Mit => "MIT",
            ModelLicenseKind.Apache20 => "Apache-2.0",
            ModelLicenseKind.CcBy40 => "CC-BY-4.0",
            ModelLicenseKind.CcByNc40 => "CC-BY-NC-4.0",
            ModelLicenseKind.NvidiaOpenModelLicense => "NVIDIA-Open-Model-License",
            ModelLicenseKind.OpenMdw11 => "OpenMDW-1.1",
            ModelLicenseKind.OpenRailPlusPlus => "openrail++",
            ModelLicenseKind.Custom => "custom",
            ModelLicenseKind.Unknown => "unknown",
            ModelLicenseKind.NonCommercial => "non-commercial",
            _ => throw new ArgumentOutOfRangeException(nameof(license), license, "Unknown model license.")
        };

    public static string ToManifestValue(this ModelLane lane) =>
        lane switch
        {
            ModelLane.Commercial => "commercial",
            ModelLane.NonCommercial => "non-commercial",
            ModelLane.Experimental => "experimental",
            _ => throw new ArgumentOutOfRangeException(nameof(lane), lane, "Unknown model lane.")
        };

    public static ModelTask ParseTask(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);

        return value.ToLowerInvariant() switch
        {
            "asr" => ModelTask.Asr,
            "translation" => ModelTask.Translation,
            "tts" => ModelTask.Tts,
            "diarization" => ModelTask.Diarization,
            "vad" => ModelTask.Vad,
            "separation" => ModelTask.Separation,
            "speech-enhancement" => ModelTask.SpeechEnhancement,
            "forced-alignment" => ModelTask.ForcedAlignment,
            "text-refinement" => ModelTask.TextRefinement,
            "overlap-rescue" => ModelTask.OverlapRescue,
            "lip-synthesis" => ModelTask.LipSynthesis,
            "face-detection" => ModelTask.FaceDetection,
            "face-landmarks" => ModelTask.FaceLandmarks,
            _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown model task.")
        };
    }

    public static ModelLicenseKind ParseLicense(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);

        return value.ToLowerInvariant() switch
        {
            "mit" => ModelLicenseKind.Mit,
            "apache-2.0" => ModelLicenseKind.Apache20,
            "apache-2" => ModelLicenseKind.Apache20,
            "apache2" => ModelLicenseKind.Apache20,
            "apache2.0" => ModelLicenseKind.Apache20,
            "cc-by-4.0" => ModelLicenseKind.CcBy40,
            "cc-by-4" => ModelLicenseKind.CcBy40,
            "ccby-4.0" => ModelLicenseKind.CcBy40,
            "ccby4" => ModelLicenseKind.CcBy40,
            "ccby40" => ModelLicenseKind.CcBy40,
            "cc-by-nc-4.0" => ModelLicenseKind.CcByNc40,
            "cc-by-nc-4" => ModelLicenseKind.CcByNc40,
            "ccbync-4.0" => ModelLicenseKind.CcByNc40,
            "ccby-nc-4.0" => ModelLicenseKind.CcByNc40,
            "ccbync4" => ModelLicenseKind.CcByNc40,
            "ccbync40" => ModelLicenseKind.CcByNc40,
            "nvidia-open-model-license" => ModelLicenseKind.NvidiaOpenModelLicense,
            "nvidiaopenmodellicense" => ModelLicenseKind.NvidiaOpenModelLicense,
            "nvidia-open-model" => ModelLicenseKind.NvidiaOpenModelLicense,
            "openmdw-1.1" => ModelLicenseKind.OpenMdw11,
            "openmdw-11" => ModelLicenseKind.OpenMdw11,
            "openmdw1.1" => ModelLicenseKind.OpenMdw11,
            "openrail++" => ModelLicenseKind.OpenRailPlusPlus,
            "openrail-plus-plus" => ModelLicenseKind.OpenRailPlusPlus,
            "custom" => ModelLicenseKind.Custom,
            "unknown" => ModelLicenseKind.Unknown,
            "non-commercial" => ModelLicenseKind.NonCommercial,
            "noncommercial" => ModelLicenseKind.NonCommercial,
            _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown model license.")
        };
    }

    public static ModelLane ParseLane(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);

        return value.ToLowerInvariant() switch
        {
            "commercial" => ModelLane.Commercial,
            "non-commercial" => ModelLane.NonCommercial,
            "noncommercial" => ModelLane.NonCommercial,
            "experimental" => ModelLane.Experimental,
            _ => throw new ArgumentOutOfRangeException(nameof(value), value, "Unknown model lane.")
        };
    }
}
