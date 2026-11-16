namespace Trackdub.Domain;

public sealed record ModelLanguagePair(string SourceLanguage, string TargetLanguage);

public sealed record LocalModelCacheRecord
{
    public LocalModelCacheRecord(
        string ModelId,
        string RootPath,
        string Revision,
        string Sha256,
        DateTimeOffset CachedAtUtc,
        bool IntegrityFailed = false,
        IReadOnlyList<LocalModelVariantRecord>? Variants = null)
    {
        this.ModelId = ModelId;
        this.RootPath = RootPath;
        this.Revision = Revision;
        this.Sha256 = Sha256;
        this.CachedAtUtc = CachedAtUtc;
        this.IntegrityFailed = IntegrityFailed;
        this.Variants = Variants ?? [];
    }

    public string ModelId { get; init; }

    public string RootPath { get; init; }

    public string Revision { get; init; }

    public string Sha256 { get; init; }

    public DateTimeOffset CachedAtUtc { get; init; }

    public bool IntegrityFailed { get; init; }

    public IReadOnlyList<LocalModelVariantRecord> Variants { get; init; } = [];
}

public sealed record LocalModelVariantRecord(
    string Alias,
    string RootPath,
    string EntryRelativePath,
    IReadOnlyList<string> ComponentRelativePaths,
    string OptimizerId,
    ExecutionProviderKind ExecutionProvider,
    string Precision,
    DateTimeOffset CreatedAtUtc,
    string SourceModelRevision,
    string? SourceModelSha256 = null,
    bool IntegrityFailed = false,
    ModelOptimizedVariantProvenance? Provenance = null);

public sealed record ModelOptimizedVariantProvenance(
    string? OliveVersion,
    string CommandKind,
    IReadOnlyList<ModelOptimizationOperation> Operations,
    string OliveProvider,
    string Device,
    string? RecipeConfigPath,
    string? RecipeConfigSha256,
    string? QuantizationMethod,
    string? Evaluator,
    ModelOptimizationExpectedOutput OutputKind,
    ModelOptimizationFallbackPolicy FallbackPolicy,
    IReadOnlyList<string>? ScriptIdentifiers = null)
{
    public IReadOnlyList<string> ScriptIdentifiers { get; init; } = ScriptIdentifiers ?? [];
}

public enum ModelCacheState
{
    Missing = 0,
    Downloading = 1,
    Installed = 2,
    Corrupt = 3,
    Blocked = 4,
    Ready = 5
}

public sealed record ModelOptimizationAvailability(
    bool HasProfile,
    bool CanOptimize,
    IReadOnlyList<string> ComponentRelativePaths,
    IReadOnlyList<ExecutionProviderKind> AvailableProviders,
    string? UnavailableReason,
    string? EntryRelativePath = null,
    string? Mode = null,
    IReadOnlyList<string>? BaseVariantAliases = null,
    IReadOnlyList<string>? SupportedPrecisions = null,
    int? DeclaredOpset = null,
    IReadOnlyList<ModelOptimizationOpsetPolicy>? OpsetPolicies = null,
    bool RequireOpsetMetadata = false,
    IReadOnlyList<ModelOptimizationRecipeBinding>? RecipeBindings = null,
    ModelOptimizationFallbackPolicy FallbackPolicy = ModelOptimizationFallbackPolicy.None)
{
    public static ModelOptimizationAvailability None { get; } = new(
        HasProfile: false,
        CanOptimize: false,
        ComponentRelativePaths: [],
        AvailableProviders: [],
        UnavailableReason: null,
        EntryRelativePath: null,
        Mode: null,
        BaseVariantAliases: [],
        SupportedPrecisions: [],
        DeclaredOpset: null,
        OpsetPolicies: [],
        RequireOpsetMetadata: false,
        RecipeBindings: []);

    public IReadOnlyList<string> BaseVariantAliases { get; init; } = BaseVariantAliases ?? [];

    public IReadOnlyList<string> SupportedPrecisions { get; init; } = SupportedPrecisions ?? [];

    public IReadOnlyList<ModelOptimizationOpsetPolicy> OpsetPolicies { get; init; } = OpsetPolicies ?? [];

    public IReadOnlyList<ModelOptimizationRecipeBinding> RecipeBindings { get; init; } = RecipeBindings ?? [];
}

public sealed record ModelOptimizationRecipeBinding(
    string ConfigRelativePath,
    string? Provider = null,
    string? Precision = null,
    IReadOnlyList<ModelOptimizationOperation>? Operations = null,
    ModelOptimizationExpectedOutput ExpectedOutput = ModelOptimizationExpectedOutput.OnnxComponents,
    ModelOptimizationFallbackPolicy? FallbackPolicy = null,
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
    public IReadOnlyList<ModelOptimizationOperation> Operations { get; init; } = Operations ?? [];
}

public enum ModelOptimizationOperation
{
    OnnxExport, QnnConversion, OpenVinoConversion, Compression,
    ProviderOptimization, GenAiPackaging, ModelSplitting,
    Evaluation, AdapterHandling, Registration
}

public enum ModelOptimizationExpectedOutput
{
    OnnxComponents, OrtGenAi, QnnModelLibrary,
    OpenVinoModel, SplitOnnxComponents, AdapterPackage
}

public enum ModelOptimizationFallbackPolicy
{
    None, AutoOptAllowed, BaseVariantAllowed, CpuRuntimeAllowed
}

public sealed record ModelOptimizationOpsetPolicy(
    ExecutionProviderKind? Provider,
    string? Precision,
    int MinimumOpset);

public sealed record ModelOptimizedVariantInfo(
    string Alias,
    string OptimizerId,
    ExecutionProviderKind ExecutionProvider,
    string Precision,
    ModelCacheState State,
    DateTimeOffset CreatedAtUtc,
    string RootPath,
    string EntryRelativePath,
    IReadOnlyList<string> ComponentRelativePaths,
    string? FailureReason = null,
    ModelOptimizedVariantProvenance? Provenance = null);

public sealed record ModelInventoryEntry(
    string ModelId,
    string DisplayName,
    string Task,
    string EngineFamily,
    string License,
    bool CommercialAllowed,
    bool CommercialUseVerified,
    ModelCacheState State,
    long? FileSizeBytes,
    DateTimeOffset? CachedAtUtc,
    string? FailureReason,
    string? LanguageCoverageDisplay = null,
    string? ExpectedRuntime = null,
    string? ExpectedRuntimeHint = null,
    bool CanAutoDownload = true,
    string? ModelRootPath = null,
    bool IsOliveOptimizable = false,
    ModelOptimizationAvailability? OptimizationAvailability = null,
    IReadOnlyList<ModelOptimizedVariantInfo>? OptimizedVariants = null,
    IReadOnlyList<string>? Aliases = null,
    IReadOnlyList<ModelLanguagePair>? LanguagePairs = null)
{
    public IReadOnlyList<ModelOptimizedVariantInfo> OptimizedVariants { get; init; } = OptimizedVariants ?? [];

    public IReadOnlyList<string> Aliases { get; init; } = Aliases ?? [];

    public IReadOnlyList<ModelLanguagePair> LanguagePairs { get; init; } = LanguagePairs ?? [];
}
