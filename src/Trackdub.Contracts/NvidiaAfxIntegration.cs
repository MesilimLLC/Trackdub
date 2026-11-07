namespace Trackdub.Contracts;

/// <summary>
/// Markers for the NVIDIA Audio Effects (AFX) integration. Readiness is probe-based: it only
/// reports Ready for a runtime whose native library, feature DLLs and models are present and
/// that creates and loads an effect on this machine. <see cref="IsStubbed"/> is the kill switch:
/// while it is true, readiness never reports Ready and downloads refuse to install packages.
/// </summary>
public static class NvidiaAfxIntegration
{
    public const string ProviderId = "nvidia-afx";

    public const string DisplayName = "NVIDIA Audio Effects (AFX)";

    public const string StudioVoiceHighQualitySelector = "studio_voice_high_quality";

    public const string StubStatusLabel = "Disabled";

    public const string StubReason =
        "NVIDIA AFX is switched off by the integration kill switch (NvidiaAfxIntegration.IsStubbed()).";

    /// <summary>License for the user-installed AFX SDK and its non-Studio-Voice feature packages.</summary>
    public const string LicenseUrl =
        "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/";

    /// <summary>License for the Trackdub-hosted Studio Voice High Quality runtime package.</summary>
    public const string StudioVoiceLicenseUrl =
        "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/";

    public const string DeveloperResourcesUrl = "https://www.nvidia.com/broadcast-sdk-resources";

    /// <summary>
    /// Early Access effects (Speaker Focus) ship under NVIDIA's evaluation license and must not reach
    /// end users. They stay disabled unless <c>TRACKDUB_AFX_ALLOW_EARLY_ACCESS=1</c> is set for
    /// development or evaluation.
    /// </summary>
    public static bool AllowEarlyAccessEffects() =>
        Environment.GetEnvironmentVariable("TRACKDUB_AFX_ALLOW_EARLY_ACCESS") == "1";

    // Method (not const) so the kill-switch gates do not create CS0162 unreachable-code failures.
    public static bool IsStubbed() => false;
}
