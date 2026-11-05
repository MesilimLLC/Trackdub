# NVIDIA AFX readiness contract

Trackdub integrates **NVIDIA Audio Effects (AFX / Maxine)** as an optional speech-enhancement backend. DeepFilterNet3 remains the default and the fallback. Layout, native behavior and verification are in [nvidia-afx-wiring.md](nvidia-afx-wiring.md).

The file keeps its original name for existing links; the integration is no longer a stub.

## Honesty contract

| Signal | Behavior |
|--------|----------|
| `INvidiaAfxRuntimeReadinessService` | `NvidiaAfxRuntimeReadinessService` (probe-based). Ready only when the license is accepted, the native library, any feature DLLs the layout requires and every model the profile needs are present, **and** an effect creates, loads and runs a frame at every supported rate on this machine. Only successful native probes are cached; file checks run on every call. |
| `NvidiaAfxIntegration.IsStubbed()` | Kill switch, `false`. While `true`, readiness never reports Ready, the installer and downloader refuse, and enhancement falls through to DeepFilterNet. |
| `NvidiaAfxSpeechAudioEnhancementService` | Falls through to DeepFilterNet when disabled, not ready, missing an AEC far-end reference, when the readiness check throws, or when the native run fails. |
| `NvidiaAfxRuntimeDownloader` / `NvidiaAfxRuntimeInstaller` | Still refuse placeholder packages, missing license acceptance and missing manifests. Settings can download the Trackdub-hosted Studio Voice High Quality runtime (NVIDIA Open Model License) for Turing, Ampere, Ada, and Blackwell; other AFX profiles still require a local SDK. |
| `nvidiaafx-runtime.manifest.json` | Pinned `downloads.trackdub.com` URLs, SHA-256 hashes, and sizes for the hosted Studio Voice runtime, split into ten parts per architecture. Readiness uses it as an architecture whitelist; the installer reads it to gate downloads and pick the architecture-matched package (`NvidiaAfxRuntimePackageGates`). |

Never treat component registration, settings fields or profile catalog entries as proof that AFX ran. The stage result's `Backend` is `NvidiaAfx` only when the native effect actually produced the audio.

## Discoverability

| Area | Location |
|------|----------|
| Provider id / kill switch | `src/Trackdub.Contracts/NvidiaAfxIntegration.cs` |
| Profile enum | `src/Trackdub.Contracts/NvidiaAfxProfile.cs` |
| Studio settings | `EnableNvidiaAfx` / `NvidiaAfxProfile` / `NvidiaAfxIntensityRatio` / `NvidiaAfxLicenseAccepted` / `NvidiaAfxRuntimeDirectory` |
| Enhancement options | `SpeechAudioEnhancementOptions` (+ `FarEndReferenceAudioPath`, `FromStudioSettings`) |
| Profile catalog | `src/Trackdub.Composition/NvidiaAfx/NvidiaAfxProfileCatalog.cs` |
| Readiness | `NvidiaAfxRuntimeReadinessService`, `NvidiaAfxInstalledRuntimeEvaluator`, `NvidiaAfxSessionEffectProbe` |
| Native seams | `NvidiaAfxNative`, `NvidiaAfxNativeLoader`, `NvidiaAfxSession` |
| On-disk layout | `NvidiaAfxRuntimeLayout` |
| GPU architecture | `NvidiaAfxArchitectureDetector` (display adapter, `TRACKDUB_NVIDIA_GPU_NAME` override) |
| Packaging gates | `NvidiaAfxRuntimePackageGates`, `NvidiaAfxRuntimePathResolver`, `NvidiaAfxRuntimeInstaller` |
| DI | `CompositionRoot.AddApplication`, `HeadlessCompositionRoot` |
| Stage provenance | `SpeechAudioEnhancementStageHandler` (`nvidia-afx` backend tag) |
| Live GPU proofs | `tests/Trackdub.Composition.Tests/NvidiaAfxLiveRuntimeTests.cs` |

## Open items

- Trackdub hosts the Studio Voice High Quality runtime under the NVIDIA Open Model License. The core AFX SDK and other feature packages remain user-installed under NVIDIA's SDK license.
- AEC needs a far-end reference that dubbing never has, so it is hidden from the UI and untested live.
- Speaker Focus (Early Access) is in the catalog but gated: it is hidden from `SelectableDefinitions` and readiness refuses it ("Early Access disabled") unless `TRACKDUB_AFX_ALLOW_EARLY_ACCESS=1` is set, because it ships under NVIDIA's evaluation license, which permits internal testing and evaluation only, not production use.
- Live proofs need a Windows NVIDIA RTX machine and never run in default CI.
