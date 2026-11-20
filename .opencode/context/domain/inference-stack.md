# Inference stack

Three projects, one direction: abstractions then implementation then native assets.

```
Trackdub.Inference                     provider-neutral: manifests, registry, download
                                       planning, EP descriptors, variant selection
        ▲
        │  (interfaces only)
Trackdub.Inference.Onnx                concrete ONNX Runtime / Windows ML sessions
                                       and mappers
        ▲
        │  (native assets, no project edge)
Trackdub.OnnxRuntime.Dnnl.Native       RID native package + checksum provenance
```

## What each layer holds

| Layer | Path | Contents |
|---|---|---|
| `Inference` | `src/Trackdub.Inference/Runtime/ModelManifest/` | `bundled-models.manifest.json`, `ModelManifestLoader`, `BundledModelManifestRegistry`, `ModelHashVerifier`, `ModelExpectedRuntime`, alias resolver, schema JSONs |
| `Inference` | `src/Trackdub.Inference/Runtime/Planning/` | `RuntimePlanner`, `RuntimePlanFactory`, `StageRuntimePlan`, `HardwareMatrix`, `AffinityRule`, `IVramMonitor`, ranking strategy |
| `Inference` | `src/Trackdub.Inference/Runtime/ExecutionProviders/` | EP descriptors + capability checks for planning (read its README; session construction is **not** here) |
| `Inference` | `src/Trackdub.Inference/Runtime/ModelRegistry/` | installed/available model lookup and state |
| `Inference` | `src/Trackdub.Inference/Runtime/ModelDownloads/` | download queue, verification, cache placement |
| `Inference` | `src/Trackdub.Inference/Runtime/VariantSelection/` | FP16/INT8/CPU/GPU variant resolution |
| `Inference` | `.../WinMlCatalog`, `.../TensorRtRtx`, `.../NativeCudaTensorRt`, `.../Migraphx` | provider-specific constants/ordering only |
| `Inference.Onnx` | `src/Trackdub.Inference.Onnx/` | concrete sessions, mappers, smoke testers |
| `OnnxRuntime.Dnnl.Native` | `src/Trackdub.OnnxRuntime.Dnnl.Native/` | `runtimes/{win-x64,linux-x64,osx-x64}/native`, `provenance/dnnl-native-assets.template.json` |

## Execution providers

`ExecutionProviderKind` (in `src/Trackdub.Domain/Common/RuntimePlanning.cs`) enumerates `Cpu`, `DirectMl`, `TensorRTRtx`, `OpenVino`, `CoreMl`, `Cuda`, `TensorRt`, `Migraphx`, `Qnn`, plus further catalog entries.

Global probe order when a stage allows multiple providers (`Milestone5PlanningPolicy.SupportedProvidersThisMilestone`, per `Runtime/Planning/README.md`):

```
TensorRTRtx → Migraphx → OpenVinoCatalog → Qnn → VitisAi → TensorRt
           → Cuda → OpenVino → DirectMl → Cpu
```

Individual stages override via `AllowedProvidersByEngineFamily` (e.g. Kokoro TTS stays CPU-only; `whisper-onnx` and `latentsync-diffusion` skip TensorRT families because those graphs cannot import). `Cuda` is the Linux path; on Windows the pin remaps to `TensorRTRtx`. CPU is the terminal fallback.

Windows strategy (ADR-0002, `docs/decisions/ADR-0002-windows-ml-provider-strategy.md`): TensorRT RTX via the standalone ORT EP ABI plugin; Windows ML as the ONNX integration surface for DirectML and catalog EPs; DirectML is legacy fallback; CPU terminal. DirectML package references are banned — `DependencyGraphTests.WindowsOnnxRuntimePackagesUseWinMlCatalogProvider` asserts `Microsoft.ML.OnnxRuntime.DirectML` appears in no `PackageVersion`/`PackageReference` and that legacy DirectML copy targets do not exist.

`StageRuntimePlanStatus` (`src/Trackdub.Domain/Common/RuntimePlanning.cs`): `Ready` = files present + integrity ok (CPU stops here, no smoke test); `Verified` = **stronger**: files present *and* a runtime smoke test passed for the selected non-CPU EP; `DownloadRequired`; `Blocked`. `StageRuntimePlanningRequest.SkipProviderSmokeTest` exists for listing/inventory only; it is part of the plan cache key so a listing `Ready` can never satisfy a later verified plan.

### Verify any version or API claim

Do not quote an ONNX Runtime, CUDA, or TensorRT version from memory.

```bash
grep -n "OnnxRuntimeVersion" Directory.Packages.props          # 1.30.0 today; single source of truth
python -c "import json;print(json.load(open('runtime/trt-rtx-ep.manifest.json'))['version'])"
dotnet list src/Trackdub.Inference.Onnx package                  # resolved graph
```

Use `trackdub-gpu-docs` (local, offline) for TRT-RTX EP ABI facts, `nvidia-cuda-docs` for CUDA toolkit internals, and `trackdub-docs-rag` for Trackdub pin policy. Vendor docs are upstream reference, not Trackdub pin policy.

## Tooling directories

| Directory | Purpose |
|---|---|
| `tools/onnxruntime-dnnl/Build-OnnxRuntimeDnnlNativePackage.ps1` | builds the DNNL-flavored ONNX Runtime native package; x64 hosts only, rejects Arm64 with an explicit message |
| `tools/olive/` | TRT-RTX conversion/validation lane: `Bootstrap-TrtRtxOliveVenv.ps1`, `Export-QwenTrtRtxGenAi.ps1`, `Flip-WhisperOnnxTrtRtx.ps1`, `Validate-*` scripts, `TrtRtxOliveCommon.ps1` (dot-sourced helpers that pin the expected ORT version and assert the TRT-RTX EP ABI surface) |
| `tools/model-lab/` | graph-surgery scripts: `decompose-mha-qkv.py`, `decompose-microsoft-contrib-ops.py`, `decompose-whisper-mha.py`, `decompose-whisper-cross-attention.py` |
| `tools/dev/model-conversion/` | per-model ONNX export/upload pipelines (whisper GenAI, Qwen text refinement, opus, latentsync) |
| `tools/mcp-trackdub-gpu-docs/` | local FastMCP over the Trackdub GPU/TRT-RTX corpus, allowlist keyed to the pinned EP ABI |
| `tools/docs-rag/` | corpus manifest (`corpus.v1.json`) for the hosted `trackdub-docs-rag` MCP; agent spec in `SPEC.md` |
| `tools/quality-baseline/` | pre/post-decompose ASR quality baselines, `compare_transcripts.py` (WER/CER gates) |
| `tools/separation-eval/` | separation evaluation assets |
| `tools/models/` | starter-pack publish + download smoke |
| `tools/espeak-ng/` | espeak-ng acquisition manifest (IPA phoneme transcription, dev-time only) |
| `tools/localization/` | `App.{culture}.resx` display languages |
| `tools/bench-per-stage.ps1`, `tools/bench-smoke-verdict-ab.ps1` | per-stage and A/B smoke benchmarks over `Benchmarks.DevHost` |
| `tools/ci/` | manifest schema/hash/license audits, dependency-graph verifier, manifest apply scripts |

`TrackdubOrtRuntimeFlavor` gates asset copying in `src/Trackdub.Composition/Trackdub.Composition.csproj`: `CopyDnnlOrtAssetsToOutput` / `ValidateDnnlOrtAssets` run only for `Dnnl`; WinML and stock-ORT-GPU targets only when `!= 'Dnnl'`, where stock ORT runtime assets are excluded from package references. Both directions are asserted by `DependencyGraphTests`.

## Model lifecycle

```
manifest entry
  → download (or import)
  → checksum verify
  → license metadata present
  → license review
  → commercial-use decision
  → hardware provider availability
  → stage enabled in the run
  → stage ran
  → output usable
```

Manifest fields that matter (see `bundled-models.manifest.json`): `model_id`, `task`, `engine_family`, `tier`, `license`, `commercial_allowed`, `redistribution_allowed`, `requires_attribution`, `requires_user_consent`, `voice_cloning`, `commercial_use_verified`, `revision`, `sha256`, `aliases`, `benchmark_entry`, `download_file_sources`, `download_file_hashes`, `variants[].alias/entry_path`.

Rules: commercial license only; unknown license is unsafe; `ModelManifestLoader` enforces license/commercial policy at load time rather than filtering at runtime. Changes affecting model governance must update the manifest and, where attribution applies, `THIRD_PARTY_NOTICES.md`.

```bash
python tools/ci/validate-manifest-schema.py
python tools/ci/audit-bundled-model-manifest.py
python tools/ci/verify-manifest-hashes.py --structural --all-families
dotnet test tests/Trackdub.Inference.Tests --filter "FullyQualifiedName~ModelManifest" -m:1
```

## Readiness ladder

Each rung is a separate claim. Never collapse them.

| Rung | Question | Evidence that proves it |
|---|---|---|
| registered | is a provider registered in DI? | registration in `CompositionRoot.cs` |
| installed | is the runtime/EP present on this machine? | runtime locator / readiness service result |
| downloaded | are the model files on disk? | manifest-relative file existence |
| checksummed | do hashes match? | `ModelHashVerifier` / `verify-manifest-hashes.py` |
| licensed | is license metadata recorded and reviewed? | manifest fields + review record |
| commercial-use decision | is `commercial_allowed` + `commercial_use_verified` satisfied? | `BundledModelManifestRegistry.CommercialSafeMode` outcome |
| provider available | did the EP probe succeed for this stage? | `StageRuntimePlanStatus.Verified` |
| enabled | was the stage enabled in this run? | `StageReadiness.Status` / run options |
| ran | did execution occur? | `StageRunRecord` with a non-skip terminal state |
| output usable | is the artifact valid and usable downstream? | `ProjectArtifact` + resume evaluator |

Contract type: `ReadinessState` in `src/Trackdub.Contracts/Pipeline/ReadinessState.cs` — blocking states `ProviderMissing`, `RuntimeMissing`, `DownloadRequired`, `ImportRequired`, `IntegrityFailed`, `LicenseReviewRequired`, `CommercialBlocked`, `CloudKeyMissing`, `ConsentRequired`, `CloudEgressConsentRequired`; non-blocking `Ready`, `Satisfied`, `SkippableOptional`, `Unverified`.

## Grounding MCP servers

From `.mcp.json`:

| Server | Transport | Use for |
|---|---|---|
| `trackdub-gpu-docs` | stdio, `uv --directory tools/mcp-trackdub-gpu-docs run trackdub-gpu-docs-mcp` | offline TRT-RTX / EP ABI lookup; tools `list_corpus`, `search_trackdub_gpu_docs`, `get_doc`. **Disabled by default** in `.mcp.json` — enable before use. |
| `nvidia-cuda-docs` | HTTP, NVIDIA endpoint | CUDA toolkit/driver internals outside the corpus |
| `trackdub-docs-rag` | HTTP + bearer `${DOCS_RAG_TOKEN}` | Trackdub implementation facts, pin policy, provider wiring. Tools `search_trackdub_docs`, `ask_trackdub_docs`, `get_trackdub_doc`. **Disabled by default**; needs the token env var. |
| `serena` | stdio | symbol-aware code navigation when installed and configured with the REPL interface |

Prefer `trackdub-docs-rag` for Trackdub facts, Context7 for third-party libraries outside the corpus, and live code over both.

## Related

- `context/standards/architecture-rules.md` — no inference leaking upward
- `context/domain/terminology.md` — disabled vs ran vs skipped
- `context/standards/validation-gates.md` — gate order