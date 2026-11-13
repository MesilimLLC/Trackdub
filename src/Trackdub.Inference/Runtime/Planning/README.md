# src/Trackdub.Inference/Runtime/Planning/README.md

## Purpose

Runtime planning: manifest-driven provider selection, smoke-test gating, and stable diagnostics shapes.

## Scope boundaries

- `RuntimePlanner` and `RuntimePlanFactory` live here.
- `Milestone5PlanningPolicy.SupportedProvidersThisMilestone` defines the **global** probe order when a stage allows multiple providers:

  `TensorRTRtx` -> `Migraphx` -> `OpenVinoCatalog` -> `Qnn` -> `VitisAi` -> `TensorRt` -> `Cuda` -> `OpenVino` -> `DirectMl` -> `Cpu`

  Individual stages may override via `AllowedProvidersByEngineFamily` (for example Kokoro TTS remains CPU-only; stock `whisper-onnx` and `latentsync-diffusion` skip TensorRT families because those graphs cannot import). Most ONNX spine stages reference the shared default list.
- `StageRuntimePlanningRequest.SkipProviderSmokeTest` is for listing and inventory. GPU routes then report `Ready` (files + EP present) instead of compiling graphs. Pipeline, `SelectRouteAsync`, and starter-pack apply keep the smoke gate and report `Verified`. The skip flag is part of the plan cache key so listing `Ready` cannot satisfy a later verified plan. Starter-pack presentation and import pass `skipProviderSmokeTest: true`; `StarterPackApplyService` uses the default (`false`).
- Planner output is a stable diagnostics/logging shape and must not expose absolute machine-local paths.
- Smoke-test execution is interface-driven (`IExecutionProviderSmokeTester`); concrete ONNX sessions live in `Trackdub.Inference.Onnx`.

## Expectations

- Reuse `BundledModelManifestRegistry` as the source of model metadata truth; commercial eligibility is enforced when manifests are loaded and in bundled-inventory tests, not via runtime filtering.
- Treat preferred aliases as soft ranking hints inside `StageRuntimeRequirements`.
- Commercial-safe ASR blocking is driven by manifest fields (commercial_allowed, commercial_use_verified, lane) read by the planner. `ModelManifest.CommercialSafeMode` is a read-only property computed from commercial_use_verified, not a runtime toggle, and there is no CommercialSafeEvaluator type. Blocking behaviour with the current bundled inventory is covered by planner tests.
- On Windows, TensorRT RTX is the standalone ORT EP ABI plugin route, while Windows ML catalog EPs (MIGraphX/OpenVINO/QNN/VitisAI) are preferred over DirectML legacy fallback per [ADR-0002](../../../../docs/adr/ADR-0002-windows-ml-provider-strategy.md); registration != readiness.
