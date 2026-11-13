# HANDOFF — TensorRT-RTX 1.7 / EP ABI 0.5.0 upgrade + adopting the new features

Date: 2026-10-22
Predecessor: `docs/archive/HANDOFF-trt-rtx-1.6-cu13.4-upgrade.md` (the 1.5 → 1.6 move; its Phase 3 "do not regress" list still applies verbatim)
Goal: move the ORT EP ABI plugin route from **EP ABI 0.4.2 (win) / 0.4.0 (linux), TRT-RTX 1.6.1, cu13** to **EP ABI 0.5.0, TRT-RTX 1.7.x, cu13** on both platforms, prove it with measurements, then adopt the 1.7 / 0.5.0 features that pay for Trackdub, each behind its own measured gate.

Do the work as a **stack** (`gh stack`), one PR per part below. Part 1 is mandatory. Parts 2–6 are opt-in features, and each one ships only if its gate measures a win.

---

## What shipped upstream (verified 2026-10-22)

**TensorRT-RTX 1.7** (<https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.7.html>)
- Null weight placeholders for weight-stripped builds, which lower build-time memory.
- **Weights offloading**: unload an inactive engine's weights and restore them later without a rebuild (CUDA VMM mapping).
- Faster JIT context creation, faster runtime-cache (de)serialization, less host memory.
- Kernel and fusion work: FMHA, GEMV heuristics, 1D conv/deconv, conv+const-mul/div, pointwise→conv, and the Windows concat→conv perf bug.
- Fixes: a **multi-engine-per-CUDA-context concurrency bug**, a strided-deconv hang (CC ≥ 8.0), FP16 overflow in ReduceLogSumExp/LpPool/LpNormalization/ReduceLogSum, causal-mask attention accuracy, **dynamic-shape `Squeeze` with omitted axes** (`Cannot infer squeeze dimensions from a dynamic shape!`), and a refittable+loop build crash on Windows.
- Shorter engine build time across all architectures.
- Removes internal `nvinfer1*InternalBuildFlags` / `setDebugOutput` exports. We don't call these, but confirm with `rg`.
- Unchanged: **engines are not forward-compatible, so rebuild everything.** The timing cache is still deprecated. Known issue: **on Windows with driver R615 or earlier, TRT-RTX inference after a DirectML `RecordDispatch` in the same process fails with `CUDA_ERROR_INVALID_VALUE`.** The workaround is a 99 KiB shared-memory pool limit. We run DML and TRT-RTX in one process, so this one matters.

**TensorRT-RTX-EP-ABI v0.5.0** (released 2026-10-21, <https://github.com/NVIDIA/TensorRT-RTX-EP-ABI/releases/tag/v0.5.0>)
- Assets: `…-v0.5.0-cu13.zip` (win-x64, 112.8 MB), `…-cu13-linux-x86_64.zip` (146.5 MB), `…-cu13-win-arm64.zip`, `…-cu13-linux-aarch64.zip`, plus cu12 variants. **Windows and Linux share one tag again.**
- README compatibility row: EP **0.5**, base ORT **1.30.0**, compatible ORT **1.25.0 – 1.30.0**, TRT RTX **1.7.x.x**.
- Driver floor (cu13 prebuilt): Windows **≥ 580.88**, Linux **≥ 590.48.01**. These are the same as 0.4.
- Features: runtime weight-streaming budget updates (`SetEpDynamicOptions`), CUDA-graph capture stability fixes (aux streams, allocator teardown), sandbox-safe CPU-only AOT (`nv_target_sm`), opt-in persistent execution-context memory, deferred/streamed weight loading through the EP-context payload, constant-folded fp16/bf16 `DequantizeLinear`, runtime-cache teardown fix, security hardening, and "standardized EP registration name to `NvTensorRTRTXExecutionProvider`".

**Provider-option diff, `src/tensorrt_rtx_provider_options.h` v0.4.2 → v0.5.0:**

| Key | Change | Trackdub impact |
|---|---|---|
| `device_id` | **Removed** | We don't pass it to TRT-RTX (only MIGraphX uses `device_id`). Add a guard test anyway (Part 1.6). |
| `nv_weight_streaming_budget` | Now **dynamic** (`SetEpDynamicOptions`). Accepts `0`, `-1` (auto), `1` (min VRAM), bytes, `512M`-style values, or `0%–100%`. | Part 3 |
| `nv_persistent_context_memory` | New | Part 2 |
| `nv_weight_stripped_engine_enable_experimental` | New, needs 1.7 | Part 4 |
| `nv_target_sm` + env config `ep_factory.NvTensorRTRTXExecutionProvider.nv_target_sm` / `nv_hardware_device_id` | New. Values are `sm86`, `sm89`, `sm120`, `sm121`. | Part 5 |
| `nv_use_sync_gpu_allocator` | Present since 0.4. Docs now call out Windows VA-space exhaustion under `cudaMallocAsync`. | Part 6 |

---

## Pre-flight blockers — resolve before writing code

### B1. ORT version: we are on **1.31.0**, and 0.5.0 lists 1.25–1.30
`Directory.Packages.props` sets `OnnxRuntimeVersion` to `1.31.0`. The 0.4 row said 1.25–1.27.1 and we already run 0.4.x on 1.31. That works because of EP ABI version negotiation (added in 0.3). So the table is "tested range", not a hard gate. Prove it again for 0.5.0:
1. The plugin registers on ORT 1.31 (`trackdub providers trt-rtx status`, check the plugin log for negotiated API version).
2. The EP-context compile/load round-trip and `SetEpDynamicOptions` both work. These are the ORT APIs that 0.5.0 leans on more than before.
3. If registration fails, **stop**. Either pin ORT to 1.30.x for the whole app (a large blast radius, so call it out in the PR) or wait for an EP ABI build against 1.31. Do **not** downgrade ORT silently.

**B1 results by host runtime** (recorded per route, not pooled):

| Route | Host runtime | Plugin registration | EP-context round trip / `SetEpDynamicOptions` |
|---|---|---|---|
| Windows ML, native Windows (RTX 5070, driver 617.42) | ORT 1.27.1 (inside the 1.25-1.30 tested range) | OK (`providers trt-rtx status`, measurements audit) | `cache warm` ran in the measurement pass; `SetEpDynamicOptions` not yet exercised |
| LinML route on WSL2 (RTX 5070, driver 621.01, Ubuntu 26.04) | ORT 1.31.0 (outside the tested range) | OK: LinML patched plugin registers 1 GPU device; `providers trt-rtx status` Ready (2026-10-24) | Not run. Only a tiny MatMul+Relu inference (max error 1.9e-6) was checked |
| Stock NVIDIA zip on WSL2 | ORT 1.31.0 | Registers, exposes 0 GPU devices (NVIDIA/TensorRT-RTX-EP-ABI#12) | n/a |

The 1.31 route still owes the EP-context compile/load round trip and `SetEpDynamicOptions` checks above.

### B2. EP registration-name standardization vs the WinML catalog filter
`src/Trackdub.Inference.Onnx/TensorRtRtx/WindowsMlCatalogProviderFilter.cs` excludes the WinML catalog's TRT-RTX EP from bulk registration by comparing to `NvTensorRTRTXExecutionProvider` with `OrdinalIgnoreCase`. The doc comment says the catalog spells it `NvTensorRtRtxExecutionProvider`. The ignore-case match is what makes the exclusion work, and it keeps working whatever casing NVIDIA standardizes to. The risk is in the other direction:
- `OnnxExecutionSessionFactory.IsTensorRtRtxDeviceCandidate` matches the **bundled** plugin with `StringComparison.Ordinal`. If 0.5.0 now registers under a different casing, device selection finds no TRT-RTX device, and every stage silently falls back to DML/CPU.
- **Verify:** after install, log the enumerated `OrtEpDevice.EpName` strings (`trackdub providers list --verbose` or a one-off test) and compare them byte-for-byte with `TensorRtRtxProviderConstants.PluginOrtExecutionProviderName`. If they differ, update the constant. Ordinal matching stays correct, because the bundled plugin must never be confused with the catalog EP. Also update the filter's doc comment and the `EpName` assertions in `OnnxExecutionSessionFactoryTests`.
- Also confirm the WinML catalog EP (`Microsoft.WinML.NVIDIA.TRT-RTX.EP.*`) still loads **excluded** when both are present (ADR-0002 point 2).

### B3. LinML WSL2 route has its own artifact
`LinMlWsl2TensorRtRtxPlugin.cs` points users to the LinML release `ep-trt-rtx-wsl2-0.4.0-wsl2p1`, a glibc-floor rebuild (see `HANDOFF-linml-trt-rtx-wsl2-glibc-floor.md`). That artifact won't move by itself. Either rebuild it from the v0.5.0 source as `ep-trt-rtx-wsl2-0.5.0-wsl2p1` in the LinML repo, or **explicitly keep WSL2 on 0.4.0 / 1.6.1**. The fingerprint already separates them, so a split is safe. Record which one you chose in the PR.

---

## Part 1 — Core upgrade (mandatory, ships alone)

### 1.1 Get and inspect the bundle
```powershell
tools/dev/Update-TrtRtxEpManifest.ps1 -Version 0.5.0 -TrtRtxRuntimeVersion 1.7.<x> -WhatIf
```
- Download both cu13 zips once and **list their contents before touching constants.** Record:
  - the exact Windows runtime/parser DLL names (expected `tensorrt_rtx_1_7.dll`, `tensorrt_onnxparser_rtx_1_7.dll`. Verify, don't assume)
  - whether `tensorrt_plugins.dll` / CUDA runtime DLLs are still present or still statically linked (the 1.6 bundle statically links cudart, and our Windows install hint claims that)
  - the Linux SONAME chain (`libtensorrt_rtx.so.1.7.x` → `.so.1` → `.so`) and whether the zip still nests a folder. `TrtRtxEpBundleDownloaderTests` mirrors the 0.4.0 layout.
  - the exact 1.7 patch version (read it from the SONAME or the DLL version resource). This becomes `BundledTrtRtxRuntimeVersion`.
- Then run the script for real. **Drop `-LinuxVersion`**, because both platforms share the 0.5.0 tag now.

### 1.2 `runtime/trt-rtx-ep.manifest.json`
- `version` → `0.5.0`, `trtRtxRuntimeVersion` → `1.7.x`
- Remove the per-package `linux-x64.version: "0.4.0"` override
- Set the real `sha256` / `sizeBytes` for both packages, from the script output
- Decide on `win-arm64` / `linux-aarch64`: **out of scope** for this part (no test hardware). Leave them out of the manifest and note it in the PR.

### 1.3 `src/Trackdub.Inference/Runtime/TensorRtRtx/TensorRtRtxProviderConstants.cs`
| Constant | Today | Target |
|---|---|---|
| `TensorRtRuntimeFileNameWindows` | `tensorrt_rtx_1_6.dll` | 1.7 name from 1.1 |
| `TensorRtOnnxParserFileNameWindows` | `tensorrt_onnxparser_rtx_1_6.dll` | 1.7 name from 1.1 |
| `BundledVersionWindows` / `BundledVersionLinux` | `0.4.2` / `0.4.0` | both `0.5.0`. Collapse to one `BundledVersion` const if nothing else needs the split, and delete the "different tags" comment. |
| `BundledTrtRtxRuntimeVersion` | `1.6.1` | `1.7.x` |
| `WindowsInstallHint` | "…TensorRT-RTX 1.6 bundle" | 1.7. Keep the "no CUDA Toolkit" claim only if 1.1 confirms static cudart. |
| `LinuxInstallHint` | `libcudart.so.13` | confirm against the 1.1 listing |

Mirror `src/Trackdub.Infrastructure/Runtime/TrtRtxEp/TrtRtxEpRequiredFiles.cs` (it hard-codes the `_1_6` names as a second list). **Better:** make it derive from `TensorRtRtxProviderConstants.RequiredPluginFileNames` so the next bump touches one place. Add a test that asserts the two lists agree, if they can't be merged because of project layering.

`BundledFingerprintVersion` becomes `0.5.0+cu13+trt-rtx-1.7.x`. That automatically invalidates smoke verdicts and EP-context stamps (the gap the 1.6 handoff closed). **Confirm with a test, don't assume** (1.6).

### 1.4 Engine/cache invalidation, which is non-negotiable here
1.7 engines can't load 1.6 engines and vice versa. Every persisted artifact keyed on the old runtime must miss:
- `nv_runtime_cache_path` directory (`ResolveTensorRtRtxRuntimeCachePath`). Check that its path includes the fingerprint version. If it doesn't, a stale runtime cache could be fed to 1.7. The safest behavior is to key the directory by `BundledFingerprintVersion`, and have `IsSupersededManagedInstallDirectory`-style cleanup delete the old one.
- `.epc.onnx` + `.stamp.json` (`EpContextArtifact.Stamp.TrtRtxEpVersion`)
- `FileSmokeVerdictStore` entries (`SmokeVerdictKey.TrtRtxEpVersion`)
- Classic-TRT engine cache directories are fingerprinted separately (`Blackwell-617.42-trt-10.13.3-ort-1.31.0`) and aren't affected.
- Superseded install dir `Providers/trt-rtx/0.4.2/cu13/win-x64`: `IsSupersededManagedInstallDirectory` should flag it for cleanup. Verify the Model Manager offers the cleanup and doesn't keep loading the old one.

### 1.5 Stale-string sweep
Update the following, and leave the dated handoffs alone:
- `docs/legal/legal.md`, `docs/legal/THIRD_PARTY_NOTICES.md` (EP ABI tag, TRT-RTX version, license URL unchanged)
- `docs/reference/tensorrt-rtx-ep-abi-plugin.md`, `docs/reference/reference.md`, `docs/reference/gpu-execution-providers.md`, `docs/reference/profiling-report.md`, `docs/reference/nvidia-afx-wiring.md`
- `docs/operations/cloud-gpu-container.md`. The live fingerprint example `0.4.0+cu13+trt-rtx-1.6.1` becomes historical. Add a new measured line after Part 1 lands, rather than editing the dated record.
- `src/Trackdub.Inference/Runtime/Planning/StageRuntimeRequirements.cs` comments that cite "EP ABI 0.4.2/cu13, TensorRT-RTX 1.6.1". Rewrite them **only after** re-testing the claim on 1.7 (see 1.7 below). Otherwise keep the version in the comment as the date of the evidence.
- `src/Trackdub.Inference.Onnx/OnnxExecutionSessionFactory.cs` and `tests/Trackdub.Inference.Tests/OnnxExecutionSessionFactoryTests.cs:1257` evidence comments: same rule.
- GPU docs MCP: `tools/mcp-trackdub-gpu-docs/corpus/manifest.v0.json` (`epAbiVersion`, `trtRtxRuntimeVersion`, `releaseAssets`, `bundledWindowsLibs`, `pinWarning`, the v0.4.2 release-notes source → v0.5.0, and add the 1.7 release notes plus the v0.5.0 `technical-notes.md` as sources), `server.py` instructions, `README.md`. Re-ingest with `uv run trackdub-gpu-docs-ingest`.
- Docs RAG: `tools/docs-rag/sync_corpus.py:340` pin sentence, `tools/docs-rag/SPEC.md`. Re-sync the R2 corpus.
- The MCP server instructions themselves say "TRT-RTX 1.6.1". After re-ingest, confirm `search_trackdub_gpu_docs` answers with 1.7.

Acceptance: `rg "tensorrt_rtx_1_6|onnxparser_rtx_1_6|1\.6\.1|0\.4\.2|v0\.4\.0"` hits only dated handoffs/audits, deliberate historical evidence comments, and test fixtures that intentionally model the *old* version (e.g. "1.6.1 → 1.7.x must invalidate").

### 1.6 Tests to add or update
- `FileSmokeVerdictStoreTests`: add a `…+trt-rtx-1.6.1` → `…+trt-rtx-1.7.x` case that must miss and prune.
- `EpContextArtifact` stamp test: a 1.6.1-stamped artifact resolves to `null` under the 1.7 constants.
- `TrtRtxBundlePinTests`: the current install dir is `0.5.0\cu13\win-x64`, and `0.4.2\cu13\win-x64` **and** `0.4.0\cu13\linux-x64` are superseded.
- `TrtRtxEpBundleDownloaderTests`: update the fixture to the 0.5.0 Linux layout found in 1.1.
- `TensorRtRtxCudaRuntimeBootstrapTests`: new DLL names.
- New: `BuildTensorRtRtxOptions` never emits `device_id`, so a caller-supplied `device_id` should be dropped with a log line. That's cheap insurance against the removed key.
- New: `TrtRtxEpRequiredFiles` ≡ `TensorRtRtxProviderConstants.RequiredPluginFileNames`.

### 1.7 Re-test the TRT-RTX denials that 1.7 claims to fix
These are evidence questions. Change policy only on a measured pass.

| Denial today | 1.7 fix that might lift it | Test |
|---|---|---|
| **silero-vad** excluded from TRT-RTX: "squeeze index (1) must be less than length (0)" then IConditionalOutputLayer rank mismatch (#329) | dynamic-shape `Squeeze` with omitted axes | `providers trt-rtx smoke --model silero-vad`. The `If`/conditional half may still fail. If so, update the comment to "still fails on 1.7: <error>" and keep the denial. |
| **Kokoro** CUDA/CPU only, "cannot infer duration-dependent shapes" | `Squeeze` fix, build-time reductions | `providers trt-rtx smoke --model kokoro`, plus `cache warm` |
| **CosyVoice int8** "Myelin rejects int8 dtype" | "error for certain supported INT8 convolutions", WoQ conv improvements | smoke on the `int8` variant. The vocoder CPU fallback is a separate issue. |
| **whisper-onnx** production-excluded (fused contrib ops; large-v3 `ShapeInferenceNotRegistered` on 1.5) | FMHA fusion, `Squeeze` | rerun the `HANDOFF-whisper-trtrtx-debug.md` matrix for tiny/base/small/large-v3 |
| **Qwen3-ASR** TRT graph falls back on unsupported plugin ops | probably none (importer plugin ops) | one smoke run to confirm; expect no change |
| sortformer `Reshape … dimension of -1 has indeterminate solution` (c13 triage) | build changes | `cache warm` on sortformer, then compare with `docs/audits/c13-trt-rtx-shape-triage-2026-10-05.md` |

For each one: record pass/fail with the exact error in a new audit `docs/audits/trt-rtx-1.7-denial-retest-<date>.md`. A lifted denial is its **own small PR** on the stack, with the planner test (`PlanAsync_*`) updated alongside.

### 1.8 DirectML + TRT-RTX same-process issue (driver ≤ R615)
The 1.7 known issue lands squarely on us: `OnnxExecutionProviderDiscovery` registers DML from the WinML catalog, and stages mix DML and TRT-RTX in one process.
- Add a driver-version check, using the same reader `MachineHardwareProfileProvider` uses. On Windows with driver **≤ R615**, emit `nv_max_shared_mem_size = 101376` (99 × 1024) in `BuildTensorRtRtxOptions`. **Verify the EP's units and semantics before shipping.** The release note expresses the workaround as `kTACTIC_SHARED_MEMORY` in the builder config, and `nv_max_shared_mem_size` is the EP option that maps to it. Check the v0.5.0 source (`src/` for `kTACTIC_SHARED_MEMORY`) to confirm the mapping.
- The current floor is ≥ 580.88, so R580–R615 drivers are both supported and affected. Gate on the driver range rather than always applying the limit, because the cap costs performance on new drivers.
- `trackdub doctor`: add a warning row when the driver is in the affected range ("update to R616+ for full TRT-RTX performance alongside DirectML").
- Test: unit-test the option emission across the driver-version boundary. If an R615 box is available, run a DML stage then a TRT-RTX stage in one `warm-host` run.

### 1.9 Part 1 measurement — before/after, same machine, same fixture
Baseline first on **0.4.2 / 1.6.1**, then upgrade, `trackdub cache clear engines`, and repeat. Use the commands from the 1.6 handoff (§4.2, §4.3):
- `cache warm` on **sortformer**, **parakeet-tdt-0.6b-v3** (encoder+decoder), **madlad400 `trt-fp16`**, **DeepFilterNet3**: record `compileMilliseconds`, epc size, EP-context marker presence.
- Fresh-process cold load: source vs `.epc.onnx`, `--runs 3`. Record cold load, warmup, and the `Cold load breakdown` (`engine cache=wrote/hit` bytes).
- **Second fresh-process load** (warm runtime cache). This is where 1.7's faster runtime-cache deserialization and JIT context creation should show. Measure `phase:onnx-session-create` specifically.
- Per-stage `controlled` runs on `baseline-v1/short.mp4` (`--mode fresh-process --reuse-engine-cache` and `--mode warm-host`) for `asr` (parakeet), `translation` (`--model madlad400`), `tts`.
- Steady-state inference: MADLAD decode tok/s with `enable_cuda_graph` on and off (`OnnxModelBenchmarkRunner` A/B). The memory note "fp16 bucketed TRT is 19x faster than int4 CPU" is the reference point. **Confirm the speedup survives 1.7.**
- Peak process VRAM per stage (`WindowsProcessGpuMemoryReader`).
- Linux: repeat `cache warm` + one parakeet smoke in the `trackdub-gpu:server` container on the RTX 5070 box. Expect the fingerprint `Blackwell|<driver>|0.5.0+cu13+trt-rtx-1.7.x|1.31.0`.

**Part 1 is done when:** no stage regresses beyond noise (±5%) on cold load, warm load or steady state; every previously-passing smoke still passes; and the before/after table is in the PR body.

---

## Part 2 — Persistent execution-context memory (`nv_persistent_context_memory`)

**What it does:** it keeps the engine-wide execution-context device memory allocated across `Run()` calls, sized for the engine's maximum. Per-run allocations go away. The cost is resident VRAM until the session dies, even for small dynamic shapes.

**Where it pays for us:** per-call stages with many short `Run()`s: the MADLAD decoder step loop, the Parakeet decoder/joint, and the CosyVoice/Kokoro flow steps. It doesn't help one-shot encoders.

**Implementation**
1. Add a per-model opt-in alongside the existing per-model TRT option sets (`MadladTrtProfiles`, `NemotronAsrEncoderTrtProfiles`). Don't add a global default.
2. **Session-pool accounting:** persistent memory is resident for the session's lifetime, so the pool's admission estimate (e.g. MADLAD ≈ 8069 MB) must include the max-shape context memory when this is on. Otherwise the 75%/85% VRAM budget overcommits. Measure the delta with `WindowsProcessGpuMemoryReader` and fold it into the model's VRAM estimate.
3. It applies in both normal and EP-context execution, so it doesn't interact with `.epc` stamping.

**Gate:** MADLAD decode step latency p50/p95 and tok/s improve by **≥ 5%**, with the VRAM delta recorded and the pool estimate updated. Turn it on only for models that pass.

---

## Part 3 — Dynamic weight-streaming budget (`nv_weight_streaming_budget` + `SetEpDynamicOptions`)

**What it does:** a nonzero budget at build time makes the engine weight-streamable. Since 0.5.0, the budget can change **on a live session** through `SetEpDynamicOptions` without rebuilding. Values: `-1` auto, `1` minimum VRAM, bytes / `512M`, or `0%–100%` of streamable weights resident.

**Why it fits Trackdub:** the pipeline runs stages one after another (VAD → ASR → MT → TTS), and the session pool holds several large TRT-RTX sessions. MADLAD alone is about 8 GB. On 8–12 GB cards we currently evict or deny. With streaming, an idle-but-pooled session can shrink its resident weights instead of being evicted, and grow back when its stage becomes active.

**We already have the plumbing:** `src/Trackdub.Inference.Onnx/WindowsMl/OrtEpDynamicOptions.cs` calls native `SetEpDynamicOptions` (today only for WinML workload type).

**Implementation**
1. Generalize `OrtEpDynamicOptions`: add `TrySetWeightStreamingBudget(InferenceSession, string budget, out string? error)`, and move it out of the `WindowsMl` namespace since it's now cross-EP and cross-platform. Confirm the reflection surface exists in the **Linux** managed ORT build too.
2. Build engines that are streaming-capable: set `nv_weight_streaming_budget` to `100%` (fully resident, but streamable) at session creation for the **large** models only (MADLAD, CosyVoice LLM part if it ever runs on TRT-RTX). Small models gain nothing. **This changes the engine**, so it must be part of the EP-context compile options and therefore of the `.epc` stamp, or a non-streamable `.epc` gets loaded where a streamable one was expected. Add the option to the stamp's option fingerprint.
3. Session pool policy: when the pool would evict a TRT-RTX session that is streaming-capable, call `TrySetWeightStreamingBudget(session, "<low>%")` first and recompute the admission. On lease (stage becomes active), restore `100%`. Keep the eviction path as the fallback.
4. **CUDA-graph interaction:** the header says graph eligibility is decided from the resolved budget *when each execution context is created*. A session that captured a graph at 100% and is then shrunk may drop graph replay or recapture. Test MADLAD (which enables `enable_cuda_graph`) through a shrink → restore cycle and check outputs stay **bit-identical** to a never-shrunk session. Stale-buffer replay is exactly the bug our `enable_cuda_graph` doc comment warns about.

**Gate:**
- On an 8 GB-class budget (`TRACKDUB_SESSION_VRAM_BUDGET_MB=8192` on the RTX 5070 to simulate), the full pipeline completes **without** a MADLAD evict-and-rebuild, and total wall time beats the evict path.
- Steady-state MADLAD tok/s at 100% resident stays within noise of the non-streamable engine. If streamable engines are measurably slower at 100%, restrict this to low-VRAM machines (detected budget < model estimate).
- Output parity test passes after shrink/restore.

---

## Part 4 — Weight-stripped (weightless) EP-context engines — experimental

**What it does:** `nv_weight_stripped_engine_enable_experimental=1` at compile produces a small engine plus an `ep_refit_table`. At load, the EP refits the weights from the EP-context model's **kept initializers**, so the original ONNX isn't needed. Requires TRT-RTX 1.7+ and ORT ≥ 1.27 (we're on 1.31). There is **no automatic fallback**: an unsupported arch or SDK fails the compile.

**Why it might pay:** today `cache warm` writes a `.epc.onnx` that embeds full engines next to the source model. For MADLAD that roughly doubles disk use. With weight-stripping we'd ship and store one copy of the weights, and models above 2 GB stop forcing `ShouldEmbedEpContext=false` (external engine files). Combined with 1.7's null-weight placeholders, peak build-time memory also drops. That matters for compiling MADLAD on 8 GB cards.

**Implementation (behind a hidden setting, `StudioSettings.TrtRtxWeightStrippedEpContext`, default off)**
1. `EpContextCompiler` / `EpContextCapture`: when enabled, add the option to the compile session.
2. **Validation must stay honest** (1.6 handoff Phase 3): the compiler currently requires `com.microsoft.ep.context` / `engine_data` markers. Extend it to require a non-empty `ep_refit_table` when stripping is on. The upstream note says "the refit table is empty after export filtering" is a known failure mode, so a stripped artifact without a refit table must be rejected, not kept.
3. Stamp the mode in `EpContextArtifact.Stamp`, so a stripped and a full artifact are never confused.
4. Read upstream's "Known limitations (current version)" section in `docs/technical-notes.md` @ v0.5.0 before starting, and copy the relevant ones into our reference doc.

**Gate:** for sortformer and MADLAD: artifact size (stripped vs full), compile time, peak compile VRAM, **cold load time** (refit cost vs full-engine deserialize), and output parity. Ship as default only if cold load is no worse than +10% and disk drops substantially. It's experimental upstream, so expect it to stay opt-in for this release.

---

## Part 5 — Sandbox-safe CPU-only AOT compile (`nv_target_sm`)

**What it does:** a compile-only session with `nv_target_sm=sm86|sm89|sm120|sm121` skips all CUDA/device init and builds engines with **no GPU driver**. To pre-describe a device at enumeration time, use the env config `ep_factory.NvTensorRTRTXExecutionProvider.nv_target_sm` (+ optional `nv_hardware_device_id`) in `OrtEnvCreationOptions`.

**Why it might pay:** (a) precompile `.epc` artifacts in CI or on a GPU-less AWS builder and ship them in model packs, so users skip the 20–90 s first-load build; (b) compile in a separate low-priority process without touching the user's GPU.

**The catch, from the 1.7 known issues:** "CPU-only AOT can lead to reduced performance for some models … best performance if they instead perform AOT on-device (`kCURRENT`)." So precompiled artifacts are a **cold-start** optimization that may cost **steady-state** throughput.

**Implementation (investigation first, no product wiring in this stack)**
1. Spike: compile parakeet + MADLAD with `nv_target_sm=sm120` on a box with no GPU visible to the process (`CUDA_VISIBLE_DEVICES=""` in the Linux container), then load on the RTX 5070.
2. Measure against an on-device (`kCURRENT`) compile: cold load, steady-state latency/tok/s, parity.
3. Stamp design: a precompiled artifact needs `target_sm` in its stamp, and the loader must reject it on a mismatched architecture. 1.7 now gives a clearer error for this, but we want to refuse before touching native code.
4. Write up the result as `docs/audits/trt-rtx-1.7-cpu-aot-spike-<date>.md`. Productizing it (shipping precompiled packs, a hybrid "load precompiled, recompile on-device in the background, swap") is a **separate follow-up plan** that depends on these numbers.

---

## Part 6 — Synchronous GPU allocator as a targeted mitigation (`nv_use_sync_gpu_allocator`)

The option has existed since 0.4. The v0.5.0 technical notes now document the exact failure mode: on Windows, `cudaMallocAsync` pools can **exhaust process virtual address space while VRAM is still free**, which gives nondeterministic OOMs that depend on process history. Long-running Studio sessions that churn sessions through the pool are the textbook trigger.

**Implementation**
1. Don't flip the default. Sync allocation disables CUDA-graph support and can cost a little performance. Add a `StudioSettings` / env override (`TRACKDUB_TRT_RTX_SYNC_ALLOCATOR=1`) that `BuildTensorRtRtxOptions` honors, and make it **mutually exclusive with `enable_cuda_graph=1`** (force graphs off and log it).
2. Classify the failure: when a TRT-RTX session create or run fails with an OOM while `IAvailableVramReader` reports headroom, surface a diagnostic that names the setting. Don't auto-retry silently.
3. Soak test: a 2-hour `warm-host` loop that cycles all stages through the pool on Windows, once with the default allocator and once with sync. Record OOM count, peak commit/VA, throughput delta.

**Gate:** if the soak reproduces VA-exhaustion OOMs with async and not with sync, make sync the automatic retry after that specific classified failure. Otherwise ship only the override.

---

## Not adopting (and why)

- **Weights offloading (TRT-RTX 1.7 runtime API).** This is a native TRT-RTX feature, and EP ABI 0.5.0 exposes no provider option for it. Part 3 (dynamic streaming budget) is the EP-level route to the same benefit. Revisit when the EP surfaces it.
- **Edge-LLM backend.** Trackdub's LLM-ish models run through ORT/GenAI. Not applicable.
- **cu12 packages.** Our driver floor is already cu13-capable (≥ 580.88), and the prebuilt cu12 assets would mean a second variant to support.
- **win-arm64 / linux-aarch64.** No test hardware. That needs a separate plan, including the Qualcomm/Snapdragon story.
- **Timing cache.** Deprecated and has no effect on TRT-RTX engines. Make sure we don't set it anywhere (`rg -i timing.?cache src`).
- **ORT bump.** We stay on 1.31. Only revisit if B1 fails.

---

## Acceptance — Part 1 (required to merge the bottom of the stack)

- [ ] B1: plugin registers and EP-context compile/load works on ORT 1.31 (evidence in the PR)
- [ ] B2: enumerated `EpName` verified byte-for-byte; constant/tests updated if it changed; WinML catalog EP still excluded
- [ ] B3: LinML WSL2 decision recorded (rebuilt to 0.5.0, or explicitly held at 0.4.0)
- [ ] Manifest at 0.5.0 for win-x64 + linux-x64 with real hashes; constants and `TrtRtxEpRequiredFiles` agree (ideally one source)
- [ ] Fingerprint `0.5.0+cu13+trt-rtx-1.7.x`; tests prove 1.6.1 verdicts, stamps and runtime cache all miss
- [ ] `cache clear engines` wipes engine cache + verdicts + `.epc.onnx`/`.stamp.json`; superseded 0.4.x install dirs are offered for cleanup
- [ ] `trackdub providers trt-rtx status` + `trackdub doctor` green on Windows and in the Linux container
- [ ] DML+TRT driver ≤ R615 mitigation emitted only in range, plus a doctor warning
- [ ] Stale-string `rg` clean except intentional history; GPU-docs MCP + docs RAG re-ingested and answering with 1.7
- [ ] Denial retest audit written; any lifted denial is its own PR
- [ ] Before/after table (1.9) in the PR body: no regression beyond ±5%, and the MADLAD TRT fp16 speedup confirmed
- [ ] `dotnet test Trackdub.slnx -m:1` green (apart from known unrelated failures, listed by name)

Parts 2–6 each carry their own gate above. A part that misses its gate ships as an audit note, not code.

## Suggested stack order

1. `trt-rtx-1.7/core`: Part 1.1–1.6 + 1.8 (code, manifest, strings, tests)
2. `trt-rtx-1.7/measure`: 1.9 numbers + 1.7 retest audit (docs only)
3. `trt-rtx-1.7/lift-<model>`: one per denial that passes (often none, which is fine)
4. `trt-rtx-1.7/persistent-ctx`: Part 2
5. `trt-rtx-1.7/ws-budget`: Part 3 (largest product change, so review it carefully)
6. `trt-rtx-1.7/sync-alloc`: Part 6
7. Part 4 and Part 5 are spikes, as docs-only PRs or local branches until their numbers justify product work

## Environment gotchas
Same as the 1.6 handoff (`ProgramFiles` missing in the shell, stale `MSBuildSDKsPath`, leftover `testhost` locks). In addition:
- Firecrawl credits are exhausted. Read NVIDIA pages with the built-in browser or `gh api repos/NVIDIA/TensorRT-RTX-EP-ABI/contents/<path>?ref=v0.5.0`.
- WSL has no internet: build on Windows, then copy (see memory "Gated UI test Linux repro").

## References
- TRT-RTX 1.7 release notes: <https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/release-notes-1/1.7.html>
- EP ABI v0.5.0 release: <https://github.com/NVIDIA/TensorRT-RTX-EP-ABI/releases/tag/v0.5.0>
- EP ABI @ v0.5.0: `README.md` (compat table), `docs/prerequisites.md` (driver floors), `docs/technical-notes.md` (sync allocator, weightless EPContext), `src/tensorrt_rtx_provider_options.h` (option keys)
- In-repo: `docs/archive/HANDOFF-trt-rtx-1.6-cu13.4-upgrade.md`, `docs/reference/tensorrt-rtx-ep-abi-plugin.md`, `docs/decisions/ADR-0002-windows-ml-provider-strategy.md`, `tools/dev/Update-TrtRtxEpManifest.ps1`, `tools/dev/Fetch-TrtRtxEp.ps1`, `src/Trackdub.Inference.Onnx/WindowsMl/OrtEpDynamicOptions.cs`
