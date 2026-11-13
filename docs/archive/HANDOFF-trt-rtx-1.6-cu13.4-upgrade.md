# HANDOFF — TensorRT-RTX 1.6 / CUDA 13.4 upgrade + proof it works

Date: 2026-09-24
Branch: `main` (work alongside an in-flight session-pool / lease-bundle refactor — do not revert unrelated diffs)
Machine: Windows 11 (build 26200), **Blackwell** GPU (driver 32.0.16.1714), pwsh, `Trackdub.slnx`
Goal: move the ORT EP ABI plugin route from **TRT-RTX 1.5 / cu12 (CUDA 12.9-class)** to **TRT-RTX 1.6 / CUDA 13.4** — code + every hardcoded string + CUDA-runtime handling — then **test and investigate until it actually works**, unlike the current pin.

---

## Status: current pin is BROKEN for EP-context AOT (measured 2026-09-24)

The plugin *registers* fine. It does **not** deliver its advertised AOT/cold-load win. Evidence on sortformer
(`cgus/diar_streaming_sortformer_4spk-v2.1-onnx`, 469 MB) with `--provider trt-rtx`:

| # | Failure | Measurement |
|---|---|---|
| 1 | **EP-context AOT emits nothing** | `cache warm` → `CompileModel()` under `TensorRTRtx` ran **86 689 ms** and the output contained **zero** `com.microsoft.ep.context` / `engine_data` markers. Artifact was source + **146 KB** (Conv 329→385, Squeeze 68→85 = reserialized graph, not engines). Discarded by `EpContextCompiler` validation. |
| 2 | **AOT silently ran under DirectML before** | earlier `cache warm`: `compiled` in 115 ms, `selected provider was 'DirectMl'` → stamped bogus artifact → cold-load **regressed +6 s**. Root cause: TRT plugin never registered before the first `OrtEnv` touch. *Fixed in-tree; must not regress.* |
| 3 | **Cold load is engine-build dominated, not model load** | CPU cold load **4 185 ms** vs trt-rtx **59 170–65 780 ms** for the same 469 MB graph. |
| 4 | **Engine cache barely helps** | `cache warm` seeded only **2 files / 6.73 MB** (`NvTensorRTRTXExecutionProvider_*`); warm-vs-cold gain ≈ **6.6 s** (65 780 → 59 170 ms). |
| 5 | **TRT-RTX shape/profile error at compile** | log: `Profile kMIN values are not self-consistent … /pre_encode/Reshape: … dimension of -1 has indeterminate solution` |
| 6 | **silero-vad is TRT-incompatible** | `cache warm` → `InvalidGraph` (`_ORT_MEM_ADDR_`); trt-rtx benchmark Failed (`If`/`Squeeze` graph) |

Current pin (all of these change):
- `runtime/trt-rtx-ep.manifest.json` → EP ABI **v0.3.0**, `cudaVariant: "cu12"`
- bundled libs: `onnxruntime_providers_nv_tensorrt_rtx.dll`, **`tensorrt_rtx_1_5.dll`**, **`tensorrt_onnxparser_rtx_1_5.dll`**, `tensorrt_plugins.dll`, **`cudart64_12.dll`**
- install path: `%LOCALAPPDATA%\Trackdub\Providers\trt-rtx\0.3.0\cu12\win-x64\`

Target: **TRT-RTX 1.6** on the **CUDA 13.4** package line.
NVIDIA 1.6 "Available Versions" (verify yourself): `1.6 CUDA 12.9` + `1.6 CUDA 13.4` for **Linux x86_64 and Windows** (13.4 is *not* ARM-only); `13.4` only for Linux aarch64 / Windows Arm64.

---

## ⚠️ Step 0 — pick the asset path BEFORE touching code

**Two version axes — do not conflate them:**
- **EP ABI plugin** (`onnxruntime_providers_nv_tensorrt_rtx.dll`) versions as `0.x.y` — currently v0.3.0, and v0.3.0 *vendors* TRT-RTX **1.5** (`tensorrt_rtx_1_5.dll`).
- **TRT-RTX runtime** versions as `1.x` — currently 1.5, target **1.6**.

Options (record which one you took in the PR):
- **(A) preferred** — NVIDIA shipped a newer `TensorRT-RTX-EP-ABI` release with **cu13** assets / `tensorrt_rtx_1_6*` libs → pin it. Check <https://github.com/NVIDIA/TensorRT-RTX-EP-ABI/releases> (our docs corpus only knows v0.3.0).
- **(B)** — no such release → **build the EP ABI from source** against TRT-RTX 1.6 + CUDA 13.4 (EP ABI README: `build.bat --cuda_home … --onnxruntime_home … --trt_rtx_home … --version 0.x.y`; EP ABI needs ORT ≥ 1.23, we are on 1.30). Ship the built set the same way (manifest + downloader + install dir).
- **(C) NOT ACCEPTABLE** — drop `tensorrt_rtx_1_6.dll` next to the existing 0.3.0 plugin with no rebuild. Lineage/ABI mismatch, untested.

Also install the standalone SDK for the CLI isolation tests (Phase 4.1):
`TensorRT for RTX 1.6 CUDA 13.4 (Windows)` x64 zip → extract to **`C:\SDK\TensorRT-RTX-1.6.0.0`**, add `lib` + `bin` to `PATH`
(NVIDIA install page is location-agnostic: "decompress … add the package `lib` and `bin` directories to your `PATH`"). Note: a PATH install does **nothing** for the app route — the plugin loads `tensorrt_rtx_*` from its own bundle directory.

**CUDA Toolkit state on this machine (verified 2026-09-24):** Toolkit **v13.3 only** (`CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`, `nvcc` on PATH) — **no 12.x has ever been installed here** (13.4 is being installed for this upgrade). This did *not* break the current cu12 pin: the bundle is self-contained — it ships its own `cudart64_12.dll` (0.56 MB) beside the plugin and dependent-DLL resolution finds it there (engines demonstrably built on this box). Consequences:
- The two real failures in the status table are **not** missing-toolkit symptoms (those would be `DllNotFoundException` / `cudart64_12.dll` not found). Don't stop at "install CUDA and retest".
- A Toolkit *is* required for: **path (B) EP ABI source build** (`--cuda_home` must point at the Toolkit matching the package line — use 13.4) and anything invoking `nvcc` (Olive / pycuda recipes).
- For the 1.6 upgrade take **TensorRT for RTX 1.6 CUDA 13.4 (Windows)** to match the 13.4 Toolkit. (`cudart64_13.dll` is major-versioned — all 13.x share it — so 13.3 would likely run it, but the supported pairs are exactly 12.9 U1 / 13.4 per the Support Matrix.)
- `TRACKDUB_CUDA12_BIN_DIR` was never set here and was never the load path for cudart (bundle-local wins). Phase 1.3's generalization still applies.

---

## Phase 1 — code + manifest + strings

### 1.1 `runtime/trt-rtx-ep.manifest.json`
- `version` → new EP ABI version (if (B) rebuilds 0.3.0 against 1.6, decide explicitly and say so)
- `cudaVariant` → `cu13`
- `packages.win-x64.archiveUrl` / `linux-x64.archiveUrl` → the cu13 assets
- `sha256` / `sizeBytes` → real values. Try `tools/dev/Update-TrtRtxEpManifest.ps1 -Version <x.y.z>`; if it only knows `cu12`, **extend it with `-CudaVariant cu13`** and compute sha/size yourself.
- Mirror copy of the pin lives in `tools/mcp-trackdub-gpu-docs/corpus/manifest.v0.json` (`epAbiVersion`, `cudaVariant`, `releaseAssets`, `bundledWindowsLibs`) — update and re-ingest the corpus (`tools/mcp-trackdub-gpu-docs`: `uv run trackdub-gpu-docs-ingest`).

### 1.2 `src/Trackdub.Inference/Runtime/TensorRtRtx/TensorRtRtxProviderConstants.cs` — the hardcoded strings
| Constant | Today | Action |
|---|---|---|
| `TensorRtRuntimeFileNameWindows` | `tensorrt_rtx_1_5.dll` | → actual 1.6 filename (**verify from the bundle**; expect `tensorrt_rtx_1_6.dll`) |
| `TensorRtOnnxParserFileNameWindows` | `tensorrt_onnxparser_rtx_1_5.dll` | → 1.6 filename |
| `BundledVersion` | `0.3.0` | → new EP ABI version (drives install path **and** both invalidation fingerprints — see 1.5) |
| `BundledCudaVariant` | `cu12` | → `cu13` |
| `CudaRuntimeBinDirectoryEnvironmentVariable` | `TRACKDUB_CUDA12_BIN_DIR` | see 1.3 |
| `WindowsInstallHint` / `LinuxInstallHint` | `0.3.0 cu12`, `cudart64_12.dll`, "CUDA Toolkit 12.x" | → new pin, `cudart64_13.dll`, CUDA 13.x |
| `TensorRtRuntimeFileNameLinux` / `TensorRtOnnxParserFileNameLinux` | `libtensorrt_rtx.so`, `libtensorrt_onnxparser_rtx.so` | un-suffixed by design — confirm still correct |

`RequiredPluginFileNames`, `GetDefaultInstallDirectory` (`Providers/trt-rtx/<BundledVersion>/<BundledCudaVariant>/<rid>`) follow these automatically — but `TensorRtRtxPluginLocator` validates **exactly** those names, so a 1.6 set is rejected as "required plugin files missing" until the constants move.

### 1.3 CUDA-runtime handling ("the cudart type stuff")
- `TRACKDUB_CUDA12_BIN_DIR` is CUDA-12-named. **Generalize**: prefer `TRACKDUB_CUDA_BIN_DIR`, keep `TRACKDUB_CUDA12_BIN_DIR` as a read-compat fallback. Update every hint/doc that names it.
- `TensorRtRtxCudaRuntimeBootstrap` expects `cudart64_12.dll` → probe `cudart64_13.dll` (or both) and fix the hint text.
- Policy decision (AGENTS.md says no end-user CUDA Toolkit dependency): **bundle `cudart64_13.dll` beside the plugin** (today's cu12 bundle ships `cudart64_12.dll`) rather than requiring a CUDA 13.4 Toolkit / `nvidia-cuda-runtime-cu13` on the user machine. Whatever you choose, make `trackdub providers trt-rtx status` report it truthfully.

### 1.4 Every other stale string (rg inventory, 2026-09-24)
| File | Stale bits |
|---|---|
| `docs/legal/legal.md` (~L239-240) | `0.3.0` / `cu12` / `releases/tag/v0.3.0` |
| `docs/legal/THIRD_PARTY_NOTICES.md` (~L142-143) | same |
| `docs/reference/tensorrt-rtx-ep-abi-plugin.md` | `tensorrt_rtx_1_5.dll`, `tensorrt_onnxparser_rtx_1_5.dll`, `0.3.0 cu12` (≈L38-42, 61, 77-79, 115-116, 206) + the bump checklist at L240 |
| `docs/reference/reference.md` | mirror of the above (≈L1090-1344, 1649) |
| `docs/reference/windows-ml-stage-provider-matrix.md` (L97, L116) | `0.3.0 cu12` path |
| `docs/reference/profiling-report.md` (L111-117) | `0.3.0/cu12` |
| `docs/decisions/decisions.md` (L169) + `docs/decisions/ADR-0002-windows-ml-provider-strategy.md` (L21) | the `_1_5` DLL names in the TRT-RTX route description |
| `tools/olive/Validate-WhisperOnnxTrtRtx.ps1`, `Validate-SortFormerTrtRtx.ps1`, `Validate-NemotronAsrTrtRtx.ps1` | default `…\Providers\trt-rtx\0.3.0\cu12\win-x64` + `cudart64_12.dll` / `tensorrt_rtx_1_5.dll` comments |
| `tools/mcp-trackdub-gpu-docs/README.md` | pin line |
| `HANDOFF-whisper-trtrtx-debug.md` | **leave** — dated historical record |

Phase-1 acceptance: `rg "tensorrt_rtx_1_5|tensorrt_onnxparser_rtx_1_5|cudart64_12|TRACKDUB_CUDA12|cu12"` hits only dated handoffs (or nothing).

### 1.5 ⚠️ Fingerprint gap you must close (this is the "invalidate with the engine cache" rule)
`SmokeVerdictKey.TrtRtxEpVersion` (`src/Trackdub.Contracts/SmokeVerdictStore.cs`) and
`EpContextArtifact.Stamp.TrtRtxEpVersion` (`src/Trackdub.Inference.Onnx/EpContext/EpContextArtifact.cs`)
both use **`TensorRtRtxProviderConstants.BundledVersion`** — the *EP ABI* version.
If the runtime lineage moves 1.5→1.6 while the EP ABI version string does not, **neither fingerprint changes** → stale smoke verdicts and stale `.epc` stamps would survive a runtime bump. That violates the rule these keys exist for.

- Add the **runtime** identity to the fingerprint (e.g. a `BundledTrtRtxRuntimeVersion` constant derived from the `tensorrt_rtx_1_6` filename, or fold the required-runtime-filename into `EnvironmentFingerprint`).
- Unit tests: a runtime-version change must (a) miss `ISmokeVerdictStore.IsVerified`, (b) make `EpContextArtifact.TryResolveValidLoadPath` return null, (c) drop the other environment's entries in `FileSmokeVerdictStore` (its prune-on-record behavior).

---

## Phase 2 — install + register

1. Get the bundle (path A download, or path B build).
2. Install for the **app** route: `%LOCALAPPDATA%\Trackdub\Providers\trt-rtx\<epAbiVer>\cu13\win-x64\` (dev shortcut: `tools/dev/Fetch-TrtRtxEp.ps1`, or set `TRACKDUB_TRT_RTX_EP_DIR` / `StudioSettings.TensorRtRtxPluginDirectory`).
   Locator order (`TensorRtRtxPluginLocator`): explicit studio setting → `TRACKDUB_TRT_RTX_EP_DIR` → installed bundle.
3. Install for **experiments**: `C:\SDK\TensorRT-RTX-1.6.0.0` + PATH.
4. `trackdub providers trt-rtx status` + `trackdub doctor`: plugin dir resolved, **all required DLLs present with the new names**, `NvTensorRTRTXExecutionProvider` GPU device visible, CUDA 13 runtime resolved.
5. `trackdub cache clear engines` after the swap (clears engine cache + smoke verdicts + `.epc.onnx`/`.stamp.json` — this is the shared-invalidation path; verify all three go away).

---

## Phase 3 — do not regress (recently fixed; keep the tests green)

- **Fatal-CTP guards above any bootstrap/native touch** in `OnnxExecutionProviderSmokeTester.SmokeTestAsync` (GenAI/opus-mt/madlad under TRT hard-crash the host). `GenAiTensorRtExclusionTests` enforce "refuse without touching native code".
- **Smoke bootstrap gate** (`OnnxExecutionSessionFactory.BootstrapForSmokeAsync` + `IsRequestFulfilled`) and **`allowTrtInitFallback: false`** on single-graph smoke probes — smoke must prove the requested EP, never a fallback, and failures are never recorded as verdicts.
- **WinML catalog registration before the first `OrtEnv` touch** in `OnnxExecutionProviderDiscovery.DiscoverAsync` — this is what made DirectML smoke pass at all (no native `OrtSessionOptionsAppendExecutionProvider_DML` export on our ORT build; DML only works via a catalog `OrtEpDevice`).
- **`FileSmokeVerdictStore`** keying (`model sha256 | EP | GPU arch | driver | ep-version`) + `EngineCacheMaintenanceService.Clear()` clearing verdicts **and** `.epc.onnx`/`.stamp.json`.
- **`EpContextCompiler.CompileAsync` honest-failure behavior**: register TRT before compile, refuse non-TRT compile, require EP-context markers before keeping an artifact. **Do not weaken this to make `cache warm` look green.**
- Pre-failure message quality: `CreateDirectMlSelection` surfaces both catalog+direct append reasons; keep them.

---

## Phase 4 — TEST & INVESTIGATE (the real deliverable)

**"Working" means the six failures in the status table are gone — not that the plugin registers.
Provider registered ≠ model ready ≠ stage ran.**

### 4.1 CLI isolation (decisive; no ORT in the loop)
```powershell
tensorrt_rtx.exe --onnx=…\sortformer\onnx\model.onnx --saveEngine=sortformer.trt   # AOT
tensorrt_rtx.exe --loadEngine=sortformer.trt --runtimeCacheFile=sortformer.cache    # JIT + cache
```
- Expect: real `sortformer.trt` (engines embedded → **hundreds of MB**, not KB), AOT roughly 20–60 s, JIT first run < 5 s, `sortformer.cache` written (docs: Build Your First Engine).
- If the `Reshape … dimension of -1 has indeterminate solution` profile error appears **here too** → it is TRT-RTX-side (not ORT). Escalate upstream (NVIDIA forum `tensorrt` tag / TensorRT-RTX GitHub) with the model + log; do not paper over in Trackdub.
- Repeat with silero-vad to see whether 1.6 accepts the `If`/`Squeeze` graph. If not, record it as TRT-unsupported (it already falls back to DirectML/CPU) and move on.

### 4.2 EP-context AOT end-to-end (the regression we actually care about)
```powershell
dotnet run --project src/Trackdub.Cli --framework net10.0-windows10.0.19041.0 -- `
  cache warm --model …\sortformer\onnx\model.onnx
```
PASS only if **all** hold:
1. status **`compiled`** (not `failed`), `compileMilliseconds` sane
2. `model.epc.onnx` **meaningfully larger** than `model.onnx` (engines embedded) and contains `com.microsoft.ep.context` / `engine_data` markers (the compiler now checks this — it must pass, not be bypassed)
3. `warmLoadMilliseconds` sane, and the runtime cache picks up kernels

Then measure both graphs (the benchmark takes a **path**, so it measures exactly what you point it at):
```powershell
dotnet run --project src/Trackdub.Benchmarks.DevHost -f net10.0-windows10.0.19041.0 -- `
  --model …\sortformer\onnx\model.onnx    --provider trt-rtx --runs 3 --output …\src.json --format json
dotnet run --project src/Trackdub.Benchmarks.DevHost -f net10.0-windows10.0.19041.0 -- `
  --model …\sortformer\onnx\model.epc.onnx --provider trt-rtx --runs 3 --output …\epc.json --format json
```
Baseline to beat (Blackwell / driver 32.0.16.1714 / EP 0.3.0 cu12, sortformer 469 MB):

| Configuration | Cold load | Warmup | Breakdown note |
|---|---|---|---|
| source, **CPU** | 4 185 ms | 65 ms | load is cheap — NOT the target |
| source, trt-rtx, **cold** engine cache | 65 780 ms | 1 658 ms | `engine cache=empty` |
| source, trt-rtx, warm cache | 59 170 ms | 591 ms | `engine cache=hit` (+0 bytes) |
| `.epc.onnx` (bogus artifact) | 64 866 ms | 1 421 ms | **regression** vs source |
| `cache warm` compile | 86 689 ms | — | 0 EP-context nodes |

Success = `.epc.onnx` cold load **clearly below** the source trt-rtx cold load (the 59–65 s), approaching the docs' AOT/JIT split (engine built once, load is deserialize + `<5 s` JIT). Record the `Cold load breakdown` notes: expect `engine cache=wrote` on the first load and `hit` with real byte counts afterwards (today: `+0 file(s), +0 byte(s)`, 6.73 MB total).

### 4.3 Pipeline-level before/after (per stage)
Fixture: `%LOCALAPPDATA%\Trackdub\benchmark-fixtures\baseline-v1\short.mp4`
(sha256 `c4640c3f8062b4d928eeef25c52f845f4867c10f26aeb4f5d5ce6be1c295bd85`, 8 s)

```powershell
dotnet run --project src/Trackdub.Benchmarks.DevHost -f net10.0-windows10.0.19041.0 -- `
  controlled $fixture --output <dir> --stage <vad|asr|translation|tts> `
  --mode fresh-process --reuse-engine-cache --provider TensorRTRtx --model <alias>
```
- Provider pin takes the **`ExecutionProviderKind` enum name** (`TensorRTRtx`, `DirectMl`, `Cpu`) — *not* `trt-rtx`.
- `--mode warm-host` for the warm-host contrast; `translation` needs `--model madlad400` (default translator is cloud and fails); CosyVoice/Chatterbox are voice-clone TTS and need a reference clip in the fixture.
- Compare `timingsMilliseconds`: `preflight`, `phase:onnx-session-create`, the stage's `durationMilliseconds`, `total`. Ready-made drivers: `tools/bench-per-stage.ps1`, `tools/bench-smoke-verdict-ab.ps1`.
- Also re-verify the smoke-verdict store writes keys carrying the **new** version fingerprint and that `cache clear engines` wipes verdicts + epc artifacts.

### 4.4 Sweep the previously-broken surfaces
```powershell
dotnet test tests/Trackdub.Inference.Tests --filter "FullyQualifiedName~TrtRtx|FullyQualifiedName~EpContext|FullyQualifiedName~Smoke|FullyQualifiedName~EngineCacheProbe"
dotnet run --project src/Trackdub.Benchmarks.DevHost -f net10.0-windows10.0.19041.0 -- --scope trt-rtx-smoke --provider trt-rtx
```
- Starter-pack smoke catalog (`TrtRtxSmokeCatalog`) targets; optional CI tier `.github/workflows/trt-rtx-smoke.yml`.
- Whisper-onnx family from `HANDOFF-whisper-trtrtx-debug.md`: **large-v3 failed on 1.5 with `ShapeInferenceNotRegistered`** — recheck on 1.6 (tiny/base passed; small/medium had unrelated asserts).
- Full suites: `dotnet test Trackdub.slnx -m:1` (expect 1 known failure unrelated to this work: `PlanAsync_TensorRTRtxNotAllowedForTtsFamilies` cosyvoice row — a concurrent change removed the cosyvoice TRT deny in `StageRuntimeRequirements.cs` while the test still asserts DirectMl).

---

## Acceptance — all required

- [ ] `rg "tensorrt_rtx_1_5|tensorrt_onnxparser_rtx_1_5|cudart64_12|TRACKDUB_CUDA12|cu12"` is clean outside dated handoffs
- [ ] `trackdub providers trt-rtx status` green on the new bundle; `trackdub doctor` shows the new path + CUDA 13 runtime
- [ ] `cache clear engines` removes engine cache **and** smoke verdicts **and** `.epc.onnx`/`.stamp.json`
- [ ] `cache warm` on sortformer: `compiled=1`, epc contains EP-context markers, epc ≫ source in size
- [ ] fresh-process cold load: `.epc.onnx` **beats** source trt-rtx, and both beat the 59–65 s baseline meaningfully
- [ ] `Cold load breakdown` shows `engine cache=wrote` then `hit`, with real persisted bytes (> the 6.73 MB we see today)
- [ ] DirectML smoke still passes and the verdict store populates (discovery-registration fix intact); verdict keys carry the new version fingerprint
- [ ] fatal-CTP guard tests + smoke-tester tests + `EngineCacheProbeTests` + `FileSmokeVerdictStoreTests` green
- [ ] `legal.md` / `THIRD_PARTY_NOTICES.md` / reference docs / olive validators / gpu-docs corpus all show the new pin
- [ ] PR body records: asset path (A/B), the measured before/after table from 4.2, and any upstream issue filed

## Out of scope
- ORT package bump (stay on `Microsoft.ML.OnnxRuntime` 1.30 unless the new EP ABI demands more; EP ABI needs ORT ≥ 1.23)
- PyPI/Python bindings (not needed — the CLI covers every experiment here)
- silero-vad TRT support if 1.6 still rejects `If`/`Squeeze` — record as unsupported rather than hacking the graph
- Windows ML catalog EP route (ADR-0002: we ship the standalone plugin)

## Environment gotchas (this machine)
- The shell **lacks `ProgramFiles`** and has a **stale `MSBuildSDKsPath`** (Scoop 10.0.400 vs `DOTNET_ROOT` 10.0.401) → NuGet restore dies with `Value cannot be null. (Parameter 'path1')`. Per command:
  ```powershell
  $env:ProgramFiles = "C:\Program Files"
  ${env:ProgramFiles(x86)} = "C:\Program Files (x86)"
  $env:ProgramW6432 = "C:\Program Files"
  Remove-Item env:MSBuildSDKsPath -ErrorAction SilentlyContinue
  ```
- Leftover `testhost` from `dotnet test` locks test bin dirs → `MSB3027` copy failures on build. `Get-Process testhost | Stop-Process -Force` first.
- The working tree is shared with other in-flight work — build all of `Trackdub.slnx` and leave unrelated diffs alone.

## References
- NVIDIA install (extract anywhere; add `lib`+`bin` to PATH): <https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/installing-tensorrt-rtx/installing.html#installing>
- TRT-RTX 1.6 "Available Versions": CUDA 12.9 + 13.4 for Linux x86_64 and Windows; 13.4 also aarch64 / Windows Arm64
- Support Matrix footnote [#f2]: "Separate TensorRT-RTX packages are provided for CUDA 12.9 Update 1 and CUDA 13.4 on x86-64 platforms…"
- EP ABI repo + release notes (v0.3.0 vendors TRT-RTX 1.5; "per-CUDA variants (cu12, cu13)" only in *wheels*; build with `--trt_rtx_home`/`--cuda_home`): <https://github.com/NVIDIA/TensorRT-RTX-EP-ABI>
- In-repo: `docs/reference/tensorrt-rtx-ep-abi-plugin.md` (bundle channel + "Bumping TRT RTX EP release"), `docs/decisions/ADR-0002-windows-ml-provider-strategy.md`, `tools/dev/Update-TrtRtxEpManifest.ps1`, `tools/dev/Fetch-TrtRtxEp.ps1`, `tools/mcp-trackdub-gpu-docs/`
- Related fixed-this-week notes: smoke verdict store (`src/Trackdub.Contracts/SmokeVerdictStore.cs`, `FileSmokeVerdictStore`), `EpContextCompiler` honest validation, `OnnxExecutionProviderDiscovery` pre-OrtEnv registration, `EngineCacheMaintenanceService.Clear()` EP-context cleanup
