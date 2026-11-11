# TensorRT RTX EP ABI plugin

TensorRT RTX is a runtime provider plugin, not a model and not a Windows ML catalog EP.

## Build targets

TRT RTX works on **both** Windows target frameworks because the EP ABI plugin is
plain ONNX Runtime extensibility: it does not need the Windows ML packages:

| Build | TRT RTX | DirectML / WinML catalog EPs |
|-------|---------|------------------------------|
| `net10.0` (portable) | Yes: `NvTensorRTRTXExecutionProvider` via `RegisterExecutionProviderLibrary` | No; unavailable in discovery, CLI warns at parse time |
| `net10.0-windows10.0.19041.0` | Yes | Yes |

On Linux `net10.0` the plugin bundle (`libonnxruntime_providers_nv_tensorrt_rtx.so` +
`libtensorrt_rtx.so` set) registers the same way.

Effective NVIDIA Windows fallback chain: **TRT RTX → DirectML → CPU**. On the
portable build the chain is TRT RTX → CPU, since DirectML requires the
Windows-targeted build.

Trackdub registers the standalone ONNX Runtime EP ABI plugin before creating TRT RTX sessions, then selects the GPU `OrtEpDevice` whose EP name is:

```text
NvTensorRTRTXExecutionProvider
```

Do not accept `NvTensorRtRtxExecutionProvider` as the primary plugin identity. That spelling belonged to the deprecated Windows ML catalog wiring and must not be used for the standalone plugin route.

## Bundle layout

The plugin directory must contain these files together.

Windows:

```text
onnxruntime_providers_nv_tensorrt_rtx.dll
tensorrt_rtx_1_7.dll
tensorrt_onnxparser_rtx_1_7.dll
```

Linux (v0.5.0 cu13, installed through LinML.Catalog; SONAME symlinks are preserved on extract):

```text
libonnxruntime_providers_nv_tensorrt_rtx.so
libtensorrt_rtx.so
libtensorrt_onnxparser_rtx.so
libtensorrt_plugins.so
```

Companion libraries such as `tensorrt_plugins.dll` / `libtensorrt_plugins.so` are copied during install but are not part of the required readiness triple.

## Locator order

`TensorRtRtxPluginLocator` resolves the bundle directory in this order:

1. `StudioSettings.TensorRtRtxPluginDirectory`
2. `TRACKDUB_TRT_RTX_EP_DIR`
3. Default installed bundle under the Trackdub user data root (see below)

If the bundle is missing, use **Install** in Model Manager (downloads v0.5.0 cu13 on Windows; on Linux LinML.Catalog installs its pinned 0.5.0 wheel, persists the studio path, then registers), or run the dev fetch script.

## Fetch script (dev/CI)

From repo root on Windows or Linux x64:

```powershell
.\tools\dev\Fetch-TrtRtxEp.ps1
```

Optional `-InstallRoot` or `TRACKDUB_DATA_ROOT` overrides the user data root. The script verifies SHA-256 + size from `runtime/trt-rtx-ep.manifest.json`, extracts native libraries into the flat plugin directory, and prints `TRACKDUB_TRT_RTX_EP_DIR` for the current shell.

## Bundle channel (in-app + manifest)

Trackdub ships the EP ABI plugin through a pinned manifest, not the Windows ML catalog:

- **Manifest:** `runtime/trt-rtx-ep.manifest.json` (version `0.5.0`, linux-x64 package at the same `0.5.0` as LinML.Catalog's wheel, CUDA `cu13`, `trtRtxRuntimeVersion` `1.7.1` for Windows, per-RID archive URL + SHA-256 + size).
- **Composition copy:** `trt-rtx-ep.manifest.json` next to the app assembly (`Trackdub.Composition` `CopyToOutputDirectory`).
- **Downloader:** `TrtRtxEpBundleDownloader` verifies checksum/size, extracts required plugin files into `%UserDataRoot%/Providers/trt-rtx/<version>/cu13/<rid>/` (0.5.0 on win-x64 and linux-x64).
- **In-app install:** Model Manager **Install** calls `ITrtRtxEpInstaller` after `NvidiaTensorRtRtx` license acceptance, persists `StudioSettings.TensorRtRtxPluginDirectory`, then registers via `ITensorRtRtxProviderBootstrap`.
- **Inference bootstrap:** registers an already-installed bundle only; it **never** downloads the EP bundle (same policy as WinML catalog EPs during session bootstrap).

## Linux: LinML catalog route

On Linux the plugin is not installed from the manifest bundle. `LinMlTensorRtRtxBootstrap`
prepares NVIDIA's pinned EP ABI wheel through `LinML.Catalog` (vendored under
`~/.local/linml-deps`), after the NVIDIA hardware and license gates; Model Manager **Install**
and `trackdub providers trt-rtx install --accept-license` skip the bundle download and call
the bootstrap with downloads allowed. `StudioSettings.TensorRtRtxPluginDirectory` and
`TRACKDUB_TRT_RTX_EP_DIR` still take precedence when they point at a complete plugin
directory. On WSL2 it registers LinML's patched plugin when that is vendored (the stock one
exposes no device there). Windows is unchanged. Details: [gpu-execution-providers.md](gpu-execution-providers.md#tensorrt-rtx-on-linux).

## Download policy (install vs session)

| Path | May download TRT RTX EP bundle? |
|------|--------------------------------|
| Model Manager **Install** / bulk catalog install | Yes (after NVIDIA license acceptance) |
| CLI `trackdub providers trt-rtx install --accept-license` | Yes |
| Inference session bootstrap (`OnnxExecutionSessionFactory`) | **No** |
| Readiness / doctor / `providers trt-rtx status` | **No** |
| Benchmark harness (`BenchmarkTensorRtRtxBootstrap` with `allowProviderDownloads: true`) | Yes, only when license already accepted and bootstrap opts in |

Portable and per-user installs use the in-app or CLI install paths. A future MSI/EXE/MSIX wizard should call the same `ITrtRtxEpInstaller` after license acceptance (see [packaging/installer/README.md](../../packaging/installer/README.md)).

## Registration flow

`TensorRtRtxPluginService` owns plugin registration:

1. Verify NVIDIA hardware eligibility (Windows registry probe or Linux `/proc/driver/nvidia/gpus`).
2. Resolve and validate the plugin bundle directory.
3. Call `OrtEnv.Instance().RegisterExecutionProviderLibrary(...)` for `onnxruntime_providers_nv_tensorrt_rtx.dll`.
4. Enumerate `OrtEnv.Instance().GetEpDevices()`.
5. Require a GPU device named `NvTensorRTRTXExecutionProvider`.
6. Session creation appends that device through `SessionOptions.AppendExecutionProvider(env, devices, options)`.

TRT RTX must not call Windows ML `ExecutionProvider.TryRegister`, `EnsureAndRegisterCertifiedAsync`, or `SessionOptions.SetEpSelectionPolicy`.

## Runtime cache

Provider binaries and runtime cache are separate.

Provider bundle (installed by Model Manager or `Fetch-TrtRtxEp.ps1`):

```text
%LOCALAPPDATA%\Trackdub\Providers\trt-rtx\0.5.0\cu13\win-x64\   # Windows
~/.local/share/Trackdub/Providers/trt-rtx/0.5.0/cu13/linux-x64/   # Linux
```

> Engine caches: TRT-RTX runtime caches live under `EngineCache/trt-rtx/<BundledFingerprintVersion>`, and classic TensorRT engines under `EngineCache/classic-trt/<fingerprint>` (no longer nested inside the TRT-RTX path). Caches written by earlier layouts (flat `EngineCache/NvTensorRTRTX*`, or `EngineCache/trt-rtx/<old-fingerprint>/classic-trt/`) are orphaned on upgrade; the next run rebuilds engines once. Run `trackdub cache clear engines` to reclaim the space.

Manifest: `runtime/trt-rtx-ep.manifest.json` (pinned NVIDIA GitHub release URLs + checksums).

### DirectML interop and the engine-build limit (Windows)

TensorRT-RTX inference launched after a DirectML dispatch in the same process fails with
`CUDA_ERROR_INVALID_VALUE` (`cuKernelSetAttribute returned error invalid argument`), which ONNX
Runtime reports as `ShapeInferenceNotRegistered`. Trackdub mixes DirectML and TensorRT-RTX stages in one
process, so every Windows engine is built with `nv_max_shared_mem_size = 101376` (99 KiB), NVIDIA's
documented workaround (TensorRT-RTX 1.7 release notes, known issues).

- NVIDIA scopes the issue to "CUDA driver R615 or earlier", but it reproduced on driver 621.14
  (RTX 5070, EP ABI 0.5.0, TensorRT-RTX 1.7.1), so the limit is not gated on the driver version.
- The limit is part of `BundledFingerprintVersion` on Windows (`...+smem99k`). Engines and runtime-cache
  kernels built without it keep failing after the limit is added, so the fingerprint change moves the
  runtime cache directory and invalidates smoke verdicts and EP-context stamps.
- A smaller caller-supplied `nv_max_shared_mem_size` is kept. Zero, larger, or unparsable values are
  clamped to the limit.
- The effect on kernel performance has not been measured.

### CUDA graphs (on by default)

TensorRT-RTX sessions capture CUDA graphs by default (`OnnxExecutionSessionFactory.DefaultEnableCudaGraph`).
The sources disagree on what that requires. The TensorRT-RTX runtime docs ("Working with RTX CUDA Graphs")
say native capture recaptures when input sizes or buffer locations change and falls back to normal
execution for graphs it cannot capture. The ONNX Runtime TensorRT-RTX EP docs say to avoid or be careful with
`enable_cuda_graph` when input shapes or device bindings change frequently, and that data-dependent control
flow is a poor fit. What has been checked here is output equality, not the mechanism: see the measurements below.

- Opt a model out by putting `enable_cuda_graph = 0` in that call site's TRT options (a caller profile
  wins over the default). The Qwen3-ASR encoder is opted out because its mel length changes almost every
  chunk, so capture would record a graph per length. The Nemotron/Parakeet runtime options used to force
  `0`; they now follow the default.
- Variable-length inputs (Parakeet and Nemotron encoders) can still create many captured graphs on long
  files. Captured graphs use device memory. This has not been measured.
- `TRACKDUB_TRT_RTX_CUDA_GRAPHS` overrides every call site and caller profile for diagnosis and A/B
  runs: `on`, `1` or `true` force capture on; `off`, `0` or `false` force it off; any other value leaves the
  call site's choice alone. Set it before the process starts; it is read each time options are built,
  including for session-pool keys.
- Measured on an RTX 5070 with graphs forced on, on one short clip: identical stage output for Whisper (tiny,
  base, small, medium), Parakeet, Sortformer and Spleeter. Nemotron and CosyVoice were not run.
- Per-graph `Trackdub.Benchmarks --cuda-graphs compare` runs (device-bound I/O, source graphs, 50 runs) showed
  57-87% lower median latency on most single-graph models. Those runs use persistent device-bound buffers, the
  condition the benchmark section above names for graphs, so they are a ceiling. The gain with managed I/O,
  which most stages use, was not measured.
- EP-context artifacts are valid whichever capture mode loads them: `BuildCompileOptionsIdentity` leaves
  `enable_cuda_graph` out, because capture is a runtime session setting rather than an engine-build option.
  Compile and warmup sessions strip capture from the model's profile (`ForEngineBuild`), so they run with
  capture off even for MADLAD, whose runtime profile turns it on.
- A call site that decides per session (Whisper turns capture on only for its fixed-address bound path)
  writes `enable_cuda_graph` explicitly with `WithCudaGraphs`, so its decision wins over the default.

Compiled TRT RTX runtime cache:

```text
%LOCALAPPDATA%\Trackdub\EngineCache\
```

Clear the engine cache after a GPU driver change, GPU swap, or TRT RTX EP version bump when inference fails with stale compiled engines:

```powershell
trackdub cache clear engines
trackdub doctor   # shows cache path and approximate size
```

Override cache roots:

```powershell
$env:TRACKDUB_ENGINE_CACHE_ROOT = "D:\TrackdubEngineCache"
$env:TRACKDUB_CACHE_ROOT = "D:\TrackdubCache"
```

`TRACKDUB_ENGINE_CACHE_ROOT` wins. Otherwise `TRACKDUB_CACHE_ROOT\EngineCache` is used. Otherwise Trackdub falls back to `%LOCALAPPDATA%\Trackdub\EngineCache`.

### EP-context artifacts

The runtime cache keeps JIT kernels, not engines, so each new process would otherwise rebuild every
engine from the source graph. An EP-context artifact (`<model>.epc.onnx`, its `<model>.epc.stamp.json`,
and for large models the serialized `*.engine` files it references) sits next to the source model.
TRT RTX sessions load it instead of the source whenever its stamp still matches the model bytes
(including `<model>.onnx.data` / `<model>.onnx_data` external weights), GPU architecture, driver,
and EP/runtime version.

Artifacts are produced two ways:

- **Captured on first build.** When a TRT RTX session is created for a graph of at least 256 MB
  (model plus external weights) without a current artifact, the session builds with EP-context
  generation on and publishes the artifact afterwards. The capture rides on the build the session
  performs anyway; it only adds writing the engine. Capture is skipped when the volume lacks room
  for twice the model bytes, and a capture failure retries the session without capture.
- **Ahead of time** with `trackdub cache warm` (or the desktop Settings warm action).

Engines above protobuf's 2 GB limit are written as separate `*.engine` files whose names and sizes
the stamp records; a missing or truncated engine invalidates the artifact. Measured on MADLAD-400 3B
fp16 (RTX 5070): the encoder session builds from source in ~48 s and loads its artifact in 5-13 s
(3.36 GB engine; the low end with the file in the OS cache). Artifacts roughly double the model's
disk footprint.

## Readiness states

Keep these states separate:

- Plugin directory resolved
- All required DLLs present
- Plugin registered with ORT
- `NvTensorRTRTXExecutionProvider` GPU device visible
- Model files downloaded and checksum verified
- Model/provider pair smoke-tested
- Pipeline stage ran and produced usable artifacts

Provider registration alone is not model readiness. A failed TRT RTX plugin route may fall back to DirectML, but the selected provider must be reported as DirectML, not TRT RTX.

## CLI / SDK execution-provider pins (soft prefer)

`--execution-provider trt-rtx` (and other Kind pins) is a **soft preference**: the planner tries TRT RTX first among stage/engine allow-lists, then falls through (for example DirectML → CPU) when the engine family forbids TRT, smoke fails, or session init fails on unsupported ops.

Hard pin (collapse allow-list / block on smoke-init failure when TRT is allowed):

```powershell
trackdub dub ... --execution-provider trt-rtx --require-execution-provider
```

SDK equivalent: `WithExecutionProvider(ExecutionProviderKind.TensorRTRtx)` soft-prefers; pass `require: true` for a hard pin.

Engine-family allow-lists deny TensorRT families for graphs that hard-fail session init under TRT RTX (examples: `whisper-onnx`, `opus-mt`, `chatterbox`, `qwen3-tts`, `latentsync-diffusion`). Those stages still run under a global `trt-rtx` soft prefer by selecting DirectML/CPU. CosyVoice is intentionally **not** denied: the per-graph `TrtRtxUnsupportedOpScanner` plus session-init fallback isolate its failures, and the TTS smoke proves all nine graph sessions before a hard pin trusts them. Do **not** treat this as hybrid VRAM spillover / `supports_partial_offload`; Trackdub does not claim partial offload for TRT RTX.

**ORT GenAI loads are excluded from TensorRT entirely** (`whisper-genai`, `phi-genai`, `qwen-instruct` engine-family overrides). ORT GenAI's `NvTensorRtRtx` device can terminate the host process with a native stack overflow during model init/generation (observed on `qwen-instruct`, Qwen2.5-1.5B). A fatal crash cannot surface as a catchable smoke failure, so the planner never offers TensorRT to GenAI-loaded families and the smoke tester refuses the combination before touching native code.

Session create also retries once with DirectML (Windows) then CPU when TensorRT RTX was selected and init fails with EP/kernel/importer errors (`Kernel not found`, `ModelImporter`, `No graph will run on TensorRT`, etc.), and records that path in `FallbackReason` / bootstrap detail. Pass `--require-execution-provider` / `allowTrtInitFallback: false` to disable that retry on hard-pin routes.

This TRT session-init retry applies to **single-session** factory paths (`CreateSingleAsync` / `CreatePooledSingleAsync`). Multi-session factories (Whisper/Opus/Qwen3-ASR/LatentSync pools) still rely on planner engine-family allow-lists and soft-prefer smoke fallthrough; they do not re-run TRT→DirectML init fallback per role in this pass.

## Quieter TRT capability probe logs

ORT may emit per-node `SkipLayerNormalization` / unsupported-op ERROR spam while probing TRT RTX capability. That spam is ORT/NVIDIA noise, not an install failure. To reduce stderr noise without hiding real session failures:

```powershell
$env:ORT_LOG_SEVERITY_LEVEL = "3"   # 0=Verbose .. 4=Fatal; 3=Error still shows failures, raise to 4 if needed
```

Trackdub does not rewrite ORT's default log severity in session bootstrap.

## Smoke commands

Catalog sweep (planner-style smoke over every cached bundled GPU target; prints
`PASS`/`FAIL`/`SKIP` per target as each completes so a native crash mid-catalog
does not lose earlier results):

```powershell
trackdub providers trt-rtx smoke
```

Readiness/probe slices:

```powershell
dotnet test tests/Trackdub.Inference.Tests --filter "FullyQualifiedName~TensorRtRtxPluginLocator|FullyQualifiedName~OnnxExecutionSessionFactory" --no-restore -m:1
```

Benchmark smoke on a Windows NVIDIA RTX machine with the plugin bundle available:

```powershell
$env:TRACKDUB_TRT_RTX_EP_DIR = "$env:LOCALAPPDATA\Trackdub\Providers\trt-rtx\0.5.0\cu13\win-x64"
dotnet run --project src/Trackdub.Benchmarks.DevHost -f net10.0-windows10.0.19041.0 -- --model <model-id> --provider trt-rtx --runs 1 --format console
```

For explicit session testing, inspect the selected provider in benchmark output. Do not infer success from plugin registration logs alone.

## Benchmark / DubBench bootstrap

`Trackdub.Benchmarks` and DubBench share `BenchmarkOnnxExecutionBootstrap`:

- **WinML registry:** `ConfigureExecution(...)` before ONNX runs (same as `Trackdub.Benchmarks Program.cs`).
- **TRT RTX runner factory:** `CreateOnnxRunner()` wires `BenchmarkTensorRtRtxBootstrap` (plugin directory providers + optional license-gated bundle ensure).

Readiness paths (pick one):

1. **Model Manager** — Install after `NvidiaTensorRtRtxLicenseAccepted` in studio settings.
2. **Dev/CI fetch** — `tools/dev/Fetch-TrtRtxEp.ps1` (sets `TRACKDUB_TRT_RTX_EP_DIR`).
3. **Auto-download (opt-in)** — when `NvidiaTensorRtRtxLicenseAccepted` is true in `%LOCALAPPDATA%\Trackdub\settings.json` and benchmark/bootstrap allows provider downloads.

Headless operators:

```powershell
trackdub providers trt-rtx status
trackdub providers trt-rtx install --accept-license
trackdub doctor   # includes tensorrt-rtx-plugin probe row (no download)
```

DubBench ONNX runs call the same bootstrap before each benchmark invocation.

## Benchmark fidelity flags

`Trackdub.Benchmarks --model <path>` accepts a sampling and timing policy beyond the
legacy `--runs <n>`:

| Flag | Effect |
|------|--------|
| `--runs <n>` | Measured pass target (default policy also applies duration/warmup floors) |
| `--duration-ms <n>` | Minimum measured wall time; the sampler keeps collecting past `n` runs until reached |
| `--warmup-runs <n>` / `--warmup-ms <n>` | Warmup floor by count and/or wall time; warmup samples are excluded from statistics |
| `--input-mode synthetic\|zeros\|profile` | Deterministic seeded synthetic inputs (default), the legacy all-zeros recipe, or `profile` = feed inputs at the model family's registered `trt_profile_opt_shapes` (nemotron encoder, sortformer, MADLAD); `--input-seed <n>` fixes the synthetic stream |
| `--input-shape <name>=<d1,d2,...>` | Repeatable per-input shape override applied on top of the selected input mode; pins dynamic dims to the shape a baked engine profile expects (e.g. `--input-shape chunk=1,3040,128`) |
| `--io-mode managed\|device-bound` | Managed `session.Run` end-to-end timing vs device-bound `OrtIoBinding` timing |
| `--cuda-graphs auto\|off\|on\|compare` | `enable_cuda_graph` for the TRT-RTX EP; `auto` (default) enables graphs when the run is device-bound on a single-graph trt-rtx session; `compare` measures both lanes in fresh sessions |
| `--artifact auto\|source\|require` | EP-context artifact selection (`auto` = production resolver, `source` = force source model, `require` = fail without a compatible artifact) |
| `--runtime-cache reuse\|isolated` | `nv_runtime_cache_path` reuse vs an isolated per-run cache directory (`--runtime-cache-dir`). Isolated dirs (`benchmark-runtime-cache-<guid>` next to the report) are retained after the run — they hold the TRT engine cache evidence for that measurement; delete them manually when no longer needed |
| `--reference-tool tensorrt-rtx\|ort-perf-test` | Optional NVIDIA reference-tool run attached to the report (`--reference-tool-path`, `--reference-tool-required`) |
| `--history` | Persist the run into the user benchmark history SQLite store |
| `--format console\|json\|both` | Console summary, JSON report, or both |

### Device-bound I/O

`--io-mode device-bound` stages inputs in pinned host tensors, copies them per pass into
bound device OrtValues with `OrtEnv.CopyTensors`, binds outputs to session-reported
memory, and fences device work with `SynchronizeBoundOutputs`. Explicit device staging is
required because `SynchronizeBoundInputs` is a no-op for host-bound OrtValues on TRT-RTX
(the engine runs on its own fixed device buffers). The measured pass therefore covers the
real host-to-device copy plus device execution, and reports
`TimingMode: DeviceBoundSynchronized` in JSON metadata.

Prerequisites and limits:

- Explicit `--provider trt-rtx` only; other routes throw rather than silently fall back
  to managed timing.
- Single-graph models only in this build; whisper/opus encoder-decoder pipelines keep
  managed I/O and fail fast if device-bound is requested.
- Requires the session to report device-resident inputs (`GetMemoryInfosForInputs`);
  host-only sessions fail with an explicit reason.
- Inputs must be dense float/fp16/int32/int64 tensors; string or non-tensor inputs are
  unsupported.

### CUDA graphs

`--cuda-graphs on|compare` requires `--io-mode device-bound` (persistent bound buffers
are the same precondition production applies for `enable_cuda_graph=1`); settings
validation rejects the combination with managed I/O. The `auto` default enables graphs
only when the run is already device-bound on a single-graph trt-rtx session and records
the resolution in report notes; on managed runs it stays off. `compare` runs a graphs-off lane
then a graphs-on lane, each in a fresh session so `enable_cuda_graph` is honestly
applied, and records per-lane sampling/statistics under `metadata.cudaGraphLanes` in
the JSON report plus a `CUDA graph comparison` block in console output.

### Reference tool

`--reference-tool ort-perf-test` invokes a resolved `onnxruntime_perf_test` binary and
`--reference-tool tensorrt-rtx` invokes `tensorrt_rtx`. Every intended argument is
checked against the tool's live `--help` output before invocation; missing tools or
unsupported flags report `NotAvailable`/`UnsupportedVersion` in
`metadata.referenceToolResults` instead of failing the primary benchmark. With
`--reference-tool-required` the command exits nonzero when the tool cannot run.
Reference metrics keep their native timing domain (native-tool numbers are not merged
into ORT measurements) and stdout/stderr are captured to bounded log files whose
identities are recorded in the report.

### Report metadata

JSON reports carry `metadata.environment` (GPU/driver/OS/ORT/plugin identity),
`metadata.graphs[]` (model SHA-256, artifact identity, effective provider options,
cache observations per phase), `metadata.sampling`, `metadata.configuration.timingMode`,
and a `metadata.comparisonKey` fingerprint so runs from different machines can be
compared honestly.

## Bumping TRT RTX EP release

When NVIDIA ships a new `TensorRT-RTX-EP-ABI` GitHub release:

1. Check which platforms the tag actually ships (v0.4.2 was Windows-only; v0.5.0 ships both). Run `tools/dev/Update-TrtRtxEpManifest.ps1 -Version <x.y.z> -TrtRtxRuntimeVersion <a.b.c> [-LinuxVersion <older tag>] [-CudaVariant cu13]` to refresh `runtime/trt-rtx-ep.manifest.json` (URLs, SHA-256, size, per-package `version` when a platform lags).
2. Inspect the archive: update `TensorRtRtxProviderConstants.BundledVersionWindows` / `BundledVersionLinux`, `BundledCudaVariant`, `BundledTrtRtxRuntimeVersion`, and the `tensorrt_rtx_<major>_<minor>.dll` / parser file names (plus `TrtRtxEpRequiredFiles` and the required-file list in `tools/dev/Fetch-TrtRtxEp.ps1`). `TrtRtxBundlePinTests` fails if constants and manifest drift. Check the plugin's imports (`dumpbin /dependents`): the cu13 Windows bundle links the CUDA runtime statically, the Linux bundle ships `libcudart.so.13`; adjust `TensorRtRtxCudaRuntimeBootstrap` if that changes.
   `BundledFingerprintVersion` (EP ABI version + TRT-RTX runtime version) keys smoke verdicts and EP-context stamps, so either bump invalidates them. It also names the TRT-RTX runtime cache directory (`EngineCache/trt-rtx/<fingerprint>`), so runtime caches from an older lineage are never fed to the new runtime.
3. Run `tools/dev/Fetch-TrtRtxEp.ps1` locally and verify `trackdub providers trt-rtx status`.
4. Run optional GPU smoke (`.github/workflows/trt-rtx-smoke.yml`) or `Trackdub.Benchmarks.DevHost --provider trt-rtx`.
5. Run `trackdub doctor` and advise users with stale engines to `trackdub cache clear engines` after upgrading the EP bundle.

## CI optional GPU tier

Default CI (`ci.yml`) stays unit/fake-backed. Optional smoke workflow: `.github/workflows/trt-rtx-smoke.yml`.

| Gate | Meaning |
|------|---------|
| Repository variable `TRACKDUB_TRT_RTX_SMOKE=true` | Enables the workflow job on self-hosted Windows with NVIDIA RTX |
| `TRACKDUB_TRT_RTX_EP_DIR` | Plugin directory after fetch (workflow sets from default install root) |
| `TRACKDUB_TRT_RTX_SMOKE=1` | Test attribute gate for optional integration tests (`RequiresTrtRtxFactAttribute`) |

The smoke job runs `Fetch-TrtRtxEp.ps1`, exports `TRACKDUB_TRT_RTX_EP_DIR`, then one `Trackdub.Benchmarks.DevHost --provider trt-rtx` invocation. A nonzero smoke exit fails the job.

## References

- [ONNX Runtime TensorRT RTX EP](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html)
- [ONNX Runtime plugin EP usage](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html)
- [ADR-0002 Windows ML provider strategy](../decisions/ADR-0002-windows-ml-provider-strategy.md)
