# ONNX session-pool memory admission

ONNX session memory admission is enabled by default. Each accelerator device gets three-quarters
of the largest adapter's dedicated VRAM (clamped to 4096–16384 MiB), and a separate host-RAM
budget of a quarter of physical RAM (same clamp) is shared by CPU and DNNL sessions (plus
OpenVINO when CPU-proxy mode is enabled). When the size cannot be detected, each budget falls
back to 4096 MiB. Multi-graph models are admitted against the sum of their graph estimates.

## Per-graph estimate

A graph's estimate is `weights × provider factor + 128 MiB`, where *weights* is the `.onnx` file
plus every external-data sidecar its tensors reference:

- The pool reads the `location` of each external-data tensor (initializers, sparse initializers,
  and node attributes, including subgraphs) from the graph without loading the weights, and sums
  the sizes of the distinct files. `<name>.onnx.data`, `<name>.onnx_data`, and any other
  location inside the model's directory count. Locations that are rooted or leave the model's
  directory are ignored, matching ONNX Runtime's path validation.
- If the graph cannot be parsed, the adjacent `<name>.onnx.data` and `<name>.onnx_data` files
  count instead. A graph that parses and references no external data counts only the `.onnx`
  file, even when a stray `.data` file sits next to it.
- The referenced locations are cached per graph file (path, length, and last-write time).
  Sidecar sizes are re-read on every key build, so a replaced sidecar is re-measured. The
  pool key's content hash still covers the `.onnx` file only.
- Two graphs that reference the same sidecar file each count the whole file, because each
  ONNX Runtime session loads its own copy. On DirectML, qwen3-asr-0.6b's decoder-init and
  decoder-step graphs share one 2867 MiB sidecar, and the measured usage matches two copies.

| Provider | Factor | Basis |
|---|---|---|
| TensorRT RTX | 1.25× | Measured: MADLAD-400 3B `trt_rtx_mixed_fp16_fp32` encoder + decoder (6.55 GB of external weights) raised process GPU usage by ~8 GB on a 12 GB RTX 5070, about 1.2× weights. The compiled engines for that pair total 7.94 GB (1.21×). |
| DirectML | 1.25× | Measured: qwen3-asr-0.6b encoder + decoder-init + decoder-step (6445 MiB of per-session weights) raised process GPU usage by 6526 MiB on a 12 GB RTX 5070 under Windows ML 2.4.89 (ORT 1.27.1), about 1.0× weights. |
| CUDA | 1.5× | Measured once: Qwen3-TTS 1.7B CustomVoice (talker prefill and decode at 5400 MiB of external weights each, code predictor 420 MiB, vocoder 435 MiB; 11655 MiB in all) on a 24 GB NVIDIA L4 (g6.xlarge). Observed process GPU usage peaked at 15736 MiB, with earlier pipeline stages still resident; residency per graph was not logged. Summing the per-graph growth steps, attributed from session-creation timestamps, estimates about 14888 MiB (1.28× weights) if both talker graphs were resident together. That is an estimate, not a bound, from one clip on one GPU. Evidence: [MesilimLLC/Trackdub#677](https://github.com/MesilimLLC/Trackdub/issues/677). |
| All others (CPU, DNNL, TensorRT, OpenVINO, …) | 2× | Not measured; conservative allowance for weights, initialization and pre-packing copies, and activation slack. |

The pool key records the provider chosen before session creation, after any TensorRT RTX
fallback decided at that point, so the factor follows that provider. A pooled single-session
model can still fall back from TensorRT RTX to DirectML or CPU while its session is being
created, after the key is admitted, so those keys reserve at 2× unless the route is hard-pinned
(`RequirePreferredExecutionProvider`). Multi-graph bundles such as MADLAD create their sessions
without that fallback and keep 1.25×.

## Sessions that fall back while they are created

A pooled single-session key is admitted against the bucket of its pre-creation provider. When
TensorRT RTX init then falls back and the session is created on a provider in a different
bucket, the pool moves the session's accounting before publishing it:

- A fallback to DirectML stays on the same device, so its reservation is unchanged.
- A fallback to CPU releases the device reservation, then admits the session against the
  host-RAM budget with the same estimate, under the same rules as a direct CPU request. If
  the estimate exceeds the host budget, the session is discarded and the request fails, as a
  direct CPU request would (see [Raising the limits](#raising-the-limits)). Otherwise idle
  host-RAM sessions are evicted to fit, or the request waits for host RAM to be released,
  without holding any device reservation. The already-created session is charged against the
  host budget for the whole wait, so while it waits its real memory is never hidden from the
  budget (concurrent CPU fallbacks for distinct keys each count toward the host ceiling).
- Once published, the session counts against host RAM, is evicted only to make room in host
  RAM, and leaves the device budget free for GPU sessions. Later pool hits on the same key reuse
  it without admitting again.

The bucket follows the provider whose session options the fallback created the session with,
not the per-input device the session reports afterwards.

The estimate is admission sizing, not pool identity: two keys that differ only in their estimate
(for example after a sidecar is re-measured) share one pooled session, which keeps the
reservation it was admitted with.

Worked examples on a 12 GB RTX 5070, which reports 11943 MB of dedicated VRAM (default
accelerator budget 8957 MiB):

| Bundle | Weights | Estimate | Result |
|---|---|---|---|
| MADLAD `trt_rtx_mixed_fp16_fp32` on TensorRT RTX | 2549 + 3702 MiB | 3314 + 4755 = 8069 MiB | Admitted; it runs with ~800 MB of VRAM to spare. |
| Same files on CUDA (1.5×) | 2549 + 3702 MiB | 3951 + 5681 = 9632 MiB | Refused; MADLAD falls back to CPU. |
| MADLAD bundled `quantized` (inline weights) on DirectML | 1275 + 1782 MiB | 1721 + 2355 = 4076 MiB | Admitted. On CPU (2×) it needs a host budget of at least 6370 MiB. |
| qwen3-asr-0.6b on DirectML | 711 + 2867 + 2867 MiB | 1016 + 3711 + 3711 = 8438 MiB | Admitted; measured 6526 MiB. At 2× it needed 13274 MiB and failed the ASR stage. |

Before external-data sidecars were counted, the `trt_rtx_mixed_fp16_fp32` pair reserved about
260 MiB, so admission never refused or evicted anything for it.

On a 24 GB NVIDIA L4 (23034 MiB reported, default accelerator budget 19578 MiB), Qwen3-TTS 1.7B
CustomVoice on CUDA reserves 8228 + 8228 + 758 + 780 = 17994 MiB for its talker prefill, talker
decode, code predictor and vocoder graphs, which are admitted together, so the set fits. At 2×
the same set needed 23822 MiB and was refused with the fail-fast admission message.

## Raising the limits

These are safety limits, not a promise that every supported model fits on every machine. For
example, a single graph with about 1985 MiB of weights (in the `.onnx` file plus its sidecars)
already exceeds the 4096 MiB host floor at 2×. Larger graphs and multi-graph bundles need more
headroom. Increase the host limit only when the machine has enough available RAM:

```powershell
$env:TRACKDUB_SESSION_RAM_BUDGET_MB = "12288"
```

```bash
export TRACKDUB_SESSION_RAM_BUDGET_MB=12288
```

The value is in MiB and must be a positive integer. Invalid or unset values keep the scaled
default. The accelerator limit can be adjusted independently with
`TRACKDUB_SESSION_VRAM_BUDGET_MB`; it applies per device. Raising either limit permits more
resident sessions and can increase memory pressure or cause the operating system to terminate
the process if the actual workload exceeds physical memory.

## Process-isolated GPU observation

Reservations are estimates. On Windows and Linux the shared pool also accounts for this process's
*actual* dedicated GPU memory: the host registers the process-isolated reading (`gpuBytes`) as the
pool's observation source, and each GPU admission decision uses its adapter's measured footprint
when attribution is available, or the full process total as a conservative upper bound when it is
not. GPU memory the pool never reserved — driver contexts, runtime arenas, non-pooled consumers —
consumes the same per-device budget instead of being invisible to admission. Other platforms have
no source and keep reservation-only accounting.

| Platform | Source | Attribution |
|---|---|---|
| Windows | The process's `GPU Process Memory` performance-counter instances. | Per adapter LUID, mapped to pool devices. |
| Linux | NVML's per-process compute list (`nvmlDeviceGetComputeRunningProcesses_v3` in `libnvidia-ml.so.1`), entries for this process id summed across devices. | Process total only, so every device is charged the full total. |

### Linux (NVML)

NVML is injected by the NVIDIA container toolkit whenever a GPU is passed through, so the reader
needs no extra setup in the server image. Measured on an NVIDIA L4 with driver 595.112.07, Docker
29.9.3 and NVIDIA Container Toolkit 1.20.1:

- In a default container (its own PID namespace) NVML lists the process under its container PID,
  the number `Environment.ProcessId` returns, with accurate memory: 186 MiB for a bare CUDA context
  and 698 MiB after a further 512 MiB allocation, within 8 MiB of the device-level delta.
- Two concurrent containers each saw only their own process, so one container never reads another's
  footprint even when both are PID 1 in their own namespace.
- With `--pid=host` the process is listed under its host PID, also with accurate memory.

The reading is reported as unavailable, and the pool falls back to reservation-only accounting,
whenever it cannot be trusted to describe this process: NVML or its v3 export is missing; processes
are listed but none carries this process id (reports on NVIDIA's issue tracker and forum describe
drivers before 555 returning host process ids to containers, which only `--pid=host` makes
matchable); the process is listed with 0 bytes or with the memory marked not available (a forum
report describes 0 bytes for container processes on driver 555, which did not reproduce on 595);
or any device cannot be queried. Neither older-driver behavior has been verified here. The reason is surfaced
as the reader's `UnavailableReason`. Empty lists on every device mean the process holds no GPU
memory. A multi-GPU host is charged the process total on each device, which can refuse work an
adapter-aware reading would admit.

What the observation means in practice:

- A GPU bucket admits a request only when `max(resident reservations, observed GPU usage) +
  pending reservations + request estimate` fits its budget. A mapped free adapter can admit work
  while another adapter is over budget. Each blocked bucket waits until usage drains. Each observation-blocked pass evicts at most one idle entry and re-observes,
  so evicted memory gets a chance to drain before more cache is discarded. Passes that free
  nothing count toward a fail-fast: an empty bucket throws after ~5 s of continuous stall, while
  a bucket holding live-but-unevictable entries (held leases, pins, live externals) gets ~60 s
  for turnover before the same diagnostic fires — a lease held for a whole stage still ends in
  an error, not a hang. The caller's cancellation token remains an escape hatch throughout,
  exactly as for an exhausted reservation budget.
- In-flight creates hold a pending reservation but have not allocated yet, so the process reading
  may contain partially allocated memory: pending reservations are charged on top of the observed floor, keeping
  concurrent admissions from overshooting the device budget.
- When the reader attributes usage per adapter (Windows) and the host registered its
  device-to-LUID map, each accelerator device is charged exactly its own adapter's footprint:
  usage on other adapters never blocks it and no sibling subtraction is needed. Without a
  breakdown or a mapping for the deciding GPU, the pool uses the full process total as an
  upper bound. It never subtracts estimated sibling reservations from measured bytes: an
  overestimate could otherwise hide real usage on the deciding GPU. This fallback can block
  a free adapter while another adapter holds memory; admission remains bounded by the stall
  limits above. Per-adapter attribution avoids that restriction when available.
  Both `HeadlessDubbingHost` and SDK `TrackdubBuilder.Build` register this mapping from
  `IDeviceEnumerator` during construction and release their own mapping on disposal without
  clearing a newer host's registration.
- Host-RAM buckets (CPU, DNNL, and OpenVINO CPU-proxy) and OpenVINO NPU buckets are never
  charged with dedicated GPU observations.
- An unavailable reading — no GPU, a driver that does not publish the counter set, a failing probe — leaves admission exactly as it was.

Set `TRACKDUB_SESSION_PROCESS_GPU_ADMISSION` to `0`, `false`, `off`, or `disabled` to turn the
observation off explicitly; anything else, including unset, keeps it on.

```powershell
$env:TRACKDUB_SESSION_PROCESS_GPU_ADMISSION = "0"
```

The process reading is reported as the `gpuBytes` evidence metric as well; that reporting is
independent of admission. See [benchmark-evidence.md](../development/benchmark-evidence.md).

## TTS synthesis concurrency

TTS synthesizes segments in parallel. The effective degree of parallelism is
`min(configured, VRAM-derived bound)`:

- **Configured** — `ttsMaxConcurrency` in `%LOCALAPPDATA%\Trackdub\settings.json`
  (or `TtsMaxConcurrency` in `SdkSessionOptions`). Unset or non-positive keeps the
  historical default of 4; values above 8 are clamped to 8.
- **VRAM-derived bound** — computed from the dedicated memory of the device the TTS plan
  selects and the memory class of the preferred model (`NormalizedPreferredModelAlias`). The
  stage plans TTS with the same request synthesis uses. If the plan names no device, the
  device sessions bind is used: ORT device 0 for the provider (the first hardware adapter for
  DirectML, the first NVIDIA GPU for CUDA and TensorRT). Integrated GPUs count shared memory.
  If that device cannot be identified or reports no memory, the smallest GPU is assumed. If no
  per-device reading exists at all, the largest adapter's memory is used. One worker is
  assumed to need approximately 1 GB (small: Kokoro-82M), 4 GB (medium: CosyVoice-300M,
  Chatterbox, Qwen3-TTS-0.6B, F5) or 16 GB (large: Qwen3-TTS-1.7B) of accelerator memory,
  with each additional concurrent worker adding ~512 MB. The bound never exceeds the
  configured value and the cap never fails a run: a budget that cannot cover even one
  worker synthesizes one segment at a time.
  - When the preferred alias is not required, the planner may fall back to a model in
    another memory class. The bound does not cover that fallback model.
  - A run whose plan selects a CPU or DNNL provider is not bounded by VRAM. If placement
    cannot be resolved, only a required CPU or DNNL pin skips the bound, since a non-required
    pin can still fall back to an accelerator.
  - Runtime device fallback after an out-of-memory error can move sessions to another device
    after the bound is set; the bound is not recomputed mid-run.

A per-device reading applies even when the host-wide probe returns 0. Only when neither a
per-device reading nor the host-wide probe is available (CPU-only machines, probe failure)
is the VRAM bound skipped and the configured value applied unchanged. The effective value is
logged at the start of every TTS stage (`TTS parallelism: N (configured …, device VRAM … MB,
host max VRAM … MB, accelerator …, model …)`).

```json
{ "ttsMaxConcurrency": 4 }
```

