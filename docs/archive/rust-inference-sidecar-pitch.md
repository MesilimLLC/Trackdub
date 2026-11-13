# Rust-supervised inference sidecar: build plan (decided, not proposed)

- Status: **Decided 2026-10-06 (Tony), authorized by ADR-0016, spiking per
  `rust-sidecar-spike-plan.md`. This doc is the plan of record — it describes
  what we are building, not what we might build.**
- Date: 2026-10-06 (rewritten 2026-10-07 to remove the hedging)
- Author: Hermes Agent (drafted with Tony)
- Scope: `docs/plans/` + `tools/rust-sidecar-spike/` sketches only. Nothing in
  `src/` changes, nothing joins any `.slnx`, CI behavior is unaffected.

## 1. Problem

Two pressures are colliding:

1. **Frontier voice models are PyTorch-only.** Chatterbox, CosyVoice, and
   friends publish `.pt` weights first (sometimes only). Our ONNX-first pipeline
   pays the export tax every time: fragile graph conversions, `If`/`Squeeze`
   graphs that won't lower (Silero on TRT-RTX), `-1` reshape profile fights
   (SortFormer), and the TTS stall investigations in `.freebuff/c13-*`.
2. **Python is the worst vehicle for shipping them.** A per-inference
   `python` spawn costs seconds (interpreter + torch import + CUDA context +
   weight load). A resident daemon fixes the per-call cost but still ships a
   multi-GB Python + torch + CUDA-matched runtime to every user — directly
   against the AGENTS.md rule: *no end-user Python, Conda, Docker, or CUDA
   Toolkit dependency* — and breaks the single-installer story on macOS/Linux.

So: we are building a sidecar story that keeps warm calls in the milliseconds,
ships the Rust supervisor as a small native binary with colocated provider
libs, and contains the one Python runtime ADR-0016 allows under that
supervisor instead of pretending we can avoid it.

## 2. Options considered (kept for the record — decision made)

| | Python resident daemon | Go sidecar | TypeScript sidecar | **Rust sidecar (chosen)** |
|---|---|---|---|---|
| Cold start | ~1–5 s first load, ms after (resident) | ms (static binary) | 100s of ms + runtime | ms process start; first inference unmeasured (spike measures) |
| Ships as | 2–4 GB env, CUDA-matched | single binary | needs Node/Bun + native addons | small binary + colocated ORT/provider libs (GPU EPs ship as shared libs, not static) |
| Runs `.pt`-only models | **yes — the only one** | no | no | no (ONNX only) |
| ORT / GPU story | torch-direct, good | cgo → ORT C API (thin, painful) | `onnxruntime-node`, no TRT-RTX worth having | `ort` crate — first-class ORT bindings |
| ORT copies / version skew | n/a (torch world) | second ORT via cgo | second ORT via node addon | **second ORT copy** — must pin the same ORT version as the C# app or engine/EP-context caches diverge (see §6.6) |
| Fits AGENTS.md dist rules | no (scoped exception granted, ADR-0016) | yes | no (new runtime) | **yes** |
| Ecosystem for audio/ML glue | best | thin | thin | good (ndarray, hound, safetensors) |

TypeScript was the weakest option: a whole JS runtime without solving GPU
execution or `.pt` coverage. Go is a fine *supervisor* but its ML bindings are
cgo wrappers around the same ORT C API our C# already drives better. Neither
runs PyTorch-native models, so neither removes the export tax. Rust won on
elimination; Python stays because only Python runs the models.

## 3. What we are building (both phases committed, sequenced)

**Build the sidecar seam once, fill it with Rust first, then put the
Rust-supervised Python worker behind the same contract.**

- **Phase 1 — Rust ONNX sidecar spike** (`tools/rust-sidecar-spike/`,
  two-week box per `rust-sidecar-spike-plan.md`). Its value is the seam
  itself, stated plainly: it proves the out-of-process contract Phase 2 needs
  (C# stage → IPC contract → sidecar lifecycle: start, health, model-ready,
  infer, shutdown) and measures its tax. It does NOT run `.pt`-only models
  and does NOT beat in-process ORT on raw latency — ONNX models already run
  resident in-process, so Phase 1's benchmark is sidecar-vs-in-process on the
  same graph, and success means IPC overhead within budget, not a speedup.
- **Phase 2 — Rust-supervised Python worker** (starts at Phase 1 go/no-go,
  not before). Implements the *same* IPC contract, used for `.pt`-only
  frontier voice models where no usable ONNX exists. Authorized by ADR-0016's
  scoped AGENTS.md exception; the pain is contained by the supervision design
  in Section 5b, not by opt-in alone.
- **What stays:** C# + in-process ONNX Runtime remains the default local path
  (Whisper, Silero, SortFormer, Kokoro, translation). The sidecar never becomes
  the only way to run anything shippable today.

This preserves everything the product promises: honest readiness
(`process alive` != `model loaded` != `accelerator ready`), resumable stages
with durable artifacts, EP fallback explained truthfully, and no silent new
runtime dependency.

## 4. IPC contract

JSON-lines over stdin/stdout for the spike (zero dependencies); graduate to
gRPC or named pipes when the spike's transport measurement says so (see
§6.1). The C# side sees only this (full sketches in
`tools/rust-sidecar-spike/`):

```jsonc
// C# → sidecar
{ "id": "req-1", "op": "load", "plan": { "model": "models/kokoro-onnx/model.onnx",
  "providers": ["TensorRTRTX", "DirectML", "CPU"], "requirePreferred": false } }
{ "id": "req-2", "op": "infer", "inputs": { "input_ids":
  { "dtype": "int64", "shape": [1, 3], "data": "<base64>" } } }
{ "id": "req-3", "op": "health" }

// sidecar → C#
{ "id": "req-1", "status": "loaded", "activeProvider": "DirectML",
  "loadMs": 1840 }
{ "id": "req-2", "status": "ok", "outputs": { "audio": "<base64 pcm>" },
  "inferMs": 61 }
```

Tensors are typed on the wire (`{ dtype, shape, data }` with base64 payload)
— the same envelope as the proto sketch's `TensorMeta`, so the spike exercises
the real contract and the gRPC graduation is a transport swap, not a redesign.
Base64 costs ~33% on payloads (notably PCM outputs); the spike measures it
before we commit to a transport.

The `load` payload is a planner-approved plan (integrity-qualified path plus
the planner's ordered provider fallback and hard-pin flag), mirroring
`StageRuntimePlan` — the sidecar executes it and never selects models or
providers itself, so `IRuntimePlanner` gates and
`RequirePreferredExecutionProvider` cannot be bypassed or silently violated.

Fallback transparency is the point: the sidecar reports which provider it
*actually* settled on, and C# surfaces that in stage evidence — never "GPU
ready" when the answer was CPU.

## 5. Cold-start story (estimates, to be measured in the spike)

Terminology, kept honest: "process start" (ms-scale for native binaries) is NOT
"first inference" — weight load plus EP/CUDA init dominates first use (the
`loadMs: 1840` example is illustrative, and the real number is an explicit
spike measurement, not a decision criterion).

| Cost | In-process ORT today (Phase 1 baseline) | Rust sidecar (resident) | Resident Python worker (Phase 2 target) |
|---|---|---|---|
| Process start | none (in-process) | ~5–20 ms, once | ~1–5 s interpreter+torch, once |
| Weight load | once per session pool | once; mmap'd ONNX | once per worker |
| EP/CUDA context | once per session pool | once, kept resident | once per worker, kept resident |
| Warm infer call | inference only | inference + IPC + base64 tax | inference + IPC + base64 tax |

Two different stories, kept separate: Phase 1 is expected to be *slightly slower*
than in-process ORT on warm calls (IPC + base64) — its deliverable is the proven
seam plus a measured overhead number. The cold-start win belongs to Phase 2,
where a resident Python collapses per-call interpreter/torch/CUDA costs into a
one-time warmup at app start / first use — the same move our ORT engine cache
already makes, with the same invalidation discipline (stale engine → re-verify,
same fingerprint rules).

## 5b. Painless Python: Rust supervision design (committed, not optional)

Python doesn't get to be a pet — it gets a supervisor. The Rust binary owns
everything users hate about Python, and Python itself only runs models:

- **Bundled interpreter, pinned.** Ship a fixed CPython via `uv python`
  (or equivalent), never the system Python. Version stamp checked at startup;
  mismatch refuses to serve, same fingerprint discipline as engine caches.
- **Locked dependencies.** One lockfile for torch + model deps per worker
  release. No pip-at-runtime, no floating versions, no "works on my machine."
- **Resident worker, warmed once.** Supervisor spawns Python at app start /
  first use, preloads weights + CUDA context, then holds it warm. Per-call
  cost is IPC + inference — the cold hit happens once, where users forgive it.
- **Honest readiness, enforced by the supervisor.** `process alive` !=
  `interpreter up` != `weights loaded` != `accelerator ready` — each gated
  separately, surfaced in stage evidence verbatim. No "GPU ready" unless CUDA
  actually initialized.
- **Crash containment with backoff.** Python segfaults or OOMs, the supervisor
  restarts with backoff, marks the stage failed with the real reason, and the
  app itself never goes down. In-process native crashes can't offer this.
- **Kill and reclaim.** Unload a 6 GB voice model by killing the worker —
  clean memory/GPU reclamation, no app-heap fragmentation.
- **Same contract or nothing.** The worker speaks the Section 4 protocol
  (typed `{dtype, shape, data}` envelopes, planner-approved load plans) —
  shared verbatim with the Rust ONNX path, so executors stay interchangeable
  and the gRPC graduation covers both.
- **Telemetry from day one.** First-use latency, warm-call overhead, VRAM held
  per worker — measured through `controlled-matrix --executor sidecar`, per
  repo policy, before any claim ships.

What Rust does NOT do: run `.pt` models, touch the UI, or own pipeline state.
It supervises. Python computes. C# orchestrates. Three jobs, three owners.

## 6. Resolved questions (was: open questions — every one now has an owner)

1. **IPC transport: decided by measurement, not preference.** Spike exit
   criterion: JSON-lines stands unless the base64-tax measurement forces the
   gRPC graduation. Avalonia sandboxing check is a spike task, week one.
2. **Sidecar lifetime: dedicated `ISidecarHost` behind
   `Trackdub.Infrastructure`** (not the Composition root directly) — decided;
   the host owns spawn, health, version-stamp refusal, backoff restart, kill.
3. **Signing/notarization: measured in the spike** (exit criterion 4:
   distribution delta sized on all three OSes). If the second binary breaks
   the installer story, that is a go/no-go input with numbers attached.
4. **Do we need Phase 2: YES — decided 2026-10-06.** `.pt`-only models are
   inevitable. Struck as a question; the remaining work is execution.
5. **Benchmark harness: yes, `controlled-matrix --executor sidecar`** — a
   spike work item, not a question. No evidence, no promotion, per repo policy.
6. **ORT pinning: single-version rule.** Supervisor and app pin the same ORT;
   either side bumping invalidates engine/EP-context fingerprints on both
   (same stamp discipline as smoke verdicts). Spike writes the exact mechanism.

## 7. Execution checklist (was: decision requested — decisions are made)

- [x] `.pt`-only problem real enough to need the seam — **decided 2026-10-06:
  yes. Contain per §5b.**
- [x] Direction accepted — **Rust spike now, Rust-supervised Python worker
  next, under ADR-0016's scoped exception.**
- [ ] Phase 1 spike complete (two-week box, `tools/` only) — judged on
  seam-proven + IPC-overhead-measured + distribution-sized + failures-behave,
  not on speedup. Go/no-go review at the end with evidence attached.
- [ ] Phase 2 worker build (starts at Phase 1 go) — first `.pt` voice model
  behind the same contract, license re-verified per model (ADR-0006 pattern).
- [ ] `src/` promotion (separate review, harness evidence required) —
  `ISidecarHost`, stage integration, UI readiness surfacing.

## Files in this PR (all non-build, decision record + spike inputs)

- `docs/plans/rust-inference-sidecar-pitch.md` — this document
- `docs/decisions/ADR-0016-supervised-python-worker-exception.md` — the
  governance exception (mirrored in `decisions.md`)
- `docs/plans/rust-sidecar-spike-plan.md` — the two-week box with exit criteria
- `tools/rust-sidecar-spike/Cargo.toml` — dependency sketch (`ort`, `serde`)
- `tools/rust-sidecar-spike/src/main.rs` — load-once / serve-forever loop sketch
- `tools/rust-sidecar-spike/proto/inference.proto` — contract sketch for the
  gRPC graduation path
- `tools/rust-sidecar-spike/csharp/SidecarClient.sketch.cs` — C# caller sketch
  (`.sketch.cs`: deliberately never compiled)

Merge means "we are building this" — the spike starts on merge.
