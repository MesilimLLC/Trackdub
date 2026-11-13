# ADR-0016: Scoped end-user Python exception for the supervised inference worker

- Status: Accepted
- Date: 2026-10-06
- Decided by: Tony, 2026-10-06 (recorded in PR #401, following pitch PR #394; `docs/archive/rust-inference-sidecar-pitch.md`)

## Context

AGENTS.md forbids end-user runtime dependencies on Python, Conda, Docker, and
the CUDA Toolkit. That rule exists to protect the single-installer,
local-first distribution story — and it stands for the application itself.

But frontier voice models (Chatterbox, CosyVoice and successors) publish
PyTorch `.pt` weights first, sometimes only. The ONNX export tax — fragile
graph conversions, `If`/`Squeeze` graphs that won't lower, reshape profile
fights, weeks waiting on community conversions — is a recurring cost
documented in the TTS stall investigations (`.freebuff/c13-*`). Decision
2026-10-06: `.pt`-only models are inevitable, so Python returns in exactly one
place, under supervision, per the design in pitch Section 5b.

Related: ADR-0006 (Chatterbox commercial-use verification — license posture
for the likeliest first worker model), pitch PR #394.

## Decision

A single scoped exception to the no-end-user-Python rule:

1. **Exactly one Python runtime may ship**, and only as a worker behind the
   sidecar IPC contract — never as an application dependency, build
   requirement, or ad-hoc subprocess (no `espeak-ng`-style one-offs through
   this door; Decision 3 of ADR-0005 stays as-is).
2. **Rust supervises, Python computes, C# orchestrates.** The supervisor owns
   the bundled pinned interpreter (`uv python` or equivalent — never system
   Python), the dependency lockfile, process lifecycle, readiness gates,
   crash-backoff restarts, and version-stamp refusal. Python code loads
   weights and runs inference, nothing else.
3. **Same contract or nothing.** The worker speaks the typed sidecar protocol
   (`{dtype, shape, data}` envelopes, planner-approved load plans mirroring
   `StageRuntimePlan`) verbatim with the Rust ONNX path. Executors stay
   interchangeable; no caller changes per executor.
4. **No silent runtime.** Worker presence, interpreter version, model-loaded
   state, and settled accelerator are stage evidence, surfaced truthfully in
   the UI — the honest-readiness rule extends to the worker, it is not waived
   for it.
5. **Proven in the harness first.** No worker model ships until
   `controlled-matrix --executor sidecar` carries first-use latency,
   warm-call overhead, and VRAM-held evidence for it, per repo benchmark
   policy. Claims without evidence are rejected the same as in-process ones.

## Consequences

### Positive

- `.pt`-only frontier voice models become shippable in days, not
  after-export-if-ever.
- All Python pain (interpreter pinning, CUDA matching, dependency drift,
  crash containment) has exactly one owner: the supervisor.
- The application keeps its no-Python posture; the exception is bounded,
  versioned, and revocable.

### Negative

- Installer grows: pinned interpreter + torch + CUDA-matched libs + provider
  shared libs alongside the C# app's ORT copy. Distribution size and the
  cross-process ORT pin (pitch Q6) are open spike measurements.
- Second binary widens installer signing/notarization surface (pitch Q3).
- A torch/CUDA security advisory now has a response path we must own —
  lockfile bump plus version-stamp invalidation.

### Neutral

- No `src/` changes ship under this ADR. It authorizes the spike
  (`docs/archive/rust-sidecar-spike-plan.md`); promotion to `src/` needs its
  own review with harness evidence attached.

## Alternatives considered

### Continued ONNX-only

Rejected 2026-10-06: export coverage for frontier voice models is not
converging fast enough, and each export is bespoke validation work.

### Unsupervised Python daemon / system Python

Rejected: reintroduces every failure mode the AGENTS.md rule was written to
prevent (version drift, CUDA mismatch, silent breakage). Supervision is the
condition of the exception, not an implementation detail.

### Full Rust rewrite of inference

Rejected: see PR #394 discussion — the working pipeline, session pool, and
harness stay in C#; Rust owns supervised processes behind the IPC contract,
never pipeline state or UI.

## Revocation criteria

Revisit (narrow or revoke this exception) when one or more is true:

1. ONNX export coverage for shippable voice models converges (no `.pt`-only
   gap for two consecutive model evaluations).
2. The worker's measured distribution or maintenance cost exceeds its model
   coverage value for two consecutive releases.
3. A Python-supply-chain incident makes the lockfile posture untenable and no
   mitigation lands within one release.
