# Spike plan: Rust-supervised inference sidecar (authorized by ADR-0016)

- Status: Planned — starts on merge of the decision record (PR #401)
- Time-box: **two weeks**, then a go/no-go review with evidence attached.
  The box is the point: this spike proves the seam and measures the tax, it
  does not ship a worker.
- Scope: `tools/` only. No `src/` changes, no `.slnx` changes. Anything that
  wants to touch `src/` waits for the go review and its own PR. That includes
  the `controlled-matrix --executor sidecar` lane: the spike records evidence
  from its own `tools/` harness in the same evidence schema (exit criterion 2),
  and the `controlled-matrix` lane itself moves to the `src/` promotion step —
  ADR-0016 Decision 5 only requires harness evidence before a worker *ships*,
  and the lane touches the `ControlledBenchmarkCliBinder` seam AGENTS.md forbids
  widening without its own review.

## Exit criteria (all required for go)

1. **Seam proven:** C# spike harness (throwaway, `tools/`) drives a Rust
   supervisor through the full lifecycle — start → health → planner-approved
   load → infer → shutdown — against a tiny ONNX graph, with typed
   `{dtype, shape, data}` envelopes both directions.
2. **Tax measured:** sidecar-vs-in-process warm-call overhead on the same
   graph (IPC + base64), first-use latency split into process-start vs
   weight-load vs EP/CUDA-init, and peak VRAM/RAM held — recorded as
   `controlled-matrix --executor sidecar` evidence, not chat claims.
3. **Contract holds across two executors:** a stub Python worker answers the
   same protocol (health/load/infer envelopes) so no caller changes per
   executor. Stub may return canned tensors; it proves interchangeability,
   not torch integration.
4. **Distribution sized:** installer delta measured on Windows, macOS, and
   Linux — supervisor binary + colocated ORT/provider libs + pinned
   interpreter + locked torch set — plus whether the second binary signs and
   notarizes cleanly (pitch Q3), with the cross-process ORT pin and
   fingerprint story written down (pitch Q6 answered with numbers, not
   adjectives).
5. **Failure behavior demonstrated:** malformed request, worker crash, and
   version-stamp mismatch each produce the specified protocol error or
   refusal — never a hang, never a fake-ready.

## Work items

| # | Item | Owner | Done when |
|---|------|-------|-----------|
| 1 | Transport decision (JSON-lines vs gRPC vs named pipes) with base64-tax measurement | spike | pitch Q1 closed, rationale in plan |
| 2 | Rust supervisor skeleton: spawn, health, version-stamp check, backoff restart, kill | spike | exit criterion 1 + 5 (supervisor half) |
| 3 | ONNX load/infer path on a tiny graph (CPU first, one GPU EP second) | spike | exit criterion 1 + 2 |
| 4 | `tools/`-harness evidence in the shared schema (plus the `controlled-matrix` lane design, built at promotion) | spike | exit criterion 2, evidence files attached to go review |
| 5 | Stub Python worker speaking the same protocol | spike | exit criterion 3 |
| 6 | Distribution + signing measurement (pitch Q3/Q6) | spike | exit criterion 4 |
| 7 | Go/no-go review: merge, redirect, or kill the direction | Tony + reviewers | decision recorded, ADR-0016 updated if scope changes |

## Explicitly out of scope

- Real `.pt` model integration (Chatterbox/CosyVoice wiring is the *next*
  spike after go, with ADR-0006 license posture re-confirmed per model).
- `src/` wiring (`ISidecarHost`, Composition, stage integration).
- Transport graduation to gRPC unless JSON-lines fails a criterion above.
- Any claim in UI, docs, or investor material that the sidecar exists.

## First-worker note (for after go, not now)

Likeliest first real worker is a Chatterbox- or CosyVoice-class voice model —
but model choice waits for go plus per-model license re-verification
(ADR-0006 pattern: commercial-safe source or blocked). No model weights enter
the repo under this plan.
