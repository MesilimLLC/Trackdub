# Supervised sidecar workers

This directory holds the supervised-Python-worker exception per
[ADR-0016](../docs/decisions/ADR-0016-supervised-python-worker-exception.md):
the Rust process supervisor, the Chatterbox protocol worker, and the wire
contract they share.

- [`PROTOCOL.md`](PROTOCOL.md) — normative v1 wire contract (JSON-lines,
  `{dtype, shape, data}` tensor envelopes, closed status/reason vocabularies).
- [`supervisor/`](supervisor/) — Rust supervisor crate: spawn, health-gate,
  version-stamp refusal, capped backoff, kill-on-drop.
- [`chatterbox/`](chatterbox/) — Python protocol worker: lazy torch/chatterbox
  imports, planner-approved load plans (`from_local`), voice-clone
  `voicePromptPath`.

## The C# sidecar host (`tools/Trackdub.SidecarHost`)

A small .NET 10 library that closes the loop from the managed side: it spawns
the worker process with piped stdio, speaks protocol v1 end-to-end
(health → load → infer → shutdown), enforces the protocol version stamp,
poisons the connection on timeout or id-mismatch, and kills the child on
dispose — mirroring `supervisor.rs` on the host side of the seam. Its test
suite lives in `tools/Trackdub.SidecarHost.Tests` (including a scripted
fake worker used for cheap host-lifecycle tests).

### Dev-only scope

**Nothing here ships or changes any shipping path.** The host is not
registered in Composition, is not a member of `Trackdub.slnx`, and no
`src/Trackdub.*` project references it. It exists to prove the seam with real
audio and measure the sidecar tax, per ADR-0016 rule 5 ("proven in the
harness first") and the spike plan's promotion criteria. Promotion (executor
lane + `ISidecarHost` + stage integration) is explicitly out of scope.

### Running the cheap suites (no model stack, no weights)

```bash
cargo test --manifest-path workers/supervisor/Cargo.toml   # Rust supervisor: 14 tests
cd workers/chatterbox && uv sync --dev && uv run pytest tests -q && cd ../..   # Python protocol: 15 tests
dotnet test tools/Trackdub.SidecarHost.Tests -m:1          # C# host: 13 pass, 3 honest skips
```

Live-worker tests (`ChatterboxLiveWorkerTests`, `SupervisorGateTests`,
`OverheadMeasurementTests`) are gated by `SidecarWorkerFact`: they skip with
an explanatory message unless `TRACKDUB_SIDECAR_TESTS=1` is set. A skip is
honest, never fake-ready — the repo's `RequiresBundledModelFact` pattern.

### Running live (real worker, real weights, real audio)

```bash
cd workers/chatterbox && uv sync --extra model && cd ../..
TRACKDUB_SIDECAR_TESTS=1 dotnet test tools/Trackdub.SidecarHost.Tests -m:1
```

**Stage the model snapshot first.** The live tests assert the snapshot
directory exists *before* spawning the worker and fail fast when it is
missing — there is no download-on-first-load here. Stage the Chatterbox
snapshot where `from_local` expects it
(`%LOCALAPPDATA%/Trackdub/model-cache/ResembleAI/chatterbox`, override with
`TRACKDUB_CHATTERBOX_SNAPSHOT`), e.g. by populating that directory from the
Trackdub model cache or a prior supervised run. The Rust supervisor gate test
additionally needs `cargo build --release --manifest-path
workers/supervisor/Cargo.toml` first.

### Evidence

Measured runs from 2026-10-08 (the raw logs were removed from the tree on
2026-11-12; they remain in git history at commit `7925368e` under
`tools/sidecar-evidence/`):

- `live-run.txt` — full live health → load → infer test output (real 24 kHz
  audio produced through the actual pipeline).
- `supervisor-gate-run.txt` — Rust supervisor `--check` accepting the real
  worker.
- `overhead-run.txt` + `warm-infer-20261008-033246.json` — warm-infer
  measurement: **median 7954.7 ms, n=30** (min 6445.6, max 12339.6) for 2.6 s
  of 24 kHz audio per call, `activeProvider: cuda`, cold CUDA load 120–200 s.

New live runs of `OverheadMeasurementTests` write their JSON evidence to
`tools/sidecar-evidence/`, which is gitignored.

### CI

[`.github/workflows/sidecar-worker.yml`](../.github/workflows/sidecar-worker.yml)
runs all three cheap suites on PRs and pushes touching `workers/**` or the
host tools. Live tests stay env-gated OFF in CI — the live evidence is the
"Evidence" summary above.
