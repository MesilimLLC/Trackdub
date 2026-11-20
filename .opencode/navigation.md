# Trackdub core — `.opencode` system index

Scope: **`Mesililm-LLC/Trackdub` public core only.** This index covers library, inference, media, benchmarks, tooling, and CI for the core repo.

**Desktop/Avalonia shell work is NOT in scope here.** That belongs to the separate `Mesililm-LLC/Trackdub-gated` repository, which carries its own `.opencode` system. Find a sibling checkout with `git rev-parse --show-toplevel` on that repo, then read `<gated-root>/.opencode/navigation.md`. Desktop-shell context is deliberately not duplicated in core.

Design rationale for every choice below: [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick start

```
/validate [scope]   # full CI-equivalent gate: restore, Release -warnaserror build, tests, architecture bounds audit
/ready <target>     # prove-or-fail readiness audit for a provider / model / pipeline stage
/bench <baseline> [workload]   # run benchmarks, diff vs a named baseline, attribute deltas
/pin <tag-or-sha>   # bump the pinned core submodule in Trackdub-gated, verify both repos
```

Most common use: `/validate` before declaring any change done. Then `/ready` when a question of the form "is X usable yet?" arises. `/bench` and `/pin` are explicit, opt-in checks — neither runs inside CI.

## Orchestrator

The orchestrator is the **primary agent** in this directory. It:

- owns architecture governance and release/ops documentation itself (does **not** delegate these),
- dispatches implementation and audit work to the four subagents below,
- decides context level (1/2/3) and which context files to load,
- maintains the conflict order from `AGENTS.md`: source/tests > task instructions > documentation.

The orchestrator's own context stays lean. Detail lives in `context/` and is loaded on demand.

## Subagents (4)

| Agent | Edit | Owns |
|---|---|---|
| `pipeline-inference` | allow | Pipeline stage work end to end: stage handlers, runtime stages, execution providers, engines, model manifest entries, through the `Contracts → Domain → Application → Composition` path. Also the only agent with write authority for `/pin`. |
| `validation-gate` | deny | The full CI-equivalent gate and the architecture bounds audit. Owns the PASS/FAIL verdict. |
| `core-diagnostics` | deny | Read-only triage: build/test failures, `packages.lock.json` conflicts, submodule pin fallout, warnings-as-errors breaks, native ONNX/CUDA/TensorRT loading, flaky tests. Produces ranked hypotheses with discriminating commands. Backs `/ready`. |
| `benchmark-perf` | deny | Benchmark execution, baseline diffing, regression attribution. Measurement only, no tuning. Backs `/bench`. |

Three of four are `edit: deny`. That is the point: a read-only agent cannot make a failing rung pass by changing it. The writer is `pipeline-inference`; the judges are the other three.

`pipeline-inference` and `validation-gate` are deliberately distinct: one *makes* the code, the other *judges* it. Never let the agent that produced a change certify it.

## Commands (4)

| Command | File | Agent | Purpose |
|---|---|---|---|
| `/validate` | [commands/validate.md](commands/validate.md) | `validation-gate` | restore → Release `-warnaserror` build → Release tests → architecture bounds audit → `packages.lock.json` integrity. PASS/FAIL table; stops at the first failure. |
| `/ready` | [commands/ready.md](commands/ready.md) | `core-diagnostics` | Walk the 16-rung readiness ladder for a provider/model/stage. `READY` / `NOT READY` / `NOT VERIFIED`. Never rounds up. |
| `/bench` | [commands/bench.md](commands/bench.md) | `benchmark-perf` | Run DubBench/Benchmarks, diff vs a **required** named baseline, attribute deltas to GPU / CPU / IO / model-or-config. |
| `/pin` | [commands/pin.md](commands/pin.md) | `pipeline-inference` | Bump `external/Trackdub` in Trackdub-gated, commit the gitlink with `-m`, verify **both** repos build, triage breaking changes. Only writing command in the set. |

Command shell-injection, per the OpenCode command format: `$ARGUMENTS` / `$1`..`$n` substitute arguments; `` !`cmd` `` injects shell output into the prompt; `@path` includes file content. `/validate` and `/bench` inject git state so the agent always reports what it was actually looking at.

## Context files

Grouped by category. Load the minimum that answers the question — do not bulk-load.

### `context/domain/` — what the system *is*

Subject-matter facts. Read-only reference; changes here need evidence from code.

- `context/domain/architecture.md` — the 20-project dependency graph, layer responsibilities, build/test commands, cross-repo relationship to the gated desktop app.
- `context/domain/inference-stack.md` — providers, execution providers, runtime flavors, model manifests, stage topology and ordering. Referenced by `/ready`.
- `context/domain/terminology.md` — pipeline stage vocabulary, artifact layout, and the state distinctions the repo treats as different facts.

### `context/processes/` — how work is *done*

Repeatable multi-step procedures.

- `context/processes/adding-pipeline-stage.md` — the Contracts → Domain → Application → Composition path for a new stage, with per-step verification and common mistakes. Backed by `workflows/new-inference-stage.md`.
- `context/processes/pr-lifecycle.md` — branch → validate → push → PR → CI → review threads → merge, using `gh`, gh-stack, GitKraken, and cubic.
- `context/processes/submodule-pin-bump.md` — the canonical core-pin bump procedure. Lives in the **gated** repo; `/pin` points here when invoked from core.

### `context/standards/` — what "correct" means here

The bar a change is measured against. Consult before editing, cite in review.

- `context/standards/validation-gates.md` — the CI-equivalent command sequence, `-m:1` rationale, `--no-restore`/`--no-build` sequencing, the non-build gates, and the PASS/FAIL report template. Backs `/validate`.
- `context/standards/coding-standards.md` — file-scoped namespaces, `sealed` where extension is not intended, `Async` suffix, immutable `record` in Domain, warnings-as-errors, cross-platform portability.
- `context/standards/architecture-rules.md` — the readiness ladder as enforceable rules, the boundary-violation catalogue, and dependency-direction rules.

### `context/standards/` — what "correct" means

Coding, testing, dependency-direction, and review rules that must hold across all changes.

- Authority order for any rule: `@AGENTS.md` (repo root) is authoritative. `BannedSymbols.txt` + `Directory.Build.props` enforce the banned-API rule; `tests/Trackdub.Architecture.Tests` enforces dependency direction, acyclicity, ADR-0011 Contracts isolation, ONNX/WinML/DNNL asset invariants, and portable-RID lock graphs; `src/Trackdub.Analyzers` carries repo analyzers.

### `context/templates/` — output shapes

Report formats so evidence is emitted consistently.

- `context/templates/evidence-report.md` — the evidence/gating report shape used by `/ready` and by audit agents. Referenced by `/ready`.

## Workflows

Multi-step compositions that chain subagents and gate their transitions. Every step ends in an observable fact, not an intention; a step returning `NOT VERIFIED` blocks the next step rather than passing through.

- **[workflows/new-inference-stage.md](workflows/new-inference-stage.md)** — new dubbing/inference stage from intake to proven readiness. Orchestrator drives; `pipeline-inference` writes layers in dependency order; `core-diagnostics` owns every failure; `validation-gate` renders the final verdict. Exists because the most expensive defect class here is a **faked readiness claim**: a provider registered, a manifest entry present, a stage that compiles, and no proof it downloaded a model, ran inference, and produced an output artifact.

- **[workflows/pr-ready-loop.md](workflows/pr-ready-loop.md)** — working tree to merge-ready: audit, dev-loop validation, full Release gate, bounds audit, commit, push, PR, CI polling, review-thread fix loop.
- **[workflows/benchmark-regression-triage.md](workflows/benchmark-regression-triage.md)** — environment capture, reproduce, diff vs baseline, attribute the delta, decide accept/fix/revert. Measured numbers only.

Two further workflows are planned (`feature-change`, `release-and-pin`) but are **not yet written**. Do not reference them as existing.

## Routing table

| Request type | Agent | Context to load |
|---|---|---|
| Add/change a pipeline stage, provider, or EP binding | `pipeline-inference` | `context/domain/inference-stack.md`, `context/standards/` |
| "Does it build? Is it safe to merge?" | `validation-gate` (via `/validate`) | `context/standards/`, `context/processes/` |
| "Is <model/provider/stage> ready?" | `core-diagnostics` (via `/ready`) | `context/domain/inference-stack.md`, `context/templates/evidence-report.md` |
| "This test/build is failing — why?" | `core-diagnostics` | `context/processes/` |
| `packages.lock.json` conflict | `core-diagnostics` | `context/processes/` |
| "Did this get slower? Why?" | `benchmark-perf` (via `/bench`) | `context/domain/` |
| "Bump the core pin" | `pipeline-inference` (via `/pin`) | `context/processes/submodule-pin-bump.md` (in gated) |
| Architecture governance, ADRs, layering policy | **orchestrator** (kept in-house) | `context/standards/` |
| Release notes, ops docs, CI policy, PR descriptions, commits | **orchestrator** (kept in-house) | `context/standards/`, `context/processes/` |
| External issue tracker | **opt-in**: use only when the user explicitly asks | n/a |
| Avalonia desktop shell, UI, playback UX, mpv, subtitle pipeline | *out of scope* — use the gated repo system | `../Trackdub-gated/.opencode/` |
| Third-party library docs (non-Trackdub) | **orchestrator** | Context7; verify retrieved docs against live code |
| Trackdub implementation facts / pin policy / provider wiring | **orchestrator** | checkout source/docs and `runtime/trt-rtx-ep.manifest.json` |

## Context allocation strategy

Three levels, chosen by blast radius:

- **Level 1 — isolation (default, ~80% of tasks).** Load nothing beyond what the task names. Fastest, and correct for most work.
- **Level 2 — filtered (multi-file work).** Load the category index, then only the specific files the task actually needs.
- **Level 3 — full (cross-cutting analysis).** Load everything. Reserve for dependency-graph changes, model governance changes, and anything touching layering.

Details and the reasoning: [ARCHITECTURE.md](ARCHITECTURE.md).

## Known gaps

- **External issue trackers are opt-in.** Do not use Linear or require issue searches/status/TS IDs by default. Track work in Git; create a persisted plan only when the user requests one. Update an external tracker only when the user explicitly asks.
- **The desktop repo has its own system.** Core and gated are deliberately separate `.opencode` trees. Cross-repo work — chiefly `/pin` — is the only seam, and it is documented on both sides.
- **Two planned workflows are unwritten.** `feature-change` and `release-and-pin` are named in the routing table but do not exist. The three shipped workflows are `new-inference-stage`, `pr-ready-loop`, and `benchmark-regression-triage`.
- **`/pin` mutates git from the writer agent.** It is the one command that edits state, so it runs on `pipeline-inference` rather than a read-only judge. A failure mid-procedure can leave the gated submodule checked out but uncommitted — recover with `git -C external/Trackdub checkout <previous-sha>`.
