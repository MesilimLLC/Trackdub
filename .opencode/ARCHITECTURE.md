# Why this system exists

## Purpose

`AGENTS.md` at the repo root already carries the rules: dependency direction, build commands, banned APIs, model governance, conflict order. It is dense, and it is read as prose. This `.opencode` system exists to turn those rules into **routable, executable procedure** — so an agent lands on the right context, runs the right command in the right order, and reports what actually happened.

Three problems it solves:

1. **Rules that are not actions.** "Treat warnings as errors" is a rule. `dotnet build Trackdub.slnx --configuration Release --no-restore -m:1 -warnaserror` is the gate. This system wires the second to the first.
2. **The readiness lie.** AGENTS.md principle 3: *provider registered ≠ model downloaded ≠ stage ran ≠ stage succeeded*. Language models compress that into a plausible sentence. `/ready` makes the compression impossible by forcing a rung-by-rung verdict with evidence.
3. **Context cost.** Every rule in `AGENTS.md`, every architecture invariant, every process, and every report shape in the orchestrator's context on every turn. Most turns need a small fraction. This system moves the rest behind an allocation decision.

## Scope: core repo only

This system governs **`Mesililm-LLC/Trackdub`** — the public core: `src/Trackdub.*`, `src/DubBench*`, `tests/Trackdub.Architecture.Tests`, `tools/`, `scripts/`, and `.github/workflows/`.

It does **not** govern the Avalonia desktop shell. That lives in the `Trackdub-gated` repo under `App.Avalonia` / `App.Avalonia.Tests` / `UI.Tests`, with its own `.opencode` system at `../Trackdub-gated/.opencode/`.

The split is not administrative convenience — see [Cross-repo split](#cross-repo-split) below.

## Three-level context allocation

The central mechanism. Context is loaded in one of three bands, decided before the task starts.

**Level 1 — isolation. Default. ~80% of tasks.**
Load nothing beyond what the task names. `/pin <sha>` in core loads one file. Fixing a stage-name constant loads the architecture tests and nothing else. Rationale: most requests are bounded, and a bounded request does not benefit from an unbounded context. Loading everything is not safer — it is noisier, and it dilutes the constraint that matters.

**Level 2 — filtered. Multi-file work.**
Load the **category index** (`domain/`, `processes/`, `standards/`, `templates/`), then only the specific files the task needs. Rationale: at this point the agent knows roughly which concern it is in but not which document holds the detail. The index resolves that in one read instead of a directory listing plus guessing.

**Level 3 — full. Cross-cutting analysis. Rare.**
Load everything. Justified only for work where a file you did not read would have changed your answer: dependency-graph edits, project renames or moves, model manifest or licensing changes, anything touching layering policy, TFM or packaging changes, and new projects entering `src/`. Rationale: at this scale, the failure mode is not "too little context" but "a rule you did not know existed." Under-reading a layering invariant costs a broken build and a reverted PR; the cost of over-reading is bounded token spend on a handful of tasks.

The level is chosen per task, not per session, and Level 3 is not a default escalated to by habit.

## Why four subagents, and what each owns

Subagents exist to **bound a concern**, not to add parallelism. Each is a single-writer owner of one kind of work, so the orchestrator never has to hold two competing mental models of the same subsystem.

| Agent | Edit | Concern | Why it is separate |
|---|---|---|---|
| `pipeline-inference` | **allow** | Producing change: pipeline stages, stage handlers, runtime stages, execution providers, engines, model manifest entries, through `Contracts → Domain → Application → Composition`. Also the sole writer for `/pin`. | Keeps mutation authority unambiguous. Exactly one agent can write; every judgement-bearing agent cannot. |
| `validation-gate` | deny | CI-equivalence and the architecture bounds audit; owns PASS/FAIL | **Separation of duties.** The agent that produced a change must not certify it. `pipeline-inference` cannot pass its own work; `validation-gate` has no write authority to make its verdict true. |
| `core-diagnostics` | deny | Read-only triage: build/test failures, `packages.lock.json` conflicts, pin fallout, warnings-as-errors breaks, native ONNX/CUDA/TensorRT loading, flaky tests. Backs `/ready`. | Enforces non-mutation structurally, not by instruction. A readiness audit that could "just fix it" is worthless — the fix would make the rung pass without proving it. |
| `benchmark-perf` | deny | Benchmark execution, baseline diffing, regression attribution | Measurement discipline. Restricting it to measure-and-report keeps tuning decisions — which are user decisions — out of the measurement path, so a number is never produced by the same loop that wanted a better one. |

The load-bearing property is not the count of four; it is that **three of four have `edit: deny`.** Read-only is enforced by permission, not by good intentions. A judge that could edit its own evidence is not a judge.

`validation-gate` is the one that prevents the worst collapse — "I built it and it works" as a single unverifiable claim.

Note the consequence for `/pin`: it mutates git state, so it cannot run on a judge. It routes to the writer and carries an explicit mid-procedure recovery instruction, because a read-only agent failing *between* the submodule checkout and the commit would leave the gated repo dirty.

## Why the orchestrator keeps governance and release/ops docs

Architecture governance, ADRs, layering policy, release notes, and CI policy are **not** delegated. Two reasons.

**Continuity.** These are the decisions that constrain every subagent. If they live in a subagent, the orchestrator holds a summary of them forever and the summary drifts from the source. Keeping them in the orchestrator means the governing context is always the same context that dispatches the work.

**Authority.** Dependency direction, conflict order, and model-licensing rules are decisions about the system, not tasks within it. A delegation boundary implies the delegate decides within it. Architecture policy is decided once, globally, and then applied — it is not delegated scope. The subagents consume governance; they do not author it.

Practically: the orchestrator reads fewer files than any other agent, and holds the most authority.

## Manager–worker, and keeping context out of the window

This is a manager–worker pattern. The orchestrator decomposes, delegates, adjudicates between results, and reports. It does not hold subagent transcripts, read the files a subagent read, or retain intermediate reasoning from work it delegated.

Why it matters concretely: a `/validate` run touches a Release build and a full test pass across the solution. The failure output, the restore log, and the per-project warnings belong to `validation-gate`'s context. They do not belong in the orchestrator's — not because tokens are precious in the abstract, but because the orchestrator's context has one job: hold the system-level constraints accurately while dispatching the next task. Every line of delegated output is a line not spent on that.

The rule is mechanical: work delegated is context not retained. If a subagent's result needs to persist, it persists as the verdict and the evidence — not as the transcript that produced it.

## Evidence and gating

**No claim without a command that proves it.**

This is not aspirational; it is the load-bearing assumption of the whole system. AGENTS.md: "never mark Done without proof." A proof is a command, an exit code, an artifact, or a file path — not a recollection, not a doc that says so, not a green badge from a previous run.

Therefore:

- **`NOT VERIFIED` is a valid and expected output**, not a failure to be papered over. It is the honest state when a rung has no evidence behind it. Every rung of `/ready`, every gate of `/validate`, every delta of `/bench` accepts it.
- **Partial evidence never becomes a pass.** `/validate` stops at the first failing gate and shows the exact command and output. `/bench` reports measured numbers only — a missing baseline means `NOT VERIFIED`, never "no regression". `/ready` never rounds a rung up.
- **Rules are enforced by the build wherever possible.** `BannedSymbols.txt` via `Microsoft.CodeAnalysis.BannedApiAnalyzers`; dependency direction, acyclicity, ADR-0011 Contracts isolation, ONNX/WinML/DNNL asset invariants and portable-RID lock graphs via `tests/Trackdub.Architecture.Tests`; analyzer diagnostics via `src/Trackdub.Analyzers`. A rule that can be a test should be a test — a test fails deterministically, and an instruction does not.
- **Documentation is not evidence.** For implementation facts, the checkout and `runtime/trt-rtx-ep.manifest.json` ground answers, Context7 covers third-party libraries, and any retrieved text is treated as evidence to be verified against live code. The conflict order still governs: source/tests > task instructions > documentation.

## Stateless design

**State lives in the repo: code, tests, git history, and the PR. Nothing lives in bookkeeping files.**

The system holds no task database, no progress ledger, no status file, no notes-to-self. Consequences:

- A fresh session with no prior context reaches the same conclusion an interrupted one would, because every fact needed to reconstruct state is in the tree.
- There is no reconciliation problem. No bookkeeping file can drift from reality, because there is no bookkeeping file.
- There is no serialization point, and therefore no lost-update class of bug.

The same reasoning explains `NOT VERIFIED` being load-bearing rather than a hedge: the system's only durable record is the evidence in the repo, so an unproven claim has nowhere to hide.

## Cross-repo split

Two repos, two `.opencode` systems, one seam.

`Trackdub` is the public core and is consumed as a **pinned read-only submodule** (`external/Trackdub`) by `Trackdub-gated`, which owns the Avalonia desktop shell. The split follows the deployment boundary, which is also a governance boundary: core can be consumed by anything, gated is application-specific.

One consequence is architectural and worth stating plainly — **the gated repo builds against the pin, never against core `HEAD`.** Editing inside `external/Trackdub` is forbidden; `/pin` reports any such edit as a hard failure. If a bump needs a core change, that change lands in the core repo, in its own PR, and is pinned afterward.

This is why `/pin` exists in **both** systems, and why this copy handles being invoked from the wrong repo: from core it explains that the submodule lives in gated and redirects to `../Trackdub-gated/.opencode/context/processes/submodule-pin-bump.md`. One procedure, documented once, reachable from both sides.

The cost is duplication at the boundary: core and gated each describe the pin procedure. Accepted — the alternative is a shared file that neither repo owns, which the stateless rule already rules out.

## Known gaps

- **External issue trackers are opt-in.** Do not use Linear or require issue searches/status/TS IDs by default. Track work in Git; create a persisted plan only when the user requests one. Update an external tracker only when the user explicitly asks. "Never mark Done without proof" stays enforced by the evidence in the handoff.
- **The desktop repo has its own system.** Core and gated are separate trees with separate context. Cross-cutting reasoning that spans both — anything touching the core/desktop contract — must load two systems. `/pin` and the breaking-change triage are the mitigation, not a solution.
- **Two planned workflows are unwritten.** `feature-change` and `release-and-pin` are named in the routing table but do not exist. The three shipped workflows are `new-inference-stage`, `pr-ready-loop`, and `benchmark-regression-triage`.
- **Factual claims decay.** Every grounding statement in `context/` was verified at authoring time against a specific revision. Counts, file paths, test names, and type names drift. Anything asserted here should be re-verified with `verify with: <command>` before being relied on, and this tree will otherwise start lying to its own agents.
