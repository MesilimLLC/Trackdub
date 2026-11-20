---
description: "Prove-or-fail readiness audit for a model, provider, or pipeline stage - refuses unproven readiness"
agent: subagents/core-diagnostics
---

Read-only readiness audit. **Modify nothing.** This runs on `core-diagnostics`, which has `edit: deny`: it inspects, proves, and reports. It cannot make a rung pass by changing it.

**Audit target:** `$1` — the provider / stage / model to audit. If `$1` is empty, ask which target to audit and stop. For a provider or stage, resolve its applicable model selection from the runtime plan; do not treat the target name itself as a model id.

## Load context first

- `.opencode/context/domain/inference-stack.md` — provider/stage topology, execution providers, runtime flavors, stage ordering.
- `.opencode/context/templates/evidence-report.md` — output shape. Use it.

Grounding sources, in conflict order (source/tests > task instructions > documentation):
- `src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json` — the bundled inventory.
- `tools/ci/audit-bundled-model-manifest.py`, `tools/ci/validate-manifest-schema.py`, `tools/ci/verify-manifest-hashes.py`.
- `AGENTS.md` "Model Governance".
- `runtime/trt-rtx-ep.manifest.json` for pin policy and `docs/reference/windows-ml-stage-provider-matrix.md` for provider wiring.

## The readiness ladder

Walk **every** rung. Each rung gets exactly one of `PASS` / `FAIL` / `NOT VERIFIED` / `N/A`, plus evidence or the reason it does not apply. Use `N/A` only when the target or run makes a rung inapplicable; do not use it for missing evidence.

| # | Rung | What proves it |
|---|------|----------------|
| 1 | Provider registered | a concrete provider-registry entry or Composition DI registration; manifest presence alone is not proof |
| 2 | Runtime installed | the runtime package/binding is present and resolves on this host |
| 3 | External binary available | the required external executable exists and its version was observed |
| 4 | Model manifest present | a manifest entry exists for the model selected for this model-backed target; N/A for a model-free provider or stage |
| 5 | Model files downloaded | the weights files exist on disk, with observed byte sizes |
| 6 | Checksum verified | sha256 matches the manifest value; cite the verifying command |
| 7 | License metadata present | the manifest declares a license field for this model |
| 8 | License reviewed | the license was actually read and classified, not merely present |
| 9 | Commercial mode allowed | commercial-use is permitted. **Unknown license = unsafe = FAIL** |
| 10 | Hardware provider available | the required EP/device is present on this host (e.g. GPU/EP probe) |
| 11 | Stage enabled in snapshot | the stage is enabled in the configuration/snapshot actually in use |
| 12 | Prerequisites satisfied | every upstream stage and every prerequisite stage completed successfully |
| 13 | Stage ran | execution evidence exists — a run report, not an intent to run |
| 14 | Stage produced usable output | a non-empty, well-formed artifact of the expected type/shape/duration |
| 15 | Stage skipped safely *(if applicable)* | an explicit skip with a recorded reason, and original artifacts preserved |
| 16 | Stage failed cleanly *(if applicable)* | failure surfaced with a recorded reason; artifacts preserved, not silently dropped |

Use `N/A` for provider/runtime/binary/hardware rungs when the target has no applicable route, for model-specific rungs when the resolved route is model-free, for stage-only rungs when the target is not a stage, and for rungs 15/16 when that skip/failure outcome did not occur.

## Non-equivalences — these are all FALSE

State each one explicitly wherever the evidence would tempt the shortcut:

- **disabled != ran.** A stage that is disabled in the snapshot has produced nothing. Rung 11 PASS does not advance rungs 13–14.
- **skipped != succeeded.** A skipped stage has no output. A skip is not a zero, and not a pass.
- **registered != installed.** A provider registry or DI entry proves registration, not presence on this host.
- **manifest present != provider registered.** Prove registration from the provider registry or Composition DI wiring, not from the model manifest.
- **repo license != weights commercial.** The repo's own LICENSE file says nothing about redistribution rights of model weights. Weights are governed by rung 9 only.
- **duration match != lip-sync quality.** Wall-clock duration within tolerance is a throughput/real-time signal. It is not a quality, accuracy, intelligibility, or lip-sync verdict. Quality needs its own metric and its own evidence.

## Verdict

Emit exactly one:

- `READY` — every applicable rung PASS, with evidence cited for each; every N/A has a stated reason. Nothing inferred.
- `NOT READY` — at least one rung FAIL. List the failing rungs with evidence.
- `NOT VERIFIED` — at least one rung could not be proven. List **exactly** what could not be proven and, per item, the command or inspection that would prove it.

**Never round up.** No FAIL may be softened to NOT VERIFIED to produce a usable verdict. No NOT VERIFIED may be reported as PASS. Absence of a counter-example is not evidence.

## Report shape

```markdown
# Readiness audit: <target>

| # | Rung | Result | Evidence |
|---|------|--------|----------|
| 1 | Provider registered | PASS | ... |
| 4 | Model manifest present | N/A | target is model-free |
| ... |

## Verdict
READY | NOT READY | NOT VERIFIED

## Blocking items
- rung N — <reason> — prove with: `<command>`

## Non-equivalences checked
- disabled != ran: <how it was checked>
- skipped != succeeded: <...>
- registered != installed: <...>
- manifest present != provider registered: <...>
- repo license != weights commercial: <...>
- duration match != lip-sync quality: <...>
```

## Rules

- Read-only. No source edits, no `git checkout`, no restores, no downloads. If a rung cannot be proven without writing, it is `NOT VERIFIED` — say so and name the command that would settle it.
- Never mark a rung PASS from documentation, a manifest field alone, or a previous audit's result. Verify against live code and live disk.
- If the target is a model, or a provider/stage resolves to a model-backed route, and that model does not exist in the bundled manifest, rung 4 is `FAIL` — stop there. For provider/stage targets, first resolve the applicable model selection; a model-free target is `N/A` at rung 4.
- Do not recommend enabling a stage, adding a model, or changing a license decision. Report only.
