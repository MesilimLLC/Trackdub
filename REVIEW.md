# Review Guide

Use this file when reviewing pull requests in the public Trackdub core
(`MesilimLLC/Trackdub`).

This is a reviewer checklist, not an authoring guide. For contribution rules,
see `CONTRIBUTING.md` and `docs/repository-policy.md`. For the required PR body
shape, see `.github/pull_request_template.md`. For build, dependency direction,
and model governance, see `AGENTS.md`.

The private desktop product lives in `MesilimLLC/Trackdub-gated` and consumes
this repo as a pinned submodule. Avalonia shell and gated licensing trust-ring
changes belong there, not here.

## Automated review (Bugbot)

If Cursor Bugbot (or similar) is configured for this repository, it should follow
this checklist. Human reviewers still own the final merge bar.

Bugbot should surface the same blocking issues as
[Automatic review stops](#automatic-review-stops) below (fake readiness, manifest
gates, planner routing, architecture boundaries).

## Review priorities

Review in this order:

1. Correctness
2. Scope control
3. Architecture and dependency direction
4. Model/license safety
5. Runtime route visibility
6. Test evidence
7. Docs and milestone alignment

## Approval standard

Do not approve a PR if any of these are unclear:

- What changed
- What did not change
- What was tested
- What was not tested
- Whether model, dependency, or runtime routing behavior changed
- Whether the change stays inside the intended milestone or slice
- Whether desktop-only work was incorrectly landed here instead of Trackdub-gated

## Checklist

### 1. Scope

- [ ] The PR solves one clear problem or one bounded milestone slice.
- [ ] The PR description explicitly states non-goals.
- [ ] Unrelated cleanup is absent or clearly separated.
- [ ] No Avalonia UI shell, gated trust-ring, or private product-only work was
      added here. Those belong in Trackdub-gated.

### 2. Hard architectural constraints

Treat these as request-changes items, not suggestions.

- [ ] `Trackdub.Domain` still depends on nothing.
- [ ] Project references still follow the repository dependency direction in
      `AGENTS.md`, and the dependency graph stays acyclic.
- [ ] Inference implementations stay in `Trackdub.Inference` /
      `Trackdub.Inference.Onnx`, not Application or above.
- [ ] Persistence implementations stay in `Trackdub.Infrastructure` (or Media
      where that boundary already owns the concern), not Domain or Contracts.
- [ ] Model wrappers and engine adapters do not mutate project state.
- [ ] CLI entry points stay in `Trackdub.Cli`; reusable automation stays in
      `Trackdub.Sdk`.
- [ ] Neutral licensing mechanisms stay portable. Private multi-key trust-ring
      / revocation product policy is not reinvented here for desktop shipping.

### 3. Runtime planning and model routing

- [ ] Runtime/provider selection goes through `IRuntimePlanner`. Engines must
      not hardcode CPU, DirectML, or cloud routing internally.
- [ ] Model selection and model metadata come from
      `BundledModelManifestRegistry` and commercial-safe evaluation, not ad hoc
      file probing or alias-only logic.
- [ ] Translation routing is manifest-driven. Use `engine_family`,
      `capabilities`, and `language_coverage`; do not infer routing behavior from
      model name strings except where the current planner or router already
      treats aliases as soft hints.
- [ ] New model usage is blocked unless the manifest, license, and
      commercial-safe handling are updated in the same change.
- [ ] Planner and diagnostics surfaces do not expose absolute machine-local
      paths unless there is a specific troubleshooting reason.

### 4. Stage and workflow discipline

- [ ] `StageRunRecord.Start(...)` uses `StageNames.*` constants only. No inline
      stage-name literals.
- [ ] New stages update the canonical stage-name source and its guard tests in
      the same PR.
- [ ] Workflow growth stays bounded. New behavior should usually extend targeted
      workflows or handlers rather than expanding a general-purpose coordinator
      or service.
- [ ] Skip/fail paths preserve original artifacts and log exact skip reasons.
- [ ] Pipeline stages prefer immutable execution snapshots.

### 5. Runtime behavior

- [ ] CPU/GPU/cloud route changes are explicit in code and called out in the PR.
- [ ] Fallback behavior is visible, not hidden behind silent routing changes.
- [ ] Startup, model, media, or CLI fixes are grounded in log or runtime
      evidence when applicable.
- [ ] Cross-platform impact is considered. Do not assume Windows unless the
      boundary is explicitly Windows-only.

### 6. Models, dependencies, and licensing

- [ ] New packages are centrally versioned in `Directory.Packages.props`.
- [ ] New models have manifest coverage before use
      (`src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json`).
- [ ] License and commercial-safe impact are called out explicitly.
- [ ] Unknown or unverified license status is not treated as safe.
- [ ] When lockfiles change (`packages.lock.json`), verify the intentional
      dependency change in `Directory.Packages.props`.
- [ ] Attribution updates (`THIRD_PARTY_NOTICES.md` or equivalent) land when
      `requires_attribution: true` models are added.

### 7. Tests

- [ ] Tests were added or updated at the correct layer.
- [ ] The commands run are listed exactly in the PR.
- [ ] Skipped tests are named and justified.
- [ ] Validation matches the risk of the change.
- [ ] The PR does not claim broader verification than was actually run.
- [ ] If a new seam, route, lifetime rule, or architectural guard is introduced,
      an enforcing test lands in the same PR.
- [ ] Pipeline changes cover success, disabled/skipped, missing-prerequisite,
      and failure paths.
- [ ] Application-layer tests use fakes from `tests/Trackdub.TestDoubles/`
      where appropriate.

### 8. Docs and milestone alignment

- [ ] Relevant docs were updated when behavior or constraints changed.
- [ ] ADR or troubleshooting docs are linked when the change affects them.
- [ ] Repository policy and AGENTS.md remain accurate if boundaries moved.

## Automatic review stops

Request changes immediately if a PR does any of the following:

- Violates Domain-depends-on-nothing or other dependency-direction rules.
- Adds provider or model routing outside the planner or router path.
- Adds translation behavior based on alias naming conventions instead of
  manifest metadata.
- Uses a model before manifest, license, and commercial-safe handling are
  present.
- Treats provider registration, download, stage run, skip, or success as the
  same readiness state.
- Adds a new stage name without updating `StageNames` and its guard tests.
- Lands desktop-shell or gated trust-ring product policy in this public core.
- Claims validation that the PR body does not actually support with exact
  commands.

## Review comment style

Prefer comments that are concrete and falsifiable:

- Point to the exact file or behavior.
- State the risk.
- State what evidence is missing.
- Suggest the narrowest acceptable correction.

Good review comments usually sound like:

- "This makes Domain depend on Infrastructure, which breaks the repo boundary
  rules."
- "The PR says CPU fallback is unchanged, but this planner path now silently
  reroutes."
- "This adds a model entry point without manifest and license-policy coverage."
- "The tests listed do not exercise the changed runtime path."
- "This trust-ring change belongs in Trackdub-gated, not the public core."

## Minimum merge bar

A PR is ready to merge when:

- The scope is still bounded.
- The architecture and dependency-direction rules still hold.
- Model, license, and runtime impacts are explicit.
- Validation is honest and adequate.
- The written docs match the shipped behavior.
