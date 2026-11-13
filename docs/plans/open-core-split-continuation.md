# Trackdub Open-Core Split Continuation Plan

Status: active execution plan  
Canonical planning repository: `Mesililm-LLC/Trackdub`
Last verified core commit: `d590533b3220edf5211ce773b7620487c29eb1e7`  
Scope: finish the public-core release gates, rebuild the private desktop repository with fresh history, then validate the cross-repository boundary.

## Purpose

Trackdub is moving from a mixed historical monorepo to a deliberate repository model:

| Repository | Visibility | Role |
|---|---|---|
| `Mesililm-LLC/Trackdub` | Public after release gates pass | Apache-2.0 reusable engine, SDK, CLI, pipeline, inference, media, neutral licensing mechanisms, tooling, and tests |
| `Mesililm-LLC/Trackdub-gated` | Private | Proprietary desktop product, product policy, activation client, packaging, signing, and releases |
| `Mesililm-LLC/Trackdub-Monorepo-Archive` | Private, archived | Full historical monorepo; never an active development target |
| `Mesililm-LLC/api.trackdub` | Private | Future server-side activation and product API |
| `Mesililm-LLC/portal.trackdub` | Private | Portal product |
| `Mesililm-LLC/trackdub.com` | Private | Marketing site |

This plan preserves the actual technical boundary. It is not a license-text-only cleanup.

## Verified Current State

### Public core

`Mesililm-LLC/Trackdub` **is public** (made public since this section was last written; Phase 1's release gates passed) and has a fresh-root history. Its root commit was `8bac38a8fa3e0343c5d10c558b64d269d15e7828`; follow-up commits have added CI and cross-platform fixes. See "STATUS: Phase 3 partly complete" under Phase 3 below for current validated state.

The core contains:

- `Trackdub.slnx` and focused `.slnx` solutions.
- Domain, Contracts, Application, Infrastructure, Media, Playback, Inference, ONNX, Composition, SDK, CLI, Licensing, Benchmarks, Tools, Analyzers, and the DNNL native package project.
- 12 test projects covering those public projects.
- An unmodified Apache-2.0 `LICENSE` and the root `NOTICE`:
  ```text
  Trackdub
  Copyright 2024-2026 Trackdub LLC
  ```
- A neutral `IExportTierGate` contract and no public concrete export-tier registration.

The split is not release-ready yet:

1. The current CI run on the latest inspected head failed.
   - Format and macOS build/test passed.
   - Windows build passed, but `Trackdub.Media.Tests.WavePcm16Tests.WriteSamplesAsync_move_failure_deletes_temp_file_and_preserves_destination` failed because the test expects exactly `Exception` while the operating system returns `UnauthorizedAccessException`.
   - The Linux job needs fresh inspection; its failed log was not retrievable in the inspected run.
2. CodeQL fails because:
   - GitHub Advanced Security is not enabled for the private repository.
   - `.github/codeql/codeql-config.yml` still includes the removed `frontend` path, which breaks the Python analyzer.
3. The Dependabot workflow claims patch/minor-only automation, but two major GitHub Actions updates were merged into `main`. Treat that policy as untrusted until verified and corrected.
4. Stale legal and boundary wording remains:
   - `docs/legal/CONTRIBUTOR-LICENSE-AGREEMENT.md` still describes GPLv3 plus commercial dual licensing.
   - `docs/legal/THIRD_PARTY_NOTICES.md` still contains desktop-only App.Avalonia and LibVLC packaging claims.
   - `src/Trackdub.Contracts/IExportTierGate.cs` says the interface is implemented in the Application layer, which is no longer the intended boundary.
   - `Directory.Build.props` has no opt-in public package ownership or Apache package-license policy.

### Desktop staging repository

`Mesililm-LLC/Trackdub-gated` is private but is explicitly a staging repository, not the final product repository. Its README confirms that it contains desktop code plus cloud and activation quarantine material. Its existing history begins with a private import and includes staging/quarantine commits. It must not become the final desktop history unchanged.

The final `Trackdub-gated` repository must receive a new fresh root. Preserve the existing staging repository separately for recovery, then reuse `Trackdub-gated` as the canonical private product repository.

### Archive and service repositories

`Mesililm-LLC/Trackdub-Monorepo-Archive` is private and archived. Do not modify it.

`api.trackdub`, `portal.trackdub`, and `trackdub.com` exist as private repositories. They are not sources to copy into the core or desktop repository.

## Final Boundary

### Public `Trackdub`

Keep reusable, product-neutral code:

- Domain, contracts, pipeline stages, orchestration, media primitives, inference implementations, model governance, SDK, CLI, tools, benchmarks, analyzers, and public tests.
- `Trackdub.Licensing` only while it stays mechanism-oriented: token validation, public-key verification, hardware fingerprinting, neutral claim parsing, and neutral result types.
- `IExportTierGate` as a neutral contract. Its documentation must say that a consuming product supplies policy when it needs one.
- Generic update-manifest interfaces and services only when they contain no desktop-specific URL, channel, or paid-entitlement policy.

Do not keep:

- Avalonia desktop shell, product views, product view models, branding, installer or signing logic, storefront integration, activation UI, entitlement persistence, concrete free/pro policy, release packaging, desktop telemetry policy, or desktop release workflows.
- Any API, worker, webhook, portal, website, Docker, or cloud deployment code.
- Concrete watermark, five-minute export restriction, or upgrade messaging.

### Private `Trackdub-gated`

Own product behavior and composition:

- Avalonia workstation, views, view models, controls, styles, branding, desktop playback integration, desktop tests, and product release workflows.
- One and only one production `IExportTierGate` implementation. It must own the watermark rule, export duration limit, tier mapping, and upgrade messaging.
- Activation client integration, local token storage, offline/grace behavior, updater configuration, packaging, signing, and binary distribution.
- Product-specific native acquisition and installer layout.
- Proprietary notice, approved EULA integration, and product-specific third-party notices.

The desktop repository may depend on the public core. The public core must never depend on the desktop repository.

### `api.trackdub`

Owns future server-side activation:

- License issuance, device and seat records, activate/reactivate/status/deactivate routes, revocation, purchase webhooks, signing keys, and server persistence.

The legacy activation service remains an archive/source of behavior until its replacement is designed, ported, parity-tested, deployed, and verified. It is not a desktop dependency and is not part of this split's critical path.

## Phase 1: Make the Public Core Release-Ready

Work on a normal branch from current `main`. Do not rewrite history or force-push.

### 1.1 Establish a green, reproducible CI baseline

1. Reproduce the Windows Media test failure locally and in CI.
2. Decide the actual contract for the move-failure path:
   - If access denial is the expected Windows result, update the test to assert that specific observable behavior.
   - If the method should normalize or avoid that exception, fix the implementation and retain a precise test.
   - Do not weaken the test to accept any exception without evidence.
3. Inspect the Linux failure from a fresh run; do not assume it is the same as Windows.
4. Run the root and focused SLNX build/test commands on all supported CI platforms.
5. Keep the cross-platform target-framework logic in `Directory.Build.targets` covered by architecture tests and lock-file tests.

Gate: CI format, Windows, Linux, and macOS must all pass on the same core commit.

### 1.2 Repair the CodeQL configuration and decide the security gate

1. Remove stale `frontend` references from `.github/codeql/codeql-config.yml`.
2. Decide whether the public core's Python tooling should be analyzed:
   - If yes, configure CodeQL Python for the retained scripts and tools without assuming a frontend application exists.
   - If no, remove the Python matrix deliberately and document the scope.
3. Enable GitHub Advanced Security for the private repository before requiring CodeQL to pass, or explicitly change the pre-public gate so that CodeQL is first validated immediately after public visibility is enabled.
4. Keep C# and Actions analysis aligned with the public-core solution and workflow paths.
5. Do not claim a green CodeQL gate while GitHub rejects SARIF uploads.

Owner decision required: enable GHAS now, or approve the alternative public-release sequence. This is a billing/security-governance decision.

### 1.3 Fix Dependabot governance

1. Disable automatic merging until required checks are genuinely green.
2. Audit why major updates to `actions/checkout` and `actions/setup-dotnet` merged despite a patch/minor-only workflow.
3. Ensure major updates never auto-merge.
4. Require the intended CI checks before any eligible update auto-merges.
5. Review the already-merged action upgrades for runner compatibility, inputs, permissions, and behavior. Revert them through ordinary commits only if evidence requires it.

### 1.4 Finish the source-license and package metadata cleanup

1. Delete `docs/legal/CONTRIBUTOR-LICENSE-AGREEMENT.md`; it is incompatible with Apache-2.0 core policy.
2. Create root `CONTRIBUTING.md` with an inbound contribution statement:
   - Contributions are submitted under Apache-2.0.
   - Contributors must have the right to submit them.
   - No separate CLA is currently required.
   - Historical ownership review is separate from this forward-looking contribution policy.
3. Create a short public `docs/legal/LICENSE-HISTORY.md` describing the post-split Apache core and the separate proprietary desktop product. Keep detailed ownership evidence private.
4. Rewrite `docs/legal/THIRD_PARTY_NOTICES.md` so it inventories only components distributed or used by the public core. Move desktop packaging, App.Avalonia, LibVLC runtime, installer, and product-native claims to the final gated repository.
5. Correct `IExportTierGate` comments to describe a neutral consuming-product policy seam.
6. Audit every publishable project:
   - Mark Trackdub-owned packages explicitly.
   - Apply `PackageLicenseExpression=Apache-2.0`, ownership metadata, repository URL, and project URL only to those owned packages.
   - Preserve actual metadata for third-party-derived or specially licensed packages, including `Trackdub.OnnxRuntime.Dnnl.Native`.
7. Add a repository-boundary document and CI scanner that rejects desktop, cloud, old BSL/GPL/dual-license claims, and prohibited product-policy wording in active core files. Allow third-party notices, model policies, and historical records only where context requires the terms.

### 1.5 Publish-readiness verification

Before making `Trackdub` public:

- Fresh clone succeeds with the documented SDK.
- CI and CodeQL gates pass according to the chosen security policy.
- No private endpoints, activation routes, pricing policy, customer data, secrets, staging remotes, or desktop source appears in the tree.
- LICENSE, NOTICE, package metadata, model policy, and core third-party notices are internally consistent.
- Root README describes the public core, not the desktop product.
- Branch protection, Actions permissions, Dependabot behavior, CodeQL settings, and security alerts are configured intentionally.

Create an annotated, immutable core pin only after these checks pass. The gated repository will use that exact full SHA.

## Phase 2: Rebuild `Trackdub-gated` With Fresh History

### 2.1 Preserve the current staging repository

1. Record the current staging repository's full SHA and repository settings.
2. Rename or mirror it to a private staging archive such as `Trackdub-gated-Staging-Archive`.
3. Preserve it for recovery only; do not continue active work there.
4. Recreate `Mesililm-LLC/Trackdub-gated` as a private fresh-root repository with no imported branches, tags, commits, or Git metadata.

The canonical `Trackdub-gated` URL is intentionally reused for the final private product repository.

### 2.2 Build from an exact source inventory

Create a split manifest before copying:

- Source revision and source repository for every imported group.
- Destination: gated, public core, api.trackdub, archive-only, or excluded.
- Dependencies and tests that must move together.
- Ownership and license status for every native component, font, icon, sample asset, model manifest, and tool.

Use the old staging repository as a source of desktop behavior, not as proof that its current architecture is correct.

Explicitly exclude from final gated history:

- `src/Trackdub.Api/`
- `src/Trackdub.Worker/`
- `src/Trackdub.WebhookDelivery/`
- cloud tests
- `services/activation-service/`
- `frontend/`, website, Docker, infra, and cloud deployment material
- copied public-core source after the submodule integration is complete
- all old migration/quarantine history

### 2.3 Establish the standalone desktop root

The final desktop repository owns its build configuration. It must include, at minimum:

- `Trackdub.Desktop.slnx`
- `Directory.Build.props`, `Directory.Build.targets`, `Directory.Packages.props`, `global.json`, and `NuGet.config`
- `.gitmodules`, `.gitignore`, `.gitattributes`, `.editorconfig`
- `README.md`, `AGENTS.md`, `LICENSE.md`
- `docs/index.md`, `docs/repository-policy.md`, and a desktop-specific documentation taxonomy
- `THIRD_PARTY_NOTICES.md` and `docs/legal/EULA.md`

Initially consume `Trackdub` through a Git submodule at `external/Trackdub`, pinned to the verified core SHA. Use project references into that pinned checkout. Do not use source copies or sibling-directory references as the official build path. Migrate to versioned public packages later.

### 2.4 Move the product policy cleanly

1. Inspect every current export-gate implementation and call site.
2. Select one final concrete implementation, for example `DesktopExportTierGate`.
3. State its project, namespace, constructor dependencies, DI registration, and test coverage.
4. Move product constants, duration checks, watermark decision, Free/Pro mapping, and upgrade text with it.
5. Keep public core export behavior unrestricted when no `IExportTierGate` is supplied.
6. Create a desktop-owned test-double project and use normal project references. Do not link source files from public core test projects.
7. Confirm free exports retain their existing watermark and duration behavior, while Pro behavior remains unchanged.

### 2.5 Desktop legal, native, and release work

1. Add a proprietary notice that does not falsely call historically public source confidential.
2. Add an EULA placeholder for internal builds only. Public production release CI must fail while it contains placeholder or counsel-review markers.
3. Build a desktop-specific third-party notice inventory:
   - FFmpeg build variant, source URL, hash, effective license, and whether downloaded or bundled.
   - libmpv, LibVLC, eSpeak-NG, fonts, icons, models, and all installed native packages.
4. Track manifests, hashes, acquisition scripts, placeholders, and notices. Do not track downloaded DLL, dylib, SO, FFmpeg, libmpv, or model payloads.
5. Ensure release artifacts include the proprietary notice, approved EULA, Apache core LICENSE and NOTICE, and relevant third-party notices.

Gate: a clean clone of the fresh gated repository builds through the pinned core submodule, runs desktop tests, and preserves end-user product behavior.

## Phase 3: Cross-Repository Validation and Publication

### STATUS: Phase 3 partly complete (2026-07-31) — validation steps run, ownership/protection steps need you

Step 5 (make `Trackdub` public, keep `Trackdub-gated` private) was already
done independent of this checklist. Steps 1–3 (dependency direction,
boundary scan, package metadata) were run this round — see
`docs/archive/phase-3-validation-report.md` for the full detail; Step 1 is
partially verified (reverse dependency direction not independently rerun
against fresh Trackdub-gated clone), Steps 2–3 passed with caveats noted
there (package metadata is source-only, no `dotnet` SDK available to inspect
built artifacts). Step 4 (branch
protection) could not be verified or configured for **either** repository —
no branch-protection/ruleset tool was available in the session that ran this
check. Private repositories support branch protection and repository rulesets
on GitHub Pro, Team, and Enterprise plans, while GitHub Free excludes these
features for private repositories — this session doesn't know `Mesililm-LLC`'s
plan, so `Trackdub-gated`'s gap is recorded as unverified, not accepted. Needs a human
to check `Settings → Branches` on both `Trackdub` and `Trackdub-gated`
directly. **Step 7 itself (recreating relevant historical issues from the
archive in their new canonical repositories) was not attempted this round —
don't read it as done.** What actually happened alongside this validation
work was unrelated general hygiene: the public core's issue tracker had
accumulated unrelated cross-project issues (a different project's Nushell
plugin-registry backlog, nothing to do with the open-core split or the
archive migration); the genuinely-Trackdub ones were kept, the rest closed
`not_planned`. That's noise removal, not the archive-to-new-repo issue
migration Step 7 describes.

1. Verify the dependency direction from independent clean clones:
   ```text
   Trackdub-gated -> Trackdub
   Trackdub -> no Trackdub-gated dependency
   ```
2. Scan both repositories for duplicate core implementations, old monorepo project references, stale repository URLs, unapproved license claims, untracked binary payloads, and accidentally retained cloud code.
3. Validate package metadata and generated artifacts, not only source trees.
4. Configure branch protection and required checks for both repositories.
5. Keep the core repository private until its release gates pass. Then make only `Mesililm-LLC/Trackdub` public.
6. Keep `Trackdub-gated` private.
7. Recreate only current, relevant issues in their new canonical repositories. Historical issues and PRs remain in the archive unless they contain context not captured elsewhere.

## Phase 4: Activation Port and Product Release Readiness

This phase is independent of the repository split but required before replacing the legacy activation service.

1. Define the desktop-to-API contract: endpoint URLs, JWT claims, public-key rotation, activation/reactivation/status/deactivation, offline grace behavior, and error semantics.
2. Implement server-side activation in `api.trackdub`.
3. Port behavior from the legacy service with parity tests and migration strategy.
4. Keep legacy activation operational until the replacement is deployed, migrated, and verified against real product flows.
5. Keep signing keys and production secrets only in the server-side trust boundary.

## Completion Criteria

The split is complete only when all of the following are true:

- Public `Trackdub` has no desktop product, cloud service, activation server, pricing policy, or copied native payloads.
- Public `Trackdub` is Apache-2.0 with accurate LICENSE, NOTICE, package metadata, contribution terms, model policy, and third-party notices.
- Public core CI is green under the chosen CodeQL/GHAS policy.
- `Trackdub-gated` has one fresh root, remains private, owns the sole concrete export gate, and builds from a pinned public-core dependency.
- Desktop product behavior is unchanged for Free and Pro users.
- Production releases cannot ship with a placeholder EULA.
- The old monorepo and prior gated staging history are preserved only in private archives.
- No component claims to be complete based only on architecture or documentation; builds, tests, artifacts, and remote checks provide the evidence.
