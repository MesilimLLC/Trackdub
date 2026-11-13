## LibVLCSharp.Avalonia Playback Plan

### Summary
Add a LibVLC-backed playback path for the Avalonia shell so cross-platform video playback works on Windows, macOS, and Linux with app-bundled `libvlc` runtimes. Keep the first slice narrow: real play/pause/seek/duration/current-position, honest runtime-unavailable errors, and optional VLC-managed subtitle tracks or external sidecar subtitles. Do not attempt arbitrary Avalonia overlays on top of the video in this slice.

### Key Changes
- Add a new playback backend kind for the shared seam:
  - Extend `PlaybackBackendKind` with `LibVlc`.
  - Keep existing `MediaFoundation`, `FfmpegFallback`, and `LibMpvFallback` values for compatibility; do not repurpose them.
  - Update backend labels/warnings so `LibVlc` reports as the selected backend and missing-runtime failures are explicit.

- Keep WinUI playback untouched; make Avalonia choose LibVLC through shell-specific composition:
  - In the WinUI app, continue using the existing `PlaybackCapabilityProbe` + `DefaultPlaybackBackendFactory`.
  - In `Trackdub.App.Avalonia`, override the playback registrations with:
    - an Avalonia-specific capability probe that prefers `LibVlc` for local video playback,
    - a LibVLC backend factory,
    - a LibVLC host control instead of WinUI `MediaPlayerElement`.
  - This avoids forcing WinUI’s Media Foundation assumptions onto the Avalonia shell.

- Add a new LibVLC playback implementation in `Trackdub.Media.Playback`:
  - `LibVlcPlaybackBackend : IPlaybackBackend, IPlaybackHostAwareBackend, IPlaybackRateBackend, IPlaybackVolumeBackend, IDisposable`
  - Host type should be Avalonia/LibVLC-specific and owned by the Avalonia app, but the backend stays in `Media.Playback`.
  - `TryAttachHost` should bind the backend to the Avalonia `VideoView`.
  - `OpenAsync` should:
    - validate the source path,
    - initialize `LibVLC`/`MediaPlayer`,
    - load local file media,
    - surface missing runtime or open failure as a warning in `PlaybackSnapshot`,
    - not fabricate `IsLoaded` when VLC failed to prepare media.
  - `GetSnapshotAsync` should report duration, current position, play/pause state, rate, and warning/error text.

- Add bundled VLC runtime resolution for the Avalonia app:
  - Introduce a small runtime locator/service that resolves the bundled `libvlc` root relative to the Avalonia app output.
  - App-bundled runtime is the only supported first-slice mode.
  - Missing or malformed runtime bundle must produce a structured unavailable/error state, not silent fallback.
  - Keep this entirely separate from inference/model readiness.

- Add Avalonia player hosting and shell wiring:
  - Replace the current placeholder playback area with a LibVLC Avalonia `VideoView` host.
  - Keep the current view model-driven commands for play, pause, seek, and playback status.
  - Do not layer sibling Avalonia controls over the native video host.
  - If simple in-video UI is needed, place it inside the LibVLC `VideoView` container only.

- Subtitle scope for this slice:
  - Support only VLC-managed subtitles:
    - embedded subtitle tracks when present,
    - optionally an external sidecar subtitle file generated from existing project subtitle data.
  - Do not port the current WinUI arbitrary CC overlay behavior yet.
  - If no usable subtitle track/sidecar is available, playback still works and the UI reports subtitles as unavailable.

- Subtitle integration shape:
  - Reuse existing project subtitle data/export logic where possible instead of inventing a parallel Avalonia subtitle model.
  - Generate a temporary SRT sidecar from the current transcript or translated cues only when the user explicitly enables subtitles in the Avalonia shell.
  - Feed that sidecar to VLC for rendering; keep the subtitle toggle honest about what source is active.
  - Clean up temporary sidecars with the same artifact/persistence discipline already used elsewhere; do not overwrite source media.

- Package and dependency changes:
  - Add central package pins in `Directory.Packages.props` for the LibVLCSharp Avalonia packages and any runtime package(s) needed for bundled desktop distribution.
  - Keep package additions scoped to the Avalonia app and playback layer.
  - Record packaging/license notes for VLC redistribution in repo docs where third-party runtime packaging is already described.

### Interfaces / Public Shape
- Shared playback seam:
  - Add `PlaybackBackendKind.LibVlc`.
  - Update `PlaybackControlViewModel` backend label/warning mapping for the new kind.
- Avalonia composition:
  - Add shell-specific playback registrations in `Trackdub.App.Avalonia` rather than changing the Windows composition root behavior globally.
- No new Avalonia-only project/session model.
- No changes to inference/provider readiness contracts.

### Test Plan
- Playback abstraction tests:
  - `PlaybackService` opens through `LibVlc` when the Avalonia probe selects it.
  - Missing VLC runtime returns `IsBackendAvailable = false` or loaded=false with explicit warning text, depending on final backend contract choice.
  - Play, pause, seek, rate, and volume flow through the LibVLC backend.
  - Open failure surfaces warning text without pretending playback loaded.

- Avalonia shell tests or focused integration checks:
  - Build `src/Trackdub.App.Avalonia`.
  - Verify project open/create still works with the LibVLC host present.
  - Verify playback state updates in the view model when media is loaded and when runtime is unavailable.

- Subtitle scenarios:
  - Embedded subtitle track present: toggle enables VLC subtitle display.
  - External generated sidecar available: VLC loads it and reports subtitles active.
  - No subtitles available: toggle is disabled or reports unavailable without affecting playback.

- Manual smoke matrix:
  - Windows Avalonia shell with bundled runtime: MP4/H.264 plays.
  - macOS Avalonia shell with bundled runtime: MP4/H.264 plays.
  - Linux Avalonia shell with bundled runtime: MP4/H.264 plays.
  - Missing bundled runtime on any platform: explicit unavailable/error state.
  - Non-MP4 format that VLC can handle: plays if runtime is valid, without changing project/media persistence behavior.

### Assumptions
- First-slice supported playback target is “reliably play local video,” not “match WinUI overlay composition.”
- App-bundled VLC runtime is acceptable for Windows/macOS/Linux packaging and licensing review.
- In-video subtitles may be VLC-rendered, but arbitrary Avalonia overlays over the video are out of scope for this slice.
- H.264/MP4 remains the minimum-confidence smoke target even if VLC can play more formats.



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

`Mesililm-LLC/Trackdub` is private, unarchived, and has a fresh-root history. Its root commit was `8bac38a8fa3e0343c5d10c558b64d269d15e7828`; follow-up commits have added CI and cross-platform fixes. The current inspected head is `d590533b3220edf5211ce773b7620487c29eb1e7`.

The core contains:

- `Trackdub.slnx` and focused `.slnx` solutions.
- Domain, Contracts, Application, Infrastructure, Media, Playback, Inference, ONNX, Composition, SDK, CLI, Licensing, Benchmarks, Tools, Analyzers, and the DNNL native package project.
- 12 test projects covering those public projects.
- An unmodified Apache-2.0 `LICENSE` and the root `NOTICE`:
  ```text
  Trackdub
  Copyright 2024-2026 Mesilim LLC
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

# Agent Handoff: Trackdub Repository Split

Use this handoff to continue the Trackdub open-core split. Treat `docs/plans/open-core-split-continuation.md` as the canonical execution plan.

## Mission

Finish the public Apache-2.0 core release gates, then rebuild the private desktop product repository with fresh history and a pinned dependency on the public core.

Do not treat the current state as complete. The core is structurally separated but still private and not green in remote CI. The existing `Trackdub-gated` repository is source staging, not the final private product repository.

## Current Repositories

| Repository | Current role | Rule |
|---|---|---|
| `Mesililm-LLC/Trackdub` | Private public-core candidate | Work here first; do not make public until gates pass |
| `Mesililm-LLC/Trackdub-gated` | Private staging lane | Read-only source until the fresh-root rebuild phase |
| `Mesililm-LLC/Trackdub-Monorepo-Archive` | Private archived history | Never modify |
| `Mesililm-LLC/api.trackdub` | Private future server activation/API | Do not change during Phase 1 |
| `Mesililm-LLC/portal.trackdub` | Private portal | Out of scope |
| `Mesililm-LLC/trackdub.com` | Private marketing site | Out of scope |

## Facts Already Verified

- `Trackdub` is private, default branch `main`, with fresh-root history beginning at `8bac38a8fa3e0343c5d10c558b64d269d15e7828`.
- The latest inspected core commit is `d590533b3220edf5211ce773b7620487c29eb1e7`.
- The core uses `Trackdub.slnx`; do not reintroduce legacy `.sln` or `.slnf` assumptions.
- The public core has the canonical Apache-2.0 `LICENSE` and root `NOTICE`.
- `IExportTierGate` remains in public Contracts. The public core must not register a concrete product policy.
- Desktop watermark, duration limit, Free/Pro mapping, activation client, packaging, and release logic belong only in the final private desktop repository.
- The legacy activation service is not a desktop dependency. `api.trackdub` will eventually replace it on the server side.

## Known Remaining Core Defects

Do not rediscover these from scratch; verify and then address them.

1. The latest remote CI run failed.
   - macOS build/test and formatting passed.
   - Windows failed at `WavePcm16Tests.WriteSamplesAsync_move_failure_deletes_temp_file_and_preserves_destination`: the test expects exactly `Exception`, but Windows reported `UnauthorizedAccessException`.
   - Linux needs a fresh, inspectable rerun before it can be declared healthy.
2. CodeQL fails.
   - GitHub Advanced Security is not enabled for this private repository.
   - `.github/codeql/codeql-config.yml` still includes the removed `frontend` path.
3. Dependabot auto-merge merged major actions updates despite a patch/minor-only policy. The policy needs evidence-based repair.
4. Core documentation still has monorepo-era contradictions:
   - `docs/legal/CONTRIBUTOR-LICENSE-AGREEMENT.md` describes GPLv3 plus commercial dual licensing.
   - `docs/legal/THIRD_PARTY_NOTICES.md` contains desktop/App.Avalonia/LibVLC runtime claims.
   - `IExportTierGate` commentary says Application implements it.
   - `Directory.Build.props` lacks explicit opt-in package ownership/license metadata.

## Operating Rules

- Read `AGENTS.md`, `docs/repository-policy.md`, and the canonical plan before editing.
- Start from current `Trackdub/main` on a normal topic branch.
- Do not rewrite history, force-push, reset remote history, make the core public, or touch the archive.
- Do not modify `Trackdub-gated`, `api.trackdub`, `portal.trackdub`, or `trackdub.com` during the first task.
- Do not weaken tests merely to turn a failure green. Establish the behavior contract first.
- Do not add a no-op export gate to public core unless compilation proves it is necessary; report the dependency instead.
- Do not add per-file SPDX headers.
- Do not track native binary payloads, downloaded models, caches, build outputs, or machine-specific agent state.
- Do not copy cloud, activation-server, portal, website, Docker, or infrastructure code into the core or final desktop repository.
- Use `AGENTS.md` as the only substantive tracked agent-instruction file. Tool-specific prompt libraries and state stay untracked.

## Authorized First Task: Phase 1 Only

Complete the public-core release-readiness work described in Phase 1 of the canonical plan.

### Required work

1. Establish a green CI baseline across Windows, Linux, and macOS.
2. Resolve the Windows Media test using the intended contract, not a broad assertion.
3. Repair CodeQL configuration for the core-only tree and report the GitHub Advanced Security decision needed to make the gate enforceable.
4. Repair Dependabot auto-merge governance so major updates cannot merge automatically and checks are genuinely required.
5. Remove stale GPL/dual-license contribution wording and replace it with Apache-2.0 contribution guidance.
6. Separate public-core third-party notices from desktop packaging notices.
7. Correct `IExportTierGate` documentation and add package metadata only through explicit opt-in for Trackdub-owned packages.
8. Add or finish repository-boundary and stale-claim checks.
9. Run full local and remote validation.

### Stop conditions

Stop and ask for owner direction if any of these becomes necessary:

- Enabling GitHub Advanced Security or another paid GitHub security feature.
- Choosing an approved consumer EULA.
- Publishing packages to NuGet.org or GitHub Packages.
- Making `Trackdub` public.
- Recreating or renaming `Trackdub-gated`.
- Any deletion or move that would discard the only retained copy of source material.
- Any uncertainty about third-party redistribution rights or model licensing.

## Required Phase 1 Report

Before proposing a merge, report:

- Current core branch, base SHA, and proposed head SHA.
- Exact test behavior and the evidence behind the Media test fix.
- CI run URLs/IDs and job conclusions for Windows, Linux, macOS, format, and CodeQL.
- CodeQL scope after removing stale monorepo assumptions.
- The exact GHAS/publication decision still needed, if any.
- Dependabot policy changes and confirmation that major updates are blocked from auto-merge.
- Every deleted, moved, or rewritten legal/attribution document.
- Every packable project and its final ownership/license metadata.
- Results of repository-boundary, stale-license, secret, private-URL, native-payload, and generated-file scans.
- Any remaining blocker.

End with exactly one of:

```text
Phase 1 ready for owner review: Yes
```

or:

```text
Phase 1 ready for owner review: No
Blocker: <specific decision or failed verification>
```

## Next Phases After Approval

After explicit owner approval of Phase 1:

1. Preserve the current gated staging repository privately and recreate `Trackdub-gated` with a fresh root.
2. Build the final desktop repository from an exact file inventory.
3. Pin `external/Trackdub` to the verified core SHA via Git submodule and project references.
4. Move desktop/product policy into one concrete desktop export gate with desktop-owned tests and test doubles.
5. Rebuild packaging, legal artifacts, and production EULA release checks.
6. Validate clean clones and make the core public only after every agreed release gate passes.
7. Port activation-server behavior into `api.trackdub` separately; it does not block the repository split.

Real-Time A/B Voice Preview - Implementation Plan

Overview

Add instant voice candidate switching during preview, eliminating the need to re-generate entire audio tracks for
comparison.

Phase 1: Extend Domain Layer for Candidate Groups

1.1 Add TtsCandidateGroup record

File: src/Trackdub.Domain/Tts/TtsCandidateGroup.cs (new)

namespace Trackdub.Domain.Tts;

public sealed record TtsCandidateGroup(
    Guid Id,
    Guid ProjectId,
    Guid TranslatedSegmentId,
    int SegmentIndex,
    Guid SelectedCandidateId,
    DateTimeOffset CreatedAtUtc)
{
    public static TtsCandidateGroup Create(
        Guid projectId,
        Guid translatedSegmentId,
        int segmentIndex,
        Guid selectedCandidateId) =>
        new(
            Guid.NewGuid(),
            projectId,
            translatedSegmentId,
            segmentIndex,
            selectedCandidateId,
            DateTimeOffset.UtcNow);

    public TtsCandidateGroup SelectCandidate(Guid candidateId) =>
        this with { SelectedCandidateId = candidateId };
}

1.2 Extend TtsTake with candidate metadata

File: src/Trackdub.Domain/Tts/TtsTake.cs (modify)

Add to the record:

Guid? CandidateGroupId,
int CandidateIndex,  // 0, 1, 2 for A/B/C variants
TtsCandidateVariant Variant  // new enum

Add new enum: File: src/Trackdub.Domain/Tts/TtsCandidateVariant.cs (new)

namespace Trackdub.Domain.Tts;

public enum TtsCandidateVariant
{
    Primary = 0,      // Default/base generation
    Alternative1 = 1, // Slight parameter variation
    Alternative2 = 2  // Different parameter variation
}

1.3 Add repository interface

File: src/Trackdub.Application/Contracts/ITtsCandidateGroupRepository.cs (new)

using Trackdub.Domain.Tts;

namespace Trackdub.Application.Contracts;

public interface ITtsCandidateGroupRepository
{
    Task<TtsCandidateGroup?> GetBySegmentAsync(Guid translatedSegmentId, CancellationToken ct);
    Task SaveAsync(TtsCandidateGroup group, CancellationToken ct);
    Task DeleteAsync(Guid groupId, CancellationToken ct);
}

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Phase 2: Modify TTS Stage to Generate Multiple Candidates

2.1 Add candidate generation configuration

File: src/Trackdub.Application/Transcripts/StartTtsStageHandler.cs (modify)

Add to StartTtsStageRequest:

bool GenerateMultipleCandidates = false,
int CandidateCount = 3  // Generate 3 variants per segment

2.2 Modify SynthesizeSegmentAsync to generate candidates

File: src/Trackdub.Application/Transcripts/StartTtsStageHandler.cs (modify)

In the synthesis loop, when GenerateMultipleCandidates is true:

if (request.GenerateMultipleCandidates)
{
    var candidates = new List<TtsTake>();
    for (int i = 0; i < request.CandidateCount; i++)
    {
        TtsSynthesisRequest candidateRequest = CreateVariantRequest(
            originalRequest,
            i,  // candidate index
            voice);

        TtsTake candidate = await SynthesizeSegmentAsync(
            request,
            stageRunId,
            translatedSegment,
            sourceSegment,
            voice,
            voiceCloneReference,
            reservedArtifactRelativePaths,
            candidateRequest,
            i,
            cancellationToken);

        candidates.Add(candidate);
    }

    // Create candidate group and select first as default
    TtsCandidateGroup group = TtsCandidateGroup.Create(
        request.ProjectId,
        translatedSegment.Id,
        translatedSegment.SegmentIndex,
        candidates[0].Id);

    await candidateGroupRepository.SaveAsync(group, cancellationToken);
    takes.AddRange(candidates);
}

2.3 Add variant request creation

File: src/Trackdub.Application/Transcripts/StartTtsStageHandler.cs (modify)

Add private method:

private TtsSynthesisRequest CreateVariantRequest(
    TtsSynthesisRequest baseRequest,
    int candidateIndex,
    VoiceCatalogEntry voice)
{
    // Vary parameters slightly for each candidate
    // Example: adjust stability, similarity, speed
    var variantOptions = baseRequest.Options with
    {
        // Adjust based on candidateIndex
        // This depends on your TTS engine's parameter support
    };

    return baseRequest with { Options = variantOptions };
}

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Phase 3: Add Candidate Selection and Preview Coordination

3.1 Add candidate selection service

File: src/Trackdub.Application/Transcripts/TtsCandidateSelectionService.cs (new)

using Trackdub.Application.Contracts;
using Trackdub.Domain.Tts;

namespace Trackdub.Application.Transcripts;

public sealed class TtsCandidateSelectionService
{
    private readonly ITtsCandidateGroupRepository candidateGroupRepository;
    private readonly ITtsTakeRepository ttsTakeRepository;
    private readonly IArtifactStore artifactStore;

    public TtsCandidateSelectionService(
        ITtsCandidateGroupRepository candidateGroupRepository,
        ITtsTakeRepository ttsTakeRepository,
        IArtifactStore artifactStore)
    {
        this.candidateGroupRepository = candidateGroupRepository;
        this.ttsTakeRepository = ttsTakeRepository;
        this.artifactStore = artifactStore;
    }

    public async Task<IReadOnlyList<TtsTake>> GetCandidatesAsync(
        Guid translatedSegmentId,
        CancellationToken ct)
    {
        TtsCandidateGroup? group = await candidateGroupRepository
            .GetBySegmentAsync(translatedSegmentId, ct);

        if (group is null)
            return [];

        IReadOnlyList<TtsTake> allTakes = await ttsTakeRepository
            .GetBySegmentAsync(translatedSegmentId, ct);

        return allTakes
            .Where(take => take.CandidateGroupId == group.Id)
            .OrderBy(take => take.CandidateIndex)
            .ToList();
    }

    public async Task<TtsTake?> GetSelectedCandidateAsync(
        Guid translatedSegmentId,
        CancellationToken ct)
    {
        TtsCandidateGroup? group = await candidateGroupRepository
            .GetBySegmentAsync(translatedSegmentId, ct);

        if (group is null)
            return null;

        return await ttsTakeRepository.GetByIdAsync(group.SelectedCandidateId, ct);
    }

    public async Task SelectCandidateAsync(
        Guid translatedSegmentId,
        Guid candidateId,
        CancellationToken ct)
    {
        TtsCandidateGroup? group = await candidateGroupRepository
            .GetBySegmentAsync(translatedSegmentId, ct);

        if (group is null)
            throw new InvalidOperationException("No candidate group exists for this segment.");

        TtsCandidateGroup updated = group.SelectCandidate(candidateId);
        await candidateGroupRepository.SaveAsync(updated, ct);
    }

    public async Task<string?> GetSelectedCandidatePathAsync(
        Guid translatedSegmentId,
        CancellationToken ct)
    {
        TtsTake? selected = await GetSelectedCandidateAsync(translatedSegmentId, ct);

        if (selected?.ArtifactId is not Guid artifactId)
            return null;

        // You'll need to add a method to get artifact by ID
        // or modify this to use the existing artifact lookup
        return null; // TODO: Implement artifact lookup
    }
}

3.2 Extend preview coordinator for instant switching

File: src/Trackdub.Application/Transcripts/TtsDubPreviewCoordinator.cs (modify)

Add methods:

private readonly TtsCandidateSelectionService? candidateSelectionService;

public TtsDubPreviewCoordinator(
    IAudioPreviewTransport transport,
    IArtifactStore artifactStore,
    TtsCandidateSelectionService? candidateSelectionService = null)
{
    this.transport = transport;
    this.artifactStore = artifactStore;
    this.candidateSelectionService = candidateSelectionService;
}

public async Task SwitchCandidateAsync(
    Guid translatedSegmentId,
    int candidateIndex,
    CancellationToken ct)
{
    if (candidateSelectionService is null)
        return;

    IReadOnlyList<TtsTake> candidates = await candidateSelectionService
        .GetCandidatesAsync(translatedSegmentId, ct);

    if (candidateIndex < 0 || candidateIndex >= candidates.Count)
        return;

    TtsTake selected = candidates[candidateIndex];
    await candidateSelectionService.SelectCandidateAsync(
        translatedSegmentId,
        selected.Id,
        ct);

    // If currently playing this segment, switch instantly
    if (inSequenceMode && sequencePaths.Count > 0)
    {
        // Reload current segment with new candidate
        await ReloadCurrentSegmentAsync(selected, ct);
    }
}

private async Task ReloadCurrentSegmentAsync(TtsTake newTake, CancellationToken ct)
{
    // Get artifact path for new take
    string? artifactPath = await GetArtifactPathAsync(newTake.ArtifactId, ct);
    if (artifactPath is null || !File.Exists(artifactPath))
        return;

    // Stop current playback, reload, resume at same position
    double currentPosition = transport.CurrentPosition;
    await StopCoreAsync(ct);
    await transport.OpenAsync(artifactPath, ct);
    if (currentPosition > 0)
    {
        await transport.SeekAsync(currentPosition, ct);
    }
    await transport.PlayAsync(ct);
}

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Phase 4: Add UI for Real-Time A/B Switching

4.1 Add candidate selector view model

File: src/Trackdub.App.Avalonia/ViewModels/TtsCandidateSelectorViewModel.cs (new)

using System.Collections.ObjectModel;
using System.ComponentModel;
using Trackdub.Application.Transcripts;
using Trackdub.Domain.Tts;

namespace Trackdub.App.Avalonia.ViewModels;

public sealed class TtsCandidateSelectorViewModel : ObservableObject
{
    private readonly TtsCandidateSelectionService selectionService;
    private readonly TtsDubPreviewCoordinator previewCoordinator;
    private int selectedCandidateIndex;
    private Guid translatedSegmentId;
    private bool hasCandidates;

    public TtsCandidateSelectorViewModel(
        TtsCandidateSelectionService selectionService,
        TtsDubPreviewCoordinator previewCoordinator)
    {
        this.selectionService = selectionService;
        this.previewCoordinator = previewCoordinator;
        Candidates = new ObservableCollection<TtsCandidateViewModel>();
    }

    public ObservableCollection<TtsCandidateViewModel> Candidates { get; }

    public int SelectedCandidateIndex
    {
        get => selectedCandidateIndex;
        set
        {
            if (SetProperty(ref selectedCandidateIndex, value))
            {
                _ = SwitchToCandidateAsync(value);
            }
        }
    }

    public bool HasCandidates
    {
        get => hasCandidates;
        private set => SetProperty(ref hasCandidates, value);
    }

    public async Task LoadCandidatesAsync(Guid segmentId, CancellationToken ct)
    {
        translatedSegmentId = segmentId;
        Candidates.Clear();

        IReadOnlyList<TtsTake> candidates = await selectionService
            .GetCandidatesAsync(segmentId, ct);

        for (int i = 0; i < candidates.Count; i++)
        {
            Candidates.Add(new TtsCandidateViewModel(
                i,
                candidates[i],
                i == 0)); // First is default
        }

        HasCandidates = candidates.Count > 0;
    }

    private async Task SwitchToCandidateAsync(int index, CancellationToken ct = default)
    {
        if (index < 0 || index >= Candidates.Count)
            return;

        await selectionService.SelectCandidateAsync(
            translatedSegmentId,
            Candidates[index].TakeId,
            ct);

        await previewCoordinator.SwitchCandidateAsync(
            translatedSegmentId,
            index,
            ct);
    }
}

public sealed record TtsCandidateViewModel(
    int Index,
    TtsTake Take,
    bool IsDefault)
{
    public string Label => IsDefault ? "Default" : $"Variant {Index}";
    public Guid TakeId => Take.Id;
    public string? VoiceId => Take.VoiceId;
    public TtsCandidateVariant Variant => Take.Variant;
}

4.2 Add keyboard shortcut handling

File: src/Trackdub.App.Avalonia/ViewModels/PreviewMixViewModel.cs (modify)

Add:

private readonly TtsCandidateSelectorViewModel candidateSelector;

// In constructor:
candidateSelector = new TtsCandidateSelectorViewModel(
    selectionService,
    previewCoordinator);

// Add method:
public async Task HandleCandidateShortcutAsync(int candidateIndex, CancellationToken ct)
{
    if (candidateIndex >= 0 && candidateIndex < candidateSelector.Candidates.Count)
    {
        candidateSelector.SelectedCandidateIndex = candidateIndex;
    }
}

4.3 Add XAML UI for candidate selector

File: src/Trackdub.App.Avalonia/Views/PreviewMixView.xaml (modify)

Add to the preview panel:

<StackPanel Orientation="Horizontal" Margin="0,8,0,0"
            Visibility="{Binding CandidateSelector.HasCandidates, Converter={StaticResource BoolToVisibilityConverte
r}}">
    <TextBlock Text="Voice Candidates:" VerticalAlignment="Center" Margin="0,0,8,0"/>
    <ItemsControl ItemsSource="{Binding CandidateSelector.Candidates}">
        <ItemsControl.ItemsPanel>
            <ItemsPanelTemplate>
                <StackPanel Orientation="Horizontal"/>
            </ItemsPanelTemplate>
        </ItemsControl.ItemsPanel>
        <ItemsControl.ItemTemplate>
            <DataTemplate>
                <Button Content="{Binding Label}"
                        Command="{Binding DataContext.SelectCandidateCommand, RelativeSource={RelativeSource AncestorType=Page}}"
                        CommandParameter="{Binding Index}"
                        Margin="0,0,4,0"
                        Padding="8,4"/>
            </DataTemplate>
        </ItemsControl.ItemTemplate>
    </ItemsControl>
    <TextBlock Text="[1] [2] [3] to switch"
               Foreground="Gray"
               VerticalAlignment="Center"
               Margin="8,0,0,0"
               FontSize="11"/>
</StackPanel>

4.4 Add keyboard handler in main window

File: src/Trackdub.App.Avalonia/MainWindow.xaml.cs (modify)

protected override void OnKeyDown(KeyRoutedEventArgs e)
{
    base.OnKeyDown(e);

    // Handle 1, 2, 3 keys for candidate switching
    if (e.Key == VirtualKey.Number1)
    {
        _ = previewMixViewModel.HandleCandidateShortcutAsync(0, CancellationToken.None);
        e.Handled = true;
    }
    else if (e.Key == VirtualKey.Number2)
    {
        _ = previewMixViewModel.HandleCandidateShortcutAsync(1, CancellationToken.None);
        e.Handled = true;
    }
    else if (e.Key == VirtualKey.Number3)
    {
        _ = previewMixViewModel.HandleCandidateShortcutAsync(2, CancellationToken.None);
        e.Handled = true;
    }
}

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Phase 5: Add Persistence for Locked-in Selections

5.1 Implement candidate group repository

File: src/Trackdub.Infrastructure/Repositories/TtsCandidateGroupRepository.cs (new)

using Dapper;
using Trackdub.Application.Contracts;
using Trackdub.Domain.Tts;
using Trackdub.Infrastructure.Database;

namespace Trackdub.Infrastructure.Repositories;

public sealed class TtsCandidateGroupRepository : ITtsCandidateGroupRepository
{
    private readonly IDbConnectionFactory dbConnectionFactory;

    public TtsCandidateGroupRepository(IDbConnectionFactory dbConnectionFactory)
    {
        this.dbConnectionFactory = dbConnectionFactory;
    }

    public async Task<TtsCandidateGroup?> GetBySegmentAsync(
        Guid translatedSegmentId,
        CancellationToken ct)
    {
        const string sql = @"
            SELECT * FROM tts_candidate_groups
            WHERE translated_segment_id = @TranslatedSegmentId";

        using var connection = dbConnectionFactory.CreateConnection();
        return await connection.QuerySingleOrDefaultAsync<TtsCandidateGroup>(
            sql,
            new { TranslatedSegmentId = translatedSegmentId });
    }

    public async Task SaveAsync(TtsCandidateGroup group, CancellationToken ct)
    {
        const string sql = @"
            INSERT INTO tts_candidate_groups
                (id, project_id, translated_segment_id, segment_index, selected_candidate_id, created_at_utc)
            VALUES
                (@Id, @ProjectId, @TranslatedSegmentId, @SegmentIndex, @SelectedCandidateId, @CreatedAtUtc)
            ON CONFLICT (translated_segment_id)
            DO UPDATE SET
                selected_candidate_id = @SelectedCandidateId";

        using var connection = dbConnectionFactory.CreateConnection();
        await connection.ExecuteAsync(sql, group);
    }

    public async Task DeleteAsync(Guid groupId, CancellationToken ct)
    {
        const string sql = "DELETE FROM tts_candidate_groups WHERE id = @GroupId";

        using var connection = dbConnectionFactory.CreateConnection();
        await connection.ExecuteAsync(sql, new { GroupId = groupId });
    }
}

5.2 Add database migration

File: src/Trackdub.Infrastructure/Database/Migrations/xxxx_add_tts_candidate_groups.sql (new)

CREATE TABLE IF NOT EXISTS tts_candidate_groups (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    translated_segment_id TEXT NOT NULL UNIQUE,
    segment_index INTEGER NOT NULL,
    selected_candidate_id TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (selected_candidate_id) REFERENCES tts_takes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tts_candidate_groups_segment
    ON tts_candidate_groups(translated_segment_id);

CREATE INDEX IF NOT EXISTS idx_tts_candidate_groups_project
    ON tts_candidate_groups(project_id);

5.3 Extend TTS takes table

File: src/Trackdub.Infrastructure/Database/Migrations/xxxx_add_candidate_metadata_to_tts_takes.sql (new)

ALTER TABLE tts_takes ADD COLUMN candidate_group_id TEXT;
ALTER TABLE tts_takes ADD COLUMN candidate_index INTEGER DEFAULT 0;
ALTER TABLE tts_takes ADD COLUMN candidate_variant INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_tts_takes_candidate_group
    ON tts_takes(candidate_group_id);

5.4 Register services in DI

File: src/Trackdub.Composition/ServiceCollectionExtensions.cs (modify)

services.AddSingleton<ITtsCandidateGroupRepository, TtsCandidateGroupRepository>();
services.AddSingleton<TtsCandidateSelectionService>();

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Testing Strategy

Unit Tests

  1. Test TtsCandidateGroup creation and candidate selection
  2. Test TtsCandidateSelectionService candidate retrieval and selection
  3. Test variant request generation with different parameters

Integration Tests

  1. Test TTS stage with GenerateMultipleCandidates = true
  2. Test candidate group repository CRUD operations
  3. Test preview coordinator candidate switching

Manual Testing

  1. Generate a project with multiple candidates
  2. Play preview and press 1, 2, 3 to switch voices
  3. Verify selection persists after app restart
  4. Verify export uses selected candidates

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Rollout Plan

Phase 1 (Foundation)

  • Implement Phase 1 (domain layer)
  • Add database migrations
  • Write unit tests

Phase 2 (Backend)

  • Implement Phase 2 (TTS stage modification)
  • Implement Phase 3 (selection service)
  • Write integration tests

Phase 3 (Frontend)

  • Implement Phase 4 (UI and keyboard shortcuts)
  • Manual testing with real content

Phase 4 (Persistence)

  • Implement Phase 5 (repository and DI)
  • End-to-end testing

Phase 5 (Polish)

  • Add visual feedback for active candidate
  • Add candidate comparison metrics (duration, quality score)
  • Add "Generate Candidates" button to existing projects

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Estimated Effort

  • Phase 1: 2-3 days (domain + migrations)
  • Phase 2: 3-4 days (TTS stage + selection service)
  • Phase 3: 2-3 days (preview coordination)
  • Phase 4: 3-4 days (UI + keyboard handling)
  • Phase 5: 1-2 days (persistence + DI)

Total: ~11-16 days for a complete implementation

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Success Metrics

  1. Time savings: Users can compare 3 voice variants in under 1 minute vs. 15+ minutes currently
  2. Adoption: >50% of users with candidate generation enabled use the feature within first week
  3. Quality: Users report higher satisfaction with final voice selection
  4. Performance: Candidate switching happens within 100ms (imperceptible delay)


