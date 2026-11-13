# Phase 3 validation report

Status: complete for what tooling in this session could verify. Ad hoc audit
run 2026-07-31, not a recurring CI gate — see "Gaps" below for what still
needs a human or different tooling.

Cross-reference `open-core-split-continuation.md`'s Phase 3 section, which
this report fulfills.

## 1. Dependency direction

**Verified: `Trackdub` has no reference to `Trackdub-gated`.**

```shell
grep -rEln "Trackdub-gated|App\.Avalonia|DesktopExportTierGate|DesktopLicensingComposition" \
  --include="*.cs" --include="*.md" --include="*.props" --include="*.targets" \
  --include="*.csproj" --include="*.sln" --include="*.slnx" \
  src/ docs/ *.props *.targets *.csproj *.sln *.slnx
```

Two source-code hits, both deliberate seams, not violations:

- `src/Trackdub.Media.Playback/PlaybackAbstractions.cs:257` — comment
  explaining why a member is `public` rather than `internal` (the presenter
  that constructs it lives in `Trackdub.App.Avalonia`, a different assembly
  in a different repo).
- `src/Trackdub.Application/Properties/AssemblyInfo.cs:4` —
  `[assembly: InternalsVisibleTo("Trackdub.App.Avalonia")]`, the actual
  friend-assembly grant that seam requires.

`Directory.Packages.props` has one comment mentioning `App.Avalonia.Tests`
(a pending-migration note about xunit v2 vs FsCheck 3.x) — informational,
not a dependency.

Every other hit is in `docs/` (plans, specs, audits, architecture, decisions,
operations, development, strategy) — historical/internal-engineering
documents that legitimately predate or describe the split. Per
`docs/repository-policy.md`'s conflict order, historical plans are evidence,
not binding implementation truth, and are explicitly out of scope for the
project's own boundary scanner (see below).

Separately checked every `<ProjectReference>` in every `*.csproj` under
`src/`, `tests/`, and `tools/` (94 entries across 25 files): all resolve to
relative paths inside this repo's own tree (`..\..\src\...`,
`..\..\..\src\...`), none point outside it. No `.gitmodules` file or git
submodule/gitlink exists in this repo — expected, since `Trackdub` is the
consumed side of the pin, not the consumer.

**Confirmed direction from this repo's side: `Trackdub` has no reference to
`Trackdub-gated`, in source, docs, or project files.** The other half of the
direction — that `Trackdub-gated`'s project references genuinely resolve
into `external/Trackdub` and nowhere else — was not independently re-verified
in this session against a fresh clean clone of `Trackdub-gated`; it's known
from `Trackdub-gated/AGENTS.md`'s explicit dependency-direction diagram and
this session's own submodule-pin work on PR #10, not from a repeated grep
here.

## 2. Boundary scan

**Ran the repo's existing scanner rather than writing a new one:**

```console
$ python3 scripts/ci/check-repository-boundary.py
Repository-boundary scan passed.
```

This checks for stale monorepo-era license claims (GPL/dual-license wording
outside legitimate third-party-notice discussion) and desktop/cloud project
names bleeding into files a public consumer would read as authoritative. It
already special-cases `src/Trackdub.Media/Process/FfmpegAutoDownloader.cs`
(legitimately discusses GPL ffmpeg builds) and excludes `docs/plans/`,
`docs/decisions/`, `docs/audits/`, `docs/architecture/`, `docs/specs/`,
`docs/operations/`, `docs/development/`, and `tools/` as historical/internal.

No stale `Trackdub-Monorepo-Archive`, `BSL`, or `Business Source License`
references found anywhere in `docs/legal/`, `src/`, or root `.props` files.

## 3. Package metadata (source-only — no `dotnet` SDK available)

**Source declarations verified; generated artifacts not inspected.** No
`dotnet` SDK was available in this session (`which dotnet` found nothing),
so `dotnet pack` could not be run for either project and the resulting
`.nupkg`/`.nuspec` license metadata was never actually inspected. What
follows is a source-only check, not artifact validation — flagging that
distinction explicitly rather than overclaiming.

Only two projects are packable (with `PackageId` and license metadata set)
in this repo: `Trackdub.Cli` (via `PackAsTool=true`, which implicitly
enables packing — it doesn't itself set `IsPackable`) and
`Trackdub.OnnxRuntime.Dnnl.Native` (via explicit `IsPackable=true`). Several
test projects also declare `IsPackable` — explicitly `false` (e.g.
`tests/Trackdub.Application.Tests/Trackdub.Application.Tests.csproj:7`) —
which correctly excludes them from packing; they aren't a metadata gap.

| Project | Declared license | Why |
|---|---|---|
| `Trackdub.Cli` | `Apache-2.0` | Trackdub-owned, matches root `LICENSE` |
| `Trackdub.OnnxRuntime.Dnnl.Native` | `MIT` | Third-party-derived native package; kept distinct per Phase 1.4's requirement to preserve real upstream metadata rather than blanket-applying Apache-2.0 |

`Directory.Build.props` sets no repo-wide package metadata (no
`PackageLicenseExpression`, `RepositoryUrl`, etc.) — correct, since it
would incorrectly apply to `Trackdub.OnnxRuntime.Dnnl.Native`'s MIT license
if it did. Each packable project sets its own metadata explicitly instead.

**Follow-up:** run `dotnet pack` for both projects on a machine with the .NET
10 SDK and confirm the emitted `.nuspec`/`.nupkg` actually carries the
license expression declared in source — MSBuild property evaluation or a
packaging-target override could in principle diverge from what the
`.csproj` states, and that hasn't been checked.

## 4. Branch protection

**Unverified for both repositories — no tool available, and no assumption
made about what plan tier would allow.** The GitHub MCP server this session
had access to exposes no branch-protection or repository-ruleset read/write
tools (checked via broad keyword search across its full tool surface). This
earlier version of this report additionally claimed that `Trackdub-gated`
being private meant branch protection/rulesets weren't available "on
GitHub's private-repo tier" — that claim was wrong and has been removed.
Private repositories support branch protection and repository rulesets on
GitHub Pro, Team, and Enterprise plans, while GitHub Free excludes these
features for private repositories. This session doesn't know which plan
`Mesililm-LLC` is on, so neither repo's branch-protection status is recorded
as an accepted gap — both are simply unverified.

Action needed from you: check `Settings → Branches` (or `Settings → Rules →
Rulesets`) directly on **both** `Mesililm-LLC/Trackdub` and
`Mesililm-LLC/Trackdub-gated`, record the org's actual GitHub plan and each
repo's actual settings, and decide from there whether the existing
compensating controls in `Trackdub-gated` (PR-merge convention in its
`AGENTS.md`, `REVIEW.md` checklist, green-CI-before-merge) are sufficient on
their own or should be supplemented with real branch protection.

## Gaps / follow-ups

- **Branch protection state on both `Trackdub` and `Trackdub-gated`** —
  needs a human check with actual repo-settings access, see above. Do not
  assume either is or isn't possible based on visibility alone.
- **Stale GitHub labels on `Trackdub`** — `repo:quickshell`,
  `repo:numan-plugins`, `repo:numan-registry`, `repo:olive-studio` still
  exist on the label set (confirmed via `get_label`) from when this repo's
  issue tracker briefly served as a catch-all for unrelated projects. No
  label-delete tool was available this session; delete manually via
  `Settings → Labels`.
- **Closed cross-project issues #8–#14** — flagged in the original audit for
  your skim (closed but world-readable on a now-public repo); not re-checked
  in this pass.
