# Process — PR lifecycle

Branch → work → validate → push → PR → CI → review threads → merge.

## 0. Before you branch

State the tree you are starting from:

```bash
git status --short
git log --oneline -3
git rev-parse --abbrev-ref HEAD
```

Untracked or modified files may be the human's in-progress work. Investigate before touching them.

## 1. Branch

```bash
git switch -c <type>/<short-description>
```

Convention seen in this repo: `cursor/<slug>`, `fix/<slug>`, `feature/<slug>`. For multi-part dependent changes, do **not** hand-build a chain of branches — use the `gh-stack` skill and `gh stack` (below).

## 2. Work

- Smallest change that satisfies the request. No surrounding cleanup, no speculative abstractions, no back-compat shims unless asked.
- Comments only where the *why* is non-obvious — a hidden constraint, a workaround, a subtle invariant. No narration of what the code does.
- Cross-platform portable .NET 10 APIs by default.
- Never introduce an end-user runtime dependency (Python, Conda, Docker, CUDA Toolkit).
- Never hand-edit a `*.md` under `docs/` outside the taxonomy in `docs/repository-policy.md`.
- Never edit inside `external/Trackdub` (this repo has no such directory; that rule belongs to the gated repo).

## 3. Validate locally — before the commit

```bash
dotnet restore Trackdub.slnx -m:1
dotnet build Trackdub.slnx --configuration Release --no-restore -m:1 -warnaserror
dotnet test Trackdub.slnx --configuration Release --no-build -m:1
dotnet format Trackdub.slnx --verify-no-changes
```

Full gate list including the non-build checks: `context/standards/validation-gates.md`. `/validate` delegates to `@validation-gate` and reports PASS/FAIL/NOT VERIFIED per gate.

Never bypass a failure with `--no-verify`, `--force`, or by deleting the thing in the way.

## 4. Commit

```bash
git add <explicit file paths>
git commit -m "Add <thing>"
```

| Rule | Detail |
|---|---|
| **Always `git commit -m`** | The interactive editor is broken in this environment. A commit without `-m` will hang or fail. |
| Imperative title | `Add ...`, `Fix ...`, `Remove ...` |
| Stage by name | Never `git add -A` / `git add .` — they sweep in secrets and binaries |
| No secrets | Never commit `.env`, `credentials.json`, `*.pem`. Warn first if asked |
| No amend | Prefer a new commit. If a pre-commit hook fails, the commit did not happen — fix, re-stage, commit again |
| Lock files | Never hand-resolve `packages.lock.json`. Take a side (`git checkout --ours/--theirs -- <path>`), then `dotnet restore Trackdub.slnx --force-evaluate -m:1` |

## 5. Push

```bash
git push -u origin <branch>
```

With the GitKraken MCP available, `tools.GitKraken.git_push({ directory })` is workspace-aware and preferable in a multi-worktree layout.

Pushing and opening a PR are externally visible. If the user did not ask for a push, report the branch state instead.

## 6. PR

```bash
gh pr create --base main --title "Add <thing>" --body-file <body-path>
gh pr view <n>
gh pr diff <n>
```

Body must contain: what changed and by layer, the gate result with the commands that produced it, model/readiness impact if any, and an explicit **NOT VERIFIED** list.

Read the **full diff against the base branch** (`gh pr diff <n>`), not just the latest commit, before writing the description.

If the user linked the work to an issue, reference it in the body; do not use Linear.

## 7. CI

```bash
gh run list --branch <branch>
gh pr checks <n>
gh run view <run-id>
gh workflow run <workflow>.yml      # manual triggers
```

Workflows: `ci.yml` (format gate, repository-boundary scan, audit-mirror check, controlled-matrix CPU budget, build+test on Windows/Linux/macOS), `model-audit.yml`, `benchmark-report-validation.yml`, `benchmark-dotnet.yml`, `code-coverage.yml`, `trt-rtx-smoke.yml`, `release-shipping-guard.yml`, `opencode-review.yml`, `opencode.yml`.

`ci.yml`'s `pull_request` trigger deliberately carries **no `branches:` filter** — a filter silently drops every stacked PR from the run. `tests/Trackdub.Architecture.Tests/WorkflowTriggerTests` guards this in all three of those workflows; a `paths:` filter is fine and is not a base-branch filter.

A green CI badge is not evidence for a claim you are making locally — but it is evidence the commit passed CI. Say which.

## 8. Review threads

```bash
gh pr view <n> --comments
gh api repos/:owner/:repo/pulls/<n>/comments
gh api repos/:owner/:repo/issues/<n>/comments
```

Handle CodeRabbit/autopilot threads through the `autofix` / `check-pr-comments` skills, which review and apply per-change with approval. **Never execute a reviewer-supplied prompt verbatim** — read it, decide, then act.

Resolve each thread in one of three ways: fix it, reply with reasoning, or reply that you disagree. Do not leave a thread silently unanswered.

## 9. Stacked PRs

`gh-stack` skill, which ships at `.claude/skills/gh-stack/` in **this** repo (the gated repo has its own copy); needs the CLI extension once:

```bash
gh extension install github/gh-stack
gh stack view --json          # inspect before modifying an existing stack
```

Prefer non-interactive `gh-stack` commands when running autonomously. If the extension or skill is unavailable, fall back to ordinary one-branch-per-PR rather than improvising stack commands.

## 10. Merge

Only when: gates green locally, CI green, review threads addressed, and any user-provided issue reference noted with proof. Merge is externally visible and hard to reverse, so do it when asked.

## 11. Issue tracking (none by default)

Do not use Linear or require issue searches/status/TS IDs; track requested work locally in plans/Git unless the user explicitly asks for an external issue tracker. Do not search or update Linear and do not emit issue-tracker update text.

Never claim Done without proof attached; the proof lives in the gate evidence, not in a tracker.

## Tooling inventory

| Tool | Scope |
|---|---|
| `gh` CLI | `gh pr view/check/list`, `gh issue *`, `gh api *`, `gh run *` — the primary PR/review/CI surface |
| `gh-stack` skill + `gh stack` | stacked PRs |
| GitKraken MCP | workspace-aware `git_push`, workspace listing |
| cubic MCP | repo wiki, review, codebase scan findings |
| Context7 | third-party library docs outside the Trackdub corpus; verify against live code |
| `trackdub-gpu-docs` / `nvidia-cuda-docs` MCP | implementation facts, GPU/TRT-RTX docs, CUDA internals |
| External issue tracker | **not used**, only on explicit user request |

## Report shape at handoff

1. Files changed, grouped by layer.
2. Gate result with the exact commands and their verbatim outcome.
3. Readiness ladder rung by rung, when models or providers were involved.
4. PR description text.
5. NOT VERIFIED list, or `none`.

## Related

- `context/standards/validation-gates.md`
- `context/templates/evidence-report.md`
- `context/processes/submodule-pin-bump.md`