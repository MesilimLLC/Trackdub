# GitHub Actions Workflows

CI/CD lives in `.github/workflows/`. All jobs run on GitHub-hosted runners (`ubuntu-latest`,
`windows-latest`, `macos-latest`), except `trt-rtx-smoke.yml`, which needs the self-hosted Windows
runner.

`ci.yml`, `model-audit.yml`, `benchmark-report-validation.yml`,
`release-shipping-guard.yml`, `dependabot-auto-merge.yml`, and `opencode-review.yml` run
automatically on their triggers. `benchmark-dotnet.yml` runs on a nightly schedule. Everything else
is manual (`workflow_dispatch`) or PR-comment triggered:

Pull-request triggers are not restricted to pull requests whose base is `main`. `ci.yml`,
`model-audit.yml`, and `benchmark-report-validation.yml` also run for stacked pull requests based on
another branch; the workflows with a `paths` filter still run only when a matching path changes.
Pre-merge CodeQL analysis comes from GitHub's default code scanning setup, which runs for pull requests into `main`; a pull request based on another branch currently gets none.

| Command (PR comment) | Workflow |
|----------------------|----------|
| `/oc` or `/opencode` | OpenCode bot |

Manual dispatch still works:

```powershell
gh workflow run ci.yml
gh workflow run code-coverage.yml
gh workflow run benchmark-dotnet.yml -f mode=cpu
gh workflow run opencode.yml -f prompt="Summarize recent pipeline changes"
```

## Active workflows

### CI (`ci.yml`)

- **Trigger:** Push to `main`, any pull request, or manual (`workflow_dispatch`)
- **Jobs:**
  - **Verify Code Format** (`ubuntu-latest`): `dotnet format Trackdub.slnx --verify-no-changes`,
    scoped to the C# files changed against the PR base (the whole solution when no base SHA is
    available)
  - **Verify Repository Boundary** (`ubuntu-latest`): `python3 scripts/ci/check-repository-boundary.py`
  - **Controlled Matrix CPU Budget** (`ubuntu-latest`): restore `Trackdub.slnx`, build
    `src/Trackdub.Benchmarks.DevHost`, then `python3 scripts/ci/check_controlled_matrix_cpu_budget.py`
  - **Build & Test (Windows / Linux / macOS):** restore, build, and test `Trackdub.slnx`
    (Release, `-m:1`) on `windows-latest`, `ubuntu-latest`, and `macos-latest`
- **Timeout:** 45 minutes per build job (format 15, boundary 5, CPU budget 20 minutes)

### Release Restore & Shipping Guard (`release-shipping-guard.yml`)

- **Trigger:** Push to `main`, any pull request, or manual (`workflow_dispatch`)
- **Runs:** `ubuntu-latest`
- **Tasks:** locked-mode restore of `Trackdub.slnx` for Debug, then Release (the Release restore is
  deliberately the last restore before the build, so the Release build resolves Release-frozen
  assets), Release build and publish of `src/DubBench.DevHost`, then assertions that no
  `AvaloniaUI.DiagnosticsSupport` output and no `Avalonia.Diagnostics.Diagnostic.IsEnabled`
  runtimeconfig flag reaches `bin/Release`

### Dependabot Auto-Merge (`dependabot-auto-merge.yml`)

- **Trigger:** Automatically runs on `pull_request_target` for PRs targeting `main` when Dependabot opens or updates a PR
- **Jobs:**
  - **Auto-merge Dependabot PR:** fetches Dependabot metadata, approves the PR, and enables auto-merge with `--squash` via `gh` CLI. It ensures that once all required status checks/tests pass on the PR, the PR is automatically and safely merged.

### Model manifest audit (`model-audit.yml`)

- **Trigger:** push to `main` or any pull request touching `src/Trackdub.Inference/Runtime/ModelManifest/**`, `tools/ci/**`, or the workflow itself; weekly schedule (Mon 06:00 UTC); manual (`workflow_dispatch`)
- **Runs:** `ubuntu-latest`
- **Tasks:** `tools/ci/audit-bundled-model-manifest.py`

### BenchmarkDotNet (`benchmark-dotnet.yml`)

- **Trigger:** nightly schedule (03:17 UTC), or manual (`workflow_dispatch` with `mode` = `cpu`, `baseline`, or `onnx`)
- **Runs:** `ubuntu-latest`
- **Tasks:** CPU microbenchmarks (the scheduled run only executes when the `TRACKDUB_BENCHMARKDOTNET` repository variable is `true`), a saved-commit baseline comparison, or an opt-in real-model ONNX benchmark; results are uploaded as workflow artifacts

### Benchmark report validation (`benchmark-report-validation.yml`)

- **Trigger:** push to `main` or `benchmark`, or any pull request, touching `src/Trackdub.Benchmarks/**`, `tests/Trackdub.Benchmarks.Tests/**`, or the workflow itself; manual (`workflow_dispatch`)
- **Runs:** `ubuntu-latest`
- **Tasks:** restore/build `Trackdub.slnx`, run the report export tests, validate the mock-matrix JSON output, upload the `benchmark-report-validation` artifact

### OpenCode review (`opencode-review.yml`)

- **Trigger:** `pull_request` (opened/reopened/synchronize/ready_for_review)
- **Runs:** `ubuntu-latest`
- **Tasks:** calls the `tonythethompson/opencode-review-threads` reusable `opencode-review.yml` (`/review-pr`); posts one structured GitHub review (summary body plus inline resolvable threads) as `opencode-agent[bot]`. After a submitted review, later pushes only diff commits since that review, so unchanged findings are not re-raised
- **Gate (enforced inside the called reusable workflow, not in this repo):** same-repository PRs only, so fork PRs receive no secrets; author must be `OWNER`/`MEMBER`/`COLLABORATOR`/`CONTRIBUTOR`, non-draft, and not a bot; a `model` input is an explicit opt-in bypass
- **Secrets:** `OPENCODE_API_KEY` (zen: probe chain) and `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_TOKEN` (cf: Workers AI probe chain)
- **Requirement:** the OpenCode GitHub App must be installed on the repo for the `opencode-agent[bot]` token exchange; otherwise reviews fail or need `use-github-token: true` (posts as `github-actions[bot]`)

### OpenCode on demand (`opencode.yml`)

- **Trigger:** issue comments and pull request review comments (`created`), or manual (`workflow_dispatch` with `prompt`)
- **Runs:** `ubuntu-latest`
- **Tasks:** calls the `tonythethompson/opencode-review-threads` reusable `opencode-bot.yml` with the comment or supplied prompt; replies as `opencode-agent[bot]`
- **Gate (enforced inside the called reusable workflow; the caller also pre-filters commenter association):** non-bot commenters with `OWNER`/`MEMBER`/`COLLABORATOR` association (exact match) whose comment starts with or contains ` /oc` or ` /opencode`; `workflow_dispatch` requires `prompt`

### TRT RTX smoke (`trt-rtx-smoke.yml`)

- **Trigger:** Manual (`workflow_dispatch`)
- **Runs:** self-hosted Windows when `TRACKDUB_TRT_RTX_SMOKE == 'true'`
- **Steps:** restore, fetch the TensorRT RTX EP plugin (`tools/dev/Fetch-TrtRtxEp.ps1`), build the benchmarks, download the starter-pack models, then run `Trackdub.Benchmarks.DevHost --scope trt-rtx-smoke --provider trt-rtx`.
- **Model download:** a "Download starter-pack models" step runs `trackdub models download` for every target in `TrtRtxSmokeCatalog.StarterPackTurboGpu` (with variants `gpu-int4`, `quantized`, and `fp16` where the catalog defines them) before the smoke run. Without it the resolver skips every target. The download step and the smoke step share one model cache via job-level `TRACKDUB_CACHE_ROOT` and `TRACKDUB_MODEL_CACHE` (kept consistent so `TRACKDUB_MODEL_CACHE == TRACKDUB_CACHE_ROOT/model-cache`).
- **Failure surfacing:** the job no longer uses `continue-on-error`, so a non-zero smoke exit fails the run. The benchmark exits non-zero when every target is skipped ("TRT RTX smoke did not run any targets (all skipped)"), so a run that downloads nothing or skips everything now turns red instead of reporting green.

### Code coverage (`code-coverage.yml`)

- **Trigger:** Manual (`workflow_dispatch`) only; the push/PR triggers are disabled so the Linux CI test run is not duplicated
- **Runs:** `ubuntu-latest`
- **Tasks:** Coverlet on `Trackdub.slnx` (`-f net10.0`), ReportGenerator merge, `actions/upload-code-coverage`

## Secrets (review + audit)

| Secret | Purpose |
|--------|---------|
| `GRAPHITE_CI_OPTIMIZER_TOKEN` | Graphite CI optimizer in `model-audit.yml` |
| `OPENCODE_API_KEY` | OpenCode review/bot zen: probe chain (`opencode-review.yml`, `opencode.yml`) |
| `CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_API_TOKEN` | OpenCode review/bot cf: Workers AI probe chain |
| `CONTEXT7_API_KEY` / `OPENROUTER_API_KEY` | OpenCode review/bot model providers (`opencode-review.yml`, `opencode.yml`) |

## Local parity

```powershell
dotnet restore Trackdub.slnx -m:1
dotnet format Trackdub.slnx --verify-no-changes
dotnet build Trackdub.slnx -c Release --no-restore -m:1
dotnet test Trackdub.slnx -c Release --no-build -m:1
python3 scripts/ci/check-repository-boundary.py
```

Last updated: 2026-09-30
