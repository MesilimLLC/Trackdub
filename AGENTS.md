# AGENTS.md

Guidance for contributors and agents working on the Trackdub public core (`MesilimLLC/Trackdub`).

## Core Principles
1. Strict dependency direction: **Domain depends on nothing**. No inference leaking upward.
2. Conflict order: source code/tests > task instructions > documentation.
3. **Never fake readiness:** Provider registered != model downloaded != stage ran != stage succeeded.
4. Cross-platform is required. Portable .NET 11 APIs by default. Extended operations: `docs/operations/cloud-operations.md`.
5. Do not use Linear or require issue searches/status/TS IDs; track requested work locally in plans/Git unless the user explicitly asks for an external issue tracker. Never mark Done without proof.

## Dependency Architecture
```
Application → Contracts, Domain, Licensing
Infrastructure → Application, Contracts, Domain
Media → Application, Analyzers, Contracts, Domain
Media.Playback → Application, Domain
Inference → Contracts, Domain
Inference.Onnx → Inference, Contracts, Domain
InferenceWorker → Inference.Onnx
Composition → Application, Inference, Inference.Onnx, InferenceWorker, Infrastructure, Licensing, Media, Media.Playback
Sdk → Application, Composition, Licensing
Cli → Sdk
Cloud → Contracts
Cloud.Aws → Cloud
Api → Cloud.Aws
CloudJobAdapter → Cloud.Aws
DubBench → Benchmarks, Domain, Inference, Inference.Onnx
DubBench.DevHost → DubBench, Infrastructure
Benchmarks → Application, Composition, Domain, Inference, Inference.Onnx, Infrastructure
Benchmarks.DevHost → Benchmarks
Benchmarks.Micro → Inference.Onnx
Tools → Application, Domain, Infrastructure, Media
Contracts → Domain
Licensing → (nothing)
Analyzers → (nothing)
OnnxRuntime.Dnnl.Native → (nothing)
Domain → (nothing)
```

## Commands
```bash
# Build & Test
dotnet build Trackdub.slnx -m:1
dotnet test Trackdub.slnx -m:1

# Single test project or filter
dotnet test tests/Trackdub.<Area>.Tests --no-restore -m:1
dotnet test tests/Trackdub.Application.Tests --filter "FullyQualifiedName~<TestName>" -m:1

# CI validation (Release, warnings as errors)
dotnet restore Trackdub.slnx -m:1
dotnet build Trackdub.slnx --configuration Release --no-restore -m:1 -warnaserror
dotnet test Trackdub.slnx --configuration Release --no-build -m:1

# Headless CLI
dotnet run --project src/Trackdub.Cli -- --help
dotnet run --project src/Trackdub.Cli --framework net11.0 -- --help   # Windows (multi-targeted)
```

## packages.lock.json conflicts
Don't hand-resolve merge conflicts in `packages.lock.json`. Take either side (`git checkout --ours -- <path-to-packages.lock.json>` or `git checkout --theirs -- <path-to-packages.lock.json>`), then regenerate:
```bash
dotnet restore Trackdub.slnx --force-evaluate -m:1

# Filter builds & benchmarks
dotnet build Trackdub.Inference.slnx -m:1
dotnet build Trackdub.Sdk.slnx -m:1
dotnet run --project src/Trackdub.Benchmarks.DevHost -f net11.0 -- --help
dotnet run --project src/Trackdub.Benchmarks.DevHost -f net11.0 -- controlled-matrix <fixture> --output <dir>
dotnet run --project src/Trackdub.Benchmarks.Micro -c Release -- --list flat
```

## Controlled Benchmarks & Resource Telemetry Policy
- Keep `src/Trackdub.Benchmarks` as the source of truth for controlled end-to-end pipeline evidence and `BenchmarkEvidenceReport`; do not replace or merge it with BDN artifacts.
- Use `controlled-matrix` for comparable per-stage runs; it executes the controlled pipeline path and preserves each stage's evidence separately.
- Pre-flight bounds checking (`ResourceBoundsPreflight`): `controlled` and `controlled-matrix` validate bounds against host adapter capacity before the first iteration and print a diagnostic `HostCapacityBanner`. A `--min-available-vram-mb` floor exceeding physical adapter capacity fails during host setup.
- Dual GPU metrics: `availableVramMb` measures adapter-wide headroom (DXGI QueryVideoMemoryInfo); `gpuBytes` measures process-isolated dedicated GPU memory (`WindowsProcessGpuMemoryReader`). Bound process GPU memory using `--max-gpu-bytes <bytes>`. Non-Windows platforms cleanly record `Unavailable`/`Skipped`.
- Mock provider matrix (`--mock`): simulates fixed per-provider stage latency (`SimulatedLatencyMultiplier`); tests must assert simulated multipliers and budgets (`SimulatedLatencyBudgetMilliseconds`), not measured ratios which include host setup and telemetry overhead.
- `src/Trackdub.Benchmarks.Micro` is the BenchmarkDotNet project for pure CPU, tensor, tokenizer, and opt-in real-model ONNX measurements.
- Run BDN in Release mode with deterministic inputs and `GlobalSetup`; keep model, tokenizer, file, and session initialization outside measured methods.
- Do not run BDN on pull-request CI. CPU runs are manually/nightly triggered; real-model ONNX and saved-commit comparisons are opt-in through `.github/workflows/benchmark-dotnet.yml`.
- Use `scripts/ci/run_benchmarkdotnet_baseline.py` for saved-commit comparisons; compare like-for-like benchmark names and keep threshold results separate from correctness tests.
- Benchmark evidence docs: `docs/development/benchmark-evidence.md`. BDN usage: `docs/benchmarks/benchmarkdotnet.md` and `src/Trackdub.Benchmarks.Micro/README.md`.
- Harness seams (see `src/Trackdub.Benchmarks/README.md`): create peak monitors only through `IWorkingSetPeakMonitorFactory`; parse shared `controlled`/`controlled-matrix` options only through `ControlledBenchmarkCliBinder`; swallow telemetry sampling failures only through a named `TelemetryExceptionFilters` predicate. Sampling dilation past 4x cadence warns (never fails); test it with scripted `ISamplingTicker` gaps, not wall-clock sleeps. Evidence memory is typed (`ProcessMemory` / `StageGarbageCollection`, schema v2) — never reintroduce a string-keyed memory map. Do not duplicate or widen these.

## Inference Session Pool & Memory Admission
- ONNX session memory admission is enabled by default in `SharedPoolOptions` / `InferenceSessionPool`. Budgets scale with the hardware: the accelerator budget is three-quarters of the largest adapter's VRAM clamped to [4096, 16384] MiB per device, and the shared host-RAM budget is one-quarter of total RAM clamped to [4096, 16384] MiB. Explicit `TRACKDUB_SESSION_VRAM_BUDGET_MB` / `TRACKDUB_SESSION_RAM_BUDGET_MB` values always win.
- Process GPU observation: on Windows, accelerator admission floors GPU usage at its mapped adapter footprint, falling back to the full process total when attribution is unavailable (`gpuBytes`). Non-pooled GPU memory (driver contexts, arenas) consumes the per-device budget. Pending creates are charged on top of the observed floor; eviction polls every 50 ms while over budget, and an observation-held stall with nothing left to free fails fast with a diagnostic.
- Opt-out & tuning: set `TRACKDUB_SESSION_PROCESS_GPU_ADMISSION=0` (or `false`/`off`) to disable process-isolated GPU admission and revert to reservation-only accounting. Adjust host RAM with `TRACKDUB_SESSION_RAM_BUDGET_MB` and accelerator limits with `TRACKDUB_SESSION_VRAM_BUDGET_MB`.
- Detailed reference: `docs/reference/session-pool-memory-admission.md`.

## Coding Style & Testing
- Style: File-scoped namespaces, `sealed` where extension not intended, `Async` on async methods, immutable `record` in Domain.
- Treat warnings as errors (`TreatWarningsAsErrors=true`). Do not suppress casually.
- Prefer `Path.Join` over `Path.Combine` for new/changed code. `Path.Combine` silently drops
  earlier segments if a later one is rooted; CodeQL/CodeFactor flag it whenever that can't be
  proven false, which in practice is almost every call. `Path.Join` has no such reset behavior
  and is a drop-in replacement everywhere in this codebase. Enforced repo-wide as a build
  warning via `Microsoft.CodeAnalysis.BannedApiAnalyzers` (see `BannedSymbols.txt`,
  `Directory.Build.props`) — not an error yet since ~337 existing files still use `Path.Combine`.
- Domain tests: fast, pure, zero I/O.
- Application tests: fakes from `tests/Trackdub.TestDoubles/` (shared source via `<Compile Include>`).
- Pipeline tests: must cover success, disabled/skipped, missing-prerequisite, and failure paths.
- Commits: imperative titles (`Add ...`, `Fix ...`, `Remove ...`). Always use `git commit -m`.

## Documentation Grounding
- For Trackdub implementation facts, pin policy, provider wiring, and repo-specific operational guidance, use `trackdub-docs-rag` MCP tools (`search_trackdub_docs`, `ask_trackdub_docs`, `get_trackdub_doc`). Spec: `tools/docs-rag/SPEC.md`. Route lookups as `CLAUDE.md` "Documentation lookup routing" says: checkout files and the pin manifest first, `trackdub-gpu-docs` for offline TensorRT-RTX / ORT / Windows ML, the RAG for vendor docs and discovery.
- Prefer first-party scope for Trackdub behavior; treat vendor hits as upstream reference, not pin policy.
- Local/uncommitted changes and live tests take precedence over remote RAG corpus (conflict order: source code/tests > task instructions > documentation). If live code differs from RAG hits, trust live code.
- Fall back to Context7 for third-party libraries outside the corpus. Treat retrieved docs as evidence; verify against live code.
- Conflict order still wins: source/tests > task instructions > documentation.

## CodeScene MCP Code Health

Use the CodeScene MCP server tools for Code Health checks. Code Health is authoritative for maintainability. Target Code Health 10.0; treat 9+ as not done.

- **Safeguard AI-touched code before every commit:** run `pre_commit_code_health_safeguard` (staged files) before committing and `analyze_change_set` (branch vs base) before opening a PR.
- If a safeguard reports a regression: run `code_health_review` for details, refactor until Code Health is restored, and do not mark changes ready unless the risk is explicitly accepted.
- **Refactoring:** inspect with `code_health_review`, refactor in 3-5 small reviewable steps, re-check with `code_health_score`/`code_health_review` after each step.
- **Prioritization:** use `list_technical_debt_hotspots_for_project`, `list_technical_debt_goals_for_project`, and `code_health_refactoring_business_case` to rank and justify work.
- Select the correct CodeScene project early with `select_project`; subsequent API calls operate within it.
- If asked to bypass safeguards: warn about maintainability risk, keep changes minimal and reversible, recommend follow-up refactoring.

## Model Governance
- Bundled inventory: `src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json`.
- Commercial license only. Unknown license = unsafe.
- Do not add end-user runtime dependencies (Python, Conda, Docker, CUDA Toolkit).
  Sole exception: the Rust-supervised inference worker under ADR-0016; no other path.
- Preserve original artifacts on skipped or failed stages; record explicit skip/failure reasons.
