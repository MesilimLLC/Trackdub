# Architecture — Trackdub public core

Repo root: the current checkout (`git rev-parse --show-toplevel`). Solution `Trackdub.slnx` (plus `Trackdub.Sdk.slnx`, `Trackdub.Inference.slnx`). .NET 10, `TreatWarningsAsErrors=true`, `RestorePackagesWithLockFile=true`, central package management in `Directory.Packages.props`.

Canonical sources: `AGENTS.md` (graph + commands), `docs/repository-policy.md` (org + doc taxonomy), `docs/architecture/ARCHITECTURE.md` (design rationale), `docs/decisions/ADR-0011-contracts-domain-coupling.md` (the one sanctioned coupling).

## Dependency graph (verbatim from `AGENTS.md`)

```
Application ────────► Contracts, Domain, Licensing
Infrastructure ─────► Application, Contracts, Domain
Media ──────────────► Application, Analyzers, Contracts, Domain
Media.Playback ─────► Application, Domain
Inference ──────────► Contracts, Domain
Inference.Onnx ─────► Inference, Contracts, Domain
Composition ────────► Application, Inference, Inference.Onnx, Infrastructure,
                       Licensing, Media, Media.Playback
Sdk ────────────────► Application, Composition, Licensing
Cli ────────────────► Sdk
DubBench ───────────► Benchmarks, Domain, Inference, Inference.Onnx
DubBench.DevHost ───► DubBench, Infrastructure
Benchmarks ─────────► Application, Composition, Domain, Inference,
                       Inference.Onnx, Infrastructure
Benchmarks.DevHost ─► Benchmarks
Benchmarks.Micro ───► Inference.Onnx
Tools ──────────────► Application, Domain, Infrastructure, Media
Contracts ──────────► Domain
Licensing ──────────► (nothing)
Analyzers ──────────► (nothing)
OnnxRuntime.Dnnl.Native ► (nothing)
Domain ─────────────► (nothing)
```

Project name vs directory name differ for two projects: `DubBench` lives in `src/DubBench/` (assembly `DubBench` — `AssemblyName` is set explicitly, it is *not* `Trackdub.DubBench`), and `DubBench.DevHost` in `src/DubBench.DevHost/`.

| Project | May depend on |
|---|---|
| `Trackdub.Domain` | nothing |
| `Trackdub.Licensing` | nothing (zero `ProjectReference`, BCL-only crypto, `net10.0` single-target) |
| `Trackdub.Analyzers` | nothing |
| `Trackdub.OnnxRuntime.Dnnl.Native` | nothing |
| `Trackdub.Contracts` | `Domain` only (ADR-0011) |
| `Trackdub.Inference` | `Contracts`, `Domain` |
| `Trackdub.Inference.Onnx` | `Inference`, `Contracts`, `Domain` |
| `Trackdub.Application` | `Contracts`, `Domain`, `Licensing` |
| `Trackdub.Infrastructure` | `Application`, `Contracts`, `Domain` |
| `Trackdub.Media` | `Application`, `Analyzers` (as analyzer reference only), `Contracts`, `Domain` |
| `Trackdub.Media.Playback` | `Application`, `Domain` |
| `Trackdub.Composition` | `Application`, `Inference`, `Inference.Onnx`, `Infrastructure`, `Licensing`, `Media`, `Media.Playback` |
| `Trackdub.Sdk` | `Application`, `Composition`, `Licensing` |
| `Trackdub.Cli` | `Sdk` |
| `Trackdub.Tools` | `Application`, `Domain`, `Infrastructure`, `Media` |
| `Trackdub.Benchmarks` | `Application`, `Composition`, `Domain`, `Inference`, `Inference.Onnx`, `Infrastructure` |
| `Trackdub.Benchmarks.DevHost` | `Benchmarks` |
| `Trackdub.Benchmarks.Micro` | `Inference.Onnx` |
| `DubBench` (assembly `DubBench`) | `Benchmarks`, `Domain`, `Inference`, `Inference.Onnx` |
| `DubBench.DevHost` | `DubBench`, `Infrastructure` |

Verify the graph against reality — it is asserted, not assumed:

```bash
python tools/ci/verify-dependency-graph.py
dotnet test tests/Trackdub.Architecture.Tests --no-restore -m:1
```

`DependencyGraphTests.AgentsMdDiagramMatchesEveryCsprojProjectReference` fails if the `AGENTS.md` diagram and the real `<ProjectReference>` sets diverge in either direction. **Editing one without the other is a hard build failure.**

## Non-negotiables

1. **Domain depends on nothing.** No Avalonia UI, SQLite, FFmpeg, Windows ML, ONNX Runtime, or machine-local paths in `src/Trackdub.Domain`.
2. **Conflict order:** source code/tests > task instructions > documentation.
3. **Never fake readiness:** provider registered != model downloaded != stage enabled != stage ran != stage succeeded.
4. **Cross-platform is required.** Portable .NET 10 APIs by default (`net10.0`; Windows legs opt in explicitly). Extended ops: `docs/operations/cloud-operations.md`.
5. **Model governance:** commercial license only, verified. Unknown license = unsafe.
6. **No end-user runtime dependencies** (Python, Conda, Docker, CUDA Toolkit) in the shipped path.

## Layer responsibilities

| Layer | Belongs | Must not contain |
|---|---|---|
| `Domain` | entities, value objects, enums, invariants, `StageNames`, `StageSkipReasonCodes`, `RuntimeStage`, `ExecutionProviderKind`, `ProjectArtifact`, `StageRunRecord`, `TransientFailureKind` | any project reference; UI, SQL, FFmpeg, ONNX |
| `Contracts` | cross-boundary DTOs and interfaces (`IArtifactStore`, `ReadinessState`, `StageReadiness`, `IPipelineReadinessService`) | domain logic or implementation |
| `Application` | use cases and orchestration: projects, sessions, pipeline stages, export, licensing, readiness service, runtime setup coordinator | UI rendering, ONNX implementation, raw SQL |
| `Infrastructure` | SQLite, filesystem, settings, logging, diagnostics, native-download plumbing | UI views, ONNX model code |
| `Media` | probe, extraction, normalization, muxing, timing, `WavePcm16` | AI model inference |
| `Inference` | provider-neutral abstractions: model registry, manifests, download planning, EP descriptors, `IRuntimePlanner`, variant selection | concrete `SessionOptions`/tensor code, UI types |
| `Inference.Onnx` | concrete ONNX Runtime / WinML sessions and mappers | UI, SQLite, broad pipeline orchestration |
| `Composition` | the single wiring root (`CompositionRoot.cs`, `TranscriptWorkspaceFactory`) | business logic |
| `Sdk` / `Cli` | headless entry points | logic that belongs in Application |
| `Benchmarks` / `Benchmarks.Micro` / `DubBench*` | controlled end-to-end evidence (Benchmarks) and BenchmarkDotNet micro (Micro) | product code |
| `Tools` | manifest builders, artifact inspectors, DB tools | product code |

## Why the edges are where they are

**Composition is the only wiring root.** Only `Composition` references both the abstractions (`Inference`) and their implementations (`Inference.Onnx`), plus `Infrastructure`, `Media`, `Media.Playback`, and `Licensing`. Every other project sees abstractions only. If `Application` referenced `Inference.Onnx`, no host could substitute a fake or an alternate provider, and `tests/Trackdub.Application.Tests` could not use `tests/Trackdub.TestDoubles/` without dragging ONNX into the test graph. Both the desktop shell and the CLI resolve the *same* registrations.

**Benchmarks reaches past the abstraction boundary, deliberately.** `Trackdub.Benchmarks` directly references both `Inference` and `Inference.Onnx` (see the graph), because it is a measurement harness that must exercise the real provider rather than a fake. That is the documented exception to "production wiring lives only in Composition" — the rule binds `Application`, `Sdk`, `Cli`, and the desktop shell, not the benchmark projects. Do not cite the rule as absolute when reviewing a `Benchmarks` change.

**Inference is provider-neutral.** `Inference` holds descriptors, ranking policy, and planning (`src/Trackdub.Inference/Runtime/Planning/`) but never constructs a session. Per `Runtime/ExecutionProviders/README.md`, session construction lives in `Inference.Onnx` behind `IExecutionProviderSmokeTester` and the session-factory abstractions. Consequence: the provider probe order is data (`Milestone5PlanningPolicy.SupportedProvidersThisMilestone`), not a chain of `if (OperatingSystem.IsWindows())` inside the planner.

**Media.Playback sits apart.** It references only `Application` and `Domain` — deliberately not `Contracts`, and not `Media`. It carries native playback interop (libmpv, LibVLCSharp, Media Foundation), `AllowUnsafeBlocks`, `WinNativeDepsManifest`, and its own multi-targeting (`net10.0;net10.0-windows10.0.19041.0`). Isolating it keeps native-playback dependencies and unsafe code out of the graph that every other consumer walks, and lets `Composition` opt into playback without `Media` dragging it along.

**`Media` → `Analyzers` is an analyzer reference, not a code edge.** In `src/Trackdub.Media/Trackdub.Media.csproj` it is `OutputItemType="Analyzer" ReferenceOutputAssembly="false"`. It appears in the diagram because the compile graph includes it; no runtime type flows that way.

## Build and test commands

```bash
# Fast loop
dotnet build Trackdub.slnx -m:1
dotnet test Trackdub.slnx -m:1

# Scoped
dotnet test tests/Trackdub.<Area>.Tests --no-restore -m:1
dotnet test tests/Trackdub.Application.Tests --filter "FullyQualifiedName~<TestName>" -m:1

# CI-equivalent (see context/standards/validation-gates.md)
dotnet restore Trackdub.slnx -m:1
dotnet build Trackdub.slnx --configuration Release --no-restore -m:1 -warnaserror
dotnet test Trackdub.slnx --configuration Release --no-build -m:1

# Format
dotnet format Trackdub.slnx --verify-no-changes

# Headless CLI
dotnet run --project src/Trackdub.Cli -- --help
dotnet run --project src/Trackdub.Cli --framework net10.0 -- --help   # Windows multi-target

# Benchmarks
dotnet run --project src/Trackdub.Benchmarks.DevHost -f net10.0 -- controlled-matrix <fixture> --output <dir>
dotnet run --project src/Trackdub.Benchmarks.Micro -c Release -- --list flat
```

`-m:1` on every `dotnet` invocation: single-node MSBuild avoids the shared output races this repo hit repeatedly.

## Cross-repo relationship

The desktop application is **not** in this repo. It lives in the sibling `Trackdub-gated` checkout (`Mesililm-LLC/Trackdub-gated`) and consumes this repo as a **pinned, read-only git submodule** at `external/Trackdub` (`.gitmodules`: `url = https://github.com/Mesililm-LLC/Trackdub.git`, `branch = main`). The gated shell references `Application`, `Composition`, `Domain`, `Licensing`, `Media.Playback`, and `Sdk` from the submodule; nothing may be edited inside `external/Trackdub`.

- Gated-side agent system and desktop boundary: `../Trackdub-gated/.opencode/navigation.md`
- Gated-side pin bump procedure: `context/processes/submodule-pin-bump.md` in this file's sibling location under `../Trackdub-gated/.opencode/`
- Gated conflict order inserts `AGENT_CONTEXT.md` ahead of `AGENTS.md`; this repo's order is source/tests > task instructions > documentation.

## Related

- `context/standards/architecture-rules.md` — hard bounds and the boundary-violation catalogue
- `context/standards/validation-gates.md` — the CI-equivalent gate order
- `context/domain/inference-stack.md` — providers, manifest, readiness ladder
- `context/domain/terminology.md` — stage/status vocabulary