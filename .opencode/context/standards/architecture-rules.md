# Architecture rules

Hard bounds. These are not preferences; the dependency graph is asserted by `tests/Trackdub.Architecture.Tests` against the real `.csproj` files, and a violation fails the build.

## Rule 1 — never fake readiness

A capability claim must name exactly one rung and no more. Never collapse two.

| # | Rung | What it means | Proved by |
|---|---|---|---|
| 1 | provider registered | a DI registration exists | `CompositionRoot.cs` |
| 2 | runtime installed | the EP/runtime package is present on this machine | runtime locator / readiness service |
| 3 | model downloaded | the manifest-relative files exist on disk | manifest file existence |
| 4 | checksum verified | hashes match | `ModelHashVerifier`, `tools/ci/verify-manifest-hashes.py` |
| 5 | license recorded | manifest license metadata present | `bundled-models.manifest.json` fields |
| 6 | license reviewed | a human/automated review acted on it | review record, `docs/legal/` |
| 7 | commercial-use decision | `commercial_allowed` + `commercial_use_verified` satisfied | `ModelManifest.CommercialSafeMode`, `BundledModelManifestRegistry.CommercialSafeMode`, `DeepFilterNetModelPaths.IsCommercialSafe` |
| 8 | provider available | EP probe succeeded for this stage | `StageRuntimePlanStatus.Verified` |
| 9 | stage enabled | enabled in this run | `StageReadiness.Status`, run options |
| 10 | stage ran | execution occurred | `StageRunRecord` terminal state, not a skip |
| 11 | output usable | the artifact is valid downstream | `ProjectArtifact` + resume evaluator |

Rules:

- "Registered" is not "installed". "Downloaded" is not "checksummed". "Files present" is not "ran". "Ran" is not "succeeded". "Succeeded" is not "usable".
- `StageRuntimePlanStatus.Ready` means files present and integrity ok; `Verified` is strictly stronger (a smoke test executed the non-CPU EP). CPU never smoke-tests, so CPU legitimately stops at `Ready`. Do not report `Ready` as `Verified`.
- `ReadinessState.Unverified` (model metadata checked, provider path not verified) is non-blocking and must be surfaced, not silently promoted to `Ready`.
- A manifest entry existing is not a working model.
- A prior green run is not evidence about the current tree.
- **Rung 7 is documented but not code-enforced.** `CommercialSafeEvaluator` is named in `docs/legal/MODEL_LICENSE_POLICY.md` as the enforcement point, but no such type exists under `src/` or `tests/` at this revision — verify with `grep -rn "CommercialSafeEvaluator" src/ tests/`. The manifest fields (`commercial_allowed`, `commercial_use_verified`) are the only thing that is actually readable today, so a rung-7 pass is a **data** claim, not an enforced gate. If that grep comes back empty, say so rather than reporting commercial safety as enforced by code.

Anything not climbed is stated in the report as `NOT VERIFIED`. Missing evidence is reported in those words.

## Rule 2 — disabled ≠ skipped ≠ succeeded

- `DISABLED_BY_OPTION` — the stage never executed. No timing, no artifact, no evidence.
- `EXISTING_ARTIFACTS_VALID` — a resume. The current run produced nothing. Never counted as a speed sample or a pass.
- `PREREQUISITE_FAILED` — a skip caused by an upstream failure. Not a pass.
- `OPTIONAL_MODEL_DECLINED` — the user declined an optional model; the stage did not run.
- `NO_TRANSCRIPT_SEGMENTS`, `NO_SPEECH_REGIONS` — benign, still not successes.

All six are "benign" per `StageSkipReasonCodes.IsBenignSkip`. Benign means *intentional gating*, not success. Preserve original artifacts on skip or failure and record the reason code.

## Rule 3 — dependency direction

Full graph and per-project table: `context/domain/architecture.md`. The enforceable subset:

| Project | Rule |
|---|---|
| `Domain` | zero project references. No Avalonia, SQLite, FFmpeg, Windows ML, ONNX Runtime, machine-local paths, or cloud providers |
| `Licensing` | zero project references, BCL-only crypto (`System.Security.Cryptography`), single-target `net10.0` — asserted by `LicensingIsolationTests` |
| `Analyzers`, `OnnxRuntime.Dnnl.Native` | zero project references |
| `Contracts` | `Domain` and nothing else (ADR-0011) |
| `Inference` | `Contracts`, `Domain`. Never `Application`. Never constructs a session |
| `Inference.Onnx` | `Inference`, `Contracts`, `Domain`. Uses `Trackdub.Contracts.ApplicationContracts` for shared EP contracts, not the root Contracts namespace |
| `Application` | `Contracts`, `Domain`, `Licensing`. Abstract use cases only |
| `Infrastructure`, `Media` | `Application` for application-defined interfaces and DTOs only — they must not call use cases back into the coordinator |
| `Composition` | the only wiring root; the only project referencing both abstractions and implementations |
| `Sdk` → `Application`, `Composition`, `Licensing`. `Cli` → `Sdk` only | no shortcuts around the SDK |

Changing a project reference means changing the `AGENTS.md` diagram **in the same commit** — `DependencyGraphTests.AgentsMdDiagramMatchesEveryCsprojProjectReference` compares them in both directions. Acyclicity is asserted too.

Verify:

```bash
python tools/ci/verify-dependency-graph.py
dotnet test tests/Trackdub.Architecture.Tests --no-restore -m:1
```

## Rule 4 — model governance

Commercial license only, verified. **Unknown license is unsafe.** Enforcement lives in `ModelManifestLoader` at load time and in bundled-inventory tests, not in runtime filtering. Attribution obligations go to `THIRD_PARTY_NOTICES.md`.

Native binaries (DLLs, dylibs, SOs, FFmpeg, libmpv, EP bundles) are not tracked in the repo. Manifests, URLs, hashes, and acquisition scripts are.

## Rule 5 — no end-user runtime dependencies

Python, Conda, Docker, and the CUDA Toolkit may never become requirements for someone running the product. Developer tooling under `tools/` may use them; the shipped path may not.

## Rule 6 — cross-platform is required

Portable .NET 10 APIs by default. Windows-specific surface needs an explicit Windows leg or a guarded seam with a real fallback. CI builds and tests Windows, Linux, and macOS.

## Boundary violation catalogue

| Violation | Looks like | Detected by | Fix |
|---|---|---|---|
| Domain project reference | `<ProjectReference>` in `src/Trackdub.Domain/Trackdub.Domain.csproj` | `DomainHasNoProjectReferences`, `verify-dependency-graph.py` | move the type into Contracts or the consuming layer |
| Contracts over-coupling | `Contracts` referencing anything but `Domain` | `ContractsReferencesOnlyDomain` | drop the edge or move the type |
| Inference leaking upward | `Domain`/`Contracts`/`Application` referencing ONNX Runtime, `SessionOptions`, `OrtValue`, a concrete session type | review; caught indirectly by `DomainHasNoProjectReferences` when it reaches Domain | abstractions into `Inference`, implementations into `Inference.Onnx` |
| Application → Inference.Onnx | `Application` referencing a concrete provider | review; graph diff | declare the interface in Contracts, wire in Composition |
| Contracts namespace misuse | `using Trackdub.Contracts;` plus `ApplicationContracts`/`IRuntimePlanningPreferences`/`IInferenceSessionPoolEvictor` in `Inference.Onnx` | `InferenceOnnxDoesNotImportApplicationContractsNamespace` | use `using Trackdub.Contracts.ApplicationContracts;` |
| Wiring outside Composition | `new`-ing a concrete dependency in Application/Cli/Sdk, or a DI registration anywhere but Composition | review | move the registration to `CompositionRoot.cs`; both hosts must resolve the same one |
| UI types below Contracts | `Avalonia.*` or view-model shapes in Inference/Infrastructure/Application | review; assembly references | Inference returns DTOs; the shell binds to Contracts |
| SQL outside Infrastructure | raw SQL or a `Sqlite*` type in Application or a view model | review | repository in Infrastructure behind a Contracts interface |
| Stage-name literal | `StageRunRecord.Start(store, id, "my-stage", ct)` | `StageNameConsistencyTests.StageRunRecord_Start_never_receives_inline_string_literal` | use the `StageNames.*` constant |
| Incomplete stage-name sync | new constant in `StageNames.cs` not added to `KnownStageNameValues` | `KnownStageNameValues_covers_all_StageNames_constants` | add the value to the test's list |
| DirectML reintroduced | `Microsoft.ML.OnnxRuntime.DirectML` in `Directory.Packages.props` or any `.csproj`, or `CopyDirectMLAssetsToOutput` / `AddDirectMLAssetsToOutputItems` / `MirrorDirectMLAssetsForProjectReferenceCopy` targets | `WindowsOnnxRuntimePackagesUseWinMlCatalogProvider` | WinML package assets (ADR-0002) |
| DNNL flavor crossing | DNNL copy targets running outside the DNNL flavor, or WinML/GPU targets running inside it | `CompositionOnlyCopiesDnnlNativeAssetsForDnnlRuntimeFlavor`, `DnnlFlavorStripsStockOrtRuntimeAssetsFromPackageReferences` | gate every asset target on `TrackdubOrtRuntimeFlavor` |
| Missing RID assets | absent `runtimes/{win-x64,linux-x64,osx-x64}/native` or the provenance template in the DNNL package | `DnnlNativePackageDeclaresInitialRidAssetsAndChecksumProvenance` | populate the package; provenance template must carry `sha256` and `onnxruntime_version` |
| Arm64 DNNL build | running `Build-OnnxRuntimeDnnlNativePackage.ps1` on a non-x64 host | `DnnlNativePackageScriptRejectsArm64Hosts` | build x64 assets on an x64 machine |
| Dropped portable RID graph | a required `net10.0/{rid}` entry missing from a lock file | `OnnxLockFilePreservesPortableRuntimeIdentifierGraphs` (Windows-only) | `dotnet restore Trackdub.slnx --force-evaluate -m:1` on Windows |
| Licensing coupling | a `ProjectReference` or a crypto package in `Trackdub.Licensing`, or multi-targeting | `LicensingIsolationTests` | keep it standalone, BCL-only, `net10.0` |
| Stacked PRs dropped from CI | a `branches:` filter on a `pull_request:` trigger | `WorkflowTriggerTests` | remove the filter (a `paths:` filter is fine) |
| Hand-merged lock file | conflict markers or a lock diff no restore reproduces | `git diff --check "$BASE_REF"...HEAD -- '*packages.lock.json'` plus `git diff --check HEAD -- '*packages.lock.json'` + reproducibility | take a side, then `dotnet restore Trackdub.slnx --force-evaluate -m:1` |
| New `Path.Combine` sites | `RS0030` count grows in touched files | `grep -c "Path\.Combine"` (warning-only by design) | use `Path.Join` |
| Silent EP swap | a "GPU" claim that actually ran on CPU | `StageRuntimePlan.RequirePreferredExecutionProvider`, reported EP | set a hard pin, or report the EP actually used |

## Escalation

When a task explicitly asks for something that violates these rules, **stop and say so.** Do not silently comply, and do not implement it "temporarily".

1. Name the rule and the specific file/edge the request would break.
2. Name the gate that will fail (`verify-dependency-graph.py`, a named `DependencyGraphTests` test, `LicensingIsolationTests`, the Release `-warnaserror` build).
3. State the compliant alternative.
4. Ask for a decision. Record the answer, and if the answer is "do it anyway", state in the handoff which invariant was broken and that the gate was not run or failed.

The same applies in reverse: if the code and the request disagree, source code/tests outrank task instructions. Say what the code actually does.

## Related

- `context/domain/architecture.md` — the graph and layer map
- `context/domain/terminology.md` — the distinction pairs
- `context/domain/inference-stack.md` — readiness ladder detail
- `context/standards/validation-gates.md` — how the bounds are enforced in the gate
- `context/processes/adding-pipeline-stage.md` — the ordered procedure