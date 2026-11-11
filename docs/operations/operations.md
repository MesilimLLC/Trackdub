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

# macOS Deployment Notes

Guidelines and constraints for shipping Trackdub on macOS.

## Build & Publish

```bash
# Self-contained publish for macOS (required for users without .NET installed)
dotnet publish src/Trackdub.App.Avalonia -r osx-arm64 -c Release --self-contained
dotnet publish src/Trackdub.App.Avalonia -r osx-x64 -c Release --self-contained
```

## Critical Constraints

### IncludeNativeLibrariesForSelfExtract

**Never set `IncludeNativeLibrariesForSelfExtract = true` for macOS targets.**

This is incompatible with macOS app bundles. Native libraries (libmpv, ONNX Runtime, espeak-ng, etc.) must be loose files inside the bundle structure, not packed into a single-file executable. macOS code signing and Gatekeeper require individual files to be inspectable.

As of July 2026, this property is not set anywhere in the Trackdub solution (verified in Directory.Build.props and all .csproj files).

### Single UI Thread

macOS allows only one UI thread. Code that spawns splash screens or secondary dispatchers on background threads will crash. Avalonia's threading model enforces this natively, and Trackdub's current code (verified July 2026) uses only `DispatcherTimer` on the main thread with no secondary UI thread creation.

### Key Mapping

Avalonia natively maps `KeyModifiers.Control` to Cmd on macOS for standard gestures (Cmd+Z, Cmd+C, etc.). Trackdub's `MainWindowShortcutRouter` checks `KeyModifiers.Control` which Avalonia translates correctly on macOS. No XPF shim layer needed (Trackdub uses native Avalonia, not XPF).

Verify custom keybindings work on macOS when adding new shortcuts.

## Code Signing (Pre-Launch)

### When to Sign

- Not required for GitHub Releases (early adopters can bypass Gatekeeper)
- Required before marketing to non-technical users (Product Hunt, ads, etc.)
- Required for Mac App Store distribution

### How to Sign

1. Apple Developer Program ($99/year) required for Developer ID certificate
2. Sign individual native libraries first, then the bundle. Do NOT use `codesign --deep` (unreliable for complex bundles)
3. Native dylibs requiring individual signing:
   - `libmpv.2.dylib`
   - `libonnxruntime.dylib` (+ provider dylibs)
   - espeak-ng libraries
   - LibVLC libraries
   - FFmpeg libraries
4. Use Avalonia Parcel for cross-platform signing (P12 cert) and notarization
5. Notarization required for macOS 10.15+ (Catalina and later)

### Entitlements

If CoreML or GPU access requires JIT, add appropriate entitlements to `Entitlements.plist`:
```xml
<key>com.apple.security.cs.allow-jit</key>
<true/>
<key>com.apple.security.cs.allow-unsigned-executable-memory</key>
<true/>
```

Verify which entitlements ONNX Runtime's CoreML EP requires before release.

## CI Screenshots

macOS headless UI screenshots are captured in CI on `macos-14` (Apple Silicon):
- Set `CAPTURE_UI_SCREENSHOTS=1` environment variable
- Tests run with `--filter "FullyQualifiedName~Screenshot|FullyQualifiedName~Layout"`
- Screenshots uploaded as `macos-ui-screenshots` artifact on each CI run
- Same headless Skia renderer as Windows (UseHeadlessDrawing = false)

## App Bundle Structure

macOS requires `.app` bundle for distribution. Use Avalonia Parcel or manual structure:

```
Trackdub.app/
  Contents/
    MacOS/
      Trackdub              (main executable)
      libmpv.2.dylib
      libonnxruntime.dylib
      ...
    Resources/
      AppIcon.icns
    Info.plist
```

Bundle identifier: `ai.trackdub.app` (or similar reverse-DNS)

# Playback native runtime layout (libmpv / LibVLC)

Avalonia video preview uses composited frame backends. Native libraries must be present where
[`LibMpvRuntimeLocator`](../../src/Trackdub.Media.Playback/LibMpvRuntimeLocator.cs) and
[`LibVlcRuntimeLocator`](../../src/Trackdub.Media.Playback/LibVlcRuntimeLocator.cs) probe,
relative to `AppContext.BaseDirectory` (the published app folder).

**Primary strategy:** ship libmpv under `native/{rid}/` in publish output. LibVLC on Windows and
macOS comes from NuGet (`VideoLAN.LibVLC.Windows` / `VideoLAN.LibVLC.Mac`). On Linux, LibVLC is
typically the system package (`vlc` / `libvlc`).

**Fallback (optional):** startup bootstrap may download libmpv into the user profile when bundled
copies are missing (Windows `%LocalAppData%\Trackdub\native\{rid}`, macOS
`~/Library/Application Support/Trackdub/native/{rid}`). Do **not** rely on bootstrap for portable
releases or CI artifacts.

**Resolution order:** `LibMpvRuntimeLocator` checks `native/{rid}/` next to the app (walking up from
`AppContext.BaseDirectory`) **before** user-profile bootstrap paths. A stale download under
`%LocalAppData%` must not override a good bundled DLL beside the executable.

## Fetch scripts (dev / release)

| OS | Command | Output |
|----|---------|--------|
| Windows x64 | `.\tools\dev\Fetch-WinNativeDeps.ps1 -Architecture X64` | `native/win-x64/libmpv-2.dll` |
| Windows ARM64 | `.\tools\dev\Fetch-WinNativeDeps.ps1 -Architecture Arm64` | `native/win-arm64/libmpv-2.dll` |
| macOS | `pwsh ./tools/dev/Fetch-MacNativeDeps.ps1` (on Mac) | `native/osx-x64/libmpv.2.dylib` or `native/osx-arm64/...` |

> [!WARNING]
> These fetch scripts currently require `runtime/win-native-deps.manifest.json`, which is not present in
> the repository. Both scripts will fail immediately with "Missing native deps manifest" until the manifest
> is restored or generated. Native dependencies must be obtained manually in the interim.

When the manifest is available, `tools/dev/Fetch-WinNativeDeps.ps1` and `tools/dev/Fetch-MacNativeDeps.ps1`
fetch the artifacts on demand when they are missing; no workflow in `.github/workflows/` runs them
automatically.

## Windows x64 (published app folder)

```text
{AppContext.BaseDirectory}/
  Trackdub.App.Avalonia.exe
  native/
    win-x64/
      libmpv-2.dll          # also accepts libmpv-1.dll, mpv-2.dll, mpv-1.dll
  libvlc/                   # from VideoLAN.LibVLC.Windows
    win-x64/
      libvlc.dll
      libvlccore.dll
      plugins/...
```

Flat `{base}/libmpv-2.dll` is also probed when walking parent directories.

## macOS

```text
{AppContext.BaseDirectory}/
  native/
    osx-arm64/              # or osx-x64
      libmpv.2.dylib        # also libmpv.1.dylib, libmpv.dylib
  libvlc/                   # from VideoLAN.LibVLC.Mac
    osx-arm64/
      libvlc.dylib
      libvlccore.dylib
      plugins/...
```

User-cache fallback: `~/Library/Application Support/Trackdub/native/{rid}/`.

## Linux

**libmpv (optional bundle):**

```text
native/
  linux-x64/
    libmpv.so.2             # also libmpv.so.1, libmpv.so
```

**LibVLC (default: system install):**

- Bundled: `libvlc/linux-x64/libvlc.so` (if you ship a tree).
- Otherwise locator checks `/usr/lib`, `/usr/lib/x86_64-linux-gnu`, `/usr/lib/aarch64-linux-gnu`,
  `/usr/lib64` for `libvlc.so` / `libvlc.so.*`.

Install example: `sudo apt install vlc` (Debian/Ubuntu).

## Backend selection

[`AvaloniaPlaybackCapabilityProbe`](../../src/Trackdub.App.Avalonia/Playback/AvaloniaPlaybackCapabilityProbe.cs)
sets `PreferredBackend = LibMpv` when `ILibMpvRuntimeLocator.ResolveRuntimeLibraryPath()` is
non-null; otherwise `LibVlc`.

[`PlaybackService`](../../src/Trackdub.Media.Playback/PlaybackAbstractions.cs) retries LibVLC if
LibMpv open fails.

## Verification checklist

1. **Locators** — After publish or `dotnet run`, paths resolve:
   - `ILibMpvRuntimeLocator.ResolveRuntimeLibraryPath()` → non-null when `native/{rid}/` is present.
   - `ILibVlcRuntimeLocator.ResolveRuntimePath()` → non-null (bundled `libvlc/` or Linux system path).

2. **Probe** — Open a project with video; UI `PlaybackSummary` should show `Backend: LibMpv` when
   libmpv is bundled.

3. **Preview** — `IsPlaybackBackendAvailable` true; compositor delivers a frame (`HasVideoFrame` true,
   playback placeholder hidden).

On failures, check `%LOCALAPPDATA%\Trackdub\trackdub.log` (Windows) or platform log path.

## Debug logging

In **Debug** / `TRACKDUB_DEV_BUILD`, [`AvaloniaPlaybackComposition`](../../src/Trackdub.App.Avalonia/Playback/AvaloniaPlaybackComposition.cs)
logs resolved libmpv and LibVLC paths once at startup (when a logger is available).
