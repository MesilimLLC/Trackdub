# Product

<!-- impeccable:product-schema 1 -->

## Platform

desktop

Confirmed. Avalonia on Windows, macOS, and Linux. This is not `web`, `ios`, `android`, or `adaptive`. The schema list does not have a desktop value. Record `desktop` anyway, because mapping it onto one of those four would be false.

## Users

Confirmed. The primary user is an editor dubbing media on their own machine: run the pipeline, inspect each stage, correct the result, and export. A second audience is a developer integrating the public SDK or CLI. The editor is primary. Target industries, pricing, and market size remain unconfirmed. The investor brief still marks those as missing.

## Product Purpose

Inferred from README.md, not confirmed. Trackdub is a local-first dubbing workstation and a reusable inference engine. It automates language detection, speech recognition, translation, text-to-speech, timing reconciliation, and audio export, and it keeps each stage inspectable. Success, as the README states it, is a workstation where every pipeline stage produces durable artifacts, users can inspect and edit intermediate results, and the UI tells the truth about what the model actually did.

## Positioning

Inferred from README.md, not confirmed. The claimed gap is between black-box cloud dubbing that sends media off-device and fragmented Python or Conda setups. The mechanism a neighbor could not copy from a slogan alone: a stage-aware local pipeline with distinct readiness states (provider registered, model downloaded, stage ran, stage succeeded), artifact preservation on skip or failure, and cloud egress only with explicit consent.

## Operating Context

Inferred from this repository, not confirmed.

- This checkout is the public core (README links to MesilimLLC/Trackdub, Apache-2.0): engine, SDK, CLI, pipeline, inference, media, licensing mechanisms, and tests.
- Confirmed. The proprietary Avalonia desktop product (shell, branding, installer, signing, activation, and tier gating) lives in private Trackdub-gated. The design record for that shell belongs in that repo, not in this public core, and it is not the DubBench theme.
- api.trackdub, portal.trackdub, and trackdub.com are named as separate private repositories. No pricing, activation secrets, or customer data live here.
- The only visual implementation in this checkout is DubBench, an Avalonia benchmark harness under src/DubBench. It requests the dark theme by default and also ships a light theme dictionary. It is not the commercial workstation shell.
- The documented pipeline runs from media ingest through export, with optional separation, lip alignment, and visual-dubbing branches. Advanced lanes (managed glossary analyzers, some tokenization, visual dubbing, generated portraits) are tracked and are not claimed as shipped.
- An opt-in cloud render tier is a proof of concept with SnapSoft on AWS. README.md does not claim a completed run on the target AWS GPU instances.

## Capabilities and Constraints

Inferred from README.md, AGENTS.md, and the docs tree, not confirmed.

- .NET 11. Domain depends on nothing. Portable APIs by default.
- Local-first inference, with fallback across TensorRT-RTX, DirectML, Windows ML, MIGraphX, and CPU on the desktop, and CUDA plus classic TensorRT on Linux datacenter GPUs. Unsupported acceleration must not block a workflow, and the app must explain the fallback.
- Only ONNX models with a verified commercial license. An unknown license is unsafe and blocked. Bundled inventory: src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json.
- No end-user runtime dependency on Python, Conda, Docker, or the CUDA Toolkit, except the Rust-supervised inference worker under ADR-0016.
- Completed stages are not recomputed. A skipped or failed stage keeps prior artifacts and records an explicit reason.
- Honest readiness: registered, downloaded, ran, and succeeded stay separate states.
- Legal entity is Mesilim LLC. Trackdub is the product name. The docs tree sometimes writes the GitHub org as Mesililm-LLC; the README link uses MesilimLLC. The live org name was not re-checked in this pass.
- Confirmed scope: future design work follows the commercial desktop shell in Trackdub-gated, not DubBench.
- Still undecided, and must not be invented: target customer segments beyond the two audiences above, pricing, traction, market size, and any accessibility standard.

## Brand Commitments

Confirmed in the naming round. Not a palette. Values live in Trackdub-gated `DESIGN.md`.

- Name: Trackdub. Legal entity: Mesilim LLC.
- Visual system name: The Editorial Bench.
- Mark on disk in this repo: assets/trackdubicon.png. It was not inspected as the shell system.

## Evidence on Hand

- README.md: product statement, pipeline, open-core split, and status.
- docs/strategy/investor-deck-brief.md: the same narrative, plus explicit missing input for customers, pricing, traction, and market size.
- The commercial shell theme is in Trackdub-gated, `src/Trackdub.App.Avalonia/Resources/ThemeResources.axaml`. The design record belongs there.
- src/DubBench is a benchmark harness. It is not the shell.
- Absences future work must not fabricate: customer quotes, pricing, revenue, user counts, market size, testimonials.

## Product Principles

Inferred from the README and repository rules, not confirmed in interview.

1. Tell the truth about readiness. Registered, downloaded, ran, and succeeded stay distinct.
2. Keep media local unless the user consents to cloud egress.
3. Preserve artifacts. A skip or a failure does not destroy prior usable work.
4. Govern models by verified commercial license. Unknown is unsafe.
5. Keep the public core separable from the proprietary shell, activation, and pricing.
