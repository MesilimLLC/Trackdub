# Product

<!-- impeccable:product-schema 1 -->

## Platform

undecided

Inferred, not confirmed. The interview timed out. README.md and docs/strategy/investor-deck-brief.md describe a cross-platform Avalonia desktop workstation for Windows, macOS, and Linux. That is not `web`, `ios`, `android`, or `adaptive`, so this field is left undecided rather than mapped onto a schema value the product does not match.

## Users

Inferred, not confirmed. docs/strategy/investor-deck-brief.md marks target customers as missing input. README.md describes the intended job as an editorial dubbing workflow: ingest media, run staged pipeline work, inspect and correct intermediate artifacts, and export. A second audience, also inferred, is a developer using the public SDK or CLI. Which audience is primary was not confirmed.

## Product Purpose

Inferred from README.md, not confirmed. Trackdub is a local-first dubbing workstation and a reusable inference engine. It automates language detection, speech recognition, translation, text-to-speech, timing reconciliation, and audio export, and it keeps each stage inspectable. Success, as the README states it, is a workstation where every pipeline stage produces durable artifacts, users can inspect and edit intermediate results, and the UI tells the truth about what the model actually did.

## Positioning

Inferred from README.md, not confirmed. The claimed gap is between black-box cloud dubbing that sends media off-device and fragmented Python or Conda setups. The mechanism a neighbor could not copy from a slogan alone: a stage-aware local pipeline with distinct readiness states (provider registered, model downloaded, stage ran, stage succeeded), artifact preservation on skip or failure, and cloud egress only with explicit consent.

## Operating Context

Inferred from this repository, not confirmed.

- This checkout is the public core (README links to MesilimLLC/Trackdub, Apache-2.0): engine, SDK, CLI, pipeline, inference, media, licensing mechanisms, and tests.
- The proprietary Avalonia desktop product (shell, branding, installer, signing, activation, and tier gating) is described as living in private Trackdub-gated. That repository is not this checkout.
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
- Undecided, and must not be invented: primary user, target customer segments, pricing, traction, market size, any accessibility standard, and whether future design work in this repo should follow DubBench or the private desktop shell.

## Brand Commitments

Inferred, not confirmed. No binding visual direction was collected.

- Name: Trackdub. Legal entity: Mesilim LLC.
- Mark on disk: assets/trackdubicon.png, referenced by README.md. It was not inspected as a design system.

## Evidence on Hand

- README.md: product statement, pipeline, open-core split, and status.
- docs/strategy/investor-deck-brief.md: the same narrative, plus explicit missing input for customers, pricing, traction, and market size.
- assets/trackdubicon.png: icon referenced by the README.
- src/DubBench/Resources/DubBenchTheme.axaml and DubBenchThemeLight.axaml: the only token dictionaries in this checkout.
- Absences future work must not fabricate: customer quotes, pricing, revenue, user counts, market size, testimonials, and a visual system for the commercial desktop shell. That shell is not in this repo.

## Product Principles

Inferred from the README and repository rules, not confirmed in interview.

1. Tell the truth about readiness. Registered, downloaded, ran, and succeeded stay distinct.
2. Keep media local unless the user consents to cloud egress.
3. Preserve artifacts. A skip or a failure does not destroy prior usable work.
4. Govern models by verified commercial license. Unknown is unsafe.
5. Keep the public core separable from the proprietary shell, activation, and pricing.
