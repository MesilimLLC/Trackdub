# Trackdub Documentation

## Categories

- [decisions/](decisions/) — Architecture Decision Records (ADRs)
- [architecture/](architecture/) — System architecture and design
- [specs/](specs/) — Technical specifications and requirements
- [audits/](audits/) — Completed investigations and reports
- [operations/](operations/) — Operational procedures
  - [operations/cloud-gpu-container.md](operations/cloud-gpu-container.md) — Linux GPU container (`rtx` and `server` targets): publish, seed models, warm TensorRT engines
  - [operations/cloud-operations.md](operations/cloud-operations.md) — Headless and cloud environment facts
  - [operations/snapsoft-poc-acceptance.md](operations/snapsoft-poc-acceptance.md) — Contract for the earlier AWS-only job stack (outside the POC scope)
  - [operations/media-cleanup.md](operations/media-cleanup.md) — Attributed media-cleanup agent skill, Trackdub stage/artifact workflow, and verification boundaries
  - [operations/linear-workflow.md](operations/linear-workflow.md) — Linear source-of-truth + agent update loop (GitHub / Notion / Figma)
- [handoff/](handoff/) — Onboarding for external delivery teams
  - [handoff/snapsoft-onboarding.md](handoff/snapsoft-onboarding.md) — SnapSoft cloud render POC: SOW mapping, success criteria, model pack, provider allow-list, design decisions
- [development/](development/) — Developer procedures and guides
- [reference/](reference/) — Technical reference material
  - [reference/design-standards.md](reference/design-standards.md) — Canonical visual design tokens; Figma Design System + Qodo Design Review
  - [reference/gpu-execution-providers.md](reference/gpu-execution-providers.md) - GPU provider routing per build (TRT RTX → DirectML → CPU), pins, exclusions
  - [reference/tensorrt-rtx-ep-abi-plugin.md](reference/tensorrt-rtx-ep-abi-plugin.md) - TensorRT RTX EP ABI plugin install, registration, smoke
  - [reference/nvidia-afx-stubs.md](reference/nvidia-afx-stubs.md) - NVIDIA AFX readiness contract (probe-based; kill switch; no fake readiness)
  - [reference/nvidia-afx-wiring.md](reference/nvidia-afx-wiring.md) - AFX runtime layout, native behavior and verification status (works with a local SDK)
- [legal/](legal/) — Legal and attribution material
- [strategy/](strategy/) — Roadmap and strategic direction
- [plans/](plans/) — Active cross-cutting implementation plans

## Governance

See [repository-policy.md](repository-policy.md) for repository organization,
contribution, and governance details.
