# Trackdub Investor Deck Brief

Use this brief alongside the combined `docs/**/combined.md` files when asking Copilot to generate an investor slide deck. It extracts the strongest investor-facing narrative from the technical docs and flags the business/financial sections you still need to supply.

---

## 1. Company and Product (from source docs)

**Mesilim LLC** is building Trackdub, a cross-platform, local-first AI dubbing workstation for Windows, macOS, and Linux. It is a desktop application, not a browser shell or a one-click SaaS demo.

The product spine is a staged editorial workflow:

```
media ingest
  -> audio preparation
  -> optional speech/noise split or dialogue/stem separation
  -> VAD
  -> diarization
  -> ASR
  -> transcript confidence review
  -> translation
  -> glossary / terminology hints
  -> speaker and voice assignment
  -> TTS
  -> timing reconciliation
  -> optional audio-level lip alignment
  -> preview mix
  -> export
  -> optional visual dubbing / generated portrait branches
```

The product philosophy: a reliable workstation where every pipeline stage produces durable artifacts, users can inspect and edit intermediate results, and the UI tells the truth about what the model actually did.

---

## 2. Problem and Opportunity (inferred from the product spine)

**Problem:** AI dubbing today is either a black-box cloud service that sends raw media files off-device with limited editability, or a collection of disconnected scripts and Python environments that require technical expertise.

**Opportunity:** There is room for a local-first, editorial-grade dubbing workstation that keeps content on the user's machine, supports professional review and correction, and routes to cloud providers only with explicit consent.

**[NEED INPUT: insert market size / TAM / SAM / SOM and target customer segments.]**

---

## 3. Solution and Key Differentiators (from architecture and strategy)

- **Local-first by default:** Media and inference run on the user's hardware. Cloud lanes exist but are gated by explicit consent and disclosure.
- **Stage-aware workflow:** Each stage has defined inputs, outputs, status, warnings, and artifacts. Projects are resumable and inspectable.
- **Model manifest governance:** Every real model or provider route is described in a machine-readable manifest with fields for model_id, provider_id, task, format, expected runtime, input/output contracts, and commercial/noncommercial permissions. 40 bundled ONNX models are tracked in `bundled-models.manifest.json`.
- **Hardware-aware inference:** The runtime stack probes and falls back across TensorRT-RTX, DirectML, Windows ML, MIGraphX, and CPU. Unsupported acceleration does not block a workflow; the app explains the fallback.
- **License-aware by design:** Only ONNX models with verified commercial licenses are used. Unknown or non-commercial licenses are treated as unsafe and blocked.
- **Cross-platform desktop shell:** Avalonia on .NET 10, with Windows, macOS, and Linux as first-class targets.
- **Testable architecture:** Fake-backed application tests, clean dependency direction (Domain depends on nothing), and immutable execution snapshots for pipeline stages.

---

## 4. Technical Moat (from architecture and specs)

- **ONNX-first inference with manifest-driven runtime planning.** The system does not hardcode model choice; it resolves provider availability, checksums, and execution providers at runtime.
- **Artifact preservation and resumability.** Completed stages are not recomputed. Failed or skipped stages leave prior artifacts in place with explicit reasons.
- **Honest readiness states.** Provider registered, model downloaded, stage ran, and stage succeeded are tracked as separate states. The UI does not claim "GPU ready" when only a DLL is present.
- **Cloud egress visibility and consent.** A dedicated design spec (G3) documents what data leaves the machine for each cloud provider and stage. Full-media cloud dubbing is scaffolded but not exposed without a consent gate.
- **Engineering discipline:** File-scoped namespaces, `sealed` where extension is not intended, `Async` suffix on async methods, immutable records in Domain, and architecture tests that enforce dependency direction.

**[NEED INPUT: if you have patents, exclusive model partnerships, or first-mover claims, add them here.]**

---

## 5. Roadmap and Status (from LONGTERM-ROADMAP.md and audits)

The roadmap is organized into milestone arcs:

- **M0-M7:** Foundation (repo structure, model manifest policy, SQLite project spine, media ingest, runtime planning, transcript generation, translation). Status: historical foundation.
- **M8-M16:** Workstation spine (video playback, segment editing, diarization, transcript confidence, Kokoro TTS, timing reconciliation, Spleeter separation, preview mix, voice cloning, export, hardware acceleration). Status: mostly implemented; current source is the source of truth.
- **M17+:** Advanced features (managed glossary analyzers, Japanese/Chinese/Arabic tokenization, visual dubbing / generated portrait branches).

Long-term exploration lanes include speech/noise split, dialogue enhancement, overlap speech rescue, premium dialogue/music separation (AudioShake Local SDK), cloud speech isolation (Auphonic), and first-party trained separation models. These are tracked but not claimed as shipped.

**[NEED INPUT: add current release version, shipped features, and any customer/user milestones.]**

---

## 6. Commercial Strategy (from open-core split plan and legal docs)

- **Public core:** `Mesililm-LLC/Trackdub` is Apache-2.0 and contains the reusable engine, SDK, CLI, pipeline, inference, media, licensing mechanisms, tooling, and tests.
- **Private product:** `Mesililm-LLC/Trackdub-gated` is the proprietary desktop product with the Avalonia shell, branding, installer, signing, activation, and tier gating.
- **Future private services:** `api.trackdub`, `portal.trackdub`, and `trackdub.com` repositories are reserved for server-side activation, product API, portal, and marketing site.
- **Contributor licensing:** A contributor license agreement lets Mesilim LLC relicense contributions under commercial terms.

This open-core model allows developer adoption through the public core while the commercial product carries tiered features, activation, and support.

**[NEED INPUT: add actual pricing tiers, target ACV/ARPU, and any pilot or paid conversion data.]**

---

## 7. Operations and Trust (from operations and audits)

- CI builds and tests run on Windows, macOS, and Linux.
- CodeQL and Dependabot are configured; security audit documents are maintained.
- Model manifests are validated in CI for schema, SHA-256 alignment, and commercial license gating.
- No secrets, customer data, pricing policy, or activation server code lives in the public core.

---

## 8. Missing Investor Deck Context (you must provide)

The source docs are almost entirely technical. To make a complete investor deck, add the following:

1. **Market size and growth:** TAM, SAM, SOM, market growth rate, and demand drivers.
2. **Target customers:** Personas, industries, use cases, and customer pain quotes.
3. **Business model and pricing:** Subscription, perpetual license, usage-based, enterprise, marketplace, or freemium.
4. **Traction:** Users, revenue, pilots, waitlist, LOIs, GitHub stars, downloads, or retention metrics.
5. **Team and founders:** Backgrounds, relevant experience, and key hires.
6. **Competitive landscape:** Direct competitors, substitutes, and Trackdub's specific differentiation.
7. **Go-to-market strategy:** Sales motion, channels, partnerships, marketing, and community.
8. **Funding ask and use of funds:** Round size, runway, headcount, and key milestones the round enables.
9. **Financial projections:** 3-5 year revenue, burn, and path to profitability.

---

## 9. Suggested Slide Outline for Copilot

Use this outline with the combined docs and this brief:

1. **Title:** Trackdub — local-first AI dubbing workstation
2. **The Problem:** Current dubbing is either black-box cloud or fragmented scripts
3. **The Solution:** Stage-aware, editable, local-first desktop workstation
4. **Product Demo / Pipeline:** ingest -> ASR -> translation -> TTS -> mix -> export
5. **Market Opportunity:** [NEED INPUT]
6. **Business Model:** [NEED INPUT]
7. **Traction:** [NEED INPUT]
8. **Competitive Advantage:** local-first, manifest governance, cross-platform Avalonia stack, open-core
9. **Roadmap:** M8-M16 shipped / M17+ advanced lanes
10. **Team:** [NEED INPUT]
11. **Financials / Ask:** [NEED INPUT]
12. **Appendix:** architecture diagram, model governance, open-core split

---

## 10. Source Documents Used

- `docs/combined.md` — documentation taxonomy and repository policy
- `docs/architecture/combined.md` — architecture, design goals, runtime stack
- `docs/strategy/combined.md` — product spine, roadmap, model governance
- `docs/specs/combined.md` — bundled model manifest, cloud egress consent design
- `docs/plans/combined.md` — open-core split and LibVLC playback plan
- `docs/legal/combined.md` — licensing and contributor terms
- `docs/operations/combined.md` — CI, deployment, and security practices
- `docs/audits/combined.md` — quality and security audit findings

This brief was generated from the repository docs. Replace all `[NEED INPUT]` placeholders before generating the final deck.
