# Model License Policy

Trackdub must track model licensing explicitly. Current source manifests
are the authority for bundled models:

```text
src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json
```

Every model should have a manifest entry with:

```json
{
  "model_id": "example/model",
  "task": "asr | translation | tts | diarization | vad | separation",
  "engine_family": "example-engine",
  "capabilities": [ "example-capability" ],
  "tier": "fast | balanced | quality | experimental",
  "license": "MIT | Apache-2.0 | CC-BY-4.0 | CC-BY-NC-4.0 | custom | unknown",
  "source_url": "https://example.invalid/model",
  "revision": "model-revision",
  "sha256": "artifact-sha256",
  "commercial_allowed": true,
  "commercial_use_verified": true,
  "redistribution_allowed": true,
  "requires_attribution": false,
  "requires_user_consent": false,
  "voice_cloning": false,
  "aliases": [ "example" ],
  "root_path": "../../../../models/example",
  "benchmark_entry": "model.onnx",
  "variants": [
    { "alias": "default", "entry_path": "model.onnx" }
  ],
  "risk_acceptance": {
    "decision_record": "record-reference",
    "accepted_risks": [
      "licence indemnity to the licensor",
      "termination and deletion on licence end",
      "licensor's right to change terms",
      "unverified training-data provenance"
    ],
    "owner": "owner-name",
    "date": "YYYY-MM-DD"
  }
}
```

Rules:

- **There is no runtime `CommercialSafeMode` flag or user toggle.** The product
  ships only commercial-safe models. Lane enforcement is done at manifest
  authoring time via `commercial_allowed`, `commercial_use_verified`, and `lane`
  fields. The model inventory and runtime planner read these fields
  (`ModelManifest.CommercialSafeMode` is computed from `commercial_use_verified`);
  nothing consumes a runtime parameter.
- Non-commercial models (`lane: "non-commercial"`, `commercial_allowed: false`)
  must never appear in `bundled-models.manifest.json`. They belong in a separate
  research or dev-tooling manifest only.
- Unknown-license models (`license: "unknown"`) must be treated as unsafe for
  any commercial lane until review sets `commercial_use_verified: true`.
- `commercial_use_verified: true` means both commercial-use license confidence
  and artifact integrity are verified. It must not be true unless `sha256` is
  non-empty.
- `commercial_allowed: true` is not enough to make a model selectable. Use
  `commercial_use_verified` for the product gate.
- Demucs/HTDemucs is a non-commercial stem-separation route only. It may be
  explored in dev/research tooling, but it must never appear in the bundled
  manifest or be selected for any commercially-shipped pipeline path.
- Voice-cloning models must require explicit consent flow.
- Attribution-required models must appear in export/project metadata where appropriate.
- Model licenses are independent from the app license.
- A repository license is not enough. Check code license, pretrained weights,
  dependency models, model-card terms, and known training-data restrictions.
- Do not mark `commercial_use_verified: true` until the model/license
  combination has been reviewed for the intended product lane and the manifest
  has a real SHA-256 for the expected artifact.
- **Accepted-risk exception (owner decision, 2026-11-19).** A model whose
  licence explicitly grants commercial use, but whose training-data provenance
  cannot be verified, may be selected for a commercial lane only if all of the
  following hold. Without them it stays out of the bundled manifest and the
  downloadable catalog.
  - `license` names the specific licence where one exists (for example
    `NVIDIA-Open-Model-License`, `OpenMDW-1.1`, or `openrail++`); use `custom`
    only as a fallback when no established identifier for the licence exists.
    The licence text is bundled with the product wherever the artifact is
    distributed.
  - `commercial_allowed: true`, `redistribution_allowed: true`, and `sha256`
    is set to the exact artifact being shipped.
  - `commercial_use_verified` stays `false`. It keeps meaning the model was
    verified, and the exception is recorded separately.
  - `risk_acceptance` names the decision record and lists the accepted risks:
    licence indemnity to the licensor, termination and deletion on licence
    end, licensor's right to change terms, and unverified training-data
    provenance. It also names the owner who accepted them and the date.
  - The exception never applies to non-commercial or unknown-licence models,
    which remain banned from the bundled manifest.
  - The product gate (`ModelManifest.CommercialSafeMode`) must be extended to
    accept `risk_acceptance` in combination with `commercial_use_verified: false`
    (not the `custom` licence string alone) before any such model can be
    selected. Until that code change lands, no accepted-risk entry may be added
    to the bundled manifest or the downloadable catalog, because the download
    and model-manager selection paths key on `commercial_allowed` and
    `can_auto_download` and do not yet enforce this gate. Once the gate lands,
    those paths must enforce the same combination so that an accepted-risk
    entry cannot become selectable through them before the gate does.
  - **Gate status.** `ModelManifest.CommercialSafeMode` is now
    `commercial_use_verified || risk_acceptance`, and the loader rejects
    `risk_acceptance` unless the licence is commercial-capable,
    `commercial_allowed`, `redistribution_allowed` and a pinned `sha256` hold,
    and `commercial_use_verified` is false. Translation routing and
    provisioning, lip-synthesis, starter packs, DeepFilterNet and the model
    inventory now gate on that combination. The model download orchestrator
    refuses any model that is not `commercial_allowed` and either verified or
    risk-accepted, and the model inventory, runtime bootstrap status and the
    CLI and TUI download-all lists only offer auto-download for models that
    pass the same check. Importing a model file by hand is not gated, and the
    runtime planner still allows an explicitly requested alias outside its
    automatic ranking. Automatic choices (the planner's ranking, the default
    Opus translation pick and the translation router when no model is named)
    consider only verified models, so an accepted-risk model is opt-in by
    explicit alias and never selected automatically.
