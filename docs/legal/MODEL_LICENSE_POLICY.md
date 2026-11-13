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
  ]
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
