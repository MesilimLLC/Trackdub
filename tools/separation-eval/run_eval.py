#!/usr/bin/env python3
"""Score a separation candidate on a generated corpus and write a results JSON.

The candidate runs out of process: this script prepares 44.1 kHz stereo PCM16 inputs (what the
shipped pipeline feeds Spleeter), writes a jobs file, runs the separator command, then scores the
outputs with metrics.py. The Spleeter command is `Trackdub.Benchmarks separation-eval`, which drives
the real engine and records wall time, RTF and peak working set per job.

Two domains are scored, on purpose:
  * separator domain (44.1 kHz mono, what the engine saw and wrote): the reconstruction gate;
  * reference domain (clip sample rate, stereo): bed leakage, bed damage, dialogue SI-SDR.
Stems are resampled back to the clip rate and mono stems are tiled to stereo, so the stereo-to-mono
collapse and the resampling the pipeline really does are part of what is measured.

    python run_eval.py --corpus D:/corpus-dev --work D:/work --out results.json \
        --hardware-label "RTX 4090 / 7950X" --provider cpu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.io import wavfile

import audiomath as am
import metrics

SEPARATOR_SR = 44100
RESULTS_SCHEMA_VERSION = 5
LENGTH_TOLERANCE_SAMPLES = 4

PINNED_MANIFEST = (Path(__file__).resolve().parents[2] / "src" / "Trackdub.Inference" / "Runtime"
                   / "ModelManifest" / "bundled-models.manifest.json")
PINNED_SPLEETER_FILES = ("vocals.onnx", "accompaniment.onnx")


def default_runner() -> list[str]:
    framework = "net11.0-windows10.0.19041.0" if platform.system() == "Windows" else "net11.0"
    return ["dotnet", "run", "--project", "src/Trackdub.Benchmarks.DevHost", "-f", framework,
            "-c", "Release", "--"]


DEFAULT_RUNNER = default_runner()

# (json path into a clip record, higher_is_better)
TRACKED_METRICS: dict[str, tuple[tuple[str, ...], bool]] = {
    "bed_leakage_to_bed_db": (("leakage", "leakage_to_bed_db"), False),
    "bed_dialogue_residual_db": (("leakage", "dialogue_residual_db"), False),
    "bed_worst_window_leakage_db": (("leakage", "worst_window_leakage_db"), False),  # relative to dialogue energy, not bed energy
    "bed_si_sdr_db": (("damage", "si_sdr_db"), True),
    "bed_si_sdr_inactive_db": (("damage", "si_sdr_inactive_db"), True),
    "bed_lsd_inactive_db": (("damage", "lsd_inactive_db"), False),
    "dialogue_si_sdr_db": (("dialogue_si_sdr_db",), True),
    "reconstruction_residual_db": (("reconstruction", "residual_db"), False),
    "reconstruction_in_band_residual_db": (("reconstruction", "in_band_residual_db"), False),
    "reconstruction_in_band_residual_with_edges_db": (("reconstruction", "in_band_residual_with_edges_db"), False),
    "reconstruction_residual_with_edges_db": (("reconstruction", "residual_with_edges_db"), False),
    "reconstruction_edge_residual_db": (("reconstruction", "edge_residual_db"), False),
    "bandwidth_retained_db": (("reconstruction", "bandwidth_retained_db"), True),
}

SeparatorRunner = Callable[[Path, Path], None]

# What each candidate declares about its own output, used by the reconstruction gate. band_limit_hz is
# the highest frequency it processes (Spleeter keeps bins 0..1023 of a 4096-point STFT at 44.1 kHz =
# 11025 Hz and zeroes the rest); output_bits is the precision of the stems it writes and output_rounding
# how WaveAudioWriter reduces to it (Spleeter's writer casts toward zero, which has 4x the noise power
# of rounding to nearest).
CANDIDATE_PROFILES: dict[str, dict] = {
    "spleeter": {"band_limit_hz": 11025.0, "output_bits": 16, "output_rounding": "truncate"},
    # Bandit v2 (multilingual DnR v3 checkpoint, 48 kHz STFT, full band) is run by candidates/bandit_v2_runner.py,
    # which writes float32 stems, so nothing is declared and nothing is relaxed. The two names differ only
    # in how the bed is formed: `sum` is music + sfx as the model estimated them (the reconstruction gate
    # then measures the model's own additivity), `residual` is mixture - dialogue computed in the separator domain.
    "bandit-v2-multi-sum": {"band_limit_hz": None, "output_bits": 32, "output_rounding": "nearest"},
    "bandit-v2-multi-residual": {"band_limit_hz": None, "output_bits": 32, "output_rounding": "nearest"},
    # KUIELab MDX-Net vocals network (candidates/mdxnet_runner.py): the network keeps 2048 of the 3073 bins of a 6144-point
    # FFT, so its estimate stops at 44100 * 2048 / 6144 = 14.7 kHz. Float32 stems; the bed is the residual.
    "mdxnet-vocals-residual": {"band_limit_hz": 14700.0, "output_bits": 32, "output_rounding": "nearest"},
    # MossFormer2_SE_48K (candidates/mossformer2_runner.py): full band, float32 stems, bed = mixture - enhanced speech.
    "mossformer2-se-48k-residual": {"band_limit_hz": None, "output_bits": 32, "output_rounding": "nearest"},
}


class EvalError(Exception):
    pass


def to_stereo(x: np.ndarray) -> np.ndarray:
    """Stereo view of a clip: mono is tiled, 5.1 (FL FR FC LFE BL BR) is downmixed, stereo is kept."""
    if x.shape[1] == 2:
        return x
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    if x.shape[1] == 6:
        g = 10.0 ** (-3.0 / 20.0)
        left = x[:, 0] + g * x[:, 2] + g * x[:, 4]
        right = x[:, 1] + g * x[:, 2] + g * x[:, 5]
        return np.stack([left, right], axis=1).astype(np.float32)
    raise EvalError(f"unsupported channel count {x.shape[1]}")


def write_pcm16(path: Path, x: np.ndarray, sr: int) -> np.ndarray:
    """Write PCM16 and return the quantised signal as float, i.e. exactly what the engine will read."""
    q = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, q)
    return q.astype(np.float32) / 32768.0


def to_reference_domain(est_mono: np.ndarray, n_ref: int, sr_ref: int) -> tuple[np.ndarray, int]:
    """Resample a 44.1 kHz mono stem to the clip rate and tile to stereo. Returns (stereo, length
    adjustment in samples). Differences beyond a few samples are an error, not something to hide."""
    est = am.resample(est_mono, SEPARATOR_SR, sr_ref)
    adjust = est.shape[0] - n_ref
    if abs(adjust) > LENGTH_TOLERANCE_SAMPLES:
        raise EvalError(f"stem length differs from the reference by {adjust} samples after resampling")
    est = est[:n_ref] if adjust >= 0 else np.concatenate(
        [est, np.zeros((-adjust, est.shape[1]), dtype=np.float32)], axis=0)
    return np.repeat(est, 2, axis=1), adjust


def prepare_jobs(corpus: Path, work: Path, clips: list[dict]) -> tuple[list[dict], dict[str, np.ndarray]]:
    jobs, mixtures = [], {}
    for clip in clips:
        cid = clip["clip_id"]
        mix = am.load_audio(corpus / cid / "mixture.wav", channels=clip["channels"], sr=clip["sample_rate"])
        stereo = am.resample(to_stereo(mix), clip["sample_rate"], SEPARATOR_SR)
        src = work / "input" / f"{cid}.wav"
        src.parent.mkdir(parents=True, exist_ok=True)
        mixtures[cid] = write_pcm16(src, stereo, SEPARATOR_SR)
        jobs.append({"id": cid, "input": str(src.resolve()),
                     "vocals_output": str((work / "output" / f"{cid}.vocals.wav").resolve()),
                     "bed_output": str((work / "output" / f"{cid}.bed.wav").resolve())})
    return jobs, mixtures


def verify_clip_files(corpus: Path, clips: list[dict]) -> None:
    """Check generated audio against the hashes recorded in the corpus manifest."""
    root = corpus.resolve()
    for clip in clips:
        clip_id = clip.get("clip_id")
        clip_dir = (root / str(clip_id)).resolve()
        if not clip_dir.is_relative_to(root):
            raise EvalError(f"clip {clip_id!r} resolves outside the corpus directory")
        files = clip.get("files")
        if not isinstance(files, dict):
            raise EvalError(f"clip {clip_id!r} has no generated-file hashes in the corpus manifest")
        for name in ("mixture", "dialogue", "bed"):
            expected = files.get(name)
            if not isinstance(expected, str) or len(expected) != 64:
                raise EvalError(f"clip {clip_id!r} has no valid {name} SHA-256 in the corpus manifest")
            path = clip_dir / f"{name}.wav"
            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise EvalError(f"clip {clip_id!r} is missing its {name} file: {exc}") from exc
            if actual != expected:
                raise EvalError(f"clip {clip_id!r} {name} SHA-256 mismatch")


def dotnet_runner(runner_cmd: list[str], provider: str | None, model_directory: str | None,
                  model: str = "spleeter", model_cache_directory: str | None = None) -> SeparatorRunner:
    def run(jobs_path: Path, results_path: Path) -> None:
        cmd = [*runner_cmd, "separation-eval", "--jobs", str(jobs_path), "--results", str(results_path),
               "--model", model]
        if provider:
            cmd += ["--provider", provider]
        if model_directory:
            cmd += ["--model-directory", model_directory]
        if model_cache_directory:
            cmd += ["--model-cache-directory", model_cache_directory]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode not in (0, 2):  # 2 = some jobs failed; those are recorded per clip
            raise EvalError(f"separator command failed with exit code {proc.returncode}")
    return run


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise EvalError(f"separator produced no results file at {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def model_provenance(timings: list[dict]) -> dict:
    """Record which model root the runner actually executed and whether its ONNX files match the
    pinned revision hashes in bundled-models.manifest.json. The engine's per-job metadata carries
    `model_root`; TRACKDUB_SPLEETER_ONNX_PATH can override the planned model, so this makes the
    executed weights (not just the candidate name) part of the report."""
    roots = sorted({t.get("metadata", {}).get("model_root") for t in timings
                    if isinstance(t.get("metadata"), dict) and t["metadata"].get("model_root")})
    pinned: dict = {}
    try:
        manifest = json.loads(PINNED_MANIFEST.read_text(encoding="utf-8"))
        for model in manifest.get("models", []):
            if model.get("engine_family") == "spleeter":
                pinned = dict(model.get("download_file_hashes") or {})
                break
    except OSError:
        pinned = {}
    executed = []
    for root in roots:
        hashes = {}
        for name in PINNED_SPLEETER_FILES:
            path = Path(root) / name
            if path.is_file():
                try:
                    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError:
                    hashes[name] = None
        executed.append({"model_root": root, "file_sha256": hashes,
                         "matches_pinned": bool(pinned) and hashes == pinned})
    return {"pinned_file_sha256": pinned or None, "executed": executed}


def candidate_provenance(timings: list[dict]) -> dict:
    """What a non-Spleeter runner says it executed: the distinct per-job metadata blocks (checkpoint hash,
    code revision, resampling and bed construction). Reported as given; there is no pinned manifest to check."""
    seen: list[dict] = []
    for t in timings:
        meta = t.get("metadata")
        if isinstance(meta, dict) and meta not in seen:
            seen.append(meta)
    return {"executed_metadata": seen}


def score_clip(clip: dict, corpus: Path, job: dict, timing: dict, mixture44: np.ndarray,
               gate: dict | None = None) -> dict:
    record = {"clip_id": clip["clip_id"], "recipe": clip["recipe"], "variant": clip.get("variant"),
              "codec": clip.get("codec"), "ok": False, "separator_ok": False, "error": None,
              "timing": {k: timing.get(k) for k in ("job_index", "wall_ms", "rtf", "audio_seconds",
                                                     "working_set_before_bytes", "peak_working_set_bytes",
                                                     "selected_provider")}}
    if not timing.get("ok"):
        record["error"] = f"separator failed: {timing.get('error')}"
        return record
    record["separator_ok"] = True
    try:
        vocals = am.load_audio(job["vocals_output"], channels=1, sr=SEPARATOR_SR)
        bed = am.load_audio(job["bed_output"], channels=1, sr=SEPARATOR_SR)
        record["reconstruction"] = metrics.check_reconstruction(
            mixture44.mean(axis=1, keepdims=True), vocals, bed, SEPARATOR_SR, **(gate or {})).as_dict()

        sr, n = clip["sample_rate"], None
        ref_d = to_stereo(am.load_audio(corpus / clip["clip_id"] / "dialogue.wav", clip["channels"], sr))
        ref_b = to_stereo(am.load_audio(corpus / clip["clip_id"] / "bed.wav", clip["channels"], sr))
        n = ref_d.shape[0]
        est_d, adj_d = to_reference_domain(vocals, n, sr)
        est_b, adj_b = to_reference_domain(bed, n, sr)
        record["length_adjust_samples"] = {"dialogue": adj_d, "bed": adj_b}

        try:
            record["leakage"] = metrics.bed_leakage(ref_d, ref_b, est_b, sr).as_dict()
        except metrics.MetricError as exc:
            record["leakage"] = {"skipped": str(exc)}
        try:
            record["damage"] = metrics.bed_damage(ref_d, ref_b, est_b, sr).as_dict()
        except metrics.MetricError as exc:
            record["damage"] = {"skipped": str(exc)}
        record["dialogue_si_sdr_db"] = metrics.si_sdr(ref_d, est_d)
        record["ok"] = record["reconstruction"]["passed"]
        if not record["ok"]:
            reasons = record["reconstruction"].get("reasons") or ["reconstruction gate failed"]
            record["error"] = "reconstruction gate failed: " + "; ".join(reasons)
    except (EvalError, am.AudioError, metrics.MetricError, OSError) as exc:
        record["error"] = str(exc)
    return record


def _at(record: dict, path: tuple[str, ...]):
    for key in path:
        if not isinstance(record, dict) or key not in record:
            return None
        record = record[key]
    return record


def _dig(record: dict, path: tuple[str, ...]):
    value = _at(record, path)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def summarise(values: list[float], higher_is_better: bool) -> dict | None:
    if not values:
        return None
    a = np.array(values, dtype=np.float64)
    worst_q = 10 if higher_is_better else 90
    return {"n": len(values), "median": float(np.median(a)), "worst_decile": float(np.percentile(a, worst_q)),
            "min": float(a.min()), "max": float(a.max())}


def summarise_diagnostic(values: list[float]) -> dict | None:
    if not values:
        return None
    a = np.array(values, dtype=np.float64)
    return {"n": len(values), "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def aggregate(records: list[dict]) -> dict:
    scored = [r for r in records if "reconstruction" in r]
    groups = {"all": scored}
    for r in scored:
        groups.setdefault(r["recipe"], []).append(r)
    per_group = {}
    for name, rows in sorted(groups.items()):
        per_group[name] = {
            "clips": len(rows),
            "reconstruction_gate_pass": sum(1 for r in rows if r["reconstruction"].get("passed")),
            "metrics": {m: summarise([v for r in rows if (v := _dig(r, path)) is not None], hib)
                        for m, (path, hib) in TRACKED_METRICS.items()},
            "bed_leakage_accounting": {
                "scored_clips": sum(1 for r in rows if _at(r, ("leakage", "status")) == "scored"),
                "not_applicable_reference_bed_below_floor": sum(
                    1 for r in rows if _at(r, ("leakage", "status")) == "not_applicable_reference_bed_below_floor"),
                "skipped_clips": sum(1 for r in rows if _at(r, ("leakage", "skipped")) is not None),
                "not_computed_clips": sum(1 for r in rows if "leakage" not in r),
                "estimated_bed_active_rms_dbfs": summarise_diagnostic(
                    [v for r in rows if (v := _dig(r, ("leakage", "estimated_bed_active_rms_dbfs"))) is not None]),
            },
        }
    timed = [r["timing"] for r in records if r["timing"].get("wall_ms") is not None]
    warm = [t["rtf"] for t in timed if t["job_index"] and t.get("rtf") is not None]
    cold = [t["rtf"] for t in timed if t["job_index"] == 0 and t.get("rtf") is not None]
    peaks = [t["peak_working_set_bytes"] for t in timed if t.get("peak_working_set_bytes") is not None]
    return {
        "clips_total": len(records), "clips_scored": len(scored),
        "clips_failed": [{"clip_id": r["clip_id"], "error": r["error"]} for r in records if not r["ok"]],
        "strata": per_group,
        "resources": {
            "rtf_cold_first_job": cold[0] if cold else None,
            "rtf_warm_median": float(np.median(warm)) if warm else None,
            "rtf_warm_p95": float(np.percentile(warm, 95)) if warm else None,
            "peak_working_set_bytes_max": max(peaks) if peaks else None,
            "selected_providers": sorted({t["selected_provider"] for t in timed if t.get("selected_provider")}),
        },
    }


def evaluate(corpus: Path, work: Path, run_separator: SeparatorRunner, *, candidate: str, provider: str | None,
             hardware_label: str, clip_limit: int | None = None, gate: dict | None = None) -> dict:
    if clip_limit is not None and clip_limit <= 0:
        raise EvalError("clip_limit must be positive")
    manifest_path = corpus / "corpus.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = manifest["clips"][:clip_limit] if clip_limit is not None else manifest["clips"]
    if not clips:
        raise EvalError("corpus has no clips")
    verify_clip_files(corpus, clips)
    work.mkdir(parents=True, exist_ok=True)
    jobs, mixtures = prepare_jobs(corpus, work, clips)
    jobs_path, results_path = work / "jobs.jsonl", work / "separator-results.jsonl"
    jobs_path.write_text("".join(json.dumps(j) + "\n" for j in jobs), encoding="utf-8")
    results_path.unlink(missing_ok=True)
    run_separator(jobs_path, results_path)

    timings = {t["id"]: t for t in read_jsonl(results_path)}
    by_id = {j["id"]: j for j in jobs}
    records = []
    for clip in clips:
        cid = clip["clip_id"]
        if cid not in timings:
            records.append({"clip_id": cid, "recipe": clip["recipe"], "variant": clip.get("variant"),
                            "codec": clip.get("codec"), "ok": False, "separator_ok": False,
                            "error": "no result from separator",
                            "timing": {}})
            continue
        records.append(score_clip(clip, corpus, by_id[cid], timings[cid], mixtures[cid], gate))

    return {
        "results_schema_version": RESULTS_SCHEMA_VERSION,
        "candidate": candidate,
        "provider_requested": provider,
        "hardware_label": hardware_label,
        "gate_config": {"threshold_db": metrics.RECONSTRUCTION_GATE_DB,
                        "edge_exclusion_s": metrics.RECONSTRUCTION_EDGE_EXCLUSION_S, **(gate or {})},
        "model_provenance": (model_provenance(list(timings.values())) if candidate == "spleeter"
                             else candidate_provenance(list(timings.values()))),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "corpus": {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   "items_sha256": manifest.get("items_sha256"), "split": manifest.get("split"),
                   "seed": manifest.get("seed"), "generator_version": manifest.get("generator_version")},
        "summary": aggregate(records),
        "clips": records,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True, help="directory with corpus.manifest.json (from mixgen.py)")
    p.add_argument("--work", type=Path, required=True, help="scratch directory for inputs and stems")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hardware-label", required=True, help="free-text machine description recorded in the results")
    p.add_argument("--provider", help="execution provider to pin (e.g. cpu, directml, tensorrtrtx)")
    p.add_argument("--model-directory", help="model root passed to the separator")
    p.add_argument("--model-cache-directory", help="machine-local model cache root holding the pinned weights and model-cache-records.json")
    p.add_argument("--candidate", choices=sorted(CANDIDATE_PROFILES), default="spleeter",
                   help="candidate profile (declared band, output precision); also passed to the runner as --model")
    p.add_argument("--runner", default=" ".join(default_runner()),
                   help="command prefix for Trackdub.Benchmarks, as one quoted string")
    p.add_argument("--band-limit-hz", type=float, help="override the candidate profile's declared processing band")
    p.add_argument("--output-bits", type=int, help="override the candidate profile's stem precision")
    p.add_argument("--output-rounding", choices=("nearest", "truncate"), help="override the profile's stem rounding")
    p.add_argument("--limit", type=int, help="score only the first N clips (development)")
    args = p.parse_args(argv)
    gate = dict(CANDIDATE_PROFILES[args.candidate])
    if args.band_limit_hz is not None:
        gate["band_limit_hz"] = args.band_limit_hz
    if args.output_bits is not None:
        gate["output_bits"] = args.output_bits
    if args.output_rounding is not None:
        gate["output_rounding"] = args.output_rounding
    try:
        result = evaluate(args.corpus, args.work, dotnet_runner(shlex.split(args.runner), args.provider, args.model_directory,
                                                      model=args.candidate,
                                                      model_cache_directory=args.model_cache_directory),
                          candidate=args.candidate, provider=args.provider, hardware_label=args.hardware_label,
                          clip_limit=args.limit, gate=gate)
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"error: could not write results to {args.out}: {exc}", file=sys.stderr)
        return 1
    s = result["summary"]
    print(f"scored {s['clips_scored']}/{s['clips_total']} clips; results in {args.out}")
    return 0 if not s["clips_failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
