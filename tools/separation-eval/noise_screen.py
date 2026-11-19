#!/usr/bin/env python3
"""Screen dialogue items for background noise and interfering voices before they become training targets.

Whatever a dialogue item contains is, to a separator in training, dialogue: room noise, hiss, music or another
person talking in a crowd-recorded clip would teach the model to pull background into the dialogue stem. Each item
is scored with DNSMOS P.835 (Reddy, Gopal and Cutler, ICASSP 2022; microsoft/DNS-Challenge, CC BY 4.0), a
no-reference model trained on human ratings, which predicts 1 to 5 scores for the speech (SIG), the background
(BAK) and overall (OVRL). The personalized model (pDNSMOS) also penalizes interfering talkers. Items whose
background score falls under the thresholds are rejected.

The models are a local screening tool only, never bundled into Trackdub. Expected files (download them yourself
from github.com/microsoft/DNS-Challenge, folder DNSMOS):
    <models-dir>/p835/sig_bak_ovr.onnx               DNSMOS/DNSMOS/sig_bak_ovr.onnx
    <models-dir>/p835-personalized/sig_bak_ovr.onnx  DNSMOS/pDNSMOS/sig_bak_ovr.onnx

Scoring follows the reference dnsmos_local.py: 16 kHz mono, 9.01 s windows one second apart (short audio is tiled
to one window), per-window scores mapped through the published polynomials and averaged. Two differences, both
recorded in the output: runs of exact digital silence of 10 ms or more are removed first (build_local_items.py puts
0.25 s of it between clips, and it would flatter BAK), and long items are scored on at most --max-windows evenly
spaced windows.

    python noise_screen.py screen --candidates cv-train.candidates.jsonl --cache D:/train-cache/corpus-cache \\
        --models-dir D:/cache/models/dnsmos --out cv-train.screened.jsonl --report cv-noise-report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

import audiomath as am
import ingest

SR = 16000
WINDOW = 144160  # 9.01 s at 16 kHz, the models' fixed input
HOP = SR
MIN_SILENCE_RUN = SR // 100
SCREEN_VERSION = 1
MODELS = {"p835": "p835/sig_bak_ovr.onnx", "personalized": "p835-personalized/sig_bak_ovr.onnx"}
# Published mappings from raw model outputs to MOS, from dnsmos_local.py (sig, bak, ovr), highest power first.
POLYNOMIALS = {
    "p835": ([-0.08397278, 1.22083953, 0.0052439], [-0.13166888, 1.60915514, -0.39604546],
             [-0.06766283, 1.11546468, 0.04602535]),
    "personalized": ([-0.01019296, 0.02751166, 1.19576786, -0.24348726],
                     [-0.04976499, 0.44276479, -0.1644611, 0.96883132],
                     [-0.00533021, 0.005101, 1.18058466, -0.11236046]),
}
# Calibrated on the corpus v1.2 eval dialogue (2026-11-19), median and 5th percentile of BAK: VCTK (studio) 4.07 and
# 3.99, LibriVox 4.07 and 3.70, Common Voice 3.80 and 3.04, Freesound shouts and whispers 2.44 and 1.14. 3.6 lies
# under the noisiest 5% of clean read speech and above the noisy tail of crowd recordings.
DEFAULT_MIN_BAK = 3.6
DEFAULT_MIN_PERSONALIZED_BAK = 3.6


class Thresholds(NamedTuple):
    min_bak: float
    min_personalized_bak: float


def drop_digital_silence(x: np.ndarray, min_run: int = MIN_SILENCE_RUN) -> np.ndarray:
    """Remove runs of exact zeros at least `min_run` samples long; shorter zero runs are kept."""
    zero = np.concatenate([[False], x == 0.0, [False]])
    edges = np.flatnonzero(np.diff(zero.astype(np.int8)))
    keep = np.ones(x.size, dtype=bool)
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start >= min_run:
            keep[start:stop] = False
    return x[keep]


def windows(x: np.ndarray, max_windows: int) -> np.ndarray:
    """The 9.01 s windows to score, shaped (n, WINDOW): tiled when short, evenly thinned when long."""
    if x.size == 0:
        raise ValueError("no audio left after removing digital silence")
    while x.size < WINDOW:
        x = np.concatenate([x, x])
    starts = np.arange(0, x.size - WINDOW + 1, HOP)
    if starts.size > max_windows:
        starts = starts[np.linspace(0, starts.size - 1, max_windows).round().astype(int)]
    return np.stack([x[s:s + WINDOW] for s in starts]).astype(np.float32)


def to_mos(raw: np.ndarray, model: str) -> np.ndarray:
    """Map raw (n, 3) outputs to MOS with the published polynomials, column by column."""
    return np.stack([np.poly1d(coeffs)(raw[:, k]) for k, coeffs in enumerate(POLYNOMIALS[model])], axis=1)


Scorer = Callable[[np.ndarray], dict]


def onnx_scorer(models_dir: Path) -> Scorer:
    import onnxruntime as ort
    sessions = {name: ort.InferenceSession(str(models_dir / rel), providers=["CPUExecutionProvider"])
                for name, rel in MODELS.items()}

    def score(batch: np.ndarray) -> dict:
        out = {}
        for name, session in sessions.items():
            raw = np.concatenate([session.run(None, {"input_1": w[np.newaxis, :]})[0] for w in batch])
            out[name] = to_mos(raw, name).mean(axis=0)
        return out
    return score


def score_item(path: Path, scorer: Scorer, max_windows: int) -> dict:
    audio = am.load_audio(path, channels=1, sr=SR)[:, 0]
    speech = drop_digital_silence(audio)
    batch = windows(speech, max_windows)
    scores = scorer(batch)
    sig, bak, ovr = (round(float(v), 3) for v in scores["p835"])
    p_sig, p_bak, p_ovr = (round(float(v), 3) for v in scores["personalized"])
    return {"sig": sig, "bak": bak, "ovrl": ovr, "personalized_sig": p_sig, "personalized_bak": p_bak,
            "personalized_ovrl": p_ovr, "windows": int(batch.shape[0]),
            "seconds_scored": round(speech.size / SR, 1), "screen_version": SCREEN_VERSION}


def verdict(scores: dict, thresholds: Thresholds) -> list[str]:
    reasons = []
    if scores["bak"] < thresholds.min_bak:
        reasons.append(f"noisy-background: DNSMOS BAK {scores['bak']} < {thresholds.min_bak}")
    if scores["personalized_bak"] < thresholds.min_personalized_bak:
        reasons.append(f"interfering-sound: personalized BAK {scores['personalized_bak']} < "
                       f"{thresholds.min_personalized_bak}")
    return reasons


def screen_record(rec: dict, cache: Path, scorer: Scorer, max_windows: int) -> dict:
    """The record's scores, or an error when it cannot be scored (not dialogue, missing audio, decode failure)."""
    if rec.get("role") != "dialogue":
        return {"error": "not a dialogue item"}
    try:
        return score_item(ingest.cache_path(cache, rec), scorer, max_windows)
    except (am.AudioError, OSError, ValueError, ingest.IngestError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def model_hashes(models_dir: Path) -> dict:
    return {name: hashlib.sha256((models_dir / rel).read_bytes()).hexdigest() for name, rel in MODELS.items()}


def distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    a = np.array(values)
    return {"n": len(values), **{f"p{q}": round(float(np.percentile(a, q)), 3) for q in (5, 10, 25, 50, 75, 90)}}


def summarize(results: list[tuple[dict, dict, list[str]]], thresholds: Thresholds, models: dict) -> dict:
    scored = [s for _, s, _ in results if "error" not in s]
    return {"screen_version": SCREEN_VERSION, "models_sha256": models, "thresholds": thresholds._asdict(),
            "items": len(results), "accepted": sum(1 for _, _, r in results if not r),
            "rejected": sum(1 for _, _, r in results if r),
            "bak": distribution([s["bak"] for s in scored]),
            "personalized_bak": distribution([s["personalized_bak"] for s in scored]),
            "sig": distribution([s["sig"] for s in scored]),
            "rejections": [{"id": rec.get("id"), "reasons": r, "scores": s} for rec, s, r in results if r]}


def screen_all(records: list[dict], cache: Path, scorer: Scorer, args: argparse.Namespace) -> list[tuple]:
    thresholds = Thresholds(args.min_bak, args.min_personalized_bak)

    def one(rec: dict) -> tuple[dict, dict, list[str]]:
        scores = screen_record(rec, cache, scorer, args.max_windows)
        reasons = [f"unscored: {scores['error']}"] if "error" in scores else verdict(scores, thresholds)
        return rec, scores, reasons

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        return list(pool.map(one, records))


def run_screen(args: argparse.Namespace, scorer: Scorer | None = None) -> dict:
    if len({args.candidates.resolve(), args.out.resolve(), args.report.resolve()}) < 3:
        raise SystemExit("--candidates, --out and --report must be three different files")
    records = ingest.read_jsonl(args.candidates)
    results = screen_all(records, args.cache, scorer or onnx_scorer(args.models_dir), args)
    accepted = [{**rec, "noise_screen": scores} for rec, scores, reasons in results if not reasons]
    args.out.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in accepted),
                        encoding="utf-8")
    models = model_hashes(args.models_dir) if scorer is None else {}
    report = summarize(results, Thresholds(args.min_bak, args.min_personalized_bak), models)
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("screen", help="score dialogue candidates and keep the clean ones")
    s.add_argument("--candidates", type=Path, required=True, help="candidate records (JSONL)")
    s.add_argument("--cache", type=Path, required=True, help="corpus cache holding the items' audio")
    s.add_argument("--models-dir", type=Path, required=True, help="folder with p835/ and p835-personalized/")
    s.add_argument("--out", type=Path, required=True, help="accepted records with their scores (JSONL)")
    s.add_argument("--report", type=Path, required=True, help="score distributions and every rejection (JSON)")
    s.add_argument("--min-bak", type=float, default=DEFAULT_MIN_BAK)
    s.add_argument("--min-personalized-bak", type=float, default=DEFAULT_MIN_PERSONALIZED_BAK)
    s.add_argument("--max-windows", type=int, default=20, help="windows scored per item at most (default 20)")
    s.add_argument("--workers", type=int, default=4)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = run_screen(args)
    print(f"{report['accepted']} accepted, {report['rejected']} rejected of {report['items']}; report in "
          f"{args.report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
