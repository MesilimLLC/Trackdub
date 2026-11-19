#!/usr/bin/env python3
"""Out-of-process separation runner for the Dynamic Band Splitting cinematic models (DBS-TIGER, DBS-BSRNN).

Speaks the command line of `Trackdub.Benchmarks separation-eval` through runner_common:

    separation-eval --jobs <jobs.jsonl> --results <results.jsonl> --model dbs-<tiger|bsrnn>-dnr-residual
                    [--provider cpu|cuda] --model-directory <local snapshot of JusperLee/DBS-<...>-DnR>

The models (Zhang, Li and Hu, arXiv 2610.14993; JusperLee/Dynamic-Band-Splitting, Apache 2.0) separate 44.1 kHz
mono audio into music, sound effects and speech. Per job:
  1. read the 44.1 kHz stereo input (the models' native rate, so nothing is resampled);
  2. separate the left and right channels as a batch of two with the authors' chunked inference and window sizes
     (DBS-TIGER 6 s windows, 3 s hop; DBS-BSRNN 3 s, 1.5 s hop);
  3. the dialogue is the speech stem and the bed is the input minus it (the residual form).
The code and weights are Apache 2.0, but the weights were trained on DnR v2 (LibriSpeech, FMA, FSD50K, which hold
non-commercial material), so they are evaluation-only. The point of scoring them is the architecture: a 1 to 5 M
parameter cinematic separator that could be retrained on cleared data for the CPU default.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner_common as rc  # noqa: E402

# candidate -> (model class name, chunk seconds, hop seconds), from the authors' README and training recipes
MODELS = {"dbs-tiger-dnr-residual": ("DBSTIGER", 6.0, 3.0), "dbs-bsrnn-dnr-residual": ("DBSBSRNN", 3.0, 1.5)}
STEMS = ("music", "sfx", "speech")


def speech_stem(output, channels: int):
    """Pick the speech stem from separate()'s output, shaped [channels, sources, samples]."""
    out = output.reshape(channels, len(STEMS), -1)
    return out[:, STEMS.index("speech")]


class Separator:
    def __init__(self, model_directory: Path, provider: str | None, model: str):
        import torch
        from dbs import models

        self.torch = torch
        self.device_type = "cuda" if provider == "cuda" else "cpu"
        if self.device_type == "cuda" and not torch.cuda.is_available():
            raise rc.RunnerError("CUDA requested but not available")
        class_name, self.chunk_seconds, self.hop_seconds = MODELS[model]
        self.model = getattr(models, class_name).from_pretrained(str(model_directory)).to(self.device_type).eval()

    def estimate(self, mix):
        from dbs.engine import separate

        torch = self.torch
        protocol = {"mode": "chunked", "chunk_seconds": self.chunk_seconds, "hop_seconds": self.hop_seconds}
        with torch.inference_mode():
            output = separate(self.model, mix.unsqueeze(1).to(self.device_type), protocol,
                              sample_rate=rc.SEPARATOR_SR)
        dialogue = rc.fix_length(torch, speech_stem(output, mix.shape[0]).float().cpu(), mix.shape[1])
        return dialogue, mix - dialogue


def code_revision() -> str:
    repo = os.environ.get("DBS_REPO")
    if not repo:
        return "unknown (DBS_REPO not set)"
    out = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def metadata(model_directory: Path, model: str, device_type: str, threads: int) -> dict:
    class_name, chunk, hop = MODELS[model]
    return {"runner": "dbs_runner", "candidate": model, "model_class": class_name,
            "weights_sha256": rc.file_hash(model_directory / "model.safetensors", "sha256"),
            "code_revision": code_revision(), "chunk_seconds": str(chunk), "hop_seconds": str(hop),
            "bed": "input mixture minus the speech stem, at 44.1 kHz", "selected_provider": device_type,
            "torch_threads": str(threads),
            "licence_note": "Apache 2.0 code and weights; weights trained on DnR v2 (non-commercial sources), "
                            "evaluation only"}


def main(argv: list[str]) -> int:
    args = rc.parse_command_line(argv, list(MODELS))
    sep = Separator(args.model_directory, args.provider, args.model)
    meta = metadata(args.model_directory, args.model, sep.device_type, sep.torch.get_num_threads())
    ctx = rc.RunContext(sep.torch, sep.device_type, args.provider or "cpu", meta)
    return rc.run_all(sep.estimate, args, ctx)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
