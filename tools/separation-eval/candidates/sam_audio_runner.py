#!/usr/bin/env python3
"""Out-of-process separation runner for Meta SAM-Audio (text-prompted, generative), scored as a GPU-only candidate.

Speaks the command line of `Trackdub.Benchmarks separation-eval` through runner_common, so `run_eval.py` drives it
with `--runner "<python> <this file>"`:

    separation-eval --jobs <jobs.jsonl> --results <results.jsonl> --model sam-audio-<size>-residual
                    [--provider cuda] --model-directory <local snapshot of facebook/sam-audio-<size>>

`--model-directory` is a local copy of the gated Hugging Face repository (checkpoint.pt and config.json), so the
runner never downloads model weights itself; the T5 text encoder named in config.json is fetched by transformers
on first use. The code comes from the installed `sam_audio` package (facebookresearch/sam-audio); its revision is
read from SAM_AUDIO_REPO when set and recorded.

SAM-Audio is generative: a flow-matching transformer produces the prompted target and a residual from noise. Per job:
  1. read the 44.1 kHz stereo input and resample to the model's 48 kHz;
  2. separate the left and right channels as a batch of two (the model mixes its input to mono), prompt PROMPT,
     no span prediction, no reranking, noise seeded per job so a rerun gives the same stems;
  3. inputs longer than WINDOW_SECONDS (the training clip length) are separated in windows overlapping by
     OVERLAP_SECONDS and joined with linear crossfades;
  4. resample the target back to 44.1 kHz; the dialogue is that target and the bed is the input minus it (the
     residual form, as for the other generative or enhancement candidates; the model's own residual is not used).
The licence (SAM License, Meta) and the training data (Meta-internal) are not cleared for shipping; this runner is
for evaluation only.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner_common as rc  # noqa: E402

os.environ.setdefault("TQDM_DISABLE", "1")

MODELS = {"sam-audio-small-residual": "small", "sam-audio-base-residual": "base"}
PROMPT = "speech"
NATIVE_SR = 48000
WINDOW_SECONDS, OVERLAP_SECONDS = 30.0, 5.0
SEED = 1234


def window_starts(n: int, window: int, overlap: int) -> list[int]:
    """Start samples of windows covering n samples; the last window ends exactly at n."""
    if n <= window:
        return [0]
    hop = window - overlap
    starts = list(range(0, n - window, hop))
    return starts + [n - window]


def crossfade_join(pieces: list, starts: list[int], n: int, overlap: int):
    """Overlap-add windows of shape (channels, w) with linear ramps on every overlapping edge."""
    import torch

    out = torch.zeros(pieces[0].shape[0], n)
    weight = torch.zeros(n)
    for k, (piece, start) in enumerate(zip(pieces, starts)):
        w = piece.shape[1]
        ramp = torch.ones(w)
        fade = min(overlap, w)
        if k > 0:
            ramp[:fade] = torch.linspace(0.0, 1.0, fade)
        if k < len(pieces) - 1:
            ramp[w - fade:] = torch.minimum(ramp[w - fade:], torch.linspace(1.0, 0.0, fade))
        out[:, start:start + w] += piece * ramp
        weight[start:start + w] += ramp
    return out / weight.clamp_min(1e-6)


class Separator:
    def __init__(self, model_directory: Path, provider: str | None):
        import torch

        if provider not in (None, "cuda") or not torch.cuda.is_available():
            raise rc.RunnerError("SAM-Audio is evaluated on CUDA only")
        from sam_audio import SAMAudio, SAMAudioProcessor

        self.torch = torch
        self.device = torch.device("cuda")
        self.model = SAMAudio.from_pretrained(str(model_directory)).to(self.device).eval()
        self.processor = SAMAudioProcessor.from_pretrained(str(model_directory))

    def separate_window(self, channels_48k):
        torch = self.torch
        audios = [channels_48k[c:c + 1] for c in range(channels_48k.shape[0])]
        batch = self.processor(audios=audios, descriptions=[PROMPT] * len(audios)).to(self.device)
        result = self.model.separate(batch, predict_spans=False, reranking_candidates=1)
        targets = [t.reshape(-1).float().cpu() for t in result.target]
        n = channels_48k.shape[1]
        return torch.stack([rc.fix_length(torch, t.unsqueeze(0), n)[0] for t in targets])

    def estimate(self, mix):
        import torchaudio

        torch = self.torch
        torch.manual_seed(SEED)
        x = torchaudio.functional.resample(mix, rc.SEPARATOR_SR, NATIVE_SR)
        window, overlap = int(WINDOW_SECONDS * NATIVE_SR), int(OVERLAP_SECONDS * NATIVE_SR)
        starts = window_starts(x.shape[1], window, overlap)
        pieces = [self.separate_window(x[:, s:s + window]) for s in starts]
        target = crossfade_join(pieces, starts, x.shape[1], overlap)
        dialogue = rc.fix_length(torch, torchaudio.functional.resample(target, NATIVE_SR, rc.SEPARATOR_SR),
                                 mix.shape[1])
        return dialogue, mix - dialogue


def code_revision() -> str:
    repo = os.environ.get("SAM_AUDIO_REPO")
    if not repo:
        return "unknown (SAM_AUDIO_REPO not set)"
    out = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return out.stdout.strip() or "unknown"


def metadata(model_directory: Path, model: str) -> dict:
    checkpoint = model_directory / "checkpoint.pt"
    return {"runner": "sam_audio_runner", "candidate": model, "size": MODELS[model], "prompt": PROMPT,
            "checkpoint_sha256": rc.file_hash(checkpoint, "sha256"), "code_revision": code_revision(),
            "window_seconds": str(WINDOW_SECONDS), "overlap_seconds": str(OVERLAP_SECONDS), "seed": str(SEED),
            "options": "predict_spans=False, reranking_candidates=1, default ODE options",
            "bed": "input mixture minus the prompted target, in the 44.1 kHz domain",
            "selected_provider": "cuda",
            "licence_note": "SAM License (Meta) and Meta-internal training data; evaluation only"}


def main(argv: list[str]) -> int:
    args = rc.parse_command_line(argv, list(MODELS))
    sep = Separator(args.model_directory, args.provider)
    ctx = rc.RunContext(sep.torch, "cuda", args.provider or "cuda", metadata(args.model_directory, args.model))
    return rc.run_all(sep.estimate, args, ctx)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
