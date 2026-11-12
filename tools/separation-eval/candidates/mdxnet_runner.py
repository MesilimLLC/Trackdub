#!/usr/bin/env python3
"""Out-of-process separation runner for the official KUIELab-MDX-Net vocals model (MDX Challenge 2021, Track A).

    separation-eval --jobs <jobs.jsonl> --results <results.jsonl> --model mdxnet-vocals-residual [--provider cpu|cuda]
                    --model-directory <dir holding onnx_A.zip and repo/>

The weights are `vocals.onnx` inside `onnx_A.zip` (Zenodo record 5717356, CC-BY-4.0; the code is MIT). The archive
md5 is checked against the Zenodo listing and the member sha256 against the value pinned below before the model is
loaded; nothing is unpickled, because the model is an ONNX graph. `repo/` is a checkout of
kuielab/mdx-net-submission (branch leaderboard_A), kept for provenance: its `models.py` and `predict_blend.py` are
the reference for the STFT and chunking reproduced here, and their hashes are recorded.

This is the vocals network only. The challenge submission also blended in Demucs (9:1 for vocals) and a mixer over all
four stems; neither is used, because the question is what this one vocals checkpoint does as a dialogue/background
separator. The network has no accompaniment output, so the bed is the residual `input - dialogue`.

Pipeline per job, recorded in the result metadata:
  1. read the 44.1 kHz stereo PCM16 input the scorer prepared (float);
  2. the official non-overlapping chunking: zero-pad `n_fft // 2` at the start and enough zeros at the end for whole
     chunks, cut chunks of `hop * (2 * dim_t - 1)` samples at a stride of `chunk - n_fft`, STFT each (n_fft 6144,
     hop 1024, Hann, centered), keep the first `dim_f` = 2048 bins (about 14.7 kHz), run the ONNX network, restore the
     bins with zeros, inverse STFT, trim `n_fft // 2` from each end, concatenate, cut to the input length;
  3. the dialogue is that estimate; the bed is the input mixture minus it.
The estimate has no content above the model's band, so the bed keeps everything above it (that is the 14.7 kHz band
declared in the candidate profile).
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner_common as rc  # noqa: E402

ARCHIVE_NAME = "onnx_A.zip"
ARCHIVE_MD5 = "72605d3722ebef18c353f831abdbfc07"  # Zenodo 5717356 file listing
MEMBER = "onnx_A/vocals.onnx"
MEMBER_SHA256 = "daba83c2ee1afee9139766ad64c9b6808d6b6f092fff04bed3338be50baac721"
REPO_REVISION = "13c98c0ed9a759b82a3a6d10514738f3e8518365"  # kuielab/mdx-net-submission, leaderboard_A
REFERENCE_FILES = ["models.py", "predict_blend.py"]
MODELS = {"mdxnet-vocals-residual": "residual"}

# configs: Conv_TDF_net_trim(target_name='vocals', dim_f=11, dim_t=8, hop=1024); n_fft = 2**11 * n_fft_scale['vocals']
DIM_F, DIM_T, HOP = 2 ** 11, 2 ** 8, 1024
N_FFT = DIM_F * 3
N_BINS = N_FFT // 2 + 1
CHUNK = HOP * (DIM_T * 2 - 1)
TRIM = N_FFT // 2
GEN_SIZE = CHUNK - 2 * TRIM
BAND_LIMIT_HZ = DIM_F * rc.SEPARATOR_SR / N_FFT
BATCH = 8


def padding_for(n: int) -> int:
    """Zeros appended so the input is a whole number of chunk strides; none when it already is."""
    return (-n) % GEN_SIZE


def verified_weights(model_directory: Path) -> tuple[bytes, str, str]:
    """The vocals graph from the archive, after the archive md5 and the member sha256 match their pins."""
    archive = model_directory / ARCHIVE_NAME
    if not archive.is_file():
        raise rc.RunnerError(f"archive not found: {archive}")
    archive_md5 = rc.file_hash(archive, "md5")
    if archive_md5 != ARCHIVE_MD5:
        raise rc.RunnerError(f"{ARCHIVE_NAME} md5 {archive_md5} does not match the Zenodo listing {ARCHIVE_MD5}")
    with zipfile.ZipFile(archive) as zf:
        weights = zf.read(MEMBER)
    member_sha256 = hashlib.sha256(weights).hexdigest()
    if member_sha256 != MEMBER_SHA256:
        raise rc.RunnerError(f"{MEMBER} sha256 {member_sha256} is not the pinned {MEMBER_SHA256}")
    return weights, archive_md5, member_sha256


def reference_hashes(repo: Path) -> dict:
    if rc.git_head(repo) != REPO_REVISION:
        raise rc.RunnerError(f"{repo} is not at the pinned revision {REPO_REVISION}")
    rc.require_unmodified(repo, REFERENCE_FILES)
    return {name: rc.file_hash(repo / name, "sha256") for name in REFERENCE_FILES}


def providers_for(ort, provider: str | None) -> list[str]:
    wanted = (provider or "").lower()
    if wanted == "cpu":
        return ["CPUExecutionProvider"]
    if wanted not in ("", "cuda", "gpu"):
        raise rc.RunnerError(f"unsupported provider {provider!r}; use cpu or cuda")
    return [p for p in ("CUDAExecutionProvider",) if p in ort.get_available_providers()] + ["CPUExecutionProvider"]


def open_session(ort, weights: bytes, provider: str | None):
    session = ort.InferenceSession(weights, providers=providers_for(ort, provider))
    selected = session.get_providers()[0]
    if (provider or "").lower() in ("cuda", "gpu"):
        require_cuda(selected)
    return session, selected


def require_cuda(selected: str) -> None:
    if selected != "CUDAExecutionProvider":
        raise rc.RunnerError(f"provider cuda requested but onnxruntime selected {selected}")


class Separator:
    def __init__(self, model_directory: Path, provider: str | None):
        import numpy as np
        import onnxruntime as ort
        import torch

        self.np, self.torch = np, torch
        weights, self.archive_md5, self.member_sha256 = verified_weights(model_directory)
        self.reference_hashes = reference_hashes(model_directory / "repo")
        self.session, self.selected = open_session(ort, weights, provider)
        self.window = torch.hann_window(N_FFT, periodic=True)
        self.device_name = "cpu" if self.selected == "CPUExecutionProvider" else self.selected
        self.versions = {"torch": torch.__version__, "onnxruntime": ort.__version__, "python": sys.version.split()[0]}

    def stft(self, x):
        torch = self.torch
        batch = x.shape[0]
        spec = torch.stft(x.reshape(batch * 2, -1), n_fft=N_FFT, hop_length=HOP, window=self.window, center=True,
                          return_complex=True)
        real = torch.view_as_real(spec).permute(0, 3, 1, 2)
        return real.reshape(batch, 2, 2, N_BINS, -1).reshape(batch, 4, N_BINS, -1)[:, :, :DIM_F]

    def istft(self, x):
        torch = self.torch
        batch = x.shape[0]
        pad = torch.zeros(batch, 4, N_BINS - DIM_F, x.shape[-1])
        full = torch.cat([x, pad], dim=-2).reshape(batch, 2, 2, N_BINS, -1).reshape(batch * 2, 2, N_BINS, -1)
        complex_spec = torch.view_as_complex(full.permute(0, 2, 3, 1).contiguous())
        wave = torch.istft(complex_spec, n_fft=N_FFT, hop_length=HOP, window=self.window, center=True)
        return wave.reshape(batch, 2, -1)

    def run_chunks(self, chunks):
        torch, np = self.torch, self.np
        pieces = []
        for start in range(0, chunks.shape[0], BATCH):
            spec = self.stft(chunks[start:start + BATCH]).numpy()
            out = self.session.run(None, {"input": spec})[0]
            wave = self.istft(torch.from_numpy(np.ascontiguousarray(out)))
            pieces.append(wave[:, :, TRIM:-TRIM])
        return torch.cat(pieces, dim=0)

    def estimate(self, mix):
        """mix: float32 (2, n) at 44.1 kHz -> (dialogue, bed)."""
        torch = self.torch
        n = mix.shape[1]
        pad = padding_for(n)
        padded = torch.nn.functional.pad(mix, (TRIM, pad + TRIM))
        chunks = torch.stack([padded[:, i:i + CHUNK] for i in range(0, n + pad, GEN_SIZE)])
        with torch.inference_mode():
            tracks = self.run_chunks(chunks)
        dialogue = tracks.transpose(0, 1).reshape(2, -1)[:, :n].float().contiguous()
        return dialogue, mix - dialogue


def metadata(sep: Separator, model: str) -> dict:
    return {"runner": "mdxnet-python", "engine_family": "kuielab-mdx-net", "model": model,
            "checkpoint": f"{ARCHIVE_NAME}:{MEMBER}", "checkpoint_archive_md5": sep.archive_md5,
            "checkpoint_sha256": sep.member_sha256, "weights_source": "zenodo.org/records/5717356",
            "weights_license": "CC-BY-4.0 (Zenodo record); code MIT", "code_repo": "github.com/kuielab/mdx-net-submission",
            "code_branch": "leaderboard_A", "code_revision": REPO_REVISION,
            "reference_file_sha256": json.dumps(sep.reference_hashes, sort_keys=True),
            "runner_sha256": rc.file_hash(Path(__file__), "sha256"), "bed_mode": MODELS[model],
            "bed_formula": "input mixture - dialogue estimate, both 44.1 kHz stereo",
            "training_data": "MUSDB18-HQ only (Track A); music source separation, not dialogue",
            "model_scope": "vocals network only; the challenge Demucs blend and mixer are not used",
            "stft": f"n_fft {N_FFT}, hop {HOP}, Hann, centered, first {DIM_F} bins kept (band {BAND_LIMIT_HZ:.0f} Hz)",
            "chunking": f"{CHUNK} samples at stride {GEN_SIZE}, {TRIM} trimmed each side, no overlap-add",
            "inference_batch": str(BATCH), "native_sample_rate": str(rc.SEPARATOR_SR),
            "separator_sample_rate": str(rc.SEPARATOR_SR), "channels": "stereo network input, no resampling",
            "output": "float32 stereo WAV", "device": sep.device_name, "selected_provider": sep.selected,
            "versions": json.dumps(sep.versions, sort_keys=True)}


def main(argv: list[str]) -> int:
    args = rc.parse_command_line(argv, list(MODELS))
    try:
        sep = Separator(args.model_directory, args.provider)
    except rc.RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    selected = "cuda" if sep.selected == "CUDAExecutionProvider" else "cpu"
    meta = metadata(sep, args.model)
    meta["selected_provider"] = selected
    return rc.run_all(sep.estimate, args, rc.RunContext(sep.torch, "cpu", args.provider or selected, meta))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
