#!/usr/bin/env python3
"""Out-of-process separation runner for Alibaba MossFormer2_SE_48K (ClearerVoice-Studio speech enhancement).

    separation-eval --jobs <jobs.jsonl> --results <results.jsonl> --model mossformer2-se-48k-residual
                    [--provider cpu|cuda] --model-directory <dir holding last_best_checkpoint.pt and repo/>

The checkpoint is `last_best_checkpoint.pt` from huggingface.co/alibabasglab/MossFormer2_SE_48K (revision pinned
below, Apache-2.0); its sha256 is checked before it is read. `repo/` is a checkout of modelscope/ClearerVoice-Studio
at the pinned revision (Apache-2.0). Only the model class and the official 48 kHz decoder are imported from it, with
a stub in place of the `clearvoice` package `__init__`, so none of the package's downloader or CLI code runs.
The checkpoint is loaded with `weights_only=True`, and every parameter of the network must be found in it (the
official loader skips a key that is missing or the wrong shape and runs on with random weights; this one fails).

This is a speech-enhancement model, not a separator: it estimates clean speech, and the bed is the residual
`input - enhanced speech` (rubric 5.4). Whatever it treats as noise (crowd, walla, room tone, music) lands in the bed,
and whatever it treats as speech stays out of it.

Pipeline per job, recorded in the result metadata:
  1. read the 44.1 kHz stereo PCM16 input the scorer prepared (float);
  2. resample each channel to the model's 48 kHz (torchaudio sinc; the official reader uses librosa);
  3. enhance each channel independently with the official decoder (no input normalisation for this model; whole
     clip up to 20 s, otherwise 4 s windows at a 3 s stride), the Kaldi filterbank dither seeded per job so a rerun is
     reproducible;
  4. resample the estimate back to 44.1 kHz and fix the length to the input length;
  5. the dialogue is that estimate; the bed is the input mixture minus it, in the 44.1 kHz domain.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runner_common as rc  # noqa: E402

CHECKPOINT_NAME = "last_best_checkpoint.pt"
CHECKPOINT_SHA256 = "03692b9f773bbd6bb43b9c5a41f96b1e28affd66e13796b7bec66ad3d8b227c6"  # Hugging Face file page
HF_REVISION = "2b49ae38eaf136f3104cc1f7addb684b63c2d984"
CODE_REVISION = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"  # modelscope/ClearerVoice-Studio
SOURCE_FILES = ["utils/decode.py", "utils/misc.py", "utils/bandwidth_sub.py", "dataloader/meldataset.py",
                "config/inference/MossFormer2_SE_48K.yaml",
                *(f"models/mossformer2_se/{name}.py" for name in ("__init__", "conv_module", "fsmn", "layer_norm", "mossformer2",
                                                                  "mossformer2_block", "mossformer2_se_wrapper"))]
MODELS = {"mossformer2-se-48k-residual": "residual"}
NATIVE_SR = 48000
# config/inference/MossFormer2_SE_48K.yaml
DECODER_ARGS = dict(sampling_rate=NATIVE_SR, one_time_decode_length=20, decode_window=4, network="MossFormer2_SE_48K",
                    win_type="hamming", win_len=1920, win_inc=384, fft_len=1920, num_mels=60)
DITHER_SEED = 1234
DECODER_WINDOW = DECODER_ARGS["decode_window"] * NATIVE_SR
DECODER_STRIDE = DECODER_WINDOW * 3 // 4
DECODER_ONE_PASS = DECODER_ARGS["one_time_decode_length"] * NATIVE_SR


def decoder_padding(n: int) -> int:
    """One zero sample for a long input whose length fits the decoder's windows exactly. The official long-input path
    leaves the last (window - stride) / 2 output samples unfilled in that case; any other length is padded by the
    decoder itself and filled, so the extra sample (cut off again afterwards) moves it onto that path."""
    aligned = n >= DECODER_WINDOW + DECODER_STRIDE and (n - DECODER_WINDOW) % DECODER_STRIDE == 0
    return 1 if n > DECODER_ONE_PASS and aligned else 0


def load_clearvoice_modules(inner: Path):
    """The decoder and the model wrapper, imported without running the package's `__init__`."""
    for key in [k for k in sys.modules if k == "clearvoice" or k.startswith("clearvoice.")]:
        del sys.modules[key]  # a cached import from another checkout would make the source hashes meaningless
    stub = types.ModuleType("clearvoice")
    stub.__path__ = [str(inner)]
    sys.modules["clearvoice"] = stub
    decode = importlib.import_module("clearvoice.utils.decode")
    wrapper = importlib.import_module("clearvoice.models.mossformer2_se.mossformer2_se_wrapper")
    return decode, wrapper


def pretrained_value(name: str, pretrained: dict, shape):
    for key in (name, name.replace("module.", ""), "module." + name):
        if key in pretrained and tuple(pretrained[key].shape) == tuple(shape):
            return pretrained[key]
    return None


def load_strict(model, checkpoint: dict) -> None:
    """The official loader's key mapping, except that a parameter it cannot fill is an error."""
    pretrained = checkpoint["model"] if "model" in checkpoint else checkpoint
    state = model.state_dict()
    missing = []
    for name, tensor in state.items():
        value = pretrained_value(name, pretrained, tensor.shape)
        if value is None:
            missing.append(name)
        else:
            state[name] = value
    if missing:
        raise rc.RunnerError(f"{len(missing)} parameters not found in the checkpoint (first: {missing[0]})")
    model.load_state_dict(state, strict=True)


def verified_checkpoint(model_directory: Path) -> tuple[Path, str]:
    path = model_directory / CHECKPOINT_NAME
    if not path.is_file():
        raise rc.RunnerError(f"checkpoint not found: {path}")
    digest = rc.file_hash(path, "sha256")
    if digest != CHECKPOINT_SHA256:
        raise rc.RunnerError(f"checkpoint sha256 {digest} is not the pinned {CHECKPOINT_SHA256}")
    return path, digest


def source_hashes(repo: Path, inner: Path) -> dict:
    if rc.git_head(repo) != CODE_REVISION:
        raise rc.RunnerError(f"{repo} is not at the pinned revision {CODE_REVISION}")
    rc.require_unmodified(repo, [str((inner / name).relative_to(repo)).replace("\\", "/") for name in SOURCE_FILES])
    return {name: rc.file_hash(inner / name, "sha256") for name in SOURCE_FILES}


def pick_device(torch, provider: str | None):
    wanted = (provider or "").lower()
    if wanted not in ("", "cpu", "cuda", "gpu"):
        raise rc.RunnerError(f"unsupported provider {provider!r}; use cpu or cuda")
    if wanted == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if wanted:
        raise rc.RunnerError("provider cuda requested but CUDA is not available")
    return torch.device("cpu")


class Separator:
    def __init__(self, model_directory: Path, provider: str | None):
        import torch
        import torchaudio

        self.torch, self.ta = torch, torchaudio
        checkpoint_path, self.checkpoint_sha256 = verified_checkpoint(model_directory)
        repo = model_directory / "repo"
        inner = repo / "clearvoice" / "clearvoice"
        self.source_hashes = source_hashes(repo, inner)
        self.device = pick_device(torch, provider)
        self.decode, wrapper = load_clearvoice_modules(inner)
        self.args = SimpleNamespace(**DECODER_ARGS, use_cuda=int(self.device.type == "cuda"))
        model = wrapper.MossFormer2_SE_48K(self.args).model
        load_strict(model, torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        self.model = model.to(self.device).eval()
        self.versions = {"torch": torch.__version__, "torchaudio": torchaudio.__version__,
                         "cuda": str(torch.version.cuda), "python": sys.version.split()[0]}
        self.device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu"

    def enhance_channel(self, channel_48k):
        """channel_48k: float32 (n,) tensor at 48 kHz -> enhanced float32 (n,) tensor."""
        torch = self.torch
        n = channel_48k.shape[0]
        padded = torch.nn.functional.pad(channel_48k.cpu(), (0, decoder_padding(n)))
        with torch.inference_mode():
            out = self.decode.decode_one_audio_mossformer2_se_48k(self.model, self.device, padded.numpy()[None, :], self.args)
        out = torch.from_numpy(out).float().reshape(-1)
        if out.shape[0] < n or not bool(torch.isfinite(out).all()):
            raise rc.RunnerError(f"enhanced channel is not finite or too short ({out.shape[0]} of {n} samples)")
        return out[:n]

    def estimate(self, mix):
        """mix: float32 (2, n) at 44.1 kHz -> (dialogue, bed)."""
        torch, ta = self.torch, self.ta
        n = mix.shape[1]
        torch.manual_seed(DITHER_SEED)
        up = ta.functional.resample(mix, rc.SEPARATOR_SR, NATIVE_SR)
        enhanced = torch.stack([self.enhance_channel(up[c].contiguous()) for c in range(2)])
        back = ta.functional.resample(enhanced, NATIVE_SR, rc.SEPARATOR_SR)
        dialogue = rc.fix_length(torch, back, n).contiguous()
        return dialogue, mix - dialogue


def metadata(sep: Separator, model: str) -> dict:
    return {"runner": "mossformer2-python", "engine_family": "mossformer2", "model": model,
            "checkpoint": CHECKPOINT_NAME, "checkpoint_sha256": sep.checkpoint_sha256,
            "weights_source": f"huggingface.co/alibabasglab/MossFormer2_SE_48K@{HF_REVISION}",
            "weights_license": "Apache-2.0 (model card)", "code_repo": "github.com/modelscope/ClearerVoice-Studio",
            "code_revision": CODE_REVISION, "source_file_sha256": json.dumps(sep.source_hashes, sort_keys=True),
            "runner_sha256": rc.file_hash(Path(__file__), "sha256"), "bed_mode": MODELS[model],
            "bed_formula": "input mixture - enhanced speech, both 44.1 kHz stereo after length restoration",
            "model_scope": "speech enhancement (clean speech estimate); not a source separator",
            "training_data": "large-scale open-source and private data (model card)",
            "decoder": "official decode_one_audio_mossformer2_se_48k: whole clip up to 20 s, else 4 s windows, 3 s stride",
            "dither": f"Kaldi filterbank dither seeded with {DITHER_SEED} per job",
            "native_sample_rate": str(NATIVE_SR), "separator_sample_rate": str(rc.SEPARATOR_SR),
            "resample": "torchaudio.functional.resample sinc, 44100->48000 on input and 48000->44100 on the estimate",
            "channels": "each input channel enhanced independently as mono", "output": "float32 stereo WAV",
            "device": sep.device_name, "selected_provider": sep.device.type, "versions": json.dumps(sep.versions, sort_keys=True)}


def main(argv: list[str]) -> int:
    args = rc.parse_command_line(argv, list(MODELS))
    try:
        sep = Separator(args.model_directory, args.provider)
    except rc.RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    ctx = rc.RunContext(sep.torch, sep.device.type, args.provider or sep.device.type, metadata(sep, args.model))
    return rc.run_all(sep.estimate, args, ctx)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
