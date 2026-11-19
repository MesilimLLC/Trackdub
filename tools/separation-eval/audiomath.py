"""Audio helpers shared by the corpus generator and the metric library.

All functions work on float32 arrays shaped (samples, channels) unless noted.
"""

from __future__ import annotations

from io import BytesIO
import math
import struct
import shutil
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000


class AudioError(Exception):
    pass


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    g = gcd(sr_from, sr_to)
    return signal.resample_poly(x, sr_to // g, sr_from // g, axis=0).astype(np.float32)


def _adapt_channels(x: np.ndarray, channels: int | None) -> np.ndarray:
    if channels in (None, x.shape[1]):
        return x
    mono = x.mean(axis=1, keepdims=True)
    return mono if channels == 1 else np.repeat(mono, channels, axis=1)


def _read_pipe_wav(data: bytes) -> tuple[int, np.ndarray]:
    """Repair the sentinel RIFF/data sizes ffmpeg must write to an unseekable pipe."""
    payload = bytearray(data)
    if payload[:4] == b"RIFF" and struct.unpack_from("<I", payload, 4)[0] == 0xFFFFFFFF:
        struct.pack_into("<I", payload, 4, len(payload) - 8)
        offset = 12
        while offset + 8 <= len(payload):
            chunk_id = payload[offset:offset + 4]
            chunk_size = struct.unpack_from("<I", payload, offset + 4)[0]
            if chunk_id == b"data":
                if chunk_size == 0xFFFFFFFF:
                    struct.pack_into("<I", payload, offset + 4, len(payload) - offset - 8)
                break
            offset += 8 + chunk_size + (chunk_size & 1)
    return wavfile.read(BytesIO(payload))


def load_audio(path: str | Path, channels: int | None = 2, sr: int = SR) -> np.ndarray:
    """Decode to float32 (n, channels) at `sr`; `channels=None` keeps the file's own channel count.
    Uses ffmpeg when present, else WAV via scipy."""
    path = Path(path)
    if have_ffmpeg():
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-c:a", "pcm_f32le", "-f", "wav", "-"],
            capture_output=True, check=False)
        if proc.returncode == 0 and proc.stdout:
            rate, data = _read_pipe_wav(proc.stdout)
            if data.ndim == 1:
                data = data[:, None]
            return resample(_adapt_channels(_to_float32(data), channels), rate, sr)
        if path.suffix.lower() != ".wav":
            raise AudioError(f"ffmpeg could not decode {path.name}: {proc.stderr.decode(errors='replace')[:200]}")
    if path.suffix.lower() != ".wav":
        raise AudioError(f"ffmpeg is required to decode {path.suffix} files")
    rate, data = wavfile.read(path)
    if data.ndim == 1:
        data = data[:, None]
    return resample(_adapt_channels(_to_float32(data), channels), rate, sr)


def _to_float32(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    return data


def write_wav(path: str | Path, x: np.ndarray, sr: int) -> None:
    wavfile.write(str(path), sr, np.ascontiguousarray(x, dtype=np.float32))


# ITU-R BS.1770-4 K-weighting coefficients, valid at 48 kHz only.
_PRE_B = [1.53512485958697, -2.69169618940638, 1.19839281085285]
_PRE_A = [1.0, -1.69065929318241, 0.73248077421585]
_RLB_B = [1.0, -2.0, 1.0]
_RLB_A = [1.0, -1.99004745483398, 0.99007225036621]
_CHANNEL_WEIGHTS_51 = [1.0, 1.0, 1.0, 0.0, 1.41, 1.41]  # FL FR FC LFE BL BR


def integrated_lufs(x: np.ndarray, sr: int = SR) -> float:
    """BS.1770-4 integrated loudness with absolute and relative gating. -inf if silent."""
    if sr != SR:
        raise AudioError("integrated_lufs is implemented for 48 kHz only")
    channels = x.shape[1]
    if channels == 6:
        weights = np.array(_CHANNEL_WEIGHTS_51)
    else:
        weights = np.ones(channels)
    y = signal.lfilter(_PRE_B, _PRE_A, x.astype(np.float64), axis=0)
    y = signal.lfilter(_RLB_B, _RLB_A, y, axis=0)
    block, step = int(0.4 * sr), int(0.1 * sr)
    n = y.shape[0]
    if n < block:
        return -math.inf
    cs = np.vstack([np.zeros((1, channels)), np.cumsum(y * y, axis=0)])
    starts = np.arange(0, n - block + 1, step)
    ms = (cs[starts + block] - cs[starts]) / block
    z = (ms * weights).sum(axis=1)
    with np.errstate(divide="ignore"):
        loud = -0.691 + 10.0 * np.log10(z)
    above = loud > -70.0
    if not above.any():
        return -math.inf
    rel_thr = -0.691 + 10.0 * math.log10(z[above].mean()) - 10.0
    keep = above & (loud > rel_thr)
    if not keep.any():
        return -math.inf
    return float(-0.691 + 10.0 * math.log10(z[keep].mean()))


def frame_activity(x: np.ndarray, sr: int, frame_ms: int = 20, floor_db: float = -40.0) -> np.ndarray:
    """Per-frame boolean mask: frames within `floor_db` of the loudest frame (and not silent)."""
    frame = max(1, int(sr * frame_ms / 1000))
    mono_power = (x.astype(np.float64) ** 2).mean(axis=1)
    frames = len(mono_power) // frame
    if frames == 0:
        return np.zeros(0, dtype=bool)
    rms = np.sqrt(mono_power[: frames * frame].reshape(frames, frame).mean(axis=1))
    peak = rms.max()
    if peak < 1e-6:
        return np.zeros(frames, dtype=bool)
    return rms > peak * 10.0 ** (floor_db / 20.0)


def active_rms(x: np.ndarray, mask: np.ndarray, sr: int, frame_ms: int = 20) -> float:
    """RMS of `x` restricted to the frames selected by `mask`."""
    frame = max(1, int(sr * frame_ms / 1000))
    sample_mask = np.repeat(mask, frame)
    n = min(len(sample_mask), x.shape[0])
    sel = x[:n][sample_mask[:n]]
    if sel.size == 0:
        return 0.0
    return float(np.sqrt((sel.astype(np.float64) ** 2).mean()))


def estimate_rt60(rir: np.ndarray, sr: int = SR) -> float:
    """Schroeder backward integration, T20 extrapolated to 60 dB. NaN if the decay is too short."""
    h = rir.astype(np.float64).reshape(-1)
    if h.size == 0:
        return math.nan
    edc = np.cumsum((h ** 2)[::-1])[::-1]
    if edc[0] <= 0:
        return math.nan
    db = 10.0 * np.log10(np.maximum(edc / edc[0], 1e-20))
    i5 = np.argmax(db <= -5.0) if (db <= -5.0).any() else None
    i25 = np.argmax(db <= -25.0) if (db <= -25.0).any() else None
    if i5 is None or i25 is None or i25 <= i5:
        return math.nan
    return float(3.0 * (i25 - i5) / sr)
