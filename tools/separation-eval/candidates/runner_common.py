"""Shared pieces of the out-of-process separation runners (MDX-Net, MossFormer2).

`run_eval.py` drives a runner with `--runner "<python> <runner.py>"`; the runner speaks the command line of
`Trackdub.Benchmarks separation-eval`:

    separation-eval --jobs <jobs.jsonl> --results <results.jsonl> --model <candidate> [--provider cpu|cuda]
                    --model-directory <dir>

Each job gives a 44.1 kHz stereo PCM16 `input` and two output paths, `vocals_output` and `bed_output`. A runner
supplies one function, `estimate(mix) -> (dialogue, bed)`, on a float32 tensor of shape (2, n) at 44.1 kHz. A job
fails, and is recorded as failed with no stems left behind, on any exception, on non-finite output or on a wrong
shape; a failure is never recorded as silent output. Stems are written as float32 WAV, so no output quantisation
is declared. (bandit_v2_runner.py predates this module and keeps its own copy of these helpers so that its
recorded runner hash stays the one the baselines were scored with.)
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple

SEPARATOR_SR = 44100
LENGTH_FIX_MAX = 4


class RunnerError(Exception):
    pass


class RunContext(NamedTuple):
    torch: object
    device_type: str
    provider: str
    meta: dict


def parse_command_line(argv: list[str], models: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("command", choices=["separation-eval"])
    p.add_argument("--jobs", required=True, type=Path)
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--model", required=True, choices=sorted(models))
    p.add_argument("--provider", default=None)
    p.add_argument("--model-directory", required=True, type=Path)
    p.add_argument("--model-cache-directory", default=None)  # accepted for command-line compatibility
    return p.parse_args(argv)


def file_hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head(repo: Path) -> str:
    import subprocess

    out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RunnerError(f"cannot read the revision of {repo}: {out.stderr.strip()}")
    return out.stdout.strip()


def _memory_counters():
    """Memory counters of this process. Returns (peak, current) bytes."""
    return _windows_counters() if sys.platform == "win32" else _posix_counters()


def _posix_counters():
    """Peak resident size from getrusage (KiB on Linux, bytes on macOS); current from /proc where there is one."""
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            return peak, int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return peak, peak


def _windows_counters():
    """Working-set counters of this process (Windows)."""
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    c = Counters()
    c.cb = ctypes.sizeof(c)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise RunnerError("GetProcessMemoryInfo failed")
    return int(c.PeakWorkingSetSize), int(c.WorkingSetSize)


def read_wav_float(path: str):
    import soundfile as sf
    import torch

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != SEPARATOR_SR or data.shape[1] != 2:
        raise RunnerError(f"expected 44.1 kHz stereo input, got {rate} Hz x {data.shape[1]} channels")
    return torch.from_numpy(data.T.copy())


def write_wav_float(path: str, x) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x.numpy().T, SEPARATOR_SR, subtype="FLOAT")


def fix_length(torch, x, n: int):
    """Trim or zero-pad a stem to the input length; a larger difference is an error, not something to hide."""
    delta = x.shape[1] - n
    if abs(delta) > LENGTH_FIX_MAX:
        raise RunnerError(f"output length differs from the input by {delta} samples")
    if delta > 0:
        return x[:, :n]
    if delta < 0:
        return torch.nn.functional.pad(x, (0, -delta))
    return x


def _new_record(job: dict, index: int, ctx: RunContext) -> dict:
    return {"id": job["id"], "job_index": index, "ok": False, "error": None, "wall_ms": 0.0,
            "audio_seconds": 0.0, "rtf": None, "working_set_before_bytes": _memory_counters()[1],
            "peak_working_set_bytes": None, "requested_provider": ctx.provider,
            "selected_provider": ctx.meta.get("selected_provider", ctx.device_type), "bootstrap_detail": None, "metadata": dict(ctx.meta)}


def _write_stems(job: dict, stems: dict, mix, torch) -> list[str]:
    written = []
    for name, stem in stems.items():
        if stem.shape != mix.shape or not bool(torch.isfinite(stem).all()):
            raise RunnerError(f"{name} is not finite or not the input's shape")
        write_wav_float(job[name], stem)
        written.append(job[name])
    return written


def _finish_record(record: dict, started: float, ctx: RunContext) -> None:
    elapsed = time.perf_counter() - started
    record["wall_ms"] = elapsed * 1000.0
    if record["ok"] and record["audio_seconds"] > 0:
        record["rtf"] = elapsed / record["audio_seconds"]
    record["peak_working_set_bytes"] = _memory_counters()[0]
    if ctx.device_type == "cuda":
        record["metadata"]["peak_cuda_allocated_bytes"] = str(ctx.torch.cuda.max_memory_allocated())
        ctx.torch.cuda.reset_peak_memory_stats()


def run_job(estimate: Callable, job: dict, index: int, ctx: RunContext) -> dict:
    record = _new_record(job, index, ctx)
    started = time.perf_counter()
    written: list[str] = []
    try:
        mix = read_wav_float(job["input"])
        record["audio_seconds"] = mix.shape[1] / SEPARATOR_SR
        dialogue, bed = estimate(mix)
        written = _write_stems(job, {"vocals_output": dialogue, "bed_output": bed}, mix, ctx.torch)
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001 - every failure is recorded per job, never as silent output
        record["error"] = f"{type(exc).__name__}: {exc}"
        for path in written:
            Path(path).unlink(missing_ok=True)
    _finish_record(record, started, ctx)
    return record


def read_jobs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def warm_up(estimate: Callable, ctx: RunContext) -> float:
    """One untimed pass so CUDA context, cuDNN and allocator start-up are not billed to the first clip."""
    torch = ctx.torch
    started = time.perf_counter()
    estimate(torch.zeros(2, SEPARATOR_SR * 2) + 1e-3 * torch.randn(2, SEPARATOR_SR * 2))
    if ctx.device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    return (time.perf_counter() - started) * 1000.0


def run_all(estimate: Callable, args: argparse.Namespace, ctx: RunContext) -> int:
    jobs = read_jobs(args.jobs)
    try:
        ctx.meta["warmup_ms"] = f"{warm_up(estimate, ctx):.0f}"
    except Exception as exc:  # noqa: BLE001
        print(f"error: warm-up failed: {exc}", file=sys.stderr)
        return 1
    failed = 0
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w", encoding="utf-8") as out:
        for index, job in enumerate(jobs):
            record = run_job(estimate, job, index, ctx)
            failed += 0 if record["ok"] else 1
            out.write(json.dumps(record) + "\n")
            out.flush()
            status = "ok" if record["ok"] else "FAILED: " + str(record["error"])
            print(f"[{index + 1}/{len(jobs)}] {job['id']} {status}  {record['wall_ms'] / 1000.0:.1f}s", file=sys.stderr)
    return 2 if failed else 0
