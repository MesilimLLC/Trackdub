#!/usr/bin/env python3
"""Run a mock controlled-matrix stage and fail if normalized process CPU exceeds budget."""

from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any


CPU_BUDGET_PERCENT = 95.0
RUN_COUNT = 3
STAGE = "audio-preparation"


def write_fixture(path: Path) -> None:
    """Create a one-second, valid mono PCM fixture without checked-in media."""
    sample_count = 16_000
    with wave.open(str(path), "wb") as fixture:
        fixture.setnchannels(1)
        fixture.setsampwidth(2)
        fixture.setframerate(16_000)
        fixture.writeframes(struct.pack("<h", 0) * sample_count)


def validate_report(path: Path) -> None:
    report: Any = json.loads(path.read_text(encoding="utf-8"))
    if report.get("Status") != "Completed":
        raise RuntimeError(f"Controlled matrix status was {report.get('Status')!r}, expected Completed.")

    results = report.get("Results")
    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError("Controlled matrix must contain exactly one focused-stage result.")
    result = results[0]
    if result.get("Stage") != STAGE:
        raise RuntimeError(f"Controlled matrix measured {result.get('Stage')!r}, expected {STAGE!r}.")

    evidence = result.get("Evidence") or {}
    if evidence.get("Status") != "Completed":
        raise RuntimeError(f"Stage evidence status was {evidence.get('Status')!r}, expected Completed.")

    samples = [
        sample
        for sample in evidence.get("ResourceTelemetry", [])
        if sample.get("Phase") == "measured"
    ]
    if len(samples) != RUN_COUNT:
        raise RuntimeError(f"Expected {RUN_COUNT} measured telemetry samples, found {len(samples)}.")

    observed: list[float] = []
    for expected_iteration, sample in enumerate(samples, start=1):
        if sample.get("Iteration") != expected_iteration:
            raise RuntimeError(
                f"Measured sample order drifted: expected iteration {expected_iteration}, "
                f"found {sample.get('Iteration')!r}."
            )
        if sample.get("ExecutionStatus") != "Completed":
            raise RuntimeError(f"Stage execution iteration {expected_iteration} did not complete.")

        validation = sample.get("Validation") or {}
        cpu_checks = [check for check in validation.get("Checks", []) if check.get("Metric") == "cpuPercent"]
        if len(cpu_checks) != 1:
            raise RuntimeError(f"Iteration {expected_iteration} must contain exactly one CPU check.")
        check = cpu_checks[0]
        value = check.get("ObservedValue")
        if check.get("Threshold") != CPU_BUDGET_PERCENT:
            raise RuntimeError(
                f"Iteration {expected_iteration} CPU threshold drifted: {check.get('Threshold')!r}."
            )
        if check.get("Status") != "Passed":
            raise RuntimeError(
                f"Iteration {expected_iteration} CPU budget failed: "
                f"observed={check.get('ObservedValue')!r}, threshold={check.get('Threshold')!r}."
            )
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RuntimeError(f"Iteration {expected_iteration} has no finite normalized CPU reading: {value!r}.")
        if value > CPU_BUDGET_PERCENT:
            raise RuntimeError(
                f"Iteration {expected_iteration} used {value:.2f}% normalized CPU, "
                f"over the {CPU_BUDGET_PERCENT:.2f}% budget."
            )
        observed.append(float(value))

    print(
        f"Controlled-matrix CPU budget passed: {STAGE}, {RUN_COUNT} runs, "
        f"max={max(observed):.2f}% / budget={CPU_BUDGET_PERCENT:.2f}% normalized CPU."
    )


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="trackdub-controlled-cpu-") as temporary:
        root = Path(temporary)
        fixture = root / "cpu-budget.wav"
        output = root / "reports"
        output.mkdir()
        write_fixture(fixture)

        command = [
            "dotnet",
            "run",
            "--project",
            "src/Trackdub.Benchmarks.DevHost",
            "--configuration",
            "Release",
            "--no-build",
            "--framework",
            "net11.0",
            "--",
            "controlled-matrix",
            str(fixture),
            "--output",
            str(output),
            "--stages",
            STAGE,
            "--runs",
            str(RUN_COUNT),
            "--max-cpu-percent",
            str(CPU_BUDGET_PERCENT),
            "--mock",
        ]
        print("+", " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=repo, text=True, capture_output=True, check=False)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        reports = sorted(output.glob("stage-matrix-*.json"))
        if len(reports) != 1:
            raise RuntimeError(
                f"controlled-matrix exited with status {completed.returncode}; "
                f"expected one stage-matrix report, found {len(reports)}."
            )
        validate_report(reports[0])
        if completed.returncode != 0:
            raise RuntimeError(f"controlled-matrix exited with status {completed.returncode}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, json.JSONDecodeError, subprocess.SubprocessError) as error:
        print(f"CPU budget gate failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
