#!/usr/bin/env python3
"""Run the same BDN filter on the current checkout and a saved commit."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


PROJECT = "src/Trackdub.Benchmarks.Micro/Trackdub.Benchmarks.Micro.csproj"


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def report_path(artifacts: Path) -> Path:
    reports = sorted(artifacts.rglob("*-report-full.json"))
    if not reports:
        raise RuntimeError(f"No BenchmarkDotNet full JSON report found under {artifacts}.")
    return reports[0]


def benchmarkdotnet_version(root: Path) -> str:
    props = root / "Directory.Packages.props"
    if not props.exists():
        raise RuntimeError(f"Directory.Packages.props not found under {root}.")
    match = re.search(r'<PackageVersion Include="BenchmarkDotNet" Version="([^"]+)"', props.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"BenchmarkDotNet is not pinned in {props}.")
    return match.group(1)


def require_matching_benchmarkdotnet(repo: Path, baseline_worktree: Path, baseline_commit: str) -> None:
    current_version = benchmarkdotnet_version(repo)
    baseline_version = benchmarkdotnet_version(baseline_worktree)
    if current_version != baseline_version:
        raise RuntimeError(
            f"BenchmarkDotNet versions differ: baseline {baseline_commit} pins {baseline_version}, "
            f"current checkout pins {current_version}. Results would come from different harnesses."
        )


def benchmark_command(root: Path, filters: list[str], job: str, artifacts: Path) -> list[str]:
    return [
        "dotnet",
        "run",
        "--project",
        str(root / PROJECT),
        "--configuration",
        "Release",
        "--",
        "--filter",
        *filters,
        "--job",
        job,
        "--exporters",
        "fulljson",
        "--artifacts",
        str(artifacts),
    ]


def _numeric_statistics(entry: dict[str, Any]) -> dict[str, float]:
    statistics = entry.get("Statistics") or {}
    return {key: float(value) for key, value in statistics.items() if isinstance(value, (int, float))}


def _entry_name(entry: dict[str, Any]) -> str | None:
    name = entry.get("FullName") or entry.get("DisplayInfo")
    if isinstance(name, str):
        return name
    return None


def _comparable_entry(entry: dict[str, Any]) -> tuple[str, dict[str, float]] | None:
    name = _entry_name(entry)
    statistics = _numeric_statistics(entry)
    if name is None or "Mean" not in statistics:
        return None
    return name, statistics


def load_report(path: Path) -> dict[str, dict[str, float]]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, float]] = {}
    for entry in payload.get("Benchmarks", []):
        comparable = _comparable_entry(entry)
        if comparable is not None:
            result[comparable[0]] = comparable[1]
    if not result:
        raise RuntimeError(f"BenchmarkDotNet report {path} contains no comparable Mean statistics.")
    return result


def _print_missing(missing: list[str]) -> None:
    if not missing:
        return
    print("Unmatched benchmark entries:")
    for name in missing:
        print(f"  {name}")


def _change_percent(baseline_mean: float, current_mean: float) -> float:
    if not baseline_mean:
        return 0.0
    return ((current_mean - baseline_mean) / baseline_mean) * 100.0


def _print_comparison_rows(
    common: list[str],
    baseline: dict[str, dict[str, float]],
    current: dict[str, dict[str, float]],
    threshold: float,
) -> list[tuple[str, float]]:
    print("\n| Benchmark | Baseline mean | Current mean | Change |")
    print("|---|---:|---:|---:|")
    regressions: list[tuple[str, float]] = []
    for name in common:
        baseline_mean = baseline[name]["Mean"]
        current_mean = current[name]["Mean"]
        change = _change_percent(baseline_mean, current_mean)
        print(f"| {name} | {baseline_mean:.3f} | {current_mean:.3f} | {change:+.2f}% |")
        if change > threshold:
            regressions.append((name, change))
    return regressions


def _report_regressions(regressions: list[tuple[str, float]], threshold: float) -> bool:
    if regressions:
        print(f"\nRegression threshold exceeded ({threshold:.2f}%):")
        for name, change in regressions:
            print(f"  {name}: {change:+.2f}%")
        return False
    print(f"\nNo benchmark exceeded the {threshold:.2f}% regression threshold.")
    return True


def compare(baseline: dict[str, dict[str, float]], current: dict[str, dict[str, float]], threshold: float) -> bool:
    common = sorted(set(baseline) & set(current))
    missing = sorted((set(baseline) | set(current)) - set(common))
    _print_missing(missing)
    if not common:
        raise RuntimeError("No benchmark names were common to the baseline and current reports.")
    regressions = _print_comparison_rows(common, baseline, current, threshold)
    return _report_regressions(regressions, threshold)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--filter", required=True, nargs="+")
    parser.add_argument("--job", default="Dry")
    parser.add_argument("--threshold-percent", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    project = repo / PROJECT
    if not project.exists():
        raise RuntimeError(f"Benchmark project not found: {project}")

    baseline_commit = args.baseline_commit.strip()
    run(["git", "rev-parse", "--verify", f"{baseline_commit}^{{commit}}"], repo)

    with tempfile.TemporaryDirectory(prefix="trackdub-bdn-") as temporary:
        temporary_root = Path(temporary)
        baseline_worktree = temporary_root / "baseline"
        current_artifacts = temporary_root / "current-artifacts"
        baseline_artifacts = temporary_root / "baseline-artifacts"

        run(["git", "worktree", "add", "--detach", str(baseline_worktree), baseline_commit], repo)
        try:
            require_matching_benchmarkdotnet(repo, baseline_worktree, baseline_commit)
            run(benchmark_command(repo, args.filter, args.job, current_artifacts), repo)
            run(benchmark_command(baseline_worktree, args.filter, args.job, baseline_artifacts), baseline_worktree)

            current = load_report(report_path(current_artifacts))
            baseline = load_report(report_path(baseline_artifacts))
            if args.output_dir is not None:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(current_artifacts, args.output_dir / "current")
                shutil.copytree(baseline_artifacts, args.output_dir / "baseline")
            if not compare(baseline, current, args.threshold_percent):
                return 1
        finally:
            # The detached worktree is disposable; its generated bin/obj folders
            # prevent a normal worktree removal, hence --force.
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(baseline_worktree)],
                cwd=repo,
                check=False,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
