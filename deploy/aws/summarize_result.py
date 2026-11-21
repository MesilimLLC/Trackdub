#!/usr/bin/env python3
"""Print a one-line verdict for a Trackdub pipeline result.json, with a suggested fix when the failure is known."""

import argparse
import json
import re
import sys

OVERRUN_PATTERN = re.compile(r"extends ([0-9]+(?:\.[0-9]+)?) s past the end")
OVERRUN_HINT = (
    "TTS overran the source; rerun with --tts-rubberband-stretch and a higher "
    "--tts-auto-stretch-max-overrun (e.g. 0.35), or shorten the translation"
)


def summarize(text, exit_code):
    if text is None:
        return f"NO RESULT: result.json was not written (exit {exit_code})"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return f"NO RESULT: result.json is not valid JSON (exit {exit_code})"

    failed = [stage for stage in (data.get("result") or {}).get("stageOutcomes") or []
              if stage.get("status") == "failed"]
    if failed:
        return _failed_stage_verdict(failed[0])
    if exit_code == 0 and data.get("status") == "succeeded":
        return "OK"
    return f"FAILED (exit {exit_code}): {data.get('status') or 'no stage outcome recorded'}"


def _failed_stage_verdict(stage):
    name = stage.get("stageName", "unknown stage")
    records = stage.get("degradationRecords") or []
    reason = records[0] if records else (stage.get("reasonCode") or "no reason recorded")
    return f"FAILED at {name}: {reason}. Fix: {_hint_for(reason)}"


def _hint_for(reason):
    overrun = OVERRUN_PATTERN.search(reason)
    if overrun:
        return f"{OVERRUN_HINT} (overrun {overrun.group(1)} s)"
    return "read result.json and run.log for the stage's error"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", help="path to a pipeline result.json (may be missing)")
    parser.add_argument("--exit-code", type=int, default=0, help="exit code of the pipeline run")
    args = parser.parse_args(argv)

    try:
        with open(args.result_json, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        text = None

    print(summarize(text, args.exit_code))
    return 0


if __name__ == "__main__":
    sys.exit(main())
