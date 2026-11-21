import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from summarize_result import summarize  # noqa: E402

OVERRUN_MESSAGE = (
    "Dubbed take extends 6.826 s past the end of the source audio (40.171 s) and would be "
    "cut off at export. Stretch the take to fit, trim it, or raise the auto-stretch limit."
)


def result_with(stages, status="partialSuccess"):
    return json.dumps({"schemaVersion": 1, "status": "failed", "result": {"overallStatus": status, "stageOutcomes": stages}})


def stage(name, status, records=None, reason=None):
    return {"stageName": name, "status": status, "degradationRecords": records, "reasonCode": reason}


class SummarizeResultTests(unittest.TestCase):
    def test_overrun_failure_names_stage_and_suggests_rubberband(self):
        text = result_with([
            stage("tts", "succeeded"),
            stage("export", "failed", [OVERRUN_MESSAGE], "STAGE_FAILED"),
        ])

        line = summarize(text, exit_code=2)

        self.assertTrue(line.startswith("FAILED at export: Dubbed take extends 6.826 s"), line)
        self.assertIn("--tts-rubberband-stretch", line)
        self.assertIn("--tts-auto-stretch-max-overrun", line)
        self.assertIn("(overrun 6.826 s)", line)

    def test_unknown_failure_points_to_logs(self):
        text = result_with([stage("asr", "failed", None, "MODEL_MISSING")])

        line = summarize(text, exit_code=2)

        self.assertEqual(
            line,
            "FAILED at asr: MODEL_MISSING. Fix: read result.json and run.log for the stage's error",
        )

    def test_success_is_ok(self):
        text = json.dumps({"schemaVersion": 1, "status": "succeeded", "result": {"stageOutcomes": [stage("export", "succeeded")]}})

        self.assertEqual(summarize(text, exit_code=0), "OK")

    def test_missing_result_is_reported_not_raised(self):
        self.assertEqual(summarize(None, exit_code=125), "NO RESULT: result.json was not written (exit 125)")

    def test_malformed_result_is_reported_not_raised(self):
        self.assertEqual(summarize("{not json", exit_code=1), "NO RESULT: result.json is not valid JSON (exit 1)")

    def test_nonzero_exit_without_failed_stage_is_not_ok(self):
        text = json.dumps({"status": "cancelled", "result": {"stageOutcomes": []}})

        self.assertEqual(summarize(text, exit_code=130), "FAILED (exit 130): cancelled")


if __name__ == "__main__":
    unittest.main()
