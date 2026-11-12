import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ingest
from test_support import running_server


def write_wav(path: Path, seconds: float = 0.5, rate: int = 16000, tone: int = 1) -> None:
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", (i * tone) % 2000) for i in range(n)))


def rec(item_id="dlg-0001", **over):
    base = {
        "id": item_id, "source": "librivox", "url": "https://example.org/a.wav",
        "role": "dialogue", "group": "librivox:r1", "license_spdx": "CC0-1.0",
        "license_evidence_url": "https://example.org/license", "attribution_text": "x",
    }
    base.update(over)
    return base


def fetcher(tone_by_url=None):
    def fetch(url, dest, range_bytes=None):
        write_wav(dest, tone=(tone_by_url or {}).get(url, 1))
    return fetch


class LicenseTests(unittest.TestCase):
    def test_allowed(self):
        self.assertIsNone(ingest.check_license("CC-BY-4.0", False))

    def test_noncommercial_and_noderivs_rejected(self):
        for spdx in ("CC-BY-NC-4.0", "CC-BY-ND-4.0", "CC-BY-NC-SA-4.0"):
            self.assertIn("noncommercial", ingest.check_license(spdx, True))

    def test_unknown_rejected(self):
        self.assertIn("allow-list", ingest.check_license("Proprietary", True))

    def test_sharealike_needs_flag(self):
        self.assertIn("share-alike", ingest.check_license("CC-BY-SA-4.0", False))
        self.assertIsNone(ingest.check_license("CC-BY-SA-4.0", True))


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_ingest(self, records, **kw):
        return ingest.ingest_items(records, self.cache, fetch=kw.pop("fetch", fetcher()), **kw)

    def test_accepts_and_records_fields(self):
        items, rejected = self.run_ingest([rec()])
        self.assertEqual(rejected, [])
        self.assertEqual(items[0]["sample_rate"], 16000)
        self.assertEqual(items[0]["channels"], 1)
        self.assertTrue(items[0]["excluded_from_training"])
        self.assertEqual(len(items[0]["sha256"]), 64)

    def test_rejects_bad_records(self):
        bad = [rec("Bad_ID"), rec("dlg-0002", url="http://x/a.wav"), rec("dlg-0003", role="vocals"),
               rec("dlg-0004", license_spdx="CC-BY-NC-4.0"), rec("dlg-0005", group="")]
        items, rejected = self.run_ingest(bad)
        self.assertEqual(items, [])
        self.assertEqual(len(rejected), 5)

    def test_rejects_missing_or_invalid_license_evidence_url(self):
        for value in (None, "", "not a URL", "http://example.org/license"):
            with self.subTest(value=value):
                self.assertIn("license_evidence_url", ingest.validate_record(rec(license_evidence_url=value), False))

    def test_rejects_malformed_tags(self):
        for value in ("instrumental", None, ["music", 5]):
            with self.subTest(value=value):
                self.assertIn("tags", ingest.validate_record(rec(tags=value), False))

    def test_normalizes_valid_tags(self):
        items, rejected = self.run_ingest([rec(tags=["LOUD", "loud", " Crowd "])])
        self.assertEqual(rejected, [])
        self.assertEqual(items[0]["tags"], ["crowd", "loud"])

    def test_rejects_nonpositive_per_host_limit_without_starting_downloads(self):
        for per_host in (0, -1):
            with self.subTest(per_host=per_host), self.assertRaisesRegex(ingest.IngestError, "positive"):
                self.run_ingest([rec()], per_host=per_host)

    def test_rejects_extensions_that_could_escape_the_cache(self):
        for ext in ("../../../../outside", "wav/../../outside", ".."):
            with self.subTest(ext=ext):
                items, rejected = self.run_ingest([rec(ext=ext)])
                self.assertEqual(items, [])
                self.assertIn("ext", rejected[0].reason)
                with self.assertRaises(ingest.IngestError):
                    ingest.cache_path(self.cache, rec(ext=ext))

    def test_hash_mismatch_rejected(self):
        items, rejected = self.run_ingest([rec(sha256="0" * 64)])
        self.assertEqual(items, [])
        self.assertIn("sha256 mismatch", rejected[0].reason)

    def test_duplicate_bytes_and_ids_rejected(self):
        items, rejected = self.run_ingest([rec("dlg-0001"), rec("dlg-0002"), rec("dlg-0001")])
        self.assertEqual([i["id"] for i in items], ["dlg-0001"])
        self.assertEqual(len(rejected), 2)

    def test_download_failure_is_per_item(self):
        def boom(url, dest, range_bytes=None):
            raise OSError("offline")
        items, rejected = self.run_ingest([rec()], fetch=boom)
        self.assertEqual(items, [])
        self.assertIn("download failed", rejected[0].reason)

    def test_split_is_group_level_and_deterministic(self):
        self.assertEqual(ingest.assign_split("g1"), "test")
        self.assertEqual(ingest.assign_split("speaker:alice"), "dev")
        splits = {ingest.assign_split(f"group-{i}") for i in range(200)}
        self.assertEqual(splits, {"dev", "test"})

    def test_same_group_same_split(self):
        urls = {"https://example.org/b.wav": 2}
        items, _ = self.run_ingest(
            [rec("dlg-0001"), rec("dlg-0002", url="https://example.org/b.wav")],
            fetch=fetcher(urls))
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["split"], items[1]["split"])


class RangeTests(unittest.TestCase):
    def test_range_bytes_is_passed_to_the_fetcher_and_kept_in_the_manifest(self):
        seen = []

        def fetch(url, dest, range_bytes=None):
            seen.append(range_bytes)
            write_wav(dest)

        with tempfile.TemporaryDirectory() as td:
            items, rejected = ingest.ingest_items([rec(range_bytes=3000000, tags=["Loud"], tag_basis="search-query",
                                                       title="t", creator="c")], Path(td), fetch=fetch)
        self.assertEqual(rejected, [])
        self.assertEqual(seen, [3000000])
        self.assertEqual(items[0]["range_bytes"], 3000000)
        self.assertEqual(items[0]["tags"], ["loud"])
        self.assertEqual(items[0]["tag_basis"], "search-query")

    def test_bad_range_bytes_rejected(self):
        for bad in (10, "100000", True, 1.5):
            self.assertIn("range_bytes", ingest.validate_record(rec(range_bytes=bad), False))

    def test_http_fetch_truncates_to_the_range_even_if_the_server_ignores_it(self):
        import http.server
        payload = bytes(range(256)) * 40

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        with running_server(Handler) as server:
            with tempfile.TemporaryDirectory() as td:
                dest = Path(td) / "a.bin"
                ingest.http_fetch(f"http://127.0.0.1:{server.server_port}/a", dest, range_bytes=2048)
                self.assertEqual(dest.read_bytes(), payload[:2048])
                ingest.http_fetch(f"http://127.0.0.1:{server.server_port}/a", dest)
                self.assertEqual(dest.read_bytes(), payload)

    def test_short_http_response_is_not_committed_as_a_complete_download(self):
        import http.server
        payload = b"partial-audio"

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload) + 20))
                self.end_headers()
                self.wfile.write(payload)
                self.close_connection = True

            def log_message(self, *a):
                pass

        with running_server(Handler) as server:
            with tempfile.TemporaryDirectory() as td:
                dest = Path(td) / "partial.bin"
                with self.assertRaisesRegex(ingest.IngestError, "download failed|incomplete download"):
                    ingest.http_fetch(f"http://127.0.0.1:{server.server_port}/audio", dest)
                self.assertFalse(dest.exists())
                self.assertFalse(dest.with_suffix(dest.suffix + ".part").exists())

    def test_https_redirect_handler_rejects_downgrade(self):
        handler = ingest._HttpsRedirectHandler()
        with self.assertRaisesRegex(ingest.IngestError, "non-HTTPS"):
            handler.redirect_request(None, None, 302, "Found", {}, "http://example.org/audio")

    def test_https_redirect_handler_rejects_loopback_target_from_a_remote_request(self):
        handler = ingest._HttpsRedirectHandler()
        req = ingest.urllib.request.Request("https://example.org/audio")
        with self.assertRaisesRegex(ingest.IngestError, "non-HTTPS"):
            handler.redirect_request(req, None, 302, "Found", {}, "http://127.0.0.1:9/audio")

    def test_bearer_token_reaches_the_requested_url_but_not_its_redirect_target(self):
        import http.server
        seen = {}

        class Target(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                seen["target"] = self.headers.get("Authorization")
                self.send_response(200)
                self.send_header("Content-Length", "5")
                self.end_headers()
                self.wfile.write(b"audio")

            def log_message(self, *a):
                pass

        with running_server(Target) as target:
            class Origin(http.server.BaseHTTPRequestHandler):
                def do_GET(self):
                    seen["origin"] = self.headers.get("Authorization")
                    self.send_response(302)
                    self.send_header("Location", f"http://127.0.0.1:{target.server_port}/file")
                    self.end_headers()

                def log_message(self, *a):
                    pass

            with running_server(Origin) as origin, tempfile.TemporaryDirectory() as td:
                dest = Path(td) / "a.bin"
                ingest.http_fetch(f"http://127.0.0.1:{origin.server_port}/download", dest, bearer_token="tok")
                self.assertEqual(dest.read_bytes(), b"audio")
        self.assertEqual(seen, {"origin": "Bearer tok", "target": None})

    def test_plain_http_is_rejected_outside_loopback_fixtures(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ingest.IngestError, "url must be https"):
                ingest.http_fetch("http://example.org/audio", Path(td) / "audio.bin")


class ProbeAndJsonlTests(unittest.TestCase):
    def test_probe_audio_wraps_missing_and_invalid_fields(self):
        payloads = (
            '{"streams":[{"sample_rate":"16000","channels":1}],"format":{}}',
            '{"streams":[{"sample_rate":"invalid","channels":1}],"format":{"duration":"2"}}',
        )
        for payload in payloads:
            with self.subTest(payload=payload), patch.object(
                    ingest.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=payload)), \
                    patch.object(ingest.shutil, "which", return_value="/usr/bin/ffprobe"):
                with self.assertRaisesRegex(ingest.IngestError, "invalid ffprobe output"):
                    ingest.probe_audio(Path("invalid.wav"))

    def test_read_jsonl_rejects_non_object_values_with_line_number(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "items.jsonl"
            path.write_text('{"id":"ok"}\nnull\n[]\n', encoding="utf-8")
            with self.assertRaisesRegex(ingest.IngestError, r"items\.jsonl:2: expected a JSON object"):
                ingest.read_jsonl(path)

            path.write_text('{"id":"ok"}\n[]\n', encoding="utf-8")
            with self.assertRaisesRegex(ingest.IngestError, r"items\.jsonl:2: expected a JSON object"):
                ingest.read_jsonl(path)


class ParallelTests(unittest.TestCase):
    def records(self, n, hosts=("a.example", "b.example")):
        return [rec(f"dlg-{i:04d}", url=f"https://{hosts[i % len(hosts)]}/{i}.wav", group=f"g{i}") for i in range(n)]

    @staticmethod
    def tone_for(url):
        return int(url.rsplit("/", 1)[1].split(".")[0]) + 1

    def run_parallel(self, records, fetch, **kw):
        with tempfile.TemporaryDirectory() as td:
            return ingest.ingest_items(records, Path(td), fetch=fetch, **kw)

    def tracking_fetch(self, delay=0.05):
        import threading
        import time
        lock, active, peak = threading.Lock(), [0], [0]

        def fetch(url, dest, range_bytes=None):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(delay)
            write_wav(dest, tone=self.tone_for(url))
            with lock:
                active[0] -= 1
        return fetch, peak

    def test_results_match_the_sequential_run_and_are_ordered(self):
        def fetch(url, dest, range_bytes=None):
            write_wav(dest, tone=self.tone_for(url))
        seq, _ = self.run_parallel(self.records(12), fetch, workers=1)
        par, _ = self.run_parallel(self.records(12), fetch, workers=6)
        self.assertEqual(seq, par)
        self.assertEqual([i["id"] for i in par], sorted(i["id"] for i in par))

    def test_actually_runs_downloads_concurrently(self):
        fetch, peak = self.tracking_fetch()
        self.run_parallel(self.records(12), fetch, workers=6, per_host=3)
        self.assertGreater(peak[0], 1)
        self.assertLessEqual(peak[0], 6)

    def test_per_host_limit_is_respected(self):
        fetch, peak = self.tracking_fetch()
        self.run_parallel(self.records(8, hosts=("only.example",)), fetch, workers=8, per_host=2)
        self.assertLessEqual(peak[0], 2)

    def test_one_failure_does_not_stop_the_others_and_progress_is_reported(self):
        seen = []

        def fetch(url, dest, range_bytes=None):
            if "/3." in url:
                raise OSError("reset")
            write_wav(dest, tone=self.tone_for(url))
        items, rejected = self.run_parallel(self.records(6), fetch, workers=4,
                                            progress=lambda d, t, i: seen.append((d, t)))
        self.assertEqual(len(items), 5)
        self.assertEqual([r.item_id for r in rejected], ["dlg-0003"])
        self.assertIn("download failed", rejected[0].reason)
        self.assertEqual(sorted(seen), [(i, 6) for i in range(1, 7)])

    def test_cached_files_are_not_fetched_again(self):
        calls = []

        def fetch(url, dest, range_bytes=None):
            calls.append(url)
            write_wav(dest, tone=self.tone_for(url))
        with tempfile.TemporaryDirectory() as td:
            ingest.ingest_items(self.records(5), Path(td), fetch=fetch, workers=3)
            first = len(calls)
            ingest.ingest_items(self.records(5), Path(td), fetch=fetch, workers=3)
        self.assertEqual(first, 5)
        self.assertEqual(len(calls), 5)


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name)
        items, _ = ingest.ingest_items([rec()], self.cache, fetch=fetcher())
        self.manifest = ingest.build_manifest(items)

    def tearDown(self):
        self.tmp.cleanup()

    def test_clean_manifest_verifies(self):
        self.assertEqual(ingest.verify_manifest(self.manifest, self.cache), [])

    def test_edited_manifest_detected(self):
        m = json.loads(json.dumps(self.manifest))
        m["items"][0]["attribution_text"] = "changed"
        self.assertTrue(any("edited" in p for p in ingest.verify_manifest(m, self.cache)))

    def test_corrupted_cache_detected(self):
        ingest.cache_path(self.cache, rec()).write_bytes(b"corrupt")
        self.assertTrue(any("hash differs" in p for p in ingest.verify_manifest(self.manifest, self.cache)))

    def test_missing_cache_detected(self):
        ingest.cache_path(self.cache, rec()).unlink()
        self.assertTrue(any("missing" in p for p in ingest.verify_manifest(self.manifest, self.cache)))


class SplitPendingTests(unittest.TestCase):
    def test_a_local_only_item_that_changed_is_never_queued_for_download(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "a.wav"
            dest.write_bytes(b"changed")
            local = {"id": "loc", "local_only": True, "sha256": "0" * 64, "url": "https://example.org/x.wav"}
            remote = {"id": "rem", "sha256": "0" * 64, "url": "https://example.org/y.wav"}
            pending, changed = ingest.split_pending([(local, dest), (remote, dest)])
        self.assertEqual([rec["id"] for rec, _ in pending], ["rem"])
        self.assertEqual(list(changed), ["loc"])

    def test_matching_files_need_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "a.wav"
            dest.write_bytes(b"abc")
            rec = {"id": "x", "local_only": True, "sha256": ingest.sha256_file(dest), "url": "https://example.org/x"}
            self.assertEqual(ingest.split_pending([(rec, dest)]), ([], {}))


if __name__ == "__main__":
    unittest.main()
