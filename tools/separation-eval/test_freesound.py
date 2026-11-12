import contextlib
import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
from pathlib import Path
from unittest import mock

import freesound
import ingest


def result(sound_id=264499, **over):
    base = {
        "id": sound_id, "name": "Crowd murmur", "username": "noah0189",
        "license": "http://creativecommons.org/publicdomain/zero/1.0/", "type": "wav", "channels": 2,
        "samplerate": 48000.0, "duration": 30.5, "filesize": 5_000_000,
        "url": f"https://freesound.org/people/noah0189/sounds/{sound_id}/",
    }
    base.update(over)
    return base


SPEC = {"name": "ambience crowd", "role": "ambience", "tags": ["crowd"], "q": "crowd",
        "min_duration_s": 20, "max_duration_s": 300, "per_creator_cap": 2, "limit": 10, "pages": 3}
CREDS = freesound.Credentials("cid", "sec")
GRANT = freesound.AuthorizationGrant("code1", "ver1")
NO_EXCLUSIONS = freesound.SearchContext("2026-11-10", frozenset())


class CandidateTests(unittest.TestCase):
    def test_accepts_a_cc0_lossless_original_and_matches_the_v1_group_format(self):
        item, reason = freesound.freesound_candidate(result(), SPEC, "2026-11-10", set())
        self.assertIsNone(reason)
        self.assertEqual(item["id"], "fsorig-amb-264499")
        self.assertEqual(item["url"], "https://freesound.org/apiv2/sounds/264499/download/")
        self.assertEqual(item["group"], "freesound:noah0189")
        self.assertEqual(item["license_spdx"], "CC0-1.0")
        self.assertEqual(item["ext"], "wav")
        self.assertEqual(item["freesound_id"], 264499)
        self.assertEqual(item["tags"], ["crowd"])
        self.assertIsNone(ingest.validate_record(item, allow_sharealike=False))

    def test_rejections(self):
        cases = {
            "exclusion": (result(), {264499}),
            "license": (result(license="http://creativecommons.org/licenses/by-nc/4.0/"), set()),
            "by-not-requested": (result(license="https://creativecommons.org/licenses/by/4.0/"), set()),
            "lossy type": (result(type="mp3"), set()),
            "ambisonic": (result(channels=4), set()),
            "low rate": (result(samplerate=22050), set()),
            "too short": (result(duration=5.0), set()),
            "empty file": (result(filesize=0), set()),
            "foreign landing": (result(url="https://example.org/x"), set()),
        }
        for name, (r, excluded) in cases.items():
            with self.subTest(name):
                item, reason = freesound.freesound_candidate(r, SPEC, "2026-11-10", excluded)
                self.assertIsNone(item)
                self.assertTrue(reason)

    def test_attribution_license_is_accepted_only_when_the_spec_asks_for_it(self):
        spec = {**SPEC, "licenses": ["CC0-1.0", "CC-BY-4.0"]}
        item, _ = freesound.freesound_candidate(
            result(license="https://www.creativecommons.org/licenses/by/4.0/"), spec, "2026-11-10", set())
        self.assertEqual(item["license_spdx"], "CC-BY-4.0")

    def test_aif_type_is_normalised(self):
        item, _ = freesound.freesound_candidate(result(type="aif"), SPEC, "2026-11-10", set())
        self.assertEqual(item["ext"], "aiff")

    def test_unexpected_result_shape_fails_loudly(self):
        for bad in ({k: v for k, v in result().items() if k != "license"}, result(channels="2"),
                    result(id=True), "not-an-object"):
            with self.subTest(bad=bad), self.assertRaisesRegex(freesound.FreesoundError, "unexpected search result shape"):
                freesound.freesound_candidate(bad, SPEC, "2026-11-10", set())

    def test_unsupported_spec_license_is_rejected(self):
        with self.assertRaisesRegex(freesound.FreesoundError, "licenses"):
            freesound.spec_licenses({"name": "x", "licenses": ["CC-BY-NC-4.0"]})


class SearchTests(unittest.TestCase):
    def test_filter_states_license_type_channels_rate_and_duration(self):
        f = freesound.search_filter(SPEC)
        self.assertIn('license:"Creative Commons 0"', f)
        self.assertIn("type:(wav OR flac OR aiff)", f)
        self.assertIn("channels:[1 TO 2]", f)
        self.assertIn("samplerate:[44100 TO *]", f)
        self.assertIn("duration:[20 TO 300]", f)

    def test_discover_pages_caps_creators_dedupes_and_limits(self):
        pages = {
            1: {"results": [result(1), result(2), result(3), result(4, username="other")], "next": "more"},
            2: {"results": [result(5, username="third"), result(1, username="fourth")], "next": None},
            3: {"results": [result(6, username="never")], "next": None},
        }
        requested = []

        def fetch_json(url):
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
            requested.append(query)
            return pages[int(query["page"][0])]

        items = freesound.discover_freesound([SPEC], fetch_json, NO_EXCLUSIONS)
        self.assertEqual([i["freesound_id"] for i in items], [1, 2, 4, 5])
        self.assertEqual(len(requested), 2)
        self.assertEqual(requested[0]["fields"], [freesound.SEARCH_FIELDS])
        self.assertTrue(all("token" not in q for q in requested))

    def test_same_sound_found_by_two_searches_is_kept_once(self):
        other = {**SPEC, "name": "second", "q": "murmur"}
        items = freesound.discover_freesound([SPEC, other], lambda url: {"results": [result(7)], "next": None},
                                             NO_EXCLUSIONS)
        self.assertEqual(len(items), 1)

    def test_response_without_results_fails_loudly(self):
        with self.assertRaisesRegex(freesound.FreesoundError, "no 'results' list"):
            freesound.discover_freesound([SPEC], lambda url: {"detail": "nope"}, NO_EXCLUSIONS)

    def test_read_excluded_ids_from_list_and_fsd50k_csv(self):
        with tempfile.TemporaryDirectory() as td:
            plain = Path(td) / "ids.txt"
            plain.write_text("# comment\n101\n102\n", encoding="utf-8")
            csv = Path(td) / "dev.csv"
            csv.write_text("fname,labels,mids,split\n64760,Electric_guitar,/m/02sgy,train\n16399,Bell,/m/0395lw,val\n",
                           encoding="utf-8")
            self.assertEqual(freesound.read_excluded_ids([plain, csv]), {101, 102, 64760, 16399})
            empty = Path(td) / "empty.csv"
            empty.write_text("fname,labels\n", encoding="utf-8")
            with self.assertRaisesRegex(freesound.FreesoundError, "no Freesound ids"):
                freesound.read_excluded_ids([empty])


class TokenTests(unittest.TestCase):
    def test_pkce_challenge_matches_the_rfc_7636_example(self):
        verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        self.assertEqual(freesound.pkce_challenge(verifier), "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM")

    def test_authorize_url_asks_for_an_s256_challenge(self):
        url = freesound.authorize_url("cid", "st", "chal")
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        self.assertEqual(query, {"client_id": ["cid"], "response_type": ["code"], "state": ["st"],
                                 "code_challenge": ["chal"], "code_challenge_method": ["S256"]})

    def test_exchange_posts_the_authorization_code_grant_with_the_verifier(self):
        posted = []

        def post(url, fields):
            posted.append((url, fields))
            return {"access_token": "a1", "refresh_token": "r1", "expires_in": 86400, "scope": "read write"}

        record = freesound.exchange_code(GRANT, CREDS, post=post, now=lambda: 1000.0)
        self.assertEqual(posted, [(freesound.TOKEN_URL, {"client_id": "cid", "client_secret": "sec",
                                                         "grant_type": "authorization_code", "code": "code1",
                                                         "code_verifier": "ver1"})])
        self.assertEqual(record["expires_at"], 87400.0)
        self.assertEqual(record["refresh_token"], "r1")

    def test_malformed_token_response_is_rejected(self):
        with self.assertRaisesRegex(freesound.FreesoundError, "unexpected token response"):
            freesound.exchange_code(GRANT, CREDS, post=lambda u, f: {"error": "invalid_grant"})

    def test_authorize_then_exchange_carries_the_verifier_through_the_pending_file(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(freesound, "credentials", lambda: CREDS):
            token = Path(td) / "t.json"
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(freesound.main(["authorize", "--token-file", str(token)]), 0)
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(out.getvalue().strip()).query)
            pending = json.loads(freesound.pending_file_for(token).read_text(encoding="utf-8"))
            self.assertEqual(query["state"], [pending["state"]])
            self.assertEqual(query["code_challenge"], [freesound.pkce_challenge(pending["code_verifier"])])
            self.assertGreaterEqual(len(pending["code_verifier"]), 43)

            sent = []

            def fake_exchange(grant, creds, **kwargs):
                sent.append(grant)
                return {"access_token": "a1", "refresh_token": "r1", "expires_at": 4_000_000_000.0}

            with mock.patch.object(freesound, "exchange_code", fake_exchange), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(freesound.main(["exchange", "--code", "abc", "--token-file", str(token)]), 0)
            self.assertEqual(sent, [freesound.AuthorizationGrant("abc", pending["code_verifier"])])
            self.assertEqual(freesound.read_token(token)["access_token"], "a1")
            self.assertFalse(freesound.pending_file_for(token).exists())

    def test_exchange_without_a_pending_authorization_asks_to_authorize_first(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(freesound, "credentials", lambda: CREDS):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = freesound.main(["exchange", "--code", "abc", "--token-file", str(Path(td) / "t.json")])
            self.assertEqual(code, 1)
            self.assertIn("no pending authorization", err.getvalue())

    def test_malformed_pending_authorization_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "p.json"
            path.write_text(json.dumps({"state": "s"}), encoding="utf-8")
            with self.assertRaisesRegex(freesound.FreesoundError, "malformed"):
                freesound.read_pending_verifier(path)

    def test_valid_token_is_returned_without_refresh(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.json"
            freesound.save_token(path, {"access_token": "a1", "refresh_token": "r1", "expires_at": 100_000.0})

            def no_post(url, fields):
                raise AssertionError("must not refresh")

            self.assertEqual(freesound.load_access_token(path, post=no_post, now=lambda: 1000.0,
                                                         creds=lambda: CREDS), "a1")

    def test_token_near_expiry_is_refreshed_and_saved(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.json"
            freesound.save_token(path, {"access_token": "old", "refresh_token": "r1", "expires_at": 2000.0})
            posted = []

            def post(url, fields):
                posted.append(fields)
                return {"access_token": "new", "refresh_token": "r2", "expires_in": 86400}

            token = freesound.load_access_token(path, post=post, now=lambda: 1000.0, creds=lambda: CREDS)
            self.assertEqual(token, "new")
            self.assertEqual(posted[0]["grant_type"], "refresh_token")
            self.assertEqual(posted[0]["refresh_token"], "r1")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["refresh_token"], "r2")

    def test_expired_token_without_refresh_token_asks_for_authorization(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.json"
            freesound.save_token(path, {"access_token": "old", "refresh_token": None, "expires_at": 0.0})
            with self.assertRaisesRegex(freesound.FreesoundError, "authorize"):
                freesound.load_access_token(path, now=lambda: 1000.0, creds=lambda: CREDS)

    def test_missing_and_malformed_token_files(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(freesound.FreesoundError, "no Freesound token"):
                freesound.read_token(Path(td) / "missing.json")
            bad = Path(td) / "bad.json"
            bad.write_text(json.dumps({"access_token": "a", "expires_at": True}), encoding="utf-8")
            with self.assertRaisesRegex(freesound.FreesoundError, "malformed"):
                freesound.read_token(bad)


class FetcherTests(unittest.TestCase):
    def test_download_url_detection(self):
        self.assertTrue(freesound.is_download_url("https://freesound.org/apiv2/sounds/12/download/"))
        for url in ("http://freesound.org/apiv2/sounds/12/download/", "https://freesound.org/apiv2/sounds/12/",
                    "https://evil.example/apiv2/sounds/12/download/", "https://cdn.freesound.org/previews/1/1.mp3",
                    "https://freesound.org/apiv2/sounds/12/download/../../x"):
            with self.subTest(url):
                self.assertFalse(freesound.is_download_url(url))

    def test_plain_items_use_the_plain_fetcher_and_never_read_the_token(self):
        plain = [{"url": "https://archive.org/download/x/y.mp3"}, "not-a-record"]
        self.assertIs(freesound.fetcher_for_records(plain, Path("missing-token.json")), ingest.http_fetch)

    def test_a_freesound_download_needs_the_stored_token(self):
        originals = [{"url": "https://freesound.org/apiv2/sounds/12/download/"}]
        with tempfile.TemporaryDirectory() as td:
            fetch = freesound.fetcher_for_records(originals, Path(td) / "missing.json")
            with self.assertRaisesRegex(freesound.FreesoundError, "no Freesound token"):
                fetch("https://freesound.org/apiv2/sounds/12/download/", Path(td) / "a", None)

    def test_the_token_is_not_read_until_a_freesound_download_starts(self):
        loads = []
        fetched = []
        token = freesound.LazyToken(lambda: loads.append(1) or "tok")
        fetch = freesound.authorized_fetcher(token, fetch=lambda url, dest, rb=None, *, bearer_token=None:
                                             fetched.append(bearer_token))
        fetch("https://archive.org/download/x/y.mp3", Path("a"), None)
        self.assertEqual(loads, [])
        fetch(DOWNLOAD, Path("b"), None)
        fetch(DOWNLOAD, Path("c"), None)
        self.assertEqual((loads, fetched), ([1], [None, "tok", "tok"]))

    def test_token_is_sent_to_freesound_downloads_only(self):
        calls = []

        def fake_fetch(url, dest, range_bytes=None, *, bearer_token=None):
            calls.append((url, bearer_token))

        fetch = freesound.authorized_fetcher("tok", fetch=fake_fetch)
        fetch("https://freesound.org/apiv2/sounds/12/download/", Path("a"), None)
        fetch("https://archive.org/download/x/y.mp3", Path("b"), 1024)
        self.assertEqual(calls, [("https://freesound.org/apiv2/sounds/12/download/", "tok"),
                                 ("https://archive.org/download/x/y.mp3", None)])


class FakeClock:
    def __init__(self):
        self.now, self.sleeps = 100.0, []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


DOWNLOAD = "https://freesound.org/apiv2/sounds/12/download/"


class BudgetTests(unittest.TestCase):
    def budget(self, interval=2.0, cap=10):
        clock = FakeClock()
        return freesound.DownloadBudget(interval, cap, clock=clock.time, sleep=clock.sleep), clock

    def test_downloads_start_at_least_one_interval_apart(self):
        budget, clock = self.budget(interval=2.0)
        for _ in range(4):
            budget.acquire()
        self.assertEqual(clock.sleeps, [2.0, 2.0, 2.0])

    def test_time_spent_downloading_counts_toward_the_interval(self):
        budget, clock = self.budget(interval=2.0)
        budget.acquire()
        clock.now += 1.5
        budget.acquire()
        self.assertEqual(clock.sleeps, [0.5])

    def test_a_download_that_takes_longer_than_the_interval_is_not_delayed(self):
        budget, clock = self.budget(interval=2.0)
        budget.acquire()
        clock.now += 30.0
        budget.acquire()
        self.assertEqual(clock.sleeps, [])

    def test_rate_stays_under_thirty_a_minute_at_the_default_interval(self):
        budget, clock = self.budget(interval=2.2, cap=100)
        start = clock.now
        for _ in range(31):
            budget.acquire()
        self.assertLess(30 / ((clock.now - start) / 60), 30)

    def test_a_stop_during_the_wait_is_honoured_and_gives_the_slot_back(self):
        budget, clock = self.budget(interval=2.0)
        budget.acquire()

        def sleep_then_stop(seconds):
            clock.sleep(seconds)
            budget.stop("quota hit")

        budget._sleep = sleep_then_stop
        with self.assertRaisesRegex(freesound.FreesoundError, "quota hit"):
            budget.acquire()
        self.assertEqual(budget.used, 1)

    def test_other_workers_can_stop_the_budget_while_one_is_waiting(self):
        budget, _ = self.budget(interval=2.0)
        budget.acquire()
        stopped = []

        def sleep_while_another_thread_stops(seconds):
            worker = threading.Thread(target=lambda: (budget.stop("quota hit"), stopped.append(True)))
            worker.start()
            worker.join(timeout=5)

        budget._sleep = sleep_while_another_thread_stops
        with self.assertRaises(freesound.FreesoundError):
            budget.acquire()
        self.assertEqual(stopped, [True])

    def test_budget_stops_after_the_cap(self):
        budget, _ = self.budget(cap=2)
        budget.acquire()
        budget.acquire()
        with self.assertRaisesRegex(freesound.FreesoundError, "budget of 2 requests"):
            budget.acquire()
        self.assertEqual(budget.used, 2)

    def test_authorized_fetcher_uses_the_budget_for_freesound_downloads_only(self):
        budget, clock = self.budget(cap=1)
        calls = []

        def fake_fetch(url, dest, range_bytes=None, *, bearer_token=None):
            calls.append(url)

        fetch = freesound.authorized_fetcher("tok", fetch=fake_fetch, budget=budget)
        fetch("https://archive.org/download/x/y.mp3", Path("a"), 1024)
        fetch("https://archive.org/download/x/z.mp3", Path("b"), 1024)
        fetch(DOWNLOAD, Path("c"), None)
        self.assertEqual(budget.used, 1)
        with self.assertRaisesRegex(freesound.FreesoundError, "budget"):
            fetch("https://freesound.org/apiv2/sounds/13/download/", Path("d"), None)
        self.assertEqual(len(calls), 3)

    def test_a_429_stops_every_later_download_without_a_request(self):
        budget, _ = self.budget()
        calls = []

        def throttled_fetch(url, dest, range_bytes=None, *, bearer_token=None):
            calls.append(url)
            cause = urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
            raise ingest.IngestError("download failed: HTTP Error 429") from cause

        fetch = freesound.authorized_fetcher("tok", fetch=throttled_fetch, budget=budget)
        with self.assertRaises(ingest.IngestError):
            fetch(DOWNLOAD, Path("a"), None)
        with self.assertRaisesRegex(freesound.FreesoundError, "429"):
            fetch("https://freesound.org/apiv2/sounds/13/download/", Path("b"), None)
        self.assertEqual(len(calls), 1)

    def test_other_failures_do_not_stop_the_run(self):
        budget, _ = self.budget()
        calls = []

        def failing_fetch(url, dest, range_bytes=None, *, bearer_token=None):
            calls.append(url)
            raise ingest.IngestError("download failed: HTTP Error 404")

        fetch = freesound.authorized_fetcher("tok", fetch=failing_fetch, budget=budget)
        for sound in (12, 13):
            with self.assertRaises(ingest.IngestError):
                fetch(f"https://freesound.org/apiv2/sounds/{sound}/download/", Path("a"), None)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
