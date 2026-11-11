import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path

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
    def test_authorize_url(self):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(freesound.authorize_url("cid", "st")).query)
        self.assertEqual(query, {"client_id": ["cid"], "response_type": ["code"], "state": ["st"]})

    def test_exchange_posts_the_authorization_code_grant(self):
        posted = []

        def post(url, fields):
            posted.append((url, fields))
            return {"access_token": "a1", "refresh_token": "r1", "expires_in": 86400, "scope": "read write"}

        record = freesound.exchange_code("code1", CREDS, post=post, now=lambda: 1000.0)
        self.assertEqual(posted, [(freesound.TOKEN_URL, {"client_id": "cid", "client_secret": "sec",
                                                         "grant_type": "authorization_code", "code": "code1"})])
        self.assertEqual(record["expires_at"], 87400.0)
        self.assertEqual(record["refresh_token"], "r1")

    def test_malformed_token_response_is_rejected(self):
        with self.assertRaisesRegex(freesound.FreesoundError, "unexpected token response"):
            freesound.exchange_code("c", CREDS, post=lambda u, f: {"error": "invalid_grant"})

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

    def test_freesound_items_need_the_stored_token(self):
        originals = [{"url": "https://freesound.org/apiv2/sounds/12/download/"}]
        with tempfile.TemporaryDirectory() as td, self.assertRaisesRegex(freesound.FreesoundError, "no Freesound token"):
            freesound.fetcher_for_records(originals, Path(td) / "missing.json")

    def test_token_is_sent_to_freesound_downloads_only(self):
        calls = []

        def fake_fetch(url, dest, range_bytes=None, *, bearer_token=None):
            calls.append((url, bearer_token))

        fetch = freesound.authorized_fetcher("tok", fetch=fake_fetch)
        fetch("https://freesound.org/apiv2/sounds/12/download/", Path("a"), None)
        fetch("https://archive.org/download/x/y.mp3", Path("b"), 1024)
        self.assertEqual(calls, [("https://freesound.org/apiv2/sounds/12/download/", "tok"),
                                 ("https://archive.org/download/x/y.mp3", None)])


if __name__ == "__main__":
    unittest.main()
