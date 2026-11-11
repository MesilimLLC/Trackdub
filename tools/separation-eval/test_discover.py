import json
import sys
import tempfile
import types
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import patch

import discover
import ingest
import recipe_coverage as coverage
from test_support import running_server

TODAY = "2026-09-30"


def result(n=1, *, license="cc0", version="1.0", creator="alice", source="freesound", duration=10000,
           filetype="mp3", url=None, filesize=500000, title="Some Sound"):
    return {"id": f"{n:012d}-0000-0000-0000-000000000000", "title": title, "creator": creator, "license": license,
            "license_version": version, "source": source, "provider": source, "duration": duration,
            "filetype": filetype, "filesize": filesize, "url": url or f"https://cdn.example.org/{n}.mp3",
            "foreign_landing_url": f"https://example.org/sounds/{n}"}


SPEC = {"name": "crowd", "role": "ambience", "tags": ["Crowd"], "q": "crowd", "source": "freesound", "pages": 2,
        "limit": 10, "min_duration_s": 5, "max_duration_s": 100, "per_creator_cap": 2}


def openverse_fetch(pages, calls=None):
    def fetch(url):
        if calls is not None:
            calls.append(url)
        page = int(urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["page"][0])
        return {"results": pages.get(page, []), "page_count": max(pages)}
    return fetch


class OpenverseTests(unittest.TestCase):
    def candidates(self, spec, results_by_page):
        return discover.discover_openverse([spec], openverse_fetch(results_by_page), TODAY)

    def test_license_mapping(self):
        rows = [result(1, license="cc0"), result(2, license="by", version="4.0", creator="bob"),
                result(3, license="by", version="3.0", creator="cy"), result(4, license="pdm", version="1.0", creator="di"),
                result(5, license="by", version="2.0", creator="ed"), result(6, license="by-sa", version="4.0", creator="fy"),
                result(7, license="by-nc", version="4.0", creator="gus")]
        got = {i["id"][-12:]: i["license_spdx"] for i in self.candidates(SPEC, {1: rows})}
        self.assertEqual(got, {"000000000001": "CC0-1.0", "000000000002": "CC-BY-4.0",
                               "000000000003": "CC-BY-3.0", "000000000004": "LicenseRef-PublicDomain"})

    def test_jamendo_mp32_is_mp3_and_ext_is_recorded(self):
        item = self.candidates(SPEC, {1: [result(1, filetype="mp32", url="https://prod-1.storage.jamendo.com/?trackid=1&format=mp32")]})[0]
        self.assertEqual(item["ext"], "mp3")
        with tempfile.TemporaryDirectory() as td:
            self.assertTrue(str(ingest.cache_path(Path(td), item)).endswith(f"{item['id']}.mp3"))

    def test_filters(self):
        rows = [result(1, duration=1000), result(2, duration=500000, creator="b"), result(3, filetype="m4a", creator="c"),
                result(4, creator="  ", source="freesound"), result(5, url="http://insecure/5.mp3", creator="e"),
                result(6, creator="f")]
        ids = [i["id"][-12:] for i in self.candidates(SPEC, {1: rows})]
        self.assertEqual(ids, ["000000000006"])

    def test_creator_cap_dedupe_and_limit(self):
        rows = [result(n, creator="alice") for n in range(1, 6)] + [result(10, creator="bob", url="https://cdn.example.org/1.mp3")]
        items = self.candidates({**SPEC, "limit": 3}, {1: rows})
        self.assertEqual(sum(1 for i in items if i["creator"] == "alice"), 2)
        self.assertNotIn("000000000010", [i["id"][-12:] for i in items])
        self.assertLessEqual(len(items), 3)

    def test_stops_requesting_pages_when_limit_is_reached(self):
        calls = []
        rows = [result(1, creator="alice"), result(2, creator="bob")]
        discover.discover_openverse([{**SPEC, "pages": 5, "limit": 1}],
                                    openverse_fetch({1: rows, 2: [result(3, creator="cy")]}, calls), TODAY)
        self.assertEqual(len(calls), 1)

    def test_item_fields_group_tags_and_attribution(self):
        item = self.candidates(SPEC, {1: [result(1, title="Big Crowd", creator="Inspector J", license="by", version="4.0")]})[0]
        self.assertEqual(item["group"], "freesound:inspector-j")
        self.assertEqual(item["tags"], ["crowd"])
        self.assertEqual(item["tag_basis"], "search-query")
        self.assertEqual(item["role"], "ambience")
        self.assertIn("Big Crowd", item["attribution_text"])
        self.assertIn("Inspector J", item["attribution_text"])
        self.assertIn("CC-BY-4.0", item["attribution_text"])
        self.assertEqual(item["license_evidence_url"], "https://example.org/sounds/1")
        self.assertEqual(item["retrieved_at"], TODAY)

    def test_non_latin_creators_keep_distinct_group_keys(self):
        left = self.candidates(SPEC, {1: [result(1, creator="東京の作者")]})[0]
        right = self.candidates(SPEC, {1: [result(2, creator="京都の作者")]})[0]
        self.assertNotEqual(left["group"], right["group"])
        self.assertIn("東京", left["group"])

    def test_range_bytes_only_when_the_file_is_larger(self):
        spec = {**SPEC, "range_bytes": 3000000}
        big = self.candidates(spec, {1: [result(1, filesize=9000000)]})[0]
        small = self.candidates(spec, {1: [result(2, filesize=100000)]})[0]
        unknown = self.candidates(spec, {1: [result(3, filesize=None)]})[0]
        self.assertEqual(big["range_bytes"], 3000000)
        self.assertNotIn("range_bytes", small)
        self.assertEqual(unknown["range_bytes"], 3000000)

    def test_paginates_and_stops_at_page_count(self):
        calls = []
        discover.discover_openverse([{**SPEC, "pages": 5}], openverse_fetch({1: [result(1)], 2: [result(2, creator="b")]}, calls), TODAY)
        self.assertEqual(len(calls), 2)

    def test_request_parameters(self):
        calls = []
        discover.discover_openverse([SPEC], openverse_fetch({1: []}, calls), TODAY)
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(calls[0]).query)
        self.assertEqual(q["license"], ["cc0,pdm,by"])
        self.assertEqual(q["source"], ["freesound"])
        self.assertEqual(q["page_size"], ["20"])

    def test_every_candidate_passes_ingest_validation(self):
        items = self.candidates(SPEC, {1: [result(n, creator=f"c{n}") for n in range(1, 8)]})
        for item in items:
            self.assertIsNone(ingest.validate_record(item, allow_sharealike=False), item["id"])


def archive_fetch(docs, files_by_id, calls=None):
    def fetch(url):
        if calls is not None:
            calls.append(url)
        if url.startswith(discover.ARCHIVE_SEARCH_URL):
            return {"response": {"docs": docs}}
        identifier = url.removeprefix(discover.ARCHIVE_METADATA_URL)
        return {"files": files_by_id.get(identifier, [])}
    return fetch


def book(identifier, title="A Novel", runtime="5:00:00", licenseurl="http://creativecommons.org/publicdomain/mark/1.0/"):
    return {"identifier": identifier, "title": title, "creator": "Some Author", "runtime": runtime, "licenseurl": licenseurl}


def mp3s(n=6, size=9_000_000):
    return [{"name": f"chapter_{i:02d}_64kb.mp3", "format": "64Kbps MP3", "size": str(size)} for i in range(n)]


class LibriVoxTests(unittest.TestCase):
    CONFIG = {"books": 3, "seed": 1, "range_bytes": 3000000, "min_runtime_minutes": 30, "section_index": 3}

    def run_librivox(self, docs, files, config=None):
        return discover.discover_librivox(config or self.CONFIG, archive_fetch(docs, files), TODAY)

    def test_selects_public_domain_solo_long_books(self):
        docs = [book("good_one"), book("good_two"), book("nc", licenseurl="http://creativecommons.org/licenses/by-nc/3.0/"),
                book("collection", title="Short Story Collection 001"), book("short", runtime="0:10:00"),
                book("dramatic", title="Dramatic Reading Hour"), book("no_mp3"),
                book("wind_in_the_willows_collab_librivox", title="The Wind in the Willows")]
        files = {d["identifier"]: mp3s() for d in docs if d["identifier"] != "no_mp3"}
        files["no_mp3"] = [{"name": "x.ogg", "format": "Ogg Vorbis", "size": "9000000"}]
        items = self.run_librivox(docs, files)
        self.assertEqual(sorted(i["group"] for i in items), ["librivox:good_one", "librivox:good_two"])

    def test_collab_identifier_is_filtered_even_when_title_is_plain(self):
        docs = [book("wind_in_the_willows_collab_librivox", title="The Wind in the Willows"),
                book("wind_in_the_willows_solo", title="The Wind in the Willows")]
        files = {d["identifier"]: mp3s() for d in docs}
        items = self.run_librivox(docs, files)
        self.assertEqual([i["group"] for i in items], ["librivox:wind_in_the_willows_solo"])

    def test_item_fields(self):
        item = self.run_librivox([book("good_one", title="Treasure Island")], {"good_one": mp3s()})[0]
        self.assertEqual(item["url"], "https://archive.org/download/good_one/chapter_03_64kb.mp3")
        self.assertEqual(item["range_bytes"], 3000000)
        self.assertEqual(item["license_spdx"], "LicenseRef-PublicDomain")
        self.assertEqual(item["role"], "dialogue")
        self.assertIn("Treasure Island", item["attribution_text"])
        self.assertEqual(item["license_evidence_url"], "https://archive.org/details/good_one")
        self.assertEqual(urllib.parse.urlsplit(item["license_evidence_url"]).scheme, "https")
        self.assertIsNone(ingest.validate_record(item, allow_sharealike=False))

    def test_section_index_is_clamped_and_small_files_skipped(self):
        files = {"short_book": mp3s(2), "tiny": mp3s(6, size=100)}
        items = self.run_librivox([book("short_book"), book("tiny")], files)
        self.assertEqual([i["url"].rsplit("/", 1)[1] for i in items], ["chapter_01_64kb.mp3"])

    def test_selection_is_deterministic_and_seeded(self):
        docs = [book(f"book_{n}") for n in range(20)]
        files = {d["identifier"]: mp3s() for d in docs}
        a = [i["group"] for i in self.run_librivox(docs, files)]
        b = [i["group"] for i in self.run_librivox(docs, files)]
        c = [i["group"] for i in self.run_librivox(docs, files, {**self.CONFIG, "seed": 2})]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 3)


class CacheAndDiscoverTests(unittest.TestCase):
    def test_https_redirect_handler_rejects_downgrade(self):
        handler = discover._HttpsRedirectHandler()
        with self.assertRaisesRegex(discover.DiscoverError, "non-HTTPS"):
            handler.redirect_request(None, None, 302, "Found", {}, "http://example.org/api")

    def test_discover_merges_sorts_and_rejects_duplicate_ids(self):
        spec = {"openverse": [SPEC], "librivox": LibriVoxTests.CONFIG}
        ov = openverse_fetch({1: [result(1), result(2, creator="b")]})
        av = archive_fetch([book("good_one")], {"good_one": mp3s()})
        items = discover.discover(spec, ov, av, TODAY)
        self.assertEqual([i["id"] for i in items], sorted(i["id"] for i in items))
        self.assertEqual({i["role"] for i in items}, {"ambience", "dialogue"})
        only = discover.discover(spec, ov, av, TODAY, only="librivox")
        self.assertEqual({i["role"] for i in only}, {"dialogue"})

    def test_write_items_roundtrips_through_ingest_reader(self):
        items = [ingest_ready() for _ in range(1)]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "items.jsonl"
            discover.write_items(path, items, "spec", TODAY)
            self.assertEqual(ingest.read_jsonl(path), items)

    def test_cached_fetch_reuses_responses_and_survives_without_network(self):
        import http.server
        hits = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                hits.append(self.path)
                body = json.dumps({"ok": len(hits)}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        with running_server(Handler) as server:
            with tempfile.TemporaryDirectory() as td:
                fetch = discover.cached_fetch_json(Path(td), pause_s=0)
                url = f"http://127.0.0.1:{server.server_port}/x"
                self.assertEqual(fetch(url), {"ok": 1})
                self.assertEqual(fetch(url), {"ok": 1})
                self.assertEqual(len(hits), 1)

    def test_rate_limit_raises_quota_error_with_guidance(self):
        import http.server

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(429)
                self.send_header("Retry-After", "60")
                self.end_headers()

            def log_message(self, *a):
                pass

        with running_server(Handler) as server:
            fetch = discover.cached_fetch_json(None, pause_s=0)
            with self.assertRaises(discover.QuotaError) as ctx:
                fetch(f"http://127.0.0.1:{server.server_port}/x")
            self.assertIn("Retry-After 60", str(ctx.exception))

    def test_shipped_spec_is_well_formed(self):
        spec = json.loads((Path(__file__).parent / "corpus-sources.v1.json").read_text())
        for q in spec["openverse"]:
            self.assertIn(q["role"], ingest.ROLES)
            self.assertLessEqual(q["pages"], 3)
        self.assertLessEqual(sum(q["pages"] for q in spec["openverse"]), 60, "stay well inside the 200/day quota")


def ingest_ready():
    return {"id": "x-dlg-000000000001", "source": "s", "url": "https://x/a.mp3", "role": "dialogue", "group": "g:1",
            "license_spdx": "CC0-1.0", "license_evidence_url": "https://example.org/license",
            "attribution_text": "a", "tags": [], "tag_basis": "none", "retrieved_at": TODAY}


class CoverageTests(unittest.TestCase):
    def items_for(self, group_prefix, role, tags, per_split=4):
        out, n = [], 0
        found = {"dev": 0, "test": 0}
        while min(found.values()) < per_split:
            n += 1
            group = f"{group_prefix}:{n}"
            split = ingest.assign_split(group)
            if found[split] < per_split:
                found[split] += 1
                item_id = f"{group_prefix}-{n:04d}"
                out.append({"id": item_id, "role": role, "group": group, "tags": tags,
                            "url": f"https://example.org/{item_id}.wav", "ext": "wav"})
        return out

    def full_set(self):
        items = []
        for prefix, role, tags in (("dlg", "dialogue", []), ("wh", "dialogue", ["whisper"]), ("mi", "music", ["instrumental"]),
                                   ("mv", "music", ["vocals"]), ("sx", "sfx", ["loud"]), ("cr", "ambience", ["crowd"]),
                                   ("rir", "rir", [])):
            items += self.items_for(prefix, role, tags)
        return items

    def test_a_complete_set_has_no_gaps(self):
        _, gaps = coverage.check(self.full_set(), min_groups=3)
        self.assertEqual([gap for gap in gaps if not gap.startswith("A3 rir")], [])
        self.assertEqual(sum("RT60 unverified" in gap for gap in gaps), 2)

    def test_rir_coverage_requires_cached_audio_with_in_range_rt60(self):
        items = self.full_set()
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            for item in items:
                if item["role"] == "rir":
                    path = ingest.cache_path(cache, item)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"cached")
            fake_audio = types.ModuleType("audiomath")
            fake_audio.SR = 48000
            fake_audio.AudioError = type("AudioError", (Exception,), {})
            fake_audio.load_audio = lambda *_args, **_kwargs: "cached audio"
            fake_audio.estimate_rt60 = lambda *_args, **_kwargs: 1.0
            with patch.dict(sys.modules, {"audiomath": fake_audio}):
                _, valid_gaps = coverage.check(items, min_groups=3, cache=cache)
                self.assertEqual(valid_gaps, [])
                fake_audio.estimate_rt60 = lambda *_args, **_kwargs: 0.2
                _, invalid_gaps = coverage.check(items, min_groups=3, cache=cache)
        self.assertTrue(any(gap.startswith("A3 rir") for gap in invalid_gaps))

    def test_missing_tags_are_reported_per_recipe_and_split(self):
        items = [i for i in self.full_set() if "whisper" not in i["tags"]]
        _, gaps = coverage.check(items, min_groups=3)
        self.assertTrue(any(g.startswith("A8 dialogue+whisper [dev]") for g in gaps))
        self.assertTrue(any(g.startswith("A8 dialogue+whisper [test]") for g in gaps))
        self.assertFalse(any(g.startswith("A1 ") for g in gaps))

    def test_overlap_needs_two_groups_even_with_a_low_minimum(self):
        items = [i for i in self.full_set() if i["role"] != "dialogue"] + [
            {"id": "d1", "role": "dialogue", "group": g, "tags": []} for g in ("only:1",)]
        _, gaps = coverage.check(items, min_groups=1)
        self.assertTrue(any(g.startswith("A6 dialogue") for g in gaps))

    def language_items(self, language, per_split=8):
        items = self.items_for(f"lang-{language}", "dialogue", [], per_split=per_split)
        return [{**i, "language": language} for i in items]

    def test_a13_is_not_checked_when_every_dialogue_item_is_english(self):
        rows, gaps = coverage.check(self.full_set(), min_groups=3)
        self.assertNotIn("A13", {r["recipe"] for r in rows})
        self.assertFalse(any(g.startswith("A13") for g in gaps))

    def test_non_english_dialogue_activates_a13_and_asks_for_five_languages(self):
        items = self.full_set() + [i for lang in ("ar", "pt", "ru") for i in self.language_items(lang)]
        rows, gaps = coverage.check(items, min_groups=3)
        self.assertIn("A13", {r["recipe"] for r in rows})
        self.assertTrue(any(g.startswith("A13 [dev]: 3 non-English language(s) with 8+ speakers, need 5") for g in gaps))
        self.assertTrue(any(g.startswith("A13 [test]: 3 non-English") for g in gaps))
        self.assertFalse(any(g.startswith("A1 ") for g in gaps))

    def test_five_languages_with_enough_speakers_close_the_a13_gap(self):
        items = self.full_set() + [i for lang in ("ar", "pt", "ru", "ja", "tr") for i in self.language_items(lang)]
        _, gaps = coverage.check(items, min_groups=3)
        self.assertFalse(any(g.startswith("A13") for g in gaps))

    def test_a_language_with_too_few_speakers_does_not_count(self):
        items = self.full_set() + [i for lang in ("ar", "pt", "ru", "ja") for i in self.language_items(lang)]
        items += self.language_items("tr", per_split=3)
        _, gaps = coverage.check(items, min_groups=3)
        self.assertTrue(any(g.startswith("A13 [dev]: 4 non-English") for g in gaps))

    def test_english_recipes_ignore_non_english_dialogue(self):
        english_gone = [i for i in self.full_set() if i["role"] != "dialogue"]
        items = english_gone + [i for lang in ("ar", "pt", "ru") for i in self.language_items(lang)]
        _, gaps = coverage.check(items, min_groups=3)
        self.assertTrue(any(g.startswith("A1 dialogue [dev]") for g in gaps))

    def test_main_exit_codes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "items.jsonl"
            path.write_text("\n".join(json.dumps(i) for i in self.full_set()))
            self.assertEqual(coverage.main(["--items", str(path)]), 1)
            path.write_text(json.dumps(self.full_set()[0]))
            self.assertEqual(coverage.main(["--items", str(path)]), 1)


if __name__ == "__main__":
    unittest.main()
