"""Offline checks for the gpu-docs search. Run from tools/mcp-trackdub-gpu-docs:
    uv run python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from trackdub_gpu_docs_mcp import search
from trackdub_gpu_docs_mcp.corpus import load_sources

EVAL_DIR = Path(__file__).resolve().parents[2] / "docs-eval"
sys.path.insert(0, str(EVAL_DIR))

import eval_gpu  # noqa: E402


class TermTests(unittest.TestCase):
    def test_identifier_parts_reach_prose_queries(self):
        weights, symbols = search._query_weights("RegisterExecutionProviderLibrary")
        self.assertIn("registerexecutionproviderlibrary", weights)
        self.assertTrue({"register", "execution", "provider", "library"} <= set(weights))
        self.assertEqual(symbols, ["registerexecutionproviderlibrary"])

    def test_snake_case_identifier_splits(self):
        self.assertEqual(search._parts("ep_refit_table"), ["ep", "refit", "table"])

    def test_plain_word_has_no_parts(self):
        self.assertEqual(search._parts("allocator"), [])

    def test_stopwords_are_dropped_from_queries(self):
        weights, _ = search._query_weights("how do I use the allocator")
        self.assertEqual(set(weights), {"use", "allocator"})


class SourceMergeTests(unittest.TestCase):
    def test_remote_sources_come_from_the_docs_rag_corpus(self):
        corpus = json.loads((EVAL_DIR.parent / "docs-rag" / "corpus.v1.json").read_text(encoding="utf-8"))
        gpu_ids = {
            entry["gpu"]["id"]
            for vendor in corpus["vendors"]
            for entry in vendor["sources"]
            if isinstance(entry, dict) and "gpu" in entry
        }
        loaded = {s.id for s in load_sources() if s.kind == "url"}
        self.assertEqual(loaded, gpu_ids)
        self.assertTrue(loaded)


class SearchTests(unittest.TestCase):
    def test_results_cap_chunks_per_source(self):
        hits = search.search_chunks("TensorRT-RTX", limit=20)
        per_source: dict[str, int] = {}
        for hit in hits:
            per_source[hit["sourceId"]] = per_source.get(hit["sourceId"], 0) + 1
        self.assertTrue(all(count <= search._MAX_PER_SOURCE for count in per_source.values()))

    def test_excerpt_is_centred_on_the_match_not_the_chunk_start(self):
        hits = search.search_chunks("nv_use_sync_gpu_allocator", limit=1)
        self.assertTrue(hits)
        self.assertIn("nv_use_sync_gpu_allocator", hits[0]["excerpt"].lower())
        self.assertLessEqual(len(hits[0]["excerpt"]), search._SNIPPET_CHARS + 4)

    def test_tag_filter_restricts_results(self):
        hits = search.search_chunks("execution provider", limit=10, tag="microsoft-winml")
        self.assertTrue(hits)
        self.assertTrue(all("microsoft-winml" in hit["tags"] for hit in hits))

    def test_empty_query_returns_nothing(self):
        self.assertEqual(search.search_chunks("the of and"), [])


class RetrievalQualityTests(unittest.TestCase):
    """Floors sit below the measured baseline so noise does not fail CI, but a real ranking regression does."""

    def test_eval_set_meets_the_floor(self):
        questions = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))["questions"]
        result = eval_gpu.score(questions)
        self.assertGreaterEqual(result["total"], 20)
        self.assertGreaterEqual(result["hit3"], 0.95, result["rows"])
        self.assertGreaterEqual(result["mrr"], 0.90)


if __name__ == "__main__":
    unittest.main()
