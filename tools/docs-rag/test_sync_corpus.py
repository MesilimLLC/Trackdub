import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

import sync_corpus


class UploadTests(unittest.TestCase):
    def test_upload_uses_metadata_helper(self):
        with patch.object(sync_corpus.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(sync_corpus.upload(Path("staging"), "corpus", Path("api"), workers=3), 0)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[0], "node")
        self.assertEqual(Path(command[1]).name, "upload_corpus.mjs")
        self.assertEqual(command[2:], ["api", "staging", "corpus", "3"])

    def test_upload_propagates_failure_returncode(self):
        with patch.object(sync_corpus.subprocess, "run", return_value=subprocess.CompletedProcess([], 3)) as run:
            self.assertEqual(sync_corpus.upload(Path("staging"), "corpus", Path("api"), workers=3), 3)
        run.assert_called_once()

    def test_prune_asks_the_uploader_to_delete_absent_objects(self):
        with patch.object(sync_corpus.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(sync_corpus.upload(Path("staging"), "corpus", Path("api"), workers=3, prune=True), 0)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][2:], ["api", "staging", "corpus", "3", "--prune"])

    def test_prune_refused_for_incomplete_refresh(self):
        self.assert_prune_refused(["--skip-fetch", "--upload", "--prune"], "--skip-fetch")

    def test_prune_refused_for_reused_staging(self):
        self.assert_prune_refused(["--reuse-staging", "--upload", "--prune"], "--reuse-staging")

    def test_prune_refused_without_upload(self):
        self.assert_prune_refused(["--prune"], "missing --upload")

    def test_full_refresh_passes_prune_to_upload(self):
        with tempfile.TemporaryDirectory() as root:
            staging = Path(root) / ".staging"
            staging.mkdir()
            with patch.object(sync_corpus, "STAGING", staging), \
                 patch.object(sync_corpus, "load_manifest", return_value={"bucket": "corpus", "aiSearchInstance": "instance"}), \
                 patch.object(sync_corpus, "refresh_staging") as refresh, \
                 patch.object(sync_corpus, "upload", return_value=0) as upload, \
                 patch.object(sync_corpus, "reindex", return_value=0), \
                 patch("sys.argv", ["sync_corpus.py", "--upload", "--prune"]):
                with self.assertRaises(SystemExit) as stopped:
                    sync_corpus.main()
            self.assertEqual(stopped.exception.code, 0)
        refresh.assert_called_once()
        self.assertTrue(upload.call_args.kwargs["prune"])

    def assert_prune_refused(self, flags, reason):
        with patch.object(sync_corpus, "load_manifest") as manifest, \
             patch.object(sync_corpus, "refresh_staging") as refresh, \
             patch.object(sync_corpus, "upload") as upload, \
             patch.object(sync_corpus, "reindex") as reindex, \
             patch("builtins.print") as output, \
             patch("sys.argv", ["sync_corpus.py", *flags]):
            with self.assertRaises(SystemExit) as stopped:
                sync_corpus.main()
        self.assertNotEqual(stopped.exception.code, 0)
        output.assert_called_once_with(
            f"--prune deletes bucket objects missing from staging; refusing because of {reason}",
            flush=True,
        )
        refresh.assert_not_called()
        manifest.assert_not_called()
        upload.assert_not_called()
        reindex.assert_not_called()

    def test_upload_reindexes_only_after_success(self):
        self.run_main(upload_status=0, reindex_status=0, expected=0)

    def test_failed_upload_does_not_reindex(self):
        self.run_main(upload_status=1, reindex_status=0, expected=1)

    def test_failed_reindex_fails_command(self):
        self.run_main(upload_status=0, reindex_status=1, expected=1)

    def test_failed_reindex_prints_job_state_advice(self):
        with patch.object(sync_corpus.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)), \
             patch("builtins.print") as output:
            self.assertEqual(sync_corpus.reindex(Path("api"), "corpus"), 1)
        output.assert_called_once_with(
            "Upload succeeded but reindex request failed; check ai-search jobs list before retrying because the job may have started.",
            flush=True,
        )

    def run_main(self, upload_status, reindex_status, expected):
        with tempfile.TemporaryDirectory() as root:
            staging = Path(root)
            (staging / "doc.md").write_text("doc", encoding="utf-8")
            manifest = {"bucket": "corpus", "aiSearchInstance": "instance-from-manifest"}
            with patch.object(sync_corpus, "STAGING", staging), \
                 patch.object(sync_corpus, "load_manifest", return_value=manifest), \
                 patch.object(sync_corpus, "upload", return_value=upload_status), \
                 patch.object(sync_corpus.subprocess, "run", return_value=subprocess.CompletedProcess([], reindex_status)) as run, \
                 patch("sys.argv", ["sync_corpus.py", "--reuse-staging", "--upload"]):
                with self.assertRaises(SystemExit) as stopped:
                    sync_corpus.main()
                self.assertEqual(stopped.exception.code, expected)
            if upload_status:
                run.assert_not_called()
            else:
                run.assert_called_once()
                command = run.call_args.args[0]
                self.assertIn("ai-search", command)
                self.assertEqual(command[-4:], ["jobs", "create", "instance-from-manifest", "--json"])


class RefreshTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "guide.md").write_text("new guide", encoding="utf-8")
        self.staging = self.root / ".staging"
        self.staging.mkdir()
        (self.staging / "old.md").write_text("last complete snapshot", encoding="utf-8")
        self.manifest = {
            "bucket": "corpus", "aiSearchInstance": "docs", "maxBytes": 3500000,
            "repos": [{"id": "core", "rootEnv": "TEST_DOCS_ROOT",
                       "defaultFromTrackdub": str(self.repo),
                       "prefix": "first-party/core", "include": ["*.md"]}],
            "vendors": [{"id": "vendor", "prefix": "vendor/example",
                         "sources": [["guide", "https://example.com/guide"]]}],
        }

    def run_refresh(self, *args, fetch_result="vendor guide", fetch_error=None):
        with patch.object(sync_corpus, "STAGING", self.staging), \
             patch.object(sync_corpus, "load_manifest", return_value=self.manifest), \
             patch.dict(sync_corpus.os.environ, {"TEST_DOCS_ROOT": str(self.repo)}), \
             patch.object(sync_corpus, "fetch_url", return_value=fetch_result, side_effect=fetch_error) as fetch, \
             patch.object(sync_corpus, "upload", return_value=0) as upload, \
             patch.object(sync_corpus, "reindex", return_value=0) as reindex, \
             patch("sys.argv", ["sync_corpus.py", *args]):
            try:
                sync_corpus.main()
                code = 0
            except SystemExit as stopped:
                code = stopped.code
        return code, fetch, upload, reindex

    def assert_failed_refresh(self, **kwargs):
        code, _, upload, reindex = self.run_refresh("--upload", **kwargs)
        self.assertNotEqual(code, 0)
        upload.assert_not_called()
        reindex.assert_not_called()
        self.assertEqual(sorted(p.name for p in self.staging.iterdir()), ["old.md"])
        self.assertEqual((self.staging / "old.md").read_text(encoding="utf-8"), "last complete snapshot")

    def test_missing_repo_preserves_cache_and_prevents_upload(self):
        self.repo = self.root / "missing"
        self.assert_failed_refresh()

    def test_repo_without_matching_docs_prevents_partial_refresh(self):
        self.manifest["repos"][0]["include"] = ["*.missing"]
        self.assert_failed_refresh()

    def test_failed_vendor_fetch_preserves_cache_and_prevents_upload(self):
        self.assert_failed_refresh(fetch_error=TimeoutError("fetch timed out"))

    def test_empty_vendor_document_preserves_cache_and_prevents_upload(self):
        self.assert_failed_refresh(fetch_result="")

    def test_failed_first_refresh_leaves_no_reusable_partial_cache(self):
        self.staging = self.root / "fresh-cache"
        code, _, upload, reindex = self.run_refresh("--upload", fetch_error=TimeoutError())
        self.assertNotEqual(code, 0)
        self.assertFalse(self.staging.exists())
        upload.assert_not_called()
        reindex.assert_not_called()

    def test_successful_refresh_replaces_snapshot(self):
        code, _, upload, reindex = self.run_refresh("--upload")
        self.assertEqual(code, 0)
        self.assertFalse((self.staging / "old.md").exists())
        self.assertEqual((self.staging / "first-party/core/guide.md").read_text(), "new guide")
        self.assertEqual((self.staging / "vendor/example/guide.md").read_text(),
                         "Source: https://example.com/guide\n\nvendor guide\n")
        self.assertTrue((self.staging / "first-party/trackdub/docs/reference/docs-rag-pin.md").is_file())
        upload.assert_called_once()
        self.assertEqual(upload.call_args.args[0], self.staging)
        reindex.assert_called_once()

    def test_exclude_drops_only_listed_repo_files(self):
        (self.repo / "mirror.md").write_text("mirror aggregate", encoding="utf-8")
        self.manifest["repos"][0]["exclude"] = ["mirror.md"]

        code, _, upload, _ = self.run_refresh("--upload")

        self.assertEqual(code, 0)
        upload.assert_called_once()
        self.assertTrue((self.staging / "first-party/core/guide.md").is_file())
        self.assertFalse((self.staging / "first-party/core/mirror.md").exists())

    def test_exclude_supports_nested_paths_and_globs(self):
        (self.repo / "docs" / "reference").mkdir(parents=True)
        (self.repo / "docs" / "specs").mkdir(parents=True)
        (self.repo / "docs" / "index.md").write_text("keep", encoding="utf-8")
        (self.repo / "docs" / "reference" / "reference.md").write_text("keep", encoding="utf-8")
        (self.repo / "docs" / "specs" / "specs.md").write_text("mirror aggregate", encoding="utf-8")
        (self.repo / "docs" / "strategy").mkdir()
        (self.repo / "docs" / "strategy" / "strategy.md").write_text("mirror aggregate", encoding="utf-8")
        self.manifest["repos"][0]["include"] = ["docs/**/*.md"]
        self.manifest["repos"][0]["exclude"] = ["docs/*/specs.md", "docs/strategy/**"]

        code, _, upload, _ = self.run_refresh("--upload")

        self.assertEqual(code, 0)
        upload.assert_called_once()
        self.assertTrue((self.staging / "first-party/core/docs/index.md").is_file())
        self.assertTrue((self.staging / "first-party/core/docs/reference/reference.md").is_file())
        self.assertFalse((self.staging / "first-party/core/docs/specs/specs.md").exists())
        self.assertFalse((self.staging / "first-party/core/docs/strategy/strategy.md").exists())

    def test_exclude_pattern_matching_nothing_fails_refresh(self):
        self.manifest["repos"][0]["exclude"] = ["docs/specs/typo.md"]

        code, _, upload, _ = self.run_refresh("--upload")

        self.assertNotEqual(code, 0)
        upload.assert_not_called()

    @unittest.skipUnless(sys.version_info >= (3, 13), "pathlib glob matching needs Python 3.13")
    def test_glob_matches_agrees_with_the_pathlib_matcher(self):
        cases = [
            ("docs/specs/specs.md", "docs/specs/specs.md"),
            ("docs/specs/other.md", "docs/specs/specs.md"),
            ("docs/specs/specs.md", "docs/*/specs.md"),
            ("docs/a/b/specs.md", "docs/*/specs.md"),
            ("docs/strategy/strategy.md", "docs/strategy/**"),
            ("docs/strategy/a/b/plan.md", "docs/strategy/**"),
            ("docs/strategy/a/b/plan.md", "docs/**/plan.md"),
            ("docs/plan.md", "docs/**/plan.md"),
            ("docs/strategy/strategy.md", "docs/reference/**"),
        ]
        for rel, pattern in cases:
            with self.subTest(rel=rel, pattern=pattern):
                self.assertEqual(sync_corpus.glob_matches(rel, pattern), PurePosixPath(rel).full_match(pattern))

    def test_skip_fetch_intentionally_promotes_repo_only_snapshot(self):
        code, fetch, upload, reindex = self.run_refresh("--skip-fetch", "--upload")
        self.assertEqual(code, 0)
        fetch.assert_not_called()
        self.assertTrue((self.staging / "first-party/core/guide.md").is_file())
        self.assertFalse((self.staging / "vendor").exists())
        upload.assert_called_once()
        reindex.assert_called_once()

    def test_failed_promotion_restores_previous_snapshot(self):
        original = Path.rename

        def rename(path, target):
            if path.name == "candidate":
                raise OSError("promotion failed")
            return original(path, target)

        with patch.object(Path, "rename", rename):
            self.assert_failed_refresh()

    def test_failed_rollback_does_not_delete_previous_snapshot(self):
        original = Path.rename

        def rename(path, target):
            if Path(target) == self.staging:
                raise OSError("target unavailable")
            return original(path, target)

        with patch.object(Path, "rename", rename):
            code, _, upload, reindex = self.run_refresh("--upload")
        self.assertNotEqual(code, 0)
        preserved = list(self.root.rglob("old.md"))
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0].read_text(), "last complete snapshot")
        upload.assert_not_called()
        reindex.assert_not_called()

    def test_staging_only_refresh_also_fails_for_missing_sources(self):
        code, _, upload, reindex = self.run_refresh(fetch_error=TimeoutError())
        self.assertNotEqual(code, 0)
        self.assertEqual((self.staging / "old.md").read_text(), "last complete snapshot")
        upload.assert_not_called()
        reindex.assert_not_called()


class ExcludeGlobTests(unittest.TestCase):
    """Exclude patterns are matched with a portable glob: PurePath.full_match needs Python 3.13."""

    def test_literal_pattern_matches_only_itself(self):
        self.assertTrue(sync_corpus.glob_matches("docs/specs/specs.md", "docs/specs/specs.md"))
        self.assertFalse(sync_corpus.glob_matches("docs/specs/other.md", "docs/specs/specs.md"))

    def test_star_stays_inside_one_path_segment(self):
        self.assertTrue(sync_corpus.glob_matches("docs/specs/specs.md", "docs/*/specs.md"))
        self.assertFalse(sync_corpus.glob_matches("docs/nested/specs/specs.md", "docs/*/specs.md"))
        self.assertFalse(sync_corpus.glob_matches("docs/specs/specs.md", "docs/specs/*.md/extra"))

    def test_double_star_matches_any_depth_below_a_directory(self):
        self.assertTrue(sync_corpus.glob_matches("docs/strategy/strategy.md", "docs/strategy/**"))
        self.assertTrue(sync_corpus.glob_matches("docs/strategy/a/b/plan.md", "docs/strategy/**"))
        self.assertFalse(sync_corpus.glob_matches("docs/reference/reference.md", "docs/strategy/**"))
        self.assertTrue(sync_corpus.glob_matches("docs/architecture/ADR-1.md", "docs/**/ADR-*.md"))
        self.assertTrue(sync_corpus.glob_matches("docs/ADR-1.md", "docs/**/ADR-*.md"))

    def test_refresh_avoids_python_3_13_only_path_api(self):
        source = inspect.getsource(sync_corpus)
        self.assertNotIn("full_match", source)


class HtmlToTextTests(unittest.TestCase):
    def test_article_is_preferred_over_main_and_page_chrome(self):
        html = """<html><body><header>Site header</header>
            <nav>Home Docs API</nav>
            <main><div class="bd-sidebar">Sidebar TOC</div>
              <article><h1>Title<a class="headerlink" href="#t">#</a></h1><p>Body text.</p>
                <footer class="prev-next-footer">previous next</footer></article>
              <aside>On this page</aside></main>
            <footer>Privacy Policy</footer></body></html>"""
        self.assertEqual(sync_corpus.html_to_text(html), "Title\nBody text.")

    def test_main_is_used_when_there_is_no_article(self):
        html = "<body><nav>menu</nav><main><h1>EP options</h1><p>enable_cuda_graph</p></main><footer>x</footer></body>"
        self.assertEqual(sync_corpus.html_to_text(html), "EP options\nenable_cuda_graph")

    def test_role_main_on_a_div_closes_at_its_own_end_tag(self):
        html = "<body><div role='main'><div><p>inner</p></div><p>outer</p></div><div>after root</div></body>"
        self.assertEqual(sync_corpus.html_to_text(html), "inner\nouter")

    def test_only_first_article_is_kept(self):
        html = "<body><article><p>page</p></article><article><p>related card</p></article></body>"
        self.assertEqual(sync_corpus.html_to_text(html), "page")

    def test_nested_chrome_with_the_root_tag_does_not_leak_the_root(self):
        html = "<body><article><p>a</p><article class='sidebar'>x</article><p>b</p></article><p>after</p></body>"
        self.assertEqual(sync_corpus.html_to_text(html), "a\nb")

    def test_page_without_content_root_keeps_body_minus_chrome(self):
        html = "<body><nav>menu</nav><p>plain page</p><script>var x;</script><footer>legal</footer></body>"
        self.assertEqual(sync_corpus.html_to_text(html), "plain page")

    def test_empty_content_root_falls_back_to_whole_page(self):
        html = "<body><main id='app'></main><p>server text</p></body>"
        self.assertEqual(sync_corpus.html_to_text(html), "server text")

    def test_small_card_article_yields_to_main(self):
        # Hugging Face model pages: <article> holds a dataset/paper card, <main> holds the model card.
        card = "<article><p>Viewer • Updated Jun 26</p></article>"
        body = "<p>" + "Streaming ASR model card with usage and limits. " * 10 + "</p>"
        html = f"<body><main><div>{card}</div><div class='model-card'>{body}</div></main></body>"
        text = sync_corpus.html_to_text(html)
        self.assertIn("Streaming ASR model card", text)

    def test_void_elements_do_not_unbalance_skipping(self):
        html = "<body><main><nav><img src='a'><br>menu</nav><p>kept<br>line</p></main></body>"
        self.assertEqual(sync_corpus.html_to_text(html), "kept\nline")

    def test_content_links_are_preserved_as_markdown(self):
        html = "<body><main><p>See the <a href='/docs/performance/best-practices.html'>Best Practices</a> guide.</p></main></body>"
        text = sync_corpus.html_to_text(html, base_url="https://docs.nvidia.com/x/")
        self.assertEqual(text, "See the [Best Practices](<https://docs.nvidia.com/docs/performance/best-practices.html>) guide.")

    def test_fragment_only_links_keep_plain_text(self):
        html = "<body><main><p>Jump to <a href='#install'>Install</a> section.</p></main></body>"
        self.assertEqual(sync_corpus.html_to_text(html), "Jump to Install section.")

    def test_links_inside_chrome_are_still_dropped(self):
        html = "<body><nav><a href='/docs/x.html'>Home</a></nav><main><p>Body text.</p></main></body>"
        self.assertEqual(sync_corpus.html_to_text(html), "Body text.")

    def test_link_label_spans_text_nodes_keeps_boundary_spaces(self):
        html = "<body><main><p><a href='/docs/x.html'>Read <strong>more</strong> now</a></p></main></body>"
        text = sync_corpus.html_to_text(html, base_url="https://docs.nvidia.com/")
        self.assertEqual(text, "[Read more now](<https://docs.nvidia.com/docs/x.html>)")

    def test_link_label_and_destination_are_escaped(self):
        html = "<body><main><p><a href='/guide a[b]'>Label [x]</a></p></main></body>"
        text = sync_corpus.html_to_text(html, base_url="https://docs.example.com/")
        self.assertEqual(text, "[Label \\[x\\]](<https://docs.example.com/guide a[b]>)")

    def test_unclosed_anchor_keeps_its_text_as_plain(self):
        html = "<body><main><p>See <a href='/guide.html'>the guide</p><p>more detail</p></main></body>"
        text = sync_corpus.html_to_text(html, base_url="https://docs.example.com/")
        self.assertEqual(text, "See the guide more detail")

    def test_anchor_straddling_chrome_flushes_label_as_plain(self):
        html = "<body><main><a href='/x.html'>label <nav>chrome</a>tail</nav></main></body>"
        text = sync_corpus.html_to_text(html, base_url="https://docs.example.com/")
        self.assertEqual(text, "label")

    def test_small_card_with_links_still_yields_to_main(self):
        # A link-heavy card must not beat <main> through inflated rendered URLs.
        links = " ".join(
            f"<a href='/datasets/nvidia/parakeet-tdt-0.6b-v3/resolve/main/README.md'>source {i}</a>"
            for i in range(4)
        )
        card = f"<article><p>{links}</p></article>"
        body = "<p>" + "Streaming ASR model card with usage and limits. " * 10 + "</p>"
        html = f"<body><main><div>{card}</div><div class='model-card'>{body}</div></main></body>"
        text = sync_corpus.html_to_text(html, base_url="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/")
        self.assertIn("Streaming ASR model card", text)


class PinAndProvenanceTests(unittest.TestCase):
    def write_runtime_manifest(self, directory: Path, win: str, linux: str, trt: str = "1.7.1") -> Path:
        base = "https://github.com/NVIDIA/TensorRT-RTX-EP-ABI/releases/download"
        path = directory / "trt-rtx-ep.manifest.json"
        path.write_text(
            json.dumps(
                {
                    "version": win,
                    "cudaVariant": "cu13",
                    "trtRtxRuntimeVersion": trt,
                    "packages": {
                        "win-x64": {"archiveUrl": f"{base}/v{win}/EP-v{win}-cu13.zip"},
                        "linux-x64": {"archiveUrl": f"{base}/v{linux}/EP-v{linux}-cu13-linux.zip"},
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_pin_text_follows_the_runtime_manifest(self):
        with tempfile.TemporaryDirectory() as raw:
            manifest = self.write_runtime_manifest(Path(raw), "0.9.0", "0.9.0", trt="1.9.2")
            text = sync_corpus.pin_text(sync_corpus.read_pin(manifest))
        self.assertIn("version 0.9.0 on Windows and Linux", text)
        self.assertIn("TensorRT-RTX 1.9.2", text)
        self.assertIn("tensorrt_rtx_1_9.dll", text)
        self.assertNotIn("0.5.0", text)

    def test_pin_text_names_both_versions_when_platforms_differ(self):
        with tempfile.TemporaryDirectory() as raw:
            manifest = self.write_runtime_manifest(Path(raw), "0.5.0", "0.4.0")
            text = sync_corpus.pin_text(sync_corpus.read_pin(manifest))
        self.assertIn("version 0.5.0 on Windows and 0.4.0 on Linux", text)

    def test_shipped_pin_text_matches_the_repository_manifest(self):
        pin = sync_corpus.read_pin()
        shipped = json.loads(sync_corpus.RUNTIME_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(pin["trt"], shipped["trtRtxRuntimeVersion"])
        self.assertEqual(pin["cuda"], shipped["cudaVariant"])

    def test_vendor_sources_accept_pairs_and_objects(self):
        vendor = {"sources": [["a", "https://a"], {"id": "b", "url": "https://b", "gpu": {"id": "x"}}]}
        self.assertEqual(sync_corpus.vendor_sources(vendor), [("a", "https://a"), ("b", "https://b")])

    def test_sync_manifest_records_repo_commits(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "repo"
            root.mkdir()
            manifest = {"repos": [{"id": "demo", "rootEnv": "DEMO_ROOT", "defaultFromTrackdub": "."}]}
            with patch.dict(sync_corpus.os.environ, {"DEMO_ROOT": str(root)}), patch.object(
                sync_corpus, "git_state", return_value={"commit": "abc123", "dirty": False}
            ):
                sync_corpus.stage_sync_manifest(manifest, Path(raw))
            payload = json.loads((Path(raw) / sync_corpus.SYNC_MANIFEST_KEY).read_text(encoding="utf-8"))
        self.assertEqual(payload["repos"]["demo"], {"commit": "abc123", "dirty": False})
        self.assertIn("syncedAt", payload)

    def test_wait_returns_verifier_exit_code_once_job_is_done(self):
        active = subprocess.CompletedProcess([], 1, stdout="Indexing job active; verification deferred", stderr="")
        done = subprocess.CompletedProcess([], 0, stdout='{"ok": true}', stderr="")
        with patch.object(sync_corpus.subprocess, "run", side_effect=[active, active, done]) as run, patch.object(
            sync_corpus.time, "sleep"
        ) as sleep:
            self.assertEqual(sync_corpus.wait_and_verify(Path("api"), timeout_minutes=5), 0)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_wait_gives_up_when_the_job_never_finishes(self):
        active = subprocess.CompletedProcess([], 1, stdout="Indexing job active", stderr="")
        with patch.object(sync_corpus.subprocess, "run", return_value=active), patch.object(
            sync_corpus.time, "sleep"
        ) as sleep, patch.object(sync_corpus.time, "monotonic", side_effect=[0, 0, 10_000]):
            self.assertEqual(sync_corpus.wait_and_verify(Path("api"), timeout_minutes=1), 1)
        sleep.assert_called_once()

    def test_wait_returns_success_even_if_output_mentions_the_sentinel(self):
        done = subprocess.CompletedProcess([], 0, stdout="warning: Indexing job active earlier", stderr="")
        with patch.object(sync_corpus.subprocess, "run", return_value=done) as run:
            self.assertEqual(sync_corpus.wait_and_verify(Path("api"), timeout_minutes=5), 0)
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
