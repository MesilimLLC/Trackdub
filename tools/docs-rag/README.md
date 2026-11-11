# Trackdub docs RAG (v1)

Cloudflare AI Search over an R2 corpus. Agents call the `trackdub-docs-rag` Worker. Local keyword MCP in `tools/mcp-trackdub-gpu-docs` stays for offline TRT-RTX lookup.

## Layout in R2 `trackdub-docs-corpus`

- `first-party/trackdub/` public core docs, pin manifest, NvTensorRtRtx olive READMEs
- `first-party/trackdub-gated/` gated docs
- `first-party/api/` api.trackdub docs
- `vendor/{nvidia,microsoft,onnxruntime,amd,intel,qualcomm,qwen,whisper}/` allowlisted pages

`--upload` uploads objects with first-party metadata and requests an indexing job after all uploads succeed. AI Search also syncs every 24 hours as a safety net; requesting a job does not mean indexing has finished.

## One-time Cloudflare setup

From `api.trackdub` (Wrangler 4, logged in):

```powershell
npx wrangler r2 bucket create trackdub-docs-corpus
npx wrangler secret put DOCS_RAG_TOKEN --config wrangler.docs-rag.jsonc
```

In the dashboard, create AI Search instance `trackdub-docs` with data source R2 bucket `trackdub-docs-corpus`. Do not enable the public unauthenticated MCP endpoint. Agents use this Worker, which requires `Authorization: Bearer <DOCS_RAG_TOKEN>`.

Confirm the generation model still exists before relying on answers:

```powershell
npx wrangler ai models
```

Default is `@cf/meta/llama-3.3-70b-instruct-fp8-fast` (`AI_SEARCH_MODEL` in `wrangler.docs-rag.jsonc`).

## Sync

```powershell
python tools/docs-rag/sync_corpus.py
python tools/docs-rag/sync_corpus.py --upload
python tools/docs-rag/sync_corpus.py --upload --wait   # also polls the reindex job, then runs verify_index
```

`--wait` (with `--wait-timeout` minutes, default 40) replaces the manual upload, poll, verify loop: the exit code is the verifier's, so a zero means "indexed and verified". `.github/workflows/docs-rag-sync.yml` runs exactly this on pushes to the indexed paths, weekly, and on demand; it never passes `--prune`.

### One allowlist

Remote pages are listed once, in `corpus.v1.json`. A vendor source is `["id", "url"]` or `{"id", "url", "gpu": {"id", "title", "tags", "priority", "pinWarning"}}`; the `gpu` block makes the page part of the offline `tools/mcp-trackdub-gpu-docs` corpus too, which reads it from here. Local repo-file sources for gpu-docs stay in its own manifest.

### Generated pin text

`first-party/trackdub/docs/reference/docs-rag-pin.md` is generated from `runtime/trt-rtx-ep.manifest.json` on every sync (unless the repo ships its own copy). The Worker's agent instructions and answer prompt come from `api.trackdub/src/docs-rag/pin.generated.ts`, written by `scripts/gen-docs-rag-pin.mjs` before `dev:docs-rag` and `deploy:docs-rag`; `check:docs-rag-pin` (and the sync workflow) fail when it is stale.

### Provenance

Each sync stages `first-party/_sync.json` with the commit of every indexed repo. The Worker stamps first-party hits with `repo`, `localPath` and `syncedAt`, and hides the manifest itself from results. It adds `syncedCommit` (and `syncedDirty` when the tree was modified) only when the hit's index timestamp is at or after its own R2 object's upload time, so the indexed bytes match that commit. Unchanged files keep their old upload time and keep their stamp. Hits that fail the check, or whose object is missing, get no commit stamp. Agents with the checkout should read `localPath` instead of the indexed copy.

### Eval

`tools/docs-eval/questions.json` holds known-item questions with the documents that should come back. `uv run python ../docs-eval/eval_gpu.py` (from `tools/mcp-trackdub-gpu-docs`, offline) and `DOCS_RAG_TOKEN=... node tools/docs-eval/eval_rag.mjs` report hit@1, hit@3 and MRR. The gpu-docs unit tests enforce a floor; the Worker eval needs a live token and a synced corpus, so it is run by hand.

Run from the Trackdub repo. Sibling checkouts `../Trackdub-gated` and `../api.trackdub` are picked up automatically. Override with `TRACKDUB_ROOT`, `TRACKDUB_GATED_ROOT`, `API_TRACKDUB_ROOT`.

Requires Node.js 22.16.0+ (the `mise.toml` baseline; `upload_corpus.mjs` uses `Dirent.parentPath`, unavailable before 18.20/20.12) and installed Wrangler in `api.trackdub`, plus Python 3.10+ for staging (verified on 3.10 through 3.14). `mise.toml` pins Node, .NET and pnpm, so the Python interpreter comes from the environment. `upload_corpus.mjs` opens a temporary Wrangler remote R2 binding, uploads bytes with `customMetadata.is_first_party: "true"` only for `first-party/**`, then disposes the proxy. Vendors omit the field, including `false`, because the boost tests existence. Wrangler's `r2 object put` still has no `--header` flag as of 4.124.0, only `--content-type` and friends, so the binding path is required for custom metadata.

The uploader lists the bucket once and skips a document whose stored size, MD5 `etag`, content type and custom metadata all already match, logging `skip` instead of `put`. A re-put refreshes `last_modified`, which makes AI Search treat the object as changed and re-embed it; re-embedding the whole corpus at once overruns Workers AI capacity and leaves items `outdated`. Skipping unchanged objects keeps a document prune down to deletions plus a reindex. Metadata is checked with a per-object `head`, because `list` rows return `customMetadata` and `httpMetadata` as null, so changing the metadata rule still re-uploads every affected object.

Both upload and reindex use Wrangler authentication. For an existing OAuth login, unset stale `CLOUDFLARE_API_TOKEN` / `CLOUDFLARE_API_KEY` overrides in the calling shell. `AI_SEARCH_API_TOKEN` is a Worker secret, not a Wrangler authentication variable. No secrets are passed as CLI arguments.

Refresh builds a separate candidate tree. Missing repositories, repositories with no eligible documents, and failed or empty vendor fetches exit nonzero before upload or reindex, retaining the previous staging snapshot. Successful refresh replaces that snapshot. Oversized repository files remain explicitly skipped under `maxBytes`; vendor content is capped at that size. If promotion and rollback both fail, the previous snapshot stays at `.staging.previous`; recover it before refreshing again.

A repository entry may carry `exclude`: repo-relative glob patterns matched segment by segment by `glob_matches`, which keeps `*` inside one path segment and lets `**` span directories. Pathlib's own pattern match needs Python 3.13, and `fnmatch` alone would let `*` cross `/`, so the matcher is local to this tool. Exclusion drops mirrored aggregate pages whose bodies duplicate the leaf docs already in the corpus. A pattern matching nothing fails the refresh, so a renamed or deleted mirror cannot silently stop being excluded.

`--reuse-staging --upload` resends the cached tree and requests reindex; it does not fetch current sources or certify a cache created by an older script. `--skip-fetch --upload` intentionally stages only repo files and does not delete existing vendor objects in R2. Failed uploads skip reindex and exit nonzero. A failed job request also exits nonzero, but a timeout can occur after the job has started: inspect `npx wrangler ai-search jobs list trackdub-docs` before retrying `jobs create` from `api.trackdub`.

Staging changes alone never remove objects from the bucket. `--upload --prune` deletes bucket objects missing from the staged tree, and only after every upload succeeds; the keys come from the single list pass taken before the first put, and that pass also drives the unchanged-object skips. Pruning is refused for `--skip-fetch`, `--reuse-staging`, and a bare `--prune`, because those trees are knowingly incomplete and would delete every vendor object. Objects already indexed from a deleted key stay retrievable until the next reindex, and how long AI Search keeps them is not specified.

## Verify indexing separately

```powershell
node tools/docs-rag/verify_index.mjs --account <cloudflare-account-id> --api-root ../api.trackdub
```

Uses `CLOUDFLARE_API_TOKEN` or the existing Wrangler login; `CLOUDFLARE_ACCOUNT_ID` and `API_TRACKDUB_ROOT` can replace the flags. This read-only check never uploads or starts a job. Authentication has a 20-second timeout; API checks share a 90-second deadline. It exits nonzero while a job is active, on unstable/duplicate pagination, missing or extra indexed keys, incorrect source/index first-party flags or indexed scope folders, stale item status, pending actions, absent chunks, size differences, or items not seen since their source upload. Source inventory and job state are checked again to detect concurrent changes.

A passing inventory also requires an uncached retrieval of the EP ABI reference with both canonical provider symbols in its content. This verifies one content canary, not byte-for-byte freshness of every chunk; same-size changes cannot be proven from index checksums, which differ from R2 ETags. Job acknowledgment and aggregate completed counts are never treated as verification. Partial upload failures can still leave some R2 objects updated; this is not a transactional remote upload.

Local tests:

```bash
python -m unittest discover -s tools/docs-rag -p test_sync_corpus.py
node --test tools/docs-rag/upload_corpus.test.mjs tools/docs-rag/verify_index.test.mjs
```

## Worker

```powershell
cd ../api.trackdub
npm run dev:docs-rag
npm run deploy:docs-rag
```

`POST /mcp` speaks MCP `initialize`, `tools/list`, and `tools/call` for `search_trackdub_docs` and `ask_trackdub_docs`. `POST /v1/search` and `POST /v1/ask` are the same operations over JSON. Optional `scope`: `all`, `first-party`, `trackdub`, `trackdub-gated`, `api`, `vendor`, `nvidia`, `microsoft`, `amd`, `intel`, `qualcomm`, `qwen`, `whisper`, `speech-models`, `onnxruntime`.

Cursor entry after deploy (workers.dev host is printed by Wrangler):

```json
"trackdub-docs-rag": {
  "url": "https://trackdub-docs-rag.<account>.workers.dev/mcp",
  "headers": { "Authorization": "Bearer <DOCS_RAG_TOKEN>" }
}
```
