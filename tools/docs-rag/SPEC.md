# Trackdub Docs RAG — Spec

Version: 1.4 (2026-10-01)
Status: Live
Endpoint: `https://trackdub-docs-rag.trackdub.workers.dev`
Owner: Tony Thompson
Code: `api.trackdub/src/docs-rag/` (Worker), `Trackdub/tools/docs-rag/corpus.v1.json` (manifest)

## 1. What it is

A hosted, bearer-authenticated MCP server backed by Cloudflare AI Search
(AutoRAG) over a curated corpus of Trackdub implementation documentation. It
answers "how does X actually work at Trackdub" from first-party docs plus
pin-accurate vendor docs, for any MCP-capable agent (Cursor, Claude Code, etc).

It is one of three complementary doc sources an agent may use:

| Source | Covers | When to prefer |
|---|---|---|
| **trackdub-docs-rag** (this) | Trackdub first-party docs + pinned vendor docs | Implementation facts, pin policy, provider wiring, "how does Trackdub do X" |
| NVIDIA CUDA MCP (`search_cuda_docs`) | CUDA toolkit internals | CUDA kernel/driver questions outside the RAG corpus |
| `tools/mcp-trackdub-gpu-docs` (local) | Offline TRT-RTX keyword lookup | No-network / fast local lookup; same Trackdub pin docs |

## 2. Architecture

```text
Agent (Cursor / Claude Code)
   |  Authorization: Bearer <DOCS_RAG_TOKEN>
   v
trackdub-docs-rag Worker (api.trackdub, agents-SDK MCP)
   |  tools: search / ask / get_doc
   |  rate limits: burst / steady / ask (per client)
   v
Cloudflare AI Search instance "trackdub-docs"
   |  Authorization: Bearer <AI_SEARCH_API_TOKEN>   (REST, server-side only)
   v
R2 bucket trackdub-docs-corpus
```

Two credentials, different jobs:

| Secret | What it is | Who presents it | Rotate by |
|---|---|---|---|
| `DOCS_RAG_TOKEN` | Random string (>=16 chars), no Cloudflare meaning | Agents -> Worker (in `mcp.json` / `.claude.json`) | `wrangler secret put DOCS_RAG_TOKEN` + update agent configs |
| `AI_SEARCH_API_TOKEN` | Real Cloudflare API token (Account > AI Search Edit + Run) | Worker -> AI Search REST API, server-side only | Rotate in dashboard, `wrangler secret put AI_SEARCH_API_TOKEN` |

## 3. The corpus

R2 bucket `trackdub-docs-corpus` contains **197 uploaded documents**. Instance
`trackdub-docs` indexes them with `@cf/qwen/qwen3-embedding-0.6b`, hybrid search
and reranking, 24h scheduled sync, and a job request after successful uploads.
An uploaded object or ended job does not prove the index is current; inspect
item status and metadata in the Cloudflare dashboard when needed.

### Instance settings

| Setting | Value |
|---|---|
| Scheduled sync | `sync_interval: 86400` (24h) |
| Keyword tokenizer | `trigram`, for code identifier substrings |
| Keyword match mode | `and`; Worker retries `or` once only on an empty successful search |
| Custom metadata | `is_first_party`, boolean |
| Relevance boost | `is_first_party`, direction `exists` |
| Chunking | 1024 tokens, 10% overlap (unchanged) |
| Similarity cache | `close_enough`, 172800 seconds (48h, unchanged) |

First-party objects carry R2 custom metadata `is_first_party: "true"`;
vendors omit that field entirely. Setting `false` would still match the
existence boost. Boosting biases candidates before reranking; it is not a
hard guarantee that every first-party hit outranks every vendor hit. Changing
the tokenizer or metadata schema requires full reindexing.

| Folder | Contents |
|---|---|
| `first-party/trackdub/` | Public core docs, AGENTS.md, TRT-RTX pin manifest, olive-recipe NvTensorRtRtx READMEs |
| `first-party/trackdub-gated/` | Gated repo docs |
| `first-party/api/` | api.trackdub docs |
| `vendor/nvidia/` | TRT-RTX docs (arch, AOT/JIT, support matrix, best practices, troubleshooting, APIs) + EP ABI releases + Sortformer/Nemotron model cards + Maxine Audio Effects (current AFX user guide: Windows/Linux setup, effects, application flow, API) |
| `vendor/microsoft/` | Windows ML (overview, EP selection, initialization), DirectML |
| `vendor/onnxruntime/` | EP docs (TRT-RTX, plugin EP, CUDA, DirectML, OpenVINO, QNN, MIGraphX), ORT GenAI, performance tuning, quantization |
| `vendor/olive/` | Microsoft Olive docs (the pinned Olive 0.13.0 recipe engine) |
| `vendor/amd/` | MIGraphX (install, driver, C++ API, operators, quantization) |
| `vendor/intel/` | OpenVINO (get started, workflow, generative) |
| `vendor/qualcomm/` | QNN (overview, integration guide, backends) |
| `vendor/qwen/` | Qwen2.5-1.5B and Qwen3-Embedding model cards |
| `vendor/whisper/` | OpenAI Whisper + faster-whisper |
| `vendor/speech-models/` | Kokoro-82M, Opus-MT, MADLAD400, Spleeter |
| `vendor/lip-sync-alignment/` | Wav2Vec2-LV60-eSpeak (CTC forced alignment), Qwen3-ForcedAligner, LatentSync 1.6, InsightFace (SCRFD-500M + 2D106det) |
| `vendor/avalonia/` | Avalonia 12 UI docs (welcome, get started, basics, MVVM, ReactiveUI, data binding, controls, cross-platform) |
| `vendor/reactiveui/` | ReactiveUI 20.x docs (getting started, binding, primitives, validation, Splat, Akavache) |

### Model-family coverage check (v1.2)

All `bundled-models.manifest.json` engine families that ship vendor docs:
whisper (genai+onnx), qwen (asr/instruct/tts), phi-genai, kokoro, chatterbox,
cosyvoice, opus-mt, madlad, nemotron-asr, sortformer, spleeter, silero-vad,
kokoro/qwen3-tts (HF cards), olive recipes. **Known thin spots**: Silero VAD
upstream wiki, DeepFilterNet3, SepFormer, CosyVoice/Chatterbox HF cards
(401-anonymous), LatentSync. Add to `corpus.v1.json` `vendors` and refresh the hosted corpus manually.

### Keeping it fresh

The Trackdub repo no longer ships automated corpus ingest (the former
`.github/workflows/docs-rag-sync.yml` and `sync_corpus.py` tooling were removed).
`corpus.v1.json` remains the manifest of intended sources for agents and for
`tools/mcp-trackdub-gpu-docs`. Refreshing the live R2 bucket and AI Search index
is manual ops in `api.trackdub` / Cloudflare when Tony chooses to run it.

To inspect or trigger indexing from `api.trackdub`:

```bash
cd ../api.trackdub
npx wrangler ai-search jobs list trackdub-docs
# Dashboard -> AI Search -> trackdub-docs -> Sync
```

Adding a vendor page — manual steps (no automated sync remains):

1. Add `[id, url]` under the vendor in `corpus.v1.json` (a `{"id", "url"}`
   object with a `gpu` block also works). This repo's manifest is the source
   of truth; `tools/mcp-trackdub-gpu-docs` reads the same entries.
2. Fetch the URL. Vendor pages are stored as extracted text, not HTML: keep
   only the main content region (`article` preferred over `main` when both
   exist), drop chrome (nav, headers/footers, sidebars, Sphinx `#` header
   links, scripts/styles), and preserve in-content links as `[text](url)`.
   Some vendors serve raw markdown or text (`.md` / GitHub blobs) — if the
   body already reads as plain text, save it as-is. The old
   `sync_corpus.py` `html_to_text` / `fetch_url` logic (deleted) is the
   reference for the extraction rules.
3. Normalize to `Source: <url>\n\n<text>\n`. If the UTF-8 payload exceeds
   `corpus.v1.json` `maxBytes` (3,500,000), truncate to `maxBytes` — the
   old staging did exactly this (byte-prefix, dropping a partial trailing
   character).
4. Upload to R2 bucket `trackdub-docs-corpus` at key
   `<vendor prefix>/<source id>.md` (prefixes are e.g. `vendor/nvidia`,
   `vendor/microsoft` — the `prefix` field on each entry in
   `corpus.v1.json`). Content type `text/markdown; charset=utf-8` (`.json`
   objects get `application/json`). Size must stay within `maxBytes`;
   nothing else about the object is checked downstream.
   Caveat: `wrangler r2 object put --header` is not implemented in
   Wrangler 4.114.0 (documented limitation), so the CLI cannot set custom
   metadata that way. Use the R2 Workers API (via `getPlatformProxy`, as
   the deleted `upload_corpus.mjs` did) or an S3-compatible client that
   sets custom metadata directly.
5. Set metadata per this convention — the same rule the retired uploader
   applied and the dashboard shows today:
   - Key starts with `first-party/` → custom metadata `is_first_party: "true"`.
   - Everywhere else (all vendor keys) → omit `is_first_party` entirely.
   Never set it to `"false"` (the existence boost in section 3 would still
   treat it as first-party). Changing the tokenizer or this schema requires
   a full reindex.
6. Index: from `api.trackdub`, `npx wrangler ai-search jobs list
   trackdub-docs` (inspect first — a timeout may still have started the
   job), then `npx wrangler ai-search jobs create trackdub-docs` only if
   no job was created, or Dashboard → AI Search → `trackdub-docs` → Sync.
   Verify the dashboard item status — an ended job does not prove the
   index is current (section 3 preamble).

Adding a new scope (agent-visible filter): update `DOC_SCOPES`, `FOLDER`, and
the `filterForScope` switch in `api.trackdub/src/docs-rag/scopes.ts`, then
add a scope test row.

## 4. Agent tools

Stateless agents-SDK MCP server (`createMcpHandler`, streamable HTTP at
`POST /mcp`). Fresh `McpServer` per request (SDK >= 1.26 CVE guidance).

| Tool | Input | Returns |
|---|---|---|
| `search_trackdub_docs` | `query`, `scope?`, `limit?` (1-20, default 8) | Chunks: `score`, `key` (R2 path), `text` (<=4000 chars), plus `matchMode`. Hybrid vector+keyword, reranked except the bounded symbol fallback below. |
| `ask_trackdub_docs` | `query`, `scope?`, `limit?` | AI answer + scope-enforced source chunks. Prompt prefers first-party over vendor; refuses to invent readiness/APIs. |
| `get_trackdub_doc` | `key`, `max_chars?` (default 50000) | Full document text straight from R2 (4MB cap). Use when chunks are truncated. |

`scope` values: `all`, `first-party`, `trackdub`, `trackdub-gated`, `api`,
`vendor`, `nvidia`, `microsoft`, `amd`, `intel`, `qualcomm`, `qwen`,
`whisper`, `speech-models`, `onnxruntime`. Olive documents are available through
`vendor` or `all`; a dedicated Olive scope is not implemented.

Scope enforcement: filters are ASCII range queries on the AI Search `folder`
metadata (`$gte: "vendor/nvidia/", $lt: "vendor/nvidia0"` — note the stripped
trailing slash; `"vendor/nvidia/0"` would sort above every real value and
match nothing). Retrieval always goes through the REST API for this; the
legacy `aiSearch` binding rejects folder range filters.

`search` explicitly requests keyword mode `and`, then retries once with `or`
only if the successful response contains no chunks. Both requests keep the
same scope and result limit. If both are empty and the query is one code-like
identifier (4-128 ASCII characters, camel-case, dotted, colon-qualified or snake-case),
one additional hybrid search disables cache and reranking for that request only.
It fetches at most 20 candidates in the same scope, retains literal case-insensitive
identifier-prefix matches in the returned text, and caps output at the requested
limit. Ordinary prose never takes this fallback. This is bounded candidate recovery,
not exhaustive substring search; candidates outside the top 20 remain unavailable.
Responses identify `matchMode: "and"`, `"or-fallback"`, or `"symbol-fallback"`.
Symbol scores are unreranked retrieval scores, not comparable to reranked scores.
Upstream errors and rate limits propagate at every step. Global models, thresholds,
and reranking configuration are unchanged.

`ask` makes one REST chat-completions call with system/user `messages`,
`ai_search_options.query_rewrite.enabled: true`, and scope/limit inside
`ai_search_options.retrieval`. Returned hits are the same chunks supplied to
generation, not a separate search's results. `searchQuery` is null when the
upstream chat response does not expose `search_query`.

**Query rewriting limitation:** Cloudflare rewrites follow-up messages, not
the first user message. This stateless, single-query tool does not supply
conversation history, so enabling the option does not establish that a rewrite
occurred. Do not fabricate history or report the original query as evidence of
rewriting. See [Cloudflare query rewriting](https://developers.cloudflare.com/ai-search/configuration/retrieval/query-rewriting/).

## 5. Rate limits

Per client (key = SHA-256-ish hash of bearer token, never the raw value):

| Binding | Window | Cap | Applies |
|---|---|---|---|
| `RL_BURST` | 10s | 30 | all endpoints |
| `RL_STEADY` | 60s | 300 | all endpoints |
| `RL_ASK` | 10s | 5 | `/v1/ask` + `ask` tool only |

Responses: HTTP 429 with `Retry-After` and `{"error":"rate_limited",
"limiter":"burst|steady|ask|upstream"}`. AI Search's own backpressure
(code 2003) is re-mapped to the same clean 429. Note the binding counters are
per-Cloudflare-location (eventual consistency), so these are abuse caps, not
exact quotas.

## 6. How to connect an agent

Endpoint: `https://trackdub-docs-rag.trackdub.workers.dev/mcp`
Auth: `Authorization: Bearer <DOCS_RAG_TOKEN>` on every request.

### Cursor (project, committed)

`Trackdub/.mcp.json` (Claude Code: `Bearer ${DOCS_RAG_TOKEN}`, ships
`"disabled": true` so contributors without the token don't break config parsing),
`Trackdub/.cursor/mcp.json` (Cursor IDE: `Bearer ${env:DOCS_RAG_TOKEN}`, enabled
as committed), and `Trackdub-gated/.mcp.json` register:

```json
"trackdub-docs-rag": {
  "type": "http",
  "url": "https://trackdub-docs-rag.trackdub.workers.dev/mcp",
  "headers": { "Authorization": "Bearer <DOCS_RAG_TOKEN>" }
}
```

Set `DOCS_RAG_TOKEN` in the local shell or system environment — Cursor expands
`${env:DOCS_RAG_TOKEN}` from there and Claude Code expands `${DOCS_RAG_TOKEN}`.
Do not commit the literal token. To use the Claude Code entry, remove
`"disabled": true` locally (opt-in).

### Cursor Cloud Agents

A committed project `mcp.json` does **not** reach Cloud Agents by itself. Add the
same HTTP server under [cursor.com/agents](https://cursor.com/agents) (personal
MCP dropdown) or **Dashboard → Integrations & MCP** (team), with header
`Authorization: Bearer <DOCS_RAG_TOKEN>`. The token is held as a Cursor Secret
(`DOCS_RAG_TOKEN`, injected into the cloud-agent runtime) and/or pasted as the
dashboard header — this is separate from, and does not replace, the local shell
or system environment used by the IDE configs above. Streamable HTTP only;
SSE/`mcp-remote` are unsupported. Tool calls are proxied; the bearer stays out
of the VM.

### Cursor (global)

`~/.cursor/mcp.json`, same shape.

### Claude Code

Already registered in user scope:

```bash
claude mcp add -s user --transport http trackdub-docs-rag \
  "https://trackdub-docs-rag.trackdub.workers.dev/mcp" \
  --header "Authorization: Bearer <DOCS_RAG_TOKEN>"
```

### Other MCP clients (generic)

Streamable HTTP at `/mcp`; initialize handshake returns serverInfo
`trackdub-docs-rag` v0.2.0. Any client that speaks
`Authorization` headers + streamable HTTP works. Raw HTTP alternative:
`POST /v1/search` and `POST /v1/ask` (`{"query","scope","limit"}`).
### Test validation with project configuration

The committed `Trackdub/.mcp.json` entry ships `"disabled": true` so contributors
without `DOCS_RAG_TOKEN` don't break config parsing. To validate that an agent
discovers the three tools, enable it locally first (remove `"disabled": true`
from `.mcp.json`), then verify with `DOCS_RAG_TOKEN` set:

**Claude Code (project context):**
```bash
# From Trackdub repo root, with DOCS_RAG_TOKEN exported
claude mcp list --context project
```

Expected output includes:
```
trackdub-docs-rag (http)
  Tools: search_trackdub_docs, ask_trackdub_docs, get_trackdub_doc
```

**Cursor IDE:**
Open MCP panel in Cursor settings; the trackdub-docs-rag server should appear with status "Connected" and show 3 tools after environment variable substitution succeeds.

**Direct HTTP initialization (any client):**
```bash
curl -X POST https://trackdub-docs-rag.trackdub.workers.dev/mcp -H "Authorization: Bearer ${DOCS_RAG_TOKEN}" -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test-client","version":"1.0"}}}'
```


### Adding a *new agent identity* (per-agent tokens)

Currently one shared token. To give an agent its own quota/revoke key:
generate a token per agent, store an allowlist mapping
`hash(token) -> name` (env var or KV), key the rate limiter on that hash, and
revoke by removal. Not built yet; the limiter key derivation already avoids
storing raw tokens so the extension is small (`auth.ts`).

## 7. Operations quick reference

```bash
# Deploy the Worker
cd api.trackdub && npm run deploy:docs-rag

# Health check
curl https://trackdub-docs-rag.trackdub.workers.dev/health

# Inspect index state (from api.trackdub)
npx wrangler ai-search get trackdub-docs
npx wrangler ai-search jobs list trackdub-docs

# Force-sync without CLI
# Dashboard -> AI Search -> trackdub-docs -> Sync
```

## 8. If something breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| `unauthorized` from Worker | Wrong `DOCS_RAG_TOKEN` header | Re-set secret + update agent config |
| 429 `limiter: upstream` often | Corpus too hot / shared key | Space out; or add per-agent tokens (section 6) |
| Empty scoped results but `all` works | Indexing lag after adding docs | Trigger `jobs create`; check `jobs list` |
| `AI Search REST 401` in Worker logs | `AI_SEARCH_API_TOKEN` rotated/expired | Create token w/ AI Search Edit+Run, `wrangler secret put` |
| `AutoRAGNotFoundError` | Instance name mismatch | `AI_SEARCH_INSTANCE` var must equal dashboard name |
| Model 404 on `ask` | Model retired | `npx wrangler ai models`, update `AI_SEARCH_MODEL` |
| Reindex request HTTP 524 | Cloudflare timed out before acknowledging the request | Check `jobs list` and job logs first; the job may already have started |

## 9. Related docs in the repos

- `Trackdub/tools/mcp-trackdub-gpu-docs/README.md` — the local v0 MCP
- `Trackdub/docs/reference/tensorrt-rtx-ep-abi-plugin.md` — the pin this corpus is keyed to (EP ABI 0.5.0 win + linux, cu13, TRT-RTX 1.7.1)
- `Trackdub-gated/docs/audits/2026-09-19-four-repo-audit/` — corpus includes this audit set
