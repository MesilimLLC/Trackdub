# Trackdub Docs RAG (MCP)

Hosted MCP server `trackdub-docs-rag` answering "how does X actually work at
Trackdub" from first-party docs + pin-accurate vendor docs. Spec:
`tools/docs-rag/SPEC.md`. Corpus: ~380 docs, Cloudflare AI Search over R2. Kept current by `.github/workflows/docs-rag-sync.yml`.

Tools: `search_trackdub_docs`, `ask_trackdub_docs`, `get_trackdub_doc`.

## Required

- **Use it for implementation facts before guessing.** EP wiring,
  model-family support, "which layer owns this decision", and "what does this
  vendor page actually say" should come from a search, not from memory. Pin
  versions come from `runtime/trt-rtx-ep.manifest.json` first; the RAG repeats
  them, it does not define them. A plausible-sounding wrong version is a
  defect, not a rounding error.
- **Read `AGENTS.md` and repo source first for policy.** The RAG indexes
  documentation, which is the *third* place in the conflict order (source
  code/tests > task instructions > Linear > documentation). It explains and
  cross-checks; it does not override what the tests assert.
- **Cite the `key` you got the fact from.** Every hit carries an R2 path
  (`first-party/trackdub/...`, `vendor/nvidia/...`). Name it when stating a
  pin or claim so the answer is checkable. First-party hits also carry `repo`,
  `localPath` and `syncedCommit`: with that checkout available, read the local
  file; the indexed copy is only as new as that commit.
- **Escalate to `get_trackdub_doc` when a chunk is cut off** (4000-char cap).
  Read the whole doc before drawing a conclusion from a truncated one.
- **Prefer first-party over vendor for Trackdub behavior.** Vendor scopes
  (`nvidia`, `microsoft`, `onnxruntime`, `amd`, `intel`, `qualcomm`, `qwen`,
  `whisper`, `speech-models`) describe the upstream product, not what Trackdub
  pins or configures. A vendor default is not a Trackdub default.
- **Never fake readiness, in docs or in code.** Registered provider != model
  downloaded != stage ran != stage succeeded. Corpus hits are documentation and
  prove nothing about the current machine or index state.

## Choosing a tool

| Need | Tool |
|---|---|
| Verbatim chunks, exact identifiers, quotes, EP APIs | `search_trackdub_docs` |
| Synthesized explanation across docs (confirm versions and flags against the cited keys) | `ask_trackdub_docs` |
| A chunk you already have but need whole | `get_trackdub_doc` |

`search` is the default. It is hybrid vector+keyword, so a code identifier
(camel-case, dotted, `Namespace:Member`, snake_case) usually resolves directly.
Use `ask` only when you want a synthesized explanation, and treat its answer as
generated text: check every version, flag and symbol against the cited keys.

## Scopes

`all`, `first-party`, `trackdub`, `trackdub-gated`, `api`, `vendor`, `nvidia`,
`microsoft`, `amd`, `intel`, `qualcomm`, `qwen`, `whisper`, `speech-models`,
`onnxruntime`.

- Default to `first-party` for Trackdub design and policy questions.
- Olive is reachable via `vendor` or `all` only; there is no `olive` scope.
- `trackdub-gated` and `api` are separate first-party surfaces — scope in
  deliberately rather than relying on `all` to surface them.

## Known limits — state these, do not paper over them

- **Empty scoped results are not proof of absence.** Indexing lags a 24h
  scheduled sync plus job-triggered reindexes. Fall back to `all`, or say the
  corpus came up empty; do not conclude the behavior does not exist.
- **Thin spots in the corpus** (known, documented gaps): Silero VAD upstream
  wiki, DeepFilterNet3, SepFormer, CosyVoice/Chatterbox HF cards, LatentSync.
  An empty result in these areas is a coverage gap, not an answer.
- **First-party is a relevance boost, not a guarantee.** Boosting biases
  candidates before reranking; a vendor hit may still outrank a first-party
  one. Check the `key`, don't infer from `score`.
- **`ask` never proves query rewriting happened.** `query_rewrite` applies to
  follow-up messages, and this tool is stateless with no history.
  `searchQuery: null` is normal. Don't report a rewrite as evidence.
- **Symbol-fallback scores are unreranked** and not comparable to normal
  scores. Don't compare them across modes.
- **Upstream 429s propagate**, including `limiter: "upstream"`. A shared token
  means one busy agent throttles all of them — space calls out rather than
  retry-looping.

## Other doc sources

Prefer the RAG for implementation facts. For CUDA toolkit internals outside
the corpus, use NVIDIA's CUDA MCP. For offline/no-network TRT-RTX lookup, use
the local `trackdub-gpu-docs` MCP.
