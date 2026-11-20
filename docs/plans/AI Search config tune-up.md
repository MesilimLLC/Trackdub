---
name: AI Search config tune-up
overview: Original delivery published; follow-up fixes verified locally and metadata repaired live. Capacity errors and follow-up deployment remain open.
todos:
  - id: fix-rewrite
    content: Use one scoped messages-format REST chat request; preserve null when upstream omits search_query and document stateless first-message limitation.
    status: completed
  - id: auto-reindex
    content: Request indexing automatically after all staged uploads succeed.
    status: completed
  - id: stamp-metadata
    content: Upload first-party R2 objects with is_first_party=true and omit the field for vendors.
    status: completed
  - id: instance-config
    content: Apply 24h sync, trigram tokenizer, boolean metadata schema and existence boost.
    status: completed
  - id: search-fallback
    content: Retry successful empty AND retrieval once with OR, preserving scope and limit and reporting matchMode.
    status: completed
  - id: repair-metadata
    content: Rebuild indexed metadata, remove temporary schema field and verify all 126 flags and vendor omission.
    status: completed
  - id: reject-partial-refresh
    content: Fail incomplete source staging before upload and preserve previous reusable cache, including failed promotion recovery.
    status: completed
  - id: verify-index-tool
    content: (Retired with sync) Bounded read-only source/index verifier with key, metadata, folder, status, size, timestamp and content-canary checks.
    status: completed
  - id: symbol-recovery
    content: Implement scoped literal-prefix fallback and test modified local function against live Cloudflare without global ranking changes.
    status: completed
  - id: verify
    content: Resolve remaining Cloudflare capacity errors and pass full index verification; retain first-party ordering caveat. Manual acceptance (2026-11-20): see Remaining limits.
    status: pending
  - id: docs-commit
    content: Publish original implementation and handoff in separate core/API commits.
    status: completed
  - id: followup-publish
    content: Commit/push and deploy local follow-up only when authorized.
    status: pending
isProject: false
---

# AI Search config tune-up: execution status

Tracked record for the supplied plan from
`Trackdub-gated/external/Trackdub/docs/plans/AI Search config tune-up.md`.
The pinned submodule is unchanged. Known deficiencies remain explicit rather
than being represented as passing acceptance checks.

## Original published implementation

| Plan step | Implementation and evidence |
|---|---|
| Messages-format ask | One scoped REST chat-completions call supplies answer and citations; query rewriting enabled; limit nested under retrieval. |
| First-party metadata | Remote-binding uploader preserves bytes and attaches metadata only to first-party objects. All 197 source objects checked: 126 flags, 71 vendors without flags. |
| Automatic indexing | All uploads must succeed before requesting an indexing job. Ambiguous timeouts require inspecting existing jobs before retrying. |
| Instance settings | 86400-second sync, trigram tokenizer, boolean `is_first_party` schema and existence boost. |
| Empty-search fallback | AND then one OR request only after successful empty retrieval; same scope and limit. Errors propagate. |
| Publish | Core `a3e3f18`, `c751d4a`, plan record `c747acf`; API `59037f8`, on respective `agent/rag-tuneup` branches. No merge or PR. |
| Deploy | Worker version `27a4fccc-6c5d-4d0a-8fc7-e64206653515` still serves API `59037f8`, not the follow-up below. |

## Follow-up implemented and verified

Follow-up code and documentation are local, uncommitted and unpushed in the core
and API task worktrees. No follow-up Worker deployment has occurred.

- **Complete staging gate:** separate candidate snapshot; missing/empty repos
  or failed/empty vendor fetches abort before upload/reindex. Previous staging
  survives; `.staging.previous` retains the backup if promotion and rollback
  both fail. Intentional repo-only staging and existing size limits remain.
- **Read-only verifier (removed with sync retire):** formerly `tools/docs-rag/verify_index.mjs` compared complete
  source/index inventories, unique keys, metadata flags and scope folders,
  status/errors, pending actions, chunks, sizes and last-seen timestamps. It
  refuses active/changing jobs, detects source changes and requires an uncached
  symbol content canary after inventory passes. It never requests a job.
- **Partial identifiers:** after empty AND and OR, code-like identifiers get at
  most 20 same-scope hybrid candidates without cache/reranking for that request.
  Only literal identifier-prefix matches within returned text survive; requested
  output limit is enforced. `matchMode: "symbol-fallback"` identifies this path.
  Global models/thresholds are unchanged; this is not exhaustive substring search.
- **Metadata repair:** temporary boolean schema field triggered a full rebuild;
  removing it triggered a second rebuild, ending 2026-09-22 at 04:05:51 UTC.
  Post-cleanup verification found all 126 indexed boolean flags, no vendor flags,
  and correct folders for all 197 keys. Source bytes were unchanged by repair.

## Evidence

- Core: 15 Python tests and 32 Node tests pass, including failed-refresh rollback,
  real R2 terminal-page pagination shape, and missing/wrong scope-folder cases.
- API: typecheck and 87 RAG tests pass. Local modified `searchDocs` against live
  Cloudflare returned three literal `RegisterExec` hits in `all`, and the EP ABI
  reference alone for `first-party`, limit 1. Both used `symbol-fallback`.
- Exact two-symbol query still returns exactly three first-party hits with `and`,
  led by the EP ABI reference and ADR-0002.
- Post-cleanup live verifier: 197 source objects, 197 unique indexed items, 126
  first-party flags, **25 outdated capacity-error items**. Exit 1 correctly
  reports incomplete index readiness; no other inventory issues were reported.
- Full API run: 136 main-suite tests plus 20 activation tests passed, but two
  baseline Better Auth unhandled rejections remain. Existing suppression permits
  exit 0; this is not a clean full-suite result.

## Remaining limits

- [ ] **Index readiness:** resolve 25 `workers_ai_out_of_capacity_error` items,
  then verify manually since the automated verifier was removed with the sync
  retire. Acceptance, from `api.trackdub`:
  `npx wrangler ai-search jobs list trackdub-docs` shows no active/changing
  job and no error items; `npx wrangler ai-search get trackdub-docs` confirms
  the 24h sync, trigram tokenizer and `is_first_party` existence boost; and
  the content canary below returns in scope. Do not use the earlier
  197-completed observation from before rebuilding as current. (Metadata,
  chunk, size and last-seen checks the retired verifier performed are now
  dashboard spot-checks on item status and metadata; the canary covers one
  document only and same-size content freshness is not proven, see
  Verification boundaries.)
  Content canary: call the hosted `search_trackdub_docs` with query
  `RegisterExecutionProviderLibrary` (the EP ABI pin doc
  `docs/reference/tensorrt-rtx-ep-abi-plugin.md` is the expected first-party
  hit) — a literal hit on that symbol in the returned text is the pass
  criterion, because plain keyword search misses long code identifiers while
  `symbol-fallback` recovers them. Check the dashboard item status/metadata
  only to confirm the hit is the current indexed copy, not an outdated one.
- **First-party preference:** uncached `runtime cache` still ranks an NVIDIA
  result first and Trackdub second after metadata recovery. Existence boost
  biases candidates before reranking; it is not a hard ordering guarantee.
- **First-message rewriting:** Cloudflare uses the first user message as-is.
  Stateless ask supplies no history and demonstrates no rewrite benefit.
  `searchQuery: null` truthfully reports omitted upstream `search_query`.
- **Verification boundaries:** same-size content freshness is not proven for
  every chunk, and the content canary covers one document. Cached staging from
  older script versions is not retroactively certified. Remote uploads are not
  transactional; already uploaded objects remain changed after partial failure.
- **Operational follow-up:** Linear access was unavailable; no issue was marked
  Done. When access is available, search team `TS` for a matching issue and update
  it, or create one if none exists; apply `repo:*`, `area:*`, and `agent-owned`
  labels, and keep the issue In Progress while the remaining acceptance work is
  active. No cloud security scan or cloud code review was performed.
Detailed commands and rebuild evidence: see `tools/docs-rag/SPEC.md` (hosted Worker ops; automated sync from Trackdub retired).
