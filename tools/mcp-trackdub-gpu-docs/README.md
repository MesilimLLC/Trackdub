# Local FastMCP for Trackdub GPU / TensorRT-RTX agent docs (v0).
#
# Complements NVIDIA CUDA MCP. Pin: runtime/trt-rtx-ep.manifest.json 0.5.0 (win + linux, TRT-RTX 1.7.1; Linux via LinML.Catalog 0.6.0), cu13.

## Layout

```text
tools/mcp-trackdub-gpu-docs/
  corpus/manifest.v0.json     # pin + local repo-file sources (EP ABI 0.5.0, win + linux)
                              # remote pages come from tools/docs-rag/corpus.v1.json ("gpu" blocks)
  src/trackdub_gpu_docs_mcp/  # FastMCP server + keyword search
  .data/chunks/               # ingested remote pages (gitignored)
  pyproject.toml
  README.md
```

## Tools

| Tool | Purpose |
|------|---------|
| `list_corpus` | Allowlist + live pin check vs `runtime/trt-rtx-ep.manifest.json` |
| `search_trackdub_gpu_docs` | Keyword search (Trackdub local always; remotes after ingest) |
| `get_doc` | Full text for one source id |

Policy precedence: Trackdub local > ORT plugin EP docs > NVIDIA TRT-RTX `/latest/` > WinML catalog docs.

## Setup

```powershell
cd tools/mcp-trackdub-gpu-docs
uv sync
uv run trackdub-gpu-docs-ingest   # optional; fetches allowlisted remote URLs (--force re-fetches cached pages)
```

## Search

BM25 over 1,800-character chunks, with identifier-aware terms (`RegisterExecutionProviderLibrary` also matches `register`, `execution`, `provider`, `library`), a title boost, a bonus for verbatim symbols, and at most three chunks per document. `excerpt` is the 800-character window with the densest matches, not the start of the chunk, and `matched` lists the query terms found. The index is cached per file mtime.

## Tests and eval

```powershell
uv run python -m unittest discover -s tests       # includes a retrieval-quality floor
uv run python ../docs-eval/eval_gpu.py            # hit@1 / hit@3 / MRR per question
```

Repo-local sources work with zero ingest (ADR-0002, EP ABI plugin doc, manifest).

## Cursor `mcp.json` entry

```json
"trackdub-gpu-docs": {
  "command": "uv",
  "args": [
    "--directory",
    "D:/Dev/Trackdub_Workspace/Trackdub/tools/mcp-trackdub-gpu-docs",
    "run",
    "trackdub-gpu-docs-mcp"
  ]
}
```

Adjust `--directory` to your checkout path.

## Pin notes

- EP ABI **0.5.0** (win-x64, and linux-x64 via LinML.Catalog), **cu13**, from `NVIDIA/TensorRT-RTX-EP-ABI`
- Windows bundle libs include `tensorrt_rtx_1_7.dll` (TRT-RTX 1.7.1 lineage); CUDA runtime statically linked
- NVIDIA `/latest/` docs may describe newer product releases; do not treat them as the shipped pin

## Not in v0

- Embeddings / Vectorize / Cloudflare hosting
- Full `docs.nvidia.com` crawl
