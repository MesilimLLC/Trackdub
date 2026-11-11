// Score trackdub-docs-rag retrieval against tools/docs-eval/questions.json via the Worker's POST /v1/search.
//   DOCS_RAG_TOKEN=... node tools/docs-eval/eval_rag.mjs [--min-hit3 0.8] [--limit 8]
// DOCS_RAG_URL overrides the Worker origin. Exits 0 without scoring when no token is set.
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: {
    "min-hit3": { type: "string", default: "0" },
    limit: { type: "string", default: "8" },
    json: { type: "boolean", default: false },
  },
});

const token = process.env.DOCS_RAG_TOKEN;
if (!token) {
  console.log("DOCS_RAG_TOKEN not set; skipping docs-rag eval");
  process.exit(0);
}
const origin = (process.env.DOCS_RAG_URL ?? "https://trackdub-docs-rag.trackdub.workers.dev").replace(/\/$/, "");
const questionsPath = path.join(path.dirname(fileURLToPath(import.meta.url)), "questions.json");
const { questions } = JSON.parse(await readFile(questionsPath, "utf8"));

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function search(query, limit) {
  for (let attempt = 0; attempt < 4; attempt++) {
    const response = await fetch(`${origin}/v1/search`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ query, limit }),
    });
    if (response.status === 429) {
      await wait(11_000);
      continue;
    }
    if (!response.ok) throw new Error(`search ${response.status}: ${(await response.text()).slice(0, 200)}`);
    return (await response.json()).hits ?? [];
  }
  throw new Error("rate limited after retries");
}

const rows = [];
for (const item of questions) {
  if (!item.rag_expect?.length) continue;
  const hits = await search(item.q, Number(values.limit));
  const keys = [...new Set(hits.map((hit) => hit.key).filter(Boolean))];
  const rank = keys.findIndex((key) => item.rag_expect.some((expected) => key.includes(expected))) + 1 || null;
  rows.push({ q: item.q, rank, top: keys.slice(0, 3) });
  await wait(700);
}

const hitAt = (k) => rows.filter((row) => row.rank !== null && row.rank <= k).length / rows.length;
const mrr = rows.reduce((sum, row) => sum + (row.rank ? 1 / row.rank : 0), 0) / rows.length;
const summary = { total: rows.length, hit1: hitAt(1), hit3: hitAt(3), mrr };
if (values.json) {
  console.log(JSON.stringify({ ...summary, rows }, null, 2));
} else {
  for (const row of rows) {
    console.log(`${row.rank && row.rank <= 3 ? "ok  " : "MISS"} rank=${row.rank} ${row.q.slice(0, 70)}  top=${JSON.stringify(row.top)}`);
  }
  console.log(`n=${summary.total} hit@1=${summary.hit1.toFixed(2)} hit@3=${summary.hit3.toFixed(2)} mrr=${mrr.toFixed(3)}`);
}
process.exit(summary.hit3 >= Number(values["min-hit3"]) ? 0 : 1);
