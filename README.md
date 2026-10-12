# TradeDocs AI

An evidence-grounded question-answering assistant over fictional, versioned documentation for a trading platform called TradeFlow. It answers only from retrieved documents, cites every sentence, prefers the current version of a policy while still answering "what changed" questions, and refuses when the documentation does not cover the question.

Available as a REST API with a web UI, and as an **MCP server**, so AI agents can use it as a tool.

## How it works

```text
question → intent + query rewrite → BM25 + LSA dense retrieval → reciprocal-rank fusion → rerank
         → evidence coverage check (refuse if the topic isn't covered) → version-aware evidence selection
         → answer generation (extractive, or OpenAI / Claude) → citation validation → SQLite telemetry
```

| Stage | Implementation |
|---|---|
| Corpus | 52 fictional documents in 49 policy families (`data/corpus.json`). Superseded versions (rate limits, authentication, withdrawals) are kept alongside current ones. |
| Retrieval | BM25 and a dense retriever using **latent semantic analysis** (TF-IDF projected with truncated SVD, scikit-learn), fused with reciprocal-rank fusion. A rerank stage adds title/tag match and demotes superseded versions unless the question asks about history. The dense vectors are LSA, not a neural embedding model, so everything runs offline. |
| Refusal | IDF-weighted **evidence coverage**: the share of the question's topic terms that appear in the best evidence document. Terms the corpus has never seen weigh the most. Below a threshold (0.55, chosen on the dev split only) the assistant refuses. |
| Version awareness | Questions about previous versions or changes cite every version of the policy, newest first. |
| Generation | Default: a deterministic **extractive** generator that only quotes evidence sentences, each with its citation. Optional: an **OpenAI-compatible** or **Anthropic Claude** model (`LLM_PROVIDER`, `LLM_API_KEY`). LLM output must be JSON, every `[n]` citation must point at supplied evidence, and any invalid output falls back to the extractive answer. |
| Telemetry | Each request (intent, rewritten query, timings, retrieved and cited IDs, refusal) and user feedback are stored in SQLite and summarised at `/metrics` and `/dashboard.html`. |

## Evaluation

`python scripts/evaluate.py` runs 85 labelled questions (65 answerable, 20 deliberately unanswerable) split into a **dev** half, used for every tuning decision, and a held-out **test** half, used only for reporting. Results are written to `data/evaluation/results.json`; the numbers below are copied from it and measured with the extractive generator (no LLM).

**Held-out test split** (32 answerable, 10 unanswerable):

| Metric | Result |
|---|---|
| Retrieval Recall@5 (hybrid + rerank) | **96.9%** |
| Retrieval MRR (hybrid + rerank) | **0.901** |
| Unanswerable questions correctly refused | **100%** (10/10) |
| Answerable questions answered (not wrongly refused) | **87.5%** |
| Citation precision (cited documents that are the expected ones) | **83.9%** |
| Answers citing at least one expected document | **96.4%** |
| Expected key facts present in the answer | **68.8%** |

Retrieval ablation on the test split:

| Mode | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| Dense (LSA) | 0.938 | 0.969 | 0.945 |
| BM25 | 0.875 | 0.969 | 0.917 |
| Hybrid (RRF) | 0.875 | 0.969 | 0.917 |
| Hybrid + rerank | 0.844 | 0.969 | 0.901 |

What the numbers say:
- Refusal works on this set. The cost is that some paraphrased answerable questions are refused, for example "leaked" when the document says "exposed".
- On the held-out split the reranker did **not** improve ranking; dense-only had the best MRR. Hybrid + rerank was marginally best on dev, which is why it is the default, but the evidence for it is weak.
- The extractive generator sometimes quotes a relevant but less useful sentence, which is why fact recall is the lowest metric. An LLM generator should do better; it is supported but not part of these measured numbers.

Limitations: the corpus and questions were written by the author, so the set is small and not independent. These results show the system behaves as designed; they are not evidence of production quality.

## Run it

```bash
pip install -r requirements.txt
uvicorn app.main:app --app-dir backend --reload      # UI at http://localhost:8000, dashboard at /dashboard.html
pytest -q                                            # unit, API, LLM-adapter, MCP and quality-floor tests
python scripts/evaluate.py                           # full evaluation → data/evaluation/results.json
python scripts/evaluate.py --tune                    # refusal threshold sweep on the dev split only
docker compose up --build
```

Optional LLM generation: copy `.env.example` to `.env` and set `LLM_PROVIDER` (`openai` or `anthropic`) and `LLM_API_KEY`.

### API

`GET /health`, `POST /chat`, `POST /search` (modes `dense`, `bm25`, `hybrid`, `hybrid_rerank`), `GET /documents`, `GET /documents/{id}` (includes its other versions), `POST /feedback`, `GET /metrics`, `GET /evaluation/results`.

### MCP server

```bash
cd backend && python -m app.mcp_server
```

Read-only tools: `search_documents`, `get_document`, and `answer_question` (returns the answer, citations and a refusal flag). Example client configuration:

```json
{"mcpServers": {"tradedocs": {"command": "python", "args": ["-m", "app.mcp_server"], "cwd": "/path/to/tradedocs-ai/backend"}}}
```

CI (GitHub Actions) runs the test suite, which includes a dev-split quality floor, and re-runs the full evaluation on every push and pull request.
