"""TradeDocs AI REST API: evidence-grounded answers over versioned TradeFlow documentation."""
import json
import math
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .corpus import ROOT, load_corpus, versions_of
from .llm import CITATION
from .pipeline import answer_question
from .retrieval import MODES, default_retriever
from .storage import TelemetryStore

app = FastAPI(title="TradeDocs AI", version="0.3.0")
STORE = TelemetryStore()
RESULTS = ROOT / "data" / "evaluation" / "results.json"


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    filters: dict[str, str] = {}


class SearchRequest(ChatRequest):
    mode: str = "hybrid_rerank"
    limit: int = Field(default=5, ge=1, le=20)


class Feedback(BaseModel):
    request_id: str
    helpful: bool


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(load_corpus())}


@app.post("/chat")
def chat(req: ChatRequest):
    request_id = str(uuid.uuid4())
    result = answer_question(req.query, req.filters)
    cited = [int(n) for n in CITATION.findall(result.answer)]
    if not result.insufficient and (not cited or max(cited) > len(result.evidence)):
        raise HTTPException(502, "Citation validation failed")
    event = {
        "request_id": request_id, "query": req.query, "intent": result.intent, "rewritten_query": result.rewritten_query,
        "retrieval_ms": result.timings["retrieval_ms"], "generation_ms": result.timings["generation_ms"],
        "total_ms": result.timings["total_ms"], "retrieved_document_ids": [h.document.document_id for h in result.retrieved],
        "citations": [h.document.document_id for h in result.evidence], "insufficient_evidence": result.insufficient,
    }
    STORE.record(event)
    return {
        "request_id": request_id, "answer": result.answer, "intent": result.intent, "rewritten_query": result.rewritten_query,
        "evidence": [h.to_dict() for h in result.evidence], "retrieved": [h.to_dict() for h in result.retrieved],
        "insufficient_evidence": result.insufficient, "evidence_coverage": round(result.coverage, 3),
        "generator": result.generator, "metrics": event,
    }


@app.post("/search")
def search(req: SearchRequest):
    if req.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    hits, ms = default_retriever().search(req.query, mode=req.mode, limit=req.limit, filters=req.filters)
    return {"results": [h.to_dict() for h in hits], "retrieval_ms": ms}


@app.get("/documents")
def documents():
    return [d.to_dict() for d in load_corpus()]


@app.get("/documents/{document_id}")
def document(document_id: str):
    doc = next((d for d in load_corpus() if d.document_id == document_id), None)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc.to_dict() | {"versions": [v.document_id for v in versions_of(doc.family)]}


@app.post("/feedback")
def feedback(item: Feedback):
    STORE.feedback(item.request_id, item.helpful)
    return {"stored": True}


@app.get("/metrics")
def metrics():
    report = STORE.metrics()
    latencies = STORE.latencies()
    report.update({
        "p50_latency_ms": latencies[len(latencies) // 2] if latencies else 0,
        "p95_latency_ms": latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)] if latencies else 0,
    })
    return report


@app.get("/evaluation/results")
def evaluation_results():
    """Serves the last measured evaluation (python scripts/evaluate.py), never a hard-coded figure."""
    if not RESULTS.exists():
        raise HTTPException(404, "Run python scripts/evaluate.py first")
    return json.loads(RESULTS.read_text())


app.mount("/", StaticFiles(directory=str(Path(ROOT) / "frontend"), html=True), name="frontend")
