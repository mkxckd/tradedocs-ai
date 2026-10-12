"""TradeDocs AI REST API: evidence-grounded answers over versioned TradeFlow documentation."""
import difflib
import json
import math
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import grounding
from .corpus import ROOT, load_corpus, versions_of
from .env import load_dotenv
from .llm import CITATION
from .pipeline import answer_question
from .retrieval import MODES, default_retriever
from .storage import TelemetryStore
from .text import sentences

load_dotenv()

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
    highlights = grounding.highlight_sentences(result.answer, result.evidence)
    evidence = [h.to_dict() | {"sentences": sentences(h.document.content), "highlight": marks}
                for h, marks in zip(result.evidence, highlights)]
    return {
        "request_id": request_id, "answer": result.answer, "intent": result.intent, "rewritten_query": result.rewritten_query,
        "evidence": evidence, "retrieved": [h.to_dict() for h in result.retrieved],
        "insufficient_evidence": result.insufficient, "evidence_coverage": round(result.coverage, 3),
        "coverage_threshold": grounding.COVERAGE_THRESHOLD, "missing_terms": result.missing_terms,
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


@app.get("/documents/{document_id}/compare/{other_id}")
def compare(document_id: str, other_id: str):
    """Sentence-level diff between two versions of the same policy: what was removed, added or kept."""
    docs = {d.document_id: d for d in load_corpus()}
    if document_id not in docs or other_id not in docs:
        raise HTTPException(404, "Document not found")
    old, new = sorted((docs[document_id], docs[other_id]), key=lambda d: d.date)
    if old.family != new.family:
        raise HTTPException(422, "Only versions of the same policy can be compared")
    a, b = sentences(old.content), sentences(new.content)
    changes = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            changes += [{"op": "same", "text": t} for t in a[i1:i2]]
        else:
            changes += [{"op": "removed", "text": t} for t in a[i1:i2]]
            changes += [{"op": "added", "text": t} for t in b[j1:j2]]
    return {"older": old.to_dict(), "newer": new.to_dict(), "changes": changes}


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
