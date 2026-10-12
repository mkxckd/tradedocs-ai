"""MCP server exposing TradeDocs retrieval and grounded answering as tools for AI agents.

Run over stdio (for Claude Desktop, IDE agents or any MCP client):

    python -m app.mcp_server            # from the backend/ directory

All tools are read-only: they never modify documents, telemetry or feedback.
"""
from mcp.server.mcpserver import MCPServer

from .corpus import load_corpus, versions_of
from .env import load_dotenv
from . import pipeline
from .retrieval import MODES, default_retriever

load_dotenv()

server = MCPServer(
    name="tradedocs",
    instructions=("Search and answer questions over versioned TradeFlow documentation. Prefer answer_question for "
                  "user questions; it cites evidence and refuses when the documentation does not cover the topic."),
)


@server.tool()
def search_documents(query: str, limit: int = 5, mode: str = "hybrid_rerank", category: str = "", version: str = "") -> list[dict]:
    """Rank TradeFlow documents for a query. Optional filters: category (api, trading, account, security, operations) and version."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    hits, _ = default_retriever().search(query, mode=mode, limit=max(1, min(limit, 20)), filters={"category": category, "version": version})
    return [{"document_id": h.document.document_id, "title": h.document.title, "version": h.document.version,
             "status": h.document.status, "score": round(h.score, 4)} for h in hits]


@server.tool()
def get_document(document_id: str) -> dict:
    """Return the full text and metadata of one document, including the IDs of its other versions."""
    doc = next((d for d in load_corpus() if d.document_id == document_id), None)
    if doc is None:
        raise ValueError(f"Unknown document_id: {document_id}")
    return doc.to_dict() | {"versions": [v.document_id for v in versions_of(doc.family)]}


@server.tool()
def answer_question(question: str) -> dict:
    """Answer a question using only TradeFlow documentation, with numbered citations, or report insufficient evidence."""
    result = pipeline.answer_question(question)
    return {"answer": result.answer, "insufficient_evidence": result.insufficient,
            "citations": [{"n": i, "document_id": h.document.document_id, "title": h.document.title, "version": h.document.version}
                          for i, h in enumerate(result.evidence, 1)],
            "evidence_coverage": round(result.coverage, 3), "generator": result.generator}


if __name__ == "__main__":
    server.run("stdio")
