from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200 and response.json()["documents"] >= 50


def test_chat_returns_cited_answer_and_records_telemetry():
    data = client.post("/chat", json={"query": "Why am I getting HTTP 429 errors?"}).json()
    assert data["evidence"] and "[1]" in data["answer"] and not data["insufficient_evidence"]
    assert client.get("/metrics").json()["total_queries"] >= 1


def test_chat_refuses_out_of_scope():
    data = client.post("/chat", json={"query": "Is there a GraphQL API?"}).json()
    assert data["insufficient_evidence"] and data["evidence"] == []


def test_validation_and_errors():
    assert client.post("/chat", json={"query": ""}).status_code == 422
    assert client.get("/documents/not-real").status_code == 404
    assert client.post("/search", json={"query": "x", "mode": "magic"}).status_code == 422


def test_document_lists_its_versions():
    data = client.get("/documents/api-rate-v32").json()
    assert set(data["versions"]) == {"api-rate-v32", "api-rate-v21"}


def test_feedback():
    assert client.post("/feedback", json={"request_id": "r1", "helpful": True}).json()["stored"]


def test_chat_returns_highlighted_source_sentences():
    body = client.post("/chat", json={"query": "Why am I getting HTTP 429 errors?"}).json()
    first = body["evidence"][0]
    assert first["highlight"], "the cited document should have at least one highlighted sentence"
    quoted = [first["sentences"][i] for i in first["highlight"]]
    assert all(q in body["answer"] for q in quoted)  # extractive answers quote the highlighted sentences verbatim
    assert body["coverage_threshold"] > 0


def test_refusal_explains_missing_terms():
    body = client.post("/chat", json={"query": "Can I stake crypto on TradeFlow?"}).json()
    assert body["insufficient_evidence"]
    assert "crypto" in body["missing_terms"]
    assert body["evidence_coverage"] < body["coverage_threshold"]


def test_compare_versions():
    body = client.get("/documents/api-rate-v32/compare/api-rate-v21").json()
    assert body["older"]["version"] == "2.1" and body["newer"]["version"] == "3.2"
    added = " ".join(c["text"] for c in body["changes"] if c["op"] == "added")
    removed = " ".join(c["text"] for c in body["changes"] if c["op"] == "removed")
    assert "180 requests" in added and "120 requests" in removed
    # argument order doesn't matter; older is always the earlier date
    assert client.get("/documents/api-rate-v21/compare/api-rate-v32").json()["older"]["version"] == "2.1"


def test_compare_rejects_different_policies_and_unknown_ids():
    assert client.get("/documents/api-rate-v32/compare/withdrawals-v20").status_code == 422
    assert client.get("/documents/api-rate-v32/compare/nope").status_code == 404
