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
