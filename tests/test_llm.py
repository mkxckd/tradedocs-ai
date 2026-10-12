import io
import json

import pytest

from app import llm
from app.pipeline import answer_question


def test_validation_rejects_citations_to_missing_evidence():
    with pytest.raises(llm.LLMError):
        llm.parse_and_validate('{"answer": "Limit is 180. [3]", "insufficient": false}', evidence_count=2)


def test_validation_rejects_uncited_answers_and_non_json():
    with pytest.raises(llm.LLMError):
        llm.parse_and_validate('{"answer": "Limit is 180.", "insufficient": false}', 2)
    with pytest.raises(llm.LLMError):
        llm.parse_and_validate("The limit is 180 [1]", 2)


def test_validation_accepts_model_refusal():
    assert llm.parse_and_validate('{"answer": "", "insufficient": true}', 2)["insufficient"]


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def fake_urlopen(body: dict):
    return lambda request, timeout: FakeResponse(json.dumps(body).encode())


def test_openai_answer_is_used_when_valid(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test")
    content = json.dumps({"answer": "Access tokens expire after 60 minutes. [1]", "insufficient": False})
    monkeypatch.setattr(llm, "urlopen", fake_urlopen({"choices": [{"message": {"content": content}}]}))
    result = answer_question("How long do access tokens last?")
    assert result.generator == "openai" and "60 minutes" in result.answer


def test_anthropic_response_shape_is_parsed(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    content = json.dumps({"answer": "Heartbeats are sent every 30 seconds. [1]", "insufficient": False})
    monkeypatch.setattr(llm, "urlopen", fake_urlopen({"content": [{"type": "text", "text": content}]}))
    result = answer_question("How often does the WebSocket send heartbeats?")
    assert result.generator == "anthropic" and "30 seconds" in result.answer


def test_invalid_llm_output_falls_back_to_extractive_answer(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test")
    bad = json.dumps({"answer": "Made-up fact without a citation.", "insufficient": False})
    monkeypatch.setattr(llm, "urlopen", fake_urlopen({"choices": [{"message": {"content": bad}}]}))
    result = answer_question("How often does the WebSocket send heartbeats?")
    assert result.generator == "extractive (LLM fallback)"
    assert "30 seconds" in result.answer
