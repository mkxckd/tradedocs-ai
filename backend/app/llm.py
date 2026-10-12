"""Optional LLM generation over retrieved evidence (OpenAI-compatible or Anthropic Messages API).

The LLM is only ever given the selected evidence. Its output must be JSON, every citation must point at
supplied evidence, and any failure falls back to the deterministic extractive answer.
"""
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SYSTEM = ("You are a strictly evidence-grounded support assistant for TradeFlow. Answer only from the numbered "
          "evidence. Cite every factual sentence with [n]. If the evidence does not answer the question, set "
          "insufficient to true. Reply with JSON only: {\"answer\": string, \"insufficient\": boolean}.")
CITATION = re.compile(r"\[(\d+)\]")


class LLMError(RuntimeError):
    pass


def build_prompt(query: str, evidence: list) -> str:
    sources = "\n".join(f"[{i}] {h.document.title} (v{h.document.version}, {h.document.status}): {h.document.content}"
                        for i, h in enumerate(evidence, 1))
    return f"Evidence:\n{sources}\n\nQuestion: {query}"


def parse_and_validate(raw: str, evidence_count: int) -> dict:
    """Parses the model's JSON and rejects answers that cite missing evidence or cite nothing."""
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise LLMError("model did not return JSON")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError("model returned invalid JSON") from exc
    answer, insufficient = str(data.get("answer", "")).strip(), bool(data.get("insufficient", False))
    if insufficient:
        return {"answer": answer, "insufficient": True}
    cited = [int(c) for c in CITATION.findall(answer)]
    if not cited:
        raise LLMError("answer has no citations")
    if any(c < 1 or c > evidence_count for c in cited):
        raise LLMError("answer cites evidence that was not supplied")
    return {"answer": answer, "insufficient": False}


class _HTTPProvider:
    name = "base"
    default_model = ""

    def __init__(self, api_key: str, model: str | None = None, base_url: str | None = None, timeout: float = 20, retries: int = 1):
        self.api_key, self.model = api_key, model or self.default_model
        self.base_url, self.timeout, self.retries = base_url, timeout, retries

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        last = None
        for _ in range(self.retries + 1):
            try:
                request = Request(url, data=json.dumps(payload).encode(), headers=headers | {"Content-Type": "application/json"}, method="POST")
                with urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                last = exc
        raise LLMError(f"{self.name} request failed: {type(last).__name__}") from last

    def generate(self, query: str, evidence: list) -> dict:
        return parse_and_validate(self._complete(build_prompt(query, evidence)), len(evidence))


class OpenAICompatibleProvider(_HTTPProvider):
    name, default_model = "openai", "gpt-4.1-mini"

    def _complete(self, prompt: str) -> str:
        url = (self.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        payload = {"model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
        data = self._post(url, {"Authorization": f"Bearer {self.api_key}"}, payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected OpenAI response shape") from exc


class AnthropicProvider(_HTTPProvider):
    name, default_model = "anthropic", "claude-haiku-5-5"

    def _complete(self, prompt: str) -> str:
        url = (self.base_url or "https://api.anthropic.com/v1").rstrip("/") + "/messages"
        payload = {"model": self.model, "max_tokens": 600, "temperature": 0, "system": SYSTEM,
                   "messages": [{"role": "user", "content": prompt}]}
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        data = self._post(url, headers, payload)
        try:
            return "".join(block["text"] for block in data["content"] if block.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise LLMError("unexpected Anthropic response shape") from exc


def provider_from_env() -> _HTTPProvider | None:
    """LLM_PROVIDER=openai|anthropic with LLM_API_KEY; optional LLM_MODEL and LLM_BASE_URL."""
    key = os.getenv("LLM_API_KEY")
    if not key:
        return None
    cls = {"openai": OpenAICompatibleProvider, "anthropic": AnthropicProvider}.get(os.getenv("LLM_PROVIDER", "openai").lower())
    if cls is None:
        raise ValueError("LLM_PROVIDER must be 'openai' or 'anthropic'")
    return cls(key, os.getenv("LLM_MODEL"), os.getenv("LLM_BASE_URL"))
