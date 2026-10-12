"""End-to-end question answering: classify, rewrite, retrieve, ground, generate, validate."""
import time
from dataclasses import dataclass, field

from .grounding import REFUSAL, extractive_answer, ground
from .llm import LLMError, provider_from_env
from .retrieval import Hit, Retriever, default_retriever

INTENTS = [
    ("payments", ("withdraw", "deposit", "fee", "wire", "card")),
    ("security", ("login", "password", "secret", "2fa", "two-factor", "phishing", "allowlist", "session", "code")),
    ("incident investigation", ("incident", "outage", "latency", "disconnected", "2025")),
    ("trading", ("order", "market", "settle", "margin", "option", "split", "trade")),
    ("api troubleshooting", ("api", "429", "401", "403", "token", "webhook", "websocket", "rate", "page", "sdk")),
]
EXPANSIONS = {"429": "HTTP 429 rate limit exceeded Retry-After", "2fa": "two-factor authentication", "kyc": "identity verification"}


def classify(query: str) -> str:
    q = query.lower()
    return next((name for name, cues in INTENTS if any(c in q for c in cues)), "documentation lookup")


def rewrite(query: str) -> str:
    """Expands short or jargon-heavy queries; the original wording is always kept."""
    extra = [v for k, v in EXPANSIONS.items() if k in query.lower().split() or query.strip().lower() == k]
    return query if not extra else f"{query} {' '.join(extra)}"


@dataclass
class Answer:
    answer: str
    insufficient: bool
    evidence: list[Hit]
    retrieved: list[Hit]
    intent: str
    rewritten_query: str
    coverage: float
    generator: str
    timings: dict = field(default_factory=dict)


def answer_question(query: str, filters: dict | None = None, retriever: Retriever | None = None, use_llm: bool = True) -> Answer:
    retriever = retriever or default_retriever()
    started = time.perf_counter()
    rewritten = rewrite(query)
    hits, retrieval_ms = retriever.search(rewritten, filters=filters)
    grounding = ground(retriever, query, hits)
    timings = {"retrieval_ms": retrieval_ms, "generation_ms": 0.0}
    if grounding.insufficient:
        return Answer(REFUSAL, True, [], hits, classify(query), rewritten, grounding.coverage, "refusal", timings | {"total_ms": _ms(started)})
    text, generator = extractive_answer(retriever, query, grounding.evidence), "extractive"
    provider = provider_from_env() if use_llm else None
    if provider:
        generated = time.perf_counter()
        try:
            result = provider.generate(query, grounding.evidence)
            if result["insufficient"]:
                return Answer(REFUSAL, True, [], hits, classify(query), rewritten, grounding.coverage, provider.name, timings | {"total_ms": _ms(started)})
            text, generator = result["answer"], provider.name
        except LLMError:
            generator = "extractive (LLM fallback)"
        timings["generation_ms"] = _ms(generated)
    return Answer(text, False, grounding.evidence, hits, classify(query), rewritten, grounding.coverage, generator, timings | {"total_ms": _ms(started)})


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 2)
