"""Evidence selection, evidence-coverage refusal and the deterministic extractive answer generator."""
import math
import re
from dataclasses import dataclass

from .corpus import versions_of
from .retrieval import Hit, Retriever, wants_history
from .text import content_terms, sentences, stem, tokens

# Selected on the *dev* split only (scripts/evaluate.py --tune); the test split is never used for tuning.
COVERAGE_THRESHOLD = 0.55
EVIDENCE_RELATIVE_SCORE = 0.75
MAX_EVIDENCE = 3

REFUSAL = "I couldn't find reliable information about that in the available TradeFlow documentation."


@dataclass
class Grounding:
    evidence: list[Hit]
    coverage: float
    insufficient: bool
    missing_terms: list[str]


def _weight(retriever: Retriever, term: str) -> float:
    # Terms never seen in the corpus get the maximum weight: they are exactly what the corpus cannot answer.
    n = len(retriever.docs)
    return retriever.idf(term) if term in retriever.doc_freq else math.log(n + 1) + 1


# Words that frame a question about versions rather than describe its topic; version handling covers them.
FRAMING_TERMS = {stem(w) for w in ("compare", "comparison", "current", "currently", "previous", "old", "older", "latest",
                                    "new", "newer", "version", "versions", "legacy", "earlier", "changed", "change")}


def coverage(retriever: Retriever, query: str, hit: Hit) -> tuple[float, list[str]]:
    """IDF-weighted share of the query's topic terms that appear in a single evidence document."""
    terms = [t for t in content_terms(query) if t not in FRAMING_TERMS]
    if not terms:
        return 0.0, []
    present = set(tokens(hit.document.searchable_text))
    total = sum(_weight(retriever, t) for t in terms)
    found = sum(_weight(retriever, t) for t in terms if t in present)
    return found / total, [t for t in terms if t not in present]


def ground(retriever: Retriever, query: str, hits: list[Hit], threshold: float | None = None) -> Grounding:
    threshold = COVERAGE_THRESHOLD if threshold is None else threshold
    if not hits:
        return Grounding([], 0.0, True, content_terms(query))
    scored = [(h, *coverage(retriever, query, h)) for h in hits[:MAX_EVIDENCE]]
    best_cov = max(c for _, c, _ in scored)
    missing = min(scored, key=lambda x: len(x[2]))[2]
    if best_cov < threshold:
        return Grounding([], best_cov, True, missing)
    top = hits[0].score
    evidence = [h for h, c, _ in scored if h is hits[0] or (h.score >= EVIDENCE_RELATIVE_SCORE * top and c >= threshold)]
    # Version awareness: for history/comparison questions make sure every version of the policy is cited.
    if wants_history(query):
        # Version questions are answered from every version of the top document's policy, newest first.
        family = evidence[0].document.family
        scores = {h.document.document_id: h.score for h in hits}
        evidence = [Hit(doc, scores.get(doc.document_id, 0.0)) for doc in versions_of(family, retriever.docs)]
    return Grounding(evidence[:MAX_EVIDENCE], best_cov, False, missing)


SUPERSESSION = re.compile(r"\bsupersede[sd]?\b|\breplaced by\b", re.I)
QUANTITY_QUESTION = re.compile(r"\b(how (long|many|much|often|soon)|what is the (maximum|minimum|limit|default)|when)\b", re.I)
HAS_QUANTITY = re.compile(r"\d|\b(one|two|three|five|seven|ten|six|hour|day|week|month|year|sunday|monday)s?\b", re.I)


def extractive_answer(retriever: Retriever, query: str, evidence: list[Hit], max_sentences: int = 3) -> str:
    """Builds an answer only from sentences in the evidence, each followed by its citation."""
    terms = set(content_terms(query)) - FRAMING_TERMS
    # Answer-type hint: quantity/time questions prefer sentences that actually state a number or period.
    wants_quantity = bool(QUANTITY_QUESTION.search(query))
    picked = []
    for n, hit in enumerate(evidence, 1):
        ranked = sorted(
            ((_sentence_score(retriever, terms, s, wants_quantity), idx, s) for idx, s in enumerate(sentences(hit.document.content))),
            key=lambda x: (-x[0], x[1]),
        )
        # Supersession notes ("This version supersedes ...") repeat metadata; prefer sentences with substance.
        substantive = [r for r in ranked if r[0] > 0 and not SUPERSESSION.search(r[2])]
        ranked = substantive or ranked
        if not ranked or ranked[0][0] == 0:
            continue
        best = [ranked[0]]
        if len(ranked) > 1 and ranked[1][0] >= 0.75 * ranked[0][0] and n == 1:
            best.append(ranked[1])
        for score, idx, sentence in sorted(best, key=lambda x: x[1]):
            picked.append(f"{sentence} [{n}]")
    return " ".join(picked[:max_sentences]) if picked else REFUSAL


def _sentence_score(retriever: Retriever, terms: set[str], sentence: str, wants_quantity: bool) -> float:
    score = sum(_weight(retriever, t) for t in terms & set(tokens(sentence)))
    if score and wants_quantity and HAS_QUANTITY.search(sentence):
        score *= 1.5
    return score
