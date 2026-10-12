"""Hybrid retrieval: BM25 + LSA dense vectors, fused with reciprocal-rank fusion, then reranked.

The "dense" retriever uses latent semantic analysis (TF-IDF projected with truncated SVD). It is a
real dense vector space that runs offline with no model download; it is not a neural embedding model.
"""
import math
import re
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .corpus import Document, load_corpus
from .text import tokens

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
LEGACY_CUES = {"previous", "old", "older", "legacy", "earlier", "former", "v1", "v2", "v2.1", "v1.3",
               "compare", "comparison", "changed", "change", "history", "deprecated", "still", "used"}


@dataclass
class Hit:
    document: Document
    score: float

    def to_dict(self) -> dict:
        return self.document.to_dict() | {"score": round(self.score, 4), "passage": self.document.content}


VERSION_REF = re.compile(r"\bv?\d+\.\d+\b")


def wants_history(query: str) -> bool:
    """True when the user asks about an older or specific version, or how a policy changed."""
    q = query.lower()
    words = set(re.findall(r"[a-z0-9.]+", q))
    return bool(words & LEGACY_CUES) or bool(VERSION_REF.search(q))


class Retriever:
    def __init__(self, docs: tuple[Document, ...] | None = None, k1: float = 1.5, b: float = 0.75):
        self.docs = docs or load_corpus()
        self.k1, self.b = k1, b
        self.doc_tokens = [tokens(d.searchable_text) for d in self.docs]
        self.title_tokens = [set(tokens(d.title + " " + " ".join(d.tags))) for d in self.docs]
        self.doc_freq = Counter(t for toks in self.doc_tokens for t in set(toks))
        self.avg_len = sum(map(len, self.doc_tokens)) / len(self.doc_tokens)
        self.term_counts = [Counter(toks) for toks in self.doc_tokens]
        # LSA dense space
        self.vectorizer = TfidfVectorizer(analyzer=tokens, sublinear_tf=True)
        tfidf = self.vectorizer.fit_transform([d.searchable_text for d in self.docs])
        self.svd = TruncatedSVD(n_components=min(48, len(self.docs) - 1), random_state=0)
        self.doc_vectors = self._normalise(self.svd.fit_transform(tfidf))

    @staticmethod
    def _normalise(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)

    def idf(self, term: str) -> float:
        n = len(self.docs)
        return math.log((n - self.doc_freq.get(term, 0) + 0.5) / (self.doc_freq.get(term, 0) + 0.5) + 1)

    def _bm25(self, query: str) -> list[float]:
        q = tokens(query)
        scores = []
        for counts, toks in zip(self.term_counts, self.doc_tokens):
            norm = 1 - self.b + self.b * len(toks) / self.avg_len
            scores.append(sum(self.idf(t) * counts[t] * (self.k1 + 1) / (counts[t] + self.k1 * norm) for t in q if t in counts))
        return scores

    def _dense(self, query: str) -> list[float]:
        vector = self._normalise(self.svd.transform(self.vectorizer.transform([query])))
        return (self.doc_vectors @ vector[0]).tolist()

    def search(self, query: str, mode: str = "hybrid_rerank", limit: int = 5, filters: dict | None = None) -> tuple[list[Hit], float]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        started = time.perf_counter()
        filters = {k: v for k, v in (filters or {}).items() if v}
        allowed = [i for i, d in enumerate(self.docs) if all(str(getattr(d, k, "")) == str(v) for k, v in filters.items())]
        bm25, dense = self._bm25(query), self._dense(query)
        by_bm25 = sorted(allowed, key=lambda i: bm25[i], reverse=True)
        by_dense = sorted(allowed, key=lambda i: dense[i], reverse=True)
        if mode == "bm25":
            ranked = [(i, bm25[i]) for i in by_bm25 if bm25[i] > 0]
        elif mode == "dense":
            ranked = [(i, dense[i]) for i in by_dense]
        else:
            fused = Counter()
            for order in (by_bm25, by_dense):
                for rank, i in enumerate(order, 1):
                    fused[i] += 1 / (60 + rank)
            if mode == "hybrid_rerank":
                ranked = [(i, self._rerank(query, i, s)) for i, s in fused.items()]
            else:
                ranked = list(fused.items())
            ranked.sort(key=lambda x: x[1], reverse=True)
        hits = [Hit(self.docs[i], float(s)) for i, s in ranked[:limit]]
        return hits, round((time.perf_counter() - started) * 1000, 2)

    def _rerank(self, query: str, i: int, fused: float) -> float:
        """Second-stage signals: title/tag match, and demoting superseded versions unless history is asked for."""
        q = set(tokens(query))
        title_match = sum(self.idf(t) for t in q & self.title_tokens[i]) / (sum(self.idf(t) for t in q) or 1)
        score = fused * (1 + title_match)
        if self.docs[i].status == "superseded" and not wants_history(query):
            score *= 0.6
        return score


_DEFAULT: Retriever | None = None


def default_retriever() -> Retriever:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Retriever()
    return _DEFAULT
