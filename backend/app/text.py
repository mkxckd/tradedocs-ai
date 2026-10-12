"""Tokenisation, light stemming and stopwords shared by retrieval and grounding."""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN = re.compile(r"[a-z0-9]+(?:[._+\-][a-z0-9]+)*")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# sklearn's list contains some words that carry meaning in a trading/support domain; keep those searchable.
_DOMAIN_WORDS = {"call", "bill", "fill", "system", "detail", "front", "move", "show", "find", "full", "empty", "part",
                 "side", "top", "bottom", "interest", "amount", "name", "found", "first", "last", "next", "fire", "mill"}
STOPWORDS = (set(ENGLISH_STOP_WORDS) - _DOMAIN_WORDS) | {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "could", "did", "do", "does", "for",
    "from", "get", "getting", "has", "have", "how", "i", "if", "in", "is", "it", "its", "long", "many",
    "me", "much", "my", "need", "of", "on", "or", "our", "should", "so", "still", "than", "that", "the",
    "their", "then", "there", "this", "to", "use", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your", "tradeflow", "about", "after", "before", "any",
    "ever", "there", "straight", "away", "into", "up", "out", "one", "happens", "happen", "give", "offer",
    "provide", "support", "supported", "allow", "allowed", "is", "am", "s",
}

_DERIVATIONAL = ("ations", "ation", "ments", "ment", "ate", "ness", "ly")


def stem(word: str) -> str:
    """Small rule-based stemmer (plural, -ed/-ing, -ies, a few derivational suffixes).

    Keeps related forms together, e.g. retry/retries/retried, key/keys, log/logged, authenticate/authentication.
    """
    if not word.isalpha() or len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("ied") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    for suffix in ("ating", "ated", "ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsz":
                word = word[:-1]
            break
    for suffix in _DERIVATIONAL:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word[:-1] if len(word) > 4 and word.endswith("e") else word


COMPOUND = re.compile(r"[._+\-]")


def tokens(text: str) -> list[str]:
    """Stemmed tokens; compounds such as next_cursor or x-request-id also yield their parts."""
    out = []
    for raw in TOKEN.findall(text.lower()):
        out.append(stem(raw))
        parts = [p for p in COMPOUND.split(raw) if p and not p.isdigit()]
        if len(parts) > 1:
            out.extend(stem(p) for p in parts)
    return out


def content_terms(text: str) -> list[str]:
    """Stemmed tokens with stopwords removed, order preserved, duplicates removed."""
    seen, out = set(), []
    for raw in TOKEN.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        term = stem(raw)
        if term not in seen:
            seen.add(term)
            out.append(term)
    return out


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE.split(text) if s.strip()]
