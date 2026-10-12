"""Loads the fictional, versioned TradeFlow documentation corpus from data/corpus.json."""
import json
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = ROOT / "data" / "corpus.json"


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    category: str
    version: str
    date: str
    status: str          # "current" or "superseded"
    family: str          # documents describing the same policy across versions share a family
    tags: tuple
    source_type: str     # "documentation" or "incident"
    content: str

    @property
    def searchable_text(self) -> str:
        return f"{self.title} {' '.join(self.tags)} {self.content}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["tags"] = list(self.tags)
        return data


@lru_cache(maxsize=1)
def load_corpus(path: str = str(CORPUS_PATH)) -> tuple[Document, ...]:
    rows = json.loads(Path(path).read_text())
    docs = tuple(Document(**{**row, "tags": tuple(row["tags"])}) for row in rows)
    ids = [d.document_id for d in docs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document_id in corpus")
    return docs


def versions_of(family: str, docs: tuple[Document, ...] | None = None) -> list[Document]:
    docs = docs or load_corpus()
    return sorted((d for d in docs if d.family == family), key=lambda d: d.date, reverse=True)
