import pytest

from app.corpus import load_corpus
from app.retrieval import MODES, default_retriever, wants_history
from app.text import stem


@pytest.mark.parametrize("a,b", [("retries", "retry"), ("keys", "key"), ("logged", "log"),
                                 ("authenticate", "authentication"), ("settled", "settle")])
def test_stemmer_groups_related_forms(a, b):
    assert stem(a) == stem(b)


def test_corpus_is_versioned_and_unique():
    docs = load_corpus()
    assert len(docs) >= 50
    assert {d.status for d in docs} == {"current", "superseded"}


def test_current_version_ranks_above_superseded_by_default():
    hits, _ = default_retriever().search("What is the Orders endpoint rate limit?")
    ids = [h.document.document_id for h in hits]
    assert ids[0] == "api-rate-v32"
    assert "api-rate-v21" not in ids[:1]


def test_history_questions_are_detected():
    assert wants_history("How did the policy change from the previous version?")
    assert wants_history("What did v2.1 allow?")
    assert not wants_history("What is the current limit?")


def test_metadata_filter_is_respected():
    hits, _ = default_retriever().search("rate limit", filters={"version": "2.1"})
    assert hits and all(h.document.version == "2.1" for h in hits)


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_ranked_results(mode):
    hits, ms = default_retriever().search("HTTP 429 retry", mode=mode)
    assert hits and ms >= 0
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        default_retriever().search("anything", mode="magic")
