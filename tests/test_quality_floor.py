"""Regression guard on the dev split, so CI fails if retrieval or refusal quality drops."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate  # noqa: E402


def test_dev_split_quality_does_not_regress():
    retrieval = evaluate.retrieval_metrics(evaluate.split("dev"), "hybrid_rerank")
    answers = evaluate.answer_metrics(evaluate.split("dev"))
    assert retrieval["recall_at_5"] >= 0.90
    assert answers["refusal_recall_on_unanswerable"] >= 0.90
    assert answers["answer_rate_on_answerable"] >= 0.75
    assert answers["citation_precision"] >= 0.75
