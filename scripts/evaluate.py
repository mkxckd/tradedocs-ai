"""Reproducible evaluation of retrieval, refusal and answer quality.

    python scripts/evaluate.py            # evaluate dev and test splits, write data/evaluation/results.json
    python scripts/evaluate.py --tune     # choose the refusal threshold using the dev split only

The test split is never used for tuning. All numbers in the README come from results.json.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import grounding  # noqa: E402
from app.pipeline import answer_question, rewrite  # noqa: E402
from app.retrieval import MODES, default_retriever  # noqa: E402

CASES = json.loads((ROOT / "data" / "evaluation" / "cases.json").read_text())
RESULTS = ROOT / "data" / "evaluation" / "results.json"


def split(name: str) -> list[dict]:
    return [c for c in CASES if name == "all" or c["split"] == name]


def mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def retrieval_metrics(cases: list[dict], mode: str) -> dict:
    retriever, ranks = default_retriever(), []
    for case in cases:
        if case["behavior"] != "answer":
            continue
        hits, _ = retriever.search(rewrite(case["question"]), mode=mode)
        ids = [h.document.document_id for h in hits]
        ranks.append(next((i + 1 for i, d in enumerate(ids) if d in case["expected_documents"]), None))
    return {"recall_at_1": mean([r == 1 for r in ranks]), "recall_at_5": mean([r is not None for r in ranks]),
            "mrr": mean([1 / r if r else 0 for r in ranks]), "cases": len(ranks)}


def answer_metrics(cases: list[dict]) -> dict:
    answerable, unanswerable = [], []
    precision, cited_expected, facts, failures = [], [], [], []
    for case in cases:
        result = answer_question(case["question"], use_llm=False)
        if case["behavior"] == "insufficient_evidence":
            unanswerable.append(result.insufficient)
            if not result.insufficient:
                failures.append({"id": case["id"], "question": case["question"], "problem": "answered an unanswerable question",
                                 "cited": [h.document.document_id for h in result.evidence]})
            continue
        answerable.append(not result.insufficient)
        text = result.answer.lower()
        fact_score = sum(f.lower() in text for f in case["expected_facts"]) / len(case["expected_facts"]) if not result.insufficient else 0.0
        facts.append(fact_score)
        if result.insufficient:
            failures.append({"id": case["id"], "question": case["question"], "problem": "refused an answerable question", "coverage": round(result.coverage, 3)})
            continue
        cited = {h.document.document_id for h in result.evidence}
        expected = set(case["expected_documents"])
        precision.append(len(cited & expected) / len(cited))
        cited_expected.append(bool(cited & expected))
        if fact_score < 1:
            failures.append({"id": case["id"], "question": case["question"], "problem": "missing expected fact", "answer": result.answer})
    return {
        "answerable_cases": len(answerable), "unanswerable_cases": len(unanswerable),
        "answer_rate_on_answerable": mean(answerable),
        "refusal_recall_on_unanswerable": mean(unanswerable),
        "citation_precision": mean(precision),
        "answers_citing_an_expected_document": mean(cited_expected),
        "expected_fact_recall": mean(facts),
        "failures": failures,
    }


def tune() -> None:
    dev, best = split("dev"), None
    print("threshold  answer_rate  refusal_recall  balanced")
    for t in [round(0.30 + 0.05 * i, 2) for i in range(13)]:
        grounding.COVERAGE_THRESHOLD = t
        m = answer_metrics(dev)
        balanced = (m["answer_rate_on_answerable"] + m["refusal_recall_on_unanswerable"]) / 2
        print(f"{t:9.2f}  {m['answer_rate_on_answerable']:11.3f}  {m['refusal_recall_on_unanswerable']:14.3f}  {balanced:8.3f}")
        if best is None or balanced > best[1]:
            best = (t, balanced)
    print(f"\nBest dev threshold: {best[0]} (balanced accuracy {best[1]:.3f}). Set COVERAGE_THRESHOLD in backend/app/grounding.py.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    if args.tune:
        return tune()
    report = {"generated": date.today().isoformat(), "generator": "extractive (no LLM)",
              "coverage_threshold": grounding.COVERAGE_THRESHOLD, "splits": {}}
    for name in ("dev", "test"):
        cases = split(name)
        report["splits"][name] = {"retrieval": {m: retrieval_metrics(cases, m) for m in MODES}, "answers": answer_metrics(cases)}
    RESULTS.write_text(json.dumps(report, indent=2))
    for name, data in report["splits"].items():
        a = data["answers"]
        r = data["retrieval"]["hybrid_rerank"]
        print(f"[{name}] hybrid+rerank Recall@5 {r['recall_at_5']:.3f} MRR {r['mrr']:.3f} | answered {a['answer_rate_on_answerable']:.3f} "
              f"| refused-unanswerable {a['refusal_recall_on_unanswerable']:.3f} | citation precision {a['citation_precision']:.3f} "
              f"| fact recall {a['expected_fact_recall']:.3f} | failures {len(a['failures'])}")
        for mode in MODES:
            print(f"    {mode:14s} R@1 {data['retrieval'][mode]['recall_at_1']:.3f}  R@5 {data['retrieval'][mode]['recall_at_5']:.3f}  MRR {data['retrieval'][mode]['mrr']:.3f}")


if __name__ == "__main__":
    main()
