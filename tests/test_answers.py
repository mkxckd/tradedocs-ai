import re

from app.pipeline import answer_question, rewrite


def citations(text):
    return [int(n) for n in re.findall(r"\[(\d+)\]", text)]


def test_answer_cites_only_supplied_evidence():
    result = answer_question("How often does the WebSocket send heartbeats?", use_llm=False)
    assert not result.insufficient
    assert "30 seconds" in result.answer
    assert citations(result.answer) and max(citations(result.answer)) <= len(result.evidence)


def test_out_of_scope_question_is_refused_without_evidence():
    result = answer_question("Does TradeFlow provide a cryptocurrency wallet?", use_llm=False)
    assert result.insufficient and not result.evidence


def test_version_comparison_cites_both_versions():
    result = answer_question("Compare the current Orders rate limit with the previous version.", use_llm=False)
    cited = {h.document.document_id for h in result.evidence}
    assert cited == {"api-rate-v32", "api-rate-v21"}
    assert "[1]" in result.answer and "[2]" in result.answer


def test_short_jargon_query_is_expanded_but_kept():
    assert rewrite("429").startswith("429 ")
    assert rewrite("How do I paginate?") == "How do I paginate?"
