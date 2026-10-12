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


def test_highlights_match_paraphrased_llm_answers():
    from app.grounding import highlight_sentences
    from app.retrieval import Hit, default_retriever
    doc = next(d for d in default_retriever().docs if d.document_id == "api-rate-v32")
    answer = "You can send 180 requests per minute to the REST Orders endpoint. [1]"
    assert highlight_sentences(answer, [Hit(doc, 1.0)]) == [[0]]
    assert highlight_sentences("Nothing relevant here. [2]", [Hit(doc, 1.0)]) == [[]]  # out-of-range citation ignored


def test_readable_terms_maps_stems_to_typed_words():
    from app.grounding import readable_terms
    from app.text import stem
    assert readable_terms("Can I stake my crypto rewards?", [stem("rewards"), "crypto"]) == ["crypto", "rewards"]


def test_dotenv_loader_respects_existing_env(tmp_path, monkeypatch):
    from app.env import load_dotenv
    env = tmp_path / ".env"
    env.write_text("# comment\nTD_TEST_A=one   # inline\nTD_TEST_B=\nTD_TEST_C='quoted'\n")
    monkeypatch.setenv("TD_TEST_C", "already set")
    monkeypatch.delenv("TD_TEST_A", raising=False)
    load_dotenv(env)
    import os
    assert os.environ["TD_TEST_A"] == "one"
    assert "TD_TEST_B" not in os.environ
    assert os.environ["TD_TEST_C"] == "already set"
    monkeypatch.delenv("TD_TEST_A")
