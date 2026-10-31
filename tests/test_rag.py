import pytest

from vector_masterclass.rag import score_answer, search_queries
from vector_masterclass.schema import Query


def test_planner_output_is_bounded_and_invalid_output_is_not_retried():
    assert search_queries("1. Atlas E104\n2. Boreal E104\n3. Extra", limit=2) == [
        "Atlas E104",
        "Boreal E104",
    ]
    assert search_queries("x" * 301) == []
    assert search_queries("") == []
    with pytest.raises(ValueError):
        search_queries("query", limit=99)


def test_fact_and_citation_scores_are_separate():
    query = Query(id="d01", text="Atlas warranty?", split="development")
    result = {
        "text": "two years",
        "citation_audit": {"labels": [], "all_resolve": False},
        "citations": {},
        "input_tokens": 4,
        "output_tokens": 2,
        "seconds": 0.1,
        "finished_with_eos": True,
    }
    score = score_answer(query, result)
    assert score["required_fact_pattern_pass"]
    assert not score["citations_resolve"]
    assert not score["cited_gold_span"]
    assert not score["exact_abstention"]
