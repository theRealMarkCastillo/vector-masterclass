import numpy as np
import pytest

from vector_masterclass.retrieval import LexicalIndex, exact_search, mmr, reciprocal_rank_fusion


def test_exact_normalizes_and_resolves_ties_by_id():
    results = exact_search(np.array([1, 0]), np.array([[2, 0], [4, 0], [0, 1]]), ["b", "a", "c"], 3)
    assert [x[0] for x in results] == ["a", "b", "c"]
    with pytest.raises(ValueError):
        exact_search(np.array([0, 0]), np.array([[1, 0]]), ["a"])


def test_fusion_deduplicates_each_list():
    result = dict(reciprocal_rank_fusion([["a", "a", "b"], ["b", "a"]], constant=0))
    assert result == {"a": 1.5, "b": 1.5}


def test_identifier_tokenization_and_lexical_retrieval():
    index = LexicalIndex(
        [
            "Atlas E104 cable disconnected",
            "Boreal E104 fan stopped",
            "Atlas E140 signature invalid",
        ],
        ["a", "b", "c"],
    )
    assert index.search("E140 signature")[0][0] == "c"


def test_mmr_diversity_extremes():
    vectors = np.array([[1, 0], [0.999, 0.001], [0, 1]])
    assert mmr(np.array([1, 0]), vectors, 2, weight=1) == [0, 1]
    assert mmr(np.array([1, 0]), vectors, 2, weight=0.1) == [0, 2]
