import numpy as np
import pytest

from vector_masterclass.applications import binary_metrics, neighbor_predict, route


def test_false_merge_metrics_and_no_positive_prediction():
    scores = binary_metrics([True, False, True, False], [True, True, False, False])
    assert scores == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert binary_metrics([True], [False])["precision"] == 0
    with pytest.raises(ValueError):
        binary_metrics([True], [False, True])


def test_route_ambiguity_abstains_and_never_authorizes():
    prototypes = np.array([[1, 0], [0, 1]], dtype=float)
    assert route(np.array([1, 1]), prototypes, ["read", "write"])["intent"] == "abstain"
    selected = route(np.array([1, 0]), prototypes, ["read", "write"])
    assert selected["intent"] == "read" and selected["authorized"] is False
    assert neighbor_predict(np.array([-1, -1]), prototypes, ["a", "b"], k=1) == "abstain"
