import numpy as np
import pytest

from vector_masterclass.late_interaction import maxsim


def test_maxsim_sum_max_not_max_sum_or_average():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    d = np.array([[1.0, 0.0], [0.6, 0.8]])
    assert maxsim(q, d) == pytest.approx(1.8)
    assert maxsim(q, np.array([[1.0, 0.0]])) == pytest.approx(1)
    assert maxsim(q, np.array([[-1.0, 0.0], [0.0, -1.0]])) == pytest.approx(0)
    assert maxsim(q[::-1], d) == pytest.approx(maxsim(q, d))


@pytest.mark.parametrize(
    "q,d",
    [
        (np.zeros((0, 2)), np.ones((1, 2))),
        (np.ones((1, 3)), np.ones((1, 2))),
        (np.array([[np.nan, 0]]), np.ones((1, 2))),
    ],
)
def test_maxsim_rejects_invalid_matrices(q, d):
    with pytest.raises(ValueError):
        maxsim(q, d)
