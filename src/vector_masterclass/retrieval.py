"""Transparent lexical, exhaustive, fusion, and diversity baselines."""

import re

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+(?:[-.]\w+)*", text.lower())


def normalized(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.isfinite(vectors).all() or np.any(norms == 0):
        raise ValueError("Vectors must be finite and nonzero")
    return vectors / norms


def exact_search(
    query: np.ndarray, vectors: np.ndarray, ids: list[str], k: int = 5
) -> list[tuple[str, float]]:
    if k <= 0 or len(ids) != len(vectors) or len(ids) != len(set(ids)):
        raise ValueError("Require positive k and unique IDs aligned with vectors")
    if not ids:
        return []
    q = normalized(query)
    matrix = normalized(vectors)
    if matrix.ndim != 2 or q.ndim != 1 or matrix.shape[1] != q.shape[0]:
        raise ValueError("Vector dimensions do not match")
    scores = matrix @ q
    # Stable explicit ID tie-break so reruns do not depend on ingestion order.
    order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))[:k]
    return [(ids[i], float(scores[i])) for i in order]


class LexicalIndex:
    def __init__(self, texts: list[str], ids: list[str]):
        if not texts or len(texts) != len(ids) or len(ids) != len(set(ids)):
            raise ValueError("Lexical index requires aligned nonempty texts and unique IDs")
        self.ids = ids
        self.index = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        scores = self.index.get_scores(tokenize(query))
        order = sorted(range(len(self.ids)), key=lambda i: (-float(scores[i]), self.ids[i]))[:k]
        return [(self.ids[i], float(scores[i])) for i in order]


def reciprocal_rank_fusion(
    rankings: list[list[str]], constant: int = 60
) -> list[tuple[str, float]]:
    """One-based ranks; duplicates within a retriever cannot earn repeated credit."""
    if constant < 0:
        raise ValueError("Fusion constant must be nonnegative")
    scores = {}
    for ranked in rankings:
        for rank, id_ in enumerate(dict.fromkeys(ranked), 1):
            scores[id_] = scores.get(id_, 0) + 1 / (constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def mmr(query: np.ndarray, vectors: np.ndarray, k: int, weight: float = 0.7) -> list[int]:
    if not 0 <= weight <= 1 or k <= 0:
        raise ValueError("Require positive k and weight in [0, 1]")
    if len(vectors) == 0:
        return []
    matrix, query = normalized(vectors), normalized(query)
    relevance = matrix @ query
    similarity = matrix @ matrix.T
    selected, remaining = [], set(range(len(vectors)))
    while remaining and len(selected) < k:
        scores = {
            i: weight * relevance[i]
            - (1 - weight) * max((similarity[i, j] for j in selected), default=0)
            for i in remaining
        }
        best = min(remaining, key=lambda i: (-float(scores[i]), i))
        selected.append(best)
        remaining.remove(best)
    return selected
