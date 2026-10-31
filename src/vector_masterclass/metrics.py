"""Retrieval metrics with explicit document, no-answer, and evidence-coverage semantics."""

import math
from collections.abc import Iterable

from .schema import Chunk, EvidenceUnit


def unique_ranked(ids: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def ranking_metrics(ranked: list[str], relevance: dict[str, int], k: int) -> dict:
    """Collapse duplicates before top-k; unjudged results score zero and are counted."""
    if k <= 0:
        raise ValueError("k must be positive")
    if any(g < 0 for g in relevance.values()):
        raise ValueError("Relevance grades cannot be negative")
    selected = unique_ranked(ranked)[:k]
    positives = {key for key, grade in relevance.items() if grade > 0}
    coverage = sum(key in relevance for key in selected) / len(selected) if selected else None
    base = {
        "judgment_coverage": coverage,
        "retrieved": len(selected),
        "relevant_count": len(positives),
    }
    if not positives:
        return base | {
            "recall": None,
            "mrr": None,
            "ndcg": None,
            "no_answer_returned_any": bool(selected),
        }
    grades = [relevance.get(key, 0) for key in selected]
    dcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(grades))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
    return base | {
        "recall": len(set(selected) & positives) / len(positives),
        "mrr": next((1 / (i + 1) for i, grade in enumerate(grades) if grade > 0), 0.0),
        "ndcg": dcg / idcg,
        "no_answer_returned_any": None,
    }


def union_length(intervals: list[tuple[int, int]]) -> int:
    """Measure union of half-open intervals; overlapping chunks cannot earn double credit."""
    if not intervals:
        return 0
    intervals = sorted(intervals)
    if any(end <= start for start, end in intervals):
        raise ValueError("Intervals must be nonempty and ordered")
    start, end = intervals[0]
    total = 0
    for left, right in intervals[1:]:
        if left > end:
            total += end - start
            start, end = left, right
        else:
            end = max(end, right)
    return total + end - start


def evidence_coverage(
    chunks: list[Chunk], units: list[EvidenceUnit], threshold: float = 0.8
) -> dict:
    if not 0 < threshold <= 1:
        raise ValueError("Coverage threshold must be in (0, 1]")
    fractions = {}
    for unit in units:
        alternatives = []
        for span in unit.alternatives:
            intervals = [
                (max(span.start, c.start), min(span.end, c.end))
                for c in chunks
                if c.document_key == span.document_key
                and c.source_hash == span.source_hash
                and c.tenant == span.tenant
                and max(span.start, c.start) < min(span.end, c.end)
            ]
            alternatives.append(union_length(intervals) / (span.end - span.start))
        fractions[unit.id] = max(alternatives)
    return {
        "unit_fractions": fractions,
        "threshold": threshold,
        "coverage": sum(v >= threshold for v in fractions.values()) / len(fractions)
        if fractions
        else None,
    }


def aggregate(rows: list[dict], fields=("recall", "mrr", "ndcg")) -> dict:
    result = {"queries": len(rows)}
    for field in fields:
        values = [r[field] for r in rows if r.get(field) is not None]
        result[field] = sum(values) / len(values) if values else None
        result[field + "_included"] = len(values)
    return result


def ann_recall(actual: list[str], exact: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    expected = set(unique_ranked(exact)[:k])
    if not expected:
        raise ValueError("ANN recall requires a nonempty oracle")
    return len(set(unique_ranked(actual)[:k]) & expected) / len(expected)
