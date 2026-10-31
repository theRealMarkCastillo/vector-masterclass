import math

import pytest

from vector_masterclass.metrics import (
    aggregate,
    ann_recall,
    evidence_coverage,
    ranking_metrics,
    union_length,
)
from vector_masterclass.schema import Chunk, Document, EvidenceUnit, Query, SourceSpan


def test_rank_metrics_hand_computed():
    result = ranking_metrics(["noise", "a", "a", "b"], {"a": 2, "b": 1}, 3)
    assert result["recall"] == 1
    assert result["mrr"] == 0.5
    assert result["ndcg"] == pytest.approx((3 / math.log2(3) + 1 / 2) / (3 + 1 / math.log2(3)))
    assert result["judgment_coverage"] == pytest.approx(2 / 3)


def test_no_answer_not_silently_zero_in_mean():
    empty = ranking_metrics(["a"], {}, 3)
    assert empty["recall"] is None and empty["no_answer_returned_any"]
    summary = aggregate([empty, ranking_metrics(["a"], {"a": 1}, 1)])
    assert summary["recall"] == 1 and summary["recall_included"] == 1
    assert ranking_metrics([], {"a": 1}, 3)["recall"] == 0


def test_overlapping_chunks_are_union_not_sum():
    doc = Document(id="d", revision="1", title="Example", text="0123456789")
    unit = EvidenceUnit(
        id="fact",
        alternatives=[SourceSpan(document_key=doc.key, source_hash=doc.sha256, start=0, end=10)],
    )
    chunks = [Chunk.from_span(doc, 0, 4, "a"), Chunk.from_span(doc, 2, 6, "a")]
    coverage = evidence_coverage(chunks, [unit])
    assert coverage["unit_fractions"]["fact"] == 0.6
    assert coverage["coverage"] == 0
    chunks.append(Chunk.from_span(doc, 6, 10, "a"))
    assert evidence_coverage(chunks, [unit])["coverage"] == 1


def test_revision_hash_prevents_false_evidence_credit():
    old = Document(id="d", revision="1", title="", text="old advice")
    new = Document(id="d", revision="2", title="", text="new advice")
    span = SourceSpan(document_key=new.key, source_hash=new.sha256, start=0, end=10)
    unit = EvidenceUnit(id="f", alternatives=[span])
    assert evidence_coverage([Chunk.from_span(old, 0, 10, "document")], [unit])["coverage"] == 0
    with pytest.raises(ValueError):
        span.validate_source(old)


def test_alternative_evidence_and_empty_gold():
    a = Document(id="a", revision="1", title="", text="alpha")
    b = Document(id="b", revision="1", title="", text="bravo")
    unit = EvidenceUnit(
        id="either",
        alternatives=[
            SourceSpan(document_key=d.key, source_hash=d.sha256, start=0, end=5) for d in [a, b]
        ],
    )
    assert evidence_coverage([Chunk.from_span(b, 0, 5, "d")], [unit])["coverage"] == 1
    assert evidence_coverage([], [])["coverage"] is None


def test_interval_and_metric_invalid_inputs():
    assert union_length([(0, 2), (2, 5), (8, 10)]) == 7
    with pytest.raises(ValueError):
        union_length([(4, 3)])
    with pytest.raises(ValueError):
        ranking_metrics([], {}, 0)
    with pytest.raises(ValueError):
        Query(id="q", text="?", split="test", answerable=False, relevance={"a": 1})
    assert ann_recall(["a", "x"], ["a", "b"], 2) == 0.5
