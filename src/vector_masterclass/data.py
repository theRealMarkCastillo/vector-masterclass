"""Versioned fictional corpus and checksummed public benchmark access."""

import csv
import json

from .config import file_hash, root
from .schema import Document, Query


def documents(*, tenant: str | None = None, active_only: bool = True) -> list[Document]:
    rows = [
        Document.model_validate_json(line)
        for line in (root() / "datasets/course/corpus.jsonl").read_text().splitlines()
        if line
    ]
    return [
        d
        for d in rows
        if (not active_only or d.active) and (tenant is None or d.tenant in {"public", tenant})
    ]


def queries(split: str | None = None) -> list[Query]:
    rows = [
        Query.model_validate_json(line)
        for line in (root() / "datasets/course/queries.jsonl").read_text().splitlines()
        if line
    ]
    return [q for q in rows if split is None or q.split == split]


def validate_gold() -> dict:
    docs = {d.key: d for d in documents(active_only=False)}
    families = {}
    for query in queries():
        for unit in query.evidence:
            for span in unit.alternatives:
                doc = docs[span.document_key]
                span.validate_source(doc)
                if doc.tenant not in {"public", query.tenant}:
                    raise ValueError("Gold evidence violates query tenant permissions")
                families.setdefault(query.split, set()).add(doc.family)
    if families.get("development", set()) & families.get("test", set()):
        raise ValueError("Development and test gold sources share product families")
    review_path = root() / "configs/gold_review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    approved = (
        review.get("status") == "approved"
        and all(
            file_hash(root() / path) == expected
            for path, expected in review.get("files", {}).items()
        )
        and len(review.get("files", {})) == 2
    )
    return {
        "queries": len(queries()),
        "source_revisions": len(docs),
        "gold_span_checks": "passed",
        "human_review": "approved" if approved else "pending_or_stale",
    }


def benchmark(split: str = "test") -> tuple[list[Document], list[Query]]:
    if split not in {"train", "test"}:
        raise ValueError("Benchmark split must be train or test")
    cfg = json.loads((root() / "configs/benchmark.json").read_text())
    folder = root() / "datasets/downloads"
    for relative, expected in cfg["files"].items():
        path = folder / relative
        if not path.exists() or file_hash(path) != expected:
            raise ValueError(
                "Public benchmark is missing or changed; run scripts/download_benchmark.py"
            )
    corpus = [
        json.loads(line) for line in (folder / "scifact/corpus.jsonl").read_text().splitlines()
    ]
    qs = {
        str(q["_id"]): q["text"]
        for q in (
            json.loads(line) for line in (folder / "scifact/queries.jsonl").read_text().splitlines()
        )
    }
    labels = {}
    with (folder / f"scifact/qrels/{split}.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            labels.setdefault(row["query-id"], {})[row["corpus-id"] + "@beir"] = int(row["score"])
    selected = sorted(labels, key=int)[:20]
    docs = [
        Document(
            id=str(d["_id"]),
            revision="beir",
            title=d["title"],
            text=d["title"] + "\n" + d["text"],
            family="scifact",
        )
        for d in corpus
    ]
    query_rows = [
        Query(
            id=q,
            text=qs[q],
            split="development" if split == "train" else "test",
            relevance=labels[q],
            rationale=cfg["judgment_convention"],
        )
        for q in selected
    ]
    return docs, query_rows
