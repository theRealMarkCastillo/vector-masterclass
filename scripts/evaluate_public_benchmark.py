"""Fixed public retrieval comparison. Real cached CPU models, full corpus, no sampled negatives."""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from vector_masterclass.config import root
from vector_masterclass.data import benchmark
from vector_masterclass.evidence import json_write
from vector_masterclass.lesson import LessonRun
from vector_masterclass.metrics import ranking_metrics
from vector_masterclass.models import dense_model, embed, rerank
from vector_masterclass.retrieval import LexicalIndex, exact_search, reciprocal_rank_fusion


def main():
    lesson = LessonRun("public_retrieval", "B")
    config = json.loads((root() / "configs/retrieval_evaluation.json").read_text())
    docs, development = benchmark("train")
    _, held_out = benchmark("test")
    texts, ids = [d.text for d in docs], [d.key for d in docs]
    by_id = dict(zip(ids, docs, strict=True))
    start = time.perf_counter()
    vectors = embed(texts, batch_size=64)
    dense_build = time.perf_counter() - start
    start = time.perf_counter()
    lexical = LexicalIndex(texts, ids)
    lexical_build = time.perf_counter() - start
    lengths = [len(dense_model().tokenizer.encode(t, truncation=False)) for t in texts]
    rows, rankings = [], []
    for q in [*development, *held_out]:
        start = time.perf_counter()
        dense = [
            i
            for i, _ in exact_search(
                embed([q.text], role="query")[0], vectors, ids, config["candidate_k"]
            )
        ]
        dense_seconds = time.perf_counter() - start
        start = time.perf_counter()
        bm25 = [i for i, _ in lexical.search(q.text, config["candidate_k"])]
        lexical_seconds = time.perf_counter() - start
        start = time.perf_counter()
        fused = [i for i, _ in reciprocal_rank_fusion([dense, bm25], config["fusion_constant"])][
            : config["candidate_k"]
        ]
        fusion_seconds = time.perf_counter() - start
        start = time.perf_counter()
        scores = rerank(q.text, [by_id[i].text for i in fused])
        rescored = sorted(fused, key=lambda id_: (-float(scores[fused.index(id_)]), id_))
        rerank_seconds = time.perf_counter() - start
        for method, ranked, seconds in [
            ("BM25", bm25, lexical_seconds),
            ("dense exact", dense, dense_seconds),
            ("dense+BM25 RRF60", fused, dense_seconds + lexical_seconds + fusion_seconds),
            (
                "RRF60+cross-encoder",
                rescored,
                dense_seconds + lexical_seconds + fusion_seconds + rerank_seconds,
            ),
        ]:
            rows.append(
                {
                    "split": q.split,
                    "query": q.id,
                    "method": method,
                    "seconds": seconds,
                    **ranking_metrics(ranked, q.relevance, config["k"]),
                }
            )
            rankings.append(
                {
                    "split": q.split,
                    "query": q.id,
                    "text": q.text,
                    "method": method,
                    "top_ids": ranked[: config["k"]],
                    "positive_ids": list(q.relevance),
                }
            )
        print(f"{q.split} query {q.id} evaluated", flush=True)
    results = lesson.table(pd.DataFrame(rows))
    summary = (
        results.groupby(["split", "method"])[
            ["recall", "mrr", "ndcg", "judgment_coverage", "seconds"]
        ]
        .mean()
        .reset_index()
    )
    lesson.table(summary, "summary")
    json_write(lesson.path / "rankings.json", rankings)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, split in zip(axes, ["development", "test"], strict=True):
        summary[summary.split.eq(split)].set_index("method")[["recall", "ndcg"]].plot.bar(ax=ax)
        ax.set(title=split, ylabel="Mean metric@10", ylim=(0, 1.05))
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    lesson.figure(fig)
    lesson.finish(
        {
            "full_corpus": len(docs) == 5183,
            "fixed_query_count": len(development) == len(held_out) == 20,
            "four_methods": results.method.nunique() == 4,
            "all_queries_scored": len(results) == 160,
        },
        observations={
            "config": config,
            "dense_build_seconds": dense_build,
            "lexical_build_seconds": lexical_build,
            "dense_truncated_documents": sum(n > 256 for n in lengths),
            "corpus_documents": len(docs),
            "cold_model_load_included_in_build": True,
            "reranker_first_query_includes_model_load": True,
            "judgments": "Unjudged equals zero for scoring; judgments are incomplete; no answerability labels supplied.",
        },
    )
    print(summary.to_string(index=False))
    print(lesson.path)


if __name__ == "__main__":
    main()
