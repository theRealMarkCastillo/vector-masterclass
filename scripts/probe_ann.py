"""M0 ANN feasibility probe. Synthetic vectors test index fidelity, not relevance."""

import time
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient, models

from vector_masterclass.config import qdrant_url
from vector_masterclass.evidence import json_write, new_run, provenance


def main():
    rng = np.random.default_rng(17)
    vectors = rng.normal(size=(6000, 64)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = rng.normal(size=(12, 64)).astype("float32")
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    client = QdrantClient(url=qdrant_url(), timeout=30)
    name = "vmc_ann_probe_" + uuid4().hex[:8]
    out = new_run("ann_probe")
    result = {"provenance": provenance(), "synthetic": True, "semantic_quality_claim": False}
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=64, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=8, ef_construct=40, full_scan_threshold=10),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=64, default_segment_number=1
        ),
    )
    try:
        start = time.perf_counter()
        for first in range(0, len(vectors), 500):
            client.upsert(
                name,
                points=[
                    models.PointStruct(id=i, vector=vectors[i].tolist())
                    for i in range(first, min(first + 500, len(vectors)))
                ],
                wait=True,
            )
        result["ingestion_seconds"] = time.perf_counter() - start
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            info = client.get_collection(name)
            if (
                info.status == models.CollectionStatus.GREEN
                and info.indexed_vectors_count >= 0.99 * len(vectors)
            ):
                break
            time.sleep(0.5)
        else:
            raise TimeoutError("Probe index did not become sufficiently indexed within 60 seconds")
        result["index_wait_seconds"] = time.perf_counter() - start - result["ingestion_seconds"]
        result["collection"] = info.model_dump(mode="json")
        rows = []
        for ef in [8, 16, 32, 64, 128]:
            latencies, recall = [], []
            for query in queries:
                exact = client.query_points(
                    name,
                    query=query.tolist(),
                    limit=10,
                    search_params=models.SearchParams(exact=True),
                ).points
                start = time.perf_counter()
                hits = client.query_points(
                    name,
                    query=query.tolist(),
                    limit=10,
                    search_params=models.SearchParams(hnsw_ef=ef, exact=False),
                ).points
                latencies.append((time.perf_counter() - start) * 1000)
                recall.append(len({h.id for h in hits} & {h.id for h in exact}) / 10)
                numpy_ids = np.argsort(-(vectors @ query), kind="stable")[:10]
                assert {h.id for h in exact} == set(numpy_ids.tolist())
            rows.append(
                {
                    "hnsw_ef": ef,
                    "mean_ann_recall_at_10": float(np.mean(recall)),
                    "median_ms": float(np.median(latencies)),
                }
            )
        result["sweep"] = rows
        result["interpretation"] = (
            "Indexed-vector coverage and minimum supported full-scan threshold make HNSW eligible for unfiltered approximate requests. API provides no per-query execution trace here; this is configuration/behavior evidence, not a traced execution-plan proof. Twelve queries are a feasibility probe, not a production benchmark."
        )
        json_write(out / "manifest.json", result)
        print(out)
        print(rows)
    finally:
        client.delete_collection(name)
        client.close()


if __name__ == "__main__":
    main()
