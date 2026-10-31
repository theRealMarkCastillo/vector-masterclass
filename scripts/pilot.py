"""M0: actual native CPU/MPS models and Compose Qdrant. Run from the uv environment."""

import json
import time
from uuid import uuid4

import numpy as np
from qdrant_client import QdrantClient, models

from vector_masterclass.config import qdrant_url
from vector_masterclass.evidence import ResourceMeter, json_write, new_run, provenance
from vector_masterclass.models import embed, generate, release_models, rerank, sparse_embed
from vector_masterclass.preflight import preflight


def main():
    out = new_run("pilot")
    result = {"preflight": preflight(), "provenance": provenance(), "live": True}
    texts = [
        "The Atlas router warranty lasts two years from the purchase date.",
        "The Boreal switch warranty lasts five years from the purchase date.",
        "Reset an Atlas router by holding the reset button for ten seconds.",
    ]
    question = "How long is the Atlas router warranty?"
    client = QdrantClient(url=qdrant_url(), timeout=10)
    collection = "vmc_pilot_" + uuid4().hex[:10]
    with ResourceMeter() as meter:
        start = time.perf_counter()
        vectors = embed(texts)
        query = embed([question], role="query")[0]
        result["dense_cold_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        warm = embed(texts)
        result["dense_warm_seconds"] = time.perf_counter() - start
        assert vectors.shape == (3, 384) and np.allclose(vectors, warm, atol=1e-6)
        client.create_collection(
            collection,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        try:
            client.upsert(
                collection,
                points=[
                    models.PointStruct(id=i, vector=v.tolist(), payload={"text": texts[i]})
                    for i, v in enumerate(vectors)
                ],
                wait=True,
            )
            hits = client.query_points(collection, query=query.tolist(), limit=3).points
            result["qdrant_dense_hits"] = [{"id": h.id, "score": h.score} for h in hits]
            assert hits[0].id == 0
            result["reranker_scores"] = rerank(question, texts).tolist()
            assert np.argmax(result["reranker_scores"]) == 0
            release_models()
            start = time.perf_counter()
            sparse = sparse_embed([*texts, question])
            result["sparse_seconds"] = time.perf_counter() - start
            result["sparse_nonzero"] = [len(v["indices"]) for v in sparse]
            client.update_vectors(
                collection,
                points=[
                    models.PointVectors(id=i, vector={"sparse": models.SparseVector(**v)})
                    for i, v in enumerate(sparse[:-1])
                ],
                wait=True,
            )
            sparse_hits = client.query_points(
                collection, query=models.SparseVector(**sparse[-1]), using="sparse", limit=3
            ).points
            result["qdrant_sparse_hits"] = [{"id": h.id, "score": h.score} for h in sparse_hits]
            assert sparse_hits[0].id == 0
            release_models()
            system = "Answer using only the supplied evidence. Cite the source ID in square brackets. If the evidence does not answer the question, reply exactly: Insufficient evidence."
            result["answerable"] = generate(
                f"Evidence:\n[D1] {texts[0]}\nQuestion: {question}", system=system
            )
            result["unanswerable"] = generate(
                f"Evidence:\n[D1] {texts[0]}\nQuestion: What is the router's waterproof rating?",
                system=system,
            )
            result["generation_checks"] = {
                "answer_mentions_two_years": "two years" in result["answerable"]["text"].lower()
                or "2 years" in result["answerable"]["text"].lower(),
                "answer_cites_D1": "[D1]" in result["answerable"]["text"],
                "unanswerable_exact_format": "insufficient evidence"
                in result["unanswerable"]["text"].lower(),
            }
            result["collection_info"] = client.get_collection(collection).model_dump(mode="json")
        except Exception as error:
            result["error"] = {"type": type(error).__name__, "message": str(error)}
            json_write(out / "failed.json", result)
            raise
        finally:
            client.delete_collection(collection)
            client.close()
    result["resources"] = meter.report()
    json_write(out / "manifest.json", result)
    print(
        json.dumps(
            {
                "artifact": str(out),
                "generation_checks": result["generation_checks"],
                "resources": result["resources"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
