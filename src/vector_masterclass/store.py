"""Real Qdrant index with representation contracts and mandatory tenant filters."""

import re
import time
from collections.abc import Callable

import numpy as np
from qdrant_client import QdrantClient, models

from .config import fingerprint, model_config, qdrant_url
from .models import embed, sparse_embed
from .schema import Chunk, Hit


def representation_contract(key: str = "dense") -> dict:
    cfg = model_config(key)
    return {k: v for k, v in cfg.items() if k not in {"device", "license", "weight_file"}}


def tenant_filter(
    tenant: str,
    document_keys: list[str] | None = None,
    authorized_sources: list[tuple[str, str]] | None = None,
) -> models.Filter:
    if not tenant:
        raise ValueError("An explicit tenant is required")
    conditions = [
        models.FieldCondition(key="tenant", match=models.MatchAny(any=sorted({"public", tenant}))),
        models.FieldCondition(key="active", match=models.MatchValue(value=True)),
    ]
    if document_keys is not None:
        conditions.append(
            models.FieldCondition(key="document_key", match=models.MatchAny(any=document_keys))
        )
    if authorized_sources is not None:
        # A document key is only unique inside its tenant. Keep each tenant/key
        # pair together; separate MatchAny lists would admit their cross product.
        groups = {}
        for source_tenant, key in authorized_sources:
            groups.setdefault(source_tenant, []).append(key)
        if groups:
            conditions.append(
                models.Filter(
                    should=[
                        models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="tenant", match=models.MatchValue(value=t)
                                ),
                                models.FieldCondition(
                                    key="document_key", match=models.MatchAny(any=keys)
                                ),
                            ]
                        )
                        for t, keys in sorted(groups.items())
                    ]
                )
            )
        else:
            conditions.append(
                models.FieldCondition(key="document_key", match=models.MatchAny(any=[]))
            )
    return models.Filter(must=conditions)


class VectorIndex:
    def __init__(
        self,
        name: str,
        *,
        key: str = "dense",
        client: QdrantClient | None = None,
        encoder: Callable = embed,
        sparse_encoder: Callable = sparse_embed,
    ):
        if not re.fullmatch(r"vmc_[a-zA-Z0-9_-]+", name):
            raise ValueError("Course collections must have an isolated vmc_ name")
        self.name, self.key = name, key
        self.client = client or QdrantClient(url=qdrant_url(), timeout=20)
        self.encoder, self.sparse_encoder = encoder, sparse_encoder
        self.contract = {
            "dense": representation_contract(key),
            "sparse": representation_contract("sparse"),
        }
        self.signature = fingerprint(self.contract)

    def create(self, *, on_disk: bool = False, quantization=None, ann: bool = False):
        if self.client.collection_exists(self.name):
            self.verify_contract()
            return
        self.client.create_collection(
            self.name,
            vectors_config={
                "dense": models.VectorParams(
                    size=model_config(self.key)["dimensions"],
                    distance=models.Distance.COSINE,
                    on_disk=on_disk,
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
            hnsw_config=models.HnswConfigDiff(full_scan_threshold=10 if ann else 10000),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=64 if ann else 10000, default_segment_number=1
            ),
            quantization_config=quantization,
            metadata={
                "course": "vector-masterclass",
                "representation_signature": self.signature,
                "representation_contract": self.contract,
            },
        )
        for field in ["tenant", "document_key", "document_id"]:
            self.client.create_payload_index(
                self.name, field, models.PayloadSchemaType.KEYWORD, wait=True
            )
        self.client.create_payload_index(
            self.name, "active", models.PayloadSchemaType.BOOL, wait=True
        )

    def verify_contract(self):
        current = {
            "dense": representation_contract(self.key),
            "sparse": representation_contract("sparse"),
        }
        if fingerprint(current) != self.signature:
            raise ValueError(
                "Representation contract mismatch: model configuration changed; recreate the index handle and migrate"
            )
        info = self.client.get_collection(self.name)
        metadata = info.config.metadata or {}
        if metadata.get("representation_signature") != self.signature:
            raise ValueError(
                "Index/query representation contract mismatch; create a new index and migrate"
            )
        return info

    def upsert(self, chunks: list[Chunk], *, document_id: str, include_sparse: bool = True):
        self.verify_contract()
        if not chunks:
            return
        texts = [c.text for c in chunks]
        dense = np.asarray(self.encoder(texts, key=self.key, role="document"))
        if (
            dense.shape != (len(chunks), model_config(self.key)["dimensions"])
            or not np.isfinite(dense).all()
        ):
            raise ValueError("Encoder violated dimensionality/finiteness contract")
        sparse = self.sparse_encoder(texts) if include_sparse else None
        points = []
        for i, chunk in enumerate(chunks):
            vector = {"dense": dense[i].tolist()}
            if sparse is not None:
                vector["sparse"] = models.SparseVector(**sparse[i])
            payload = chunk.model_dump() | {
                "document_id": document_id,
                "active": True,
                "representation_signature": self.signature,
            }
            points.append(models.PointStruct(id=chunk.id, vector=vector, payload=payload))
        for start in range(0, len(points), 128):
            self.client.upsert(self.name, points=points[start : start + 128], wait=True)

    def delete_ids(self, ids: list[str]):
        if ids:
            self.client.delete(
                self.name, points_selector=models.PointIdsList(points=ids), wait=True
            )

    def search(
        self,
        text: str,
        *,
        tenant: str,
        mode: str = "dense",
        k: int = 5,
        candidate_k: int = 20,
        document_keys: list[str] | None = None,
        authorized_sources: list[tuple[str, str]] | None = None,
        exact: bool = False,
        timings: dict | None = None,
    ) -> list[Hit]:
        if (
            mode not in {"dense", "sparse", "hybrid"}
            or not 0 < k <= 100
            or not k <= candidate_k <= 1000
        ):
            raise ValueError("Invalid search mode or candidate budget")
        started = time.perf_counter()
        self.verify_contract()
        if timings is not None:
            timings["contract_check"] = time.perf_counter() - started
        started = time.perf_counter()
        query_filter = tenant_filter(tenant, document_keys, authorized_sources)
        dense = (
            self.encoder([text], key=self.key, role="query")[0].tolist()
            if mode != "sparse"
            else None
        )
        sparse = models.SparseVector(**self.sparse_encoder([text])[0]) if mode != "dense" else None
        if timings is not None:
            timings["embedding"] = time.perf_counter() - started
        started = time.perf_counter()
        if mode == "hybrid":
            response = self.client.query_points(
                self.name,
                prefetch=[
                    models.Prefetch(
                        query=dense,
                        using="dense",
                        filter=query_filter,
                        limit=candidate_k,
                        params=models.SearchParams(exact=exact),
                    ),
                    models.Prefetch(
                        query=sparse, using="sparse", filter=query_filter, limit=candidate_k
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=k,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                self.name,
                query=dense if mode == "dense" else sparse,
                using=mode,
                query_filter=query_filter,
                search_params=models.SearchParams(exact=exact),
                limit=k,
                with_payload=True,
            )
        if timings is not None:
            timings["qdrant_request"] = time.perf_counter() - started
        started = time.perf_counter()
        hits = []
        for point in response.points:
            payload = point.payload or {}
            if payload.get("representation_signature") != self.signature:
                raise ValueError("Retrieved point has a different representation contract")
            chunk = Chunk.model_validate({key: payload[key] for key in Chunk.model_fields})
            if chunk.tenant not in {"public", tenant}:
                raise PermissionError("Unexpected cross-tenant result")
            hits.append(Hit(id=str(point.id), score=point.score, chunk=chunk))
        if timings is not None:
            timings["decode"] = time.perf_counter() - started
        return hits

    def all_points(self):
        result, offset = [], None
        while True:
            points, offset = self.client.scroll(
                self.name, limit=256, offset=offset, with_payload=True, with_vectors=True
            )
            result.extend(points)
            if offset is None:
                return result

    def drop(self):
        self.verify_contract()
        self.client.delete_collection(self.name)

    def close(self):
        self.client.close()
