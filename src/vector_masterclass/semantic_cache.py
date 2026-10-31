"""Qdrant-backed semantic answer cache with exact contextual eligibility before similarity.

Callers must obtain scopes from trusted policy state. The course stores labeled answer
fixtures here; cache hits do not claim new model generation or guaranteed equivalence.
"""

import math
import re
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from .config import fingerprint, model_config, qdrant_url
from .models import embed
from .store import representation_contract


class CacheScope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_min_length=1)
    tenant: str
    authorization_revision: str
    source_id: str
    source_revision: str
    conversation_hash: str
    answer_model_revision: str


class SemanticCache:
    def __init__(self, name, *, encoder=embed):
        if not re.fullmatch(r"vmc_[a-zA-Z0-9_-]+", name):
            raise ValueError("Use an isolated course collection name")
        self.name, self.encoder = name, encoder
        self.client = QdrantClient(url=qdrant_url(), timeout=10)
        self.signature = fingerprint(
            {"kind": "semantic-cache", "encoder": representation_contract()}
        )
        if not self.client.collection_exists(name):
            self.client.create_collection(
                name,
                vectors_config=qm.VectorParams(
                    size=model_config("dense")["dimensions"], distance=qm.Distance.COSINE
                ),
                metadata={"representation_signature": self.signature},
            )
            self.client.create_payload_index(name, "scope", qm.PayloadSchemaType.KEYWORD, wait=True)
            self.client.create_payload_index(
                name, "source_id", qm.PayloadSchemaType.KEYWORD, wait=True
            )
            self.client.create_payload_index(name, "expires", qm.PayloadSchemaType.FLOAT, wait=True)
        self.verify()

    def verify(self):
        current = fingerprint({"kind": "semantic-cache", "encoder": representation_contract()})
        if current != self.signature:
            raise ValueError("Model configuration changed; recreate the semantic cache and migrate")
        if (self.client.get_collection(self.name).config.metadata or {}).get(
            "representation_signature"
        ) != self.signature:
            raise ValueError("Semantic cache representation contract mismatch")

    def put(self, query, answer, scope: CacheScope, *, now: float, ttl: float):
        if ttl <= 0 or not math.isfinite(ttl) or not math.isfinite(now):
            raise ValueError("Require finite clock and positive finite TTL")
        self.verify()
        scope_hash = fingerprint(scope.model_dump())
        id_ = str(uuid5(NAMESPACE_URL, f"{scope_hash}:{query}"))
        vector = self.encoder([query], role="query")[0]
        self.client.upsert(
            self.name,
            points=[
                qm.PointStruct(
                    id=id_,
                    vector=vector.tolist(),
                    payload={
                        "scope": scope_hash,
                        "scope_fields": scope.model_dump(),
                        "source_id": scope.source_id,
                        "query": query,
                        "answer": answer,
                        "expires": now + ttl,
                        "answer_kind": "explicit fixture or previously generated value; no generation at lookup",
                    },
                )
            ],
            wait=True,
        )
        return id_

    def get(self, query, scope: CacheScope, *, now: float, threshold=0.9):
        if not -1 <= threshold <= 1 or not math.isfinite(now):
            raise ValueError("Invalid threshold or clock")
        self.verify()
        vector = self.encoder([query], role="query")[0]
        hits = self.client.query_points(
            self.name,
            query=vector.tolist(),
            query_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="scope", match=qm.MatchValue(value=fingerprint(scope.model_dump()))
                    ),
                    qm.FieldCondition(key="expires", range=qm.Range(gt=now)),
                ]
            ),
            score_threshold=threshold,
            limit=1,
            with_payload=True,
            search_params=qm.SearchParams(exact=True),
        ).points
        if not hits:
            return None
        hit = hits[0]
        if hit.payload["scope_fields"] != scope.model_dump() or hit.payload["expires"] <= now:
            raise PermissionError("Cache returned an ineligible entry")
        return {"id": str(hit.id), "score": hit.score, **hit.payload}

    def invalidate_source(self, source_id):
        self.verify()
        self.client.delete(
            self.name,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="source_id", match=qm.MatchValue(value=source_id))]
                )
            ),
            wait=True,
        )

    def close(self, *, drop=False):
        if drop:
            self.verify()
            self.client.delete_collection(self.name)
        self.client.close()
