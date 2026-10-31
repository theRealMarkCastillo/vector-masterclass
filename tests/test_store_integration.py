from uuid import uuid4

import numpy as np
import pytest

from vector_masterclass.ingestion import Ingestor
from vector_masterclass.schema import Document
from vector_masterclass.store import VectorIndex

pytestmark = pytest.mark.integration


def fixture_encoder(texts, **kwargs):
    # Isolate storage/filter correctness from model quality; real-model checks are separate.
    out = np.zeros((len(texts), 384), dtype=np.float32)
    out[:, 0] = 1
    return out


def fixture_sparse(texts):
    return [{"indices": [1], "values": [1.0]} for _ in texts]


@pytest.fixture
def index():
    index = VectorIndex(
        "vmc_test_" + uuid4().hex[:10], encoder=fixture_encoder, sparse_encoder=fixture_sparse
    )
    index.create()
    yield index
    index.drop()
    index.close()


def test_interrupted_upsert_retry_update_and_delete(index, tmp_path):
    pipeline = Ingestor(index, tmp_path / "manifest.sqlite")
    original = Document(id="d", revision="1", title="", text="old source text")
    try:
        with pytest.raises(InterruptedError):
            pipeline.ingest([original], fail_after=0)
        assert len(pipeline.pending()) == 1
        assert not pipeline.reconcile()["consistent"]
        assert pipeline.replay() == 1
        assert pipeline.reconcile()["consistent"]
        assert pipeline.ingest([original])["enqueued"] == 0
        old_ids = {p.id for p in index.all_points()}
        revised = original.model_copy(update={"revision": "2", "text": "new source text"})
        pipeline.ingest([revised])
        assert not old_ids & {p.id for p in index.all_points()}
        assert pipeline.reconcile()["consistent"]
        assert pipeline.delete("d", "public") == 1
        assert index.all_points() == []
        with pytest.raises(ValueError, match="resurrect"):
            pipeline.ingest([revised])
    finally:
        pipeline.close()


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_tenant_filters_at_all_retrieval_stages(index, tmp_path, mode):
    pipeline = Ingestor(index, tmp_path / "manifest.sqlite")
    docs = [
        Document(id=tenant, revision="1", title="", text="same words", tenant=tenant)
        for tenant in ["public", "red", "blue"]
    ]
    try:
        pipeline.ingest(docs)
        hits = index.search("same words", tenant="red", mode=mode)
        assert {h.chunk.tenant for h in hits} == {"public", "red"}
        assert index.search("same words", tenant="red", mode=mode, document_keys=[]) == []
    finally:
        pipeline.close()


def test_same_dimension_different_model_contract_rejected(index):
    bad = VectorIndex(index.name)
    bad.signature = "different-model-same-384-dimensions"
    try:
        with pytest.raises(ValueError, match="contract mismatch"):
            bad.verify_contract()
    finally:
        bad.close()


def test_same_source_id_and_text_do_not_overwrite_other_tenants(index, tmp_path):
    pipeline = Ingestor(index, tmp_path / "manifest.sqlite")
    docs = [
        Document(id="shared-id", revision="1", title="", text="identical content", tenant=tenant)
        for tenant in ["public", "red", "blue"]
    ]
    try:
        pipeline.ingest(docs)
        assert len(index.all_points()) == 3
        assert {h.chunk.tenant for h in index.search("identical content", tenant="red")} == {
            "public",
            "red",
        }
        pipeline.delete("shared-id", "red")
        assert {p.payload["tenant"] for p in index.all_points()} == {"public", "blue"}
    finally:
        pipeline.close()


def test_source_revision_and_pending_pipeline_guards(index, tmp_path):
    path = tmp_path / "manifest.sqlite"
    pipeline = Ingestor(index, path)
    doc = Document(id="d", revision="1", title="", text="original")
    pipeline.ingest([doc])
    with pytest.raises(ValueError, match="revision change"):
        pipeline.ingest([doc.model_copy(update={"text": "changed"})])
    new = doc.model_copy(update={"revision": "2", "text": "new"})
    with pytest.raises(InterruptedError):
        pipeline.ingest([new], fail_after=0)
    pipeline.close()
    other = Ingestor(index, path, words=20)
    try:
        with pytest.raises(ValueError, match="another pipeline"):
            other.replay()
    finally:
        other.close()


def test_resumed_ingestion_equals_clean_remote_state(index, tmp_path):
    clean = VectorIndex(
        "vmc_test_clean_" + uuid4().hex[:10], encoder=fixture_encoder, sparse_encoder=fixture_sparse
    )
    clean.create()
    resumed = Ingestor(index, tmp_path / "resumed.sqlite", words=4, overlap=1)
    baseline = Ingestor(clean, tmp_path / "clean.sqlite", words=4, overlap=1)
    docs = [
        Document(
            id=str(i),
            revision="1",
            title="",
            text=f"source {i} has several words crossing chunk boundaries",
        )
        for i in range(3)
    ]
    try:
        with pytest.raises(InterruptedError):
            resumed.ingest(docs, fail_after=1)
        resumed.close()
        resumed = Ingestor(index, tmp_path / "resumed.sqlite", words=4, overlap=1)
        resumed.replay()
        baseline.ingest(docs)

        def state(store):
            return sorted(
                [(p.id, p.payload, p.vector) for p in store.all_points()], key=lambda r: r[0]
            )

        assert state(index) == state(clean)
        assert resumed.reconcile()["consistent"] and baseline.reconcile()["consistent"]
    finally:
        resumed.close()
        baseline.close()
        clean.drop()
        clean.close()


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_qualified_authorization_excludes_colliding_private_key(index, tmp_path, mode):
    pipeline = Ingestor(index, tmp_path / "manifest.sqlite")
    docs = [
        Document(id="same", revision="1", title="", text="same text", tenant=t)
        for t in ("public", "red")
    ]
    try:
        pipeline.ingest(docs)
        hits = index.search(
            "same", tenant="red", mode=mode, authorized_sources=[("public", docs[0].key)]
        )
        assert [h.chunk.tenant for h in hits] == ["public"]
        assert index.search("same", tenant="red", mode=mode, authorized_sources=[]) == []
    finally:
        pipeline.close()


def test_ingestion_batch_conflict_is_atomic(index, tmp_path):
    pipeline = Ingestor(index, tmp_path / "manifest.sqlite")
    original = Document(id="same", revision="1", title="", text="original")
    try:
        with pytest.raises(ValueError, match="revision change"):
            pipeline.ingest([original, original.model_copy(update={"text": "changed"})])
        assert pipeline.events() == []
        assert pipeline.current_documents() == []
        assert index.all_points() == []
        assert pipeline.ingest([original, original])["enqueued"] == 1
    finally:
        pipeline.close()


def test_ingestion_rejects_historical_revision_after_update_and_delete(index, tmp_path):
    path = tmp_path / "manifest.sqlite"
    original = Document(id="same", revision="1", title="", text="original")
    pipeline = Ingestor(index, path)
    pipeline.ingest([original])
    pipeline.delete(original.id, original.tenant)
    pipeline.ingest([original.model_copy(update={"revision": "2"})])
    pipeline.close()
    pipeline = Ingestor(index, path)
    try:
        with pytest.raises(ValueError, match="historical revision"):
            pipeline.ingest([original])
        assert pipeline.current_documents()[0].revision == "2"
        assert pipeline.reconcile()["consistent"]
    finally:
        pipeline.close()


def test_open_index_rejects_configuration_drift_before_encoding(index, monkeypatch):
    import vector_masterclass.store as store

    original = store.representation_contract
    monkeypatch.setattr(
        store, "representation_contract", lambda key: original(key) | {"revision": "changed"}
    )
    with pytest.raises(ValueError, match="configuration changed"):
        index.search("query", tenant="public")
    monkeypatch.undo()


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_api_serves_public_collision_after_private_grant_revoked(index, tmp_path, mode):
    from fastapi.testclient import TestClient

    from vector_masterclass.policy import PolicyStore, seed_demo_policy
    from vector_masterclass.service import SearchService, create_app

    pipeline = Ingestor(index, tmp_path / "manifest.sqlite")
    policy = PolicyStore(tmp_path / "policy.sqlite")
    docs = [
        Document(id="same", revision="1", title="", text="same text", tenant=t)
        for t in ("public", "red")
    ]
    pipeline.ingest(docs)
    seed_demo_policy(policy, docs)
    service = SearchService(index, policy)
    try:
        with TestClient(create_app(service)) as client:
            kwargs = {
                "json": {"query": "same", "mode": mode},
                "headers": {"Authorization": "Bearer demo-red-token"},
            }
            assert len(client.post("/search", **kwargs).json()["results"]) == 2
            policy.revoke_grant("alice", "red", "same")
            response = client.post("/search", **kwargs)
            assert response.status_code == 200
            assert [h["chunk"]["tenant"] for h in response.json()["results"]] == ["public"]
    finally:
        service.close()
        policy.close()
        pipeline.close()


def test_identity_pipeline_upgrade_removes_previous_mapped_points(index, tmp_path):
    from vector_masterclass.chunking import chunk_document

    pipeline = Ingestor(index, tmp_path / "manifest.sqlite", include_sparse=False)
    doc = Document(id="d", revision="1", title="", text="source text")
    pipeline.ingest([doc])
    current = chunk_document(doc)[0]
    legacy = current.model_copy(update={"id": str(uuid4())})
    # Model an existing manifest whose earlier identity algorithm emitted another UUID.
    index.delete_ids([current.id])
    index.upsert([legacy], document_id=doc.id, include_sparse=False)
    with pipeline.db:
        pipeline.db.execute("UPDATE point_map SET point_id=?", (legacy.id,))
        pipeline.db.execute("UPDATE events SET pipeline_signature='legacy'")
        pipeline.db.execute("UPDATE applied SET pipeline_signature='legacy'")
    try:
        assert pipeline.ingest([doc])["enqueued"] == 1
        assert {p.id for p in index.all_points()} == {current.id}
        assert pipeline.reconcile()["consistent"]
        assert pipeline.ingest([doc])["enqueued"] == 0
    finally:
        pipeline.close()
