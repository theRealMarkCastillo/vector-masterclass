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
