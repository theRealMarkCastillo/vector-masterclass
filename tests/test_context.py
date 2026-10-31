import pytest

from vector_masterclass.chunking import chunk_document
from vector_masterclass.context import assemble, citation_audit
from vector_masterclass.schema import Document


class Words:
    def encode(self, text, **kwargs):
        return text.split()


def test_packing_counts_headers_and_preserves_provenance():
    doc = Document(id="a", revision="1", title="A", text="one two three")
    chunk = chunk_document(doc, "document")[0]
    packed = assemble([chunk, chunk], [doc], Words(), 4, tenant="public")
    assert packed["tokens"] == 4
    assert packed["citations"]["E1"]["source_hash"] == doc.sha256
    assert len(packed["chunks"]) == 1
    assert not assemble([chunk], [doc], Words(), 3, tenant="public")["chunks"]


def test_stale_poisoned_or_cross_tenant_sources_fail():
    doc = Document(id="a", revision="1", title="A", text="one two three", tenant="red")
    chunk = chunk_document(doc, "document")[0]
    with pytest.raises(PermissionError):
        assemble([chunk], [doc], Words(), 10, tenant="blue")
    with pytest.raises(PermissionError):
        assemble([chunk], [doc.model_copy(update={"active": False})], Words(), 10, tenant="red")
    with pytest.raises(ValueError):
        assemble([chunk.model_copy(update={"text": "forged"})], [doc], Words(), 10, tenant="red")


def test_citation_integrity_is_not_support():
    assert not citation_audit("A claim without citations", {"E1": {}})["all_resolve"]
    assert citation_audit("A false claim [E1]", {"E1": {}})["all_resolve"]
    assert citation_audit("Claim [E2]", {"E1": {}})["unknown"] == ["E2"]
