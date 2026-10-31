import pytest

from vector_masterclass.chunking import chunk_document, pack_chunks
from vector_masterclass.schema import Document


@pytest.mark.parametrize("strategy", ["fixed", "structural", "parent_child", "document"])
def test_source_offsets_and_stable_ids(strategy):
    doc = Document(
        id="a", revision="1", title="", text="One two three four.\n\nFive six seven eight."
    )
    chunks = chunk_document(doc, strategy, words=3, overlap=1)
    assert chunks == chunk_document(doc, strategy, words=3, overlap=1)
    assert all(c.text == doc.text[c.start : c.end] for c in chunks)
    if strategy == "parent_child":
        assert all(c.parent_id for c in chunks)


def test_invalid_overlap():
    doc = Document(id="a", revision="1", title="", text="abc")
    with pytest.raises(ValueError):
        chunk_document(doc, words=3, overlap=3)


def test_packing_counts_headers_and_deduplicates():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    doc = Document(id="a", revision="1", title="", text="one two\n\nthree four")
    chunks = chunk_document(doc, "structural")
    packed = pack_chunks([chunks[0], chunks[0], chunks[1]], Tokenizer(), budget=4)
    assert len(packed["chunks"]) == 1
    assert packed["tokens"] == 3
