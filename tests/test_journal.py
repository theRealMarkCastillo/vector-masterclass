import pytest

from vector_masterclass.journal import SourceJournal
from vector_masterclass.schema import Document


def test_revision_cas_and_opaque_revision_snapshot(tmp_path):
    journal = SourceJournal(tmp_path / "journal.sqlite")
    doc = Document(id="d", revision="z", title="", text="first")
    assert journal.append(doc, expected_revision=None) == 1
    corrected = doc.model_copy(update={"revision": "a", "text": "corrected"})
    assert journal.append(corrected, expected_revision="z") == 2
    with pytest.raises(ValueError, match="stale writer"):
        journal.append(doc, expected_revision="z")
    assert journal.snapshot() == (2, [corrected])
    journal.close()


def test_gap_rejects_replay_and_checkpoint_crash_replays(tmp_path):
    journal = SourceJournal(tmp_path / "journal.sqlite")
    doc = Document(id="d", revision="1", title="", text="first")
    journal.append(doc, expected_revision=None)

    class Consumer:
        index = type("Index", (), {"name": "vmc_test"})()
        pipeline_signature = "a"
        calls = 0

        def ingest(self, docs):
            self.calls += 1

    consumer = Consumer()
    with pytest.raises(InterruptedError):
        journal.replay(consumer, fail_before_checkpoint=True)
    assert journal.cursor(consumer) == 0
    journal.replay(consumer)
    assert consumer.calls == 2 and journal.cursor(consumer) == 1
    journal.append(doc.model_copy(update={"revision": "2"}), expected_revision="1")
    journal.append(doc.model_copy(update={"revision": "3"}), expected_revision="2")
    with journal.db:
        journal.db.execute("DELETE FROM changes WHERE sequence=2")
    with pytest.raises(ValueError, match="gap"):
        journal.replay(consumer)
    journal.close()


def test_historical_revision_cannot_be_reused(tmp_path):
    journal = SourceJournal(tmp_path / "journal.sqlite")
    original = Document(id="d", revision="one", title="", text="original")
    try:
        journal.append(original, expected_revision=None)
        journal.append(original.model_copy(update={"revision": "two"}), expected_revision="one")
        with pytest.raises(ValueError, match="immutable revision"):
            journal.append(original, expected_revision="two")
        assert journal.snapshot()[0] == 2
    finally:
        journal.close()
