"""Retryable ingestion and explicit deletion across SQLite and Qdrant.

The stores do not share a transaction. Pending events retain enough desired state to
replay deterministic writes and removals after interruption. This is a single-writer
teaching implementation; production concurrency requires a durable worker/lease design.
"""

import sqlite3
from pathlib import Path

from .chunking import chunk_document
from .config import fingerprint
from .schema import Document
from .store import VectorIndex


class Ingestor:
    def __init__(
        self,
        index: VectorIndex,
        manifest: Path,
        *,
        strategy: str = "fixed",
        words: int = 48,
        overlap: int = 8,
        include_sparse: bool = True,
    ):
        manifest.parent.mkdir(parents=True, exist_ok=True)
        self.index = index
        self.db = sqlite3.connect(manifest, timeout=5)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.strategy, self.words, self.overlap = strategy, words, overlap
        self.include_sparse = include_sparse
        self.pipeline_signature = fingerprint(
            {
                "representation": index.signature,
                "strategy": strategy,
                "words": words,
                "overlap": overlap,
                "include_sparse": include_sparse,
            }
        )
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS events (
              sequence INTEGER PRIMARY KEY AUTOINCREMENT, collection TEXT NOT NULL,
              source_id TEXT NOT NULL, tenant TEXT NOT NULL, operation TEXT NOT NULL,
              source_json TEXT NOT NULL, pipeline_signature TEXT NOT NULL,
              state TEXT NOT NULL DEFAULT 'pending');
            CREATE TABLE IF NOT EXISTS applied (
              collection TEXT NOT NULL, source_id TEXT NOT NULL, tenant TEXT NOT NULL,
              source_hash TEXT NOT NULL, revision TEXT NOT NULL, pipeline_signature TEXT NOT NULL,
              deleted INTEGER NOT NULL DEFAULT 0, source_json TEXT NOT NULL,
              PRIMARY KEY(collection, source_id, tenant));
            CREATE TABLE IF NOT EXISTS point_map (
              collection TEXT NOT NULL, source_id TEXT NOT NULL, tenant TEXT NOT NULL,
              point_id TEXT NOT NULL, PRIMARY KEY(collection, point_id));
        """)

    def pending(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM events WHERE collection=? AND state='pending' ORDER BY sequence",
            (self.index.name,),
        ).fetchall()

    def _enqueue(self, doc: Document, operation: str):
        with self.db:
            self.db.execute(
                "INSERT INTO events(collection,source_id,tenant,operation,source_json,pipeline_signature) VALUES (?,?,?,?,?,?)",
                (
                    self.index.name,
                    doc.id,
                    doc.tenant,
                    operation,
                    doc.model_dump_json(),
                    self.pipeline_signature,
                ),
            )

    def ingest(self, documents: list[Document], *, fail_after: int | None = None) -> dict:
        self.replay()
        enqueued = 0
        for doc in documents:
            if not doc.active:
                continue
            existing = self.db.execute(
                "SELECT * FROM applied WHERE collection=? AND source_id=? AND tenant=?",
                (self.index.name, doc.id, doc.tenant),
            ).fetchone()
            if existing:
                if existing["revision"] == doc.revision and existing["source_hash"] != doc.sha256:
                    raise ValueError("Source text changed without a revision change")
                if existing["deleted"] and existing["revision"] == doc.revision:
                    raise ValueError("Refusing to resurrect a deleted source revision")
                if (
                    not existing["deleted"]
                    and existing["source_json"] == doc.model_dump_json()
                    and existing["pipeline_signature"] == self.pipeline_signature
                ):
                    continue
            self._enqueue(doc, "upsert")
            enqueued += 1
        applied = self.replay(fail_after=fail_after)
        return {"enqueued": enqueued, "applied": applied, "pending": len(self.pending())}

    def replay(self, *, fail_after: int | None = None) -> int:
        count = 0
        for event in self.pending():
            if event["pipeline_signature"] != self.pipeline_signature:
                raise ValueError(
                    "Pending event belongs to another pipeline; replay with its original configuration"
                )
            doc = Document.model_validate_json(event["source_json"])
            old_ids = [
                r[0]
                for r in self.db.execute(
                    "SELECT point_id FROM point_map WHERE collection=? AND source_id=? AND tenant=?",
                    (self.index.name, doc.id, doc.tenant),
                )
            ]
            chunks = (
                chunk_document(doc, self.strategy, self.words, self.overlap)
                if event["operation"] == "upsert"
                else []
            )
            new_ids = [chunk.id for chunk in chunks]
            if chunks:
                self.index.upsert(chunks, document_id=doc.id, include_sparse=self.include_sparse)
            # Fault injection occurs at the difficult boundary: remote write succeeded,
            # but the local manifest has not committed. Stable point IDs make replay safe.
            if fail_after is not None and count == fail_after:
                raise InterruptedError(
                    "Injected interruption after Qdrant write, before manifest commit"
                )
            self.index.delete_ids(sorted(set(old_ids) - set(new_ids)))
            with self.db:
                self.db.execute(
                    "DELETE FROM point_map WHERE collection=? AND source_id=? AND tenant=?",
                    (self.index.name, doc.id, doc.tenant),
                )
                self.db.executemany(
                    "INSERT INTO point_map VALUES (?,?,?,?)",
                    [(self.index.name, doc.id, doc.tenant, id_) for id_ in new_ids],
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO applied VALUES (?,?,?,?,?,?,?,?)",
                    (
                        self.index.name,
                        doc.id,
                        doc.tenant,
                        doc.sha256,
                        doc.revision,
                        self.pipeline_signature,
                        int(event["operation"] == "delete"),
                        doc.model_dump_json(),
                    ),
                )
                self.db.execute(
                    "UPDATE events SET state='applied' WHERE sequence=?", (event["sequence"],)
                )
            count += 1
        return count

    def delete(self, source_id: str, tenant: str):
        self.replay()
        row = self.db.execute(
            "SELECT source_json,deleted FROM applied WHERE collection=? AND source_id=? AND tenant=?",
            (self.index.name, source_id, tenant),
        ).fetchone()
        if row is None or row["deleted"]:
            return 0
        self._enqueue(Document.model_validate_json(row["source_json"]), "delete")
        return self.replay()

    def current_documents(self) -> list[Document]:
        return [
            Document.model_validate_json(row[0])
            for row in self.db.execute(
                "SELECT source_json FROM applied WHERE collection=? AND deleted=0 ORDER BY source_id,tenant",
                (self.index.name,),
            )
        ]

    def events(self) -> list[dict]:
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT * FROM events WHERE collection=? ORDER BY sequence", (self.index.name,)
            )
        ]

    def reconcile(self) -> dict:
        expected = {
            r[0]
            for r in self.db.execute(
                "SELECT point_id FROM point_map WHERE collection=?", (self.index.name,)
            )
        }
        actual = {str(p.id) for p in self.index.all_points()}
        return {
            "missing": sorted(expected - actual),
            "unexpected": sorted(actual - expected),
            "pending": len(self.pending()),
            "consistent": expected == actual and not self.pending(),
        }

    def close(self):
        self.db.close()
