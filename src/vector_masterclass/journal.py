"""Ordered source change log for a single-writer migration teaching exercise.

Opaque source revisions are compared with CAS, not lexicographically ordered.
The global sequence supplies replay order. Qdrant and SQLite are not atomic.
"""

import sqlite3
from pathlib import Path

from .schema import Document


class SourceJournal:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
          PRAGMA journal_mode=WAL;
          CREATE TABLE IF NOT EXISTS changes(sequence INTEGER PRIMARY KEY AUTOINCREMENT, source_json TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS current(tenant TEXT NOT NULL, source_id TEXT NOT NULL, source_json TEXT NOT NULL, PRIMARY KEY(tenant,source_id));
          CREATE TABLE IF NOT EXISTS consumers(name TEXT PRIMARY KEY, pipeline TEXT NOT NULL, sequence INTEGER NOT NULL);
        """)

    def append(self, doc: Document, *, expected_revision: str | None):
        with self.db:
            row = self.db.execute(
                "SELECT source_json FROM current WHERE tenant=? AND source_id=?",
                (doc.tenant, doc.id),
            ).fetchone()
            prior = Document.model_validate_json(row[0]) if row else None
            if (prior.revision if prior else None) != expected_revision:
                raise ValueError("Source revision conflict; stale writer rejected")
            if prior and prior.revision == doc.revision:
                raise ValueError("Each source event requires a new immutable revision")
            sequence = self.db.execute(
                "INSERT INTO changes(source_json) VALUES (?)", (doc.model_dump_json(),)
            ).lastrowid
            self.db.execute(
                "INSERT OR REPLACE INTO current VALUES (?,?,?)",
                (doc.tenant, doc.id, doc.model_dump_json()),
            )
        return sequence

    def snapshot(self):
        # One read transaction; the course has one authoritative writer.
        with self.db:
            self.db.execute("BEGIN")
            head = self.db.execute("SELECT COALESCE(MAX(sequence),0) FROM changes").fetchone()[0]
            docs = [
                Document.model_validate_json(row[0])
                for row in self.db.execute(
                    "SELECT source_json FROM current ORDER BY tenant,source_id"
                )
            ]
        return head, docs

    def cursor(self, ingestor):
        row = self.db.execute(
            "SELECT pipeline,sequence FROM consumers WHERE name=?", (ingestor.index.name,)
        ).fetchone()
        if row and row[0] != ingestor.pipeline_signature:
            raise ValueError("Consumer pipeline changed; create another index")
        return row[1] if row else 0

    def replay(self, ingestor, *, stop_after=None, fail_before_checkpoint=False):
        cursor = self.cursor(ingestor)
        events = self.db.execute(
            "SELECT sequence,source_json FROM changes WHERE sequence>? ORDER BY sequence", (cursor,)
        ).fetchall()
        count = 0
        for sequence, payload in events:
            if sequence != cursor + 1:
                raise ValueError("Journal gap; cannot certify complete replay")
            doc = Document.model_validate_json(payload)
            if doc.active:
                ingestor.ingest([doc])
            else:
                ingestor.delete(doc.id, doc.tenant)
            if fail_before_checkpoint:
                raise InterruptedError("Applied index update before journal cursor commit")
            with self.db:
                self.db.execute(
                    "INSERT OR REPLACE INTO consumers VALUES (?,?,?)",
                    (ingestor.index.name, ingestor.pipeline_signature, sequence),
                )
            cursor = sequence
            count += 1
            if stop_after is not None and count >= stop_after:
                break
        return count

    def close(self):
        self.db.close()
