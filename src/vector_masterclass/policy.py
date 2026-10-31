"""Persistent local identity simulation and authoritative teaching source/grant fixture.

Known demo bearer tokens simulate authentication; this is not a production identity provider.
"""

import hashlib
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from .schema import Document


@dataclass(frozen=True)
class Identity:
    subject: str
    tenant: str
    epoch: int


class PolicyStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS epoch (id INTEGER PRIMARY KEY CHECK(id=1), value INTEGER NOT NULL);
            INSERT OR IGNORE INTO epoch VALUES (1,0);
            CREATE TABLE IF NOT EXISTS identities (subject TEXT PRIMARY KEY, tenant TEXT NOT NULL, token_hash TEXT UNIQUE NOT NULL, active INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS sources (tenant TEXT NOT NULL, source_id TEXT NOT NULL, document_json TEXT NOT NULL, PRIMARY KEY(tenant,source_id));
            CREATE TABLE IF NOT EXISTS grants (subject TEXT NOT NULL, tenant TEXT NOT NULL, source_id TEXT NOT NULL, PRIMARY KEY(subject,tenant,source_id));
        """)
        self.db.commit()

    def _bump(self):
        self.db.execute("UPDATE epoch SET value=value+1 WHERE id=1")

    def epoch(self):
        with self.lock:
            return self.db.execute("SELECT value FROM epoch WHERE id=1").fetchone()[0]

    def add_identity(self, subject: str, tenant: str, token: str):
        if not subject or not tenant or not token:
            raise ValueError("Identity fields cannot be empty")
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO identities VALUES (?,?,?,1) ON CONFLICT(subject) DO UPDATE SET tenant=excluded.tenant,token_hash=excluded.token_hash,active=1",
                (subject, tenant, hashlib.sha256(token.encode()).hexdigest()),
            )
            self._bump()

    def authenticate(self, token: str) -> Identity:
        with self.lock:
            row = self.db.execute(
                "SELECT subject,tenant FROM identities WHERE token_hash=? AND active=1",
                (hashlib.sha256(token.encode()).hexdigest(),),
            ).fetchone()
            if row is None:
                raise PermissionError("Invalid or revoked local demo identity")
            return Identity(row[0], row[1], self.epoch())

    def put_source(self, doc: Document):
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO sources VALUES (?,?,?) ON CONFLICT(tenant,source_id) DO UPDATE SET document_json=excluded.document_json",
                (doc.tenant, doc.id, doc.model_dump_json()),
            )
            self._bump()

    def grant(self, subject: str, tenant: str, source_id: str):
        with self.lock, self.db:
            row = self.db.execute(
                "SELECT tenant FROM identities WHERE subject=? AND active=1", (subject,)
            ).fetchone()
            if row is None or row[0] != tenant:
                raise PermissionError("Local fixture grants cannot cross tenant membership")
            self.db.execute(
                "INSERT OR IGNORE INTO grants VALUES (?,?,?)", (subject, tenant, source_id)
            )
            self._bump()

    def revoke_grant(self, subject: str, tenant: str, source_id: str):
        with self.lock, self.db:
            self.db.execute(
                "DELETE FROM grants WHERE subject=? AND tenant=? AND source_id=?",
                (subject, tenant, source_id),
            )
            self._bump()

    def revoke_identity(self, subject: str):
        with self.lock, self.db:
            self.db.execute("UPDATE identities SET active=0 WHERE subject=?", (subject,))
            self._bump()

    def assert_current(self, identity: Identity):
        with self.lock:
            row = self.db.execute(
                "SELECT tenant,active FROM identities WHERE subject=?", (identity.subject,)
            ).fetchone()
            if row != (identity.tenant, 1) or self.epoch() != identity.epoch:
                raise PermissionError(
                    "Policy/source state changed during request; retry with current authorization"
                )

    def sources(self, identity: Identity) -> list[Document]:
        with self.lock:
            self.assert_current(identity)
            rows = self.db.execute(
                "SELECT s.document_json FROM sources s WHERE s.tenant='public' OR (s.tenant=? AND EXISTS (SELECT 1 FROM grants g WHERE g.subject=? AND g.tenant=s.tenant AND g.source_id=s.source_id))",
                (identity.tenant, identity.subject),
            ).fetchall()
            return [d for row in rows if (d := Document.model_validate_json(row[0])).active]

    def close(self):
        with self.lock:
            self.db.close()


def seed_demo_policy(policy: PolicyStore, docs: list[Document]):
    """Explicit setup only; never reseed on each request or silently undo revocations."""
    for subject, tenant in [("alice", "red"), ("bob", "blue"), ("visitor", "public")]:
        policy.add_identity(subject, tenant, f"demo-{tenant}-token")
    for doc in docs:
        policy.put_source(doc)
        if doc.tenant in {"red", "blue"}:
            policy.grant("alice" if doc.tenant == "red" else "bob", doc.tenant, doc.id)
