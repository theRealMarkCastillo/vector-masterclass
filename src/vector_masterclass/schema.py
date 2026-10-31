"""Immutable source identity and evidence anchored before chunking."""

import hashlib
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Document(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    title: str
    text: str = Field(min_length=1)
    tenant: str = "public"
    family: str = "general"
    published: str = "2026-01-01"
    active: bool = True

    @property
    def key(self) -> str:
        return f"{self.id}@{self.revision}"

    @property
    def sha256(self) -> str:
        return text_hash(self.text)


class SourceSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    document_key: str
    source_hash: str
    tenant: str = "public"
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def valid_order(self):
        if self.end <= self.start:
            raise ValueError("Evidence spans must be nonempty half-open intervals")
        return self

    def validate_source(self, document: Document):
        if (
            self.document_key != document.key
            or self.source_hash != document.sha256
            or self.tenant != document.tenant
            or self.end > len(document.text)
        ):
            raise ValueError("Evidence does not match the immutable canonical source")


class EvidenceUnit(BaseModel):
    """Any one alternative span supports this fact; distinct units are distinct required facts."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    alternatives: list[SourceSpan] = Field(min_length=1)


class Query(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    text: str
    split: Literal["development", "test"]
    tenant: str = "public"
    relevance: dict[str, int] = Field(default_factory=dict)
    evidence: list[EvidenceUnit] = Field(default_factory=list)
    answerable: bool = True
    rationale: str = ""

    @model_validator(mode="after")
    def valid_labels(self):
        if any(grade < 0 for grade in self.relevance.values()):
            raise ValueError("Relevance grades must be nonnegative")
        if not self.answerable and (any(self.relevance.values()) or self.evidence):
            raise ValueError("Unanswerable queries cannot have positive gold evidence")
        return self


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    document_key: str
    source_hash: str
    start: int
    end: int
    text: str
    tenant: str
    strategy: str
    parent_id: str | None = None

    @classmethod
    def from_span(
        cls, doc: Document, start: int, end: int, strategy: str, parent_id: str | None = None
    ):
        if not 0 <= start < end <= len(doc.text):
            raise ValueError("Chunk span is outside source text")
        identity = f"{doc.tenant}:{doc.key}:{doc.sha256}:{strategy}:{start}:{end}"
        return cls(
            id=str(uuid5(NAMESPACE_URL, identity)),
            document_key=doc.key,
            source_hash=doc.sha256,
            start=start,
            end=end,
            text=doc.text[start:end],
            tenant=doc.tenant,
            strategy=strategy,
            parent_id=parent_id,
        )


class Hit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    score: float
    chunk: Chunk
