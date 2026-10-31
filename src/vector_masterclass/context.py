"""Bounded context with request-local citations resolving to immutable source spans."""

import re

from .schema import Chunk, Document


def validate_chunk(chunk: Chunk, sources: list[Document], tenant: str):
    by_key = {(d.tenant, d.key): d for d in sources if d.active}
    doc = by_key.get((chunk.tenant, chunk.document_key))
    if chunk.tenant not in {"public", tenant} or doc is None:
        raise PermissionError("Evidence is not in the authorized active source snapshot")
    if (
        chunk.source_hash != doc.sha256
        or not 0 <= chunk.start < chunk.end <= len(doc.text)
        or chunk.text != doc.text[chunk.start : chunk.end]
    ):
        raise ValueError("Evidence text/hash/span does not match its canonical source")


def assemble(chunks: list[Chunk], sources: list[Document], tokenizer, budget: int, *, tenant: str):
    if budget <= 0 or not tenant:
        raise ValueError("Require a positive token budget and an explicit tenant")
    selected, parts, citations, seen = [], [], {}, set()
    for chunk in chunks:
        # Validate even duplicates/oversize chunks: reject poisoned inputs before packing.
        validate_chunk(chunk, sources, tenant)
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        label = f"E{len(selected) + 1}"
        candidate = "\n\n".join([*parts, f"[{label}] {chunk.text}"])
        if len(tokenizer.encode(candidate, add_special_tokens=False)) > budget:
            continue
        selected.append(chunk)
        parts.append(f"[{label}] {chunk.text}")
        citations[label] = chunk.model_dump()
    text = "\n\n".join(parts)
    return {
        "text": text,
        "chunks": selected,
        "citations": citations,
        "tokens": len(tokenizer.encode(text, add_special_tokens=False)),
        "budget": budget,
    }


def citation_audit(answer: str, citations: dict) -> dict:
    """Check referential validity only; a resolvable citation need not support the claim."""
    labels = list(dict.fromkeys(re.findall(r"\[(E\d+)\]", answer)))
    return {
        "labels": labels,
        "unknown": [label for label in labels if label not in citations],
        "has_citation": bool(labels),
        "all_resolve": bool(labels) and all(label in citations for label in labels),
    }
