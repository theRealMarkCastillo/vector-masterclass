"""Inspectable chunk boundaries with character offsets into immutable canonical text."""

import re

from .schema import Chunk, Document


def chunk_document(
    doc: Document, strategy: str = "fixed", words: int = 48, overlap: int = 8
) -> list[Chunk]:
    if words <= 0 or not 0 <= overlap < words:
        raise ValueError("Require words > 0 and 0 <= overlap < words")
    if strategy == "document":
        return [Chunk.from_span(doc, 0, len(doc.text), "document")]
    if strategy == "structural":
        return [
            Chunk.from_span(doc, m.start(), m.end(), strategy)
            for m in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", doc.text)
        ]
    if strategy not in {"fixed", "parent_child"}:
        raise ValueError(f"Unknown chunk strategy: {strategy}")
    tokens = list(re.finditer(r"\S+", doc.text))
    parent = Chunk.from_span(doc, 0, len(doc.text), "document")
    chunks = []
    for index in range(0, len(tokens), words - overlap):
        group = tokens[index : index + words]
        start, end = group[0].start(), group[-1].end()
        chunks.append(
            Chunk.from_span(
                doc,
                start,
                end,
                f"{strategy}:{words}:{overlap}",
                parent.id if strategy == "parent_child" else None,
            )
        )
        if index + words >= len(tokens):
            break
    return chunks


def pack_chunks(chunks: list[Chunk], tokenizer, budget: int) -> dict:
    """Whole-chunk greedy packing includes citation header tokens and separators."""
    if budget <= 0:
        raise ValueError("Context budget must be positive")
    parts, selected, seen = [], [], set()
    for chunk in chunks:
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        part = f"[{chunk.id}] {chunk.text}"
        candidate = "\n\n".join([*parts, part])
        count = len(tokenizer.encode(candidate, add_special_tokens=False))
        if count <= budget:
            parts.append(part)
            selected.append(chunk)
    text = "\n\n".join(parts)
    return {
        "text": text,
        "chunks": selected,
        "tokens": len(tokenizer.encode(text, add_special_tokens=False)),
        "budget": budget,
    }
