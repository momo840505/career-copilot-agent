"""Shared formatting: turn retrieved evidence into one flat, deduplicated chunk_id ->
text listing usable in a prompt. draft_writer and critic both need the exact same view
of "what evidence exists" (the critic is re-checking the writer's citations against
it), so this lives in one place instead of two copies drifting apart.
"""
from __future__ import annotations

from career_copilot.graph.retrieve_evidence import EvidenceBundle, RetrievedChunk


def format_evidence_lookup(bundles: list[EvidenceBundle]) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for b in bundles:
        for c in b.chunks:
            if c.chunk_id in seen:
                continue
            seen.add(c.chunk_id)
            lines.append(f"- chunk_id={c.chunk_id} (from {c.doc_title}): {c.text}")
    return "\n".join(lines)


def build_chunk_lookup(bundles: list[EvidenceBundle]) -> dict[str, RetrievedChunk]:
    """chunk_id -> chunk, so a caller can join a citation directly to its text instead
    of asking the model to cross-reference two separately-shaped lists itself."""
    lookup: dict[str, RetrievedChunk] = {}
    for b in bundles:
        for c in b.chunks:
            lookup[c.chunk_id] = c
    return lookup
