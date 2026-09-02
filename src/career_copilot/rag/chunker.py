"""Split each SourceDoc into retrievable chunks with stable, citeable IDs.

Why chunk at all instead of embedding the whole document? Two reasons that matter for
this project specifically:
  1. Retrieval precision — a JD requirement like "SQL" should pull back the 2-3 sentences
     about SQL/PostgreSQL, not the entire 500-word project write-up diluting the match.
  2. Citation granularity — Phase 4's critic checks "does this sentence in the draft trace
     back to a real chunk?". That check is only meaningful if chunks are small enough that
     citing one actually commits to a specific claim.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from career_copilot.rag.loader import SourceDoc


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    tags: list[str]
    text: str


def chunk_docs(
    docs: list[SourceDoc],
    chunk_size: int = 600,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks: list[Chunk] = []
    for doc in docs:
        pieces = splitter.split_text(doc.content)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::chunk{i}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    tags=doc.tags,
                    text=piece.strip(),
                )
            )
    return chunks
