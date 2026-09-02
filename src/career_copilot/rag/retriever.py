"""Query-time retrieval: turn a question/requirement into the top-k most relevant chunks.

This is what Phase 3's retrieve_evidence node will call once per JD requirement (e.g. once
for "SQL", once for "cloud deployment", once for "stakeholder communication").
"""
from __future__ import annotations

from dataclasses import dataclass

from career_copilot.config import Settings
from career_copilot.rag.store import get_collection


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    distance: float  # lower = more similar (cosine distance)


def search(settings: Settings, query: str, k: int = 4) -> list[RetrievedChunk]:
    collection = get_collection(settings)
    result = collection.query(query_texts=[query], n_results=k)

    hits: list[RetrievedChunk] = []
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
        hits.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=meta["doc_id"],
                doc_title=meta["doc_title"],
                text=text,
                distance=dist,
            )
        )
    return hits
