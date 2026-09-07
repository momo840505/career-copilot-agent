from __future__ import annotations

import math
from dataclasses import dataclass

from career_copilot.config import Settings
from career_copilot.rag.store import get_collection


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    distance: float


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mmr_select(
    candidate_ids: list[str],
    embeddings: list[list[float]],
    distances: list[float],
    k: int,
    lambda_mult: float = 0.7,
) -> list[str]:
    if k >= len(candidate_ids):
        return list(candidate_ids)

    relevance = [1.0 - distance for distance in distances]
    remaining = list(range(len(candidate_ids)))
    selected: list[int] = []

    while remaining and len(selected) < k:
        if not selected:
            best_idx = max(remaining, key=lambda i: relevance[i])
        else:

            def score(i: int) -> float:
                redundancy = max(
                    _cosine_similarity(embeddings[i], embeddings[j]) for j in selected
                )
                return lambda_mult * relevance[i] - (1.0 - lambda_mult) * redundancy

            best_idx = max(remaining, key=score)

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidate_ids[i] for i in selected]


def search(
    settings: Settings,
    query: str,
    k: int = 4,
    fetch_k: int | None = None,
    use_mmr: bool = True,
) -> list[RetrievedChunk]:
    collection = get_collection(settings)
    effective_fetch_k = max(fetch_k or min(4 * k, 20), k)

    result = collection.query(
        query_texts=[query],
        n_results=effective_fetch_k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    rows = list(
        zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            result["embeddings"][0],
        )
    )
    if settings.rag_max_distance is not None:
        rows = [row for row in rows if row[3] <= settings.rag_max_distance]
    if not rows:
        return []

    ids = [row[0] for row in rows]
    embeddings = [row[4] for row in rows]
    distances = [row[3] for row in rows]

    by_id = {
        chunk_id: RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=metadata["doc_id"],
            doc_title=metadata["doc_title"],
            text=document,
            distance=distance,
        )
        for chunk_id, document, metadata, distance, _embedding in rows
    }

    if use_mmr and len(ids) > k:
        ordered_ids = _mmr_select(ids, embeddings, distances, k)
    else:
        ordered_ids = ids[:k]

    return [by_id[chunk_id] for chunk_id in ordered_ids]
