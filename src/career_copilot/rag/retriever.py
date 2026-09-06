"""Query-time retrieval: turn a question/requirement into the top-k most relevant chunks.

This is what retrieve_evidence.py calls once per JD requirement (e.g. once for "SQL",
once for "cloud deployment", once for "stakeholder communication").

Plain top-k cosine similarity has one recurring failure mode on a portfolio this small:
a project doc gets split into several overlapping chunks (see rag/chunker.py's 80-char
overlap), so the k nearest neighbours to a query are often 3 near-duplicate slices of
the SAME paragraph rather than k genuinely different pieces of evidence. Citation
diversity matters here specifically because draft_writer only ever sees what gets
retrieved -- if all 4 "evidence" chunks for a requirement say almost the same sentence,
the critic has nothing but repetition to judge groundedness against.

MMR (Maximal Marginal Relevance) fixes this without a second model call: pull back a
larger candidate pool than we actually need (fetch_k), then greedily pick chunks that
are still relevant to the query but not too similar to what's already been picked.
`_mmr_select` is the actual selection logic, factored out so it's testable on plain
vectors without touching the embeddings API (see tests/test_retriever_mmr.py).
"""
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
    distance: float  # lower = more similar (cosine distance)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity -- no numpy import needed just for this, and
    chromadb (a real dependency already) pulls numpy in transitively anyway, but
    there's no reason to depend on that indirectly rather than just computing a dot
    product over two same-length lists directly."""
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
    """Greedily select `k` of `candidate_ids`, balancing relevance to the query
    (higher = closer to the front of `distances`) against redundancy with whatever's
    already been picked.

    `lambda_mult` is the relevance/diversity trade-off: 1.0 collapses back to plain
    top-k-by-distance (no diversity effect at all); 0.0 ignores the query entirely and
    just spreads picks as far apart from each other as possible. 0.7 favours relevance
    but still lets a clearly redundant near-duplicate lose to a more different,
    slightly-less-relevant chunk.

    `distances` are cosine distances from chromadb (hnsw:space="cosine", see
    rag/store.py), so `1 - distance` is used as the relevance term to keep both sides
    of the trade-off on the same cosine-similarity scale.
    """
    if k >= len(candidate_ids):
        return list(candidate_ids)

    relevance = [1.0 - d for d in distances]
    remaining = list(range(len(candidate_ids)))
    selected: list[int] = []

    while remaining and len(selected) < k:
        if not selected:
            # First pick is always the single most relevant candidate -- there's
            # nothing selected yet to be redundant with.
            best_idx = max(remaining, key=lambda i: relevance[i])
        else:
            def mmr_score(i: int) -> float:
                redundancy = max(
                    _cosine_similarity(embeddings[i], embeddings[j]) for j in selected
                )
                return lambda_mult * relevance[i] - (1.0 - lambda_mult) * redundancy

            best_idx = max(remaining, key=mmr_score)
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
    """Retrieve the top `k` chunks for `query`.

    By default over-fetches `fetch_k` candidates (min(4 * k, 20) unless given
    explicitly) and re-ranks them with MMR so the final `k` favour genuinely
    different evidence over near-duplicate slices of the same paragraph. Pass
    `use_mmr=False` for plain top-k-by-distance (e.g. to compare behavior, or if a
    future caller genuinely wants the raw ranking).
    """
    collection = get_collection(settings)
    effective_fetch_k = fetch_k if fetch_k is not None else min(4 * k, 20)
    effective_fetch_k = max(effective_fetch_k, k)

    result = collection.query(
        query_texts=[query],
        n_results=effective_fetch_k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    embeddings = result["embeddings"][0]

    by_id = {
        chunk_id: RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=meta["doc_id"],
            doc_title=meta["doc_title"],
            text=text,
            distance=dist,
        )
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists)
    }

    if use_mmr and len(ids) > k:
        ordered_ids = _mmr_select(list(ids), list(embeddings), list(dists), k)
    else:
        ordered_ids = list(ids)[:k]

    return [by_id[chunk_id] for chunk_id in ordered_ids]
