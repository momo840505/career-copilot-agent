"""_mmr_select is pure vector math -- testable on synthetic embeddings with no
real API call, unlike the rest of retriever.py (see test_retrieval_smoke.py, which
is the one that actually hits the embeddings API)."""
from career_copilot.rag.retriever import _cosine_similarity, _mmr_select


def test_mmr_prefers_diverse_candidates_over_a_near_duplicate():
    # "a" and "b" are near-identical (both close to the query direction, close to
    # each other); "c" is a bit further from the query but points somewhere very
    # different. Plain top-k-by-distance would pick a, b, dropping c entirely --
    # MMR should swap b for c once a is already selected, since b adds almost no
    # new information once a is picked.
    candidate_ids = ["a", "b", "c"]
    embeddings = [
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
    ]
    # Cosine distances to the query, consistent with each vector's closeness above.
    distances = [0.0, 0.01, 0.5]

    selected = _mmr_select(candidate_ids, embeddings, distances, k=2, lambda_mult=0.5)

    assert selected[0] == "a"  # most relevant candidate always goes first
    assert "c" in selected  # diverse-but-relevant beats a near-duplicate of "a"
    assert "b" not in selected


def test_mmr_with_lambda_1_matches_plain_top_k_by_distance():
    candidate_ids = ["x", "y", "z"]
    embeddings = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    distances = [0.05, 0.02, 0.9]  # y is actually closer than x here

    selected = _mmr_select(candidate_ids, embeddings, distances, k=2, lambda_mult=1.0)

    # lambda_mult=1.0 means the redundancy term is fully switched off -- pure
    # relevance ranking, so this should just be the two smallest distances.
    assert selected == ["y", "x"]


def test_mmr_returns_everything_when_k_covers_all_candidates():
    candidate_ids = ["p", "q"]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    distances = [0.1, 0.2]

    assert _mmr_select(candidate_ids, embeddings, distances, k=2) == candidate_ids
    assert _mmr_select(candidate_ids, embeddings, distances, k=5) == candidate_ids


def test_cosine_similarity_handles_zero_vector_without_crashing():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
