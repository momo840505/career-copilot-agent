"""all_chunk_ids is pure aggregation logic — testable without hitting the embedding API."""
from career_copilot.graph.retrieve_evidence import EvidenceBundle, all_chunk_ids
from career_copilot.rag.retriever import RetrievedChunk


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, doc_id="d", doc_title="D", text="t", distance=0.1)


def test_all_chunk_ids_deduplicates_across_requirements():
    bundles = [
        EvidenceBundle(requirement="Python", chunks=[_chunk("skills::chunk0"), _chunk("skills::chunk1")]),
        EvidenceBundle(requirement="SQL", chunks=[_chunk("skills::chunk0")]),  # overlaps with above
        EvidenceBundle(requirement="LangGraph", chunks=[]),  # nothing retrieved
    ]
    ids = all_chunk_ids(bundles)
    assert ids == {"skills::chunk0", "skills::chunk1"}


def test_all_chunk_ids_empty_when_no_evidence_anywhere():
    bundles = [EvidenceBundle(requirement="LangGraph", chunks=[])]
    assert all_chunk_ids(bundles) == set()
