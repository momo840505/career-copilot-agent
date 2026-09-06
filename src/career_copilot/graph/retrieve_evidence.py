"""For every JD requirement, retrieves the most relevant portfolio evidence.

One retrieval call per requirement (not one big query for the whole JD) is deliberate:
gap_analysis needs to see, requirement by requirement, whether *anything* relevant came
back at all — that per-requirement signal is what lets it tell matched vs. partial vs.
missing apart, rather than one blurry "here's some stuff that's kind of related".
"""
from __future__ import annotations

from dataclasses import dataclass

from career_copilot.config import Settings, get_settings
from career_copilot.rag.retriever import RetrievedChunk, search
from career_copilot.schemas.jd import JDRequirements


@dataclass
class EvidenceBundle:
    requirement: str
    chunks: list[RetrievedChunk]


def retrieve_evidence(
    jd: JDRequirements,
    settings: Settings | None = None,
    k_per_requirement: int = 3,
) -> list[EvidenceBundle]:
    settings = settings or get_settings()
    requirements = jd.must_have_skills + jd.nice_to_have_skills
    return [
        EvidenceBundle(requirement=req, chunks=search(settings, req, k=k_per_requirement))
        for req in requirements
    ]


def all_chunk_ids(bundles: list[EvidenceBundle]) -> set[str]:
    """Every chunk_id actually shown to the model — the ground truth gap_analysis's
    citation check validates against."""
    return {chunk.chunk_id for bundle in bundles for chunk in bundle.chunks}
