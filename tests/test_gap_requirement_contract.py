import pytest

from career_copilot.graph.gap_analysis import check_requirement_contract
from career_copilot.graph.retrieve_evidence import EvidenceBundle
from career_copilot.rag.retriever import RetrievedChunk
from career_copilot.schemas.gap import GapItem, GapReport


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="doc",
        doc_title="Doc",
        text="evidence",
        distance=0.1,
    )


def test_requires_every_jd_requirement_exactly_once():
    report = GapReport(
        matched=[
            GapItem(
                requirement="SQL",
                evidence_chunk_ids=["sql::0"],
                note="Direct match.",
            )
        ],
        overall_fit_summary="ok",
    )
    bundles = [
        EvidenceBundle(requirement="SQL", chunks=[_chunk("sql::0")]),
        EvidenceBundle(requirement="AWS", chunks=[]),
    ]

    with pytest.raises(ValueError, match="not classified"):
        check_requirement_contract(report, bundles, ["SQL", "AWS"])


def test_rejects_an_invented_requirement():
    report = GapReport(
        matched=[
            GapItem(
                requirement="SQL",
                evidence_chunk_ids=["sql::0"],
                note="Direct match.",
            )
        ],
        missing=[
            GapItem(requirement="Kafka", evidence_chunk_ids=[], note="No evidence.")
        ],
        overall_fit_summary="ok",
    )
    bundles = [EvidenceBundle(requirement="SQL", chunks=[_chunk("sql::0")])]

    with pytest.raises(ValueError, match="not present in the JD"):
        check_requirement_contract(report, bundles, ["SQL"])


def test_rejects_cross_requirement_citations():
    report = GapReport(
        matched=[
            GapItem(
                requirement="AWS",
                evidence_chunk_ids=["sql::0"],
                note="Incorrect citation.",
            )
        ],
        missing=[
            GapItem(requirement="SQL", evidence_chunk_ids=[], note="No evidence.")
        ],
        overall_fit_summary="ok",
    )
    bundles = [
        EvidenceBundle(requirement="SQL", chunks=[_chunk("sql::0")]),
        EvidenceBundle(requirement="AWS", chunks=[_chunk("aws::0")]),
    ]

    with pytest.raises(ValueError, match="another requirement"):
        check_requirement_contract(report, bundles, ["SQL", "AWS"])
