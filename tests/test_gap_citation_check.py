"""check_citations_are_real is the anti-hallucination guard for gap_analysis —
pure logic, testable without any LLM call."""
import pytest

from career_copilot.graph.gap_analysis import check_citations_are_real
from career_copilot.schemas.gap import GapItem, GapReport


def test_passes_when_all_citations_are_valid():
    report = GapReport(
        matched=[GapItem(requirement="Python", evidence_chunk_ids=["skills::chunk0"], note="ok")],
        overall_fit_summary="fine",
    )
    result = check_citations_are_real(report, valid_ids={"skills::chunk0", "skills::chunk1"})
    assert result is report


def test_raises_on_a_hallucinated_chunk_id():
    report = GapReport(
        matched=[
            GapItem(
                requirement="Python",
                evidence_chunk_ids=["skills::chunk0", "made_up::chunk99"],  # chunk99 was never retrieved
                note="ok",
            )
        ],
        overall_fit_summary="fine",
    )
    with pytest.raises(ValueError, match="made_up::chunk99"):
        check_citations_are_real(report, valid_ids={"skills::chunk0"})
