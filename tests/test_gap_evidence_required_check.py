"""check_matched_and_partial_have_evidence closes the gap that let a matched/partial
item through with zero real evidence to cite — the schema itself never required one
(only 'missing' is required to have none). Left uncaught, draft_writer later gets
handed that item as `[evidence: []]` and, despite its own prompt forbidding it, has
been observed inventing a placeholder chunk_id like "_" rather than dropping the claim
(the golden-eval failure that motivated this check). Pure logic, no LLM call needed.
"""
import pytest

from career_copilot.graph.gap_analysis import check_matched_and_partial_have_evidence
from career_copilot.schemas.gap import GapItem, GapReport


def test_passes_when_matched_and_partial_both_cite_evidence():
    report = GapReport(
        matched=[GapItem(requirement="Python", evidence_chunk_ids=["skills::chunk0"], note="ok")],
        partial=[GapItem(requirement="Tableau", evidence_chunk_ids=["skills::chunk1"], note="adjacent")],
        overall_fit_summary="fine",
    )
    result = check_matched_and_partial_have_evidence(report)
    assert result is report


def test_passes_when_missing_has_no_evidence():
    # 'missing' is explicitly allowed -- required, even -- to cite nothing.
    report = GapReport(
        missing=[GapItem(requirement="Kubernetes", evidence_chunk_ids=[], note="no evidence retrieved")],
        overall_fit_summary="fine",
    )
    result = check_matched_and_partial_have_evidence(report)
    assert result is report


def test_raises_when_a_matched_item_cites_no_evidence():
    report = GapReport(
        matched=[GapItem(requirement="RPA", evidence_chunk_ids=[], note="candidate has this")],
        overall_fit_summary="fine",
    )
    with pytest.raises(ValueError, match="RPA"):
        check_matched_and_partial_have_evidence(report)


def test_raises_when_a_partial_item_cites_no_evidence():
    report = GapReport(
        partial=[GapItem(requirement="企業流程自動化", evidence_chunk_ids=[], note="somewhat related")],
        overall_fit_summary="fine",
    )
    with pytest.raises(ValueError, match="企業流程自動化"):
        check_matched_and_partial_have_evidence(report)


def test_error_lists_every_offending_requirement():
    report = GapReport(
        matched=[GapItem(requirement="A", evidence_chunk_ids=[], note="x")],
        partial=[GapItem(requirement="B", evidence_chunk_ids=[], note="y")],
        overall_fit_summary="fine",
    )
    with pytest.raises(ValueError, match="A") as exc_info:
        check_matched_and_partial_have_evidence(report)
    assert "B" in str(exc_info.value)
