import pytest

from career_copilot.graph.draft_writer import (
    _allowed_evidence_ids,
    _format_gap_report,
    check_claims_cite_real_evidence,
)
from career_copilot.schemas.draft import Claim, CoverLetterDraft
from career_copilot.schemas.gap import GapItem, GapReport


def _draft(chunk_ids: list[str]) -> CoverLetterDraft:
    return CoverLetterDraft(
        greeting="Dear Hiring Manager,",
        body="Body text.",
        closing="Best,\nMo",
        claims=[Claim(text="A claim.", evidence_chunk_ids=chunk_ids)],
    )


def test_passes_when_all_citations_are_valid():
    draft = _draft(["skills::chunk0"])
    result = check_claims_cite_real_evidence(
        draft,
        valid_ids={"skills::chunk0", "skills::chunk1"},
    )
    assert result is draft


def test_raises_on_a_hallucinated_chunk_id():
    draft = _draft(["skills::chunk0", "made_up::chunk99"])
    with pytest.raises(ValueError, match="made_up::chunk99"):
        check_claims_cite_real_evidence(
            draft,
            valid_ids={"skills::chunk0"},
        )


def test_format_gap_report_never_shows_missing_items_or_talking_points():
    report = GapReport(
        matched=[
            GapItem(
                requirement="Python",
                evidence_chunk_ids=["skills::chunk0"],
                note="matched note",
            )
        ],
        partial=[
            GapItem(
                requirement="Google Sheets",
                evidence_chunk_ids=["proj::chunk1"],
                note="partial note",
            )
        ],
        missing=[
            GapItem(
                requirement="Automation",
                evidence_chunk_ids=[],
                note="no evidence found",
            )
        ],
        suggested_talking_points=["Bridge missing experience."],
        overall_fit_summary="Strong fit overall.",
    )
    formatted = _format_gap_report(report)

    assert "Python" in formatted
    assert "Google Sheets" in formatted
    assert "Automation" not in formatted
    assert "no evidence found" not in formatted
    assert "Bridge missing experience." not in formatted


def test_only_matched_and_partial_citations_are_allowed_for_drafting():
    report = GapReport(
        matched=[
            GapItem(
                requirement="Python",
                evidence_chunk_ids=["python::chunk0"],
                note="direct evidence",
            )
        ],
        partial=[
            GapItem(
                requirement="Terraform",
                evidence_chunk_ids=["terraform::chunk0"],
                note="adjacent evidence",
            )
        ],
        missing=[
            GapItem(
                requirement="Kubernetes",
                evidence_chunk_ids=[],
                note="not demonstrated",
            )
        ],
        suggested_talking_points=[],
        overall_fit_summary="Mixed fit.",
    )

    assert _allowed_evidence_ids(report) == {
        "python::chunk0",
        "terraform::chunk0",
    }
