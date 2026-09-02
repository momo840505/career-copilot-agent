"""check_claims_cite_real_evidence is draft_writer's anti-hallucination guard —
same pattern as gap_analysis's citation check. Pure logic, no LLM call needed."""
import pytest

from career_copilot.graph.draft_writer import _format_gap_report, check_claims_cite_real_evidence
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
    result = check_claims_cite_real_evidence(draft, valid_ids={"skills::chunk0", "skills::chunk1"})
    assert result is draft


def test_raises_on_a_hallucinated_chunk_id():
    draft = _draft(["skills::chunk0", "made_up::chunk99"])
    with pytest.raises(ValueError, match="made_up::chunk99"):
        check_claims_cite_real_evidence(draft, valid_ids={"skills::chunk0"})


def test_format_gap_report_never_shows_missing_items_or_talking_points():
    # Regression test for the Phase 4b bug: draft_writer was shown "missing" items and
    # ungrounded suggested_talking_points, and tried to write citable claims about them
    # anyway — inventing placeholder chunk_ids ("_", "") when it couldn't find real
    # evidence. The fix is structural (don't show it these fields at all), so lock that
    # down directly rather than only re-testing the symptom.
    report = GapReport(
        matched=[GapItem(requirement="Python", evidence_chunk_ids=["skills::chunk0"], note="matched note")],
        partial=[GapItem(requirement="Google Sheets", evidence_chunk_ids=["proj::chunk1"], note="partial note")],
        missing=[GapItem(requirement="企業流程自動化", evidence_chunk_ids=[], note="no evidence found")],
        suggested_talking_points=["Although I haven't done X, my experience with Y demonstrates..."],
        overall_fit_summary="Strong fit overall.",
    )
    formatted = _format_gap_report(report)
    assert "Python" in formatted
    assert "Google Sheets" in formatted
    assert "企業流程自動化" not in formatted
    assert "no evidence found" not in formatted
    assert "demonstrates" not in formatted  # the talking point text must not leak through
    assert "talking point" not in formatted.lower()
