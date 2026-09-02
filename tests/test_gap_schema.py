"""GapReport schema tests — pure Pydantic logic, no API key needed."""
import pytest
from pydantic import ValidationError

from career_copilot.schemas.gap import GapItem, GapReport


def test_valid_report_parses():
    report = GapReport(
        matched=[GapItem(requirement="Python", evidence_chunk_ids=["skills::chunk0"], note="Has it.")],
        partial=[],
        missing=[GapItem(requirement="LangGraph", evidence_chunk_ids=[], note="Not found.")],
        suggested_talking_points=["Frame X as adjacent experience."],
        overall_fit_summary="Reasonable fit with one clear gap.",
    )
    assert report.matched[0].requirement == "Python"
    assert report.missing[0].evidence_chunk_ids == []


def test_missing_item_cannot_cite_evidence():
    with pytest.raises(ValidationError):
        GapReport(
            missing=[
                GapItem(
                    requirement="LangGraph",
                    evidence_chunk_ids=["skills::chunk0"],  # contradiction: cites evidence but "missing"
                    note="Not found.",
                )
            ],
            overall_fit_summary="x",
        )


def test_empty_report_is_valid_shape():
    # No matched/partial/missing items yet is a legitimate (if useless) state —
    # only overall_fit_summary is required.
    report = GapReport(overall_fit_summary="No requirements evaluated.")
    assert report.matched == []
    assert report.missing == []
