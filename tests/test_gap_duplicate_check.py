"""check_no_duplicate_requirements guards against the model classifying the same
requirement into more than one bucket — an observed failure mode (see the real run
that motivated this: 技術文件撰寫 came back in both 'matched' AND 'missing').
Pure logic, no LLM call needed."""
import pytest

from career_copilot.graph.gap_analysis import check_no_duplicate_requirements
from career_copilot.schemas.gap import GapItem, GapReport


def test_passes_when_every_requirement_appears_once():
    report = GapReport(
        matched=[GapItem(requirement="Excel", evidence_chunk_ids=[], note="ok")],
        partial=[GapItem(requirement="Google Sheets", evidence_chunk_ids=[], note="ok")],
        missing=[GapItem(requirement="LangGraph", evidence_chunk_ids=[], note="ok")],
        overall_fit_summary="fine",
    )
    result = check_no_duplicate_requirements(report)
    assert result is report


def test_raises_when_a_requirement_appears_in_two_buckets():
    # This is exactly what happened live: "技術文件撰寫" classified as both matched and missing.
    report = GapReport(
        matched=[GapItem(requirement="技術文件撰寫", evidence_chunk_ids=[], note="has it")],
        missing=[GapItem(requirement="技術文件撰寫", evidence_chunk_ids=[], note="doesn't have it")],
        overall_fit_summary="fine",
    )
    with pytest.raises(ValueError, match="技術文件撰寫"):
        check_no_duplicate_requirements(report)


def test_raises_when_a_requirement_appears_in_three_buckets():
    report = GapReport(
        matched=[GapItem(requirement="AI工具應用", evidence_chunk_ids=[], note="a")],
        partial=[GapItem(requirement="AI工具應用", evidence_chunk_ids=[], note="b")],
        missing=[GapItem(requirement="AI工具應用", evidence_chunk_ids=[], note="c")],
        overall_fit_summary="fine",
    )
    with pytest.raises(ValueError, match="AI工具應用"):
        check_no_duplicate_requirements(report)
