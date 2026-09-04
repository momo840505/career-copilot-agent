"""dedupe_requirements guards against the model classifying the same requirement into
more than one bucket — an observed failure mode (see the real run that motivated this:
技術文件撰寫 came back in both 'matched' AND 'missing', and separately, a golden-eval
run where a temperature-bumped retry kept moving the same conflict onto a *different*
requirement attempt after attempt instead of converging).

Originally this was a raise-and-retry check (check_no_duplicate_requirements) like
gap_analysis's other validators. It was rewritten to resolve the conflict
deterministically instead: a duplicate-bucket conflict has an uncontroversial correct
answer (see dedupe_requirements's docstring for the 'partial' > 'matched' > 'missing'
reasoning), so there's no need to spend another LLM round-trip hoping a retry lands on
it — and the golden-eval failure showed that retries don't reliably converge here
anyway. Pure logic, no LLM call needed.
"""
from career_copilot.graph.gap_analysis import dedupe_requirements
from career_copilot.schemas.gap import GapItem, GapReport


def test_returns_the_same_report_when_every_requirement_appears_once():
    report = GapReport(
        matched=[GapItem(requirement="Excel", evidence_chunk_ids=["skills::chunk0"], note="ok")],
        partial=[GapItem(requirement="Google Sheets", evidence_chunk_ids=["skills::chunk1"], note="ok")],
        missing=[GapItem(requirement="LangGraph", evidence_chunk_ids=[], note="ok")],
        overall_fit_summary="fine",
    )
    result = dedupe_requirements(report)
    assert result is report


def test_matched_vs_missing_conflict_keeps_matched_and_drops_missing():
    # This is exactly what happened live: "技術文件撰寫" classified as both matched and
    # missing. "missing" requires zero evidence, so if the SAME requirement also has a
    # matched entry (which does have evidence), the missing entry was simply wrong.
    report = GapReport(
        matched=[GapItem(requirement="技術文件撰寫", evidence_chunk_ids=["portfolio::chunk2"], note="has it")],
        missing=[GapItem(requirement="技術文件撰寫", evidence_chunk_ids=[], note="doesn't have it")],
        overall_fit_summary="fine",
    )
    result = dedupe_requirements(report)
    assert [item.requirement for item in result.matched] == ["技術文件撰寫"]
    assert result.missing == []


def test_partial_vs_matched_conflict_keeps_the_more_conservative_partial():
    # The observed golden-eval failure mode: a requirement the model couldn't commit
    # to a single bucket for. "matched" requires the unambiguous case (per the
    # gap_analysis system prompt's own tie-breaker rule) -- a requirement double-listed
    # here was, by construction, not unambiguous, so 'partial' is the honest one to keep.
    report = GapReport(
        matched=[GapItem(requirement="系統測試經驗", evidence_chunk_ids=["portfolio::chunk4"], note="strong")],
        partial=[GapItem(requirement="系統測試經驗", evidence_chunk_ids=["portfolio::chunk4"], note="adjacent")],
        overall_fit_summary="fine",
    )
    result = dedupe_requirements(report)
    assert result.matched == []
    assert [item.requirement for item in result.partial] == ["系統測試經驗"]


def test_three_way_conflict_keeps_partial():
    report = GapReport(
        matched=[GapItem(requirement="AI工具應用", evidence_chunk_ids=["a::chunk0"], note="a")],
        partial=[GapItem(requirement="AI工具應用", evidence_chunk_ids=["a::chunk0"], note="b")],
        missing=[GapItem(requirement="AI工具應用", evidence_chunk_ids=[], note="c")],
        overall_fit_summary="fine",
    )
    result = dedupe_requirements(report)
    assert result.matched == []
    assert result.missing == []
    assert [item.requirement for item in result.partial] == ["AI工具應用"]


def test_same_bucket_duplicate_is_collapsed_to_one_entry():
    # Regression: the model can also list the same requirement TWICE inside the SAME
    # bucket (no cross-bucket conflict at all -- both copies are "matched"). The old
    # filter only decided which *bucket* wins a cross-bucket conflict, then kept every
    # item whose bucket matched that winner -- so two same-bucket copies both matched
    # that check and both survived, contradicting "each requirement classified exactly
    # once". Only the first copy should remain.
    report = GapReport(
        matched=[
            GapItem(requirement="SQL", evidence_chunk_ids=["skills::chunk0"], note="first"),
            GapItem(requirement="SQL", evidence_chunk_ids=["skills::chunk1"], note="second"),
        ],
        overall_fit_summary="fine",
    )
    result = dedupe_requirements(report)
    assert len(result.matched) == 1
    assert result.matched[0].note == "first"


def test_only_the_duplicated_requirement_is_touched():
    # A conflict on one requirement must not disturb an unrelated, correctly-classified
    # one sitting in the same buckets.
    report = GapReport(
        matched=[
            GapItem(requirement="SQL", evidence_chunk_ids=["skills::chunk0"], note="ok"),
            GapItem(requirement="重複項目", evidence_chunk_ids=["skills::chunk1"], note="a"),
        ],
        partial=[GapItem(requirement="重複項目", evidence_chunk_ids=["skills::chunk1"], note="b")],
        overall_fit_summary="fine",
    )
    result = dedupe_requirements(report)
    assert [item.requirement for item in result.matched] == ["SQL"]
    assert [item.requirement for item in result.partial] == ["重複項目"]
