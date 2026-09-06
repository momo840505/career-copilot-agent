"""reclassify_uncited_as_missing closes the gap that let a matched/partial item
through with zero real evidence to cite — the schema itself never required one (only
'missing' is required to have none). Left uncaught, draft_writer later gets handed
that item as `[evidence: []]` and, despite its own prompt forbidding it, has been
observed inventing a placeholder chunk_id like "_" rather than dropping the claim
(the golden-eval failure that motivated this check).

This used to raise ValueError and rely on the model fixing itself on retry. That
didn't reliably converge on live golden JDs (the model kept re-asserting the same
uncited matched/partial classification attempt after attempt), so it now
deterministically moves the offending item into 'missing' instead — no LLM call
needed, and it never invents a citation to do it.
"""

from career_copilot.graph.gap_analysis import reclassify_uncited_as_missing
from career_copilot.schemas.gap import GapItem, GapReport


def test_passes_through_when_matched_and_partial_both_cite_evidence():
    report = GapReport(
        matched=[GapItem(requirement="Python", evidence_chunk_ids=["skills::chunk0"], note="ok")],
        partial=[GapItem(requirement="Tableau", evidence_chunk_ids=["skills::chunk1"], note="adjacent")],
        overall_fit_summary="fine",
    )
    result = reclassify_uncited_as_missing(report)
    assert result is report
    assert [i.requirement for i in result.matched] == ["Python"]
    assert [i.requirement for i in result.partial] == ["Tableau"]
    assert result.missing == []


def test_passes_through_when_missing_already_has_no_evidence():
    # 'missing' is explicitly allowed -- required, even -- to cite nothing.
    report = GapReport(
        missing=[GapItem(requirement="Kubernetes", evidence_chunk_ids=[], note="no evidence retrieved")],
        overall_fit_summary="fine",
    )
    result = reclassify_uncited_as_missing(report)
    assert result is report


def test_moves_an_uncited_matched_item_to_missing():
    report = GapReport(
        matched=[GapItem(requirement="RPA", evidence_chunk_ids=[], note="candidate has this")],
        overall_fit_summary="fine",
    )
    result = reclassify_uncited_as_missing(report)
    assert result.matched == []
    assert [i.requirement for i in result.missing] == ["RPA"]
    assert result.missing[0].evidence_chunk_ids == []
    # The original note is preserved -- this is a reclassification, not a rewrite.
    assert result.missing[0].note == "candidate has this"


def test_moves_an_uncited_partial_item_to_missing():
    report = GapReport(
        partial=[GapItem(requirement="企業流程自動化", evidence_chunk_ids=[], note="somewhat related")],
        overall_fit_summary="fine",
    )
    result = reclassify_uncited_as_missing(report)
    assert result.partial == []
    assert [i.requirement for i in result.missing] == ["企業流程自動化"]


def test_moves_every_offending_requirement_and_leaves_cited_ones_alone():
    report = GapReport(
        matched=[
            GapItem(requirement="A", evidence_chunk_ids=[], note="x"),
            GapItem(requirement="Python", evidence_chunk_ids=["skills::chunk0"], note="ok"),
        ],
        partial=[GapItem(requirement="B", evidence_chunk_ids=[], note="y")],
        overall_fit_summary="fine",
    )
    result = reclassify_uncited_as_missing(report)
    assert [i.requirement for i in result.matched] == ["Python"]
    assert result.partial == []
    assert {i.requirement for i in result.missing} == {"A", "B"}


def test_returned_report_still_satisfies_the_schemas_own_missing_validator():
    # GapReport's own field_validator rejects a 'missing' item that cites anything --
    # confirm the reclassified items really do land with an empty citation list, not
    # just that this function's own bookkeeping thinks they do.
    report = GapReport(
        matched=[GapItem(requirement="RPA", evidence_chunk_ids=[], note="candidate has this")],
        overall_fit_summary="fine",
    )
    result = reclassify_uncited_as_missing(report)
    GapReport.model_validate(result.model_dump())
